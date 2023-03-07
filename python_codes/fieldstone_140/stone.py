import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import random

#------------------------------------------------------------------------------

def NN(r,s):
    return np.array([1-r-s,r,s],dtype=np.float64)

def dNNdr(r,s):
    return np.array([-1,1,0],dtype=np.float64)

def dNNds(r,s):
    return np.array([-1,0,1],dtype=np.float64)

#------------------------------------------------------------------------------

def export_elements_to_vtu(x,y,z,zc,A,sorted_indices,icon,filename,area):
    N=np.size(x)
    m,nel=np.shape(icon)

    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],z[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='elevation' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e \n" %(z[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='zc' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (zc[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='drainage area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (A[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sorted_indices' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d\n" % (sorted_indices[iel]))
    vtufile.write("</DataArray>\n")

    vtufile.write("</CellData>\n")

    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#------------------------------------------------------------------------------

def compute_segs(InputCoords):
    segs = np.stack([np.arange(len(InputCoords)),np.arange(len(InputCoords))+1],axis=1)%len(InputCoords)
    return segs

#------------------------------------------------------------------------------

def compute_triangles_area(coords,nodesArray):
    
    tx=coords[:,0]
    ty=coords[:,1]
    
    # Triangle Area is calculated via Heron's formula, see wikipedia
    
    a=np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,1]])**2)
    b=np.sqrt((tx[nodesArray[:,2]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,2]]-ty[nodesArray[:,1]])**2)
    c=np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,2]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,2]])**2)
    
    area = 0.5 * np.sqrt(a**2 * c**2 - (( a**2 + c**2 - b**2) / 2)**2)
    area = area.reshape(-1,1) #Transposing the 1xN matrix into Nx1 shape
    
    return area

###############################################################################
###############################################################################

eps=1e-8
year=365.25*24*3600

Lx=50e3
Ly=50e3

rexp = 2       # drainage area exponent
mexp = 1       # slope exponent
w = 1e-3/year  # rock uplift rate (m/s)
c = 1e-11/year # fluvial erosion coeff. (mˆ(2-2rexp)/s)
c0 = 0.1/year  # linear diffusion coeff.(mˆ2/s)
Ac = 0         # critical drainage area (mˆ2)
scale = 10     # amplitude random initial topography (m)

nstep = 3000  # number of time steps
ndim = 2      # number of spatial dimensions
m = 3         # number of nodes in 1 element
dt = 100*year # time step (s)
tol = 1e-3    # iteration tolerance

#------------------------------------------------------------------------------
# Gauss quadrature setup

nqel=3

#qcoords_r=np.array([1/6,2/3,1/6],dtype=np.float64) 
#qcoords_s=np.array([1/6,1/6,2/3],dtype=np.float64) 

qcoords_r=np.array([1/2,1/2,0],dtype=np.float64) 
qcoords_s=np.array([1/2,0,1/2],dtype=np.float64) 
qweights =np.array([1/6,1/6,1/6],dtype=np.float64) 

#WHYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY?


#------------------------------------------------------------------------------
# build mesh
#------------------------------------------------------------------------------
 
square_vertices = np.array([[0,0],[0,Lx],[Lx,Ly],[Ly,0]])
square_edges = compute_segs(square_vertices)

O1 = {'vertices' : square_vertices, 'segments' : square_edges}
T1 = tr.triangulate(O1, 'pqa60000000') # tr.triangulate() computes the main dictionary 

tr.compare(plt, O1, T1) # The tr.compare() function always takes plt as its 1st argument
#plt.savefig('ex1.pdf', bbox_inches='tight')
#plt.show()

#print('vertices:',T1['vertices'])
#print('segments:',T1['segments'])
#print('triangles connectivity:',T1['triangles']) # this is icon!
#print('vertices on hull:',T1['vertex_markers'])
#print('segments on hull:',T1['segment_markers'])

area=compute_triangles_area(T1['vertices'], T1['triangles'])
icon=T1['triangles'] ; icon=icon.T
m,nel=np.shape(icon)
x=T1['vertices'][:,0] 
y=T1['vertices'][:,1] 

print(np.sum(area),Lx*Ly)

N=np.size(x)

#------------------------------------------------------------------------------
# initial topography 
#------------------------------------------------------------------------------
z=np.zeros(N,dtype=np.float64) 

for i in range(0,N):
    #z[i]=x[i]/10000+y[i]/11000 # random.uniform(0,10)
    z[i]= random.uniform(0,10)

#------------------------------------------------------------------------------
# compute elevation of middle of element zc 
#------------------------------------------------------------------------------
zc=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    zc[iel]=(z[icon[0,iel]]+z[icon[1,iel]]+z[icon[2,iel]])/3


#------------------------------------------------------------------------------
# boundary conditions
#------------------------------------------------------------------------------

bc_fix = np.zeros(N,dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(N,dtype=np.float64)  # boundary condition, value

for i in range(0,N):
    if abs(x[i]/Lx)<eps:
       bc_fix[i] = True ; bc_val[i]=0
    if abs(x[i]/Lx-1)<eps:
       bc_fix[i] = True ; bc_val[i]=0
    if abs(y[i]/Ly)<eps:
       bc_fix[i] = True ; bc_val[i]=0
    if abs(y[i]/Ly-1)<eps:
       bc_fix[i] = True ; bc_val[i]=0

#------------------------------------------------------------------------------
# compute gnei (step 2)
# Find and save the three elements adjacent to each element. Elements on
# the boundary that have only two neighbors are assigned their own element 
# index to the missing third neighbor.
# face0: nodes 0-1
# face1: nodes 1-2
# face2: nodes 2-0
# node1,2 are nodes making given face
#------------------------------------------------------------------------------

gnei = np.zeros((3,nel),dtype=np.int32) 

for iel in range(0,nel):
    #print('-------------------')
    for iface in range(0,3):
        if iface==0:
           iel_node1=icon[0,iel] 
           iel_node2=icon[1,iel] 
           found0=False
        if iface==1:
           iel_node1=icon[1,iel] 
           iel_node2=icon[2,iel] 
           found1=False
        if iface==2:
           iel_node1=icon[2,iel] 
           iel_node2=icon[0,iel] 
           found2=False
        #print('iel=',iel,'iface=',iface,iel_node1,iel_node2)
        for jel in range(0,nel):
            if jel!=iel:
               for jface in range(0,3):
                   if jface==0:
                      jel_node1=icon[0,jel] 
                      jel_node2=icon[1,jel] 
                   if jface==1:
                      jel_node1=icon[1,jel] 
                      jel_node2=icon[2,jel] 
                   if jface==2:
                      jel_node1=icon[2,jel] 
                      jel_node2=icon[0,jel] 
                   #print('   -> jel=',jel,'jface=',jface,jel_node1,jel_node2)

                   if iel_node1==jel_node2 and iel_node2==jel_node1:
                      gnei[iface,iel]=jel
                      if iface==0: found0=True
                      if iface==1: found1=True
                      if iface==2: found2=True
                   #end if

               #end for jface
            #end if jel/=iel       
        #end for jel
    #end for iface 
    if not found0: gnei[0,iel]=iel 
    if not found1: gnei[1,iel]=iel 
    if not found2: gnei[2,iel]=iel 
    #print('element behind face 0 of element',iel,'is element',gnei[0,iel])
    #print('element behind face 1 of element',iel,'is element',gnei[1,iel])
    #print('element behind face 2 of element',iel,'is element',gnei[2,iel])
#end for iel

#print(gnei)

#------------------------------------------------------------------------------
# step4: Sort the average element elevation for the entire mesh from highest 
# to lowest. The indices of the sorted elevations are saved in the array sorted indices. 
#------------------------------------------------------------------------------

sorted_indices=zc.argsort()
sorted_indices=np.flip(sorted_indices[:])

#print(sorted_indices)

#print(zc)
print(zc[sorted_indices])
print('----')

#------------------------------------------------------------------------------
# step 5: Using the sorted indices, sort the neighboring elements, the result of 
# which is saved in sorted gnei.
# At this stage, all elements in the mesh have been ordered, along with their 
# three adjacent neighbors, from highest to lowest.
#------------------------------------------------------------------------------

sorted_gnei = np.zeros((3,nel),dtype=np.int32) 

sorted_gnei[:,:] = gnei[:,sorted_indices]

print(gnei)
print('----')
print(sorted_gnei)

#------------------------------------------------------------------------------
# step 6: For each element, find and save the local index of the lowest 
# of the three adjacent elements. The result is 0, 1, or 2
#------------------------------------------------------------------------------

min_index = np.zeros(nel,dtype=np.int32) 

for iel in range(0,nel):
    zc_0=zc[gnei[0,iel]]
    zc_1=zc[gnei[1,iel]]
    zc_2=zc[gnei[2,iel]]
    if zc_0<zc_1 and zc_0<zc_2: 
       min_index[iel]=0
       print('I am element',iel,'and I give to element',min_index[iel],gnei[0,iel])
    if zc_1<zc_0 and zc_1<zc_2: 
       min_index[iel]=1
       print('I am element',iel,'and I give to element',min_index[iel],gnei[1,iel])
    if zc_2<zc_0 and zc_2<zc_1: 
       min_index[iel]=2
       print('I am element',iel,'and I give to element',min_index[iel],gnei[2,iel])
    #print(gnei[:,iel],zc[gnei[:,iel]])

#------------------------------------------------------------------------------
# step 7

A = np.zeros(nel,dtype=np.float64) 

A[:]=area[:,0]

#------------------------------------------------------------------------------
# step 8
#In a loop over all ordered elements from highest to lowest, “pass” the accumulated drainage area from
#each (donor) element to its lowest adjacent neighbor (receiver). The result after the loop has been
#completed is the accumulated surface area that “drains” to each element in the landscape, A. Note
#that this procedure implicitly assumes a spatially uniform rainfall, which can easily be accounted for
#if desired.

for iel in range(0,nel):
   donor = sorted_indices[iel] 
   #receiver = sorted_gnei[min_index[iel],iel] suspicious line in book?!
   receiver=gnei[min_index[donor],donor]
   print('loop index ',iel,'is donor',donor,'gives to ',receiver)
   A[receiver] += A[donor] 

export_elements_to_vtu(x,y,z,zc,A,sorted_indices,icon,'solution_0.vtu',area)

#------------------------------------------------------------------------------
# solve system
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# export to vtu
#------------------------------------------------------------------------------




###############################################################################
# compute area of elements
###############################################################################

area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV=NN(rq,sq)
        dNNNVdr=dNNdr(rq,sq)
        dNNNVds=dNNds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,m):
            jcb[0,0] += dNNNVdr[k]*x[icon[k,iel]]
            jcb[0,1] += dNNNVdr[k]*y[icon[k,iel]]
            jcb[1,0] += dNNNVds[k]*x[icon[k,iel]]
            jcb[1,1] += dNNNVds[k]*y[icon[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq

print(area,np.sum(area))

