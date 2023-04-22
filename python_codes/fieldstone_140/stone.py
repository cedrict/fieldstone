import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import random
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as timing

#--------------------------------------------------------------------------------------------------

cm=0.01
km=1e3
eps=1e-6
year=365.25*24*3600
ndim=2      # number of spatial dimensions
m=3         # number of nodes in 1 element

#--------------------------------------------------------------------------------------------------
# P1 basis functions
#--------------------------------------------------------------------------------------------------

def NN(r,s):
    return np.array([1-r-s,r,s],dtype=np.float64)

def dNNdr(r,s):
    return np.array([-1,1,0],dtype=np.float64)

def dNNds(r,s):
    return np.array([-1,0,1],dtype=np.float64)

#--------------------------------------------------------------------------------------------------

def export_elements_to_vtu(x,y,z,u,v,w,zc,A,sorted_indices,dhdx,dhdy,slope,kappa,icon,filename,area,border,catchment,dz):
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
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],z[i]*zscale))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='elevation' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e \n" %(z[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='elevation change (dz)' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e \n" %(dz[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e %10e %10e \n" %(u[i]*year/cm,v[i]*year/cm,w[i]*year/cm))
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
    vtufile.write("<DataArray type='Float32' Name='dhdx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (dhdx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='dhdy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (dhdy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='slope (%)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (slope[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='slope (angle)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (np.arctan(slope[iel])/np.pi*180))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='kappa' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (kappa[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='border' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (border[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='catchment' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (catchment[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sorted_indices' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d\n" % (sorted_indices[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#--------------------------------------------------------------------------------------------------

def export_network_to_vtu(x,y,z,xc,yc,zc,A,icon,min_index,filename):
    m,nel=np.shape(icon)
    vx=np.zeros(nel,dtype=np.float64) 
    vy=np.zeros(nel,dtype=np.float64) 
    vz=np.zeros(nel,dtype=np.float64) 
    for iel in range(0,nel):
        jel=gnei[min_index[iel],iel]
        if iel==jel:
           vx[iel]=0
           vy[iel]=0
           vz[iel]=0
        else:
           vx[iel]=xc[jel]-xc[iel]
           vy[iel]=yc[jel]-yc[iel]
           vz[iel]=zc[jel]-zc[iel]

    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nel,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e %10e %10e \n" %(xc[i],yc[i],zc[i]*zscale))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='network' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e %10e %10e \n" %(vx[i],vy[i],vz[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='drainage area' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%e \n" % A[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%d " % i) 
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%d " % (i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range(0,nel):
        vtufile.write("%d " % 1) 
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#--------------------------------------------------------------------------------------------------

def compute_segs(InputCoords):
    segs = np.stack([np.arange(len(InputCoords)),np.arange(len(InputCoords))+1],axis=1)%len(InputCoords)
    return segs

#--------------------------------------------------------------------------------------------------
# Triangle Area is calculated via Heron's formula, see wikipedia

def compute_triangles_area(coords,nodesArray):
    tx=coords[:,0]
    ty=coords[:,1]
    a=np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,1]])**2)
    b=np.sqrt((tx[nodesArray[:,2]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,2]]-ty[nodesArray[:,1]])**2)
    c=np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,2]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,2]])**2)
    area = 0.5 * np.sqrt(a**2 * c**2 - (( a**2 + c**2 - b**2) / 2)**2)
    area = area.reshape(-1,1) #Transposing the 1xN matrix into Nx1 shape
    return area

#--------------------------------------------------------------------------------------------------
# compute slope

def compute_element_slope(x1,x2,x3,y1,y2,y3,h1,h2,h3):
    maat = np.array([[1,x1,y1],[1,x2,y2],[1,x3,y3]],dtype=np.float64) 
    rhhs = np.array([h1,h2,h3],dtype=np.float64)
    sool = np.linalg.solve(maat,rhhs)
    dh_dx=sool[1]
    dh_dy=sool[2]
    slope=np.sqrt(dh_dx**2+dh_dy**2)
    return dh_dx,dh_dy,slope

#--------------------------------------------------------------------------------------------------

def uplift(exp,xq,yq):
    if exp==-1:
       val=0
    elif exp==0:
       val=w_uplift
    elif exp==1:
       val=w_uplift
    elif exp==2:
       val=w_uplift
    elif exp==3:
       val=0
    elif exp==4:
       r=np.sqrt((xq-Lx/2)**2+(yq-Ly/2)**2)
       if r<Lx/2:
          val=(1-r/(Lx/2))* 1e-2/year
       else:
          val=0 
    elif exp==5:
       val=w_uplift*xq/Lx*((1-np.cos(2*np.pi*xq/Lx))/2)**2
    elif exp==6:
       if xq<Lx/2-thickness:
          val=0
       elif xq<Lx/2+thickness:
          val=w_uplift
       else:
          val=0 
    else:
       exit('pb with exp in uplift function')
    return val 

#--------------------------------------------------------------------------------------------------

def tectonic_velocity(exp,xq,yq):
    if exp==-1:
       return 0,0
    elif exp==0:
       return 0,0
    elif exp==1:
       return 0,0
    elif exp==2:
       return 0,0
    elif exp==3:
       return 0,0
    elif exp==4:
       return 0,0
    elif exp==5:
       return 0,0
    elif exp==6:
       if xq<Lx/2-thickness:
          return 0,0
       elif xq<Lx/2+thickness:
          return u_tecto/1.414,0
       else:
          return u_tecto,0
    else:
       exit('pb with exp in tectonic_velocity function')

#--------------------------------------------------------------------------------------------------

def med(a, b, c):
    return sorted([a, b, c])[1]

###################################################################################################
###################################################################################################

zscale=1

lumping=True

unstructured=True
   
tol = 1e-3    # iteration tolerance

build_gnei='2b'

#------------------------------------------------------------------------------

#experiment=-1 #test
#experiment=0 # simpson's book example
#experiment=1 # simple slope high at x=Lx
#experiment=2 # benchmark uplift 
#experiment=3 # benchmark linear diffusion
#experiment=4 # volcano-like
experiment=5 # orogen-like
#experiment=6 # crustal orogen with tect adv

if experiment==-1: #test
   Lx=50e3
   Ly=50e3
   rexp = 2       # drainage area exponent
   mexp = 1       # slope exponent
   w_uplift = 1e-3/year  # rock uplift rate (m/s)
   c = 1e-11/year # fluvial erosion coeff. (mˆ(2-2rexp)/s)
   c0 = 0.1/year  # linear diffusion coeff.(mˆ2/s)
   Ac = 0         # critical drainage area (mˆ2)
   scale = 10     # amplitude random initial topography (m)
   nstep = 1  # number of time steps
   dt = 0*year # time step (s)
   #target_area='60000000' #ref
   target_area='4500000' 
   every=1


if experiment==0:
   Lx=50e3
   Ly=50e3
   rexp = 2       # drainage area exponent
   mexp = 1       # slope exponent
   w_uplift = 1e-3/year  # rock uplift rate (m/s)
   c = 1e-11/year # fluvial erosion coeff. (mˆ(2-2rexp)/s)
   c0 = 0.1/year  # linear diffusion coeff.(mˆ2/s)
   Ac = 0         # critical drainage area (mˆ2)
   scale = 10     # amplitude random initial topography (m)
   nstep = 3000  # number of time steps
   dt = 100*year # time step (s)
   target_area='4000000'
   every=20

if experiment==1:
   Lx=50e3
   Ly=50e3
   rexp = 2       # drainage area exponent
   mexp = 1       # slope exponent
   w_uplift = 0.1e-3/year  # rock uplift rate (m/s)
   c = 1e-11/year # fluvial erosion coeff. (mˆ(2-2rexp)/s)
   c0 = 0.1/year  # linear diffusion coeff.(mˆ2/s)
   Ac = 0         # critical drainage area (mˆ2)
   scale = 10     # amplitude random initial topography (m)
   nstep = 3000  # number of time steps
   dt = 100*year # time step (s)
   target_area='500000'
   nsegments=80
   every=50

if experiment==2:
   Lx=50e3
   Ly=50e3
   rexp = 2       # drainage area exponent
   mexp = 1       # slope exponent
   w_uplift = 1e-3/year  # rock uplift rate (m/s)
   c = 0.         # fluvial erosion coeff. (mˆ(2-2rexp)/s)
   c0 = 0.        # linear diffusion coeff.(mˆ2/s)
   Ac = 0         # critical drainage area (mˆ2)
   scale = 10     # amplitude random initial topography (m)
   nstep = 100  # number of time steps
   dt = 100*year # time step (s)
   target_area='3500000'
   every=10

if experiment==3:
   Lx=50e3
   Ly=50e3
   rexp = 2       # drainage area exponent
   mexp = 1       # slope exponent
   c = 0.         # fluvial erosion coeff. (mˆ(2-2rexp)/s)
   c0 = 200/year  # linear diffusion coeff.(mˆ2/s)
   Ac = 0         # critical drainage area (mˆ2)
   scale = 0     # amplitude random initial topography (m)
   nstep = 5000  # number of time steps
   dt = 100*year # time step (s)
   target_area='4000000'
   every=100
   every=10

if experiment==4: #volcano-like
   Lx=50e3
   Ly=50e3
   rexp = 2       # drainage area exponent
   mexp = 1       # slope exponent
   c = 1e-12/year # fluvial erosion coeff. (mˆ(2-2rexp)/s)
   c0 = 0.1/year  # linear diffusion coeff.(mˆ2/s)
   Ac = 0         # critical drainage area (mˆ2)
   scale = 1      # amplitude random initial topography (m)
   nstep = 3000   # number of time steps
   dt = 100*year  # time step (s)
   target_area='700000'
   every=10

if experiment==5: # orogen-like
   Lx=200e3
   Ly=100e3
   rexp = 2       # drainage area exponent
   mexp = 1       # slope exponent
   w_uplift = 2e-2/year  # rock uplift rate (m/s)
   c = 1e-13/year # fluvial erosion coeff. (mˆ(2-2rexp)/s)
   c0 = 0.001/year  # linear diffusion coeff.(mˆ2/s)
   Ac = 0         # critical drainage area (mˆ2)
   scale = 0.1     # amplitude random initial topography (m)
   nstep = 3001  # number of time steps
   dt = 100*year # time step (s)
   target_area='1500000'
   nsegments=80
   every=10

if experiment==6: # upper crustal wedge with advection
   Lx=200e3
   Ly=50e3
   rexp = 2       # drainage area exponent
   mexp = 1       # slope exponent
   c = 0 # fluvial erosion coeff. (mˆ(2-2rexp)/s)
   c0 = 0.001/year  # linear diffusion coeff.(mˆ2/s)
   Ac = 0         # critical drainage area (mˆ2)
   scale = 0.1     # amplitude random initial topography (m)
   nstep = 1001  # number of time steps
   dt = 100*year # time step (s)
   target_area='25000000'
   nsegments=80
   every=25
   u_tecto=-0.02/year
   w_uplift=-u_tecto/1.414
   thickness=25e3


print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")
print("experiment=",experiment)
print("Lx=",Lx)
print("Ly=",Ly)

#------------------------------------------------------------------------------
# Gauss quadrature setup
#------------------------------------------------------------------------------

nqel=3

qcoords_r=np.array([1/6,2/3,1/6],dtype=np.float64) 
qcoords_s=np.array([1/6,1/6,2/3],dtype=np.float64) 
qweights =np.array([1/6,1/6,1/6],dtype=np.float64) 

#qcoords_r=np.array([1/2,1/2,0],dtype=np.float64) 
#qcoords_s=np.array([1/2,0,1/2],dtype=np.float64) 
#qweights =np.array([1/6,1/6,1/6],dtype=np.float64) 

#------------------------------------------------------------------------------
# build mesh
#------------------------------------------------------------------------------
start = timing.time()

if unstructured:
   print('   unstructured mesh')
 
   square_vertices = np.array([[0,0],[0,Ly],[Lx,Ly],[Lx,0]])
   square_edges = compute_segs(square_vertices)

   O1 = {'vertices' : square_vertices, 'segments' : square_edges}
   #T1 = tr.triangulate(O1, 'pqa60000000') #for testing
   T1 = tr.triangulate(O1, 'pqa'+target_area) 

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

   print('   -> sum(area)',np.sum(area),'Lx*Ly=',Lx*Ly)

   N=np.size(x)


else:

   print('   structured mesh')

   hmean=np.sqrt(float(target_area))

   print('   -> hmean  =',hmean)

   nelx=int(Lx/hmean)
   nely=int(Ly/hmean)

   print('   -> nelx  =',nelx)
   print('   -> nely  =',nely)

   hx=Lx/nelx
   hy=Lx/nely

   nel=nelx*nely*2
   N=(nelx+1)*(nely+1)

   print('   -> hx  =',hx)
   print('   -> hy  =',hy)

   x = np.empty(N,dtype=np.float64)  # x coordinates
   y = np.empty(N,dtype=np.float64)  # y coordinates

   counter=0
   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           x[counter]=i*Lx/float(nelx)
           y[counter]=j*Ly/float(nely)
           counter += 1

   icon =np.zeros((m,nel),dtype=np.int32)
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           i0=i + j * (nelx + 1)
           i1=i + 1 + j * (nelx + 1)
           i2=i + 1 + (j + 1) * (nelx + 1)
           i3=i + (j + 1) * (nelx + 1)
           nb= random.uniform(0,10)
           if nb<5:
              icon[0,counter]=i0 
              icon[1,counter]=i1 
              icon[2,counter]=i3 
              counter+=1
              icon[0,counter]=i1 
              icon[1,counter]=i2 
              icon[2,counter]=i3 
              counter+=1
           else:
              icon[0,counter]=i0 
              icon[1,counter]=i1 
              icon[2,counter]=i2 
              counter+=1
              icon[0,counter]=i0
              icon[1,counter]=i2 
              icon[2,counter]=i3 
              counter+=1


print('   -> N  =',N)
print('   -> nel=',nel)

print("generate mesh: %.3f s" % (timing.time() - start))
    
dhdx = np.zeros(nel,dtype=np.float64) 
dhdy = np.zeros(nel,dtype=np.float64) 
slope = np.zeros(nel,dtype=np.float64) 

rad = np.zeros(N,dtype=np.float64) 
rad[:]=np.sqrt((x[:]-Lx/2)**2+(y[:]-Ly/2)**2)


#------------------------------------------------------------------------------
# initial topography 
#------------------------------------------------------------------------------
start = timing.time()

z=np.zeros(N,dtype=np.float64) 
z_prev=np.zeros(N,dtype=np.float64) 

#random.seed(12)

if experiment==-1:
   for i in range(0,N):
       z[i]=x[i]/10000+y[i]/11000 

elif experiment==0 or experiment==4 or experiment==5 or experiment==6:
   for i in range(0,N):
       z[i]= random.uniform(0,scale)

elif experiment==1:
   for i in range(0,N):
       z[i]=x[i]/Lx*scale

elif experiment==2 or experiment==3:
   for i in range(0,N):
       z[i]=0

else:
   exit("unknown experiment in initial topography")

#for i in range(0,N):
   #z[i]=x[i]/10000+y[i]/11000 #for testing
   #z[i]=y[i]/Ly/3
   #z[i]=0
   #z[i]=np.cos((x[i]-Lx/2)/Lx*np.pi)*np.cos((y[i]-Ly/2)/Ly*np.pi)*10
   #z[i]=2*x[i]/km+3*y[i]/km

z_prev[:]=z[:]
       
np.savetxt('initial_topography.ascii',np.array([x,y,z]).T)

print("prescribe initial elevation: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# tectonic velocity
#------------------------------------------------------------------------------

u=np.zeros(N,dtype=np.float64) 
v=np.zeros(N,dtype=np.float64) 
w=np.zeros(N,dtype=np.float64) 

for i in range(0,N):
    u[i],v[i]=tectonic_velocity(experiment,x[i],y[i])
    w[i]=uplift(experiment,x[i],y[i])

#------------------------------------------------------------------------------
# compute area of elements with basis functions
#------------------------------------------------------------------------------
start = timing.time()

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
    #end for
#end for

print('   -> sum(area)',np.sum(area),'Lx*Ly=',Lx*Ly)

print("compute element area: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# boundary conditions
#------------------------------------------------------------------------------
start = timing.time()

bc_fix = np.zeros(N,dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(N,dtype=np.float64)  # boundary condition, value

if experiment==-1:

   1

elif experiment==0 or experiment==2 or experiment==4:
   for i in range(0,N):
       if abs(x[i]/Lx)<eps:
          bc_fix[i] = True ; bc_val[i]=0
       if abs(x[i]/Lx-1)<eps:
          bc_fix[i] = True ; bc_val[i]=0
       if abs(y[i]/Ly)<eps:
          bc_fix[i] = True ; bc_val[i]=0
       if abs(y[i]/Ly-1)<eps:
          bc_fix[i] = True ; bc_val[i]=0

elif experiment==1:
   for i in range(0,N):
       if abs(x[i]/Lx)<eps:
          bc_fix[i] = True ; bc_val[i]=0

elif experiment==3:
   for i in range(0,N):
       if abs(x[i]/Lx)<eps:
          bc_fix[i] = True ; bc_val[i]=0
       if abs(x[i]/Lx-1)<eps:
          bc_fix[i] = True ; bc_val[i]=100

elif experiment==5 or experiment==6:
   for i in range(0,N):
       if abs(x[i]/Lx)<eps:
          bc_fix[i] = True ; bc_val[i]=0
       if abs(x[i]/Lx-1)<eps:
          bc_fix[i] = True ; bc_val[i]=0

else:

   exit('unknown experiment in boundary conditions')

print("define boundary conditions: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# step 2b: same as step 2, but much much faster
#------------------------------------------------------------------------------
start = timing.time()

if build_gnei=='2b':

   edges = np.empty((N,N,2),dtype=np.int32) ; edges[:,:,:]=-1

   for iel in range(0,nel):
       min_vertex = min(icon[0,iel],icon[1,iel],icon[2,iel]) #edge 0
       mid_vertex = med(icon[0,iel],icon[1,iel],icon[2,iel]) #edge 1
       max_vertex = max(icon[0,iel],icon[1,iel],icon[2,iel]) #edge 2

       #edge 0
       if edges[min_vertex,mid_vertex,0]<0:
          edges[min_vertex,mid_vertex,0]=iel
       else:
          edges[min_vertex,mid_vertex,1]=iel

       #edge 1
       if edges[mid_vertex,max_vertex,0]<0:
          edges[mid_vertex,max_vertex,0]=iel
       else:
          edges[mid_vertex,max_vertex,1]=iel

       #edge 2
       if edges[min_vertex,max_vertex,0]<0:
          edges[min_vertex,max_vertex,0]=iel
       else:
          edges[min_vertex,max_vertex,1]=iel


   gnei = np.zeros((3,nel),dtype=np.int32) ; gnei[:]=-1

   for iel in range(0,nel):
       min_vertex = min(icon[0,iel],icon[1,iel],icon[2,iel]) #edge 0
       mid_vertex = med(icon[0,iel],icon[1,iel],icon[2,iel]) #edge 1
       max_vertex = max(icon[0,iel],icon[1,iel],icon[2,iel]) #edge 2

       #edge 0 - min-mid we assume the edge has two neighbours
       el1=edges[min_vertex,mid_vertex,0] 
       el2=edges[min_vertex,mid_vertex,1] 
       if el2<0: el2=iel

       if iel==el1:
          gnei[0,el1]=el2
       else:
          gnei[0,el2]=el1

       #edge 1 - mid-max we assume the edge has two neighbours
       el1=edges[mid_vertex,max_vertex,0]
       el2=edges[mid_vertex,max_vertex,1]
       if el2<0: el2=iel

       if iel==el1:
          gnei[1,el1]=el2
       else:
          gnei[1,el2]=el1

       #edge 2 - min-max we assume the edge has two neighbours
       el1=edges[min_vertex,max_vertex,0]
       el2=edges[min_vertex,max_vertex,1]
       if el2<0: el2=iel

       if iel==el1:
          gnei[2,el1]=el2
       else:
          gnei[2,el2]=el1

   if experiment==-1:
      print(gnei[0,0:15],gnei[0,-13:])
      print(gnei[1,0:15],gnei[1,-13:])
      print(gnei[2,0:15],gnei[2,-13:])

   print("compute gnei 2b: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# step 2a: same as step 2, but about twice as fast and more compact.
#------------------------------------------------------------------------------
start = timing.time()

if build_gnei=='2a':

   faces=np.zeros((3,2,nel),dtype=np.int32) # 3 face per elt, each face 2 nodes

   for iel in range(0,nel):
       faces[0,0,iel]=icon[0,iel]
       faces[0,1,iel]=icon[1,iel]
       faces[1,0,iel]=icon[1,iel]
       faces[1,1,iel]=icon[2,iel]
       faces[2,0,iel]=icon[2,iel]
       faces[2,1,iel]=icon[0,iel]

   gnei = np.zeros((3,nel),dtype=np.int32) ; gnei[:]=-1

   for iel in range(0,nel):
       if iel%200==0: print('iel=',iel)
       found=np.zeros(3,dtype=np.bool) 
       for iface in range(0,3):
           if gnei[iface,iel]>=0: continue
           for jel in range(0,nel):
               if jel==iel: continue
               for jface in range(0,3):
                   if faces[iface,0,iel]==faces[jface,1,jel] and \
                      faces[iface,1,iel]==faces[jface,0,jel]: 
                      gnei[iface,iel]=jel
                      gnei[jface,jel]=iel
                      found[iface]=True
                      break 
                   #end if
               #end for jface
               if found[iface]: break
           #end for jel
           if not found[iface]: gnei[iface,iel]=iel 
       #end for
       #print('element behind face 0 of element',iel,'is element',gnei[0,iel])
       #print('element behind face 1 of element',iel,'is element',gnei[1,iel])
       #print('element behind face 2 of element',iel,'is element',gnei[2,iel])
   #end for

   if experiment==-1:
      print(gnei[0,0:15],gnei[0,-13:])
      print(gnei[1,0:15],gnei[1,-13:])
      print(gnei[2,0:15],gnei[2,-13:])

   print("compute gnei 2a: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# compute gnei (step 2)
# Find and save the three elements adjacent to each element. Elements on
# the boundary that have only two neighbors are assigned their own element 
# index to the missing third neighbor.
# face0: nodes 0-1, face1: nodes 1-2, face2: nodes 2-0
# iel_node1,2 are nodes making given face of element iel
# jel_node1,2 are nodes making given face of element jel
#------------------------------------------------------------------------------
start = timing.time()

if build_gnei=='2':

   gnei = np.zeros((3,nel),dtype=np.int32) 

   for iel in range(0,nel):
       if iel%200==0: print('iel=',iel)
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

   if experiment==-1:
      print(gnei[0,0:15],gnei[0,-13:])
      print(gnei[1,0:15],gnei[1,-13:])
      print(gnei[2,0:15],gnei[2,-13:])

   print("compute gnei 2 : %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# establish list of elements on border of domain 
#------------------------------------------------------------------------------
start = timing.time()
    
border_element = np.zeros(nel,dtype=np.int16)

for iel in range(nel):
    if abs(x[icon[0,iel]])/Lx<eps and abs(x[icon[1,iel]])/Lx<eps:
       border_element[iel]=1
    if abs(x[icon[1,iel]])/Lx<eps and abs(x[icon[2,iel]])/Lx<eps:
       border_element[iel]=1
    if abs(x[icon[2,iel]])/Lx<eps and abs(x[icon[0,iel]])/Lx<eps:
       border_element[iel]=1
    if abs(x[icon[0,iel]]-Lx)/Lx<eps and abs(x[icon[1,iel]]-Lx)/Lx<eps:
       border_element[iel]=1
    if abs(x[icon[1,iel]]-Lx)/Lx<eps and abs(x[icon[2,iel]]-Lx)/Lx<eps:
       border_element[iel]=1
    if abs(x[icon[2,iel]]-Lx)/Lx<eps and abs(x[icon[0,iel]]-Lx)/Lx<eps:
       border_element[iel]=1
    if abs(y[icon[0,iel]])/Ly<eps and abs(y[icon[1,iel]])/Ly<eps:
       border_element[iel]=1
    if abs(y[icon[1,iel]])/Ly<eps and abs(y[icon[2,iel]])/Ly<eps:
       border_element[iel]=1
    if abs(y[icon[2,iel]])/Ly<eps and abs(y[icon[0,iel]])/Ly<eps:
       border_element[iel]=1
    if abs(y[icon[0,iel]]-Ly)/Ly<eps and abs(y[icon[1,iel]]-Ly)/Ly<eps:
       border_element[iel]=1
    if abs(y[icon[1,iel]]-Ly)/Ly<eps and abs(y[icon[2,iel]]-Ly)/Ly<eps:
       border_element[iel]=1
    if abs(y[icon[2,iel]]-Ly)/Ly<eps and abs(y[icon[0,iel]]-Ly)/Ly<eps:
       border_element[iel]=1

nborder=np.count_nonzero(border_element==1)

print('   -> nborder=',nborder)

print("establish border elts: %.3f s" % (timing.time() - start))

###################################################################################################
###################################################################################################
# time stepping 
###################################################################################################
###################################################################################################

time=0

for istep in range(0,nstep):

    print('=============== istep=',istep,'==============')

    time+=dt

    #------------------------------------------------------------------------------
    # compute elevation of middle of element zc 
    #------------------------------------------------------------------------------
    start = timing.time()

    xc = np.zeros(nel,dtype=np.float64)
    yc = np.zeros(nel,dtype=np.float64)
    zc = np.zeros(nel,dtype=np.float64)
    for iel in range(0,nel):
        xc[iel]=(x[icon[0,iel]]+x[icon[1,iel]]+x[icon[2,iel]])/3
        yc[iel]=(y[icon[0,iel]]+y[icon[1,iel]]+y[icon[2,iel]])/3
        zc[iel]=(z[icon[0,iel]]+z[icon[1,iel]]+z[icon[2,iel]])/3

    print("compute xc,yc,zc: %.3f s" % (timing.time() - start))

    #------------------------------------------------------------------------------
    # compute slope
    #------------------------------------------------------------------------------
    start = timing.time()

    for iel in range(0,nel):
        dhdx[iel],dhdy[iel],slope[iel]=compute_element_slope(x[icon[0,iel]],x[icon[1,iel]],x[icon[2,iel]],\
                                                             y[icon[0,iel]],y[icon[1,iel]],y[icon[2,iel]],\
                                                             z[icon[0,iel]],z[icon[1,iel]],z[icon[2,iel]])

    print("compute slope: %.3f s" % (timing.time() - start))

    #------------------------------------------------------------------------------
    # step4: Sort the average element elevation for the entire mesh from highest 
    # to lowest. The indices of the sorted elevations are saved in the array sorted indices. 
    #------------------------------------------------------------------------------
    start = timing.time()

    sorted_indices=zc.argsort()
    sorted_indices=np.flip(sorted_indices[:])

    print("compute sorted_indices: %.3f s" % (timing.time() - start))

    #------------------------------------------------------------------------------
    # step 5: Using the sorted indices, sort the neighboring elements, the result of 
    # which is saved in sorted gnei.
    # At this stage, all elements in the mesh have been ordered, along with their 
    # three adjacent neighbors, from highest to lowest.
    #------------------------------------------------------------------------------
    #sorted_gnei = np.zeros((3,nel),dtype=np.int32) 
    #sorted_gnei[:,:] = gnei[:,sorted_indices]
    #print(sorted_gnei)

    #------------------------------------------------------------------------------
    # step 6: For each element, find and save the local index of the lowest 
    # of the three adjacent elements. The result is 0, 1, or 2
    #------------------------------------------------------------------------------
    start = timing.time()

    min_index = np.zeros(nel,dtype=np.int32) 

    for iel in range(0,nel):
        min_index[iel]=-1
        zc_0=zc[gnei[0,iel]]
        zc_1=zc[gnei[1,iel]]
        zc_2=zc[gnei[2,iel]]
        if zc_0<zc_1 and zc_0<zc_2 and zc[iel]>zc_0: 
           min_index[iel]=0
        if zc_1<zc_0 and zc_1<zc_2 and zc[iel]>zc_1: 
           min_index[iel]=1
        if zc_2<zc_0 and zc_2<zc_1 and zc[iel]>zc_2: 
           min_index[iel]=2

        #if iel==16:
        #   print('iel=16',zc_0,zc_1,zc_2,'  ',zc[iel])
        #   print(min_index[iel],gnei[min_index[iel],iel])
        #if iel==134:
        #   print('iel=134',zc_0,zc_1,zc_2,'  ',zc[iel])
        #   print(min_index[iel],gnei[min_index[iel],iel])
        #   exit()

        #print('I am element',iel,'and I give to element',min_index[iel],gnei[min_index[iel],iel])

    print("compute min_index: %.3f s" % (timing.time() - start))

    #------------------------------------------------------------------------------
    # step 7+8: In a loop over all ordered elements from highest to lowest, “pass” 
    # the accumulated drainage area from each (donor) element to its lowest adjacent 
    # neighbor (receiver). The result after the loop has been completed is the 
    # accumulated surface area that “drains” to each element in the landscape, A. 
    # Note that this procedure implicitly assumes a spatially uniform rainfall, 
    # which can easily be accounted for if desired.
    #receiver = sorted_gnei[min_index[iel],iel] suspicious line in book?!
    #------------------------------------------------------------------------------
    start = timing.time()

    A = np.zeros(nel,dtype=np.float64) 

    A[:]=area[:]

    for iel in range(0,nel):
       donor = sorted_indices[iel] 
       #if min_index[donor]>0: #if there is a lower neighbour 
       receiver=gnei[min_index[donor],donor]
       #print('loop index ',iel,'is donor',donor,'gives to ',receiver)
       A[receiver] += A[donor] 

    print("compute drainage area: %.3f s" % (timing.time() - start))

    if istep%every==0:
       export_network_to_vtu(x,y,z,xc,yc,zc,A,icon,min_index,'network_'+str(istep)+'.vtu')

    ###################################################
    start = timing.time()
    catchment = np.zeros(nel,dtype=np.float64) 
    catchment[:]=-1

    if istep%every==0:
       counter=0
       for iel in range(0,nel):
           if border_element[iel]==1:
              catchment[iel]=counter
              counter+=1


       #for iel in range(0,5):
       #    print('---------------')
       #    donor = sorted_indices[iel] 
       #    river=[]
       #    for i in range(0,10):
       #        receiver=gnei[min_index[donor],donor]
       #        print(donor,receiver)
       #        if receiver==donor:
       #           print('dead end')
       #           break
       #        #if catchment[receiver]>0:
       #        #   print('reached')
       #        #   break
       #        #river.append(receiver)
       #        donor=receiver
       #    #print(river)
       #exit()

       #ncmem=0
       #for it in range(0,1):
       #    for iel in range(0,nel):
       #        if catchment[iel]>-1:
       #           for jel in range(0,nel):
       #               if gnei[min_index[jel],jel]==iel:
       #                  catchment[jel]=catchment[iel]
       #    nc=np.count_nonzero(catchment!=-1)
       #    print('   -> ',it,nc)
       #    if ncmem==nc: 
       #       print('   -> conv!')
       #       break
       #    ncmem=nc

    print("compute catchments: %.3f s" % (timing.time() - start))

    #------------------------------------------------------------------------------
    # build FEM matrix and rhs
    #------------------------------------------------------------------------------
    start = timing.time()

    A_mat = np.zeros((N,N),dtype=np.float64)    # FE matrix 
    B_mat = np.zeros((ndim,m),dtype=np.float64) # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)    # shape functions
    dNNNdr = np.zeros(m,dtype=np.float64)       # shape functions derivatives
    dNNNds = np.zeros(m,dtype=np.float64)       # shape functions derivatives
    dNNNdx = np.zeros(m,dtype=np.float64)       # shape functions derivatives
    dNNNdy = np.zeros(m,dtype=np.float64)       # shape functions derivatives
    rhs   = np.zeros(N,dtype=np.float64)        # FE rhs 
    hvect = np.zeros(m,dtype=np.float64)   
    kappa = np.zeros(nel,dtype=np.float64)   

    for iel in range (0,nel):

        b_el=np.zeros(m,dtype=np.float64)
        a_el=np.zeros((m,m),dtype=np.float64)
        f_el=np.zeros(m,dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m):
            hvect[k]=z[icon[k,iel]]

        for kq in range (0,nqel):

            # position & weight of quad. point
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            N_mat[0:m,0]=NN(rq,sq)
            dNNNdr[0:m]=dNNdr(rq,sq)
            dNNNds[0:m]=dNNds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNNNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNNNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNNNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNNNds[k]*y[icon[k,iel]]
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)

            for k in range(0,m):
                dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
                dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]
                B_mat[0,k]=dNNNdx[k]
                B_mat[1,k]=dNNNdy[k]

            xq=0
            yq=0
            vel[0,0]=0.
            vel[0,1]=0.
            for k in range(0,m):
                xq+=N_mat[k,0]*x[icon[k,iel]]
                yq+=N_mat[k,0]*y[icon[k,iel]]
                vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]

            if A[iel]>Ac:
               kappa[iel]=c0+c*(A[iel]-Ac)**rexp * slope[iel]**(mexp-1)
            else:
               kappa[iel]=c0
            #kappa[iel]=0

            MM+=N_mat.dot(N_mat.T)*weightq*jcob

            Ka+=N_mat.dot(vel.dot(B_mat))*weightq*jcob

            Kd+=B_mat.T.dot(B_mat)*weightq*jcob*kappa[iel]

            f_el+=N_mat[:,0]*uplift(experiment,xq,yq)*weightq*jcob

        # end for kq

        if lumping:
           MM[0,0]+=MM[0,1]+MM[0,2] ; MM[0,1]=0 ; MM[0,2]=0
           MM[1,1]+=MM[1,0]+MM[1,2] ; MM[1,0]=0 ; MM[1,2]=0
           MM[2,2]+=MM[2,0]+MM[2,1] ; MM[2,0]=0 ; MM[2,1]=0

        a_el=MM+(Kd+Ka)*dt

        b_el=MM.dot(hvect)+f_el*dt

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon[k1,iel]
            if bc_fix[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_val[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               # end for k2
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val[m1]
            # end if
        # end for k1

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,m):
            m1=icon[k1,iel]
            for k2 in range(0,m):
                m2=icon[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            # end for k2
            rhs[m1]+=b_el[k1]
        # end for k1

    # end for iel

    print('   -> A (m/M)',np.min(A_mat),np.max(A_mat))
    print('   -> b (m/M)',np.min(rhs),np.max(rhs))

    print("build matrix: %.3f s" % (timing.time() - start))

    #------------------------------------------------------------------------------
    # solve system
    #------------------------------------------------------------------------------
    start = timing.time()

    z[:] = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print('   -> z (m/M)',np.min(z),np.max(z))

    if istep%every==0:
       np.savetxt('solution_'+str(istep)+'.ascii',np.array([x,y,z,rad]).T)

    print("solve FEM system: %.3f s" % (timing.time() - start))

    #------------------------------------------------------------------------------
    # export min/max profiles
    #------------------------------------------------------------------------------

    if experiment==1 or experiment==5 or experiment==6:
       start = timing.time()

       dx=Lx/nsegments
       xprofile=np.zeros(nsegments,dtype=np.float64) 
       zprofile_min=np.zeros(nsegments,dtype=np.float64) 
       zprofile_max=np.zeros(nsegments,dtype=np.float64) 
       zprofile_min[:]=+1e50
       zprofile_max[:]=-1e50
       for i in range(0,N):
           iindex=int(min(x[i],0.99999*Lx)/dx)
           #print(iindex)       
           zprofile_min[iindex]=min(z[i],zprofile_min[iindex])
           zprofile_max[iindex]=max(z[i],zprofile_max[iindex])
       for i in range(0,nsegments):
           xprofile[i]=(i+0.5)*dx
       if istep%every==0:
          np.savetxt('profile_x_'+str(istep)+'.ascii',np.array([xprofile,zprofile_min,zprofile_max]).T)

       print("compute orogen profiles: %.3f s" % (timing.time() - start))

    #------------------------------------------------------------------------------
    # export to vtu
    #------------------------------------------------------------------------------
    start = timing.time()

    dz=z-z_prev

    if istep%every==0:
       export_elements_to_vtu(x,y,z,u,v,w,zc,A,sorted_indices,dhdx,dhdy,slope,kappa,icon,\
                              'solution_'+str(istep)+'.vtu',area,border_element,catchment,dz)

    print("export to vtu: %.3f s" % (timing.time() - start))


    z_prev[:]=z[:]


#end for istep

print("-----------------------------")
print("-----------------------------")
