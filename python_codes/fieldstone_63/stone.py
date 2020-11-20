import numpy as np
import sys as sys
import scipy
import time 
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= (1.-rq-sq)
    NV_1= rq
    NV_2= sq
    return NV_0,NV_1,NV_2

def dNNVdr(rq,sq):
    dNVdr_0= -1. 
    dNVdr_1= +1.
    dNVdr_2=  0.
    return dNVdr_0,dNVdr_1,dNVdr_2

def dNNVds(rq,sq):
    dNVds_0= -1. 
    dNVds_1=  0.
    dNVds_2= +1.
    return dNVds_0,dNVds_1,dNVds_2

#------------------------------------------------------------------------------

def get_area_cement(R,a_fac,h_fac):
    if squarification == True:
        h_dash = R*h_fac
        val = 1
    else:
        h_dash = R*(h_fac+(1-np.sqrt(1-a_fac**2)))
        area1 = a_fac*R**2*(h_fac+(1-np.sqrt(1-a_fac**2)))
        area2 = 0.5*R**2*a_fac*(1-np.sqrt(1-a_fac**2))
        area = area1-area2
        val = area/(R**2*a_fac*h_fac)
    print('h_dash (normalized) = ',h_dash/(R*h_fac))
    print('cement area (normalized) = ',val)
    return val
        
#------------------------------------------------------------------------------

def export_mesh(NV,nel,x,y,icon,name):
    vtufile=open(name,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
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
        vtufile.write("%d \n" %((iel+1)*3))
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
    print('     -> created '+name)

#------------------------------------------------------------------------------
print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")
#------------------------------------------------------------------------------

MPa=1e6

######################################################
#parameters
######################################################

nLayers = 101   # must be odd !
nel_h=10

#grain geometrical properties
outer_radius=9e-5  #radius (m)  
a_fac = 20/90      # ratio of a to R 
h_fac = 5/90       # ratio of h to R
a = outer_radius*a_fac
h = outer_radius*h_fac

#Quartz mechanical properties
E1 = 95e9                #Young's modulus (Pa)
nu1=0.08                 # poisson ratio
mu1= E1/(2*(1+nu1))      # shear modulus
lambdaa1=2.*mu1*nu1/(1.-2.*nu1)  

##cement mechanical properties
E2 = 84e9                #Young's modulus (Pa)
nu2=0.31                 # poisson ratio
mu2= E2/(2*(1+nu2))      # shear modulus
lambdaa2=2.*mu2*nu2/(1.-2.*nu2)  

gx=0.  # gravity vector x-component
gy=0.  # gravity vector y-component
visu=1
squarification  = True
    
t_bc = 1e8  # traction at contact (Pa)

    
#nb of eighths of disk
nsection=2
nel = nsection * nLayers * nLayers                  # number of elements per disk
nel_minus = nsection * (nLayers-1) * (nLayers-1)    # number of elements per disk
NV  = (nLayers+1)**2                                # number of mesh nodes per disk
NV_minus  = nLayers**2  # number of mesh nodes per disk when outer layer is not considered

qcoords_r=[1./6.,1./6.,2./3.] # coordinates & weights 
qcoords_s=[2./3.,1./6.,1./6.] # of quadrature points
qweights =[1./6.,1./6.,1./6.]

m=3
nqel=3
ndof=2
ndim=2

eps=1e-9

axisymmetric=False

###################################################################
#area_norm = get_area_cement(outer_radius,a_fac,h_fac)

print ('nLayers  =',nLayers)
print ('NV       =',NV)
print ('nel      =',nel)
        
#np.savetxt('info_grain_prop.ascii',np.array([outer_radius,a_fac,h_fac,E1,nu1,rho1,E2,mu2,rho2]).T \
#           ,header='# diameter,a_fac,h_fac,E_qtz,nu_qtz,density_qtz,E_cement,mu_cement,density_cement')

print("-----------------------------")

#################################################################
# grid point setup
#################################################################
start_grid = time.time()
start_all = time.time()

x=np.zeros(NV,dtype=np.float64)          # x coordinates
y=np.zeros(NV,dtype=np.float64)          # y coordinates
outer_node = np.zeros(NV, dtype=np.bool) # on the outer hull yes/no 
contact_node = np.zeros(NV, dtype=np.bool) # on the outer hull yes/no 
contact_iNV=np.zeros(NV,dtype=np.float64)   
counter=0
counter_contact=0

#create coordinates x,y of nodes of disk centered at 0,0 
x[counter]=0#xc
y[counter]=0#yc
counter += 1 
for iLayer in range(1,nLayers+1):
    radius = outer_radius * float(iLayer)/float(nLayers)
    nPointsOnCircle = 2*iLayer+1
    for iPoint in range (0,nPointsOnCircle):
        # Coordinates are created, starting at twelve o'clock, 
        # going in clockwise direction
        x[counter] =  radius * np.cos(0.5 * np.pi * float(iPoint) / float(nPointsOnCircle-1)) #+ xc
        y[counter] =  -radius * np.sin(0.5 * np.pi * float(iPoint) / float(nPointsOnCircle-1)) #+ yc
        #print(x[counter],y[counter])
        if iLayer==nLayers:
            outer_node[counter]=True
            if x[counter] < outer_radius*a_fac:
                contact_node[counter]=True
                contact_iNV[counter_contact] = counter
                counter_contact += 1
            #end if
        #end if
        counter += 1 
    #end for 
#end for 
contact_iNV = contact_iNV[0:counter_contact]
contact_iNV = contact_iNV[::-1]

#####################
#apply squarification
#####################
if squarification == True:
    theta = np.arcsin(a_fac)
    ncontactnodes = 0
    counter = 0
    x_contact = np.zeros(NV, dtype=np.float64)  
    y_contact = np.zeros(NV, dtype=np.float64)
    for i in range(0,NV): #loop for all nodes
       if abs(x[i])<eps : #The node is on x- or y-axis
          fac_shrink = np.cos(theta)
          x[i] = fac_shrink * x[i]
          y[i] = fac_shrink * y[i]  
          if i >= NV_minus and i < NV:
              x_contact[counter] = x[i]
              y_contact[counter] = y[i]
              counter += 1
          #endif
       elif abs(y[i]) > eps and abs(x[i]/y[i]) < np.tan(theta):    #The node near y axis
           theta2 = np.arctan(abs(x[i]/y[i]))
           fac_shrink = np.cos(theta)/np.cos(theta2)
           x[i] = fac_shrink * x[i]
           y[i] = fac_shrink * y[i]
           if i >= NV_minus and i < NV:
               x_contact[counter] = x[i]
               y_contact[counter] = y[i]
               counter += 1
           #endif
       #end if
    #end for i
    ncontactnodes = int(counter)
#end if

##############################################
# grid point setup for cement centered at 0,0
##############################################

num = int(counter_contact) #number of nodes that fit within length a
num_h = nel_h +1
NV_cement = num * num_h
nel_cement = 2*(num-1)*(num_h-1)
x_cement = np.zeros(NV_cement, dtype=np.float64)
y_cement = np.zeros(NV_cement, dtype=np.float64)
contact_iNV_cement = np.zeros(NV_cement, dtype=np.float64)
for i in range(0,num):
    j = int(contact_iNV[i])
    x_cement[i] = x[j]
#end for

counter = 0
for i in range(0,num_h):
    x_cement[i*num:(i+1)*num] = x_cement[0:num]
    y_cement[i*num:(i+1)*num] = i /nel_h * h_fac * outer_radius
#end for

if squarification == False:
    #######
    #modify shape of cement
    #######
    mag_fac = np.zeros(NV_cement, dtype=np.float64)
    counter = 0
    for i in range(0,num):
        j = int(contact_iNV[i])
        mag_fac[i] = (y[j]+outer_radius*(1+h_fac))/(outer_radius*h_fac)
    #end for
        
    for i in range(0,num_h):
        mag_fac[i*num:(i+1)*num] = mag_fac[0:num]
    #end for
           
    for i in range(0,NV_cement):
        y_cement[i] = mag_fac[i] * y_cement[i]
    #end for
#end if squarification == False
    
##shift cement to its right position
if squarification == False:
    y_cement = y_cement - (1 + h_fac)*outer_radius
elif squarification == True:
    y_cement = y_cement - (np.cos(theta) + h_fac)*outer_radius
               
print('NV_cement = ',NV_cement)
print('nel cement = ',nel_cement)
print("-----------------------------")

###################
##compute cement iNV
###################
contact_iNV_cement = np.zeros(num, dtype=np.float64)
counter = (num_h-1)*num
for i in range(0,num):
    contact_iNV_cement[i] = counter
    counter = counter + 1
#end for
    
###############################################################################
#combine all the node information into a single vector
###############################################################################
NV_new = NV+NV_cement
x_new = np.zeros(NV_new, dtype=np.float64) 
y_new = np.zeros(NV_new, dtype=np.float64)
x_new[0:NV] = x
x_new[NV:NV_new] = x_cement
y_new[0:NV] = y
y_new[NV:NV_new] = y_cement
len_iNV = len(contact_iNV) + len(contact_iNV_cement)
contact_iNV_new = np.zeros(len_iNV,dtype=np.float) 
contact_iNV_new[0:len(contact_iNV)] = contact_iNV
contact_iNV_new[len(contact_iNV):len_iNV] = contact_iNV_cement + NV
outer_node_new = np.zeros(NV_new, dtype=np.bool) 
outer_node_new[0:NV] = outer_node
 
x = x_new
y = y_new
contact_iNV = contact_iNV_new
outer_node = outer_node_new

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start_grid))

#################################################################
# build connectivity of single disk
#################################################################
start_connectivity = time.time()

icon=np.zeros((m,nel),dtype=np.int32)
outer_elem = np.zeros(nel, dtype=np.bool)

iInner = 1 
iOuter = nsection+2 

for i in range (0,nsection):
   icon[:,i] = (0,2+i,1+i)
#end for

storedElems = nsection 

for iLayer in range(2,nLayers+1):  
    nPointsOnCircle = nsection*iLayer+1
    iInner = (iLayer-1)**2
    iOuter = iLayer**2
    for iSection in range (0,nsection):     
        for iBlock in range(0,iLayer-1):
            icon[:,storedElems] = (iInner, iOuter +1, iOuter )
            if iLayer == nLayers:
                outer_elem[storedElems] = True
            #end if
            storedElems = storedElems + 1    
            icon[:,storedElems] = (iInner, iInner+1, iOuter + 1) 
            if iLayer == nLayers:
                outer_elem[storedElems] = True
            #end if
            storedElems = storedElems + 1 
            iInner += 1 
            iOuter += 1 
        #end for
        icon[:,storedElems] = (iInner, iOuter+1, iOuter )
        if iLayer == nLayers:
            outer_elem[storedElems] = True
        #end if
        storedElems = storedElems + 1 
        iOuter = iOuter + 1 
    #end for
    
###############################
#build connectivity for cement
###############################
icon_cement=np.zeros((m,nel_cement),dtype=np.int32)
      
iInner = 0 
iOuter = num
    
storedElems = 0 
    
for iLayer in range(0,num_h-1):
    icon_cement[:,storedElems] = (iInner, iInner+1, iOuter)
    storedElems = storedElems + 1
    iInner += 1 
    iOuter += 1 
    for iBlock in range(0,num-2):
        icon_cement[:,storedElems] = (iInner, iOuter, iOuter-1)
        storedElems = storedElems + 1    
        icon_cement[:,storedElems] = (iInner, iInner+1, iOuter) 
        storedElems = storedElems + 1 
        iInner += 1 
        iOuter += 1 
    #end for
    icon_cement[:,storedElems] = (iInner, iOuter, iOuter-1)
    storedElems = storedElems + 1 
    iInner += 1
    iOuter += 1 
#end for
    
###############################################################################
#combine all the connectivity arrays information into a single matrix
#and compute element type. 1 for quartz, 2 for cement 
###############################################################################
nel_new = nel+nel_cement
icon_new = np.zeros((m,nel_new),dtype=np.int32) 
icon_new[0:3,0:nel] = icon
icon_new[0:3,nel:nel_new] = icon_cement + NV
outer_elem_new = np.zeros(nel_new, dtype=np.bool)
outer_elem_new[0:nel] = outer_elem
    
el_type = np.ones(nel_new, dtype=np.int)
el_type[nel:nel_new] = 2  #2 for cement

mu = np.zeros(nel_new, dtype=np.float64)
lambdaa = np.zeros(nel_new, dtype=np.float64)

mu[0:nel] = mu1  
mu[nel:nel_new] = mu2 
lambdaa[0:nel] = lambdaa1  
lambdaa[nel:nel_new] = lambdaa2 
    
icon = icon_new
outer_elem = outer_elem_new
NV = NV_new
nel = nel_new

print("-----------------------------")
print ('new NV (before merge)  =',NV)
print ('new nel (before merge) =',nel)
print("-----------------------------")     

print("setup: disk connectivity: %.3f s" % (time.time() - start_connectivity))

#export_mesh(NV,nel,x,y,icon,'mesh_before_merge.vtu')

#################################################################
# merging all disks into one mesh, removing duplicate nodes 
#################################################################
start = time.time()

doubble = np.zeros(NV, dtype=np.bool)
pointto = np.zeros(NV, dtype=np.int32) 

for ip in range(0,NV):
    pointto[ip]=ip
#end for

counter = 0    
for ip in range(0,len(contact_iNV)):
    num_i = int(contact_iNV[ip])
    for jp in range (0,ip):
        num_j = int(contact_iNV[jp])
        if abs(x[num_i]-x[num_j])<eps and abs(y[num_i]-y[num_j])<eps:
           doubble[num_i]=True
           pointto[num_i]=num_j
           #print(x[ip],y[ip])
           counter+=1
        #end if
    #end for
#end for
#print (counter,np.count_nonzero(doubble))

NV_new=NV-num

xnew=np.zeros(NV_new,dtype=np.float64) # x coordinates
ynew=np.zeros(NV_new,dtype=np.float64) # y coordinates
counter=0
for ip in range(0,NV):
   if not doubble[ip]: 
      xnew[counter]=x[ip]
      ynew[counter]=y[ip]
      counter=counter+1
   #end if
#end do

#np.savetxt('grid_new.ascii',np.array([xnew,ynew]).T,header='# x,y')

for iel in range(0,nel):
    for i in range (0,m):
        icon[i,iel]=pointto[icon[i,iel]]
    #end for
#end for

compact = np.zeros(NV, dtype=np.int32) 
counter=0
for ip in range(0,NV):
   if not doubble[ip]: 
      compact[ip]=counter
      counter=counter+1
   #end if
#end for

for iel in range(0,nel):
    for i in range (0,m):
        icon[i,iel]=compact[icon[i,iel]]
    #end for
#end for

print("setup: merging all sub-meshes: %.3f s" % (time.time() - start))

#export_mesh(NV_new,nel,xnew,ynew,icon,'mesh_after_merge.vtu')

#################################################################
# resize x,y to receive xnew,ynew
#################################################################

NV=NV_new

x=np.zeros(NV,dtype=np.float64)          # x coordinates
y=np.zeros(NV,dtype=np.float64)          # y coordinates

x[:]=xnew[:]
y[:]=ynew[:]

Nfem=NV*ndof  

print("-----------------------------")
print ('new NV (after merge)  =',NV_new)  # Total number of degrees of freedom
print ('Nfem     =',Nfem)
print("-----------------------------")

#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value
  
ymin=np.min(y)
 
for i in range(0,NV):
    if abs(x[i])<eps: # left side
        bc_fix[i*ndof] = True ; bc_val[i*ndof] = 0.
    #end if
    if abs(y[i]-ymin)<eps: # bottom 
        #bc_fix[i*ndof] = True ; bc_val[i*ndof] = 0.
        bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
    #end if
#    if abs(y[i])<eps:  # top
#        bc_fix[i*ndof] = True ; bc_val[i*ndof] = 0
    #endif
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# compute area of elements
#################################################################
start = time.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(m,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(m,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(m,dtype=np.float64)          # shape functions derivatives

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV[0:m]=NNV(rq,sq)
        dNNNVdr[0:m]=dNNVdr(rq,sq)
        dNNNVds[0:m]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,m):
            jcb[0,0] += dNNNVdr[k]*x[icon[k,iel]]
            jcb[0,1] += dNNNVdr[k]*y[icon[k,iel]]
            jcb[1,0] += dNNNVds[k]*x[icon[k,iel]]
            jcb[1,1] += dNNNVds[k]*y[icon[k,iel]]
        #end for
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6e " %(area.sum()))

print("compute elements areas: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

a_mat = sps.lil_matrix((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
u     = np.zeros(NV,dtype=np.float64)           # x-component displacement 
v     = np.zeros(NV,dtype=np.float64)           # y-component displacement 
NNNV    = np.zeros(m,dtype=np.float64)          # shape functions V
dNNNVdx = np.zeros(m,dtype=np.float64)          # shape functions derivatives
dNNNVdy = np.zeros(m,dtype=np.float64)          # shape functions derivatives
dNNNVdr = np.zeros(m,dtype=np.float64)          # shape functions derivatives
dNNNVds = np.zeros(m,dtype=np.float64)          # shape functions derivatives

if axisymmetric:
   k_mat = np.array([[1,1,1,0],[1,1,1,0],[1,1,1,0],[0,0,0,0]],dtype=np.float64) 
   c_mat = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 
   b_mat = np.zeros((4,ndof*m),dtype=np.float64)   # gradient matrix B 
else:
   k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
   c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
   b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    K_el = np.zeros((m*ndof,m*ndof),dtype=np.float64)
    f_el = np.zeros(m*ndof)

    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV[0:m]=NNV(rq,sq)
        dNNNVdr[0:m]=dNNVdr(rq,sq)
        dNNNVds[0:m]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,m):
            jcb[0,0] += dNNNVdr[k]*x[icon[k,iel]]
            jcb[0,1] += dNNNVdr[k]*y[icon[k,iel]]
            jcb[1,0] += dNNNVds[k]*x[icon[k,iel]]
            jcb[1,1] += dNNNVds[k]*y[icon[k,iel]]
        #end for
        jcob = np.linalg.det(jcb)
        jcbi = np.linalg.inv(jcb)

        # compute dNdx & dNdy
        xq=0.0
        yq=0.0
        for k in range(0,m):
            xq+=NNNV[k]*x[icon[k,iel]]
            yq+=NNNV[k]*y[icon[k,iel]]
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for

        if axisymmetric:
           for i in range(0, m):
               b_mat[0:4, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                        [NNNV[i]/xq,0.       ],
                                        [0.        ,dNNNVdy[i]],
                                        [dNNNVdy[i],dNNNVdx[i]]]
           K_el += 2.*np.pi*b_mat.T.dot((mu[iel]*c_mat+lambdaa[iel]*k_mat).dot(b_mat))*weightq*jcob * xq
        else:
           for i in range(0, m):
               b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                        [0.        ,dNNNVdy[i]],
                                        [dNNNVdy[i],dNNNVdx[i]]]
           K_el += b_mat.T.dot((mu[iel]*c_mat+lambdaa[iel]*k_mat).dot(b_mat))*weightq*jcob
        
    #end for kq

    # impose dirichlet b.c. 
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m*ndof):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
            #end if
        #end for 
    #end for

    # impose Neumann b.c.
    inode0=icon[0,iel]
    inode1=icon[1,iel]
    inode2=icon[2,iel]
    if abs(y[inode0])<eps and abs(y[inode2])<eps:  # nodes 0 & 2 on top boundary 
       if axisymmetric:
          f_el[1]-=t_bc*(x[inode2]-x[inode0])/2 * 2*np.pi*(x[inode2]+x[inode0])/2
          f_el[5]-=t_bc*(x[inode2]-x[inode0])/2 * 2*np.pi*(x[inode2]+x[inode0])/2
       else:
          f_el[1]-=t_bc*(x[inode2]-x[inode0])/2
          f_el[5]-=t_bc*(x[inode2]-x[inode0])/2
     #end if 

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon[k2,iel]+i2
                    a_mat[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            rhs[m1]+=f_el[ikk]
        #end for
    #end for
#end for

print("     -> rhs (m,M) %.4f %.4f " %(np.min(rhs),np.max(rhs)))

print("build FE matrix: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
start = time.time()

u,v=np.reshape(sol,(NV,2)).T

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

#np.savetxt('displacement.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')
           
#####################################################################
# compute elemental strain components
#####################################################################
start = time.time()

qcoords2_r=[0.1012865073235,0.7974269853531,0.1012865073235,\
            0.4701420641051,0.4701420641051,0.0597158717898,0.3333333333333]
qcoords2_s=[0.1012865073235,0.1012865073235,0.7974269853531,\
            0.0597158717898,0.4701420641051,0.4701420641051,0.3333333333333]
qweights2 =[0.0629695902724,0.0629695902724,0.0629695902724,\
            0.0661970763942,0.0661970763942,0.0661970763942,0.1125000000000]

xc  = np.zeros(nel,dtype=np.float64)  
yc  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
ett = np.zeros(nel,dtype=np.float64) # theta-theta 
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  
divv= np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]=np.sum(x[icon[:,iel]])/3.
    yc[iel]=np.sum(y[icon[:,iel]])/3.
    for kq in range(0,7):
        rq=qcoords2_r[kq]
        sq=qcoords2_s[kq]
        weightq=qweights2[kq]
        NNNV[0:m]=NNV(rq,sq)
        dNNNVdr[0:m]=dNNVdr(rq,sq)
        dNNNVds[0:m]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,m):
            jcb[0,0]+=dNNNVdr[k]*x[icon[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*y[icon[k,iel]]
            jcb[1,0]+=dNNNVds[k]*x[icon[k,iel]]
            jcb[1,1]+=dNNNVds[k]*y[icon[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)
        xq=0
        yq=0
        for k in range(0,m):
            xq+=NNNV[k]*x[icon[k,iel]]
            yq+=NNNV[k]*y[icon[k,iel]]
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for
        for k in range(0,m):
            exx[iel] += dNNNVdx[k]*u[icon[k,iel]]*jcob*weightq
            eyy[iel] += dNNNVdy[k]*v[icon[k,iel]]*jcob*weightq
            exy[iel] += 0.5*(dNNNVdy[k]*u[icon[k,iel]]+dNNNVdx[k]*v[icon[k,iel]])*jcob*weightq
            if axisymmetric:
               ett[iel] += NNNV[k]*u[icon[k,iel]]/xq*jcob*weightq
            else:
               ett[iel] += 0 
        #end for
    #end for
    exx[iel]/=area[iel]
    eyy[iel]/=area[iel]
    exy[iel]/=area[iel]
    divv[iel]=exx[iel]+eyy[iel]+ett[iel]
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
#end for

print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))

#np.savetxt('strain.ascii',np.array([xc,yc,exx,eyy,exy,divv]).T,header='# xc,yc,exx,eyy,exy,divv')

print("compute strain components: %.3f s" % (time.time() - start))

#################################################################
# compute elemental stress 
#################################################################

sigma_xx = np.zeros(nel,dtype=np.float64)  
sigma_yy = np.zeros(nel,dtype=np.float64)  
sigma_xy = np.zeros(nel,dtype=np.float64)  
sigma_angle = np.zeros(nel,dtype=np.float64)  

sigma_xx[:]=lambdaa[:]*divv[:]+2*mu[:]*exx[:]
sigma_yy[:]=lambdaa[:]*divv[:]+2*mu[:]*eyy[:]
sigma_xy[:]=                  +2*mu[:]*exy[:]

sigma_angle[:]=0.5*np.arctan(2*sigma_xy[:]/(sigma_yy[:]-sigma_xx[:])) #* (180/np.pi)

sigma_1=0.5*(sigma_xx[:]+sigma_yy[:]) \
       +np.sqrt( 0.25*(sigma_xx[:]-sigma_yy[:])**2+sigma_xy[:]**2 )

sigma_2=0.5*(sigma_xx[:]+sigma_yy[:]) \
       -np.sqrt( 0.25*(sigma_xx[:]-sigma_yy[:])**2+sigma_xy[:]**2 )
           
#################################################################

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")

    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain (2nd inv)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_xx (MPa)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_xx[iel]/MPa))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_yy (MPa)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_yy[iel]/MPa))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_xy (MPa)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_xy[iel]/MPa))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='stress principal angle' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_angle[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_1 (MPa)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_1[iel]/MPa))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_2 (MPa)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_2[iel]/MPa))
    vtufile.write("</DataArray>\n")  
    vtufile.write("<DataArray type='Float32' Name='maximum shear stress (MPa)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (0.5*(sigma_1[iel]-sigma_2[iel])/MPa))
    vtufile.write("</DataArray>\n")  
    vtufile.write("<DataArray type='Float32' Name='volumetric strain div(u)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" %divv[iel]) 
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='mu' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (mu[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='lambda' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (lambdaa[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='tensile_zone (sigma1 gt 0)' Format='ascii'> \n")
    for iel in range(0,nel):
        if sigma_1[iel]>0:
            vtufile.write("%10e \n" % 1)
        else:
            vtufile.write("%10e \n" % 0)
        #end if
    #end for
    vtufile.write("</DataArray>\n")  

    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='n1' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e %10e %10e\n" % (np.cos(sigma_angle[iel]),np.sin(sigma_angle[iel]),0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='n2' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e %10e %10e\n" % (np.cos(sigma_angle[iel]+np.pi/2),np.sin(sigma_angle[iel]+np.pi/2),0.))
    vtufile.write("</DataArray>\n")

    vtufile.write("</CellData>\n")

    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    
    #vtufile.write("<DataArray type='Float32' Name='outer_node' Format='ascii'> \n")
    #for i in range(0,NV):
    #    if outer_node[i]:
    #      vtufile.write("%10e \n" % 1)
    #    else:
    #       vtufile.write("%10e \n" % 0)
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='bc_fix (u)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*ndof]:
           vtufile.write("%10e \n" % 1)
        else:
           vtufile.write("%10e \n" % 0)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='bc_fix (v)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*ndof+1]:
           vtufile.write("%10e \n" % 1)
        else:
           vtufile.write("%10e \n" % 0)
    vtufile.write("</DataArray>\n")
    
    #--
    vtufile.write("</PointData>\n")
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
        vtufile.write("%d \n" %((iel+1)*3))
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

print("Total time: %.3f s" % (time.time() - start_all))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
