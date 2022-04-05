import numpy as np
import math as math
import sys as sys
import scipy
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as time
import random

bench=3

grid='symm'

#------------------------------------------------------------------------------

def bx(x, y):
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if bench==2:
       val=0
    if bench==3:
       val=0
    return val

def by(x, y):
    if bench==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if bench==2:
       val=0
    if bench==3:
       if abs(x-0.5)<0.125 and abs(y-0.5)<0.125:
          val=-15
       else: 
          val=-1
    return val

#------------------------------------------------------------------------------

def velocity_x(x,y):
    if bench==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if bench==2:
       val=20*x*y**3
    if bench==3:
       val=0
    return val

def velocity_y(x,y):
    if bench==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if bench==2:
       val=5*x**4-5*y**4
    if bench==3:
       val=0
    return val

def pressure(x,y):
    if bench==1:
       val=x*(1.-x)-1./6.
    if bench==2:
       val=60*x**2*y-20*y**3-5
    if bench==3:
       val=0
    return val

#------------------------------------------------------------------------------

eps=1.e-10
sqrt3=np.sqrt(3.)

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=3     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = 32
   visu = 1
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

NV=nnx*nny  # number of nodes

nel=2*nelx*nely  # number of elements, total

eta=1.  # dynamic viscosity 

NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

pnormalise=False

Gscaling=eta/(Ly/nely)

print(nel,NV)

#################################################################
# quadrature
#################################################################

nq=3

val_r = np.zeros(nq,dtype=np.float64) 
val_s = np.zeros(nq,dtype=np.float64) 
val_w = np.zeros(nq,dtype=np.float64) 

if nq==3:
   val_r[0]=1/6 ; val_s[0]=1/6 ; val_w[0]=1/3/2
   val_r[1]=2/3 ; val_s[1]=1/6 ; val_w[1]=1/3/2
   val_r[2]=1/6 ; val_s[2]=2/3 ; val_w[2]=1/3/2
if nq==6:
   val_r[0]=0.091576213509771 ; val_s[0]=0.091576213509771 ; val_w[0]=0.109951743655322/2.0 
   val_r[1]=0.816847572980459 ; val_s[1]=0.091576213509771 ; val_w[1]=0.109951743655322/2.0 
   val_r[2]=0.091576213509771 ; val_s[2]=0.816847572980459 ; val_w[2]=0.109951743655322/2.0 
   val_r[3]=0.445948490915965 ; val_s[3]=0.445948490915965 ; val_w[3]=0.223381589678011/2.0 
   val_r[4]=0.108103018168070 ; val_s[4]=0.445948490915965 ; val_w[4]=0.223381589678011/2.0 
   val_r[5]=0.445948490915965 ; val_s[5]=0.108103018168070 ; val_w[5]=0.223381589678011/2.0 

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        if j>0 and j<nny-1 and i>0 and i<nnx-1:
           x[counter]+=random.uniform(-1,+1)/5*Lx/nelx
           y[counter]+=random.uniform(-1,+1)/5*Ly/nely
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((mV,nel),dtype=np.int32)

if grid=='symm':

   counter = 0  
   for j in range(0,nely):
       for i in range(0,nelx):
           inode0=i+j*(nelx+1)
           inode1=i+1+j*(nelx+1)
           inode2=i+1+(j+1)*(nelx+1)
           inode3=i+(j+1)*(nelx+1)
           if (i<nelx/2 and j<nely/2) or\
              (i>=nelx/2 and j>=nely/2):
              #C
              icon[0,counter]=inode1 
              icon[1,counter]=inode2
              icon[2,counter]=inode0
              counter += 1 
              #D
              icon[0,counter]=inode3 
              icon[1,counter]=inode0
              icon[2,counter]=inode2
              counter += 1 
           else: 
              #A
              icon[0,counter]=inode0 
              icon[1,counter]=inode1
              icon[2,counter]=inode3
              counter += 1 
              #B
              icon[0,counter]=inode2 
              icon[1,counter]=inode3
              icon[2,counter]=inode1
              counter += 1 
else:
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           icon[0, counter] = i + 1 + j * (nelx + 1)
           icon[1, counter] = i + 1 + (j + 1) * (nelx + 1)
           icon[2, counter] = i + j * (nelx + 1)
           counter += 1
   
           icon[0, counter] = i + (j + 1) * (nelx + 1)
           icon[1, counter] = i + j * (nelx + 1)
           icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
           counter += 1


# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i])
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i])
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i])
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i])
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

if pnormalise:
   A_mat = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)# matrix A 
   rhs   = np.zeros((Nfem+1),dtype=np.float64)         # right hand side 
   A_mat[Nfem,NfemV:Nfem]=1
   A_mat[NfemV:Nfem,Nfem]=1
else:
   A_mat = np.zeros((Nfem,Nfem),dtype=np.float64)# matrix A 
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
N     = np.zeros(mV,dtype=np.float64)            # shape functions
dNdx  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)            # x-component velocity
v     = np.zeros(NV,dtype=np.float64)            # y-component velocity
p     = np.zeros(nel,dtype=np.float64)           # pressure field 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) # a
#c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64)  #b



for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nq):

            # position & weight of quad. point
            rq=val_r[iq]
            sq=val_s[iq]
            weightq=val_w[iq]

            # calculate shape functions
            N[0]=1.-rq-sq
            N[1]=rq
            N[2]=sq

            # calculate shape function derivatives
            dNdr[0]=-1 ; dNds[0]=-1
            dNdr[1]=+1 ; dNds[1]=0
            dNdr[2]=0  ; dNds[2]=+1

            # calculate jacobian matrix
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3,2*i:2*i+2] = [[dNdx[i],0.     ],
                                        [0.     ,dNdy[i]],
                                        [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=N[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=N[i]*jcob*weightq*by(xq,yq)
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq

    #end for iq

    G_el*=Gscaling

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    A_mat[m1,m2]+=K_el[ikk,jkk]
            rhs[m1]+=f_el[ikk]
            A_mat[m1,NfemV+iel]+=G_el[ikk,0]
            A_mat[NfemV+iel,m1]+=G_el[ikk,0]
    rhs[NfemV+iel]+=h_el[0]

#A_mat=A_mat.tocsr()

#print(np.min(A_mat))
#print(np.max(A_mat))

#plt.spy(A_mat,markersize=1)
#plt.savefig('matrix.pdf', bbox_inches='tight')

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# solve system
######################################################################
start = time.time()

sol=sps.linalg.spsolve(A_mat,rhs)
#sol=rhs

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*Gscaling

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

######################################################################
# compute nodal pressure q
######################################################################

q=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q[icon[0,iel]]+=p[iel]
    q[icon[1,iel]]+=p[iel]
    q[icon[2,iel]]+=p[iel]
    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
#end for

q=q/count

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')


#####################################################################
# plot of solution export to vtu format
#####################################################################
start = time.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nel*3,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%10f %10f %10f \n" %(x[icon[i,iel]],y[icon[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   #vtufile.write("<CellData Scalars='scalars'>\n")
   #vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%.6e %.6e %.6e \n" %(u[icon[i,iel]],v[icon[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" %p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   counter=0
   for iel in range(0,nel):
       vtufile.write("%d %d %d  \n" %(counter,counter+1,counter+2))
       counter+=3
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
   print("export to vtu: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
