import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def density(x,y):
    if (x)**2+(y-0.5*Ly)**2<50e3**2:
       val=3000-300
    else:
       val=3000
    return val

def viscosity(x,y):
    if (x)**2+(y-0.5*Ly)**2<50e3**2:
       val=1.e24
    else:
       val=1e21
    return val

#------------------------------------------------------------------------------
#160x160 is maximum resolution for full square on 32Gb RAM laptop

print("-----------------------------")
print("--------fieldstone 02--------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

#0: Free slip
#1: No slip
#2: Open top
bc_type=0

nelx=150
nely=150
Lx=500e3 
Ly=500e3
    
nnx=nelx+1     # number of elements, x direction
nny=nely+1     # number of elements, y direction
NV=nnx*nny     # number of nodes
nel=nelx*nely  # number of elements, total
Nfem=NV*ndof   # Total number of degrees of freedom

penalty=1.e25  # penalty coefficient value

eps=1.e-10

gx=0.  # gravity vector, x component
gy=-9.81  # gravity vector, y component

sqrt3=np.sqrt(3.)

print('nelx=',nelx)
print('nely=',nely)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("mesh setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m,nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#     print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

print("build icon: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(Nfem,dtype=np.bool)    # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

if bc_type==0:
   for i in range(0,NV):
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]>(Ly-eps):
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

if bc_type==1:
   for i in range(0,NV):
       if x[i]<eps:
          bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]<eps:
          bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]>(Ly-eps):
          bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

if bc_type==2:
   for i in range(0,NV):
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

print("define b.c.: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)          # x-component velocity
v     = np.zeros(NV,dtype=np.float64)          # y-component velocity
k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros(m * ndof)
    a_el = np.zeros((m * ndof, m * ndof), dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb = np.zeros((2, 2),dtype=float)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq)*wq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                b_el[2*i  ]+=N[i]*jcob*wq*density(xq,yq)*gx
                b_el[2*i+1]+=N[i]*jcob*wq*density(xq,yq)*gy

        #end for
    #end for

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    wq=2.*2.

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    # compute the jacobian
    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob = np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi = np.linalg.inv(jcb)

    # compute dNdx and dNdy
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    # compute gradient matrix
    for i in range(0,m):
        b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                              [0.     ,dNdy[i]],
                              [dNdy[i],dNdx[i]]]

    # compute elemental matrix
    a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*wq*jcob

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon[k2,iel]+i2
                    a_mat[m1,m2]+=a_el[ikk,jkk]
            rhs[m1]+=b_el[ikk]

#end for iel 

print("build matrix: %.3f s" % (time.time() - start))

#################################################################
# impose boundary conditions
#################################################################
start = time.time()

for i in range(0, Nfem):
    if bc_fix[i]:
       a_matref = a_mat[i,i]
       for j in range(0,Nfem):
           rhs[j]-= a_mat[i, j] * bc_val[i]
           a_mat[i,j]=0.
           a_mat[j,i]=0.
           a_mat[i,i] = a_matref
       rhs[i]=a_matref*bc_val[i]
    #end if
#end for

print("impose b.c.: %.3f s" % (time.time() - start))

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

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split solution: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure and strain rate tensor 
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
eta = np.zeros(nel,dtype=np.float64)  
dens = np.zeros(nel,dtype=np.float64)  
sr = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    wq = 2.0 * 2.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    for k in range(0, m):
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

    p[iel]=-penalty*(exx[iel]+eyy[iel])
    eta[iel]=viscosity(xc[iel],yc[iel])
    dens[iel]=density(xc[iel],yc[iel])
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
print("     -> eta (m,M) %.4e %.4e " %(np.min(eta),np.max(eta)))
print("     -> dens (m,M) %.4e %.4e " %(np.min(dens),np.max(dens)))

#np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p & sr: %.3f s" % (time.time() - start))

#####################################################################
# compute vrms 
#####################################################################
start = time.time()

vrms=0.

for iel in range(0, nel):
    for iq in [-1, 1]:
        for jq in [-1, 1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb = np.zeros((2, 2),dtype=np.float64)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # compute dNdx & dNdy
            uq=0.0
            vq=0.0
            for k in range(0, m):
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]

            vrms+=(uq**2+vq**2)*jcob*weightq

        #end for
    #end for
#end for

vrms=np.sqrt(vrms/Lx/Ly)
       
print('     -> vrms=',vrms)

print("compute vrms: %.3f s" % (time.time() - start))

#####################################################################
# compute surface vrms and average velocity
#####################################################################

hx=Lx/nelx

vrms_surf=0.
avrgu_surf=0.
for iel in range(0,nel):
    if y[icon[3,iel]]/Ly>1-eps: 
       for iq in [-1, 1]:
           rq=iq/sqrt3
           weightq=1.
           N[0]=0.5*(1-rq)
           N[1]=0.5*(1+rq)
           uq=N[0]*u[icon[3,iel]]+N[1]*u[icon[2,iel]]
           jcob=hx/2
           vrms_surf+=jcob*weightq*uq**2
           avrgu_surf+=jcob*weightq*abs(uq)

vrms_surf=np.sqrt(vrms_surf/Lx)
avrgu_surf/=Lx

print('     -> vrms_surf=',vrms_surf)
print('     -> avrgu_surf=',avrgu_surf)

#####################################################################
# export to vtu 
#####################################################################
start = time.time()

vtufile=open("solution.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % p[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % exx[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % eyy[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % exy[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sr' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % sr[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % eta[iel])
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
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
