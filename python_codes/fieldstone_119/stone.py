import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as time

###############################################################################

def rho(x,y):
    if experiment==1:
       val=1
    if experiment==2:
       if y>0.5:
          val=1
       else:
          val=2
    if experiment==3:
       if x<0.5:
          val=1
       else:
          val=2
    if experiment==4:
       if (x-0.5)**2+(y-0.5)**2<0.25**2:
          val=2
       else:
          val=1
    if experiment==5:
       if (x-0.5*Lx)**2+(y-0.5*Ly)**2<0.1**2:
          val=10
       else:
          val=1
    return val

###############################################################################

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

nelx = 100
nely = 100
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

NV=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

Nfem=NV  # Total number of degrees of freedom

eps=1.e-10

sqrt3=np.sqrt(3.)

gy=-10

experiment=5

###############################################################################
# grid point setup
###############################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

###############################################################################
# build connectivity array
###############################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
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

print("setup: connectivity: %.3f s" % (time.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value
for i in range(0,NV):
    if y[i]>(Ly-eps):
       bc_fix[i]   = True ; bc_val[i]   = 0.

print("setup: boundary conditions: %.3f s" % (time.time() - start))

###############################################################################
# build FE matrix
# r,s are the reduced coordinates in the [-1:1]x[-1:1] ref elt
###############################################################################
start = time.time()

A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros(m,dtype=np.float64)
    a_el = np.zeros((m,m),dtype=np.float64)
    B_mat=np.zeros((2,m),dtype=np.float64)     # gradient matrix B 

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
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
            jcb = np.zeros((2,2),dtype=np.float64)
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
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                B_mat[0,k]=dNdx[k]
                B_mat[1,k]=dNdy[k]
            #end for

            # compute diffusion matrix
            a_el+=B_mat.T.dot(B_mat)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,m):
                b_el[i]+=dNdy[i]*jcob*weightq*rho(xq,yq)*gy

        #end for
    #end for

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
           a_el[k1,k1]=Aref
           b_el[k1]=Aref*bc_val[m1]
        #end if
    #end for

    # assemble matrix A_mat and right hand side rhs
    for k1 in range(0,m):
        m1=icon[k1,iel]
        for k2 in range(0,m):
            m2=icon[k2,iel]
            A_mat[m1,m2]+=a_el[k1,k2]
        #end for
        rhs[m1]+=b_el[k1]
    #end for

#end for iel

print("build FE matrix: %.3f s" % (time.time() - start))

###############################################################################
# solve system
###############################################################################
start = time.time()

sol1 = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

np.savetxt('pressure1.ascii',np.array([x,y,sol1]).T)

print("solve time: %.3f s" % (time.time() - start))

###############################################################################
# integrate columns
###############################################################################

rhon = np.zeros(Nfem,dtype=np.float64) 

for i in range(0,NV):
    rhon[i]=rho(x[i],y[i])

sol2 = np.zeros(Nfem,dtype=np.float64) 

hy=Ly/nely

for i in range(0,nnx):
    for j in range(nny-2,-1,-1):
        k=j*nnx+i
        k_above=(j+1)*nnx+i
        sol2[k]=sol2[k_above]+hy*(rhon[k]+rhon[k_above])/2*abs(gy)


np.savetxt('pressure2.ascii',np.array([x,y,sol2]).T)
        
###############################################################################
# export middle and side profiles
###############################################################################

bottomfile=open("press_bottom.ascii","w")
leftfile=open("press_left.ascii","w")
rightfile=open("press_right.ascii","w")

for i in range(0,NV):
    if y[i]/Ly<eps:
       bottomfile.write("%10f %10f %10f \n" %(x[i],sol1[i],sol2[i]))
    if x[i]/Ly<eps:
       leftfile.write("%10f %10f %10f \n" %(y[i],sol1[i],sol2[i]))
    if x[i]/Ly>1-eps:
       rightfile.write("%10f %10f %10f \n" %(y[i],sol1[i],sol2[i]))

#####################################################################
# retrieve pressure
# we compute the pressure and strain rate components in the middle 
# of the elements.
#####################################################################
start = time.time()

dp1dx = np.zeros(nel,dtype=np.float64)  
dp1dy = np.zeros(nel,dtype=np.float64)  
dp2dx = np.zeros(nel,dtype=np.float64)  
dp2dy = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    for k in range(0, m):
        dp1dx[iel] += dNdx[k]*sol1[icon[k,iel]]
        dp1dy[iel] += dNdy[k]*sol1[icon[k,iel]]
        dp2dx[iel] += dNdx[k]*sol2[icon[k,iel]]
        dp2dy[iel] += dNdy[k]*sol2[icon[k,iel]]

print("     -> dp1dx (m,M) %.4f %.4f " %(np.min(dp1dx),np.max(dp1dx)))
print("     -> dp1dy (m,M) %.4f %.4f " %(np.min(dp1dy),np.max(dp1dy)))
print("     -> dp2dx (m,M) %.4f %.4f " %(np.min(dp2dx),np.max(dp2dx)))
print("     -> dp2dy (m,M) %.4f %.4f " %(np.min(dp2dy),np.max(dp2dy)))

###############################################################################
# export to vtu
###############################################################################
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
#
vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
for iel in range (0,nel):
    xc=(x[icon[0,iel]]+x[icon[2,iel]])/2
    yc=(y[icon[0,iel]]+y[icon[2,iel]])/2
    vtufile.write("%e \n" % rho(xc,yc))
vtufile.write("</DataArray>\n")
#
vtufile.write("<DataArray type='Float32' Name='dp1dx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % dp1dx[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dp1dy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % dp1dy[iel])
vtufile.write("</DataArray>\n")
#
vtufile.write("<DataArray type='Float32' Name='dp2dx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % dp2dx[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dp2dy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % dp2dy[iel])
vtufile.write("</DataArray>\n")
#
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='pressure1' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e \n" % sol1[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='pressure2' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e \n" % sol2[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='p1-p2' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e \n" % (sol1[i]-sol2[i]))
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
