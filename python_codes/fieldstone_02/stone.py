import numpy as np
import sys as sys
import scipy.sparse as sps
import time as clock 
import matplotlib.pyplot as plt

###############################################################################

def density(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123456789**2:
       val=1.01
    else:
       val=1.
    return val

def viscosity(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123456789**2:
       val=1.e3
    else:
       val=1.
    return val

###############################################################################
#160x160 is maximum resolution for full square on 32Gb RAM laptop

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 001 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

#0: Free slip
#1: No slip
#2: Open top
bc_type=2

half=True

# allowing for argument parsing through command line
if int(len(sys.argv) == 3):
   nely = int(sys.argv[1])
   visu = int(sys.argv[2])
else:
   nely = 100
   visu = 1

if half:
   nelx=nely//2
   Lx=0.5 
   Ly=1. 
else:
   nelx=nely
   Lx=1. 
   Ly=1. 
    
nnx=nelx+1         # number of nodes, x direction
nny=nely+1         # number of nodes, y direction
nn_V=nnx*nny       # number of nodes
nel=nelx*nely      # number of elements, total
Nfem_V=nn_V*ndof_V # number of velocity degrees of freedom
Nfem=Nfem_V        # Total number of degrees of freedom

penalty=1.e7  # penalty coefficient value

gx=0.  # gravity vector, x component
gy=-1. # gravity vector, y component

debug=False

###############################################################################

print('nelx=',nelx)
print('nely=',nely)
print('nel=',nel)
print('nn_V=',nn_V)
print('Nfem=',Nfem)
print('penalty=',penalty)

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*Lx/float(nelx)
        y_V[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# build connectivity array
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx+1)
        icon_V[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0,iel],"at pos.",x[icon[0,iel]], y[icon[0,iel]])
#     print ("node 2",icon[1,iel],"at pos.",x[icon[1,iel]], y[icon[1,iel]])
#     print ("node 3",icon[2,iel],"at pos.",x[icon[2,iel]], y[icon[2,iel]])
#     print ("node 4",icon[3,iel],"at pos.",x[icon[3,iel]], y[icon[3,iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))


###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V =np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V =np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if bc_type==0:
   for i in range(0,nn_V):
       if x[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       if x[i]>(Lx-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       if y[i]<eps:
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y[i]>(Ly-eps):
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.

if bc_type==1:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V+0] = True ; bc_val_V[i*ndof_V+0] = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V+0] = True ; bc_val_V[i*ndof_V+0] = 0.
          if not half:
             bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V+0] = True ; bc_val_V[i*ndof_V+0] = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V+0] = True ; bc_val_V[i*ndof_V+0] = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.

if bc_type==2:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.

print("define b.c.: %.3f s" % (clock.time()-start))

#################################################################
###############################################################################
# build FE matrix
###############################################################################
#################################################################
start=clock.time()

A_fem=np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
b_fem=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)   # gradient matrix B 
N_V=np.zeros(m_V,dtype=np.float64)            # shape functions
dNdx_V=np.zeros(m_V,dtype=np.float64)            # shape functions derivatives
dNdy_V=np.zeros(m_V,dtype=np.float64)            # shape functions derivatives
dNdr_V=np.zeros(m_V,dtype=np.float64)            # shape functions derivatives
dNds_V=np.zeros(m_V,dtype=np.float64)            # shape functions derivatives
H=np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)
    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N_V[0]=0.25*(1.-rq)*(1.-sq)
            N_V[1]=0.25*(1.+rq)*(1.-sq)
            N_V[2]=0.25*(1.+rq)*(1.+sq)
            N_V[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
            dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
            dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
            dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

            # calculate the determinant of the jacobian
            JxWq=np.linalg.det(jcb)*weightq

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute coords of quad point
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])

            # compute dNdx & dNdy
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            # construct 3x8 B matrix
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.       ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental matrix
            A_el+=B.T.dot(C.dot(B))*viscosity(xq,yq)*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                b_el[2*i  ]+=N_V[i]*density(xq,yq)*gx*JxWq
                b_el[2*i+1]+=N_V[i]*density(xq,yq)*gy*JxWq

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    weightq=2.*2.

    dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
    dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
    dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
    dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

    # compute the jacobian
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

    # calculate determinant of the jacobian
    JxWq=np.linalg.det(jcb)*weightq

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    # compute dNdx and dNdy
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    # compute gradient matrix
    for i in range(0,m_V):
        B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.       ],
                          [0.       ,dNdy_V[i]],
                          [dNdy_V[i],dNdx_V[i]]]

    # compute elemental matrix
    A_el+=B.T.dot(H.dot(B))*penalty*JxWq

    # assemble matrix and right hand side 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
            b_fem[m1]+=b_el[ikk]

print("Build FE matrix: %.5f s | Nfem= %d" % (clock.time()-start,Nfem))

###############################################################################
# impose boundary conditions
# for now it is done outside of the previous loop, we will see
# later in the course how it can be incorporated seamlessly in it.
###############################################################################
start=clock.time()

for i in range(0, Nfem):
    if bc_fix_V[i]:
       A_femref = A_fem[i,i]
       for j in range(0,Nfem):
           b_fem[j]-= A_fem[i, j] * bc_val_V[i]
           A_fem[i,j]=0.
           A_fem[j,i]=0.
           A_fem[i,i] = A_femref
       b_fem[i]=A_femref*bc_val_V[i]

#print("A_fem (m,M) = %.4e %.4e" %(np.min(A_fem),np.max(A_fem)))
#print("b_fem (m,M) = %.4e %.4e" %(np.min(b_fem),np.max(b_fem)))

print("impose b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol = sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("Solve linear system: %.5f s | Nfem= %d " % (clock.time()-start,Nfem))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# compute pressure and strain rate components in the middle of the elements
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
eta=np.zeros(nel,dtype=np.float64)  
dens=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    weightq = 2.0 * 2.0

    # calculate shape functions
    N_V[0]=0.25*(1.-rq)*(1.-sq)
    N_V[1]=0.25*(1.+rq)*(1.-sq)
    N_V[2]=0.25*(1.+rq)*(1.+sq)
    N_V[3]=0.25*(1.-rq)*(1.+sq)

    # calculate shape function derivatives
    dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
    dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
    dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
    dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

    # calculate jacobian matrix
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

    # calculate inverse of the jacobian matrix
    jcbi = np.linalg.inv(jcb)

    # compute dNdx & dNdy
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    # compute coords of center 
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    eta[iel]=viscosity(x_e[iel],y_e[iel])
    dens[iel]=density(x_e[iel],y_e[iel])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5

    p[iel]=-penalty*(exx[iel]+eyy[iel])

e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("     -> eta (m,M) %.4f %.4f " %(np.min(eta),np.max(eta)))
print("     -> dens (m,M) %.4f %.4f " %(np.min(dens),np.max(dens)))

if debug:
   np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x_e,y_e,exx,eyy,exy')

print("compute press & strain rate: %.3f s" % (clock.time()-start))

###############################################################################
# compute vrms 
###############################################################################
start=clock.time()

vrms=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N_V[0]=0.25*(1.-rq)*(1.-sq)
            N_V[1]=0.25*(1.+rq)*(1.-sq)
            N_V[2]=0.25*(1.+rq)*(1.+sq)
            N_V[3]=0.25*(1.-rq)*(1.+sq)
            dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
            dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
            dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
            dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            vrms+=(uq**2+vq**2)*JxWq

vrms=np.sqrt(vrms/Lx/Ly)

print("compute vrms: %.3f s" % (clock.time()-start))

#####################################################################
###############################################################################
# export various measurements for stokes sphere benchmark 
###############################################################################
#####################################################################
start=clock.time()

vel=np.sqrt(u**2+v**2)
print('bench ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      0,0,\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),
      vrms)

print("measurements: %.3f s" % (clock.time()-start))

#####################################################################
# export to vtu 
#####################################################################
start=clock.time()

vtufile=open("solution.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10f %10f %10f \n" %(x_V[i],y_V[i],0.))
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
vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % e[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % eta[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % dens[iel])
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
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

print("export to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
