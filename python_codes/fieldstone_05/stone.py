import numpy as np
import sys as sys
import time as clock 
import solcx as solcx
import scipy.sparse as sps

###############################################################################

def viscosity(x,y):
    if x<0.5:
       val=1.
    else:
       val=1.e6
    return val

def density(x,y):
    val=np.sin(np.pi*y)*np.cos(np.pi*x)
    return val

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 005 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

gx=0
gy=1

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 50 
   nely = 50
   visu = 1
    
nnx=nelx+1         # number of nodes, x direction
nny=nely+1         # number of nodes, y direction
nn_V=nnx*nny       # number of velocity nodes
nel=nelx*nely      # number of elements, total
Nfem_V=nn_V*ndof_V # number of velocity degrees of freedom
Nfem=Nfem_V        # Total number of degrees of freedom

penalty=1.e7  # penalty coefficient value

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

print("setup node coordinates: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
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

print("setup connectivity array: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
#end for

print("define b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

#A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)

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

print("impose b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("reshape sol. vector: %.3f s" % (clock.time()-start))

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
rho=np.zeros(nel,dtype=np.float64)  

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
    rho[iel]=density(x_e[iel],y_e[iel])

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
print("     -> rho (m,M) %.4f %.4f " %(np.min(rho),np.max(rho)))

if debug:
   np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x_e,y_e,exx,eyy,exy')

print("compute press & strain rate: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

error_u=np.zeros(nn_V,dtype=np.float64)
error_v=np.zeros(nn_V,dtype=np.float64)
error_p=np.zeros(nel,dtype=np.float64)

for i in range(0,nn_V):
    ui,vi,pi=solcx.SolCxSolution(x_V[i],y_V[i]) 
    error_u[i]=u[i]-ui
    error_v[i]=v[i]-vi
#end for

for iel in range(0,nel): 
    ui,vi,pi=solcx.SolCxSolution(x_e[iel],y_e[iel]) 
    error_p[iel]=p[iel]-pi
#end for

if debug: np.savetxt('error_pressure.ascii',np.array([x_e,y_e,error_p]).T,header='# x,y,p')

errv=0.
errp=0.
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
            jcbi = np.linalg.inv(jcb)
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            ui,vi,pi=solcx.SolCxSolution(xq,yq) 
            errv+=((uq-ui)**2+(vq-vi)**2)*JxWq
            errp+=(p[iel]-pi)**2*JxWq

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute discr. errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate xx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain rate yy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain rate xy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain rate (effective)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (p[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='pressure (error)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (error_p[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eta[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (rho[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(error_u[i],error_v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*4))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

print("export to vtu: %.3f s" % (clock.time()-start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
