import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix
import time as clock 

###############################################################################

def ubc(x,y,ldc):
    if ldc==0:
       val=1.
    elif ldc==1: 
       val=x**2*(1.-x)**2 *16
    elif ldc==2:
       xx=0.1
       if x<xx:
          val=1-0.25*(1-np.cos((xx-x)/xx*np.pi))**2
       elif x>1-xx:
          val=1-0.25*(1-np.cos(((1-xx)-x)/xx*np.pi))**2
       else:
          val=1.
    return val

def vbc(x,y):
    val=0.
    return val

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 004 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 64
   nely = nelx
   visu = 1
    
nnx=nelx+1         # number of elements, x direction
nny=nely+1         # number of elements, y direction
nn_V=nnx*nny       # number of nodes
nel=nelx*nely      # number of elements, total
Nfem_V=nn_V*ndof_V # number of velocity degrees of freedom
Nfem=Nfem_V        # Total number of degrees of freedom

penalty=1.e7  # penalty coefficient value

viscosity=1.  # dynamic viscosity \mu

ldc=2

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

if debug:
   for iel in range (0,nel):
       print ("iel=",iel)
       print ("node 1",icon[0,iel],"at pos.",x[icon[0,iel]], y[icon[0,iel]])
       print ("node 2",icon[1,iel],"at pos.",x[icon[1,iel]], y[icon[1,iel]])
       print ("node 3",icon[2,iel],"at pos.",x[icon[2,iel]], y[icon[2,iel]])
       print ("node 4",icon[3,iel],"at pos.",x[icon[3,iel]], y[icon[3,iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =ubc(x_V[i],y_V[i],ldc)
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vbc(x_V[i],y_V[i])

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

#################################################################
# build FE matrix
#################################################################
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
            A_el+=B.T.dot(C.dot(B))*viscosity*JxWq

            # compute elemental rhs vector
            #for i in range(0,m_V):
            #    b_el[2*i  ]+=N_V[i]*rho*gx*JxWq
            #    b_el[2*i+1]+=N_V[i]*rho*gy*JxWq

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

if debug:
   print("A_fem (m,M) = %.4e %.4e" %(np.min(A_fem),np.max(A_fem)))
   print("b_fem (m,M) = %.4e %.4e" %(np.min(b_fem),np.max(b_fem)))

print("impose b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

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

for iel in range(0,nel):

    rq=0.0
    sq=0.0
    weightq=2.0*2.0

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

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5

    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x_e,y_e,exx,eyy,exy')

np.savetxt('p_top.ascii',np.array([x_e[nel-nelx:nel],p[nel-nelx:nel]]).T,header='# x,p')

print("compute press & sr: %.3f s" % (clock.time()-start))

######################################################################
# compute nodal pressure
######################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    q[icon_V[0,iel]]+=p[iel]
    q[icon_V[1,iel]]+=p[iel]
    q[icon_V[2,iel]]+=p[iel]
    q[icon_V[3,iel]]+=p[iel]
    count[icon_V[0,iel]]+=1
    count[icon_V[1,iel]]+=1
    count[icon_V[2,iel]]+=1
    count[icon_V[3,iel]]+=1
#end for

q/=count

np.savetxt('q_top.ascii',np.array([x_V[nn_V-nnx:nn_V],q[nn_V-nnx:nn_V]]).T,header='# x,q')

if debug: np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

print("compute p on V-nodes: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if visu==1:

   filename = 'solution.vtu'
   vtufile=open(filename,"w")
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
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % p[iel])
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % exx[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % eyy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % exy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e (2nd inv.)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % (0.5*(np.sqrt(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)))
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
   #-------------
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10e \n" % q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
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

print("generate vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")
