import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def disp_x(x,y):
    val=(1+nu)/E*x*((1-2*nu)*Bcoeff-Acoeff/x**2)
    return val

def disp_y(x,y):
    val=0
    return val


#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

r1=0.5
r2=1

p1=1.
p2=0.

Lx=r2-r1  # horizontal extent of the domain 
Ly=0.5      # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 48
   nely = 20
   visu = 1
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
Nfem=nnp*ndof  # Total number of degrees of freedom

gx=0
gy=0

E=1 # Young's modulus
nu=0.25 # Poisson ratio

mu=E/2/(1+nu)
lambdaa=E*nu/(1+nu)/(1-2*nu)

print ('mu=',mu)
print('lambda=',lambdaa)

eps=1.e-10
sqrt3=np.sqrt(3.)

Acoeff=r1**2*r2**2*(p2-p1)/(r2**2-r1**2)
Bcoeff=(r1**2*p1-r2**2*p2)/(r2**2-r1**2)

axisymmetric=True

#################################################################

nqperdim=3

if nqperdim==1:
   qcoords=[0]
   qweights=[2.]

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)+r1
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
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

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value
for i in range(0,nnp):
    if abs(x[i]-r1)<eps:
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = disp_x(x[i],y[i])
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = disp_y(x[i],y[i])
    if abs(x[i]-r2)<eps:
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = disp_x(x[i],y[i])
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = disp_y(x[i],y[i])
    if y[i]<eps:
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = disp_y(x[i],y[i])
    if y[i]>(Ly-eps):
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = disp_y(x[i],y[i])

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

#a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)

a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives

if axisymmetric:
   k_mat = np.array([[1,1,1,0],[1,1,1,0],[1,1,1,0],[0,0,0,0]],dtype=np.float64) 
   c_mat = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 
   b_mat = np.zeros((4,ndof*m),dtype=np.float64)   # gradient matrix B 
else:
   k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
   c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
   b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros(m*ndof)
    K_el = np.zeros((m*ndof,m*ndof), dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

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
                jcb[0,0] += dNdr[k]*x[icon[k,iel]]
                jcb[0,1] += dNdr[k]*y[icon[k,iel]]
                jcb[1,0] += dNds[k]*x[icon[k,iel]]
                jcb[1,1] += dNds[k]*y[icon[k,iel]]

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
            #for i in range(0,m):
            #    b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
            #                             [0.     ,dNdy[i]],
            #                             [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            #a_el += b_mat.T.dot(c_mat.dot(b_mat))*wq*jcob

            if axisymmetric:
               for i in range(0,m):
                   b_mat[0:4, 2*i:2*i+2] = [[dNdx[i],0.       ],
                                            [N[i]/xq,0.       ],
                                            [0.     ,dNdy[i]],
                                            [dNdy[i],dNdx[i]]]
               K_el += 2*np.pi*b_mat.T.dot((mu*c_mat+lambdaa*k_mat).dot(b_mat))*weightq*jcob * xq
            else:
               for i in range(0,m):
                   b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.       ],
                                            [0.     ,dNdy[i]],
                                            [dNdy[i],dNdx[i]]]
               K_el += b_mat.T.dot((mu*c_mat+lambdaa*k_mat).dot(b_mat))*weightq*jcob

        #end for
    #end for

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
            rhs[m1]+=b_el[ikk]

print("build FE matrix: %.3f s" % (time.time() - start))

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

#print("a_mat (m,M) = %.4f %.4f" %(np.min(a_mat),np.max(a_mat)))
#print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

print("impose b.c.: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y displacement arrays
#####################################################################
start = time.time()

u,v=np.reshape(sol,(nnp,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

np.savetxt('displacement.ascii',np.array([x,u,v]).T,header='# x,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

#####################################################################
# retrieve 
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
uc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
ett = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
divv = np.zeros(nel,dtype=np.float64)  
sigmaxx = np.zeros(nel,dtype=np.float64)  
sigmayy = np.zeros(nel,dtype=np.float64)  
sigmatt = np.zeros(nel,dtype=np.float64)  
sigmaxy = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    weightq = 2.0 * 2.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

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
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        uc[iel] += N[k]*u[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

    ett[iel]= uc[iel]/xc[iel] 
    divv[iel]=exx[iel]+ett[iel]+eyy[iel]
    sigmaxx[iel]=lambdaa*divv[iel]+2*mu*exx[iel] 
    sigmatt[iel]=lambdaa*divv[iel]+2*mu*ett[iel] 
    sigmayy[iel]=lambdaa*divv[iel]+2*mu*eyy[iel] 
    sigmaxy[iel]=                 +  mu*exy[iel] 

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('profiles.ascii',np.array([xc,exx,ett,eyy,exy,sigmaxx,sigmatt,sigmayy,sigmaxy]).T)

print("compute press & sr: %.3f s" % (time.time() - start))

#################################################################
# compute error
#################################################################
start = time.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            jcob=np.linalg.det(jcb)
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
            #errv+=((uq-velocity_x(xq,yq,lambdaa,mu,Ly))**2\
            #      +(vq-velocity_y(xq,yq,lambdaa,mu,Ly))**2)*wq*jcob

errv=np.sqrt(errv)

print("     -> nel= %6d ; errv= %.8f " %(nel,errv))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################
       
if visu==1:
       vtufile=open('solution.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nnp):
          vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='err' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='ett' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % ett[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='ezz' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % eyy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='erz' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (divv[iel]))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' Name='sigmaxx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % sigmaxx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigmatt' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % sigmatt[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigmayy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % sigmayy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigmaxy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % sigmaxy[iel])
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(disp_x(x[i],y[i]),disp_y(x[i],y[i]),0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
