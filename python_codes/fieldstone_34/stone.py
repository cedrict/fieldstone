import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as time
from scipy.sparse import lil_matrix

# exp=1: simple shear
# exp=2: pure shear
# exp=3: aquarium 
# exp=4: strip load

experiment=4

#------------------------------------------------------------------------------

def disp_x(x,y,rho,g,lambdaa,mu,L):
    if experiment==1:
       val=y
    if experiment==2:
       val=2*(x-0.5)
    if experiment==3:
       val=0
    if experiment==4:
       val=0
    return val

def disp_y(x,y,rho,g,lambdaa,mu,L):
    if experiment==1:
       val=0
    if experiment==2:
       val=-2*(y-0.5)
    if experiment==3:
       val=rho*g/(lambdaa+2*mu)*(0.5*y**2-L*y)
    if experiment==4:
       val=0
    return val

def pressure(x,y,rho,g,lambdaa,mu,L):
    if experiment==1:
       val=0.
    if experiment==2:
       val=0.
    if experiment==3:
       val=(lambdaa+2./3.*mu)/(lambdaa+2*mu)*rho*g*(L-y)
    if experiment==4:
       val=0
    return val

def sigma_xx(x,y,p0,a):
    xR=Lx/2+a
    xL=Lx/2-a
    theta1=np.arctan((x-xR)/(Ly-y))
    theta2=np.arctan((x-xL)/(Ly-y))
    val=p0/np.pi*(theta2-theta1-0.5*(np.sin(2*theta2)-np.sin(2*theta1)))
    return -val

def sigma_xy(x,y,p0,a):
    xR=Lx/2+a
    xL=Lx/2-a
    theta1=np.arctan((x-xR)/(Ly-y))
    theta2=np.arctan((x-xL)/(Ly-y))
    val=p0/np.pi*((np.sin(theta2))**2-(np.sin(theta1))**2)
    return val

def sigma_yy(x,y,p0,a):
    xR=Lx/2+a
    xL=Lx/2-a
    theta1=np.arctan((x-xR)/(Ly-y))
    theta2=np.arctan((x-xL)/(Ly-y))
    val=p0/np.pi*(theta2-theta1+0.5*(np.sin(2*theta2)-np.sin(2*theta1)))
    return -val

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node


# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 300
   nely = 200
   visu = 1
    
nnx=nelx+1 
nny=nely+1 
NV=nnx*nny 
nel=nelx*nely 
Nfem=NV*ndof 

if experiment==1:
   Lx=1
   Ly=1
   gx=0
   gy=0
   rho=0
   mu=1
   nu=0.25   
   lambdaa=2*mu*nu/(1-2*nu)

if experiment==2:
   Lx=1
   Ly=1
   gx=0
   gy=0
   rho=0
   mu=1
   nu=0.25   
   lambdaa=2*mu*nu/(1-2*nu)

if experiment==3:
   Lx=1000.  
   Ly=1000.
   gx=0
   gy=9.81
   E=6e10 # Young's modulus
   nu=0.25 # Poisson ratio
   rho=2800
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)

if experiment==4:
   Lx=3000.  
   Ly=2000.
   gx=0
   gy=0
   E=6e10 # Young's modulus
   nu=0.25 # Poisson ratio
   rho=2800
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   a=50
   p0=1e8

hx=Lx/nelx
hy=Ly/nely

eps=1.e-10
sqrt3=np.sqrt(3.)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV, dtype=np.float64)  # x coordinates
y = np.empty(NV, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
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

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    for k in range(0,m):
        xc[iel]+=x[icon[k,iel]]*0.25
        yc[iel]+=y[icon[k,iel]]*0.25


#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value

if experiment==1:
   for i in range(0, NV):
       if x[i]<eps:
          bc_fix[i*ndof+1]   = True ; bc_val[i*ndof+1]   = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof+1]   = True ; bc_val[i*ndof+1]   = 0.
       if y[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]>(Ly-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 1
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

if experiment==2:
   for i in range(0, NV):
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = +1
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = -1
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -1
       if y[i]>(Ly-eps):
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = +1

if experiment==3 or experiment==4:
   for i in range(0, NV):
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)

#a_mat = np.zeros((Nfem,Nfem),dtype=np.float64) # matrix of Ax=b
b_mat = np.zeros((3,ndof*m),dtype=np.float64)  # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)        # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)           # shape functions
dNdx  = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)           # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)          # x-component displacement 
v     = np.zeros(NV,dtype=np.float64)          # y-component displacement 
c_mat = np.array([[2*mu+lambdaa,lambdaa,0],[lambdaa,2*mu+lambdaa,0],[0,0,mu]],dtype=np.float64) 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros(m*ndof)
    a_el = np.zeros((m*ndof,m*ndof), dtype=np.float64)

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
            a_el += b_mat.T.dot(c_mat.dot(b_mat))*wq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                b_el[2*i  ]-=N[i]*jcob*wq*gx*rho
                b_el[2*i+1]-=N[i]*jcob*wq*gy*rho

        #end for
    #end for

    if experiment==4 and abs(xc[iel]-Lx/2)<a and yc[iel]>Ly-hy:
       b_el[2*2+1]-=0.5*p0*hx
       b_el[2*3+1]-=0.5*p0*hx

    # apply boundary conditions
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            m1 =ndof*icon[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndof*k1+i1
               aref=a_el[ikk,ikk]
               for jkk in range(0,m*ndof):
                   b_el[jkk]-=a_el[jkk,ikk]*fixt
                   a_el[ikk,jkk]=0.
                   a_el[jkk,ikk]=0.
               a_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt


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

print("build FE matrix: %.3f s" % (time.time() - start))

#################################################################
# impose boundary conditions
#################################################################
#start = time.time()
#for i in range(0, Nfem):
#    if bc_fix[i]:
#       a_matref = a_mat[i,i]
#       for j in range(0,Nfem):
#           rhs[j]-= a_mat[i, j] * bc_val[i]
#           a_mat[i,j]=0.
#           a_mat[j,i]=0.
#           a_mat[i,i] = a_matref
#       rhs[i]=a_matref*bc_val[i]
#print("a_mat (m,M) = %.4f %.4f" %(np.min(a_mat),np.max(a_mat)))
#print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))
#print("impose b.c.: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y arrays
#####################################################################
start = time.time()

u,v=np.reshape(sol,(NV,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

np.savetxt('displacement.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure
#####################################################################
start = time.time()

p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  

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
    jcbi=np.linalg.inv(jcb)

    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    for k in range(0, m):
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

    p[iel]=-(lambdaa+2./3.*mu)*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')

np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

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
            wq=1.*1.
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
            errv+=((uq-disp_x(xq,yq,rho,gy,lambdaa,mu,Ly))**2\
                  +(vq-disp_y(xq,yq,rho,gy,lambdaa,mu,Ly))**2)*wq*jcob
            errp+=(p[iel]-pressure(xq,yq,rho,gy,lambdaa,mu,Ly))**2*wq*jcob

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################
       
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
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % pressure(xc[iel],yc[iel],rho,gy,lambdaa,mu,Ly))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (p[iel]-pressure(xc[iel],yc[iel],rho,gy,lambdaa,mu,Ly)))
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
       vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (lambdaa*(exx[iel]+eyy[iel])+2*mu*exx[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (lambdaa*(exx[iel]+eyy[iel])+2*mu*eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (2*mu*exy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xx (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (sigma_xx(xc[iel],yc[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xy (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (sigma_xy(xc[iel],yc[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_yy (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (sigma_yy(xc[iel],yc[iel],p0,a)))
       vtufile.write("</DataArray>\n")



       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(0.,disp_y(x[i],y[i],rho,gy,lambdaa,mu,Ly),0.))
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
