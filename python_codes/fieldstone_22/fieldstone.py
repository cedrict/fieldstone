import numpy as np
import math as math
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix 
import time as time

#------------------------------------------------------------------------------

def bx(x,y,case):
    if case==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if case==2:
       val=-(1+y-3*x**2*y**2)
    return val

def by(x,y,case):
    if case==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if case==2:
       val=-(1-3*x-2*x**3*y)
    return val

def velocity_x(x,y,case):
    if case==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if case==2:
       val=x + x**2 - 2*x*y + x**3 -3*x*y**2 + x**2*y
    return val

def velocity_y(x,y,case):
    if case==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if case==2:
       val = -y - 2*x*y +y**2 - 3*x**2*y + y**3 -x*y**2
    return val

def pressure(x,y,case):
    if case==1:
       val=x*(1.-x)-1./6.
    if case==2:
       val = x*y+x+y+x**3*y**2 -4./3.
    return val

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element
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
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemV=nnp*ndofV   # number of velocity dofs
NfemP=nnp*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

case=2
viscosity=1.  # dynamic viscosity 
pnormalise=True

eps=1.e-10
sqrt3=np.sqrt(3.)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

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

icon =np.zeros((m, nel),dtype=np.int16)
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

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, nnp):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i],case)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i],case)
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i],case)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i],case)
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i],case)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i],case)
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i],case)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i],case)

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT -C][p] [h]
#################################################################
start = time.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix G
C_mat = np.zeros((NfemP,NfemP),dtype=np.float64) # matrix C
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
b_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)             # shape functions
dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)           # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)           # y-component velocity
p     = np.zeros(nnp,dtype=np.float64)           # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

Navrg = np.zeros(m,dtype=np.float64)
Nvect = np.zeros((1,m),dtype=np.float64)
N_mat = np.zeros((3,m),dtype=np.float64)

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m*ndofV),dtype=np.float64)
    K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
    G_el=np.zeros((m*ndofV,m*ndofP),dtype=np.float64)
    C_el=np.zeros((m*ndofP,m*ndofP),dtype=np.float64)
    h_el=np.zeros((m*ndofP),dtype=np.float64)

    #compute Navrg 
    Navrg[0]=0.25#/(Lx*Ly/nel)
    Navrg[1]=0.25#/(Lx*Ly/nel)
    Navrg[2]=0.25#/(Lx*Ly/nel)
    Navrg[3]=0.25#/(Lx*Ly/nel)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb = np.zeros((2, 2),dtype=np.float64)
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
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,m):
                f_el[ndofV*i  ]+=N[i]*jcob*weightq*bx(xq,yq,case)
                f_el[ndofV*i+1]+=N[i]*jcob*weightq*by(xq,yq,case)

            # compute G_el matrix
            for i in range(0,m):
                N_mat[0,i]=N[i]
                N_mat[1,i]=N[i]
                N_mat[2,i]=0
            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            # compute C_el matrix
            Nvect[0,0:m]=N[0:m]-Navrg[0:m]
            C_el+=Nvect.T.dot(Nvect)*jcob*weightq/viscosity

    # impose b.c. 
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1  # from 0 to 7 
            m1 =ndofV*icon[k1,iel]+i1
            # assemble K block
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            # assemble f vector 
            f_rhs[m1]+=f_el[ikk]
            # assemble G block
            for k2 in range(0,m):
                m2 = icon[k2,iel]
                jkk=k2                       # from 0 to 3
                G_mat[m1,m2]+=G_el[ikk,jkk]
        for k2 in range(0,m):
            C_mat[icon[k1,iel],icon[k2,iel]]+=C_el[k1,k2] 

    for k2 in range(0,m):
        m2=icon[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=N[k2]

print("     -> K (m,M) %.4e %.4e " %(np.min(K_mat),np.max(K_mat)))
print("     -> G (m,M) %.4e %.4e " %(np.min(G_mat),np.max(G_mat)))
print("     -> C (m,M) %.4e %.4e " %(np.min(C_mat),np.max(C_mat)))

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

if pnormalise:
   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   a_mat[NfemV:Nfem,NfemV:Nfem]=-C_mat
   a_mat[Nfem,NfemV:Nfem]=constr
   a_mat[NfemV:Nfem,Nfem]=constr
else:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   a_mat[NfemV:Nfem,NfemV:Nfem]=-C_mat

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (time.time() - start))

######################################################################
# solve system
######################################################################
start = time.time()

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

######################################################################
# compute strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    N[0:m]=NNV(rq,sq)
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)

    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
    jcob=np.linalg.det(jcb)
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

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

np.savetxt('p.ascii',np.array([x,y,p]).T,header='# x,y,p')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute error
######################################################################
start = time.time()

error_u = np.empty(nnp,dtype=np.float64)
error_v = np.empty(nnp,dtype=np.float64)
error_p = np.empty(nnp,dtype=np.float64)

for i in range(0,nnp): 
    error_u[i]=u[i]-velocity_x(x[i],y[i],case)
    error_v[i]=v[i]-velocity_y(x[i],y[i],case)
    error_p[i]=p[i]-pressure(x[i],y[i],case)

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)
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
            pq=0.0
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
                pq+=N[k]*p[icon[k,iel]]
            errv+=((uq-velocity_x(xq,yq,case))**2+(vq-velocity_y(xq,yq,case))**2)*weightq*jcob
            errp+=(pq-pressure(xq,yq,case))**2*weightq*jcob

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################

if visu==1:

       filename = 'solution.vtu'
       vtufile=open(filename,"w")
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
           vtufile.write("%10e\n" % e[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %( velocity_x(x[i],y[i],case) , velocity_y(x[i],y[i],case) , 0))
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='p' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % p[i])
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='p (th)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % pressure(x[i],y[i],case) )
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='error(u)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % error_u[i])
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='error(v)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % error_v[i])
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='error(p)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % error_p[i])
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

print("generate vtu: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
