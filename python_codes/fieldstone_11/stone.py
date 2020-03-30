import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def rho(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
       val=2.
    else:
       val=1.
    return val

def mu(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
       val=1.e2
    else:
       val=1.
    return val

#------------------------------------------------------------------------------

def NNV(rq,sq,tq):
    N_0=0.125*(1-rq)*(1-sq)*(1-tq)
    N_1=0.125*(1+rq)*(1-sq)*(1-tq)
    N_2=0.125*(1+rq)*(1+sq)*(1-tq)
    N_3=0.125*(1-rq)*(1+sq)*(1-tq)
    N_4=0.125*(1-rq)*(1-sq)*(1+tq)
    N_5=0.125*(1+rq)*(1-sq)*(1+tq)
    N_6=0.125*(1+rq)*(1+sq)*(1+tq)
    N_7=0.125*(1-rq)*(1+sq)*(1+tq)
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7

def dNNVdr(rq,sq,tq):
    dNdr_0=-0.125*(1-sq)*(1-tq) 
    dNdr_1=+0.125*(1-sq)*(1-tq)
    dNdr_2=+0.125*(1+sq)*(1-tq)
    dNdr_3=-0.125*(1+sq)*(1-tq)
    dNdr_4=-0.125*(1-sq)*(1+tq)
    dNdr_5=+0.125*(1-sq)*(1+tq)
    dNdr_6=+0.125*(1+sq)*(1+tq)
    dNdr_7=-0.125*(1+sq)*(1+tq)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7

def dNNVds(rq,sq,tq):
    dNds_0=-0.125*(1-rq)*(1-tq) 
    dNds_1=-0.125*(1+rq)*(1-tq)
    dNds_2=+0.125*(1+rq)*(1-tq)
    dNds_3=+0.125*(1-rq)*(1-tq)
    dNds_4=-0.125*(1-rq)*(1+tq)
    dNds_5=-0.125*(1+rq)*(1+tq)
    dNds_6=+0.125*(1+rq)*(1+tq)
    dNds_7=+0.125*(1-rq)*(1+tq)
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7

def dNNVdt(rq,sq,tq):
    dNdt_0=-0.125*(1-rq)*(1-sq) 
    dNdt_1=-0.125*(1+rq)*(1-sq)
    dNdt_2=-0.125*(1+rq)*(1+sq)
    dNdt_3=-0.125*(1-rq)*(1+sq)
    dNdt_4=+0.125*(1-rq)*(1-sq)
    dNdt_5=+0.125*(1+rq)*(1-sq)
    dNdt_6=+0.125*(1+rq)*(1+sq)
    dNdt_7=+0.125*(1-rq)*(1+sq)
    return dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------fieldstone 11--------")
print("-----------------------------")

m=8      # number of nodes making up an element
ndofV=3  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 
assert (Lz>0.), "Lz should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
else:
   nelx =20   # do not exceed 20 
   nely =nelx
   nelz =nelx
#end if

gx=0
gy=0
gz=-1

visu=1

pnormalise=True
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

nnp=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

NfemV=nnp*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eps=1.e-10

sqrt3=np.sqrt(3.)

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("nnp=",nnp)
print("------------------------------")

######################################################################
# grid point setup
######################################################################
start = time.time()

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates
z = np.empty(nnp, dtype=np.float64)  # z coordinates

counter=0
for i in range(0, nnx):
    for j in range(0, nny):
        for k in range(0, nnz):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            z[counter]=k*Lz/float(nelz)
            counter += 1
        #end for
    #end for
#end for

print("grid points setup: %.3f s" % (time.time() - start))

######################################################################
# connectivity
######################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for i in range(0, nelx):
    for j in range(0, nely):
        for k in range(0, nelz):
            icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("build connectivity: %.3f s" % (time.time() - start))

######################################################################
# define boundary conditions
######################################################################
start = time.time()

bc_fix=np.zeros(Nfem,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=float)  # boundary condition, value

for i in range(0,nnp):
    if x[i]<eps:
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=0
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=0
    if y[i]<eps:
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=0
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=0
    if z[i]<eps:
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=0
    if z[i]>(Lz-eps):
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=0 
#end for

print("define b.c.: %.3f s" % (time.time() - start))

######################################################################
# build FE matrix
######################################################################
start = time.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 

b_mat = np.zeros((6,ndofV*m),dtype=np.float64)   # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdz  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdt  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
w     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
p     = np.zeros(nel,dtype=np.float64)          # y-component velocity
c_mat = np.zeros((6,6),dtype=np.float64) 

c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m*ndofV),dtype=np.float64)
    K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
    G_el=np.zeros((m*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:
            for kq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                wq=1.*1.*1.

                # calculate shape functions
                N[0:8]=NNV(rq,sq,tq)
                dNdr[0:8]=dNNVdr(rq,sq,tq)
                dNds[0:8]=dNNVds(rq,sq,tq)
                dNdt[0:8]=dNNVdt(rq,sq,tq)

                # calculate jacobian matrix
                jcb=np.zeros((3,3),dtype=np.float64)
                for k in range(0,m):
                    jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                    jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                    jcb[0, 2] += dNdr[k]*z[icon[k,iel]]
                    jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                    jcb[1, 1] += dNds[k]*y[icon[k,iel]]
                    jcb[1, 2] += dNds[k]*z[icon[k,iel]]
                    jcb[2, 0] += dNdt[k]*x[icon[k,iel]]
                    jcb[2, 1] += dNdt[k]*y[icon[k,iel]]
                    jcb[2, 2] += dNdt[k]*z[icon[k,iel]]
                #end for

                # calculate the determinant of the jacobian
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi = np.linalg.inv(jcb)

                # compute dNdx, dNdy, dNdz
                xq=0.0
                yq=0.0
                zq=0.0
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    zq+=N[k]*z[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
                    dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
                #end for

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                             [0.     ,dNdy[i],0.     ],
                                             [0.     ,0.     ,dNdz[i]],
                                             [dNdy[i],dNdx[i],0.     ],
                                             [dNdz[i],0.     ,dNdx[i]],
                                             [0.     ,dNdz[i],dNdy[i]]]
                #end for

                K_el += b_mat.T.dot(c_mat.dot(b_mat))*mu(xq,yq,zq)*wq*jcob

                for i in range(0, m):
                    f_el[ndofV*i+0]+=N[i]*jcob*wq*rho(xq,yq,zq)*gx
                    f_el[ndofV*i+1]+=N[i]*jcob*wq*rho(xq,yq,zq)*gy
                    f_el[ndofV*i+2]+=N[i]*jcob*wq*rho(xq,yq,zq)*gz
                    G_el[ndofV*i+0,0]-=dNdx[i]*jcob*wq
                    G_el[ndofV*i+1,0]-=dNdy[i]*jcob*wq
                    G_el[ndofV*i+2,0]-=dNdz[i]*jcob*wq
                #end for

            #end for kq
        #end for jq
    #end for iq

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
               #end for
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0
            #end if
        #end for
    #end for

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
        #end for
    #end for
    h_rhs[iel]+=h_el[0]

#end for iel

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
   a_mat[Nfem,NfemV:Nfem]=1
   a_mat[NfemV:Nfem,Nfem]=1
else:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
#end if

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (time.time() - start))

plt.spy(a_mat)
plt.savefig('matrix.pdf', bbox_inches='tight')

######################################################################
# solve system
######################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v,w=np.reshape(sol[0:NfemV],(nnp,3)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.4f %.4f " %(np.min(w),np.max(w)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4es" % sol[Nfem])

np.savetxt('velocity.ascii',np.array([x,y,z,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (time.time() - start))

#####################################################################
# compute strainrate 
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
zc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
ezz = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
exz = np.zeros(nel,dtype=np.float64)  
eyz = np.zeros(nel,dtype=np.float64)  
visc = np.zeros(nel,dtype=np.float64)  
sr = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    tq=0.
    wq=2.*2.*2.

    N[0:8]=NNV(rq,sq,tq)
    dNdr[0:8]=dNNVdr(rq,sq,tq)
    dNds[0:8]=dNNVds(rq,sq,tq)
    dNdt[0:8]=dNNVdt(rq,sq,tq)

    # calculate jacobian matrix
    jcb=np.zeros((3,3),dtype=np.float64)
    for k in range(0,m):
        jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
        jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
        jcb[0, 2] += dNdr[k]*z[icon[k,iel]]
        jcb[1, 0] += dNds[k]*x[icon[k,iel]]
        jcb[1, 1] += dNds[k]*y[icon[k,iel]]
        jcb[1, 2] += dNds[k]*z[icon[k,iel]]
        jcb[2, 0] += dNdt[k]*x[icon[k,iel]]
        jcb[2, 1] += dNdt[k]*y[icon[k,iel]]
        jcb[2, 2] += dNdt[k]*z[icon[k,iel]]
    #end for
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
        dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
    #end for

    for k in range(0, m):
        xc[iel]+=N[k]*x[icon[k,iel]]
        yc[iel]+=N[k]*y[icon[k,iel]]
        zc[iel]+=N[k]*z[icon[k,iel]]
        exx[iel]+=dNdx[k]*u[icon[k,iel]]
        eyy[iel]+=dNdy[k]*v[icon[k,iel]]
        ezz[iel]+=dNdz[k]*w[icon[k,iel]]
        exy[iel]+=0.5*dNdy[k]*u[icon[k,iel]]+0.5*dNdx[k]*v[icon[k,iel]]
        exz[iel]+=0.5*dNdz[k]*u[icon[k,iel]]+0.5*dNdx[k]*w[icon[k,iel]]
        eyz[iel]+=0.5*dNdz[k]*v[icon[k,iel]]+0.5*dNdy[k]*w[icon[k,iel]]
    #end for

    visc[iel]=mu(xc[iel],yc[iel],zc[iel])
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
                    +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])

#end for iel

print("exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("ezz (m,M) %.4f %.4f " %(np.min(ezz),np.max(ezz)))
print("exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("exz (m,M) %.4f %.4f " %(np.min(exz),np.max(exz)))
print("eyz (m,M) %.4f %.4f " %(np.min(eyz),np.max(eyz)))
print("visc (m,M) %.4f %.4f " %(np.min(visc),np.max(visc)))

np.savetxt('strainrate.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
np.savetxt('p.ascii',np.array([xc,yc,zc,p]).T,header='# xc,yc,p')

print("compute strainrate: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = time.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % visc[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % sr[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f %f %f %f %f %f\n" % (exx[iel], eyy[iel], ezz[iel], exy[iel], eyz[iel], exz[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                       icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %12)
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

