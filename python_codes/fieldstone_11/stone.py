import numpy as np
import sys as sys
import scipy.sparse as sps
import time as clock 
import matplotlib.pyplot as plt

###############################################################################

def density(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
       val=2.
    else:
       val=1.
    return val

def viscosity(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
       val=1.e2
    else:
       val=1.
    return val

###############################################################################

def NNV(rq,sq,tq):
    N_0=0.125*(1-rq)*(1-sq)*(1-tq)
    N_1=0.125*(1+rq)*(1-sq)*(1-tq)
    N_2=0.125*(1+rq)*(1+sq)*(1-tq)
    N_3=0.125*(1-rq)*(1+sq)*(1-tq)
    N_4=0.125*(1-rq)*(1-sq)*(1+tq)
    N_5=0.125*(1+rq)*(1-sq)*(1+tq)
    N_6=0.125*(1+rq)*(1+sq)*(1+tq)
    N_7=0.125*(1-rq)*(1+sq)*(1+tq)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7],dtype=np.float64)

def dNNVdr(rq,sq,tq):
    dNdr_0=-0.125*(1-sq)*(1-tq) 
    dNdr_1=+0.125*(1-sq)*(1-tq)
    dNdr_2=+0.125*(1+sq)*(1-tq)
    dNdr_3=-0.125*(1+sq)*(1-tq)
    dNdr_4=-0.125*(1-sq)*(1+tq)
    dNdr_5=+0.125*(1-sq)*(1+tq)
    dNdr_6=+0.125*(1+sq)*(1+tq)
    dNdr_7=-0.125*(1+sq)*(1+tq)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7],dtype=np.float64)

def dNNVds(rq,sq,tq):
    dNds_0=-0.125*(1-rq)*(1-tq) 
    dNds_1=-0.125*(1+rq)*(1-tq)
    dNds_2=+0.125*(1+rq)*(1-tq)
    dNds_3=+0.125*(1-rq)*(1-tq)
    dNds_4=-0.125*(1-rq)*(1+tq)
    dNds_5=-0.125*(1+rq)*(1+tq)
    dNds_6=+0.125*(1+rq)*(1+tq)
    dNds_7=+0.125*(1-rq)*(1+tq)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7],dtype=np.float64)

def dNNVdt(rq,sq,tq):
    dNdt_0=-0.125*(1-rq)*(1-sq) 
    dNdt_1=-0.125*(1+rq)*(1-sq)
    dNdt_2=-0.125*(1+rq)*(1+sq)
    dNdt_3=-0.125*(1-rq)*(1+sq)
    dNdt_4=+0.125*(1-rq)*(1-sq)
    dNdt_5=+0.125*(1+rq)*(1-sq)
    dNdt_6=+0.125*(1+rq)*(1+sq)
    dNdt_7=+0.125*(1-rq)*(1+sq)
    return np.array([dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7],dtype=np.float64)

###############################################################################

print("-----------------------------")
print("--------- stone 11 ----------")
print("-----------------------------")

m=8      # number of nodes making up an element
ndofV=3  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
else:
   nelx =16   # do not exceed ~20 
   nely =nelx
   nelz =nelx
#end if

gx=0
gy=0
gz=-1

visu=1

pnormalise=True

nel=nelx*nely*nelz # number of elements, total
nnx=nelx+1         # number of nodes, x direction
nny=nely+1         # number of nodes, y direction
nnz=nelz+1         # number of nodes, z direction
NV=nnx*nny*nnz     # number of velocity nodes
NP=nel             # number of pressure nodes
NfemV=NV*ndofV     # number of velocity dofs
NfemP=NP*ndofP     # number of pressure dofs
Nfem=NfemV+NfemP   # total number of dofs

eps=1.e-10
sqrt3=np.sqrt(3.)

###############################################################################
#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("NV=",NV)
print("NP=",NP)
print("Nfem=",Nfem)
print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates
z=np.empty(NV,dtype=np.float64)  # z coordinates

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

print("grid points setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon =np.zeros((m, nel),dtype=np.int32)
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

print("build connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
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

print("define b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat = np.zeros((6,ndofV*m),dtype=np.float64)   # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)             # shape functions
dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdz  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdt  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)            # x-component velocity
v     = np.zeros(NV,dtype=np.float64)            # y-component velocity
w     = np.zeros(NV,dtype=np.float64)            # z-component velocity
p     = np.zeros(nel,dtype=np.float64)           # pressure 
jcb=np.zeros((3,3),dtype=np.float64)

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
                jcb[0,0]=dNdr[:].dot(x[icon[:,iel]])
                jcb[0,1]=dNdr[:].dot(y[icon[:,iel]])
                jcb[0,2]=dNdr[:].dot(z[icon[:,iel]])
                jcb[1,0]=dNds[:].dot(x[icon[:,iel]])
                jcb[1,1]=dNds[:].dot(y[icon[:,iel]])
                jcb[1,2]=dNds[:].dot(z[icon[:,iel]])
                jcb[2,0]=dNdt[:].dot(x[icon[:,iel]])
                jcb[2,1]=dNdt[:].dot(y[icon[:,iel]])
                jcb[2,2]=dNdt[:].dot(z[icon[:,iel]])

                # calculate the determinant of the jacobian
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi = np.linalg.inv(jcb)

                # compute coords of quad point
                xq=N[:].dot(x[icon[:,iel]])
                yq=N[:].dot(y[icon[:,iel]])
                zq=N[:].dot(z[icon[:,iel]])

                # compute dNdx, dNdy, dNdz
                dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]+jcbi[0,2]*dNdt[:]
                dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]+jcbi[1,2]*dNdt[:]
                dNdz[:]=jcbi[2,0]*dNdr[:]+jcbi[2,1]*dNds[:]+jcbi[2,2]*dNdt[:]

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                             [0.     ,dNdy[i],0.     ],
                                             [0.     ,0.     ,dNdz[i]],
                                             [dNdy[i],dNdx[i],0.     ],
                                             [dNdz[i],0.     ,dNdx[i]],
                                             [0.     ,dNdz[i],dNdy[i]]]
                #end for

                K_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq,zq)*wq*jcob

                for i in range(0, m):
                    f_el[ndofV*i+0]+=N[i]*jcob*wq*density(xq,yq,zq)*gx
                    f_el[ndofV*i+1]+=N[i]*jcob*wq*density(xq,yq,zq)*gy
                    f_el[ndofV*i+2]+=N[i]*jcob*wq*density(xq,yq,zq)*gz
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
    h_rhs[iel]+=h_el[0,0]

#end for iel

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start=clock.time()

if pnormalise:
   A_fem = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   b_fem = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   A_fem[0:NfemV,0:NfemV]=K_mat
   A_fem[0:NfemV,NfemV:Nfem]=G_mat
   A_fem[NfemV:Nfem,0:NfemV]=G_mat.T
   A_fem[Nfem,NfemV:Nfem]=1
   A_fem[NfemV:Nfem,Nfem]=1
else:
   A_fem = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   b_fem = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   A_fem[0:NfemV,0:NfemV]=K_mat
   A_fem[0:NfemV,NfemV:Nfem]=G_mat
   A_fem[NfemV:Nfem,0:NfemV]=G_mat.T
#end if

#I tried this, it makes minimum difference
#del K_mat
#del G_mat

b_fem[0:NfemV]=f_rhs
b_fem[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))

#plt.spy(A_fem)
#plt.savefig('matrix.pdf', bbox_inches='tight')

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

u,v,w=np.reshape(sol[0:NfemV],(NV,3)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.6f %.6f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.6f %.6f " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.6f %.6f " %(np.min(w),np.max(w)))
print("     -> p (m,M) %.6f %.6f " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

#np.savetxt('velocity.ascii',np.array([x,y,z,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate 
###############################################################################
start=clock.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
zc = np.zeros(nel,dtype=np.float64)  
sr = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
ezz = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
exz = np.zeros(nel,dtype=np.float64)  
eyz = np.zeros(nel,dtype=np.float64)  
eta = np.zeros(nel,dtype=np.float64)  
rho = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    tq=0.
    wq=2.*2.*2.

    N[0:8]=NNV(rq,sq,tq)
    dNdr[0:8]=dNNVdr(rq,sq,tq)
    dNds[0:8]=dNNVds(rq,sq,tq)
    dNdt[0:8]=dNNVdt(rq,sq,tq)

    jcb[0,0]=dNdr[:].dot(x[icon[:,iel]])
    jcb[0,1]=dNdr[:].dot(y[icon[:,iel]])
    jcb[0,2]=dNdr[:].dot(z[icon[:,iel]])
    jcb[1,0]=dNds[:].dot(x[icon[:,iel]])
    jcb[1,1]=dNds[:].dot(y[icon[:,iel]])
    jcb[1,2]=dNds[:].dot(z[icon[:,iel]])
    jcb[2,0]=dNdt[:].dot(x[icon[:,iel]])
    jcb[2,1]=dNdt[:].dot(y[icon[:,iel]])
    jcb[2,2]=dNdt[:].dot(z[icon[:,iel]])
    jcbi=np.linalg.inv(jcb)

    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]+jcbi[0,2]*dNdt[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]+jcbi[1,2]*dNdt[:]
    dNdz[:]=jcbi[2,0]*dNdr[:]+jcbi[2,1]*dNds[:]+jcbi[2,2]*dNdt[:]

    xc[iel]=N[:].dot(x[icon[:,iel]])
    yc[iel]=N[:].dot(y[icon[:,iel]])
    zc[iel]=N[:].dot(z[icon[:,iel]])

    exx[iel]=dNdx[:].dot(u[icon[:,iel]])
    eyy[iel]=dNdy[:].dot(v[icon[:,iel]])
    ezz[iel]=dNdz[:].dot(w[icon[:,iel]])
    exy[iel]=0.5*dNdy[:].dot(u[icon[:,iel]])+0.5*dNdx[:].dot(v[icon[:,iel]])
    exz[iel]=0.5*dNdz[:].dot(u[icon[:,iel]])+0.5*dNdx[:].dot(w[icon[:,iel]])
    eyz[iel]=0.5*dNdz[:].dot(v[icon[:,iel]])+0.5*dNdy[:].dot(w[icon[:,iel]])

    eta[iel]=viscosity(xc[iel],yc[iel],zc[iel])
    rho[iel]=density(xc[iel],yc[iel],zc[iel])
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
                        +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])

#end for iel

print("     -> exx (m,M) %.6f %.6f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.6f %.6f " %(np.min(eyy),np.max(eyy)))
print("     -> ezz (m,M) %.6f %.6f " %(np.min(ezz),np.max(ezz)))
print("     -> exy (m,M) %.6f %.6f " %(np.min(exy),np.max(exy)))
print("     -> exz (m,M) %.6f %.6f " %(np.min(exz),np.max(exz)))
print("     -> eyz (m,M) %.6f %.6f " %(np.min(eyz),np.max(eyz)))
print("     -> eta (m,M) %.6f %.6f " %(np.min(eta),np.max(eta)))
print("     -> rho (m,M) %.6f %.6f " %(np.min(rho),np.max(rho)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
#np.savetxt('p.ascii',np.array([xc,yc,zc,p]).T,header='# xc,yc,p')

print("compute strainrate: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
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
       vtufile.write("%f\n" % eta[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % rho[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % sr[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f %f %f %f %f %f\n" % (exx[iel],eyy[iel],ezz[iel],exy[iel],eyz[iel],exz[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
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
   print("export to vtu: %.3f s" % (clock.time()-start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
