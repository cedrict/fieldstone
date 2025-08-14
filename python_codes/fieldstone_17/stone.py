import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
import time as clock
from basis_functions import *

###############################################################################
beta=0

experiment=2

if experiment==1:
   from functions_experiment1 import*
if experiment==2:
   from functions_experiment2 import*

###############################################################################

print("-----------------------------")
print("--------- stone 17 ----------")
print("-----------------------------")

mV=27    # number of velocity nodes making up an element
mP=8     # number of pressure nodes making up an element
ndofV=3  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
   visu = int(sys.argv[4])
else:
   nelx = 9
   nely = nelx
   nelz = nelx
   visu=1

nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction
nnz=2*nelz+1  # number of elements, z direction

NV=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

NfemV=NV*ndofV                        # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*(nelz+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP                       # total number of dofs

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

pnormalise=True

###############################################################################
# quadrature points and weights
###############################################################################

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("NV=",NV)
print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x=np.empty(NV,dtype=np.float64) # x coordinates
y=np.empty(NV,dtype=np.float64) # y coordinates
z=np.empty(NV,dtype=np.float64) # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x[counter]=i*hx/2
            y[counter]=j*hy/2
            z[counter]=k*hz/2
            counter += 1
        #end for
    #end for
#end for

print("grid points setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            iconV[ 0,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            iconV[ 1,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            iconV[ 2,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            iconV[ 3,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            iconV[ 4,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            iconV[ 5,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            iconV[ 6,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            iconV[ 7,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            iconV[ 8,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            iconV[ 9,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            iconV[10,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            iconV[11,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            iconV[12,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            iconV[13,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            iconV[14,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            iconV[15,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            iconV[16,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            iconV[17,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            iconV[18,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            iconV[19,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            iconV[20,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            iconV[21,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            iconV[22,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            iconV[23,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            iconV[24,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            iconV[25,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            iconV[26,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            counter += 1
        #end for
    #end for
#end for

#print ('=======iconV=======')
#for iel in range (0,nel):
#    print ("iel=",iel)
#    for i in range(0,27):
#        print ("node",i,iconV[i,iel],"at pos.",x[iconV[i,iel]],y[iconV[i,iel]],z[iconV[i,iel]])

iconP=np.zeros((mP,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            iconP[0,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j-1+1)+k  
            iconP[1,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j-1+1)+k  
            iconP[2,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j  +1)+k  
            iconP[3,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j  +1)+k  
            iconP[4,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j-1+1)+k+1  
            iconP[5,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j-1+1)+k+1  
            iconP[6,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j  +1)+k+1  
            iconP[7,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j  +1)+k+1  
            counter += 1
        #end for
    #end for
#end for

#print ('=======iconP=======')
#for iel in range (0,nel):
#    print ("iel=",iel)
#    for i in range(0,8):
#        print ("node",i,iconP[i,iel])

print("build connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

eps=1.e-10

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if x[i]<eps:
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=wth(x[i],y[i],z[i])
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=wth(x[i],y[i],z[i])
    if y[i]<eps:
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=wth(x[i],y[i],z[i])
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=wth(x[i],y[i],z[i])
    if z[i]<eps:
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=wth(x[i],y[i],z[i])
    if z[i]>(Lz-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=wth(x[i],y[i],z[i])
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
constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
b_mat = np.zeros((6,ndofV*mV),dtype=np.float64)  # gradient matrix B 
N_mat = np.zeros((6,ndofP*mP),dtype=np.float64)  # matrix  
dNVdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNVdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNVdz = np.zeros(mV,dtype=np.float64)            # shape functions derivatives

c_mat = np.zeros((6,6),dtype=np.float64) 
c_mat[0,0]=2.; c_mat[1,1]=2.; c_mat[2,2]=2.
c_mat[3,3]=1.; c_mat[4,4]=1.; c_mat[5,5]=1.

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el=np.zeros((mV*ndofV),dtype=np.float64)
    K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros(mP*ndofP,dtype=np.float64)
    NNNP=np.zeros(mP*ndofP,dtype=np.float64) # int of shape functions P

    # integrate viscous term at 3*3*3 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:
            for kq in [0,1,2]:

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]

                # calculate shape functions
                NNNV=NNV(rq,sq,tq)
                dNVdr=dNNVdr(rq,sq,tq)
                dNVds=dNNVds(rq,sq,tq)
                dNVdt=dNNVdt(rq,sq,tq)
                NP=NNP(rq,sq,tq)

                # calculate jacobian matrix
                jcb=np.zeros((3,3),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
                    jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
                    jcb[0,2] += dNVdr[k]*z[iconV[k,iel]]
                    jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
                    jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
                    jcb[1,2] += dNVds[k]*z[iconV[k,iel]]
                    jcb[2,0] += dNVdt[k]*x[iconV[k,iel]]
                    jcb[2,1] += dNVdt[k]*y[iconV[k,iel]]
                    jcb[2,2] += dNVdt[k]*z[iconV[k,iel]]
                #end for

                # calculate the determinant of the jacobian
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi = np.linalg.inv(jcb)

                # compute dNdx, dNdy, dNdz
                xq=0.0
                yq=0.0
                zq=0.0
                for k in range(0,mV):
                    xq+=NNNV[k]*x[iconV[k,iel]]
                    yq+=NNNV[k]*y[iconV[k,iel]]
                    zq+=NNNV[k]*z[iconV[k,iel]]
                    dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]+jcbi[0,2]*dNVdt[k]
                    dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]+jcbi[1,2]*dNVdt[k]
                    dNVdz[k]=jcbi[2,0]*dNVdr[k]+jcbi[2,1]*dNVds[k]+jcbi[2,2]*dNVdt[k]
                #end for

                # construct 6x24 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:6, 3*i:3*i+3] = [[dNVdx[i],0.      ,0.     ],
                                             [0.      ,dNVdy[i],0.     ],
                                             [0.      ,0.      ,dNVdz[i]],
                                             [dNVdy[i],dNVdx[i],0.     ],
                                             [dNVdz[i],0.      ,dNVdx[i]],
                                             [0.      ,dNVdz[i],dNVdy[i]]]
                #end for

                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq,zq,beta)*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i+0]-=NNNV[i]*jcob*weightq*bx(xq,yq,zq,beta)
                    f_el[ndofV*i+1]-=NNNV[i]*jcob*weightq*by(xq,yq,zq,beta)
                    f_el[ndofV*i+2]-=NNNV[i]*jcob*weightq*bz(xq,yq,zq,beta)
                #end for

                for i in range(0,mP):
                    N_mat[0,i]=NP[i]
                    N_mat[1,i]=NP[i]
                    N_mat[2,i]=NP[i]
                    N_mat[3,i]=0.
                    N_mat[4,i]=0.
                    N_mat[5,i]=0.
                #end for

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                NNNP[:]+=NP[:]*jcob*weightq

            #end for
        #end for
    #end for

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
            #end if
        #end for
    #end for

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                G_mat[m1,m2]+=G_el[ikk,jkk]
            #end for
            f_rhs[m1]+=f_el[ikk]
        #end for
    #end for
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNP[k2]
    #end for

#end for iel

print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))
print("     -> f_mat (m,M) %.4f %.4f " %(np.min(f_rhs),np.max(f_rhs)))
print("     -> h_mat (m,M) %.4f %.4f " %(np.min(h_rhs),np.max(h_rhs)))

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start=clock.time()

if pnormalise:
   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   a_mat[Nfem,NfemV:Nfem]=constr
   a_mat[NfemV:Nfem,Nfem]=constr
else:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
#end if

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v,w=np.reshape(sol[0:NfemV],(NV,3)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %e %e " %(np.min(w),np.max(w)))
print("     -> p (m,M) %e %e " %(np.min(p),np.max(p)))
    
if pnormalise:
   print("     -> Lagrange multiplier: %e" % sol[Nfem])

#np.savetxt('velocity.ascii',np.array([x,y,z,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental strainrate 
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
visc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    tq=0.

    NNNV=NNV(rq,sq,tq)
    dNVdr=dNNVdr(rq,sq,tq)
    dNVds=dNNVds(rq,sq,tq)
    dNVdt=dNNVdt(rq,sq,tq)

    # calculate jacobian matrix
    jcb=np.zeros((3,3),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
        jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
        jcb[0,2] += dNVdr[k]*z[iconV[k,iel]]
        jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
        jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
        jcb[1,2] += dNVds[k]*z[iconV[k,iel]]
        jcb[2,0] += dNVdt[k]*x[iconV[k,iel]]
        jcb[2,1] += dNVdt[k]*y[iconV[k,iel]]
        jcb[2,2] += dNVdt[k]*z[iconV[k,iel]]
    #end for 
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,mV):
        dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]+jcbi[0,2]*dNVdt[k]
        dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]+jcbi[1,2]*dNVdt[k]
        dNVdz[k]=jcbi[2,0]*dNVdr[k]+jcbi[2,1]*dNVds[k]+jcbi[2,2]*dNVdt[k]
    #end for 

    for k in range(0, mV):
        xc[iel]+=NNNV[k]*x[iconV[k,iel]]
        yc[iel]+=NNNV[k]*y[iconV[k,iel]]
        zc[iel]+=NNNV[k]*z[iconV[k,iel]]
        exx[iel]+=dNVdx[k]*u[iconV[k,iel]]
        eyy[iel]+=dNVdy[k]*v[iconV[k,iel]]
        ezz[iel]+=dNVdz[k]*w[iconV[k,iel]]
        exy[iel]+=0.5*dNVdy[k]*u[iconV[k,iel]]+0.5*dNVdx[k]*v[iconV[k,iel]]
        exz[iel]+=0.5*dNVdz[k]*u[iconV[k,iel]]+0.5*dNVdx[k]*w[iconV[k,iel]]
        eyz[iel]+=0.5*dNVdz[k]*v[iconV[k,iel]]+0.5*dNVdy[k]*w[iconV[k,iel]]
    #end for 

    visc[iel]=viscosity(xc[iel],yc[iel],zc[iel],beta)
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
                    +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])

#end for 

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> ezz (m,M) %.4e %.4e " %(np.min(ezz),np.max(ezz)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
print("     -> exz (m,M) %.4e %.4e " %(np.min(exz),np.max(exz)))
print("     -> eyz (m,M) %.4e %.4e " %(np.min(eyz),np.max(eyz)))
print("     -> visc (m,M) %.4e %.4e " %(np.min(visc),np.max(visc)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute strainrate: %.3f s" % (clock.time()-start))

###############################################################################
# interpolate pressure onto velocity grid points
###############################################################################
start=clock.time()

q=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    q[iconV[ 0,iel]]= p[iconP[0,iel]]
    q[iconV[ 1,iel]]= p[iconP[1,iel]]
    q[iconV[ 2,iel]]= p[iconP[2,iel]]
    q[iconV[ 3,iel]]= p[iconP[3,iel]]
    q[iconV[ 4,iel]]= p[iconP[4,iel]]
    q[iconV[ 5,iel]]= p[iconP[5,iel]]
    q[iconV[ 6,iel]]= p[iconP[6,iel]]
    q[iconV[ 7,iel]]= p[iconP[7,iel]]
    q[iconV[ 8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    q[iconV[ 9,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    q[iconV[10,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
    q[iconV[11,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
    q[iconV[12,iel]]=(p[iconP[4,iel]]+p[iconP[5,iel]])*0.5
    q[iconV[13,iel]]=(p[iconP[5,iel]]+p[iconP[6,iel]])*0.5
    q[iconV[14,iel]]=(p[iconP[6,iel]]+p[iconP[7,iel]])*0.5
    q[iconV[15,iel]]=(p[iconP[7,iel]]+p[iconP[4,iel]])*0.5
    q[iconV[16,iel]]=(p[iconP[0,iel]]+p[iconP[4,iel]])*0.5
    q[iconV[17,iel]]=(p[iconP[1,iel]]+p[iconP[5,iel]])*0.5
    q[iconV[18,iel]]=(p[iconP[2,iel]]+p[iconP[6,iel]])*0.5
    q[iconV[19,iel]]=(p[iconP[3,iel]]+p[iconP[7,iel]])*0.5
#end for

print("     -> q (m,M) %.4f %.4f " %(np.min(q),np.max(q)))

print("compute q : %.3f s" % (clock.time() - start))

########################################################################
# compute error in L2 norm 
#################################################################
start=clock.time()

errv=0.
errp=0.
errexx=0.
erreyy=0.
errezz=0.
errexy=0.
errexz=0.
erreyz=0.
for iel in range (0,nel):
    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:
            for kq in [0,1,2]:

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]

                # calculate shape functions
                NNNV[0:mV]=NNV(rq,sq,tq)
                dNVdr[0:mV]=dNNVdr(rq,sq,tq)
                dNVds[0:mV]=dNNVds(rq,sq,tq)
                dNVdt[0:mV]=dNNVdt(rq,sq,tq)
                NP[0:mP]=NNP(rq,sq,tq)

                # calculate jacobian matrix
                jcb=np.zeros((3,3),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
                    jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
                    jcb[0,2] += dNVdr[k]*z[iconV[k,iel]]
                    jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
                    jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
                    jcb[1,2] += dNVds[k]*z[iconV[k,iel]]
                    jcb[2,0] += dNVdt[k]*x[iconV[k,iel]]
                    jcb[2,1] += dNVdt[k]*y[iconV[k,iel]]
                    jcb[2,2] += dNVdt[k]*z[iconV[k,iel]]
                #end for

                # calculate the determinant of the jacobian
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                for k in range(0,mV):
                    dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]+jcbi[0,2]*dNVdt[k]
                    dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]+jcbi[1,2]*dNVdt[k]
                    dNVdz[k]=jcbi[2,0]*dNVdr[k]+jcbi[2,1]*dNVds[k]+jcbi[2,2]*dNVdt[k]
                #end for 

                xq=0.0
                yq=0.0
                zq=0.0
                uq=0.0
                vq=0.0
                wq=0.0
                pq=0.0
                exxq=0.0
                eyyq=0.0
                ezzq=0.0
                exyq=0.0
                exzq=0.0
                eyzq=0.0
                for k in range(0,mV):
                    xq+=NNNV[k]*x[iconV[k,iel]]
                    yq+=NNNV[k]*y[iconV[k,iel]]
                    zq+=NNNV[k]*z[iconV[k,iel]]
                    uq+=NNNV[k]*u[iconV[k,iel]]
                    vq+=NNNV[k]*v[iconV[k,iel]]
                    wq+=NNNV[k]*w[iconV[k,iel]]
                    exxq+=dNVdx[k]*u[iconV[k,iel]]
                    eyyq+=dNVdy[k]*v[iconV[k,iel]]
                    ezzq+=dNVdz[k]*w[iconV[k,iel]]
                    exyq+=0.5*dNVdy[k]*u[iconV[k,iel]]+0.5*dNVdx[k]*v[iconV[k,iel]]
                    exzq+=0.5*dNVdz[k]*u[iconV[k,iel]]+0.5*dNVdx[k]*w[iconV[k,iel]]
                    eyzq+=0.5*dNVdz[k]*v[iconV[k,iel]]+0.5*dNVdy[k]*w[iconV[k,iel]]
                #end for
                for k in range(0,mP):
                    pq+=NP[k]*p[iconP[k,iel]]
                #end for

                errv+=((uq-uth(xq,yq,zq))**2+\
                       (vq-vth(xq,yq,zq))**2+\
                       (wq-wth(xq,yq,zq))**2)*weightq*jcob

                errp+=(pq-pth(xq,yq,zq))**2*weightq*jcob

                errexx+=(exxq-exx_th(xq,yq,zq))**2*weightq*jcob
                erreyy+=(eyyq-eyy_th(xq,yq,zq))**2*weightq*jcob
                errezz+=(ezzq-ezz_th(xq,yq,zq))**2*weightq*jcob
                errexy+=(exyq-exy_th(xq,yq,zq))**2*weightq*jcob
                errexz+=(exzq-exz_th(xq,yq,zq))**2*weightq*jcob
                erreyz+=(eyzq-eyz_th(xq,yq,zq))**2*weightq*jcob

            #end for kq
        #end for jq
    #end for iq
#end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errexx=np.sqrt(errexx)
erreyy=np.sqrt(erreyy)
errezz=np.sqrt(errezz)
errexy=np.sqrt(errexy)
errexz=np.sqrt(errexz)
erreyz=np.sqrt(erreyz)

print("     -> nel= %6d ; errv: %e ; p: %e ; exx,eyy,ezz,exy,exz,eyz= %e %e %e %e %e %e"\
       %(nel,errv,errp,errexx,erreyy,errezz,errexy,errexz,erreyz))

print("compute errors: %.3f s" % (clock.time() - start))

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
       vtufile.write("%e %e %e \n" %(x[i],y[i],z[i]))
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
       vtufile.write("%e %e %e %e %e %e\n" % (exx[iel],eyy[iel],ezz[iel],exy[iel],eyz[iel],exz[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate (analytical)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e %e %e %e %e %e\n" % (exx_th(xc[iel],yc[iel],zc[iel]), \
                                              eyy_th(xc[iel],yc[iel],zc[iel]), \
                                              ezz_th(xc[iel],yc[iel],zc[iel]), \
                                              exy_th(xc[iel],yc[iel],zc[iel]), \
                                              eyz_th(xc[iel],yc[iel],zc[iel]), \
                                              exz_th(xc[iel],yc[iel],zc[iel]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate (error)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e %e %e %e %e %e\n" % (exx[iel]-exx_th(xc[iel],yc[iel],zc[iel]), \
                                              eyy[iel]-eyy_th(xc[iel],yc[iel],zc[iel]), \
                                              ezz[iel]-ezz_th(xc[iel],yc[iel],zc[iel]), \
                                              exy[iel]-exy_th(xc[iel],yc[iel],zc[iel]), \
                                              eyz[iel]-eyz_th(xc[iel],yc[iel],zc[iel]), \
                                              exz[iel]-exz_th(xc[iel],yc[iel],zc[iel]) ))
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
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (analytical)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(uth(x[i],y[i],z[i]), vth(x[i],y[i],z[i]), wth(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i]-uth(x[i],y[i],z[i]),\
                                     v[i]-vth(x[i],y[i],z[i]),\
                                     w[i]-wth(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate (analytical)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e %e %e %e %e %e\n" % (exx_th(x[i],y[i],z[i]), \
                                              eyy_th(x[i],y[i],z[i]), \
                                              ezz_th(x[i],y[i],z[i]), \
                                              exy_th(x[i],y[i],z[i]), \
                                              eyz_th(x[i],y[i],z[i]), \
                                              exz_th(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (analytical)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%f\n" % pth(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%f\n" % (q[i]-pth(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel],iconV[8,iel],iconV[9,iel],iconV[10,iel],iconV[11,iel],iconV[12,iel],iconV[13,iel],iconV[14,iel],iconV[15,iel],iconV[16,iel],iconV[17,iel],iconV[18,iel],iconV[19,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*20))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %25)

       #THIS SHOULD BE CHANGED TO 33!

   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu: %.3f s" % (clock.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

