import sys 
import numpy as np
import time as timing
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

###############################################################################
#   Vspace=Q2     Pspace=Q1       
#
#  3----6----2   3---------2  
#  |    |    |   |         |  
#  |    |    |   |         |  
#  7----8----5   |         |  
#  |    |    |   |         |  
#  |    |    |   |         |  
#  0----4----1   0---------1  
#
###############################################################################

def NNV(rq,sq):
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8],dtype=np.float64)

def dNNVdr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

def dNNVds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)

###############################################################################

def viscosity(x,y,Ly,eta_um,eta_c,eta_o,xA,yB,xC,xE,yE,xF,yF,yG,yI):

    if y<yI:
       val=1e21
    elif y<yG:
       val=5e22
    else:
       val=eta_um

    if x>xA and x<xC and y>yB:
       val=eta_c

    if x>xC and x<=xE and y>=yK:
       val=eta_o
    if x>xE and y>(yF-yE)/(xF-xE)*(x-xE)+yE:
       val=eta_o

    return val

###############################################################################

def density(x,y,yB,xL,xM,xN,rhod,rhou,rho0):
    val=rho0
    if x>xL and x<xM and y<yB:
       val=rhod
    if x>xN and y<Ly-30e3:
       val=rhou
    return val

###############################################################################

ndim=2
year=365.25*3600*24
cm=0.01
km=1e3
eps=1e-6

print("-----------------------------")
print("--------- stone 143 ---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

eta_ref=1e22 # numerical parameter for FEM

Lx=6000*km
Ly=3000*km

gy=-9.81

# allowing for argument parsing through command line
if int(len(sys.argv) == 7): 
   um = int(sys.argv[1])
   c = int(sys.argv[2])
   o = int(sys.argv[3])
   Fu = float(sys.argv[4])
   Fd = float(sys.argv[5])
   nelx=int(sys.argv[6])
   eta_um=10.**um
   eta_c=10.**c
   eta_o=10.**o
   Fu*=1e13
   Fd*=1e13
else:
   eta_um=1e20
   eta_c=1e22 
   eta_o=1e23
   Fu=1e13
   Fd=1e13
   nelx=300

###############################################################################

nely=int(nelx*Ly/Lx)
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

NV=nnx*nny    # number of nodes
nel=nelx*nely # number of elements, total
NP=(nelx+1)*(nely+1)

NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

hx=Lx/nelx # size of element
hy=Ly/nely

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("hx=",hx)
print("hy=",hy)
print("------------------------------")

compute_area=False

###############################################################################

lr=500*km
lc=2000*km

L=0*km

xA=L   ; yA=Ly
xB=xA        ; yB=2800*km
xC=xA+lc     ; yC=Ly
xD=xB+lc     ; yD=yB
xE=Lx-lr     ; yE=Ly-100*km
xF=Lx-hx     ; yF=Ly
xG=0         ; yG=Ly-660*km
xH=Lx        ; yH=Ly-660*km
xI=0         ; yI=350*km
xJ=Lx        ; yJ=350*km
xK=xD        ; yK=yE
xL=xA        ; yL=0
xM=xL+100*km ; yM=0
xN=Lx-100*km ; yN=0

###############################################################################

Au=100e3*Ly
Ad=100e3*yB
rho_ref=3250. 
rho_u=rho_ref-Fu/(abs(gy)*Au)#-3250 
rho_d=rho_ref+Fd/(abs(gy)*Ad)#-3250 
#rho_ref-=3250

print('rho_ref',rho_ref)
print('rho_u',rho_u)
print('rho_d',rho_d)
print('eta_um',eta_um)
print('eta_o',eta_o)
print('eta_c',eta_c)
print('Fu',Fu)
print('Fd',Fd)

###############################################################################
# integration coeffs and weights 
###############################################################################

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

###############################################################################
# grid point setup
###############################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates
top=np.empty(NV,dtype=bool)  

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        top[counter]= (j==nny-1)
        counter += 1
    #end for
#end for

iconV=np.zeros((mV,nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV]).T) 

print("velocity grid points: %.3f s" % (timing.time() - start))

###############################################################################
# pressure connectivity array
###############################################################################
start = timing.time()

xP=np.zeros(NP,dtype=np.float64)     # x coordinates
yP=np.zeros(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

counter = 0
for j in range(0, nely+1):
    for i in range(0, nelx+1):
        xP[counter]=i*Lx/float(nelx)
        yP[counter]=j*Ly/float(nely)
        counter += 1

#np.savetxt('gridP.ascii',np.array([xP,yP]).T) 

print("pressure connectivity & nodes: %.3f s" % (timing.time() - start))

###############################################################################
# compute element center coordinates
###############################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]= (xP[iconP[0,iel]]+xP[iconP[2,iel]])/2
    yc[iel]= (yP[iconP[0,iel]]+yP[iconP[2,iel]])/2

print("     -> xc (m,M) %.6e %.6e " %(np.min(xc),np.max(xc)))
print("     -> yc (m,M) %.6e %.6e " %(np.min(yc),np.max(yc)))

#np.savetxt('grid_centers.ascii',np.array([xc,yc]).T) 

print("compute element center coords: %.3f s" % (timing.time() - start))

###############################################################################
# assign viscosity to elements
###############################################################################
start = timing.time()

eta=np.zeros(nel,dtype=np.float64)  
rho=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    eta[iel]=viscosity(xc[iel],yc[iel],Ly,eta_um,eta_c,eta_o,xA,yB,xC,xE,yE,xF,yF,yG,yI)
    rho[iel]=density(xc[iel],yc[iel],yB,xL,xM,xN,rho_d,rho_u,rho_ref)

print("     -> rho (m,M) %.6e %.6e " %(np.min(rho),np.max(rho)))
print("     -> eta (m,M) %.6e %.6e " %(np.min(eta),np.max(eta)))

np.savetxt('viscosity.ascii',np.array([xc,yc,np.log10(eta),rho]).T) 

print("assign density,viscosity: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
###############################################################################
start = timing.time()

if compute_area:
   area=np.zeros(nel,dtype=np.float64) 
   NNNV=np.zeros(mV,dtype=np.float64)       
   dNNNVdr=np.zeros(mV,dtype=np.float64)   
   dNNNVds= np.zeros(mV,dtype=np.float64) 

   for iel in range(0,nel):
       for iq in range(0,nqperdim):
           for jq in range(0,nqperdim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV[0:mV]=NNV(rq,sq)
               dNNNVdr[0:mV]=dNNVdr(rq,sq)
               dNNNVds[0:mV]=dNNVds(rq,sq)
               jcb=np.zeros((2,2),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               jcob = np.linalg.det(jcb)
               area[iel]+=jcob*weightq
           #end for
       #end for
   #end for

   print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
   print("     -> total area meas %.8e " %(area.sum()))
   print("     -> total area anal %.8e " %(Lx*Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

######################################################################
# define boundary conditions
######################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)
bc_val=np.zeros(NfemV,dtype=np.float64)

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0. 
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0. 
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. 
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. 

print("define bc: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
f_rhs = np.zeros(NfemV,dtype=np.float64)            # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)            # right hand side h 
b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)     # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64)     # matrix N 
NNNV = np.zeros(mV,dtype=np.float64)                # shape functions V
NNNP = np.zeros(mP,dtype=np.float64)                # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
u = np.zeros(NV,dtype=np.float64)                   # x-component velocity
v = np.zeros(NV,dtype=np.float64)                   # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64)
           
#only ok if elements are rectangles 
jcbi=np.zeros((2,2),dtype=np.float64)
jcob=hx*hy/4
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy

for iel in range(0,nel):

    f_el=np.zeros((mV*ndofV),dtype=np.float64)
    K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)

            # calculate jacobian matrix
            #jcb=np.zeros((2,2),dtype=np.float64)
            #for k in range(0,mV):
            #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #jcob = np.linalg.det(jcb)
            #jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            # construct b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta[iel]*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rho[iel]

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        # end for jq
    # end for iq

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
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
        #end for
    #end for

    G_el*=eta_ref/Ly
    h_el*=eta_ref/Ly

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    A_sparse[m1,m2] += K_el[ikk,jkk]
                #end for
            #end for
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
            #end for
            f_rhs[m1]+=f_el[ikk]
        #end for
    #end for
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        #constr[m2]+=NNNNP[k2]
        #if pnormalise:
        #   A_sparse[Nfem,NfemV+m2]+=constr[m2]
        #   A_sparse[NfemV+m2,Nfem]+=constr[m2]
    #end for
#end for iel

print("     build FE matrix: %.3f s" % (timing.time() - start))

######################################################################
# solve system
######################################################################
start = timing.time()

rhs = np.zeros(Nfem,dtype=np.float64)
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

sol=sps.linalg.spsolve(sps.csr_matrix(A_sparse),rhs)

print("solve time: %.3f s" % (timing.time() - start))


######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %.4f %.4f " %(np.min(u/cm*year),np.max(u/cm*year)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v/cm*year),np.max(v/cm*year)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("split vel into u,v: %.3f s" % (timing.time() - start))

######################################################################
# normalise pressure
######################################################################
start = timing.time()
            
jcob=hx*hy/4

avrg_p=0
for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNP[0:mP]=NNP(rq,sq)
            pq=NNNP.dot(p[iconP[0:mP,iel]])
            avrg_p+=pq*jcob*weightq

print('          -> avrg_p=',avrg_p)

p-=avrg_p/Lx/Ly

print("          -> p (m,M) %.4e %.4e (Pa)     " %(np.min(p),np.max(p)))

#np.savetxt('pressure_aft.ascii',np.array([xP,yP,p]).T,header='# x,y,p')
            
print("pressure normalisation: %.3f s" % (timing.time() - start))

######################################################################
# project pressure onto Q2 mesh for plotting
######################################################################
start = timing.time()

q=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    q[iconV[0,iel]]=p[iconP[0,iel]]
    q[iconV[1,iel]]=p[iconP[1,iel]]
    q[iconV[2,iel]]=p[iconP[2,iel]]
    q[iconV[3,iel]]=p[iconP[3,iel]]
    q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
    q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
    q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])*0.25

print("project pressure onto V grid: %.3f s" % (timing.time() - start))


######################################################################
# compute nodal strainrate 
######################################################################
start = timing.time()

rVnodes=[-1,+1,+1,-1, 0,+1, 0,-1,0]
sVnodes=[-1,-1,+1,+1,-1,0,+1,0,0]
 
exx = np.zeros(NV,dtype=np.float64)  
eyy = np.zeros(NV,dtype=np.float64)  
exy = np.zeros(NV,dtype=np.float64)  
ee  = np.zeros(NV,dtype=np.float64)  
ccc = np.zeros(NV,dtype=np.float64)  
 
for iel in range(0,nel):
    for k in range(0,mV):
        rq = rVnodes[k]
        sq = sVnodes[k]
        inode=iconV[k,iel]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        #jcb=np.zeros((2,2),dtype=np.float64)
        #for k in range(0,mV):
        #    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        #    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        #    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        #    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        #jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        ccc[inode]+=1
        exx[inode]+=dNNNVdx.dot(u[iconV[:,iel]])
        eyy[inode]+=dNNNVdy.dot(v[iconV[:,iel]])
        exy[inode]+=dNNNVdx.dot(v[iconV[:,iel]])*0.5+dNNNVdy.dot(u[iconV[:,iel]])*0.5
    #end for
#end for
exx[:]/=ccc[:]
eyy[:]/=ccc[:]
exy[:]/=ccc[:]
ee[:]=np.sqrt(0.5*(exx[:]*exx[:]+eyy[:]*eyy[:])+exy[:]*exy[:])
 
print("          -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("          -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("          -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
 
print("compute strain rate: %.3f s" % (timing.time() - start))

######################################################################
# compute vrms 
######################################################################
start = timing.time()

vrms=0.
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[0:9]=NNV(rq,sq)
            dNNNVdr[0:9]=dNNVdr(rq,sq)
            dNNNVds[0:9]=dNNVds(rq,sq)

            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)

            uq=0.0
            vq=0.0
            for k in range(0,mV):
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
            #end for
            vrms+=(uq**2+vq**2)*weightq*jcob 

        #end for jq
    #end for iq
#end for iel

vrms=np.sqrt(vrms/Lx/Ly)

print("     -> nel= %6d ; vrms= %e " %(nel,vrms/cm*year))

print("compute vrms: %.3f s" % (timing.time() - start))

#####################################################################
# plot of solution
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 
#####################################################################
start = timing.time()

np.savetxt('solution_surface.ascii',np.array([xV[top],u[top],\
                                    q[top],exx[top],ee[top]]).T,header='# x,u,p,exx,e')

if True:
    filename = 'solution.vtu'
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    if compute_area:
       vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (area[iel]))
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (rho[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eta[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % ((p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3   ))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %exx[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %eyy[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %exy[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2+1]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")

    vtufile.write("</PointData>\n")
    #####

    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],\
                                                    iconV[5,iel],iconV[6,iel],iconV[7,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*8))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %23)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    print("write data: %.3fs" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
