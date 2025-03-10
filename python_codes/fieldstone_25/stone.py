import numpy as np
import time as time
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix 

###############################################################################
# rhs for Donea & Huerta manufactured solution, used to benchmark FE bit

def bx(x, y): 
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val 

def by(x, y): 
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val 

###############################################################################

def density(x,y,Lx):
    yinterface=0.2+0.02*np.cos(np.pi*x/Lx)
    if y<yinterface:
       val=0+1000
    else:
       val=10+1000
    return val

def viscosity(x,y,Lx):
    yinterface=0.2+0.02*np.cos(np.pi*x/Lx)
    if y<yinterface:
       val=eta_bottom
    else:
       val=100
    return val

###############################################################################
#      Q2            Q1           P-1
#
#  3----6----2   3---------2  +----2----+
#  |    |    |   |         |  |    |    |
#  |    |    |   |         |  |    |    |
#  7----8----5   |         |  |    0----1
#  |    |    |   |         |  |         |
#  |    |    |   |         |  |         |
#  0----4----1   0---------1  +---------+
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
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,\
                     NV_5,NV_6,NV_7,NV_8],dtype=np.float64)

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
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,\
                     dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

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
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,\
                     dNVds_5,dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

def NNP(rq,sq):
    if disc:
       NP_0=1-rq-sq
       NP_1=rq
       NP_2=sq
       return np.array([NP_0,NP_1,NP_2],dtype=np.float64)
    else:
       NP_0=0.25*(1-rq)*(1-sq)
       NP_1=0.25*(1+rq)*(1-sq)
       NP_2=0.25*(1+rq)*(1+sq)
       NP_3=0.25*(1-rq)*(1+sq)
       return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)

###############################################################################

eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=0.9142 # horizontal extent of the domain 
Ly=1.     # vertical extent of the domain 

gx=0
gy=-10

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   eta_bottom=float(sys.argv[4])
else:
   nelx = 32
   nely = nelx
   visu = 1
   eta_bottom=100

curved=True

disc=True

pnormalise=False

###########################################################
nel=nelx*nely # number of elements

if disc:
   NP=3*nel
   mP=3  
else:
   NP=(nelx+1)*(nely+1)
   mP=4 
    
nnx=2*nelx+1     # number of V nodes, x direction
nny=2*nely+1     # number of V nodes, y direction
NV=nnx*nny       # number of V nodes
NfemV=NV*ndofV   # number of V dofs
NfemP=NP*ndofP   # number of P dofs
Nfem=NfemV+NfemP # total number of dofs

hx=Lx/nelx
hy=Ly/nely

###########################################################

nqperdim=3

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

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("eta_bottom",eta_bottom)
print("curved=",curved)
print("p disc=",disc)
print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start = time.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

on_interface=np.zeros(NV,dtype=bool) 
jtarget=2*int(nely/5)+1 -1 
counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        yinterface=0.2+0.02*np.cos(np.pi*xV[counter]/Lx)
        if j==jtarget:
           yV[counter]=yinterface
           on_interface[counter]=True
        if j<jtarget:
           yV[counter]=yinterface*(j+1-1.)/(jtarget+1-1.)
        if j>jtarget:
           dy=(Ly-yinterface)/(nny-jtarget-1)
           yV[counter]=yinterface+dy*(j-jtarget)
        if j==nny-1:
           yV[counter]=1.
        if j==0:
           yV[counter]=0.
        counter += 1

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

if not curved: # straighten edges
   for iel in range(0,nel):
       xV[iconV[4,iel]]= (xV[iconV[0,iel]]+xV[iconV[1,iel]])/2
       yV[iconV[4,iel]]= (yV[iconV[0,iel]]+yV[iconV[1,iel]])/2
       xV[iconV[6,iel]]= (xV[iconV[2,iel]]+xV[iconV[3,iel]])/2
       yV[iconV[6,iel]]= (yV[iconV[2,iel]]+yV[iconV[3,iel]])/2
       xV[iconV[5,iel]]= (xV[iconV[1,iel]]+xV[iconV[2,iel]])/2
       yV[iconV[5,iel]]= (yV[iconV[1,iel]]+yV[iconV[2,iel]])/2
       xV[iconV[7,iel]]= (xV[iconV[0,iel]]+xV[iconV[3,iel]])/2
       yV[iconV[7,iel]]= (yV[iconV[0,iel]]+yV[iconV[3,iel]])/2
       xV[iconV[8,iel]]= (xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]]+xV[iconV[3,iel]])/4
       yV[iconV[8,iel]]= (yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]]+yV[iconV[3,iel]])/4

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("velocity grid points: %.3f s" % (time.time() - start))

###############################################################################
# pressure connectivity array
###############################################################################
start = time.time()

xP=np.zeros(NP,dtype=np.float64)     # x coordinates
yP=np.zeros(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

if disc:
   for iel in range(nel):
       iconP[0,iel]=3*iel
       iconP[1,iel]=3*iel+1
       iconP[2,iel]=3*iel+2

   NNNV = np.zeros(mV,dtype=np.float64)           # shape functions V
   counter=0
   for iel in range(nel):
       #pressure node 0
       rq=0.0
       sq=0.0
       NNNV[0:mV]=NNV(rq,sq)
       xq=NNNV[:].dot(xV[iconV[:,iel]])
       yq=NNNV[:].dot(yV[iconV[:,iel]])
       xP[counter]=xq
       yP[counter]=yq
       counter+=1
       #pressure node 1
       rq=1.0
       sq=0.0
       NNNV[0:mV]=NNV(rq,sq)
       xq=NNNV[:].dot(xV[iconV[:,iel]])
       yq=NNNV[:].dot(yV[iconV[:,iel]])
       xP[counter]=xq
       yP[counter]=yq
       counter+=1
       #pressure node 2
       rq=0.0
       sq=1.0
       NNNV[0:mV]=NNV(rq,sq)
       xq=NNNV[:].dot(xV[iconV[:,iel]])
       yq=NNNV[:].dot(yV[iconV[:,iel]])
       xP[counter]=xq
       yP[counter]=yq
       counter+=1

else:
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


#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("pressure connectivity & nodes: %.3f s" % (time.time() - start))

###############################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
###############################################################################
start = time.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

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
        if area[iel]<0: 
           for k in range(0,mV):
               print (xV[iconV[k,iel]],yV[iconV[k,iel]])
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.8e " %(area.sum()))
print("     -> total area anal %.8e " %(Lx*Ly))

print("compute elements areas: %.3f s" % (time.time() - start))

###############################################################################
# compute density on nodes and elements
###############################################################################
start = time.time()

rho_el=np.zeros(nel,dtype=np.float64) 
eta_el=np.zeros(nel,dtype=np.float64) 
NNNV=np.zeros(mV,dtype=np.float64)       

for iel in range(0,nel):
    rq=0
    sq=0
    NNNV[0:9]=NNV(rq,sq)
    xc=NNNV[:].dot(xV[iconV[:,iel]])
    yc=NNNV[:].dot(yV[iconV[:,iel]])
    rho_el[iel]=density(xc,yc,Lx)
    eta_el[iel]=viscosity(xc,yc,Lx)
#end for

#np.savetxt('rho.ascii',np.array([x,y,rho]).T,header='# x,y')

print("assign rho,eta per elt: %.3f s" % (time.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, NV):
    if xV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # dohu03
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # dohu03
    if yV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = time.time()

eta_ref=100

if pnormalise:
   A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
   rhs=np.zeros(Nfem+1,dtype=np.float64)
else:
   A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
   rhs=np.zeros(Nfem,dtype=np.float64)  

f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
constr  = np.zeros(NfemP,dtype=np.float64)        # constraint matrix/vector
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNNP= np.zeros(mP*ndofP,dtype=np.float64)   

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            NNNP=NNP(rq,sq)

            # calculate jacobian matrix,det, and inverse
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            # construct b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta_el[iel]*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*rho_el[iel]
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rho_el[iel]
                #f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                #f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            NNNNP[:]+=NNNP[:]*jcob*weightq

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
            #end if
        #end for
    #end for

    G_el*=eta_ref/Ly
    h_el*=eta_ref/Ly

    # assemble matrix and rhs
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
                A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]#*eta_ref/Ly
                A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]#*eta_ref/Ly
            #end for
            rhs[m1]+=f_el[ikk]
        #end for
    #end for
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        rhs[NfemV+m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]
        if pnormalise:
           A_sparse[Nfem,NfemV+m2]+=constr[m2]
           A_sparse[NfemV+m2,Nfem]+=constr[m2]
    #end for

#end for

print("build FE matrix: %.3f s" % (time.time() - start))

###############################################################################
# solve system
###############################################################################
start = time.time()

sol=sps.linalg.spsolve(A_sparse.tocsr(),rhs)

print("solve time: %.3f s" % (time.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %e %e " %(np.min(p),np.max(p)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
#np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (time.time() - start))

###############################################################################
# compute strainrate in the middle of each element (r=s=0) 
###############################################################################
start = time.time()

rmid=0.0
smid=0.0

e=np.zeros(nel,dtype=np.float64)  
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    NNNV=NNV(rmid,smid)
    dNNNVdr=dNNVdr(rmid,smid)
    dNNNVds=dNNVds(rmid,smid)

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

    xc[iel]=NNNV.dot(xV[iconV[:,iel]])
    yc[iel]=NNNV.dot(yV[iconV[:,iel]])
    exx[iel]=dNNNVdx.dot(u[iconV[:,iel]])
    eyy[iel]=dNNNVdy.dot(v[iconV[:,iel]])
    exy[iel]=dNNNVdx.dot(v[iconV[:,iel]])*0.5+\
             dNNNVdy.dot(u[iconV[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
#end for

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

###############################################################################
# compute vrms 
###############################################################################
start = time.time()

vrms=0.
avrg_p=0.
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            NNNP=NNP(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)
            uq=NNNV.dot(u[iconV[:,iel]])
            vq=NNNV.dot(v[iconV[:,iel]])
            vrms+=(uq**2+vq**2)*weightq*jcob
            pq=NNNP.dot(p[iconP[:,iel]])
            avrg_p+=pq*jcob*weightq
        #end for
    #end for
#end for

vrms=np.sqrt(vrms/(Lx*Ly))
avrg_p/=(Lx*Ly)

print('     -> avrg_p=',avrg_p)

p-=avrg_p

print("     -> hx= %.6e  vrms= %.7e " % (hx,vrms))

print("compute vrms: %.3f s" % (time.time() - start))

###############################################################################
# interpolate pressure onto velocity grid points
###############################################################################
start = time.time()

q=np.zeros(NV,dtype=np.float64)
counter=np.zeros(NV,dtype=np.float64)

if disc:
   rVnodes=[-1,1,1,-1,0,1,0,-1,0]
   sVnodes=[-1,-1,1,1,-1,0,1,0,0]
   for iel in range(0,nel):
       for k in range(0,mV):
          NNNP=NNP(rVnodes[k],sVnodes[k])
          pq=NNNP.dot(p[iconP[:,iel]])
          q[iconV[k,iel]]+=pq
          counter[iconV[k,iel]]+=1
       #end for
   #end for
   q[:]/=counter[:]

else:
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
   #end for
#end if 

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')
#np.savetxt('qbottom'+str(nelx)+'.ascii',np.array([xV[0:nnx],yV[0:nnx],q[0:nnx]]).T,header='# x,y,q')

print("compute q: %.3f s" % (time.time() - start))

###############################################################################
# compute pressure at the bottom of the domain
###############################################################################
start = time.time()

pbottomfile=open('pbottom'+str(nelx)+'.ascii',"w")

for iel in range(0,nelx):
    rq=-1 ;sq=-1
    NNNP=NNP(rq,sq)
    xq=NNNP.dot(xP[iconP[:,iel]])
    pq=NNNP.dot(p[iconP[:,iel]])
    pbottomfile.write("%e %e \n" %(xq,pq))
    rq=-0.5 ;sq=-1
    NNNP=NNP(rq,sq)
    xq=NNNP.dot(xP[iconP[:,iel]])
    pq=NNNP.dot(p[iconP[:,iel]])
    pbottomfile.write("%e %e \n" %(xq,pq))
    rq= 0 ; sq=-1
    NNNP=NNP(rq,sq)
    xq=NNNP.dot(xP[iconP[:,iel]])
    pq=NNNP.dot(p[iconP[:,iel]])
    pbottomfile.write("%e %e \n" %(xq,pq))
    rq=+0.5 ; sq=-1
    NNNP=NNP(rq,sq)
    xq=NNNP.dot(xP[iconP[:,iel]])
    pq=NNNP.dot(p[iconP[:,iel]])
    pbottomfile.write("%e %e \n" %(xq,pq))
    rq=+1 ; sq=-1
    NNNP=NNP(rq,sq)
    xq=NNNP.dot(xP[iconP[:,iel]])
    pq=NNNP.dot(p[iconP[:,iel]])
    pbottomfile.write("%e %e \n" %(xq,pq))

print("compute pbottom: %.3f s" % (time.time() - start))

###############################################################################

vinterface_file=open('vel_interface'+str(nelx)+'.ascii',"w")
for i in range(0,NV):
    if on_interface[i]:
       vinterface_file.write("%e %e %e \n" %(xV[i],u[i],v[i]))

###############################################################################
# print measurements later to be grep-ed by bash scripts
###############################################################################

vel=np.sqrt(u**2+v**2)
print('benchmark ',nel,Nfem,hx,\
np.min(u),np.max(u),\
np.min(v),np.max(v),\
np.min(vel),np.max(vel),\
np.min(p),np.max(p),
vrms)

###############################################################################
# export solution to vtu format for Paraview
###############################################################################
start = time.time()

filename = 'solution'+str(nelx)+'.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exx[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % eyy[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exy[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % rho_el[iel]) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % eta_el[iel]) 
vtufile.write("</DataArray>\n")
#--
if disc:
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % p[3*iel]) 
   vtufile.write("</DataArray>\n")

vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='interface' Format='ascii'> \n")
for i in range(0,NV):
    if on_interface[i]:
       vtufile.write("%e \n" % 1.)
    else:
       vtufile.write("%e \n" % 0.)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                   iconV[3,iel],iconV[4,iel],iconV[5,iel],\
                                                   iconV[6,iel],iconV[7,iel],iconV[8,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*9))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %28)
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

###############################################################################
###############################################################################
###############################################################################
