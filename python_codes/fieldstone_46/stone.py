import numpy as np
import sys as sys
import time as timing
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix

#------------------------------------------------------------------------------

def NNV(r,s):
    NV_0= (1-r-s)*(1-2*r-2*s+ 3*r*s)
    NV_1= r*(2*r -1 + 3*s-3*r*s-3*s**2 )
    NV_2= s*(2*s -1 + 3*r-3*r**2-3*r*s )
    NV_3= 4*(1-r-s)*r*(1-3*s)
    NV_4= 4*r*s*(-2+3*r+3*s)
    NV_5= 4*(1-r-s)*s*(1-3*r)
    NV_6= 27*(1-r-s)*r*s
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6],dtype=np.float64)

def dNNVdr(r,s):
    dNdr_0= r*(4-6*s)-3*s**2+7*s-3
    dNdr_1= r*(4-6*s)-3*s**2+3*s-1
    dNdr_2= -3*s*(2*r+s-1)  
    dNdr_3= 4*(3*s-1)*(2*r+s-1) 
    dNdr_4= 4*s*(6*r+3*s-2) 
    dNdr_5= 4*s*(6*r+3*s-4)
    dNdr_6=-27*s*(2*r+s-1)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6],dtype=np.float64)

def dNNVds(r,s):
    dNds_0= -3*r**2+r*(7-6*s)+4*s-3
    dNds_1= -3*r*(r+2*s-1)
    dNds_2= -3*r**2+r*(3-6*s)+4*s-1 
    dNds_3= 4*r*(3*r+6*s-4)  
    dNds_4= 4*r*(3*r+6*s-2) 
    dNds_5= 4*(3*r-1)*(r+2*s-1)
    dNds_6= -27*r*(r+2*s-1)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6],dtype=np.float64)

def NNP(r,s):
    NP_0=1-r-s
    NP_1=r
    NP_2=s
    return np.array([NP_0,NP_1,NP_2],dtype=np.float64)

#------------------------------------------------------------------------------

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

def velocity_x(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def velocity_y(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pressure(x,y):
    val=x*(1.-x)-1./6.
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=7     # number of velocity nodes making up an element
mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.
Ly=1.

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = 32
   visu = 1

nel=nelx*nely*2
nnx=2*nelx+1
nny=2*nely+1
NV=nnx*nny+nel
NP=nel*mP

ndofV=2
ndofP=1

NfemV=NV*ndofV       # number of velocity dofs
NfemP=nel*mP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP     # total nb of dofs

print ('nnx  =',nnx)
print ('nny  =',nny)
print ('NV   =',NV)
print ('NP   =',NP)
print ('nel  =',nel)
print ('NfemV=',NfemV)
print ('NfemP=',NfemP)
print ('Nfem =',Nfem)
print("-----------------------------")

eps=1e-6

eta=1.

#---------------------------------------
# 6 point integration coeffs and weights 

nqel=6

qcoords_r=np.empty(6,dtype=np.float64)  
qcoords_s=np.empty(6,dtype=np.float64)  
qweights=np.empty(6,dtype=np.float64)  

qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 

#################################################################
# checking that all shape functions are 1 on their node and 
# zero elsewhere
#print ('node1:',NNV(0,0))
#print ('node2:',NNV(1,0))
#print ('node3:',NNV(0,1))
#print ('node4:',NNV(0.5,0))
#print ('node5:',NNV(0.5,0.5))
#print ('node6:',NNV(0,0.5))
#print ('node7:',NNV(1/3.,1/3.))

#################################################################
# build velocity nodes coordinates and connectivity array 
#################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates
iconV=np.zeros((mV,nel),dtype=np.int32)

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*Lx/(2*nelx) 
        yV[counter]=j*Ly/(2*nely) 
        counter+=1

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
          # lower left triangle
          iconV[0,counter]=(i)*2+1+(j)*2*nnx      -1  # 1 of q2
          iconV[1,counter]=(i)*2+3+(j)*2*nnx      -1  # 3 of q2
          iconV[2,counter]=(i)*2+1+(j)*2*nnx+nnx*2-1  # 7 of q2
          iconV[3,counter]=(i)*2+2+(j)*2*nnx      -1  # 2 of q2
          iconV[4,counter]=(i)*2+2+(j)*2*nnx+nnx  -1  # 5 of q2
          iconV[5,counter]=(i)*2+1+(j)*2*nnx+nnx  -1  # 4 of q2
          iconV[6,counter]=nnx*nny+counter
          counter=counter+1
          # upper right triangle
          iconV[0,counter]=(i)*2+3+(j)*2*nnx+nnx*2-1  # 9 of Q2
          iconV[1,counter]=(i)*2+1+(j)*2*nnx+nnx*2-1  # 7 of Q2
          iconV[2,counter]=(i)*2+3+(j)*2*nnx      -1  # 3 of Q2
          iconV[3,counter]=(i)*2+2+(j)*2*nnx+nnx*2-1  # 8 of Q2
          iconV[4,counter]=(i)*2+2+(j)*2*nnx+nnx  -1  # 5 of Q2
          iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx  -1  # 6 of Q2
          iconV[6,counter]=nnx*nny+counter
          counter=counter+1

for iel in range (0,nel): #bubble nodes
    xV[nnx*nny+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
    yV[nnx*nny+iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconV[0,iel],"at pos.",xV[iconV[0,iel]], yV[iconV[0,iel]])
#    print ("node 1",iconV[1,iel],"at pos.",xV[iconV[1,iel]], yV[iconV[1,iel]])
#    print ("node 2",iconV[2,iel],"at pos.",xV[iconV[2,iel]], yV[iconV[2,iel]])
#    print ("node 3",iconV[3,iel],"at pos.",xV[iconV[3,iel]], yV[iconV[3,iel]])
#    print ("node 4",iconV[4,iel],"at pos.",xV[iconV[4,iel]], yV[iconV[4,iel]])
#    print ("node 5",iconV[5,iel],"at pos.",xV[iconV[5,iel]], yV[iconV[5,iel]])
#    print ("node 6",iconV[6,iel],"at pos.",xV[iconV[6,iel]], yV[iconV[6,iel]])

#print("iconV (min/max): %d %d" %(np.min(iconV[0,:]),np.max(iconV[0,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[1,:]),np.max(iconV[1,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[2,:]),np.max(iconV[2,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[3,:]),np.max(iconV[3,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[4,:]),np.max(iconV[4,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[5,:]),np.max(iconV[5,:])))

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("grid and connectivity V: %.3f s" % (timing.time() - start))

#################################################################
# build pressure grid (nodes and icon)
#################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int32)
xP=np.empty(NfemP,dtype=np.float64)     # x coordinates
yP=np.empty(NfemP,dtype=np.float64)     # y coordinates

counter=0
for iel in range(0,nel):
    xP[counter]=xV[iconV[0,iel]]
    yP[counter]=yV[iconV[0,iel]]
    iconP[0,iel]=counter
    counter+=1
    xP[counter]=xV[iconV[1,iel]]
    yP[counter]=yV[iconV[1,iel]]
    iconP[1,iel]=counter
    counter+=1
    xP[counter]=xV[iconV[2,iel]]
    yP[counter]=yV[iconV[2,iel]]
    iconP[2,iel]=counter
    counter+=1

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], yP[iconP[0][iel]])
#    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], yP[iconP[1][iel]])
#    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], yP[iconP[2][iel]])

print("grid and connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]<eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]<eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]>(Ly-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area    = np.zeros(nel,dtype=np.float64) 
dNNNVdr = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)  # shape functions derivatives

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
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

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = timing.time()

a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)
K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat=np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
N_mat=np.zeros((3,ndofP*mP),dtype=np.float64)  # matrix  
dNNNVdx=np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdy=np.zeros(mV,dtype=np.float64)          # shape functions derivatives
c_mat=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
#NNNV=np.zeros(mV,dtype=np.float64)             # shape functions V
#NNNP=np.zeros(mP,dtype=np.float64)             # shape functions P
#dNNNVdr=np.zeros(mV,dtype=np.float64)          # shape functions derivatives
#dNNNVds=np.zeros(mV,dtype=np.float64)          # shape functions derivatives
#u=np.zeros(NV,dtype=np.float64)                # x-component velocity
#v=np.zeros(NV,dtype=np.float64)                # y-component velocity

for iel in range(0,nel):

    f_el=np.zeros((mV*ndofV),dtype=np.float64)
    K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    for kq in range (0,nqel):

        # position & weight of quad. point
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        NNNV=NNV(rq,sq)
        dNNNVdr=dNNVdr(rq,sq)
        dNNNVds=dNNVds(rq,sq)
        NNNP=NNP(rq,sq)

        # calculate jacobian matrix
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        # compute dNdx & dNdy
        xq=0.0
        yq=0.0
        for k in range(0,mV):
            xq+=NNNV[k]*xV[iconV[k,iel]]
            yq+=NNNV[k]*yV[iconV[k,iel]]
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

        # construct 3x8 b_mat matrix
        for i in range(0,mV):
            b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                     [0.        ,dNNNVdy[i]],
                                     [dNNNVdy[i],dNNNVdx[i]]]

        # compute elemental a_mat matrix
        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

        # compute elemental rhs vector
        for i in range(0,mV):
            f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
            f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)

        for i in range(0,mP):
            N_mat[0,i]=NNNP[i]
            N_mat[1,i]=NNNP[i]
            N_mat[2,i]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

    #end for kq

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
               #end if
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
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]

print("build FE matrix: %.3f s" % (timing.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = timing.time()

rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
a_mat[0:NfemV,0:NfemV]=K_mat
a_mat[0:NfemV,NfemV:Nfem]=G_mat
a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# solve system
######################################################################
start = timing.time()

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

print("split solution into u,v,p: %.3f s" % (timing.time() - start))

######################################################################
# normalise pressure
######################################################################
start = timing.time()

pavrg=0
for iel in range(0,nel):

    for kq in range (0,nqel):

        # position & weight of quad. point
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        dNNNVdr=dNNVdr(rq,sq)
        dNNNVds=dNNVds(rq,sq)
        NNNP=NNP(rq,sq)

        # calculate jacobian matrix
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        jcob=np.linalg.det(jcb)

        pavrg+=NNNP.dot(p[iconP[:,iel]])*weightq*jcob

p-=pavrg

np.savetxt('pressure_after.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("normalise pressure: %.3f s" % (timing.time() - start))

######################################################################
# compute elemental strainrate 
######################################################################
start = timing.time()

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.33333
    sq = 0.33333
    NNNV=NNV(rq,sq)
    dNNNVdr=dNNVdr(rq,sq)
    dNNNVds=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
    #end for
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)
    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
    #end for
    for k in range(0,mV):
        xc[iel]+=NNNV[k]*xV[iconV[k,iel]]
        yc[iel]+=NNNV[k]*yV[iconV[k,iel]]
        exx[iel]+=dNNNVdx[k]*u[iconV[k,iel]]
        eyy[iel]+=dNNNVdy[k]*v[iconV[k,iel]]
        exy[iel]+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                  0.5*dNNNVdx[k]*v[iconV[k,iel]]
    #end for
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
#end for

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (timing.time() - start))

#####################################################################
# interpolate pressure onto velocity grid points
#####################################################################
#
#  02          #  02
#  ||\\        #  ||\\
#  || \\       #  || \\
#  ||  \\      #  ||  \\
#  05   04     #  ||   \\
#  || 06 \\    #  ||    \\
#  ||     \\   #  ||     \\
#  00==03==01  #  00======01
#
#####################################################################
start = timing.time()

q=np.zeros(NV,dtype=np.float64)
temp=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    q[iconV[0,iel]]+=p[iconP[0,iel]]
    q[iconV[1,iel]]+=p[iconP[1,iel]]
    q[iconV[2,iel]]+=p[iconP[2,iel]]
    temp[iconV[0,iel]]+=1
    temp[iconV[1,iel]]+=1
    temp[iconV[2,iel]]+=1

    q[iconV[3,iel]]+=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    q[iconV[4,iel]]+=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    q[iconV[5,iel]]+=(p[iconP[0,iel]]+p[iconP[2,iel]])*0.5
    temp[iconV[3,iel]]+=1
    temp[iconV[4,iel]]+=1
    temp[iconV[5,iel]]+=1

    q[iconV[6,iel]]+=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3
    temp[iconV[6,iel]]+=1

q=q/temp

#np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

print("interp p onto V nodes: %.3f s" % (timing.time() - start))

#################################################################
# compute error fields for plotting
#################################################################
start = timing.time()

error_u = np.empty(NV,dtype=np.float64)
error_v = np.empty(NV,dtype=np.float64)
error_q = np.empty(NV,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(xV[i],yV[i])
    error_v[i]=v[i]-velocity_y(xV[i],yV[i])
    error_q[i]=q[i]-pressure(xV[i],yV[i])

print("compute error fields: %.3f s" % (timing.time() - start))

#################################################################
# compute L2 errors
#################################################################
start = timing.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for kq in range (0,nqel):
        # position & weight of quad. point
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        NNNP[0:mP]=NNP(rq,sq)
        # calculate jacobian matrix
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        jcbi = np.linalg.inv(jcb)

        # compute dNdx & dNdy
        xq=0.
        yq=0.
        uq=0.
        vq=0.
        pq=0.
        for k in range(0,mV):
            xq+=NNNV[k]*xV[iconV[k,iel]]
            yq+=NNNV[k]*yV[iconV[k,iel]]
            uq+=NNNV[k]*u[iconV[k,iel]]
            vq+=NNNV[k]*v[iconV[k,iel]]
        errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
        # end for k
        xq=0.
        yq=0.
        pq=0.
        for k in range(0,mP):
            xq+=NNNP[k]*xP[iconP[k,iel]]
            yq+=NNNP[k]*yP[iconP[k,iel]]
            pq+=NNNP[k]*p[iconP[k,iel]]
        errp+=(pq-pressure(xq,yq))**2*weightq*jcob
        # end for k
    # end for kq
# end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (timing.time() - start))

#####################################################################
# plot of solution
#####################################################################
# the 7-node P2+ element does not exist in vtk, but the 6-node one 
# does, i.e. type=22. 

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnx*nny,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p (el)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % ((p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error u' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%10e \n" %error_u[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error v' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%10e \n" %error_v[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error q' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%10e \n" %error_q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                              iconV[3,iel],iconV[4,iel],iconV[5,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*6))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22)
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
