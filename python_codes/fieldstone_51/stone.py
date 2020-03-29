import numpy as np
import math as math
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import time as timing
import random

#------------------------------------------------------------------------------

def NNV(r,s):
    NV_0=1-r-s-9*(1-r-s)*r*s 
    NV_1=  r  -9*(1-r-s)*r*s
    NV_2=    s-9*(1-r-s)*r*s
    NV_3=     27*(1-r-s)*r*s
    return NV_0,NV_1,NV_2,NV_3

def dNNVdr(r,s):
    dNdr_0= -1-9*(1-2*r-s)*s 
    dNdr_1=  1-9*(1-2*r-s)*s
    dNdr_2=   -9*(1-2*r-s)*s
    dNdr_3=   27*(1-2*r-s)*s
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(r,s):
    dNds_0= -1-9*(1-r-2*s)*r 
    dNds_1=   -9*(1-r-2*s)*r
    dNds_2=  1-9*(1-r-2*s)*r
    dNds_3=   27*(1-r-2*s)*r
    return dNds_0,dNds_1,dNds_2,dNds_3

def NNP(r,s):
    return 1-r-s,r,s 

def dNNPdr(r,s):
    return -1,1,0 

def dNNPds(r,s):
    return -1,0,1 

def NNT(r,s):
    return 1-r-s,r,s 

def dNNTdr(r,s):
    return -1,1,0 

def dNNTds(r,s):
    return -1,0,1 

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
mV=4     # number of velocity nodes making up an element
mP=3     # number of pressure nodes making up an element
mT=3     # number of temperature nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
ndofT=1  # number of temperature degrees of freedom 

n=25

Lx=1 # horizontal extent of the domain 
Ly=1 # vertical extent of the domain 

nel=(n-1)**2
NV=int(n*(n+1)/2)+nel
NP=int(n*(n+1)/2)
NT=int(n*(n+1)/2)

NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
NfemT=NT*ndofT       # number of pressure dofs
Nfem=NfemV+NfemP     # total number of dofs

visu=1

h=Lx/(n-1)

eps=1e-8

nqel=6


Ra=1e6

eta=1.

nstep=100

scramble = True
rand=False

if rand:
   deltax=h/4.
   deltay=h/4.
else:
   deltax=0.
   deltay=0.

relax=0.2

rVnodes=[0.,1.,0.,1./3.]
sVnodes=[0.,0.,1.,1./3.]

rTnodes=[0.,1.,0.]
sTnodes=[0.,0.,1.]

#---------------------------------------
# 6 point integration coeffs and weights 

qcoords_r=np.empty(nqel,dtype=np.float64)  
qcoords_s=np.empty(nqel,dtype=np.float64)  
qweights=np.empty(nqel,dtype=np.float64)  

if nqel==3:
   qcoords_r[0]=1./6.; qcoords_s[0]=1./6.; qweights[0]=1./6.
   qcoords_r[1]=2./3.; qcoords_s[1]=1./6.; qweights[1]=1./6.
   qcoords_r[2]=1./6.; qcoords_s[2]=2./3.; qweights[2]=1./6.

if nqel==6:
   qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
   qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
   qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
   qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
   qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
   qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 

if nqel==7:
   qcoords_r[0]=0.1012865073235 ; qcoords_s[0]=0.1012865073235 ; qweights[0]=0.0629695902724 
   qcoords_r[1]=0.7974269853531 ; qcoords_s[1]=0.1012865073235 ; qweights[1]=0.0629695902724 
   qcoords_r[2]=0.1012865073235 ; qcoords_s[2]=0.7974269853531 ; qweights[2]=0.0629695902724 
   qcoords_r[3]=0.4701420641051 ; qcoords_s[3]=0.0597158717898 ; qweights[3]=0.0661970763942 
   qcoords_r[4]=0.4701420641051 ; qcoords_s[4]=0.4701420641051 ; qweights[4]=0.0661970763942 
   qcoords_r[5]=0.0597158717898 ; qcoords_s[5]=0.4701420641051 ; qweights[5]=0.0661970763942 
   qcoords_r[6]=0.3333333333333 ; qcoords_s[6]=0.3333333333333 ; qweights[6]=0.1125000000000 

#################################################################

print('nel=',nel)
print('NV=',NV)
print('NT',NT)
print('NP',NP)
       
filename = 'Nusselt_bottom.ascii'
Nu_bot=open(filename,"w")

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.zeros(NV,dtype=np.float64)  # x coordinates
yV=np.zeros(NV,dtype=np.float64)  # y coordinates
iconV=np.zeros((mV,nel),dtype=np.int32)

counter=0    
for j in range(0,n):
    for i in range(0,n-j):
        if i==0 or i==n-j-1 or j==0 or j==n-1:
           xV[counter]=i*h 
           yV[counter]=j*h
        else:
           xV[counter]=i*h + random.randrange(-100,100,1)/100*deltax
           yV[counter]=j*h + random.randrange(-100,100,1)/100*deltay
        counter+=1

iel = 0
for iY in range(0,n-1):
    startLower = NP - int(0.5 * (n-iY) * (n-iY+1))+1
    startUpper = NP - int(0.5 * (n-iY-1) * (n-iY))+1
    for iX in range(0,n-iY-2):
        # add a square of two elements
        topleft     = startUpper+iX
        topright    = startUpper+iX+1
        bottomleft  = startLower+iX
        bottomright = startLower+iX+1
        if (scramble == True and iX%2 == 0 and iY%2==0):
            # switch the diagonal, in order to have more
            # mesh points with an odd number of elements adjacent to them
            iconV[0:3,iel]   = bottomleft, bottomright, topright
            iconV[0:3,iel+1] = bottomleft, topright, topleft
        else:
            iconV[0:3,iel]   = bottomleft, bottomright, topleft
            iconV[0:3,iel+1] = bottomright, topright, topleft
        iel = iel + 2
    # add the tail element
    iconV[0:3,iel] = startLower+n-iY-2, startLower+n-iY-1, startUpper+n-iY-2
    iel = iel + 1
# add the top element
iconV[0:3,nel-1] = NP-2, NP-1, NP


iconV[0:3,0:nel] -=1

for iel in range(0,nel):
    iconV[3,iel]=NP+iel

for iel in range (0,nel): #bubble nodes
    xV[NP+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
    yV[NP+iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconV[0,iel],"at pos.",xV[iconV[0,iel]], yV[iconV[0,iel]])
#    print ("node 1",iconV[1,iel],"at pos.",xV[iconV[1,iel]], yV[iconV[1,iel]])
#    print ("node 2",iconV[2,iel],"at pos.",xV[iconV[2,iel]], yV[iconV[2,iel]])
#    print ("node 3",iconV[3,iel],"at pos.",xV[iconV[3,iel]], yV[iconV[3,iel]])

#print("iconV (min/max): %d %d" %(np.min(iconV[0,:]),np.max(iconV[0,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[1,:]),np.max(iconV[1,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[2,:]),np.max(iconV[2,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[3,:]),np.max(iconV[3,:])))

print("grid and connectivity V: %.3f s" % (timing.time() - start))

#################################################################
# build pressure grid (nodes and icon)
#################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int32)
xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates

xP[0:NP]=xV[0:NP]
yP[0:NP]=yV[0:NP]

iconP[0:mP,0:nel]=iconV[0:mP,0:nel]

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], yP[iconP[0][iel]])
#    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], yP[iconP[1][iel]])
#    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], yP[iconP[2][iel]])

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("grid and connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# build temperature grid (nodes and icon)
#################################################################
start = timing.time()

iconT=np.zeros((mT,nel),dtype=np.int32)
xT=np.empty(NT,dtype=np.float64)     # x coordinates
yT=np.empty(NT,dtype=np.float64)     # y coordinates

xT[0:NT]=xV[0:NT]
yT[0:NT]=yV[0:NT]

iconT[0:mT,0:nel]=iconV[0:mT,0:nel]

#np.savetxt('gridT.ascii',np.array([xT,yT]).T,header='# x,y')

print("grid and connectivity T: %.3f s" % (timing.time() - start))

#################################################################
# define velocity boundary conditions
#################################################################
start = timing.time()

bc_fixV=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_valV=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fixV[i*ndofV  ] = True ; bc_valV[i*ndofV  ] = 0. 
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0. 
    if abs(xV[i]+yV[i]-1)<eps:
       bc_fixV[i*ndofV  ] = True ; bc_valV[i*ndofV  ] = 0. 
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0. 
    if yV[i]/Ly<eps:
       bc_fixV[i*ndofV  ] = True ; bc_valV[i*ndofV  ] = 0. 
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0. 

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# define temperature boundary conditions
#################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  # boundary condition, yes/no
bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

for i in range(0,NT):
    if xT[i]/Lx<eps:
       bc_fixT[i] = True ; bc_valT[i] = 0. 
    if yT[i]/Ly<eps:
       bc_fixT[i] = True ; bc_valT[i] = 2*(1.-np.cos(2*np.pi*xT[i]))

print("setup: boundary conditions T: %.3f s" % (timing.time() - start))

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
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
    # end for kq
# end for iel

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# define temperature field
#################################################################

T=np.zeros(NT,dtype=np.float64) 
T_old=np.zeros(NT,dtype=np.float64) 

for i in range(0,NT):
    if yT[i]/Ly<eps:
       T[i]=2*(1.-np.cos(2*np.pi*xT[i]))

T_old=T

#np.savetxt('temperatureinit.ascii',np.array([xT,yT,T]).T,header='# x,y,T')

#==============================================================================
# timestepping
#==============================================================================

vrms=np.zeros(nstep,dtype=np.float64) 
avrgT=np.zeros(nstep,dtype=np.float64) 
u=np.zeros(NV,dtype=np.float64)          # x-component velocity
v=np.zeros(NV,dtype=np.float64)          # y-component velocity
u_old=np.zeros(NV,dtype=np.float64)      # x-component velocity
v_old=np.zeros(NV,dtype=np.float64)      # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for istep in range(0,nstep):

    print("--------------------------------------------------------")
    print("----------- istep=",istep,"-----------------------------------")
    print("--------------------------------------------------------")

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
    b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
    NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
    NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
    NNNT    = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNNNVdx  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)

        for kq in range (0,nqel):

            # position & weight of quad. point
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)
            NNNT[0:mP]=NNT(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
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

            Tq=0.0
            for k in range(0,mT):
                Tq+=NNNT[k]*T[iconT[k,iel]]

            #print (xq,yq,Tq)

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*Ra*Tq

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        # end for kq

        # impose b.c. 
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                if bc_fixV[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,mV*ndofV):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_valV[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_valV[m1]
                   h_el[:]-=G_el[ikk,:]*bc_valV[m1]
                   G_el[ikk,:]=0

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

    # end for iel

    print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
    print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))

    print("build FE system V,P: %.3f s" % (timing.time() - start))

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

    #assign extra pressure b.c. to remove null space
    a_mat[Nfem-1,:]=0
    a_mat[:,Nfem-1]=0
    a_mat[Nfem-1,Nfem-1]=1
    rhs[Nfem-1]=0

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

    print("     -> u (m,M) %4e %4e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %4e %4e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %4e %4e " %(np.min(p),np.max(p)))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
    #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("split sol into u,v,p: %.3f s" % (timing.time() - start))

    #####################################################################
    # relaxation step
    #####################################################################

    u=relax*u+(1-relax)*u_old
    v=relax*v+(1-relax)*v_old

    #####################################################################
    # compute strain rate components
    #####################################################################
    start = timing.time()

    exx = np.zeros(nel,dtype=np.float64)  
    eyy = np.zeros(nel,dtype=np.float64)  
    exy = np.zeros(nel,dtype=np.float64)  
    xc = np.zeros(nel,dtype=np.float64)  
    yc = np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):

        rq=1./3.
        sq=1./3.
        weightq=0.5

        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)

        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            xc[iel] += NNNV[k]*xV[iconV[k,iel]]
            yc[iel] += NNNV[k]*yV[iconV[k,iel]]
            exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
            eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
            exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
        # end for k

    # end for iel

    print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

    #np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

    print("compute el strain rate: %.3f s" % (timing.time() - start))

    #####################################################################
    # project strainrate onto velocity grid
    #####################################################################
    start = timing.time()

    exxn=np.zeros(NV,dtype=np.float64)
    eyyn=np.zeros(NV,dtype=np.float64)
    exyn=np.zeros(NV,dtype=np.float64)
    c=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,mV):
            NNNV[0:mV]=NNV(rVnodes[i],sVnodes[i])
            dNNNVdr[0:mV]=dNNVdr(rVnodes[i],sVnodes[i])
            dNNNVds[0:mV]=dNNVds(rVnodes[i],sVnodes[i])
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
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
            e_xx=0.
            e_yy=0.
            e_xy=0.
            for k in range(0,mV):
                e_xx += dNNNVdx[k]*u[iconV[k,iel]]
                e_yy += dNNNVdy[k]*v[iconV[k,iel]]
                e_xy += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                        0.5*dNNNVdx[k]*v[iconV[k,iel]]
            exxn[iconV[i,iel]]+=e_xx
            eyyn[iconV[i,iel]]+=e_yy
            exyn[iconV[i,iel]]+=e_xy
            c[iconV[i,iel]]+=1.
        # end for i
    # end for iel

    exxn/=c
    eyyn/=c
    exyn/=c

    print("     -> exxn (m,M) %.4f %.4f " %(np.min(exxn),np.max(exxn)))
    print("     -> eyyn (m,M) %.4f %.4f " %(np.min(eyyn),np.max(eyyn)))
    print("     -> exyn (m,M) %.4f %.4f " %(np.min(exyn),np.max(exyn)))

    print("compute nod strain rate: %.3f s" % (timing.time() - start))

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64)   # FE matrix 
    B_mat = np.zeros((ndim,ndofT*mT),dtype=np.float64) # gradient matrix B 
    N_mat = np.zeros((mT,1),dtype=np.float64)          # shape functions
    dNNNTdr = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTds = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdx = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdy = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    rhs   = np.zeros(NfemT,dtype=np.float64)           # FE rhs 
    Tvect = np.zeros(mT,dtype=np.float64)   

    for iel in range (0,nel):

        b_el=np.zeros(mT*ndofT,dtype=np.float64)
        a_el=np.zeros((mT*ndofT,mT*ndofT),dtype=np.float64)
        Ka=np.zeros((mT,mT),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,mT):
            Tvect[k]=T[iconT[k,iel]]

        for kq in range (0,nqel):

            # position & weight of quad. point
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            N_mat[0:mT,0]=NNT(rq,sq)
            dNNNTdr[0:mT]=dNNTdr(rq,sq)
            dNNNTds[0:mT]=dNNTds(rq,sq)
            NNNV[0:mV]=NNV(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mT):
                jcb[0,0]+=dNNNTdr[k]*xT[iconT[k,iel]]
                jcb[0,1]+=dNNNTdr[k]*yT[iconT[k,iel]]
                jcb[1,0]+=dNNNTds[k]*xT[iconT[k,iel]]
                jcb[1,1]+=dNNNTds[k]*yT[iconT[k,iel]]

            # calculate the determinant of the jacobian
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)

            # compute dNdx & dNdy
            vel[0,0]=0.
            vel[0,1]=0.
            for k in range(0,mV):
                vel[0,0]+=NNNV[k]*u[iconV[k,iel]]
                vel[0,1]+=NNNV[k]*v[iconV[k,iel]]

            for k in range(0,mT):
                dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                B_mat[0,k]=dNNNTdx[k]
                B_mat[1,k]=dNNNTdy[k]

            # compute diffusion matrix
            Kd=B_mat.T.dot(B_mat)*weightq*jcob

            # compute advection matrix
            Ka=N_mat.dot(vel.dot(B_mat))*weightq*jcob

            a_el+=(Kd+Ka)

        # end for kq

        # apply boundary conditions
        for k1 in range(0,mT):
            m1=iconT[k1,iel]
            if bc_fixT[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mT):
                   m2=iconT[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               # end for k2
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            # end if
        # end for k1

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mT):
            m1=iconT[k1,iel]
            for k2 in range(0,mT):
                m2=iconT[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            # end for k2
            rhs[m1]+=b_el[k1]
        # end for k1

    # end for iel

    print("build FE system T: %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %4f %.4f " %(np.min(T),np.max(T)))

    print("solve T time: %.3f s" % (timing.time() - start))

    #################################################################
    # relax
    #################################################################

    T=relax*T+(1-relax)*T_old

    #################################################################
    # compute vrms 
    #################################################################
    start = timing.time()

    for iel in range (0,nel):
        for kq in range (0,nqel):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            uq=0.
            vq=0.
            for k in range(0,mV):
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
            vrms[istep]+=(uq**2+vq**2)*weightq*jcob
        # end for kq
    # end for iel

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly))

    print("     -> vrms   = %.6f" %(vrms[istep]))

    print("compute vrms: %.3fs" % (timing.time() - start))

    #################################################################
    # compute average temperature 
    #################################################################
    start = timing.time()

    for iel in range (0,nel):
        for kq in range (0,nqel):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            NNNT[0:mT]=NNT(rq,sq)
            dNNNTdr[0:mT]=dNNTdr(rq,sq)
            dNNNTds[0:mT]=dNNTds(rq,sq)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mT):
                jcb[0,0] += dNNNTdr[k]*xT[iconT[k,iel]]
                jcb[0,1] += dNNNTdr[k]*yT[iconT[k,iel]]
                jcb[1,0] += dNNNTds[k]*xT[iconT[k,iel]]
                jcb[1,1] += dNNNTds[k]*yT[iconT[k,iel]]
            jcob=np.linalg.det(jcb)

            Tq=0.
            for k in range(0,mT):
                Tq+=NNNT[k]*T[iconT[k,iel]]
            avrgT[istep]+=Tq*weightq*jcob
        # end for kq
    # end for iel

    avrgT[istep]/=0.5

    print("     -> avrgT  = %.6f" %(avrgT[istep]))

    print("compute avrg T: %.3fs" % (timing.time() - start))

    #################################################################
    # compute average pressure 
    #################################################################
    start = timing.time()

    dNNNPdr = np.zeros(mP,dtype=np.float64)  
    dNNNPds = np.zeros(mP,dtype=np.float64)  

    avrgp=0.
    for iel in range (0,nel):
        for kq in range (0,nqel):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            NNNP[0:mP]=NNP(rq,sq)
            dNNNPdr[0:mP]=dNNPdr(rq,sq)
            dNNNPds[0:mP]=dNNPds(rq,sq)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mP):
                jcb[0,0] += dNNNPdr[k]*xP[iconP[k,iel]]
                jcb[0,1] += dNNNPdr[k]*yP[iconP[k,iel]]
                jcb[1,0] += dNNNPds[k]*xP[iconP[k,iel]]
                jcb[1,1] += dNNNPds[k]*yP[iconP[k,iel]]
            jcob=np.linalg.det(jcb)
            pq=0.
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            avrgp+=pq*weightq*jcob
        # end for kq
    # end for iel
    avrgp/=0.5

    p-=avrgp

    print("     -> p (m,M) %4e %4e " %(np.min(p),np.max(p)))

    print("compute avrg p: %.3fs" % (timing.time() - start))

    #####################################################################
    # compute nodal heat flux
    #####################################################################
    start = timing.time()

    qxn=np.zeros(NT,dtype=np.float64)
    qyn=np.zeros(NT,dtype=np.float64)
    c=np.zeros(NT,dtype=np.float64)

    for iel in range (0,nel):
        for i in range(0,mT):
            NNNT[0:mT]=NNT(rTnodes[i],sTnodes[i])
            dNNNTdr[0:mT]=dNNTdr(rTnodes[i],sTnodes[i])
            dNNNTds[0:mT]=dNNTds(rTnodes[i],sTnodes[i])
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mT):
                jcb[0,0]+=dNNNTdr[k]*xT[iconT[k,iel]]
                jcb[0,1]+=dNNNTdr[k]*yT[iconT[k,iel]]
                jcb[1,0]+=dNNNTds[k]*xT[iconT[k,iel]]
                jcb[1,1]+=dNNNTds[k]*yT[iconT[k,iel]]
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)
            for k in range(0,mT):
                dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
            q_x=0.
            q_y=0.
            for k in range(0,mT):
                q_x+=dNNNTdx[k]*T[iconT[k,iel]]
                q_y+=dNNNTdy[k]*T[iconT[k,iel]]
            qxn[iconT[i,iel]]-=q_x#*hcond
            qyn[iconT[i,iel]]-=q_y#*hcond
            c[iconT[i,iel]]+=1.
        # end for i
    # end for iel

    qxn/=c
    qyn/=c

    print("compute nodal heat flux: %.3fs" % (timing.time() - start))

    #####################################################################

    if istep%visu==0:

       filename = 'temperature_hypotenuse_{:04d}.ascii'.format(istep) 
       temp_hyp=open(filename,"w")
       for i in range(0,NT):
           if abs(xT[i]+yT[i]-1)<eps:
              temp_hyp.write("%6e %6e %6e \n" %(xT[i],yT[i],T[i]))

       filename = 'heatflux_bottom_{:04d}.ascii'.format(istep) 
       hf_bot=open(filename,"w")
       for i in range(0,NT):
           if yT[i]<eps:
              hf_bot.write("%6e %6e \n" %(xT[i],qyn[i]))

    Nu=0.
    for iel in range(0,nel):
        if yT[iconT[0,iel]]<eps and yT[iconT[1,iel]]<eps:
           Nu+=h*(qyn[iconT[0,iel]]+qyn[iconT[1,iel]])/2.
    Nu_bot.write("%6e\n" %(Nu))
    Nu_bot.flush()

    #####################################################################
    # plot of solution
    #####################################################################
    start = timing.time()

    if istep%visu==0:
       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
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
       vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")

       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e %10e %10e \n" %(qxn[i],qyn[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.5e \n" %exxn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.5e \n" %eyyn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.5e \n" %exyn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e \n" %p[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='fix u' Format='ascii'> \n")
       for i in range(0,NP):
           if bc_fixV[i*ndofV  ]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='fix v' Format='ascii'> \n")
       for i in range(0,NP):
           if bc_fixV[i*ndofV+1]:
              vtufile.write("%3e" %1.)
           else:
              vtufile.write("%3e" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='fix T' Format='ascii'> \n")
       for i in range(0,NT):
           if bc_fixT[i]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d \n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*3))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %5)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("make vtu file: %.3fs" % (timing.time() - start))

    # end if

    #####################################################################

    np.savetxt('vrms.ascii',np.array(vrms[0:istep]).T,header='# time,vrms')
    np.savetxt('avrgT.ascii',np.array(avrgT[0:istep]).T,header='# time,avrgT')

    #####################################################################

    u_old=u
    v_old=v
    T_old=T

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
