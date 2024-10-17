import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import ticker, cm
#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
    NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
    NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
    NV_3= 4.*rq*sq*(-2.+3.*rq+3.*sq)
    NV_4= 4.*(1.-rq-sq)*sq*(1.-3.*rq)
    NV_5= 4.*(1.-rq-sq)*rq*(1.-3.*sq)
    NV_6= 27*(1.-rq-sq)*rq*sq
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6

def dNNVdr(rq,sq):
    dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
    dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
    dNVdr_2= 3*sq-6*rq*sq-3*sq**2
    dNVdr_3= -8*sq+24*rq*sq+12*sq**2
    dNVdr_4= -16*sq+24*rq*sq+12*sq**2
    dNVdr_5= -8*rq+24*rq*sq+4-16*sq+12*sq**2
    dNVdr_6= -54*rq*sq+27*sq-27*sq**2
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6

def dNNVds(rq,sq):
    dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
    dNVds_1= rq*(3-3*rq-6*sq)
    dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
    dNVds_3= -8*rq+12*rq**2+24*rq*sq
    dNVds_4= 4-16*rq-8*sq+24*rq*sq+12*rq**2
    dNVds_5= -16*rq+24*rq*sq+12*rq**2
    dNVds_6= -54*rq*sq+27*rq-27*rq**2
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6

def NNP(rq,sq):
    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return NP_0,NP_1,NP_2

def gx(xq,yq,grav):
    return -xq/np.sqrt(xq**2+yq**2)*grav

def gy(xq,yq,grav):
    return -yq/np.sqrt(xq**2+yq**2)*grav

#------------------------------------------------------------------------------

cm=0.01
year=365.25*3600.*24.

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

         # Crouzeix-Raviart elements
mV=7     # number of velocity nodes making up an element
mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

nnp=193785
nel=63590

#nnp=3230462/2
#nel=534363

NfemV=nnp*ndofV     # number of velocity dofs
NfemP=nel*3*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP    # total number of dofs

print ('nel  ', nel)
print ('NfemV', NfemV)
print ('NfemP', NfemP)
print ('Nfem ', Nfem)

pressure_scaling=1e22/6371e3

grav=9.81

#---------------------------------------
# 6 point integration coeffs and weights 

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322/2.
nb6=0.223381589678011/2.

qcoords_r=[nb1,nb2,nb2,nb4,nb3,nb4]
qcoords_s=[nb2,nb1,nb2,nb3,nb4,nb4]
qweights=[nb5,nb5,nb5,nb6,nb6,nb6]

#---------------------------------------------------------------
# Because of the format of the files containing the connectivity
# the internal numbering of the nodes for this stone is as follows:
#
#  P_2^+           P_-1
#
#  02          #  02
#  ||\\        #  ||\\
#  || \\       #  || \\
#  ||  \\      #  ||  \\
#  04   03     #  ||   \\
#  || 06 \\    #  ||    \\
#  ||     \\   #  ||     \\
#  00==05==01  #  00======01
#

rVnodes=[0,1,0,0.5,0.0,0.5,1./3.]
sVnodes=[0,0,1,0.5,0.5,0.0,1./3.]

#################################################################
# checking that all velocity shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i]))

#################################################################
# grid point setup
#################################################################
start = timing.time()

x=np.empty(nnp,dtype=np.float64)     # x coordinates
y=np.empty(nnp,dtype=np.float64)     # y coordinates
r=np.empty(nnp,dtype=np.float64)     # cylindrical coordinate r
theta=np.empty(nnp,dtype=np.float64) # cylindrical coordinate theta 

f = open('lowres/GCOORD_lowres.txt', 'r')
counter=0
for line in f:
    line = line.strip()
    columns = line.split()
    if counter==0:
       for i in range(0,nnp):
           x[i]=columns[i]
    if counter==1:
       for i in range(0,nnp):
           y[i]=columns[i]
    counter+=1

x[:]*=1e6
y[:]*=1e6

for i in range(0,nnp):
    r[i]=np.sqrt(x[i]**2+y[i]**2)
    theta[i]=math.atan2(y[i],x[i])

print("x (min/max): %.4f %.4f" %(np.min(x),np.max(x)))
print("y (min/max): %.4f %.4f" %(np.min(y),np.max(y)))
print("r (min/max): %.4f %.4f" %(np.min(r),np.max(r)))
print("theta (min/max): %.4f %.4f" %(np.min(theta)/np.pi*180,np.max(theta)/np.pi*180))

#np.savetxt('gridV.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
boundary_bottom=np.zeros(nnp,dtype=bool) 
boundary_top=np.zeros(nnp,dtype=bool)  
boundary_left=np.zeros(nnp,dtype=bool) 
boundary_right=np.zeros(nnp,dtype=bool)

for i in range(0, nnp):
    if r[i]<4875e3:
       boundary_bottom[i]=True
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       r[i]=4871e3
       x[i]=r[i]*np.cos(theta[i])
       y[i]=r[i]*np.sin(theta[i])
    if r[i]>6370.85e3:
       boundary_top[i]=True
       #bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       #put points on perfect radius
       #r[i]=6371e3
       #x[i]=r[i]*np.cos(theta[i])
       #y[i]=r[i]*np.sin(theta[i])
    if theta[i]<0.5237:
       boundary_right[i]=True
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       theta[i]=0.52359877559
       x[i]=r[i]*np.cos(theta[i])
       y[i]=r[i]*np.sin(theta[i])

    if theta[i]>2.3561:
       boundary_left[i]=True
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       x[i]=-r[i]*np.sqrt(2.)/2.
       y[i]= r[i]*np.sqrt(2.)/2.

print("x (min/max): %.4f %.4f" %(np.min(x),np.max(x)))
print("y (min/max): %.4f %.4f" %(np.min(y),np.max(y)))
print("r (min/max): %.4f %.4f" %(np.min(r),np.max(r)))
print("theta (min/max): %.4f %.4f" %(np.min(theta)/np.pi*180,np.max(theta)/np.pi*180))

print("define boundary conditions: %.3f s" % (timing.time() - start))

#np.savetxt('gridV2.ascii',np.array([x,y]).T,header='# x,y')

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int64)

f = open('lowres/ELEM2NODE_lowres.txt', 'r')
counter=0
for line in f:
    line = line.strip()
    columns = line.split()
    for i in range(0,nel):
        #print (columns[i])
        iconV[counter,i]=float(columns[i])-1
    #print (counter)
    counter+=1

#for iel in range (0,1):
#    print ("iel=",iel)
#    print ("node 0",iconV[0][iel],"at pos.",x[iconV[0][iel]], y[iconV[0][iel]])
#    print ("node 1",iconV[1][iel],"at pos.",x[iconV[1][iel]], y[iconV[1][iel]])
#    print ("node 2",iconV[2][iel],"at pos.",x[iconV[2][iel]], y[iconV[2][iel]])
#    print ("node 3",iconV[3][iel],"at pos.",x[iconV[3][iel]], y[iconV[3][iel]])
#    print ("node 4",iconV[4][iel],"at pos.",x[iconV[4][iel]], y[iconV[4][iel]])
#    print ("node 5",iconV[5][iel],"at pos.",x[iconV[5][iel]], y[iconV[5][iel]])
#    print ("node 6",iconV[6][iel],"at pos.",x[iconV[6][iel]], y[iconV[6][iel]])

#print("iconV (min/max): %d %d" %(np.min(iconV[0,:]),np.max(iconV[0,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[1,:]),np.max(iconV[1,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[2,:]),np.max(iconV[2,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[3,:]),np.max(iconV[3,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[4,:]),np.max(iconV[4,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[5,:]),np.max(iconV[5,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[6,:]),np.max(iconV[6,:])))

#print (iconV[0:6,0])
#print (iconV[0:6,1])
#print (iconV[0:6,2])

print("setup: connectivity V: %.3f s" % (timing.time() - start))

#################################################################
# build pressure grid (nodes and icon)
#################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int64)
xP=np.empty(NfemP,dtype=np.float64)     # x coordinates
yP=np.empty(NfemP,dtype=np.float64)     # y coordinates

counter=0
for iel in range(0,nel):
    xP[counter]=x[iconV[0,iel]]
    yP[counter]=y[iconV[0,iel]]
    iconP[0,iel]=counter
    counter+=1
    xP[counter]=x[iconV[1,iel]]
    yP[counter]=y[iconV[1,iel]]
    iconP[1,iel]=counter
    counter+=1
    xP[counter]=x[iconV[2,iel]]
    yP[counter]=y[iconV[2,iel]]
    iconP[2,iel]=counter
    counter+=1

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], yP[iconP[0][iel]])
#    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], yP[iconP[1][iel]])
#    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], yP[iconP[2][iel]])

print("setup: connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# read in material properties
#################################################################
start = timing.time()

rho=np.zeros(nel,dtype=np.float64)  # boundary condition, value
eta=np.zeros(nel,dtype=np.float64)  # boundary condition, value

f = open('lowres/Rho_lowres.txt', 'r')
counter=0
for line in f:
    line = line.strip()
    columns = line.split()
    rho[counter]=float(columns[0])*1e21*1e3
    counter+=1

#rho[:]=2500

print("     -> rho (m,M) %.6e %.6e " %(np.min(rho),np.max(rho)))

f = open('lowres/Eta_lowres.txt', 'r')
counter=0
for line in f:
    line = line.strip()
    columns = line.split()
    eta[counter]=float(columns[0])*1e21
    counter+=1

#eta[:]=1e22

print("     -> eta (m,M) %.6e %.6e " %(np.min(eta),np.max(eta)))

print("read in density, viscosity: %.3f s" % (timing.time() - start))



#################################################################
# compute area of elements
#################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
NV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NV[0:mV]=NNV(rq,sq)
        dNVdr[0:mV]=dNNVdr(rq,sq)
        dNVds[0:mV]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
            jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
            jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
            jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
    if area[iel]<0: 
       for k in range(0,mV):
           print (x[iconV[k,iel]],y[iconV[k,iel]])
   #    print(iel,iconV[:,iel])

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))
print("     -> total area %.6f " %(105./360.*np.pi*(6371e3**2-4871e3**2)   ))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

#a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)
#K_mat = lil_matrix((NfemV,NfemV),dtype=np.float64) # matrix K 
#G_mat = lil_matrix((NfemV,NfemP),dtype=np.float64) # matrix GT

f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNVdx  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVdy  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    if iel%5000==0:
       print('iel=',iel)

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNP= np.zeros(mP*ndofP,dtype=np.float64)   

    for kq in range (0,nqel):

        # position & weight of quad. point
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        NV[0:mV]=NNV(rq,sq)
        dNVdr[0:mV]=dNNVdr(rq,sq)
        dNVds[0:mV]=dNNVds(rq,sq)
        NP[0:4]=NNP(rq,sq)

        # calculate jacobian matrix
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
            jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
            jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
            jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        jcbi = np.linalg.inv(jcb)

        # compute dNdx & dNdy
        xq=0.0
        yq=0.0
        for k in range(0,mV):
            xq+=NV[k]*x[iconV[k,iel]]
            yq+=NV[k]*y[iconV[k,iel]]
            dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
            dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

        # compute etaq, rhoq

        etaq=eta[iel]
        rhoq=rho[iel]

        # construct 3x8 b_mat matrix
        for i in range(0,mV):
            b_mat[0:3, 2*i:2*i+2] = [[dNVdx[i],0.     ],
                                     [0.      ,dNVdy[i]],
                                     [dNVdy[i],dNVdx[i]]]

        # compute elemental a_mat matrix
        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

        # compute elemental rhs vector
        for i in range(0,mV):
            f_el[ndofV*i  ]+=NV[i]*jcob*weightq*gx(xq,yq,grav)*rhoq
            f_el[ndofV*i+1]+=NV[i]*jcob*weightq*gy(xq,yq,grav)*rhoq

        for i in range(0,mP):
            N_mat[0,i]=NP[i]
            N_mat[1,i]=NP[i]
            N_mat[2,i]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        NNNP[:]+=NP[:]*jcob*weightq

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

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    #K_mat[m1,m2]+=K_el[ikk,jkk]
                    A_sparse[m1,m2] += K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                #G_mat[m1,m2]+=G_el[ikk,jkk]
                A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*pressure_scaling
                A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*pressure_scaling
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]*pressure_scaling

print("build FE matrix: %.3f s" % (timing.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = timing.time()

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

#for i in range(0,Nfem):
#    A_sparse[Nfem-1,i]=0
#    A_sparse[i,Nfem-1]=0
#A_sparse[Nfem-1,Nfem-1]=1e20
#rhs[Nfem-1]=0

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# solve system
######################################################################
start = timing.time()

sol = np.zeros(Nfem,dtype=np.float64) 

sparse_matrix=A_sparse.tocsr()
sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]*pressure_scaling

print("     -> u (m,M) %.6e %.6e (cm/yr)" %(np.min(u)/cm*year,np.max(u)/cm*year))
print("     -> v (m,M) %.6e %.6e (cm/yr)" %(np.min(v)/cm*year,np.max(v)/cm*year))
print("     -> p (m,M) %.6e %.6e (MPa)" %(np.min(p)/1e6,np.max(p)/1e6))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (timing.time() - start))

######################################################################
# compute nodal strainrate and pressure  
######################################################################
start = timing.time()

exx_n = np.zeros(nnp,dtype=np.float64)  
eyy_n = np.zeros(nnp,dtype=np.float64)  
exy_n = np.zeros(nnp,dtype=np.float64)  
count = np.zeros(nnp,dtype=np.int32)  
NNNV = np.zeros(mV,dtype=np.float64) 
dNNNVdr = np.zeros(mV,dtype=np.float64) 
dNNNVds = np.zeros(mV,dtype=np.float64) 
dNNNVdx = np.zeros(mV,dtype=np.float64) 
dNNNVdy = np.zeros(mV,dtype=np.float64) 
NNNP = np.zeros(mP,dtype=np.float64) 
q=np.zeros(nnp,dtype=np.float64)

#p=xP
#u[:]=x[:]
#v[:]=y[:]

for iel in range(0,nel):
    for i in range(0,mV):
        rq=rVnodes[i]
        sq=sVnodes[i]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        NNNP[0:mP]=NNP(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*x[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*y[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*x[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*y[iconV[k,iel]]
        #end for
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for
        inode=iconV[i,iel]
        q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        exx_n[inode]+=np.dot(dNNNVdx[0:mV],u[iconV[0:mV,iel]])
        eyy_n[inode]+=np.dot(dNNNVdy[0:mV],v[iconV[0:mV,iel]])
        exy_n[inode]+=0.5*np.dot(dNNNVdx[0:mV],v[iconV[0:mV,iel]])+\
                      0.5*np.dot(dNNNVdy[0:mV],u[iconV[0:mV,iel]])
        count[inode]+=1
    #end for
#end for
 
exx_n/=count
eyy_n/=count
exy_n/=count
q/=count

print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
print("     -> q (m,M) %.6e %.6e (MPa)" %(np.min(q)/1e6,np.max(q)/1e6))

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


#for iel in range(0,nel):
#    q[iconV[0,iel]]=p[iconP[0,iel]]
#    q[iconV[1,iel]]=p[iconP[1,iel]]
#    q[iconV[2,iel]]=p[iconP[2,iel]]
#    q[iconV[3,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
#    q[iconV[4,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
#    q[iconV[5,iel]]=(p[iconP[0,iel]]+p[iconP[2,iel]])*0.5

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

#####################################################################
# plot of solution
#####################################################################
# the 7-node P2+ element does not exist in vtk, but the 6-node one 
# does, i.e. type=22. 

if True:
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
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (rho[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eta[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/yr)' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e %10e %10e \n" %(gx(x[i],y[i],grav),gy(x[i],y[i],grav),0.))
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e \n" %exx_n[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e \n" %eyy_n[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e \n" %exy_n[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e \n" %r[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta (deg)' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e \n" %(theta[i]/np.pi*180.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
    for i in range(0,nnp):
        if bc_fix[i*2]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
    for i in range(0,nnp):
        if bc_fix[i*2+1]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[5,iel],iconV[3,iel],iconV[4,iel]))
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

##############################################################
# naive matplotlib-based visualisation
##############################################################

triangles = np.zeros( (nel,3) ) 
for iel in range(0,nel):
    for i in range(0,mP):
        triangles[iel,i]=iconV[i,iel]

fig = plt.figure(figsize=(10,3))
#fig = plt.figure(figsize=(10,10))
# Set the frame box 'aspect' - defined from axis scaling as ratio y-unit/x-unit.
frame_h_over_w = 1.0 
cb_shrink = 0.40
cb_pad    = 0.04
cb_aspect = 10

triang = mtri.Triangulation(x, y, triangles)
frame = fig.add_subplot(131,adjustable='box',aspect=frame_h_over_w)  # nrows,ncols,index
#frame = fig.add_subplot(131)  # nrows,ncols,index
frame.contour=plt.tricontourf(triang, theta, cmap='rainbow')
frame.triplot(triang, 'k-',linewidth=0.1)
frame.set_title('theta')
frame.set_ylabel('y coordinate')
cbar = plt.colorbar(frame.contour,ax=frame,shrink=cb_shrink,pad=cb_pad,aspect=cb_aspect)

triang = mtri.Triangulation(x, y, triangles)
frame = fig.add_subplot(132,adjustable='box',aspect=frame_h_over_w)  # nrows,ncols,index
# viscosity with reversed color map
frame.contour=plt.tricontourf(triang, r, locator=ticker.LogLocator(),cmap='rainbow_r')
#frame.contour=plt.tricontourf(triang, r, locator=ticker.LogLocator())
frame.triplot(triang, 'k-',linewidth=0.1)
frame.set_title('radius')
frame.set_yticks([]) # no y-ticks
frame.set_xlabel('x coordinate')
cbar = plt.colorbar(frame.contour,ax=frame,shrink=cb_shrink,pad=cb_pad,aspect=cb_aspect)

triang = mtri.Triangulation(x, y, triangles)
frame = fig.add_subplot(133,adjustable='box',aspect=frame_h_over_w)  # nrows,ncols,index
#frame = fig.add_subplot(133)  # nrows,ncols,index
frame.contour=plt.tricontourf(triang, q)
frame.triplot(triang, 'k-',linewidth=0.1)
frame.set_title('pressure')
frame.set_yticks([]) # no y-ticks
cbar = plt.colorbar(frame.contour,ax=frame,shrink=cb_shrink,pad=cb_pad,aspect=cb_aspect)

plt.tight_layout()
plt.savefig('plot.pdf')
plt.show()


print("-----------------------------")
print("------------the end----------")
print("-----------------------------")




