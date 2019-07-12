import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing

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

print ('NfemV', NfemV)
print ('NfemP', NfemP)
print ('Nfem ', Nfem)

#---------------------------------------
# 6 point integration coeffs and weights 

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322
nb6=0.223381589678011

qcoords_r=[nb1,nb2,nb2,nb4,nb3,nb4]
qcoords_s=[nb2,nb1,nb2,nb3,nb4,nb4]
qweights=[nb5,nb5,nb5,nb6,nb6,nb6]

grav=9.81

#################################################################
# grid point setup
#################################################################
start = timing.time()

x=np.empty(nnp,dtype=np.float64)     # x coordinates
y=np.empty(nnp,dtype=np.float64)     # y coordinates
r=np.empty(nnp,dtype=np.float64)     # cylindrical coordinate r
theta=np.empty(nnp,dtype=np.float64) # cylindrical coordinate theta 

f = open('GCOORD_lowres.txt', 'r')
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

for i in range(0,nnp):
    r[i]=np.sqrt(x[i]**2+y[i]**2)
    theta[i]=math.atan2(y[i],x[i])

print("x (min/max): %.4f %.4f" %(np.min(x),np.max(x)))
print("y (min/max): %.4f %.4f" %(np.min(y),np.max(y)))
print("r (min/max): %.4f %.4f" %(np.min(r),np.max(r)))
print("theta (min/max): %.4f %.4f" %(np.min(theta),np.max(theta)))

np.savetxt('gridV.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
#
#  P_2^+           P_-1
#
#  02              02
#  ||\\            ||\\
#  || \\           || \\
#  ||  \\          ||  \\
#  05   04         ||   \\
#  || 06 \\        ||    \\
#  ||     \\       ||     \\
#  00==03==01      00======01
#
#################################################################
start = timing.time()
iconV=np.zeros((mV,nel),dtype=np.int32)

f = open('ELEM2NODE_lowres.txt', 'r')
counter=0
for line in f:
    line = line.strip()
    columns = line.split()
    for i in range(0,nel):
        #print (columns[i])
        iconV[counter,i]=float(columns[i])-1
    #print (counter)
    counter+=1

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 1",iconV[0][iel],"at pos.",x[iconV[0][iel]], y[iconV[0][iel]])
#    print ("node 2",iconV[1][iel],"at pos.",x[iconV[1][iel]], y[iconV[1][iel]])
#    print ("node 3",iconV[2][iel],"at pos.",x[iconV[2][iel]], y[iconV[2][iel]])
#    print ("node 4",iconV[3][iel],"at pos.",x[iconV[3][iel]], y[iconV[3][iel]])
#    print ("node 5",iconV[4][iel],"at pos.",x[iconV[4][iel]], y[iconV[4][iel]])
#    print ("node 6",iconV[5][iel],"at pos.",x[iconV[5][iel]], y[iconV[5][iel]])

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

iconP=np.zeros((mP,nel),dtype=np.int32)
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

np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

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

f = open('Rho_lowres.txt', 'r')
counter=0
for line in f:
    line = line.strip()
    columns = line.split()
    rho[counter]=columns[0]
    counter+=1

print("     -> rho (m,M) %.6e %.6e " %(np.min(rho),np.max(rho)))

f = open('Eta_lowres.txt', 'r')
counter=0
for line in f:
    line = line.strip()
    columns = line.split()
    eta[counter]=columns[0]
    counter+=1

print("     -> eta (m,M) %.6e %.6e " %(np.min(eta),np.max(eta)))

print("read in density, viscosity: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
boundary_bottom=np.zeros(nnp,dtype=np.bool) 
boundary_top=np.zeros(nnp,dtype=np.bool)  
boundary_left=np.zeros(nnp,dtype=np.bool) 
boundary_right=np.zeros(nnp,dtype=np.bool)

for i in range(0, nnp):
    if r[i]<4.875:
       boundary_bottom[i]=True
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    if r[i]>6.37085:
       boundary_top[i]=True
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       #print(i,r[i])

    if theta[i]<0.5237:
       boundary_right[i]=True
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    if theta[i]>2.3561:
       boundary_left[i]=True
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

print("define boundary conditions: %.3f s" % (timing.time() - start))

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

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = timing.time()

a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)
K_mat = lil_matrix((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = lil_matrix((NfemV,NfemP),dtype=np.float64) # matrix GT

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

for iel in range(0,0):

    if iel%100==0:
       print(iel)

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

#a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
#rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
#a_mat[0:NfemV,0:NfemV]=K_mat
#a_mat[0:NfemV,NfemV:Nfem]=G_mat
#a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
#rhs[0:NfemV]=f_rhs
#rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# solve system
######################################################################
start = timing.time()

sol = np.zeros(Nfem,dtype=np.float64) 

#sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (timing.time() - start))

######################################################################
# compute elemental strainrate 
######################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.0
    sq = 0.0
    weightq = 2.0 * 2.0
    NV[0:mV]=NNV(rq,sq)
    dNVdr[0:mV]=dNNVdr(rq,sq)
    dNVds[0:mV]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNVdr[k]*x[iconV[k,iel]]
        jcb[0,1]+=dNVdr[k]*y[iconV[k,iel]]
        jcb[1,0]+=dNVds[k]*x[iconV[k,iel]]
        jcb[1,1]+=dNVds[k]*y[iconV[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)
    for k in range(0,mV):
        dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
        dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]
    for k in range(0,mV):
        xc[iel] += NV[k]*x[iconV[k,iel]]
        yc[iel] += NV[k]*y[iconV[k,iel]]
        exx[iel] += dNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNVdy[k]*u[iconV[k,iel]]+ 0.5*dNVdx[k]*v[iconV[k,iel]]
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

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

q=np.zeros(nnp,dtype=np.float64)

for iel in range(0,nel):
    q[iconV[0,iel]]=p[iconP[0,iel]]
    q[iconV[1,iel]]=p[iconP[1,iel]]
    q[iconV[2,iel]]=p[iconP[2,iel]]
    q[iconV[3,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    q[iconV[4,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    q[iconV[5,iel]]=(p[iconP[0,iel]]+p[iconP[2,iel]])*0.5

np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

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
    for i in range(0,nnp):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
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



print("-----------------------------")
print("------------the end----------")
print("-----------------------------")




