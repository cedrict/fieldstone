import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
from scipy.sparse import lil_matrix
from parameters import *

from inside  import Polygon

#http://code.activestate.com/recipes/578381-a-point-in-polygon-program-sw-sloan-algorithm/

#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
    NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
    NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
    NV_3= 4.*(1.-rq-sq)*rq*(1.-3.*sq) 
    NV_4= 4.*rq*sq*(-2.+3.*rq+3.*sq)
    NV_5= 4.*(1.-rq-sq)*sq*(1.-3.*rq) 
    NV_6= 27*(1.-rq-sq)*rq*sq
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6

def dNNVdr(rq,sq):
    dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
    dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
    dNVdr_2= 3*sq-6*rq*sq-3*sq**2
    dNVdr_3= -8*rq+24*rq*sq+4-16*sq+12*sq**2
    dNVdr_4= -8*sq+24*rq*sq+12*sq**2
    dNVdr_5= -16*sq+24*rq*sq+12*sq**2
    dNVdr_6= -54*rq*sq+27*sq-27*sq**2
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6

def dNNVds(rq,sq):
    dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
    dNVds_1= rq*(3-3*rq-6*sq)
    dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
    dNVds_3= -16*rq+24*rq*sq+12*rq**2
    dNVds_4= -8*rq+12*rq**2+24*rq*sq
    dNVds_5= 4-16*rq-8*sq+24*rq*sq+12*rq**2
    dNVds_6= -54*rq*sq+27*rq-27*rq**2
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6

def NNP(rq,sq):
    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return NP_0,NP_1,NP_2

#------------------------------------------------------------------------------
print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=7     # number of velocity nodes making up an element
mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

nel=115731 #6000x3000
NV0=232206
#nel=81544  #5000x2500
#NV0=163735 
#nel=53826  #4000
#NV0=108149
#nel=31765   #3000
#NV0=63914
#nel=16400 #2000
#NV0=33099


NV=NV0+nel

NfemV=NV*ndofV     # number of velocity dofs
NfemP=nel*3*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP    # total number of dofs

print ('NfemV', NfemV)
print ('NfemP', NfemP)
print ('Nfem ', Nfem)

eta_ref=1e22

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
qweights =[nb5,nb5,nb5,nb6,nb6,nb6]

gx=0
gy=-9.81

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)     # x coordinates
yV=np.empty(NV,dtype=np.float64)     # y coordinates

xV[0:NV0],yV[0:NV0]=np.loadtxt('subd.1.node',unpack=True,usecols=[1,2],skiprows=1)

print("xV (min/max): %.4f %.4f" %(np.min(xV),np.max(xV)))
print("yV (min/max): %.4f %.4f" %(np.min(yV),np.max(yV)))

#np.savetxt('gridV0.ascii',np.array([xV,yV]).T,header='# xV,yV')

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
# note that the ordering of nodes returned by triangle is different
# than mine: https://www.cs.cmu.edu/~quake/triangle.highorder.html.
# note also that triangle returns nodes 0-5, but not 6.
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

iconV[0,:],iconV[1,:],iconV[2,:],iconV[4,:],iconV[5,:],iconV[3,:]=\
np.loadtxt('subd.1.ele',unpack=True, usecols=[1,2,3,4,5,6],skiprows=1)

iconV[0,:]-=1
iconV[1,:]-=1
iconV[2,:]-=1
iconV[3,:]-=1
iconV[4,:]-=1
iconV[5,:]-=1

for iel in range (0,nel):
    iconV[6,iel]=NV0+iel

print("setup: connectivity V: %.3f s" % (timing.time() - start))

for iel in range (0,nel): #bubble nodes
    xV[NV0+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
    yV[NV0+iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# xV,yV')

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

print("setup: connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# assigning material properties to elements
#################################################################
start = timing.time()

xmid=np.empty(np_mid,dtype=np.float64)   
ymid=np.empty(np_mid,dtype=np.float64)   
xmid[0:np_mid],ymid[0:np_mid]=np.loadtxt('cedric1_xmid.dat',unpack=True,usecols=[0,1])
for i in range (0,np_mid):
    xmid[i]=xmid[i]*1e5+xL-L
    ymid[i]=ymid[i]*1e5+Ly

xperim=np.empty(np_perim,dtype=np.float64)   
yperim=np.empty(np_perim,dtype=np.float64)   
xperim[0:np_perim],yperim[0:np_perim]=np.loadtxt('cedric1_shape.dat',unpack=True,usecols=[0,1])

for i in range (0,np_perim):
    xperim[i]=xperim[i]*1e5+xL-L
    yperim[i]=yperim[i]*1e5+Ly

xmin=np.min(xperim) ; xmax=np.max(xperim)
ymin=np.min(yperim) ; ymax=np.max(yperim)

poly = Polygon(xperim, yperim) # see inside.py in same folder

rho=np.zeros(nel,dtype=np.float64) 
eta=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    x_c=xV[iconV[6,iel]]
    y_c=yV[iconV[6,iel]]
    # if element is inside a box containing slab
    if x_c>xmin and x_c<xmax and y_c>ymin and y_c<ymax:
       dist = poly.is_inside(x_c, y_c)
       if dist>0: 
          rho[iel]=drho
          eta[iel]=eta1*gamma
       else:
          rho[iel]=0.
          eta[iel]=eta1
    else:
       rho[iel]=0.
       eta[iel]=eta1
    # end if
#end for

print("assign density, viscosity: %.3f s" % (timing.time() - start))

#################################################################
# generate regular grid with composition for ASPECT ascii plugin
#################################################################

if False:
   n_n=1499
   aspectfile=open('aspect_slab.ascii',"w")
   aspectfile.write("# POINTS: %8d %8d \n" %((n_n+1),(n_n+1)))
   aspectfile.write("# Columns: x y phase\n")
   for j in range(0,n_n+1):
       for i in range(0,n_n+1):
           x_c=i/n_n*Lx
           y_c=j/n_n*Ly
           if x_c>xmin and x_c<xmax and y_c>ymin and y_c<ymax:
              dist = poly.is_inside(x_c,y_c)
              if dist>0:
                 comp=1
              else:
                 comp=0 
           else:
              comp=0
           # end if
           aspectfile.write("%10e %10e %10e \n" %(x_c,y_c,comp))
      #end for
   #end for

   #np.savetxt('aspect_midpositions.ascii',np.array([xmid,ymid]).T)

   #xxxp=np.concatenate([xmid,xperim])
   #yyyp=np.concatenate([ymid,yperim])
   #np.savetxt('aspect_allpositions.ascii',np.array([xxxp,yyyp]).T)
   exit()

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, NV):
    if yV[i]/Ly>0.9999999:
       #bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    if i==NV-1: 
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

    #if xV[i]/Lx<0.0000001:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0

    #if xV[i]/Lx>0.9999999:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0

    #if yV[i]/Ly<0.0000001:
    #   #bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0


print("define boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

Rx=0
Ry=0
slab_area=0
for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
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

        xq=NNNV.dot(xV[iconV[:,iel]])
        yq=NNNV.dot(yV[iconV[:,iel]])
        if rho[iel]>0:
           Rx+=jcob*weightq*xq
           Ry+=jcob*weightq*yq
           slab_area+=jcob*weightq

Rx/=slab_area
Ry/=slab_area

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))
print("     -> total area (anal) %.6f " %(Lx*Ly))

print("     -> coords center mass: %e %e" %(Rx,Ry))

#print( np.sum(area*rho))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
f_rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs   = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u       = np.zeros(NV,dtype=np.float64)          # x-component velocity
v       = np.zeros(NV,dtype=np.float64)          # y-component velocity
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    if iel%5000==0:
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
        xq=0.0
        yq=0.0
        for k in range(0,mV):
            xq+=NNNV[k]*xV[iconV[k,iel]]
            yq+=NNNV[k]*yV[iconV[k,iel]]
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

        # compute etaq, rhoq

        etaq=eta[iel]
        rhoq=rho[iel]

        # construct 3x8 b_mat matrix
        for i in range(0,mV):
            b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                     [0.        ,dNNNVdy[i]],
                                     [dNNNVdy[i],dNNNVdx[i]]]

        # compute elemental a_mat matrix
        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

        # compute elemental rhs vector
        for i in range(0,mV):
            f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*rhoq
            f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rhoq

        for i in range(0,mP):
            N_mat[0,i]=NNNP[i]
            N_mat[1,i]=NNNP[i]
            N_mat[2,i]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

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
                A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
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
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

#a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)
#a_mat[0:NfemV,0:NfemV]=K_mat
#a_mat[0:NfemV,NfemV:Nfem]=G_mat
#a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# solve system
######################################################################
start = timing.time()

sol = np.zeros(Nfem,dtype=np.float64) 

sparse_matrix=A_sparse.tocsr()

print(sparse_matrix.min(),sparse_matrix.max())

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*eta_ref/Ly

print("     -> u (m,M) %.6e %.6e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.6e %.6e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.6e %.6e " %(np.min(p),np.max(p)))

np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (timing.time() - start))

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

q=np.zeros(NV,dtype=np.float64)
p_el=np.zeros(nel,dtype=np.float64)
cc=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    q[iconV[0,iel]]+=p[iconP[0,iel]]
    cc[iconV[0,iel]]+=1.
    q[iconV[1,iel]]+=p[iconP[1,iel]]
    cc[iconV[1,iel]]+=1.
    q[iconV[2,iel]]+=p[iconP[2,iel]]
    cc[iconV[2,iel]]+=1.
    q[iconV[3,iel]]+=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    cc[iconV[3,iel]]+=1.
    q[iconV[4,iel]]+=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    cc[iconV[4,iel]]+=1.
    q[iconV[5,iel]]+=(p[iconP[0,iel]]+p[iconP[2,iel]])*0.5
    cc[iconV[5,iel]]+=1.
    p_el[iel]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.

for i in range(0,NV):
    if cc[i] != 0:
       q[i]=q[i]/cc[i]

#np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

######################################################################
# compute nodal strain rate
######################################################################

#u[:]=xV[:]**2
#v[:]=yV[:]**2

exx_nodal = np.zeros(NV,dtype=np.float64)  
eyy_nodal = np.zeros(NV,dtype=np.float64)  
exy_nodal = np.zeros(NV,dtype=np.float64)  
tau_xx_nodal = np.zeros(NV,dtype=np.float64)  
tau_yy_nodal = np.zeros(NV,dtype=np.float64)  
tau_xy_nodal = np.zeros(NV,dtype=np.float64)  
sigma_xx_nodal = np.zeros(NV,dtype=np.float64)  
sigma_yy_nodal = np.zeros(NV,dtype=np.float64)  
sigma_xy_nodal = np.zeros(NV,dtype=np.float64)  
e_nodal   = np.zeros(NV,dtype=np.float64)  
ccc       = np.zeros(NV,dtype=np.float64)  

rVnodes=[0,1,0,0.5,0.5,0,1./3.]
sVnodes=[0,0,1,0,0.5,0.5,1./3.]

for iel in range(0,nel):
    for i in range(0,mV):
        rq=rVnodes[i]
        sq=sVnodes[i]
        inode=iconV[i,iel]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

        exxq=dNNNVdx.dot(u[iconV[:,iel]])
        eyyq=dNNNVdy.dot(v[iconV[:,iel]])
        exyq=0.5*dNNNVdy.dot(u[iconV[:,iel]])+0.5*dNNNVdx.dot(v[iconV[:,iel]])

        exx_nodal[inode]+=exxq
        eyy_nodal[inode]+=eyyq
        exy_nodal[inode]+=exyq

        tau_xx_nodal[inode]+=2*eta[iel]*exxq
        tau_yy_nodal[inode]+=2*eta[iel]*eyyq
        tau_xy_nodal[inode]+=2*eta[iel]*exyq

        sigma_xx_nodal[inode]+=-q[inode]+2*eta[iel]*exxq
        sigma_yy_nodal[inode]+=-q[inode]+2*eta[iel]*eyyq
        sigma_xy_nodal[inode]+=          2*eta[iel]*exyq
        ccc[inode]+=1

exx_nodal[:]/=ccc[:]
eyy_nodal[:]/=ccc[:]
exy_nodal[:]/=ccc[:]
tau_xx_nodal[:]/=ccc[:]
tau_yy_nodal[:]/=ccc[:]
tau_xy_nodal[:]/=ccc[:]
sigma_xx_nodal[:]/=ccc[:]
sigma_yy_nodal[:]/=ccc[:]
sigma_xy_nodal[:]/=ccc[:]

print("     -> exx_nodal (m,M) %.6e %.6e " %(np.min(exx_nodal),np.max(exx_nodal)))
print("     -> eyy_nodal (m,M) %.6e %.6e " %(np.min(eyy_nodal),np.max(eyy_nodal)))
print("     -> exy_nodal (m,M) %.6e %.6e " %(np.min(exy_nodal),np.max(exy_nodal)))
print("     -> tau_xx_nodal (m,M) %.6e %.6e " %(np.min(tau_xx_nodal),np.max(tau_xx_nodal)))
print("     -> tau_yy_nodal (m,M) %.6e %.6e " %(np.min(tau_yy_nodal),np.max(tau_yy_nodal)))
print("     -> tau_xy_nodal (m,M) %.6e %.6e " %(np.min(tau_xy_nodal),np.max(tau_xy_nodal)))

print("     -> sigma_xx_nodal (m,M) %.6e %.6e " %(np.min(sigma_xx_nodal),np.max(sigma_xx_nodal)))
print("     -> sigma_yy_nodal (m,M) %.6e %.6e " %(np.min(sigma_yy_nodal),np.max(sigma_yy_nodal)))
print("     -> sigma_xy_nodal (m,M) %.6e %.6e " %(np.min(sigma_xy_nodal),np.max(sigma_xy_nodal)))

np.savetxt('strainrate.ascii',np.array([xV,yV,exx_nodal,eyy_nodal,exy_nodal]).T,header='# xc,yc,exx,eyy,exy')

######################################################################
# compute elemental strainrate 
######################################################################
start = timing.time()

#u[:]=xV[:]**2
#v[:]=yV[:]**2

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
    NNNV[0:mV]=NNV(rq,sq)
    dNNNVdr[0:mV]=dNNVdr(rq,sq)
    dNNNVds[0:mV]=dNNVds(rq,sq)
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
    for k in range(0,mV):
        xc[iel] += NNNV[k]*xV[iconV[k,iel]]
        yc[iel] += NNNV[k]*yV[iconV[k,iel]]
        exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute sr: %.3f s" % (timing.time() - start))


#####################################################################
# compute vrms 
#####################################################################
start = timing.time()

vrms=0.
avrg_u=0.
avrg_u_slab=0.
avrg_v_slab=0.

for iel in range (0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
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
        uq=0.
        vq=0.
        for k in range(0,mV):
            uq+=NNNV[k]*u[iconV[k,iel]]
            vq+=NNNV[k]*v[iconV[k,iel]]
        vrms+=(uq**2+vq**2)*weightq*jcob
        avrg_u+=uq*weightq*jcob
        if rho[iel]>0:
           avrg_u_slab+=uq*weightq*jcob
           avrg_v_slab+=vq*weightq*jcob
    # end for kq
# end for iel

avrg_u/=(Lx*Ly)
avrg_u_slab/=slab_area
avrg_v_slab/=slab_area

print("     -> vrms        = %.6e m/s" %(vrms))
print("     -> avrg u      = %.6e m/s" %(avrg_u))
print("     -> avrg u slab = %.6e m/s" %(avrg_u_slab))
print("     -> avrg v slab = %.6e m/s" %(avrg_v_slab))

print("compute vrms: %.3fs" % (timing.time() - start))

#####################################################################
# export velocity on perimeter and midsurface to file
#####################################################################

perimfile=open('perimeter.ascii',"w")
for j in range (0,np_perim):
    for i in range(0,NV):
        if abs(xV[i]-xperim[j])<1 and abs(yV[i]-yperim[j])<1:
           perimfile.write("%e %e %e %e %e %e %e %e %e %e %e %e\n" %(xperim[j],yperim[j],\
                                                              u[i],v[i],u[i]-avrg_u_slab,\
                                                              tau_xx_nodal[i],tau_yy_nodal[i],tau_xy_nodal[i],\
                                                              sigma_xx_nodal[i],sigma_yy_nodal[i],sigma_xy_nodal[i],\
                                                              q[i]))
        #end if
    #end for
#end for

midfile=open('midsurface.ascii',"w")
for j in range (0,np_mid):
    for i in range(0,NV):
        if abs(xV[i]-xmid[j])<1 and abs(yV[i]-ymid[j])<1:
           midfile.write("%e %e %e %e %e %e %e %e %e %e %e %e\n" %(xmid[j],ymid[j],\
                                                           u[i],v[i],u[i]-avrg_u_slab,\
                                                           tau_xx_nodal[i],tau_yy_nodal[i],tau_xy_nodal[i],\
                                                           sigma_xx_nodal[i],sigma_yy_nodal[i],sigma_xy_nodal[i],\
                                                           q[i]))
        #end if
    #end for
#end for

#####################################################################
# plot of solution
#####################################################################
# the 7-node P2+ element does not exist in vtk, but the 6-node one 
# does, i.e. type=22. 

u[:]-=avrg_u

if True:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
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
    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%7e\n" % (rho[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%7e\n" % (eta[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p (el)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%7e\n" % (p_el[iel]))
    vtufile.write("</DataArray>\n")

    #--
    #vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (exx[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (eyy[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (exy[iel]))
    #vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e[iel]))
    vtufile.write("</DataArray>\n")

    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p (nod)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (exx_nodal[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (eyy_nodal[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (exy_nodal[i]))
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xx_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (tau_xx_nodal[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='tau_yy_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (tau_yy_nodal[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='tau_xy_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (tau_xy_nodal[i]))
    vtufile.write("</DataArray>\n")






    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xx_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (sigma_xx_nodal[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_yy_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (sigma_yy_nodal[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_xy_nodal' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % (sigma_xy_nodal[i]))
    vtufile.write("</DataArray>\n")





    #--
    #vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
    #for i in range(0,NV):
    #    if bc_fix[i*2]:
    #       val=1
    #    else:
    #       val=0
    #    vtufile.write("%10e \n" %val)
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
    #for i in range(0,NV):
    #    if bc_fix[i*2+1]:
    #       val=1
    #    else:
    #       val=0
    #    vtufile.write("%10e \n" %val)
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel]))
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
