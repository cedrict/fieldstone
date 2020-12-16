import numpy as np
import numpy.ma as ma
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
from scipy.sparse import lil_matrix
from parameters import *

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
print("---------------------------------------")
print("---------------fieldstone--------------")
print("---------------------------------------")

         # Crouzeix-Raviart elements
mV=7     # number of velocity nodes making up an element
mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

#read nb of elements and nb of nodes from temp file 

counter=0
file=open("temp", "r")
for line in file:
    fields = line.strip().split()
    #print(fields[0], fields[1], fields[2])
    if counter==0:
       nel=int(fields[0])
    if counter==1:
       NV0=int(fields[0])
    counter+=1

NV=NV0+nel

NfemV=NV*ndofV     # number of velocity dofs
NfemP=nel*3*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP    # total number of dofs

print ('nel', nel)
print ('NV0', NV0)
print ('NfemV', NfemV)
print ('NfemP', NfemP)
print ('Nfem ', Nfem)

eta_ref=1

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

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.zeros(NV,dtype=np.float64)     # x coordinates
yV=np.zeros(NV,dtype=np.float64)     # y coordinates

xV[0:NV0],yV[0:NV0]=np.loadtxt('mesh.1.node',unpack=True,usecols=[1,2],skiprows=1)

print("xV (min/max): %.4f %.4f" %(np.min(xV[0:NV0]),np.max(xV[0:NV0])))
print("yV (min/max): %.4f %.4f" %(np.min(yV[0:NV0]),np.max(yV[0:NV0])))

np.savetxt('gridV0.ascii',np.array([xV,yV]).T,header='# xV,yV')

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
np.loadtxt('mesh.1.ele',unpack=True, usecols=[1,2,3,4,5,6],skiprows=1)

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

np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# xV,yV')

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

np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], yP[iconP[0][iel]])
#    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], yP[iconP[1][iel]])
#    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], yP[iconP[2][iel]])

print("setup: connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# flag nodes for benchmark
# one at 0.5,0.6 , one at 0.5,0 , at startup
#################################################################

for i in range(0,NV):
    if abs(xV[i]-xobject)<1e-6 and abs(yV[i]-yobject)<1e-6:
       node_center=i
    if abs(xV[i]-0.5*Lx)<1e-6 and abs(yV[i])<1e-6:
       node_bottom=i

#################################################################
# assigning material properties to elements
#################################################################
start = timing.time()

rho=np.zeros(nel,dtype=np.float64) 
eta=np.zeros(nel,dtype=np.float64) 
mat=np.zeros(nel,dtype=np.int16) 

if experiment==1:
   for iel in range(0,nel):
       x_c=xV[iconV[6,iel]]
       y_c=yV[iconV[6,iel]]
       mat[iel]=1
       rho[iel]=1
       eta[iel]=1
       if (x_c-0.5)**2+(y_c-0.5)**2<rad**2:
          mat[iel]=2
          rho[iel]=1.01
          eta[iel]=1e3

if experiment==2:
   for iel in range(0,nel):
       x_c=xV[iconV[6,iel]]
       y_c=yV[iconV[6,iel]]
       mat[iel]=1
       rho[iel]=1
       eta[iel]=1
       if y_c>0.75:
          mat[iel]=3
          rho[iel]=0
          eta[iel]=0.001
       if (x_c-0.5)**2+(y_c-0.6)**2<rad**2:
          mat[iel]=2
          rho[iel]=2
          eta[iel]=1e3

if experiment==3:
   for iel in range(0,nel):
       x_c=xV[iconV[6,iel]]
       y_c=yV[iconV[6,iel]]
       mat[iel]=1
       rho[iel]=1
       eta[iel]=1
       if abs(x_c-0.5)<size/2 and abs(y_c-0.5)<size/2:
          mat[iel]=2
          rho[iel]=1.01
          eta[iel]=1e3

if experiment==4:
   for iel in range(0,nel):
       x_c=xV[iconV[6,iel]]
       y_c=yV[iconV[6,iel]]
       mat[iel]=1
       rho[iel]=1010
       eta[iel]=100
       if y_c<0.2+0.02*np.cos(x_c*np.pi/Lx):
          mat[iel]=2
          rho[iel]=1000
          eta[iel]=100

print("material layout: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, NV):
    #Left boundary  
    if xV[i]/Lx<0.0000001:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       if bc=='noslip':
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0  
    #right boundary  
    if xV[i]/Lx>0.9999999:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if bc=='noslip':
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0  
    #bottom boundary  
    if yV[i]/Lx<1e-6:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0   
       if bc=='noslip' or experiment==4:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
    #top boundary  
    if yV[i]/Ly>0.9999999:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0  
       if bc=='noslip'or experiment==4:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0

print("define boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# flag surface nodes
#################################################################

on_surf=np.zeros(NV,dtype=np.bool)  # boundary condition, yes/no

for i in range(0, NV):
    if abs(yV[i]-0.75)<1e-6:
       on_surf[i]=True

xs = np.zeros(np.count_nonzero(on_surf),dtype=np.float64)  
ys = np.zeros(np.count_nonzero(on_surf),dtype=np.float64)  
us = np.zeros(np.count_nonzero(on_surf),dtype=np.float64)  
vs = np.zeros(np.count_nonzero(on_surf),dtype=np.float64)  
ps = np.zeros(np.count_nonzero(on_surf),dtype=np.float64)  


################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################

model_time=0

for istep in range(0,nstep):

    print("--------------------------------------------")
    print("istep= ", istep, '; time=',model_time)
    print("--------------------------------------------")

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = timing.time()

    A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    NNNV     = np.zeros(mV,dtype=np.float64)           # shape functions V
    dNNNVdr  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    rhs      = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    f_rhs    = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs    = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
    b_mat    = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat    = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
    NNNV     = np.zeros(mV,dtype=np.float64)           # shape functions V
    NNNP     = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNNNVdx  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVdr  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives

    c_mat    = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    for iel in range(0,nel):

        if iel%500==0:
           print('     ',iel)

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

            if jcob<0:
               exit("jacobian is negative - bad triangle")

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
            #print (etaq,rhoq)

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
                    #end for
                #end for
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    #G_mat[m1,m2]+=G_el[ikk,jkk]
                    A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk] 
                    A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk] 
                #end for
                f_rhs[m1]+=f_el[ikk] 
            #end for
        #end for
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]  
        #end for
    #end for

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("build FE matrix: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    sol = np.zeros(Nfem,dtype=np.float64) 

    sparse_matrix=A_sparse.tocsr()

    #print(sparse_matrix.min(),sparse_matrix.max())

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*eta_ref/Ly

    print("     -> u (m,M) %.6e %.6e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.6e %.6e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.6e %.6e " %(np.min(p),np.max(p)))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

    print("solve time: %.3f s" % (timing.time() - start))

    ######################################################################
    # normalise pressure
    # no need to divide by Lx=1
    ######################################################################
    if experiment==2:
       start = timing.time()

       avrg_p=0
       for iel in range(0,nel):
           if yP[iconP[0,iel]]>0.99999 and yP[iconP[1,iel]]>0.99999:
              avrg_p+=0.5*(p[iconP[0,iel]]+p[iconP[1,iel]])*(xP[iconP[0,iel]]-xP[iconP[1,iel]])
           if yP[iconP[1,iel]]>0.99999 and yP[iconP[2,iel]]>0.99999:
              avrg_p+=0.5*(p[iconP[1,iel]]+p[iconP[2,iel]])*(xP[iconP[1,iel]]-xP[iconP[2,iel]])
           if yP[iconP[2,iel]]>0.99999 and yP[iconP[0,iel]]>0.99999:
              avrg_p+=0.5*(p[iconP[2,iel]]+p[iconP[0,iel]])*(xP[iconP[2,iel]]-xP[iconP[0,iel]])

       print('     -> <p> at surface: ',avrg_p)

       p-=avrg_p

       print("normalise pressure: %.3f s" % (timing.time() - start))

    #################################################################
    # compute area of elements
    #################################################################
    start = timing.time()

    area = np.zeros(nel,dtype=np.float64) 

    avrg_p=0
    vrms=0
    vrms_a=0
    vrms_f=0
    vrms_s=0
    vrms_fs=0
    vol_a=0
    vol_f=0
    vol_s=0
    vol_fs=0
    for iel in range(0,nel):
        for kq in range (0,nqel):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            area[iel]+=jcob*weightq

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
 
            vrms+=(uq**2+vq**2)*jcob*weightq 

            if mat[iel]==1:
               vrms_f+=(uq**2+vq**2)*jcob*weightq 
               vrms_fs+=(uq**2+vq**2)*jcob*weightq 
               vol_f+=jcob*weightq
               vol_fs+=jcob*weightq
               
            if mat[iel]==2:
               vrms_s+=(uq**2+vq**2)*jcob*weightq 
               vrms_fs+=(uq**2+vq**2)*jcob*weightq 
               vol_s+=jcob*weightq
               vol_fs+=jcob*weightq

            if mat[iel]==3:
               vrms_a+=(uq**2+vq**2)*jcob*weightq 
               vol_a+=jcob*weightq

            pq=0.0
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            avrg_p+=pq*jcob*weightq

        #end for
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))
    vrms_f=np.sqrt(vrms_f/vol_f)
    vrms_s=np.sqrt(vrms_s/vol_s)
    vrms_fs=np.sqrt(vrms_fs/vol_fs)
    if experiment==2: 
       vrms_a=np.sqrt(vrms_a/vol_a)

    avrg_p/=(Lx*Ly)
    if experiment==1 or experiment==3 or experiment==4: 
       p-=avrg_p

    print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
    print("     -> total area (meas) %.6f " %(area.sum()))
    print("     -> total area (anal) %.6f " %(Lx*Ly))
    print("     -> vrms   = %e " %(vrms))
    print("     -> vrms_a = %e " %(vrms_a))
    print("     -> vrms_f = %e " %(vrms_f))
    print("     -> vrms_s = %e " %(vrms_s))
    print("     -> vrms_fs= %e " %(vrms_fs))
    print("     -> avrg_p = %e " %(avrg_p))

    print("compute area & vrms: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute time stepping
    ######################################################################
    start = timing.time()

    dt=(CFL*np.min(np.sqrt(area))/np.max(np.sqrt(u**2+v**2)))
    print("     -> dt=", dt)

    print("compute time step: %.3f s" % (timing.time() - start))

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

    print("compute sr and stress: %.3f s" % (timing.time() - start))

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

    #####################################################################
    # interpolate strain rate onto velocity grid points
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

    sr=np.zeros(NV,dtype=np.float64)
    sr_el=np.zeros(nel,dtype=np.float64)
    cc=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        sr[iconV[0,iel]]+=e[iel]
        cc[iconV[0,iel]]+=1.
        sr[iconV[1,iel]]+=e[iel]
        cc[iconV[1,iel]]+=1.
        sr[iconV[2,iel]]+=e[iel]
        cc[iconV[2,iel]]+=1.
        sr[iconV[3,iel]]+=e[iel]
        cc[iconV[3,iel]]+=1.
        sr[iconV[4,iel]]+=e[iel]
        cc[iconV[4,iel]]+=1.
        sr[iconV[5,iel]]+=e[iel]
        cc[iconV[5,iel]]+=1.
        sr_el[iel]=e[iel]

    for i in range(0,NV):
        if cc[i] != 0:
           sr[i]=sr[i]/cc[i]

    #####################################################################
    # compure dev stress tensor and stress tensor 
    #####################################################################

    tauxx = np.zeros(nel,dtype=np.float64)  
    tauyy = np.zeros(nel,dtype=np.float64)  
    tauxy = np.zeros(nel,dtype=np.float64)  
    sigmaxx = np.zeros(nel,dtype=np.float64)  
    sigmayy = np.zeros(nel,dtype=np.float64)  
    sigmaxy = np.zeros(nel,dtype=np.float64)  
    
    tauxx[:]=2*eta[:]*exx[:]
    tauyy[:]=2*eta[:]*eyy[:]
    tauxy[:]=2*eta[:]*exy[:]

    sigmaxx[:]=-p_el[:]+2*eta[:]*exx[:]
    sigmayy[:]=-p_el[:]+2*eta[:]*eyy[:]
    sigmaxy[:]=        +2*eta[:]*exy[:]

    #####################################################################
    # locate (0.5,0.6) point ands record fields on it
    #####################################################################

    if experiment==1 or experiment==3:

       p_u=0
       p_v=0
       p_p=0
       p_mat=0
       iel_target=0

       profile=open('profile.ascii',"w")
       for i in range(0,NV):
           if abs(xV[i]-0.5)<1e-6:
              profile.write("%10e %10e %10e %10e \n" %(yV[i],q[i]-(0.5-yV[i]),u[i],v[i]))
              

    if experiment==2:
       start = timing.time()

       px=0.5
       py=0.6
       for iel in range(0,nel):
           if abs(xc[iel]-px)<0.05 and  abs(yc[iel]-py)<0.05: 
              p0x=xV[iconV[0,iel]]
              p0y=yV[iconV[0,iel]]
              p1x=xV[iconV[1,iel]]
              p1y=yV[iconV[1,iel]]
              p2x=xV[iconV[2,iel]]
              p2y=yV[iconV[2,iel]]
              r = 1/(2*area[iel])*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py)
              s = 1/(2*area[iel])*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py)
              if r >=0 and s>= 0 and 1-r-s>=0 :
                 NNNV[0:mV]=NNV(r,s)
                 NNNP[0:mP]=NNP(r,s)
                 xq=0.
                 yq=0.
                 uq=0.
                 vq=0.
                 for k in range(0,mV):
                     xq+=NNNV[k]*xV[iconV[k,iel]]
                     yq+=NNNV[k]*yV[iconV[k,iel]]
                     uq+=NNNV[k]*u[iconV[k,iel]]
                     vq+=NNNV[k]*v[iconV[k,iel]]
                 pq=0.
                 for k in range(0,mP):
                     pq+=NNNP[k]*p[iconP[k,iel]]
                 p_u=uq
                 p_v=vq
                 p_p=pq
                 p_mat=mat[iel]
                 iel_target=iel
                 break

       #print(iel_target,xq,yq,p_u,p_v,p_p,p_mat)

       print("measure at 0.5,0.6: %.3f s" % (timing.time() - start))

    if experiment==4:

       p_u=0
       p_v=0
       p_p=0
       p_mat=0
       iel_target=0


    #####################################################################
    # carry out measurements for benchmark
    #####################################################################
    start = timing.time()

    avrg_rho=0.
    avrg_eta=0.
    for iel in range(0,nel):
        avrg_rho+=rho[iel]*area[iel]
        avrg_eta+=eta[iel]*area[iel]
    avrg_rho/=(Lx*Ly)
    avrg_eta/=(Lx*Ly)

    counter=0
    for i in range(0,NV):
        if on_surf[i]:
           xs[counter]=xV[i]
           ys[counter]=yV[i]
           us[counter]=u[i]
           vs[counter]=v[i]
           ps[counter]=q[i]
           counter+=1

    np.savetxt('surf_{:04d}.ascii'.format(istep),np.array([xs,ys,us,vs,ps]).T)

    vel=np.sqrt(u**2+v**2)
    print('benchmark ',nel,Nfem,model_time,\
    np.min(u),np.max(u),\
    np.min(v),np.max(v),\
    np.min(vel),np.max(vel),\
    np.min(p),np.max(p),
    vrms,vrms_a,vrms_f,vrms_s,vrms_fs,\
    avrg_rho,avrg_eta,\
    vol_a,vol_f,vol_s,vol_fs,\
    dt,\
    xV[node_center],yV[node_center],u[node_center],v[node_center],\
    q[node_bottom],\
    np.min(ys),np.max(ys),
    p_u,p_v,p_p,p_mat)

    print("export measurements: %.3f s" % (timing.time() - start))

    #####################################################################
    # plot of solution
    # the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
    #####################################################################
    start = timing.time()

    filename = 'solution_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
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
    vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d\n" % (mat[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p (el)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%7e\n" % (p_el[iel]))
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
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % tauxx[iel])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % tauyy[iel])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % tauxy[iel])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % sigmaxx[iel])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % sigmayy[iel])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % sigmaxy[iel])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta_p(dev stress)' Format='ascii'> \n")
    for iel in range(0,nel):
        theta_p=0.5*np.arctan(2*tauxy[iel]/(tauxx[iel]-tauyy[iel]))
        vtufile.write("%10e \n" % (theta_p/np.pi*180.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta_p(stress)' Format='ascii'> \n")
    for iel in range(0,nel):
        theta_p=0.5*np.arctan(2*sigmaxy[iel]/(sigmaxx[iel]-sigmayy[iel]))
        vtufile.write("%10e \n" % (theta_p/np.pi*180.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_max' Format='ascii'> \n")
    for iel in range(0,nel):
        tau_max=np.sqrt( (tauxx[iel]-tauyy[iel])**2/4 +tauxy[iel]**2 )
        vtufile.write("%10e \n" % tau_max)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_max' Format='ascii'> \n")
    for iel in range(0,nel):
        sigma_max=np.sqrt( (sigmaxx[iel]-sigmayy[iel])**2/4 +sigmaxy[iel]**2 )
        vtufile.write("%10e \n" % sigma_max)
    vtufile.write("</DataArray>\n")
    #--
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
    vtufile.write("<DataArray type='Float32' Name='on_surf' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %on_surf[i])
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

    #------------------------------------------------------

    filename = 'stress_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nel,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        x_c=xV[iconV[6,i]]
        y_c=yV[iconV[6,i]]
        vtufile.write("%10e %10e %10e \n" %(x_c,y_c,0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_1 (dir)' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        theta_p=0.5*np.arctan(2*tauxy[i]/(tauxx[i]-tauyy[i]))
        vtufile.write("%10e %10e %10e \n" %( np.cos(theta_p),np.sin(theta_p),0) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_2 (dir)' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        theta_p=0.5*np.arctan(2*tauxy[i]/(tauxx[i]-tauyy[i])) + np.pi/2.
        vtufile.write("%10e %10e %10e \n" %( np.cos(theta_p),np.sin(theta_p),0) )
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_1 (dir+mag)' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        theta_p=0.5*np.arctan(2*sigmaxy[i]/(sigmaxx[i]-sigmayy[i]))
        sigma1=(sigmaxx[iel]+sigmayy[iel])/2. + np.sqrt( (sigmaxx[iel]-sigmayy[iel])**2/4 +sigmaxy[iel]**2 ) 
        vtufile.write("%10e %10e %10e \n" %( np.cos(theta_p)*sigma1,np.sin(theta_p)*sigma1,0.) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_2 (dir+mag)' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        theta_p=0.5*np.arctan(2*sigmaxy[i]/(sigmaxx[i]-sigmayy[i]))
        sigma2=(sigmaxx[iel]+sigmayy[iel])/2. - np.sqrt( (sigmaxx[iel]-sigmayy[iel])**2/4 +sigmaxy[iel]**2 ) 
        vtufile.write("%10e %10e %10e \n" %( np.cos(theta_p)*sigma2,np.sin(theta_p)*sigma2,0.) )
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range (0,nel):
        vtufile.write("%d\n" % i )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range (0,nel):
        vtufile.write("%d \n" % (i+1) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range (0,nel):
        vtufile.write("%d \n" % 1)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    print("write data: %.3fs" % (timing.time() - start))

    #################################################################
    # mesh evolution
    #################################################################

    for i in range(0,NV):
        xV[i]+=u[i]*dt
        yV[i]+=v[i]*dt

    for iel in range(0,nel):
        # node 3 is between nodes 0 and 1
        xV[iconV[3,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[1,iel]]) 
        yV[iconV[3,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[1,iel]]) 
        # node 4 is between nodes 1 and 2
        xV[iconV[4,iel]]=0.5*(xV[iconV[1,iel]]+xV[iconV[2,iel]]) 
        yV[iconV[4,iel]]=0.5*(yV[iconV[1,iel]]+yV[iconV[2,iel]]) 
        # node 5 is between nodes 0 and 2
        xV[iconV[5,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[2,iel]]) 
        yV[iconV[5,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[2,iel]]) 
        # recenter middle node
        xV[iconV[6,iel]]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
        yV[iconV[6,iel]]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.

    for iel in range(0,nel):
        xP[iconP[0,iel]]=xV[iconV[0,iel]]
        yP[iconP[0,iel]]=yV[iconV[0,iel]]
        xP[iconP[1,iel]]=xV[iconV[1,iel]]
        yP[iconP[1,iel]]=yV[iconV[1,iel]]
        xP[iconP[2,iel]]=xV[iconV[2,iel]]
        yP[iconP[2,iel]]=yV[iconV[2,iel]]

    #####################################################################

    model_time+=dt

    #####################################################################

    if model_time>=end_time: 
       print("***********************************")
       print("*********end time reached**********")
       print("***********************************")
       break

    #####################################################################
