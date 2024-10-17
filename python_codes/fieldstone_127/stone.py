import numpy as np
import time as timing
import sys as sys
from parameters import *
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from numpy import linalg as LA
import math

###############################################################################

Rgas=8.3145
cm=0.01
year=365.25*24*3600

###############################################################################
# basis functions for Crouzeix-Raviart elements

def NNV(rq,sq):
    NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
    NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
    NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
    NV_3= 4.*(1.-rq-sq)*rq*(1.-3.*sq) 
    NV_4= 4.*rq*sq*(-2.+3.*rq+3.*sq)
    NV_5= 4.*(1.-rq-sq)*sq*(1.-3.*rq) 
    NV_6= 27*(1.-rq-sq)*rq*sq
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6],dtype=np.float64)

def dNNVdr(rq,sq):
    dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
    dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
    dNVdr_2= 3*sq-6*rq*sq-3*sq**2
    dNVdr_3= -8*rq+24*rq*sq+4-16*sq+12*sq**2
    dNVdr_4= -8*sq+24*rq*sq+12*sq**2
    dNVdr_5= -16*sq+24*rq*sq+12*sq**2
    dNVdr_6= -54*rq*sq+27*sq-27*sq**2
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6],dtype=np.float64)

def dNNVds(rq,sq):
    dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
    dNVds_1= rq*(3-3*rq-6*sq)
    dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
    dNVds_3= -16*rq+24*rq*sq+12*rq**2
    dNVds_4= -8*rq+12*rq**2+24*rq*sq
    dNVds_5= 4-16*rq-8*sq+24*rq*sq+12*rq**2
    dNVds_6= -54*rq*sq+27*rq-27*rq**2
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6],dtype=np.float64)

def NNP(rq,sq):
    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return np.array([NP_0,NP_1,NP_2],dtype=np.float64)

###############################################################################

etas_diff=[1e19,8e19,2e20,5e20,1e21]

def viscosity(imat,exx,eyy,exy,temp,rheology):

    ee=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
    ee=max(ee,1e-20)

    # diffusion creep viscosity (gatt20)
    #Ediff_UM=410e3
    #Adiff_UM=1e-7
    #eta_diff=Adiff_UM**(-1.)*np.exp(Ediff_UM/(Rgas*temp)) #no 1/2 to match Garel
    eta_diff=etas_diff[idiff]

    #--------------
    if rheology==0: # constant viscosity

       eta_eff=1e22 

    #--------------
    if rheology==1: #diffusion creep only

       eta_eff=eta_diff

    #--------------
    if rheology==2: #diff + disl TANH 
       a0=4.40e+08
       b0=-52600
       a1=0.0211
       b1=0.000174
       a2=-41.8
       b2=0.0421
       c2=-1.14e-05
       stress = (a0+b0*temp)*(1+np.tanh((a1+b1*temp)*(np.log10(ee)-(a2+b2*temp+c2*temp**2))))
       eta_TANH = stress/(2.0*ee)
       eta_eff=1/(1/eta_diff+1/eta_TANH)

    #--------------
    if rheology==3: #diff + disl ERF 
       a0=4.39e+08
       b0=-2.21e4
       a1=3.01e-2
       b1=1.26e-4
       a2=-41.5
       b2=4.22e-2
       c2=-1.14e-05
       stress= (a0+b0*temp)*(1+math.erf((a1+b1*temp)*(np.log10(ee)-(a2+b2*temp+c2*temp**2))))
       eta_ERF = stress/(2.0*ee)
       eta_eff=1/(1/eta_diff+1/eta_ERF)

    #--------------
    if rheology==4: #diff + disl HT Gouriet 2019,gatt20 :
       A_disl=5.27e-29
       n_disl=4.5
       E_disl=443e3
       eta_disl=0.5*A_disl**(-1./n_disl)*ee**(-1.+1./n_disl)*np.exp(E_disl/(n_disl*Rgas*temp))
       eta_eff=1/(1/eta_diff+1/eta_disl)
       #print('ee=',ee,'disl=',eta_disl,'diff=',eta_diff,'eff=',eta_eff)

    #--------------
    if rheology==5: #diff + disl HT Garel 2014 subd
       n_disl=3.5
       A_disl=5e-16
       E_disl=540e3
       eta_disl=0.5*A_disl**(-1/n_disl)*ee**(-1.+1./n_disl)*np.exp(E_disl/(n_disl*Rgas*temp))
       eta_eff=1./(1./eta_diff+1./eta_disl)
       #print('ee=',ee,'disl=',eta_disl,'diff=',eta_diff,'eff=',eta_eff)

    #--------------
    #inclusion
    if imat==2: eta_eff=1e25

    #eta_eff=min(1e26,eta_eff)
    #eta_eff=max(1e18,eta_eff)

    return eta_eff

###############################################################################
# testing rheologies

if False:
   rheology=1
   exx=0
   eyy=0
   imat=1
   rheofile=open('rheology_test.ascii',"w")
   for exy in (-14,-14.2,-14.4,-14.6,-14.8,-15,-15.2,-15.4,-15.6,-15.8,-16):
       for temp in (1300,1325,1350,1375,1400,1425,1450,1475,1500,1525,1550,1575,1600):
           rheofile.write("%10e %10e %10e \n" %(exy,temp,viscosity(imat,exx,eyy,10**exy,temp,rheology)))
   rheofile.close()
   exit()

###############################################################################
# allowing for argument parsing through command line

if int(len(sys.argv) == 5):
   temperature = float(sys.argv[1])
   strain_rate_background=float(sys.argv[2])
   rheology= int(sys.argv[3])
   idiff= int(sys.argv[4])
else:
   temperature = 1600.0 # range 1300:100:1600
   strain_rate_background=-14 # range 1e-14, 1e-15, 1e-16
   rheology=1
   idiff= 2 

strain_rate_background=10**strain_rate_background

u_bc = Ly*strain_rate_background

print(temperature,strain_rate_background,rheology,idiff)

###############################################################################

print("---------------------------------------")
print("---------------fieldstone--------------")
print("---------------------------------------")

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

eta_ref=1e23

tol=5e-3

###############################################################################

print("temperature=",temperature)
print("strain_rate_background=",strain_rate_background)
print("u_bc=",u_bc)
print('nel', nel)
print('NV0', NV0)
print('NfemV', NfemV)
print('NfemP', NfemP)
print('Nfem ', Nfem)
print("---------------------------------------")
print("mesh size on inclusion=",2*np.pi*rad/np_object,'m')
print("---------------------------------------")

###############################################################################
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

###############################################################################
# grid point setup
###############################################################################

start = timing.time()

xV=np.zeros(NV,dtype=np.float64)     # x coordinates
yV=np.zeros(NV,dtype=np.float64)     # y coordinates

xV[0:NV0],yV[0:NV0]=np.loadtxt('mesh.1.node',unpack=True,usecols=[1,2],skiprows=1)

#print("xV (min/max): %.4f %.4f" %(np.min(xV[0:NV0]),np.max(xV[0:NV0])))
#print("yV (min/max): %.4f %.4f" %(np.min(yV[0:NV0]),np.max(yV[0:NV0])))

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

print("setup: connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# assigning material properties to elements
#################################################################
start = timing.time()

mat=np.zeros(nel,dtype=np.int16) 

for iel in range(0,nel):
    x_c=xV[iconV[6,iel]]
    y_c=yV[iconV[6,iel]]
    mat[iel]=1
    if (x_c-0.5*Lx)**2+(y_c-0.5*Ly)**2<rad**2:
       mat[iel]=2

print("material layout: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, NV):
    #Left boundary  
    if xV[i]/Lx<0.0000001:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0  
    #right boundary  
    if xV[i]/Lx>0.9999999:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0  
    #bottom boundary  
    if yV[i]/Lx<1e-6:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -u_bc   
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
    #top boundary  
    if yV[i]/Ly>0.9999999:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = +u_bc  
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 

    #if (xV[i]-0.5*Lx)**2+(yV[i]-0.5*Ly)**2<rad**2*1.01 and\
    #   (xV[i]-0.5*Lx)**2+(yV[i]-0.5*Ly)**2>rad**2*0.99:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 


print("define boundary conditions: %.3f s" % (timing.time() - start))

################################################################################################
# initial velocity 
# not strictly necessary but it will help with nl iterations
################################################################################################

u=np.zeros(NV,dtype=np.float64)
v=np.zeros(NV,dtype=np.float64)
umem=np.zeros(NV,dtype=np.float64)
vmem=np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    u[i]=(yV[i]-Ly/2)/(Ly/2)*u_bc

#np.savetxt('velocity0.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

################################################################################################
################################################################################################
# non linear iterations 
################################################################################################
################################################################################################

convfile=open('conv.ascii',"w")

for iiter in range(0,25):

    print('***** iteration: ',iiter,' *****')

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

        if iel%1000==0:
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
            exxq=0.
            eyyq=0.
            exyq=0.
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                exyq+=0.5*dNNNVdx[k]*v[iconV[k,iel]]+\
                      0.5*dNNNVdy[k]*u[iconV[k,iel]]

            # compute etaq
            etaq=viscosity(mat[iel],exxq,eyyq,exyq,temperature,rheology)

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

            # compute elemental rhs vector
            #for i in range(0,mV):
            #    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*rhoq
            #    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rhoq

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
    #sol=rhs

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*eta_ref/Ly

    print("     -> u (m,M) %.6e %.6e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.6e %.6e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.6e %.6e " %(np.min(p),np.max(p)))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

    print("solve time: %.3f s" % (timing.time() - start))

    #################################################################
    # compute area of elements
    #################################################################
    start = timing.time()

    area = np.zeros(nel,dtype=np.float64) 

    avrg_p=0
    vrms=0
    vrms1=0
    vrms2=0
    vol1=0
    vol2=0
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
               vrms1+=(uq**2+vq**2)*jcob*weightq 
               vol1+=jcob*weightq
               
            if mat[iel]==2:
               vrms2+=(uq**2+vq**2)*jcob*weightq 
               vol2+=jcob*weightq

            pq=0.0
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            avrg_p+=pq*jcob*weightq

        #end for
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))
    vrms1=np.sqrt(vrms1/vol1)
    vrms2=np.sqrt(vrms2/vol2)

    avrg_p/=(Lx*Ly)
    p-=avrg_p

    np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
    print("     -> total area (meas) %.6f " %(area.sum()))
    print("     -> total area (anal) %.6f " %(Lx*Ly))
    print("     -> vrms (cm/year)  = %e " %(vrms/cm*year))
    print("     -> vrms1 (cm/year) = %e " %(vrms1/cm*year))
    print("     -> vrms2 (cm/year) = %e " %(vrms2/cm*year))
    print("     -> avrg_p = %e " %(avrg_p))

    print("compute area & vrms: %.3f s" % (timing.time() - start))

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
        for k in range(0,mV):
            xc[iel]+=NNNV[k]*xV[iconV[k,iel]]
            yc[iel]+=NNNV[k]*yV[iconV[k,iel]]
            exx[iel]+=dNNNVdx[k]*u[iconV[k,iel]]
            eyy[iel]+=dNNNVdy[k]*v[iconV[k,iel]]
            exy[iel]+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
        e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

    #np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

    print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))

    print("compute sr and stress: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute nodal strainrate 
    ######################################################################
    start = timing.time()

    e_nodal = np.zeros(3*nel,dtype=np.float64)  
    exx_nodal = np.zeros(3*nel,dtype=np.float64)  
    eyy_nodal = np.zeros(3*nel,dtype=np.float64)  
    exy_nodal = np.zeros(3*nel,dtype=np.float64)  
    eta_nodal = np.zeros(3*nel,dtype=np.float64)  
    sigma_xx_nodal = np.zeros(3*nel,dtype=np.float64)  
    sigma_yy_nodal = np.zeros(3*nel,dtype=np.float64)  
    sigma_xy_nodal = np.zeros(3*nel,dtype=np.float64)  

    rVnodes=[0,1,0,0.5,0.5,0,1./3.]
    sVnodes=[0,0,1,0,0.5,0.5,1./3.]

    counter=0
    for iel in range(0,nel):
        for i in range(0,3):
            inode=iconV[i,iel]
            rq = rVnodes[i]
            sq = sVnodes[i]
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
            for k in range(0,mV):
                exx_nodal[counter]+=dNNNVdx[k]*u[iconV[k,iel]]
                eyy_nodal[counter]+=dNNNVdy[k]*v[iconV[k,iel]]
                exy_nodal[counter]+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                                    0.5*dNNNVdx[k]*v[iconV[k,iel]]
            eta_nodal[counter]+=viscosity(mat[iel],exx_nodal[counter],eyy_nodal[counter],\
                                                   exy_nodal[counter],temperature,rheology)
            sigma_xx_nodal[counter]=-p[counter]+2*eta_nodal[counter]*exx_nodal[counter]
            sigma_yy_nodal[counter]=-p[counter]+2*eta_nodal[counter]*eyy_nodal[counter]
            sigma_xy_nodal[counter]=            2*eta_nodal[counter]*exy_nodal[counter]
            counter+=1
        #end for
    #end for iel

    e_nodal[:]=np.sqrt(0.5*(exx_nodal[:]*exx_nodal[:]+eyy_nodal[:]*eyy_nodal[:])+exy_nodal[:]*exy_nodal[:])

    np.savetxt('strainrate.ascii',np.array([xP-Lx/2,yP-Ly/2,e_nodal,eta_nodal]).T,header='# x,y,e')

    print("     -> exx (m,M) %e %e" %(np.min(exx_nodal),np.max(exx_nodal)))
    print("     -> eyy (m,M) %e %e" %(np.min(eyy_nodal),np.max(eyy_nodal)))
    print("     -> exy (m,M) %e %e" %(np.min(exy_nodal),np.max(exy_nodal)))
    print("     -> e   (m,M) %e %e" %(np.min(e_nodal)  ,np.max(e_nodal)))
    print("     -> eta (m,M) %e %e" %(np.min(eta_nodal),np.max(eta_nodal)))
        
    print("compute nodal: %.3f s" % (timing.time() - start))

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
    #start = timing.time()

    #q=np.zeros(NV,dtype=np.float64)
    #p_el=np.zeros(nel,dtype=np.float64)
    #cc=np.zeros(NV,dtype=np.float64)

    #for iel in range(0,nel):
    #    q[iconV[0,iel]]+=p[iconP[0,iel]]
    #    cc[iconV[0,iel]]+=1.
    #    q[iconV[1,iel]]+=p[iconP[1,iel]]
    #    cc[iconV[1,iel]]+=1.
    #    q[iconV[2,iel]]+=p[iconP[2,iel]]
    #    cc[iconV[2,iel]]+=1.
    #    q[iconV[3,iel]]+=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    #    cc[iconV[3,iel]]+=1.
    #    q[iconV[4,iel]]+=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    #    cc[iconV[4,iel]]+=1.
    #    q[iconV[5,iel]]+=(p[iconP[0,iel]]+p[iconP[2,iel]])*0.5
    #    cc[iconV[5,iel]]+=1.
    #    p_el[iel]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.

    #for i in range(0,NV):
    #    if cc[i] != 0:
    #       q[i]=q[i]/cc[i]

    #print("compute nodal pressure: %.3f s" % (timing.time() - start))

    #####################################################################
    # interpolate strain rate onto velocity grid points
    #####################################################################
    #start = timing.time()

    #sr=np.zeros(NV,dtype=np.float64)
    #sr_el=np.zeros(nel,dtype=np.float64)
    #cc=np.zeros(NV,dtype=np.float64)

    #for iel in range(0,nel):
    #    sr[iconV[0,iel]]+=e[iel]
    #    cc[iconV[0,iel]]+=1.
    #    sr[iconV[1,iel]]+=e[iel]
    #    cc[iconV[1,iel]]+=1.
    #    sr[iconV[2,iel]]+=e[iel]
    #    cc[iconV[2,iel]]+=1.
    #    sr[iconV[3,iel]]+=e[iel]
    #    cc[iconV[3,iel]]+=1.
    #    sr[iconV[4,iel]]+=e[iel]
    #    cc[iconV[4,iel]]+=1.
    #    sr[iconV[5,iel]]+=e[iel]
    #    cc[iconV[5,iel]]+=1.
    #    sr_el[iel]=e[iel]

    #for i in range(0,NV):
    #    if cc[i] != 0:
    #       sr[i]=sr[i]/cc[i]

    #print("compute nodal strain rate: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute stress tensor 
    #####################################################################
    #start = timing.time()

    #sigma_xx=np.zeros(nel,dtype=np.float64)
    #sigma_yy=np.zeros(nel,dtype=np.float64)
    #sigma_xy=np.zeros(nel,dtype=np.float64)
    #for iel in range(0,nel):
    #    sigma_xx[iel]=-p_el[iel]+2*viscosity(mat[iel],exx[iel],eyy[iel],exy[iel],temperature,rheology)*exx[iel]
    #    sigma_yy[iel]=-p_el[iel]+2*viscosity(mat[iel],exx[iel],eyy[iel],exy[iel],temperature,rheology)*eyy[iel]
    #    sigma_xy[iel]=           2*viscosity(mat[iel],exx[iel],eyy[iel],exy[iel],temperature,rheology)*exy[iel]

    #np.savetxt('stress.ascii',np.array([xc,yc,sigma_xx,sigma_yy,sigma_xy]).T,header='# x,y,sigma_xx,sigma_yy,sigma_xy')

    #print("compute stress: %.3f s" % (timing.time() - start))

    #####################################################################
    # plot of solution
    # the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
    #####################################################################
    #start = timing.time()

    #filename = 'solution_vel_{:04d}.vtu'.format(iiter)
    #vtufile=open(filename,"w")
    #vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    #vtufile.write("<UnstructuredGrid> \n")
    #vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    #vtufile.write("<Points> \n")
    #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("</Points> \n")
    #####
    #vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (area[iel]))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%7e\n" % (viscosity(mat[iel],exx[iel],eyy[iel],exy[iel],temperature,rheology)))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%d\n" % (mat[iel]))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='p (el)' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%7e\n" % (p_el[iel]))
    #vtufile.write("</DataArray>\n")
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
    #vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (e[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%e\n" % (sigma_xx[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%e\n" % (sigma_yy[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%e\n" % (sigma_xy[iel]))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("</CellData>\n")
    #####
    #vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='p (nod)' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%10e \n" %q[i])
    #vtufile.write("</DataArray>\n")
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
    #vtufile.write("</PointData>\n")
    #####
    #vtufile.write("<Cells>\n")
    #--
    #vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
    #                                          iconV[3,iel],iconV[4,iel],iconV[5,iel]))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%d \n" %((iel+1)*6))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    #for iel in range (0,nel):
    #    vtufile.write("%d \n" %22)
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("</Cells>\n")
    #####
    #vtufile.write("</Piece>\n")
    #vtufile.write("</UnstructuredGrid>\n")
    #vtufile.write("</VTKFile>\n")
    #vtufile.close()

    filename = 'solution_{:04d}.vtu'.format(iiter)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(3*nel,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for iel in range(nel):
        vtufile.write("%10e %10e %10e \n" %(xP[iconP[0,iel]],yP[iconP[0,iel]],0.))
        vtufile.write("%10e %10e %10e \n" %(xP[iconP[1,iel]],yP[iconP[1,iel]],0.))
        vtufile.write("%10e %10e %10e \n" %(xP[iconP[2,iel]],yP[iconP[2,iel]],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for iel in range(nel):
        vtufile.write("%10e \n" %(p[iconP[0,iel]]))
        vtufile.write("%10e \n" %(p[iconP[1,iel]]))
        vtufile.write("%10e \n" %(p[iconP[2,iel]]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e \n" %eta_nodal[3*i])
        vtufile.write("%10e \n" %eta_nodal[3*i+1])
        vtufile.write("%10e \n" %eta_nodal[3*i+2])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e \n" %e_nodal[3*i])
        vtufile.write("%10e \n" %e_nodal[3*i+1])
        vtufile.write("%10e \n" %e_nodal[3*i+2])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e \n" %exx_nodal[3*i])
        vtufile.write("%10e \n" %exx_nodal[3*i+1])
        vtufile.write("%10e \n" %exx_nodal[3*i+2])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e \n" %eyy_nodal[3*i])
        vtufile.write("%10e \n" %eyy_nodal[3*i+1])
        vtufile.write("%10e \n" %eyy_nodal[3*i+2])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e \n" %exy_nodal[3*i])
        vtufile.write("%10e \n" %exy_nodal[3*i+1])
        vtufile.write("%10e \n" %exy_nodal[3*i+2])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e \n" %sigma_xx_nodal[3*i])
        vtufile.write("%10e \n" %sigma_xx_nodal[3*i+1])
        vtufile.write("%10e \n" %sigma_xx_nodal[3*i+2])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e \n" %sigma_yy_nodal[3*i])
        vtufile.write("%10e \n" %sigma_yy_nodal[3*i+1])
        vtufile.write("%10e \n" %sigma_yy_nodal[3*i+2])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%10e \n" %sigma_xy_nodal[3*i])
        vtufile.write("%10e \n" %sigma_xy_nodal[3*i+1])
        vtufile.write("%10e \n" %sigma_xy_nodal[3*i+2])
    vtufile.write("</DataArray>\n")


    vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d\n" % (mat[iel]))
        vtufile.write("%d\n" % (mat[iel]))
        vtufile.write("%d\n" % (mat[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (area[iel]))
        vtufile.write("%e\n" % (area[iel]))
        vtufile.write("%e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='diameter' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (np.sqrt(area[iel])))
        vtufile.write("%e\n" % (np.sqrt(area[iel])))
        vtufile.write("%e\n" % (np.sqrt(area[iel])))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%10e %10e %10e \n" %(u[iconV[0,iel]]/cm*year,v[iconV[0,iel]]/cm*year,0.))
        vtufile.write("%10e %10e %10e \n" %(u[iconV[1,iel]]/cm*year,v[iconV[1,iel]]/cm*year,0.))
        vtufile.write("%10e %10e %10e \n" %(u[iconV[2,iel]]/cm*year,v[iconV[2,iel]]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    counter=0
    for iel in range(0,nel):
        vtufile.write("%d %d %d \n" %(counter,counter+1,counter+2))
        counter+=3
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*3))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5) 
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    ######################################################################
    # convergence criterion is based on difference between two consecutively
    # obtained velocity fields, normalised by the boundary condition velocity

    chi_u=LA.norm(u-umem,2)/u_bc # vx convergence indicator
    chi_v=LA.norm(v-vmem,2)/u_bc # vy convergence indicator

    print('     -> convergence u,v: %.3e %.3e | tol= %.2e' %(chi_u,chi_v,tol))

    convfile.write("%f %10e %10e %10e\n" %(iiter,chi_u,chi_v,tol))
    convfile.flush()

    umem[:]=u[:]
    vmem[:]=v[:]

    if chi_u<tol and chi_v<tol:
       print('     ***converged***')
       break

#end for time step

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
