import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def NNT(r,s,order):
    if order==1:
       N_0=0.25*(1.-r)*(1.-s)
       N_1=0.25*(1.+r)*(1.-s)
       N_2=0.25*(1.-r)*(1.+s)
       N_3=0.25*(1.+r)*(1.+s)
       return N_0,N_1,N_2,N_3
    if order==2:
       N_0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       N_1=    (1.-r**2) * 0.5*s*(s-1.)
       N_2= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       N_3= 0.5*r*(r-1.) *    (1.-s**2)
       N_4=    (1.-r**2) *    (1.-s**2)
       N_5= 0.5*r*(r+1.) *    (1.-s**2)
       N_6= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       N_7=    (1.-r**2) * 0.5*s*(s+1.)
       N_8= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8

def dNNTdr(r,s,order):
    if order==1:
       dNdr_0=-0.25*(1.-s)
       dNdr_1=+0.25*(1.-s)
       dNdr_2=-0.25*(1.+s)
       dNdr_3=+0.25*(1.+s)
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3
    if order==2:
       dNdr_0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       dNdr_1=       (-2.*r) * 0.5*s*(s-1)
       dNdr_2= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       dNdr_3= 0.5*(2.*r-1.) *   (1.-s**2)
       dNdr_4=       (-2.*r) *   (1.-s**2)
       dNdr_5= 0.5*(2.*r+1.) *   (1.-s**2)
       dNdr_6= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       dNdr_7=       (-2.*r) * 0.5*s*(s+1)
       dNdr_8= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8

def dNNTds(r,s,order):
    if order==1:
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.-r)
       dNds_3=+0.25*(1.+r)
       return dNds_0,dNds_1,dNds_2,dNds_3
    if order==2:
       dNds_0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       dNds_1=    (1.-r**2) * 0.5*(2.*s-1.)
       dNds_2= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       dNds_3= 0.5*r*(r-1.) *       (-2.*s)
       dNds_4=    (1.-r**2) *       (-2.*s)
       dNds_5= 0.5*r*(r+1.) *       (-2.*s)
       dNds_6= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       dNds_7=    (1.-r**2) * 0.5*(2.*s+1.)
       dNds_8= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8

#------------------------------------------------------------------------------

sqrt3=np.sqrt(3.)
sqrt2=np.sqrt(2.)
sqrt15=np.sqrt(15.)
eps=1.e-10 
cm=0.01
year=365.*24.*3600.

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2       # number of space dimensions
ndofT=1      # number of degrees of freedom per node
hcond=0.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density


if int(len(sys.argv) == 4):
   experiment=int(sys.argv[1])
   order     =int(sys.argv[2])
   supg_type =int(sys.argv[3])
else:
   experiment=6
   order=2
   supg_type=1

if order==1:
   m=4          # number of nodes making up an element
if order==2:
   m=9

use_bdf=False
bdf_order=2

if experiment==1: # rotating cone
   nelx = 30
   nely = 30
   Lx=1.  
   Ly=1.  
   tfinal=2.*np.pi
   CFLnb=0.5
   xmin=0.
   ymin=0.
   every=10

if experiment==2: # rotating 3 objects
   nelx=64
   nely=64
   Lx=2.   
   Ly=2.   
   tfinal=2.*np.pi
   CFLnb=0.5
   every=10
   xmin=-1.
   ymin=-1.

if experiment==3: # front advection
   nelx=64
   nely=16
   Lx=1.  
   Ly=0.25 
   tfinal=0.5
   CFLnb=0.25
   xmin=0.
   ymin=0.
   every=10

if experiment==4: # skew advection
   nelx=10
   nely=10
   Lx=1.   
   Ly=1.   
   tfinal=3.
   CFLnb=0.1
   xmin=0.
   ymin=0.
   every=10

if experiment==5: # quarter circle
   nelx=16
   nely=16
   Lx=1.   
   Ly=1.   
   tfinal=4.
   CFLnb=0.1
   xmin=0.
   ymin=0.
   every=10

if experiment==6: # elastic slab
   nelx=50
   nely=50
   Lx=1e6
   Ly=1e6
   tfinal=15e6*year 
   CFLnb=0.25
   xmin=0.
   ymin=0.
   every=1

if experiment==7: # elastic slab
   nelx=50
   nely=50
   Lx=1e6
   Ly=1e6
   tfinal=30e6*year 
   CFLnb=0.25
   xmin=0.
   ymin=0.
   every=5

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
NV=nnx*nny        # number of nodes
nel=nelx*nely     # number of elements, total
NfemT=NV*ndofT    # Total number of degrees of temperature freedom

# alphaT=1: implicit
# alphaT=0: explicit
# alphaT=0.5: Crank-Nicolson

alphaT=0.5

#####################################################################

if order==1:
   nqperdim=2
   qcoords=[-1./sqrt3,1./sqrt3]
   qweights=[1.,1.]

if order==2:
   nqperdim=3
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

#####################################################################

stats_T_file=open('stats_T.ascii',"w")
avrg_T_file=open('avrg_T.ascii',"w")
ET_file=open('ET.ascii',"w")

#####################################################################

print ('experiment =',experiment)
print ('order      =',order)
print ('supg_type  =',supg_type)
print ('nnx        =',nnx)
print ('nny        =',nny)
print ('NV         =',NV)
print ('nel        =',nel)
print ('NfemT      =',NfemT)
print ('nqperdim   =',nqperdim)
print("-----------------------------")

#####################################################################
# grid point setup 
#####################################################################
start = timing.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
u = np.zeros(NV,dtype=np.float64)  # x-component velocity
v = np.zeros(NV,dtype=np.float64)  # y-component velocity

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx/order+xmin
        y[counter]=j*hy/order+ymin
        if experiment==1:
           u[counter]=-(y[counter]-Ly/2)
           v[counter]=+(x[counter]-Lx/2)
        if experiment==2:
           u[counter]=-y[counter]
           v[counter]=+x[counter]
        if experiment==3:
           u[counter]=1
           v[counter]=0
        if experiment==4:
           u[counter]=np.cos(30./180.*np.pi)
           v[counter]=np.sin(30./180.*np.pi)
        if experiment==5:
           u[counter]=y[counter]
           v[counter]=1-x[counter]
        if experiment==6:
           u[counter]=0
           v[counter]=-x[counter]/Lx*cm/year
        if experiment==7:
           xx=x[counter]/Lx
           yy=y[counter]/Ly
           u[counter]=(xx*xx*(1.-xx)**2*(2.*yy-6.*yy*yy+4*yy*yy*yy))*cm/year  *100
           v[counter]=(-yy*yy*(1.-yy)**2*(2.*xx-6.*xx*xx+4*xx*xx*xx))*cm/year *100
        counter += 1
    #end for
#end for

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("mesh (%.3fs)" % (timing.time() - start))

#####################################################################
# connectivity
#####################################################################
start = timing.time()

icon =np.zeros((m,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                icon[counter2,counter]=i*order+l+j*order*nnx+nnx*k
                counter2+=1
            #end for
        #end for
        counter += 1
    #end for
#end for

#connectivity array for plotting
nel2=(nnx-1)*(nny-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int16)
counter = 0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        iconQ1[0,counter]=i+j*nnx
        iconQ1[1,counter]=i+1+j*nnx
        iconQ1[2,counter]=i+1+(j+1)*nnx
        iconQ1[3,counter]=i+(j+1)*nnx
        counter += 1 
    #end for
#end for

print("connectivity (%.3fs)" % (timing.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

if experiment==1:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if x[i]/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y[i]/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y[i]/Ly>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

if experiment==2:
   for i in range(0,NV):
       if (x[i]-xmin)/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if (x[i]-xmin)/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
       if (y[i]-ymin)/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if (y[i]-ymin)/Ly>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

if experiment==3:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=1.
       if x[i]/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

if experiment==4:
   for i in range(0,NV):
       if y[i]/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if x[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=1.
   #end for

if experiment==5:
   for i in range(0,NV):
       if y[i]/Ly<eps:                        #bottom
          if x[i]<1./3.:
             bc_fixT[i]=True ; bc_valT[i]=0.
          else:
             bc_fixT[i]=True ; bc_valT[i]=1.
       #if y[i]/Ly>(1-eps):
       #   bc_fixT[i]=True ; bc_valT[i]=0.
       if x[i]/Lx<eps:                        # left bc
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

if experiment==6 or experiment==17:
   for i in range(0,NV):
       if x[i]/Lx<eps and np.abs(y[i]-Ly/2)<=300e3:
          bc_fixT[i]=True ; bc_valT[i]=1.
       if x[i]/Lx<eps and np.abs(y[i]-Ly/2)>300e3:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y[i]/Ly<eps:               
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y[i]/Ly>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

print("boundary conditions (%.3fs)" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################
start = timing.time()

T = np.zeros(NV,dtype=np.float64)
Tm1 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-1
Tm2 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-2
Tm3 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-3
Tm4 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-4
Tm5 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-5

if experiment==1:
   xc=2./3.
   yc=2./3.
   sigma=0.2
   for i in range(0,NV):
       if (x[i]-xc)**2+(y[i]-yc)**2<=sigma**2:
          T[i]=0.25*(1+np.cos(np.pi*(x[i]-xc)/sigma))*(1+np.cos(np.pi*(y[i]-yc)/sigma))
       #end if
   #end for

if experiment==2:
   for i in range(0,NV):
       xi=x[i]
       yi=y[i]
       if np.sqrt(xi**2+(yi-0.5)**2)<0.3 and (np.abs(xi)>=0.05 or yi>=0.7):
          T[i]=1
       #end if
       if np.sqrt((x[i])**2+(y[i]+0.5)**2)<0.3:
          T[i]=1-np.sqrt((x[i])**2+(y[i]+0.5)**2)/0.3
       #end if
       if np.sqrt((x[i]+0.5)**2+(y[i])**2)<0.3:
          T[i]=0.25*(1+np.cos(np.pi*np.sqrt((xi+0.5)**2+yi**2)/0.3))
       #end if
   #end for

if experiment==3:
   for i in range(0,NV):
       if x[i]<0.25:
          T[i]=1
       #end if
   #end for

if experiment==4:
   T[i]=0.

if experiment==5:
   T[i]=0.

if experiment==6 or experiment==7:
   for i in range(0,NV):
       if x[i]<=800e3 and np.abs(y[i]-Ly/2)<=300e3:
          T[i]=1
       else:
          T[i]=0
       #end if
    #end for

Tm1[:]=T[:]
Tm2[:]=T[:]
Tm3[:]=T[:]
Tm4[:]=T[:]
Tm5[:]=T[:]

#np.savetxt('T_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

print("initial temperature (%.3fs)" % (timing.time() - start))

#################################################################
# compute timestep
#################################################################
start = timing.time()

dt=CFLnb*hx/np.max(np.sqrt(u**2+v**2))/order
print('dt=',dt)
nstep=int(tfinal/dt)
print('nstep=',nstep)

print("compute timestep (%.3fs)" % (timing.time() - start))

#####################################################################
# create necessary arrays 
#####################################################################
start = timing.time()

N     = np.zeros(m,dtype=np.float64)    # shape functions
dNdx  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
Tvectm1 = np.zeros(m,dtype=np.float64)   
Tvectm2 = np.zeros(m,dtype=np.float64)   
Tvectm3 = np.zeros(m,dtype=np.float64)   
Tvectm4 = np.zeros(m,dtype=np.float64)   
Tvectm5 = np.zeros(m,dtype=np.float64)   
NNNT    = np.zeros(m,dtype=np.float64)           # shape functions 
dNNNTdx = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTdy = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTdr = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTds = np.zeros(m,dtype=np.float64)           # shape functions derivatives
    
print("create arrays (%.3fs)" % (timing.time() - start))

#==============================================================================
# time stepping loop
#==============================================================================

model_time=0.

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")


    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    N_mat_supg = np.zeros((m,1),dtype=np.float64)         # shape functions
    tau_supg = np.zeros(nel*nqperdim**ndim,dtype=np.float64)    

    counterq=0
    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m):
            Tvectm1[k]=Tm1[icon[k,iel]]
            Tvectm2[k]=Tm2[icon[k,iel]]
            Tvectm3[k]=Tm3[icon[k,iel]]
            Tvectm4[k]=Tm4[icon[k,iel]]
            Tvectm5[k]=Tm5[icon[k,iel]]
        #end for

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNT[0:m]=NNT(rq,sq,order)
                dNNNTdr[0:m]=dNNTdr(rq,sq,order)
                dNNNTds[0:m]=dNNTds(rq,sq,order)
                N_mat[0:m,0]=NNT(rq,sq,order)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0] += dNNNTdr[k]*x[icon[k,iel]]
                    jcb[0,1] += dNNNTdr[k]*y[icon[k,iel]]
                    jcb[1,0] += dNNNTds[k]*x[icon[k,iel]]
                    jcb[1,1] += dNNNTds[k]*y[icon[k,iel]]
                #end for

                # calculate the determinant of the jacobian
                jcob=np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,m):
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                    dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                    B_mat[0,k]=dNNNTdx[k]
                    B_mat[1,k]=dNNNTdy[k]
                #end for

                if supg_type==0:
                   tau_supg[counterq]=0.
                elif supg_type==1:
                      tau_supg[counterq]=(hx*sqrt2)/2/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)
                elif supg_type==2:
                      tau_supg[counterq]=(hx*sqrt2)/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)/sqrt15
                else:
                   exit("supg_type: wrong value")
                     
                N_mat_supg=N_mat+tau_supg[counterq]*np.transpose(vel.dot(B_mat))

                # compute mass matrix
                MM=N_mat_supg.dot(N_mat.T)*rho0*hcapa*weightq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka=N_mat_supg.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

                if use_bdf and istep>bdf_order:
                   if bdf_order==1:
                      a_el+=MM+1.*dt*(Ka+Kd)
                      b_el+=MM.dot(Tvectm1)
                   #end if
                   if bdf_order==2:
                      a_el+=MM+2./3.*dt*(Ka+Kd)
                      b_el+=4./3.*MM.dot(Tvectm1)\
                           -1./3.*MM.dot(Tvectm2)
                   #end if
                   if bdf_order==3:
                      a_el+=MM+6./11.*dt*(Ka+Kd)
                      b_el+=18./11.*MM.dot(Tvectm1)\
                           -9./11.*MM.dot(Tvectm2)\
                           +2./11.*MM.dot(Tvectm3)
                   #end if
                   if bdf_order==4:
                      a_el+=MM+12./25.*dt*(Ka+Kd)
                      b_el+=48./25.*MM.dot(Tvectm1)\
                           -36./25.*MM.dot(Tvectm2)\
                           +16./25.*MM.dot(Tvectm3)\
                           -3./25.*MM.dot(Tvectm4)
                   #end if
                   if bdf_order==5:
                      a_el+=MM+60./137.*dt*(Ka+Kd)
                      b_el+=300./137.*MM.dot(Tvectm1)\
                           -300./137.*MM.dot(Tvectm2)\
                           +200./137.*MM.dot(Tvectm3)\
                           -75./137.*MM.dot(Tvectm4)\
                           +12./137.*MM.dot(Tvectm5)
                   #end if
                else:
                   a_el+=MM+alphaT*(Ka+Kd)*dt
                   b_el+=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvectm1)
                #end if

                counterq+=1
            #end for jq
        #end for iq

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon[k1,iel]
            if bc_fixT[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            #end if
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,m):
            m1=icon[k1,iel]
            for k2 in range(0,m):
                m2=icon[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    #end for iel
    
    print("     -> matrix (m,M) %.4e %.4e " %(np.min(A_mat),np.max(A_mat)))
    print("     -> rhs (m,M) %.4e %.4e " %(np.min(rhs),np.max(rhs)))

    print("     -> tau_supg (m,M) %e %e " %(np.min(tau_supg),np.max(tau_supg)))

    if istep==0:
       np.savetxt('tau_supg.ascii',np.array(tau_supg).T,header='# x,y,T')

    print("build FEM matrix: %.3fs" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    stats_T_file.write("%e %e %e \n" %(model_time,np.min(T),np.max(T))) ; stats_T_file.flush()

    print("solve T time: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute average of temperature using a 4x4 quadrature
    #####################################################################
    start = timing.time()

    qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
    qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
    qw4a=(18-np.sqrt(30.))/36.
    qw4b=(18+np.sqrt(30.))/36.
    qcoords4=[-qc4a,-qc4b,qc4b,qc4a]
    qweights4=[qw4a,qw4b,qw4b,qw4a]

    ET=0.
    avrg_T=0.
    for iel in range (0,nel):
        for iq in range(0,4):
            for jq in range(0,4):
                rq=qcoords4[iq]
                sq=qcoords4[jq]
                weightq=qweights4[iq]*qweights4[jq]
                NNNT[0:m]=NNT(rq,sq,order)
                dNNNTdr[0:m]=dNNTdr(rq,sq,order)
                dNNNTds[0:m]=dNNTds(rq,sq,order)
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNNNTdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNNNTdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNNNTds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNNNTds[k]*y[icon[k,iel]]
                jcob=np.linalg.det(jcb)
                Tq=0.
                for k in range(0,m):
                    Tq+=NNNT[k]*T[icon[k,iel]]
                avrg_T+=Tq*weightq*jcob
                ET+=rho0*hcapa*(abs(Tq))*weightq*jcob
            #end for
        #end for
    #end for
    avrg_T/=Lx*Ly

    ET_file.write("%e %.10e \n" %(model_time,ET))         ; ET_file.flush()
    avrg_T_file.write("%e %.10e \n" %(model_time,avrg_T)) ; avrg_T_file.flush()

    print("     -> avrg T= %.6e" % avrg_T)

    print("compute <T>,M: %.3f s" % (timing.time() - start))

    #################################################################
    # visualisation 
    #################################################################

    if istep%every==0:

       start = timing.time()

       #filename = 'T_{:04d}.ascii'.format(istep) 
       #np.savetxt(filename,np.array([x,y,T]).T,header='# x,y,T')

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel2))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau' Format='ascii'> \n")
       for i in range(0,NV):
           if np.sqrt(u[i]**2+v[i]**2)<eps:
              taunode=0
           else:
              taunode=(hx*sqrt2)/2/order/np.sqrt(u[i]**2+v[i]**2)
           vtufile.write("%e \n" %taunode)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if order==1:
          for iel in range (0,nel2):
              vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[3,iel],icon[2,iel]))
       if order==2:
          for iel in range (0,nel2):
              vtufile.write("%d %d %d %d \n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       filename = 'solution_{:04d}.pdf'.format(istep) 
       fig = plt.figure ()
       ax = fig.gca(projection='3d')
       ax.plot_surface(x.reshape ((nny,nnx)),y.reshape((nny,nnx)),T.reshape((nny,nnx)),color = 'darkseagreen')
       ax.set_xlabel ( 'X [ m ] ')
       ax.set_ylabel ( 'Y [ m ] ')
       ax.set_zlabel ( ' Temperature  [ C ] ')
       plt.title('Timestep  %.2d' %(istep),loc='right')
       plt.grid ()
       plt.savefig(filename)
       #plt.show ()
       plt.close()

       print("export to files: %.3f s" % (timing.time() - start))

    #end if

    Tm5=Tm4
    Tm4=Tm3
    Tm3=Tm2
    Tm2=Tm1
    Tm1=T

    model_time+=dt
    print ("model_time=",model_time)
    
#end for istep

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
