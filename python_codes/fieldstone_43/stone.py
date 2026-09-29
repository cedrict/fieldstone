import numpy as np
import sys as sys
import scipy
import time as clock
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix

###############################################################################

def rhs(x,y,experiment):
    match(experiment):
        case(1|2|3|4|5|6|7|8):
            val=0.
        case(9):
            if (x+0.75)**2+(y+0.75)**2<0.01:
               val=10.
            else:
               val=0.
    return val

###############################################################################

def basis_functions_T(r,s,order):
    if order==1:
       N0=0.25*(1.-r)*(1.-s)
       N1=0.25*(1.+r)*(1.-s)
       N2=0.25*(1.-r)*(1.+s)
       N3=0.25*(1.+r)*(1.+s)
       return np.array([N0,N1,N2,N3],dtype=np.float64)
    if order==2:
       N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       N1=    (1.-r**2) * 0.5*s*(s-1.)
       N2= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       N3= 0.5*r*(r-1.) *    (1.-s**2)
       N4=    (1.-r**2) *    (1.-s**2)
       N5= 0.5*r*(r+1.) *    (1.-s**2)
       N6= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       N7=    (1.-r**2) * 0.5*s*(s+1.)
       N8= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

def basis_functions_T_dr(r,s,order):
    if order==1:
       dNdr0=-0.25*(1.-s)
       dNdr1=+0.25*(1.-s)
       dNdr2=-0.25*(1.+s)
       dNdr3=+0.25*(1.+s)
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)
    if order==2:
       dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       dNdr1=       (-2.*r) * 0.5*s*(s-1)
       dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       dNdr3= 0.5*(2.*r-1.) *   (1.-s**2)
       dNdr4=       (-2.*r) *   (1.-s**2)
       dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
       dNdr6= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       dNdr7=       (-2.*r) * 0.5*s*(s+1)
       dNdr8= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_T_ds(r,s,order):
    if order==1:
       dNds0=-0.25*(1.-r)
       dNds1=-0.25*(1.+r)
       dNds2=+0.25*(1.-r)
       dNds3=+0.25*(1.+r)
       return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)
    if order==2:
       dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       dNds1=    (1.-r**2) * 0.5*(2.*s-1.)
       dNds2= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       dNds3= 0.5*r*(r-1.) *       (-2.*s)
       dNds4=    (1.-r**2) *       (-2.*s)
       dNds5= 0.5*r*(r+1.) *       (-2.*s)
       dNds6= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       dNds7=    (1.-r**2) * 0.5*(2.*s+1.)
       dNds8= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

###############################################################################

sqrt3=np.sqrt(3.)
sqrt2=np.sqrt(2.)
sqrt15=np.sqrt(15.)
eps=1.e-10 
cm=0.01
year=365.25*24.*3600.

print("*******************************")
print("********** stone 043 **********")
print("*******************************")

ndim=2       # number of space dimensions
hcond=0.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density

if int(len(sys.argv) == 4):
   experiment=int(sys.argv[1])
   order     =int(sys.argv[2])
   supg_type =int(sys.argv[3])
else:
   experiment=2
   order=2
   supg_type=1

if order==1: m=4
if order==2: m=9

use_bdf=False
bdf_order=2

if experiment==1: # rotating cone
   nelx=30
   nely=nelx
   Lx=1.  
   Ly=1.  
   tfinal=2.*np.pi
   CFLnb=0.5
   xmin=0.
   ymin=0.
   every=10
   steady_state=False

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
   steady_state=False

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
   steady_state=False

if experiment==4: # skew advection
   nelx=16
   nely=nelx
   Lx=1.   
   Ly=1.   
   tfinal=3.
   CFLnb=0.1
   xmin=0.
   ymin=0.
   every=10
   steady_state=False

if experiment==5: # quarter circle
   nelx=16
   nely=16
   Lx=1.   
   Ly=1.   
   tfinal=2.
   CFLnb=0.25
   xmin=0.
   ymin=0.
   every=25
   steady_state=False

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
   steady_state=False

if experiment==7: # elastic slab
   nelx=32
   nely=32
   Lx=1e6
   Ly=1e6
   tfinal=30e6*year 
   CFLnb=0.25
   xmin=0.
   ymin=0.
   every=5
   steady_state=False

if experiment==8: # advection cone Li book
   nelx=200
   nely=4
   Lx=1
   Ly=0.05
   tfinal=8
   CFLnb=0.5
   xmin=0.
   ymin=0.
   every=10
   steady_state=False

if experiment==9: #step-9
   nelx=128
   nely=nelx
   Lx=2
   Ly=2
   xmin=-1.
   ymin=-1.
   every=5
   tfinal=1.25
   CFLnb=0.5
   steady_state=True

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
nn_T=nnx*nny      # number of nodes
nel=nelx*nely     # number of elements, total
Nfem_T=nn_T       # Total number of degrees of temperature freedom

debug=False

# alphaT=1: implicit
# alphaT=0: explicit
# alphaT=0.5: Crank-Nicolson

alphaT=0.5

###############################################################################

if order==1:
   nq_per_dim=2
   qcoords=[-1./sqrt3,1./sqrt3]
   qweights=[1.,1.]

if order==2:
   nq_per_dim=3
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

###############################################################################

stats_T_file=open('stats_T.ascii',"w")
avrg_T_file=open('avrg_T.ascii',"w")
ET_file=open('ET.ascii',"w")

###############################################################################

print ('experiment =',experiment)
print ('order      =',order)
print ('supg_type  =',supg_type)
print ('nnx        =',nnx)
print ('nny        =',nny)
print ('nn_T       =',nn_T)
print ('nel        =',nel)
print ('Nfem_T     =',Nfem_T)
print ('nq_per_dim =',nq_per_dim)
print ('CFLnb      =',CFLnb)
print("-----------------------------")

###############################################################################
# grid point setup 
###############################################################################
start=clock.time()

x_T=np.zeros(nn_T,dtype=np.float64)
y_T=np.zeros(nn_T,dtype=np.float64)
u=np.zeros(nn_T,dtype=np.float64)
v=np.zeros(nn_T,dtype=np.float64)

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_T[counter]=i*hx/order+xmin
        y_T[counter]=j*hy/order+ymin
        if experiment==1:
           u[counter]=-(y_T[counter]-Ly/2)
           v[counter]=+(x_T[counter]-Lx/2)
        if experiment==2:
           u[counter]=-y_T[counter]
           v[counter]=+x_T[counter]
        if experiment==3:
           u[counter]=1
           v[counter]=0
        if experiment==4:
           u[counter]=np.cos(30./180.*np.pi)
           v[counter]=np.sin(30./180.*np.pi)
        if experiment==5:
           u[counter]=y_T[counter]
           v[counter]=1-x_T[counter]
        if experiment==6:
           u[counter]=0
           v[counter]=-x_T[counter]/Lx*cm/year
        if experiment==7:
           xx=x_T[counter]/Lx
           yy=y_T[counter]/Ly
           u[counter]=(xx*xx*(1.-xx)**2*(2.*yy-6.*yy*yy+4*yy*yy*yy))*cm/year  *100
           v[counter]=(-yy*yy*(1.-yy)**2*(2.*xx-6.*xx*xx+4*xx*xx*xx))*cm/year *100
        if experiment==8:
           u[counter]=0.1
           v[counter]=0
        if experiment==9:
           u[counter]=2
           v[counter]=1+4./5.*np.sin(8*np.pi*x_T[counter])
        counter += 1
    #end for
#end for

if debug: np.savetxt('grid.ascii',np.array([x_T,y_T]).T,header='# x,y')

print("mesh (%.3fs)" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_T=np.zeros((m,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                icon_T[counter2,counter]=i*order+l+j*order*nnx+nnx*k
                counter2+=1
            #end for
        #end for
        counter += 1
    #end for
#end for

#connectivity array for plotting
nel2=(nnx-1)*(nny-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int32)
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

print("connectivity (%.3fs)" % (clock.time()-start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start=clock.time()

bc_fixT=np.zeros(Nfem_T,dtype=bool)  
bc_valT=np.zeros(Nfem_T,dtype=np.float64) 

if experiment==1 or experiment==2:
   for i in range(0,nn_T):
       if (x_T[i]-xmin)/Lx<eps and u[i]>0:     bc_fixT[i]=True ; bc_valT[i]=0.
       if (x_T[i]-xmin)/Lx>(1-eps) and u[i]<0: bc_fixT[i]=True ; bc_valT[i]=0.
       if (y_T[i]-ymin)/Ly<eps and v[i]>0:     bc_fixT[i]=True ; bc_valT[i]=0.
       if (y_T[i]-ymin)/Ly>(1-eps) and v[i]<0: bc_fixT[i]=True ; bc_valT[i]=0.

if experiment==3:
   for i in range(0,nn_T):
       if x_T[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=1.
   #end for

if experiment==4:
   for i in range(0,nn_T):
       if y_T[i]/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if x_T[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=1.
   #end for

if experiment==5:
   for i in range(0,nn_T):
       if y_T[i]/Ly<eps:                        #bottom
          if x_T[i]<1./3.:
             bc_fixT[i]=True ; bc_valT[i]=0.
          else:
             bc_fixT[i]=True ; bc_valT[i]=1.
       #if y[i]/Ly>(1-eps):
       #   bc_fixT[i]=True ; bc_valT[i]=0.
       if x_T[i]/Lx<eps:                        # left bc
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

if experiment==6:
   for i in range(0,nn_T):
       if x_T[i]/Lx<eps and np.abs(y_T[i]-Ly/2)<=300e3:
          bc_fixT[i]=True ; bc_valT[i]=1.
       if x_T[i]/Lx<eps and np.abs(y_T[i]-Ly/2)>300e3:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y_T[i]/Ly>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

if experiment==7:
   for i in range(0,nn_T):
       if x_T[i]/Lx<eps and np.abs(y_T[i]-Ly/2)<=300e3:
          bc_fixT[i]=True ; bc_valT[i]=1.
       if x_T[i]/Lx<eps and np.abs(y_T[i]-Ly/2)>300e3:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if (x_T[i]-xmin)/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y_T[i]/Ly>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
       if (y_T[i])/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

if experiment==8:
   for i in range(0,nn_T):
       if x_T[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.

if experiment==9:
   for i in range(0,nn_T):
       r2=x_T[i]**2+y_T[i]**2
       if (x_T[i]+1)/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=np.exp(5*(1-r2))*np.sin(16*np.pi*r2)
       if (y_T[i]+1)/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=np.exp(5*(1-r2))*np.sin(16*np.pi*r2)

print("boundary conditions (%.3fs)" % (clock.time()-start))

###############################################################################
# initial temperature
###############################################################################
start=clock.time()

T=np.zeros(nn_T,dtype=np.float64)
Tm1=np.zeros(nn_T,dtype=np.float64) # temperature at timestep n-1
Tm2=np.zeros(nn_T,dtype=np.float64) # temperature at timestep n-2
Tm3=np.zeros(nn_T,dtype=np.float64) # temperature at timestep n-3
Tm4=np.zeros(nn_T,dtype=np.float64) # temperature at timestep n-4
Tm5=np.zeros(nn_T,dtype=np.float64) # temperature at timestep n-5

if experiment==1:
   xc=2./3.
   yc=2./3.
   sigma=0.2
   for i in range(0,nn_T):
       if (x_T[i]-xc)**2+(y_T[i]-yc)**2<=sigma**2:
          T[i]=0.25*(1+np.cos(np.pi*(x_T[i]-xc)/sigma))*(1+np.cos(np.pi*(y_T[i]-yc)/sigma))

if experiment==2:
   for i in range(0,nn_T):
       xi=x_T[i]
       yi=y_T[i]
       if np.sqrt(xi**2+(yi-0.5)**2)<0.3 and (np.abs(xi)>=0.05 or yi>=0.7):
          T[i]=1
       if np.sqrt((x_T[i])**2+(y_T[i]+0.5)**2)<0.3:
          T[i]=1-np.sqrt((x_T[i])**2+(y_T[i]+0.5)**2)/0.3
       if np.sqrt((x_T[i]+0.5)**2+(y_T[i])**2)<0.3:
          T[i]=0.25*(1+np.cos(np.pi*np.sqrt((xi+0.5)**2+yi**2)/0.3))

if experiment==3:
   for i in range(0,nn_T):
       if x_T[i]<0.25:
          T[i]=1

if experiment==6 or experiment==7:
   for i in range(0,nn_T):
       if x_T[i]<=800e3 and np.abs(y_T[i]-Ly/2)<=300e3:
          T[i]=1

if experiment==8:
   for i in range(0,nn_T):
       if x_T[i]<0.1:
          T[i]=np.sin(10*np.pi*x_T[i])

Tm1[:]=T[:]
Tm2[:]=T[:]
Tm3[:]=T[:]
Tm4[:]=T[:]
Tm5[:]=T[:]

if debug: np.savetxt('T_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

print("initial temperature (%.3fs)" % (clock.time()-start))

###############################################################################
# compute timestep
###############################################################################
start=clock.time()

if steady_state:
   dt=0.
   nstep=1
else:
   dt=CFLnb*hx/np.max(np.sqrt(u**2+v**2))/order
   print('dt=',dt)
   nstep=int(tfinal/dt)
   print('nstep=',nstep)

print("compute timestep (%.3fs)" % (clock.time()-start))

###############################################################################
###############################################################################
# time stepping loop
###############################################################################
###############################################################################

model_time=0.
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")

    ###########################################################################
    # all elements are rectangles of size hx,hy

    jcob=hx*hy/4
    jcbi=np.zeros((2,2),dtype=np.float64)
    jcbi[0,0]=2/hx
    jcbi[1,1]=2/hy

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start=clock.time()

    A_fem=lil_matrix((Nfem_T,Nfem_T),dtype=np.float64)
    b_fem=np.zeros(Nfem_T,dtype=np.float64)  
    B_mat=np.zeros((2,m),dtype=np.float64)    
    N_mat=np.zeros((m,1),dtype=np.float64)       
    N_mat_supg=np.zeros((m,1),dtype=np.float64)    
    tau_supg=np.zeros(nel*nq_per_dim**ndim,dtype=np.float64)    

    counterq=0
    for iel in range (0,nel):

        b_el=np.zeros(m,dtype=np.float64)
        A_el=np.zeros((m,m),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        Tvectm1=Tm1[icon_T[0:m,iel]]
        Tvectm2=Tm2[icon_T[0:m,iel]]
        Tvectm3=Tm3[icon_T[0:m,iel]]
        Tvectm4=Tm4[icon_T[0:m,iel]]
        Tvectm5=Tm5[icon_T[0:m,iel]]

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):

                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_T=basis_functions_T(rq,sq,order)
                N_mat[0:m,0]=N_T
                dNdr_T=basis_functions_T_dr(rq,sq,order)
                dNds_T=basis_functions_T_ds(rq,sq,order)
                jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
                jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
                jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
                jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq

                xq=np.dot(N_T,x_T[icon_T[:,iel]])
                yq=np.dot(N_T,y_T[icon_T[:,iel]])
                vel[0,0]=np.dot(N_T,u[icon_T[:,iel]])
                vel[0,1]=np.dot(N_T,v[icon_T[:,iel]])
                dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
                dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
                B_mat[0,:]=dNdx_T
                B_mat[1,:]=dNdy_T

                if supg_type==0:
                   tau_supg[counterq]=0.
                elif supg_type==1:
                      tau_supg[counterq]=(hx*sqrt2) / (2*np.sqrt(vel[0,0]**2+vel[0,1]**2)*order)
                elif supg_type==2:
                      tau_supg[counterq]=(hx*sqrt2)/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)/sqrt15
                else:
                   exit("supg_type: wrong value")
                     
                N_mat_supg=N_mat+tau_supg[counterq]*np.transpose(vel.dot(B_mat))

                # compute mass matrix
                MM=N_mat_supg.dot(N_mat.T)*rho0*hcapa*JxWq

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*JxWq

                # compute advection matrix
                Ka=N_mat_supg.dot(vel.dot(B_mat))*rho0*hcapa*JxWq

                if use_bdf and istep>bdf_order:
                   if bdf_order==1:
                      A_el+=MM+1.*dt*(Ka+Kd)
                      b_el+=MM.dot(Tvectm1)
                   #end if
                   if bdf_order==2:
                      A_el+=MM+2./3.*dt*(Ka+Kd)
                      b_el+=4./3.*MM.dot(Tvectm1)\
                           -1./3.*MM.dot(Tvectm2)
                   #end if
                   if bdf_order==3:
                      A_el+=MM+6./11.*dt*(Ka+Kd)
                      b_el+=18./11.*MM.dot(Tvectm1)\
                           -9./11.*MM.dot(Tvectm2)\
                           +2./11.*MM.dot(Tvectm3)
                   #end if
                   if bdf_order==4:
                      A_el+=MM+12./25.*dt*(Ka+Kd)
                      b_el+=48./25.*MM.dot(Tvectm1)\
                           -36./25.*MM.dot(Tvectm2)\
                           +16./25.*MM.dot(Tvectm3)\
                           -3./25.*MM.dot(Tvectm4)
                   #end if
                   if bdf_order==5:
                      A_el+=MM+60./137.*dt*(Ka+Kd)
                      b_el+=300./137.*MM.dot(Tvectm1)\
                           -300./137.*MM.dot(Tvectm2)\
                           +200./137.*MM.dot(Tvectm3)\
                           -75./137.*MM.dot(Tvectm4)\
                           +12./137.*MM.dot(Tvectm5)
                   #end if
                else:
                   if steady_state:
                      A_el+=Ka
                      b_el+=N_mat[:,0]*JxWq*rhs(xq,yq,experiment)
                   else:
                      A_el+=MM+alphaT*(Ka+Kd)*dt
                      b_el+=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvectm1) +\
                            N_mat[:,0]*JxWq*rhs(xq,yq,experiment)*dt
                #end if

                #print(xq,yq,rhs(xq,yq,experiment))

                counterq+=1
            #end for jq
        #end for iq

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon_T[k1,iel]
            if bc_fixT[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon_T[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_valT[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            #end if
        #end for

        # assemble matrix A_fem and right hand side b_fem
        for k1 in range(0,m):
            m1=icon_T[k1,iel]
            for k2 in range(0,m):
                m2=icon_T[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        #end for

    #end for iel

    print("     -> tau_supg (m,M) %e %e " %(np.min(tau_supg),np.max(tau_supg)))

    if istep==0:
       np.savetxt('tau_supg.ascii',np.array(tau_supg).T,header='# x,y,T')

    print("build FEM matrix: %.3fs" % (clock.time() - start))

    ###########################################################################
    # solve system
    ###########################################################################
    start = clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    stats_T_file.write("%e %e %e \n" %(model_time,np.min(T),np.max(T))) ; stats_T_file.flush()

    print("solve T time: %.3f s" % (clock.time() - start))

    ###########################################################################
    # compute average of temperature using a 4x4 quadrature
    ###########################################################################
    start=clock.time()

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

                N_T=basis_functions_T(rq,sq,order)
                dNdr_T=basis_functions_T_dr(rq,sq,order)
                dNds_T=basis_functions_T_ds(rq,sq,order)
                jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
                jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
                jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
                jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq

                Tq=np.dot(N_T,T[icon_T[:,iel]])
                avrg_T+=Tq*JxWq
                ET+=rho0*hcapa*(abs(Tq))*JxWq
            #end for
        #end for
    #end for
    avrg_T/=Lx*Ly

    ET_file.write("%e %.10e \n" %(model_time,ET))         ; ET_file.flush()
    avrg_T_file.write("%e %.10e \n" %(model_time,avrg_T)) ; avrg_T_file.flush()

    print("     -> avrg T= %.6e" % avrg_T)

    print("compute <T>,M: %.3f s" % (clock.time()-start))

    ###########################################################################
    # visualisation 
    ###########################################################################

    if istep%every==0:
       start=clock.time()

       if debug:
          filename = 'T_{:04d}.ascii'.format(istep) 
          np.savetxt(filename,np.array([x_T,y_T,T]).T,header='# x,y,T')

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_T,nel2))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e %e %e \n" %(x_T[i],y_T[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='bc T' Format='ascii'> \n")
       for i in range(0,nn_T):
           if bc_fixT[i]: 
              vtufile.write("%e \n" % 1)
           else:
              vtufile.write("%e \n" % 0)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau' Format='ascii'> \n")
       for i in range(0,nn_T):
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
              vtufile.write("%d %d %d %d \n" %(icon_T[0,iel],icon_T[1,iel],icon_T[3,iel],icon_T[2,iel]))
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

       #filename = 'solution_{:04d}.pdf'.format(istep) 
       #fig = plt.figure ()
       #ax = fig.gca(projection='3d')
       #ax.plot_surface(x.reshape ((nny,nnx)),y.reshape((nny,nnx)),T.reshape((nny,nnx)),color = 'darkseagreen')
       #ax.set_xlabel ( 'X [ m ] ')
       #ax.set_ylabel ( 'Y [ m ] ')
       #ax.set_zlabel ( ' Temperature  [ C ] ')
       #plt.title('Timestep  %.2d' %(istep),loc='right')
       #plt.grid ()
       #plt.savefig(filename)
       #plt.show ()
       #plt.close()

       print("export to files: %.3f s" % (clock.time()-start))

    #end if

    Tm5=np.copy(Tm4)
    Tm4=np.copy(Tm3)
    Tm3=np.copy(Tm2)
    Tm2=np.copy(Tm1)
    Tm1=np.copy(T)

    model_time+=dt
    print ("model_time=",model_time)
    
#end for istep

if experiment==9 or experiment==7:
   diagonal_file=open('diagonal.ascii',"w")
   for i in range(0,nn_T):
       if np.abs(y_T[i]-Ly+x_T[i])<eps*Lx:
          diagonal_file.write("%4e %6e %7e \n" %(x_T[i],y_T[i],T[i]))

if experiment==6:
   diagonal_file=open('diagonal.ascii',"w")
   for i in range(0,nn_T):
       if np.abs(x_T[i]-Lx/2)<eps*Lx:
          diagonal_file.write("%4e %6e %7e \n" %(x_T[i],y_T[i],T[i]))

if experiment==8 or experiment==3:
   diagonal_file=open('diagonal.ascii',"w")
   for i in range(0,nn_T):
       if np.abs(y_T[i]-Ly/2)<eps*Ly:
          diagonal_file.write("%4e %6e %7e \n" %(x_T[i],y_T[i],T[i]))

###############################################################################
###############################################################################
# end time stepping loop
###############################################################################
###############################################################################

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
