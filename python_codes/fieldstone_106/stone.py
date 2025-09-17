import numpy as np
import sys as sys
import time as clock
import scipy.sparse as sps
from scipy.sparse import lil_matrix
from scipy.interpolate import interp1d

###############################################################################
# viscosity function
###############################################################################

def eta(T,rrr):
    if viscosity_model==1:
       Tused=min(T,Tpatch)
       Tused=max(Tused,Tsurf)
       val=eta0*np.exp(E*(1./(Tused+T0)-1./(1+T0)))    
       val=min(val,eta_max)
    elif viscosity_model==2:
       rad=3481e3+(rrr-Rinner)*(6370e3-3480e3)
       rad=min(6379999,rad)
       rad=max(3481001,rad)
       if rad>6371e3-200e3:
          val=1e23
       else:
          val=f_cubic(rad)
          val=10**val
       val/=1e21
    return val

###############################################################################
# velocity shape functions
###############################################################################
# Q2          Q1
# 6---7---8   2-------3
# |       |   |       |
# 3   4   5   |       |
# |       |   |       |
# 0---1---2   0-------1
###############################################################################

def basis_functions_V(r,s,order):
    if order==1:
       N0=0.25*(1.-r)*(1.-s)
       N1=0.25*(1.+r)*(1.-s)
       N2=0.25*(1.-r)*(1.+s)
       N3=0.25*(1.+r)*(1.+s)
       return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)
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

def basis_functions_V_dr(r,s,order):
    if order==1:
       dNdr0=-0.25*(1.-s) 
       dNdr1=+0.25*(1.-s) 
       dNdr2=-0.25*(1.+s) 
       dNdr3=+0.25*(1.+s) 
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)
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

def basis_functions_V_ds(r,s,order):
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

def basis_functions_P(r,s,order):
    if order==1:
       return 1
    if order==2:
       N0=0.25*(1-r)*(1-s)
       N1=0.25*(1+r)*(1-s)
       N2=0.25*(1-r)*(1+s)
       N3=0.25*(1+r)*(1+s)
       return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

sqrt2=np.sqrt(2)
eps=1e-9
R=8.3145

print("*******************************")
print("********** stone 106 **********")
print("*******************************")

ndim=2   # number of dimensions
ndof_V=2  # number of velocity degrees of freedom per node

order=2

if int(len(sys.argv) == 6):
   nelr  = int(sys.argv[1])
   visu  = int(sys.argv[2])
   nstep = int(sys.argv[3])
   exp   = int(sys.argv[4])
   every = int(sys.argv[5])
else:
   nelr = 48 
   visu = 1
   nstep= 1000
   exp  = 1
   every= 5

axisymmetric=True

Rinner=1.2222 
Router=2.2222 
Ra=1e7
Tsurf=0
Tpatch=1
eta0=1
kappa=1
eta_max=1e3*eta0
T0=0.1

if exp==1: E=0
if exp==2: E=0.25328
if exp==3: E=3

viscosity_model=1 # do not change for now

tfinal=1

CFL_nb=0.5
apply_filter=False
supg_type=0

nelt=int(0.75*nelr)
nel=nelt*nelr
nnt=order*nelt+1  # number of elements, x direction
nnr=order*nelr+1  # number of elements, y direction
nn_V=nnt*nnr

if order==1:
   nn_P=nelt*nelr
   m_V=4
   m_P=1
   r_V=[-1,+1,-1,+1]
   s_V=[-1,-1,+1,+1]
   r_P=[0]
   s_P=[0]

if order==2:
   nn_P=(nelt+1)*(nelr+1)
   m_V=9 
   m_P=4 
   r_V=[-1,0,+1,-1,0,+1,-1,0,+1]
   s_V=[-1,-1,-1,0,0,0,+1,+1,+1]
   r_P=[-1,+1,-1,+1]
   s_P=[-1,-1,+1,+1]

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nn_P        # number of pressure dofs
Nfem_T=nn_V        # number of temperature dofs
Nfem=Nfem_V+Nfem_P # total nb of dofs

ht=np.pi/8/nelt         # element size in tangential direction (radians)
hr=(Router-Rinner)/nelr # element size in radial direction (meters)

sparse=True # storage of FEM matrix 

eta_ref=eta0

use_fs_on_sides=True

debug=False

#################################################################
# Gauss quadrature parameters
#################################################################

nqperdim=order+1

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

#################################################################
# open output files

vrms_file=open('vrms.ascii',"w")
dt_file=open('dt.ascii',"w")
Tavrg_file=open('Tavrg.ascii',"w")
Tstats_file=open('stats_T.ascii',"w")
vrstats_file=open('stats_vr.ascii',"w")
vtstats_file=open('stats_vt.ascii',"w")
Tcorner_file=open('Tcorner.ascii',"w")

#################################################################
#Ra=1e6
#Rinner=3480e3
#Router=6371e3
#DeltaT=3000
#hcond=2.25
#hcapa=1250
#rho0=3250
#g0=9.81
#alpha=3e-5
#eta0=rho0**2*hcapa*g0*alpha*DeltaT*(Router-Rinner)**3/Ra/hcond
#print(eta0)
#print(np.exp(-DeltaT/3273*E))
#print(np.exp(-DeltaT/3273*E)*eta0)
#exit()
#################################################################

print('Ra          =',Ra)
print('Rinner/Router=',Rinner/Router)
print('nelr        =',nelr)
print('nelt        =',nelt)
print('nn_V        =',nn_V)
print('nn_P        =',nn_P)
print('nel         =',nel)
print('Nfem_V      =',Nfem_V)
print('Nfem_P      =',Nfem_P)
print('Nfem        =',Nfem)
print('nqperdim    =',nqperdim)
print('eta0        =',eta0)
print('E           =',E)
print("*******************************")

#################################################################
# reading in steinberger profile 
#################################################################

profile_r=np.zeros(2821,dtype=np.float64)
profile_eta=np.zeros(2821,dtype=np.float64)
###profile_r[1:11511],profile_rho[1:11511]=np.loadtxt('data/rho_prem.ascii',unpack=True,usecols=[0,1])
profile_r,profile_eta=np.loadtxt('../../images/viscosity_profile/steinberger2/visc_sc06.d',unpack=True,usecols=[1,0])
profile_r=(6371-profile_r)*1000
profile_r=np.flip(profile_r)
profile_eta=np.flip(profile_eta)
#print(np.min(profile_r),np.max(profile_r))
#print(np.min(profile_eta),np.max(profile_eta))
f_cubic=interp1d(profile_r, profile_eta, kind='cubic')

#################################################################
# checking that all velocity shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i],order))

#################################################################
# checking that all pressure shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mP):
#   print ('node',i,':',NNP(rPnodes[i],sPnodes[i],order))

#################################################################
# build velocity nodes coordinates 
#################################################################
start = clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)      # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)      # y coordinates
rad_V=np.zeros(nn_V,dtype=np.float64)    # r coordinates
theta_V=np.zeros(nn_V,dtype=np.float64)  # theta coordinates

counter=0    
for j in range(0,nnr):
    for i in range(0,nnt):
        #tV[counter]=5*np.pi/8-i*ht/order
        theta_V[counter]=4*np.pi/8-i*ht/order
        rad_V[counter]=Rinner+j*hr/order
        x_V[counter]=rad_V[counter]*np.cos(theta_V[counter])
        y_V[counter]=rad_V[counter]*np.sin(theta_V[counter])
        counter+=1
    #end for
#end for

if debug:
   np.savetxt('gridV.ascii',np.array([x_V,y_V,theta_V,rad_V]).T,header='# x,y')

print("build V grid: %.3f s" % (clock.time()-start))

#################################################################
# flag elements
#################################################################
start = clock.time()

flag_el_1=np.zeros(nel,dtype=bool)  
flag_el_2=np.zeros(nel,dtype=bool)  
flag_el_3=np.zeros(nel,dtype=bool)  
flag_el_4=np.zeros(nel,dtype=bool)  

counter=0
for j in range(0,nelr):
    for i in range(0,nelt):
        if i==0: flag_el_1[counter]=True
        if i==nelt-1: flag_el_2[counter]=True
        if j==0: flag_el_3[counter]=True
        if j==nelr-1: flag_el_4[counter]=True
        counter+=1
    #end for
#end for

print("flagging boundary elements: %.3f s" % (clock.time()-start))

#################################################################
# connectivity
#################################################################
start = clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nelr):
    for i in range(0,nelt):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                icon_V[counter2,counter]=i*order+l+j*order*nnt+nnt*k
                counter2+=1
            #end for
        #end for
        counter+=1
    #end for
#end for

print("build icon_V: %.3f s" % (clock.time() - start))

#################################################################
# build pressure grid 
#################################################################
start = clock.time()

x_P=np.zeros(nn_P,dtype=np.float64) # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64) # y coordinates
theta_P=np.zeros(nn_P,dtype=np.float64) # theta coordinates
rad_P=np.zeros(nn_P,dtype=np.float64) # rad coordinates

if order==1:
   for iel in range(0,nel):
       x_P[iel]=sum(x_V[icon_V[0:m_V,iel]])*0.25
       y_P[iel]=sum(y_V[icon_V[0:m_V,iel]])*0.25
    #end for
#end if 
      
if order>1:
   counter=0    
   for j in range(0,(order-1)*nelr+1):
       for i in range(0,(order-1)*nelt+1):
           theta_P[counter]=5*np.pi/8-i*ht/(order-1)
           rad_P[counter]=Rinner+j*hr/(order-1)
           x_P[counter]=rad_P[counter]*np.cos(theta_P[counter])
           y_P[counter]=rad_P[counter]*np.sin(theta_P[counter])
           counter+=1
       #end for
    #end for
#end if

if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time()-start))

#################################################################
# build pressure connectivity array 
#################################################################
start = clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)

if order==1:
   counter=0
   for j in range(0,nelr):
       for i in range(0,nelt):
           icon_P[0,counter]=counter
           counter += 1
       #end for
   #end for

if order>1:
   om1=order-1
   counter=0
   for j in range(0,nelr):
       for i in range(0,nelt):
           counter2=0
           for k in range(0,order):
               for l in range(0,order):
                   icon_P[counter2,counter]=i*om1+l+j*om1*(om1*nelt+1)+(om1*nelt+1)*k 
                   counter2+=1
               #end for
           #end for
           counter += 1
       #end for
   #end for

print("build icon_P: %.3f s" % (clock.time()-start))

#################################################################
# define boundary conditions
#################################################################
start = clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

flag_1=np.zeros(nn_V,dtype=bool)  
flag_2=np.zeros(nn_V,dtype=bool)  
flag_3=np.zeros(nn_V,dtype=bool)  
flag_4=np.zeros(nn_V,dtype=bool)  

for i in range(0,nn_V):
    if abs(theta_V[i]-(4*np.pi/8))<eps: flag_1[i]=True
    if abs(theta_V[i]-(3*np.pi/8))<eps: flag_2[i]=True
    if abs(rad_V[i]-Rinner)<eps: flag_3[i]=True
    if abs(rad_V[i]-Router)<eps: flag_4[i]=True
    #if flag_1[i]:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
    #if flag_2[i]:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
    #if flag_3[i]:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
    #if flag_4[i]:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
    if not use_fs_on_sides: # must remove nullspace
       if abs(x_V[i])<0.001 and flag_3[i]:
          bc_fix_V[i*ndof_V  ] = True ; bc_val[i*ndof_V  ]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val[i*ndof_V+1]   = 0.

print("velocity b.c.: %.3f s" % (clock.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = clock.time()

bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

for i in range(0,nn_V):
    if flag_4[i]:
       bc_fix_T[i]=True ; bc_val_T[i]=Tsurf
    if flag_3[i] and abs(theta_V[i]-np.pi/2)<np.pi/16:
       bc_fix_T[i]=True ; bc_val_T[i]=Tpatch
#end for

print("temperature b.c.: %.3f s" % (clock.time() - start))

#####################################################################
# initial temperature
#####################################################################

T=np.zeros(nn_V,dtype=np.float64)
T_mem=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V):
    T[i]=0.25 #*DeltaT+Tsurf
#end for

T_mem[:]=T[:]

if debug: np.savetxt('temperature_init.ascii',np.array([x_V,y_V,T]).T,header='# x,y,T')

#################################################################
# compute area of elements
#################################################################
start = clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
       #end for
   #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

area=np.pi/8*(Router**2-Rinner**2)
print("     -> total area %.6f " %( area  ))

print("compute elements areas: %.3f s" % (clock.time() - start))


#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================
u_mem=np.zeros(nn_V,dtype=np.float64) # not used yet
v_mem=np.zeros(nn_V,dtype=np.float64)
Tvect=np.zeros(m_V,dtype=np.float64)   

time=0

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start = clock.time()

    if axisymmetric:
       C=np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 
       B=np.zeros((4,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
       N_mat=np.zeros((4,m_P),dtype=np.float64) 
    else:
       C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
       B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
       N_mat=np.zeros((3,m_P),dtype=np.float64)

    if sparse:
       A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    else:   
       K_mat = np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
       G_mat = np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT

    f_rhs   = np.zeros(Nfem_V,dtype=np.float64)        # right hand side f 
    h_rhs   = np.zeros(Nfem_P,dtype=np.float64)        # right hand side h 

    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
        K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_V=basis_functions_V(rq,sq,order)
                N_P=basis_functions_P(rq,sq,order)
                dNdr_V=basis_functions_V_dr(rq,sq,order)
                dNds_V=basis_functions_V_ds(rq,sq,order)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq

                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])

                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                radq=np.sqrt(xq**2+yq**2)
                etaq=eta(Tq,radq)

                #Cartesian components of -\vec{e}_r
                erxq=-xq/np.sqrt(xq**2+yq**2)  
                eryq=-yq/np.sqrt(xq**2+yq**2) 

                if axisymmetric:

                   for i in range(0,m_V):
                       B[0:4, 2*i:2*i+2] = [[dNdx_V[i],0.       ],
                                            [N_V[i]/xq,0.       ],
                                            [0.       ,dNdy_V[i]],
                                            [dNdy_V[i],dNdx_V[i]]]

                   K_el+=B.T.dot(C.dot(B))*etaq*JxWq * 2*np.pi*xq

                   for i in range(0,m_V):
                       f_el[ndof_V*i  ]-=N_V[i]*JxWq*Tq*erxq*Ra *2*np.pi*xq
                       f_el[ndof_V*i+1]-=N_V[i]*JxWq*Tq*eryq*Ra *2*np.pi*xq

                   for i in range(0,m_P):
                       N_mat[0,i]=N_P[i]
                       N_mat[1,i]=N_P[i]
                       N_mat[2,i]=N_P[i]
                       N_mat[3,i]=0.

                   G_el-=B.T.dot(N_mat)*JxWq *2*np.pi*xq

                else:

                   for i in range(0,m_V):
                       B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                         [0.       ,dNdy_V[i]],
                                         [dNdy_V[i],dNdx_V[i]]]

                   K_el+=B.T.dot(C.dot(B))*etaq*JxWq

                   for i in range(0,m_V):
                       f_el[ndof_V*i  ]-=N_V[i]*jcob*weightq*Tq*erxq*Ra
                       f_el[ndof_V*i+1]-=N_V[i]*jcob*weightq*Tq*eryq*Ra

                   for i in range(0,m_P):
                       N_mat[0,i]=N_P[i]
                       N_mat[1,i]=N_P[i]
                       N_mat[2,i]=0.

                   G_el-=B.T.dot(N_mat)*JxWq

                #end if

            # end for jq
        # end for iq

        #impose dirichlet b.c. 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                if bc_fix_V[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m_V*ndof_V):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   #end for
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val_V[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                   G_el[ikk,:]=0
                #end if
            #end for
        #end for

        #free slip at bottom and top
        if flag_el_3[iel] or flag_el_4[iel]:
           for k in range(0,m_V):
               inode=icon_V[k,iel]
               if flag_3[inode] or flag_4[inode]:
                  RotMat=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
                  for i in range(0,m_V*ndof_V):
                      RotMat[i,i]=1.
                  angle=theta_V[inode]
                  RotMat[2*k  ,2*k]= np.cos(angle) ; RotMat[2*k  ,2*k+1]=np.sin(angle)
                  RotMat[2*k+1,2*k]=-np.sin(angle) ; RotMat[2*k+1,2*k+1]=np.cos(angle)
                  # apply counter rotation 
                  K_el=RotMat.dot(K_el.dot(RotMat.T))
                  f_el=RotMat.dot(f_el)
                  G_el=RotMat.dot(G_el)
                  # apply boundary conditions
                  # x-component set to 0
                  ikk=ndof_V*k     
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,m_V*ndof_V):
                      K_el[ikk,jkk]=0
                      K_el[jkk,ikk]=0
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=0#K_ref*bc_val[m1]
                  #h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0
                  # rotate back 
                  K_el=RotMat.T.dot(K_el.dot(RotMat))
                  f_el=RotMat.T.dot(f_el)
                  G_el=RotMat.T.dot(G_el)
               #end if
           #end for
        #end if

        if (flag_el_1[iel] or flag_el_2[iel]) and use_fs_on_sides:
           for k in range(0,m_V):
               inode=icon_V[k,iel]
               if flag_1[inode] or flag_2[inode]:
                  RotMat=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
                  for i in range(0,m_V*ndof_V):
                      RotMat[i,i]=1.
                  angle=theta_V[inode]
                  RotMat[2*k  ,2*k]= np.cos(angle) ; RotMat[2*k  ,2*k+1]=np.sin(angle)
                  RotMat[2*k+1,2*k]=-np.sin(angle) ; RotMat[2*k+1,2*k+1]=np.cos(angle)
                  # apply counter rotation 
                  K_el=RotMat.dot(K_el.dot(RotMat.T))
                  f_el=RotMat.dot(f_el)
                  G_el=RotMat.dot(G_el)
                  # apply boundary conditions
                  # x-component set to 0
                  ikk=ndof_V*k   +1  
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,m_V*ndof_V):
                      K_el[ikk,jkk]=0
                      K_el[jkk,ikk]=0
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=0#K_ref*bc_val[m1]
                  #h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0
                  # rotate back 
                  K_el=RotMat.T.dot(K_el.dot(RotMat))
                  f_el=RotMat.T.dot(f_el)
                  G_el=RotMat.T.dot(G_el)
               #end if
           #end for
        #end if

        G_el*=eta_ref/Rinner
        h_el*=eta_ref/Rinner

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        if sparse:
                           A_sparse[m1,m2] += K_el[ikk,jkk]
                        else:
                           K_mat[m1,m2]+=K_el[ikk,jkk]
                        #end if
                    #end for
                #end for
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    if sparse:
                       A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                       A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                    else:
                       G_mat[m1,m2]+=G_el[ikk,jkk]
                    #end if
                f_rhs[m1]+=f_el[ikk]
            #end for
        #end for
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            h_rhs[m2]+=h_el[k2]
        #end for

    #end for iel

    if not sparse:
       print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
       print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))

    print("build FE matrix: %.3fs" % (clock.time()-start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = clock.time()

    rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    rhs[0:Nfem_V]=f_rhs
    rhs[Nfem_V:Nfem]=h_rhs

    if not sparse:
       a_mat = np.zeros((Nfem,Nfem),dtype=np.float64) 
       a_mat[0:Nfem_V,0:Nfem_V]=K_mat
       a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
       a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T

    print("assemble blocks: %.3f s" % (clock.time() - start))

    ######################################################################
    # assign extra pressure b.c. to remove null space
    ######################################################################

    if sparse:
       A_sparse[Nfem-1,:]=0
       A_sparse[:,Nfem-1]=0
       A_sparse[Nfem-1,Nfem-1]=1
       rhs[Nfem-1]=0
    else:
       a_mat[Nfem-1,:]=0
       a_mat[:,Nfem-1]=0
       a_mat[Nfem-1,Nfem-1]=1
       rhs[Nfem-1]=0
    #end if

    ######################################################################
    # solve system
    ######################################################################
    start = clock.time()

    if sparse:
       sparse_matrix=A_sparse.tocsr()
    else:
       sparse_matrix=sps.csr_matrix(a_mat)

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (clock.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*eta_ref/Rinner

    print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

    if debug:
       np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
       np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (clock.time()-start))

    #################################################################
    # Compute velocity in polar coordinates
    #################################################################
    start = clock.time()

    vr=np.zeros(nn_V,dtype=np.float64)   
    vt=np.zeros(nn_V,dtype=np.float64)  

    vr= u*np.cos(theta_V)+v*np.sin(theta_V)
    vt=-u*np.sin(theta_V)+v*np.cos(theta_V)

    vrstats_file.write("%6e %6e %6e\n" % (time,np.min(vr),np.max(vr)))
    vtstats_file.write("%6e %6e %6e\n" % (time,np.min(vt),np.max(vt)))
    vrstats_file.flush()
    vtstats_file.flush()

    print("     -> vr (m,M) %.4e %.4e " %(np.min(vr),np.max(vr)))
    print("     -> vt (m,M) %.4e %.4e " %(np.min(vt),np.max(vt)))

    print("compute vr,vtheta: %.3f s" % (clock.time()-start))

    #################################################################
    # compute timestep value
    #################################################################
    start = clock.time()

    dt1=CFL_nb*hr/np.max(np.sqrt(u**2+v**2))
    dt2=CFL_nb*hr**2 / kappa #(hcond/hcapa/rho0)

    dt_candidate=np.min([dt1,dt2])

    if istep==0:
       dt=1e-6
    else:
       dt=min(dt_candidate,2*dt)

    print('     -> dt1,dt2 = %e %e ' %(dt1,dt2))
    print('     -> dt  = %.8f ' %(dt))

    time+=dt

    print('     -> time= %.6f; tfinal= %.6f' %(time,tfinal))

    dt_file.write("%10e %10e %10e %10e\n" % (time,dt1,dt2,dt))
    dt_file.flush()

    #################################################################
    # build temperature matrix
    #################################################################
    start = clock.time()

    A_fem=np.zeros((Nfem_T,Nfem_T),dtype=np.float64)
    b_fem=np.zeros(Nfem_T,dtype=np.float64)         
    B=np.zeros((2,m_V),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m_V,1),dtype=np.float64)         # shape functions
    N_mat_supg = np.zeros((m_V,1),dtype=np.float64)         # shape functions
    tau_supg = np.zeros(nel*nqperdim**ndim,dtype=np.float64)

    counterq=0   
    for iel in range(0,nel):

        b_el=np.zeros(m_V,dtype=np.float64)
        a_el=np.zeros((m_V,m_V),dtype=np.float64)
        Ka=np.zeros((m_V,m_V),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m_V,m_V),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m_V,m_V),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m_V):
            Tvect[k]=T[icon_V[k,iel]]
        #end for

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_mat[:,0]=basis_functions_V(rq,sq,order)
                dNdr_V=basis_functions_V_dr(rq,sq,order)
                dNds_V=basis_functions_V_ds(rq,sq,order)

                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq

                uq=np.dot(N_V,u[icon_V[:,iel]]) ; vel[0,0]=uq
                vq=np.dot(N_V,v[icon_V[:,iel]]) ; vel[0,1]=vq

                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                B[0,:]=dNdx_V[:]
                B[1,:]=dNdy_V[:]

                if supg_type==0:
                   tau_supg[counterq]=0.
                elif supg_type==1:
                   tau_supg[counterq]=(hx*sqrt2)/2/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)
                elif supg_type==2:
                   tau_supg[counterq]=(hx*sqrt2)/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)/sqrt15
                else:
                   exit("supg_type: wrong value")
    
                N_mat_supg=N_mat+tau_supg[counterq]*np.transpose(vel.dot(B))

                # compute mass matrix
                MM+=N_mat_supg.dot(N_mat.T)*JxWq #*hcapa*rho0

                # compute diffusion matrix
                Kd+=B.T.dot(B)*JxWq*kappa #hcond

                # compute advection matrix
                Ka+=N_mat_supg.dot(vel.dot(B))*JxWq #*hcapa*rho0

                counterq+=1

            #end for
        #end for

        A_el=MM+0.5*(Ka+Kd)*dt
        b_el=(MM-0.5*(Ka+Kd)*dt).dot(Tvect)

        # apply boundary conditions
        for k1 in range(0,m_V):
            m1=icon_V[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_V):
                   m2=icon_V[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            #end for
        #end for

        # assemble matrix and right hand side
        for k1 in range(0,m_V):
            m1=icon_V[k1,iel]
            for k2 in range(0,m_V):
                m2=icon_V[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        #end for

    #end for iel

    print("     -> tau_supg (m,M) %e %e " %(np.min(tau_supg),np.max(tau_supg)))

    print("build FE matrix : %.3f s" % (clock.time() - start))

    #################################################################
    # solve system
    #################################################################
    start=clock.time()

    Traw=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     T (m,M) %.4f %.4f " %(np.min(Traw),np.max(Traw)))

    Tstats_file.write("%6e %6e %6e\n" % (time,np.min(Traw),np.max(Traw))) ; Tstats_file.flush()

    print("solve T: %.3f s" % (clock.time()-start))

    #################################################################
    # measure T in corner 
    #################################################################
    start=clock.time()

    for i in range(0,nn_V):
        if flag_2[i] and flag_3[i]:
           T_23=Traw[i]

    Tcorner_file.write("%10e %10e \n" % (time,T_23)) ; Tcorner_file.flush()

    print("measure T in corner: %.3f s" % (clock.time()-start))

    #################################################################
    # apply Lenardic & Kaula filter
    #################################################################
    start=clock.time()

    if apply_filter:

       # step 1: compute the initial sum 'sum0' of all values of T

       sum0=np.sum(Traw)

       # step 2: find the minimum value Tmin of T
  
       minT=np.min(Traw)
  
       # step 3: find the maximum value Tmax of T
  
       maxT=np.max(Traw)

       # step 4: set T=0 if T<=|Tmin|

       for i in range(0,nn_V):
           if Traw[i]<=abs(minT):
              Traw[i]=0

       # step 5: set T=1 if T>=2-Tmax

       for i in range(0,nn_V):
           if Traw[i]>=2-maxT:
              Traw[i]=1

       # step 6: compute the sum sum1 of all values of T

       sum1=np.sum(Traw)

       # step 7: compute the number num of 0<T<1

       num=0
       for i in range(0,nn_V):
           if Traw[i]>0 and Traw[i]<1:
              num+=1

       # step 8: add (sum1-sum0)/num to all 0<T<1
       
       for i in range(0,nn_V):
           if Traw[i]>0 and Traw[i]<1:
              Traw[i]+=(sum1-sum0)/num 

       print("     T (m,M) %.4f %.4f " %(np.min(Traw),np.max(Traw)))

    #end if
       
    T[:]=Traw[:]

    print("apply L&K filter: %.3f s" % (clock.time() - start))

    #################################################################
    # compute vrms and average temperature
    #################################################################
    start=clock.time()

    vrms=0.
    Tavrg=0.
    for iel in range (0,nel):
        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                N_V=basis_functions_V(rq,sq,order)
                dNdr_V=basis_functions_V_dr(rq,sq,order)
                dNds_V=basis_functions_V_ds(rq,sq,order)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                JxWq=np.linalg.det(jcb)*weightq
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                vrms+=(uq**2+vq**2)*JxWq
                Tavrg+=Tq*JxWq
            #end for jq
        #end for iq
    #end for iel

    vrms=np.sqrt(vrms/area)
    Tavrg/=area

    Tavrg_file.write("%10e %10e\n" % (time,Tavrg)) ; Tavrg_file.flush()
    vrms_file.write("%10e %.10e\n" % (time,vrms))  ; vrms_file.flush()

    print("     istep= %.6d ; vrms  = %.6f" %(istep,vrms))

    print("compute vrms: %.3f s" % (clock.time()-start))

    #####################################################################
    # compute nodal strainrate and pressure 
    #####################################################################
    start = clock.time()

    q=np.zeros(nn_V,dtype=np.float64)
    c=np.zeros(nn_V,dtype=np.float64)
    count=np.zeros(nn_V,dtype=np.int32)  
    exx_n=np.zeros(nn_V,dtype=np.float64)  
    eyy_n=np.zeros(nn_V,dtype=np.float64)  
    ett_n=np.zeros(nn_V,dtype=np.float64)  
    exy_n=np.zeros(nn_V,dtype=np.float64)  
    eta_n=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,m_V):
            rq=r_V[i]
            sq=s_V[i]

            N_V=basis_functions_V(rq,sq,order)
            N_P=basis_functions_P(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
            eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
            exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
            ettq=np.dot(N_V,u[icon_V[:,iel]])               # really?
            pq=np.dot(N_P,p[icon_P[:,iel]])

            inode=icon_V[i,iel]
            q[inode]+=pq
            exx_n[inode]+=exxq
            eyy_n[inode]+=eyyq
            exy_n[inode]+=exyq
            ett_n[inode]+=ettq
            count[inode]+=1
        #end for
    #end for
    
    exx_n/=count
    eyy_n/=count
    exy_n/=count
    ett_n/=count
    q/=count

    print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
    print("     -> ett_n (m,M) %.6e %.6e " %(np.min(ett_n),np.max(ett_n)))

    if debug: np.savetxt('strainrate.ascii',np.array([x_V,y_V,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute press & sr: %.3f s" % (clock.time()-start))

    #####################################################################
    # compute nodal viscosity
    #####################################################################
    start=clock.time()

    for i in range(0,nn_V):
        eta_n[i]=eta(T[i],rad_V[i])

    print("     -> eta_n (m,M) %.6e %.6e " %(np.min(eta_n),np.max(eta_n)))

    print("compute nodal viscosity: %.3f s" % (clock.time()-start))

    #####################################################################
    # normalise pressure at surface
    #####################################################################
    start=clock.time()

    qavrg=0
    for i in range(0,nn_V):
        if flag_4[i]:
           qavrg+=q[i]
    q[:]-=qavrg/nnt

    if debug: np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

    print("normalise pressure: %.3f s" % (clock.time()-start))

    #####################################################################
    # compute (dev) stress tensor 
    #####################################################################
    start = clock.time()

    tauxx_n=2*eta_n*exx_n
    tauyy_n=2*eta_n*eyy_n
    tauxy_n=2*eta_n*exy_n
    tautt_n=2*eta_n*ett_n

    sigmaxx_n=-q+tauxx_n[:]
    sigmayy_n=-q+tauyy_n[:]
    sigmatt_n=-q+tautt_n[:]
    sigmaxy_n=   tauxy_n[:]

    print("compute stress: %.3f s" % (clock.time()-start))

    #####################################################################
    # compute traction at the surface
    #####################################################################
    start=clock.time()

    if istep%every==0:
       nx=np.zeros(nn_V,dtype=np.float64)  
       ny=np.zeros(nn_V,dtype=np.float64)  
       for i in range(0,nn_V):
           if flag_4[i]:
              nx[i]=np.cos(theta_V[i])
              ny[i]=np.sin(theta_V[i])

       tx_n=np.zeros(nn_V,dtype=np.float64)  
       ty_n=np.zeros(nn_V,dtype=np.float64)  
       tx_n=sigmaxx_n*nx+sigmaxy_n*ny
       tx_n=sigmaxy_n*nx+sigmayy_n*ny

       tractfile=open('surface_traction.ascii',"w")
       t_n=np.zeros(nnt,dtype=np.float64)  
       counter=0
       for i in range(0,nn_V):
           if flag_4[i]:
              t_n[counter]=tx_n[i]*nx[i]+ty_n[i]*ny[i]
              tractfile.write("%10e %10e \n" %(np.pi/2-theta_V[i],t_n[counter]))
              counter+=1
       tractfile.close()

       #in order to compute topo I need dimensioned values!
       sigma_ref=829.55
       tnavrg=np.sum(t_n)/nnt
       topofile=open('surface_topography.ascii',"w")
       counter=0
       for i in range(0,nn_V):
           if flag_4[i]:
              topofile.write("%10e %10e \n" %(np.pi/2-theta_V[i],-(t_n[counter]-tnavrg)*sigma_ref/9.81/3250))
              counter+=1
       topofile.close()

    print("compute tractions and dyn topo: %.3f s" % (clock.time()-start))

    #####################################################################
    # compute temperature & visc profile
    #####################################################################
    start = clock.time()

    if istep%every==0:
       vprofile=open('profile.ascii',"w")
       for i in range(0,nn_V):
           if abs(x_V[i])<0.001:
              vprofile.write("%10e %10e %10e\n" % (y_V[i],T[i],eta_n[i]))
       vprofile.close()

       bot_file=open('bottom_temp.ascii',"w")
       for i in range(0,nn_V):
           if flag_3[i]:
              bot_file.write("%10e %10e %10e\n" % (theta_V[i],T[i],rad_V[i]))
       bot_file.close()

    print("compute profiles: %.3f s" % (clock.time() - start))

    #####################################################################
    # plot of solution
    #####################################################################
    start = clock.time()

    if istep%every==0:
       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Normal vector' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(nx[i],ny[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Traction' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(tx_n[i],ty_n[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='vr' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %(vr[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='vt' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %(vt[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pressure' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Temperature' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %eta_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_1' Format='ascii'> \n")
       for i in range(0,nn_V):
           if flag_1[i]:
              vtufile.write("%e \n" %1.)
           else:
              vtufile.write("%e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_2' Format='ascii'> \n")
       for i in range(0,nn_V):
           if flag_2[i]:
              vtufile.write("%e \n" %1.)
           else:
              vtufile.write("%e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_3' Format='ascii'> \n")
       for i in range(0,nn_V):
           if flag_3[i]:
              vtufile.write("%e \n" %1.)
           else:
              vtufile.write("%e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_4' Format='ascii'> \n")
       for i in range(0,nn_V):
           if flag_4[i]:
              vtufile.write("%e \n" %1.)
           else:
              vtufile.write("%e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='ett' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %ett_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_1' Format='ascii'> \n")
       for iel in range(0,nel):
           if flag_el_1[iel]:
              vtufile.write("%e \n" %1.)
           else:
              vtufile.write("%e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_2' Format='ascii'> \n")
       for iel in range(0,nel):
           if flag_el_2[iel]:
              vtufile.write("%e \n" %1.)
           else:
              vtufile.write("%e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_3' Format='ascii'> \n")
       for iel in range(0,nel):
           if flag_el_3[iel]:
              vtufile.write("%e \n" %1.)
           else:
              vtufile.write("%e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_4' Format='ascii'> \n")
       for iel in range(0,nel):
           if flag_el_4[iel]:
              vtufile.write("%e \n" %1.)
           else:
              vtufile.write("%e \n" %0.)
       vtufile.write("</DataArray>\n")
       #-
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[2,iel],icon_V[8,iel],\
                                                          icon_V[6,iel],icon_V[1,iel],icon_V[5,iel],\
                                                          icon_V[7,iel],icon_V[3,iel],icon_V[4,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m_V))
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

       print("export to vtu file: %.3f s" % (clock.time() - start))

    ###################################

    T_mem[:]=T[:]
    u_mem[:]=u[:]
    v_mem[:]=v[:]

    if time>tfinal:
       print("*****tfinal reached*****")
       break

#end for istep

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
