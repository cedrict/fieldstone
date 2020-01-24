import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time

#------------------------------------------------------------------------------

def density(rho0,alphaT,T,T0):
    val=rho0*(1.-alphaT*(T-T0)) #-rho0
    return val

#------------------------------------------------------------------------------

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

#------------------------------------------------------------------------------

def viscosity(T,exx,eyy,exy,gamma_T,gamma_y,sigma_y,eta_star,case,r,R2):
    y=R2-r # hijacking vertical coordinate y 
    #-------------------
    # blankenbach et al, case 1
    #-------------------
    if case==0: 
       val=1.
    #-------------------
    # tosi et al, case 1
    #-------------------
    elif case==1:
       val=np.exp(-gamma_T*T)
    #-------------------
    # tosi et al, case 2
    #-------------------
    elif case==2:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       e=max(e,1e-12)
       eta_lin=np.exp(-gamma_T*T)
       eta_plast=eta_star + sigma_y/(np.sqrt(2.)*e)
       val=2./(1./eta_lin + 1./eta_plast)
    #-------------------
    # tosi et al, case 3
    #-------------------
    elif case==3:
       val=np.exp(-gamma_T*T+gamma_y*(1-y))
    #-------------------
    # tosi et al, case 4
    #-------------------
    elif case==4:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    #-------------------
    # tosi et al, case 5
    #-------------------
    elif case==5:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    val=min(2.0,val)
    val=max(1.e-5,val)
    return val

#------------------------------------------------------------------------------

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2  # number of physical dimensions
m=4     # number of nodes making up an element
ndofV=2 # number of V degrees of freedom per node
ndofT=1 # number of T degrees of freedom per node

gamma_T=np.log(1e5) # rheology parameter 
eta_star=1e-3       # rheology parameter 
alphaT=1e-4         # thermal expansion coefficient
hcond=1.            # thermal conductivity
hcapa=1.            # heat capacity
rho0=1.             # reference density
T0=0                # reference temperature

CFL_nb=0.5   # CFL number 
every=1      # vtu output frequency
every_ps=100      # vtu output frequency
nstep=1000  # maximum number of timestep   
tol_nl=1.e-3  # nonlinear convergence coeff.

#--------------------------------------

case=1

if case==0:
   Ra=1e4  
   sigma_y=0.
   gamma_y=np.log(1.)  # rheology parameter 
   niter_nl=1
   nonlinear=False

if case==1:
   Ra=1e2 
   sigma_y=1.
   gamma_y=np.log(1.)  # rheology parameter 
   niter_nl=1
   nonlinear=False

if case==2:
   Ra=1e2 
   sigma_y = 1
   gamma_y=np.log(1.)  # rheology parameter 
   niter_nl=8
   nonlinear=True

if case==3:
   Ra=1e2 
   sigma_y=0.
   gamma_y=np.log(10.)  # rheology parameter 
   nonlinear=False

if case==4:
   Ra=1e2 
   sigma_y = 1
   gamma_y=np.log(10.)  # rheology parameter 
   niter_nl=10
   nonlinear=True

if case==5:
   Ra=1e2 
   sigma_y=4
   gamma_y=np.log(10.)  # rheology parameter 
   niter_nl=10
   nonlinear=True

g0=Ra/alphaT

#--------------------------------------

if int(len(sys.argv) == 4):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
   N0   = int(sys.argv[3])
else:
   nelr = 20
   visu = 1
   N0=7

R1=1.22
R2=2.22
dr=(R2-R1)/nelr
area=np.pi*(R2**2-R1**2)
nelt=int(2.*math.pi*R2/dr)
nel=nelr*nelt  # number of elements, total
nnr=nelr+1
nnt=nelt
nnp=nnr*nnt  # number of nodes
NfemV=nnp*ndofV # Total number of V degrees of freedom
NfemT=nnp*ndofT # Total number of T degrees of freedom

f=R1/R2

penalty=1.e8 

Temp_surf=0
Temp_cmb=1

use_BA=True
use_EBA=False

alpha=0.5

eps=1.e-10
sqrt3=np.sqrt(3.)

use_freeslip_outer=True
use_freeslip_inner=True

pin_one_node = use_freeslip_inner and use_freeslip_outer # needed bc of nullspace
pin_one_node = True 
        
remove_net_rotation=False

compute_eigenvalues=False

end_time=5e-2

if use_BA:
   use_shearheating=False
   use_adiabatic_heating=False

if use_EBA:
   use_shearheating=True
   use_adiabatic_heating=True

find_ss=False
if find_ss:
   relax=0.1
   alpha=1

#################################################################
#################################################################

print("nelr",nelr)
print("nelt",nelt)
print("nel",nel)
print("nnr=",nnr)
print("nnt=",nnt)
print("nnp=",nnp)
print("NfemV=",NfemV)
print("NfemT=",NfemT)
print("------------------------------")

#################################################################


convfile=open("conv_nl.ascii","w")
niterfile=open("niter_nl.ascii","w")
psT_file=open('power_spectrum_T.ascii',"w")
psV_file=open('power_spectrum_V.ascii',"w")
psvr_file=open('power_spectrum_vr.ascii',"w")
psvt_file=open('power_spectrum_vt.ascii',"w")
omega_z_file=open('omega_z.ascii',"w")
avrg_vt_file=open('avrg_vt.ascii',"w")
avrg_vr_file=open('avrg_vr.ascii',"w")
avrg_T_file=open('avrg_T.ascii',"w")
avrg_sr_file=open('avrg_sr.ascii',"w")
avrg_eta_file=open('avrg_eta.ascii',"w")
vrms_file=open('vrms.ascii',"w")
vrrms_file=open('vrrms.ascii',"w")
vtrms_file=open('vtrms.ascii',"w")
EK_file=open('EK.ascii',"w")
ET_file=open('ET.ascii',"w")
EG_file=open('EG.ascii',"w")
dETdt_file=open('dETdt.ascii',"w")
Nu1_file=open('Nu_boundary1.ascii',"w")
Nu2_file=open('Nu_boundary2.ascii',"w")
stats_vr_file=open('stats_vr.ascii',"w")
stats_vt_file=open('stats_vt.ascii',"w")
stats_u_file=open('stats_u.ascii',"w")
stats_v_file=open('stats_v.ascii',"w")
stats_T_file=open('stats_T.ascii',"w")
dt_file=open('dt.ascii',"w")
visc_diss_file=open('viscous_dissipation.ascii',"w")
work_grav_file=open('work_against_gravity.ascii',"w")
adiab_heat_file=open('adiabatic_heating.ascii',"w")
hf1_file=open('heatflux_boundary1.ascii',"w")
hf2_file=open('heatflux_boundary2.ascii',"w")
cons_file=open('conservation.ascii',"w")

#################################################################
# grid point setup
#################################################################

x=np.empty(nnp,dtype=np.float64)     # x coordinates
y=np.empty(nnp,dtype=np.float64)     # y coordinates
r=np.empty(nnp,dtype=np.float64)     # cylindrical coordinate r 
theta=np.empty(nnp,dtype=np.float64) # cylindrical coordinate theta 
node_inner = np.zeros(nnp, dtype=np.bool)  
node_outer = np.zeros(nnp, dtype=np.bool)  

Louter=2.*math.pi*R2
Lr=R2-R1
sx = Louter/float(nelt)
sy = Lr    /float(nelr)

counter=0
for j in range(0,nnr):
    for i in range(0,nelt):
        x[counter]=i*sx
        y[counter]=j*sy
        if j==0:
           node_inner[counter]=True
        if j==nnr-1:
           node_outer[counter]=True
        counter += 1

counter=0
for j in range(0,nnr):
    for i in range(0,nnt):
        xi=x[counter]
        yi=y[counter]
        t=xi/Louter*2.*math.pi    
        x[counter]=math.cos(t)*(R1+yi)
        y[counter]=math.sin(t)*(R1+yi)
        r[counter]=R1+yi
        theta[counter]=math.atan2(y[counter],x[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*math.pi
        counter+=1

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)
elt_inner = np.zeros(nel, dtype=np.bool)  
elt_outer = np.zeros(nel, dtype=np.bool)  

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        icon[0, counter] = icon2 
        icon[1, counter] = icon1
        icon[2, counter] = icon4
        icon[3, counter] = icon3
        if j==0:
           elt_inner[counter]=True
        if j==nelr-1:
           elt_outer[counter]=True
        counter += 1
    #end for
#end for

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#    print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#    print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#    print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])
    

print("connectivity (%.3fs)" % (time.time() - start))

#################################################################
# define velocity boundary conditions
#################################################################
start = time.time()

bc_fixV = np.zeros(NfemV, dtype=np.bool)  
bc_valV = np.zeros(NfemV, dtype=np.float64) 

for i in range(0,nnp):
    # inner boundary
    if r[i]<R1+eps and not use_freeslip_inner:
          bc_fixV[i*ndofV]   = True ; bc_valV[i*ndofV]   = y[i] 
          bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = -x[i]
    #end if
    # outer boundary
    if r[i]>(R2-eps) and not use_freeslip_outer:
          bc_fixV[i*ndofV]   = True ; bc_valV[i*ndofV]   = 0#y[i] 
          bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0#-x[i]
    #end if
#end for

if pin_one_node:
   bc_fixV[0] = True ; bc_valV[0] = 0 
   bc_fixV[1] = True ; bc_valV[1] = 0


print("defining V b.c. (%.3fs)" % (time.time() - start))

#################################################################
# define temperature boundary conditions
#################################################################
start = time.time()

bc_fixT = np.zeros(NfemT, dtype=np.bool)  
bc_valT = np.zeros(NfemT, dtype=np.float64) 

for i in range(0,nnp):
    if r[i]<R1+eps:
       bc_fixT[i] = True ; bc_valT[i] = Temp_cmb
    if r[i]>(R2-eps):
       bc_fixT[i] = True ; bc_valT[i] = Temp_surf
#end for

print("defining T b.c. (%.3fs)" % (time.time() - start))

#################################################################
# initial temperature field 
#################################################################
start = time.time()

T = np.zeros(nnp,dtype=np.float64)          # temperature 
T_old = np.zeros(nnp,dtype=np.float64)          # y-component velocity

for i in range(0,nnp):
    s=(R2-r[i])/(R2-R1)
    T[i]=(np.log(r[i]/R2))/(np.log(R1/R2))+0.2*s*(1.-s)*np.cos(N0*theta[i])
#end for

T_old=T

print("temperature layout (%.3fs)" % (time.time() - start))

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
q     = np.zeros(nnp,dtype=np.float64)          # nodal pressure 
q_old = np.zeros(nnp,dtype=np.float64)
u_bef = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v_bef = np.zeros(nnp,dtype=np.float64)          # y-component velocity
dqdt  = np.zeros(nnp,dtype=np.float64)
eta   = np.zeros(nel,dtype=np.float64)          # elemental visc for visu
Res   = np.zeros(NfemV,dtype=np.float64)         # non-linear residual 
sol   = np.zeros(NfemV,dtype=np.float64)         # solution vector 
u_old = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v_old = np.zeros(nnp,dtype=np.float64)          # y-component velocity
k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
Nu_boundary1_old=0.
Nu_boundary2_old=0.
model_time=0.

for istep in range(0,nstep):
    print("----------------------------------")
    print("istep= ", istep)
    print("----------------------------------")

    #################################################################
    # build FE matrix
    #################################################################
    start = time.time()

    a_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix of Ax=b
    b_mat = np.zeros((3,ndofV*m),dtype=np.float64)    # gradient matrix B 
    rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side of Ax=b
    N     = np.zeros(m,dtype=np.float64)             # shape functions
    dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    JxWq  = np.zeros(4*nel,dtype=np.float64)         # weight*jacobian at qpoint 
    etaq  = np.zeros(4*nel,dtype=np.float64)         # viscosity at q points
    rhoq  = np.zeros(4*nel,dtype=np.float64)         # density at q points

    for iter_nl in range(0,niter_nl):

        print("iter_nl= ", iter_nl)

        iiq=0
        for iel in range(0, nel):

            # set 2 arrays to 0 every loop
            b_el = np.zeros(m*ndofV,dtype=np.float64)
            a_el = np.zeros((m*ndofV,m*ndofV),dtype=np.float64)

            # integrate viscous term at 4 quadrature points
            for iq in [-1, 1]:
                for jq in [-1, 1]:

                    # position & weight of quad. point
                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    weightq=1.*1.

                    # calculate shape functions
                    N[0:m]=NNV(rq,sq)
                    dNdr[0:m]=dNNVdr(rq,sq)
                    dNds[0:m]=dNNVds(rq,sq)

                    # calculate jacobian matrix
                    jcb = np.zeros((2, 2),dtype=np.float64)
                    for k in range(0,m):
                        jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                        jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                        jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                        jcb[1, 1] += dNds[k]*y[icon[k,iel]]
                    #end for
                    jcob = np.linalg.det(jcb)
                    jcbi = np.linalg.inv(jcb)

                    JxWq[iiq]=jcob*weightq

                    # compute dNdx & dNdy
                    xq=0.0
                    yq=0.0
                    Tq=0.0
                    exxq=0.
                    eyyq=0.
                    exyq=0.
                    for k in range(0, m):
                        xq+=N[k]*x[icon[k,iel]]
                        yq+=N[k]*y[icon[k,iel]]
                        Tq+=N[k]*T[icon[k,iel]]
                        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                        exxq+=dNdx[k]*u[icon[k,iel]]
                        eyyq+=dNdy[k]*v[icon[k,iel]]
                        exyq+=0.5*dNdy[k]*u[icon[k,iel]]+\
                              0.5*dNdx[k]*v[icon[k,iel]]
                    #end for
                    rq=np.sqrt(xq*xq+yq*yq)
                    rhoq[iiq]=density(rho0,alphaT,Tq,T0)
                    etaq[iiq]=viscosity(Tq,exxq,eyyq,exyq,gamma_T,gamma_y,sigma_y,eta_star,case,rq,R2)

                    # construct 3x8 b_mat matrix
                    for i in range(0, m):
                        b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                                 [0.     ,dNdy[i]],
                                                 [dNdy[i],dNdx[i]]]
                    #end for

                    # compute elemental a_mat matrix
                    a_el += b_mat.T.dot(c_mat.dot(b_mat))*etaq[iiq]*JxWq[iiq]

                    # compute elemental rhs vector
                    for i in range(0, m):
                        b_el[2*i  ]+=N[i]*jcob*weightq*gx(xq,yq,g0)*rhoq[iiq]
                        b_el[2*i+1]+=N[i]*jcob*weightq*gy(xq,yq,g0)*rhoq[iiq]
                    #end for

                    iiq+=1

                #end for jq
            #end for iq

            # integrate penalty term at 1 point
            rq=0.
            sq=0.
            weightq=2.*2.

            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)

            # compute the jacobian
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            #end for
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx and dNdy
            for k in range(0,m):
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
            #end for

            # compute gradient matrix
            for i in range(0,m):
                b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                                      [0.     ,dNdy[i]],
                                      [dNdy[i],dNdx[i]]]
            #end for

            # compute elemental matrix
            a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*weightq*jcob

            # apply boundary conditions
            for k1 in range(0,m):
                for i1 in range(0,ndofV):
                    m1 =ndofV*icon[k1,iel]+i1
                    if bc_fixV[m1]: 
                       fixt=bc_valV[m1]
                       ikk=ndofV*k1+i1
                       aref=a_el[ikk,ikk]
                       for jkk in range(0,m*ndofV):
                           b_el[jkk]-=a_el[jkk,ikk]*fixt
                           a_el[ikk,jkk]=0.
                           a_el[jkk,ikk]=0.
                       #end for 
                       a_el[ikk,ikk]=aref
                       b_el[ikk]=aref*fixt
                    #end if
                #end for 
            #end for

            if use_freeslip_inner:
               if elt_inner[iel]: # if element is on inner boundary
                  for iii in range(0,m): # loop over corners of element
                      inode=icon[iii,iel]
                      if node_inner[inode] and not bc_fixV[inode]: # if node is on inner boundary 
                         # define rotation matrix for -theta angle
                         RotMat=np.zeros((8,8),dtype=np.float64)
                         for i in range(0,8):
                             RotMat[i,i]=1.
                         if iii==0:
                            RotMat[0,0]= np.cos(theta[inode]) ; RotMat[0,1]=np.sin(theta[inode])  
                            RotMat[1,0]=-np.sin(theta[inode]) ; RotMat[1,1]=np.cos(theta[inode])  
                         if iii==1:  
                            RotMat[2,2]= np.cos(theta[inode]) ; RotMat[2,3]=np.sin(theta[inode])  
                            RotMat[3,2]=-np.sin(theta[inode]) ; RotMat[3,3]=np.cos(theta[inode])  
                         # apply counter rotation 
                         a_el=RotMat.dot(a_el.dot(RotMat.T))
                         b_el=RotMat.dot(b_el)
                         # apply boundary conditions
                         ikk=iii*ndofV
                         fixt=0
                         aref=a_el[ikk,ikk]
                         for jkk in range(0,m*ndofV):
                             b_el[jkk]-=a_el[jkk,ikk]*fixt
                             a_el[ikk,jkk]=0.
                             a_el[jkk,ikk]=0.
                         a_el[ikk,ikk]=aref
                         b_el[ikk]=aref*fixt
                         # apply positive rotation 
                         a_el=RotMat.T.dot(a_el.dot(RotMat))
                         b_el=RotMat.T.dot(b_el)
                      #end if
                  #end for
               #end if
            #end if

            if use_freeslip_outer:
               if elt_outer[iel]: # if element is on outer boundary
                  for iii in range(0,m): # loop over corners of element
                      inode=icon[iii,iel]
                      if node_outer[inode]: # if node is on outer boundary 
                         # define rotation matrix for -theta angle
                         RotMat=np.zeros((8,8),dtype=np.float64)
                         for i in range(0,8):
                             RotMat[i,i]=1.
                         if iii==2:
                            RotMat[4,4]= np.cos(theta[inode]) ; RotMat[4,5]=np.sin(theta[inode])  
                            RotMat[5,4]=-np.sin(theta[inode]) ; RotMat[5,5]=np.cos(theta[inode])  
                         if iii==3:  
                            RotMat[6,6]= np.cos(theta[inode]) ; RotMat[6,7]=np.sin(theta[inode])  
                            RotMat[7,6]=-np.sin(theta[inode]) ; RotMat[7,7]=np.cos(theta[inode])  
                         # apply counter rotation 
                         a_el=RotMat.dot(a_el.dot(RotMat.T))
                         b_el=RotMat.dot(b_el)
                         # apply boundary conditions
                         ikk=iii*ndofV
                         fixt=0
                         aref=a_el[ikk,ikk]
                         for jkk in range(0,m*ndofV):
                             b_el[jkk]-=a_el[jkk,ikk]*fixt
                             a_el[ikk,jkk]=0.
                             a_el[jkk,ikk]=0.
                         a_el[ikk,ikk]=aref
                         b_el[ikk]=aref*fixt
                         # apply positive rotation 
                         a_el=RotMat.T.dot(a_el.dot(RotMat))
                         b_el=RotMat.T.dot(b_el)
                      #end if
                  #end for
               #end if
            #end if

            # assemble matrix a_mat and right hand side rhs
            for k1 in range(0,m):
                for i1 in range(0,ndofV):
                    ikk=ndofV*k1          +i1
                    m1 =ndofV*icon[k1,iel]+i1
                    for k2 in range(0,m):
                        for i2 in range(0,ndofV):
                            jkk=ndofV*k2          +i2
                            m2 =ndofV*icon[k2,iel]+i2
                            a_mat[m1,m2]+=a_el[ikk,jkk]
                        #end for 
                    #end for 
                    rhs[m1]+=b_el[ikk]
                #end for 
            #end for 

        #end for iel

        print("build FE matrix & rhs (%.3fs)" % (time.time() - start))

        #np.savetxt('etaq.ascii',np.array(etaq).T,header='# r,T')
        #np.savetxt('rhoq.ascii',np.array(rhoq).T,header='# r,T')

        #################################################################
        # compute non-linear residual
        #################################################################
        start = time.time()

        Res=a_mat.dot(sol)-rhs

        if iter_nl==0:
           Res0=np.max(abs(rhs))

        print("Nonlinear normalised residual (inf. norm) %.7e" % (np.max(abs(Res))/Res0))
          
        convfile.write("%e %e %e %e %e %e  \n" %( istep+iter_nl/200.,  np.max(abs(Res))/Res0, np.min(u),np.max(u),np.min(v),np.max(v) ))
        convfile.flush()

        if nonlinear and (np.max(abs(Res))/Res0 < tol_nl or iter_nl==niter_nl-1):
           niterfile.write("%d %d \n" %( istep, iter_nl ))
           niterfile.flush()
           break 

        print("computing nl residual (%.3fs)" % (time.time() - start))

        #################################################################
        # compute eigenvalues
        #################################################################
        start = time.time()

        if compute_eigenvalues:
           eigenvalues = np.linalg.eigvals(a_mat)
           np.savetxt('eigenvalues.ascii',np.array([eigenvalues.real,eigenvalues.imag]).T)

        print("compute eigenvalues (%.3fs)" % (time.time() - start))

        #################################################################
        # solve system
        #################################################################
        start = time.time()

        sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
        print("solving system (%.3fs)" % (time.time() - start))

        #####################################################################
        # put solution into separate x,y velocity arrays
        #####################################################################
        start = time.time()

        u,v=np.reshape(sol,(nnp,2)).T

        stats_u_file.write("%e %e %e \n" %(istep+iter_nl/200.,np.min(u),np.max(u))) ; stats_u_file.flush()
        stats_v_file.write("%e %e %e \n" %(istep+iter_nl/200.,np.min(v),np.max(v))) ; stats_v_file.flush()

        print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
        print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

        print("reshape solution (%.3fs)" % (time.time() - start))

        #####################################################################
        # compute angular momentum omega_z 
        #####################################################################
        start = time.time()

        Izz=0.
        Hz=0.
        iiq=0
        for iel in range(0,nel):
            for iq in [-1, 1]:
                for jq in [-1, 1]:
                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    weightq=1.*1.
                    N[0:m]=NNV(rq,sq)
                    dNdr[0:m]=dNNVdr(rq,sq)
                    dNds[0:m]=dNNVds(rq,sq)
                    jcb=np.zeros((2,2),dtype=np.float64)
                    for k in range(0,m):
                        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                    #end for
                    jcob = np.linalg.det(jcb)
                    xq=0.
                    yq=0.
                    uq=0.
                    vq=0.
                    for k in range(0,m):
                        xq+=N[k]*x[icon[k,iel]]
                        yq+=N[k]*y[icon[k,iel]]
                        uq+=N[k]*u[icon[k,iel]]
                        vq+=N[k]*v[icon[k,iel]]
                    #end for
                    #Hz+=rhoq[iiq]*(xq*vq-yq*uq)*jcob*weightq
                    #Izz+=rhoq[iiq]*(xq**2+yq**2)*jcob*weightq
                    Hz+=(xq*vq-yq*uq)*jcob*weightq
                    Izz+=(xq**2+yq**2)*jcob*weightq
                    iiq+=1
                #end for jq
            #end for iq
        #end for iel
        omega_z=Hz/Izz

        print("     -> ang. momentum omega_z %e " % omega_z)

        omega_z_file.write("%e %e \n" %(istep+iter_nl/200.,omega_z))
        omega_z_file.flush()

        print("compute ang. momentum omega_z (%.3fs)" % (time.time() - start))

        #####################################################################
        # remove net rotation: correct velocity field 
        #####################################################################
        start = time.time()

        u_bef[:]=u[:]
        v_bef[:]=v[:]

        if remove_net_rotation:
           for i in range(0,nnp):
               u[i]-= -y[i]*omega_z
               v[i]-= +x[i]*omega_z
           #end for
        #end if 

        print("remove net rotation (%.3fs)" % (time.time() - start))

    # end of nonlinear iterations

    #####################################################################
    # relaxation step
    #####################################################################

    if find_ss:
       u=relax*u+(1-relax)*u_old
       v=relax*v+(1-relax)*v_old

    #####################################################################
    start = time.time()

    vr= np.cos(theta)*u+np.sin(theta)*v
    vt=-np.sin(theta)*u+np.cos(theta)*v

    stats_vr_file.write("%d %e %e \n" %(istep,np.min(vr),np.max(vr))) ; stats_vr_file.flush()
    stats_vt_file.write("%d %e %e \n" %(istep,np.min(vt),np.max(vt))) ; stats_vt_file.flush()

    print("     -> vr (m,M) %.4f %.4f " %(np.min(vr),np.max(vr)))
    print("     -> vt (m,M) %.4f %.4f " %(np.min(vt),np.max(vt)))

    print("compute vr,vt (%.3fs)" % (time.time() - start))

    #####################################################################
    # retrieve pressure
    #####################################################################
    start = time.time()

    xc = np.zeros(nel,dtype=np.float64)  
    yc = np.zeros(nel,dtype=np.float64)  
    p  = np.zeros(nel,dtype=np.float64)  
    exx = np.zeros(nel,dtype=np.float64)  
    eyy = np.zeros(nel,dtype=np.float64)  
    exy = np.zeros(nel,dtype=np.float64)  
    dTdx = np.zeros(nel,dtype=np.float64)  
    dTdy = np.zeros(nel,dtype=np.float64)  
    dTdr = np.zeros(nel,dtype=np.float64)  
    dTdtheta = np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):
        rq = 0.0
        sq = 0.0
        N[0:m]=NNV(rq,sq)
        dNdr[0:m]=dNNVdr(rq,sq)
        dNds[0:m]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,m):
            jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
            jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
            jcb[1,0]+=dNds[k]*x[icon[k,iel]]
            jcb[1,1]+=dNds[k]*y[icon[k,iel]]
        #end for
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)
        for k in range(0, m):
            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
        #end for
        for k in range(0, m):
            xc[iel] += N[k]*x[icon[k,iel]]
            yc[iel] += N[k]*y[icon[k,iel]]
            exx[iel] += dNdx[k]*u[icon[k,iel]]
            eyy[iel] += dNdy[k]*v[icon[k,iel]]
            exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+\
                        0.5*dNdx[k]*v[icon[k,iel]]
            dTdx[iel]+=dNdx[k]*T[icon[k,iel]]
            dTdy[iel]+=dNdy[k]*T[icon[k,iel]]
            thetac=math.atan2(yc[iel],xc[iel])
            dTdr    [iel]= np.cos(thetac)*dTdx[iel]+np.sin(thetac)*dTdy[iel]
            dTdtheta[iel]=-np.sin(thetac)*dTdx[iel]+np.cos(thetac)*dTdy[iel]
        #end for
        p[iel]=-penalty*(exx[iel]+eyy[iel])
    #end for

    print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))
    print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
    print("     -> dTdx (m,M) %.4e %.4e " %(np.min(dTdx),np.max(dTdx)))
    print("     -> dTdy (m,M) %.4e %.4e " %(np.min(dTdy),np.max(dTdy)))
    print("     -> dTdr (m,M) %.4e %.4e " %(np.min(dTdr),np.max(dTdr)))
    print("     -> dTdtheta (m,M) %.4e %.4e " %(np.min(dTdtheta),np.max(dTdtheta)))

    print("compute p & sr (%.3fs)" % (time.time() - start))

    ######################################################################
    # compute time step value 
    ######################################################################
    start = time.time()

    dt1=CFL_nb*min(sx,sy)/np.max(np.sqrt(u**2+v**2))
    dt2=CFL_nb*min(sx,sy)**2/(hcond/hcapa/rho0)
    dt=min(dt1,dt2)

    model_time+=dt

    print('     -> dt1= %.6e ' % dt1)
    print('     -> dt2= %.6e ' % dt2)
    print('     -> dt = %.6e ' % dt)

    if find_ss:
       dt=1.
       print('     -> dt find_ss = %.6e ' % dt)

    dt_file.write("%d %e \n" %(istep,dt)) ; dt_file.flush()

    print("compute timestep (%.3fs)" % (time.time() - start))

    ######################################################################
    # compute total heat flux on boundaries
    ######################################################################
    start = time.time()

    hf1=0.
    hf2=0.
    for iel in range(0,nel):
        if elt_inner[iel]:
           surf=np.sqrt( (x[icon[1,iel]]-x[icon[0,iel]])**2+\
                         (y[icon[1,iel]]-y[icon[0,iel]])**2 )
           hf1+=dTdr[iel]*surf
        if elt_outer[iel]:
           surf=np.sqrt( (x[icon[3,iel]]-x[icon[2,iel]])**2+\
                         (y[icon[3,iel]]-y[icon[2,iel]])**2 )
           hf2-=dTdr[iel]*surf
 
    Nu_boundary1=abs(hf1/(2*math.pi*R1)*np.log(f)/(1-f)*f)
    Nu_boundary2=abs(hf2/(2*math.pi*R2)*np.log(f)/(1-f)  )

    Nu1_file.write("%e %e \n" %(model_time,Nu_boundary1)) ; Nu1_file.flush()
    Nu2_file.write("%e %e \n" %(model_time,Nu_boundary2)) ; Nu2_file.flush()
    hf1_file.write("%e %e \n" %(model_time,hf1))          ; hf1_file.flush()
    hf2_file.write("%e %e \n" %(model_time,hf2))          ; hf2_file.flush()

    print("     -> heat flux inner boundary %.4e " % hf1 )
    print("     -> heat flux outer boundary %.4e " % hf2 )
    print("     -> Nusselt nb inner boundary %.4e " % Nu_boundary1)
    print("     -> Nusselt nb outer boundary %.4e " % Nu_boundary2)

    print("compute heat flux on boundaries (%.3fs)" % (time.time() - start))

    ######################################################################
    # compute nodal pressure
    ######################################################################
    start = time.time()

    count=np.zeros(nnp,dtype=np.float64)
    q[:]=0
    for iel in range(0,nel):
        q[icon[0,iel]]+=p[iel]
        q[icon[1,iel]]+=p[iel]
        q[icon[2,iel]]+=p[iel]
        q[icon[3,iel]]+=p[iel]
        count[icon[0,iel]]+=1
        count[icon[1,iel]]+=1
        count[icon[2,iel]]+=1
        count[icon[3,iel]]+=1
    q=q/count

    dqdt=(q[:]-q_old[:])/dt

    print("     -> q (m,M) %.4e %.4e " %(np.min(q),np.max(q)))

    print("compute nodal press (%.3fs)" % (time.time() - start))

    ######################################################################
    # build FE matrix for Temperature 
    ######################################################################
    start = time.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    Tvect = np.zeros(m,dtype=np.float64)

    iiq=0
    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)
        f_el=np.zeros(m*ndofT,dtype=np.float64)

        for k in range(0,m):
            Tvect[k]=T[icon[k,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                # calculate shape functions
                N_mat[0,0]=0.25*(1.-rq)*(1.-sq)
                N_mat[1,0]=0.25*(1.+rq)*(1.-sq)
                N_mat[2,0]=0.25*(1.+rq)*(1.+sq)
                N_mat[3,0]=0.25*(1.-rq)*(1.+sq)
                dNdr[0:m]=dNNVdr(rq,sq)
                dNds[0:m]=dNNVds(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((2, 2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                #end for
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy and Phi
                Tq=0.
                vel[0,0]=0.
                vel[0,1]=0.
                exxq=0.
                eyyq=0.
                exyq=0.
                dqdxq=0.
                dqdyq=0.
                for k in range(0,m):
                    Tq+=N_mat[k,0]*T[icon[k,iel]]
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                    exxq+=dNdx[k]*u[icon[k,iel]]
                    eyyq+=dNdy[k]*v[icon[k,iel]]
                    exyq+=(dNdy[k]*u[icon[k,iel]]+dNdx[k]*v[icon[k,iel]])*.5
                    dqdxq+=dNdx[k]*q[icon[k,iel]]
                    dqdyq+=dNdy[k]*q[icon[k,iel]]
                #end for
                Phiq=2.*etaq[iiq]*(exxq**2+eyyq**2+2*exyq**2)

                if use_BA or use_EBA:
                   rho_lhs=rho0
                else:
                   rho_lhs=rhoq[iiq]
                #end if

                # compute mass matrix
                if not find_ss:
                   MM=N_mat.dot(N_mat.T)*rho_lhs*hcapa*weightq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho_lhs*hcapa*weightq*jcob

                # compute shear heating rhs term
                if use_shearheating:
                   f_el[:]+=N_mat[:,0]*Phiq*jcob*weightq
                else:
                   f_el[:]+=0

                # compute adiabatic heating rhs term 
                if use_adiabatic_heating:
                   f_el[:]+=N_mat[:,0]*alphaT*Tq*(vel[0,0]*dqdxq+vel[0,1]*dqdyq)*weightq*jcob
                else:
                   f_el[:]+=0

                a_el=MM+(Ka+Kd)*dt*alpha

                b_el=(MM-(Ka+Kd)*(1.-alpha)*dt).dot(Tvect) + f_el*dt

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

                iiq+=1

            #end for jq
        #end for iq
    #end for iel

    #print("A_mat (m,M) = %.4f %.4f" %(np.min(A_mat),np.max(A_mat)))
    #print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

    print("build FEM matrix T (%.3fs)" % (time.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = time.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    stats_T_file.write("%d %e %e \n" %(istep,np.min(T),np.max(T))) ; stats_T_file.flush()

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T (%.3fs)" % (time.time() - start))

    #################################################################
    # relax
    #################################################################

    if find_ss:
       T=relax*T+(1-relax)*T_old

    #####################################################################
    ######################################################################
    # compute nodal temperature gradient
    ######################################################################
    start = time.time()

    dTdr_n_1 = np.zeros(nnp,dtype=np.float64) 
    dTdtheta_n_1 = np.zeros(nnp,dtype=np.float64) 
    count=np.zeros(nnp,dtype=np.float64)

    for iel in range(0,nel):
        dTdr_n_1[icon[0,iel]]+=dTdr[iel]
        dTdr_n_1[icon[1,iel]]+=dTdr[iel]
        dTdr_n_1[icon[2,iel]]+=dTdr[iel]
        dTdr_n_1[icon[3,iel]]+=dTdr[iel]
        dTdtheta_n_1[icon[0,iel]]+=dTdtheta[iel]
        dTdtheta_n_1[icon[1,iel]]+=dTdtheta[iel]
        dTdtheta_n_1[icon[2,iel]]+=dTdtheta[iel]
        dTdtheta_n_1[icon[3,iel]]+=dTdtheta[iel]
        count[icon[0,iel]]+=1
        count[icon[1,iel]]+=1
        count[icon[2,iel]]+=1
        count[icon[3,iel]]+=1
    #end for

    dTdr_n_1=dTdr_n_1/count
    dTdtheta_n_1=dTdtheta_n_1/count

    print("     -> dTdr_n_1     (m,M) %.4e %.4e " %(np.min(dTdr_n_1),np.max(dTdr_n_1)))
    print("     -> dTdtheta_n_1 (m,M) %.4e %.4e " %(np.min(dTdtheta_n_1),np.max(dTdtheta_n_1)))

    print("compute temperature gradient (%.3fs)" % (time.time() - start))

    ######################################################################
    # compute vrms and other averages 
    ######################################################################
    start = time.time()

    avrg_vr=0.
    avrg_vt=0.
    avrg_sr=0.
    avrg_T=0.
    avrg_eta=0.
    omega=0.
    vrms=0.
    vrrms=0.
    vtrms=0.
    EK=0.
    ET=0.
    EG=0.
    visc_diss=0.
    work_grav=0.
    adiab_heating=0.
    iiq=0
    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.
                N[0:m]=NNV(rq,sq)
                dNdr[0:m]=dNNVdr(rq,sq)
                dNds[0:m]=dNNVds(rq,sq)
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                #end for
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)
                for k in range(0,m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                #end for
                xq=0. ; yq=0.
                uq=0. ; vq=0.
                vrq=0. ; vtq=0.
                Tq=0.
                exxq=0. ; eyyq=0. ; exyq=0.
                dqdxq=0. ; dqdyq=0.
                for k in range(0,m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    vrq+=N[k]*vr[icon[k,iel]]
                    vtq+=N[k]*vt[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
                    dqdxq+=dNdx[k]*q[icon[k,iel]]
                    dqdyq+=dNdy[k]*q[icon[k,iel]]
                    exxq+=dNdx[k]*u[icon[k,iel]]
                    eyyq+=dNdy[k]*v[icon[k,iel]]
                    exyq+=0.5*dNdy[k]*u[icon[k,iel]]+\
                          0.5*dNdx[k]*v[icon[k,iel]]
                #end for
                avrg_vr+=vrq*weightq*jcob
                avrg_vt+=vtq*weightq*jcob
                avrg_T+=Tq*weightq*jcob
                vrms+=(uq**2+vq**2)*weightq*jcob
                vrrms+=(vrq**2)*weightq*jcob
                vtrms+=(vtq**2)*weightq*jcob
                visc_diss+=2.*etaq[iiq]*(exxq**2+eyyq**2+2*exyq**2)*weightq*jcob
                avrg_sr+=0.5*(exxq**2+eyyq**2+2*exyq**2)*weightq*jcob
                EK+=0.5*rhoq[iiq]*(uq**2+vq**2)*weightq*jcob
                EG+=rhoq[iiq]*(gx(xq,yq,g0)*xq+gy(xq,yq,g0)*yq)*weightq*jcob
                work_grav+=(rhoq[iiq]-rho0)*(gx(xq,yq,g0)*uq+gy(xq,yq,g0)*vq)*weightq*jcob
                adiab_heating+=alphaT*Tq*(uq*dqdxq+vq*dqdyq)*weightq*jcob
                if use_BA or use_EBA:
                   ET+=rho0*hcapa*Tq*weightq*jcob
                else:
                   ET+=rhoq[iiq]*hcapa*Tq*weightq*jcob
                #end if
                #omega+=rhoq[iiq]*(xq*vq-yq*uq)*weightq*jcob
                omega+=(xq*vq-yq*uq)*weightq*jcob
                iiq+=1
            #end for jq
        #end for iq
    #end for iel
    avrg_eta=visc_diss*0.25/avrg_sr

    avrg_vr/=area 
    avrg_vt/=area 
    avrg_T/=area
    avrg_sr/=area
    vrms=np.sqrt(vrms/area)
    vrrms=np.sqrt(vrrms/area)
    vtrms=np.sqrt(vtrms/area)

    omega_z_file.write("%e %e \n" %(istep+iter_nl/200.,omega))
    omega_z_file.flush()

    if istep==0:
       EG_0=EG

    avrg_sr_file.write("%d %e \n" %(istep,avrg_sr))                 ; avrg_sr_file.flush()
    avrg_vr_file.write("%d %e %e \n" %(istep,avrg_vr,avrg_vr/vrms)) ; avrg_vr_file.flush()
    avrg_vt_file.write("%d %e %e \n" %(istep,avrg_vt,avrg_vt/vrms)) ; avrg_vt_file.flush()
    avrg_T_file.write("%e %e \n" %(model_time,avrg_T))              ; avrg_T_file.flush()
    avrg_eta_file.write("%e %e \n" %(model_time,avrg_eta))          ; avrg_eta_file.flush()
    vrms_file.write("%e %e \n" %(model_time,vrms))                  ; vrms_file.flush()
    vrrms_file.write("%e %e \n" %(model_time,vrrms))                ; vrrms_file.flush()
    vtrms_file.write("%e %e \n" %(model_time,vtrms))                ; vtrms_file.flush()
    EK_file.write("%e %e \n" %(model_time,EK))                      ; EK_file.flush()
    ET_file.write("%e %e \n" %(model_time,ET))                      ; ET_file.flush()
    EG_file.write("%e %e %e \n" %(model_time,EG,EG-EG_0))           ; EG_file.flush()
    visc_diss_file.write("%e %e \n" %(model_time,visc_diss))        ; visc_diss_file.flush()
    work_grav_file.write("%e %e \n" %(model_time,work_grav))        ; work_grav_file.flush()
    adiab_heat_file.write("%e %e \n" %(model_time,adiab_heating))   ; adiab_heat_file.flush()

    if istep>0:
       dETdt_file.write("%e %e \n" %(model_time,(ET-ET_old)/dt )) ; dETdt_file.flush()

    cons_file.write("%e %e %e %e %e\n" %(model_time,visc_diss,adiab_heating,hf1,hf2))

    print("     -> avrg vr= %.6e" % avrg_vr)
    print("     -> avrg vt= %.6e" % avrg_vt)
    print("     -> avrg  T= %.6e" % avrg_T)
    print("     -> vrms= %.6e ; Ra= %.4e " % (vrms,Ra))

    print("compute vrms, Tavrg, EK, EG, WAG (%.3fs)" % (time.time() - start))

    #####################################################################
    # compute power spectrum 
    #####################################################################
    start = time.time()

    if istep%every_ps==0:

       PS_T = np.zeros(2,dtype=np.float64) 
       PS_V = np.zeros(2,dtype=np.float64) 
       PS_vr = np.zeros(2,dtype=np.float64) 
       PS_vt = np.zeros(2,dtype=np.float64) 

       for kk in range (1,13): 
           PS_T[:]=0.
           PS_V[:]=0.
           PS_vr[:]=0.
           PS_vt[:]=0.
           iiq=0
           for iel in range (0,nel):
               for iq in [-1,1]:
                   for jq in [-1,1]:
                       rq=iq/sqrt3
                       sq=jq/sqrt3
                       N[0:m]=NNV(rq,sq)
                       xq=0. ; yq=0.
                       uq=0. ; vq=0.
                       vrq=0. ; vtq=0.
                       Tq=0.
                       for k in range(0,m):
                           xq+=N[k]*x[icon[k,iel]]
                           yq+=N[k]*y[icon[k,iel]]
                           uq+=N[k]*u[icon[k,iel]]
                           vq+=N[k]*v[icon[k,iel]]
                           Tq+=N[k]*T[icon[k,iel]]
                           vrq+=N[k]*vr[icon[k,iel]]
                           vtq+=N[k]*vt[icon[k,iel]]
                       #end for
                       thetaq=math.atan2(yq,xq)
                       coskkthetaq=np.cos(kk*thetaq)
                       sinkkthetaq=np.sin(kk*thetaq)
                       PS_T[0]+=Tq*JxWq[iiq]*coskkthetaq
                       PS_T[1]+=Tq*JxWq[iiq]*sinkkthetaq
                       PS_V[0]+=np.sqrt(uq**2+vq**2)*JxWq[iiq]*coskkthetaq
                       PS_V[1]+=np.sqrt(uq**2+vq**2)*JxWq[iiq]*sinkkthetaq
                       PS_vr[0]+=vrq*JxWq[iiq]*coskkthetaq
                       PS_vr[1]+=vrq*JxWq[iiq]*sinkkthetaq
                       PS_vt[0]+=vtq*JxWq[iiq]*coskkthetaq
                       PS_vt[1]+=vtq*JxWq[iiq]*sinkkthetaq
                       iiq+=1
                   #end for jq
               #end for iq
           #end for iel
           psT_file.write("%d %d %e %e %e %e \n " %(kk,istep,PS_T[0],PS_T[1],np.sqrt(PS_T[0]**2+PS_T[1]**2),model_time) )
           psV_file.write("%d %d %e %e %e %e \n " %(kk,istep,PS_V[0],PS_V[1],np.sqrt(PS_V[0]**2+PS_V[1]**2),model_time) )
           psvr_file.write("%d %d %e %e %e %e \n " %(kk,istep,PS_vr[0],PS_vr[1],np.sqrt(PS_vr[0]**2+PS_vr[1]**2),model_time) )
           psvt_file.write("%d %d %e %e %e %e \n " %(kk,istep,PS_vt[0],PS_vt[1],np.sqrt(PS_vt[0]**2+PS_vt[1]**2),model_time) )
       #end for kk

       psT_file.write(" \n") ; psT_file.flush()
       psV_file.write(" \n") ; psV_file.flush()
       psvr_file.write(" \n") ; psvr_file.flush()
       psvt_file.write(" \n") ; psvt_file.flush()

    #end if

    print("compute power spectrum (%.3fs)" % (time.time() - start))

    #####################################################################
    # plot of solution
    #####################################################################
    start = time.time()

    if visu==1 and istep%every==0:

       rho_el= np.zeros(nel,dtype=np.float64)
       eta_el= np.zeros(nel,dtype=np.float64)

       for iel in range(0,nel):
           rho_el[iel]=(rhoq[iel*4]+rhoq[iel*4+1]+rhoq[iel*4+2]+rhoq[iel*4+3])*0.25
           eta_el[iel]=(etaq[iel*4]+etaq[iel*4+1]+etaq[iel*4+2]+etaq[iel*4+3])*0.25

       #np.savetxt('eta_el.ascii',np.array(eta_el).T,header='# r,T')
       #np.savetxt('rho_el.ascii',np.array(rho_el).T,header='# r,T')

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%f\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='elt_inner' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    if elt_inner[iel]:
       #       vtufile.write("%f\n" % 1.)
       #    else:
       #       vtufile.write("%f\n" % 0.)
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='elt_outer' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    if elt_outer[iel]:
       #       vtufile.write("%f\n" % 1.)
       #    else:
       #       vtufile.write("%f\n" % 0.)
       #vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='viscosity' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10f \n" %eta_el[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='strain rate' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10f \n" %    (np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10f \n" %rho_el[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='grad T (x,y)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e %e %e\n" % (dTdx[iel],dTdy[iel],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='grad T (r,theta)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e %e %e\n" % (dTdr[iel],dTdtheta[iel],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
       #for i in range(0,nnp):
       #    vtufile.write("%10f %10f %10f \n" %(gx(x[i],y[i],g0),gy(x[i],y[i],g0),0.))
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (x,y)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       if remove_net_rotation:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (x,y) - bef remov' Format='ascii'> \n")
          for i in range(0,nnp):
              vtufile.write("%10e %10e %10e \n" %(u_bef[i],v_bef[i],0.))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel removed' Format='ascii'> \n")
          for i in range(0,nnp):
              vtufile.write("%10e %10e %10e \n" %(u_bef[i]-u[i],v_bef[i]-v[i],0.))
          vtufile.write("</DataArray>\n")
       #end if

       #--
       vtufile.write("<DataArray type='Float32' Name='vr' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" %vr[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32'  Name='vt' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" %vt[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='grad T (r,theta)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f %10f %10f \n" %(dTdr_n_1[i],dTdtheta_n_1[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='T' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='node_outer' Format='ascii'> \n")
       #for i in range (0,nnp):
       #    if node_outer[i]:
       #       vtufile.write("%f\n" % 1.)
       #    else:
       #       vtufile.write("%f\n" % 0.)
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='node_inner' Format='ascii'> \n")
       #for i in range (0,nnp):
       #    if node_inner[i]:
       #       vtufile.write("%f\n" % 1.)
       #    else:
       #       vtufile.write("%f\n" % 0.)
       #vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
       #for i in range(0,nnp):
       #    vtufile.write("%10f \n" %r[i])
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
       #for i in range(0,nnp):
       #    vtufile.write("%10f \n" %theta[i])
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()
       print("export to vtu (%.3fs)" % (time.time() - start))

    #####################################################################
    # depth average
    #####################################################################

    start = time.time()

    Tdepthavrg=np.zeros(nnr,dtype=np.float64)
    rdepthavrg=np.zeros(nnr,dtype=np.float64)
    veldepthavrg=np.zeros(nnr,dtype=np.float64)
    vrdepthavrg=np.zeros(nnr,dtype=np.float64)
    vtdepthavrg=np.zeros(nnr,dtype=np.float64)

    counter=0
    for j in range(0,nnr):
        for i in range(0,nelt):
            veldepthavrg[j]+=np.sqrt(u[counter]**2+v[counter]**2)/nelt
            Tdepthavrg[j]+=T[counter]/nelt
            vrdepthavrg[j]+=vr[counter]/nelt
            vtdepthavrg[j]+=vt[counter]/nelt
            rdepthavrg[j]=r[counter]
            counter += 1

    np.savetxt('Tdepthavrg.ascii',np.array([rdepthavrg[0:nnr],Tdepthavrg[0:nnr]]).T,header='# r,T')
    np.savetxt('veldepthavrg.ascii',np.array([rdepthavrg[0:nnr],veldepthavrg[0:nnr]]).T,header='# r,vel')
    np.savetxt('vrdepthavrg.ascii',np.array([rdepthavrg[0:nnr],vrdepthavrg[0:nnr]]).T,header='# r,vr')
    np.savetxt('vtdepthavrg.ascii',np.array([rdepthavrg[0:nnr],vtdepthavrg[0:nnr]]).T,header='# r,vt')

    etadepthavrg=np.zeros(nelr,dtype=np.float64)
    rdepthavrg=np.zeros(nelr,dtype=np.float64)

    counter=0
    for j in range(0, nelr):
        for i in range(0, nelt):
            etadepthavrg[j]+=eta_el[counter]/nelt
            rdepthavrg[j]=(r[icon[0,counter]]+r[icon[3,counter]])/2.
            counter+=1

    np.savetxt('etadepthavrg.ascii',np.array([rdepthavrg[0:nelr],etadepthavrg[0:nelr]]).T,header='# r,T')

    print("export depth averages (%.3fs)" % (time.time() - start))

    #####################################################################

    if np.abs(Nu_boundary1-Nu_boundary1_old) and \
       np.abs(Nu_boundary2-Nu_boundary2_old) <1.e-6:
       print("***********************************")
       print("********Nu converged to 1e-6*******")
       print("***********************************")
       break

    #####################################################################

    if model_time>end_time: 
       print("***********************************")
       print("*********end time reached**********")
       print("***********************************")
       break

    #####################################################################

    ET_old=ET
    Nu_boundary1_old=Nu_boundary1
    Nu_boundary2_old=Nu_boundary2
    q_old=q
    u_old=u
    v_old=v
    T_old=T

#end istep

################################################################################################
# END OF TIMESTEPPING
################################################################################################
       
psV_file.close()
psT_file.close()
psvr_file.close()
psvt_file.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
