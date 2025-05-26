import numpy as np
import sys as sys
import scipy.sparse as sps
import time as timing
import random
from scipy.sparse import lil_matrix
from mpmath import coth
import numba

debug=False

order=2

###############################################################################
# analytical solution
###############################################################################

def solution(x,y):
    dpdx = (9*x**2+4*x+1)*(4*y**3-3*y**2+2*y+1)
    dpdy = (3*x**3+2*x**2+x+4)*(12*y**2-6*y+2)
    rho = -(18*x+4)*(y**4-y**3+y**2+y)-(3*x**3+2*x**2+x+4)*(12*y**2-6*y) 
    u = -dpdx
    v = -dpdy - rho
    p = (3*x**3+2*x**2+x+4)*(4*y**3-3*y**2+2*y+1) 
    return u,v,p,rho

###############################################################################
# Q2 velocity shape functions in ref element [-1:1]x[-1:1]
# rnodes,snodes: coordinates of nodes inside element
###############################################################################
# 2-------3  6---7---8  
# |       |  |       |  
# |       |  3   4   5  
# |       |  |       |  
# 0-------1  0---1---2  

if order==1:
   rnodes=[-1,+1,-1,+1]
   snodes=[-1,-1,+1,+1]
   m=4
if order==2:
   rnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
   snodes=[-1,-1,-1,0,0,0,+1,+1,+1]
   m=9

@numba.jit(nopython=True)
def NN(r,s,order):
    if order==1:
       N_0=0.25*(1.-r)*(1.-s)
       N_1=0.25*(1.+r)*(1.-s)
       N_2=0.25*(1.-r)*(1.+s)
       N_3=0.25*(1.+r)*(1.+s)
       return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)
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
       return np.array([N_0,N_1,N_2,N_3,N_4,N_5,\
                        N_6,N_7,N_8],dtype=np.float64)

@numba.jit(nopython=True)
def dNNdr(r,s,order):
    if order==1:
       dNdr_0=-0.25*(1.-s) 
       dNdr_1=+0.25*(1.-s) 
       dNdr_2=-0.25*(1.+s) 
       dNdr_3=+0.25*(1.+s) 
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)
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
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,\
                        dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

@numba.jit(nopython=True)
def dNNds(r,s,order):
    if order==1:
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.-r)
       dNds_3=+0.25*(1.+r)
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)
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
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,\
                        dNds_6,dNds_7,dNds_8],dtype=np.float64)

###############################################################################

def mygauss_exp1(x,y,Lx):
    return 200*np.exp(-(x-Lx/2)**2/2/1000**2-y**2/2/1000**2)

###############################################################################
# constants

eps=1e-9
year=365.25*3600*24
sqrt2=np.sqrt(2)
ndim=2  
TKelvin=273.15

###############################################################################
###############################################################################
###############################################################################

experiment=4

print("-----------------------------")
print("--------- stone 107 ---------")
print("-----------------------------")

match (experiment):

    case(0):
        Lx=100e3   ; Ly=100e3
        nelx=32    ; nely=32
        Ttop=TKelvin ; Tbottom=500
        ptop=0     ; pbottom=1e7
        eta_f=1e-4 ; T0_f=277.15 ; hcapa_f=4184 ; hcond_f=0.598 ; rho0_f=1000 ; alpha_f=1e-4
        gx=0       ; gy=0
        K_s=1e-13  ; rho_s=2700  ; hcapa_s=1000    ; hcond_s=2
        dt=1*year  ; CFL_nb=0.   ; tfinal=1e6*year ; nstep=1
        phi=0.15
        periodicx=False
        solve_T=False

    case(1):
        Lx=100e3   ; Ly=100e3
        nelx=32    ; nely=32
        Ttop=TKelvin   ; Tbottom=500
        ptop=0     ; pbottom=1000*9.81*Ly
        eta_f=1e-4 ; T0_f=277.15 ; hcapa_f=4184 ; hcond_f=0.598 ; rho0_f=1000 ; alpha_f=0
        gx=0       ; gy=-9.81
        K_s=1e-13  ; rho_s=2700  ; hcapa_s=1000    ; hcond_s=2
        dt=1*year  ; CFL_nb=0.   ; tfinal=1e6*year ; nstep=1
        phi=0.15
        periodicx=False
        solve_T=False

    case(2):
        Lx=10e3      ; Ly=10e3
        nelx=80      ; nely=nelx
        Ttop=TKelvin  ; Tbottom=200+TKelvin
        ptop=1e5      ; pbottom=99e6
        eta_f=1.33e-4 ; T0_f=Ttop   ; hcapa_f=4184 ; hcond_f=0.598 ; rho0_f=1000 ; alpha_f=1e-4
        gx=0          ; gy=-10
        K_s=1e-13     ; rho_s=2700 ; hcapa_s=1000 ; hcond_s=2
        dt=1*year     ; CFL_nb=-0.1 ; tfinal=1e6*year ; nstep=1001
        phi=0.15
        periodicx=False
        solve_T=True

    case(3): #onset of convection
        Lx=4000 ; Ly=1000
        nelx=64  ; nely=16
        Ttop=100+TKelvin ; Tbottom=110+TKelvin
        eta_f=1e-3 ; T0_f=Ttop   ; hcapa_f=4000 ; hcond_f=0.5 ; rho0_f=1000 ; alpha_f=1e-4
        gx=0          ; gy=-10
        K_s=1e-12     ; rho_s=2700 ; hcapa_s=1000 ; hcond_s=2
        dt=500*year    ; CFL_nb=0.8 ; tfinal=1e8*year ; nstep=10000 ; dtmax=500*year
        phi=1
        periodicx=True
        solve_T=True

    case(4): #fault a la Simpson
        Lx=50e3 ; Ly=20e3
        nelx=500  ; nely=200
        Ttop=100+TKelvin ; Tbottom=101+TKelvin
        ptop=0        ; pbottom=2e8*1.5
        eta_f=1e-3    ; T0_f=Ttop   ; hcapa_f=4000 ; hcond_f=0.5 ; rho0_f=1000 ; alpha_f=0# 1e-4
        gx=0          ; gy=-10
        K_s=0         ; rho_s=2700 ; hcapa_s=1000 ; hcond_s=2
        dt=500*year    ; CFL_nb=0.8 ; tfinal=1e8*year ; nstep=1 ; dtmax=500*year
        phi=0.1
        periodicx=False
        solve_T=True

    case(5): # manufactured solution for Darcy
        Lx=1 ; Ly=1
        nelx=64  ; nely=64
        eta_f=1. ; rho0_f=1000 
        gx=0     ; gy=-1
        K_s=1       
        nstep=1 
        periodicx=False
        solve_T=False

    case(6): # advection benchmark
        Lx=1 ; Ly=1
        nelx=50  ; nely=50
        Ttop=0    ; Tbottom=1
        eta_f=1 ; T0_f=Ttop   ; hcapa_f=1 ; hcond_f=0 ; rho0_f=1 ; alpha_f=0
        gx=0     ; gy=0
        ptop=0    ; pbottom=1
        K_s=1     ; rho_s=0   ; hcapa_s=0 ; hcond_s=0
        dt=0.002  ; CFL_nb=-1 ; tfinal=1e8 ; nstep=251 ; dtmax=1
        phi=1
        periodicx=False
        solve_T=True
        TKelvin=0
 


    case _:
        3

every=10
   
visu = 1 
                   
supg_type=2

###############################################################################
# allowing for argument passing through command line,
# thereby overwriting the ones above

if int(len(sys.argv) == 4): 
   nelx = int(sys.argv[1])
   Tbottom = float(sys.argv[2])
   Tbottom+=TKelvin
   visu = int(sys.argv[3])
   print(sys.argv) 
   
   match (experiment):
       case(3):
          nely=int(nelx/4)
       case(5 | 6):
          nely=nelx
       case _:
          exit('plz set nely')

###############################################################################
#compute coeffs based on s,f with phi

if solve_T:
   rho_hcapa_m=(1-phi)*rho_s*hcapa_s+phi*rho0_f*hcapa_f
   hcond_m=(1-phi)*hcond_s+phi*hcond_f
   kappa=hcond_m/rho_hcapa_m
   if kappa>0:
      Ra=K_s*rho0_f*abs(gy)*alpha_f*(Tbottom-Ttop)*Ly/kappa/eta_f 
   else:
      Ra=0
else:
   rho_hcapa_m=0
   hcond_m=0
   Ra=0
   phi=0
   rho_s=0
   T0_f=0
   alpha_f=0

###############################################################################
   
nel=nelx*nely # total number of elements
nny=order*nely+1  # number of nodes, y direction
if periodicx:
   nnx=order*nelx  # number of nodes, x direction
else:
   nnx=order*nelx+1  # number of elements, x direction
N=nnx*nny     # number of nodes in the mesh
NfemP=N       # number of pressure dofs
NfemT=N       # number of temperature dofs
hx=Lx/nelx    # element size in x direction
hy=Ly/nely    # element size in y direction

###############################################################################
# quadrature points coords & weights
###############################################################################

if order==1:
   nqperdim=2
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if order==2:
   nqperdim=3
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

###############################################################################
# open output/statistic files
###############################################################################

dt_file=open('dt.ascii',"w")
Tstats_file=open('stats_T.ascii',"w")
pstats_file=open('stats_p.ascii',"w")
ustats_file=open('stats_u.ascii',"w")
vstats_file=open('stats_v.ascii',"w")
RaNu_file=open('RaNu.ascii',"w")
meascenter_file=open('measurements_center.ascii',"w")

###############################################################################

print('experiment =',experiment)
print('order      =',order)
print('periodicx  =',periodicx)
print('supg_type  =',supg_type)
print('solve_T    =',solve_T)
print('nnx        =',nnx)
print('nny        =',nny)
print('N          =',N)
print('nel        =',nel)
print('NfemT      =',NfemT)
print('NfemP      =',NfemP)
print('nqperdim   =',nqperdim)
print('Ra         =',Ra)
print('phi        =',phi)
print('Lx         =',Lx)
print('Ly         =',Ly)
print('-----------------------------')
print('fluid: eta   =',eta_f)
print('fluid: rho   =',rho0_f)
print('K_s          =',K_s)
print('rho_s        =',rho_s)
print('-----------------------------')
print('-----------------------------')
if solve_T:
   print('hcapa_s      =',hcapa_s)
   print('hcond_s      =',hcond_s)
   print('-----------------------------')
   print('fluid: hcond =',hcond_f)
   print('fluid: hcapa =',hcapa_f)
   print('fluid: alpha =',alpha_f)
   print('-----------------------------')
   print('m: rho*hcapa =',rho_hcapa_m)
   print('m: hcond     =',hcond_m)
   print('kappa        =',kappa)

###############################################################################
# build nodes coordinates 
###############################################################################
start = timing.time()

x=np.zeros(N,dtype=np.float64)  # x coordinates
y=np.zeros(N,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx/order
        y[counter]=j*hy/order
        counter+=1
    #end for
#end for

if debug: np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("build V grid: %.3f s" % (timing.time() - start))

###############################################################################
# connectivity
###############################################################################
start = timing.time()

icon=np.zeros((m,nel),dtype=np.int32)

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

if periodicx and order==1:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           if i==nelx-1: #last column of elts
              icon[1,counter]-=nnx
              icon[3,counter]-=nnx
           counter += 1
       #end for
   #end for

if periodicx and order==2:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           if i==nelx-1: #last column of elts
              icon[2,counter]-=nnx
              icon[5,counter]-=nnx
              icon[8,counter]-=nnx
           counter += 1
       #end for
   #end for

if debug and order==1:
   for iel in range (0,nel):
        print ("iel=",iel)
        print ("node 1",icon[0,iel],"at pos.",x[icon[0,iel]], y[icon[0,iel]])
        print ("node 2",icon[1,iel],"at pos.",x[icon[1,iel]], y[icon[1,iel]])
        print ("node 3",icon[2,iel],"at pos.",x[icon[2,iel]], y[icon[2,iel]])
        print ("node 4",icon[3,iel],"at pos.",x[icon[3,iel]], y[icon[3,iel]])

if debug and order==2:
   for iel in range (0,nel):
        print ("iel=",iel)
        print ("node 1",icon[0,iel],"at pos.",x[icon[0,iel]], y[icon[0,iel]])
        print ("node 2",icon[1,iel],"at pos.",x[icon[1,iel]], y[icon[1,iel]])
        print ("node 3",icon[2,iel],"at pos.",x[icon[2,iel]], y[icon[2,iel]])
        print ("node 4",icon[3,iel],"at pos.",x[icon[3,iel]], y[icon[3,iel]])
        print ("node 5",icon[4,iel],"at pos.",x[icon[4,iel]], y[icon[4,iel]])
        print ("node 6",icon[5,iel],"at pos.",x[icon[5,iel]], y[icon[5,iel]])
        print ("node 7",icon[6,iel],"at pos.",x[icon[6,iel]], y[icon[6,iel]])
        print ("node 8",icon[7,iel],"at pos.",x[icon[7,iel]], y[icon[7,iel]])
        print ("node 9",icon[8,iel],"at pos.",x[icon[8,iel]], y[icon[8,iel]])

print("build icon: %.3f s" % (timing.time() - start))

###############################################################################
#
###############################################################################
start = timing.time()

if experiment==5:
   uth=np.zeros(N,dtype=np.float64) 
   vth=np.zeros(N,dtype=np.float64) 
   pth=np.zeros(N,dtype=np.float64) 
   for i in range(0,N):
       uth[i],vth[i],pth[i],xxx=solution(x[i],y[i])

   print("compute analytical solution: %.3f s" % (timing.time() - start))

###############################################################################
# define pressure boundary conditions
###############################################################################
start = timing.time()

bc_fixP=np.zeros(NfemP,dtype=bool)  # boundary condition, yes/no
bc_valP=np.zeros(NfemP,dtype=np.float64)  # boundary condition, value

match (experiment):
    case(0 | 1 | 2 | 4 | 6):
        for i in range(0,N):
            if y[i]/Ly>(1-eps):
               bc_fixP[i]=True ; bc_valP[i]=ptop
            if y[i]/Ly<eps:
               bc_fixP[i]=True ; bc_valP[i]=pbottom
        #end for
    case(3):
        print('no pressure b.c.')
    case(5):
        for i in range(0,N):
            if y[i]/Ly>(1-eps):
               bc_fixP[i]=True ; bc_valP[i]=pth[i]
            if y[i]/Ly<eps:
               bc_fixP[i]=True ; bc_valP[i]=pth[i]
            if x[i]/Lx>(1-eps):
               bc_fixP[i]=True ; bc_valP[i]=pth[i]
            if x[i]/Lx<eps:
               bc_fixP[i]=True ; bc_valP[i]=pth[i]

print("pressure b.c.: %.3f s" % (timing.time() - start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

match (experiment):
    case(0 | 1 | 3 | 4 | 6):
        for i in range(0,N):
            if y[i]/Ly<eps:
               bc_fixT[i]=True ; bc_valT[i]=Tbottom
            if y[i]/Ly>(1-eps):
               bc_fixT[i]=True ; bc_valT[i]=Ttop
        #end for
    case(2):
        for i in range(0,N):
            if y[i]/Ly<eps:
               bc_fixT[i]=True ; bc_valT[i]=Tbottom + mygauss_exp1(x[i],y[i],Lx)
            if y[i]/Ly>(1-eps):
               bc_fixT[i]=True ; bc_valT[i]=Ttop
        #end for

print("temperature b.c.: %.3f s" % (timing.time() - start))

###############################################################################
# initial temperature
###############################################################################
start = timing.time()

T = np.zeros(N,dtype=np.float64)
T_init = np.zeros(N,dtype=np.float64)

match (experiment):
    case(0 | 1 | 4):
        for i in range(0,N):
            T[i]= Tbottom -y[i]/Ly*(Tbottom-Ttop) # conductive profile
        #end for
    case(2):
        for i in range(0,N):
            T[i]= Tbottom -y[i]/Ly*(Tbottom-Ttop) # conductive profile
            T[i]+=mygauss_exp1(x[i],y[i],Lx)
        #end for
    case(3): 
        for i in range(0,N):
            T[i]=Tbottom -y[i]/Ly*(Tbottom-Ttop) # conductive profile
            if abs(y[i]-Ly/2)/Ly<0.3:
               T[i]+=random.uniform(-1,+1)
        #end for
    case(6): 
        for i in range(0,N):
            if y[i]<0.25: 
               T[i]=1
            else:
               T[i]=0

#            T[i] += np.sin(x[i]/Lx*np.pi)*np.sin(y[i]/Ly*np.pi)
#        #end for

if experiment==3 and order>1: # project T field to Q1 space
   # 6---7---8  
   # |       |  
   # 3   4   5  
   # |       |  
   # 0---1---2  
   for iel in range(0,nel):
       T[icon[1,iel]]=0.5*(T[icon[0,iel]]+T[icon[2,iel]])
       T[icon[7,iel]]=0.5*(T[icon[6,iel]]+T[icon[8,iel]])
       T[icon[5,iel]]=0.5*(T[icon[2,iel]]+T[icon[8,iel]])
       T[icon[3,iel]]=0.5*(T[icon[0,iel]]+T[icon[6,iel]])
       T[icon[4,iel]]=0.25*(T[icon[0,iel]]+T[icon[2,iel]]+\
                            T[icon[6,iel]]+T[icon[8,iel]])

T_init[:]=T[:]

print("     -> Tinit (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

if debug: np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

print("temperature init: %.3f s" % (timing.time() - start))

###############################################################################
# permeability setup
###############################################################################
start = timing.time()

permeability=np.zeros(nel,dtype=np.float64) 
xc=np.zeros(nel,dtype=np.float64) 
yc=np.zeros(nel,dtype=np.float64) 

match(experiment):
    case(0 | 1 | 2 | 3 | 5 | 6):
        permeability[:]=K_s   
    case(4):
        a=0.5
        b=Ly/2
        thickness=200
        for iel in range(0,nel):
            xc[iel]=x[icon[0,iel]]+hx/2
            yc[iel]=y[icon[0,iel]]+hy/2
            if yc[iel]<a*(xc[iel]-0.5*Lx)+b+thickness and\
               yc[iel]>a*(xc[iel]-0.5*Lx)+b-thickness and\
               abs(xc[iel]-0.5*Lx)<10e3 and\
               abs(yc[iel]-0.5*Ly)<5e3:
               permeability[iel]=1e-12
            else:
               permeability[iel]=1e-16

print("     -> permeability (m,M) %e %e " %(np.min(permeability),np.max(permeability)))

print("permeability init: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements (sanity check)
###############################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            dNNNdr=dNNdr(rq,sq,order)
            dNNNds=dNNds(rq,sq,order)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNNNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNNNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNNNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNNNds[k]*y[icon[k,iel]]
            #end for
            jcob = np.linalg.det(jcb)
            #print(jcob)
            area[iel]+=1*jcob*weightq
       #end for
   #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %e %e " %(area.sum(),Lx*Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================

Nu_mem=1e50
time=0

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    ###########################################################################
    # update density on nodes based on current temp field
    ###########################################################################
    start = timing.time()

    rho=np.zeros(N,dtype=np.float64)   
    if solve_T:
       rho[:]=rho0_f*(1-alpha_f*(T[:]-T0_f))
    else:
       for i in range(0,N):
           tmp,tmp,tmp,rho[i]=solution(x[i],y[i])

    print("     -> rho (m,M) %.4e %.4e " %(np.min(rho),np.max(rho)))

    ###########################################################################
    # assemble pressure eq 
    # because elements are rectangles, no need to use mapping to 
    # compute jacobian determinant and inverse
    ###########################################################################
    start = timing.time()

    A_mat=lil_matrix((NfemP,NfemP),dtype=np.float64) # FE matrix 
    rhs=np.zeros(NfemP,dtype=np.float64)             # FE rhs 
    B_mat=np.zeros((2,m),dtype=np.float64)           # gradient matrix B 
    dNNNdx = np.zeros(m,dtype=np.float64)            # basis fct derivatives
    dNNNdy = np.zeros(m,dtype=np.float64)            # basis fct derivatives

    jcob=hx*hy/4
    jcbi=np.zeros((ndim,ndim),dtype=np.float64)
    jcbi[0,0]=2/hx
    jcbi[1,1]=2/hy

    for iel in range (0,nel):

        a_el=np.zeros((m,m),dtype=np.float64)
        b_el=np.zeros(m,dtype=np.float64)

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNN=NN(rq,sq,order)
                dNNNdr=dNNdr(rq,sq,order)
                dNNNds=dNNds(rq,sq,order)

                # calculate jacobian matrix
                #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                #for k in range(0,m):
                #    jcb[0,0]+=dNNNdr[k]*x[icon[k,iel]]
                #    jcb[0,1]+=dNNNdr[k]*y[icon[k,iel]]
                #    jcb[1,0]+=dNNNds[k]*x[icon[k,iel]]
                #    jcb[1,1]+=dNNNds[k]*y[icon[k,iel]]
                #end for
                #jcob=np.linalg.det(jcb)
                #jcbi=np.linalg.inv(jcb)

                # compute rho, dNdx & dNdy at q point
                rhoq=0.
                for k in range(0,m):
                    dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
                    dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]
                    B_mat[0,k]=dNNNdx[k]
                    B_mat[1,k]=dNNNdy[k]
                    rhoq+=NNN[k]*rho[icon[k,iel]]
                #end for

                a_el+=B_mat.T.dot(B_mat)*weightq*jcob*permeability[iel]/eta_f
                b_el+=(dNNNdx*gx+dNNNdy*gy)*permeability[iel]/eta_f*rhoq*weightq*jcob

            #end for
        #end for

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon[k1,iel]
            if bc_fixP[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valP[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               #end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valP[m1]
            #end for
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

    print("build FE matrix : %.3f s" % (timing.time() - start))

    ###########################################################################
    # solve system
    ###########################################################################
    start = timing.time()

    p=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    pstats_file.write("%6e %6e %6e\n" % (time,np.min(p),np.max(p))) ; pstats_file.flush()

    print("solve for p: %.3f s" % (timing.time() - start))

    ###########################################################################
    # compure pressure average
    # this is very rough (i.e. ~1 point quadrature)
    ###########################################################################
    #start = timing.time()

    #pavrg=0
    #for iel in range(0,nel):
    #    pavrg+=np.sum(p[icon[:,iel]])/m*hx*hy
    #pavrg/=(Lx*Ly)

    #print("     -> <p>=",pavrg)

    #if experiment==3:
    #   p-=pavrg

    #print("normalise p: %.3f s" % (timing.time() - start))

    ###########################################################################
    # compute pressure and temperature gradients
    # gradients are computed for each element at each node, and added to a 
    # nodal field, which is later averaged. 
    ###########################################################################
    start = timing.time()
    
    dpdx_nodal=np.zeros(N,dtype=np.float64)  
    dpdy_nodal=np.zeros(N,dtype=np.float64)  
    dTdx_nodal=np.zeros(N,dtype=np.float64)  
    dTdy_nodal=np.zeros(N,dtype=np.float64)  
    count =np.zeros(N,dtype=np.int32)  

    for iel in range(0,nel):
        for i in range(0,m):
            rq=rnodes[i]
            sq=snodes[i]
            dNNNdr[0:m]=dNNdr(rq,sq,order)
            dNNNds[0:m]=dNNds(rq,sq,order)
            #jcb=np.zeros((ndim,ndim),dtype=np.float64)
            #for k in range(0,m):
            #    jcb[0,0]+=dNNNdr[k]*x[icon[k,iel]]
            #    jcb[0,1]+=dNNNdr[k]*y[icon[k,iel]]
            #    jcb[1,0]+=dNNNds[k]*x[icon[k,iel]]
            #    jcb[1,1]+=dNNNds[k]*y[icon[k,iel]]
            #end for
            #jcbi=np.linalg.inv(jcb)
            for k in range(0,m):
                dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
                dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]
            #end for
            dpdx=0.
            dpdy=0.
            dTdx=0.
            dTdy=0.
            for k in range(0,m):
                dTdx+=dNNNdx[k]*T[icon[k,iel]]
                dTdy+=dNNNdy[k]*T[icon[k,iel]]
                dpdx+=dNNNdx[k]*p[icon[k,iel]]
                dpdy+=dNNNdy[k]*p[icon[k,iel]]
            #end for
            inode=icon[i,iel]
            dpdx_nodal[inode]+=dpdx
            dpdy_nodal[inode]+=dpdy
            dTdx_nodal[inode]+=dTdx
            dTdy_nodal[inode]+=dTdy
            count[inode]+=1
        #end for
    #end for
    
    dTdx_nodal/=count
    dTdy_nodal/=count
    dpdx_nodal/=count
    dpdy_nodal/=count

    print("     -> dpdx_nodal (m,M) %e %e " %(np.min(dpdx_nodal),np.max(dpdx_nodal)))
    print("     -> dpdy_nodal (m,M) %e %e " %(np.min(dpdy_nodal),np.max(dpdy_nodal)))
    print("     -> dTdx_nodal (m,M) %e %e " %(np.min(dTdx_nodal),np.max(dTdx_nodal)))
    print("     -> dTdy_nodal (m,M) %e %e " %(np.min(dTdy_nodal),np.max(dTdy_nodal)))

    print("compute nodal p,T gradients: %.3f s" % (timing.time() - start))

    ###########################################################################
    # compute velocity field
    # this approach has been abandonned bc Ks can be spatially variable
    ###########################################################################
    #start = timing.time()
    #uu=np.zeros(N,dtype=np.float64)   # x-component velocity
    #vv=np.zeros(N,dtype=np.float64)   # y-component velocity
    #uu[:]=-K_s/eta_f*(dpdx_nodal[:]-rho[:]*gx)
    #vv[:]=-K_s/eta_f*(dpdy_nodal[:]-rho[:]*gy)
    #ustats_file.write("%6e %6e %6e\n" % (time,np.min(uu),np.max(uu))) ; ustats_file.flush()
    #vstats_file.write("%6e %6e %6e\n" % (time,np.min(vv),np.max(vv))) ; vstats_file.flush()
    #if debug: np.savetxt('velocity.ascii',np.array([x,y,uu,vv]).T,header='# x,y,u,v')
    #print("     -> u (m,M) %.6e %.6e " %(np.min(u),np.max(u)))
    #print("     -> v (m,M) %.6e %.6e " %(np.min(v),np.max(v)))
    #print("compute velocity: %.3f s" % (timing.time() - start))

    ###########################################################################
    # compute velocity field
    ###########################################################################
    start = timing.time()

    AA_mat=lil_matrix((NfemP,NfemP),dtype=np.float64) # FE matrix 
    rhsx=np.zeros(NfemP,dtype=np.float64)             # FE rhs 
    rhsy=np.zeros(NfemP,dtype=np.float64)             # FE rhs 
    N_mat=np.zeros((m,1),dtype=np.float64)            # shape functions vector
    
    jcob=hx*hy/4
    jcbi=np.zeros((ndim,ndim),dtype=np.float64)
    jcbi[0,0]=2/hx
    jcbi[1,1]=2/hy

    for iel in range (0,nel):
        a_el=np.zeros((m,m),dtype=np.float64)
        bx_el=np.zeros(m,dtype=np.float64)
        by_el=np.zeros(m,dtype=np.float64)

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_mat[0:m,0]=NN(rq,sq,order)
                dNNNdr=dNNdr(rq,sq,order)
                dNNNds=dNNds(rq,sq,order)

                # calculate jacobian matrix
                #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                #for k in range(0,m):
                #    jcb[0,0]+=dNNNdr[k]*x[icon[k,iel]]
                #    jcb[0,1]+=dNNNdr[k]*y[icon[k,iel]]
                #    jcb[1,0]+=dNNNds[k]*x[icon[k,iel]]
                #    jcb[1,1]+=dNNNds[k]*y[icon[k,iel]]
                #end for
                #jcob=np.linalg.det(jcb)
                #jcbi=np.linalg.inv(jcb)

                # compute rho, dNdx & dNdy at q point
                dpdxq=0.
                dpdyq=0.
                rhoq=0.
                for k in range(0,m):
                    dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
                    dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]
                    rhoq+=N_mat[k,0]*rho[icon[k,iel]]
                    dpdxq+=dNNNdx[k]*p[icon[k,iel]]
                    dpdyq+=dNNNdy[k]*p[icon[k,iel]]
                #end for

                a_el+=N_mat.dot(N_mat.T)*weightq*jcob

                #print(permeability[iel],eta_f,dpdxq,rhoq,gx,weightq,jcob)

                bx_el-=N_mat[:,0]*permeability[iel]/eta_f*(dpdxq-rhoq*gx)*weightq*jcob
                by_el-=N_mat[:,0]*permeability[iel]/eta_f*(dpdyq-rhoq*gy)*weightq*jcob

            #end for
        #end for

        # assemble matrix AA_mat and right hand side rhs
        for k1 in range(0,m):
            m1=icon[k1,iel]
            for k2 in range(0,m):
                m2=icon[k2,iel]
                AA_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhsx[m1]+=bx_el[k1]
            rhsy[m1]+=by_el[k1]
        #end for

    #end for iel

    print("build FE matrix : %.3f s" % (timing.time() - start))

    ###########################################################################
    start = timing.time()

    uu=sps.linalg.spsolve(sps.csr_matrix(AA_mat),rhsx)
    vv=sps.linalg.spsolve(sps.csr_matrix(AA_mat),rhsy)

    print("     -> u (m,M) %e %e " %(np.min(uu),np.max(uu)))
    print("     -> v (m,M) %e %e " %(np.min(vv),np.max(vv)))

    ustats_file.write("%6e %6e %6e\n" % (time,np.min(uu),np.max(uu))) ; ustats_file.flush()
    vstats_file.write("%6e %6e %6e\n" % (time,np.min(vv),np.max(vv))) ; vstats_file.flush()

    if debug: np.savetxt('velocity.ascii',np.array([x,y,uu,vv]).T,header='# x,y,u,v')

    print("solve for u,v: %.3f s" % (timing.time() - start))

    ###########################################################################
    # compute vrms and errors
    ###########################################################################
    start = timing.time()

    vrms_th=np.sqrt(469887./35./45.)
    pavrg_th=71./6.

    errv=0
    errp=0
    vrms=0.
    pavrg=0
    for iel in range(0,nel):
        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNN=NN(rq,sq,order)
                xq=0.
                yq=0.
                uq=0.
                vq=0.
                pq=0.
                for k in range(0,m):
                    xq+=NNN[k]*x[icon[k,iel]]
                    yq+=NNN[k]*y[icon[k,iel]]
                    uq+=NNN[k]*uu[icon[k,iel]]
                    vq+=NNN[k]*vv[icon[k,iel]]
                    pq+=NNN[k]*p[icon[k,iel]]
                uthq,vthq,pthq,tmp=solution(xq,yq)
                errv+=(uq-uthq)**2*jcob*weightq+\
                      (vq-vthq)**2*jcob*weightq
                errp+=(pq-pthq)**2*jcob*weightq
                vrms+=(uq**2+vq**2)*jcob*weightq
                pavrg+=pq*jcob*weightq 
           #end for
       #end for
    #end for

    errv=np.sqrt(errv)
    errp=np.sqrt(errp)
    pavrg/=(Lx*Ly)
    vrms=np.sqrt(vrms/Lx/Ly)
    
    print('     -> vrms=',vrms,'vrms_th=',vrms_th,'nelx=',nelx)
    print('     -> errv=',errv,'errp=',errp,'nelx=',nelx)
    print('     -> pavrg=',pavrg,'pavrg_th=',pavrg_th,'nelx=',nelx)

    if experiment==3:
       p-=pavrg

    print("compute errors & vrms: %.3f s" % (timing.time() - start))

    ###########################################################################
    # export measurements for benchmarking
    ###########################################################################
    start = timing.time()

    if istep%every==0:
       hfile=open('measurements_hline_{:04d}.ascii'.format(istep),"w")
       vfile=open('measurements_vline_{:04d}.ascii'.format(istep),"w")

       for i in range(0,N):
           if y[i]/Ly<eps:
              hfile.write("%e %e %e %e %e\n" % (x[i],T[i]-TKelvin,uu[i],vv[i],p[i])) ; hfile.flush()
           if abs(x[i]-Lx/2)/Lx<eps:
              vfile.write("%e %e %e %e %e\n" % (y[i],T[i]-TKelvin,uu[i],vv[i],p[i])) ; vfile.flush()
           if abs(x[i]-Lx/2)/Lx<eps and abs(y[i]-Ly/2)/Ly<eps:
              meascenter_file.write("%e %e %e %e %e\n" % (time/year,T[i]-TKelvin,uu[i],vv[i],p[i]))
              meascenter_file.flush()

       hfile.close()
       vfile.close()

    print("export measurements: %.3f s" % (timing.time() - start))

    ###########################################################################
    # compute timestep value
    # if CFL_nb is negative it means we do not use it
    # dt1,dt2 are timestep values due to CFL condition and diffusion time
    # note that the timestep is limited to dt_max no matter what
    ###########################################################################
    start = timing.time()

    if nstep>1:

       dt1=abs(CFL_nb)*hx/np.max(np.sqrt(uu**2+vv**2))
       if solve_T and kappa>0:
          dt2=abs(CFL_nb)*hx**2/kappa
       else:
          dt2=1e50

       if CFL_nb>0:
          print('     using CFL condition timestep')
          dt=np.min([dt1,dt2])
          dt=min(dt,dtmax)

       print('     -> dt1 = %.6f (year)' %(dt1/year))
       print('     -> dt2 = %.6f (year)' %(dt2/year))
       print('     -> dt  = %.6f (year)' %(dt/year))

       time+=dt
       print('     -> time= %.6f; tfinal= %.6f (year)' %(time/year,tfinal/year))

       dt_file.write("%e %e %e %e %e\n" % (time,dt/year,dt1/year,dt2/year,CFL_nb))
       dt_file.flush()

    else:

       dt=0.

    print("compute time step: %.3f s" % (timing.time() - start))

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start = timing.time()

    if solve_T:

       AAA_mat=lil_matrix((NfemT,NfemT),dtype=np.float64) # FE matrix 
       rhs=np.zeros(NfemT,dtype=np.float64)             # FE rhs 
       B_mat=np.zeros((2,m),dtype=np.float64)           # gradient matrix B 
       N_mat=np.zeros((m,1),dtype=np.float64)           # shape functions vector
       N_mat_supg=np.zeros((m,1),dtype=np.float64)      # shape functions vector
       Tvect=np.zeros(m,dtype=np.float64)               # T vales at nodes of elt

       jcob=hx*hy/4
       jcbi=np.zeros((ndim,ndim),dtype=np.float64)
       jcbi[0,0]=2/hx
       jcbi[1,1]=2/hy

       for iel in range (0,nel):

           a_el=np.zeros((m,m),dtype=np.float64)   # elemental matrix
           b_el=np.zeros(m,dtype=np.float64)       # elemental rhs
           Ka=np.zeros((m,m),dtype=np.float64)     # elemental advection matrix 
           Kd=np.zeros((m,m),dtype=np.float64)     # elemental diffusion matrix 
           MM=np.zeros((m,m),dtype=np.float64)     # elemental mass matrix 
           vel=np.zeros((1,ndim),dtype=np.float64) # velocity at q point

           for k in range(0,m):
               Tvect[k]=T[icon[k,iel]]
           #end for

           for iq in range(0,nqperdim):
               for jq in range(0,nqperdim):
                   rq=qcoords[iq]
                   sq=qcoords[jq]
                   weightq=qweights[iq]*qweights[jq]

                   N_mat[0:m,0]=NN(rq,sq,order)
                   dNNNdr[0:m]=dNNdr(rq,sq,order)
                   dNNNds[0:m]=dNNds(rq,sq,order)

                   # calculate jacobian matrix
                   #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                   #for k in range(0,m):
                   #    jcb[0,0]+=dNNNdr[k]*x[icon[k,iel]]
                   #    jcb[0,1]+=dNNNdr[k]*y[icon[k,iel]]
                   #    jcb[1,0]+=dNNNds[k]*x[icon[k,iel]]
                   #    jcb[1,1]+=dNNNds[k]*y[icon[k,iel]]
                   #end for
                   #jcob=np.linalg.det(jcb)
                   #jcbi=np.linalg.inv(jcb)

                   # compute dNdx & dNdy
                   for k in range(0,m):
                       dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
                       dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]
                       B_mat[0,k]=dNNNdx[k]
                       B_mat[1,k]=dNNNdy[k]
                   #end for

                   #compute velocity at q point
                   dpdxq=0.
                   dpdyq=0.
                   rhoq=0.
                   for k in range(0,m):
                       rhoq+=N_mat[k,0]*rho[icon[k,iel]]
                       dpdxq+=dNNNdx[k]*p[icon[k,iel]]
                       dpdyq+=dNNNdy[k]*p[icon[k,iel]]
                   #end for
                   vel[0,0]=-permeability[iel]/eta_f*(dpdxq-rhoq*gx)
                   vel[0,1]=-permeability[iel]/eta_f*(dpdyq-rhoq*gy)

                   hh=np.sqrt(hx*hy)
                   velnorm=np.sqrt(vel[0,0]**2+vel[0,1]**2)
                   if supg_type==0:
                      tau_supg=0.
                   elif supg_type==1:
                         tau_supg=hh/order/velnorm /2
                   elif supg_type==2: #(simpson book p159)
                         if kappa>0:
                            Pe=velnorm*hh/2/kappa
                            alpha=coth(Pe)-1./Pe
                         else:
                            alpha=1
                         tau_supg=alpha*hh/2/velnorm
                   else:
                      exit("supg_type: wrong value")
                     
                   N_mat_supg=N_mat+tau_supg*np.transpose(vel.dot(B_mat))

                   # compute mass matrix
                   MM+=N_mat_supg.dot(N_mat.T)*weightq*jcob*rho_hcapa_m

                   # compute diffusion matrix
                   Kd+=B_mat.T.dot(B_mat)*weightq*jcob*hcond_m

                   # compute advection matrix
                   Ka+=N_mat_supg.dot(vel.dot(B_mat))*weightq*jcob*rho0_f*hcapa_f

               #end for
           #end for

           #1st order backward euler
           #a_el=MM+ (Ka+Kd)*dt
           #b_el=MM.dot(Tvect)

           #Crank-Nicolson
           a_el=MM+0.5*(Ka+Kd)*dt
           b_el=(MM-0.5*(Ka+Kd)*dt).dot(Tvect)

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
                  #end for
                  a_el[k1,k1]=Aref
                  b_el[k1]=Aref*bc_valT[m1]
               #end for
           #end for

           # assemble matrix A_mat and right hand side rhs
           for k1 in range(0,m):
               m1=icon[k1,iel]
               for k2 in range(0,m):
                   m2=icon[k2,iel]
                   AAA_mat[m1,m2]+=a_el[k1,k2]
               #end for
               rhs[m1]+=b_el[k1]
           #end for

       #end for iel

       print("build FE matrix : %.3f s" % (timing.time() - start))

       ###########################################################################
       # solve system
       ###########################################################################
       start = timing.time()

       T=sps.linalg.spsolve(sps.csr_matrix(AAA_mat),rhs)

       print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

       Tstats_file.write("%6e %6e %6e\n" % (time,np.min(T),np.max(T)))
       Tstats_file.flush()

       print("solve for T: %.3f s" % (timing.time() - start))

       ###########################################################################
       # compute Nusselt number
       # I should technically recompute dTdy here with new T, better accuracy
       ###########################################################################
       start = timing.time()

       if order==1:
          top_flux=0
          for iel in range (0,nel):
              if y[icon[2,iel]]>0.999*Ly:
                 top_flux-=hcond_m*((dTdy_nodal[icon[2,iel]]+dTdy_nodal[icon[3,iel]])/2)*hx

       if order==2:
          top_flux=0
          for iel in range (0,nel):
              if y[icon[7,iel]]>0.999*Ly:
                 top_flux-=hcond_m*((dTdy_nodal[icon[6,iel]]+dTdy_nodal[icon[7,iel]])/2)*hx/2
                 top_flux-=hcond_m*((dTdy_nodal[icon[7,iel]]+dTdy_nodal[icon[8,iel]])/2)*hx/2
       top_flux/=Lx

       Nu=top_flux/((Tbottom-Ttop)/Ly*hcond_m)

       RaNu_file.write("%e %e %e\n" % (time/year,Nu,Ra))
       RaNu_file.flush()

       print('     -> Nusselt number:',Nu,abs(Nu-Nu_mem)/Nu)

       print("compute Nusselt nb: %.3f s" % (timing.time() - start))

    # end if solve_T

    ###########################################################################
    # plot of solution
    ###########################################################################
    start = timing.time()

    if istep%every==0 and visu==1:

       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
       ####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       ####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='permeability' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (permeability[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       ####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity (m/s)' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%15e %15e %3e \n" %(uu[i],vv[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity (normalized)' Format='ascii'> \n")
       for i in range(0,N):
           vnorm=np.sqrt(uu[i]**2+vv[i]**2)
           vtufile.write("%15e %15e %3e \n" %(uu[i]/vnorm,vv[i]/vnorm,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e \n" %rho[i])
       vtufile.write("</DataArray>\n")
       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' Name='T (C)' Format='ascii'> \n")
          for i in range(0,N):
              vtufile.write("%10e \n" %(T[i]-TKelvin))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='dTdx' Format='ascii'> \n")
          for i in range(0,N):
              vtufile.write("%10e \n" %dTdx_nodal[i])
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='dTdy' Format='ascii'> \n")
          for i in range(0,N):
              vtufile.write("%10e \n" %dTdy_nodal[i])
          vtufile.write("</DataArray>\n")
          #--
          if istep==0:
             vtufile.write("<DataArray type='Float32' Name='T init (C)' Format='ascii'> \n")
             for i in range(0,N):
                 vtufile.write("%10e \n" %(T_init[i]-TKelvin))
             vtufile.write("</DataArray>\n")
          #--
       vtufile.write("<DataArray type='Float32' Name='dpdx' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e \n" %dpdx_nodal[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dpdy' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e \n" %dpdy_nodal[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e \n" %p[i])
       vtufile.write("</DataArray>\n")
       #--
       if experiment==5:
          vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
          for i in range(0,N):
              vtufile.write("%10e \n" %pth[i])
          vtufile.write("</DataArray>\n")
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity (th)' Format='ascii'> \n")
          for i in range(0,N):
              vtufile.write("%15e %15e %3e \n" %(uth[i],vth[i],0.))
          vtufile.write("</DataArray>\n")

          vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
          for i in range(0,N):
              vtufile.write("%10e \n" % (p[i]-pth[i]))
          vtufile.write("</DataArray>\n")
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity (error)' Format='ascii'> \n")
          for i in range(0,N):
              vtufile.write("%15e %15e %3e \n" %(uu[i]-uth[i],vv[i]-vth[i],0.))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if periodicx:
          if order==2:
             iel=0
             for j in range(0,nely):
                 for i in range(0,nelx):
                     if i==nelx-1:
                        vtufile.write("%d %d %d %d %d %d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[7,iel],\
                                                                        icon[6,iel],icon[1,iel],icon[4,iel],\
                                                                        icon[7,iel],icon[3,iel],icon[4,iel]))
                     else:
                        vtufile.write("%d %d %d %d %d %d %d %d %d \n" %(icon[0,iel],icon[2,iel],icon[8,iel],\
                                                                        icon[6,iel],icon[1,iel],icon[5,iel],\
                                                                        icon[7,iel],icon[3,iel],icon[4,iel]))
                     iel+= 1
                 #end for
             #end for
          #end if

          if order==1:
             iel=0
             for j in range(0,nely):
                 for i in range(0,nelx):
                     if i==nelx-1:
                        vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[0,iel],icon[2,iel],icon[2,iel]))
                     else:
                        vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[3,iel],icon[2,iel]))
                     iel+= 1
                 #end for
             #end for
          #end if
       else:
          if order==2:
             for iel in range (0,nel):
                 vtufile.write("%d %d %d %d %d %d %d %d %d \n" %(icon[0,iel],icon[2,iel],icon[8,iel],\
                                                                 icon[6,iel],icon[1,iel],icon[5,iel],\
                                                                 icon[7,iel],icon[3,iel],icon[4,iel]))
          if order==1:
             for iel in range (0,nel):
                 vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[3,iel],icon[2,iel]))


       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           if order==2:
              vtufile.write("%d \n" %28)
           if order==1:
              vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to vtu file: %.3f s" % (timing.time() - start))

    ###########################################################################

    if nstep>1 and time>tfinal:
       print("*****tfinal reached*****")
       break

    if nstep>1 and istep%20==0:
       if abs(Nu_mem-Nu)<1e-6: 
          print("*****steady state reached*****")
          break
       Nu_mem=Nu

#end for istep

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
