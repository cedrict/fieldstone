import sys as sys
import numpy as np
import time as clock
import scipy.sparse as sps
from scipy.sparse import lil_matrix
import random
from numpy import linalg 
import solcx 
import solkz 
import solvi 
from bench1 import *
from bench8 import *

###############################################################################

def bx(x,y):
    if bench==1:
       return dpdx_th(x,y)-2*app(x)*b(y) -(a(x)*bpp(y)+cp(x)*dp(y))
    if bench==9:
       return 3*x**2*y**2-y-1

def by(x,y):
    if bench==1:
       return dpdy_th(x,y)-(ap(x)*bp(y)+cpp(x)*d(y)) -2*c(x)*dpp(y) 
    if bench==9:
       return 2*x**3*y+3*x-1

###############################################################################

def velocity_x(x,y):
    if bench==1:
       return a(x)*b(y)
    if bench==4:
       ui,vi,pi=solcx.SolCxSolution(x,y) 
       return ui
    if bench==5:
       ui,vi,pi=solkz.SolKzSolution(x,y) 
       return ui
    if bench==6:
       ui,vi,pi=solvi.solution(x,y) 
       return ui
    if bench==9:
       return x+x**2-2*x*y+x**3-3*x*y**2+x**2*y

def velocity_y(x,y):
    if bench==1:
       return c(x)*d(y)
    if bench==4:
       ui,vi,pi=solcx.SolCxSolution(x,y) 
       return vi
    if bench==5:
       ui,vi,pi=solkz.SolKzSolution(x,y) 
       return vi
    if bench==6:
       ui,vi,pi=solvi.solution(x,y) 
       return vi
    if bench==9:
       return -y-2*x*y+y**2-3*x**2*y+y**3-x*y**2

def pressure(x,y):
    if bench==1:
       return x*(1-x)*(1-2*y)
    if bench==4:
       ui,vi,pi=solcx.SolCxSolution(x,y) 
       return pi
    if bench==5:
       ui,vi,pi=solkz.SolKzSolution(x,y) 
       return pi
    if bench==6:
       ui,vi,pi=solvi.solution(x,y) 
       return pi
    if bench==9:
       return x*y+x+y+x**3*y**2-4/3

###############################################################################

def Bubble(r,s):
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)
    elif bubble==2:
       return (1-r**2)*(1-s**2)*(1+beta*(r+s))
    else:
       return (1-r**2)*(1-s**2)

def dBdr(r,s):
    if bubble==1:
       return (1-s**2)*(1-s)*(-1-2*r+3*r**2)
    elif bubble==2:
       return (s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))
    else:
       return (-2*r)*(1-s**2)

def dBds(r,s):
    if bubble==1:
       return (1-r**2)*(1-r)*(-1-2*s+3*s**2) 
    elif bubble==2:
       return (r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)
    else:
       return (1-r**2)*(-2*s)

###############################################################################

def basis_functions_V(r,s):
    N0= 0.25*(1-r)*(1-s) - 0.25*Bubble(r,s)
    N1= 0.25*(1+r)*(1-s) - 0.25*Bubble(r,s)
    N2= 0.25*(1+r)*(1+s) - 0.25*Bubble(r,s)
    N3= 0.25*(1-r)*(1+s) - 0.25*Bubble(r,s)
    N4= Bubble(r,s)
    return np.array([N0,N1,N2,N3,N4],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-s) -0.25*dBdr(r,s)
    dNdr1=+0.25*(1.-s) -0.25*dBdr(r,s)
    dNdr2=+0.25*(1.+s) -0.25*dBdr(r,s)
    dNdr3=-0.25*(1.+s) -0.25*dBdr(r,s)
    dNdr4=dBdr(r,s) 
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-r) -0.25*dBds(r,s)
    dNds1=-0.25*(1.+r) -0.25*dBds(r,s)
    dNds2=+0.25*(1.+r) -0.25*dBds(r,s)
    dNds3=+0.25*(1.-r) -0.25*dBds(r,s)
    dNds4=dBds(r,s) 
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4],dtype=np.float64)

def basis_functions_P(r,s):
    N0= 0.25*(1-r)*(1-s)
    N1= 0.25*(1+r)*(1-s)
    N2= 0.25*(1+r)*(1+s)
    N3= 0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

def eta(x,y):
    if bench==1:
       val=1.
    if bench==2 or bench==3:
       if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
          val=eta2
       else:
          val=eta1
    if bench==4:
       if x<0.5:
          val=1.
       else:
          val=1.e6
    if bench==5:
       val= np.exp(13.8155*y) 
    if bench==6:
       if (np.sqrt(x*x+y*y) < 0.2):
          val=1e3
       else:
          val=1.
    if bench==7:
       if ((x-0.5)**2+(y-0.5)**2 < 0.123456789**2):
          val=1000.
       else:
          val=1.
    if bench==8:
       if y>256e3+amplitude*np.cos(2*np.pi*x/llambda):
          val=eta1
       else:
          val=eta2
    if bench==9:
       val=1
    if bench==10:
       if y>600e3:
          val=1e23
       else:
          val=1e21
    return val

###############################################################################

def rho(x,y):
    if bench==2:
       if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
          val=rho2 
       else:
          val=rho1 
    if bench==3:
       if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
          val=rho2-rho1
       else:
          val=rho1-rho1
    if bench==4:
       val=np.sin(np.pi*y)*np.cos(np.pi*x)
    if bench==5:
       val=np.sin(2.*y)*np.cos(3.*np.pi*x)
    if bench==6:
       val=0.
    if bench==7:
       if ((x-0.5)**2+(y-0.5)**2 < 0.123456789**2):
          val=1.01
       else:
          val=1.
    if bench==8:
       if y>256e3+amplitude*np.cos(2*np.pi*x/llambda):
          val=3300
       else:
          val=3000
    if bench==10:
       val=3300
    return val

###############################################################################
###############################################################################

cm=0.01
year=365.25*24*3600

ndim=2
ndof_V=2
m_V=5
m_P=4

# bench=1 : mms (lami17)
# bench=2 : block full density
# bench=3 : block reduced density
# bench=4 : solcx
# bench=5 : solkz
# bench=6 : solvi
# bench=7 : Stokes sphere
# bench=8 : RT-instability
# bench=9 : mms (lami17)
# bench=10: free surf. crsg12

bench=1

if bench==1 or bench==4 or bench==5 or bench==6 or bench==7 or bench==9:
   Lx=1
   Ly=1
if bench==2 or bench==3 or bench==8:
   Lx=512e3
   Ly=512e3
if bench==10:
   Lx=2800e3
   Ly=700e3

bubble=1

debug=False

if int(len(sys.argv) == 9):
   nelx=int(sys.argv[1])
   nely=int(sys.argv[2])
   visu=int(sys.argv[3])
   drho=float(sys.argv[4])
   eta1=10.**(float(sys.argv[5]))
   eta2=10.**(float(sys.argv[6]))
   nq_per_dim=int(sys.argv[7])
   beta=float(sys.argv[8])
else:
   nelx = 32
   nely = nelx
   visu = 1
   drho = 8
   eta1 = 1e21
   eta2 = 1e22
   nq_per_dim = 2
   beta = 0.25

compute_eigenvalues=False

nel=nelx*nely
nn_V=(nelx+1)*(nely+1)+nel
nn_P=(nelx+1)*(nely+1)
Nfem_V=nn_V*ndof_V
Nfem_P=nn_P
Nfem=Nfem_V+Nfem_P
hx=Lx/nelx
hy=Ly/nely

print('bench=',bench)
print('nelx =',nelx)
print('nely =',nely)
print('nel  =',nel)
print('nn_V =',nn_V)
print('nn_P =',nn_P)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print('bubble=',bubble)
print('beta=',beta)

nq_per_dim=2

if nq_per_dim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nq_per_dim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nq_per_dim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

if nq_per_dim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

if nq_per_dim==6:
   qcoords=[-0.932469514203152,\
            -0.661209386466265,\
            -0.238619186083197,\
            +0.238619186083197,\
            +0.661209386466265,\
            +0.932469514203152]
   qweights=[0.171324492379170,\
             0.360761573048139,\
             0.467913934572691,\
             0.467913934572691,\
             0.360761573048139,\
             0.171324492379170]


eps=1e-8

if bench==1 or bench==9:
   eta_ref=1.
   pnormalise=True
if bench==2 or bench==3:
   gy=-10.
   rho1=3200.
   rho2=rho1+drho
   eta_ref=1e21      # scaling of G blocks
   xc_block=256e3
   yc_block=384e3
   d_block=64e3
   print('rho1=',rho1)
   print('rho2=',rho2)
   print('eta1=',eta1)
   print('eta2=',eta2)
   pnormalise=True
if bench==4 or bench==5 or bench==6:
   eta_ref=1.
   gy=1
   pnormalise=True
if bench==7:
   eta_ref=1.
   gy=-1
   pnormalise=True
if bench==8:
   llambda=256e3
   amplitude=2000
   eta_ref=1e21      # scaling of G blocks
   gy=-10
   phi1=2.*np.pi*(Ly/2.)/llambda
   phi2=2.*np.pi*(Ly/2.)/llambda
   pnormalise=True
if bench==10:
   gy=-10
   eta_ref=1e22
   amplitude=7e3
   pnormalise=False

sparse=True

xi=0.0 # controls level of mesh randomness (between 0 and 0.5 max)

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)

counter=0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_V[counter]=i*hx
        y_V[counter]=j*hy
        counter += 1

for j in range(0,nely):
    for i in range(0,nelx):
        x_V[counter]=i*hx+1/2.*hx
        y_V[counter]=j*hy+1/2.*hy
        counter += 1

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("setup grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start = clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]= i + j * (nelx + 1)
        icon_V[1,counter]= i + 1 + j * (nelx + 1)
        icon_V[2,counter]= i + 1 + (j + 1) * (nelx + 1)
        icon_V[3,counter]= i + (j + 1) * (nelx + 1)
        icon_V[4,counter]= (nelx+1)*(nely+1)+counter
        counter+=1

print("make icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# add random noise to node positions
# we make sure bubble node is in the middle
###############################################################################
start = clock.time()

for i in range(0,nn_V):
    if x_V[i]>0 and x_V[i]<Lx and y_V[i]>0 and y_V[i]<Ly:
       x_V[i]+=random.uniform(-1.,+1)*hx*xi
       y_V[i]+=random.uniform(-1.,+1)*hy*xi
    #end if
#end for

for iel in range(0,nel):
    x_V[icon_V[4,iel]]=0.25*x_V[icon_V[0,iel]]+\
                      +0.25*x_V[icon_V[1,iel]]+\
                      +0.25*x_V[icon_V[2,iel]]+\
                      +0.25*x_V[icon_V[3,iel]]
    y_V[icon_V[4,iel]]=0.25*y_V[icon_V[0,iel]]+\
                      +0.25*y_V[icon_V[1,iel]]+\
                      +0.25*y_V[icon_V[2,iel]]+\
                      +0.25*y_V[icon_V[3,iel]]

print("randomize mesh: %.3f s" % (clock.time()-start))

###############################################################################
# add sine perturbation for RT-instability
###############################################################################

if bench==8: 
   for i in range(0,nn_V):
       if abs(y_V[i]-Ly/2.)/Ly<eps:
          y_V[i]+=amplitude*np.cos(2*np.pi*x_V[i]/llambda)

   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           k=j*(nelx+1)+i
           ya=256e3+amplitude*np.cos(2*np.pi*x_V[k]/llambda)
           if j<(nely+1)/2:
              dy=ya/(nely/2)
              y_V[k]=j*dy
           else:
              dy=(Ly-ya)/(nely/2)
              y_V[k]=ya+(j-nely/2)*dy

   for iel in range(0,nel):
       x_V[icon_V[4,iel]]=0.25*x_V[icon_V[0,iel]]+\
                         +0.25*x_V[icon_V[1,iel]]+\
                         +0.25*x_V[icon_V[2,iel]]+\
                         +0.25*x_V[icon_V[3,iel]]
       y_V[icon_V[4,iel]]=0.25*y_V[icon_V[0,iel]]+\
                         +0.25*y_V[icon_V[1,iel]]+\
                         +0.25*y_V[icon_V[2,iel]]+\
                         +0.25*y_V[icon_V[3,iel]]

###############################################################################
# add sine perturbation for free surface benchmark 
###############################################################################

if bench==10:
   for i in range(0,nn_V):
       if abs(y_V[i]-Ly)/Ly<eps:
          y_V[i]+=amplitude*np.cos(2*np.pi*x_V[i]/Lx)

   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           k=j*(nelx+1)+i
           ymax=Ly+amplitude*np.cos(2*np.pi*x_V[k]/Lx)-600e3
           dy=ymax/10
           if y_V[k]>600e3:
              y_V[k]=600e3+(j-60)*dy

   for iel in range(0,nel):
       x_V[icon_V[4,iel]]=0.25*x_V[icon_V[0,iel]]+\
                         +0.25*x_V[icon_V[1,iel]]+\
                         +0.25*x_V[icon_V[2,iel]]+\
                         +0.25*x_V[icon_V[3,iel]]
       y_V[icon_V[4,iel]]=0.25*y_V[icon_V[0,iel]]+\
                         +0.25*y_V[icon_V[1,iel]]+\
                         +0.25*y_V[icon_V[2,iel]]+\
                         +0.25*y_V[icon_V[3,iel]]

#################################################################
###############################################################################
# build pressure grid and icon_P 
###############################################################################
#################################################################
start = clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)     # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64)     # y coordinates
icon_P=np.zeros((m_P,nel),dtype=np.int32)

x_P[0:nn_P]=x_V[0:nn_P]
y_P[0:nn_P]=y_V[0:nn_P]

icon_P[0:m_P,0:nel]=icon_V[0:m_P,0:nel]

if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
###############################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
    if area[iel]<0: 
       for k in range(0,mV):
           print (x_V[icon_V[k,iel]],y_V[icon_V[k,iel]])
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.6f " %(area.sum()))
print("     -> total area anal %.6f " %(Lx*Ly))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

if bench==1:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V]   = 0.
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V]   = 0.
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V]   = 0.
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
       if y_V[i]/Ly>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V]   = 0.
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.

elif (bench==6 or bench==9):
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = velocity_x(x_V[i],y_V[i])
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = velocity_y(x_V[i],y_V[i])
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = velocity_x(x_V[i],y_V[i])
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = velocity_y(x_V[i],y_V[i])
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = velocity_x(x_V[i],y_V[i])
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = velocity_y(x_V[i],y_V[i])
       if y_V[i]/Ly>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = velocity_x(x_V[i],y_V[i])
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = velocity_y(x_V[i],y_V[i])

elif bench==8:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V  ] = 0 
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V  ] = 0
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0 
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
       if y_V[i]/Ly>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0 
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0

elif bench==10:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V]   = 0.
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V]   = 0.
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V]   = 0.
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.

else: # free slip 
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
       if y_V[i]/Ly>(1-eps):
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

if sparse:
   if pnormalise:
      A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
   else:
      A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat = np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
   G_mat = np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT

constr= np.zeros(Nfem_P,dtype=np.float64)  # constraint matrix/vector
f_rhs= np.zeros(Nfem_V,dtype=np.float64)  # right hand side f 
h_rhs= np.zeros(Nfem_P,dtype=np.float64)  # right hand side h 
B= np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
N_mat= np.zeros((3,m_P),dtype=np.float64) # matrix  

C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)
    NNNNP= np.zeros(m_P,dtype=np.float64)           # int of shape functions P

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]



            K_el+=B.T.dot(C.dot(B))*eta(xq,yq)*JxWq

            if bench==1 or bench==9:
               for i in range(0,m_V):
                   f_el[ndof_V*i  ]+=N_V[i]*bx(xq,yq)*JxWq
                   f_el[ndof_V*i+1]+=N_V[i]*by(xq,yq)*JxWq
            else:
               for i in range(0,m_V):
                   f_el[ndof_V*i+1]+=N_V[i]*rho(xq,yq)*gy*JxWq

            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]
                N_mat[2,i]=0.

            G_el-=B.T.dot(N_mat)*JxWq

            NNNNP[:]+=N_P[:]*JxWq

        # end for jq
    # end for iq

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

    G_el*=eta_ref/Ly
    h_el*=eta_ref/Ly

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
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                if sparse:
                   A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                   A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]
        if sparse and pnormalise:
           A_sparse[Nfem,Nfem_V+m2]=constr[m2]
           A_sparse[Nfem_V+m2,Nfem]=constr[m2]

if not sparse:
   print("     -> K_mat (m,M) %.4e %.4e " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4e %.4e " %(np.min(G_mat),np.max(G_mat)))

print("build FE matrix: %.3fs - %d elts" % (clock.time()-start, nel))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start=clock.time()

if not sparse:
   if pnormalise:
      a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
      rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
      a_mat[0:Nfem_V,0:Nfem_V]=K_mat
      a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
      a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
      a_mat[Nfem,Nfem_V:Nfem]=constr
      a_mat[Nfem_V:Nfem,Nfem]=constr
   else:
      a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
      rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
      a_mat[0:Nfem_V,0:Nfem_V]=K_mat
      a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
      a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
   #end if
else:
   if pnormalise:
      rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   else:
      rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
#else:

rhs[0:Nfem_V]=f_rhs
rhs[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time() - start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (clock.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

if debug:
   np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
   np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# look at eigenvalues of K_mat
###############################################################################

if compute_eigenvalues:
   eigvals = np.linalg.eigvals(K_mat)
   #np.savetxt('eigenvals.ascii',np.array([eigvals.real,eigvals.imag]).T)
   #eigvals, eigvecs = linalg.eig(K_mat)
   print('eig.vals:',nel,eigvals.min(),eigvals.max(),linalg.cond(K_mat))

###############################################################################
# measure vel at center of block
###############################################################################

if bench==2 or bench==3:
   for i in range(0,nn_V):
       if abs(x_V[i]-xc_block)<1 and abs(y_V[i]-yc_block)<1:
          print('vblock=',eta1/eta2,np.abs(v[i])*eta1/drho,u[i]*year,v[i]*year)
   for i in range(0,nn_P):
       if abs(x_P[i]-xc_block)<1 and abs(y_P[i]-yc_block)<1:
          print('pblock=',eta1/eta2,p[i]/drho/np.abs(gy)/128e3)

if bench==2 or bench==3 or bench==7:
   pline_file=open('pline.ascii',"w")
   for i in range(0,nn_P):
       if abs(x_P[i]-Lx/2)<Lx/10000:
          pline_file.write("%10e %10e \n" %(y_P[i],p[i]))
   pline_file.close()
         
if bench==7: print(" pstats %d %.4e %.4e " %(nel,np.min(p),np.max(p)))

if bench==8: print(" RT %.8e %.8e %.8e %.8e" %(np.max(abs(v)),phi1,vy_th(phi1,phi2,3300,3000),eta2))

if bench==10: print("     -> elevation: %.8e" %  y_V[(nelx+1)*(nely+1)-1] )

###############################################################################
# compute vrms 
###############################################################################
start = clock.time()

vrms=0.
for iel in range (0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            vrms+=(uq**2+vq**2)*JxWq
        # end for jq
    # end for iq
# end for iel

vrms=np.sqrt(vrms/(Lx*Ly))

if bench==2 or bench==3 or bench==10:
   vrms/=(cm/year)

print("     -> nel= %6d ; vrms= %.8f ; beta= %4e" %(nel,vrms,beta))

print("compute v_rms : %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

if bench==1 or bench==4 or bench==5 or bench==6 or bench==9:

   error_u=np.zeros(nn_V,dtype=np.float64)
   error_v=np.zeros(nn_V,dtype=np.float64)
   error_p=np.zeros(nn_P,dtype=np.float64)

   for i in range(0,nn_V): 
       error_u[i]=u[i]-velocity_x(x_V[i],y_V[i])
       error_v[i]=v[i]-velocity_y(x_V[i],y_V[i])

   for i in range(0,nn_P): 
       error_p[i]=p[i]-pressure(x_P[i],y_P[i])

   print("compute nodal error for plot: %.3f s" % (clock.time()-start))

###############################################################################
# compute error in L2 norm
###############################################################################
start = clock.time()

if bench==1 or bench==4 or bench==5 or bench==6 or bench==9:

   errv=0.
   errp=0.
   for iel in range (0,nel):
       for iq in range(0,nq_per_dim):
           for jq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               weightq=qweights[iq]*qweights[jq]
               N_V=basis_functions_V(rq,sq)
               N_P=basis_functions_P(rq,sq)
               dNdr_V=basis_functions_V_dr(rq,sq)
               dNds_V=basis_functions_V_ds(rq,sq)
               jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
               jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
               jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
               jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
               jcbi=np.linalg.inv(jcb)
               JxWq=np.linalg.det(jcb)*weightq
               uq=np.dot(N_V,u[icon_V[:,iel]])
               vq=np.dot(N_V,v[icon_V[:,iel]])
               xq=np.dot(N_V,x_V[icon_V[:,iel]])
               yq=np.dot(N_V,y_V[icon_V[:,iel]])
               errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*JxWq
               xq=np.dot(N_P,x_P[icon_P[:,iel]])
               yq=np.dot(N_P,y_P[icon_P[:,iel]])
               pq=np.dot(N_P,p[icon_P[:,iel]])
               errp+=(pq-pressure(xq,yq))**2*JxWq
           #end for
       #end for
   #end for
   errv=np.sqrt(errv)
   errp=np.sqrt(errp)

   print("     -> nel= %6d ; errv= %.10f ; errp= %.10f; beta= %4e" %(nel,errv,errp,beta))

   print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# export various measurements for stokes sphere benchmark 
###############################################################################

vel=np.sqrt(u**2+v**2)
print('benchmark ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      0,0,\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),
      vrms)

###############################################################################
# plot of solution | using in fact only 4 Vnodes and leaving the bubble out. 
###############################################################################
start=clock.time()

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_P,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_P):
        vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%10e \n" %(area[iel]))
    vtufile.write("</DataArray>\n")


    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
    if bench==1 or bench==4 or bench==5 or bench==6 or bench==7 or bench==9:
       for i in range(0,nn_P):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    if bench==2 or bench==3 or bench==8 or bench==10:
       for i in range(0,nn_P):
           vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    for i in range(0,nn_P):
        vtufile.write("%10e \n" %p[i])
    vtufile.write("</DataArray>\n")

    #--
    if bench==1 or bench==4 or bench==5 or bench==6 or bench==9:
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='error vel' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e %10e %10e \n" %(error_u[i],error_v[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='error p' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e \n" %(error_p[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel th' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e %10e %10e \n" %(velocity_x(x_V[i],y_V[i]),velocity_y(x_V[i],y_V[i]),0))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='p th' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e \n" %(pressure(x_V[i],y_V[i])))
       vtufile.write("</DataArray>\n")
    else:
       vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e \n" %(rho(x_V[i],y_V[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e \n" %(eta(x_V[i],y_V[i])))
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],\
                                         icon_V[2,iel],icon_V[3,iel]))
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

    print("export to vtu: %.3f s" % (clock.time()-start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
