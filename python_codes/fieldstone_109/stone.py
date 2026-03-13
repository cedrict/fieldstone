import numpy as np
import time as clock 
import scipy.sparse as sps
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import math
from numba import jit

###############################################################################
# ordering of nodes is a bit weird but it follows the VTK format 25
###############################################################################

@jit(nopython=True)
def basis_functions_V(r,s,t):
    N00= 0.5*r*(r-1) * 0.5*s*(s-1) * 0.5*t*(t-1) # -1,-1,-1 
    N01= 0.5*r*(r+1) * 0.5*s*(s-1) * 0.5*t*(t-1) # +1,-1,-1
    N02= 0.5*r*(r+1) * 0.5*s*(s+1) * 0.5*t*(t-1) # +1, 1,-1
    N03= 0.5*r*(r-1) * 0.5*s*(s+1) * 0.5*t*(t-1) # -1, 1,-1
    N04= 0.5*r*(r-1) * 0.5*s*(s-1) * 0.5*t*(t+1) # -1,-1,+1
    N05= 0.5*r*(r+1) * 0.5*s*(s-1) * 0.5*t*(t+1) # +1,-1,+1
    N06= 0.5*r*(r+1) * 0.5*s*(s+1) * 0.5*t*(t+1) # +1, 1,+1
    N07= 0.5*r*(r-1) * 0.5*s*(s+1) * 0.5*t*(t+1) # -1, 1,+1

    N08= (1.-r**2)   * 0.5*s*(s-1) * 0.5*t*(t-1) #  0,-1,-1 
    N09= 0.5*r*(r+1) * (1.-s**2)   * 0.5*t*(t-1) # +1, 0,-1
    N10= (1.-r**2)   * 0.5*s*(s+1) * 0.5*t*(t-1) #  0,+1,-1
    N11= 0.5*r*(r-1) * (1.-s**2)   * 0.5*t*(t-1) # -1, 0,-1

    N12= (1.-r**2)   * 0.5*s*(s-1) * 0.5*t*(t+1) #  0,-1,+1
    N13= 0.5*r*(r+1) * (1.-s**2)   * 0.5*t*(t+1) # +1, 0,+1
    N14= (1.-r**2)   * 0.5*s*(s+1) * 0.5*t*(t+1) #  0,+1,+1
    N15= 0.5*r*(r-1) * (1.-s**2)   * 0.5*t*(t+1) # -1, 0,+1 

    N16= 0.5*r*(r-1) * 0.5*s*(s-1) * (1.-t**2)   # -1,-1, 0
    N17= 0.5*r*(r+1) * 0.5*s*(s-1) * (1.-t**2)   # +1,-1, 0
    N18= 0.5*r*(r+1) * 0.5*s*(s+1) * (1.-t**2)   # +1,+1, 0
    N19= 0.5*r*(r-1) * 0.5*s*(s+1) * (1.-t**2)   # -1,+1, 0

    N20= (1.-r**2)   * (1.-s**2)   * 0.5*t*(t-1) #  0, 0,-1
    N21= (1.-r**2)   * 0.5*s*(s-1) * (1.-t**2)   #  0,-1, 0
    N22= 0.5*r*(r+1) * (1.-s**2)   * (1.-t**2)   # +1, 0, 0
    N23= (1.-r**2)   * 0.5*s*(s+1) * (1.-t**2)   #  0,+1, 0
    N24= 0.5*r*(r-1) * (1.-s**2)   * (1.-t**2)   # -1, 0, 0
    N25= (1.-r**2)   * (1.-s**2)   * 0.5*t*(t+1) #  0, 0,+1
    N26= (1.-r**2)   * (1.-s**2)   * (1.-t**2)   #  0, 0, 0
    return np.array([N00,N01,N02,N03,N04,N05,N06,N07,N08,
                     N09,N10,N11,N12,N13,N14,N15,N16,N17,
                     N18,N19,N20,N21,N22,N23,N24,N25,N26],dtype=np.float64)

@jit(nopython=True)
def basis_functions_V_dr(r,s,t):
    dNdr00= 0.5*(2*r-1) * 0.5*s*(s-1.) * 0.5*t*(t-1.) 
    dNdr01= 0.5*(2*r+1) * 0.5*s*(s-1.) * 0.5*t*(t-1.) 
    dNdr02= 0.5*(2*r+1) * 0.5*s*(s+1.) * 0.5*t*(t-1.) 
    dNdr03= 0.5*(2*r-1) * 0.5*s*(s+1.) * 0.5*t*(t-1.) 
    dNdr04= 0.5*(2*r-1) * 0.5*s*(s-1.) * 0.5*t*(t+1.) 
    dNdr05= 0.5*(2*r+1) * 0.5*s*(s-1.) * 0.5*t*(t+1.) 
    dNdr06= 0.5*(2*r+1) * 0.5*s*(s+1.) * 0.5*t*(t+1.) 
    dNdr07= 0.5*(2*r-1) * 0.5*s*(s+1.) * 0.5*t*(t+1.) 
    dNdr08= (-2*r)      * 0.5*s*(s-1.) * 0.5*t*(t-1.) 
    dNdr09= 0.5*(2*r+1) * (1.-s**2)    * 0.5*t*(t-1.) 
    dNdr10= (-2*r)      * 0.5*s*(s+1.) * 0.5*t*(t-1.) 
    dNdr11= 0.5*(2*r-1) * (1.-s**2)    * 0.5*t*(t-1.) 
    dNdr12= (-2*r)      * 0.5*s*(s-1.) * 0.5*t*(t+1.) 
    dNdr13= 0.5*(2*r+1) * (1.-s**2)    * 0.5*t*(t+1.) 
    dNdr14= (-2*r)      * 0.5*s*(s+1.) * 0.5*t*(t+1.) 
    dNdr15= 0.5*(2*r-1) * (1.-s**2)    * 0.5*t*(t+1.) 
    dNdr16= 0.5*(2*r-1) * 0.5*s*(s-1.) * (1.-t**2) 
    dNdr17= 0.5*(2*r+1) * 0.5*s*(s-1.) * (1.-t**2) 
    dNdr18= 0.5*(2*r+1) * 0.5*s*(s+1.) * (1.-t**2) 
    dNdr19= 0.5*(2*r-1) * 0.5*s*(s+1.) * (1.-t**2) 
    dNdr20= (-2*r)      * (1.-s**2)    * 0.5*t*(t-1.) 
    dNdr21= (-2*r)      * 0.5*s*(s-1.) * (1.-t**2) 
    dNdr22= 0.5*(2*r+1) * (1.-s**2)    * (1.-t**2) 
    dNdr23= (-2*r)      * 0.5*s*(s+1.) * (1.-t**2) 
    dNdr24= 0.5*(2*r-1) * (1.-s**2)    * (1.-t**2) 
    dNdr25= (-2*r)      * (1.-s**2)    * 0.5*t*(t+1.) 
    dNdr26= (-2*r)      * (1.-s**2)    * (1.-t**2)
    return np.array([dNdr00,dNdr01,dNdr02,dNdr03,dNdr04,dNdr05,
                     dNdr06,dNdr07,dNdr08,dNdr09,dNdr10,dNdr11,
                     dNdr12,dNdr13,dNdr14,dNdr15,dNdr16,dNdr17,
                     dNdr18,dNdr19,dNdr20,dNdr21,dNdr22,dNdr23,
                     dNdr24,dNdr25,dNdr26],dtype=np.float64)

@jit(nopython=True)
def basis_functions_V_ds(r,s,t):
    dNds00= 0.5*r*(r-1.) * 0.5*(2*s-1.) * 0.5*t*(t-1.) 
    dNds01= 0.5*r*(r+1.) * 0.5*(2*s-1.) * 0.5*t*(t-1.) 
    dNds02= 0.5*r*(r+1.) * 0.5*(2*s+1.) * 0.5*t*(t-1.) 
    dNds03= 0.5*r*(r-1.) * 0.5*(2*s+1.) * 0.5*t*(t-1.) 
    dNds04= 0.5*r*(r-1.) * 0.5*(2*s-1.) * 0.5*t*(t+1.) 
    dNds05= 0.5*r*(r+1.) * 0.5*(2*s-1.) * 0.5*t*(t+1.) 
    dNds06= 0.5*r*(r+1.) * 0.5*(2*s+1.) * 0.5*t*(t+1.) 
    dNds07= 0.5*r*(r-1.) * 0.5*(2*s+1.) * 0.5*t*(t+1.) 
    dNds08= (1.-r**2)    * 0.5*(2*s-1.) * 0.5*t*(t-1.) 
    dNds09= 0.5*r*(r+1.) * (-2*s)       * 0.5*t*(t-1.) 
    dNds10= (1.-r**2)    * 0.5*(2*s+1.) * 0.5*t*(t-1.) 
    dNds11= 0.5*r*(r-1.) * (-2*s)       * 0.5*t*(t-1.) 
    dNds12= (1.-r**2)    * 0.5*(2*s-1.) * 0.5*t*(t+1.) 
    dNds13= 0.5*r*(r+1.) * (-2*s)       * 0.5*t*(t+1.) 
    dNds14= (1.-r**2)    * 0.5*(2*s+1.) * 0.5*t*(t+1.) 
    dNds15= 0.5*r*(r-1.) * (-2*s)       * 0.5*t*(t+1.) 
    dNds16= 0.5*r*(r-1.) * 0.5*(2*s-1.) * (1.-t**2) 
    dNds17= 0.5*r*(r+1.) * 0.5*(2*s-1.) * (1.-t**2) 
    dNds18= 0.5*r*(r+1.) * 0.5*(2*s+1.) * (1.-t**2) 
    dNds19= 0.5*r*(r-1.) * 0.5*(2*s+1.) * (1.-t**2) 
    dNds20= (1.-r**2)    * (-2*s)       * 0.5*t*(t-1.) 
    dNds21= (1.-r**2)    * 0.5*(2*s-1.) * (1.-t**2) 
    dNds22= 0.5*r*(r+1.) * (-2*s)       * (1.-t**2) 
    dNds23= (1.-r**2)    * 0.5*(2*s+1.) * (1.-t**2) 
    dNds24= 0.5*r*(r-1.) * (-2*s)       * (1.-t**2) 
    dNds25= (1.-r**2)    * (-2*s)       * 0.5*t*(t+1.) 
    dNds26= (1.-r**2)    * (-2*s)       * (1.-t**2) 
    return np.array([dNds00,dNds01,dNds02,dNds03,dNds04,dNds05,
                     dNds06,dNds07,dNds08,dNds09,dNds10,dNds11,
                     dNds12,dNds13,dNds14,dNds15,dNds16,dNds17,
                     dNds18,dNds19,dNds20,dNds21,dNds22,dNds23,
                     dNds24,dNds25,dNds26],dtype=np.float64)

@jit(nopython=True)
def basis_functions_V_dt(r,s,t):
    dNdt00= 0.5*r*(r-1.) * 0.5*s*(s-1.) * 0.5*(2*t-1.) 
    dNdt01= 0.5*r*(r+1.) * 0.5*s*(s-1.) * 0.5*(2*t-1.) 
    dNdt02= 0.5*r*(r+1.) * 0.5*s*(s+1.) * 0.5*(2*t-1.) 
    dNdt03= 0.5*r*(r-1.) * 0.5*s*(s+1.) * 0.5*(2*t-1.) 
    dNdt04= 0.5*r*(r-1.) * 0.5*s*(s-1.) * 0.5*(2*t+1.) 
    dNdt05= 0.5*r*(r+1.) * 0.5*s*(s-1.) * 0.5*(2*t+1.) 
    dNdt06= 0.5*r*(r+1.) * 0.5*s*(s+1.) * 0.5*(2*t+1.) 
    dNdt07= 0.5*r*(r-1.) * 0.5*s*(s+1.) * 0.5*(2*t+1.) 
    dNdt08= (1.-r**2)    * 0.5*s*(s-1.) * 0.5*(2*t-1.) 
    dNdt09= 0.5*r*(r+1.) * (1.-s**2)    * 0.5*(2*t-1.) 
    dNdt10= (1.-r**2)    * 0.5*s*(s+1.) * 0.5*(2*t-1.) 
    dNdt11= 0.5*r*(r-1.) * (1.-s**2)    * 0.5*(2*t-1.) 
    dNdt12= (1.-r**2)    * 0.5*s*(s-1.) * 0.5*(2*t+1.) 
    dNdt13= 0.5*r*(r+1.) * (1.-s**2)    * 0.5*(2*t+1.) 
    dNdt14= (1.-r**2)    * 0.5*s*(s+1.) * 0.5*(2*t+1.) 
    dNdt15= 0.5*r*(r-1.) * (1.-s**2)    * 0.5*(2*t+1.) 
    dNdt16= 0.5*r*(r-1.) * 0.5*s*(s-1.) * (-2*t) 
    dNdt17= 0.5*r*(r+1.) * 0.5*s*(s-1.) * (-2*t) 
    dNdt18= 0.5*r*(r+1.) * 0.5*s*(s+1.) * (-2*t) 
    dNdt19= 0.5*r*(r-1.) * 0.5*s*(s+1.) * (-2*t) 
    dNdt20= (1.-r**2)    * (1.-s**2)    * 0.5*(2*t-1.) 
    dNdt21= (1.-r**2)    * 0.5*s*(s-1.) * (-2*t) 
    dNdt22= 0.5*r*(r+1.) * (1.-s**2)    * (-2*t) 
    dNdt23= (1.-r**2)    * 0.5*s*(s+1.) * (-2*t) 
    dNdt24= 0.5*r*(r-1.) * (1.-s**2)    * (-2*t) 
    dNdt25= (1.-r**2)    * (1.-s**2)    * 0.5*(2*t+1.) 
    dNdt26= (1.-r**2)    * (1.-s**2)    * (-2*t) 
    return np.array([dNdt00,dNdt01,dNdt02,dNdt03,dNdt04,dNdt05,
                     dNdt06,dNdt07,dNdt08,dNdt09,dNdt10,dNdt11,
                     dNdt12,dNdt13,dNdt14,dNdt15,dNdt16,dNdt17,
                     dNdt18,dNdt19,dNdt20,dNdt21,dNdt22,dNdt23,
                     dNdt24,dNdt25,dNdt26],dtype=np.float64)

@jit(nopython=True)
def basis_functions_P(r,s,t):
    N0=0.125*(1-r)*(1-s)*(1-t)    
    N1=0.125*(1+r)*(1-s)*(1-t)    
    N2=0.125*(1+r)*(1+s)*(1-t)    
    N3=0.125*(1-r)*(1+s)*(1-t)    
    N4=0.125*(1-r)*(1-s)*(1+t)    
    N5=0.125*(1+r)*(1-s)*(1+t)    
    N6=0.125*(1+r)*(1+s)*(1+t)    
    N7=0.125*(1-r)*(1+s)*(1+t)    
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

###############################################################################

@jit(nopython=True)
def eta(x,y,z):
    if (x-Lx/2)**2+(y)**2<200e3**2:
       val=2e21
    else:
       val=2e18
    return val

###############################################################################

eps=1e-8
cm=0.01
year=365.25*24*3600
MPa=1e6

print("*******************************")
print("********** stone 109 **********")
print("*******************************")

ndim=3
ndof_V=3
m_V=27
m_P=8

Lx=1000e3
Ly=500e3
Lz=15e3

nelx=30
nely=15
nelz=5

nnx=2*nelx+1
nny=2*nely+1
nnz=2*nelz+1
nel=nelx*nely*nelz
nn_V=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
nn_P=(nelx+1)*(nely+1)*(nelz+1)
Nfem_V=nn_V*ndof_V
Nfem_P=nn_P
Nfem=Nfem_V+Nfem_P

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

U0=8*cm/year

rho=3000

sparse=True

eta_ref=1e21

Pi2=8*U0*2e18/Lz**2 * Lx /2
print('Pi/2=',Pi2)

debug=False

###############################################################################

nq_per_dim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

###############################################################################

print('nelx  =',nelx)
print('nely  =',nely)
print('nelz  =',nelz)
print('nnx   =',nnx)
print('nny   =',nny)
print('nnz   =',nnz)
print('nel   =',nel)
print('nn_V  =',nn_V)
print('nn_P  =',nn_P)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)
z_V=np.zeros(nn_V,dtype=np.float64)

counter=0    
for i in range(0,2*nelx+1):
    for j in range(0,2*nely+1):
        for k in range(0,2*nelz+1):
            x_V[counter]=i*hx/2.
            y_V[counter]=j*hy/2.
            z_V[counter]=k*hz/2.
            counter += 1
        #end for
    #end for
#end for

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V,z_V]).T,header='# x,y,z')

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0    
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon_V[ 0,counter]=(2*(k)+1)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+0) -1 
            icon_V[ 1,counter]=(2*(k)+1)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+2) -1 
            icon_V[ 2,counter]=(2*(k)+1)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+2) -1 
            icon_V[ 3,counter]=(2*(k)+1)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+0) -1 
            icon_V[ 4,counter]=(2*(k)+3)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+0) -1 
            icon_V[ 5,counter]=(2*(k)+3)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+2) -1 
            icon_V[ 6,counter]=(2*(k)+3)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+2) -1 
            icon_V[ 7,counter]=(2*(k)+3)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+0) -1 
            icon_V[ 8,counter]=(2*(k)+1)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+1) -1 
            icon_V[ 9,counter]=(2*(k)+1)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+2) -1 
            icon_V[10,counter]=(2*(k)+1)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+1) -1 
            icon_V[11,counter]=(2*(k)+1)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+0) -1 
            icon_V[12,counter]=(2*(k)+3)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+1) -1 
            icon_V[13,counter]=(2*(k)+3)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+2) -1 
            icon_V[14,counter]=(2*(k)+3)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+1) -1 
            icon_V[15,counter]=(2*(k)+3)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+0) -1 
            icon_V[16,counter]=(2*(k)+2)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+0) -1 
            icon_V[17,counter]=(2*(k)+2)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+2) -1 
            icon_V[18,counter]=(2*(k)+2)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+2) -1 
            icon_V[19,counter]=(2*(k)+2)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+0) -1 
            icon_V[20,counter]=(2*(k)+1)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+1) -1 
            icon_V[21,counter]=(2*(k)+2)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+1) -1 
            icon_V[22,counter]=(2*(k)+2)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+2) -1 
            icon_V[23,counter]=(2*(k)+2)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+1) -1 
            icon_V[24,counter]=(2*(k)+2)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+0) -1 
            icon_V[25,counter]=(2*(k)+3)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+1) -1 
            icon_V[26,counter]=(2*(k)+2)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+1) -1 
            counter=counter+1   
        #end for
    #end for
#end for

print("setup: connectivity V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid and icon_P 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64) 
y_P=np.zeros(nn_P,dtype=np.float64)  
z_P=np.zeros(nn_P,dtype=np.float64)   

counter=0    
for i in range(0,nelx+1):
    for j in range(0,nely+1):
        for k in range(0,nelz+1):
            x_P[counter]=i*Lx/nelx    
            y_P[counter]=j*Ly/nely
            z_P[counter]=k*Lz/nelz
            counter+=1    
        #end for
    #end for
#end for

icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter = 0 
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon_P[0,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j-1+1)+k
            icon_P[1,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j-1+1)+k
            icon_P[2,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j  +1)+k
            icon_P[3,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j  +1)+k
            icon_P[4,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j-1+1)+k+1
            icon_P[5,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j-1+1)+k+1
            icon_P[6,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j  +1)+k+1
            icon_P[7,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P,z_P]).T,header='# x,y,z')

print("setup: build P grid: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions (no slip top & bottom, free slip sides)
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V+0] = U0*z_V[i]*(Lz-z_V[i])*4/Lz**2
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       bc_fix_V[i*ndof_V+2] = True ; bc_val_V[i*ndof_V+2] = 0.
    if x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V+0] = U0*z_V[i]*(Lz-z_V[i])*4/Lz**2
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       bc_fix_V[i*ndof_V+2] = True ; bc_val_V[i*ndof_V+2] = 0.
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]/Ly>(1-eps):
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if z_V[i]/Lz<eps:
       bc_fix_V[i*ndof_V+0] = True ; bc_val_V[i*ndof_V+0] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       bc_fix_V[i*ndof_V+2] = True ; bc_val_V[i*ndof_V+2] = 0.
    if z_V[i]/Lz>(1-eps):
       bc_fix_V[i*ndof_V+0] = True ; bc_val_V[i*ndof_V+0] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       bc_fix_V[i*ndof_V+2] = True ; bc_val_V[i*ndof_V+2] = 0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# sanity check, computing volume
###############################################################################
start=clock.time()
   
jcb=np.zeros((ndim,ndim),dtype=np.float64)
volume=np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]
                N_V=basis_functions_V(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                JxWq=np.linalg.det(jcb)*weightq
                volume[iel]+=JxWq
            #end for
        #end for
    #end for
#end for

print("     -> volume (m,M) %.6e %.6e " %(np.min(volume),np.max(volume)))
print("     -> cell volume (anal) %.6e" %(hx*hy*hz))
print("     -> total volume (meas) %e " %(volume.sum()))
print("     -> total volume (anal) %e " %(Lx*Ly*Lz))

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

if sparse:
   A_sparse=lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
   G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT

b_fem=np.zeros(Nfem,dtype=np.float64) # right hand side of Ax=b
B=np.zeros((6,ndof_V*m_V),dtype=np.float64)  # gradient matrix B 
N_mat=np.zeros((6,m_P),dtype=np.float64)        # matrix  
C=np.zeros((6,6),dtype=np.float64)
C[0,0]=2. ; C[1,1]=2. ; C[2,2]=2.
C[3,3]=1. ; C[4,4]=1. ; C[5,5]=1.

for iel in range(0,nel):

    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):

                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]

                N_V=basis_functions_V(rq,sq,tq)
                N_P=basis_functions_P(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                zq=np.dot(N_V,z_V[icon_V[:,iel]])

                for i in range(0,m_V):
                    B[0:6,3*i:3*i+3] = [[dNdx_V[i],0.       ,0.       ],
                                        [0.       ,dNdy_V[i],0.       ],
                                        [0.       ,0.       ,dNdz_V[i]],
                                        [dNdy_V[i],dNdx_V[i],0.       ],
                                        [dNdz_V[i],0.       ,dNdx_V[i]],
                                        [0.       ,dNdz_V[i],dNdy_V[i]]]

                K_el+=B.T.dot(C.dot(B))*eta(xq,yq,zq)*JxWq

                #for i in range(0,m_V):
                #    f_el[ndof_V*i+2]+=N_V[i]**rho*gz*JxWq
                
                N_mat[0,:]=N_P
                N_mat[1,:]=N_P
                N_mat[2,:]=N_P

                G_el-=B.T.dot(N_mat)*JxWq

            # end for kq
        # end for jq
    # end for iq

    # impose b.c. 
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
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0

    G_el*=eta_ref/Ly
    h_el*=eta_ref/Ly

    # assemble matrix and right hand side
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
            b_fem[m1]+=f_el[ikk]
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        b_fem[Nfem_V+m2]+=h_el[k2]

print("build FE matrix: %.3fs - %d elts" % (clock.time()-start,nel))

###############################################################################
# assemble K, G, GT, f, h into A and hs
###############################################################################
start=clock.time()

if not sparse:
   A_fem=np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T

print("assemble blocks: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

if sparse:
   A_fem=A_sparse.tocsr()
else:
   A_fem=sps.csr_matrix(A_fem)

if False:
   plt.spy(A_fem, markersize=0.1)
   plt.savefig('matrix.png', bbox_inches='tight')
   plt.clf()

sol=sps.linalg.spsolve(A_fem,b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v,w=np.reshape(sol[0:Nfem_V],(nn_V,3)).T
p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %.4e %.4e cm/yr" %(np.min(u)/cm*year,np.max(u)/cm*year))
print("     -> v (m,M) %.4e %.4e cm/yr" %(np.min(v)/cm*year,np.max(v)/cm*year))
print("     -> w (m,M) %.4e %.4e cm/yr" %(np.min(w)/cm*year,np.max(w)/cm*year))
print("     -> p (m,M) %.4e %.4e MPa" %(np.min(p)/MPa,np.max(p)/MPa))

if debug:
   np.savetxt('velocity.ascii',np.array([x_V,y_V,z_V,u,v,w]).T,header='# x,y,z,u,v,w')
   np.savetxt('pressure.ascii',np.array([x_P,y_P,z_P,p]).T,header='# x,y,z,p')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# normalise pressure
###############################################################################
start=clock.time()

pavrg=0.
for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]
                N_P=basis_functions_P(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                JxWq=np.linalg.det(jcb)*weightq
                pq=np.dot(N_P,p[icon_P[:,iel]])
                pavrg+=pq*JxWq
            #end for
        #end for
    #end for
#end for

p-=pavrg/(Lx*Ly*Lz)

print("     -> p (m,M) %.4e %.4e MPa" %(np.min(p)/MPa,np.max(p)/MPa))

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# project pressure onto velocity grid
###############################################################################
start=clock.time()

r_V=[-1,1,1,-1,-1,1,1,-1, 0,1,0,-1,0,1,0,-1,-1,1,1,-1, 0,0,1,0,-1,0,0 ] 
s_V=[-1,-1,1,1,-1,-1,1,1, -1,0,1,0,-1,0,1,0,-1,-1,1,1, 0,-1,0,1,0,0,0 ]  
t_V=[-1,-1,-1,-1,1,1,1,1, -1,-1,-1,-1,1,1,1,1,0,0,0,0, -1,0,0,0,0,1,0 ] 

q=np.zeros(nn_V,dtype=np.float64)
c=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,m_V):
        N_P=basis_functions_P(r_V[i],s_V[i],t_V[i])
        q[icon_V[i,iel]]+=np.dot(N_P,p[icon_P[:,iel]])
        c[icon_V[i,iel]]+=1.
    # end for i
# end for iel

q/=c

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,z_V,q]).T,header='# x,y,q')

print("project p onto Vnodes: %.3f s" % (clock.time()-start))

###############################################################################
# create analytical pressure field
###############################################################################
start=clock.time()

p_analytical=np.zeros(nn_V,dtype=np.float64)
p_analytical1=np.zeros(nn_V,dtype=np.float64)
p_analytical2=np.zeros(nn_V,dtype=np.float64)
theta=np.zeros(nn_V,dtype=np.float64)

kappa=(Lz/2)**2/3/eta(0,0,0)

a=200e3

U=U0*2/3
    
plinefile=open('p_line.ascii',"w")
psurffile=open('p_surf.ascii',"w")
plinefile.write("# x q panal panal1 panal2 q-panal panal/q")

for i in range(0,nn_V):
    ri=np.sqrt((x_V[i]-Lx/2)**2+(y_V[i])**2)
    theta[i]=math.atan2(y_V[i],x_V[i]-Lx/2)    
    if ri>=a:
       p_analytical[i]=-U/kappa*(ri+a**2/ri)*np.cos(theta[i])
       p_analytical1[i]=-U/kappa*(ri)*np.cos(theta[i])
       p_analytical2[i]=-U/kappa*(a**2/ri)*np.cos(theta[i])

       if abs(y_V[i]/Ly)<eps and abs(z_V[i]-Lz)/Lz<eps:
          plinefile.write("%e %e %e %e %e %e %e \n" %(x_V[i],q[i],p_analytical[i],p_analytical1[i],\
                                                p_analytical2[i],q[i]-p_analytical[i],p_analytical[i]/q[i]))
       if abs(z_V[i]-Lz)/Lz<eps:
          psurffile.write("%e %e %e %e %e %e %e %e\n" %(x_V[i],y_V[i],q[i],p_analytical[i],p_analytical1[i],\
                                                p_analytical2[i],q[i]-p_analytical[i],p_analytical[i]/q[i]))
    else:
       p_analytical[i]=0
       p_analytical1[i]=0
       p_analytical2[i]=0

plinefile.close()
psurffile.close()

print("compute analytical pressure: %.3f s" % (clock.time()-start))

###############################################################################

vlinefile=open('v_line.ascii',"w")
for i in range(0,nn_V):
    if abs(y_V[i]-Ly/2)/Ly<eps and abs(z_V[i]-Lz/2)/Lz<eps:
       vlinefile.write("%e %e %e %e %e %e %e \n" %(x_V[i],y_V[i],z_V[i],u[i],v[i],w[i],q[i]))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if True:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],z_V[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    #vtufile.write("<CellData Scalars='scalars'>\n")
    #vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    #for iel in range(0,nel):
    #    vtufile.write("%10e \n" % np.sum(p[icon_P[0:8,iel]]*0.125))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/yr)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,w[i]/cm*year))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    q.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p_analytical' Format='ascii'> \n")
    p_analytical.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p_analytical(1)' Format='ascii'> \n")
    p_analytical1.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p_analytical(2)' Format='ascii'> \n")
    p_analytical2.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
    theta.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" %(eta(x_V[i],y_V[i],z_V[i])))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n" \
                       %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel],\
                         icon_V[4,iel],icon_V[5,iel],icon_V[6,iel],icon_V[7,iel],\
                         icon_V[8,iel],icon_V[9,iel],icon_V[10,iel],icon_V[11,iel],\
                         icon_V[12,iel],icon_V[13,iel],icon_V[14,iel],icon_V[15,iel],\
                         icon_V[16,iel],icon_V[17,iel],icon_V[18,iel],icon_V[19,iel],\
                         icon_V[20,iel],icon_V[21,iel],icon_V[22,iel],icon_V[23,iel],\
                         icon_V[24,iel],icon_V[25,iel],icon_V[26,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*27))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %25)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    print("export to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
