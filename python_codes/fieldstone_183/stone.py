import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix
import numba

eps=1e-8
TKelvin=273.15
year=3600*24*365.25

###############################################################################

##@numba.njit
def basis_functions(r,order):
    if order==1:
        N0 = 1 / 2 - r / 2
        N1 = r / 2 + 1 / 2
        return np.array([N0,N1],dtype=np.float64)
    if order==2:
        N0 = r * (r - 1) / 2
        N1 = 1 - r ** 2
        N2 = r * (r + 1) / 2
        return np.array([N0,N1,N2],dtype=np.float64)
    if order==3:
        N0 = -9 * r ** 3 / 16 + 9 * r ** 2 / 16 + r / 16 - 1 / 16
        N1 = 27 * r ** 3 / 16 - 9 * r ** 2 / 16 - 27 * r / 16 + 9 / 16
        N2 = -27 * r ** 3 / 16 - 9 * r ** 2 / 16 + 27 * r / 16 + 9 / 16
        N3 = 9 * r ** 3 / 16 + 9 * r ** 2 / 16 - r / 16 - 1 / 16
        return np.array([N0,N1,N2,N3],dtype=np.float64)
    if order==4:
        N0 = 2 * r ** 4 / 3 - 2 * r ** 3 / 3 - r ** 2 / 6 + r / 6
        N1 = -8 * r ** 4 / 3 + 4 * r ** 3 / 3 + 8 * r ** 2 / 3 - 4 * r / 3
        N2 = 4 * r ** 4 - 5 * r ** 2 + 1
        N3 = -8 * r ** 4 / 3 - 4 * r ** 3 / 3 + 8 * r ** 2 / 3 + 4 * r / 3
        N4 = 2 * r ** 4 / 3 + 2 * r ** 3 / 3 - r ** 2 / 6 - r / 6
        return np.array([N0,N1,N2,N3,N4], dtype=np.float64)
    if order==5:
        N0 = -625 * r ** 5 / 768 + 625 * r ** 4 / 768 + 125 * r ** 3 / 384 - 125 * r ** 2 / 384 - 3 * r / 256 + 3 / 256
        N1 = 3125 * r ** 5 / 768 - 625 * r ** 4 / 256 - 1625 * r ** 3 / 384 + 325 * r ** 2 / 128 + 125 * r / 768 - 25 / 256
        N2 = -3125 * r ** 5 / 384 + 625 * r ** 4 / 384 + 2125 * r ** 3 / 192 - 425 * r ** 2 / 192 - 375 * r / 128 + 75 / 128
        N3 = 3125 * r ** 5 / 384 + 625 * r ** 4 / 384 - 2125 * r ** 3 / 192 - 425 * r ** 2 / 192 + 375 * r / 128 + 75 / 128
        N4 = -3125 * r ** 5 / 768 - 625 * r ** 4 / 256 + 1625 * r ** 3 / 384 + 325 * r ** 2 / 128 - 125 * r / 768 - 25 / 256
        N5 = 625 * r ** 5 / 768 + 625 * r ** 4 / 768 - 125 * r ** 3 / 384 - 125 * r ** 2 / 384 + 3 * r / 256 + 3 / 256
        return np.array([N0,N1,N2,N3,N4,N5], dtype=np.float64)
    if order==6:
        N0 = 81 * r ** 6 / 80 - 81 * r ** 5 / 80 - 9 * r ** 4 / 16 + 9 * r ** 3 / 16 + r ** 2 / 20 - r / 20
        N1 = -243 * r ** 6 / 40 + 81 * r ** 5 / 20 + 27 * r ** 4 / 4 - 9 * r ** 3 / 2 - 27 * r ** 2 / 40 + 9 * r / 20
        N2 = 243 * r ** 6 / 16 - 81 * r ** 5 / 16 - 351 * r ** 4 / 16 + 117 * r ** 3 / 16 + 27 * r ** 2 / 4 - 9 * r / 4
        N3 = -81 * r ** 6 / 4 + 63 * r ** 4 / 2 - 49 * r ** 2 / 4 + 1
        N4 = 243 * r ** 6 / 16 + 81 * r ** 5 / 16 - 351 * r ** 4 / 16 - 117 * r ** 3 / 16 + 27 * r ** 2 / 4 + 9 * r / 4
        N5 = -243 * r ** 6 / 40 - 81 * r ** 5 / 20 + 27 * r ** 4 / 4 + 9 * r ** 3 / 2 - 27 * r ** 2 / 40 - 9 * r / 20
        N6 = 81 * r ** 6 / 80 + 81 * r ** 5 / 80 - 9 * r ** 4 / 16 - 9 * r ** 3 / 16 + r ** 2 / 20 + r / 20
        return np.array([N0,N1,N2,N3,N4,N5,N6],dtype=np.float64)


###############################################################################

##@numba.njit
def basis_functions_r(r,order):
    if order==1:
        B = np.array([ -1 / 2 , 1 / 2 ], dtype=np.float64)
        return B
    if order==2:
        B = np.array([r - 1 / 2,  -2 * r,  r + 1 / 2 ], dtype=np.float64)
        return B
    if order==3:
        B = np.array([
            -27 * r ** 2 / 16 + 9 * r / 8 + 1 / 16,
            81 * r ** 2 / 16 - 9 * r / 8 - 27 / 16,
            -81 * r ** 2 / 16 - 9 * r / 8 + 27 / 16,
            27 * r ** 2 / 16 + 9 * r / 8 - 1 / 16   ], dtype=np.float64)
        return B
    if order==4:
        B = np.array([
            [8 * r ** 3 / 3 - 2 * r ** 2 - r / 3 + 1 / 6],
            [-32 * r ** 3 / 3 + 4 * r ** 2 + 16 * r / 3 - 4 / 3],
            [16 * r ** 3 - 10 * r],
            [-32 * r ** 3 / 3 - 4 * r ** 2 + 16 * r / 3 + 4 / 3],
            [8 * r ** 3 / 3 + 2 * r ** 2 - r / 3 - 1 / 6]
        ], dtype=np.float64)
        return B
    if order==5:
        B = np.array([
            [-3125 * r ** 4 / 768 + 625 * r ** 3 / 192 + 125 * r ** 2 / 128 - 125 * r / 192 - 3 / 256],
            [15625 * r ** 4 / 768 - 625 * r ** 3 / 64 - 1625 * r ** 2 / 128 + 325 * r / 64 + 125 / 768],
            [-15625 * r ** 4 / 384 + 625 * r ** 3 / 96 + 2125 * r ** 2 / 64 - 425 * r / 96 - 375 / 128],
            [15625 * r ** 4 / 384 + 625 * r ** 3 / 96 - 2125 * r ** 2 / 64 - 425 * r / 96 + 375 / 128],
            [-15625 * r ** 4 / 768 - 625 * r ** 3 / 64 + 1625 * r ** 2 / 128 + 325 * r / 64 - 125 / 768],
            [3125 * r ** 4 / 768 + 625 * r ** 3 / 192 - 125 * r ** 2 / 128 - 125 * r / 192 + 3 / 256]
        ], dtype=np.float64)
        return B
    if order==6:
        B = np.array([
            [243 * r ** 5 / 40 - 81 * r ** 4 / 16 - 9 * r ** 3 / 4 + 27 * r ** 2 / 16 + r / 10 - 1 / 20],
            [-729 * r ** 5 / 20 + 81 * r ** 4 / 4 + 27 * r ** 3 - 27 * r ** 2 / 2 - 27 * r / 20 + 9 / 20],
            [729 * r ** 5 / 8 - 405 * r ** 4 / 16 - 351 * r ** 3 / 4 + 351 * r ** 2 / 16 + 27 * r / 2 - 9 / 4],
            [-243 * r ** 5 / 2 + 126 * r ** 3 - 49 * r / 2],
            [729 * r ** 5 / 8 + 405 * r ** 4 / 16 - 351 * r ** 3 / 4 - 351 * r ** 2 / 16 + 27 * r / 2 + 9 / 4],
            [-729 * r ** 5 / 20 - 81 * r ** 4 / 4 + 27 * r ** 3 + 27 * r ** 2 / 2 - 27 * r / 20 - 9 / 20],
            [243 * r ** 5 / 40 + 81 * r ** 4 / 16 - 9 * r ** 3 / 4 - 27 * r ** 2 / 16 + r / 10 + 1 / 20]
        ], dtype=np.float64)
        return B

###############################################################################

#@numba.njit
def nq_points_weights(order):

    match order:
     case 1:     
        num=1
        coords = [0]
        weights = [2]
     case 2 | 3:
        num=2
        coords = [-1./np.sqrt(3.),1./np.sqrt(3.)]
        weights = [1.,1.]
     case 4 | 5:
        num=3
        coords = [-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
        weights = [5./9.,8./9.,5./9.]
     case 6 | 7:
        num=4
        coords = [-np.sqrt((3./7.)+(2./7.)*np.sqrt(6./5.)),
                  -np.sqrt((3./7.)-(2./7.)*np.sqrt(6./5.)),
                   np.sqrt((3./7.)-(2./7.)*np.sqrt(6./5.)),
                   np.sqrt((3./7.)+(2./7.)*np.sqrt(6./5.))]
        weights = [(18.-np.sqrt(30.))/36.,
                   (18.+np.sqrt(30.))/36.,
                   (18.+np.sqrt(30.))/36.,
                   (18.-np.sqrt(30.))/36.]
     case 8 | 9:
        num=5
        coords = [-1./3. * np.sqrt(5.+2.*np.sqrt(10./7.)),
                  -1./3. * np.sqrt(5.-2.*np.sqrt(10./7.)),
                  0.,
                  1./3. * np.sqrt(5.-2.*np.sqrt(10./7.)),
                  1./3. * np.sqrt(5.+2.*np.sqrt(10./7.))]
        weights = [(322.-13.*np.sqrt(70.))/900.,
                   (322.+13.*np.sqrt(70.))/900.,
                   128./225.,
                   (322.+13.*np.sqrt(70.))/900.,
                   (322.-13.*np.sqrt(70.))/900.]
     case 10 | 11:
        num=6
        coords = [-0.932469514203152,
                  -0.661209386466265,
                  -0.238619186083197,
                  0.238619186083197,
                  0.661209386466265,
                  0.932469514203152]
        weights = [0.171324492379170,
                   0.360761573048139,
                   0.467913934572691,
                   0.467913934572691,
                   0.360761573048139,
                   0.171324492379170]
     case 12 | 13:
        num=7
        coords = [-0.949107912342759,
                  -0.741531185599394,
                  -0.405845151377397,
                  0.,
                  0.405845151377397,
                  0.741531185599394,
                  0.949107912342759]
        weights = [0.129484966168870,
                   0.279705391489277,
                   0.381830050505119,
                   0.417959183673469,
                   0.381830050505119,
                   0.279705391489277,
                   0.129484966168870]

    return num, coords, weights

###############################################################################
# define physical variables
###############################################################################

Lx=100e3
rho=3000
hcapa=1000
hcond=3
Tleft=200+TKelvin
Tright=100+TKelvin
kappa=hcond/rho/hcapa
nelx=9
order=1
m_T=order+1
nn_T=order*nelx+1
hx=Lx/nelx
dt=hx**2/2/kappa * 0.1
nstep=1000
epsilon=0.001 # in Kelvin

###############################################################################

print('nelx=',nelx)
print('order=',order)
print('nn_T=',nn_T)
print('hx=',hx)
print('m_T=',m_T)
print('dt=',dt/year,'year')
print('kappa=',kappa)

###############################################################################
# make mesh
###############################################################################

x_T=np.zeros(nn_T,dtype=np.float64)

for i in range(0,nn_T):
    x_T[i]=i*Lx/(nn_T-1)

np.savetxt('mesh.ascii',np.array([x_T]).T,header='#x')

###############################################################################
# build icon array
###############################################################################

icon_T=np.zeros((m_T,nelx),dtype=np.int32)

for iel in range(0,nelx):
    for k in range(0,m_T):
        icon_T[k,iel]=order*iel+k

#print(icon_T)

###############################################################################
# define T init
###############################################################################

T_old=np.zeros(nn_T)

for i in range(nn_T):
    if x_T[i]<Lx/2:
       T_old[i]=200+TKelvin
    else:
       T_old[i]=100+TKelvin

fig,ax=plt.subplots()
ax.plot(x_T,T_old-TKelvin)
plt.savefig('T_init.pdf')

###############################################################################
# define T analytical 
###############################################################################

T_analytical=np.zeros(nn_T)

for i in range(nn_T):
    T_analytical[i]= x_T[i]/Lx*(Tright-Tleft) +Tleft

ax.plot(x_T,T_analytical-TKelvin)
plt.savefig('T_analytical.pdf')

###############################################################################
# define bc  
###############################################################################

bc_fix_T=np.zeros(nn_T,dtype=bool)  
bc_val_T=np.zeros(nn_T,dtype=np.float64) 

for i in range(0,nn_T):
    if x_T[i]/Lx<eps:
       bc_fix_T[i]=True ; bc_val_T[i]=Tleft
    if x_T[i]/Lx>(1-eps):
       bc_fix_T[i]=True ; bc_val_T[i]=Tright

###############################################################################
# select appropriate quadrature
###############################################################################


# nq quadrature points -> we can exactly integrate
# polynomials of order up to 2*nq-1 !!


nq_per_dim,qcoords,qweights=nq_points_weights(2*order)

print('nq_per_dim=',nq_per_dim)

###############################################################################
# time stepping
###############################################################################

time=0

for istep in range(nstep):

    ###########################################################################
    # build FEM matrix & rhs
    ###########################################################################

    A_fem=lil_matrix((nn_T,nn_T)) # global FEM matrix
    b_fem=np.zeros(nn_T)          # global FEM rhs

    for iel in range(nelx):

        MM=np.zeros((m_T,m_T),dtype=np.float64)
        Kd=np.zeros((m_T,m_T),dtype=np.float64)
        A_el=np.zeros((m_T,m_T),dtype=np.float64)
        b_el=np.zeros(m_T,dtype=np.float64)
        Tvect=np.zeros(m_T,dtype=np.float64)
        
        Tvect[:]=T_old[icon_T[0:m_T,iel]]

        # compute Ael, bel with GQ
        for iq in range(0,nq_per_dim):
            rq=qcoords[iq]
            weightq=qweights[iq]
            N=basis_functions(rq,order)
            dNdr=basis_functions_r(rq,order)
            jcob=hx/2.
            dNdx=dNdr*2./hx
            MM+=np.outer(N,N)*rho*hcapa*jcob*weightq # mass matrix
            Kd+=np.outer(dNdx,dNdx)*hcond*jcob*weightq # diffusion matrix
        #end for k

        # IMPLEMENT !!!

        A_el+=MM+Kd*dt
        b_el+=MM.dot(Tvect)

        # apply boundary conditions
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_T):
                   m2=icon_T[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            #end for
        #end for

        # assembly 
        for k1 in range(0,m_T):
             m1=icon_T[k1,iel]
             for k2 in range(0,m_T):
                 m2=icon_T[k2,iel]
                 A_fem[m1,m2]+=A_el[k1,k2]
             #end for
             b_fem[m1]+=b_el[k1]
         #end for

    #end for

    ###########################################################################
    # solve linear system 
    ###########################################################################
    
    T_new=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    time+=dt

    print('istep=',istep,'time=',time/year,'yr | T (m/M): ',np.min(T_new)-TKelvin,np.max(T_new)-TKelvin)

    ax.plot(x_T,T_new-TKelvin)
    plt.savefig('T.pdf')

    np.savetxt('T_'+str(istep)+'.ascii',np.array([x_T,T_new]).T,header='#x,T')

    ###########################################################################
    #assess steady state is reached
    ###########################################################################
    if np.linalg.norm(T_old-T_new)<epsilon :
       print('**** steady state reached! ****')
       #print(np.linalg.norm(T_new-T_analytical))   
       #print((T_new-T_analytical))
       break

    ###########################################################################
    #copy new solution into 'old' vector
    ###########################################################################

    T_old[:]=T_new[:]

#end time stepping




