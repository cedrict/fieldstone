import numpy as np
import scipy.sparse as sps
import time as timing

###############################################################################

def u_th(x,t):
    return np.cos(2*np.pi*t)*np.sin(2*np.pi*x)

def udot_th(x,t):
    return -2*np.pi*np.sin(2*np.pi*t)*np.sin(2*np.pi*x)

###############################################################################

eps=1e-8
m=2

Lx=1
nelx=20
c=1
dt=1e-3
nstep=5000

nnx=nelx+1
hx=Lx/nelx

Nfem=nnx

method=3

###############################################################################
# node layout
###############################################################################

x=np.linspace(0,Lx,nnx)

###############################################################################
# connectivity 
###############################################################################

icon=np.zeros((m,nelx),dtype=np.int32)

for iel in range(0,nelx):
    icon[0,iel]=iel
    icon[1,iel]=iel+1

###############################################################################
# initial field value
# methods 1,2 start at t=2dt. uprev contains u at t=dt and uprevprev 
# contains u at t=0
# method 3 starts at t=dt. uprev contains u at t=0, uprevprev
# is not used and udotprev contains the time derivative of u at t=0
###############################################################################

u=np.zeros(nnx,dtype=np.float64)    
uprevprev=np.zeros(nnx,dtype=np.float64) 
uprev=np.zeros(nnx,dtype=np.float64)     
udot=np.zeros(nnx,dtype=np.float64)      
udotprev=np.zeros(nnx,dtype=np.float64) 

if method==1 or method==2:
   t=2*dt
   for i in range(0,nnx):
       uprevprev[i]=u_th(x[i],0)
       uprev[i]=u_th(x[i],0)+dt*udot_th(x[i],0)

if method==3:
   t=dt
   for i in range(0,nnx):
       uprev[i]=u_th(x[i],0)
       udotprev[i]=udot_th(x[i],0)

###############################################################################
# define temperature boundary conditions
###############################################################################

bc_fix=np.zeros(Nfem,dtype=bool)  
bc_val=np.zeros(Nfem,dtype=np.float64) 

for i in range(0,nnx):
    if x[i]/Lx<eps:
       bc_fix[i]=True ; bc_val[i]=0.
    if x[i]/Lx>(1-eps):
       bc_fix[i]=True ; bc_val[i]=0.
#end for

#******************************************************************************
#******************************************************************************
# time stepping loop
#******************************************************************************
#******************************************************************************

a_el=np.zeros((2,2),dtype=np.float64)    
b_el=np.zeros(2,dtype=np.float64)    
Me=np.array([[hx/3,hx/6],[hx/6,hx/3]],dtype=np.float64)
Ke=np.array([[1/hx,-1/hx],[-1/hx,1/hx]],dtype=np.float64)

for istep in range(0,nstep):

    print('=====istep=',istep,'================')

    ###########################################################################
    # build matrix
    ###########################################################################
    start = timing.time()

    A_mat=np.zeros((Nfem,Nfem),dtype=np.float64) # FE matrix 
    rhs=np.zeros(Nfem,dtype=np.float64)        # FE rhs 

    for iel in range(0,nelx):

        #print('====== iel=',iel,'======')

        up=np.zeros(2,dtype=np.float64)
        upp=np.zeros(2,dtype=np.float64)

        for k in range(0,m):
            up[k]=uprev[icon[k,iel]]
            upp[k]=uprevprev[icon[k,iel]]

        # build elemental matrix and rhs

        a_el[:,:]=Me[:,:]

        if method==1:
           b_el[:]=(2*Me-c**2*dt**2*Ke).dot(up)-Me.dot(upp)

        if method==2 or method==3:
           b_el=-c**2*Ke.dot(up)

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon[k1,iel]
            if bc_fix[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,m):
                   b_el[k2]-=a_el[k2,k1]*bc_val[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val[m1]
        #    #end if
        #end for

        # assemble
        for k1 in range(0,m):
            m1=icon[k1,iel]
            for k2 in range(0,m):
                m2=icon[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    # end for

    print("build matrix: %.3f s" % (timing.time() - start))

    ###########################################################################
    # solve system
    ###########################################################################
    start = timing.time()

    if method==1:
       u=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    if method==2:
       uu=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
       u=np.zeros(nnx,dtype=np.float64)  
       u[:]=dt**2*uu[:]+2*uprev[:]-uprevprev[:]

    if method==3:
       u[:]=uprev[:]+udotprev[:]*dt
       R=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
       udot[:]=udotprev[:]+dt*R[:]

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))

    print("solve time: %.3f s" % (timing.time() - start))

    ###########################################################################

    uth=np.zeros(nnx,dtype=np.float64)
    for i in range(0,nnx):
        uth[i]=u_th(x[i],t)

    filename = 'u_{:04d}.ascii'.format(istep) 
    np.savetxt(filename,np.array([x,u,uth]).T,header='# x,u')
    print('time t=',t)
    print('export solution to',filename)

    ###########################################################################

    t+=dt

    if method==1 or method==2:
       uprevprev[:]=uprev[:]
       uprev[:]=u[:]

    if method==3:
       uprev[:]=u[:]
       udotprev[:]=udot[:]

#end for


