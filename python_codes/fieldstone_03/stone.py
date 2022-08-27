import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def density(rho0,alpha,T,T0):
    val=rho0*(1.-alpha*(T-T0))
    return val

def viscosity(T):
    val=1.
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------fieldstone 03--------")
print("-----------------------------")

sqrt3=np.sqrt(3.)
eps=1.e-10 

ndim=2       # number of space dimensions
m=4          # number of nodes making up an element
ndofV=2      # number of degrees of freedom per node
ndofT=1      # number of degrees of freedom per node
Lx=1.        # horizontal extent of the domain 
Ly=1.        # vertical extent of the domain 
Ra=1e6       # Rayleigh number
alpha=1e-2   # thermal expansion coefficient
hcond=1.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density
T0=0         # reference temperature
CFL=1        # CFL number 
gy=-Ra/alpha # vertical component of gravity vector
penalty=1.e7 # penalty coefficient value
nstep=250000   # maximum number of timestep   

tol=1e-6

# allowing for argument parsing through command line
if int(len(sys.argv) == 3):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
else:
   nelx = 48
   nely = 48

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=nelx+1      # number of elements, x direction
nny=nely+1      # number of elements, y direction
NV=nnx*nny      # number of nodes
nel=nelx*nely   # number of elements, total
NfemV=NV*ndofV  # Total number of degrees of velocity freedom
NfemT=NV*ndofT  # Total number of degrees of temperature freedom

Nu_vrms_file=open('Nu_vrms.ascii',"w")
dt_file=open('dt.ascii',"w")
Tavrg_file=open('Tavrg.ascii',"w")
conv_file=open('conv.ascii',"w")

#####################################################################
# grid point setup 
#####################################################################

x = np.empty(NV, dtype=np.float64)  # x coordinates
y = np.empty(NV, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1
    #end for
#end for

#####################################################################
# connectivity
#####################################################################

icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1
    #end for
#end for

#####################################################################
# define velocity boundary conditions
#####################################################################

bc_fixV=np.zeros(NfemV,dtype=np.bool) 
bc_valV=np.zeros(NfemV,dtype=np.float64) 

for i in range(0,NV):
    if x[i]<eps:
       bc_fixV[i*ndofV]   = True ; bc_valV[i*ndofV]   = 0.
    if x[i]>(Lx-eps):
       bc_fixV[i*ndofV]   = True ; bc_valV[i*ndofV]   = 0.
    if y[i]<eps:
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0.
    if y[i]>(Ly-eps):
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0.
#end for

#####################################################################
# define temperature boundary conditions
#####################################################################

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if y[i]<eps:
       bc_fixT[i]=True ; bc_valT[i]=1.
    if y[i]>(Ly-eps):
       bc_fixT[i]=True ; bc_valT[i]=0.
#end for

#####################################################################
# initial temperature
#####################################################################

T = np.zeros(NV,dtype=np.float64)
T_prev = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T[i]=1.-y[i]-0.01*np.cos(np.pi*x[i])*np.sin(np.pi*y[i])
#end for

T_prev[:]=T[:]

#np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

#####################################################################
# create necessary arrays 
#####################################################################

N     = np.zeros(m,dtype=np.float64)   # shape functions
dNdx  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)  # x-component velocity
v     = np.zeros(NV,dtype=np.float64)  # y-component velocity
u_prev= np.zeros(NV,dtype=np.float64)  # x-component velocity
v_prev= np.zeros(NV,dtype=np.float64)  # y-component velocity
Tvect = np.zeros(4,dtype=np.float64)   
k_mat = np.array([[1.,1.,0.],[1.,1.,0.],[0.,0.,0.]],dtype=np.float64) 
c_mat = np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],dtype=np.float64) 

#==============================================================================
# time stepping loop
#==============================================================================

time=0.

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    #################################################################
    # build FE matrix
    #################################################################
    start = timing.time()


    A_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemV,dtype=np.float64)         # FE rhs 
    B_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix 

    for iel in range(0, nel):

        # set 2 arrays to 0 every loop
        b_el=np.zeros(m*ndofV,dtype=np.float64)
        a_el=np.zeros((m*ndofV,m*ndofV),dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [-1, 1]:
            for jq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.

                # calculate shape functions
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)

                # calculate shape function derivatives
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

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

                # compute dNdx & dNdy
                xq=0.0
                yq=0.0
                Tq=0.0
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                #end for

                # construct 3x8 B_mat matrix
                for i in range(0, m):
                    B_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                             [0.     ,dNdy[i]],
                                             [dNdy[i],dNdx[i]]]
                #end for

                # compute elemental A_mat matrix
                a_el += B_mat.T.dot(c_mat.dot(B_mat))*viscosity(Tq)*wq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    b_el[2*i+1]+=N[i]*jcob*wq*density(rho0,alpha,Tq,T0)*gy
                #end for

            #end for
        #end for

        # integrate penalty term at 1 point
        rq=0.
        sq=0.
        wq=2.*2.

        N[0]=0.25*(1.-rq)*(1.-sq)
        N[1]=0.25*(1.+rq)*(1.-sq)
        N[2]=0.25*(1.+rq)*(1.+sq)
        N[3]=0.25*(1.-rq)*(1.+sq)

        dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
        dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
        dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
        dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

        # compute the jacobian
        jcb=np.zeros((2,2),dtype=float)
        for k in range(0, m):
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
            B_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                                  [0.     ,dNdy[i]],
                                  [dNdy[i],dNdx[i]]]
        #end for

        # compute elemental matrix
        a_el += B_mat.T.dot(k_mat.dot(B_mat))*penalty*wq*jcob

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

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,m):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*icon[k1,iel]+i1
                for k2 in range(0,m):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*icon[k2,iel]+i2
                        A_mat[m1,m2]+=a_el[ikk,jkk]
                    #end for
                #end for
                rhs[m1]+=b_el[ikk]
            #end for
        #end for

    #end for iel

    print("building Stokes matrix and rhs: %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    sol = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    u,v=np.reshape(sol,(NV,2)).T

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

    print("solve V time: %.3f s" % (timing.time() - start))

    #################################################################
    # compute timestep
    #################################################################
    start = timing.time()

    dt1=CFL*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))

    dt2=CFL*(Lx/nelx)**2/(hcond/hcapa/rho0)

    dt=np.min([dt1,dt2])

    time+=dt

    print('     -> dt1= %.6f' %dt1)
    print('     -> dt2= %.6f' %dt2)
    print('     -> dt = %.6f' %dt)

    dt_file.write("%10e %10e %10e %10e\n" % (time,dt1,dt2,dt))
    dt_file.flush()

    print("compute time step: %.3f s" % (timing.time() - start))

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions

    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m):
            Tvect[k]=T[icon[k,iel]]
        #end for

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.

                # calculate shape functions
                N_mat[0,0]=0.25*(1.-rq)*(1.-sq)
                N_mat[1,0]=0.25*(1.+rq)*(1.-sq)
                N_mat[2,0]=0.25*(1.+rq)*(1.+sq)
                N_mat[3,0]=0.25*(1.-rq)*(1.+sq)

                # calculate shape function derivatives
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

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

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,m):
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                #end for

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho0*hcapa*wq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*wq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*wq*jcob

                #a_el+=MM+(Ka+Kd)*dt
                #b_el+=MM.dot(Tvect)

                a_el+=MM+(Ka+Kd)*dt*0.5
                b_el+=(MM-(Ka+Kd)*dt*0.5).dot(Tvect)

            #end for
        #end for

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

    print("building temperature matrix and rhs: %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T time: %.3f s" % (timing.time() - start))

    #################################################################
    # compute vrms 
    #################################################################
    start = timing.time()

    vrms=0.
    Tavrg=0.
    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                #end for
                jcob = np.linalg.det(jcb)
                uq=0.
                vq=0.
                Tq=0.
                for k in range(0,m):
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
                #end for
                vrms+=(uq**2+vq**2)*wq*jcob
                Tavrg+=Tq*wq*jcob
            #end for
        #end for
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))
    Tavrg/=(Lx*Ly)

    Tavrg_file.write("%10e %10e\n" % (time,Tavrg))
    Tavrg_file.flush()

    print("     -> time= %.6f ; vrms   = %.6f" %(time,vrms))

    print("compute vrms and <T>: %.3f s" % (timing.time() - start))

    #################################################################
    # compute Nusselt number at top
    #################################################################
    start = timing.time()

    Nusselt=0
    for iel in range(0,nel):
        qy=0.
        rq=0.
        sq=0.
        wq=2.*2.
        dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
        dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
        dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
        dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
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
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
        #end for
        for k in range(0,m):
            qy+=-hcond*dNdy[k]*T[icon[k,iel]]
        #end for
        if y[icon[3,iel]]>Ly-eps:
           Nusselt+=qy*hx
        #end if
    #end for

    Nu_vrms_file.write("%10e %10e %10e\n" % (time,Nusselt,vrms))
    Nu_vrms_file.flush()

    print("     -> time= %.6f ; Nusselt= %.6f" %(time,Nusselt))

    print("compute Nu: %.3f s" % (timing.time() - start))

    #################################################################
    # has steady state been reached?
    #################################################################
    start = timing.time()

    T_diff=np.sum(abs(T-T_prev))/NV
    u_diff=np.sum(abs(u-u_prev))/NV
    v_diff=np.sum(abs(v-v_prev))/NV

    print("     -> time= %.6f ; <T_diff>= %.6f" %(time,T_diff))
    print("     -> time= %.6f ; <u_diff>= %.6f" %(time,u_diff))
    print("     -> time= %.6f ; <v_diff>= %.6f" %(time,v_diff))

    print("     -> T,u,v conv" , T_diff<tol*Tavrg,u_diff<tol*vrms,v_diff<tol*vrms)

    conv_file.write("%e %e %e %e\n" % (time,T_diff/Tavrg,u_diff/vrms,v_diff/vrms))
    conv_file.flush()

    if T_diff<tol*Tavrg and u_diff<tol*vrms and v_diff<tol*vrms:
       print("convergence reached")
       break

    T_prev[:]=T[:]
    u_prev[:]=u[:]
    v_prev[:]=v[:]

    print("assess steady state: %.3f s" % (timing.time() - start))
    
#end for

#==============================================================================
# end time stepping loop
#==============================================================================


#####################################################################
# retrieve pressure
#####################################################################
start = timing.time()

p=np.zeros(nel,dtype=np.float64)  
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
qx=np.zeros(nel,dtype=np.float64)  
qy=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
rho=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
    #end for

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    #end for

    Tc=0.
    for k in range(0,m):
        Tc += N[k]*T[icon[k,iel]]
        xc[iel]+=N[k]*x[icon[k,iel]]
        yc[iel]+=N[k]*y[icon[k,iel]]
        qx[iel]+=-hcond*dNdx[k]*T[icon[k,iel]]
        qy[iel]+=-hcond*dNdy[k]*T[icon[k,iel]]
        exx[iel]+=dNdx[k]*u[icon[k,iel]]
        eyy[iel]+=dNdy[k]*v[icon[k,iel]]
        exy[iel]+=0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]
    #end for

    p[iel]=-penalty*(exx[iel]+eyy[iel])
    rho[iel]=density(rho0,alpha,Tc,T0)
    
#end for

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("     -> rho (m,M) %.4f %.4f " %(np.min(rho),np.max(rho)))
print("     -> qx (m,M) %.4f %.4f " %(np.min(qx),np.max(qx)))
print("     -> qy (m,M) %.4f %.4f " %(np.min(qy),np.max(qy)))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')
#np.savetxt('temperature.ascii',np.array([x,y,T]).T,header='# x,y,T')
#np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
#np.savetxt('heatflux.ascii',np.array([xc,yc,qx,qy]).T,header='# xc,yc,qx,qy')
    
print("compute fields: %.3f s" % (timing.time() - start))

#####################################################################
# plot of solution
#####################################################################

u_temp=np.reshape(u,(nnx,nny))
v_temp=np.reshape(v,(nnx,nny))
T_temp=np.reshape(T,(nnx,nny))
p_temp=np.reshape(p,(nelx,nely))
exx_temp=np.reshape(exx,(nelx,nely))
eyy_temp=np.reshape(eyy,(nelx,nely))
exy_temp=np.reshape(exy,(nelx,nely))
rho_temp=np.reshape(rho,(nelx,nely))
qx_temp=np.reshape(qx,(nelx,nely))
qy_temp=np.reshape(qy,(nelx,nely))

fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(18,18))

uextent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y))
pextent=(np.amin(xc),np.amax(xc),np.amin(yc),np.amax(yc))

im = axes[0][0].imshow(u_temp,extent=uextent,cmap='Spectral',interpolation='nearest')
axes[0][0].set_title('$v_x$', fontsize=10, y=1.01)
axes[0][0].set_xlabel('x')
axes[0][0].set_ylabel('y')
fig.colorbar(im,ax=axes[0][0])

im = axes[0][1].imshow(v_temp,extent=uextent,cmap='Spectral',interpolation='nearest')
axes[0][1].set_title('$v_y$', fontsize=10, y=1.01)
axes[0][1].set_xlabel('x')
axes[0][1].set_ylabel('y')
fig.colorbar(im,ax=axes[0][1])

im = axes[0][2].imshow(p_temp,extent=pextent,cmap='RdGy',interpolation='nearest')
axes[0][2].set_title('$p$', fontsize=10, y=1.01)
axes[0][2].set_xlim(0,Lx)
axes[0][2].set_ylim(0,Ly)
axes[0][2].set_xlabel('x')
axes[0][2].set_ylabel('y')
fig.colorbar(im,ax=axes[0][2])

im = axes[0][3].imshow(T_temp,extent=uextent,cmap='jet',interpolation='nearest')
axes[0][3].set_title('$T$', fontsize=10, y=1.01)
axes[0][3].set_xlabel('x')
axes[0][3].set_ylabel('y')
fig.colorbar(im,ax=axes[0][3])

im = axes[1][0].imshow(exx_temp,extent=pextent, cmap='viridis',interpolation='nearest')
axes[1][0].set_title('$\dot{\epsilon}_{xx}$',fontsize=10, y=1.01)
axes[1][0].set_xlim(0,Lx)
axes[1][0].set_ylim(0,Ly)
axes[1][0].set_xlabel('x')
axes[1][0].set_ylabel('y')
fig.colorbar(im,ax=axes[1][0])

im = axes[1][1].imshow(eyy_temp,extent=pextent,cmap='viridis',interpolation='nearest')
axes[1][1].set_title('$\dot{\epsilon}_{yy}$',fontsize=10,y=1.01)
axes[1][1].set_xlim(0,Lx)
axes[1][1].set_ylim(0,Ly)
axes[1][1].set_xlabel('x')
axes[1][1].set_ylabel('y')
fig.colorbar(im,ax=axes[1][1])

im = axes[1][2].imshow(exy_temp,extent=pextent,cmap='viridis',interpolation='nearest')
axes[1][2].set_title('$\dot{\epsilon}_{xy}$',fontsize=10,y=1.01)
axes[1][2].set_xlim(0,Lx)
axes[1][2].set_ylim(0,Ly)
axes[1][2].set_xlabel('x')
axes[1][2].set_ylabel('y')
fig.colorbar(im,ax=axes[1][2])

im = axes[1][3].imshow(rho_temp,extent=uextent,cmap='RdYlBu',interpolation='nearest')
axes[1][3].set_title('$rho$', fontsize=10, y=1.01)
axes[1][3].set_xlabel('x')
axes[1][3].set_ylabel('y')
fig.colorbar(im,ax=axes[1][3])

im = axes[2][0].imshow(qx_temp,extent=uextent,cmap='RdYlBu',interpolation='nearest')
axes[2][0].set_title('$q_x$', fontsize=10, y=1.01)
axes[2][0].set_xlim(0,Lx)
axes[2][0].set_ylim(0,Ly)
axes[2][0].set_xlabel('x')
axes[2][0].set_ylabel('y')
fig.colorbar(im,ax=axes[2][0])

im = axes[2][1].imshow(qy_temp,extent=uextent,cmap='RdYlBu',interpolation='nearest')
axes[2][1].set_title('$q_y$', fontsize=10, y=1.01)
axes[2][1].set_xlim(0,Lx)
axes[2][1].set_ylim(0,Ly)
axes[2][1].set_xlabel('x')
axes[2][1].set_ylabel('y')
fig.colorbar(im,ax=axes[2][1])

plt.subplots_adjust(hspace=0.5)

plt.savefig('solution.pdf', bbox_inches='tight')
#plt.show()

#####################################################################
# export to vtu 
#####################################################################
start = timing.time()

vtufile=open("solution.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % p[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % exx[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % eyy[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % exy[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='qx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % qx[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='qy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % qy[iel])
vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='sr' Format='ascii'> \n")
#for iel in range (0,nel):
#    vtufile.write("%e \n" % sr[iel])
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
#for iel in range (0,nel):
#    vtufile.write("%e \n" % eta[iel])
#vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e \n" % T[i])
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export to vtu: %.3f s" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
