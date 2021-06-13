import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def rho(rho0,alpha,T,T0):
    val=rho0*(1.-alpha*(T-T0)) - rho0
    return val

def T_anal(x,y):
    return T0*np.cos(np.pi/Lx*x)*np.sinh(np.pi/Lx*y)/np.sinh(np.pi/Lx*Ly)

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------fieldstone 03--------")
print("-----------------------------")

cm=0.01
year=365.25*3600*24
sqrt3=np.sqrt(3.)
eps=1.e-10 

ndim=2       # number of space dimensions
m=4          # number of nodes making up an element
ndofV=2      # number of degrees of freedom per node
ndofT=1      # number of degrees of freedom per node
Lx=700e3     # horizontal extent of the domain 
Ly=Lx        # vertical extent of the domain 
alpha=2e-5   # thermal expansion coefficient
hcond=6.66   # thermal conductivity
hcapa=1200   # heat capacity
rho0=3700    # reference density
T0=100       # reference temperature
CFL=1        # CFL number 
gy=-10       # vertical component of gravity vector
eta=1e17*rho0
penalty=1e6*eta # penalty coefficient value
nstep=2500   # maximum number of timestep   

dt_max=1e6*year

tol=1e-6

# allowing for argument parsing through command line
if int(len(sys.argv) == 3):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
else:
   nelx = 96
   nely = 96

Ra= rho0*abs(gy)*alpha*T0*Lx**3/(hcond/rho0/hcapa)/eta

print('Ra=',Ra)

kappa=hcond/rho0/hcapa

print('kappa=',kappa)

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

NfemV=nnp*ndofV  # Total number of degrees of velocity freedom
NfemT=nnp*ndofT  # Total number of degrees of temperature freedom

Nu_vrms_file=open('Nu_vrms.ascii',"w")
dt_file=open('dt.ascii',"w")
Tavrg_file=open('Tavrg.ascii',"w")

#####################################################################
# grid point setup 
#####################################################################

print("grid point setup")

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates
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

print("connectivity array")

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

print("defining velocity boundary conditions")

bc_fixV=np.zeros(NfemV,dtype=np.bool) 
bc_valV=np.zeros(NfemV,dtype=np.float64) 

for i in range(0,nnp):
    if x[i]/Lx<eps:
       bc_fixV[i*ndofV]   = True ; bc_valV[i*ndofV]   = 0.
    if x[i]/Lx>(1-eps):
       bc_fixV[i*ndofV]   = True ; bc_valV[i*ndofV]   = 0.
    if y[i]/Ly<eps:
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0.
    if y[i]/Ly>(1-eps):
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0.
#end for

#####################################################################
# define temperature boundary conditions
#####################################################################

print("defining temperature boundary conditions")

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,nnp):
    if y[i]/Ly<eps:
       bc_fixT[i]=True ; bc_valT[i]=0
    if y[i]/Ly>(1-eps):
       bc_fixT[i]=True ; bc_valT[i]=T0*np.cos(np.pi*x[i]/Lx)
#end for

#####################################################################
# initial temperature
#####################################################################

T = np.zeros(nnp,dtype=np.float64)
T_prev = np.zeros(nnp,dtype=np.float64)

for i in range(0,nnp):
    T[i]=0.
#end for

T_prev[:]=T[:]

#np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

#####################################################################
# create necessary arrays 
#####################################################################

N     = np.zeros(m,dtype=np.float64)    # shape functions
dNdx  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)  # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)  # y-component velocity
u_prev= np.zeros(nnp,dtype=np.float64)  # x-component velocity
v_prev= np.zeros(nnp,dtype=np.float64)  # y-component velocity
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

    print("building Stokes matrix and rhs")

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

                # calculate the determinant of the jacobian
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
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
                a_el += B_mat.T.dot(c_mat.dot(B_mat))*eta*wq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    b_el[2*i+1]+=N[i]*jcob*wq*rho(rho0,alpha,Tq,T0)*gy
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

        # calculate determinant of the jacobian
        jcob = np.linalg.det(jcb)

        # calculate the inverse of the jacobian
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

    #################################################################
    # impose boundary conditions
    #################################################################

    print("imposing boundary conditions")

    for i in range(0,NfemV):
        if bc_fixV[i]:
           A_matref = A_mat[i,i]
           for j in range(0,NfemV):
               rhs[j]-= A_mat[i,j]*bc_valV[i]
               A_mat[i,j]=0.
               A_mat[j,i]=0.
               A_mat[i,i]=A_matref
           #end for
           rhs[i]=A_matref*bc_valV[i]
        #end if
    #end for

    #################################################################
    # solve system
    #################################################################

    start = timing.time()
    sol = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
    print("solve V time: %.3f s" % (timing.time() - start))

    #################################################################
    # put solution into separate x,y velocity arrays
    #################################################################

    u,v=np.reshape(sol,(nnp,2)).T

    print("u (m,M) %.6f %.6f (cm/yr)" %(np.min(u)/cm*year,np.max(u)/cm*year))
    print("v (m,M) %.6f %.6f (cm/yr)" %(np.min(v)/cm*year,np.max(v)/cm*year))

    #################################################################
    # compute timestep
    #################################################################

    dt1=CFL*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))

    dt2=CFL*(Lx/nelx)**2/(hcond/hcapa/rho0)

    dt=np.min([dt1,dt2,dt_max])

    time+=dt

    print('dt1= %.6f (yr)' %(dt1/year))
    print('dt2= %.6f (yr)' %(dt2/year))
    print('dt = %.6f (yr)' %(dt/year))

    dt_file.write("%10e %10e %10e %10e\n" % (time,dt1,dt2,dt))
    dt_file.flush()

    #################################################################
    # build temperature matrix
    #################################################################

    print("building temperature matrix and rhs")

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

                a_el=MM+(Ka+Kd)*dt

                b_el=MM.dot(Tvect)

                # assemble matrix A_mat and right hand side rhs
                for k1 in range(0,m):
                    m1=icon[k1,iel]
                    for k2 in range(0,m):
                        m2=icon[k2,iel]
                        A_mat[m1,m2]+=a_el[k1,k2]
                    #end for
                    rhs[m1]+=b_el[k1]
                #end for
            #end for
        #end for

    #end for iel

    #################################################################
    # apply boundary conditions
    #################################################################

    print("imposing boundary conditions temperature")

    for i in range(0,NfemT):
        if bc_fixT[i]:
           A_matref = A_mat[i,i]
           for j in range(0,NfemT):
               rhs[j]-= A_mat[i, j] * bc_valT[i]
               A_mat[i,j]=0.
               A_mat[j,i]=0.
               A_mat[i,i] = A_matref
           #end for
           rhs[i]=A_matref*bc_valT[i]
        #end if 
    #end for

    #print("A_mat (m,M) = %.4f %.4f" %(np.min(A_mat),np.max(A_mat)))
    #print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

    #################################################################
    # solve system
    #################################################################

    start = timing.time()
    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
    print("solve T time: %.3f s" % (timing.time() - start))

    print("T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    #################################################################
    # compute vrms 
    #################################################################

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

    Tavrg_file.write("%10e %10e\n" % (time/year,Tavrg))
    Tavrg_file.flush()

    print("time= %.6f ; vrms   = %.6f" %(time,vrms))

    #################################################################
    # compute Nusselt number at top
    # actually it is not the Nusselt number, only int_Lx qy dx
    #################################################################

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
        if y[icon[3,iel]]/Ly>1-eps:
           Nusselt+=qy*hx
        #end if
    #end for

    Nu_vrms_file.write("%10e %10e %10e\n" % (time/year,Nusselt,vrms))
    Nu_vrms_file.flush()

    print("time= %.6f ; Nusselt= %.6f" %(time,Nusselt))

    #############################333

    T_diff=np.sum(abs(T-T_prev))/nnp
    u_diff=np.sum(abs(u-u_prev))/nnp
    v_diff=np.sum(abs(v-v_prev))/nnp

    print("time= %.3e yr ; <T_diff>= %.3e ; tol= %.3e" %(time/year,T_diff,tol))
    print("time= %.3e yr ; <u_diff>= %.3e ; tol= %.3e" %(time/year,u_diff/cm*year,tol))
    print("time= %.3e yr ; <v_diff>= %.3e ; tol= %.3e" %(time/year,v_diff/cm*year,tol))

    print("T conv" , T_diff<tol*Tavrg)
    print("u conv" , u_diff<tol*vrms)
    print("v conv" , v_diff<tol*vrms)

    if T_diff<tol*Tavrg and u_diff<tol*vrms and v_diff<tol*vrms:
       print("convergence reached")
       break

    T_prev[:]=T[:]
    u_prev[:]=u[:]
    v_prev[:]=v[:]
    
    #####################################################################
    # retrieve pressure
    #####################################################################

    p=np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):

        rq = 0.0
        sq = 0.0

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
            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
        #end for

        exx=0
        eyy=0
        for k in range(0,m):
            exx+=dNdx[k]*u[icon[k,iel]]
            eyy+=dNdy[k]*v[icon[k,iel]]
        #end for
        p[iel]=-penalty*(exx+eyy)
    
    #end for

    print("p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    #####################################################################
    # compute field derivatives 
    #####################################################################

    qx=np.zeros(nnp,dtype=np.float64)  
    qy=np.zeros(nnp,dtype=np.float64)  
    exx=np.zeros(nnp,dtype=np.float64)  
    eyy=np.zeros(nnp,dtype=np.float64)  
    exy=np.zeros(nnp,dtype=np.float64)  
    ccc=np.zeros(nnp,dtype=np.float64)  

    for iel in range(0,nel):

        for kk in range(0,4):
            if kk==0:
               rq = -1.0
               sq = -1.0
            if kk==1:
               rq = +1.0
               sq = -1.0
            if kk==2:
               rq = +1.0
               sq = +1.0
            if kk==3:
               rq = -1.0
               sq = +1.0

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
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
            #end for

            q_x=0
            q_y=0
            e_xx=0
            e_yy=0
            e_xy=0
            for k in range(0,m):
                q_x+=-hcond*dNdx[k]*T[icon[k,iel]]
                q_y+=-hcond*dNdy[k]*T[icon[k,iel]]
                e_xx+=dNdx[k]*u[icon[k,iel]]
                e_yy+=dNdy[k]*v[icon[k,iel]]
                e_xy+=0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]
            #end for

            qx[icon[kk,iel]]+=q_x
            qy[icon[kk,iel]]+=q_y
            exx[icon[kk,iel]]+=e_xx
            eyy[icon[kk,iel]]+=e_yy
            exy[icon[kk,iel]]+=e_xy
            ccc[icon[kk,iel]]+=1

        #end for k

    #end for iel

    qx/=ccc
    qy/=ccc
    exx/=ccc
    eyy/=ccc
    exy/=ccc

    print("exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
    print("eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
    print("exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
    print("qx  (m,M) %.4e %.4e " %(np.min(qx),np.max(qx)))
    print("qy  (m,M) %.4e %.4e " %(np.min(qy),np.max(qy)))

    #np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')
    #np.savetxt('temperature.ascii',np.array([x,y,T]).T,header='# x,y,T')
    #np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
    #np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
    #np.savetxt('heatflux.ascii',np.array([xc,yc,qx,qy]).T,header='# xc,yc,qx,qy')

    #####################################################################
    # plot of solution
    #####################################################################

    if istep%10==0:

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nnp):
          vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='viscosity' Format='ascii'> \n")
       #for iel in range(0,nel):
       #    vtufile.write("%10f \n" %eta_el[iel])
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='e (2nd inv.)' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    vtufile.write("%10e\n" % (np.sqrt(exx[iel]**2+eyy[iel]**2+2*exy[iel]**2)))
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/yr)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %(rho(rho0,alpha,T[i],T0)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='qx' Format='ascii'> \n")
       for i in range (0,nnp):
           vtufile.write("%e\n" % qx[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='qy' Format='ascii'> \n")
       for i in range (0,nnp):
           vtufile.write("%e\n" % qy[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range (0,nnp):
           vtufile.write("%e\n" % exx[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range (0,nnp):
           vtufile.write("%e\n" % eyy[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range (0,nnp):
           vtufile.write("%e\n" % exy[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='T (anal)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (T_anal(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--



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



#end for

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
