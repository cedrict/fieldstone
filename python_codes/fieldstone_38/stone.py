import numpy as np
import sys as sys
import time as clock
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve

###############################################################################

def basis_functions_V(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-s) 
    dNdr1=+0.25*(1.-s) 
    dNdr2=+0.25*(1.+s) 
    dNdr3=-0.25*(1.+s) 
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

def rho(rho0,alpha,T,T0):
    val=rho0*(1.-alpha*(T-T0)) - rho0
    return val

def T_anal(x,y):
    return T0*np.cos(np.pi/Lx*x)*np.sinh(np.pi/Lx*y)/np.sinh(np.pi/Lx*Ly)

###############################################################################

cm=0.01
year=365.25*3600*24
Myear=year*1e6
sqrt3=np.sqrt(3.)
eps=1.e-10 

print("*******************************")
print("********** stone 38 ***********")
print("*******************************")

ndim=2       # number of space dimensions
m_V=4        # number of nodes making up an element
ndof_V=2     # number of degrees of freedom per node
Lx=700e3     # horizontal extent of the domain 
Ly=Lx        # vertical extent of the domain 
alpha=2e-5   # thermal expansion coefficient
hcond=6.66   # thermal conductivity
hcapa=1200   # heat capacity
rho0=3700    # reference density
CFL=1        # CFL number 
gy=-10       # vertical component of gravity vector
eta=1e17*rho0
penalty=1e6*eta # penalty coefficient value
nstep=10   # maximum number of timestep   

dt_max=1e6*year

tol=1e-6

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   T0   = float(sys.argv[3])
else:
   nelx = 32
   nely = 32
   T0   = 1 # reftemperature

Ra=rho0*abs(gy)*alpha*T0*Lx**3/(hcond/rho0/hcapa)/eta
kappa=hcond/rho0/hcapa

hx=Lx/nelx
hy=Ly/nely

nn_V=(nelx+1)*(nely+1) # number of nodes
nel=nelx*nely # number of elements, total
Nfem_V=nn_V*ndof_V  # Total number of degrees of velocity freedom
Nfem_T=nn_V         # Total number of degrees of temperature freedom

vrms_file=open('vrms.ascii',"w")
Q_file=open('Q.ascii',"w")
dt_file=open('dt.ascii',"w")
Tavrg_file=open('Tavrg.ascii',"w")

debug=False

###############################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('hx=',hx)
print('hy=',hy)
print('nelx=',nelx)
print('nely=',nely)
print('Ra=',Ra)
print('kappa=',kappa)

###############################################################################
# grid point setup 
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates

counter=0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_V[counter]=i*hx
        y_V[counter]=j*hy
        counter += 1
    #end for
#end for

print("build mesh coordinates: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx + 1)
        icon_V[2,counter]=i+1+(j+1)*(nelx + 1)
        icon_V[3,counter]=i+(j+1)*(nelx + 1)
        counter += 1
    #end for
#end for

print("build mesh connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define velocity boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool) 
bc_val_V=np.zeros(Nfem_V,dtype=np.float64) 

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
    if x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]/Ly>(1-eps):
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
#end for

print("define vel boundary conditions: %.3f s" % (clock.time()-start))

#####################################################################
###############################################################################
# define temperature boundary conditions
###############################################################################
start=clock.time()

bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

for i in range(0,nn_V):
    if y_V[i]/Ly<eps:
       bc_fix_T[i]=True ; bc_val_T[i]=0
    if y_V[i]/Ly>(1-eps):
       bc_fix_T[i]=True ; bc_val_T[i]=T0*np.cos(np.pi*x_V[i]/Lx)
#end for

print("define T boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature
###############################################################################

T=np.zeros(nn_V,dtype=np.float64)
T_prev=np.zeros(nn_V,dtype=np.float64)

#T[:]=bc_fix_T[:]

#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================

jcb=np.zeros((2,2),dtype=np.float64)   
Tvect=np.zeros(4,dtype=np.float64)   
H=np.array([[1.,1.,0.],[1.,1.,0.],[0.,0.,0.]],dtype=np.float64) 
C=np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],dtype=np.float64) 
u_prev=np.zeros(nn_V,dtype=np.float64)   
v_prev=np.zeros(nn_V,dtype=np.float64)   

time=0.

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    #################################################################
    # build FE matrix
    #################################################################
    start=clock.time()

    A_fem=lil_matrix((Nfem_V,Nfem_V),dtype=np.float64)
    b_fem=np.zeros(Nfem_V,dtype=np.float64)    
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64)   # gradient matrix 

    for iel in range(0, nel):

        # set 2 arrays to 0 every loop
        b_el=np.zeros(m_V*ndof_V,dtype=np.float64)
        A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                # calculate basis functions & derivatives
                N_V=basis_functions_V(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)

                # calculate jacobian matrix
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                JxWq=np.linalg.det(jcb)*weightq 
                jcbi=np.linalg.inv(jcb)

                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
 
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                A_el+=B.T.dot(C.dot(B))*eta*JxWq

                for i in range(0,m_V):
                    b_el[2*i+1]+=N_V[i]*rho(rho0,alpha,Tq,T0)*gy*JxWq

            #end for
        #end for

        # integrate penalty term at 1 point
        rq=0.
        sq=0.
        weightq=2.*2.
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        JxWq=np.linalg.det(jcb)*weightq # avoid jcob
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

        for i in range(0,m_V):
            B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                              [0.       ,dNdy_V[i]],
                              [dNdy_V[i],dNdx_V[i]]]

        A_el+=B.T.dot(H.dot(B))*penalty*JxWq

        # apply boundary conditions
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                m1 =ndof_V*icon_V[k1,iel]+i1
                if bc_fix_V[m1]: 
                   ikk=ndof_V*k1+i1
                   aref=A_el[ikk,ikk]
                   for jkk in range(0,m_V*ndof_V):
                       b_el[jkk]-=A_el[jkk,ikk]*bc_val_V[m1]
                       A_el[ikk,jkk]=0.
                       A_el[jkk,ikk]=0.
                   #end for
                   A_el[ikk,ikk]=aref
                   b_el[ikk]=aref*bc_val_V[m1]
                #end if
            #end for
        #end for

        # assemble matrix and right hand side 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1+i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2+i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        A_fem[m1,m2]+=A_el[ikk,jkk]
                    #end for
                #end for
                b_fem[m1]+=b_el[ikk]
            #end for
        #end for

    #end for iel

    print("build stokes matrix & rhs: %.3f s" % (clock.time()-start))

    #################################################################
    # solve system
    #################################################################
    start = clock.time()

    sol=spsolve(A_fem.tocsr(),b_fem)

    print("solve V time: %.3f s" % (clock.time()-start))

    #################################################################
    # put solution into separate x,y velocity arrays
    #################################################################
    start=clock.time()

    u,v=np.reshape(sol,(nn_V,2)).T

    if debug:
       np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

    print("    -> u (m,M) %.2e %.2e (cm/yr)" %(np.min(u)/cm*year,np.max(u)/cm*year))
    print("    -> v (m,M) %.2e %.2e (cm/yr)" %(np.min(v)/cm*year,np.max(v)/cm*year))

    print("split solution: %.3f s" % (clock.time()-start))

    #################################################################
    # compute timestep
    #################################################################
    start = clock.time()

    dt1=CFL*hx/np.max(np.sqrt(u**2+v**2))

    dt2=CFL*hx**2/kappa

    dt=np.min([dt1,dt2,dt_max])

    print('    -> dt1= %e (Myear)' %(dt1/Myear))
    print('    -> dt2= %e (Myear)' %(dt2/Myear))
    print('    -> dt = %e (Myear)' %(dt/Myear))

    dt_file.write("%e %e %e %e\n" % (time,dt1/Myear,dt2/Myear,dt/Myear)) ; dt_file.flush()

    print("compute timestep: %.3f s" % (clock.time()-start))

    #################################################################
    # build temperature matrix
    #################################################################
    start = clock.time()

    A_fem=lil_matrix((Nfem_T,Nfem_T),dtype=np.float64)
    b_fem=np.zeros(Nfem_T,dtype=np.float64)       
    B=np.zeros((2,m_V),dtype=np.float64)  

    for iel in range (0,nel):

        b_el=np.zeros(m_V,dtype=np.float64)
        A_el=np.zeros((m_V,m_V),dtype=np.float64)
        Ka=np.zeros((m_V,m_V),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m_V,m_V),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m_V,m_V),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m_V):
            Tvect[k]=T[icon_V[k,iel]]
        #end for

        for iq in [-1,1]:
            for jq in [-1,1]:

                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                N_V=basis_functions_V(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                JxWq=np.linalg.det(jcb)*weightq 
                jcbi=np.linalg.inv(jcb)
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
                B[0,:]=dNdx_V[:]
                B[1,:]=dNdy_V[:]

                # compute mass matrix
                MM=np.outer(N_V,N_V)*rho0*hcapa*JxWq

                # compute diffusion matrix
                Kd=B.T.dot(B)*hcond*JxWq

                # compute advection matrix
                vel[0,0]=np.dot(N_V,u[icon_V[:,iel]])
                vel[0,1]=np.dot(N_V,v[icon_V[:,iel]])
                Ka=np.outer(N_V,vel.dot(B))*rho0*hcapa*JxWq

                A_el+=MM+(Ka+Kd)*dt*0.5
                b_el+=(MM-(Ka+Kd)*dt*0.5).dot(Tvect)

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
                       A_el[k1,k1]=Aref
                       b_el[k1]=Aref*bc_val_T[m1]
                    #end if
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
            #end for
        #end for

    #end for iel

    print("build energy matrix & rhs: %.3f s" % (clock.time()-start))

    #################################################################
    # solve system
    #################################################################
    start=clock.time()

    T=spsolve(A_fem.tocsr(),b_fem)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    if debug:
       np.savetxt('temperature.ascii',np.array([x_V,y_V,T]).T,header='# x,y,T')

    print("solve T time: %.3f s" % (clock.time()-start))

    #################################################################
    # compute vrms 
    #################################################################
    start=clock.time()

    vrms=0.
    Tavrg=0.
    for iel in range(0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.
                N_V=basis_functions_V(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)
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
            #end for
        #end for
    #end for
    vrms=np.sqrt(vrms/(Lx*Ly))
    Tavrg/=(Lx*Ly)

    vrms_file.write("%e %e\n" % (time/year,vrms/cm*year)) ; vrms_file.flush()
    Tavrg_file.write("%e %e\n" % (time/year,Tavrg)) ; Tavrg_file.flush()

    print("     -> time= %.6f ; vrms   = %.6f" %(time,vrms))

    print("compute vrms: %.3f s" % (clock.time()-start))

    ###########################################################################
    start = clock.time()

    T_diff=np.sum(abs(T-T_prev))/nn_V
    u_diff=np.sum(abs(u-u_prev))/nn_V
    v_diff=np.sum(abs(v-v_prev))/nn_V

    print("     -> <T_diff>= %.3e ; tol= %.3e" %(T_diff,tol))
    print("     -> <u_diff>= %.3e ; tol= %.3e" %(u_diff/cm*year,tol))
    print("     -> <v_diff>= %.3e ; tol= %.3e" %(v_diff/cm*year,tol))

    print("     -> T conv" , T_diff<tol*Tavrg)
    print("     -> u conv" , u_diff<tol*vrms)
    print("     -> v conv" , v_diff<tol*vrms)

    if T_diff<tol*Tavrg and u_diff<tol*vrms and v_diff<tol*vrms:
       print("convergence reached")
       break

    T_prev[:]=T[:]
    u_prev[:]=u[:]
    v_prev[:]=v[:]
    
    #####################################################################
    # retrieve pressure
    #####################################################################
    start = clock.time()

    xc=np.zeros(nel,dtype=np.float64)  
    yc=np.zeros(nel,dtype=np.float64)  
    p=np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):
        rq = 0.0
        sq = 0.0
        N_V=basis_functions_V(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
        eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
        p[iel]=-penalty*(exxq+eyyq)
        xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
        yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    #end for

    if debug:
       np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')

    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    print("compute pressure: %.3f s" % (clock.time()-start))

    #####################################################################
    # compute nodal field derivatives 
    #####################################################################
    start=clock.time()

    qx=np.zeros(nn_V,dtype=np.float64)  
    qy=np.zeros(nn_V,dtype=np.float64)  
    exx=np.zeros(nn_V,dtype=np.float64)  
    eyy=np.zeros(nn_V,dtype=np.float64)  
    exy=np.zeros(nn_V,dtype=np.float64)  
    ccc=np.zeros(nn_V,dtype=np.float64)  

    r_V=np.array([-1, 1, 1,-1],np.float64)
    s_V=np.array([-1,-1, 1, 1],np.float64)

    for iel in range(0,nel):
        for k in range(0,m_V):
            rq=r_V[k]
            sq=s_V[k]
            inode=icon_V[k,iel]
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            qx[inode]-=hcond*np.dot(dNdx_V,T[icon_V[:,iel]])
            qy[inode]-=hcond*np.dot(dNdy_V,T[icon_V[:,iel]])
            exx[inode]+=np.dot(dNdx_V,u[icon_V[:,iel]])
            eyy[inode]+=np.dot(dNdy_V,v[icon_V[:,iel]])
            exy[inode]+=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                        np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
            ccc[inode]+=1
        #end for k
    #end for iel

    qx/=ccc
    qy/=ccc
    exx/=ccc
    eyy/=ccc
    exy/=ccc

    print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
    print("     -> qx  (m,M) %.4e %.4e " %(np.min(qx),np.max(qx)))
    print("     -> qy  (m,M) %.4e %.4e " %(np.min(qy),np.max(qy)))

    if debug:
       np.savetxt('strainrate.ascii',np.array([x_V,y_V,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')
       np.savetxt('heatflux.ascii',np.array([x_V,y_V,qx,qy]).T,header='# x,y,qx,qy')

    print("compute sr & heat flux: %.3f s" % (clock.time()-start))

    #################################################################
    # compute Q= int_Lx qy dx at the surface
    #################################################################
    start = clock.time()

    Q=0
    for iel in range(0,nel):
        if y_V[icon_V[3,iel]]/Ly>1-eps: # top row elts
           Q+=(qy[icon_V[3,iel]]+qy[icon_V[2,iel]])/2*hx

    Q_file.write("%e %e\n" % (time/year,Q)) ; Q_file.flush()

    print("     -> Q= %f" %(Q))

    print("compute Q: %.3f s" % (clock.time()-start))

    #####################################################################
    # plot of solution
    #####################################################################
    start=clock.time()

    if istep%10==0:

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
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pressure' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity (cm/yr)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %(rho(rho0,alpha,T[i],T0)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" % (T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='qx' Format='ascii'> \n")
       for i in range (0,nn_V):
           vtufile.write("%e\n" % qx[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='qy' Format='ascii'> \n")
       for i in range (0,nn_V):
           vtufile.write("%e\n" % qy[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range (0,nn_V):
           vtufile.write("%e\n" % exx[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range (0,nn_V):
           vtufile.write("%e\n" % eyy[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range (0,nn_V):
           vtufile.write("%e\n" % exy[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (anal)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e\n" % (T_anal(x_V[i],y_V[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
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


    time+=dt

#end for

#==============================================================================
#==============================================================================
# end time stepping loop
#==============================================================================
#==============================================================================

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
