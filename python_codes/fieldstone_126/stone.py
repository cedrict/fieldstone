import numpy as np
import sys as sys
import time as timing
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
from numba import jit



sqrt3=np.sqrt(3.)
sqrt2=np.sqrt(2.)
eps=1.e-8 
m=0.01
year=365.*24.*3600.
percent=0.01

###############################################################################

@jit(nopython=True)
def tau_fct(y,srcoeff):
    if y<5.e3:
       val=1e4+y/5e3*(1e5-1e4)
    elif y<7e3:
       val=1e5-(y-5e3)/(7e3-5e3)*(1e5-7000)
    elif y<8e3:
       val=7000
    else:
       val=7000+(y-8000)/(10e3-8e3)*(50000-7000)
    return val*year*srcoeff

@jit(nopython=True)
def Kx_fct(xq,tauq,time):
    return k_0/rho_m/g*np.exp(-abs(xq)/L)*np.exp(-3*time/tauq)

@jit(nopython=True)
def Ky_fct(xq,tauq,time):
    return 100*k_0/rho_m/g*np.exp(-abs(xq)/L)*np.exp(-3*time/tauq)

@jit(nopython=True)
def Phi_fct(xq,tauq,time):
    return Phi_0*np.exp(-abs(xq)/L)*np.exp(-time/tauq)

@jit(nopython=True)
def dPhidt_fct(xq,tauq,time):
    return -Phi_0/tauq*np.exp(-abs(xq)/L)*np.exp(-time/tauq)

###############################################################################

@jit(nopython=True)
def NN(rq,sq):
    N_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    N_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    N_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    N_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    N_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    N_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    N_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    N_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    N_8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

@jit(nopython=True)
def dNNdr(rq,sq):
    dNdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNdr_8=       (-2.*rq) *    (1.-sq**2)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,\
                     dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

@jit(nopython=True)
def dNNds(rq,sq):
    dNds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNds_8=     (1.-rq**2) *       (-2.*sq)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,\
                     dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)

###############################################################################

print("-----------------------------")
print("----------stone 126----------")
print("-----------------------------")

ndim=2       # number of space dimensions
mP=9         # number of nodes per Q2 elt

Lx=4e3
Ly=10e3
nelx= 20
nely= int(nelx*Ly/Lx)

dt=0.05*year
nstep=5000
every=10

Phi_0=10*percent
rho_m=880
C=1e-9
L=100
g=10
rho_litho=2800

model=3

if model==1: 
   k_0=1e-9
   sealing_rate_coeff=1
   tfinal=270*year
if model==2: 
   k_0=1e-9
   sealing_rate_coeff=0.1
   tfinal=270*year
if model==3: 
   k_0=1e-9
   sealing_rate_coeff=0.01
   tfinal=108*year
if model==4: 
   k_0=1e-8
   sealing_rate_coeff=0.1
   tfinal=270*year

###############################################################################
###############################################################################

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction
NP=nnx*nny    # number of nodes
nel=nelx*nely # number of elements, total
NfemP=NP      # Total number of degrees of pressure freedom

# alpha=1: implicit
# alpha=0: explicit
# alpha=0.5: Crank-Nicolson

alpha=0.5

###############################################################################

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

###############################################################################

stats_p_file=open('stats_p.ascii',"w")
stats_gradp_file=open('stats_gradp.ascii',"w")
stats_vel_file=open('stats_vel.ascii',"w")
stats_cfl_file=open('stats_cfl.ascii',"w")

###############################################################################

print('nelx      =',nelx)
print('nely      =',nely)
print('NP        =',NP)
print('nel       =',nel)
print('nqperdim  =',nqperdim)
print('dt(yr)    =',dt/year)
print('nstep     =',nstep)
print('C         =',C)
print('L         =',L)
print('g         =',g)
print('Phi_0     =',Phi_0)
print('rho_m     =',rho_m)
print('rho_litho =',rho_litho)
print("-----------------------------")

###############################################################################
# grid point setup 
###############################################################################
start=timing.time()

xP=np.zeros(NP,dtype=np.float64)  # x coordinates
yP=np.zeros(NP,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xP[counter]=i*hx/2 -Lx/2
        yP[counter]=j*hy/2
        counter += 1
    #end for
#end for

print("build mesh (%.3fs)" % (timing.time()-start))

###############################################################################
# connectivity
###############################################################################
start=timing.time()

iconP=np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconP[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconP[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconP[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconP[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconP[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconP[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconP[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconP[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

print("build connectivity (%.3fs)" % (timing.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=timing.time()

bc_fixP=np.zeros(NfemP,dtype=bool)  
bc_valP=np.zeros(NfemP,dtype=np.float64) 

for i in range(0,NP):
    #left
    #if abs(xP[i]-Lx/2)/Lx<eps:
    #   bc_fixP[i]=True ; bc_valP[i]=0
    #right
    #if abs(xP[i]+Lx/2)/Lx<eps:
    #   bc_fixP[i]=True ; bc_valP[i]=0
    #bottom
    if yP[i]/Ly<eps:
       bc_fixP[i]=True ; bc_valP[i]=g*Ly*(rho_litho-rho_m)*np.exp(-abs(xP[i])/L)
    #top
    if yP[i]/Ly>(1-eps):
       bc_fixP[i]=True ; bc_valP[i]=0
#end for

print("boundary conditions (%.3fs)" % (timing.time()-start))

#==============================================================================
# time stepping loop
#==============================================================================
p=np.zeros(NP,dtype=np.float64)
dNNNdx=np.zeros(mP,dtype=np.float64) # shape functions derivatives
dNNNdy=np.zeros(mP,dtype=np.float64) # shape functions derivatives
jcb=np.array([[hx/2,0],[0,hy/2]],dtype=np.float64) 
jcbi=np.array([[2/hx,0],[0,2/hy]],dtype=np.float64) 
jcob=hx*hy/4

model_time=0.

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    ###########################################################################
    start = timing.time()

    A_mat=np.zeros((NfemP,NfemP),dtype=np.float64) # FE matrix 
    rhs=np.zeros(NfemP,dtype=np.float64)           # FE rhs 
    B_mat=np.zeros((2,mP),dtype=np.float64)        # gradient matrix B 
    N_mat=np.zeros((mP,1),dtype=np.float64)        # shape functions
    p_old=np.zeros(mP,dtype=np.float64)            # current pressure

    counterq=0
    for iel in range (0,nel):

        b_el=np.zeros(mP,dtype=np.float64)
        a_el=np.zeros((mP,mP),dtype=np.float64)
        Kd=np.zeros((mP,mP),dtype=np.float64)     # elemental diffusion matrix 
        MM=np.zeros((mP,mP),dtype=np.float64)     # elemental mass matrix 

        for k in range(0,mP):
            p_old[k]=p[iconP[k,iel]]
        #end for

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNN=NN(rq,sq)
                dNNNdr=dNNdr(rq,sq)
                dNNNds=dNNds(rq,sq)
                N_mat[:,0]=NN(rq,sq)

                # calculate jacobian matrix
                #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                #for k in range(0,mP):
                #    jcb[0,0]+=dNNNdr[k]*xP[iconP[k,iel]]
                #    jcb[0,1]+=dNNNdr[k]*yP[iconP[k,iel]]
                #    jcb[1,0]+=dNNNds[k]*xP[iconP[k,iel]]
                #    jcb[1,1]+=dNNNds[k]*yP[iconP[k,iel]]
                #end for
                #jcob=np.linalg.det(jcb)
                #jcbi=np.linalg.inv(jcb)
                #print(jcb,hx/2,hy/2,jcob,hx*hy/4)

                # compute dNdx & dNdy
                for k in range(0,mP):
                    dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
                    dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]
                    B_mat[0,k]=dNNNdx[k]
                    B_mat[1,k]=dNNNdy[k]
                #end for

                xq=NNN.dot(xP[iconP[:,iel]])
                yq=NNN.dot(yP[iconP[:,iel]])

                tauq=tau_fct(yq,sealing_rate_coeff)

                # compute mass matrix
                Phi_q=Phi_fct(xq,tauq,model_time)
                MM=N_mat.dot(N_mat.T)*weightq*jcob * C*Phi_q

                # compute diffusion matrix
                Kx_q=Kx_fct(xq,tauq,model_time)
                Ky_q=Ky_fct(xq,tauq,model_time)
                Kmat=np.array([[Kx_q,0],[0,Ky_q]],dtype=np.float64) 
                Kd=B_mat.T.dot(Kmat.dot(B_mat))*weightq*jcob 

                # source term f
                dPhidt_q=dPhidt_fct(xq,tauq,model_time)
                b_el[:]-=NNN[:]*dPhidt_q*weightq*jcob

                #print(xq,yq,tauq,Phi_q,Kx_q,Ky_q,dPhidt_q)

                # elemental matrix and rhs
                a_el+=MM+alpha*Kd*dt
                b_el+=(MM-(1-alpha)*Kd*dt).dot(p_old) 

                counterq+=1
            #end for jq
        #end for iq

        # apply boundary conditions
        for k1 in range(0,mP):
            m1=iconP[k1,iel]
            if bc_fixP[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mP):
                   m2=iconP[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valP[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valP[m1]
            #end if
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mP):
            m1=iconP[k1,iel]
            for k2 in range(0,mP):
                m2=iconP[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    #end for iel
    
    #print("     -> matrix (m,M) %.4e %.4e " %(np.min(A_mat),np.max(A_mat)))
    #print("     -> rhs (m,M) %.4e %.4e " %(np.min(rhs),np.max(rhs)))

    print("build FEM matrix: %.3fs" % (timing.time() - start))

    ###########################################################################
    # solve system
    ###########################################################################
    start = timing.time()

    p=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    #np.savetxt('solution.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("     -> p (m,M) %e %e  (MPa)" %(np.min(p)/1e6,np.max(p)/1e6))

    stats_p_file.write("%e %e %e \n" %(model_time,np.min(p)/1e6,np.max(p)/1e6)) 
    stats_p_file.flush()

    print("solve T time: %.3f s" % (timing.time() - start))

    ###########################################################################
    # compute nodal pressure gradient
    ###########################################################################
    start = timing.time()

    rnodes=[-1,1,1,-1,0,1,0,-1,0]
    snodes=[-1,-1,1,1,-1,0,1,0,0]
    
    dpdx_n=np.zeros(NP,dtype=np.float64)  
    dpdy_n=np.zeros(NP,dtype=np.float64)  
    count=np.zeros(NP,dtype=np.int32)  

    for iel in range(0,nel):
        for i in range(0,mP):
            rq=rnodes[i]
            sq=snodes[i]
            NNN=NN(rq,sq)
            dNNNdr=dNNdr(rq,sq)
            dNNNds=dNNds(rq,sq)
            #jcb=np.zeros((ndim,ndim),dtype=np.float64)
            #for k in range(0,mP):
            #    jcb[0,0]+=dNNNdr[k]*xP[iconP[k,iel]]
            #    jcb[0,1]+=dNNNdr[k]*yP[iconP[k,iel]]
            #    jcb[1,0]+=dNNNds[k]*xP[iconP[k,iel]]
            #    jcb[1,1]+=dNNNds[k]*yP[iconP[k,iel]]
            #end for
            #jcbi=np.linalg.inv(jcb)
            for k in range(0,mP):
                dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
                dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]
            #end for

            dpdx=dNNNdx.dot(p[iconP[:,iel]])
            dpdy=dNNNdy.dot(p[iconP[:,iel]])

            inode=iconP[i,iel]
            dpdx_n[inode]+=dpdx
            dpdy_n[inode]+=dpdy
            count[inode]+=1
        #end for
    #end for
    
    dpdx_n/=count
    dpdy_n/=count

    print("     -> dpdx_n (m,M) %.6e %.6e " %(np.min(dpdx_n),np.max(dpdx_n)))
    print("     -> dpdy_n (m,M) %.6e %.6e " %(np.min(dpdy_n),np.max(dpdy_n)))

    stats_gradp_file.write("%e %e %e %e %e\n" %(model_time/year,np.min(dpdx_n),np.max(dpdx_n),\
                                                                np.min(dpdy_n),np.max(dpdy_n))) 
    stats_gradp_file.flush()

    print("compute nodal press gradient: %.3f s" % (timing.time()-start))

    ###########################################################################
    # computing nodal fields
    ###########################################################################
    start = timing.time()

    Phi_m=np.zeros(NP,dtype=np.float64)
    Kx=np.zeros(NP,dtype=np.float64)
    Ky=np.zeros(NP,dtype=np.float64) 
    tau=np.zeros(NP,dtype=np.float64) 
    P_litho=np.zeros(NP,dtype=np.float64) 
    P_hydro=np.zeros(NP,dtype=np.float64) 
    for i in range(0,NP):
        tau[i]=tau_fct(yP[i],sealing_rate_coeff)
        Kx[i]=Kx_fct(xP[i],tau[i],model_time)
        Ky[i]=Ky_fct(xP[i],tau[i],model_time)
        Phi_m[i]=Phi_fct(xP[i],tau[i],model_time)
        P_litho[i]=rho_litho*g*(Ly-yP[i])
        P_hydro[i]=rho_m*g*(Ly-yP[i])

    u_darcy=np.zeros(NP,dtype=np.float64)
    v_darcy=np.zeros(NP,dtype=np.float64) 
    u_darcy[:]=-Kx[:]*dpdx_n[:]
    v_darcy[:]=-Ky[:]*dpdy_n[:]

    print("     -> u (m,M) %.6e %.6e " %(np.min(u_darcy),np.max(u_darcy)))
    print("     -> v (m,M) %.6e %.6e " %(np.min(v_darcy),np.max(v_darcy)))

    stats_vel_file.write("%e %e %e %e %e\n" %(model_time/year,np.min(u_darcy),np.max(u_darcy),\
                                                              np.min(v_darcy),np.max(v_darcy))) 
    stats_vel_file.flush()

    print("compute Darcy flow rate: %.3f s" % (timing.time()-start))

    ###########################################################################
    # compute CFL_nb
    ###########################################################################

    CFL_nb=dt/min(hx,hy)*np.max(np.sqrt(u_darcy**2+v_darcy**2)) 

    print('     -> CFL_nb=',CFL_nb)

    stats_cfl_file.write("%e %e \n" %(model_time/year,CFL_nb)) ; stats_cfl_file.flush()

    ###########################################################################
    # visualisation 
    ###########################################################################

    if istep%every==0:

       start=timing.time()

       filename = 'solution_{:06d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e %e %e \n" %(xP[i],yP[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(p[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Ovp' Format='ascii'> \n")
       for i in range(0,NP):
           if abs(yP[i]-Ly)/Ly<eps:
              vtufile.write("%e \n" %(0))
           else:
              vtufile.write("%e \n" %(p[i]/(P_litho[i]-P_hydro[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='P-Plitho' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(p[i]+P_hydro[i]-P_litho[i]))
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='bc_valP' Format='ascii'> \n")
       #for i in range(0,NP):
       #    vtufile.write("%e \n" %(bc_valP[i]))
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='permeability Kx' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(Kx[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='permeability Ky' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(Ky[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='porosity Phi_m' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(Phi_m[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dp_dx' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(dpdx_n[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dp_dy' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(dpdy_n[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='P_litho' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(P_litho[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='P_hydro' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(P_hydro[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Darcy velocity' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e %e %e \n" %(u_darcy[i],v_darcy[i],0))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Darcy velocity (m/year)' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e %e %e \n" %(u_darcy[i]/m*year,v_darcy[i]/m*year,0))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sealing rate tau' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%e \n" %(tau[i]/year))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(iconP[0,iel],iconP[1,iel],iconP[2,iel],\
                                                          iconP[3,iel],iconP[4,iel],iconP[5,iel],\
                                                          iconP[6,iel],iconP[7,iel],iconP[8,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*9))
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

       print("export to files: %.3f s" % (timing.time()-start))

    #end if

    model_time+=dt
    print ("model_time=",model_time/year,'yr')
    
#end for istep

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
