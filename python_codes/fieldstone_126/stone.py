import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing

#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8

def dNNVdr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8

def dNNVds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8

#------------------------------------------------------------------------------

sqrt3=np.sqrt(3.)
sqrt2=np.sqrt(2.)
eps=1.e-10 
cm=0.01
year=365.*24.*3600.

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2       # number of space dimensions
mV=9         # number of nodes per Q2 elt

Lx=2e-2
Ly=1e-2
nelx= 40
nely= 20

p_f=0.8
depth=50e3
background_pressure=3000*9.81*depth
p_0=p_f*background_pressure

tfinal=100e6*year

dt=1e-2*year
nstep=150
every=1

K_0=1e-16
porosity_0=1e-2
rho_m=1000
C_f=1e-9
porosity_max=5e-2
f_0=0.01

how_often=50

varphi=5e7  # phi in f evolution equation

alpha= 5e5  # alpha in dphi/dt equation

#------------------------------------------------------------------------------

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction
NV=nnx*nny    # number of nodes
nel=nelx*nely # number of elements, total
NfemPf=NV      # Total number of degrees of temperature freedom

# alphaT=1: implicit
# alphaT=0: explicit
# alphaT=0.5: Crank-Nicolson

alphaT=1 # for now keep it implicit?

#####################################################################

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

#####################################################################

stats_Pf_file=open('stats_Pf.ascii',"w")
stats_porosity_file=open('stats_porosity.ascii',"w")
stats_f_file=open('stats_source.ascii',"w")
stats_gradP_file=open('stats_gradP.ascii',"w")
stats_vel_file=open('stats_vel.ascii',"w")

#####################################################################

print ('nnx           =',nnx)
print ('nny           =',nny)
print ('NV            =',NV)
print ('nel           =',nel)
print ('nqperdim      =',nqperdim)
print ('dt(yr)        =',dt/year)
print ('nstep         =',nstep)
print ('p_bckgr (MPa) =',background_pressure/1e6)
print ('p_0 (MPa)     =',p_0/1e6)
print("-----------------------------")

#####################################################################
# grid point setup 
#####################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2
        yV[counter]=j*hy/2
        counter += 1
    #end for
#end for

print("mesh (%.3fs)" % (timing.time() - start))

#####################################################################
# connectivity
#####################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

print("connectivity (%.3fs)" % (timing.time() - start))

#####################################################################
# define boundary conditions
#####################################################################
start = timing.time()

bc_fixPf=np.zeros(NfemPf,dtype=np.bool)  
bc_valPf=np.zeros(NfemPf,dtype=np.float64) 

for i in range(0,NV):
    #left
    #if xV[i]/Lx<eps:
    #   bc_fixT[i]=True ; bc_valT[i]=2*p_0
    #right
    #if xV[i]/Lx>(1-eps):
    #   bc_fixT[i]=True ; bc_valT[i]=p_0
    #bottom
    if yV[i]/Ly<eps:
       bc_fixPf[i]=True ; bc_valPf[i]=p_0
    #top
    if yV[i]/Ly>(1-eps):
       bc_fixPf[i]=True ; bc_valPf[i]=p_0
#end for

print("boundary conditions (%.3fs)" % (timing.time() - start))

#####################################################################
# initial pressure
#####################################################################
start = timing.time()

T = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T[i]=p_0

print("initial temperature (%.3fs)" % (timing.time() - start))

#####################################################################
# create porosity and permeability nodal arrays
#####################################################################

permeability = np.zeros(NV,dtype=np.float64) # K
porosity     = np.zeros(NV,dtype=np.float64) # phi
dporosity_dt = np.zeros(NV,dtype=np.float64) # dphi/dt
f_source     = np.zeros(NV,dtype=np.float64) # f
strainrate   = np.zeros(NV,dtype=np.float64) # nodal strainrate

porosity[:]=porosity_0

f_source[:]=f_0 # min 0.01 , max 0.06

for i in range(0,NV):
    if abs(yV[i]-Ly/2)/Ly<0.15 and abs(xV[i]-Lx/2)/Lx<0.333:
       strainrate[i]=1e-16

#####################################################################
# create necessary arrays 
#####################################################################
start = timing.time()

N     = np.zeros(mV,dtype=np.float64)    # shape functions
dNdx  = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNdy  = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNdr  = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNds  = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
P_old = np.zeros(mV,dtype=np.float64)    # previously obtained pressure
NNNT    = np.zeros(mV,dtype=np.float64)  # shape functions 
dNNNTdx = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNTdy = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNTdr = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNTds = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
    
print("create few arrays (%.3fs)" % (timing.time() - start))

#==============================================================================
# time stepping loop
#==============================================================================

model_time=0.

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")

    #################################################################
    # update PDE coefficients
    #################################################################

    if istep%how_often==0:
       print('***** update porosity and permeability *****') 
       strainrate*=2
       dporosity_dt[:]=alpha*strainrate[:]
       porosity[:]+=dporosity_dt[:]*dt*how_often
       for i in range(0,NV):
           porosity[i]=min(porosity[i],porosity_max)
       permeability[:]=K_0*porosity[:]**3
       f_source[:]*=(1+varphi*strainrate[:]*dt*how_often)

    stats_porosity_file.write("%e %f %f \n" %(model_time/year,np.min(porosity),np.max(porosity))) 
    stats_porosity_file.flush()
    stats_f_file.write("%e %f %f \n" %(model_time/year,np.min(f_source),np.max(f_source))) 
    stats_f_file.flush()

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemPf,NfemPf),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemPf,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,mV),dtype=np.float64)           # gradient matrix B 
    N_mat = np.zeros((mV,1),dtype=np.float64)         # shape functions

    counterq=0
    for iel in range (0,nel):

        b_el=np.zeros(mV,dtype=np.float64)
        a_el=np.zeros((mV,mV),dtype=np.float64)
        Kd=np.zeros((mV,mV),dtype=np.float64)     # elemental diffusion matrix 
        MM=np.zeros((mV,mV),dtype=np.float64)     # elemental mass matrix 

        for k in range(0,mV):
            P_old[k]=T[iconV[k,iel]]
        #end for

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNT[0:mV]=NNV(rq,sq)
                dNNNTdr[0:mV]=dNNVdr(rq,sq)
                dNNNTds[0:mV]=dNNVds(rq,sq)
                N_mat[0:mV,0]=NNV(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0]+=dNNNTdr[k]*xV[iconV[k,iel]]
                    jcb[0,1]+=dNNNTdr[k]*yV[iconV[k,iel]]
                    jcb[1,0]+=dNNNTds[k]*xV[iconV[k,iel]]
                    jcb[1,1]+=dNNNTds[k]*yV[iconV[k,iel]]
                #end for
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                permeability_q=0.
                porosity_q=0.
                f_source_q=0.
                dporosity_dt_q=0.
                for k in range(0,mV):
                    dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                    dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                    B_mat[0,k]=dNNNTdx[k]
                    B_mat[1,k]=dNNNTdy[k]
                    permeability_q+=permeability[iconV[k,iel]]*NNNT[k]
                    porosity_q+=porosity[iconV[k,iel]]*NNNT[k]
                    dporosity_dt_q+=dporosity_dt[iconV[k,iel]]*NNNT[k]
                    f_source_q+=f_source[iconV[k,iel]]*NNNT[k]
                #end for

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*weightq*jcob * rho_m*C_f*porosity_q

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*weightq*jcob * rho_m*permeability_q

                # source term f
                b_el[:]+=NNNT[:]*f_source_q*weightq*jcob

                #source term rho_m*dphi/dt
                b_el-= NNNT[:]*dporosity_dt_q*weightq*jcob *rho_m

                # elemental matrix and rhs
                a_el+=MM+alphaT*Kd*dt
                b_el+=(MM-(1-alphaT)*Kd*dt).dot(P_old) 

                counterq+=1
            #end for jq
        #end for iq

        # apply boundary conditions
        for k1 in range(0,mV):
            m1=iconV[k1,iel]
            if bc_fixPf[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mV):
                   m2=iconV[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valPf[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valPf[m1]
            #end if
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mV):
            m1=iconV[k1,iel]
            for k2 in range(0,mV):
                m2=iconV[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    #end for iel
    
    print("     -> matrix (m,M) %.4e %.4e " %(np.min(A_mat),np.max(A_mat)))
    print("     -> rhs (m,M) %.4e %.4e " %(np.min(rhs),np.max(rhs)))

    print("build FEM matrix: %.3fs" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    np.savetxt('solution.ascii',np.array([xV,yV,T]).T,header='# x,y')

    print("     -> Pf (m,M) %e %e  (MPa)" %(np.min(T)/1e6,np.max(T)/1e6))

    stats_Pf_file.write("%e %f30 %f30 \n" %(model_time,np.min(T)/1e6,np.max(T)/1e6)) ; stats_Pf_file.flush()

    print("solve T time: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute nodal pressure gradient
    #####################################################################
    start = timing.time()

    NNNV     = np.zeros(mV,dtype=np.float64)    # shape functions
    dNNNVdx  = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
    dNNNVdr  = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)    # shape functions derivatives

    rVnodes=[-1,1,1,-1,0,1,0,-1,0]
    sVnodes=[-1,-1,1,1,-1,0,1,0,0]
    
    dPfdx_n = np.zeros(NV,dtype=np.float64)  
    dPfdy_n = np.zeros(NV,dtype=np.float64)  
    count = np.zeros(NV,dtype=np.int32)  

    for iel in range(0,nel):
        for i in range(0,mV):
            rq=rVnodes[i]
            sq=sVnodes[i]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcbi=np.linalg.inv(jcb)
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for
            dPfdx=0.
            dPfdy=0.
            for k in range(0,mV):
                dPfdx += dNNNVdx[k]*T[iconV[k,iel]]
                dPfdy += dNNNVdy[k]*T[iconV[k,iel]]
            #end for
            inode=iconV[i,iel]
            dPfdx_n[inode]+=dPfdx
            dPfdy_n[inode]+=dPfdy
            count[inode]+=1
        #end for
    #end for
    
    dPfdx_n/=count
    dPfdy_n/=count

    print("     -> dPfdx_n (m,M) %.6e %.6e " %(np.min(dPfdx_n),np.max(dPfdx_n)))
    print("     -> dPfdy_n (m,M) %.6e %.6e " %(np.min(dPfdy_n),np.max(dPfdy_n)))

    stats_gradP_file.write("%e %e %e %e %e\n" %(model_time/year,np.min(dPfdx_n),np.max(dPfdx_n),\
                                                                np.min(dPfdy_n),np.max(dPfdy_n))) 
    stats_gradP_file.flush()

    print("compute nodal press gradient: %.3f s" % (timing.time() - start))

    #################################################################
    #################################################################
    start = timing.time()

    u_darcy = np.zeros(NV,dtype=np.float64)
    v_darcy = np.zeros(NV,dtype=np.float64) 

    u_darcy[:]=-permeability[:]*dPfdx_n[:]
    v_darcy[:]=-permeability[:]*dPfdy_n[:]

    stats_vel_file.write("%e %e %e %e %e\n" %(model_time/year,np.min(u_darcy),np.max(u_darcy),\
                                                              np.min(v_darcy),np.max(v_darcy))) 
    stats_vel_file.flush()

    print("compute Darcy flow rate: %.3f s" % (timing.time() - start))

    #################################################################
    # visualisation 
    #################################################################

    if istep%every==0:

       start = timing.time()

       filename = 'solution_{:06d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pf' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%f30 \n" %(T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pf-p_0' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%f30 \n" %(T[i]-p_0))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='permeability' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e \n" %(permeability[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='porosity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%f30 \n" %(porosity[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dPf_dx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%f30 \n" %(dPfdx_n[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dPf_dy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%f30 \n" %(dPfdy_n[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Darcy velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(u_darcy[i],v_darcy[i],0))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Darcy velocity (cm/year)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(u_darcy[i]/cm*year,v_darcy[i]/cm*year,0))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strainrate' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e \n" %(strainrate[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='f_source' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e \n" %(f_source[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                       iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %23)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to files: %.3f s" % (timing.time() - start))

    #end if

    model_time+=dt
    print ("model_time=",model_time/year/1e6,'Myr')
    
#end for istep

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
