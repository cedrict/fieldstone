import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import temperature_dependent_variables

#------------------------------------------------------------------------------

def NNT(r,s,order):
    if order==1:
       N_0=0.25*(1.-r)*(1.-s)
       N_1=0.25*(1.+r)*(1.-s)
       N_2=0.25*(1.-r)*(1.+s)
       N_3=0.25*(1.+r)*(1.+s)
       return N_0,N_1,N_2,N_3
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
       return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8

def dNNTdr(r,s,order):
    if order==1:
       dNdr_0=-0.25*(1.-s)
       dNdr_1=+0.25*(1.-s)
       dNdr_2=-0.25*(1.+s)
       dNdr_3=+0.25*(1.+s)
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3
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
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8

def dNNTds(r,s,order):
    if order==1:
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.-r)
       dNds_3=+0.25*(1.+r)
       return dNds_0,dNds_1,dNds_2,dNds_3
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
       return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8

#------------------------------------------------------------------------------
# this a 2D domain but we are actually only interested in the 1D solution

sqrt3=np.sqrt(3.)
sqrt2=np.sqrt(2.)
eps=1.e-10 
cm=0.01
year=365.*24.*3600.

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2       # number of space dimensions
ndofT=1      # number of degrees of freedom per node


order=2
if order==1:
   m=4          # number of nodes making up an element
if order==2:
   m=9

nelx = 2
nely = 60
tfinal=100e6*year
Ttop=0+273
Tbottom=1350+273
dt=10e3*year
nstep=int(170e6*year/dt)
every=1000

#0: Parsons and Sclater (1977) 
#1: McKenzie et al (2015)

model=0

if model==0:
   option_k=0
   option_C_p=0
   option_rho=0
   Lx=2e3
   Ly=125e3

if model==1:
   option_k=1
   option_C_p=6
   option_rho=1
   Lx=2e3
   Ly=106e3


hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
NV=nnx*nny        # number of nodes
nel=nelx*nely     # number of elements, total
NfemT=NV*ndofT    # Total number of degrees of temperature freedom

# alphaT=1: implicit
# alphaT=0: explicit
# alphaT=0.5: Crank-Nicolson

alphaT=0.5

#####################################################################

if order==1:
   nqperdim=2
   qcoords=[-1./sqrt3,1./sqrt3]
   qweights=[1.,1.]

if order==2:
   nqperdim=3
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

#####################################################################

stats_T_file=open('stats_T.ascii',"w")

#####################################################################

print ('order      =',order)
print ('nnx        =',nnx)
print ('nny        =',nny)
print ('NV         =',NV)
print ('nel        =',nel)
print ('NfemT      =',NfemT)
print ('nqperdim   =',nqperdim)
print ('dt(yr)     =',dt/year)
print ('nstep      =',nstep)
print("-----------------------------")

#####################################################################
# generate ascii files for k, rho, CP as a fct of T
#####################################################################

hcond_file=open('hcond.ascii',"w")
hcapa_file=open('hcapa.ascii',"w")
rho_file=open('rho.ascii',"w")
for i in range(0,1000):
    T=Ttop+(Tbottom-Ttop)/999*i
    hcond=temperature_dependent_variables.heat_conductivity(T,0,0,option_k)
    hcapa=temperature_dependent_variables.heat_capacity(T,0,0,option_C_p)
    rho=temperature_dependent_variables.density(T,0,0,option_rho)
    hcond_file.write("%e %e \n" %(T,hcond)) 
    hcapa_file.write("%e %e \n" %(T,hcapa)) 
    rho_file.write("%e %e \n" %(T,rho)) 
#end for
hcond_file.close
hcapa_file.close
rho_file.close


#####################################################################
# grid point setup 
#####################################################################
start = timing.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx/order
        y[counter]=j*hy/order
        counter += 1
    #end for
#end for

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("mesh (%.3fs)" % (timing.time() - start))

#####################################################################
# connectivity
#####################################################################
start = timing.time()

icon =np.zeros((m,nel),dtype=np.int32)

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

#connectivity array for plotting
nel2=(nnx-1)*(nny-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int32)
counter = 0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        iconQ1[0,counter]=i+j*nnx
        iconQ1[1,counter]=i+1+j*nnx
        iconQ1[2,counter]=i+1+(j+1)*nnx
        iconQ1[3,counter]=i+(j+1)*nnx
        counter += 1 
    #end for
#end for

print("connectivity (%.3fs)" % (timing.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if y[i]/Ly<eps:
       bc_fixT[i]=True ; bc_valT[i]=Tbottom
    if y[i]/Ly>(1-eps):
       bc_fixT[i]=True ; bc_valT[i]=Ttop
#end for

print("boundary conditions (%.3fs)" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################
start = timing.time()

T = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T[i]=Tbottom


#np.savetxt('T_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

print("initial temperature (%.3fs)" % (timing.time() - start))

#####################################################################
# create necessary arrays 
#####################################################################
start = timing.time()

N     = np.zeros(m,dtype=np.float64)    # shape functions
dNdx  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
Tvectm1 = np.zeros(m,dtype=np.float64)   
NNNT    = np.zeros(m,dtype=np.float64)           # shape functions 
dNNNTdx = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTdy = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTdr = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTds = np.zeros(m,dtype=np.float64)           # shape functions derivatives
Tlithosphere=np.zeros((nny,nstep),dtype=np.float64)
    
print("create arrays (%.3fs)" % (timing.time() - start))

#==============================================================================
# time stepping loop
#==============================================================================

model_time=0.

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions

    counterq=0
    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m):
            Tvectm1[k]=T[icon[k,iel]]
        #end for

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNT[0:m]=NNT(rq,sq,order)
                dNNNTdr[0:m]=dNNTdr(rq,sq,order)
                dNNNTds[0:m]=dNNTds(rq,sq,order)
                N_mat[0:m,0]=NNT(rq,sq,order)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNNNTdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNNNTdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNNNTds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNNNTds[k]*y[icon[k,iel]]
                #end for

                # calculate the determinant of the jacobian
                jcob=np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                Tq=0.
                for k in range(0,m):
                    dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                    dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                    B_mat[0,k]=dNNNTdx[k]
                    B_mat[1,k]=dNNNTdy[k]
                    Tq+=NNNT[k]*T[icon[k,iel]]
                #end for

                rhoq=temperature_dependent_variables.density(Tq,0,0,option_rho)
                hcapaq=temperature_dependent_variables.heat_capacity(Tq,0,0,option_C_p)
                hcondq=temperature_dependent_variables.heat_conductivity(Tq,0,0,option_k)

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rhoq*hcapaq*weightq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcondq*weightq*jcob

                # elemental matrix and rhs
                a_el+=MM+alphaT*Kd*dt
                b_el+=(MM-(1-alphaT)*Kd*dt).dot(Tvectm1)

                counterq+=1
            #end for jq
        #end for iq

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
    
    print("     -> matrix (m,M) %.4e %.4e " %(np.min(A_mat),np.max(A_mat)))
    print("     -> rhs (m,M) %.4e %.4e " %(np.min(rhs),np.max(rhs)))

    print("build FEM matrix: %.3fs" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    stats_T_file.write("%e %e %e \n" %(model_time,np.min(T),np.max(T))) ; stats_T_file.flush()

    print("solve T time: %.3f s" % (timing.time() - start))

    #################################################################
    # save temperature at x=Lx/2 in array for later plot
    #################################################################

    counter=0
    for i in range(0,NV):
        if abs(x[i]-Lx/2)/Lx<0.0001:
           Tlithosphere[counter,istep]=T[i]
           counter+=1

    #################################################################
    # visualisation 
    #################################################################

    if istep%every==0:

       start = timing.time()

       filename = 'T_{:06d}.ascii'.format(istep) 
       np.savetxt(filename,np.array([y,T,x]).T,header='# y,T,x')

       filename = 'solution_{:06d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel2))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f \n" %(T[i]-273))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='k(T)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f \n" % temperature_dependent_variables.heat_conductivity(T[i],0,0,option_k))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Cp(T)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f \n" % temperature_dependent_variables.heat_capacity(T[i],0,0,option_C_p))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho(T)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f \n" % temperature_dependent_variables.density(T[i],0,0,option_rho))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if order==1:
          for iel in range (0,nel2):
              vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[3,iel],icon[2,iel]))
       if order==2:
          for iel in range (0,nel2):
              vtufile.write("%d %d %d %d \n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %9)
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

####################################################
#generate vtu file for lithosphere temperature
#aspect ratio 2:1
####################################################

npts=nstep*nny
ncell=(nny-1)*(nstep-1)
filename = 'Tlithosphere.vtu'.format(istep) 
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(npts,ncell))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for j in range(0,nny):
    for i in range(0,nstep):
        vtufile.write("%e %e %e \n" %(i*dt/model_time*2,float(j)/float(nny-1),0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
for j in range(0,nny):
    for i in range(0,nstep):
        vtufile.write("%10f \n" %(Tlithosphere[j,i]-273))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='k(T)' Format='ascii'> \n")
for j in range(0,nny):
    for i in range(0,nstep):
        vtufile.write("%10f \n" %(temperature_dependent_variables.heat_conductivity(Tlithosphere[j,i],0,0,option_k)))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Cp(T)' Format='ascii'> \n")
for j in range(0,nny):
    for i in range(0,nstep):
        vtufile.write("%10f \n" %(temperature_dependent_variables.heat_capacity(Tlithosphere[j,i],0,0,option_C_p)))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='rho(T)' Format='ascii'> \n")
for j in range(0,nny):
    for i in range(0,nstep):
        vtufile.write("%10f \n" %(temperature_dependent_variables.density(Tlithosphere[j,i],0,0,option_rho)))
vtufile.write("</DataArray>\n")

vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
counter = 0
for j in range(0,nny-1):
    for i in range(0,nstep-1):
        vtufile.write("%d %d %d %d \n" %(i+j*nstep,i+1+j*nstep,i+1+(j+1)*nstep,i+(j+1)*nstep))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,ncell):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,ncell):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
