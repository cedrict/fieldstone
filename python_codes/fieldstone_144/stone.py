import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse import lil_matrix
import time as clock

###############################################################################
# basis functions
###############################################################################

def basis_functions_T(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_T_dr(r,s):
    dNdr0=-0.25*(1.-s) 
    dNdr1=+0.25*(1.-s) 
    dNdr2=+0.25*(1.+s) 
    dNdr3=-0.25*(1.+s) 
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_T_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################
# constants
###############################################################################

cm=1e-2
km=1e3
eps=1e-9
year=365.25*24*3600
sqrt2=np.sqrt(2)
ndim=2   # number of dimensions

###############################################################################

print("*******************************")
print("********** stone 144 **********")
print("*******************************")

#####################################################################
# TODO:
# - make function for velocity profile
# - stretch mesh in vertical direction - careful with jacobian!!
# - matrix does not change in time, precompute!
# - benchmark advection and diffusion
# - use Kelvin !
# - better ramps element size 
# hx,hy fcts of element id
# choose hy in the channel, hy outside the channel -> compute nely from that 
# poiseuille flow in narrowing channel 
# powerlaw or Herschel-Bulkley vel profile ?
###############################################################################
Lx=100*km          # horizontal dimension of domain
Ly=25*km           # vertical dimension of domain
Tsurf=0            # temperature at the top
Tbase=1480         # temperature at the bottom
Tintrusion=1250    # temperature prescribed at intrusion
Hmagma=1000        # thickness of magma channel
Umagma=20*cm/year  # maximum velocity
Ymagma=10.5*km     # channel middle depth 
hcapa_rock=800     # heat capacity
hcond_rock=1.5     # heat conductivity
rho_rock=2900      # density
hcapa_magma=1100   # heat capacity
hcond_magma=1.5    # heat conductivity
rho_magma=2700     # density
tfinal=10e6*year   # duration of simulation
nelx = 64          # number of elements in x direction
nely = 128         # number of elements in y direction
nstep= 2000        # maximum number of time steps
CFL_nb = 1         # CFL number
supg_type=0        # toggle switch for SUPG advection stabilisation
every=25           # how often outputs are generated
###############################################################################

dojcb=False

nel=nelx*nely  # total number of elements
nnx=nelx+1     # number of nodes, x direction
nny=nely+1     # number of nodes, y direction
nn_T=nnx*nny   # number of nodes
Nfem_T=nn_T    # nb of temperature dofs

m_T=4             # number of velocity nodes making up an element
r_T=[-1,+1,-1,+1]
s_T=[-1,-1,+1,+1]

nq_per_dim=2
qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
qweights=[1.,1.]

debug=False

###############################################################################
# open output files

dt_file=open('dt.ascii',"w")
Tstats_file=open('stats_T.ascii',"w")

###############################################################################

kappa_rock  =hcond_rock /rho_rock /hcapa_rock
kappa_magma =hcond_magma/rho_magma/hcapa_magma

print ('nnx         =',nnx)
print ('nny         =',nny)
print ('nel         =',nel)
print ('Nfem_T      =',Nfem_T)
print ('tfinal      =',tfinal/year,' year')
print ('kappa_rock  =',kappa_rock)
print ('kappa_magma =',kappa_magma)
print ('Umagma      =',Umagma/cm*year,'cm/year')
print ('Hmagma      =',Hmagma,'m')
print ('-----------------------------')

#################################################################
# build velocity nodes coordinates 
#################################################################
start=clock.time()

x_T=np.zeros(nn_T,dtype=np.float64)  # x coordinates
y_T=np.zeros(nn_T,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        x_T[counter]=i*Lx/nelx
        y_T[counter]=j*Ly/nely
        counter+=1
    #end for
#end for

if debug: np.savetxt('mesh.ascii',np.array([x_T,y_T]).T)

print("build grid: %.3f s" % (clock.time()-start))

###############################################################################

beta1=0.125
beta2=0.5
y1=Ymagma-Hmagma/2
y2=Ymagma+Hmagma/2
a=2

x1=y1/a
x2=(y2-Ly)/a+Ly

#for i in range(0,nn_T):
#    if y_T[i]<x1:
#       y_T[i]=a*y_T[i]
#    elif y_T[i]<x2:
#       y_T[i]=Hmagma/(x2-x1)*(y_T[i]-x1)+y1
#    else:
#       y_T[i]=a*(y_T[i]-Ly)+Ly

np.savetxt('grid_stretched',np.array([x_T,y_T]).T)

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_T=np.zeros((m_T,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_T[0,counter]=i+j*(nelx+1)
        icon_T[1,counter]=i+1+j*(nelx+1)
        icon_T[2,counter]=i+1+(j+1)*(nelx+1)
        icon_T[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

if debug:
   for iel in range (0,nel):
       print ("iel=",iel)
       print ("node 0",icon_T[0,iel],"at pos.",x_T[icon_T[0,iel]],y_T[icon_T[0,iel]])
       print ("node 1",icon_T[1,iel],"at pos.",x_T[icon_T[1,iel]],y_T[icon_T[1,iel]])
       print ("node 2",icon_T[2,iel],"at pos.",x_T[icon_T[2,iel]],y_T[icon_T[2,iel]])
       print ("node 3",icon_T[3,iel],"at pos.",x_T[icon_T[3,iel]],y_T[icon_T[3,iel]])

print("build mesh connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# compute hx,hy per element
###############################################################################

hx=np.zeros(nel,dtype=np.float64) 
hy=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    hx[iel]=x_T[icon_T[2,iel]]-x_T[icon_T[0,iel]]
    hy[iel]=y_T[icon_T[2,iel]]-y_T[icon_T[0,iel]]

#print(hx)
#print(hy)

###############################################################################
# define temperature boundary conditions
###############################################################################
start=clock.time()

bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

for i in range(0,nn_T):
    if y_T[i]/Ly<eps:
       bc_fix_T[i]=True ; bc_val_T[i]=Tbase
    if y_T[i]/Ly>(1-eps):
       bc_fix_T[i]=True ; bc_val_T[i]=Tsurf
    if x_T[i]/Lx<eps and abs(y_T[i]-Ymagma)<Hmagma/2:
       bc_fix_T[i]=True ; bc_val_T[i]=Tintrusion

print("temperature b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature
###############################################################################
start=clock.time()

T=np.zeros(nn_T,dtype=np.float64)
T_init=np.zeros(nn_T,dtype=np.float64)

for i in range(0,nn_T):
    T[i]=(Tsurf-Tbase)/Ly*y_T[i]+Tbase

T_init[:]=T[:]

print("initial temperature: %.3f s" % (clock.time()-start))

#################################################################
# compute area of elements
# not strictly necessary, but good test of basis functions
#################################################################
start = clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
jcbi=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_T=basis_functions_T(rq,sq)
            dNdr_T=basis_functions_T_dr(rq,sq)
            dNds_T=basis_functions_T_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
            jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
            jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
            jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.4e " %(area.sum()))
print("     -> real total area %.6f " %(Lx*Ly))

print("compute elements areas: %.3f s" % (clock.time() - start))

#################################################################
# define velocity field 
#################################################################
start = clock.time()

u=np.zeros(nn_T,dtype=np.float64) # x-component velocity
v=np.zeros(nn_T,dtype=np.float64) # y-component velocity

for i in range(0,nn_T):
    if abs(y_T[i]-Ymagma)<Hmagma/2:
       Y=y_T[i]-Ymagma+Hmagma/2
       u[i]=-(Y**2-Y*Hmagma)/Hmagma**2*4 * Umagma

print("define velocity field: %.3f s" % (clock.time() - start))

#################################################################
# define elemental parameters 
#################################################################
start = clock.time()
    
x_e=np.zeros(nel,dtype=np.float64)
y_e=np.zeros(nel,dtype=np.float64)
hcond=np.zeros(nel,dtype=np.float64)
hcapa=np.zeros(nel,dtype=np.float64)
rho=np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    x_e[iel]=np.sum(x_T[icon_T[:,iel]])/m_T
    y_e[iel]=np.sum(y_T[icon_T[:,iel]])/m_T
    if abs(y_e[iel]-Ymagma)<Hmagma/2:
       hcapa[iel]=hcapa_magma
       hcond[iel]=hcond_magma
       rho[iel]=rho_magma
    else:
       hcapa[iel]=hcapa_rock
       hcond[iel]=hcond_rock
       rho[iel]=rho_rock

print("elemental params: %.3f s" % (clock.time() - start))

#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================
Tvect=np.zeros(m_T,dtype=np.float64)   

time=0

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    #################################################################
    # compute timestep value
    #################################################################

    dt1=CFL_nb*min(hx) /np.max(np.sqrt(u**2))
    dt2=CFL_nb*min(min(hx),min(hy))**2 / max(kappa_rock,kappa_magma)
    dt=np.min([dt1,dt2])
    time+=dt

    print('     -> dt1  = %.6f yr' %(dt1/year))
    print('     -> dt2  = %.6f yr' %(dt2/year))
    print('     -> dt  = %.6f yr' %(dt/year))
    print('     -> time= %.6f yr' %(time/year))

    dt_file.write("%10e %10e %10e %10e\n" % (time/year,dt1/year,dt2/year,dt/year))
    dt_file.flush()

    #################################################################
    # build temperature matrix
    #################################################################
    start=clock.time()

    A_fem=np.zeros((Nfem_T,Nfem_T),dtype=np.float64) # FE matrix 
    b_fem=np.zeros(Nfem_T,dtype=np.float64)            # FE rhs 
    B=np.zeros((2,m_T),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m_T,1),dtype=np.float64)         # shape functions
    N_mat_supg = np.zeros((m_T,1),dtype=np.float64)         # shape functions
    tau_supg = np.zeros(nel*nq_per_dim**ndim,dtype=np.float64)

    counterq=0   
    for iel in range (0,nel):

        b_el=np.zeros(m_T,dtype=np.float64)
        A_el=np.zeros((m_T,m_T),dtype=np.float64)
        Ka=np.zeros((m_T,m_T),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m_T,m_T),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m_T,m_T),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        Tvect[:]=T[icon_T[:,iel]]

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_T=basis_functions_T(rq,sq)
                N_mat[:,0]=N_T[:]

                if dojcb:
                   dNdr_T=basis_functions_T_dr(rq,sq)
                   dNds_T=basis_functions_T_ds(rq,sq)
                   jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
                   jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
                   jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
                   jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
                   jcbi=np.linalg.inv(jcb)
                   JxWq=np.linalg.det(jcb)*weightq
                else:
                   jcbi[0,0]=2./hx[iel]
                   jcbi[1,1]=2./hy[iel]
                   JxWq=hx[iel]*hy[iel]/4*weightq

                xq=np.dot(N_T,x_T[icon_T[:,iel]])
                yq=np.dot(N_T,y_T[icon_T[:,iel]])

                vel[0,0]=np.dot(N_T,u[icon_T[:,iel]])
                vel[0,1]=np.dot(N_T,v[icon_T[:,iel]])

                dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
                dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T

                B[0,:]=dNdx_T[:]
                B[1,:]=dNdy_T[:]

                if supg_type==0:
                   tau_supg[counterq]=0.
                elif supg_type==1:
                      tau_supg[counterq]=(hx*sqrt2)/2/np.sqrt(vel[0,0]**2+vel[0,1]**2)
                elif supg_type==2:
                      tau_supg[counterq]=(hx*sqrt2)/np.sqrt(vel[0,0]**2+vel[0,1]**2)/sqrt15
                else:
                   exit("supg_type: wrong value")
    
                N_mat_supg=N_mat+tau_supg[counterq]*np.transpose(vel.dot(B))

                # compute mass matrix
                MM+=N_mat_supg.dot(N_mat.T)*rho[iel]*hcapa[iel]*JxWq

                # compute diffusion matrix
                Kd+=B.T.dot(B)*hcond[iel]*JxWq

                # compute advection matrix
                Ka+=N_mat_supg.dot(vel.dot(B))*rho[iel]*hcapa[iel]*JxWq

                counterq+=1

            #end for
        #end for

        A_el=MM+0.5*(Ka+Kd)*dt

        b_el=(MM-0.5*(Ka+Kd)*dt).dot(Tvect)

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

        # assemble matrix A_fem and right hand side b_fem
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            for k2 in range(0,m_T):
                m2=icon_T[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        #end for

    #end for iel

    #print("     -> tau_supg (m,M) %e %e " %(np.min(tau_supg),np.max(tau_supg)))

    print("build FE matrix : %.3f s" % (clock.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     T (m,M): %.4f %.4f " %(np.min(T),np.max(T)))

    Tstats_file.write("%6e %6e %6e\n" % (time,np.min(T),np.max(T)))
    Tstats_file.flush()

    print("solve linear system: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute heat flux
    ###########################################################################
    start=clock.time()

    qx=np.zeros(nel,dtype=np.float64)
    qy=np.zeros(nel,dtype=np.float64)

    for iel in range(0,nel):
        if dojcb:
           rq = 0.0 
           sq = 0.0 
           dNdr_T=basis_functions_T_dr(rq,sq)
           dNds_T=basis_functions_T_ds(rq,sq)
           jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
           jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
           jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
           jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
           jcbi=np.linalg.inv(jcb)
        else:
           jcbi[0,0]=2./hx[iel]
           jcbi[1,1]=2./hy[iel]

        dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
        dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
        qx[iel]=-hcond[iel]*np.dot(dNdx_T,T[icon_T[:,iel]])
        qy[iel]=-hcond[iel]*np.dot(dNdy_T,T[icon_T[:,iel]])

    print("     qx (m,M): %.4f %.4f " %(np.min(qx),np.max(qx)))
    print("     qy (m,M): %.4f %.4f " %(np.min(qy),np.max(qy)))

    if istep%every==0:
       filename = 'surface_heat_flux_{:04d}.ascii'.format(istep)
       np.savetxt(filename,np.array([x_e[nel-nelx:nel],qy[nel-nelx:nel]]).T)

    print("compute heat flux: %.3f s" % (clock.time()-start))

    ###########################################################################
    # plot of solution
    ###########################################################################
    start = clock.time()

    if istep==0:
       filename='solution.pvd'
       pvdfile=open(filename,"w")
       pvdfile.write('<?xml version="1.0" ?> \n')
       pvdfile.write('<VTKFile type="Collection" version="0.1" ByteOrder="LittleEndian"> \n')
       pvdfile.write('<Collection> \n')

    if istep%every==0:
       filename = 'solution_{:04d}.vtu'.format(istep)

       pvdfile.write('<DataSet timestep="'+str(time/year)+'" file="'+filename+'" /> \n')

       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_T,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%10e %10e %10e \n" %(x_T[i],y_T[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (K)' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%10e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T-T_init' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%10e \n" %(T[i]-T_init[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='qx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % qx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='qy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % qy[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % rho[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='hcapa' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % hcapa[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='hcond' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % hcond[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % area[iel])
       vtufile.write("</DataArray>\n")

       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(icon_T[0,iel],icon_T[1,iel],icon_T[2,iel],icon_T[3,iel]))
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

       print("export to vtu: %.3f s" % (clock.time() - start))

    ###################################

    if time>tfinal:
       print("*****tfinal reached*****")
       break

#end for istep
       
pvdfile.write('</Collection> \n')
pvdfile.write('</VTKFile> \n')
pvdfile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
