import numpy as np
import sys as sys
import time as clock
import random
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def basis_functions_P(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_P_dr(r,s):
    dNdr0=-0.25*(1.-s)
    dNdr1=+0.25*(1.-s)
    dNdr2=+0.25*(1.+s)
    dNdr3=-0.25*(1.+s)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_P_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

year=365.25*24*3600
sqrt3=np.sqrt(3.)
eps=1.e-10 

print("*******************************")
print("********** stone 128 **********")
print("*******************************")

ndim=2 # number of space dimensions
m=4    # number of nodes making up an element (Q1)

#experiment 1: Fluid Flow Around a circular inclusion 
#experiment 2: Fluid Flow Around a Fault
#experiment 3: Fluid Flow in a random medium 
#experiment 4: Antoine setup ??

experiment=2

if experiment==1 or experiment==2 or experiment==3:
   Lx=50e3             # horizontal extent of the domain 
   Ly=20e3             # vertical extent of the domain 
   rho=2700            # rock density (km/mˆ3) 
   rhof=1000           # water density (km/mˆ3)
   laambda = 0.6       # pore pressure ratio
   g=9.8               # acceleration due to gravity (m/sˆ2)
   Pb=laambda*rho*g*Ly # fixed fluid pressure on base (Pa)
   Ph=rhof*g*Ly        # hydrostatic fluid pressure on base (Pa)
   Peb=Pb-Ph           # excess overpressure on base (Pa)
   beta=1e-10          # bulk compresibility (1/Pa)
   eta=1.33e-4         # fluid viscosity (Pa s)
   nstep=25            # maximum number of timestep   
   dt=0.1*year
   nelx = 200 #800
   nely = int(nelx*Ly/Lx)

if experiment==4: 
   Lx=2e-2             # horizontal extent of the domain 
   Ly=1e-2             # vertical extent of the domain 
   depth=40.e3         # depth in m
   rho=2700            # rock density (km/mˆ3) 
   rhof=1000           # water density (km/mˆ3)
   laambda = 0.8       # pore pressure ratio
   g=9.8               # acceleration due to gravity (m/sˆ2)
   Pb=laambda*rho*g*depth # fixed fluid pressure on base and on top (Pa)
   Ph=rhof*g*depth        # hydrostatic fluid pressure on base (Pa)
   Peb=Pb-Ph           # excess overpressure on base (Pa)
   beta=1e-10          # bulk compresibility (1/Pa)
   eta=1.33e-4         # fluid viscosity (Pa s)
   nstep=25            # maximum number of timestep   
   dt=0.1*year
   nelx = 80 #800
   nely = int(nelx*Ly/Lx)

hx=Lx/nelx
hy=Ly/nely
    
nel=nelx*nely   # number of elements, total
nnx=nelx+1      # number of elements, x direction
nny=nely+1      # number of elements, y direction
nn_P=nnx*nny    # number of nodes
Nfem=nn_P       # Total number of pressure degrees of freedom

debug=False

###############################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('nelx=',nelx)
print('nely=',nely)
print('nel=',nel)
print('nn_P=',nn_P)
print('Nfem=',Nfem)
print('hx=',hx)
print('hy=',hy)
print('experiment=',experiment)
print("-----------------------------")

stats_vel_file=open('stats_vel.ascii',"w")
stats_p_file=open('stats_p.ascii',"w")
stats_vrms_file=open('stats_vrms.ascii',"w")
stats_pavrg_file=open('stats_pavrg.ascii',"w")

###############################################################################
# grid point setup 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)  # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_P[counter]=i*hx
        y_P[counter]=j*hy
        counter += 1
    #end for
#end for

print("mesh setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_P=np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

print("connectivity setup: %.3f s" % (clock.time()-start))

###############################################################################
# define pressure boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool) 
bc_val=np.zeros(Nfem,dtype=np.float64) 

for i in range(0,nn_P):
    if y_P[i]/Ly<eps:
       bc_fix[i]=True ; bc_val[i]=Peb
    if y_P[i]/Ly>(1-eps):
       bc_fix[i]=True ; bc_val[i]=0.
       if experiment==4: bc_val[i]=Peb   

print("define boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# initial pressure
###############################################################################
start=clock.time()

p=np.zeros(nn_P,dtype=np.float64)

for i in range(0,nn_P):
    p[i]=(Ly-y_P[i])/Ly*Peb
    if experiment==4:p[i]=Peb 

if debug: np.savetxt('pressure_init.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("initial pressure: %.3f s" % (clock.time()-start))

###############################################################################
# porosity phi and permeability K setup
###############################################################################
start=clock.time()

K=np.zeros(nel,dtype=np.float64) 
H=np.empty(nel,dtype=np.float64)
xc=np.zeros(nel,dtype=np.float64) 
yc=np.zeros(nel,dtype=np.float64) 
phi=np.empty(nel,dtype=np.float64)

if experiment==1:
   phi[:]=0.1             # porosity
   H[:]=0. 
   for iel in range(0,nel):
       xc[iel]=0.5*(x_P[icon_P[0,iel]]+x_P[icon_P[2,iel]])
       yc[iel]=0.5*(y_P[icon_P[0,iel]]+y_P[icon_P[2,iel]])
       if (xc[iel]-Lx/2)**2+(yc[iel]-Ly/2)**2<5e3**2:
          K[iel]=1e-14
       else:
          K[iel]=1e-16

if experiment==2:
   phi[:]=0.1             # porosity
   H[:]=0. 
   a=0.5
   b=Ly/2
   for iel in range(0,nel):
       xc[iel]=0.5*(x_P[icon_P[0,iel]]+x_P[icon_P[2,iel]])
       yc[iel]=0.5*(y_P[icon_P[0,iel]]+y_P[icon_P[2,iel]])
       if yc[iel]<a*(xc[iel]-0.5*Lx)+b+100 and\
          yc[iel]>a*(xc[iel]-0.5*Lx)+b-100 and\
          abs(xc[iel]-0.5*Lx)<10e3 and\
          abs(yc[iel]-0.5*Ly)<5e3:
          K[iel]=1e-12
       else:
          K[iel]=1e-16

if experiment==3:
   phi[:]=0.1             # porosity
   H[:]=0. 
   nvo=111
   xvo = np.empty(nvo,dtype=np.float64) 
   yvo = np.empty(nvo,dtype=np.float64) 
   Kvo = np.empty(nvo,dtype=np.float64) 
   for i in range(0,nvo):
       xvo[i]=random.uniform(0.01,0.99)*Lx
       yvo[i]=random.uniform(0.01,0.99)*Ly
       Kvo[i]=random.uniform(1e-16,1e-14)

   voronoi_cell=np.empty(nel,dtype=np.int) # associated V cell 
   voronoi_cell[:]=-1

   iin= np.empty(nvo,dtype=bool) 

   for iel in range(0,nel):
       xc[iel]=0.5*(x_P[icon_P[0,iel]]+x_P[icon_P[2,iel]])
       yc[iel]=0.5*(y_P[icon_P[0,iel]]+y_P[icon_P[2,iel]])
       iin[:]=True
       for i in range(0,nvo):
           for j in range(0,nvo):
               if i != j:
                  xim=xc[iel]-xvo[i]
                  yim=yc[iel]-yvo[i]
                  xjm=xc[iel]-xvo[j]
                  yjm=yc[iel]-yvo[j]
                  dim=np.sqrt(xim**2+yim**2)
                  djm=np.sqrt(xjm**2+yjm**2)
                  if dim>djm:
                     iin[i]=False 
                     break
                  #end if
               #end if
           #end for
       #end for
       where=np.where(iin)[0]
       voronoi_cell[iel]=where[0]

       K[iel]=Kvo[voronoi_cell[iel]]

       if yc[iel]<1*Ly/6: K[iel]=1e-15
       if yc[iel]>5*Ly/6: K[iel]=1e-15

   #end for iel

if experiment==4: 
   for iel in range(0,nel):
       xc[iel]=0.5*(x_P[icon_P[0,iel]]+x_P[icon_P[2,iel]])
       yc[iel]=0.5*(y_P[icon_P[0,iel]]+y_P[icon_P[2,iel]])
       if (xc[iel]-Lx/2)**2+(yc[iel]-Ly/2)**2<2e-3**2:
          phi[iel]=2
          H[iel]= 1e-9   # FG
       else:
          phi[iel]=1
          H[iel]= 0.0  # FG
       #phi[iel]= 1 # initial porosity  FG
       K[iel]= 1e-26*(phi[iel])**3  # FG

phi_mem = np.empty(nel,dtype=np.float64)
phi_mem[:]=phi[:]

print("permeability & porosity setup: %.3f s" % (clock.time()-start))

#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================

time=0.

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    ###########################################################################
    start=clock.time()

    A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64) # FE matrix 
    b_fem=np.zeros(Nfem,dtype=np.float64)          # FE rhs 
    B=np.zeros((2,m),dtype=np.float64)             # gradient matrix B 
    jcb=np.zeros((ndim,ndim),dtype=np.float64)

    for iel in range (0,nel):

        A_el=np.zeros((m,m),dtype=np.float64) # elemental matrix 
        b_el=np.zeros(m,dtype=np.float64)     # elemental rhs
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 

        pvect=p[icon_P[0:m,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:

                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                N_P=basis_functions_P(rq,sq)
                dNdr_P=basis_functions_P_dr(rq,sq)
                dNds_P=basis_functions_P_ds(rq,sq)

                jcb[0,0]=np.dot(dNdr_P,x_P[icon_P[:,iel]])
                jcb[0,1]=np.dot(dNdr_P,y_P[icon_P[:,iel]])
                jcb[1,0]=np.dot(dNds_P,x_P[icon_P[:,iel]])
                jcb[1,1]=np.dot(dNds_P,y_P[icon_P[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq

                dNdx_P=jcbi[0,0]*dNdr_P+jcbi[0,1]*dNds_P
                dNdy_P=jcbi[1,0]*dNdr_P+jcbi[1,1]*dNds_P

                B[0,:]=dNdx_P
                B[1,:]=dNdy_P

                # compute mass matrix
                MM=np.outer(N_P,N_P)*beta*phi[iel]*JxWq

                # compute diffusion matrix
                Kd=B.T.dot(B)*K[iel]/eta*JxWq

                #crank-nicolson does not work?!
                #a_el+=MM+Kd*dt*0.5
                #b_el+=(MM-Kd*dt*0.5).dot(pvect)

                A_el+=MM+Kd*dt
                b_el+=MM.dot(pvect)+N_P[:]*JxWq*(H[iel]*dt-(phi[iel]-phi_mem[iel]))

            #end for
        #end for

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon_P[k1,iel]
            if bc_fix[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon_P[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val[m1]
            #end if
        #end for

        # assemble matrix and right hand side
        for k1 in range(0,m):
            m1=icon_P[k1,iel]
            for k2 in range(0,m):
                m2=icon_P[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        #end for

    #end for iel

    print("building pressure matrix and rhs: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    p=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    stats_p_file.write("%e %e %e\n" % (time,np.min(p),np.max(p)))
    stats_p_file.flush()

    print("solve time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute velocity
    ###########################################################################
    start=clock.time()

    u=np.zeros(nel,dtype=np.float64)  
    v=np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):
        rq = 0.0
        sq = 0.0
        dNdr_P=basis_functions_P_dr(rq,sq)
        dNds_P=basis_functions_P_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_P,x_P[icon_P[:,iel]])
        jcb[0,1]=np.dot(dNdr_P,y_P[icon_P[:,iel]])
        jcb[1,0]=np.dot(dNds_P,x_P[icon_P[:,iel]])
        jcb[1,1]=np.dot(dNds_P,y_P[icon_P[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_P=jcbi[0,0]*dNdr_P+jcbi[0,1]*dNds_P
        dNdy_P=jcbi[1,0]*dNdr_P+jcbi[1,1]*dNds_P
        u[iel]=-dNdx_P.dot(p[icon_P[:,iel]])*K[iel]/eta
        v[iel]=-dNdy_P.dot(p[icon_P[:,iel]])*K[iel]/eta

    print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))

    stats_vel_file.write("%e %e %e %e %e\n" % (time,np.min(u),np.max(u),np.min(v),np.max(v)))
    stats_vel_file.flush()

    print("compute velocity: %.3f s" % (clock.time() - start))

    ###########################################################################
    # compute vrms 
    ###########################################################################
    start = clock.time()

    vrms=0.
    pavrg=0.
    for iel in range(0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.
                N_P=basis_functions_P(rq,sq)
                dNdr_P=basis_functions_P_dr(rq,sq)
                dNds_P=basis_functions_P_ds(rq,sq)
                jcb[0,0]=np.dot(dNdr_P,x_P[icon_P[:,iel]])
                jcb[0,1]=np.dot(dNdr_P,y_P[icon_P[:,iel]])
                jcb[1,0]=np.dot(dNds_P,x_P[icon_P[:,iel]])
                jcb[1,1]=np.dot(dNds_P,y_P[icon_P[:,iel]])
                JxWq=np.linalg.det(jcb)*weightq
                pq=N_P.dot(p[icon_P[:,iel]])
                vrms+=(u[iel]**2+v[iel]**2)*JxWq
                pavrg+=pq*JxWq
            #end for
        #end for
    #end for
    vrms=np.sqrt(vrms/(Lx*Ly))
    pavrg/=(Lx*Ly)

    stats_pavrg_file.write("%e %e\n" % (time,pavrg)) ; stats_pavrg_file.flush()
    stats_vrms_file.write("%e %e\n" % (time,vrms))   ; stats_vrms_file.flush()

    print("     -> time= %.6f ; vrms   = %e" %(time,vrms))

    print("compute vrms and <p>: %.3f s" % (clock.time()-start))

    ###########################################################################
    # export to vtu 
    ###########################################################################
    start=clock.time()

    filename = 'solution_{:04d}.vtu'.format(istep) 
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_P,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_P):
        vtufile.write("%.4e %.4e %.1e \n" %(x_P[i],y_P[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    ##
    vtufile.write("<DataArray type='Float32' Name='porosity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%.4e \n" % phi[iel])
    vtufile.write("</DataArray>\n")
    ##
    vtufile.write("<DataArray type='Float32' Name='source' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%.4e \n" % H[iel])
    vtufile.write("</DataArray>\n")
    ##
    vtufile.write("<DataArray type='Float32' Name='permeability' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%.4e \n" % K[iel])
    vtufile.write("</DataArray>\n")
    ##
    if experiment==3:
       vtufile.write("<DataArray type='Float32' Name='cell' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%.4e \n" % voronoi_cell[iel])
       vtufile.write("</DataArray>\n")
    ##
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%.4e %.4e %.1e \n" %(u[iel]*year,v[iel]*year,0.))
    vtufile.write("</DataArray>\n")
    ##
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='pressure (MPa)' Format='ascii'> \n")
    for i in range(0,nn_P):
        vtufile.write("%.4e \n" % (p[i]/1e6))
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d\n" %(icon_P[0,iel],icon_P[1,iel],icon_P[2,iel],icon_P[3,iel]))
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

    print("export to vtu: %.3f s" % (clock.time()-start))

    time+=dt
   
    ###########################################################################
    # evolution law for phi and H
    ###########################################################################

    phi_mem[:]=phi[:]
    # to be implemented
    # Antoine
    #if experiment==4:
    #   for iel in range(0,nel):
           #xc[iel] & yc[iel] are element center coords
           #phi[iel]=.... evolve value of phi
           #phi=(phi_f-phi_0)*time+phi_0
           #K[iel]=1e-26*phi[iel]**3
           #H[iel]=... evolve value of H
 
#end for

#==============================================================================
#==============================================================================
# end time stepping loop
#==============================================================================
#==============================================================================

if experiment==3:
    vtufile=open("voronoi_centers.vtu","w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nvo,nvo))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nvo):
        vtufile.write("%10e %10e %10e \n" %(xvo[i],yvo[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range(0,nvo):
        vtufile.write("%d " % i)
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range(0,nvo):
        vtufile.write("%d " % (i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range(0,nvo):
        vtufile.write("%d " % 1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
