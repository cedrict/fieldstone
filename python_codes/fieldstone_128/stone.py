import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
from scipy.sparse import csr_matrix, lil_matrix
import random

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------- stone 128 ---------")
print("-----------------------------")

year=365.25*24*3600
sqrt3=np.sqrt(3.)
eps=1.e-10 

ndim=2              # number of space dimensions
m=4                 # number of nodes making up an element
ndof=1              # number of degrees of freedom per node
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
phi=0.1             # porosity
eta=1.33e-4         # fluid viscosity (Pa s)

nstep=25   # maximum number of timestep   

dt=0.1*year

nelx = 800
nely = int(nelx*Ly/Lx)

experiment=3

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=nelx+1      # number of elements, x direction
nny=nely+1      # number of elements, y direction
NP=nnx*nny      # number of nodes
nel=nelx*nely   # number of elements, total
Nfem=NP*ndof    # Total number of pressure degrees of freedom

#####################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('nelx=',nelx)
print('nely=',nely)
print('nel=',nel)
print('Nfem=',Nfem)
print('hx=',hx)
print('hy=',hy)
print("-----------------------------")

stats_vel_file=open('stats_vel.ascii',"w")
stats_p_file=open('stats_p.ascii',"w")
stats_vrms_file=open('stats_vrms.ascii',"w")
stats_pavrg_file=open('stats_pavrg.ascii',"w")

#####################################################################
# grid point setup 
#####################################################################

x = np.empty(NP, dtype=np.float64)  # x coordinates
y = np.empty(NP, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1
    #end for
#end for

#####################################################################
# connectivity
#####################################################################

icon =np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1
    #end for
#end for

#####################################################################
# define temperature boundary conditions
#####################################################################

bc_fix=np.zeros(Nfem,dtype=np.bool)  
bc_val=np.zeros(Nfem,dtype=np.float64) 

for i in range(0,NP):
    if y[i]/Ly<eps:
       bc_fix[i]=True ; bc_val[i]=Peb
    if y[i]/Ly>(1-eps):
       bc_fix[i]=True ; bc_val[i]=0.
#end for

#####################################################################
# initial temperature
#####################################################################

p = np.zeros(NP,dtype=np.float64)

for i in range(0,NP):
    p[i]=(Ly-y[i])/Ly*Peb
#end for

#np.savetxt('pressure_init.ascii',np.array([x,y,p]).T,header='# x,y,p')

#####################################################################
# permeability K setup
#####################################################################
start = timing.time()

K = np.zeros(nel,dtype=np.float64) 
xc = np.zeros(nel,dtype=np.float64) 
yc = np.zeros(nel,dtype=np.float64) 

if experiment==1:
   for iel in range(0,nel):
       xc[iel]=0.5*(x[icon[0,iel]]+x[icon[2,iel]])
       yc[iel]=0.5*(y[icon[0,iel]]+y[icon[2,iel]])
       if (xc[iel]-Lx/2)**2+(yc[iel]-Ly/2)**2<5e3**2:
          K[iel]=1e-14
       else:
          K[iel]=1e-16

if experiment==2:
   a=0.5
   b=Ly/2
   for iel in range(0,nel):
       xc[iel]=0.5*(x[icon[0,iel]]+x[icon[2,iel]])
       yc[iel]=0.5*(y[icon[0,iel]]+y[icon[2,iel]])
       if yc[iel]<a*(xc[iel]-0.5*Lx)+b+100 and\
          yc[iel]>a*(xc[iel]-0.5*Lx)+b-100 and\
          abs(xc[iel]-0.5*Lx)<10e3 and\
          abs(yc[iel]-0.5*Ly)<5e3:
          K[iel]=1e-12
       else:
          K[iel]=1e-16

if experiment==3:
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

   iin= np.empty(nvo,dtype=np.bool) 

   for iel in range(0,nel):
       xc[iel]=0.5*(x[icon[0,iel]]+x[icon[2,iel]])
       yc[iel]=0.5*(y[icon[0,iel]]+y[icon[2,iel]])
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

print("permeability setup: %.3f s" % (timing.time() - start))

#####################################################################
# create necessary arrays 
#####################################################################

N     = np.zeros(m,dtype=np.float64)   # shape functions
dNdx  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
pvect = np.zeros(m,dtype=np.float64)   

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

    A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(Nfem,dtype=np.float64)          # FE rhs 
    B_mat=np.zeros((2,m),dtype=np.float64)           # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions

    for iel in range (0,nel):

        b_el=np.zeros(m,dtype=np.float64)
        a_el=np.zeros((m,m),dtype=np.float64)
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 

        pvect=p[icon[:,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

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
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                #end for
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                for k in range(0,m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                #end for

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*weightq*jcob*beta*phi

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*weightq*jcob*K[iel]/eta

                #crank-nicolson does not work?!
                #a_el+=MM+Kd*dt*0.5
                #b_el+=(MM-Kd*dt*0.5).dot(pvect)

                a_el+=MM+Kd*dt
                b_el+=MM.dot(pvect)

            #end for
        #end for

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon[k1,iel]
            if bc_fix[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_val[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val[m1]
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

    p = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    stats_p_file.write("%e %e %e\n" % (time,np.min(p),np.max(p)))
    stats_p_file.flush()

    print("solve time: %.3f s" % (timing.time() - start))

    #################################################################
    # compute velocity
    #################################################################
    start = timing.time()

    u=np.zeros(nel,dtype=np.float64)  
    v=np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):

        rq = 0.0
        sq = 0.0

        dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
        dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
        dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
        dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

        jcb=np.zeros((ndim,ndim),dtype=np.float64)
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

        u[iel]=-dNdx.dot(p[icon[:,iel]])*K[iel]/eta
        v[iel]=-dNdy.dot(p[icon[:,iel]])*K[iel]/eta
        #for k in range(0,m):
        #    u[iel]-=dNdx[k]*p[icon[k,iel]]*K[iel]/eta
        #    v[iel]-=dNdy[k]*p[icon[k,iel]]*K[iel]/eta
        #end for
    
    #end for

    print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))

    stats_vel_file.write("%e %e %e %e %e\n" % (time,np.min(u),np.max(u),np.min(v),np.max(v)))
    stats_vel_file.flush()

    print("compute velocity: %.3f s" % (timing.time() - start))

    #################################################################
    # compute vrms 
    #################################################################
    start = timing.time()

    vrms=0.
    pavrg=0.
    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                #end for
                jcob = np.linalg.det(jcb)
                pq=N.dot(p[icon[:,iel]])
                vrms+=(u[iel]**2+v[iel]**2)*weightq*jcob
                pavrg+=pq*weightq*jcob
            #end for
        #end for
    #end for
    vrms=np.sqrt(vrms/(Lx*Ly))
    pavrg/=(Lx*Ly)

    stats_pavrg_file.write("%e %e\n" % (time,pavrg)) ; stats_pavrg_file.flush()
    stats_vrms_file.write("%e %e\n" % (time,vrms))   ; stats_vrms_file.flush()

    print("     -> time= %.6f ; vrms   = %e" %(time,vrms))

    print("compute vrms and <p>: %.3f s" % (timing.time() - start))

    #####################################################################
    #CFL=0.5
    #dt1=CFL*(hx)/np.max(np.sqrt(u**2+v**2))
    #print('     -> dt1= %.6f' %dt1)
    #print('     -> dt = %.6f' %dt)

    #####################################################################
    # export to vtu 
    #####################################################################
    start = timing.time()

    filename = 'solution_{:04d}.vtu'.format(istep) 
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NP):
        vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='permeability' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e \n" % K[iel])
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' Name='cell' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e \n" % voronoi_cell[iel])
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%e %e %e \n" %(u[iel]*year,v[iel]*year,0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (dir)' Format='ascii'> \n")
    for iel in range(0,nel):
        vel=np.sqrt(u[iel]**2+v[iel]**2)
        vtufile.write("%e %e %e \n" %(u[iel]/vel,v[iel]/vel,0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='pressure (MPa)' Format='ascii'> \n")
    for i in range(0,NP):
        vtufile.write("%e \n" % (p[i]/1e6))
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

    time+=dt
    
#end for

#==============================================================================
# end time stepping loop
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
