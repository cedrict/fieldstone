import numpy as np
import time as timing
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

print("-----------------------------")
print("--------fieldstone 141-------")
print("-----------------------------")

year=365.25*3600*24
sqrt3=np.sqrt(3.)
eps=1.e-10 

ndim=2       # number of space dimensions
m=4          # number of nodes making up an element
ndofT=1

#################################################################

test=0

if test==0:
   Lx=5e3
   Ly=600e3
   h_seds=5e3
   h_crust=30e3
   h_lith=90e3
   h_mantle=Ly-h_seds-h_crust-h_lith
   nelx=2
   nely=int(Ly/1000)
   bc_top='dirichlet'
   bc_bottom='dirichlet'
   T_top=0+273
   T_moho=550+273
   T_lith=1330+273
   T_bottom=1520+273
   q_bottom=0.0208
   nstep=1000
   tfinal=5e6*year
   every=20
   Ttarget=1330+273

if test==1: # T prescribed top and bottom
   Lx=5e3
   Ly=100e3
   nelx=2
   nely=20
   bc_top='dirichlet'
   bc_bottom='dirichlet'
   T_top=0+273
   T_bottom=100+273
   nstep=1000
   tfinal=200e6*year
   every=100

if test==2: # T prescribed at top, flux at bottom
   Lx=5e3
   Ly=100e3
   nelx=2
   nely=25
   bc_top='dirichlet'
   bc_bottom='neumann'
   T_top=0+273
   nstep=5000
   tfinal=200e6*year
   every=100
   q_bottom=0.03

#################################################################

nel=nelx*nely
nnx=nelx+1
nny=nely+1
NV=nnx*nny
NfemT=NV
hx=Lx/nelx
hy=Ly/nely

depthfile=open('geotherm_depth.ascii',"w")

#################################################################
# grid point setup
#################################################################
start = timing.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# build connectivity array
#################################################################
start = timing.time()

icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (timing.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)
bc_valT=np.zeros(NfemT,dtype=np.float64)

for i in range(0,NV):
    if bc_bottom=='dirichlet':
       if y[i]/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=T_bottom
    if y[i]/Ly>(1-eps):
       bc_fixT[i]=True ; bc_valT[i]=T_top
#end for

print("setup: define boundary conditions: %.3f s" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################
start = timing.time()

T = np.zeros(NV,dtype=np.float64)
T_prev = np.zeros(NV,dtype=np.float64)

if test==0:
   for i in range(0,NV):
       if y[i]>=Ly-(h_seds+h_crust):
          T[i]=(T_moho-T_top)/(h_seds+h_crust)*(Ly-y[i])+T_top
       elif y[i]>Ly-(h_seds+h_crust+h_lith):
          T[i]=(T_lith-T_moho)/h_lith*(Ly-h_seds-h_crust-y[i])+T_moho
       else:
          T[i]=(T_bottom-T_lith)/h_mantle*(h_mantle-y[i])+T_lith
   #end for

if test==1 or test==2:
   T[:]=T_top

np.savetxt('temperature_init.ascii',np.array([x,y,T-273]).T,header='# x,y,T')

print("setup: define initial temperature: %.3f s" % (timing.time() - start))

#####################################################################
# geometry/layering
#####################################################################
start = timing.time()

hcond = np.zeros(nel,dtype=np.float64)
hcapa = np.zeros(nel,dtype=np.float64)
rho   = np.zeros(nel,dtype=np.float64)
kappa = np.zeros(nel,dtype=np.float64)

if test==0:
   for iel in range(0,nel):
       xc=np.sum(x[icon[:,iel]])*0.25
       yc=np.sum(y[icon[:,iel]])*0.25
       if yc>Ly-h_seds: #sediments
          hcond[iel]=2.25
          hcapa[iel]=750
          rho[iel]=2800
       elif yc>Ly-(h_seds+h_crust): #crust
          hcond[iel]=2.25
          hcapa[iel]=750
          rho[iel]=2800
       elif yc>Ly-(h_seds+h_crust+h_lith): #mantle lithosphere
          hcond[iel]=2.25
          hcapa[iel]=1250
          rho[iel]=3300
       else:
          hcond[iel]=5 #2
          hcapa[iel]=1250
          rho[iel]=3300

if test==1:
   hcond[:]=2.5
   hcapa[:]=1000
   rho[:]=3000

if test==2:
   hcond[:]=50
   hcapa[:]=1000
   rho[:]=3000


kappa[:]=hcond[:]/rho[:]/hcapa[:]
    
print("     -> kappa (m,M) %.4e %.4e " %(np.min(kappa),np.max(kappa)))

print("setup: elemental material properties: %.3f s" % (timing.time() - start))

#####################################################################
# compute time step
#####################################################################

dt=min(0.5*hy**2/max(kappa),1e5*year)

print('time step:',dt/year,' yr')

#==================================================================================================
#==================================================================================================
# time stepping loop
#==================================================================================================
#==================================================================================================

time=0.

for istep in range(0,nstep):

    time+=dt

    print("-------------------------------------")
    print("istep= ", istep, 'time=',time/year/1e6,'Myr')
    print("-------------------------------------")

    T_prev[:]=T[:]

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat = np.zeros((2,ndofT*m),dtype=np.float64)   # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    Tvect = np.zeros(m,dtype=np.float64)   
    dNdr=np.zeros(m,dtype=np.float64)  
    dNds=np.zeros(m,dtype=np.float64)  
    dNdx=np.zeros(m,dtype=np.float64)  
    dNdy=np.zeros(m,dtype=np.float64)  

    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 

        for k in range(0,m):
            Tvect[k]=T[icon[k,iel]]
        #end for

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
                for k in range(0,m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                #end for

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho[iel]*hcapa[iel]*weightq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond[iel]*weightq*jcob

                a_el=MM+Kd*dt

                b_el=MM.dot(Tvect)

                if bc_bottom=='neumann' and y[icon[0,iel]]/Ly<eps:
                   b_el[0]+=hx*q_bottom/8*dt
                   b_el[1]+=hx*q_bottom/8*dt

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
            #end for
        #end for

    #end for iel

    print("build matrix: %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T)-273,np.max(T)-273))

    print("solve: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute field derivatives 
    #####################################################################
    start = timing.time()

    qx=np.zeros(NV,dtype=np.float64)  
    qy=np.zeros(NV,dtype=np.float64)  
    ccc=np.zeros(NV,dtype=np.float64)  

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
            for k in range(0,m):
                q_x+=-hcond[iel]*dNdx[k]*T[icon[k,iel]]
                q_y+=-hcond[iel]*dNdy[k]*T[icon[k,iel]]
            #end for

            qx[icon[kk,iel]]+=q_x
            qy[icon[kk,iel]]+=q_y
            ccc[icon[kk,iel]]+=1

        #end for k

    #end for iel

    qx/=ccc
    qy/=ccc

    print("     -> qx  (m,M) %.4e %.4e " %(np.min(qx),np.max(qx)))
    print("     -> qy  (m,M) %.4e %.4e " %(np.min(qy),np.max(qy)))

    if istep%every==0:
       np.savetxt('solution_{:04d}.ascii'.format(istep),np.array([y,T-273,qy]).T,header='# y,T,qy')

    print("compute heat flux: %.3f s" % (timing.time() - start))

    #####################################################################
    # track depth of geotherm
    #####################################################################

    if test==0:
       for iel in range(0,nel):
           Tmin=min(T[icon[:,iel]])
           Tmax=max(T[icon[:,iel]])
           yc=np.sum(y[icon[:,iel]])*0.25
           if Ttarget>Tmin and Ttarget<Tmax:
              depthfile.write("%e %e %e \n" %(time/year,Ly-yc,yc))
              depthfile.flush()
              print('depth of geotherm:',yc)
              break

    #####################################################################
    # plot of solution
    #####################################################################

    if istep%every==0:

       start = timing.time()

       stretch=20

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
          vtufile.write("%10e %10e %10e \n" %(x[i]*stretch,y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='hcond' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e \n" % (hcond[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='hcapa' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e \n" % (hcapa[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e \n" % (rho[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='kappa' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e \n" % (kappa[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (C)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" % (T[i]-273))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='qx' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%e\n" % qx[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='qy' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%e\n" % qy[i])
       vtufile.write("</DataArray>\n")
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

       print("export to vtu: %.3f s" % (timing.time() - start))

    #end if 

    if time>tfinal:
       print('**************')
       print('tfinal reached')
       print('**************')
       break 

#end for istep

#==================================================================================================
# end time stepping loop
#==================================================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")





