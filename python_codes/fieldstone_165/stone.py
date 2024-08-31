import numpy as np
import scipy.sparse as sps
import time as timing
from scipy.sparse import lil_matrix

###############################################################################

def u_th(x,y,exp):
    if exp==1:
       return np.cos((x-1)*np.pi/2)*np.cos((y-1)*np.pi/2)
    if exp==2 or exp==3:
       if x<=0.5 and y<=0.5:
          return (np.cos((x-0.25)*2*np.pi)*np.cos((y-0.25)*2*np.pi))**2
       else:
          return 0
    if exp==4 or exp==5:
       if abs(x-0.25)<0.125:
          return (np.cos((x-0.25)*4*np.pi))**4
       else:
          return 0

def udot_th(x,y,exp):
    if exp==1 or exp==2 or exp==3 or exp==4 or exp==5:
       return 0 

###############################################################################

def NNT(r,s,order):
    if order==1:
       N_0=0.25*(1.-r)*(1.-s)
       N_1=0.25*(1.+r)*(1.-s)
       N_2=0.25*(1.-r)*(1.+s)
       N_3=0.25*(1.+r)*(1.+s)
       return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)
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
       return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

def dNNTdr(r,s,order):
    if order==1:
       dNdr_0=-0.25*(1.-s)
       dNdr_1=+0.25*(1.-s)
       dNdr_2=-0.25*(1.+s)
       dNdr_3=+0.25*(1.+s)
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)
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
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

def dNNTds(r,s,order):
    if order==1:
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.-r)
       dNds_3=+0.25*(1.+r)
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)
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
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)

#------------------------------------------------------------------------------

sqrt3=np.sqrt(3.)
eps=1.e-10 

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

method=1
experiment=5
order=1

if order==1:
   m=4          # number of nodes making up an element
if order==2:
   m=9


if experiment==1: 
   Lx=2
   Ly=2
   c=1
   dt=1e-2
   nstep=250
   nelx=20
   nely=20
   every=1

if experiment==2 or experiment==3: 
   Lx=2
   Ly=2
   c=1
   dt=1e-2
   nstep=301
   nelx=64
   nely=nelx
   every=2

if experiment==4 or experiment==5:
   Lx=1
   Ly=1
   c=1
   dt=1e-3
   nstep=1001
   nelx=128
   nely=nelx
   every=2

###############################################################################

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
N=nnx*nny         # number of nodes
nel=nelx*nely     # number of elements, total
Nfem=N           # Total number of degrees of freedom

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

print ('experiment =',experiment)
print ('order      =',order)
print ('nnx        =',nnx)
print ('nny        =',nny)
print ('N          =',N)
print ('nel        =',nel)
print ('Nfem       =',Nfem)
print ('nqperdim   =',nqperdim)
print("-----------------------------")

#####################################################################
# grid point setup 
#####################################################################
start = timing.time()

x = np.empty(N,dtype=np.float64)  # x coordinates
y = np.empty(N,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx/order
        y[counter]=j*hy/order
        counter += 1
    #end for
#end for

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

print("connectivity (%.3fs)" % (timing.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fix=np.zeros(Nfem,dtype=bool)  
bc_val=np.zeros(Nfem,dtype=np.float64) 

if experiment==1 or experiment==2:
   for i in range(0,N):
       if x[i]/Lx<eps:
          bc_fix[i]=True ; bc_val[i]=u_th(x[i],y[i],experiment)
       if x[i]/Lx>(1-eps):
          bc_fix[i]=True ; bc_val[i]=u_th(x[i],y[i],experiment)
       if y[i]/Ly<eps:
          bc_fix[i]=True ; bc_val[i]=u_th(x[i],y[i],experiment)
       if y[i]/Ly>(1-eps):
          bc_fix[i]=True ; bc_val[i]=u_th(x[i],y[i],experiment)
   #end for

if experiment==3:
   for i in range(0,N):
       if x[i]/Lx<eps:
          bc_fix[i]=True ; bc_val[i]=u_th(x[i],y[i],experiment)
       if x[i]/Lx>(1-eps):
          bc_fix[i]=True ; bc_val[i]=u_th(x[i],y[i],experiment)
       if y[i]/Ly<eps:
          bc_fix[i]=True ; bc_val[i]=u_th(x[i],y[i],experiment)
       if y[i]/Ly>(1-eps):
          bc_fix[i]=True ; bc_val[i]=u_th(x[i],y[i],experiment)
       if abs(x[i]-Lx/2)<eps and y[i]<Ly/2:
          bc_fix[i]=True ; bc_val[i]=0
   #end for

if experiment==4:
   for i in range(0,N):
       if x[i]/Lx<eps:
          bc_fix[i]=True ; bc_val[i]=0
       if x[i]/Lx>(1-eps):
          bc_fix[i]=True ; bc_val[i]=0
       if abs(x[i]-Lx/2)<eps and y[i]<0.43*Ly:
          bc_fix[i]=True ; bc_val[i]=0
       if abs(x[i]-Lx/2)<eps and y[i]>0.57*Ly:
          bc_fix[i]=True ; bc_val[i]=0
   #end for

if experiment==5:
   for i in range(0,N):
       #if x[i]/Lx<eps:
       #   bc_fix[i]=True ; bc_val[i]=0
       if x[i]/Lx>(1-eps):
          bc_fix[i]=True ; bc_val[i]=0
       if abs(x[i]-Lx/2)<eps and y[i]<0.4*Ly:
          bc_fix[i]=True ; bc_val[i]=0
       if abs(x[i]-Lx/2)<eps and abs(y[i]-Ly/2)<0.07*Ly:
          bc_fix[i]=True ; bc_val[i]=0
       if abs(x[i]-Lx/2)<eps and y[i]>0.6*Ly:
          bc_fix[i]=True ; bc_val[i]=0
   #end for







print("boundary conditions (%.3fs)" % (timing.time() - start))

#####################################################################
# initialise field values 
# methods 1,2 start at t=2dt. uprev contains u at t=dt and uprevprev 
# contains u at t=0
# method 3 starts at t=dt. uprev contains u at t=0, uprevprev
# is not used and udotprev contains the time derivative of u at t=0
#####################################################################
start = timing.time()

u=np.zeros(N,dtype=np.float64)    
uprevprev=np.zeros(N,dtype=np.float64) 
uprev=np.zeros(N,dtype=np.float64)     
udot=np.zeros(N,dtype=np.float64)      
udotprev=np.zeros(N,dtype=np.float64) 

if method==1 or method==2:
   t=2*dt
   for i in range(0,N):
       uprevprev[i]=u_th(x[i],y[i],experiment)
       uprev[i]=u_th(x[i],y[i],experiment)+dt*udot_th(x[i],y[i],experiment)

if method==3:
   t=dt
   for i in range(0,N):
       uprev[i]=u_th(x[i],y[i],experiment)
       udotprev[i]=udot_th(x[i],y[i],experiment)

print("initialse fields (%.3fs)" % (timing.time() - start))

#####################################################################
# create necessary arrays 
#####################################################################
start = timing.time()

dNNNTdx = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTdy = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTdr = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTds = np.zeros(m,dtype=np.float64)           # shape functions derivatives
    
print("create arrays (%.3fs)" % (timing.time() - start))

#******************************************************************************
#******************************************************************************
# time stepping loop
#******************************************************************************
#******************************************************************************

model_time=0.
statsfile=open('u_stats.ascii',"w")

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")


    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(Nfem,dtype=np.float64)        # FE rhs 
    B_mat=np.zeros((2,m),dtype=np.float64)         # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)       # shape functions

    for iel in range (0,nel):

        b_el=np.zeros(m,dtype=np.float64)
        a_el=np.zeros((m,m),dtype=np.float64)
        Ke=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        Me=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 

        up=np.zeros(m,dtype=np.float64)
        upp=np.zeros(m,dtype=np.float64)
        for k in range(0,m):
            up[k]=uprev[icon[k,iel]]
            upp[k]=uprevprev[icon[k,iel]]

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_mat[0:m,0]=NNT(rq,sq,order)
                dNNNTdr[0:m]=dNNTdr(rq,sq,order)
                dNNNTds[0:m]=dNNTds(rq,sq,order)

                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0] += dNNNTdr[k]*x[icon[k,iel]]
                    jcb[0,1] += dNNNTdr[k]*y[icon[k,iel]]
                    jcb[1,0] += dNNNTds[k]*x[icon[k,iel]]
                    jcb[1,1] += dNNNTds[k]*y[icon[k,iel]]
                #end for

                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                for k in range(0,m):
                    dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                    dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                    B_mat[0,k]=dNNNTdx[k]
                    B_mat[1,k]=dNNNTdy[k]
                #end for

                Me=N_mat.dot(N_mat.T)*weightq*jcob
                Ke=B_mat.T.dot(B_mat)*weightq*jcob

                a_el[:,:]+=Me[:,:]
                if method==1:
                   b_el[:]+=(2*Me-c**2*dt**2*Ke).dot(up)-Me.dot(upp)
                if method==2 or method==3:
                   b_el-=c**2*Ke.dot(up)

            #end for jq
        #end for iq

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

    print("build FEM matrix: %.3fs" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    if method==1:
       u=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    if method==2:
       uu=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
       u=np.zeros(nnx,dtype=np.float64)  
       u[:]=dt**2*uu[:]+2*uprev[:]-uprevprev[:]

    if method==3:
       u[:]=uprev[:]+udotprev[:]*dt
       R=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
       udot[:]=udotprev[:]+dt*R[:]

    statsfile.write("%e %e %e \n" %(t,np.min(u),np.max(u))) ; statsfile.flush()

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))

    print("solve time: %.3f s" % (timing.time() - start))

    #################################################################
    # visualisation 
    #################################################################

    if istep%every==0:

       start = timing.time()

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%e %e %e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='u' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10f \n" %u[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='bc_fix' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d  \n" %(int(bc_fix[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if order==1:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[3,iel],icon[2,iel]))
       if order==2:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(\
                            icon[0,iel],icon[2,iel],icon[8,iel],\
                            icon[6,iel],icon[1,iel],icon[5,iel],
                            icon[7,iel],icon[3,iel],icon[4,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       if order==1:
          for iel in range (0,nel):
              vtufile.write("%d \n" %((iel+1)*4))
       if order==2:
          for iel in range (0,nel):
              vtufile.write("%d \n" %((iel+1)*9))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       if order==1:
          for iel in range (0,nel):
              vtufile.write("%d \n" %9)
       if order==2:
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

       print("export to files: %.3f s" % (timing.time() - start))

    #end if

    ###########################################################################

    model_time+=dt
    print ("model_time= %.4f" %model_time)

    if method==1 or method==2:
       uprevprev[:]=uprev[:]
       uprev[:]=u[:]

    if method==3:
       uprev[:]=u[:]
       udotprev[:]=udot[:]
    
#end for istep

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
