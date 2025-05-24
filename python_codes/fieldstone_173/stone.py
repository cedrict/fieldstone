import numpy as np
import sys as sys
import scipy
import time as timing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix

#------------------------------------------------------------------------------

def T_analytical(x,y):
    return (2*x+y)**2

def qx_analytical(x,y):
    return -4*(2*x+y)

def qy_analytical(x,y):
    return -2*(2*x+y)

def rhs_f(x,y,experiment):
    return -10

#------------------------------------------------------------------------------

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

ndim=2       # number of space dimensions
ndofT=1      # number of degrees of freedom per node
hcond=1.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density


if int(len(sys.argv) == 4):
   experiment=int(sys.argv[1])
   order     =int(sys.argv[2])
else:
   experiment=1
   order=1

if order==1: m=4 # number of nodes making up an element
if order==2: m=9

if experiment==1: # rotating cone
   nelx=8
   nely=nelx
   Lx=1.  
   Ly=1.  
   steady_state=True

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
   rnodes=[-1,1,1,-1]
   snodes=[-1,-1,1,1]

if order==2:
   nqperdim=3
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

#####################################################################

print ('experiment =',experiment)
print ('order      =',order)
print ('nnx        =',nnx)
print ('nny        =',nny)
print ('NV         =',NV)
print ('nel        =',nel)
print ('NfemT      =',NfemT)
print ('nqperdim   =',nqperdim)
print("-----------------------------")

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

np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("mesh (%.3fs)" % (timing.time() - start))

#####################################################################
# stretch mesh
#####################################################################

for i in range(0,NV):
    x[i]=x[i]**0.66
    y[i]=y[i]**0.66

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

#################################################################
# flag nodes and elements on boundary
#################################################################
start = timing.time()

surface_element=np.zeros(nel,dtype=bool)  
boundary_node=np.zeros(NV,dtype=bool)  

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        if i==0 or i==nelx-1 or j==0 or j==nely-1: 
           surface_element[counter]=True 
        counter+=1

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        if i==0 or i==nnx-1 or j==0 or j==nny-1: 
           boundary_node[counter]=True 
        counter+=1

print("flag boundary nodes & elements: %.3f s" % (timing.time() - start))


#################################################################
# compute normal to domain at the nodes
# this is also implemented in stone 151
#################################################################
start = timing.time()

nx=np.zeros(NV,dtype=np.float64) 
ny=np.zeros(NV,dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):
    if surface_element[iel]: 
       for iq in range(0,nqperdim):
           for jq in range(0,nqperdim):

               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               dNNNTdr=dNNTdr(rq,sq,order)
               dNNNTds=dNNTds(rq,sq,order)
               jcb[0,0]=np.dot(dNNNTdr[:],x[icon[:,iel]])
               jcb[0,1]=np.dot(dNNNTdr[:],y[icon[:,iel]])
               jcb[1,0]=np.dot(dNNNTds[:],x[icon[:,iel]])
               jcb[1,1]=np.dot(dNNNTds[:],y[icon[:,iel]])
               jcob = np.linalg.det(jcb)
               jcbi=np.linalg.inv(jcb)
               dNNNTdx=jcbi[0,0]*dNNNTdr[:]+jcbi[0,1]*dNNNTds[:]
               dNNNTdy=jcbi[1,0]*dNNNTdr[:]+jcbi[1,1]*dNNNTds[:]
               for k in range(0,m):
                   if boundary_node[icon[k,iel]]:
                      nx[icon[k,iel]]+=dNNNTdx[k]*jcob*weightq
                      ny[icon[k,iel]]+=dNNNTdy[k]*jcob*weightq
           #end for
       #end for
    #end if
#end for

for i in range(0,NV):
    if boundary_node[i]:
       norm=np.sqrt(nx[i]**2+ny[i]**2)
       nx[i]/=norm
       ny[i]/=norm

print("compute normal: %.3f s" % (timing.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

if experiment==1:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=T_analytical(x[i],y[i])
       if x[i]/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=T_analytical(x[i],y[i])
       if y[i]/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=T_analytical(x[i],y[i])
       if y[i]/Ly>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=T_analytical(x[i],y[i])
   #end for

print("boundary conditions (%.3fs)" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################
start = timing.time()

T = np.zeros(NV,dtype=np.float64)

print("initial temperature (%.3fs)" % (timing.time() - start))

#################################################################
# compute timestep
#################################################################
start = timing.time()

if steady_state:
   dt=0.
   nstep=1
else:
   dt=0
   print('dt=',dt)
   nstep=int(tfinal/dt)
   print('nstep=',nstep)

print("compute timestep (%.3fs)" % (timing.time() - start))

#####################################################################
# create necessary arrays 
#####################################################################
start = timing.time()

Tvectm1 = np.zeros(m,dtype=np.float64)   
dNNNTdx = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNNNTdy = np.zeros(m,dtype=np.float64)           # shape functions derivatives
N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    
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

    A_mat = lil_matrix((NfemT,NfemT),dtype=np.float64)
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 

    counterq=0
    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_mat[0:m,0]=NNT(rq,sq,order)
                dNNNTdr=dNNTdr(rq,sq,order)
                dNNNTds=dNNTds(rq,sq,order)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNNNTdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNNNTdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNNNTds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNNNTds[k]*y[icon[k,iel]]
                #end for
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                xq=0.
                yq=0.
                for k in range(0,m):
                    xq+=N_mat[k,0]*x[icon[k,iel]]
                    yq+=N_mat[k,0]*y[icon[k,iel]]
                    dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                    dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                    B_mat[0,k]=dNNNTdx[k]
                    B_mat[1,k]=dNNNTdy[k]
                #end for

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho0*hcapa*weightq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                if steady_state:
                   a_el+=Kd
                   b_el+=N_mat[:,0]*weightq*jcob*rhs_f(xq,yq,experiment)
                else:
                      a_el+=MM+alphaT*(Kd)*dt
                      b_el+=(MM-(1-alphaT)*(Kd)*dt).dot(Tvectm1) +\
                            N_mat[:,0]*weightq*jcob*rhs_f(xq,yq,experiment)*dt
                #end if

                #print(xq,yq,rhs_f(xq,yq,experiment))

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

    print("build FEM matrix: %.3fs" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T time: %.3f s" % (timing.time() - start))

    #################################################################
    # compute heat flux
    #################################################################
    start = timing.time()

    qx=np.zeros(NV,dtype=np.float64) 
    qy=np.zeros(NV,dtype=np.float64) 
    qn=np.zeros(NV,dtype=np.float64) 
    cc=np.zeros(NV,dtype=np.float64) 

    for iel in range(0,nel):
        for k in range(0,m):
            rq=rnodes[k]
            sq=snodes[k]
            inode=icon[k,iel]
            cc[inode]+=1
            dNNNTdr=dNNTdr(rq,sq,order)
            dNNNTds=dNNTds(rq,sq,order)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            jcb[0,0]=np.sum(dNNNTdr[:]*x[icon[:,iel]])
            jcb[0,1]=np.sum(dNNNTdr[:]*y[icon[:,iel]])
            jcb[1,0]=np.sum(dNNNTds[:]*x[icon[:,iel]])
            jcb[1,1]=np.sum(dNNNTds[:]*y[icon[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNNNTdx[:]=jcbi[0,0]*dNNNTdr[:]+jcbi[0,1]*dNNNTds[:]
            dNNNTdy[:]=jcbi[1,0]*dNNNTdr[:]+jcbi[1,1]*dNNNTds[:]
            qx[inode]-=np.sum(dNNNTdx[:]*T[icon[:,iel]])
            qy[inode]-=np.sum(dNNNTdy[:]*T[icon[:,iel]])
        #end for
    #end for
    qx/=cc
    qy/=cc

    qn[:]=qx[:]*nx[:]+qy[:]*ny[:]

    print("     -> qx (m,M) %.4f %.4f " %(np.min(qx),np.max(qx)))
    print("     -> qy (m,M) %.4f %.4f " %(np.min(qy),np.max(qy)))
    print("     -> qn (m,M) %.4f %.4f " %(np.min(qn),np.max(qn)))

    print("compute heat flux: %.3f s" % (timing.time() - start))

    #################################################################
    # export boundary heat flux values
    #################################################################
    qn_file=open('heat_flux_boundary.ascii',"w")

    counter=0

    for i in range(0,nny):
        inode=nnx*(i+1)-1
        qn_file.write("%d %d %e %e %e %e  \n" %(counter,inode,qx[inode],qy[inode],qn[inode],\
               qx_analytical(x[inode],y[inode])*nx[inode]
              +qy_analytical(x[inode],y[inode])*ny[inode]))
        counter+=1
         
    counter-=1
    for i in range(0,nnx): 
        inode=NV-i-1
        qn_file.write("%d %d %e %e %e %e  \n" %(counter,inode,qx[inode],qy[inode],qn[inode],\
               qx_analytical(x[inode],y[inode])*nx[inode]
              +qy_analytical(x[inode],y[inode])*ny[inode]))
        counter+=1

    counter-=1
    for i in range(0,nny):
        inode=NV-(i+1)*nnx
        qn_file.write("%d %d %e %e %e %e  \n" %(counter,inode,qx[inode],qy[inode],qn[inode],
               qx_analytical(x[inode],y[inode])*nx[inode]
              +qy_analytical(x[inode],y[inode])*ny[inode]))
        counter+=1

    counter-=1
    for i in range(0,nnx): 
        inode=i
        qn_file.write("%d %d %e %e %e %e  \n" %(counter,inode,qx[inode],qy[inode],qn[inode],
               qx_analytical(x[inode],y[inode])*nx[inode]
              +qy_analytical(x[inode],y[inode])*ny[inode]))
        counter+=1

    qn_file.close()

    #################################################################
    # visualisation 
    #################################################################

    if True: 

       start = timing.time()

       filename = 'solution_{:04d}.vtu'.format(istep) 
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
           vtufile.write("%12.4e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (analytical)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%12.4e \n" % (T_analytical(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (error)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%12.4e \n" % (T[i]-T_analytical(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(qx[i],qy[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux (analytical)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(qx_analytical(x[i],y[i]),qy_analytical(x[i],y[i]),0))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux (error)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(qx[i]-qx_analytical(x[i],y[i]),qy[i]-qy_analytical(x[i],y[i]),0))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(nx[i],ny[i],0.))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Int32' Name='boundary' Format='ascii'> \n")
       for i in range(0,NV):
           if boundary_node[i]:
              vtufile.write("%d \n" % 1)
           else:
              vtufile.write("%d \n" % 0)
       vtufile.write("</DataArray>\n")
       #--


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

       #filename = 'solution_{:04d}.pdf'.format(istep) 
       #fig = plt.figure ()
       #ax = fig.gca(projection='3d')
       #ax.plot_surface(x.reshape ((nny,nnx)),y.reshape((nny,nnx)),T.reshape((nny,nnx)),color = 'darkseagreen')
       #ax.set_xlabel ( 'X [ m ] ')
       #ax.set_ylabel ( 'Y [ m ] ')
       #ax.set_zlabel ( ' Temperature  [ C ] ')
       #plt.title('Timestep  %.2d' %(istep),loc='right')
       #plt.grid ()
       #plt.savefig(filename)
       #plt.show ()
       #plt.close()

       print("export to files: %.3f s" % (timing.time() - start))

    #end if

    model_time+=dt
    print ("model_time=",model_time)
    
#end for istep

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
