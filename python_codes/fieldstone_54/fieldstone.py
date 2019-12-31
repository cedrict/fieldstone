import sys as sys
import numpy as np
import time as timing
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix

#------------------------------------------------------------------------------

def NNV(r,s):
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

def dNNVdr(r,s):
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

def dNNVds(r,s):
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

def NNP(r,s):
    return 0.25*(1-r)*(1-s),\
           0.25*(1+r)*(1-s),\
           0.25*(1+r)*(1+s),\
           0.25*(1-r)*(1+s)

#------------------------------------------------------------------------------

#experiment=1: relaxation of topo
#experiment=2: extension symmetric
#experiment=3: extension asymmetric
#experiment=4: extension symmetric + bottom inflow
#experiment=5: extension asymmetric + bottom inflow
#experiment=6: pure advection
#experiment=7: pure advection + cosine bump
#experiment=8: pure advection + pyramid bump
#experiment=9: rayleigh-taylor 

experiment=8

ndim=2
ndofV=2
ndofP=1

if experiment==1:
   Lx=512e3
   Ly=512e3
   nelx = 24
   nely = 24

if experiment==2 or \
   experiment==3 or \
   experiment==4 or \
   experiment==5 or \
   experiment==6 or \
   experiment==7 or \
   experiment==8:
   Lx=128e3
   Ly=32e3
   nelx = 100
   nely = 25

if experiment==9:
   Lx=500e3
   Ly=500e3
   nelx = 25
   nely = 25

visu = 1

nel=nelx*nely

NV=(2*nelx+1)*(2*nely+1)
mV=9
mP=4

NP=(nelx+1)*(nely+1)
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP

hx=Lx/nelx
hy=Ly/nely

print('nelx =',nelx)
print('nely =',nely)
print('nel  =',nel)
print('NV   =',NV)
print('NP   =',NP)
print('NfemV=',NfemV)
print('NfemP=',NfemP)

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

eps=1e-8

gy=-10.
rho1=3200.
eta1=1e22
eta_ref=1e22

if experiment==7 or experiment==8:
   eta1=1e26
   eta_ref=1e26
   width=15e3
   xbump=0.345678*Lx
   #width=16e3
   #xbump=0.5*Lx

if experiment==9:
   eta_ref=1e20

sparse=False

rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]

cm=0.01
km=1e3
year=3600.*24.*365.
Myear=1e6*year
dt=2.5e3*year
nstep=2

method=3
vertical_only=False

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1

np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

nnx=2*nelx+1
nny=2*nely+1
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

#################################################################
# build pressure grid and iconP 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

counter = 0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        xP[counter]=i*hx
        yP[counter]=j*hy
        counter += 1

np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix = np.zeros(NfemV, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(NfemV, dtype=np.float64)  # boundary condition, value
on_left_boundary = np.zeros(NV, dtype=np.bool)  
on_right_boundary = np.zeros(NV, dtype=np.bool) 
on_bottom_boundary = np.zeros(NV, dtype=np.bool)
on_top_boundary = np.zeros(NV, dtype=np.bool)

if experiment==1 or experiment==9:
   uleft=0.
   uright=0.
if experiment==2 or experiment==4:
   uleft=-1.*cm/year
   uright=+1.*cm/year
if experiment==3 or experiment==5:
   uleft=0
   uright=+2.*cm/year
if experiment==6 or experiment==7 or experiment==8:
   uleft=+1.*cm/year
   uright=+1.*cm/year

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = uleft
       on_left_boundary[i]=True
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = uright
       on_right_boundary[i]=True
    if yV[i]/Ly<eps:
       if experiment==4 or experiment==5:
          if xV[i]>= Lx/2.+1e-5:
             bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = uright
          elif xV[i]<= Lx/2.-1e-5:
             bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = uleft
          else:
             bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          #end if 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.5*cm/year
       else:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bottom_boundary[i]=True
    if yV[i]/Ly>1-eps:
       on_top_boundary[i]=True

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

#################################################################

eta_e=np.zeros(nel,dtype=np.float64)
rho_e=np.zeros(nel,dtype=np.float64)

if experiment==1 or \
   experiment==2 or \
   experiment==3 or \
   experiment==4 or \
   experiment==5 or \
   experiment==6:
   eta_e[:]=eta1
   rho_e[:]=rho1
   
if experiment==7 or \
   experiment==8:
   eta_e[:]=1e26
   rho_e[:]=rho1

if experiment==9:
   for iel in range(0,nel):
       if yV[iconV[0,iel]]<400e3:
          eta_e[iel]=1e20
          rho_e[iel]=3200
       else:
          eta_e[iel]=1e21
          rho_e[iel]=3300
       #end if
   #end for

#################################################################
# define initial elevation 
#################################################################

if experiment==1:
   for i in range(NV-(2*nelx+1),NV):
       yV[i]+=1000.*np.cos(xV[i]/Lx*np.pi)

if experiment==7:
   for i in range(NV-(2*nelx+1),NV):
       if xV[i]>xbump-width and xV[i]<xbump+width:
          #yV[i]+=1000.*np.cos((xV[i]-xbump)/width*np.pi/2 )
          yV[i]+=1000.*(1+np.cos((xV[i]-xbump)/width*np.pi ))

if experiment==8:
   for i in range(NV-(2*nelx+1),NV):
       if xV[i]>xbump-width and xV[i]<=xbump:
          yV[i]+=(xV[i]-(xbump-width))*0.2
       if xV[i]>xbump and xV[i]<xbump+width:
          yV[i]+=-(xV[i]-(xbump+width))*0.2

if experiment==9:
   for i in range(0,NV):
       if np.abs(yV[i]-400e3)<1:
          yV[i]-=5e3*np.cos(xV[i]/Lx*np.pi*2)
   for iel in range(0,iel):
       yV[iconV[5,iel]]=(yV[iconV[1,iel]]+yV[iconV[2,iel]])/2.
       yV[iconV[7,iel]]=(yV[iconV[0,iel]]+yV[iconV[3,iel]])/2.
       yV[iconV[8,iel]]=(yV[iconV[4,iel]]+yV[iconV[6,iel]])/2.

   counter = 0
   for j in range(0,2*nely+1):
       for i in range(0,2*nelx+1):
           xV[counter]=i*hx/2.
           yV[counter]=j*hy/2. 
           if yV[counter]<=400e3:
              yV[counter]-= 5e3*yV[counter]/400e3  *np.cos(xV[counter]/Lx*np.pi*2)
           else: 
              yV[counter]-= 5e3*(Ly-yV[counter])/100e3  *np.cos(xV[counter]/Lx*np.pi*2)
           counter += 1

#==============================================================================
#==============================================================================
# time stepping
#==============================================================================
#==============================================================================

elevation=np.zeros((nstep,3),dtype=np.float64)
vrms=np.zeros((nstep,2),dtype=np.float64) 
volume=np.zeros((nstep,3),dtype=np.float64) 

u = np.zeros(NV,dtype=np.float64)           # x-component velocity
v = np.zeros(NV,dtype=np.float64)           # y-component velocity

for istep in range(0,nstep):

   print("----------------------------------")
   print("istep= ", istep)
   print("----------------------------------")

   filename = 'bcsurf_{:04d}.ascii'.format(istep) 
   bcsurffile=open(filename,"w")

   #################################################################
   # mesh deformation
   #################################################################

   if method==1: # Lagrangian approach

      for i in range(0,NV):
          if on_left_boundary[i] or on_right_boundary[i]: # only vertical movement
             yV[i]+=dt*v[i]
          else:
             if not vertical_only:
                xV[i]+=dt*u[i]
             #end if
             yV[i]+=dt*v[i]
          #end if
      #end for

   if method==2: # my ALE
      for i in range(0,NV):
          if on_top_boundary[i]:
             if not vertical_only:
                xV[i]+=u[i]*dt
             yV[i]+=v[i]*dt

   if method==3: # aspect ALE

      print('computing vmesh for bc with Q1 elements')

      #first compute the L2 projection of the mesh velocity

      A_fs = np.zeros(((nelx+1)*ndim,(nelx+1)*ndim),dtype=np.float64)
      b_fs = np.zeros((nelx+1)*ndim,dtype=np.float64)
      A_el =np.zeros((4,4),dtype=np.float64)
      b_el =np.zeros((4),dtype=np.float64)
      iconFS=np.zeros((2,nelx),dtype=np.int32)
      umesh_fs = np.zeros((nelx+1),dtype=np.float64)
      vmesh_fs = np.zeros((nelx+1),dtype=np.float64)
      x_FS = np.zeros((nelx+1),dtype=np.float64)
      y_FS = np.zeros((nelx+1),dtype=np.float64)

      counter=0
      for j in range(0,nely):
          for i in range(0,nelx):
              if j==nely-1: # if element is top row

                 # compute normal to element edge
                 x1=xV[iconV[3,counter]] ; y1=yV[iconV[3,counter]]
                 x2=xV[iconV[2,counter]] ; y2=yV[iconV[2,counter]]
                 n_x=-y2+y1
                 n_y= x2-x1
                 nnorm=np.sqrt(n_x**2+n_y**2)
                 n_x/=nnorm
                 n_y/=nnorm
                 d12=np.sqrt((x1-x2)**2+(y1-y2)**2)
                 jcob=d12/2

                 #build connectivity array on the fly
                 iconFS[0,i]=i+0 
                 iconFS[1,i]=i+1 

                 x_FS[iconFS[0,i]]=xV[iconV[3,counter]]
                 x_FS[iconFS[1,i]]=xV[iconV[2,counter]]
                 y_FS[iconFS[0,i]]=yV[iconV[3,counter]]
                 y_FS[iconFS[1,i]]=yV[iconV[2,counter]]


                 A_el[:,:]=0.
                 b_el[:]=0.
                 for iq in [-1,1]:
                     rq=iq/np.sqrt(3.)
                     weightq=1. ; JxW=weightq*jcob
                     N1= 0.5*(1-rq) 
                     N2= 0.5*(1+rq) 
                     xq=N1*xV[iconV[3,counter]]+N2*xV[iconV[2,counter]]
                     yq=N1*yV[iconV[3,counter]]+N2*yV[iconV[2,counter]]
                     uq=N1*u[iconV[3,counter]]+N2*u[iconV[2,counter]]
                     vq=N1*v[iconV[3,counter]]+N2*v[iconV[2,counter]]
                     bob=uq*n_x+vq*n_y
                     #print (xq,bob,'aaa')
                     A_el[0,0]+=N1*N1*JxW ; A_el[0,2]+=N1*N2*JxW 
                     A_el[1,1]+=N1*N1*JxW ; A_el[1,3]+=N1*N2*JxW 
                     A_el[2,0]+=N2*N1*JxW ; A_el[2,2]+=N2*N2*JxW 
                     A_el[3,1]+=N2*N1*JxW ; A_el[3,3]+=N2*N2*JxW 
                     b_el[0]+=bob*N1*n_x*JxW
                     b_el[1]+=bob*N1*n_y*JxW
                     b_el[2]+=bob*N2*n_x*JxW
                     b_el[3]+=bob*N2*n_y*JxW
                 #end for iq
      
                 # assemble matrix A_fs and rhs b_fs
                 for k1 in range(0,2):
                     for i1 in range(0,ndofV):
                         ikk=ndofV*k1          +i1
                         m1 =ndofV*iconFS[k1,i]+i1
                         for k2 in range(0,2):
                             for i2 in range(0,ndofV):
                                 jkk=ndofV*k2          +i2
                                 m2 =ndofV*iconFS[k2,i]+i2
                                 A_fs[m1,m2] += A_el[ikk,jkk]
                             #end for
                         #end for
                         b_fs[m1]+=b_el[ikk]
                     #end for i1
                 #end for k1
              #end if
              counter+=1
          #end for
      #end for

      print("     -> A_fs (m,M) %.4e %.4e " %(np.min(A_fs),np.max(A_fs)))
      print("     -> b_fs (m,M) %.4e %.4e " %(np.min(b_fs),np.max(b_fs)))

      sparse_matrix=sps.csr_matrix(A_fs)
      sol=sps.linalg.spsolve(sparse_matrix,b_fs)
      umesh_fs,vmesh_fs=np.reshape(sol[0:2*(nelx+1)],(nelx+1,2)).T

      print("     -> umesh_fs (m,M) %.4e %.4e " %(np.min(umesh_fs),np.max(umesh_fs)))
      print("     -> vmesh_fs (m,M) %.4e %.4e " %(np.min(vmesh_fs),np.max(vmesh_fs)))
   
      filename = 'velmesh_Q1_{:04d}.ascii'.format(istep) 
      np.savetxt(filename,np.array([x_FS,y_FS,umesh_fs,vmesh_fs]).T)


      ####################################################3
      #3--6--2  <--- surface element
      #|  |  |
      #7--8--5 
      #|  |  |
      #0--4--1

      print('computing vmesh for bc with Q2 elements')

      A_fs = np.zeros((nnx*ndim,nnx*ndim),dtype=np.float64)
      b_fs = np.zeros(nnx*ndim,dtype=np.float64)
      A_el =np.zeros((6,6),dtype=np.float64)
      b_el =np.zeros((6),dtype=np.float64)
      iconFS=np.zeros((3,nelx),dtype=np.int32)
      umesh_fs = np.zeros(nnx,dtype=np.float64)
      vmesh_fs = np.zeros(nnx,dtype=np.float64)

      counter=0
      for j in range(0,nely):
          for i in range(0,nelx):
              if j==nely-1: # if element is top row

                 # compute normal to element edge
                 x1=xV[iconV[3,counter]] ; y1=yV[iconV[3,counter]]
                 x2=xV[iconV[2,counter]] ; y2=yV[iconV[2,counter]]
                 n_x=-y2+y1
                 n_y= x2-x1
                 nnorm=np.sqrt(n_x**2+n_y**2)
                 n_x/=nnorm
                 n_y/=nnorm
                 d12=np.sqrt((x1-x2)**2+(y1-y2)**2)
                 jcob=d12/2

                 #build connectivity array on the fly
                 iconFS[0,i]=2*i+0 
                 iconFS[1,i]=2*i+1 
                 iconFS[2,i]=2*i+2

                 A_el[:,:]=0.
                 b_el[:]=0.
                 for iq in range(0,nqperdim):
                     rq=qcoords[iq]
                     weightq=qweights[iq] ; JxW=weightq*jcob
                     N1= 0.5*rq*(rq-1.) 
                     N2=     (1.-rq**2) 
                     N3= 0.5*rq*(rq+1.) 
                     xq=N1*xV[iconV[3,counter]]+N2*xV[iconV[6,counter]]+N3*xV[iconV[2,counter]]
                     yq=N1*yV[iconV[3,counter]]+N2*yV[iconV[6,counter]]+N3*yV[iconV[2,counter]]
                     uq=N1*u[iconV[3,counter]]+N2*u[iconV[6,counter]]+N3*u[iconV[2,counter]]
                     vq=N1*v[iconV[3,counter]]+N2*v[iconV[6,counter]]+N3*v[iconV[2,counter]]
                     bob=uq*n_x+vq*n_y
                     #print (xq,bob)
                     A_el[0,0]+=N1*N1*JxW ; A_el[0,2]+=N1*N2*JxW ; A_el[0,4]+=N1*N3*JxW
                     A_el[1,1]+=N1*N1*JxW ; A_el[1,3]+=N1*N2*JxW ; A_el[1,5]+=N1*N3*JxW
                     A_el[2,0]+=N2*N1*JxW ; A_el[2,2]+=N2*N2*JxW ; A_el[2,4]+=N2*N3*JxW
                     A_el[3,1]+=N2*N1*JxW ; A_el[3,3]+=N2*N2*JxW ; A_el[3,5]+=N2*N3*JxW
                     A_el[4,0]+=N3*N1*JxW ; A_el[4,2]+=N3*N2*JxW ; A_el[4,4]+=N3*N3*JxW
                     A_el[5,1]+=N3*N1*JxW ; A_el[5,3]+=N3*N2*JxW ; A_el[5,5]+=N3*N3*JxW
                     b_el[0]+=bob*N1*n_x*JxW
                     b_el[1]+=bob*N1*n_y*JxW
                     b_el[2]+=bob*N2*n_x*JxW
                     b_el[3]+=bob*N2*n_y*JxW
                     b_el[4]+=bob*N3*n_x*JxW
                     b_el[5]+=bob*N3*n_y*JxW
                 #end for iq
          
                 # assemble matrix A_fs and rhs b_fs
                 for k1 in range(0,3):
                     for i1 in range(0,ndofV):
                         ikk=ndofV*k1          +i1
                         m1 =ndofV*iconFS[k1,i]+i1
                         for k2 in range(0,3):
                             for i2 in range(0,ndofV):
                                 jkk=ndofV*k2          +i2
                                 m2 =ndofV*iconFS[k2,i]+i2
                                 A_fs[m1,m2] += A_el[ikk,jkk]
                             #end for
                         #end for
                         b_fs[m1]+=b_el[ikk]
                     #end for i1
                 #end for k1
              #end if
              counter+=1
          #end for
      #end for

      print("     -> A_fs (m,M) %.4e %.4e " %(np.min(A_fs),np.max(A_fs)))
      print("     -> b_fs (m,M) %.4e %.4e " %(np.min(b_fs),np.max(b_fs)))

      sparse_matrix=sps.csr_matrix(A_fs)
      sol=sps.linalg.spsolve(sparse_matrix,b_fs)
      #sol=sps.linalg.spsolve(A_fs,b_fs)
      umesh_fs,vmesh_fs=np.reshape(sol[0:2*nnx],(nnx,2)).T

      print("     -> umesh_fs (m,M) %.4e %.4e " %(np.min(umesh_fs),np.max(umesh_fs)))
      print("     -> vmesh_fs (m,M) %.4e %.4e " %(np.min(vmesh_fs),np.max(vmesh_fs)))
   
      filename = 'velmesh_Q2_{:04d}.ascii'.format(istep) 
      np.savetxt(filename,np.array([xV[NV-(2*nelx+1):NV],yV[NV-(2*nelx+1):NV],umesh_fs,vmesh_fs]).T)

      #################################################

      print('solving Laplace problem for vmesh')

      NNNV   =np.zeros(mV,dtype=np.float64)           # shape functions V
      dNNNVdx=np.zeros(mV,dtype=np.float64)           # shape functions derivatives
      dNNNVdy=np.zeros(mV,dtype=np.float64)           # shape functions derivatives
      dNNNVdr=np.zeros(mV,dtype=np.float64)           # shape functions derivatives
      dNNNVds=np.zeros(mV,dtype=np.float64)           # shape functions derivatives
      b_mat  =np.zeros((4,ndofV*mV),dtype=np.float64) # gradient matrix B 

      A_sparse = lil_matrix((NfemV,NfemV),dtype=np.float64)
      f_rhs    = np.zeros(NfemV,dtype=np.float64)  # right hand side f 
      u2 = np.zeros(NV,dtype=np.float64)           # x-component velocity
      v2 = np.zeros(NV,dtype=np.float64)           # y-component velocity

      # generate bc for left, ritgh, bottom boundaries 
      bc_fix2 = np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
      bc_val2 = np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
      for i in range(0,NV):
          if on_left_boundary[i] or on_right_boundary[i]:
             bc_fix2[i*ndofV  ] = True              ; bc_val2[i*ndofV  ] = 0.
             bc_fix2[i*ndofV+1] = bc_fix[i*ndofV+1] ; bc_val2[i*ndofV+1] = bc_val[i*ndofV+1] 
          if on_bottom_boundary[i]:
             bc_fix2[i*ndofV  ] = True              ; bc_val2[i*ndofV  ] = 0.
             bc_fix2[i*ndofV+1] = True              ; bc_val2[i*ndofV+1] = 0.

      # compute normal vector to surface
      nx = np.zeros(NV,dtype=np.float64) 
      ny = np.zeros(NV,dtype=np.float64) 
      
      if vertical_only:
         nx[:]=0.
         ny[:]=1.
      else:
         # for each segment between two consecutive points at the surface
         # its normal is computed and the resulting vector is added to 
         # each point. In a second phase the resulting vector on each point
         # is normalised. 
         counter=0
         for j in range(0,2*nely+1):
             for i in range(0,2*nelx+1):
                 if j==2*nely and i<2*nelx:
                    n_x=-yV[counter+1]+yV[counter]
                    n_y= xV[counter+1]-xV[counter]
                    nnorm=np.sqrt(n_x**2+n_y**2)
                    n_x/=nnorm
                    n_y/=nnorm
                    nx[counter]+=n_x
                    ny[counter]+=n_y
                    nx[counter+1]+=n_x
                    ny[counter+1]+=n_y
                 #end if
                 counter += 1
             #end for
         #end for
         for i in range(NV-(2*nelx+1),NV):
             nnorm=np.sqrt(nx[i]**2+ny[i]**2)
             nx[i]/=nnorm
             ny[i]/=nnorm
             if i==NV-(2*nelx+1): #left point of surface
                nx[i]=0.
                ny[i]=1.
             #end if
             if i==NV-1: #right point of surface
                nx[i]=0.
                ny[i]=1.
             #end if
             #print(xV[i],yV[i],nx[i],ny[i])

      #generate bc for top surface
      counter=0
      for j in range(0,2*nely+1):
          for i in range(0,2*nelx+1):
              if j==2*nely:
                 bob=u[counter]*nx[counter]+v[counter]*ny[counter]
                 bc_fix2[2*counter  ]=True ; bc_val2[2*counter  ]=bob*nx[counter]
                 bc_fix2[2*counter+1]=True ; bc_val2[2*counter+1]=bob*ny[counter]
                 bcsurffile.write("%10e %10e %10e %10e %10e %10e %10e %10e %10e\n"
                                   %(xV[counter],yV[counter],
                                     nx[counter],ny[counter],
                                     bob,
                                     bob*nx[counter],bob*ny[counter],
                                     u[counter]*nx[counter],
                                     v[counter]*ny[counter]))
              counter += 1

      # build FE system
      for iel in range(0,nel):
          K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
          f_el =np.zeros((mV*ndofV),dtype=np.float64)
          for iq in range(0,nqperdim):
              for jq in range(0,nqperdim):
                  rq=qcoords[iq]
                  sq=qcoords[jq]
                  weightq=qweights[iq]*qweights[jq]
                  NNNV[0:mV]=NNV(rq,sq)
                  dNNNVdr[0:mV]=dNNVdr(rq,sq)
                  dNNNVds[0:mV]=dNNVds(rq,sq)
                  # calculate jacobian matrix
                  jcb=np.zeros((ndim,ndim),dtype=np.float64)
                  for k in range(0,mV):
                      jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                      jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                      jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                      jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                  jcob = np.linalg.det(jcb)
                  jcbi = np.linalg.inv(jcb)
                  # compute dNdx & dNdy
                  for k in range(0,mV):
                      dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                      dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                  # construct 3x8 b_mat matrix
                  for i in range(0,mV):
                      b_mat[0:4, 2*i:2*i+2] = [[dNNNVdx[i],0],
                                               [dNNNVdy[i],0],
                                               [0,dNNNVdx[i]],
                                               [0,dNNNVdy[i]]]
                  # compute elemental a_mat matrix
                  K_el+=b_mat.T.dot(b_mat)*weightq*jcob
              # end for jq
          # end for iq

          # impose b.c. 
          for k1 in range(0,mV):
              for i1 in range(0,ndofV):
                  ikk=ndofV*k1          +i1
                  m1 =ndofV*iconV[k1,iel]+i1
                  if bc_fix2[m1]:
                     K_ref=K_el[ikk,ikk] 
                     for jkk in range(0,mV*ndofV):
                         f_el[jkk]-=K_el[jkk,ikk]*bc_val2[m1]
                         K_el[ikk,jkk]=0
                         K_el[jkk,ikk]=0
                     K_el[ikk,ikk]=K_ref
                     f_el[ikk]=K_ref*bc_val2[m1]

          # assemble matrix K_mat and right hand side rhs
          for k1 in range(0,mV):
              for i1 in range(0,ndofV):
                  ikk=ndofV*k1          +i1
                  m1 =ndofV*iconV[k1,iel]+i1
                  for k2 in range(0,mV):
                      for i2 in range(0,ndofV):
                          jkk=ndofV*k2          +i2
                          m2 =ndofV*iconV[k2,iel]+i2
                          A_sparse[m1,m2] += K_el[ikk,jkk]
                      #end for
                  #end for
                  f_rhs[m1]+=f_el[ikk]

      sparse_matrix=A_sparse.tocsr()
      sol=sps.linalg.spsolve(sparse_matrix,f_rhs)
      u2,v2=np.reshape(sol[0:NfemV],(NV,2)).T

      print("     -> umesh (m,M) %.4e %.4e " %(np.min(u2),np.max(u2)))
      print("     -> vmesh (m,M) %.4e %.4e " %(np.min(v2),np.max(v2)))

      #filename = 'velocity_mesh_{:04d}.ascii'.format(istep) 
      #np.savetxt(filename,np.array([xV,yV,u2,v2]).T,header='# x,y,u,v')

      xV+=dt*u2
      yV+=dt*v2

   filename = 'surface_topography_{:04d}.ascii'.format(istep) 
   np.savetxt(filename,np.array([xV[NV-(2*nelx+1):NV],yV[NV-(2*nelx+1):NV]]).T,header='# x,y')

   print("     -> max(yV) %.7e " %(np.max(yV)/Ly))

   elevation[istep,0]=istep*dt/Myear
   elevation[istep,1]=np.min(yV[NV-(2*nelx+1):NV])-Ly
   elevation[istep,2]=np.max(yV[NV-(2*nelx+1):NV])-Ly
                 
   bcsurffile.close()

   #################################################################
   # build FE matrix
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   #################################################################
   start = timing.time()

   print('solving FEM problem')

   if sparse:
      A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
   else:   
      K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
      G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

   f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
   h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
   b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
   N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
   NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
   NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
   dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
   dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
   dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
   dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
   c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

   for iel in range(0,nel):

       # set arrays to 0 every loop
       f_el =np.zeros((mV*ndofV),dtype=np.float64)
       K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
       G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
       h_el=np.zeros((mP*ndofP),dtype=np.float64)

       for iq in range(0,nqperdim):
           for jq in range(0,nqperdim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]

               NNNV[0:mV]=NNV(rq,sq)
               dNNNVdr[0:mV]=dNNVdr(rq,sq)
               dNNNVds[0:mV]=dNNVds(rq,sq)
               NNNP[0:mP]=NNP(rq,sq)

               # calculate jacobian matrix
               jcb=np.zeros((ndim,ndim),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               jcob = np.linalg.det(jcb)
               jcbi = np.linalg.inv(jcb)

               # compute dNdx & dNdy
               for k in range(0,mV):
                   dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                   dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

               # construct 3x8 b_mat matrix
               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]

               # compute elemental a_mat matrix
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta_e[iel]*weightq*jcob

               # compute elemental rhs vector
               for i in range(0,mV):
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rho_e[iel]*gy

               for i in range(0,mP):
                   N_mat[0,i]=NNNP[i]
                   N_mat[1,i]=NNNP[i]
                   N_mat[2,i]=0.

               G_el-=b_mat.T.dot(N_mat)*weightq*jcob

           # end for jq
       # end for iq

       # impose b.c. 
       for k1 in range(0,mV):
           for i1 in range(0,ndofV):
               ikk=ndofV*k1          +i1
               m1 =ndofV*iconV[k1,iel]+i1
               if bc_fix[m1]:
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,mV*ndofV):
                      f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                      K_el[ikk,jkk]=0
                      K_el[jkk,ikk]=0
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=K_ref*bc_val[m1]
                  h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0

       G_el*=eta_ref/Ly
       h_el*=eta_ref/Ly

       # assemble matrix K_mat and right hand side rhs
       for k1 in range(0,mV):
           for i1 in range(0,ndofV):
               ikk=ndofV*k1          +i1
               m1 =ndofV*iconV[k1,iel]+i1
               for k2 in range(0,mV):
                   for i2 in range(0,ndofV):
                       jkk=ndofV*k2          +i2
                       m2 =ndofV*iconV[k2,iel]+i2
                       if sparse:
                          A_sparse[m1,m2] += K_el[ikk,jkk]
                       else:
                          K_mat[m1,m2]+=K_el[ikk,jkk]
               for k2 in range(0,mP):
                   jkk=k2
                   m2 =iconP[k2,iel]
                   if sparse:
                      A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                      A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                   else:
                      G_mat[m1,m2]+=G_el[ikk,jkk]
               f_rhs[m1]+=f_el[ikk]
       for k2 in range(0,mP):
           m2=iconP[k2,iel]
           h_rhs[m2]+=h_el[k2]

   if not sparse:
      print("     -> K_mat (m,M) %.4e %.4e " %(np.min(K_mat),np.max(K_mat)))
      print("     -> G_mat (m,M) %.4e %.4e " %(np.min(G_mat),np.max(G_mat)))

   print("build FE matrix: %.3fs - %d elts" % (timing.time()-start, nel))

   ######################################################################
   # assemble K, G, GT, f, h into A and rhs
   ######################################################################
   start = timing.time()

   rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   rhs[0:NfemV]=f_rhs
   rhs[NfemV:Nfem]=h_rhs

   if not sparse:
      a_mat = np.zeros((Nfem,Nfem),dtype=np.float64) 
      a_mat[0:NfemV,0:NfemV]=K_mat
      a_mat[0:NfemV,NfemV:Nfem]=G_mat
      a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

   print("assemble blocks: %.3f s" % (timing.time() - start))

   ######################################################################
   # assign extra pressure b.c. to remove null space
   ######################################################################
   #if sparse:
   #   A_sparse[Nfem-1,:]=0
   #   A_sparse[:,Nfem-1]=0
   #   A_sparse[Nfem-1,Nfem-1]=1
   #   rhs[Nfem-1]=0
   #else:
   #   a_mat[Nfem-1,:]=0
   #   a_mat[:,Nfem-1]=0
   #   a_mat[Nfem-1,Nfem-1]=1
   #   rhs[Nfem-1]=0

   ######################################################################
   # solve system
   ######################################################################
   start = timing.time()

   if sparse:
      sparse_matrix=A_sparse.tocsr()
   else:
      sparse_matrix=sps.csr_matrix(a_mat)


   sol=sps.linalg.spsolve(sparse_matrix,rhs)

   print("solve time: %.3f s" % (timing.time() - start))

   ######################################################################
   # put solution into separate x,y velocity arrays
   ######################################################################
   start = timing.time()

   u,v=np.reshape(sol[0:NfemV],(NV,2)).T
   p=sol[NfemV:Nfem]*(eta_ref/Ly)

   print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
   np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

   print("split vel into u,v: %.3f s" % (timing.time() - start))

   #####################################################################
   # project pressure onto velocity grid
   #####################################################################
   start = timing.time()

   q=np.zeros(NV,dtype=np.float64)
   c=np.zeros(NV,dtype=np.float64)

   for iel in range(0,nel):
       for i in range(0,mV):
           NNNP[0:mP]=NNP(rVnodes[i],sVnodes[i])
           q[iconV[i,iel]]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
           c[iconV[i,iel]]+=1.
       # end for i
   # end for iel

   q=q/c

   np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

   print("project p onto Vnodes: %.3f s" % (timing.time() - start))

   #####################################################################
   # compute vrms and total volume 
   #####################################################################
   start = timing.time()

   vol = np.zeros(nel,dtype=np.float64)        # right hand side f 

   for iel in range (0,nel):
       for iq in range(0,nqperdim):
           for jq in range(0,nqperdim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV[0:mV]=NNV(rq,sq)
               dNNNVdr[0:mV]=dNNVdr(rq,sq)
               dNNNVds[0:mV]=dNNVds(rq,sq)
               jcb=np.zeros((ndim,ndim),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               jcob = np.linalg.det(jcb)
               uq=0.
               vq=0.
               for k in range(0,mV):
                   uq+=NNNV[k]*u[iconV[k,iel]]
                   vq+=NNNV[k]*v[iconV[k,iel]]
               vrms[istep,1]+=(uq**2+vq**2)*weightq*jcob
               volume[istep,1]+=weightq*jcob
               vol[iel]+=weightq*jcob
            # end for jq
        # end for iq
    # end for iel

   vrms[istep,0]=istep*dt/Myear
   vrms[istep,1]=np.sqrt(vrms[istep,1]/(Lx*Ly))

   volume[istep,0]=istep*dt/Myear
   volume[istep,1]/=Lx*Ly
   volume[istep,2]=volume[istep,1]-1.

   print("     -> vrms   = %.6e" %(vrms[istep,1]))

   print("compute vrms: %.3fs" % (timing.time() - start))

   #####################################################################
   # plot of solution
   #####################################################################
   start = timing.time()

   filename = 'solution_{:04d}.vtu'.format(istep) 
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='volume' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" % vol[iel] )
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" % (eta_e[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" % (rho_e[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d \n" %(iconV[0,iel],iconV[1,iel],\
                                                    iconV[2,iel],iconV[3,iel],\
                                                    iconV[4,iel],iconV[5,iel],\
                                                    iconV[6,iel],iconV[7,iel]))
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

   print("export to vtu: %.3f s" % (timing.time() - start))

   np.savetxt('elevation.ascii',np.array([elevation[0:istep,0],elevation[0:istep,1],elevation[0:istep,2]]).T)
   np.savetxt('vrms.ascii',np.array([vrms[0:istep,0],vrms[0:istep,1]]).T)
   np.savetxt('volume.ascii',np.array([volume[0:istep,0],volume[0:istep,1],volume[0:istep,2]]).T)

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
