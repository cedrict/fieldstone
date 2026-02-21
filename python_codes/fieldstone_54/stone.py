import sys as sys
import numpy as np
import time as clock
import scipy.sparse as sps
from scipy.sparse import lil_matrix

eps=1e-8
cm=0.01
km=1e3
year=3600.*24.*365.
Myear=1e6*year
MPa=1e6

###############################################################################

def basis_functions_V(r,s):
    N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N4=    (1.-r**2) * 0.5*s*(s-1.)
    N5= 0.5*r*(r+1.) *    (1.-s**2)
    N6=    (1.-r**2) * 0.5*s*(s+1.)
    N7= 0.5*r*(r-1.) *    (1.-s**2)
    N8=    (1.-r**2) *    (1.-s**2)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr4=       (-2.*r) * 0.5*s*(s-1)
    dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr6=       (-2.*r) * 0.5*s*(s+1)
    dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds5= 0.5*r*(r+1.) *       (-2.*s)
    dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds7= 0.5*r*(r-1.) *       (-2.*s)
    dNds8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

#experiment=1:  relaxation of topo
#experiment=2:  extension symmetric
#experiment=3:  extension asymmetric
#experiment=4:  extension symmetric + bottom inflow
#experiment=5:  extension asymmetric + bottom inflow
#experiment=6:  pure advection
#experiment=7:  pure advection + cosine bump
#experiment=8:  pure advection + pyramid bump
#experiment=9:  rayleigh-taylor 
#experiment=10: multilayer relaxation

experiment=10

gy=-10.
rho1=3200.
eta1=1e22
eta_ref=1e22

match experiment:
 case 1:
   Lx=512e3
   Ly=512e3
   nelx = 24
   nely = 24
 case 2 | 3 | 4 |5 | 6 | 7 | 8:
   Lx=128e3
   Ly=32e3
   nelx = 100
   nely = 25
 case 7 | 8:
   eta1=1e26
   eta_ref=1e26
   width=15e3
   xbump=0.345678*Lx
   #width=16e3
   #xbump=0.5*Lx
 case 9:
   Lx=500e3
   Ly=500e3
   nelx=25
   nely=25
   eta_ref=1e20
 case 10:
   #Ly=3000e3 
   #nely=30 
   #Ly=4000e3
   #nely=40  
   Ly=5000e3
   nely=50  
   gy=-9.8
   eta_ref=1e21
   w0=5*km
   if int(len(sys.argv)==4):
      nelx=int(sys.argv[1])
      Lx=float(sys.argv[2])
      rrr=float(sys.argv[3])
   else:
      nelx=10
      Lx=500e3
      rrr=3  # viscosities ratio
   laambda=2*Lx

hx=Lx/nelx
hy=Ly/nely

#print (nelx,Lx,hx)
#exit()

###############################################################################

print("*******************************")
print("********** stone 054 **********")
print("*******************************")

ndim=2
ndof_V=2
m_V=9
m_P=4

nel=nelx*nely
nn_V=(2*nelx+1)*(2*nely+1)
nn_P=(nelx+1)*(nely+1)
Nfem_V=nn_V*ndof_V
Nfem_P=nn_P
Nfem=Nfem_V+Nfem_P

nq_per_dim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

dt=1e3*year
nstep=3

method=1
vertical_only=False

every=2

debug=False

print('nelx  =',nelx)
print('nely  =',nely)
print('nel   =',nel)
print('nn_V  =',nn_V)
print('nn_P  =',nn_P)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print('hx    =',hx)
print('hy    =',hy)
print("*******************************")

r_V=[-1,1,1,-1,0,1,0,-1,0]
s_V=[-1,-1,1,1,-1,0,1,0,0]

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)
x_P=np.zeros(nn_P,dtype=np.float64)
y_P=np.zeros(nn_P,dtype=np.float64)

counter=0
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter+=1

counter=0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_P[counter]=i*hx
        y_P[counter]=j*hy
        counter+=1

if debug: np.savetxt('grid_V.ascii',np.array([x_V,y_V]).T,header='# x,y')
if debug: np.savetxt('grid_P.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

nnx=2*nelx+1
nny=2*nely+1

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
        icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
        icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
        icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

print("connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions 
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

match experiment:
 case 1 | 9 | 10:
   uleft=0.
   uright=0.
 case 2 | 4:
   uleft=-1.*cm/year
   uright=+1.*cm/year
 case 3 | 5:
   uleft=0
   uright=+2.*cm/year
 case 6 | 7 | 8:
   uleft=+1.*cm/year
   uright=+1.*cm/year

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = uleft
    if x_V[i]/Lx>(1-eps):
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = uright
    if y_V[i]/Ly<eps:
       if experiment==4 or experiment==5:
          if x_V[i]>= Lx/2.+1e-5:
             bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = uright
          elif x_V[i]<= Lx/2.-1e-5:
             bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = uleft
          else:
             bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0.
          #end if 
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.5*cm/year
       else:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.

print("boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# flag nodes on boundaries
###############################################################################
start=clock.time()

on_left_boundary=np.zeros(nn_V,dtype=bool)  
on_right_boundary=np.zeros(nn_V,dtype=bool) 
on_bottom_boundary=np.zeros(nn_V,dtype=bool)
on_top_boundary=np.zeros(nn_V,dtype=bool)

for i in range(0,nn_V):
    if x_V[i]/Lx<eps: on_left_boundary[i]=True
    if x_V[i]/Lx>(1-eps): on_right_boundary[i]=True
    if y_V[i]/Ly<eps: on_bottom_boundary[i]=True
    if y_V[i]/Ly>1-eps: on_top_boundary[i]=True

print("flag nodes on boundaries: %.3f s" % (clock.time()-start))

###############################################################################
# compute element center coords
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64) 
y_e=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    rq=0 ; sq=0
    N_V=basis_functions_V(rq,sq)
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

print("compute elt center cords: %.3f s" % (clock.time()-start))

###############################################################################
# assign elemental rho and eta
###############################################################################
start=clock.time()

eta_e=np.zeros(nel,dtype=np.float64)
rho_e=np.zeros(nel,dtype=np.float64)

match experiment:
 case 1 | 2 | 3 | 4 | 5 |6: 
      eta_e[:]=eta1
      rho_e[:]=rho1
 case 7 | 8: 
      eta_e[:]=1e26
      rho_e[:]=rho1
 case 9: 
      for iel in range(0,nel):
          if y_V[icon_V[0,iel]]<400e3:
             eta_e[iel]=1e20
             rho_e[iel]=3200
          else:
             eta_e[iel]=1e21
             rho_e[iel]=3300
 case 10: 
      for iel in range(0,nel):
          if y_e[iel]<Ly-600e3:
             eta_e[iel]=1e21*rrr
             rho_e[iel]=3300
          else:
             eta_e[iel]=1e21
             rho_e[iel]=3300
 case _ :
      exit('unknown experiment')

print("assign rho & eta to elements: %.3f s" % (clock.time()-start))

###############################################################################
# define initial elevation 
###############################################################################
start=clock.time()

match experiment:
 case 1:
   for i in range(nn_V-(2*nelx+1),nn_V):
       y_V[i]+=1000.*np.cos(x_V[i]/Lx*np.pi)

 case 7:
   for i in range(nn_V-(2*nelx+1),nn_V):
       if x_V[i]>xbump-width and x_V[i]<xbump+width:
          #y_V[i]+=1000.*np.cos((x_V[i]-xbump)/width*np.pi/2 )
          y_V[i]+=1000.*(1+np.cos((x_V[i]-xbump)/width*np.pi ))

 case 8:
   for i in range(nn_V-(2*nelx+1),nn_V):
       if x_V[i]>xbump-width and x_V[i]<=xbump:
          y_V[i]+=(x_V[i]-(xbump-width))*0.2
       if x_V[i]>xbump and x_V[i]<xbump+width:
          y_V[i]+=-(x_V[i]-(xbump+width))*0.2

 case 9:
   for i in range(0,nn_V):
       if np.abs(y_V[i]-400e3)<1:
          y_V[i]-=5e3*np.cos(x_V[i]/Lx*np.pi*2)
   for iel in range(0,iel):
       y_V[icon_V[5,iel]]=(y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])/2.
       y_V[icon_V[7,iel]]=(y_V[icon_V[0,iel]]+y_V[icon_V[3,iel]])/2.
       y_V[icon_V[8,iel]]=(y_V[icon_V[4,iel]]+y_V[icon_V[6,iel]])/2.

   counter=0
   for j in range(0,2*nely+1):
       for i in range(0,2*nelx+1):
           x_V[counter]=i*hx/2.
           y_V[counter]=j*hy/2. 
           if y_V[counter]<=400e3:
              y_V[counter]-= 5e3*y_V[counter]/400e3  *np.cos(x_V[counter]/Lx*np.pi*2)
           else: 
              y_V[counter]-= 5e3*(Ly-y_V[counter])/100e3  *np.cos(x_V[counter]/Lx*np.pi*2)
           counter += 1

 case 10:
      #3--6--2
      #|  |  |
      #7--8--5 
      #|  |  |
      #0--4--1

   for i in range(0,nn_V):
       if on_top_boundary[i]:
          y_V[i]+=-w0*np.cos(2*np.pi/laambda*x_V[i]) 
   for iel in range(0,iel):
       y_V[icon_V[6,iel]]=(y_V[icon_V[2,iel]]+y_V[icon_V[3,iel]])/2. # straighten edge
       y_V[icon_V[5,iel]]=(y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])/2. # move node to middle
       y_V[icon_V[7,iel]]=(y_V[icon_V[0,iel]]+y_V[icon_V[3,iel]])/2. # move node to middle
       y_V[icon_V[8,iel]]=(y_V[icon_V[4,iel]]+y_V[icon_V[6,iel]])/2. # move node to middle

print("assign initial elevation: %.3f s" % (clock.time()-start))

#==============================================================================
#==============================================================================
# time stepping
#==============================================================================
#==============================================================================

elevation=np.zeros((nstep,4),dtype=np.float64)
vrms=np.zeros((nstep,2),dtype=np.float64) 
volume=np.zeros((nstep,3),dtype=np.float64) 

u=np.zeros(nn_V,dtype=np.float64) 
v=np.zeros(nn_V,dtype=np.float64) 

for istep in range(0,nstep):

   print("----------------------------------")
   print("istep= ", istep)
   print("----------------------------------")

   if istep%every==0:
      filename = 'bcsurf_{:04d}.ascii'.format(istep) 
      bcsurffile=open(filename,"w")

   #################################################################
   # mesh deformation
   #################################################################
   start=clock.time()

   if method==1: # Lagrangian approach
      for i in range(0,nn_V):
          if on_left_boundary[i] or on_right_boundary[i]: # only vertical movement
             y_V[i]+=dt*v[i]
          else:
             if not vertical_only: x_V[i]+=dt*u[i]
             y_V[i]+=dt*v[i]
          #end if
      #end for

   if method==2: # my ALE
      for i in range(0,nn_V):
          if on_top_boundary[i]:
             if not vertical_only: x_V[i]+=u[i]*dt
             y_V[i]+=v[i]*dt

   if method==3: # aspect ALE

      print('computing vmesh for bc with Q1 elements')

      #first compute the L2 projection of the mesh velocity

      A_FS=np.zeros(((nelx+1)*ndim,(nelx+1)*ndim),dtype=np.float64)
      b_FS=np.zeros((nelx+1)*ndim,dtype=np.float64)
      icon_FS=np.zeros((2,nelx),dtype=np.int32)
      x_FS=np.zeros((nelx+1),dtype=np.float64)
      y_FS=np.zeros((nelx+1),dtype=np.float64)

      counter=0
      for j in range(0,nely):
          for i in range(0,nelx):
              if j==nely-1: # if element is top row
                 # compute normal to element edge
                 x1=x_V[icon_V[3,counter]] ; y1=y_V[icon_V[3,counter]]
                 x2=x_V[icon_V[2,counter]] ; y2=y_V[icon_V[2,counter]]
                 n_x=-y2+y1
                 n_y= x2-x1
                 nnorm=np.sqrt(n_x**2+n_y**2)
                 n_x/=nnorm
                 n_y/=nnorm
                 d12=np.sqrt((x1-x2)**2+(y1-y2)**2)
                 jcob=d12/2
                 #build connectivity array on the fly
                 icon_FS[0,i]=i+0 
                 icon_FS[1,i]=i+1 
                 x_FS[icon_FS[0,i]]=x_V[icon_V[3,counter]]
                 x_FS[icon_FS[1,i]]=x_V[icon_V[2,counter]]
                 y_FS[icon_FS[0,i]]=y_V[icon_V[3,counter]]
                 y_FS[icon_FS[1,i]]=y_V[icon_V[2,counter]]

                 A_el=np.zeros((4,4),dtype=np.float64)
                 b_el=np.zeros((4),dtype=np.float64)
                 for iq in [-1,1]:
                     rq=iq/np.sqrt(3.)
                     weightq=1. 
                     JxWq=weightq*jcob
                     N1= 0.5*(1-rq) 
                     N2= 0.5*(1+rq) 
                     xq=N1*x_V[icon_V[3,counter]]+N2*x_V[icon_V[2,counter]]
                     yq=N1*y_V[icon_V[3,counter]]+N2*y_V[icon_V[2,counter]]
                     uq=N1*u[icon_V[3,counter]]+N2*u[icon_V[2,counter]]
                     vq=N1*v[icon_V[3,counter]]+N2*v[icon_V[2,counter]]
                     bob=uq*n_x+vq*n_y
                     A_el[0,0]+=N1*N1*JxWq ; A_el[0,2]+=N1*N2*JxWq 
                     A_el[1,1]+=N1*N1*JxWq ; A_el[1,3]+=N1*N2*JxWq 
                     A_el[2,0]+=N2*N1*JxWq ; A_el[2,2]+=N2*N2*JxWq 
                     A_el[3,1]+=N2*N1*JxWq ; A_el[3,3]+=N2*N2*JxWq 
                     b_el[0]+=bob*N1*n_x*JxWq
                     b_el[1]+=bob*N1*n_y*JxWq
                     b_el[2]+=bob*N2*n_x*JxWq
                     b_el[3]+=bob*N2*n_y*JxWq
                 #end for iq
      
                 # assemble matrix A_FS and rhs b_FS
                 for k1 in range(0,2):
                     for i1 in range(0,ndof_V):
                         ikk=ndof_V*k1          +i1
                         m1 =ndof_V*icon_FS[k1,i]+i1
                         for k2 in range(0,2):
                             for i2 in range(0,ndof_V):
                                 jkk=ndof_V*k2          +i2
                                 m2 =ndof_V*icon_FS[k2,i]+i2
                                 A_FS[m1,m2] += A_el[ikk,jkk]
                             #end for
                         #end for
                         b_FS[m1]+=b_el[ikk]
                     #end for i1
                 #end for k1
              #end if
              counter+=1
          #end for
      #end for

      print("     -> A_FS (m,M) %.4e %.4e " %(np.min(A_FS),np.max(A_FS)))
      print("     -> b_FS (m,M) %.4e %.4e " %(np.min(b_FS),np.max(b_FS)))

      #sparse_matrix=sps.csr_matrix(A_FS)
      sol=sps.linalg.spsolve(A_FS.tocsr(),b_FS)
      umesh_fs,vmesh_fs=np.reshape(sol[0:2*(nelx+1)],(nelx+1,2)).T

      print("     -> umesh_fs (m,M) %.4e %.4e " %(np.min(umesh_fs),np.max(umesh_fs)))
      print("     -> vmesh_fs (m,M) %.4e %.4e " %(np.min(vmesh_fs),np.max(vmesh_fs)))
   
      if istep%every==0:
         filename = 'velmesh_Q1_{:04d}.ascii'.format(istep) 
         np.savetxt(filename,np.array([x_FS,y_FS,umesh_fs,vmesh_fs]).T)

      ####################################################3
      #3--6--2  <--- surface element
      #|  |  |
      #7--8--5 
      #|  |  |
      #0--4--1

      print('computing vmesh for bc with Q2 elements')

      A_FS=np.zeros((nnx*ndim,nnx*ndim),dtype=np.float64)
      b_FS=np.zeros(nnx*ndim,dtype=np.float64)
      A_el=np.zeros((6,6),dtype=np.float64)
      b_el=np.zeros((6),dtype=np.float64)
      icon_FS=np.zeros((3,nelx),dtype=np.int32)
      umesh_fs=np.zeros(nnx,dtype=np.float64)
      vmesh_fs=np.zeros(nnx,dtype=np.float64)

      counter=0
      for j in range(0,nely):
          for i in range(0,nelx):
              if j==nely-1: # if element is top row

                 # compute normal to element edge
                 x1=x_V[icon_V[3,counter]] ; y1=y_V[icon_V[3,counter]]
                 x2=x_V[icon_V[2,counter]] ; y2=y_V[icon_V[2,counter]]
                 n_x=-y2+y1
                 n_y= x2-x1
                 nnorm=np.sqrt(n_x**2+n_y**2)
                 n_x/=nnorm
                 n_y/=nnorm
                 d12=np.sqrt((x1-x2)**2+(y1-y2)**2)
                 jcob=d12/2

                 #build connectivity array on the fly
                 icon_FS[0,i]=2*i+0 
                 icon_FS[1,i]=2*i+1 
                 icon_FS[2,i]=2*i+2

                 A_el[:,:]=0.
                 b_el[:]=0.
                 for iq in range(0,nq_per_dim):
                     rq=qcoords[iq]
                     weightq=qweights[iq] ; JxW=weightq*jcob
                     N1= 0.5*rq*(rq-1.) 
                     N2=     (1.-rq**2) 
                     N3= 0.5*rq*(rq+1.) 
                     xq=N1*x_V[icon_V[3,counter]]+N2*x_V[icon_V[6,counter]]+N3*x_V[icon_V[2,counter]]
                     yq=N1*y_V[icon_V[3,counter]]+N2*y_V[icon_V[6,counter]]+N3*y_V[icon_V[2,counter]]
                     uq=N1*u[icon_V[3,counter]]+N2*u[icon_V[6,counter]]+N3*u[icon_V[2,counter]]
                     vq=N1*v[icon_V[3,counter]]+N2*v[icon_V[6,counter]]+N3*v[icon_V[2,counter]]
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
          
                 # assemble matrix A_FS and rhs b_FS
                 for k1 in range(0,3):
                     for i1 in range(0,ndof_V):
                         ikk=ndof_V*k1          +i1
                         m1 =ndof_V*icon_FS[k1,i]+i1
                         for k2 in range(0,3):
                             for i2 in range(0,ndof_V):
                                 jkk=ndof_V*k2          +i2
                                 m2 =ndof_V*icon_FS[k2,i]+i2
                                 A_FS[m1,m2]+=A_el[ikk,jkk]
                             #end for
                         #end for
                         b_FS[m1]+=b_el[ikk]
                     #end for i1
                 #end for k1
              #end if
              counter+=1
          #end for
      #end for

      print("     -> A_FS (m,M) %.4e %.4e " %(np.min(A_FS),np.max(A_FS)))
      print("     -> b_FS (m,M) %.4e %.4e " %(np.min(b_FS),np.max(b_FS)))

      sparse_matrix=sps.csr_matrix(A_FS)
      sol=sps.linalg.spsolve(sparse_matrix,b_FS)
      #sol=sps.linalg.spsolve(A_FS,b_FS)
      umesh_fs,vmesh_fs=np.reshape(sol[0:2*nnx],(nnx,2)).T

      print("     -> umesh_fs (m,M) %.4e %.4e " %(np.min(umesh_fs),np.max(umesh_fs)))
      print("     -> vmesh_fs (m,M) %.4e %.4e " %(np.min(vmesh_fs),np.max(vmesh_fs)))
   
      if istep%every==0:
         filename = 'velmesh_Q2_{:04d}.ascii'.format(istep) 
         np.savetxt(filename,np.array([x_V[nn_V-(2*nelx+1):nn_V],y_V[nn_V-(2*nelx+1):nn_V],umesh_fs,vmesh_fs]).T)

      #################################################

      print('solving Laplace problem for vmesh')

      B=np.zeros((4,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
      jcb=np.zeros((ndim,ndim),dtype=np.float64)
      A_fem=lil_matrix((Nfem_V,Nfem_V),dtype=np.float64)
      b_fem=np.zeros(Nfem_V,dtype=np.float64) 
      u2=np.zeros(nn_V,dtype=np.float64)   
      v2=np.zeros(nn_V,dtype=np.float64)  

      # generate bc for left, ritgh, bottom boundaries 
      bc_fix2 = np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
      bc_val2 = np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value
      for i in range(0,nn_V):
          if on_left_boundary[i] or on_right_boundary[i]:
             bc_fix2[i*ndof_V  ] = True               ; bc_val2[i*ndof_V  ] = 0.
             bc_fix2[i*ndof_V+1] = bc_fix[i*ndof_V+1] ; bc_val2[i*ndof_V+1] = bc_val[i*ndof_V+1] 
          if on_bottom_boundary[i]:
             bc_fix2[i*ndof_V  ] = True ; bc_val2[i*ndof_V  ] = 0.
             bc_fix2[i*ndof_V+1] = True ; bc_val2[i*ndof_V+1] = 0.

      # compute normal vector to surface
      nx=np.zeros(nn_V,dtype=np.float64) 
      ny=np.zeros(nn_V,dtype=np.float64) 
      
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
                    n_x=-y_V[counter+1]+y_V[counter]
                    n_y= x_V[counter+1]-x_V[counter]
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
         for i in range(nn_V-(2*nelx+1),nn_V):
             nnorm=np.sqrt(nx[i]**2+ny[i]**2)
             nx[i]/=nnorm
             ny[i]/=nnorm
             if i==nn_V-(2*nelx+1): #left point of surface
                nx[i]=0.
                ny[i]=1.
             #end if
             if i==nn_V-1: #right point of surface
                nx[i]=0.
                ny[i]=1.
             #end if
             #print(x_V[i],y_V[i],nx[i],ny[i])

      #generate bc for top surface
      counter=0
      for j in range(0,2*nely+1):
          for i in range(0,2*nelx+1):
              if j==2*nely:
                 bob=u[counter]*nx[counter]+v[counter]*ny[counter]
                 bc_fix2[2*counter  ]=True ; bc_val2[2*counter  ]=bob*nx[counter]
                 bc_fix2[2*counter+1]=True ; bc_val2[2*counter+1]=bob*ny[counter]
                 if istep%every==0: 
                    bcsurffile.write("%e %e %e %e %e %e %e %e %e\n"
                                      %(x_V[counter],y_V[counter],
                                        nx[counter],ny[counter],
                                        bob,
                                        bob*nx[counter],bob*ny[counter],
                                        u[counter]*nx[counter],
                                        v[counter]*ny[counter]))
                 #end if
              #end if
              counter += 1
          #end for i
      #end for j

      # build FE system
      for iel in range(0,nel):
          K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
          f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
          for iq in range(0,nq_per_dim):
              for jq in range(0,nq_per_dim):
                  rq=qcoords[iq]
                  sq=qcoords[jq]
                  weightq=qweights[iq]*qweights[jq]
                  N_V=basis_functions_V(rq,sq)
                  dNdr_V=basis_functions_V_dr(rq,sq)
                  dNds_V=basis_functions_V_ds(rq,sq)
                  jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                  jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                  jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                  jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                  jcbi=np.linalg.inv(jcb)
                  JxWq=np.linalg.det(jcb)*weightq
                  dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                  dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                  for i in range(0,m_V):
                      B[0:4,2*i:2*i+2] = [[dNdx_V[i],0],
                                          [dNdy_V[i],0],
                                          [0,dNdx_V[i]],
                                          [0,dNdy_V[i]]]

                  K_el+=B.T.dot(B)*JxWq
              # end for jq
          # end for iq

          # impose b.c. 
          for k1 in range(0,m_V):
              for i1 in range(0,ndof_V):
                  ikk=ndof_V*k1          +i1
                  m1 =ndof_V*icon_V[k1,iel]+i1
                  if bc_fix2[m1]:
                     K_ref=K_el[ikk,ikk] 
                     for jkk in range(0,m_V*ndof_V):
                         f_el[jkk]-=K_el[jkk,ikk]*bc_val2[m1]
                         K_el[ikk,jkk]=0
                         K_el[jkk,ikk]=0
                     K_el[ikk,ikk]=K_ref
                     f_el[ikk]=K_ref*bc_val2[m1]

          # assemble matrix K_mat and right hand side rhs
          for k1 in range(0,m_V):
              for i1 in range(0,ndof_V):
                  ikk=ndof_V*k1          +i1
                  m1 =ndof_V*icon_V[k1,iel]+i1
                  for k2 in range(0,m_V):
                      for i2 in range(0,ndof_V):
                          jkk=ndof_V*k2          +i2
                          m2 =ndof_V*icon_V[k2,iel]+i2
                          A_fem[m1,m2] += K_el[ikk,jkk]
                      #end for
                  #end for
                  b_fem[m1]+=f_el[ikk]

      sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)
      u2,v2=np.reshape(sol[0:Nfem_V],(nn_V,2)).T

      print("     -> umesh (m,M) %.4e %.4e " %(np.min(u2),np.max(u2)))
      print("     -> vmesh (m,M) %.4e %.4e " %(np.min(v2),np.max(v2)))

      if debug:
         filename = 'velocity_mesh_{:04d}.ascii'.format(istep) 
         np.savetxt(filename,np.array([x_V,y_V,u2,v2]).T,header='# x,y,u,v')

      x_V+=dt*u2
      y_V+=dt*v2

   #end if method=3

   if istep%every==0:
      bcsurffile.close()
      filename = 'surface_topography_{:04d}.ascii'.format(istep) 
      np.savetxt(filename,np.array([x_V[nn_V-(2*nelx+1):nn_V],\
                                    y_V[nn_V-(2*nelx+1):nn_V]]).T,header='# x,y')

   print("     -> max(y_V) %.7e " %(np.max(y_V)/Ly))

   elevation[istep,0]=istep*dt/Myear
   elevation[istep,1]=np.min(y_V[nn_V-(2*nelx+1):nn_V])-Ly
   elevation[istep,2]=np.max(y_V[nn_V-(2*nelx+1):nn_V])-Ly
   elevation[istep,3]=y_V[nn_V-1]-Ly

   print("compute mesh deformation: %.3fs" % (clock.time()-start))

   #################################################################
   # build FE matrix
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   #################################################################
   start=clock.time()

   A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
   b_fem=np.zeros(Nfem,dtype=np.float64)
   B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
   C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
   N_mat= np.zeros((3,m_P),dtype=np.float64) # matrix  
   jcb=np.zeros((ndim,ndim),dtype=np.float64)

   for iel in range(0,nel):

       f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
       K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
       G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
       h_el=np.zeros(m_P,dtype=np.float64)

       for iq in range(0,nq_per_dim):
           for jq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]

               N_V=basis_functions_V(rq,sq)
               N_P=basis_functions_P(rq,sq)
               dNdr_V=basis_functions_V_dr(rq,sq)
               dNds_V=basis_functions_V_ds(rq,sq)
               jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
               jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
               jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
               jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
               jcbi=np.linalg.inv(jcb)
               JxWq=np.linalg.det(jcb)*weightq

               dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
               dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

               for i in range(0,m_V):
                   B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                     [0.       ,dNdy_V[i]],
                                     [dNdy_V[i],dNdx_V[i]]]

               # compute elemental a_mat matrix
               K_el+=B.T.dot(C.dot(B))*eta_e[iel]*JxWq

               # compute elemental rhs vector
               for i in range(0,m_V):
                   f_el[ndof_V*i+1]+=N_V[i]*rho_e[iel]*gy*JxWq

               for i in range(0,m_P):
                   N_mat[0,i]=N_P[i]
                   N_mat[1,i]=N_P[i]
                   N_mat[2,i]=0.

               G_el-=B.T.dot(N_mat)*JxWq

           # end for jq
       # end for iq

       # impose b.c. 
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k1          +i1
               m1 =ndof_V*icon_V[k1,iel]+i1
               if bc_fix[m1]:
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,m_V*ndof_V):
                      f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                      K_el[ikk,jkk]=0
                      K_el[jkk,ikk]=0
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=K_ref*bc_val[m1]
                  h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0

       G_el*=eta_ref/Ly
       h_el*=eta_ref/Ly

       # assemble matrix and right hand side 
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k1          +i1
               m1 =ndof_V*icon_V[k1,iel]+i1
               for k2 in range(0,m_V):
                   for i2 in range(0,ndof_V):
                       jkk=ndof_V*k2          +i2
                       m2 =ndof_V*icon_V[k2,iel]+i2
                       A_fem[m1,m2]+=K_el[ikk,jkk]
               for k2 in range(0,m_P):
                   jkk=k2
                   m2 =icon_P[k2,iel]
                   A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                   A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]
               b_fem[m1]+=f_el[ikk]
       for k2 in range(0,m_P):
           m2=icon_P[k2,iel]
           b_fem[Nfem_V+m2]+=h_el[k2]

   print("build Stokes FE matrix: %.3fs - %d elts" % (clock.time()-start,nel))

   ######################################################################
   # solve system
   ######################################################################
   start=clock.time()

   sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

   print("solve time: %.3f s" % (clock.time()-start))

   ######################################################################
   # put solution into separate x,y velocity arrays
   ######################################################################
   start=clock.time()

   u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
   p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

   print("     -> u (m,M) %.4e %.4e cm/yr" %(np.min(u)/cm*year,np.max(u)/cm*year))
   print("     -> v (m,M) %.4e %.4e cm/yr" %(np.min(v)/cm*year,np.max(v)/cm*year))
   print("     -> p (m,M) %.4e %.4e MPa" %(np.min(p)/MPa,np.max(p)/MPa))

   if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
   if debug: np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

   print("split vel into u,v: %.3f s" % (clock.time()-start))

   #####################################################################
   # project pressure onto velocity grid
   #####################################################################
   start=clock.time()

   q=np.zeros(nn_V,dtype=np.float64)
   c=np.zeros(nn_V,dtype=np.float64)

   for iel in range(0,nel):
       for i in range(0,m_V):
           N_P=basis_functions_P(r_V[i],s_V[i])
           q[icon_V[i,iel]]+=np.dot(N_P,p[icon_P[0:m_P,iel]])
           c[icon_V[i,iel]]+=1.
       # end for i
   # end for iel

   q/=c

   if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

   print("project p onto Vnodes: %.3f s" % (clock.time() - start))

   #####################################################################
   # compute vrms and total volume 
   #####################################################################
   start=clock.time()

   vol=np.zeros(nel,dtype=np.float64) 

   for iel in range (0,nel):
       for iq in range(0,nq_per_dim):
           for jq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               N_V=basis_functions_V(rq,sq)
               dNdr_V=basis_functions_V_dr(rq,sq)
               dNds_V=basis_functions_V_ds(rq,sq)
               jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
               jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
               jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
               jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
               jcbi=np.linalg.inv(jcb)
               JxWq=np.linalg.det(jcb)*weightq
               uq=np.dot(N_V,u[icon_V[:,iel]])
               vq=np.dot(N_V,v[icon_V[:,iel]])
               vrms[istep,1]+=(uq**2+vq**2)*JxWq
               volume[istep,1]+=JxWq
               vol[iel]+=JxWq
            # end for jq
        # end for iq
    # end for iel

   vrms[istep,0]=istep*dt/Myear
   vrms[istep,1]=np.sqrt(vrms[istep,1]/(Lx*Ly))

   volume[istep,0]=istep*dt/Myear
   volume[istep,1]/=Lx*Ly
   volume[istep,2]=volume[istep,1]-1.

   print("     -> vrms   = %.6e cm/yr" %(vrms[istep,1]/cm*year))

   print("compute vrms: %.3fs" % (clock.time()-start))

   ############################################################################
   # plot of solution
   ############################################################################
   start=clock.time()

   if istep%every==0:

      filename = 'solution_{:04d}.vtu'.format(istep) 
      vtufile=open(filename,"w")
      vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
      vtufile.write("<UnstructuredGrid> \n")
      vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
      #####
      vtufile.write("<Points> \n")
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
      for i in range(0,nn_V):
          vtufile.write("%.4e %.4e %.4e \n" %(x_V[i],y_V[i],0.))
      vtufile.write("</DataArray>\n")
      vtufile.write("</Points> \n")
      #####
      vtufile.write("<CellData Scalars='scalars'>\n")
      vtufile.write("<DataArray type='Float32' Name='volume' Format='ascii'> \n")
      vol.tofile(vtufile,sep=' ',format='%.4e')
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
      eta_e.tofile(vtufile,sep=' ',format='%.4e')
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
      rho_e.tofile(vtufile,sep=' ',format='%.4e')
      vtufile.write("</DataArray>\n")
      vtufile.write("</CellData>\n")
      #####
      vtufile.write("<PointData Scalars='scalars'>\n")
      #--
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
      for i in range(0,nn_V):
          vtufile.write("%.4e %.4e %.4e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
      q.tofile(vtufile,sep=' ',format='%.4e')
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("</PointData>\n")
      #####
      vtufile.write("<Cells>\n")
      #--
      vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                         icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                         icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%d \n" %((iel+1)*m_V))
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

   print("export to vtu: %.3f s" % (clock.time() - start))

   ############################################################################

   np.savetxt('elevation.ascii',np.array([elevation[0:istep,0],elevation[0:istep,1],\
                                          elevation[0:istep,2],elevation[0:istep,3]]).T)
   np.savetxt('vrms.ascii',np.array([vrms[0:istep,0],vrms[0:istep,1]]).T)
   np.savetxt('volume.ascii',np.array([volume[0:istep,0],volume[0:istep,1],volume[0:istep,2]]).T)

#==============================================================================
#==============================================================================
# end for istep
#==============================================================================
#==============================================================================

if experiment==10:

   np.savetxt('elevation_exp10.ascii',np.array([elevation[0:nstep,0],np.log(np.abs(elevation[0:nstep,1])/w0)]).T)

   
   print('tau=',4*np.pi*1e21/Lx/3300/abs(gy),'lambda=',laambda,\
         'slope=',np.log(np.abs(elevation[nstep-1,1])/w0)/elevation[nstep-1,0])

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
