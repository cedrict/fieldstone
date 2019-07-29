import numpy as np
import math as math
import sys as sys
import time as time
import solcx as solcx
import random

#------------------------------------------------------------------------------
#def viscosity(x,y):
#    if x<0.5:
#       val=1.
#    else:
#       val=1.e6
#    return val

#def density(x,y):
#    val=math.sin(math.pi*y)*math.cos(math.pi*x)
#    return val

def NQ1(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNQ1dr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNQ1ds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

def NQ2(rq,sq):
    N_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    N_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    N_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    N_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    N_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    N_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    N_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    N_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    N_8=     (1.-rq**2) *     (1.-sq**2)
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8

def dNQ2dr(rq,sq):
    dNdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNdr_8=       (-2.*rq) *    (1.-sq**2)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8

def dNQ2ds(rq,sq):
    dNds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNds_8=     (1.-rq**2) *       (-2.*sq)
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8

def interpolate_vel_on_pt(xm,ym,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q):
    ielx=int(xm/Lx*nelx)
    iely=int(ym/Ly*nely)
    iel=nelx*(iely)+ielx
    xmin=x[icon[0,iel]] ; xmax=x[icon[2,iel]]
    ymin=y[icon[0,iel]] ; ymax=y[icon[2,iel]]
    rm=((xm-xmin)/(xmax-xmin)-0.5)*2
    sm=((ym-ymin)/(ymax-ymin)-0.5)*2
    if Q==1:
       N[0:m]=NQ1(rm,sm)
    if Q==2:
       N[0:m]=NQ2(rm,sm)
    um=0.
    vm=0.
    for k in range(0,m):
        um+=N[k]*u[icon[k,iel]]
        vm+=N[k]*v[icon[k,iel]]
    return um,vm,rm,sm,iel 

def compute_CVI_corr (u,v,icon,rm,sm,iel,use_cvi,Q):
    if use_cvi==1 and Q==1:
       u01=(u[icon[0,iel]]-u[icon[1,iel]])*0.25
       u23=(u[icon[2,iel]]-u[icon[3,iel]])*0.25
       v03=(v[icon[0,iel]]-v[icon[3,iel]])*0.25
       v21=(v[icon[2,iel]]-v[icon[1,iel]])*0.25
       u_corr=0.5*(v03+v21)*(1.-rm)*(1+rm) 
       v_corr=0.5*(u01+u23)*(1.-sm)*(1+sm) 
    elif use_cvi==1 and Q==2:
       hx=0.125
       hy=0.125
       Jxx=2./hx ; Jxy=0.
       Jyx=0.    ; Jyy=2./hy
       #
       U1=Jxx*u[icon[0,iel]]+Jyx*v[icon[0,iel]]
       U2=Jxx*u[icon[1,iel]]+Jyx*v[icon[1,iel]]
       U3=Jxx*u[icon[2,iel]]+Jyx*v[icon[2,iel]]
       U4=Jxx*u[icon[3,iel]]+Jyx*v[icon[3,iel]]
       U5=Jxx*u[icon[4,iel]]+Jyx*v[icon[4,iel]]
       U6=Jxx*u[icon[5,iel]]+Jyx*v[icon[5,iel]]
       U7=Jxx*u[icon[6,iel]]+Jyx*v[icon[6,iel]]
       U8=Jxx*u[icon[7,iel]]+Jyx*v[icon[7,iel]]
       U9=Jxx*u[icon[8,iel]]+Jyx*v[icon[8,iel]]

       V1=Jxy*u[icon[0,iel]]+Jyy*v[icon[0,iel]]
       V2=Jxy*u[icon[1,iel]]+Jyy*v[icon[1,iel]]
       V3=Jxy*u[icon[2,iel]]+Jyy*v[icon[2,iel]]
       V4=Jxy*u[icon[3,iel]]+Jyy*v[icon[3,iel]]
       V5=Jxy*u[icon[4,iel]]+Jyy*v[icon[4,iel]]
       V6=Jxy*u[icon[5,iel]]+Jyy*v[icon[5,iel]]
       V7=Jxy*u[icon[6,iel]]+Jyy*v[icon[6,iel]]
       V8=Jxy*u[icon[7,iel]]+Jyy*v[icon[7,iel]]
       V9=Jxy*u[icon[8,iel]]+Jyy*v[icon[8,iel]]

       D0=(-2*U8+2*U6-2*V5+2*V7)*0.25
       D1=(4*U8-8*U9+4*U6+V1-V4-V2+V3)*0.25
       D2=(U1-U4-U2+U3+4*V5-8*V9+4*V7)*0.25
       D3=(-2*U1+2*U4+4*U5-4*U7-2*U2+2*U3\
           -2*V1+4*V8-2*V4+2*V2-4*V6+2*V3)*0.25
       D4=(-V1+V4+2*V5-2*V7-V2+V3)*0.25
       D5=(-U1+2*U8-U4+U2-2*U6+U3)*0.25
       D6=(2*V1-4*V8+2*V4-4*V5+8*V9-4*V7+2*V2-4*V6+2*V3)*0.25
       D7=(2*U1-4*U8+2*U4-4*U5+8*U9-4*U7+2*U2-4*U6+2*U3)*0.25

       alpha4=D3/2./(Jxx+Jyy)
       a1=(D4-Jxy*alpha4)/(3*Jxx)
       b1=(D5-Jyx*alpha4)/(3*Jyy)
       b3=(Jxx*D6-Jxy*D7)/(Jxx*Jyy-Jxy*Jyx)/2.
       a3=(D7-2*Jyx*b3)/(2*Jxx)
       a0=(D1+2*Jyx*b3)/(2*Jxx)
       b0=(D2+2*Jxy*a3)/(2*Jyy)

       #verification
       #print (-2*Jxx*a0+2*Jyx*b3+D1)
       #print (-2*Jxy*a3-2*Jyy*b0+D2)
       #print (-2*Jxy*a3-2*Jyy*b3+D6)
       #print (-2*Jxx*a3-2*Jyx*b3+D7)
       #print (-2*Jxx*alpha4-2*Jyy*alpha4+D3)
       #print (-3*Jxx*a1-Jxy*alpha4+D4)
       #print (-Jyx*alpha4-3*Jyy*b1+D5)

       u_corr=(1-rm**2)*(a0+a1*rm+a3*sm**2+alpha4*sm)
       v_corr=(1-sm**2)*(b0+b1*sm+b3*rm**2+alpha4*rm)

    else:
       u_corr=0.
       v_corr=0.
    return u_corr,v_corr

def stay_in (x,y):
    if (y>=-x+1. and x>1. ):
       delta=2.-(x+y)
       xnew=x-delta
       ynew=y+delta
       while xnew>1.:
          xnew-=delta
          ynew+=delta
          #print (x,y,'->',xnew,ynew)
    elif (y<=-x+1. and y<0):
       delta=x+y
       xnew=x-delta
       ynew=y+delta
       while ynew<0.:
          xnew-=delta
          ynew+=delta
          #print (x,y,'->',xnew,ynew)
    else:
       xnew=x
       ynew=y
    #if xnew>1.:
    #   print ('ARG:',x,y,'->',xnew,ynew)
    return xnew,ynew

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndof=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 10):
   nelx           =int(sys.argv[1])
   nely           =int(sys.argv[2])
   visu           =int(sys.argv[3])
   nmarker_per_dim=int(sys.argv[4])
   random_markers =int(sys.argv[5])
   CFL_nb         =float(sys.argv[6])
   RKorder        =int(sys.argv[7])
   use_cvi        =int(sys.argv[8])
   Q              =int(sys.argv[9])
else:
   nelx = 32
   nely = 32
   visu = 1
   nmarker_per_dim=8
   random_markers=0
   CFL_nb=0.25
   RKorder=2
   use_cvi=1
   Q=1
    
if Q==1:
   nnx=nelx+1    # number of elements, x direction
   nny=nely+1    # number of elements, y direction
   m=4           # number of nodes making up an element
if Q==2:
   nnx=2*nelx+1  # number of elements, x direction
   nny=2*nely+1  # number of elements, y direction
   m=9           # number of nodes making up an element

nnp=nnx*nny      # number of nodes
nel=nelx*nely    # number of elements, total

hx=Lx/float(nelx)
hy=Ly/float(nely)

nstep=1001
every=100      # vtu output frequency

rkf_c2=1./4.      
rkf_c3=3./8.    
rkf_c4=12./13.   
rkf_c5=1.    
rkf_c6=1./2.    
rkf_a21=1./4.    
rkf_a31=3./32.   
rkf_a32=9./32.                                       
rkf_a41= 1932./2197.                                  
rkf_a42=-7200./2197.         
rkf_a43= 7296./2197.        
rkf_a51= 439./216.         
rkf_a52=  -8.             
rkf_a53=3680./513.       
rkf_a54=-845./4104.     
rkf_a61=   -8./27.     
rkf_a62=    2.        
rkf_a63=-3544./2565. 
rkf_a64= 1859./4104.
rkf_a65=  -11./40.  
rkf_b1=16./135.    
rkf_b3= 6656./12825.  
rkf_b4=28561./56430. 
rkf_b5=   -9./50.   
rkf_b6=    2./55.  

tijd=0.


print("markercount_stats_nelx"+str(nelx)+\
                                  '_nm'+str(nmarker_per_dim)+\
                                "_rand"+str(random_markers)+\
                                "_CFL_"+str(CFL_nb)+\
                                  "_rk"+str(RKorder)+\
                                 "_cvi"+str(use_cvi)+\
                                   "_Q"+str(Q)+".ascii")


countfile=open("markercount_stats_nelx"+str(nelx)+\
                                  '_nm'+str(nmarker_per_dim)+\
                                "_rand"+str(random_markers)+\
                                "_CFL_"+str(CFL_nb)+\
                                  "_rk"+str(RKorder)+\
                                 "_cvi"+str(use_cvi)+\
                                   "_Q"+str(Q)+".ascii","w")

#################################################################
# grid point setup
#################################################################

print("grid point setup")

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

if Q==1:
   counter = 0
   for j in range(0, nny):
       for i in range(0, nnx):
           x[counter]=i*hx
           y[counter]=j*hy
           counter += 1

if Q==2:
   counter = 0
   for j in range(0, nny):
       for i in range(0, nnx):
           x[counter]=i*hx/2.
           y[counter]=j*hy/2.
           counter += 1

#################################################################
# connectivity
#
#  04===07===03
#  ||   ||   ||
#  08===09===06
#  ||   ||   ||
#  01===05===02
#
#################################################################
print("connectivity")

icon =np.zeros((m, nel),dtype=np.int32)

if Q==1:
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           icon[0, counter] = i + j * (nelx + 1)
           icon[1, counter] = i + 1 + j * (nelx + 1)
           icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
           icon[3, counter] = i + (j + 1) * (nelx + 1)
           counter += 1

if Q==2:
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon[0,counter]=(i)*2+1+(j)*2*nnx -1
           icon[1,counter]=(i)*2+3+(j)*2*nnx -1
           icon[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
           icon[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
           icon[4,counter]=(i)*2+2+(j)*2*nnx -1
           icon[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
           icon[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
           icon[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
           icon[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
           counter += 1

#################################################################
# assign nodal field values 
#################################################################
u = np.empty(nnp,dtype=np.float64)
v = np.empty(nnp,dtype=np.float64)
p = np.empty(nnp,dtype=np.float64)

for i in range(0,nnp):
    u[i],v[i],p[i]=solcx.SolCxSolution(x[i],y[i]) 

#################################################################

dt=CFL_nb*min(Lx/nelx,Ly/nely)/np.max(np.sqrt(u**2+v**2))
    
print('     -> dt= %.3e ' % dt)

#################################################################
# marker setup
#################################################################
start = time.time()

nmarker_per_element=nmarker_per_dim**2
nmarker=nel*nmarker_per_element

swarm_x=np.empty(nmarker,dtype=np.float64)  
swarm_y=np.empty(nmarker,dtype=np.float64)  
swarm_u=np.zeros(nmarker,dtype=np.float64)  
swarm_v=np.zeros(nmarker,dtype=np.float64)  
swarm_u_corr=np.zeros(nmarker,dtype=np.float64)  
swarm_v_corr=np.zeros(nmarker,dtype=np.float64)  

if random_markers==1:
   counter=0
   for iel in range(0,nel):
       x1=x[icon[0,iel]] ; y1=y[icon[0,iel]]
       x2=x[icon[1,iel]] ; y2=y[icon[1,iel]]
       x3=x[icon[2,iel]] ; y3=y[icon[2,iel]]
       x4=x[icon[3,iel]] ; y4=y[icon[3,iel]]
       for im in range(0,nmarker_per_element):
           # generate random numbers r,s between 0 and 1
           r=random.uniform(-1.,+1)
           s=random.uniform(-1.,+1)
           N1=0.25*(1-r)*(1-s)
           N2=0.25*(1+r)*(1-s)
           N3=0.25*(1+r)*(1+s)
           N4=0.25*(1-r)*(1+s)
           swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
           swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
           counter+=1

else:
   counter=0
   for iel in range(0,nel):
       x1=x[icon[0,iel]] ; y1=y[icon[0,iel]]
       x2=x[icon[1,iel]] ; y2=y[icon[1,iel]]
       x3=x[icon[2,iel]] ; y3=y[icon[2,iel]]
       x4=x[icon[3,iel]] ; y4=y[icon[3,iel]]
       for j in range(0,nmarker_per_dim):
           for i in range(0,nmarker_per_dim):
               r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
               s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
               N1=0.25*(1-r)*(1-s)
               N2=0.25*(1+r)*(1-s)
               N3=0.25*(1+r)*(1+s)
               N4=0.25*(1-r)*(1+s)
               swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
               swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
               counter+=1

print("     -> nmarker %d " % nmarker)
print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

#################################################################
# compute population stats
#################################################################

count=np.zeros(nel,dtype=np.int16)
for im in range (0,nmarker):
    ielx=int(swarm_x[im]/Lx*nelx)
    iely=int(swarm_y[im]/Ly*nely)
    iel=nelx*(iely)+ielx
    count[iel]+=1

print("     -> count (m,M) %.5d %.5d " %(np.min(count),np.max(count)))
    
countfile.write(" %e %d %d %e %e\n" % (tijd, np.min(count),np.max(count),\
                                             np.min(count)/nmarker_per_dim**2,\
                                             np.max(count)/nmarker_per_dim**2 ))

#################################################################
# marker paint
#################################################################
swarm_mat=np.zeros(nmarker,dtype=np.int16)  

for i in [0,2,4,6,8,10,12,14]:
    dx=Lx/16
    for im in range (0,nmarker):
        if swarm_x[im]>i*dx and swarm_x[im]<(i+1)*dx:
           swarm_mat[im]+=1

for i in [0,2,4,6,8,10,12,14]:
    dy=Ly/16
    for im in range (0,nmarker):
        if swarm_y[im]>i*dy and swarm_y[im]<(i+1)*dy:
           swarm_mat[im]+=1

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################
N = np.zeros(m,dtype=np.float64) # shape functions

for istep in range (0,nstep):

    print("----------------------------------")
    print("istep= ", istep)
    print("----------------------------------")

    if RKorder==0:

       for im in range(0,nmarker):

           swarm_u[im],swarm_v[im],ptemp=solcx.SolCxSolution(swarm_x[im],swarm_y[im]) 
           swarm_x[im]+=swarm_u[im]*dt
           swarm_y[im]+=swarm_v[im]*dt

    if RKorder==1:

       for im in range(0,nmarker):

           swarm_u[im],swarm_v[im],rm,sm,iel =interpolate_vel_on_pt(swarm_x[im],swarm_y[im],\
                                                                 x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           swarm_u_corr[im],swarm_v_corr[im] =compute_CVI_corr (u,v,icon,rm,sm,iel,use_cvi,Q)

           swarm_x[im]+=(swarm_u[im]+swarm_u_corr[im])*dt
           swarm_y[im]+=(swarm_v[im]+swarm_v_corr[im])*dt

           swarm_x[im],swarm_y[im]= stay_in (swarm_x[im],swarm_y[im])

       # end for im

    elif RKorder==2:

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           uA,vA,rm,sm,iel = interpolate_vel_on_pt(xA,yA,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uAcorr,vAcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uA+=uAcorr
           vA+=vAcorr
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           uB,vB,rm,sm,iel = interpolate_vel_on_pt(xB,yB,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uBcorr,vBcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uB+=uBcorr
           vB+=vBcorr
           #--------------
           swarm_x[im]=xA+uB*dt
           swarm_y[im]=yA+vB*dt
       # end for im

    elif RKorder==3:

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           uA,vA,rm,sm,iel = interpolate_vel_on_pt(xA,yA,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uAcorr,vAcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uA+=uAcorr
           vA+=vAcorr
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           uB,vB,rm,sm,iel = interpolate_vel_on_pt(xB,yB,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uBcorr,vBcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uB+=uBcorr
           vB+=vBcorr
           #--------------
           xC=xA+(2*uB-uA)*dt/2.
           yC=yA+(2*vB-vA)*dt/2.
           uC,vC,rm,sm,iel = interpolate_vel_on_pt(xC,yC,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uCcorr,vCcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uC+=uCcorr
           vC+=vCcorr
           #--------------
           swarm_x[im]=xA+(uA+4*uB+uC)*dt/6.
           swarm_y[im]=yA+(vA+4*vB+vC)*dt/6.
       # end for im

    elif RKorder==4:

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           uA,vA,rm,sm,iel = interpolate_vel_on_pt(xA,yA,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uAcorr,vAcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uA+=uAcorr
           vA+=vAcorr
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           uB,vB,rm,sm,iel = interpolate_vel_on_pt(xB,yB,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uBcorr,vBcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uB+=uBcorr
           vB+=vBcorr
           #--------------
           xC=xA+(2*uB-uA)*dt/2.
           yC=yA+(2*vB-vA)*dt/2.
           uC,vC,rm,sm,iel = interpolate_vel_on_pt(xC,yC,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uCcorr,vCcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uC+=uCcorr
           vC+=vCcorr
           #--------------
           xD=xA+uC*dt
           yD=yA+vC*dt
           uD,vD,rm,sm,iel = interpolate_vel_on_pt(xD,yD,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uDcorr,vDcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uD+=uDcorr
           vD+=vDcorr
           #--------------
           swarm_x[im]=xA+(uA+2*uB+2*uC+uD)*dt/6.
           swarm_y[im]=yA+(vA+2*vB+2*vC+vD)*dt/6.
       # end for im

    elif RKorder==5: # Runge-Kutta Fehlberg method

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           uA,vA,rm,sm,iel = interpolate_vel_on_pt(xA,yA,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uAcorr,vAcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uA+=uAcorr
           vA+=vAcorr
           #--------------
           xB=xA+(uA*rkf_a21)*dt
           yB=yA+(vA*rkf_a21)*dt
           uB,vB,rm,sm,iel = interpolate_vel_on_pt(xB,yB,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uBcorr,vBcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uB+=uBcorr
           vB+=vBcorr
           #--------------
           xC=xA+(uA*rkf_a31+uB*rkf_a32)*dt
           yC=yA+(vA*rkf_a31+vB*rkf_a32)*dt
           uC,vC,rm,sm,iel = interpolate_vel_on_pt(xC,yC,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uCcorr,vCcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uC+=uCcorr
           vC+=vCcorr
           #--------------
           xD=xA+(uA*rkf_a41+uB*rkf_a42+uC*rkf_a43)*dt
           yD=yA+(vA*rkf_a41+vB*rkf_a42+vC*rkf_a43)*dt
           uD,vD,rm,sm,iel = interpolate_vel_on_pt(xD,yD,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uDcorr,vDcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uD+=uDcorr
           vD+=vDcorr
           #--------------
           xE=xA+(uA*rkf_a51+uB*rkf_a52+uC*rkf_a53+uD*rkf_a54)*dt
           yE=yA+(vA*rkf_a51+vB*rkf_a52+vC*rkf_a53+vD*rkf_a54)*dt
           uE,vE,rm,sm,iel = interpolate_vel_on_pt(xE,yE,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uEcorr,vEcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uE+=uEcorr
           vE+=vEcorr
           #--------------
           xF=xA+(uA*rkf_a61+uB*rkf_a62+uC*rkf_a63+uD*rkf_a64+uE*rkf_a65)*dt
           yF=yA+(vA*rkf_a61+vB*rkf_a62+vC*rkf_a63+vD*rkf_a64+vE*rkf_a65)*dt
           uF,vF,rm,sm,iel = interpolate_vel_on_pt(xF,yF,x,y,u,v,icon,Lx,Ly,nelx,nely,m,Q)
           uFcorr,vFcorr = compute_CVI_corr(u,v,icon,rm,sm,iel,use_cvi,Q)
           uF+=uFcorr
           vF+=vFcorr
           #--------------
           swarm_x[im]=xA+(uA*rkf_b1+uC*rkf_b3+uD*rkf_b4+uE*rkf_b5+uF*rkf_b6)*dt
           swarm_y[im]=yA+(vA*rkf_b1+vC*rkf_b3+vD*rkf_b4+vE*rkf_b5+vF*rkf_b6)*dt

       # end for im

    #endif

    tijd+=dt

    #############################
    # compute population stats
    #############################

    count=np.zeros(nel,dtype=np.int16)
    for im in range (0,nmarker):
        ielx=int(swarm_x[im]/Lx*nelx)
        iely=int(swarm_y[im]/Ly*nely)
        if ielx<0:
           exit('ielx<0')
        if iely<0:
           exit('iely<0')
        if ielx>nelx-1:
           print (swarm_x[im],swarm_y[im],ielx,iely)
           exit('ielx>nelx-1')
        if iely>nely-1:
           exit('iely>nely-1')
        iel=nelx*(iely)+ielx
        count[iel]+=1

    print("     -> count (m,M) %.5d %.5d " %(np.min(count),np.max(count)))

    countfile.write(" %e %d %d %e %e\n" % (tijd, np.min(count),np.max(count),\
                                                 np.min(count)/nmarker_per_dim**2,\
                                                 np.max(count)/nmarker_per_dim**2 ))

    countfile.flush()

    #############################
    # export markers to vtk file
    #############################

    if visu==1 and istep%every==0:

       velfile=open("velocity.ascii","w")
       for im in range(0,nmarker):
           if swarm_x[im]<0.5:
              ui,vi,pi=solcx.SolCxSolution(swarm_x[im],swarm_y[im]) 
              velfile.write("%e %e %e %e %e %e %e %e\n " % (swarm_x[im],swarm_y[im],\
                                                            swarm_u[im],swarm_u_corr[im],ui,\
                                                            swarm_v[im],swarm_v_corr[im],vi))
       velfile.close()

       filename = 'markers_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[im],swarm_y[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_u[im],swarm_v[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (correction)' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_u_corr[im],swarm_v_corr[im],0.))
       vtufile.write("</DataArray>\n")
       #--
#       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (analytical)' Format='ascii'> \n")
#       for im in range(0,nmarker):
#           ui,vi,pi=solcx.SolCxSolution(swarm_x[im],swarm_y[im]) 
#           vtufile.write("%10e %10e %10e \n" %(ui,vi,0.))
#       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='error' Format='ascii'> \n")
       for im in range(0,nmarker):
           ui,vi,pi=solcx.SolCxSolution(swarm_x[im],swarm_y[im]) 
           vtufile.write("%10e %10e %10e \n" %(swarm_u[im]+swarm_u_corr[im]-ui,swarm_v[im]+swarm_v_corr[im]-vi,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='paint' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e \n" % swarm_mat[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d\n" % im )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % (im+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % 1)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()



       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" %p[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='count' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % count[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if Q==1:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
       if Q==2:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],\
                                                          icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))

       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       if Q==1:
          for iel in range (0,nel):
              vtufile.write("%d \n" %((iel+1)*m))
       if Q==2:
          for iel in range (0,nel):
              vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       if Q==1:
          for iel in range (0,nel):
              vtufile.write("%d \n" %9)
       if Q==2:
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

    # end if

    if tijd>20000: 
       break

#end for istep

countfile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")




