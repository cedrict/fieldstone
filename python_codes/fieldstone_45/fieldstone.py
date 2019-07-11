import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
from scipy.special import erf
import time as timing

#------------------------------------------------------------------------------

def NNT(r,s):
    N_0=1-3*r-3*s+2*r**2+4*r*s+2*s**2
    N_1=-r+2*r**2
    N_2=-s+2*s**2
    N_3=4*r-4*r**2-4*r*s
    N_4=4*r*s
    N_5=4*s-4*r*s-4*s**2
    return N_0,N_1,N_2,N_3,N_4,N_5

def dNNTdr(r,s):
    dNdr_0=-3+4*r+4*s
    dNdr_1=-1+4*r
    dNdr_2=0
    dNdr_3=4-8*r-4*s
    dNdr_4=4*s
    dNdr_5=-4*s
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5

def dNNTds(r,s):
    dNds_0=-3+4*r+4*s
    dNds_1=0
    dNds_2=-1+4*s
    dNds_3=-4*r
    dNds_4=4*r
    dNds_5=4-4*r-8*s
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5

#------------------------------------------------------------------------------

def compute_corner_flow_velocity(x,y,l1,l2,l3,angle,v0,Lx,Ly):
    v1=-v0
    theta0=angle
    theta1=np.pi-theta0
    l4=l3*np.tan(theta0)
    A0 = (- theta0 * np.sin(theta0))/(theta0**2-np.sin(theta0)**2 ) *v0 
    B0=0
    C0=(np.sin(theta0)-theta0*np.cos(theta0))/(theta0**2-np.sin(theta0)**2 ) * v0
    D0=-A0
    A1 =1./(theta1**2-np.sin(theta1)**2 ) * \
        ( -v0*theta1*np.sin(theta1)-v1*theta1*np.cos(theta1)*(np.sin(theta1)+theta1*np.cos(theta1))\
        +v1*(np.cos(theta1)-theta1*np.sin(theta1))*theta1*np.sin(theta1) )   
    B1=0
    C1=1./(theta1**2-np.sin(theta1)**2 ) * \
       ( v0*(np.sin(theta1)-theta1*np.cos(theta1)) + v1*theta1**2*np.cos(theta1)*np.sin(theta1) \
       - v1*(np.cos(theta1)-theta1*np.sin(theta1))*(np.sin(theta1)-theta1*np.cos(theta1)) )   
    D1=-A1

    u=0.
    v=0.

    #------------------------
    # slab left 
    #------------------------
    if y>=Ly-l1 and x<=l3:
       u=v0
       v=0.

    #------------------------
    # slab 
    #------------------------
    if x>=l3 and y<=Ly+l4-x*np.tan(theta0) and y>=Ly+l4-x*np.tan(theta0)-l1:
       u=v0*np.cos(theta0)
       v=-v0*np.sin(theta0)

    #------------------------
    # overriding plate
    #------------------------
    if y>Ly+l4-x*np.tan(theta0) and y>Ly-l2:
       u=0.0
       v=0.0

    #------------------------
    # wedge
    #------------------------
    xC=l3+l2/np.tan(theta0)
    yC=Ly-l2
    if x>xC and y<yC:
       xt=x-xC 
       yt=yC-y 
       theta=np.arctan(yt/xt) 
       r=np.sqrt((xt)**2+(yt)**2)
       if theta<theta0:
          # u_r=f'(theta)
          ur = A0*np.cos(theta)-B0*np.sin(theta) +\
               C0* (np.sin(theta)+theta*np.cos(theta)) + D0 * (np.cos(theta)-theta*np.sin(theta))
          # u_theta=-f(theta)
          utheta=- ( A0*np.sin(theta) + B0*np.cos(theta) + C0*theta*np.sin(theta) + D0*theta*np.cos(theta))
          ur=-ur
          utheta=-utheta
          u=  ur*np.cos(theta)-utheta*np.sin(theta)
          v=-(ur*np.sin(theta)+utheta*np.cos(theta)) # because of reverse orientation

    #------------------------
    # under subducting plate
    #------------------------
    xD=l3
    yD=Ly-l1
    if y<yD and y<Ly+l4-x*np.tan(theta0)-l1:
       xt=xD-x 
       yt=yD-y 
       theta=np.arctan2(yt,xt) #!; write(6548,*) theta/pi*180
       r=np.sqrt((xt)**2+(yt)**2)
       #u_r=f'(theta)
       ur = A1*np.cos(theta) - B1*np.sin(theta) + C1* (np.sin(theta)+theta*np.cos(theta)) \
            + (D1-v1) * (np.cos(theta)-theta*np.sin(theta))
       #u_theta=-f(theta)
       utheta=- ( A1*np.sin(theta) + B1*np.cos(theta) + C1*theta*np.sin(theta) + (D1-v1)*theta*np.cos(theta))
       ur=-ur
       utheta=-utheta
       u=-(ur*np.cos(theta)-utheta*np.sin(theta))
       v=-(ur*np.sin(theta)+utheta*np.cos(theta)) #! because of reverse orientation

    return u,v

#------------------------------------------------------------------------------
# Crouzeix-Raviart for Stokes
# P2 elements for temperature

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
mV=7
mP=3
mT=6

Lx=660e3
Ly=600e3

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 110
   nely = 100
   visu = 1

nel=nelx*nely*2
nnx=2*nelx+1
nny=2*nely+1
NV=nnx*nny+nel
NT=nnx*nny
NP=(nelx+1)*(nely+1)

ndofV=2
ndofP=1
ndofT=1

NfemT=NT*ndofT

print ('nelx=',nelx)
print ('nely=',nely)
print ('nnx =',nnx)
print ('nny =',nny)
print ('nel =',nel)
print ('NV  =',NV)
print ('NT  =',NT)

cm=0.01
year=31536000.
l1=1000.e3
l2=50.e3
l3=0.e3
v0=5*cm/year          # vack08
angle=45./180.*np.pi  # vack08

hcapa=1250 # vack08
hcond=3    # vack08
rho0=3300  # vack08
kappa=hcond/rho0/hcapa

#print(kappa)

eps=1e-9

#---------------------------------------
# 6 point integration coeffs and weights 

nqel=6

qcoords_r=np.empty(6,dtype=np.float64)  
qcoords_s=np.empty(6,dtype=np.float64)  
qweights=np.empty(6,dtype=np.float64)  

qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 

#################################################################
# checking that all shape functions are 1 on their node and 
# zero elsewhere
#print ('node1:',NNT(0,0))
#print ('node2:',NNT(1,0))
#print ('node3:',NNT(0,1))
#print ('node4:',NNT(0.5,0))
#print ('node5:',NNT(0.5,0.5))
#print ('node6:',NNT(0,0.5))

#################################################################
# build velocity nodes coordinates and connectivity array 
#################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates
iconV=np.zeros((mV,nel),dtype=np.int32)

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*Lx/(2*nelx) 
        yV[counter]=j*Ly/(2*nely) 
        counter+=1

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
          # lower left triangle
          iconV[0,counter]=(i)*2+1+(j)*2*nnx      -1  # 1 of q2
          iconV[1,counter]=(i)*2+3+(j)*2*nnx      -1  # 3 of q2
          iconV[2,counter]=(i)*2+1+(j)*2*nnx+nnx*2-1  # 7 of q2
          iconV[3,counter]=(i)*2+2+(j)*2*nnx      -1  # 2 of q2
          iconV[4,counter]=(i)*2+2+(j)*2*nnx+nnx  -1  # 5 of q2
          iconV[5,counter]=(i)*2+1+(j)*2*nnx+nnx  -1  # 4 of q2
          iconV[6,counter]=nnx*nny+counter
          counter=counter+1
          # upper right triangle
          iconV[0,counter]=(i)*2+3+(j)*2*nnx+nnx*2-1  # 9 of Q2
          iconV[1,counter]=(i)*2+1+(j)*2*nnx+nnx*2-1  # 7 of Q2
          iconV[2,counter]=(i)*2+3+(j)*2*nnx      -1  # 3 of Q2
          iconV[3,counter]=(i)*2+2+(j)*2*nnx+nnx*2-1  # 8 of Q2
          iconV[4,counter]=(i)*2+2+(j)*2*nnx+nnx  -1  # 5 of Q2
          iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx  -1  # 6 of Q2
          iconV[6,counter]=nnx*nny+counter
          counter=counter+1

for iel in range (0,nel): #bubble nodes
    xV[nnx*nny+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
    yV[nnx*nny+iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconV[0,iel],"at pos.",xV[iconV[0,iel]], yV[iconV[0,iel]])
#    print ("node 1",iconV[1,iel],"at pos.",xV[iconV[1,iel]], yV[iconV[1,iel]])
#    print ("node 2",iconV[2,iel],"at pos.",xV[iconV[2,iel]], yV[iconV[2,iel]])
#    print ("node 3",iconV[3,iel],"at pos.",xV[iconV[3,iel]], yV[iconV[3,iel]])
#    print ("node 4",iconV[4,iel],"at pos.",xV[iconV[4,iel]], yV[iconV[4,iel]])
#    print ("node 5",iconV[5,iel],"at pos.",xV[iconV[5,iel]], yV[iconV[5,iel]])
#    print ("node 6",iconV[6,iel],"at pos.",xV[iconV[6,iel]], yV[iconV[6,iel]])

#print("iconV (min/max): %d %d" %(np.min(iconV[0,:]),np.max(iconV[0,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[1,:]),np.max(iconV[1,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[2,:]),np.max(iconV[2,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[3,:]),np.max(iconV[3,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[4,:]),np.max(iconV[4,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[5,:]),np.max(iconV[5,:])))

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("grid and connectivity V: %.3f s" % (timing.time() - start))

#################################################################
# build pressure nodes coordinates 
#################################################################

#do j=0,gridP%nelz    
#   do i=0,gridP%nelx    
#      counter=counter+1    
#      gridP%x(counter)=dble(i)*Lx/gridP%nelx
#      gridP%z(counter)=dble(j)*Lz/gridP%nelz

#################################################################
# build pressure nodes connectivity array 
#################################################################

#iconP=np.zeros((mP,nel),dtype=np.int32)
#counter=0
#do j=1,gridP%nelz
#   do i=1,gridP%nelx
#      ! lower left triangle
#      counter=counter+1
#      gridP%icon(1,counter)=i+(j-1)*(gridP%nelx+1)   ! 1 in Q1 
#      gridP%icon(2,counter)=i+1+(j-1)*(gridP%nelx+1) ! 2 in Q1 
#      gridP%icon(3,counter)=i+j*(gridP%nelx+1)       ! 4 in Q1 
##      ! upper right triangle
#      counter=counter+1
#      gridP%icon(1,counter)=i+1+j*(gridP%nelx+1)     ! 3 in Q1 
#      gridP%icon(2,counter)=i+j*(gridP%nelx+1)       ! 4 in Q1 
#      gridP%icon(3,counter)=i+1+(j-1)*(gridP%nelx+1) ! 2 in Q1 

#################################################################
#################################################################
start = timing.time()

xT=np.empty(NT,dtype=np.float64)  # x coordinates
yT=np.empty(NT,dtype=np.float64)  # y coordinates
T =np.zeros(NT,dtype=np.float64)
iconT=np.zeros((mT,nel),dtype=np.int32)

xT[0:NT]=xV[0:NT]
yT[0:NT]=yV[0:NT]

iconT[0:6,:]=iconV[0:6,:]

#np.savetxt('gridT.ascii',np.array([xT,yT]).T,header='# x,y')

print("grid and connectivity T: %.3f s" % (timing.time() - start))

#################################################################
# compute velocity on temperature grid 
#################################################################

u=np.empty(NT,dtype=np.float64)  # x coordinates
v=np.empty(NT,dtype=np.float64)  # y coordinates

for ip in range(0,NT):
    u[ip],v[ip]=compute_corner_flow_velocity(xT[ip],yT[ip],l1,l2,l3,angle,v0,Lx,Ly)

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NT):
    # top boundary - vack08
    if yT[i]/Ly>(1.-eps): #
       bc_fixT[i]=True ; bc_valT[i]=273
    # left boundary 
    if xT[i]/Lx<eps:
       if yT[i]>Ly-l1:
          #bc_fixT[i]=True ; bc_valT[i]=(Ly-yT[i])/l1*1200+273
          bc_fixT[i]=True ; bc_valT[i]=273+(1573-273)*erf((Ly-yT[i])/(2*np.sqrt(kappa*50e6*year)))
       elif u[ip]<0:
          bc_fixT[i]=True ; bc_valT[i]=1200.+273
    # right boundary 
    if xT[i]/Lx>1-eps:
       if yT[i]>Ly-l2:
          bc_fixT[i]=True ; bc_valT[i]=(Ly-yT[i])/l2*1300+273
       else:
          bc_fixT[i]=True ; bc_valT[i]=1300.+273 

#    if yT[i]/Ly<eps:
#       bc_fixT[i]=True ; bc_valT[i]=1200.+273

print("temperature b.c.: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area    = np.zeros(nel,dtype=np.float64) 
dNdx  = np.zeros(mT,dtype=np.float64)    # shape functions derivatives
dNdy  = np.zeros(mT,dtype=np.float64)    # shape functions derivatives
dNdr  = np.zeros(mT,dtype=np.float64)    # shape functions derivatives
dNds  = np.zeros(mT,dtype=np.float64)    # shape functions derivatives

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNdr[0:mT]=dNNTdr(rq,sq)
        dNds[0:mT]=dNNTds(rq,sq)
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mT):
            jcb[0,0] += dNdr[k]*xT[iconV[k,iel]]
            jcb[0,1] += dNdr[k]*yT[iconV[k,iel]]
            jcb[1,0] += dNds[k]*xT[iconV[k,iel]]
            jcb[1,1] += dNds[k]*yT[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> sum area %.6f %.6f" %(area.sum(),Lx*Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# build temperature matrix
#################################################################
start = timing.time()

A_mat = lil_matrix((NfemT,NfemT),dtype=np.float64)
#A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
B_mat = np.zeros((ndim,ndofT*mT),dtype=np.float64)     # gradient matrix B 
N_mat = np.zeros((mT,1),dtype=np.float64)         # shape functions

for iel in range (0,nel):

    b_el=np.zeros(mT*ndofT,dtype=np.float64)
    a_el=np.zeros((mT*ndofT,mT*ndofT),dtype=np.float64)
    Ka=np.zeros((mT,mT),dtype=np.float64)   # elemental advection matrix 
    Kd=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
    vel=np.zeros((1,ndim),dtype=np.float64)

    for kq in range(0,nqel):

        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        N_mat[0:mT,0]=NNT(rq,sq)
        dNdr[0:mT]=dNNTdr(rq,sq)
        dNds[0:mT]=dNNTds(rq,sq)

        # calculate jacobian matrix
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mT):
            jcb[0,0]+=dNdr[k]*xT[iconT[k,iel]]
            jcb[0,1]+=dNdr[k]*yT[iconT[k,iel]]
            jcb[1,0]+=dNds[k]*xT[iconT[k,iel]]
            jcb[1,1]+=dNds[k]*yT[iconT[k,iel]]

        # calculate the determinant of the jacobian
        jcob=np.linalg.det(jcb)

        # calculate inverse of the jacobian matrix
        jcbi=np.linalg.inv(jcb)

        # compute dNdx & dNdy
        vel[0,0]=0.
        vel[0,1]=0.
        xq=0.
        yq=0.
        for k in range(0,mT):
            vel[0,0]+=N_mat[k,0]*u[iconT[k,iel]]
            vel[0,1]+=N_mat[k,0]*v[iconT[k,iel]]
            xq+=N_mat[k,0]*xT[iconT[k,iel]]
            yq+=N_mat[k,0]*yT[iconT[k,iel]]
            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
            B_mat[0,k]=dNdx[k]
            B_mat[1,k]=dNdy[k]

        # compute diffusion matrix
        Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

        # compute advection matrix
        Ka+=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

    # end for kq

    a_el=Ka+Kd

    # apply boundary conditions

    for k1 in range(0,mT):
        m1=iconT[k1,iel]
        if bc_fixT[m1]:
           Aref=a_el[k1,k1]
           for k2 in range(0,mT):
               m2=iconT[k2,iel]
               b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
               a_el[k1,k2]=0
               a_el[k2,k1]=0
           a_el[k1,k1]=Aref
           b_el[k1]=Aref*bc_valT[m1]

    # assemble matrix A_mat and right hand side rhs
    for k1 in range(0,mT):
        m1=iconT[k1,iel]
        for k2 in range(0,mT):
            m2=iconT[k2,iel]
            A_mat[m1,m2]+=a_el[k1,k2]
        rhs[m1]+=b_el[k1]

# end for iel

print("build FEM matrix T: %.3f s" % (timing.time() - start))

#################################################################
# solve system
#################################################################
start = timing.time()

T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

print("solve T: %.3f s" % (timing.time() - start))

#################################################################
# post-processing 
#################################################################

#measuring T_{11,11}
for i in range(0,NT):
    if abs(xT[i]-60e3)<1 and abs(yT[i]-540e3)<1:
       print ('result1:',xT[i],yT[i],T[i]-273)

diagfile=open('tempdiag.ascii',"w")
for i in range(0,NT):
    if abs(xT[i] - (600e3-yT[i]) ) <1:
       diagfile.write("%10e %10e %10e \n " %(xT[i],yT[i],T[i]-273))
diagfile.close()

# compute average temperature

Tavrg=0.
for iel in range(0,nel):
    for kq in range(0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        N_mat[0:mT,0]=NNT(rq,sq)
        dNdr[0:mT]=dNNTdr(rq,sq)
        dNds[0:mT]=dNNTds(rq,sq)
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mT):
            jcb[0,0]+=dNdr[k]*xT[iconT[k,iel]]
            jcb[0,1]+=dNdr[k]*yT[iconT[k,iel]]
            jcb[1,0]+=dNds[k]*xT[iconT[k,iel]]
            jcb[1,1]+=dNds[k]*yT[iconT[k,iel]]
        jcob=np.linalg.det(jcb)
        Tq=0.
        for k in range(0,mT):
            Tq+=N_mat[k,0]*T[iconT[k,iel]]
        Tavrg+=Tq*weightq*jcob

Tavrg/=(Lx*Ly)

print ('Tavrg=',Tavrg)

#################################################################

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NT,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %(xT[i],yT[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    #vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (area[iel]))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (rho[iel]))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (eta[iel]))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (exx[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (eyy[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (exy[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %(u[i]/(cm/year),v[i]/(cm/year),0.))
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
    #for i in range(0,nnp):
    #    vtufile.write("%10e %10e %10e \n" %(gx(x[i],y[i],grav),gy(x[i],y[i],grav),0.))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    #for i in range(0,nnp):
    #    vtufile.write("%10e \n" %q[i])
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e \n" %(T[i]-273))
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='theta (deg)' Format='ascii'> \n")
    #for i in range(0,nnp):
    #    vtufile.write("%10e \n" %(theta[i]/np.pi*180.))
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*6))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()



print("-----------------------------")
print("------------the end----------")
print("-----------------------------")




