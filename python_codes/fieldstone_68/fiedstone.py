import numpy as np
import time as timing
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

##########################################################
# defining velocity and pressure shape functions
##########################################################

def NNV(rq,sq):
    NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
    NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
    NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
    NV_3= 4.*rq*sq*(-2.+3.*rq+3.*sq)
    NV_4= 4.*(1.-rq-sq)*sq*(1.-3.*rq)
    NV_5= 4.*(1.-rq-sq)*rq*(1.-3.*sq)
    NV_6= 27*(1.-rq-sq)*rq*sq
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6

def dNNVdr(rq,sq):
    dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
    dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
    dNVdr_2= 3*sq-6*rq*sq-3*sq**2
    dNVdr_3= -8*sq+24*rq*sq+12*sq**2
    dNVdr_4= -16*sq+24*rq*sq+12*sq**2
    dNVdr_5= -8*rq+24*rq*sq+4-16*sq+12*sq**2
    dNVdr_6= -54*rq*sq+27*sq-27*sq**2
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6

def dNNVds(rq,sq):
    dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
    dNVds_1= rq*(3-3*rq-6*sq)
    dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
    dNVds_3= -8*rq+12*rq**2+24*rq*sq
    dNVds_4= 4-16*rq-8*sq+24*rq*sq+12*rq**2
    dNVds_5= -16*rq+24*rq*sq+12*rq**2
    dNVds_6= -54*rq*sq+27*rq-27*rq**2
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6

def NNP(rq,sq):
    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return NP_0,NP_1,NP_2

##########################################################
# Crouzeix-Raviart elements
# the internal numbering of the nodes is as follows:
#
#  P_2^+           P_-1
#
#  02          #  02
#  ||\\        #  ||\\
#  || \\       #  || \\
#  ||  \\      #  ||  \\
#  04   03     #  ||   \\
#  || 06 \\    #  ||    \\
#  ||     \\   #  ||     \\
#  00==05==01  #  00======01
#

rVnodes=[0,1,0,0.5,0.0,0.5,1./3.]
sVnodes=[0,0,1,0.5,0.5,0.0,1./3.]

# checking that all velocity shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i]))
#exit()

##########################################################
# useful constants
##########################################################

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

cm=0.01
year=365.25*3600.*24.

mV=7     # number of velocity nodes making up an element
mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
ndofT=1  # number of temperature degrees of freedom 

##########################################################
# input parameters

Lx=600e3
eta0=1e22
vel=3*cm/year

##########################################################
#find NV and nel as indicated in msh file
##########################################################

f = open('subduction_mesh.msh', 'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter==4:
       NV=int(columns[0])
       print ('NV=',NV)
    counter+=1
#end for

f = open('subduction_mesh.msh', 'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter==NV+7:
       nel=int(columns[0])
       print ('nel=',nel)
    counter+=1
#end for

##########################################################
# find exact number of real triangles
# i.e. removing the interface related lines which 
# fall under triangles in input file but which aren't.
# The lines in question count 7 numbers per line 
# instead of 11 for real triangles.
# Once this number is found it should be removed from 
# the read-in number of elements and added to NV (the 
# C-R element has a 7th node in the middle of each triangle).  
##########################################################

f = open('subduction_mesh.msh', 'r')
counter=0
counter2=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter>NV+7 and counter<NV+8+nel:
       l=len(columns)
       if l==8:
          counter2+=1
    counter+=1
#end for

print('nb of nodes on interface:',counter2)

nel-=counter2

NVold=NV

NV+=nel #adding 7th node in the middle of each element

print ('new nel=',nel)
print ('new NV=',NV)

##########################################################
# read in coordinates of nodes
##########################################################

xV=np.zeros(NV,dtype=np.float64)     # x coordinates
yV=np.zeros(NV,dtype=np.float64)     # y coordinates

f = open('subduction_mesh.msh', 'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter>4 and counter<NVold+5:
       xV[counter-5]=np.float64(columns[1])
       yV[counter-5]=np.float64(columns[2])
    counter+=1
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV]).T)

##########################################################
# read in connectivity array
##########################################################

iconV=np.zeros((mV,nel),dtype=np.int64)
mat=np.zeros(nel,dtype=np.int64)

f = open('subduction_mesh.msh', 'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter>NVold+7+counter2 and counter<NVold+8+nel+counter2:
       l=len(columns)
       iconV[0,counter-NVold-8-counter2]=np.int64(columns[l-6])-1
       iconV[1,counter-NVold-8-counter2]=np.int64(columns[l-5])-1
       iconV[2,counter-NVold-8-counter2]=np.int64(columns[l-4])-1
       iconV[5,counter-NVold-8-counter2]=np.int64(columns[l-3])-1
       iconV[3,counter-NVold-8-counter2]=np.int64(columns[l-2])-1
       iconV[4,counter-NVold-8-counter2]=np.int64(columns[l-1])-1
       mat[counter-NVold-8-counter2]=np.int64(columns[4])/1e6
    #end if
    counter+=1
#end for

##########################################################
# compute coordinate of middle node
# The 7th node of the C-R element is at the barycenter 
# of the element. This node needs to be added to the 
# connectivity array too.
##########################################################

for iel in range(0,nel):
    iconV[6,iel]=NVold+iel
    xV[iconV[6,iel]]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
    yV[iconV[6,iel]]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.
#end for

#np.savetxt('gridV2.ascii',np.array([xV,yV]).T)

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconV[0,iel],"at pos.",xV[iconV[0][iel]], yV[iconV[0][iel]])
#    print ("node 1",iconV[1,iel],"at pos.",xV[iconV[1][iel]], yV[iconV[1][iel]])
#    print ("node 2",iconV[2,iel],"at pos.",xV[iconV[2][iel]], yV[iconV[2][iel]])
#    print ("node 3",iconV[3,iel],"at pos.",xV[iconV[3][iel]], yV[iconV[3][iel]])
#    print ("node 4",iconV[4,iel],"at pos.",xV[iconV[4][iel]], yV[iconV[4][iel]])
#    print ("node 5",iconV[5,iel],"at pos.",xV[iconV[5][iel]], yV[iconV[5][iel]])
#    print ("node 6",iconV[6,iel],"at pos.",xV[iconV[6][iel]], yV[iconV[6][iel]])

##########################################################
##########################################################

NfemV=NV*ndofV     # number of velocity dofs
NfemP=nel*3*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP   # total number of dofs

print ('NfemV', NfemV)
print ('NfemP', NfemP)
print ('Nfem ', Nfem)

pressure_scaling=1e22/1000e3

##########################################################
# 6 point integration coeffs and weights 
##########################################################

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322/2.
nb6=0.223381589678011/2.

qcoords_r=[nb1,nb2,nb2,nb4,nb3,nb4]
qcoords_s=[nb2,nb1,nb2,nb3,nb4,nb4]
qweights=[nb5,nb5,nb5,nb6,nb6,nb6]

#################################################################
# build pressure grid (nodes and icon)
# because the pressure is dicontinuous (P_-1) then the total 
# number of pressure dofs is nel*3 (see above)
# We use the iconV connectivity to build the iconP array
#################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int64)
xP=np.empty(NfemP,dtype=np.float64)     # x coordinates
yP=np.empty(NfemP,dtype=np.float64)     # y coordinates

counter=0
for iel in range(0,nel):
    xP[counter]=xV[iconV[0,iel]]
    yP[counter]=yV[iconV[0,iel]]
    iconP[0,iel]=counter
    counter+=1
    xP[counter]=xV[iconV[1,iel]]
    yP[counter]=yV[iconV[1,iel]]
    iconP[1,iel]=counter
    counter+=1
    xP[counter]=xV[iconV[2,iel]]
    yP[counter]=yV[iconV[2,iel]]
    iconP[2,iel]=counter
    counter+=1

np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], yP[iconP[0][iel]])
#    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], yP[iconP[1][iel]])
#    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], yP[iconP[2][iel]])

print("setup: connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
#################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
    if area[iel]<0: 
       for k in range(0,mV):
           print (xV[iconV[k,iel]],yV[iconV[k,iel]])
   #    print(iel,iconV[:,iel])

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# read from msh file the identity of the nodes on the slab interface
#################################################################

interface=np.zeros(NV,dtype=np.bool)  

f = open('subduction_mesh.msh', 'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter>NVold+7 and counter<NVold+8+nel:
       l=len(columns)
       if l==8:
          #print ('opla',columns[5],columns[6],columns[7])
          interface[np.int64(columns[5])-1]=True
          interface[np.int64(columns[6])-1]=True
          interface[np.int64(columns[7])-1]=True
    counter+=1

#interface[0:counter2]=True
#for iel in range(0,nel):
#    inode0=iconV[0,iel]
#    inode1=iconV[1,iel]
#    inode2=iconV[2,iel]
#    if inode0<=counter2 and inode1<=counter2:
#       interface[iconV[5,iel]]=True    
#    if inode0<=counter2 and inode2<=counter2:
#       interface[iconV[4,iel]]=True    
#    if inode1<=counter2 and inode2<=counter2:
#       interface[iconV[3,iel]]=True    

#################################################################
# compute normal vector along the interface
# Loop over all triangles, find edges on the interface, 
# compute their normal, add it to all three nodes on the edge. 
# In the end renormalise the normal vectors. 
#################################################################

nx=np.zeros(NV,dtype=np.float64) 
ny=np.zeros(NV,dtype=np.float64) 

for iel in range(0,nel):
    inode0=iconV[0,iel]
    inode1=iconV[1,iel]
    inode2=iconV[2,iel]
    x0=xV[inode0]
    x1=xV[inode1]
    x2=xV[inode2]
    y0=yV[inode0]
    y1=yV[inode1]
    y2=yV[inode2]
    if inode0<=counter2 and inode1<=counter2:
       vx=x1-x0
       vy=y1-y0
       vnorm=np.sqrt(vx**2+vy**2)
       ax=-vy/vnorm
       ay= vx/vnorm
       nx[inode0]+=ax
       ny[inode0]+=ay
       nx[iconV[5,iel]]=ax
       ny[iconV[5,iel]]=ay
       nx[inode1]+=ax
       ny[inode1]+=ay
    if inode0<=counter2 and inode2<=counter2:
       vx=x2-x0
       vy=y2-y0
       vnorm=np.sqrt(vx**2+vy**2)
       ax=-vy/vnorm
       ay= vx/vnorm
       nx[inode0]+=ax
       ny[inode0]+=ay
       nx[iconV[4,iel]]=ax
       ny[iconV[4,iel]]=ay
       nx[inode2]+=ax
       ny[inode2]+=ay
    if inode1<=counter2 and inode2<=counter2:
       vx=x2-x1
       vy=y2-y1
       vnorm=np.sqrt(vx**2+vy**2)
       ax=-vy/vnorm
       ay= vx/vnorm
       nx[inode1]+=ax
       ny[inode1]+=ay
       nx[iconV[3,iel]]=ax
       ny[iconV[3,iel]]=ay
       nx[inode2]+=ax
       ny[inode2]+=ay
    
for i in range(0,NV):
    norm=np.sqrt(nx[i]**2+ny[i]**2)
    if norm>1e-6:
       nx[i]/=norm
       ny[i]/=norm

#################################################################
# use normal vector to impose velocity bcs: velocity on the interface 
# should be perpendicular to normal.
#################################################################

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for iel in range(0,nel):
    if mat[iel]==2: # over-riding plate
       skip_element=False
       if interface[iconV[0,iel]]:
          skip_element=True
       if interface[iconV[1,iel]]:
          skip_element=True
       if interface[iconV[2,iel]]:
          skip_element=True
       if interface[iconV[3,iel]]:
          skip_element=True
       if interface[iconV[4,iel]]:
          skip_element=True
       if interface[iconV[5,iel]]:
          skip_element=True
       if interface[iconV[6,iel]]:
          skip_element=True
       if not skip_element:
          for k in range(0,mV):
              inode=iconV[k,iel]
              bc_fix[inode*ndofV  ] = True ; bc_val[inode*ndofV  ] = 0
              #bc_fix[inode*ndofV+1] = True ; bc_val[inode*ndofV+1] = 0


for i in range(0,NV):
    if yV[i]>-1:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    # right lithosphere
    #if abs(xV[i]-Lx)/Lx<1e-6:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

    if interface[i]:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ny[i]*vel 
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[i]*vel

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = timing.time()

c_mat  = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
A_sparse=lil_matrix((Nfem,Nfem),dtype=np.float64)
rhs    = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
f_rhs  = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
h_rhs  = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
b_mat  = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat  = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix to build G_el 
NNNV   = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP   = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx= np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy= np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr= np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds= np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u      = np.zeros(NV,dtype=np.float64)           # x-component velocity
v      = np.zeros(NV,dtype=np.float64)           # y-component velocity

for iel in range(0,nel):

    if iel%200==0:
       print('iel=',iel)

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    for kq in range (0,nqel):

        # position & weight of quad. point
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        NNNP[0:4]=NNP(rq,sq)

        # calculate jacobian matrix
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        jcbi = np.linalg.inv(jcb)

        # compute dNdx & dNdy
        xq=0.0
        yq=0.0
        for k in range(0,mV):
            xq+=NNNV[k]*xV[iconV[k,iel]]
            yq+=NNNV[k]*yV[iconV[k,iel]]
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

        # construct 3x8 b_mat matrix
        for i in range(0,mV):
            b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                     [0.        ,dNNNVdy[i]],
                                     [dNNNVdy[i],dNNNVdx[i]]]

        # compute elemental a_mat matrix
        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta0*weightq*jcob

        # compute elemental rhs vector
        #for i in range(0,mV):
        #    f_el[ndofV*i  ]+=NV[i]*jcob*weightq*gx(xq,yq,grav)*rhoq
        #    f_el[ndofV*i+1]+=NV[i]*jcob*weightq*gy(xq,yq,grav)*rhoq

        for i in range(0,mP):
            N_mat[0,i]=NNNP[i]
            N_mat[1,i]=NNNP[i]
            N_mat[2,i]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

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

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    #K_mat[m1,m2]+=K_el[ikk,jkk]
                    A_sparse[m1,m2] += K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                #G_mat[m1,m2]+=G_el[ikk,jkk]
                A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*pressure_scaling
                A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*pressure_scaling
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]*pressure_scaling

print("build FE matrix: %.3f s" % (timing.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = timing.time()

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

#for i in range(0,Nfem):
#    A_sparse[Nfem-1,i]=0
#    A_sparse[i,Nfem-1]=0
#A_sparse[Nfem-1,Nfem-1]=1e20
#rhs[Nfem-1]=0

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# solve system
######################################################################
start = timing.time()

sol = np.zeros(Nfem,dtype=np.float64) 

sparse_matrix=A_sparse.tocsr()
sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*pressure_scaling

print("     -> u (m,M) %.6e %.6e (cm/yr)" %(np.min(u)/cm*year,np.max(u)/cm*year))
print("     -> v (m,M) %.6e %.6e (cm/yr)" %(np.min(v)/cm*year,np.max(v)/cm*year))
print("     -> p (m,M) %.6e %.6e (MPa)" %(np.min(p)/1e6,np.max(p)/1e6))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (timing.time() - start))


























#################################################################

if True:
    vtufile=open('solution.vtu',"w")
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
    #--
    #vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (area[iel]))
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (mat[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(nx[i],ny[i],0))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='interface' Format='ascii'> \n")
    for i in range(0,NV):
        #if i<=counter2: 
        if interface[i]: 
           vtufile.write("%10e \n" %1.)
        else:
           vtufile.write("%10e \n" %0.)
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u (bool)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2+1]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u (value)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2]:
           val=bc_val[i*2]/cm*year
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='fix_v (value)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2+1]:
           val=bc_val[i*2+1]/cm*year
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")

    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        #vtufile.write("%d %d %d\n" %(iconV[0,iel]-1,iconV[1,iel]-1,iconV[2,iel]-1))
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                              iconV[5,iel],iconV[3,iel],iconV[4,iel])    )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*6))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22) #5
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()



    vtufile=open('gridP.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NfemP,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NfemP):
        vtufile.write("%10e %10e %10e \n" %(xP[i],yP[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d\n" %(iconP[0,iel],iconP[1,iel],iconP[2,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*3))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

