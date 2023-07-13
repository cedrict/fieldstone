import numpy as np
import time as time
from tools import *
import velocity
from scipy.special import erf
import time as timing
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

###############################################################################
# Q1 basis functions in 2D
###############################################################################

def NNT(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

def dNNTdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)

def dNNTds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)

###############################################################################
# this function receives the coordinates of the corners of a block
# as well as its desired resolution and returns the position of nodes.
# nodes on the boundary are flagged.
###############################################################################

def laypts4(x1,y1,x2,y2,x3,y3,x4,y4,x,y,hull,level):
    counter=0
    for j in range(0,level+1):
        for i in range(0,level+1):
            #equidistant
            r=-1.+2./level*i
            s=-1.+2./level*j
            N1=0.25*(1.-r)*(1.-s)
            N2=0.25*(1.+r)*(1.-s)
            N3=0.25*(1.+r)*(1.+s)
            N4=0.25*(1.-r)*(1.+s)
            x[counter]=x1*N1+x2*N2+x3*N3+x4*N4
            y[counter]=y1*N1+y2*N2+y3*N3+y4*N4
            if i==0 or i==level: hull[counter]=True
            if j==0 or j==level: hull[counter]=True
            counter+=1
        #end for
    #end for

###############################################################################
# 12---------------13
# | \               |
# |  10------------11 
# |   | \           | 
# |   |  \          | 
# |   |   \         | 
# |   |    \        | 
# 5---6     8-------9 
# |   | \-  / \     | 
# |   |   7-   \    | 
# |   |     \   \   | 
# 0---1-----2----3--4
###############################################################################
print("-----------------------------")
print("--------- stone 149 ---------")
print("-----------------------------")

m=4   # number of nodes per element
nel=8 # number of elements
NV=14 # number of nodes

x=np.empty(NV,dtype=np.float64) 
y=np.empty(NV,dtype=np.float64) 
icon =np.zeros((m,nel),dtype=np.int32)
hull=np.zeros(14,dtype=bool)

x[ 0]=0   ; y[ 0]=0
x[ 1]=50  ; y[ 1]=0
x[ 2]=330 ; y[ 2]=0
x[ 3]=600 ; y[ 3]=0
x[ 4]=660 ; y[ 4]=0
x[ 5]=0   ; y[ 5]=300
x[ 6]=50  ; y[ 6]=300
x[ 8]=330 ; y[ 8]=270
x[ 9]=660 ; y[ 9]=270 
x[10]=50  ; y[10]=550
x[11]=660 ; y[11]=550
x[12]=0   ; y[12]=600
x[13]=660 ; y[13]=600

x[ 7]=(x[1]+x[3]+x[10])/3    
y[ 7]=(y[1]+y[3]+y[10])/3    

icon[0:m,0]=[0,1,6,5]
icon[0:m,1]=[1,2,7,6]
icon[0:m,2]=[2,3,8,7]
icon[0:m,3]=[3,4,9,8]
icon[0:m,4]=[5,6,10,12]
icon[0:m,5]=[6,7,8,10]
icon[0:m,6]=[8,9,11,10]
icon[0:m,7]=[10,11,13,12]

export_to_vtu('initial.vtu',x,y,icon,hull)

###############################################################################
# assigning level (resolution) of each block
###############################################################################
level=32

nelx=level
nely=nelx
nel=nelx*nely

nnx=level+1
nny=nnx
NV=nnx*nny

###############################################################################
# build generic connectivity array for a block
###############################################################################
start = time.time()

block_icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        block_icon[0,counter]=i+j*(nelx+1)
        block_icon[1,counter]=i+1+j*(nelx+1)
        block_icon[2,counter]=i+1+(j+1)*(nelx+1)
        block_icon[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# build each individual block
#################################################################

block1_x=np.empty(NV,dtype=np.float64) 
block1_y=np.empty(NV,dtype=np.float64) 
block1_icon =np.zeros((m, nel),dtype=np.int32)
block1_hull=np.zeros(NV,dtype=bool)
block1_icon[:,:]=block_icon[:,:]
laypts4(x[0],y[0],x[1],y[1],x[6],y[6],x[5],y[5],block1_x,block1_y,block1_hull,level)
export_to_vtu('block1.vtu',block1_x,block1_y,block1_icon,block1_hull)

block2_x=np.empty(NV,dtype=np.float64) 
block2_y=np.empty(NV,dtype=np.float64) 
block2_icon =np.zeros((m, nel),dtype=np.int32)
block2_hull=np.zeros(NV,dtype=bool)
block2_icon[:,:]=block_icon[:,:]
laypts4(x[1],y[1],x[2],y[2],x[7],y[7],x[6],y[6],block2_x,block2_y,block2_hull,level)
export_to_vtu('block2.vtu',block2_x,block2_y,block2_icon,block2_hull)

block3_x=np.empty(NV,dtype=np.float64) 
block3_y=np.empty(NV,dtype=np.float64) 
block3_icon =np.zeros((m, nel),dtype=np.int32)
block3_hull=np.zeros(NV,dtype=bool)
block3_icon[:,:]=block_icon[:,:]
laypts4(x[2],y[2],x[3],y[3],x[8],y[8],x[7],y[7],block3_x,block3_y,block3_hull,level)
export_to_vtu('block3.vtu',block3_x,block3_y,block3_icon,block3_hull)

block4_x=np.empty(NV,dtype=np.float64) 
block4_y=np.empty(NV,dtype=np.float64) 
block4_icon =np.zeros((m, nel),dtype=np.int32)
block4_hull=np.zeros(NV,dtype=bool)
block4_icon[:,:]=block_icon[:,:]
laypts4(x[3],y[3],x[4],y[4],x[9],y[9],x[8],y[8],block4_x,block4_y,block4_hull,level)
export_to_vtu('block4.vtu',block4_x,block4_y,block4_icon,block4_hull)

block5_x=np.empty(NV,dtype=np.float64) 
block5_y=np.empty(NV,dtype=np.float64) 
block5_icon =np.zeros((m, nel),dtype=np.int32)
block5_hull=np.zeros(NV,dtype=bool)
block5_icon[:,:]=block_icon[:,:]
laypts4(x[5],y[5],x[6],y[6],x[10],y[10],x[12],y[12],block5_x,block5_y,block5_hull,level)
export_to_vtu('block5.vtu',block5_x,block5_y,block5_icon,block5_hull)

block6_x=np.empty(NV,dtype=np.float64) 
block6_y=np.empty(NV,dtype=np.float64) 
block6_icon =np.zeros((m, nel),dtype=np.int32)
block6_hull=np.zeros(NV,dtype=bool)
block6_icon[:,:]=block_icon[:,:]
laypts4(x[6],y[6],x[7],y[7],x[8],y[8],x[10],y[10],block6_x,block6_y,block6_hull,level)
export_to_vtu('block6.vtu',block6_x,block6_y,block6_icon,block6_hull)

block7_x=np.empty(NV,dtype=np.float64) 
block7_y=np.empty(NV,dtype=np.float64) 
block7_icon =np.zeros((m, nel),dtype=np.int32)
block7_hull=np.zeros(NV,dtype=bool)
block7_icon[:,:]=block_icon[:,:]
laypts4(x[8],y[8],x[9],y[9],x[11],y[11],x[10],y[10],block7_x,block7_y,block7_hull,level)
export_to_vtu('block7.vtu',block7_x,block7_y,block7_icon,block7_hull)

block8_x=np.empty(NV,dtype=np.float64) 
block8_y=np.empty(NV,dtype=np.float64) 
block8_icon =np.zeros((m, nel),dtype=np.int32)
block8_hull=np.zeros(NV,dtype=bool)
block8_icon[:,:]=block_icon[:,:]
laypts4(x[10],y[10],x[11],y[11],x[13],y[13],x[12],y[12],block8_x,block8_y,block8_hull,level)
export_to_vtu('block8.vtu',block8_x,block8_y,block8_icon,block8_hull)

###############################################################################
# assemble blocks into single mesh
###############################################################################

print('-----merging 1+2----------------')
x12,y12,icon12,hull12=merge_two_blocks(block1_x,block1_y,block1_icon,block1_hull,\
                                       block2_x,block2_y,block2_icon,block2_hull)

export_to_vtu('blocks_1-2.vtu',x12,y12,icon12,hull12)
print('produced blocks_1-2.vtu')

print('-----merging 1+2+3--------------')
x13,y13,icon13,hull13=merge_two_blocks(x12,y12,icon12,hull12,\
                                       block3_x,block3_y,block3_icon,block3_hull)

export_to_vtu('blocks_1-3.vtu',x13,y13,icon13,hull13)
print('produced blocks_1-3.vtu')

print('-----merging 1+2+3+4------------')

x14,y14,icon14,hull14=merge_two_blocks(x13,y13,icon13,hull13,\
                                       block4_x,block4_y,block4_icon,block4_hull)

export_to_vtu('blocks_1-4.vtu',x14,y14,icon14,hull14)
print('produced blocks_1-4.vtu')

print('-----merging 1+2+3+4+5----------')

x15,y15,icon15,hull15=merge_two_blocks(x14,y14,icon14,hull14,\
                                       block5_x,block5_y,block5_icon,block5_hull)

export_to_vtu('blocks_1-5.vtu',x15,y15,icon15,hull15)
print('produced blocks_1-5.vtu')

print('-----merging 1+2+3+4+5+6--------')

x16,y16,icon16,hull16=merge_two_blocks(x15,y15,icon15,hull15,\
                                       block6_x,block6_y,block6_icon,block6_hull)

export_to_vtu('blocks_1-6.vtu',x16,y16,icon16,hull16)
print('produced blocks_1-6.vtu')

print('-----merging 1+2+3+4+5+6+7------')

x17,y17,icon17,hull17=merge_two_blocks(x16,y16,icon16,hull16,\
                                       block7_x,block7_y,block7_icon,block7_hull)

export_to_vtu('blocks_1-7.vtu',x17,y17,icon17,hull17)
print('produced blocks_1-7.vtu')

print('-----merging 1+2+3+4+5+6+7+8----')

x18,y18,icon18,hull18=merge_two_blocks(x17,y17,icon17,hull17,\
                                       block8_x,block8_y,block8_icon,block8_hull)

export_to_vtu('blocks_1-8.vtu',x18,y18,icon18,hull18)
print('produced blocks_1-8.vtu')

print("-----------------------------")
print(" meshing completed           ")
print("-----------------------------")

#################################################################
mT=4

NT=np.size(x18)
m,nel=np.shape(icon18)

xT=np.empty(NT,dtype=np.float64)  # x coordinates
yT=np.empty(NT,dtype=np.float64)  # y coordinates
iconT=np.zeros((mT,nel),dtype=np.int32)

xT[:]=x18*1e3
yT[:]=y18*1e3
iconT[:,:]=icon18[:,:]

NfemT=NT

print('nel=',nel)
print('NT=',NT)
print('NfemT=',NfemT)

#################################################################

Kelvin=273.15
ndim=2
sqrt3=np.sqrt(3.)
cm=0.01 
year=365.25*24*3600

Lx=660e3
Ly=600e3
hcond=3    # heat conductivity
hcapa=1250 # heat capacity
rho=3300   # density
l1=1000.e3
l2=50.e3
l3=0.e3
vel=5*cm/year
angle=45./180.*np.pi  

#################################################################
# assign velocity to nodes, corner flow
#################################################################

u=np.empty(NT,dtype=np.float64) 
v=np.empty(NT,dtype=np.float64) 

for i in range(0,NT):
    u[i],v[i]=velocity.compute_corner_flow_velocity(xT[i],yT[i],l1,l2,l3,angle,vel,Lx,Ly)

#################################################################
# this is a steady state code, but I leave the loop so that 
# one could re-introduce time stepping.
#################################################################

T=np.empty(NT,dtype=np.float64) 

for iter in range(0,1):

    ######################################################################
    # temperature boundary conditions
    # which depend on velocity 
    ######################################################################

    bc_fixT=np.zeros(NfemT,dtype=bool)  # boundary condition, yes/no
    bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

    kappa=3./3300./1250. #hcond/rho/hcapa

    for i in range(0,NT):
        # top boundary - vack08
        if abs(yT[i]-Ly)<1: #
           bc_fixT[i]=True ; bc_valT[i]=Kelvin
        # left boundary 
        if xT[i]<1:
           bc_fixT[i]=True ; bc_valT[i]=Kelvin+(1573-Kelvin)*erf(((Ly-yT[i]))/(2*np.sqrt(kappa*50e6*year)))
        # right boundary 
        if abs(xT[i]-Lx)<1:
           if yT[i]>=Ly-l2:
              bc_fixT[i]=True ; bc_valT[i]=((Ly-yT[i]))/l2*1300+Kelvin
           elif u[i]<0:
              bc_fixT[i]=True ; bc_valT[i]=1300.+Kelvin


    ######################################################################
    # build FE matrix for temperature 
    ######################################################################
    start = timing.time()

    A_mat = lil_matrix((NfemT,NfemT),dtype=np.float64) # FE matrix
    rhs   = np.zeros(NfemT,dtype=np.float64)           # FE rhs 
    B_mat = np.zeros((ndim,mT),dtype=np.float64)       # gradient matrix B 
    N_mat = np.zeros((mT,1),dtype=np.float64)          # shape functions vector
    dNNNTdr = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTds = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdx = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdy = np.zeros(mT,dtype=np.float64)            # shape functions derivatives

    for iel in range (0,nel):

        b_el=np.zeros(mT,dtype=np.float64)      # elemental rhs
        a_el=np.zeros((mT,mT),dtype=np.float64) # elemental matrix
        Ka=np.zeros((mT,mT),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
        velq=np.zeros((1,ndim),dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [-1, 1]:
            for jq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                N_mat[0:mT,0]=NNT(rq,sq)
                dNNNTdr[0:mT]=dNNTdr(rq,sq)
                dNNNTds[0:mT]=dNNTds(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mT):
                    jcb[0,0]+=dNNNTdr[k]*xT[iconT[k,iel]]
                    jcb[0,1]+=dNNNTdr[k]*yT[iconT[k,iel]]
                    jcb[1,0]+=dNNNTds[k]*xT[iconT[k,iel]]
                    jcb[1,1]+=dNNNTds[k]*yT[iconT[k,iel]]
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                velq[0,0]=0.
                velq[0,1]=0.
                xq=0.
                yq=0.
                Tq=0.
                for k in range(0,mT):
                    velq[0,0]+=N_mat[k,0]*u[iconT[k,iel]]
                    velq[0,1]+=N_mat[k,0]*v[iconT[k,iel]]
                    xq+=N_mat[k,0]*xT[iconT[k,iel]]
                    yq+=N_mat[k,0]*yT[iconT[k,iel]]
                    Tq+=N_mat[k,0]*T[iconT[k,iel]]
                    dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                    dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                    B_mat[0,k]=dNNNTdx[k]
                    B_mat[1,k]=dNNNTdy[k]

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka+=N_mat.dot(velq.dot(B_mat))*rho*hcapa*weightq*jcob

	    # end for jq
        # end for iq

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
               # end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            # end if
        # end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mT):
            m1=iconT[k1,iel]
            for k2 in range(0,mT):
                m2=iconT[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            # end for
            rhs[m1]+=b_el[k1]
        # end for

    # end for iel

    print("build FEM matrix T: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system 
    ######################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    np.savetxt('T.ascii',np.array([xT,yT,T]).T,header='# x,y')

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T: %.3f s" % (timing.time() - start))

    ######################################################################
    diag=np.zeros(NT,dtype=np.float64) # size too large
    dist=np.zeros(NT,dtype=np.float64)

    counter=0
    for i in range(0,NT):
        if abs(yT[i]-Ly+xT[i])/Lx<1e-4:
           diag[counter]=T[i]
           dist[counter]=np.sqrt( (xT[i]-0)**2+(yT[i]-Ly)**2  )
           counter+=1

    np.savetxt('diagT.ascii',np.array([dist[0:counter],diag[0:counter]]).T)

    ######################################################################

    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NT,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %(xT[i],yT[i],0))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--  
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3'  Name='velocity' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" % (u[i]/cm*year,v[i]/cm*year,0))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32'   Name='T' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e \n" % (T[i]-Kelvin))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d \n" %(iconT[0,iel],iconT[1,iel],iconT[2,iel],iconT[3,iel]))
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

print("-----------------------------")

###############################################################################
