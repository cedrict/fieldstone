import numpy as np
import time as time
from tools import *

#######################################################################
# this function receives the coordinates of the corners of a block
# as well as its desired resolution and returns the position of nodes
#######################################################################

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

#######################################################################
print("-----------------------------")
print("-------- stone 149(2)--------")
print("-----------------------------")

NV=14

x = np.empty(NV,dtype=np.float64) 
y = np.empty(NV,dtype=np.float64) 

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

#np.savetxt('points.ascii',np.array([x,y]).T)

m=4
nel=8

icon =np.zeros((m, nel),dtype=np.int32)

icon[0:m,0]=[0,1,6,5]
icon[0:m,1]=[1,2,7,6]
icon[0:m,2]=[2,3,8,7]
icon[0:m,3]=[3,4,9,8]
icon[0:m,4]=[5,6,10,12]
icon[0:m,5]=[6,7,8,10]
icon[0:m,6]=[8,9,11,10]
icon[0:m,7]=[10,11,13,12]

hull=np.zeros(14,dtype=bool)

export_to_vtu('initial.vtu',x,y,icon,hull)

#######################################################################
# assigning level (resolution) of each block
#######################################################################
level=20

nelx=level
nely=nelx
nel=nelx*nely

nnx=level+1
nny=nnx
NV=nnx*nny

m=4

#################################################################
# build generic connectivity array for a block
#################################################################
start = time.time()

block_icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        block_icon[0, counter] = i + j * (nelx + 1)
        block_icon[1, counter] = i + 1 + j * (nelx + 1)
        block_icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        block_icon[3, counter] = i + (j + 1) * (nelx + 1)
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

#################################################################
# assemble blocks into single mesh
#################################################################

print('-----merging 1+2------------------------------')
x12,y12,icon12,hull12=merge_two_blocks(block1_x,block1_y,block1_icon,block1_hull,\
                                       block2_x,block2_y,block2_icon,block2_hull)

export_to_vtu('blocks_1-2.vtu',x12,y12,icon12,hull12)
print('produced blocks_1-2.vtu')

print('-----merging 1+2+3----------------------------')
x13,y13,icon13,hull13=merge_two_blocks(x12,y12,icon12,hull12,\
                                       block3_x,block3_y,block3_icon,block3_hull)

export_to_vtu('blocks_1-3.vtu',x13,y13,icon13,hull13)
print('produced blocks_1-3.vtu')

print('-----merging 1+2+3+4--------------------------')

x14,y14,icon14,hull14=merge_two_blocks(x13,y13,icon13,hull13,\
                                       block4_x,block4_y,block4_icon,block4_hull)

export_to_vtu('blocks_1-4.vtu',x14,y14,icon14,hull14)
print('produced blocks_1-4.vtu')

print('-----merging 1+2+3+4+5------------------------')

x15,y15,icon15,hull15=merge_two_blocks(x14,y14,icon14,hull14,\
                                       block5_x,block5_y,block5_icon,block5_hull)

export_to_vtu('blocks_1-5.vtu',x15,y15,icon15,hull15)
print('produced blocks_1-5.vtu')

print('-----merging 1+2+3+4+5+6----------------------')

x16,y16,icon16,hull16=merge_two_blocks(x15,y15,icon15,hull15,\
                                       block6_x,block6_y,block6_icon,block6_hull)

export_to_vtu('blocks_1-6.vtu',x16,y16,icon16,hull16)
print('produced blocks_1-6.vtu')

print('-----merging 1+2+3+4+5+6+7--------------------')

x17,y17,icon17,hull17=merge_two_blocks(x16,y16,icon16,hull16,\
                                       block7_x,block7_y,block7_icon,block7_hull)

export_to_vtu('blocks_1-7.vtu',x17,y17,icon17,hull17)
print('produced blocks_1-7.vtu')

print('-----merging 1+2+3+4+5+6+7+8------------------')

x18,y18,icon18,hull18=merge_two_blocks(x17,y17,icon17,hull17,\
                                       block8_x,block8_y,block8_icon,block8_hull)

export_to_vtu('blocks_1-8.vtu',x18,y18,icon18,hull18)
print('produced blocks_1-8.vtu')

print("-----------------------------")
