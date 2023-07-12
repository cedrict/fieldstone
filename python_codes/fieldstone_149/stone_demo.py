import numpy as np
import time as time
from tools import *

#################################################################
print("-----------------------------")
print("-------- stone 149(1)--------")
print("-----------------------------")

Lx=1
Ly=1

nelx=20
nely=nelx
nel=nelx*nely

nnx=nelx+1
nny=nely+1
nnp=nnx*nny

m=4

print('nelx=',nelx)
print('nely=',nely)
print('nel=',nel)
print('nnx=',nnx)
print('nny=',nny)
print('nnp=',nnp)

#################################################################
# grid point setup
#################################################################
start = time.time()

block1_x = np.empty(nnp,dtype=np.float64)  # x coordinates
block1_y = np.empty(nnp,dtype=np.float64)  # y coordinates
block1_hull=np.zeros(nnp,dtype=bool)

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        block1_x[counter]=i*Lx/float(nelx)
        block1_y[counter]=j*Ly/float(nely)
        if i==0 or i==nnx-1: block1_hull[counter]=True
        if j==0 or j==nny-1: block1_hull[counter]=True
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# build connectivity array
#################################################################
start = time.time()

block1_icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        block1_icon[0, counter] = i + j * (nelx + 1)
        block1_icon[1, counter] = i + 1 + j * (nelx + 1)
        block1_icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        block1_icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
#################################################################

block2_x = np.empty(nnp,dtype=np.float64) 
block2_y = np.empty(nnp,dtype=np.float64) 
block2_icon =np.zeros((m, nel),dtype=np.int32)
block2_hull=np.zeros(nnp,dtype=bool)

block2_x[:]=block1_x[:] 
block2_y[:]=block1_y[:] 
block2_icon[:,:]=block1_icon[:,:]
block2_hull[:]=block1_hull[:]

block2_x+=Lx

#np.savetxt('block1.ascii',np.array([block1_x,block1_y,block1_hull]).T)
#np.savetxt('block2.ascii',np.array([block2_x,block2_y,block2_hull]).T)

export_to_vtu('block1.vtu',block1_x,block1_y,block1_icon,block1_hull)
export_to_vtu('block2.vtu',block2_x,block2_y,block2_icon,block2_hull)

print('produced block1.vtu')
print('produced block2.vtu')

#################################################################

x,y,icon,hull=merge_two_blocks(block1_x,block1_y,block1_icon,block1_hull,\
                          block2_x,block2_y,block2_icon,block2_hull)


export_to_vtu('merged.vtu',x,y,icon,hull)

print('produced merged.vtu')
print("-----------------------------")

