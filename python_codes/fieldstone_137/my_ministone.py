import numpy as np
import matplotlib.pyplot as plt
import random
import sys as sys

###############################################################################
###############################################################################

if int(len(sys.argv) == 2):
   test = int(sys.argv[1])
else:
   test=1

###############################################################################
###############################################################################

if test==1: # left of board
   ncolor=2
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[:,:]=2

#if test==2:
#   ncolor=2
#   ncell=4
#   njc=np.zeros((ncolor,ncell),dtype=np.int16)
#   njc[0,:]=[0,0,4,4]
#   njc[1,:]=[4,4,0,0]

#if test==3:
#   ncolor=2
#   ncell=4
#   njc=np.zeros((ncolor,ncell),dtype=np.int16)
#   njc[0,:]=[0,2,2,2]
#   njc[1,:]=[0,2,2,2]

if test==4: #right of board
   ncolor=3
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[3,1,0,2]
   njc[1,:]=[1,0,2,1]
   njc[2,:]=[0,2,1,1]

###########################################################

if test==5:
   ncolor=2
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[3,3,3,3]
   njc[1,:]=[3,3,3,3]
   
if test==6:
   ncolor=3
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[2,2,2,2]
   njc[1,:]=[2,2,2,2]
   njc[2,:]=[2,2,2,2]

###########################################################
   
if test==7:
   ncolor=1
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[12,12,6,6]

if test==8:
   ncolor=2
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[6,6,3,0]
   njc[1,:]=[6,6,3,6]
   
if test==9:
   ncolor=3
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[4,4,2,0]
   njc[1,:]=[4,4,2,3]
   njc[2,:]=[4,4,2,3]
 
###########################################################
# test with Erik's data against his program

if test==10:
   ncolor=2
   ncell=539
   njctemp=np.transpose(np.loadtxt('Njc_c2.txt',ndmin=2))
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   for icell in range(0,ncell):
       for icolor in range(0,ncolor):
          njc[icolor,icell]=int(njctemp[icolor,icell])

if test==11:
   ncolor=3
   ncell=539
   njctemp=np.transpose(np.loadtxt('Njc_c3.txt',ndmin=2))
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   for icell in range(0,ncell):
       for icolor in range(0,ncolor):
          njc[icolor,icell]=int(njctemp[icolor,icell])

###############################################################################

print('*******************************')
print('test=',test)
print('*******************************')
print('number of tracers:',njc.sum())
print('njc=\n',njc)

###############################################################################
# compute Pc
###############################################################################

Pc=np.zeros((ncolor,1),dtype=np.float64)

for icolor in range(0,ncolor):
    Pc[icolor,0]=np.sum(njc[icolor,:])/ncell

print('Pc=\n',Pc)

###############################################################################
# compute A
###############################################################################

A=np.zeros((ncolor,ncell),dtype=np.float64)

for icell in range(0,ncell):
    A[:,icell]=njc[:,icell]/Pc[:,0]

print('A=\n',A)

###############################################################################
# compute B
###############################################################################

B=np.zeros(ncell,dtype=np.float64)

for icell in range(0,ncell):
    B[icell]=np.sum(A[:,icell])

print('B=\n',B)

###############################################################################
# compute C
###############################################################################

C=np.sum(B)

print('C=',C)

###############################################################################
# compute Pj
###############################################################################

Pj=np.zeros(ncell,dtype=np.float64)

Pj[:]=B[:]/C

print('Pj=\n',Pj)

###############################################################################
# compute Pjc
###############################################################################

Pjc=np.zeros((ncolor,ncell),dtype=np.float64)

Pjc[:,:]=A[:,:]/C

print('Pjc=\n',Pjc)

###############################################################################
# compute Pcj
###############################################################################

Pcj=np.zeros((ncolor,ncell),dtype=np.float64)

for icolor in range(0,ncolor):
    Pcj[icolor,:]=A[icolor,:]/B[:]

print('Pcj=\n',Pcj)

###############################################################################
# compute Slocation
###############################################################################

Slocation=0
for icell in range(0,ncell):
    Slocation-=Pj[icell]*np.log(Pj[icell])

print('Slocation=',Slocation)

###############################################################################
# compute Sj 
###############################################################################

Sj=np.zeros(ncell,dtype=np.float64)

for icell in range(0,ncell):
    for icolor in range(0,ncolor):
        if Pcj[icolor,icell]>0:
           Sj[icell]-=Pcj[icolor,icell] * np.log(Pcj[icolor,icell])

print('Sj=',Sj)

###############################################################################
# compute Slocation_c 
###############################################################################

Slocation_c=0
for icell in range(0,ncell):
    Slocation_c+=Pj[icell]*Sj[icell]

print('Slocation_c=',Slocation_c,' (=-log(1/2)')

###############################################################################
# apply normalisation
###############################################################################

Slocation/=np.log(ncell)

print('after normalisation: Slocation=',Slocation)

Sj/=np.log(ncolor)

print('after normalisation: Sj=',Sj)

Slocation_c/=np.log(ncolor)

print('after normalisation: Slocation_c=',Slocation_c)

###############################################################################

if ncell==4:
   ncellx=2
   ncelly=2
   Lx=1
   Ly=1

if ncell==539:
   ncellx=49
   ncelly=11
   Lx=50
   Ly=10

nnx=ncellx+1
nny=ncelly+1   
npts=nnx*nny
hx=Lx/ncellx
hy=Ly/ncelly

x = np.empty(npts,dtype=np.float64)  # x coordinates
y = np.empty(npts,dtype=np.float64)  # y coordinates
counter = 0 
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1

nswarm=int(np.sum(njc))

print('nswarm=',nswarm)
swarm_x = np.zeros(nswarm,dtype=np.float64)  # x coordinates
swarm_y = np.zeros(nswarm,dtype=np.float64)  # y coordinates
swarm_c = np.zeros(nswarm,dtype=np.float64)  # color

icell=0
counter=0
for icelly in range(0,ncelly): 
    for icellx in range(0,ncellx): 
        for icolor in range(ncolor):
            for i in range(0,njc[icolor,icell]):
                swarm_x[counter]=icellx*hx+random.uniform(0,hx)
                swarm_y[counter]=icelly*hy+random.uniform(0,hy)
                swarm_c[counter]=icolor
                counter+=1
        icell+=1


fig = plt.figure()
plt.scatter(swarm_x, swarm_y, c=swarm_c, s=5)
ax = fig.gca()
plt.xlim([0,Lx])
plt.ylim([0,Ly])
#ax.set_xticks([0,0.5,1])
#ax.set_yticks([0,0.5,1])
plt.grid()
plt.show()

print('*******************************')
