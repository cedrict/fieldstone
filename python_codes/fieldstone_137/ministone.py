import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

###############################################################################
# choose testcase to run
# tests 5 & 6 are not shown in the Appendix but can be used to test 
# your own entropies in a 2x2 (test 5) or a 3x3 (test 6) grid
###############################################################################

test = 1

###############################################################################
# We fill the cells by hand through matrix njc, 
# where rows represent different compositions
###############################################################################

if test==1: # left of board
   ncolor=2
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]= [3,3,3,3]
   njc[1,:]= [3,3,3,3]

if test==2:
   ncolor=2
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[0,0,4,4]
   njc[1,:]=[4,4,0,0]

if test==3:
   ncolor=3
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[3,0,2,1]
   njc[1,:]=[1,2,1,0]
   njc[2,:]=[0,1,1,2]

if test==4: 
   ncolor=2
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]= [20,1,20,1]
   njc[1,:]= [20,1,0,4]

if test==5: 
   ncolor=2
   ncell=4
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[100,1,1,0]
   njc[1,:]=[100,0,1,1]

if test==6: 
   ncolor=3
   ncell=9
   njc=np.zeros((ncolor,ncell),dtype=np.int16)
   njc[0,:]=[4,2,1,1,4,6,1,2,0]
   njc[1,:]=[0,1,2,3,4,5,6,7,8] 
   njc[2,:]=[2,0,1,1,3,1,0,2,3]

print('*******************************')
print('test=',test)
print('*******************************')
print('number of tracers:',njc.sum())
print('njc=\n',njc)

###############################################################################
# compute Nc
###############################################################################

Nc=np.zeros((ncolor,1),dtype=np.float64)

for icolor in range(0,ncolor):
    Nc[icolor,0]=np.sum(njc[icolor,:])/ncell

print('Nc=\n',Nc)

###############################################################################
# compute A  - equation 1
# the numbering of equations corresponds to those in the Appendix
# of van der Wiel, Thieulot & van Hinsbergen (submitted).
###############################################################################

A=np.zeros((ncolor,ncell),dtype=np.float64)

for icell in range(0,ncell):
    A[:,icell]=njc[:,icell]/Nc[:,0]

print('A=\n',A)

###############################################################################
# compute B - denominator equation 2
###############################################################################

B=np.zeros(ncell,dtype=np.float64)

for icell in range(0,ncell):
    B[icell]=np.sum(A[:,icell])

print('B=\n',B)

###############################################################################
# compute C - denominator equation 3
###############################################################################

C=np.sum(B)

print('C=',C)

###############################################################################
# compute Pj - equation 3
###############################################################################

Pj=np.zeros(ncell,dtype=np.float64)

Pj[:]=B[:]/C

print('Pj=\n',Pj)

###############################################################################
# compute Pcj - not used here but can be used to verify total entropy 
# see for instance Camesasca et al., 2006
###############################################################################

Pcj=np.zeros((ncolor,ncell),dtype=np.float64)

Pcj[:,:]=A[:,:]/C

#print('Pcj=\n',Pcj)

###############################################################################
# compute Pjc - equation 2
###############################################################################

Pjc=np.zeros((ncolor,ncell),dtype=np.float64)

for icolor in range(0,ncolor):
    Pjc[icolor,:]=A[icolor,:]/B[:]

print('Pjc=\n',Pjc)

###############################################################################
# compute Spd - equation 4
###############################################################################

Spd=0
for icell in range(0,ncell):
    Spd-=Pj[icell]*np.log(Pj[icell])

print('Spd=',Spd)

###############################################################################
# compute Sj - equation 5
###############################################################################

Sj=np.zeros(ncell,dtype=np.float64)

for icell in range(0,ncell):
    for icolor in range(0,ncolor):
        if Pcj[icolor,icell]>0:
           Sj[icell]-=Pjc[icolor,icell] * np.log(Pjc[icolor,icell])

print('Sj=',Sj)

###############################################################################
# compute S - equation 6
###############################################################################

S_global=0
for icell in range(0,ncell):
    S_global+=Pj[icell]*Sj[icell]

print('S_global=',S_global,' (=-log(1/2)')

###############################################################################
# apply normalisation
###############################################################################

Spd/=np.log(ncell)

print('after normalisation: Spd=',Spd)

Sj/=np.log(ncolor)

print('after normalisation: Sj=',Sj)

S_global/=np.log(ncolor)

print('after normalisation: S_global=',S_global)

###############################################################################

if ncell==4:
   ncellx=2
   ncelly=2

elif ncell==9:
   ncellx=3
   ncelly=3 
   
nnx=ncellx+1
nny=ncelly+1   
npts=nnx*nny

Lx=1
Ly=1

hx=Lx/ncellx
hy=Ly/ncelly

###############################################################################
# We now create a grid for visualization purposes
# the grid counts ncell cells. 
# x,y are the coordinates of the npts nodes
# icon is the connectivity array (see stone 0).
# swarm_x,y are the coordinates of all points and swarm_c their color
###############################################################################
   
x = np.empty(npts,dtype=np.float64)  # x coordinates
y = np.empty(npts,dtype=np.float64)  # y coordinates
counter = 0 
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1

nswarm=np.sum(njc)

print('nswarm=',nswarm)

swarm_x = np.zeros(nswarm,dtype=np.float64)  # x coordinates
swarm_y = np.zeros(nswarm,dtype=np.float64)  # y coordinates
swarm_c = np.zeros(nswarm,dtype=np.float64)  # color

icell=0
counter=0
for icelly in range(0,ncelly): 
    for icellx in range(0,ncelly): 
        for icolor in range(ncolor):
            for i in range(0,njc[icolor,icell]):
                swarm_x[counter]=icellx*hx+random.uniform(0.05,hx-0.05)
                swarm_y[counter]=icelly*hy+random.uniform(0.05,hy-0.05)
                swarm_c[counter]=icolor
                counter+=1
            #end for
        #end for
        icell+=1
    #end for
#end for

icon =np.zeros((4,ncell),dtype=np.int32)
counter = 0
for j in range(0, ncelly):
    for i in range(0, ncellx):
        icon[0, counter] = i + j * (ncellx + 1)
        icon[1, counter] = i + 1 + j * (ncellx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (ncellx + 1)
        icon[3, counter] = i + (j + 1) * (ncellx + 1)
        counter += 1

###############################################################################
# Create plot figure (svg format)
###############################################################################

fig = plt.figure()
mpl.rcParams['font.size'] = 12
ax = fig.gca()
plt.xlim([0,1])
plt.ylim([0,1])
plt.axis('equal')
plt.axis('off')
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])

for jj in range(0,ncell):
    x_grid = ( x[icon[0,jj]], x[icon[1,jj]],x[icon[2,jj]], x[icon[3,jj]]  )
    y_grid = ( y[icon[0,jj]], y[icon[1,jj]],y[icon[2,jj]], y[icon[3,jj]]  )
    
    plt.fill(x_grid,y_grid, c=str(Sj[jj]),edgecolor='gray',linewidth=1)
plt.scatter(swarm_x, swarm_y, c=swarm_c, s=50)

plt.grid()
fig.savefig('test_'+str(test)+'.svg',bbox_inches='tight',pad_inches=0.1)

print('*******************************')
###############################################################################
