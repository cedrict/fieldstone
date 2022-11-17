import numpy as np
from magnetostatics import *

hx=1
hy=1.5
hz=1.9

Mx=1
My=1
Mz=1


###########################################################
# create cell coordinates and connectivity array
#     z
#     |
#     4---7---y  
#    /   /
#   5---6
#     |
#     0---3---y  
#    /   /
#   1---2
#  /
# x

m=8 # number of vertices

x = np.empty(m,dtype=np.float64) 
y = np.empty(m,dtype=np.float64) 
z = np.empty(m,dtype=np.float64) 

x[0]=0*hx ; y[0]=0*hy ; z[0]=0*hz
x[1]=1*hx ; y[1]=0*hy ; z[1]=0*hz
x[2]=1*hx ; y[2]=1*hy ; z[2]=0*hz
x[3]=0*hx ; y[3]=1*hy ; z[3]=0*hz
x[4]=0*hx ; y[4]=0*hy ; z[4]=1*hz
x[5]=1*hx ; y[5]=0*hy ; z[5]=1*hz
x[6]=1*hx ; y[6]=1*hy ; z[6]=1*hz
x[7]=0*hx ; y[7]=1*hy ; z[7]=1*hz

np.savetxt('vertices.ascii',np.array([x,y,z]).T)

icon =np.zeros(m,dtype=np.int32)
icon[0]=0
icon[1]=1
icon[2]=2
icon[3]=3
icon[4]=4
icon[5]=5
icon[6]=6
icon[7]=7

###########################################################
# first test: a point in space
###########################################################

print('*****test1*****')

xmeas=3
ymeas=4
zmeas=5

nqdim=2

B1=compute_B_quadrature(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,Mx,My,Mz,nqdim)

print('Vol quad B:',B1)

B2=compute_B_surface_integral(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,Mx,My,Mz)

print('Surf int B:',B2)

print('Difference:',B1+B2)

###########################################################
# second test: a point in space in a face plane
###########################################################

print('*****test2*****')

xmeas=1
ymeas=1.5
zmeas=5

nqdim=2

B1=compute_B_quadrature(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,Mx,My,Mz,nqdim)

print('Vol quad B:',B1)

B2=compute_B_surface_integral(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,Mx,My,Mz)

print('Surf int B:',B2)

print('Difference:',B1+B2)

###########################################################
# third test: a point in space, varying quad degrees 
###########################################################

print('*****test3*****')

xmeas=3
ymeas=4
zmeas=5

B2=compute_B_surface_integral(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,Mx,My,Mz)
print('Surf int B:',B2)

for nqdim in range(2,9):

    print('nqdim=',nqdim)

    B1=compute_B_quadrature(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,Mx,My,Mz,nqdim)
    print('Vol quad B:',B1)

    print('Difference:',B1+B2)





