import numpy as np

Ggrav = 6.67430e-11

R=1.123
rho=1e6
M=2*np.pi*rho
N=1000

dm=M/N

case=1

###############################################################################

x = np.empty(N,dtype=np.float64)  # x coordinates
y = np.empty(N,dtype=np.float64)  # y coordinates
z = np.empty(N,dtype=np.float64)  # z coordinates

for i in range(0,N):
    theta_i=i*2*np.pi/N
    x[i]=R*np.cos(theta_i)
    y[i]=R*np.sin(theta_i)
    z[i]=0.

np.savetxt('circle.ascii',np.array([x,y,z]).T,header='# x,y')

###############################################################################

n=200

xP = np.empty(n,dtype=np.float64)
yP = np.empty(n,dtype=np.float64)
zP = np.empty(n,dtype=np.float64)

if case==1:
   for i in range(0,n):
       xP[i]=0
       yP[i]=0
       zP[i]=i*(2*R)/(n-1)

if case==2:
   for i in range(0,n):
       xP[i]=i*(2*R)/(n-1)
       yP[i]=0
       zP[i]=0

np.savetxt('line.ascii',np.array([xP,yP,zP]).T,header='# x,y')

###############################################################################

gx = np.zeros(n,dtype=np.float64)
gy = np.zeros(n,dtype=np.float64) 
gz = np.zeros(n,dtype=np.float64) 

for im in range(0,n):
    for i in range(0,N):

        dist=np.sqrt((x[i]-xP[im])**2+(y[i]-yP[im])**2+(z[i]-zP[im])**2)
        Kernel=Ggrav/dist**3*dm

        gx[im]-=Kernel*(xP[im]-x[i])
        gy[im]-=Kernel*(yP[im]-y[i])
        gz[im]-=Kernel*(zP[im]-z[i])

np.savetxt('gravity.ascii',np.array([xP,yP,zP,gx,gy,gz]).T,header='# x,y,z,gx,gy,gz')

###############################################################################
gg = np.zeros(n,dtype=np.float64) 

if case==2:
   for im in range(1,n):
       xtilde=xP[im]/R
       ztilde=zP[im]/R
       gg[im]=-Ggrav*M/R**2/np.sqrt( (1+xtilde**2+ztilde**2)**2 - 4*xtilde**2 )


if case==1:
   for im in range(1,n):
       gg[im]=-Ggrav*M*zP[im]/(R**2+zP[im]**2)**1.5

np.savetxt('gravity_th.ascii',np.array([xP,yP,zP,gg]).T,header='# x,y,z,gx,gy,gz')

