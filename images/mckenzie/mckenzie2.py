import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# 
#  |----------------T-------------------|
#  |                | \ phi             |  
#  |--------------- L  \ ---------------|
#  |                 \  \               |
#  |                  \  |              |
#  |                   \ E              |
#  |                                    |
#  |------------------------------------|

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

Lx=1000e3  # horizontal extent of the domain 
Ly=250e3  # vertical extent of the domain 

nelx = 1000
nely = 250
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total

TMantle=1300
xT=Lx/2
xE=xT+450e3
phi=30./180.*np.pi
yL=Ly-120e3

rho=3000
hcapa=1250
hcond=2.5
vel=0.5*0.01/3600/24/365
R=rho*hcapa*vel*(Ly-yL)/2/hcond
nmax=200

#################################################################
# grid point setup
#################################################################
x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

#################################################################
# connectivity
#################################################################
icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

#################################################################
# compute temperature field 
#################################################################
T = np.empty(nnp, dtype=np.float64)  

for i in range (0,nnp):

    T[i]=TMantle

    if y[i]>= yL: 
       T[i]=(1-(y[i]-yL)/(Ly-yL))*TMantle

    if x[i]>=xT and x[i]<=xE and \
       y[i]<=Ly-(x[i]-xT)*np.tan(phi) and \
       y[i]>=yL-(x[i]-xT)*np.tan(phi):
       ytop=Ly-(x[i]-xT)*np.tan(phi)
       ybot=yL-(x[i]-xT)*np.tan(phi)
       yp=(y[i]-ybot)/(ytop-ybot)
       xp=(x[i]-xT)/(ytop-ybot)
       Ti=0.
       for n in range(1,nmax):
           Ti+=(-1.)**n/n/np.pi*np.exp( (R-np.sqrt(R*R+n*n*np.pi*np.pi))*xp)*np.sin(n*np.pi*yp)
       #end for
       T[i]=(1+2*Ti)*TMantle
    # end if
# end for

#np.savetxt('temperature.ascii',np.array([x,y,T]).T,header='# x,y,T')

#####################################################################
# plot of solution
#####################################################################

T_temp=np.reshape(T,(nny,nnx))

fig,axes = plt.subplots()

im = axes.imshow(T_temp, interpolation='bilinear', cmap='Spectral_r', #cm.RdYlGn,
               origin='lower', extent=[0, Lx/1000, 0, Ly/1000],
               vmax=abs(T).max(), vmin=(T).min())
fig.colorbar(im,orientation="horizontal")
plt.contourf(T_temp,cmap='Spectral_r',levels=[0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300])

plt.savefig('temperature.pdf')
plt.show()

