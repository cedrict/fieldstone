import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

Lx=250e3  # horizontal extent of the domain 
Ly=50e3  # vertical extent of the domain 

nelx = 250
nely = 50
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total

l=Ly
rho=3000
hcapa=1250
hcond=2.5
vel=0.5*0.01/3600/24/365
Tl=800+273
R=rho*hcapa*vel*l/2/hcond
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
    xp=x[i]/l
    yp=y[i]/l

    Ti=0.
    for n in range(1,nmax):
        Ti+=(-1.)**n/n/np.pi*np.exp( (R-np.sqrt(R*R+n*n*np.pi*np.pi))*xp)*np.sin(n*np.pi*yp)
    #end for
    T[i]=1+2*Ti
# end for

#####################################################################
# plot of solution
#####################################################################

T_temp=np.reshape(T,(nny,nnx))

fig,axes = plt.subplots()

im = axes.imshow(T_temp, interpolation='bilinear', cmap='Spectral_r', 
               origin='lower', extent=[0, Lx/1000, 0, Ly/1000],
               vmax=abs(T).max(), vmin=(T).min())
fig.colorbar(im,orientation="horizontal")
plt.contourf(T_temp,cmap='Spectral_r',levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.savefig('temperature.pdf')
plt.show()

