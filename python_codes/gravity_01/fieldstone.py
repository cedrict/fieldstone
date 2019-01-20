import numpy as np
import math as math
import matplotlib.pyplot as plt

###############################################################################

def onePlot(variable, plotX, plotY, title, labelX, labelY, extVal, limitX, limitY, colorMap):
    im = axes[plotX][plotY].imshow(np.flipud(variable),extent=extVal, cmap=colorMap, interpolation="nearest")
    axes[plotX][plotY].set_title(title,fontsize=6,y=1.)
    axes[plotX][plotY].set_xlabel(labelX)
    axes[plotX][plotY].set_ylabel(labelY)
    fig.colorbar(im,ax=axes[plotX][plotY])
    return

def density(x,y,xsphere,ysphere,rsphere,rhosphere,rhomedium):
    if np.sqrt( (x-xsphere)**2 + (y-ysphere)**2 ) < rsphere:
       val=rhosphere
    else:
       val=rhomedium
    return val

###############################################################################

nsurf = 50

Lx=1000.e3
Ly=500.e3

nnx=200
nny=100
nnp=nnx*nny

nelx=nnx-1
nely=nny-1
nel=nelx*nely

hx=Lx/(nnx-1)
hy=Ly/(nny-1)

xsphere=500.e3
ysphere=250.e3
rsphere=50.e3
rhosphere=3200.
rhomedium=3000.
rhozero=3000.

Ggrav =6.6738480e-11

###############################################################################
xsurf = np.empty(nsurf, dtype=np.float64)  
ysurf = np.empty(nsurf, dtype=np.float64)  

dx=Lx/(nsurf-1)

for i in range(0,nsurf):
   xsurf[i]=i*dx
   ysurf[i]=Ly

###############################################################################
x = np.empty(nnp, dtype=np.float64)  
y = np.empty(nnp, dtype=np.float64)  

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1

np.savetxt('grid_points.ascii',np.array([x,y]).T,header='# x,y')

###############################################################################
xc = np.empty(nel, dtype=np.float64) 
yc = np.empty(nel, dtype=np.float64) 

counter=0
for j in range(0, nely):
    for i in range(0, nelx):
        xc[counter]=i*hx +hx/2.
        yc[counter]=j*hy +hy/2.
        counter += 1

###############################################################################
rho = np.empty(nel, dtype=np.float64) 

for iel in range(0,nel):
    rho[iel]=density(xc[iel],yc[iel],\
                     xsphere,ysphere,rsphere,rhosphere,rhomedium)

np.savetxt('density.ascii',np.array([xc,yc,rho]).T,header='# x,y')

###############################################################################
gsurf    = np.zeros(nsurf, dtype=np.float64) 
gsurf_th = np.zeros(nsurf, dtype=np.float64) 
gsurf_err= np.zeros(nsurf, dtype=np.float64) 

for i in range(0,nsurf):
    for iel in range(0,nel):
        dist2=(xsurf[i]-xc[iel])**2 + (ysurf[i]-yc[iel])**2
        gsurf[i]+= 2.*Ggrav/dist2*(rho[iel]-rhozero)*(hx*hy) * (ysurf[i]-yc[iel])

    gsurf_th[i]=2*math.pi*Ggrav*(rhosphere-rhozero)*rsphere**2*(ysurf[i]-ysphere)/ \
                ((xsurf[i]-xsphere)**2+(ysurf[i]-ysphere)**2)

    gsurf_err[i]=gsurf[i]-gsurf_th[i]

np.savetxt('gravity.ascii',np.array([xsurf,ysurf,gsurf,gsurf_th,gsurf_err]).T,header='# x,y,gy,gy_th,err(gy)')

#####################################################################
# plot of solution
#####################################################################

rho_temp=np.reshape(rho,(nely,nelx))

f,axarr = plt.subplots(3)#nrows=3,ncols=3,figsize=(80,80))

pos=axarr[0].imshow(np.flipud(rho_temp),interpolation="nearest",cmap='viridis')
f.colorbar(pos, ax=axarr[0])


axarr[1].plot(xsurf,gsurf)
axarr[1].plot(xsurf,gsurf_th)
axarr[1].set_xlabel('x')
axarr[1].set_ylabel('$g_y$')

axarr[2].plot(xsurf,gsurf_err)
axarr[2].set_xlabel('x')
axarr[2].set_ylabel('$error(g_y)$')

plt.savefig('solution.pdf')
plt.show()
