import numpy as np
import random

#define 3 vertices of triangle

x=[1.,3.,2.]
y=[1.,1.5,2.9]

for npts in (10,100,1000,10000,100000,1000000):

    print('npts:',npts)

    xcoords=np.zeros(npts,dtype=np.float64)
    ycoords=np.zeros(npts,dtype=np.float64)

    # generate point in triangle 
    r=random.uniform(0,1)
    s=random.uniform(0,1-r)
    N=np.zeros(3,dtype=np.float64)
    N[0]=1-r-s
    N[1]=r
    N[2]=s
    xpt=N.dot(x)
    ypt=N.dot(y)

    for k in range(0,npts):

        # select vertex 0, 1, or 2
        ivertex=random.randint(0,2)

        xcoords[k]=(xpt+x[ivertex])/2
        ycoords[k]=(ypt+y[ivertex])/2

        xpt=xcoords[k]
        ypt=ycoords[k]

    # end for

    np.savetxt('pts_'+str(npts)+'.ascii',np.array([xcoords,ycoords]).T)

#end for npts
