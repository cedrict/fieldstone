import numpy as np
import numba
from basis_functions_numba import *
#from material_model_numba import *
from numba import jit

Ggrav = 6.67430e-11

#@numba.njit(parallel=True)

@jit(nopython=True)
def compute_gravity_at_point3(xM,yM,zM,nel,nqel,qcoords_r,qcoords_s,qweights,\
                              rhoq,jcob,xq,zq,nel_phi,thetaq,radq):

    dphi=2*np.pi/nel_phi

    gx=0.
    gy=0.
    gz=0.
    counterq=0
    for iel in range(0,nel):
        for kq in range (0,nqel):
            massq=rhoq[counterq]*jcob[counterq]*qweights[kq]*xq[counterq]*dphi
            for jel in range(0,nel_phi):
                x_c=radq[counterq]*np.sin(thetaq[counterq])*np.cos(jel*dphi)
                y_c=radq[counterq]*np.sin(thetaq[counterq])*np.sin(jel*dphi)
                dist=np.sqrt((xM-x_c)**2 + (yM-y_c)**2 + (zM-zq[counterq])**2)

                #print(x_c,y_c,zq[counterq],radq[counterq],thetaq[counterq])

                Kernel=Ggrav/dist**3*massq
                gx-= Kernel*(xM-x_c)
                gy-= Kernel*(yM-y_c)
                gz-= Kernel*(zM-zq[counterq] )
            #end for
            counterq+=1
        #end for
    #end for


    return gx,gy,gz

##################################################################################























