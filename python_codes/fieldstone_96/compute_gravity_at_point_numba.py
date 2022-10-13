import numpy as np
import numba
from basis_functions_numba import *

Ggrav = 6.67430e-11

###############################################################################
# this simple approach considers the middle of the triangle as a point mass.

@numba.njit(parallel=True)
def compute_gravity_at_point1(xM,yM,zM,nel,xV,zV,iconV,rho,arear,dphi,nel_phi):

    gx=0.
    gy=0.
    gz=0.
    #VOL=0
    for iel in numba.prange(0,nel):
        xc=np.sum(xV[iconV[0:3,iel]])/3
        zc=np.sum(zV[iconV[0:3,iel]])/3
        rc=np.sqrt(xc**2+zc**2)
        theta=np.arccos(zc/rc)
        for jel in numba.prange(0,nel_phi):
            x_c=rc*np.sin(theta)*np.cos(jel*dphi)
            y_c=rc*np.sin(theta)*np.sin(jel*dphi)
            vol=arear[iel]/2/np.pi*dphi #arear contains 2pi already!
            #VOL+=vol
            mass=vol*rho[iel]
            dist=np.sqrt((xM-x_c)**2 + (yM-y_c)**2 + (zM-zc)**2)
            Kernel=Ggrav/dist**3*mass
            gx-= Kernel*(xM-x_c)
            gy-= Kernel*(yM-y_c)
            gz-= Kernel*(zM-zc)
            #print(x_c,y_c,zc)
        #end for
    #end for
    return gx,gy,gz

###############################################################################

@numba.njit(parallel=True)
def compute_gravity_at_point2(xM,yM,zM,nel,xV,zV,iconV,rho,dphi,nel_phi,qcoords_r,qcoords_s,qweights,CR,mV,nqel):

    #NNNV=np.zeros(6,dtype=np.float64)           # shape functions derivatives
    gx=0.
    gy=0.
    gz=0.
    for iel in numba.prange(0,nel):
        for kq in numba.prange (0,nqel):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            NNNV=NNV(rq,sq,CR)
            dNNNVdr=dNNVdr(rq,sq,CR)
            dNNNVds=dNNVds(rq,sq,CR)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in numba.prange(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*zV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*zV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            xq=NNNV[:].dot(xV[iconV[:,iel]])
            zq=NNNV[:].dot(zV[iconV[:,iel]])
            rq=np.sqrt(xq**2+zq**2)
            thetaq=np.arccos(zq/rq)
            massq=rho[iel]*jcob*weightq*xq*dphi
            for jel in numba.prange(0,nel_phi):
                x_c=rq*np.sin(thetaq)*np.cos(jel*dphi)
                y_c=rq*np.sin(thetaq)*np.sin(jel*dphi)
                dist=np.sqrt((xM-x_c)**2 + (yM-y_c)**2 + (zM-zq)**2)
                Kernel=Ggrav/dist**3*massq
                gx-= Kernel*(xM-x_c)
                gy-= Kernel*(yM-y_c)
                gz-= Kernel*(zM-zq )
            #end for
        #end for
    #end for

    return gx,gy,gz

###############################################################################
