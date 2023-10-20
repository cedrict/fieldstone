import numpy as np
from numba import jit

#------------------------------------------------------------------------------
# basis functions for Crouzeix-Raviart element and Taylor-Hood element
#------------------------------------------------------------------------------

bubble=1

def B(r,s):
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)
    elif bubble==2:
       return (1-r**2)*(1-s**2)*(1+beta*(r+s))
    else:
       return (1-r**2)*(1-s**2)

def dBdr(r,s):
    if bubble==1:
       return (1-s**2)*(1-s)*(-1-2*r+3*r**2)
    elif bubble==2:
       return (s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))
    else:
       return (-2*r)*(1-s**2)

def dBds(r,s):
    if bubble==1:
       return (1-r**2)*(1-r)*(-1-2*s+3*s**2)
    elif bubble==2:
       return (r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)
    else:
       return (1-r**2)*(-2*s)


#@jit(nopython=True)
def NNV(rq,sq,elt):
    if elt=='CR':
       NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
       NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
       NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
       NV_3= 4.*(1.-rq-sq)*rq*(1.-3.*sq) 
       NV_4= 4.*rq*sq*(-2.+3.*rq+3.*sq)
       NV_5= 4.*(1.-rq-sq)*sq*(1.-3.*rq) 
       NV_6= 27*(1.-rq-sq)*rq*sq
       return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6],dtype=np.float64)
    if elt=='P2P1':
       NV_0= 1-3*rq-3*sq+2*rq**2+4*rq*sq+2*sq**2 
       NV_1= -rq+2*rq**2
       NV_2= -sq+2*sq**2
       NV_3= 4*rq-4*rq**2-4*rq*sq
       NV_4= 4*rq*sq 
       NV_5= 4*sq-4*rq*sq-4*sq**2
       return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5],dtype=np.float64)
    if elt=='Q2Q1':
       NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
       NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
       NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
       NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
       NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
       NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
       NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
       NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
       NV_8=     (1.-rq**2) *     (1.-sq**2)
       return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8],dtype=np.float64)
    if elt=='Q1+Q1':
       NV_0= 0.25*(1-rq)*(1-sq) - 0.25*B(rq,sq)
       NV_1= 0.25*(1+rq)*(1-sq) - 0.25*B(rq,sq)
       NV_2= 0.25*(1+rq)*(1+sq) - 0.25*B(rq,sq)
       NV_3= 0.25*(1-rq)*(1+sq) - 0.25*B(rq,sq)
       NV_4= B(rq,sq)
       return np.array([NV_0,NV_1,NV_2,NV_3,NV_4],dtype=np.float64)

#@jit(nopython=True)
def dNNVdr(rq,sq,elt):
    if elt=='CR':
       dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
       dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
       dNVdr_2= 3*sq-6*rq*sq-3*sq**2
       dNVdr_3= -8*rq+24*rq*sq+4-16*sq+12*sq**2
       dNVdr_4= -8*sq+24*rq*sq+12*sq**2
       dNVdr_5= -16*sq+24*rq*sq+12*sq**2
       dNVdr_6= -54*rq*sq+27*sq-27*sq**2
       arr=np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6],dtype=np.float64)
       return arr
    if elt=='P2P1':
       dNVdr_0= -3+4*rq+4*sq 
       dNVdr_1= -1+4*rq
       dNVdr_2= 0
       dNVdr_3= 4-8*rq-4*sq
       dNVdr_4= 4*sq
       dNVdr_5= -4*sq
       arr=np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5],dtype=np.float64)
       return arr
    if elt=='Q2Q1':
       dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
       dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
       dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
       dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
       dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
       dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
       dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
       dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
       dNVdr_8=       (-2.*rq) *    (1.-sq**2)
       arr=np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)
       return arr
    if elt=='Q1+Q1':
       dNVdr_0=-0.25*(1.-sq) -0.25*dBdr(rq,sq)
       dNVdr_1=+0.25*(1.-sq) -0.25*dBdr(rq,sq)
       dNVdr_2=+0.25*(1.+sq) -0.25*dBdr(rq,sq)
       dNVdr_3=-0.25*(1.+sq) -0.25*dBdr(rq,sq)
       dNVdr_4=dBdr(rq,sq) 
       return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4],dtype=np.float64)

#@jit(nopython=True)
def dNNVds(rq,sq,elt):
    if elt=='CR':
       dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
       dNVds_1= rq*(3-3*rq-6*sq)
       dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
       dNVds_3= -16*rq+24*rq*sq+12*rq**2
       dNVds_4= -8*rq+12*rq**2+24*rq*sq
       dNVds_5= 4-16*rq-8*sq+24*rq*sq+12*rq**2
       dNVds_6= -54*rq*sq+27*rq-27*rq**2
       arr=np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6],dtype=np.float64)
       return arr
    if elt=='P2P1':
       dNVds_0= -3+4*rq+4*sq 
       dNVds_1= 0
       dNVds_2= -1+4*sq
       dNVds_3= -4*rq
       dNVds_4= +4*rq
       dNVds_5= 4-4*rq-8*sq
       arr=np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5],dtype=np.float64)
       return arr
    if elt=='Q2Q1':
       dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
       dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
       dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
       dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
       dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
       dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
       dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
       dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
       dNVds_8=     (1.-rq**2) *       (-2.*sq)
       arr=np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)
       return arr
    if elt=='Q1+Q1':
       dNVds_0=-0.25*(1.-rq) -0.25*dBds(rq,sq)
       dNVds_1=-0.25*(1.+rq) -0.25*dBds(rq,sq)
       dNVds_2=+0.25*(1.+rq) -0.25*dBds(rq,sq)
       dNVds_3=+0.25*(1.-rq) -0.25*dBds(rq,sq)
       dNVds_4=dBds(rq,sq) 
       return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4],dtype=np.float64)

#@jit(nopython=True)
def NNP(rq,sq,elt):
    if elt=='CR' or elt=='P2P1':
       NP_0=1.-rq-sq
       NP_1=rq
       NP_2=sq
       arr=np.array([NP_0,NP_1,NP_2],dtype=np.float64)
       return arr
    if elt=='Q2Q1' or elt=='Q1+Q1':
       NP_0=0.25*(1-rq)*(1-sq)
       NP_1=0.25*(1+rq)*(1-sq)
       NP_2=0.25*(1+rq)*(1+sq)
       NP_3=0.25*(1-rq)*(1+sq)
       return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)
