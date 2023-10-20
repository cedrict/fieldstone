import numpy as np
from numba import jit

#------------------------------------------------------------------------------
# basis functions for Crouzeix-Raviart element and Taylor-Hood element
#------------------------------------------------------------------------------
def NNV(rq,sq,CR):
    if CR:
       NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
       NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
       NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
       NV_3= 4.*(1.-rq-sq)*rq*(1.-3.*sq) 
       NV_4= 4.*rq*sq*(-2.+3.*rq+3.*sq)
       NV_5= 4.*(1.-rq-sq)*sq*(1.-3.*rq) 
       NV_6= 27*(1.-rq-sq)*rq*sq
       return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6])
    else:
       NV_0= 1-3*rq-3*sq+2*rq**2+4*rq*sq+2*sq**2 
       NV_1= -rq+2*rq**2
       NV_2= -sq+2*sq**2
       NV_3= 4*rq-4*rq**2-4*rq*sq
       NV_4= 4*rq*sq 
       NV_5= 4*sq-4*rq*sq-4*sq**2
       return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5])

def dNNVdr(rq,sq,CR):
    if CR:
       dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
       dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
       dNVdr_2= 3*sq-6*rq*sq-3*sq**2
       dNVdr_3= -8*rq+24*rq*sq+4-16*sq+12*sq**2
       dNVdr_4= -8*sq+24*rq*sq+12*sq**2
       dNVdr_5= -16*sq+24*rq*sq+12*sq**2
       dNVdr_6= -54*rq*sq+27*sq-27*sq**2
       return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6])
    else:
       dNVdr_0= -3+4*rq+4*sq 
       dNVdr_1= -1+4*rq
       dNVdr_2= 0
       dNVdr_3= 4-8*rq-4*sq
       dNVdr_4= 4*sq
       dNVdr_5= -4*sq
       return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5])

def dNNVds(rq,sq,CR):
    if CR:
       dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
       dNVds_1= rq*(3-3*rq-6*sq)
       dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
       dNVds_3= -16*rq+24*rq*sq+12*rq**2
       dNVds_4= -8*rq+12*rq**2+24*rq*sq
       dNVds_5= 4-16*rq-8*sq+24*rq*sq+12*rq**2
       dNVds_6= -54*rq*sq+27*rq-27*rq**2
       return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6])
    else:
       dNVds_0= -3+4*rq+4*sq 
       dNVds_1= 0
       dNVds_2= -1+4*sq
       dNVds_3= -4*rq
       dNVds_4= +4*rq
       dNVds_5= 4-4*rq-8*sq
       return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5])

def NNP(rq,sq):
    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return np.array([NP_0,NP_1,NP_2])



