import numpy as np

def basis_functions_V(r,s):
    N_0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N_1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N_2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N_3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N_4=    (1.-r**2) * 0.5*s*(s-1.)
    N_5= 0.5*r*(r+1.) *    (1.-s**2)
    N_6=    (1.-r**2) * 0.5*s*(s+1.)
    N_7= 0.5*r*(r-1.) *    (1.-s**2)
    N_8=    (1.-r**2) *    (1.-s**2)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr_0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr_1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr_2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr_3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr_4=       (-2.*r) * 0.5*s*(s-1)
    dNdr_5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr_6=       (-2.*r) * 0.5*s*(s+1)
    dNdr_7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr_8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,\
                     dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds_0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds_1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds_2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds_3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds_4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds_5= 0.5*r*(r+1.) *       (-2.*s)
    dNds_6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds_7= 0.5*r*(r-1.) *       (-2.*s)
    dNds_8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,\
                     dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)

###################################################################################################

def basis_functions_P(r,s):
    N_0=0.25*(1-r)*(1-s)
    N_1=0.25*(1+r)*(1-s)
    N_2=0.25*(1+r)*(1+s)
    N_3=0.25*(1-r)*(1+s)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

def basis_functions_P_dr(r,s):
    dNdr_0=-0.25*(1-s)
    dNdr_1= 0.25*(1-s)
    dNdr_2= 0.25*(1+s)
    dNdr_3=-0.25*(1+s)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)

def basis_functions_P_ds(r,s):
    dNds_0=-0.25*(1-r)
    dNds_1=-0.25*(1+r)
    dNds_2= 0.25*(1+r)
    dNds_3= 0.25*(1-r)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)

###################################################################################################
