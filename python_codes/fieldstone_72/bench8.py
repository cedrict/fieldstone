import numpy as np

###############################################################################

def vy_th(phi1,phi2,rho1,rho2):
    c11 = (eta1*2*phi1**2)/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) - (2*phi2**2)/(np.cosh(2*phi2)-1-2*phi2**2)
    d12 = (eta1*(np.sinh(2*phi1) -2*phi1))/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) + (np.sinh(2*phi2)-2*phi2)/(np.cosh(2*phi2)-1-2*phi2**2)
    i21 = (eta1*phi2*(np.sinh(2*phi1)+2*phi1))/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) + (phi2*(np.sinh(2*phi2)+2*phi2))/(np.cosh(2*phi2)-1-2*phi2**2) 
    j22 = (eta1*2*phi1**2*phi2)/(eta2*(np.cosh(2*phi1)-1-2*phi1**2))-(2*phi2**3)/(np.cosh(2*phi2)-1-2*phi2**2 )
    K=-d12/(c11*j22-d12*i21)
    val=K*(rho1-rho2)/2/eta2*(Ly/2.)*abs(gy)*amplitude
    return val
