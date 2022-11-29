import numpy as np
import numba

###############################################################################
# linear interpolation between points in profile!

@numba.njit
def material_model(x,z,eta_blob,rho_blob,z_blob,R_blob,npt_rho,npt_eta,prof_rho,prof_eta,blobtype):
    rr=np.sqrt(x**2+z**2)
    
    for i in range(0,npt_rho-1):
        if rr>=prof_rho[0,i] and rr<=prof_rho[0,i+1]:
           rho=prof_rho[1,i]+(rr-prof_rho[0,i])*(prof_rho[1,i+1]-prof_rho[1,i])/(prof_rho[0,i+1]-prof_rho[0,i])
           #rho=prof_rho[1,i]
           break
    for i in range(0,npt_eta-1):
        if rr>=prof_eta[0,i] and rr<=prof_eta[0,i+1]:
           eta=prof_eta[1,i]+(rr-prof_eta[0,i])*(prof_eta[1,i+1]-prof_eta[1,i])/(prof_eta[0,i+1]-prof_eta[0,i])
           #eta=prof_eta[1,i]
           break

    #print (x,z,rho)
    if x**2+(z-z_blob)**2<1.001*R_blob**2:
       rho=rho_blob
       eta=eta_blob

    return eta,rho
