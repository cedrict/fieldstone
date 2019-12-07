import numpy as np
import matplotlib.pyplot as plt

Y=20e6
eta_v=1.e25
eta_m=1.e21

vfile=open('values.ascii',"w")
wfile=open('values_old.ascii',"w")

for m in range(-190,-119):
    eps_T=10.**(m/10.)
    if 2.*eta_v*eps_T < Y:
       eta_eff=eta_v
       tau=2.*eta_v*eps_T
       eps_v=eps_T
       eps_vp=0.
       eta_vp=0.

       tau_old=tau
       eta_eff_old=eta_eff
       eta_vp_old=eta_vp

    else:
       tau=(Y+2.*eta_m*eps_T)/(eta_m/eta_v+1.)
       eps_v=tau/2./eta_v
       eps_vp=eps_T-eps_v
       eta_vp=Y/2/eps_vp+eta_m
       eta_eff=1./(1./eta_v+1./eta_vp)

       #using eps_T instead of eps_vp 
       eta_vp_old=Y/2/eps_T+eta_m
       eta_eff_old=1./(1./eta_v+1./eta_vp_old)
       tau_old=2.*eta_eff_old*eps_T

    #end if
    vfile.write("%9e %9e %9e %9e %9e %9e %9e %9e %9e \n" %(eps_T,tau,eps_v,eps_vp,eta_v,eta_vp,eta_eff,eta_m,Y))
    wfile.write("%9e %9e %9e %9e %9e %9e %9e \n" %(eps_T,tau_old,eta_v,eta_vp_old,eta_eff_old,eta_m,Y))
#end for

vfile.close()
wfile.close()

