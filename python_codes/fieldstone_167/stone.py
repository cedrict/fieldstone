import numpy as np
import scipy as scp

TKelvin=273. # set to zero to get correct results
R=8.314
Cp=1000
rho=2750
e=1

#----------------------------------------------------------
#brace & kohlstedt 1980
#n=3.
#A=5e-6*1e-18
#Q=1.9e5

#----------------------------------------------------------
#stuwe & sandiford 1994
#n=3.
#A=2e-4*1e-18
#Q=2.5e5

#----------------------------------------------------------
#brace & kohlstedt 1980
#n=3.
#A=7e4*1e-18
#Q=5.2e5

#----------------------------------------------------------
#gleason Tullis 1995
#A=1.1e-28
#n=4.
#Q=223e3

#----------------------------------------------------------
# Alyssa OK
# Hirth & Kohlstedt 2003 - olivine wet diffusion
n=1
m=3
d=1e-5
A=10**7.4 * 10**(-6*n-6*m) * d**-m
Q=375e3

#----------------------------------------------------------
# Alyssa OK
#Hirth & Kohlstedt 2003 - olivine dry dislocation 
# r=0 so fugacity plays no role
#n=3.5
#A=10**5 *10**(-6*n) 
#Q=530e3

#----------------------------------------------------------
# Alyssa OK
#Rutter & Brodie 2004 - quartz wet diffusion
# r=0 so fugacity plays no role
#m=2
#n=1
#d=1e-5
#A=10**(-0.4) *10**(-6*n-6*m)  * d**(-m)
#Q=220e3

#----------------------------------------------------------
# Hirth et al 2001 (hitd01)- quartz wet dislocation
# 'A fH2O of ~37 MPa was calculated for these conditions'

#n=4
#fH2O=37e6
#r=1
#A=10**(-11.2)*10**(-6*n) * fH2O**r
#Q=135e3

#----------------------------------------------------------
# Wang et al 2023 - Garnet amphibolite dislocation creep (80% amph/20% garnet)
# https://doi.org/10.1029/2022GL102320
#n=3.1
#A=10**-6.2*10**(-6*n)
#Q=154e3

#----------------------------------------------------------
# Wang et al 2012 - Mafic granulite dislocation creep 
# https://doi.org/10.1016/j.epsl.2012.08.004
#n=3.2
#A=1e-2*10**(-6*n)
#Q=244e3


Trange=np.arange(400,1400,1)

###############################################################################

def f(T,T0,D,t):
    TT=T+TKelvin
    x=D/TT
    TT0=T0+TKelvin
    x0=D/TT0
    return  t/C/D - ( np.exp(-x)/x -np.exp(-x0)/x0 -scp.special.expi(-x0) +scp.special.expi(-x) ) 

def feq13(T,T0,D,t):
    TT=T+TKelvin
    x=D/TT
    TT0=T0+TKelvin
    x0=D/TT0
    return  t - C*D*( np.exp(-x)/x -np.exp(-x0)/x0 +scp.special.expi(x0) -scp.special.expi(x) ) 

###############################################################################

for sr in (1e-12,1e-13,1e-14,1e-15,1e-16):
    D=Q/n/R
    C=rho*Cp/sr*(sr/A)**(-1./n)
    print ('sr=',sr)
    srfile=open('sr'+str(sr)+'.ascii',"w")
    t=e/sr

    for T0 in range (400,1400,1):
        res=f(Trange,T0,D,t)
        # find the zero
        for i in range(0,len(Trange)-1):
            if res[i]*res[i+1]<0: 
               srfile.write("%e %e \n" %(Trange[i],T0))
    
        #np.savetxt('res'+str(T0)+'.ascii',np.array([Trange,res]).T)

    #end for

#end for

###############################################################################
# disprove eq 13 of paper

if False:

   sr =1e-14
   D=Q/n/R
   C=rho*Cp/sr*(sr/A)**(-1./n)

   for T0 in (400,500,600):
       res=feq13(Trange,T0,D,t)
       np.savetxt('feq13_'+str(T0)+'.ascii',np.array([Trange,res]).T)

###############################################################################

