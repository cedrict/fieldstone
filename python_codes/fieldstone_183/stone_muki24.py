import numpy as np
from scipy import special

npts=1000 # number of points

Rsurf=3.3895e6
Rcmb=1.830e6
rho0=3500
g=3.72
eta0=1e21
R=8.314
V=6.6e-6
alpha=2e-5
kappa=1e-6
hcapa=1250
Tsurf=220
DeltaT=1500
Tm=1720
year=365.25*24*3600

dr=(Rsurf-Rcmb)/(npts-1) # spacing between points

###############################################################################
# function borrowed from MEEUUW

def initial_temperature_hsc(r,Rcmb,Rsurf,Tcmb,Tsurf,age_cmb,age_surf,Tm,kappa):
    val=Tsurf+(Tm-Tsurf)*special.erf((Rsurf-r)/2/np.sqrt(age_surf*kappa))\
             -(Tcmb-Tm)*(-1+special.erf((r-Rcmb)/2/np.sqrt(age_cmb*kappa)))
    return val

###############################################################################
# compute coordinates: equidistant points between Rcmb and Rsurf
###############################################################################

r=np.zeros(npts,dtype=np.float64)

for i in range(0,npts):
    r[i]=Rcmb+dr*i

#np.savetxt('r.ascii',np.array([r]).T)

###############################################################################
###############################################################################
###############################################################################

for E in (117e3,350e3):
    for coeff in (1.0,1.1,1.2):
        for age in (50e6,100e6,200e6,300e6):
            print('E=',E,'DeltaT=',DeltaT,'age=',age)

            age_cmb=age*year
            age_surf=age*year

            Tcmb=Tsurf+coeff*DeltaT

            ###############################################################################
            # assign initial temperature
            # note that the equations 10&11 of the paper are somewhat wrong :)
            ###############################################################################

            T=np.zeros(npts,dtype=np.float64)
    
            for i in range(0,npts):
                T[i]=initial_temperature_hsc(r[i],Rcmb,Rsurf,Tcmb,Tsurf,age_cmb,age_surf,Tm,kappa)

            np.savetxt('T_'+str(int(coeff*10))+'_'+str(int(age/1e6))+'.ascii',np.array([r,T]).T)

            ###############################################################################
            # compute density
            ###############################################################################

            rho=np.zeros(npts,dtype=np.float64)
    
            for i in range(0,npts):
                rho[i]=rho0*(1-alpha*T[i])

            np.savetxt('rho_'+str(int(coeff*10))+'_'+str(int(age/1e6))+'.ascii',np.array([r,rho]).T)

            ###############################################################################
            # compute pressure: we need to integrate downwards, starting with p=0 at surf
            ###############################################################################

            p=np.zeros(npts,dtype=np.float64)
    
            p[npts-1]=0

            for i in range(npts-2,-1,-1):
                p[i]=p[i+1]+(rho[i]+rho[i+1])/2*g*dr

            np.savetxt('p_'+str(int(coeff*10))+'_'+str(int(age/1e6))+'.ascii',np.array([r,p]).T)

            ###############################################################################
            # compute viscosity 
            ###############################################################################

            eta=np.zeros(npts,dtype=np.float64)

            for i in range(0,npts):
                depth=Rsurf-r[i]
                if depth<100e3: 
                   A=10
                elif depth<1000e3:
                   A=0.1
                else: 
                   A=10

                eta[i]=A*eta0*np.exp((E+p[i]*V)/R/T[i]-(E+p[i]*V)/R/(DeltaT+Tsurf))

                eta[i]=min(eta[i],1e25) # viscosity limiter, see section 2.3.4

            np.savetxt('eta_'+str(int(coeff*10))+'_'+str(int(age/1e6))+'_E'+str(int(E/1000))+'.ascii',np.array([r,np.log10(eta)]).T)

        #end for
    #end for
#end for



