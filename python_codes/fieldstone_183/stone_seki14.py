import numpy as np
from scipy import special

npts=1000 # number of points

Rsurf=3.398e6
Rcmb=Rsurf-1780e3 
rho0=3400
g=3.72
eta0=1e21
R=8.314
V=3.61e-6
alpha=3e-5
kappa=1e-6
hcapa=1200
Tsurf=220
DeltaT=1600
E=157e3 
Tcmb=Tsurf+DeltaT
Tm=Tsurf+0.95*(Tcmb-Tsurf) # not 100% sure
year=365.25*24*3600
age=250e6

Ra=alpha*rho0*g*(Tcmb-Tsurf)*(Rsurf-Rcmb)**3/kappa/eta0
print(Ra)

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

###############################################################################
###############################################################################
###############################################################################

if True:

            age_cmb=age*year
            age_surf=age*year


            ###############################################################################
            # assign initial temperature
            # note that the equations 10&11 of the paper are somewhat wrong :)
            ###############################################################################

            T=np.zeros(npts,dtype=np.float64)
    
            for i in range(0,npts):
                T[i]=initial_temperature_hsc(r[i],Rcmb,Rsurf,Tcmb,Tsurf,age_cmb,age_surf,Tm,kappa)

            np.savetxt('T.ascii',np.array([r,T]).T)

            ###############################################################################
            # compute density
            ###############################################################################

            rho=np.zeros(npts,dtype=np.float64)
    
            for i in range(0,npts):
                rho[i]=rho0*(1-alpha*T[i])

            np.savetxt('rho.ascii',np.array([r,rho]).T)

            ###############################################################################
            # compute pressure: we need to integrate downwards, starting with p=0 at surf
            ###############################################################################

            p=np.zeros(npts,dtype=np.float64)
    
            p[npts-1]=0

            for i in range(npts-2,-1,-1):
                p[i]=p[i+1]+(rho[i]+rho[i+1])/2*g*dr

            np.savetxt('p.ascii',np.array([r,p]).T)

            ###############################################################################
            # compute viscosity 
            ###############################################################################

            eta=np.zeros(npts,dtype=np.float64)

            for i in range(0,npts):
                depth=Rsurf-r[i]
                if depth<996e3: 
                   A=1
                else: 
                   A=8

                eta[i]=A*eta0*np.exp((E+p[i]*V)/R/T[i]-(E+p[i]*V)/R/(DeltaT+Tsurf))

                eta[i]=min(eta[i],1e25) # viscosity limiter, see section 2.3.4

            np.savetxt('eta.ascii',np.array([r,np.log10(eta)]).T)

        #end for
    #end for
#end for



