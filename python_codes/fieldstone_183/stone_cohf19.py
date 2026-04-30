import numpy as np
from scipy import special

###############################################################################
# Coltice private communication: "je me suis rendu compte que j'ai fait une erreur
# dans le redimensionnement du volume d'activation dans le modèle de 2019. Dans le 
# sup. mat. tu trouveras 13.8cm3/mol alors qu'en refaisant les calculs on trouve 
# en réalité 0.7cm3/mol. Dans une exponentielle, ça fait un petit paquet... 
# Donc si tu tentes de refaire la simu, fais gaffe à ça"
# Note that Arnould et al use the lithostatic pressure in the viscosity

npts=1000 # number of points

Rsurf=6371e3
Rcmb=3485e3
rho0=4000
g=9.8
eta0=1e22
R=8.314
V=0.7e-6 #13.8e-6
alpha=3e-5
kappa=1e-6
hcond=3.15
hcapa=hcond/kappa/rho0 ; print('Cp=',hcapa)
Tsurf=255
Tcmb=2645
year=365.25*24*3600
E=160e3
T0=1530 # for viscosity 

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

for age in (50e6,100e6,200e6,300e6):

    for coeff in (0.5,0.6,0.7,0.8,0.9):

            Tm=Tsurf+(Tcmb-Tsurf)*coeff 
            print('age=',age,'Tm=',Tm)

            age_cmb=age*year
            age_surf=age*year

            ###############################################################################
            # assign initial temperature
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

            np.savetxt('p_'+str(int(age/1e6))+'.ascii',np.array([r,p]).T)

            ###############################################################################
            # compute viscosity 
            ###############################################################################

            eta=np.zeros(npts,dtype=np.float64)

            for i in range(0,npts):
                depth=Rsurf-r[i]
                if depth<660e3: 
                   A=1#/30
                else: 
                   A=30

                eta[i]=A*eta0*np.exp((E+p[i]*V)/R/T[i]-E/R/T0)

                eta[i]=min(eta[i],1e26) 

            np.savetxt('eta_'+str(int(coeff*10))+'_'+str(int(age/1e6))+'.ascii',np.array([r,np.log10(eta)]).T)

    #end for coeff

#end for age



