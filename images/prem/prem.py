import numpy as np
import sys as sys

#####################################################################
# PREM density
#####################################################################
# obtained from dzan81

def prem_density(radius):
    x=radius/6371.e3
    if radius>6371e3:
       densprem=0
    elif radius<=1221.5e3:
       densprem=13.0885-8.8381*x**2
    elif radius<=3480e3:
       densprem=12.5815-1.2638*x-3.6426*x**2-5.5281*x**3
    elif radius<=3630.e3:
       densprem=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
    elif radius<=5600.e3:
       densprem=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
    elif radius<=5701.e3:
       densprem=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
    elif radius<=5771.e3:
       densprem=5.3197-1.4836*x
    elif radius<=5971.e3:
       densprem=11.2494-8.0298*x
    elif radius<=6151.e3:
       densprem=7.1089-3.8045*x
    elif radius<=6291.e3:
       densprem=2.6910+0.6924*x
    elif radius<=6346.e3:
       densprem=2.6910+0.6924*x
    elif radius<=6356.e3:
       densprem=2.9
    elif radius<=6368.e3:
       densprem=2.6
    else:
       densprem=1.020
    return densprem*1000

#####################################################################

R=6371e3
Ggrav=6.6738480e-11

gfile=open('g.ascii',"w")

for Ncell in range(10,100000,250):

    h=R/Ncell

    #####################################################################
    # assign coordinate to center of cells
    #####################################################################

    r = np.empty(Ncell,dtype=np.float64) 
    counter=0
    for i in range(0,Ncell):
        r[counter]=i*h+h/2.
        counter+=1
    #end for

    #####################################################################
    # assign density to cells
    #####################################################################

    rho = np.empty(Ncell,dtype=np.float64) 
    for i in range(0,Ncell):
       rho[i]=prem_density(r[i])
    #end for

    np.savetxt('rho.ascii',np.array([r,rho]).T,header='# x,rho')

    #####################################################################
    # compute gr 
    #####################################################################
  
    g=0.
    for i in range(0,Ncell):
        g+=rho[i]*h*r[i]**2     
    #end for
    g*=4.*np.pi*Ggrav/R**2

    gfile.write("%e %e \n" %(h, g))

#end for

