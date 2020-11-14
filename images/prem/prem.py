import numpy as np
import sys as sys

R=6371e3
Ggrav=6.6738480e-11

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

def g_ic(radius):
    return 0.0302907*4*np.pi*Ggrav*R**3/radius**2

def g_oc(radius):
    return 0.56663*4*np.pi*Ggrav*R**3/radius**2

def g_lm(radius):
    return 0.904793*4*np.pi*Ggrav*R**3/radius**2

def g_tz1(radius):
    return 0.0354823*4*np.pi*Ggrav*R**3/radius**2

def g_tz2(radius):
    return 0.1026*4*np.pi*Ggrav*R**3/radius**2

def g_tz3(radius):
    return 0.0892215*4*np.pi*Ggrav*R**3/radius**2

def g_lvz(radius):
    return 0.0705516*4*np.pi*Ggrav*R**3/radius**2

def g_lid(radius):
    return 0.0289968*4*np.pi*Ggrav*R**3/radius**2

def g_lc(radius):
    return 0.00425234*4*np.pi*Ggrav*R**3/radius**2

def g_uc(radius):
    return 0.00488337*4*np.pi*Ggrav*R**3/radius**2

def g_o(radius):
    return 0.000480075*4*np.pi*Ggrav*R**3/radius**2

def prem_gravity(radius):

    if radius<1221.5e3:
       val=g_ic(radius)

    elif radius<=3480e3:
       val=g_ic(1221.5e3)+g_oc(radius)

    elif radius<=3630.e3:
       val=g_ic(1221.5e3)+g_oc(3480e3)+g_lm(radius)

    elif radius<=5600.e3:
       val=g_ic(1221.5e3)+g_oc(3480e3)+g_lm(3630e3)+g_tz1(radius)

    elif radius<=5701.e3:

       val=g_ic(1221.5e3)+g_oc(3480e3)+g_lm(3630e3)+g_tz1(6701e3)+g_tz2(radius)

    elif radius<=5771.e3:
       val=0
    elif radius<=5971.e3:
       val=0
    elif radius<=6151.e3:
       val=0
    elif radius<=6291.e3:
       val=0
    elif radius<=6346.e3:
       val=0
    elif radius<=6356.e3:
       val=0
    elif radius<=6368.e3:
       val=0
    else:
       val=0

    return val


#####################################################################

Ncell=1000

h=R/Ncell

r = np.empty(Ncell,dtype=np.float64) 
g = np.empty(Ncell,dtype=np.float64) 
counter=0
for i in range(0,Ncell):
    r[counter]=i*h+h/2.
    g[i]=prem_gravity(r[counter])
    counter+=1
#end for

np.savetxt('grav.ascii',np.array([r,g]).T,header='# x,g')

exit()

#####################################################################


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

