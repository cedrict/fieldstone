###############################################################################
#rho_model=0 # constant
#rho_model=1 # PREM
#rho_model=2 # ST105W
#rho_model=3 # AK135F
###############################################################################

from numba import jit
import numpy as np
import math as math

@jit(nopython=True)
def density(x,y,R1,R2,k,rho0,rho_model,exp,rhoblobstar,yblob,Rblob):

    #print(x,y,rho0,rho_model)

    if rho_model==0:
       val=rho0

    elif rho_model==1: #PREM
       radius=np.sqrt(x*x+y*y)
       xx=radius/6371.e3
       if radius>6371e3:
          densprem=0
       elif radius<=1221.5e3:
           densprem=13.0885-8.8381*xx**2
       elif radius<=3480e3:
           densprem=12.5815-1.2638*xx-3.6426*xx**2-5.5281*xx**3
       elif radius<=3630.e3:
          densprem=7.9565-6.4761*xx+5.5283*xx**2-3.0807*xx**3
       elif radius<=5600.e3:
          densprem=7.9565-6.4761*xx+5.5283*xx**2-3.0807*xx**3
       elif radius<=5701.e3:
          densprem=7.9565-6.4761*xx+5.5283*xx**2-3.0807*xx**3
       elif radius<=5771.e3:
          densprem=5.3197-1.4836*xx
       elif radius<=5971.e3:
          densprem=11.2494-8.0298*xx
       elif radius<=6151.e3:
          densprem=7.1089-3.8045*xx
       elif radius<=6291.e3:
          densprem=2.6910+0.6924*xx
       elif radius<=6346.e3:
          densprem=2.6910+0.6924*xx
       elif radius<=6356.e3:
          densprem=2.9
       elif radius<=6368.e3:
          densprem=2.6
       else:
          densprem=1.020
       val=densprem*1000

    #-------------------------------------
    elif rho_model==2: #ST105W
       depth=R2-np.sqrt(x**2+y**2)
       cell_index=49
       #for kk in range(0,50):
       #    if depth<depth_st105w[kk+1]:
       #       cell_index=kk
       #       break
       #    #end if
       #end for
       #val=(depth-depth_st105w[cell_index])/(depth_st105w[cell_index+1]-depth_st105w[cell_index])\
       #   *(rho_st105w[cell_index+1]-rho_st105w[cell_index])+rho_st105w[cell_index]

    #-------------------------------------
    elif rho_model==3: #AK135F
       depth=R2-np.sqrt(x**2+y**2)
       cell_index=144
       #for kk in range(0,145):
       #    if depth<depth_ak135f[kk+1]:
       #       cell_index=kk
       #       break
       #    #end if
       #end for
       #val=(depth-depth_ak135f[cell_index])/(depth_ak135f[cell_index+1]-depth_ak135f[cell_index])\
       #   *(rho_ak135f[cell_index+1]-rho_ak135f[cell_index])+rho_ak135f[cell_index]

    #-------------------------------------
    else:
       print('unknwon rho_model')

    #-------------------------------------
    if exp==0:
       r=np.sqrt(x*x+y*y)
       theta=math.atan2(y,x)
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       fr=A*r+B/r
       fpr=A-B/r**2
       gr=A/2.*r + B/r*math.log(r) - 1./r
       gpr=A/2.+B/r**2*(1.-math.log(r))+1./r**2
       gppr=-B/r**3*(3.-2.*math.log(r))-2./r**3
       alephr=gppr - gpr/r -gr/r**2*(k**2-1.) +fr/r**2  +fpr/r
       val=k*math.sin(k*theta)*alephr + rho0 
    elif exp==3:
       r=np.sqrt(x*x+y*y)
       theta=np.pi/2-np.arctan2(y,x)
       if theta<np.pi/8 and r>R1+3*(R2-R1)/8 and r<R1+5*(R2-R1)/8:
          val*=rhoblobstar
    else:
       if np.sqrt(x**2+(y-yblob)**2)<Rblob:
          val*=rhoblobstar
       #val-=rho0
       #val=abs(val)

    return val

###############################################################################
