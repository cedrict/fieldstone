import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg import *
import time as timing
from scipy import sparse
from numba import jit

###############################################################################

axisymmetric=False

surface_free_slip=False

###############################################################################

planet_is_Earth=False
use_rho_PREM=False
use_eta_civs12a=False
use_eta_civs12b=False
use_eta_stho08=False
use_eta_yohk01=True

planet_is_Mars=False
use_steinberger=True
use_samuelA=True
use_samuelB=True

###############################################################################

@jit(nopython=True)
def NNN(r,s,space):
    if space=='Q1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=0.25*(1.-r)*(1.-s)
       val[1]=0.25*(1.+r)*(1.-s)
       val[2]=0.25*(1.+r)*(1.+s)
       val[3]=0.25*(1.-r)*(1.+s)
    if space=='Q2':
       val = np.zeros(9,dtype=np.float64)
       val[0]= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       val[1]= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       val[2]= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       val[3]= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       val[4]=    (1.-r**2) * 0.5*s*(s-1.)
       val[5]= 0.5*r*(r+1.) *    (1.-s**2)
       val[6]=    (1.-r**2) * 0.5*s*(s+1.)
       val[7]= 0.5*r*(r-1.) *    (1.-s**2)
       val[8]=    (1.-r**2) *    (1.-s**2)
    if space=='Q3':
       val = np.zeros(16,dtype=np.float64)
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       N1t=(-1    +s +9*s**2 - 9*s**3)/16
       N2t=(+9 -27*s -9*s**2 +27*s**3)/16
       N3t=(+9 +27*s -9*s**2 -27*s**3)/16
       N4t=(-1    -s +9*s**2 + 9*s**3)/16
       val[ 0]= N1r*N1t ; val[ 1]= N2r*N1t ; val[ 2]= N3r*N1t ; val[ 3]= N4r*N1t
       val[ 4]= N1r*N2t ; val[ 5]= N2r*N2t ; val[ 6]= N3r*N2t ; val[ 7]= N4r*N2t
       val[ 8]= N1r*N3t ; val[ 9]= N2r*N3t ; val[10]= N3r*N3t ; val[11]= N4r*N3t
       val[12]= N1r*N4t ; val[13]= N2r*N4t ; val[14]= N3r*N4t ; val[15]= N4r*N4t
    if space=='Q4':
       val = np.zeros(25,dtype=np.float64)
       N1r=(    r -   r**2 -4*r**3 + 4*r**4)/6
       N2r=( -8*r +16*r**2 +8*r**3 -16*r**4)/6
       N3r=(1     - 5*r**2         + 4*r**4)
       N4r=(  8*r +16*r**2 -8*r**3 -16*r**4)/6
       N5r=(   -r -   r**2 +4*r**3 + 4*r**4)/6
       N1s=(    s -   s**2 -4*s**3 + 4*s**4)/6
       N2s=( -8*s +16*s**2 +8*s**3 -16*s**4)/6
       N3s=(1     - 5*s**2         + 4*s**4)
       N4s=(  8*s +16*s**2 -8*s**3 -16*s**4)/6
       N5s=(   -s -   s**2 +4*s**3 + 4*s**4)/6
       val[0] = N1r*N1s ; val[1] = N2r*N1s ; val[2] = N3r*N1s ; val[3] = N4r*N1s ; val[4] = N5r*N1s
       val[5] = N1r*N2s ; val[6] = N2r*N2s ; val[7] = N3r*N2s ; val[8] = N4r*N2s ; val[9] = N5r*N2s
       val[10]= N1r*N3s ; val[11]= N2r*N3s ; val[12]= N3r*N3s ; val[13]= N4r*N3s ; val[14]= N5r*N3s
       val[15]= N1r*N4s ; val[16]= N2r*N4s ; val[17]= N3r*N4s ; val[18]= N4r*N4s ; val[19]= N5r*N4s
       val[20]= N1r*N5s ; val[21]= N2r*N5s ; val[22]= N3r*N5s ; val[23]= N4r*N5s ; val[24]= N5r*N5s

    if space=='Q5':
       val = np.zeros(36,dtype=np.float64)
       N1r=-(625*r**5-625*r**4-250*r**3+250*r**2 +9*r-9)/768 
       N2r= (125*r**5 -75*r**4-130*r**3 +78*r**2 +5*r-3)*25/768
       N3r=-(125*r**5 -25*r**4-170*r**3 +34*r**2+45*r-9)*25/384
       N4r= (125*r**5 +25*r**4-170*r**3 -34*r**2+45*r+9)*25/384
       N5r=-(125*r**5 +75*r**4-130*r**3 -78*r**2 +5*r+3)*25/768
       N6r= (625*r**5+625*r**4-250*r**3-250*r**2 +9*r+9)/768 
       #print(N1r+N2r+N3r+N4r+N5r+N6r)

       N1s=-(625*s**5-625*s**4-250*s**3+250*s**2 +9*s-9)/768 
       N2s= (125*s**5 -75*s**4-130*s**3 +78*s**2 +5*s-3)*25/768
       N3s=-(125*s**5 -25*s**4-170*s**3 +34*s**2+45*s-9)*25/384
       N4s= (125*s**5 +25*s**4-170*s**3 -34*s**2+45*s+9)*25/384
       N5s=-(125*s**5 +75*s**4-130*s**3 -78*s**2 +5*s+3)*25/768
       N6s= (625*s**5+625*s**4-250*s**3-250*s**2 +9*s+9)/768 
       #print(N1s+N2s+N3s+N4s+N5s+N6s)

       val[0]=N1r*N1s  ; val[1]=N2r*N1s  ; val[2]=N3r*N1s  ; val[3]=N4r*N1s  ; val[4]=N5r*N1s  ; val[5]=N6r*N1s  
       val[6]=N1r*N2s  ; val[7]=N2r*N2s  ; val[8]=N3r*N2s  ; val[9]=N4r*N2s  ; val[10]=N5r*N2s ; val[11]=N6r*N2s 
       val[12]=N1r*N3s ; val[13]=N2r*N3s ; val[14]=N3r*N3s ; val[15]=N4r*N3s ; val[16]=N5r*N3s ; val[17]=N6r*N3s 
       val[18]=N1r*N4s ; val[19]=N2r*N4s ; val[20]=N3r*N4s ; val[21]=N4r*N4s ; val[22]=N5r*N4s ; val[23]=N6r*N4s 
       val[24]=N1r*N5s ; val[25]=N2r*N5s ; val[26]=N3r*N5s ; val[27]=N4r*N5s ; val[28]=N5r*N5s ; val[29]=N6r*N5s 
       val[30]=N1r*N6s ; val[31]=N2r*N6s ; val[32]=N3r*N6s ; val[33]=N4r*N6s ; val[34]=N5r*N6s ; val[35]=N6r*N6s 

    if space=='Q6':
       val = np.zeros(49,dtype=np.float64)
       N1r=  (81*r**6 - 81*r**5 -  45*r**4 + 45*r**3 +  4*r**2 - 4*r)/80    #ok 
       N2r= -(27*r**6 - 18*r**5 -  30*r**4 + 20*r**3 +  3*r**2 - 2*r)*9./40 #ok
       N3r=  (27*r**6 -  9*r**5 -  39*r**4 + 13*r**3 + 12*r**2 - 4*r)*9./16 #ok
       N4r= -(81*r**6           - 126*r**4           + 49*r**2       -4)/4  
       N5r=  (27*r**6 +  9*r**5 -  39*r**4 - 13*r**3 + 12*r**2 + 4*r)*9./16
       N6r= -(27*r**6 + 18*r**5 -  30*r**4 - 20*r**3 +  3*r**2 + 2*r)*9./40 
       N7r=  (81*r**6 + 81*r**5 -  45*r**4 - 45*r**3 +  4*r**2 + 4*r)/80
       #print(N1r+N2r+N3r+N4r+N5r+N6r+N7r)

       N1s=  (81*s**6 - 81*s**5 -  45*s**4 + 45*s**3 +  4*s**2 - 4*s)/80 
       N2s= -(27*s**6 - 18*s**5 -  30*s**4 + 20*s**3 +  3*s**2 - 2*s)*9/40
       N3s=  (27*s**6 -  9*s**5 -  39*s**4 + 13*s**3 + 12*s**2 - 4*s)*9/16
       N4s= -(81*s**6           - 126*s**4           + 49*s**2       -4)/4 
       N5s=  (27*s**6 +  9*s**5 -  39*s**4 - 13*s**3 + 12*s**2 + 4*s)*9/16
       N6s= -(27*s**6 + 18*s**5 -  30*s**4 - 20*s**3 +  3*s**2 + 2*s)*9/40 
       N7s=  (81*s**6 + 81*s**5 -  45*s**4 - 45*s**3 +  4*s**2 + 4*s)/80
       #print(N1s+N2s+N3s+N4s+N5s+N6s+N7s)

       val[0]=N1r*N1s  ; val[1]=N2r*N1s  ; val[2]=N3r*N1s  ; val[3]=N4r*N1s  ; val[4]=N5r*N1s  ; val[5]=N6r*N1s  ; val[6]=N7r*N1s
       val[7]=N1r*N2s  ; val[8]=N2r*N2s  ; val[9]=N3r*N2s  ; val[10]=N4r*N2s ; val[11]=N5r*N2s ; val[12]=N6r*N2s ; val[13]=N7r*N2s
       val[14]=N1r*N3s ; val[15]=N2r*N3s ; val[16]=N3r*N3s ; val[17]=N4r*N3s ; val[18]=N5r*N3s ; val[19]=N6r*N3s ; val[20]=N7r*N3s
       val[21]=N1r*N4s ; val[22]=N2r*N4s ; val[23]=N3r*N4s ; val[24]=N4r*N4s ; val[25]=N5r*N4s ; val[26]=N6r*N4s ; val[27]=N7r*N4s
       val[28]=N1r*N5s ; val[29]=N2r*N5s ; val[30]=N3r*N5s ; val[31]=N4r*N5s ; val[32]=N5r*N5s ; val[33]=N6r*N5s ; val[34]=N7r*N5s
       val[35]=N1r*N6s ; val[36]=N2r*N6s ; val[37]=N3r*N6s ; val[38]=N4r*N6s ; val[39]=N5r*N6s ; val[40]=N6r*N6s ; val[41]=N7r*N6s
       val[42]=N1r*N7s ; val[43]=N2r*N7s ; val[44]=N3r*N7s ; val[45]=N4r*N7s ; val[46]=N5r*N7s ; val[47]=N6r*N7s ; val[48]=N7r*N7s

    return val

@jit(nopython=True)
def dNNNdr(r,s,space):
   if space=='Q1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=-0.25*(1.-s)
       val[1]=+0.25*(1.-s)
       val[2]=+0.25*(1.+s)
       val[3]=-0.25*(1.+s)
   if space=='Q2':
       val = np.zeros(9,dtype=np.float64)
       val[0]= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       val[1]= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       val[2]= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       val[3]= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       val[4]=       (-2.*r) * 0.5*s*(s-1)
       val[5]= 0.5*(2.*r+1.) *   (1.-s**2)
       val[6]=       (-2.*r) * 0.5*s*(s+1)
       val[7]= 0.5*(2.*r-1.) *   (1.-s**2)
       val[8]=       (-2.*r) *   (1.-s**2)
   if space=='Q3':
       val=np.zeros(16,dtype=np.float64)
       dN1rdr=( +1 +18*r -27*r**2)/16
       dN2rdr=(-27 -18*r +81*r**2)/16
       dN3rdr=(+27 -18*r -81*r**2)/16
       dN4rdr=( -1 +18*r +27*r**2)/16
       N1s=(-1    +s +9*s**2 - 9*s**3)/16
       N2s=(+9 -27*s -9*s**2 +27*s**3)/16
       N3s=(+9 +27*s -9*s**2 -27*s**3)/16
       N4s=(-1    -s +9*s**2 + 9*s**3)/16
       val[ 0]=dN1rdr*N1s ; val[ 1]=dN2rdr*N1s ; val[ 2]=dN3rdr*N1s ; val[ 3]=dN4rdr*N1s
       val[ 4]=dN1rdr*N2s ; val[ 5]=dN2rdr*N2s ; val[ 6]=dN3rdr*N2s ; val[ 7]=dN4rdr*N2s
       val[ 8]=dN1rdr*N3s ; val[ 9]=dN2rdr*N3s ; val[10]=dN3rdr*N3s ; val[11]=dN4rdr*N3s
       val[12]=dN1rdr*N4s ; val[13]=dN2rdr*N4s ; val[14]=dN3rdr*N4s ; val[15]=dN4rdr*N4s
   if space=='Q4':
       val=np.zeros(25,dtype=np.float64)
       dN1dr=(    1 - 2*r -12*r**2 +16*r**3)/6
       dN2dr=(   -8 +32*r +24*r**2 -64*r**3)/6
       dN3dr=(      -10*r          +16*r**3)
       dN4dr=(  8   +32*r -24*r**2 -64*r**3)/6
       dN5dr=(   -1 - 2*r +12*r**2 +16*r**3)/6
       N1s=(    s -   s**2 -4*s**3 + 4*s**4)/6
       N2s=( -8*s +16*s**2 +8*s**3 -16*s**4)/6
       N3s=(1     - 5*s**2         + 4*s**4)
       N4s=(  8*s +16*s**2 -8*s**3 -16*s**4)/6
       N5s=(   -s -   s**2 +4*s**3 + 4*s**4)/6
       val[ 0]=dN1dr*N1s ; val[ 1]=dN2dr*N1s ; val[ 2]=dN3dr*N1s  
       val[ 3]=dN4dr*N1s ; val[ 4]=dN5dr*N1s ; val[ 5]=dN1dr*N2s  
       val[ 6]=dN2dr*N2s ; val[ 7]=dN3dr*N2s ; val[ 8]=dN4dr*N2s  
       val[ 9]=dN5dr*N2s ; val[10]=dN1dr*N3s ; val[11]=dN2dr*N3s 
       val[12]=dN3dr*N3s ; val[13]=dN4dr*N3s ; val[14]=dN5dr*N3s 
       val[15]=dN1dr*N4s ; val[16]=dN2dr*N4s ; val[17]=dN3dr*N4s  
       val[18]=dN4dr*N4s ; val[19]=dN5dr*N4s ; val[20]=dN1dr*N5s  
       val[21]=dN2dr*N5s ; val[22]=dN3dr*N5s ; val[23]=dN4dr*N5s  
       val[24]=dN5dr*N5s
   if space=='Q5':
       val = np.zeros(36,dtype=np.float64)

       N1r= (-3125*r**4+2500*r**3+750*r**2-500*r-9 )/768
       N2r= (  625*r**4- 300*r**3-390*r**2+156*r+5 )*25/768
       N3r=-(  625*r**4- 100*r**3-510*r**2+ 68*r+45)*25/384
       N4r= (  625*r**4+ 100*r**3-510*r**2- 68*r+45)*25/384 
       N5r=-(  625*r**4+ 300*r**3-390*r**2-156*r+5 )*25/768
       N6r= ( 3125*r**4+2500*r**3-750*r**2-500*r+9 )/768 

       N1s=-(625*s**5-625*s**4-250*s**3+250*s**2 +9*s-9)/768 
       N2s= (125*s**5 -75*s**4-130*s**3 +78*s**2 +5*s-3)*25/768
       N3s=-(125*s**5 -25*s**4-170*s**3 +34*s**2+45*s-9)*25/384
       N4s= (125*s**5 +25*s**4-170*s**3 -34*s**2+45*s+9)*25/384
       N5s=-(125*s**5 +75*s**4-130*s**3 -78*s**2 +5*s+3)*25/768
       N6s= (625*s**5+625*s**4-250*s**3-250*s**2 +9*s+9)/768 
       #print(N1s+N2s+N3s+N4s+N5s+N6s)

       val[0]=N1r*N1s  ; val[1]=N2r*N1s  ; val[2]=N3r*N1s  ; val[3]=N4r*N1s  ; val[4]=N5r*N1s  ; val[5]=N6r*N1s  
       val[6]=N1r*N2s  ; val[7]=N2r*N2s  ; val[8]=N3r*N2s  ; val[9]=N4r*N2s  ; val[10]=N5r*N2s ; val[11]=N6r*N2s 
       val[12]=N1r*N3s ; val[13]=N2r*N3s ; val[14]=N3r*N3s ; val[15]=N4r*N3s ; val[16]=N5r*N3s ; val[17]=N6r*N3s 
       val[18]=N1r*N4s ; val[19]=N2r*N4s ; val[20]=N3r*N4s ; val[21]=N4r*N4s ; val[22]=N5r*N4s ; val[23]=N6r*N4s 
       val[24]=N1r*N5s ; val[25]=N2r*N5s ; val[26]=N3r*N5s ; val[27]=N4r*N5s ; val[28]=N5r*N5s ; val[29]=N6r*N5s 
       val[30]=N1r*N6s ; val[31]=N2r*N6s ; val[32]=N3r*N6s ; val[33]=N4r*N6s ; val[34]=N5r*N6s ; val[35]=N6r*N6s 

   if space=='Q6':
       val = np.zeros(49,dtype=np.float64)

       N1r=  ( 486*r**5 -405*r**4 - 180*r**3 + 135*r**2 +  8*r -4)/80  #ok 
       N2r= -(  81*r**5 - 45*r**4 -  60*r**3 +  30*r**2 +  3*r -1)*9./20 #ok
       N3r=  ( 162*r**5 - 45*r**4 - 156*r**3 +  39*r**2 + 24*r -4)*9./16 #ok
       N4r=  (-243*r**5           + 252*r**3            - 49*r   )/2  #OK
       N5r=  ( 162*r**5 + 45*r**4 - 156*r**3 -  39*r**2 + 24*r +4)*9./16 #OK
       N6r= -(  81*r**5 + 45*r**4 -  60*r**3 -  30*r**2 +  3*r +1)*9./20 #ok
       N7r=  ( 486*r**5 +405*r**4 - 180*r**3 - 135*r**2 +  8*r +4)/80 #OK

       N1s=  (81*s**6 - 81*s**5 -  45*s**4 + 45*s**3 +  4*s**2 - 4*s)/80 
       N2s= -(27*s**6 - 18*s**5 -  30*s**4 + 20*s**3 +  3*s**2 - 2*s)*9./40
       N3s=  (27*s**6 -  9*s**5 -  39*s**4 + 13*s**3 + 12*s**2 - 4*s)*9./16
       N4s= -(81*s**6           - 126*s**4           + 49*s**2       -4)/4 
       N5s=  (27*s**6 +  9*s**5 -  39*s**4 - 13*s**3 + 12*s**2 + 4*s)*9./16
       N6s= -(27*s**6 + 18*s**5 -  30*s**4 - 20*s**3 +  3*s**2 + 2*s)*9./40 
       N7s=  (81*s**6 + 81*s**5 -  45*s**4 - 45*s**3 +  4*s**2 + 4*s)/80
       #print(N1s+N2s+N3s+N4s+N5s+N6s+N7s)
       val[0]=N1r*N1s  ; val[1]=N2r*N1s  ; val[2]=N3r*N1s  ; val[3]=N4r*N1s  ; val[4]=N5r*N1s  ; val[5]=N6r*N1s  ; val[6]=N7r*N1s
       val[7]=N1r*N2s  ; val[8]=N2r*N2s  ; val[9]=N3r*N2s  ; val[10]=N4r*N2s ; val[11]=N5r*N2s ; val[12]=N6r*N2s ; val[13]=N7r*N2s
       val[14]=N1r*N3s ; val[15]=N2r*N3s ; val[16]=N3r*N3s ; val[17]=N4r*N3s ; val[18]=N5r*N3s ; val[19]=N6r*N3s ; val[20]=N7r*N3s
       val[21]=N1r*N4s ; val[22]=N2r*N4s ; val[23]=N3r*N4s ; val[24]=N4r*N4s ; val[25]=N5r*N4s ; val[26]=N6r*N4s ; val[27]=N7r*N4s
       val[28]=N1r*N5s ; val[29]=N2r*N5s ; val[30]=N3r*N5s ; val[31]=N4r*N5s ; val[32]=N5r*N5s ; val[33]=N6r*N5s ; val[34]=N7r*N5s
       val[35]=N1r*N6s ; val[36]=N2r*N6s ; val[37]=N3r*N6s ; val[38]=N4r*N6s ; val[39]=N5r*N6s ; val[40]=N6r*N6s ; val[41]=N7r*N6s
       val[42]=N1r*N7s ; val[43]=N2r*N7s ; val[44]=N3r*N7s ; val[45]=N4r*N7s ; val[46]=N5r*N7s ; val[47]=N6r*N7s ; val[48]=N7r*N7s

   return val

@jit(nopython=True)
def dNNNds(r,s,space):
   if space=='Q1':
       val=np.zeros(4,dtype=np.float64)
       val[0]=-0.25*(1.-r)
       val[1]=-0.25*(1.+r)
       val[2]=+0.25*(1.+r)
       val[3]=+0.25*(1.-r)
   if space=='Q2':
       val=np.zeros(9,dtype=np.float64)
       val[0]= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       val[1]= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       val[2]= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       val[3]= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       val[4]=    (1.-r**2) * 0.5*(2.*s-1.)
       val[5]= 0.5*r*(r+1.) *       (-2.*s)
       val[6]=    (1.-r**2) * 0.5*(2.*s+1.)
       val[7]= 0.5*r*(r-1.) *       (-2.*s)
       val[8]=    (1.-r**2) *       (-2.*s)
   if space=='Q3':
       val=np.zeros(16,dtype=np.float64)
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       dN1sds=( +1 +18*s -27*s**2)/16
       dN2sds=(-27 -18*s +81*s**2)/16
       dN3sds=(+27 -18*s -81*s**2)/16
       dN4sds=( -1 +18*s +27*s**2)/16
       val[ 0]=N1r*dN1sds ; val[ 1]=N2r*dN1sds ; val[ 2]=N3r*dN1sds ; val[ 3]=N4r*dN1sds
       val[ 4]=N1r*dN2sds ; val[ 5]=N2r*dN2sds ; val[ 6]=N3r*dN2sds ; val[ 7]=N4r*dN2sds
       val[ 8]=N1r*dN3sds ; val[ 9]=N2r*dN3sds ; val[10]=N3r*dN3sds ; val[11]=N4r*dN3sds
       val[12]=N1r*dN4sds ; val[13]=N2r*dN4sds ; val[14]=N3r*dN4sds ; val[15]=N4r*dN4sds
   if space=='Q4':
       val=np.zeros(25,dtype=np.float64)
       N1r=(    r -   r**2 -4*r**3 + 4*r**4)/6
       N2r=( -8*r +16*r**2 +8*r**3 -16*r**4)/6
       N3r=(1     - 5*r**2         + 4*r**4)
       N4r=(  8*r +16*r**2 -8*r**3 -16*r**4)/6
       N5r=(   -r -   r**2 +4*r**3 + 4*r**4)/6
       dN1ds=(    1 - 2*s -12*s**2 +16*s**3)/6
       dN2ds=( -8*1 +32*s +24*s**2 -64*s**3)/6
       dN3ds=(      -10*s          +16*s**3)
       dN4ds=(  8   +32*s -24*s**2 -64*s**3)/6
       dN5ds=(   -1 - 2*s +12*s**2 +16*s**3)/6
       val[ 0]=N1r*dN1ds ; val[ 1]=N2r*dN1ds ; val[ 2]=N3r*dN1ds 
       val[ 3]=N4r*dN1ds ; val[ 4]=N5r*dN1ds ; val[ 5]=N1r*dN2ds 
       val[ 6]=N2r*dN2ds ; val[ 7]=N3r*dN2ds ; val[ 8]=N4r*dN2ds 
       val[ 9]=N5r*dN2ds ; val[10]=N1r*dN3ds ; val[11]=N2r*dN3ds 
       val[12]=N3r*dN3ds ; val[13]=N4r*dN3ds ; val[14]=N5r*dN3ds
       val[15]=N1r*dN4ds ; val[16]=N2r*dN4ds ; val[17]=N3r*dN4ds 
       val[18]=N4r*dN4ds ; val[19]=N5r*dN4ds ; val[20]=N1r*dN5ds 
       val[21]=N2r*dN5ds ; val[22]=N3r*dN5ds ; val[23]=N4r*dN5ds 
       val[24]=N5r*dN5ds
   if space=='Q5':
       val = np.zeros(36,dtype=np.float64)

       N1r=-(625*r**5-625*r**4-250*r**3+250*r**2 +9*r-9)/768 
       N2r= (125*r**5 -75*r**4-130*r**3 +78*r**2 +5*r-3)*25/768
       N3r=-(125*r**5 -25*r**4-170*r**3 +34*r**2+45*r-9)*25/384
       N4r= (125*r**5 +25*r**4-170*r**3 -34*r**2+45*r+9)*25/384
       N5r=-(125*r**5 +75*r**4-130*r**3 -78*r**2 +5*r+3)*25/768
       N6r= (625*r**5+625*r**4-250*r**3-250*r**2 +9*r+9)/768 
       #print(N1r+N2r+N3r+N4r+N5r+N6r)

       N1s= (-3125*s**4+2500*s**3+750*s**2-500*s-9 )/768
       N2s= (  625*s**4- 300*s**3-390*s**2+156*s+5 )*25/768
       N3s=-(  625*s**4- 100*s**3-510*s**2+ 68*s+45)*25/384
       N4s= (  625*s**4+ 100*s**3-510*s**2- 68*s+45)*25/384 
       N5s=-(  625*s**4+ 300*s**3-390*s**2-156*s+5 )*25/768
       N6s= ( 3125*s**4+2500*s**3-750*s**2-500*s+9 )/768 

       val[0]=N1r*N1s  ; val[1]=N2r*N1s  ; val[2]=N3r*N1s  ; val[3]=N4r*N1s  ; val[4]=N5r*N1s  ; val[5]=N6r*N1s  
       val[6]=N1r*N2s  ; val[7]=N2r*N2s  ; val[8]=N3r*N2s  ; val[9]=N4r*N2s  ; val[10]=N5r*N2s ; val[11]=N6r*N2s 
       val[12]=N1r*N3s ; val[13]=N2r*N3s ; val[14]=N3r*N3s ; val[15]=N4r*N3s ; val[16]=N5r*N3s ; val[17]=N6r*N3s 
       val[18]=N1r*N4s ; val[19]=N2r*N4s ; val[20]=N3r*N4s ; val[21]=N4r*N4s ; val[22]=N5r*N4s ; val[23]=N6r*N4s 
       val[24]=N1r*N5s ; val[25]=N2r*N5s ; val[26]=N3r*N5s ; val[27]=N4r*N5s ; val[28]=N5r*N5s ; val[29]=N6r*N5s 
       val[30]=N1r*N6s ; val[31]=N2r*N6s ; val[32]=N3r*N6s ; val[33]=N4r*N6s ; val[34]=N5r*N6s ; val[35]=N6r*N6s 

   if space=='Q6':
       val = np.zeros(49,dtype=np.float64)
       N1r=  (81*r**6 - 81*r**5 -  45*r**4 + 45*r**3 +  4*r**2 - 4*r)/80 
       N2r= -(27*r**6 - 18*r**5 -  30*r**4 + 20*r**3 +  3*r**2 - 2*r)*9./40
       N3r=  (27*r**6 -  9*r**5 -  39*r**4 + 13*r**3 + 12*r**2 - 4*r)*9./16
       N4r= -(81*r**6           - 126*r**4           + 49*r**2       -4)/4 
       N5r=  (27*r**6 +  9*r**5 -  39*r**4 - 13*r**3 + 12*r**2 + 4*r)*9./16
       N6r= -(27*r**6 + 18*r**5 -  30*r**4 - 20*r**3 +  3*r**2 + 2*r)*9./40 
       N7r=  (81*r**6 + 81*r**5 -  45*r**4 - 45*r**3 +  4*r**2 + 4*r)/80
       #print(N1r+N2r+N3r+N4r+N5r+N6r+N7r)
       N1s=  ( 486*s**5 -405*s**4 - 180*s**3 + 135*s**2 +  8*s -4)/80 
       N2s= -(  81*s**5 - 45*s**4 -  60*s**3 +  30*s**2 +  3*s -1)*9./20
       N3s=  ( 162*s**5 - 45*s**4 - 156*s**3 +  39*s**2 + 24*s -4)*9./16
       N4s=  (-243*s**5           + 252*s**3            - 49*s   )/2
       N5s=  ( 162*s**5 + 45*s**4 - 156*s**3 -  39*s**2 + 24*s +4)*9./16
       N6s= -(  81*s**5 + 45*s**4 -  60*s**3 -  30*s**2 +  3*s +1)*9./20
       N7s=  ( 486*s**5 +405*s**4 - 180*s**3 - 135*s**2 +  8*s +4)/80
       val[0]=N1r*N1s  ; val[1]=N2r*N1s  ; val[2]=N3r*N1s  ; val[3]=N4r*N1s  ; val[4]=N5r*N1s  ; val[5]=N6r*N1s  ; val[6]=N7r*N1s
       val[7]=N1r*N2s  ; val[8]=N2r*N2s  ; val[9]=N3r*N2s  ; val[10]=N4r*N2s ; val[11]=N5r*N2s ; val[12]=N6r*N2s ; val[13]=N7r*N2s
       val[14]=N1r*N3s ; val[15]=N2r*N3s ; val[16]=N3r*N3s ; val[17]=N4r*N3s ; val[18]=N5r*N3s ; val[19]=N6r*N3s ; val[20]=N7r*N3s
       val[21]=N1r*N4s ; val[22]=N2r*N4s ; val[23]=N3r*N4s ; val[24]=N4r*N4s ; val[25]=N5r*N4s ; val[26]=N6r*N4s ; val[27]=N7r*N4s
       val[28]=N1r*N5s ; val[29]=N2r*N5s ; val[30]=N3r*N5s ; val[31]=N4r*N5s ; val[32]=N5r*N5s ; val[33]=N6r*N5s ; val[34]=N7r*N5s
       val[35]=N1r*N6s ; val[36]=N2r*N6s ; val[37]=N3r*N6s ; val[38]=N4r*N6s ; val[39]=N5r*N6s ; val[40]=N6r*N6s ; val[41]=N7r*N6s
       val[42]=N1r*N7s ; val[43]=N2r*N7s ; val[44]=N3r*N7s ; val[45]=N4r*N7s ; val[46]=N5r*N7s ; val[47]=N6r*N7s ; val[48]=N7r*N7s

   return val

###############################################################################

@jit(nopython=True)
def density(x,y,R1,R2,k,rho0,g0):

    if use_rho_PREM:
       radius=np.sqrt(x*x+y*y)
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
       val=densprem*1000
    else:
       val=rho0


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

    return val

###############################################################################
###############################################################################

@jit(nopython=True)
def viscosity(x,y,R1,R2):
       
    depth=R2-np.sqrt(x**2+y**2)

    #--------------------------------------
    if planet_is_Earth and use_eta_civs12a: 
       cell_index=49
       for kk in range(0,50):
           if depth<depths_civs12[kk+1]:
              cell_index=kk
              break
           #end if
       #end for
       val=(depth-depths_civs12[cell_index])/(depths_civs12[cell_index+1]-depths_civs12[cell_index])\
          *(viscA_civs12[cell_index+1]-viscA_civs12[cell_index])+viscA_civs12[cell_index]
       val=10**val

    #--------------------------------------
    elif planet_is_Earth and use_eta_civs12b: 
       cell_index=49
       for kk in range(0,50):
           if depth<depths_civs12[kk+1]:
              cell_index=kk
              break
           #end if
       #end for
       val=(depth-depths_civs12[cell_index])/(depths_civs12[cell_index+1]-depths_civs12[cell_index])\
          *(viscB_civs12[cell_index+1]-viscB_civs12[cell_index])+viscB_civs12[cell_index]
       val=10**val

    #--------------------------------------
    elif planet_is_Earth and use_eta_stho08: 
       cell_index=21
       for kk in range(0,22):
           if depth<depths_stho08[kk+1]:
              cell_index=kk
              break
           #end if
       #end for
       val=(depth-depths_stho08[cell_index])/(depths_stho08[cell_index+1]-depths_stho08[cell_index])\
          *(visc_stho08[cell_index+1]-visc_stho08[cell_index])+visc_stho08[cell_index]

    #--------------------------------------
    elif planet_is_Earth and use_eta_yohk01: 
       val = 3e21
       if depth < 150e3:
          val *= 1e3 
       elif depth > 670e3:
          val *= 70

    else:
       val=eta_m

    return val       


###############################################################################
###############################################################################

@jit(nopython=True)
def velocity_x(x,y,R1,R2,k):
    if exp==0:
       r=np.sqrt(x*x+y*y)
       theta=math.atan2(y,x)
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       fr=A*r+B/r
       fpr=A-B/r**2
       gr=A/2.*r + B/r*math.log(r) - 1./r
       hr=(2*gr-fr)/r
       vr=k *gr * math.sin (k * theta)
       vtheta = fr *math.cos(k* theta)
       val=vr*math.cos(theta)-vtheta*math.sin(theta)
    else:
       val=0
    return val

@jit(nopython=True)
def velocity_y(x,y,R1,R2,k):
    if exp==0:
       r=np.sqrt(x*x+y*y)
       theta=math.atan2(y,x)
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       fr=A*r+B/r
       fpr=A-B/r**2
       gr=A/2.*r + B/r*math.log(r) - 1./r
       hr=(2*gr-fr)/r
       vr=k *gr * math.sin (k * theta)
       vtheta = fr *math.cos(k* theta)
       val=vr*math.sin(theta)+vtheta*math.cos(theta)
    else:
       val=0
    return val

@jit(nopython=True)
def pressure(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    if exp==0:
       theta=math.atan2(y,x)
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       fr=A*r+B/r
       gr=A/2.*r + B/r*math.log(r) - 1./r
       hr=(2*gr-fr)/r
       val=k*hr*math.sin(k*theta) + rho0*g0*(r-R2)
    else: 
       val=rho0*g0*(R2-r)
    return val

@jit(nopython=True)
def sr_xx(x,y,R1,R2,k):
    if exp==0:
       r=np.sqrt(x*x+y*y)
       theta=math.atan2(y,x)
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       gr=A/2.*r + B/r*math.log(r) - 1./r
       gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
       fr=A*r+B/r
       fpr=A-B/r**2
       err=gpr*k*math.sin(k*theta)
       ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
       ett=(gr-fr)/r*k*math.sin(k*theta)
       val=err*(math.cos(theta))**2\
          +ett*(math.sin(theta))**2\
          -2*ert*math.sin(theta)*math.cos(theta)
    else: 
       val=0
    return val

@jit(nopython=True)
def sr_yy(x,y,R1,R2,k):
    if exp==0:
       r=np.sqrt(x*x+y*y)
       theta=math.atan2(y,x)
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       gr=A/2.*r + B/r*math.log(r) - 1./r
       gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
       fr=A*r+B/r
       fpr=A-B/r**2
       err=gpr*k*math.sin(k*theta)
       ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
       ett=(gr-fr)/r*k*math.sin(k*theta)
       val=err*(math.sin(theta))**2\
          +ett*(math.cos(theta))**2\
          +2*ert*math.sin(theta)*math.cos(theta)
    else: 
       val=0
    return val

@jit(nopython=True)
def sr_xy(x,y,R1,R2,k):
    if exp==0:
       r=np.sqrt(x*x+y*y)
       theta=math.atan2(y,x)
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       gr=A/2.*r + B/r*math.log(r) - 1./r
       gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
       fr=A*r+B/r
       fpr=A-B/r**2
       err=gpr*k*math.sin(k*theta)
       ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
       ett=(gr-fr)/r*k*math.sin(k*theta)
       val=ert*(math.cos(theta)**2-math.sin(theta)**2)\
          +(err-ett)*math.cos(theta)*math.sin(theta)
    else: 
       val=0
    return val

###############################################################################

@jit(nopython=True)
def gx(x,y,g0):
    val= -x/np.sqrt(x*x+y*y)*g0
    return val

@jit(nopython=True)
def gy(x,y,g0):
    val= -y/np.sqrt(x*x+y*y)*g0
    return val

###############################################################################
      

print("-----------------------------")
print("--------stone 152------------")
print("-----------------------------")

ndim=2   # number of dimensions
mV=9     # number of nodes making up an element
mP=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

stop_here=False

if int(len(sys.argv) == 12):
   exp         = int(sys.argv[1])
   nelr        = int(sys.argv[2])
   visu        = int(sys.argv[3])
   nqperdim    = int(sys.argv[4])
   mapping     = int(sys.argv[5])
   xi          = int(sys.argv[6])
   etablobstar = float(sys.argv[7])
   rhoblobstar = float(sys.argv[8])
   yblob       = float(sys.argv[9])
   Rblob       = float(sys.argv[10])
   etalithstar = float(sys.argv[11])

   if mapping==1: mapping='Q1'
   if mapping==2: mapping='Q2'
   if mapping==3: mapping='Q3'
   if mapping==4: mapping='Q4'
   if mapping==5: mapping='Q5'
   if mapping==6: mapping='Q6'

   if exp==-1:
      exp=1
      stop_here=True

   print(sys.argv)

else:
   # exp=0: cyl benchmark
   # exp=1: aquarium
   # exp=2: blob
   # exp=3: pancake pi/8
   exp         = 0
   nelr        = 16 # Q1 cells!
   visu        = 1
   nqperdim    = 3
   mapping     = 'Q2' 
   xi          = 4
   etablobstar = 1
   rhoblobstar = 1 #.99
   yblob       = 4900e3
   Rblob       = 400e3
   etalithstar = 100

normal_type=1

if exp==0:
   R1=1.
   R2=2.
   rho_m=0.
   kk=4
   g0=1.
   eta_ref=1
   eta_m=1
   vel_unit=1
   velunit=' '
else:
   R1=3400e3
   R2=6400e3
   kk=0
   g0=10.
   eta_ref=1e21
   eta_m=1e21
   rho_m=4000.
   vel_unit=0.01/365.25/3600/24
   velunit='cm/year'

eps=1.e-10

rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]

debug=False

compute_sr1=False
compute_sr3=False

###############################################################################
if planet_is_Earth:
   R1=3480e3
   R2=6371e3
   g0=9.81

   print('max depth',R2-R1)

   #--------------------------------------
   if use_eta_civs12a or use_eta_civs12b :
      # reading data from civs12
      # file is 153 lines long 
      # first 51 lines are viscA, then 51 lines are viscB 
      # and last 51 lines are depths, from 0 to 2900km 
      # I have removed all ",&"

      viscA_civs12 = np.empty(51,dtype=np.float64)
      viscB_civs12 = np.empty(51,dtype=np.float64)
      depths_civs12 = np.empty(51,dtype=np.float64)

      f = open('DATA/EARTH/eta_civs12.ascii','r')
      counter=0
      for line in f:
          line=line.strip()
          columns=line.split()
          if counter<51:
             viscA_civs12[counter]=columns[0]
          elif counter<102:
             viscB_civs12[counter-51]=columns[0]
          else:
             depths_civs12[counter-102]=columns[0]
          counter+=1

      depths_civs12[:]=np.flip(depths_civs12)
      viscA_civs12[:]=np.flip(viscA_civs12)
      viscB_civs12[:]=np.flip(viscB_civs12)

      np.savetxt('civs12.ascii',np.array([depths_civs12,viscA_civs12,viscB_civs12]).T)

      print('read eta_civs12.ascii ok') 

   #-----------------
   if use_eta_stho08:
      # reading data from  Steinberger & Holmes 2008
      # file counts 22 lines
      # first column is number between 0 and 1 (normalised radii)
      # second column is viscosity
      # I have added a last line R=1, eta=1e24

      depths_stho08 = np.empty(23,dtype=np.float64)
      visc_stho08 = np.empty(23,dtype=np.float64)
      f = open('DATA/EARTH/eta_stho08.ascii','r')
      counter=0
      for line in f:
          line=line.strip()
          columns=line.split()
          depths_stho08[counter]=columns[0]
          visc_stho08[counter]=columns[1]
          counter+=1

      depths_stho08[:]=6371e3*(1-depths_stho08[:])
      depths_stho08[:]=np.flip(depths_stho08)
      visc_stho08[:]=np.flip(visc_stho08)

      np.savetxt('stho08.ascii',np.array([depths_stho08,visc_stho08]).T)

      print(depths_stho08[22]-depths_stho08[0])

      print('read eta_stho08.ascii ok')



###############################################################################
if planet_is_Mars:
   R1=1700e3
   R2=3390e3
   g0=3.71 #https://en.wikipedia.org/wiki/Mars   

   if use_steinberger:
      R1=R2-1967e3
      R_disc1 = R_outer-49.5e3
      R_disc2 = R_outer-1111.5e3
      R_disc3 = R_outer-1160e3
      R_disc4 = R_outer-1951.5e3
      eta_max=1e25

   if use_samuelA:
      R1=1839.5976879540331e3
      R_disc1 = 3317.7417442558781e3
      R_disc2 = 2836.6008937146739e3
      R_disc3 = 2350.4998282194360e3
      R_disc4 = 1918.9611272185618e3
      eta_max=1e25

   if use_samuelB:
      R1=1624.2975658322634e3
      R_disc1 = 3324.3388640802909e3
      R_disc2 = 3090.3851276356227e3
      R_disc3 = 2313.0549710614014e3
      R_disc4 = 1822.5068139999998e3
      eta_max=1e25





###############################################################################

if nqperdim==2:
   coords=[-1/np.sqrt(3),1/np.sqrt(3)]
   weights=[1,1]

if nqperdim==3:
   coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   weights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   coords=[-qc4a,-qc4b,qc4b,qc4a]
   weights=[qw4a,qw4b,qw4b,qw4a]

if nqperdim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   coords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   weights=[qw5a,qw5b,qw5c,qw5b,qw5a]

if nqperdim==6:
   coords=[-0.932469514203152,\
           -0.661209386466265,\
           -0.238619186083197,\
           +0.238619186083197,\
           +0.661209386466265,\
           +0.932469514203152]
   weights=[0.171324492379170,\
            0.360761573048139,\
            0.467913934572691,\
            0.467913934572691,\
            0.360761573048139,\
            0.171324492379170]
if nqperdim==7:
   coords=[-0.949107912342759,\
           -0.741531185599394,\
           -0.405845151377397,\
            0.000000000000000,\
            0.405845151377397,\
            0.741531185599394,\
            0.949107912342759]
   weights=[0.129484966168870,\
            0.279705391489277,\
            0.381830050505119,\
            0.417959183673469,\
            0.381830050505119,\
            0.279705391489277,\
            0.129484966168870]

nqel=nqperdim**2
qcoords_r=np.empty(nqel,dtype=np.float64)
qcoords_s=np.empty(nqel,dtype=np.float64)
qweights=np.empty(nqel,dtype=np.float64)

counterq=0
for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):
        qcoords_r[counterq]=coords[iq]
        qcoords_s[counterq]=coords[jq]
        qweights[counterq]=weights[iq]*weights[jq]
        counterq+=1
    #end for
#end for

###############################################################################
# grid point setup
###############################################################################
start = timing.time()

if not axisymmetric:

   nelt=xi*nelr 
   nel=nelr*nelt  
   nnr=nelr+1
   nnt=nelt
   NV=nnr*nnt  # number of V nodes

   xV=np.zeros(NV,dtype=np.float64) 
   yV=np.zeros(NV,dtype=np.float64) 
   rad=np.zeros(NV,dtype=np.float64)  
   theta=np.zeros(NV,dtype=np.float64) 

   Louter=2.*math.pi*R2
   Lr=R2-R1
   sx=Louter/float(nelt)
   sz=Lr/float(nelr)

   counter=0
   for j in range(0,nnr):
       for i in range(0,nelt):
           xV[counter]=i*sx
           yV[counter]=j*sz
           counter += 1

   counter=0
   for j in range(0,nnr):
       for i in range(0,nnt):
           xi=xV[counter]
           yi=yV[counter]
           t=xi/Louter*2.*math.pi    
           xV[counter]=math.cos(t)*(R1+yi)
           yV[counter]=math.sin(t)*(R1+yi)
           rad[counter]=R1+yi
           #theta[counter]=np.arctan2(yV[counter],xV[counter])
           theta[counter]=np.pi/2-np.arctan2(yV[counter],xV[counter])
           if theta[counter]<0.:
              theta[counter]+=2.*math.pi
           counter+=1

else:

   nelt=xi*nelr 
   nel=nelr*nelt  
   nnr=nelr+1
   nnt=nelt+1
   NV=nnr*nnt  # number of V nodes

   xV=np.zeros(NV,dtype=np.float64) 
   yV=np.zeros(NV,dtype=np.float64) 
   rad=np.zeros(NV,dtype=np.float64)  
   theta=np.zeros(NV,dtype=np.float64) 

   Louter=math.pi*R2
   Lr=R2-R1
   sx=Louter/float(nelt)
   sz=Lr/float(nelr)

   counter=0
   for j in range(0,nnr):
       for i in range(0,nnt):
           xV[counter]=i*sx
           yV[counter]=j*sz
           counter += 1

   counter=0
   for j in range(0,nnr):
       for i in range(0,nnt):
           x_i=xV[counter]
           yi=yV[counter]
           t=math.pi/2-x_i/Louter*math.pi 
           xV[counter]=math.cos(t)*(R1+yi)
           yV[counter]=math.sin(t)*(R1+yi)
           rad[counter]=R1+yi
           theta[counter]=np.pi/2-np.arctan2(yV[counter],xV[counter])
           if i==0:
              theta[counter]=0
              xV[counter]=0
           if i==nnt-1:
              theta[counter]=np.pi
              xV[counter]=0
           counter+=1

   if debug:
      np.savetxt('grid.ascii',np.array([xV,yV,theta]).T,header='# x,y')

print("coordinate arrays (%.3fs)" % (timing.time() - start))

###############################################################################
# build iconQ1 array needed for vtu file
###############################################################################

iconQ1 =np.zeros((4,nel),dtype=np.int32)

if not axisymmetric:

   counter = 0
   for j in range(0, nelr):
       for i in range(0, nelt):
           icon1=counter
           icon2=counter+1
           icon3=i+(j+1)*nelt+1
           icon4=i+(j+1)*nelt
           if i==nelt-1:
              icon2-=nelt
              icon3-=nelt
           iconQ1[0,counter] = icon2 
           iconQ1[1,counter] = icon1
           iconQ1[2,counter] = icon4
           iconQ1[3,counter] = icon3
           counter += 1
       #end for
   #end for

else:

   counter = 0
   for j in range(0,nelr):
       for i in range(0,nelt):
           iconQ1[0,counter] = i + j * (nelt + 1)
           iconQ1[1,counter] = i + 1 + j * (nelt + 1)
           iconQ1[2,counter] = i + 1 + (j + 1) * (nelt + 1)
           iconQ1[3,counter] = i + (j + 1) * (nelt + 1)
           counter += 1


if debug:
   vtufile=open("mesh_Q1.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

###############################################################################
# now that the grid has been built as if it was a Q1 grid, 
# we can simply use these same points to arrive at a Q2 
# connectivity array with 4 times less elements.
###############################################################################

nelr=nelr//2
nelt=nelt//2
nel=nel//4

if not axisymmetric:
   NP=nelt*(nelr+1)
else:
   NP=(nelt+1)*(nelr+1)

NfemV=NV*ndofV   # Total number of degrees of V freedom 
NfemP=NP*ndofP   # Total number of degrees of P freedom
Nfem=NfemV+NfemP # total number of dofs

print('nelr=',nelr)
print('nelt=',nelt)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('Nfem=',Nfem)
print('nqel=',nqel)
print('mapping=',mapping)
print('axisymmetric=',axisymmetric)
print('exp=',exp)
print('xi=',xi)
print("-----------------------------")

h_r=(R2-R1)/nelr

###############################################################################
# connectivity
###############################################################################
start = timing.time()

iconV =np.zeros((mV,nel),dtype=np.int32)
iconP =np.zeros((mP,nel),dtype=np.int32)

if not axisymmetric:

   counter = 0
   for j in range(0, nelr):
       for i in range(0, nelt):
           iconV[0,counter]=2*counter+2 +2*j*nelt
           iconV[1,counter]=2*counter   +2*j*nelt
           iconV[2,counter]=iconV[1,counter]+4*nelt
           iconV[3,counter]=iconV[1,counter]+4*nelt+2
           iconV[4,counter]=iconV[0,counter]-1
           iconV[5,counter]=iconV[1,counter]+2*nelt
           iconV[6,counter]=iconV[2,counter]+1
           iconV[7,counter]=iconV[5,counter]+2
           iconV[8,counter]=iconV[5,counter]+1
           if i==nelt-1:
              iconV[0,counter]-=2*nelt
              iconV[7,counter]-=2*nelt
              iconV[3,counter]-=2*nelt
           #print(j,i,counter,'|',iconV[0:mV,counter])
           counter += 1
       #end for
   #end for

   counter = 0
   for j in range(0, nelr):
       for i in range(0, nelt):
           icon1=counter
           icon2=counter+1
           icon3=i+(j+1)*nelt+1
           icon4=i+(j+1)*nelt
           if i==nelt-1:
              icon2-=nelt
              icon3-=nelt
           iconP[0,counter] = icon2 
           iconP[1,counter] = icon1
           iconP[2,counter] = icon4
           iconP[3,counter] = icon3
           counter += 1
       #end for
   #end for

else:

   counter = 0
   for j in range(0,nelr):
       for i in range(0,nelt):
           iconV[0,counter]=(i)*2+1+(j)*2*nnt -1
           iconV[1,counter]=(i)*2+3+(j)*2*nnt -1
           iconV[2,counter]=(i)*2+3+(j)*2*nnt+nnt*2 -1
           iconV[3,counter]=(i)*2+1+(j)*2*nnt+nnt*2 -1
           iconV[4,counter]=(i)*2+2+(j)*2*nnt -1
           iconV[5,counter]=(i)*2+3+(j)*2*nnt+nnt -1
           iconV[6,counter]=(i)*2+2+(j)*2*nnt+nnt*2 -1
           iconV[7,counter]=(i)*2+1+(j)*2*nnt+nnt -1
           iconV[8,counter]=(i)*2+2+(j)*2*nnt+nnt -1
           counter += 1
       #end for
   #end for

   counter = 0
   for j in range(0,nelr):
       for i in range(0,nelt):
           iconP[0,counter]=i+j*(nelt+1)
           iconP[1,counter]=i+1+j*(nelt+1)
           iconP[2,counter]=i+1+(j+1)*(nelt+1)
           iconP[3,counter]=i+(j+1)*(nelt+1)
           counter += 1
       #end for
   #end for

print("connectivity array (%.3fs)" % (timing.time() - start))

###############################################################################

#for iel in range(0,nel):
#    r0=rad[iconV[0,iel]]
#    r3=rad[iconV[3,iel]]
#    r8=rad[iconV[8,iel]]
    #center 0
    #xV[iconV[8,iel]]=0.25*(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]]+xV[iconV[3,iel]])
    #yV[iconV[8,iel]]=0.25*(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]]+yV[iconV[3,iel]])
    #center 1
    #xV[iconV[8,iel]]=0.125*(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]]+xV[iconV[3,iel]]+\
    #                        xV[iconV[4,iel]]+xV[iconV[5,iel]]+xV[iconV[6,iel]]+xV[iconV[7,iel]])
    #yV[iconV[8,iel]]=0.125*(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]]+yV[iconV[3,iel]]+\
    #                        yV[iconV[4,iel]]+yV[iconV[5,iel]]+yV[iconV[6,iel]]+yV[iconV[7,iel]])
    #center 2
    #xV[iconV[8,iel]]=0.0625*(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]]+xV[iconV[3,iel]]+\
    #                         3*xV[iconV[4,iel]]+3*xV[iconV[5,iel]]+3*xV[iconV[6,iel]]+3*xV[iconV[7,iel]])
    #yV[iconV[8,iel]]=0.0625*(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]]+yV[iconV[3,iel]]+\
    #                         3*yV[iconV[4,iel]]+3*yV[iconV[5,iel]]+3*yV[iconV[6,iel]]+3*yV[iconV[7,iel]])


###############################################################################
#now that I have both connectivity arrays I can easily build xP,yP
###############################################################################

xP=np.empty(NP,dtype=np.float64)  # x coordinates
yP=np.empty(NP,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xP[iconP[0,iel]]=xV[iconV[0,iel]]
    xP[iconP[1,iel]]=xV[iconV[1,iel]]
    xP[iconP[2,iel]]=xV[iconV[2,iel]]
    xP[iconP[3,iel]]=xV[iconV[3,iel]]
    yP[iconP[0,iel]]=yV[iconV[0,iel]]
    yP[iconP[1,iel]]=yV[iconV[1,iel]]
    yP[iconP[2,iel]]=yV[iconV[2,iel]]
    yP[iconP[3,iel]]=yV[iconV[3,iel]]

print("connectivity array (%.3fs)" % (timing.time() - start))

###############################################################################
# Q1 nodes for mapping are corners of Q2 basis functions
# Q2 nodes for mapping are same as Q2 basis functions
# Q3 nodes for mapping are built
# Q4 nodes for mapping are built
###############################################################################
start = timing.time()

if mapping=='Q1':
   xmapping=np.zeros((4,nel),dtype=np.float64)
   ymapping=np.zeros((4,nel),dtype=np.float64)
   for iel in range(0,nel):
       xmapping[0,iel]=xV[iconV[0,iel]] ; ymapping[0,iel]=yV[iconV[0,iel]]
       xmapping[1,iel]=xV[iconV[1,iel]] ; ymapping[1,iel]=yV[iconV[1,iel]]
       xmapping[2,iel]=xV[iconV[2,iel]] ; ymapping[2,iel]=yV[iconV[2,iel]]
       xmapping[3,iel]=xV[iconV[3,iel]] ; ymapping[3,iel]=yV[iconV[3,iel]]

if mapping=='Q2':
   xmapping=np.zeros((9,nel),dtype=np.float64)
   ymapping=np.zeros((9,nel),dtype=np.float64)
   for iel in range(0,nel):
       xmapping[0,iel]=xV[iconV[0,iel]] ; ymapping[0,iel]=yV[iconV[0,iel]]
       xmapping[1,iel]=xV[iconV[1,iel]] ; ymapping[1,iel]=yV[iconV[1,iel]]
       xmapping[2,iel]=xV[iconV[2,iel]] ; ymapping[2,iel]=yV[iconV[2,iel]]
       xmapping[3,iel]=xV[iconV[3,iel]] ; ymapping[3,iel]=yV[iconV[3,iel]]
       xmapping[4,iel]=xV[iconV[4,iel]] ; ymapping[4,iel]=yV[iconV[4,iel]]
       xmapping[5,iel]=xV[iconV[5,iel]] ; ymapping[5,iel]=yV[iconV[5,iel]]
       xmapping[6,iel]=xV[iconV[6,iel]] ; ymapping[6,iel]=yV[iconV[6,iel]]
       xmapping[7,iel]=xV[iconV[7,iel]] ; ymapping[7,iel]=yV[iconV[7,iel]]
       xmapping[8,iel]=xV[iconV[8,iel]] ; ymapping[8,iel]=yV[iconV[8,iel]]

if mapping=='Q3':
   if not axisymmetric:
      dtheta=2*np.pi/nelt/3
   else:
      dtheta=np.pi/nelt/3
   xmapping=np.zeros((16,nel),dtype=np.float64)
   ymapping=np.zeros((16,nel),dtype=np.float64)
   for iel in range(0,nel):
       thetamin=theta[iconV[0,iel]]
       rmin=rad[iconV[0,iel]]
       rmax=rad[iconV[2,iel]]
       counter=0
       for j in range(0,4):
           for i in range(0,4):
               ttt=thetamin+i*dtheta
               rrr=rmin+j*(rmax-rmin)/3
               xmapping[counter,iel]=math.sin(ttt)*rrr
               ymapping[counter,iel]=math.cos(ttt)*rrr
               #print(xmapping[counter,iel],ymapping[counter,iel])
               counter+=1


if mapping=='Q4':
   if not axisymmetric:
      dtheta=2*np.pi/nelt/4
   else:
      dtheta=np.pi/nelt/4
   xmapping=np.zeros((25,nel),dtype=np.float64)
   ymapping=np.zeros((25,nel),dtype=np.float64)
   for iel in range(0,nel):
       thetamin=theta[iconV[0,iel]]
       rmin=rad[iconV[0,iel]]
       rmax=rad[iconV[2,iel]]
       counter=0
       for j in range(0,5):
           for i in range(0,5):
               ttt=thetamin+i*dtheta
               rrr=rmin+j*(rmax-rmin)/4
               xmapping[counter,iel]=math.sin(ttt)*rrr
               ymapping[counter,iel]=math.cos(ttt)*rrr
               #print(xmapping[counter,iel],ymapping[counter,iel])
               counter+=1

if mapping=='Q5':
   if not axisymmetric:
      dtheta=2*np.pi/nelt/5
   else:
      dtheta=np.pi/nelt/5
   xmapping=np.zeros((36,nel),dtype=np.float64)
   ymapping=np.zeros((36,nel),dtype=np.float64)
   for iel in range(0,nel):
       thetamin=theta[iconV[0,iel]]
       rmin=rad[iconV[0,iel]]
       rmax=rad[iconV[2,iel]]
       counter=0
       for j in range(0,6):
           for i in range(0,6):
               ttt=thetamin+i*dtheta
               rrr=rmin+j*(rmax-rmin)/5
               xmapping[counter,iel]=math.sin(ttt)*rrr
               ymapping[counter,iel]=math.cos(ttt)*rrr
               #print(xmapping[counter,iel],ymapping[counter,iel])
               counter+=1

if mapping=='Q6':
   if not axisymmetric:
      dtheta=2*np.pi/nelt/6
   else:
      dtheta=np.pi/nelt/6
   xmapping=np.zeros((49,nel),dtype=np.float64)
   ymapping=np.zeros((49,nel),dtype=np.float64)
   for iel in range(0,nel):
       thetamin=theta[iconV[0,iel]]
       rmin=rad[iconV[0,iel]]
       rmax=rad[iconV[2,iel]]
       counter=0
       for j in range(0,7):
           for i in range(0,7):
               ttt=thetamin+i*dtheta
               rrr=rmin+j*(rmax-rmin)/6
               xmapping[counter,iel]=math.sin(ttt)*rrr
               ymapping[counter,iel]=math.cos(ttt)*rrr
               #print(xmapping[counter,iel],ymapping[counter,iel])
               counter+=1
if debug:
   np.savetxt('xymapping'+mapping+'.ascii',np.array([xmapping[0,:],ymapping[0,:]]).T)

print("define mapping (%.3fs)" % (timing.time() - start))

###############################################################################
# find out nodes on hull
###############################################################################
start = timing.time()

hull=np.zeros(NV,dtype=bool) 

for i in range(0,NV):
    if rad[i]/R1<1+eps:
       hull[i]=True
    if rad[i]/R2>1-eps:
       hull[i]=True
    if axisymmetric and xV[i]/R2<eps:
       hull[i]=True

print("flag nodes on hull (%.3fs)" % (timing.time() - start))

###############################################################################
# flag surface nodes and elements
###############################################################################
start = timing.time()

surfaceV=np.zeros(NV,dtype=bool) 
surface_element=np.zeros(nel,dtype=bool) 
cmbV=np.zeros(NV,dtype=bool) 
cmb_element=np.zeros(nel,dtype=bool) 

for i in range(0,NV):
    if rad[i]/R2>1-eps:
       surfaceV[i]=True
    if rad[i]/R1<1+eps:
       cmbV[i]=True

for iel in range(0,nel):
    if surfaceV[iconV[2,iel]]:
       surface_element[iel]=True 
    if cmbV[iconV[0,iel]]:
       cmb_element[iel]=True 

print("flag surf and cmb nodes and elts (%.3fs)" % (timing.time() - start))

###############################################################################
# compute normal vectors
###############################################################################
start = timing.time()

nx1=np.zeros(NV,dtype=np.float64) 
ny1=np.zeros(NV,dtype=np.float64) 
nx2=np.zeros(NV,dtype=np.float64) 
ny2=np.zeros(NV,dtype=np.float64) 
   

if normal_type==1:

   #compute normal 1 type
   for i in range(0,NV):
       if axisymmetric and xV[i]/R2<eps:
          nx1[i]=-1
          ny1[i]=0
       if rad[i]/R1<1+eps:
          nx1[i]=-np.sin(theta[i])
          ny1[i]=-np.cos(theta[i])
       if rad[i]/R2>1-eps:
          nx1[i]=np.sin(theta[i])
          ny1[i]=np.cos(theta[i])

   np.savetxt('normal1a.ascii',np.array([xV[surfaceV],yV[surfaceV],\
                                      nx1[surfaceV],ny1[surfaceV],theta[surfaceV]]).T)

if normal_type==2:
   #compute normal 2 type
   dNNNVdx=np.zeros(mV,dtype=np.float64) 
   dNNNVdy=np.zeros(mV,dtype=np.float64) 
   jcb=np.zeros((2,2),dtype=np.float64)
   for iel in range(0,nel):
       if True: #hull[iconV[0,iel]] or hull[iconV[2,iel]]: 
          for kq in range(0,nqel):
              rq=qcoords_r[kq]
              sq=qcoords_s[kq]
              weightq=qweights[kq]
              #compute jacobian matrix
              dNNNVdr=dNNNdr(rq,sq,mapping)
              dNNNVds=dNNNds(rq,sq,mapping)
              jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
              jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
              jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
              jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
              jcob=np.linalg.det(jcb)
              jcbi=np.linalg.inv(jcb)
              #basis functions
              dNNNVdr=dNNNdr(rq,sq,'Q2')
              dNNNVds=dNNNds(rq,sq,'Q2')
              # compute dNdx & dNdy
              for k in range(0,mV):
                  dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                  dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
              #end for 
              nx2[iconV[0,iel]]+=dNNNVdx[0]*jcob*weightq
              ny2[iconV[0,iel]]+=dNNNVdy[0]*jcob*weightq
              nx2[iconV[1,iel]]+=dNNNVdx[1]*jcob*weightq
              ny2[iconV[1,iel]]+=dNNNVdy[1]*jcob*weightq
              nx2[iconV[2,iel]]+=dNNNVdx[2]*jcob*weightq
              ny2[iconV[2,iel]]+=dNNNVdy[2]*jcob*weightq
              nx2[iconV[3,iel]]+=dNNNVdx[3]*jcob*weightq
              ny2[iconV[3,iel]]+=dNNNVdy[3]*jcob*weightq
              nx2[iconV[4,iel]]+=dNNNVdx[4]*jcob*weightq
              ny2[iconV[4,iel]]+=dNNNVdy[4]*jcob*weightq
              nx2[iconV[5,iel]]+=dNNNVdx[5]*jcob*weightq
              ny2[iconV[5,iel]]+=dNNNVdy[5]*jcob*weightq
              nx2[iconV[6,iel]]+=dNNNVdx[6]*jcob*weightq
              ny2[iconV[6,iel]]+=dNNNVdy[6]*jcob*weightq
              nx2[iconV[7,iel]]+=dNNNVdx[7]*jcob*weightq
              ny2[iconV[7,iel]]+=dNNNVdy[7]*jcob*weightq
          #end for
       #end if
   #end for

   for i in range(0,NV):
       if hull[i]:
          norm=np.sqrt(nx2[i]**2+ny2[i]**2)
          nx2[i]/=norm
          ny2[i]/=norm

   np.savetxt('normal2a.ascii',np.array([xV[surfaceV],yV[surfaceV],\
                                         nx2[surfaceV],ny2[surfaceV],theta[surfaceV]]).T)


for i in range(0,NV):
    if xV[i]/R1<eps and yV[i]/R2>1-eps:
       nx1[i]=0
       ny1[i]=0
       nx2[i]=0
       ny2[i]=0
    if xV[i]/R1<eps and yV[i]/R2<-1+eps:
       nx1[i]=0
       ny1[i]=0
       nx2[i]=0
       ny2[i]=0

np.savetxt('normal_diff.ascii',np.array([xV[surfaceV],yV[surfaceV],\
                                         nx2[surfaceV]-nx1[surfaceV],\
                                         ny2[surfaceV]-ny1[surfaceV],theta[surfaceV]]).T)

np.savetxt('normal_dot.ascii',np.array([xV[surfaceV],yV[surfaceV],\
                                        nx2[surfaceV]*nx1[surfaceV]+\
                                        ny2[surfaceV]*ny1[surfaceV],theta[surfaceV]]).T)

nx=np.zeros(NV,dtype=np.float64) 
ny=np.zeros(NV,dtype=np.float64) 
if normal_type==1:
   nx[:]=nx1[:]
   ny[:]=ny1[:]
else:
   nx[:]=nx2[:]
   ny[:]=ny2[:]

print("compute surface normals (%.3fs)" % (timing.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix = np.zeros(Nfem,dtype=bool)  
bc_val = np.zeros(Nfem,dtype=np.float64) 

for i in range(0,NV):
    if rad[i]/R1<1+eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = velocity_x(xV[i],yV[i],R1,R2,kk)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],R1,R2,kk)
    if rad[i]/R2>(1-eps) and not surface_free_slip:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = velocity_x(xV[i],yV[i],R1,R2,kk)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],R1,R2,kk)

    #vertical wall x=0
    if axisymmetric and xV[i]/R1<eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0 #u=0
       if rad[i]/R2>1-eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 #also v=0 for 2 pts
          #bc_fix[i*ndofV] = False 

#for iel in range(0,nel):
#    if surface_element[iel]:
#       if xV[iconV[0,iel]]/R1<eps:
          #i=iconV[0,iel]        
          #bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0
          #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
          #i=iconV[3,iel]        
          #bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0
          #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
          #i=iconV[6,iel]        
          #bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0
          #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

print("defining boundary conditions (%.3fs)" % (timing.time() - start))

###############################################################################
# compute area and volume of elements
# note that for the volume calculation we only consider the elements x>0
###############################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
vol=np.zeros(nel,dtype=np.float64) 

if True:
   jcb=np.zeros((2,2),dtype=np.float64)
   for iel in range(0,nel):
       for kq in range(0,nqel):
           rq=qcoords_r[kq]
           sq=qcoords_s[kq]
           weightq=qweights[kq]
           NNNV=NNN(rq,sq,mapping)
           dNNNVdr=dNNNdr(rq,sq,mapping)
           dNNNVds=dNNNds(rq,sq,mapping)
           xq=np.dot(NNNV[:],xmapping[:,iel])
           jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
           jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
           jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
           jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
           jcob = np.linalg.det(jcb)
           area[iel]+=jcob*weightq
           if xq>0: vol[iel]+=jcob*weightq*2*np.pi*xq
       #end for
   #end for

   print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
   print("     -> total area (meas) %.12e | nel= %d" %(area.sum(),nel))
   if not axisymmetric:
      print("     -> total area (anal) %e " %(np.pi*(R2**2-R1**2)))
   else:
      print("     -> total area (anal) %e " %(np.pi*(R2**2-R1**2)/2))

   print("     -> total volume (meas) %.12e | nel= %d" %(vol.sum(),nel))
   print("     -> total volume (anal) %.12e" %(4*np.pi/3*(R2**3-R1**3)))

print("compute elements areas: %.3f s" % (timing.time() - start))

if stop_here: exit()

###############################################################################
# compute array for assembly
###############################################################################
start = timing.time()

ndofV_el=mV*ndofV
local_to_globalV=np.zeros((ndofV_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            local_to_globalV[ikk,iel]=m1
    
print("compute local_to_global: %.3f s" % (timing.time() - start))

###############################################################################
# fill I,J arrays
###############################################################################
start = timing.time()

bignb=nel*( (mV*ndofV)**2 + 2*(mV*ndofV*mP) )

I=np.zeros(bignb,dtype=np.int32)
J=np.zeros(bignb,dtype=np.int32)
V=np.zeros(bignb,dtype=np.float64)

counter=0
for iel in range(0,nel):
    for ikk in range(ndofV_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndofV_el):
            m2=local_to_globalV[jkk,iel]
            I[counter]=m1
            J[counter]=m2
            counter+=1
        for jkk in range(0,mP):
            m2 =iconP[jkk,iel]+NfemV
            I[counter]=m1
            J[counter]=m2
            counter+=1
            I[counter]=m2
            J[counter]=m1
            counter+=1

print("fill I,J arrays: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix
###############################################################################
start = timing.time()

f_rhs = np.zeros(NfemV,dtype=np.float64) 
h_rhs = np.zeros(NfemP,dtype=np.float64) 
dNNNVdx  = np.zeros(mV,dtype=np.float64) 
dNNNVdy  = np.zeros(mV,dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

if axisymmetric:
   #c_mat=np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 
   c_mat=np.array([[4/3,-2/3,-2/3,0],[-2/3,4/3,-2/3,0],[-2/3,-2/3,4/3,0],[0,0,0,1]],dtype=np.float64) 
   b_mat= np.zeros((4,ndofV*mV),dtype=np.float64) # gradient matrix B 
   N_mat= np.zeros((4,ndofP*mP),dtype=np.float64) # matrix  
else:
   c_mat=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
   b_mat= np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
   N_mat= np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  

counter=0
for iel in range(0,nel):

    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    for kq in range(0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        #compute coords of quadrature points
        NNNV=NNN(rq,sq,mapping)
        xq=np.dot(NNNV[:],xmapping[:,iel])
        yq=np.dot(NNNV[:],ymapping[:,iel])

        #compute jacobian matrix
        dNNNVdr=dNNNdr(rq,sq,mapping)
        dNNNVds=dNNNds(rq,sq,mapping)
        jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
        jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
        jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
        jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        #basis functions
        NNNV=NNN(rq,sq,'Q2')
        dNNNVdr=dNNNdr(rq,sq,'Q2')
        dNNNVds=dNNNds(rq,sq,'Q2')
        NNNP=NNN(rq,sq,'Q1')

        # compute dNdx & dNdy
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for 

        if axisymmetric:

           coeffq=weightq*jcob*2*np.pi*xq

           for i in range(0,mV):
               b_mat[0:4,2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                       [NNNV[i]/xq,0.       ],
                                       [0.        ,dNNNVdy[i]],
                                       [dNNNVdy[i],dNNNVdx[i]]]
           for i in range(0,mP):
               N_mat[0,i]=NNNP[i]
               N_mat[1,i]=NNNP[i]
               N_mat[2,i]=NNNP[i]
        else:

           coeffq=weightq*jcob

           for i in range(0,mV):
               b_mat[0:3,2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                       [0.        ,dNNNVdy[i]],
                                       [dNNNVdy[i],dNNNVdx[i]]]
           for i in range(0,mP):
               N_mat[0,i]=NNNP[i]
               N_mat[1,i]=NNNP[i]

        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq,R1,R2)*coeffq

        G_el-=b_mat.T.dot(N_mat)*coeffq

        # compute elemental rhs vector
        for i in range(0,mV):
            f_el[ndofV*i  ]+=NNNV[i]*coeffq*gx(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho_m,g0)
            f_el[ndofV*i+1]+=NNNV[i]*coeffq*gy(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho_m,g0)
        #end for 
        print(xq,yq,gx(xq,yq,g0),gy(xq,yq,g0),jcob,coeffq,viscosity(xq,yq,R1,R2))

    #end for kq

    G_el*=eta_ref/h_r
    #print('K',np.min(K_el),np.max(K_el),
    #      'G',np.min(G_el),np.max(G_el),
    #      'f',np.min(f_el),np.max(f_el))

    #Kmax=np.max(abs(K_el))
    #Gmax=np.max(abs(G_el))
    #fmax=np.max(abs(f_el))
    #K_el/=Kmax
    #G_el/=Gmax
    #f_el/=fmax
    #for i in range(0,mV): #counter rotation by angle alpha
    #    RotMat[2*i  ,2*i]= np.cos(alpha) ; RotMat[2*i  ,2*i+1]=np.sin(alpha)
    #    RotMat[2*i+1,2*i]=-np.sin(alpha) ; RotMat[2*i+1,2*i+1]=np.cos(alpha)

    if surface_free_slip and surface_element[iel]==1:
       for k in range(0,mV):
           inode=iconV[k,iel]
           if surfaceV[inode] and xV[inode]>0 and (not bc_fix[inode*ndofV]):

              #print(xV[inode],yV[inode],np.arctan2(nx[inode],ny[inode]),theta[inode])
              #alpha=-np.arctan2(nx[inode],ny[inode])

              #if theta[inode]<np.pi/4:
              #alpha=-theta[inode]
              #o=1 #y-component set to 0
              #else:
              alpha=np.pi/2-theta[inode]
              o=0 #x-component set to 0

              RotMat=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
              for i in range(0,mV*ndofV):
                  RotMat[i,i]=1.
              RotMat[2*k  ,2*k]= np.cos(alpha) ; RotMat[2*k  ,2*k+1]=np.sin(alpha)
              RotMat[2*k+1,2*k]=-np.sin(alpha) ; RotMat[2*k+1,2*k+1]=np.cos(alpha)

              # apply counter rotation
              K_el=RotMat.dot(K_el.dot(RotMat.T))
              f_el=RotMat.dot(f_el)
              G_el=RotMat.dot(G_el)

              # apply boundary conditions,
              ikk=ndofV*k+o
              K_ref=K_el[ikk,ikk]
              for jkk in range(0,mV*ndofV):
                  K_el[ikk,jkk]=0
                  K_el[jkk,ikk]=0
              K_el[ikk,ikk]=K_ref
              f_el[ikk]=0#K_ref*bc_val[m1]
              #h_el[:]-=G_el[ikk,:]*bc_val[m1]
              G_el[ikk,:]=0

              # rotate back
              K_el=RotMat.T.dot(K_el.dot(RotMat))
              f_el=RotMat.T.dot(f_el)
              G_el=RotMat.T.dot(G_el)

           #end if
       #end for

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1+i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
            #end if 
        #end for 
    #end for 

    #print(np.min(K_el),np.max(K_el))
    #print(np.min(G_el),np.max(G_el))

    # assemble matrix K_mat and right hand side rhs
    for ikk in range(ndofV_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndofV_el):
            V[counter]=K_el[ikk,jkk]
            counter+=1
        for jkk in range(0,mP):
            V[counter]=G_el[ikk,jkk]
            counter+=1
            V[counter]=G_el[ikk,jkk]
            counter+=1
        f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]

#end for iel

print("build FE matrixs & rhs ( %.4fs ) | nel %d" % (timing.time() - start,nel))

###############################################################################
# solve system
###############################################################################
start = timing.time()

rhs=np.zeros(Nfem,dtype=np.float64)
rhs[0:NfemV]=f_rhs
rhs[NfemV:NfemV+NfemP]=h_rhs
   
sparse_matrix = sparse.coo_matrix((V,(I,J)),shape=(Nfem,Nfem)).tocsr()

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solving system ( %.3fs ) | nel= %d" % (timing.time() - start,nel))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:NfemV+NfemP]*eta_ref/h_r

print("     -> u (m,M) %.7e %.7e " %(np.min(u)/vel_unit,np.max(u)/vel_unit),velunit)
print("     -> v (m,M) %.7e %.7e " %(np.min(v)/vel_unit,np.max(v)/vel_unit),velunit)

if debug:
   np.savetxt('velocity.ascii',np.array([xV,yV,u/vel_unit,v/vel_unit]).T,header='# x,y,u,v')

#vr= np.cos(theta)*u+np.sin(theta)*v
#vt=-np.sin(theta)*u+np.cos(theta)*v
 
vr= np.sin(theta)*u+np.cos(theta)*v
vt= np.cos(theta)*u-np.sin(theta)*v
   
print("     -> nelr= %d | v_r (m,M) %.7e %.7e " %(nelr,np.min(vr)/vel_unit,np.max(vr)/vel_unit),velunit)
print("     -> nelr= %d | v_t (m,M) %.7e %.7e " %(nelr,np.min(vt)/vel_unit,np.max(vt)/vel_unit),velunit)

vel=np.zeros(NV,dtype=np.float64)  
vel[:]=np.sqrt(u[:]**2+v[:]**2)

print("reshape solution (%.3fs)" % (timing.time() - start))

###############################################################################
# compute element center coordinates
###############################################################################
start = timing.time()

xc=np.zeros(nel,dtype=np.float64)
yc=np.zeros(nel,dtype=np.float64)
pc=np.zeros(nel,dtype=np.float64)
thetac=np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
       rq=0
       sq=0
       #compute coords of quadrature points
       NNNV=NNN(rq,sq,mapping)
       xq=np.dot(NNNV[:],xmapping[:,iel])
       yq=np.dot(NNNV[:],ymapping[:,iel])
       xc[iel]=xq
       yc[iel]=yq
       thetac[iel]=np.pi/2-math.atan2(yq,xq)
       pc[iel]=np.sum(p[iconP[0:4,iel]])/4
       #if (not axisymmetric) and thetac[iel]<0.: thetac[iel]+=2.*math.pi


print("compute center coords (%.3fs)" % (timing.time() - start))

###############################################################################
# compute strain rate - center to nodes - method 1
###############################################################################
start = timing.time()
   
exxc=np.zeros(nel,dtype=np.float64)
eyyc=np.zeros(nel,dtype=np.float64)
exyc=np.zeros(nel,dtype=np.float64)

if compute_sr1:

   count=np.zeros(NV,dtype=np.int32)  
   Lxx1=np.zeros(NV,dtype=np.float64)  
   Lxy1=np.zeros(NV,dtype=np.float64)  
   Lyx1=np.zeros(NV,dtype=np.float64)  
   Lyy1=np.zeros(NV,dtype=np.float64)  

   for iel in range(0,nel):

       rq=0
       sq=0

       #compute coords of quadrature points
       NNNV=NNN(rq,sq,mapping)
       xq=np.dot(NNNV[:],xmapping[:,iel])
       yq=np.dot(NNNV[:],ymapping[:,iel])

       #compute jacobian matrix
       dNNNVdr=dNNNdr(rq,sq,mapping)
       dNNNVds=dNNNds(rq,sq,mapping)
       jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
       jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
       jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
       jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
       jcob=np.linalg.det(jcb)
       jcbi=np.linalg.inv(jcb)

       #basis functions
       dNNNVdr=dNNNdr(rq,sq,'Q2')
       dNNNVds=dNNNds(rq,sq,'Q2')

       for k in range(0,mV):
           dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
           dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
       #end for

       L_xx=np.dot(dNNNVdx[:],u[iconV[:,iel]])
       L_xy=np.dot(dNNNVdx[:],v[iconV[:,iel]])
       L_yx=np.dot(dNNNVdy[:],u[iconV[:,iel]])
       L_yy=np.dot(dNNNVdy[:],v[iconV[:,iel]])

       exxc[iel]=L_xx
       eyyc[iel]=L_yy
       exyc[iel]=(L_xy+L_yx)/2

       for i in range(0,mV):
           inode=iconV[i,iel]
           Lxx1[inode]+=L_xx
           Lxy1[inode]+=L_xy
           Lyx1[inode]+=L_yx
           Lyy1[inode]+=L_yy
           count[inode]+=1
       #end for
   #end for
   Lxx1/=count
   Lxy1/=count
   Lyx1/=count
   Lyy1/=count

   print("     -> Lxx1 (m,M) %e %e " %(np.min(Lxx1),np.max(Lxx1)))
   print("     -> Lyy1 (m,M) %e %e " %(np.min(Lyy1),np.max(Lyy1)))
   print("     -> Lxy1 (m,M) %e %e " %(np.min(Lxy1),np.max(Lxy1)))
   print("     -> Lxy1 (m,M) %e %e " %(np.min(Lyx1),np.max(Lyx1)))

   exx1 = np.zeros(NV,dtype=np.float64)  
   eyy1 = np.zeros(NV,dtype=np.float64)  
   exy1 = np.zeros(NV,dtype=np.float64)  

   exx1[:]=Lxx1[:]
   eyy1[:]=Lyy1[:]
   exy1[:]=0.5*(Lxy1[:]+Lyx1[:])

else:

   exx1 = np.zeros(NV,dtype=np.float64)  
   eyy1 = np.zeros(NV,dtype=np.float64)  
   exy1 = np.zeros(NV,dtype=np.float64)  

print("compute strain rate meth-1 (%.3fs)" % (timing.time() - start))

###############################################################################
# compute strain rate - corners to nodes - method 2
###############################################################################
start = timing.time()

count= np.zeros(NV,dtype=np.int32)  
Lxx2 = np.zeros(NV,dtype=np.float64)  
Lxy2 = np.zeros(NV,dtype=np.float64)  
Lyx2 = np.zeros(NV,dtype=np.float64)  
Lyy2 = np.zeros(NV,dtype=np.float64)  
q    = np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,mV):
        inode=iconV[i,iel]
        rq=rVnodes[i]
        sq=sVnodes[i]

        #compute coords of quadrature points
        NNNV=NNN(rq,sq,mapping)
        xq=np.dot(NNNV[:],xmapping[:,iel])
        yq=np.dot(NNNV[:],ymapping[:,iel])

        #compute jacobian matrix
        dNNNVdr=dNNNdr(rq,sq,mapping)
        dNNNVds=dNNNds(rq,sq,mapping)
        jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
        jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
        jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
        jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        #basis functions
        NNNV=NNN(rq,sq,'Q2')
        dNNNVdr=dNNNdr(rq,sq,'Q2')
        dNNNVds=dNNNds(rq,sq,'Q2')
        NNNP=NNN(rq,sq,'Q1')

        # compute dNdx & dNdy
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for 

        L_xx=np.dot(dNNNVdx[:],u[iconV[:,iel]])
        L_xy=np.dot(dNNNVdx[:],v[iconV[:,iel]])
        L_yx=np.dot(dNNNVdy[:],u[iconV[:,iel]])
        L_yy=np.dot(dNNNVdy[:],v[iconV[:,iel]])

        Lxx2[inode]+=L_xx
        Lxy2[inode]+=L_xy
        Lyx2[inode]+=L_yx
        Lyy2[inode]+=L_yy
        q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        count[inode]+=1
    #end for
#end for
Lxx2/=count
Lxy2/=count
Lyx2/=count
Lyy2/=count
q/=count

print("     -> Lxx2 (m,M) %e %e " %(np.min(Lxx2),np.max(Lxx2)))
print("     -> Lyy2 (m,M) %e %e " %(np.min(Lyy2),np.max(Lyy2)))
print("     -> Lxy2 (m,M) %e %e " %(np.min(Lxy2),np.max(Lxy2)))
print("     -> Lxy2 (m,M) %e %e " %(np.min(Lyx2),np.max(Lyx2)))

#np.savetxt('pressure.ascii',np.array([xV,yV,q]).T)
#np.savetxt('strainrate.ascii',np.array([xV,yV,Lxx,Lyy,Lxy,Lyx]).T)

exx2 = np.zeros(NV,dtype=np.float64)  
eyy2 = np.zeros(NV,dtype=np.float64)  
exy2 = np.zeros(NV,dtype=np.float64)  

exx2[:]=Lxx2[:]
eyy2[:]=Lyy2[:]
exy2[:]=0.5*(Lxy2[:]+Lyx2[:])

print("compute strain rate meth-2 (%.3fs)" % (timing.time() - start))

###############################################################################
# convert strain rate tensor to spherical coordinates
# Note that there is a pb when we run the model in plane strain. In that 
# case I set the err,ert,ett to zero for x<0
###############################################################################
start = timing.time()

e_rr2=np.zeros(NV,dtype=np.float64)  
e_tt2=np.zeros(NV,dtype=np.float64)  
e_rt2=np.zeros(NV,dtype=np.float64)  

for i in range(0,NV):
    if xV[i]>=0:
       e_rr2[i]=exx2[i]*np.sin(theta[i])**2+\
               2*exy2[i]*np.sin(theta[i])*np.cos(theta[i])+\
               eyy2[i]*np.cos(theta[i])**2
       e_tt2[i]=exx2[i]*np.cos(theta[i])**2-\
               2*exy2[i]*np.sin(theta[i])*np.cos(theta[i])+\
               eyy2[i]*np.sin(theta[i])**2
       e_rt2[i]=(exx2[i]-eyy2[i])*np.sin(theta[i])*np.cos(theta[i])+\
               exy2[i]*(-np.sin(theta[i])**2+\
               np.cos(theta[i])**2)

e_rrc=np.zeros(nel,dtype=np.float64)  
for iel in range(0,nel):
    if xc[iel]>=0:
       e_rrc[iel]=exxc[iel]*np.sin(thetac[iel])**2+\
                  2*exyc[iel]*np.sin(thetac[iel])*np.cos(thetac[iel])+\
                  eyyc[iel]*np.cos(thetac[iel])**2

print("     -> e_rr (m,M) %e %e | nel= %d" %(np.min(e_rr2),np.max(e_rr2),nel))
print("     -> e_tt (m,M) %e %e | nel= %d" %(np.min(e_tt2),np.max(e_tt2),nel))
print("     -> e_rt (m,M) %e %e | nel= %d" %(np.min(e_rt2),np.max(e_rt2),nel))

print("compute strain rate in sph. coords. (%.3fs)" % (timing.time() - start))

###############################################################################
start = timing.time()

if compute_sr3:

   M_mat=lil_matrix((NV,NV),dtype=np.float64)
   rhsLxx=np.zeros(NV,dtype=np.float64)
   rhsLyy=np.zeros(NV,dtype=np.float64)
   rhsLxy=np.zeros(NV,dtype=np.float64)
   rhsLyx=np.zeros(NV,dtype=np.float64)

   for iel in range(0,nel):

       M_el =np.zeros((mV,mV),dtype=np.float64)
       fLxx_el=np.zeros(mV,dtype=np.float64)
       fLyy_el=np.zeros(mV,dtype=np.float64)
       fLxy_el=np.zeros(mV,dtype=np.float64)
       fLyx_el=np.zeros(mV,dtype=np.float64)
       NNNV1 =np.zeros((mV,1),dtype=np.float64) 

       for kq in range(0,nqel):
           rq=qcoords_r[kq]
           sq=qcoords_s[kq]
           weightq=qweights[kq]

           #compute coords of quadrature points
           NNNV=NNN(rq,sq,mapping)
           xq=np.dot(NNNV[:],xmapping[:,iel])
           yq=np.dot(NNNV[:],ymapping[:,iel])

           #compute jacobian matrix
           dNNNVdr=dNNNdr(rq,sq,mapping)
           dNNNVds=dNNNds(rq,sq,mapping)
           jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
           jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
           jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
           jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
           jcob=np.linalg.det(jcb)
           jcbi=np.linalg.inv(jcb)

           #basis functions
           NNNV1[:,0]=NNN(rq,sq,'Q2')
           dNNNVdr=dNNNdr(rq,sq,'Q2')
           dNNNVds=dNNNds(rq,sq,'Q2')

           # compute dNdx & dNdy
           Lxxq=0.
           Lyyq=0.
           Lxyq=0.
           Lyxq=0.
           for k in range(0,mV):
               dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
               dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
               Lxxq+=dNNNVdx[k]*u[iconV[k,iel]]
               Lyyq+=dNNNVdy[k]*v[iconV[k,iel]]
               Lxyq+=dNNNVdx[k]*v[iconV[k,iel]]
               Lyxq+=dNNNVdy[k]*u[iconV[k,iel]]
           #end for 

           M_el +=NNNV1.dot(NNNV1.T)*weightq*jcob

           fLxx_el[:]+=NNNV1[:,0]*Lxxq*jcob*weightq
           fLyy_el[:]+=NNNV1[:,0]*Lyyq*jcob*weightq
           fLxy_el[:]+=NNNV1[:,0]*Lxyq*jcob*weightq
           fLyx_el[:]+=NNNV1[:,0]*Lyxq*jcob*weightq

       #end for kq

       for k1 in range(0,mV):
           m1=iconV[k1,iel]
           for k2 in range(0,mV):
               m2=iconV[k2,iel]
               M_mat[m1,m2]+=M_el[k1,k2]
           #end for
           rhsLxx[m1]+=fLxx_el[k1]
           rhsLyy[m1]+=fLyy_el[k1]
           rhsLxy[m1]+=fLxy_el[k1]
           rhsLyx[m1]+=fLyx_el[k1]
       #end for

   #end for
   #sparse_matrix=A_sparse.tocsr()

   Lxx3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxx)
   Lyy3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyy)
   Lxy3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxy)
   Lyx3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyx)

   print("     -> Lxx3 (m,M) %e %e " %(np.min(Lxx3),np.max(Lxx3)))
   print("     -> Lyy3 (m,M) %e %e " %(np.min(Lyy3),np.max(Lyy3)))
   print("     -> Lxy3 (m,M) %e %e " %(np.min(Lxy3),np.max(Lxy3)))
   print("     -> Lxy3 (m,M) %e %e " %(np.min(Lyx3),np.max(Lyx3)))

   exx3=np.zeros(NV,dtype=np.float64)  
   eyy3=np.zeros(NV,dtype=np.float64)  
   exy3=np.zeros(NV,dtype=np.float64)  

   exx3[:]=Lxx3[:]
   eyy3[:]=Lyy3[:]
   exy3[:]=0.5*(Lxy3[:]+Lyx3[:])

else:

   exx3=np.zeros(NV,dtype=np.float64)  
   eyy3=np.zeros(NV,dtype=np.float64)  
   exy3=np.zeros(NV,dtype=np.float64)  


print("compute strain rate meth-3 (%.3fs)" % (timing.time() - start))

###############################################################################
# compute elemental and nodal viscosity
###############################################################################
start = timing.time()

viscosity_elemental=np.zeros(nel,dtype=np.float64)
density_elemental=np.zeros(nel,dtype=np.float64)
viscosity_nodal=np.zeros(NV,dtype=np.float64)
counter=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    viscosity_elemental[iel]=viscosity(xc[iel],yc[iel],R1,R2)
    density_elemental[iel]=density(xc[iel],yc[iel],R1,R2,kk,rho_m,g0)
    for i in range(0,mV):
        counter[iconV[i,iel]]+=1
        viscosity_nodal[iconV[i,iel]]+=viscosity_elemental[iel]
    #end for
#end for

viscosity_nodal/=counter

if debug:
   np.savetxt('xycenter'+mapping+'.ascii',np.array([xc,yc,density_elemental]).T)

print("compute viscosity on mesh (%.3fs)" % (timing.time() - start))

###############################################################################
# normalise pressure
# note that the integration could be improved (I currently only sample the 
# pressure in 1 point in the middle of the edge)
###############################################################################
start = timing.time()

if not axisymmetric:
   if exp==0:
      poffset=np.sum(q[0:2*nelt])/(2*nelt)
   else:
      poffset=0
      for iel in range(0,nel):
          if surface_element[iel]:
             dtheta=2*np.pi/nelt 
             pmean=0.5*(p[iconP[2,iel]]+p[iconP[3,iel]])
             poffset+=dtheta*pmean
      poffset/=2*np.pi

   q-=poffset
   p-=poffset

else: #zero average pressure on surface

   poffset=0
   for iel in range(0,nel):
       if surface_element[iel]:
          dtheta=theta[iconV[2,iel]]-theta[iconV[3,iel]]
          pmean=0.5*(p[iconP[2,iel]]+p[iconP[3,iel]])
          poffset+=np.sin((theta[iconV[2,iel]]+theta[iconV[3,iel]])/2)*dtheta\
                   *2*np.pi*R2**2 * pmean
   poffset/=4*np.pi*R2**2
   q-=poffset
   p-=poffset

   poffset=0
   for iel in range(0,nel):
       if surface_element[iel]:
          dtheta=theta[iconV[2,iel]]-theta[iconV[3,iel]]
          poffset+=np.sin((theta[iconV[2,iel]]+theta[iconV[3,iel]])/2)*dtheta * 2*np.pi*R2**2 * pc[iel]
   poffset/=4*np.pi*R2**2
   pc-=poffset

print("     -> p (m,M) %e %e " %(np.min(p),np.max(p)))
print("     -> q (m,M) %e %e " %(np.min(q),np.max(q)))

#np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

print("normalise pressure (%.3fs)" % (timing.time() - start))

###############################################################################
start = timing.time()

u_err = np.zeros(NV,dtype=np.float64) 
v_err = np.zeros(NV,dtype=np.float64)    
p_err = np.zeros(NP,dtype=np.float64)    

if exp==0:
   for i in range(0,NV):
       u_err[i]=u[i]-velocity_x(xV[i],yV[i],R1,R2,kk)
       v_err[i]=v[i]-velocity_y(xV[i],yV[i],R1,R2,kk)

   for i in range(0,NP):
       p_err[i]=p[i]-pressure(xP[i],yP[i],R1,R2,kk,rho_m,g0)

   print("     -> u_err (m,M) %.10e %.10e | nelr= %d" %(np.min(u_err),np.max(u_err),nelr))
   print("     -> v_err (m,M) %.10e %.10e | nelr= %d" %(np.min(v_err),np.max(v_err),nelr))
   print("     -> p_err (m,M) %.10e %.10e | nelr= %d" %(np.min(p_err),np.max(p_err),nelr))

print("compute error fields (%.3fs)" % (timing.time() - start))

###############################################################################
# since I have zeroed the the spherical components of the strain rate for x<0
# then I also zero the dynamic topography
# note that this is only valid for constant viscosity!
###############################################################################
start = timing.time()

dyn_topo_nodal=np.zeros(NV,dtype=np.float64)
if exp>0:
   for i in range(0,NV):
       if surfaceV[i] and xV[i]>=0:
          dyn_topo_nodal[i]= -(2*viscosity_nodal[i]*e_rr2[i]-q[i])/(rho_m*g0) 

dyn_topo_eltal=np.zeros(nel,dtype=np.float64)
if exp>0:
   for iel in range(0,nel):
       if surface_element[iel] and xc[iel]>=0:
          dyn_topo_eltal[iel]= -(2*viscosity_elemental[iel]*e_rrc[iel]-pc[iel])/(rho_m*g0) 

print("compute dynamic topography: %.3f s" % (timing.time() - start))

###############################################################################
# export fields at both surfaces
###############################################################################
start = timing.time()

surfaceQ1=np.zeros(NV,dtype=bool) 
surfaceQ2=np.zeros(NV,dtype=bool) 
for iel in range(0,nel):
    if surface_element[iel]:
       surfaceQ1[iconV[2,iel]]=True
       surfaceQ1[iconV[3,iel]]=True
       surfaceQ2[iconV[2,iel]]=True
       surfaceQ2[iconV[3,iel]]=True
       surfaceQ2[iconV[6,iel]]=True

leftV=np.zeros(NV,dtype=bool) 
for i in range(0,NV):
    if xV[i]<eps and yV[i]<0: leftV[i]=True

sr1=np.zeros(NV,dtype=np.float64)  
sr2=np.zeros(NV,dtype=np.float64)  
sr3=np.zeros(NV,dtype=np.float64)  
src=np.zeros(nel,dtype=np.float64)  
sr1[:]=np.sqrt(0.5*(exx1[:]**2+eyy1[:]**2)+exy1[:]**2)
sr2[:]=np.sqrt(0.5*(exx2[:]**2+eyy2[:]**2)+exy2[:]**2)
sr3[:]=np.sqrt(0.5*(exx3[:]**2+eyy3[:]**2)+exy3[:]**2)
src[:]=np.sqrt(0.5*(exxc[:]**2+eyyc[:]**2)+exyc[:]**2)

np.savetxt('pc_R2.ascii',np.array([thetac[surface_element],pc[surface_element]]).T)
np.savetxt('src_R2.ascii',np.array([thetac[surface_element],src[surface_element]]).T)
np.savetxt('errc_R2.ascii',np.array([thetac[surface_element],e_rrc[surface_element]]).T)
np.savetxt('d_tc_R2.ascii',np.array([thetac[surface_element],dyn_topo_eltal[surface_element]]).T)

#np.savetxt('qqq_R1.ascii',np.array([xV[0:nnt],yV[0:nnt],q[0:nnt],theta[0:nnt]]).T)
#np.savetxt('sr2_R1.ascii',np.array([xV[0:nnt],yV[0:nnt],sr2[0:nnt],theta[0:nnt]]).T)
#np.savetxt('vel_R1.ascii',np.array([xV[0:nnt],yV[0:nnt],vel[0:nnt],theta[0:nnt]]).T)

np.savetxt('qqq_R2.ascii',np.array([theta[surfaceQ1],q[surfaceQ1]]).T)
np.savetxt('sr2_R2.ascii',np.array([theta[surfaceQ2],sr2[surfaceQ2]]).T)
np.savetxt('vel_R2.ascii',np.array([theta[surfaceQ2],vel[surfaceQ2],vr[surfaceQ2],vt[surfaceQ2]]).T)
np.savetxt('v_r_R2.ascii',np.array([theta[surfaceQ2],vr[surfaceQ2]]).T)
np.savetxt('v_t_R2.ascii',np.array([theta[surfaceQ2],vt[surfaceQ2]]).T)
np.savetxt('err_R2.ascii',np.array([theta[surfaceQ2],e_rr2[surfaceQ2]]).T)
np.savetxt('d_t_R2.ascii',np.array([theta[surfaceQ2],dyn_topo_nodal[surfaceQ2]]).T)

if compute_sr1:
   np.savetxt('sr1_R1.ascii',np.array([xV[0:nnt],yV[0:nnt],sr1[0:nnt],theta[0:nnt]]).T)
   np.savetxt('sr1_R2.ascii',np.array([xV[surfaceQ2],yV[surfaceQ2],sr1[surfaceQ2],theta[surfaceQ2]]).T)

if compute_sr3:
   np.savetxt('sr3_R1.ascii',np.array([xV[0:nnt],yV[0:nnt],sr3[0:nnt],theta[0:nnt]]).T)
   np.savetxt('sr3_R2.ascii',np.array([xV[surfaceQ2],yV[surfaceQ2],sr3[surfaceQ2],theta[surfaceQ2]]).T)

np.savetxt('sr2_left.ascii',np.array([xV[leftV],yV[leftV],sr2[leftV],rad[leftV]]).T)
np.savetxt('vel_left.ascii',np.array([xV[leftV],yV[leftV],vel[leftV],rad[leftV]]).T)

#np.savetxt('sr2_Q1_R2.ascii',np.array([xV[surfaceQ1],yV[surfaceQ1],sr2[surfaceQ1],theta[surfaceQ1]]).T)
#np.savetxt('vel_Q1_R2.ascii',np.array([xV[surfaceQ1],yV[surfaceQ1],vel[surfaceQ1],theta[surfaceQ1],vr[surfaceQ1],vt[surfaceQ1]]).T)
#np.savetxt('err_Q1_R2.ascii',np.array([xV[surfaceQ1],yV[surfaceQ1],e_rr2[surfaceQ1],theta[surfaceQ1]]).T)
#np.savetxt('d_t_Q1_R2.ascii',np.array([xV[surfaceQ1],yV[surfaceQ1],dyn_topo_nodal[surfaceQ1],theta[surfaceQ1]]).T)

print("export p&q on R1,R2 (%.3fs)" % (timing.time() - start))

###############################################################################
# compute error
###############################################################################
start = timing.time()

errv=0.
errp=0.
errq=0.
errexx1=0.
erreyy1=0.
errexy1=0.
errexx2=0.
erreyy2=0.
errexy2=0.
errexx3=0.
erreyy3=0.
errexy3=0.
vrms=0.
for iel in range (0,nel):

    for kq in range(0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]

        #compute coords of quadrature points
        NNNV=NNN(rq,sq,mapping)
        xq=np.dot(NNNV[:],xmapping[:,iel])
        yq=np.dot(NNNV[:],ymapping[:,iel])

        #compute jacobian matrix
        dNNNVdr=dNNNdr(rq,sq,mapping)
        dNNNVds=dNNNds(rq,sq,mapping)
        jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
        jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
        jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
        jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
        jcob=np.linalg.det(jcb)
        JxW=jcob*qweights[kq]

        #basis functions
        NNNV=NNN(rq,sq,'Q2')
        dNNNVdr=dNNNdr(rq,sq,'Q2')
        dNNNVds=dNNNds(rq,sq,'Q2')
        NNNP=NNN(rq,sq,'Q1')

        uq=np.dot(NNNV[:],u[iconV[:,iel]])
        vq=np.dot(NNNV[:],v[iconV[:,iel]])
        qq=np.dot(NNNV[:],q[iconV[:,iel]])

        #exx1q=np.dot(NNNV[:],exx1[iconV[:,iel]])
        #eyy1q=np.dot(NNNV[:],eyy1[iconV[:,iel]])
        #exy1q=np.dot(NNNV[:],exy1[iconV[:,iel]])

        #exx2q=np.dot(NNNV[:],exx2[iconV[:,iel]])
        #eyy2q=np.dot(NNNV[:],eyy2[iconV[:,iel]])
        #exy2q=np.dot(NNNV[:],exy2[iconV[:,iel]])

        #exx3q=np.dot(NNNV[:],exx3[iconV[:,iel]])
        #eyy3q=np.dot(NNNV[:],eyy3[iconV[:,iel]])
        #exy3q=np.dot(NNNV[:],exy3[iconV[:,iel]])

        errv+=((uq-velocity_x(xq,yq,R1,R2,kk))**2+\
               (vq-velocity_y(xq,yq,R1,R2,kk))**2)*JxW
        errq+=(qq-pressure(xq,yq,R1,R2,kk,rho_m,g0))**2*JxW

        #errexx1+=(exx1q-sr_xx(xq,yq,R1,R2,kk))**2*JxW
        #erreyy1+=(eyy1q-sr_yy(xq,yq,R1,R2,kk))**2*JxW
        #errexy1+=(exy1q-sr_xy(xq,yq,R1,R2,kk))**2*JxW
        #errexx2+=(exx2q-sr_xx(xq,yq,R1,R2,kk))**2*JxW
        #erreyy2+=(eyy2q-sr_yy(xq,yq,R1,R2,kk))**2*JxW
        #errexy2+=(exy2q-sr_xy(xq,yq,R1,R2,kk))**2*JxW
        #errexx3+=(exx3q-sr_xx(xq,yq,R1,R2,kk))**2*JxW
        #erreyy3+=(eyy3q-sr_yy(xq,yq,R1,R2,kk))**2*JxW
        #errexy3+=(exy3q-sr_xy(xq,yq,R1,R2,kk))**2*JxW

        if axisymmetric: 
           JxW*=2*np.pi*xq

        vrms+=(uq**2+vq**2)*JxW

        pq=np.dot(NNNP[:],p[iconP[:,iel]])
        errp+=(pq-pressure(xq,yq,R1,R2,kk,rho_m,g0))**2*JxW

    # end for kq
# end for iel

errv=np.sqrt(errv)       ; errp=np.sqrt(errp)       ; errq=np.sqrt(errq)
#errexx1=np.sqrt(errexx1) ; erreyy1=np.sqrt(erreyy1) ; errexy1=np.sqrt(errexy1)
#errexx2=np.sqrt(errexx2) ; erreyy2=np.sqrt(erreyy2) ; errexy2=np.sqrt(errexy2)
#errexx3=np.sqrt(errexx3) ; erreyy3=np.sqrt(erreyy3) ; errexy3=np.sqrt(errexy3)

if axisymmetric :
   vrms=np.sqrt(vrms/  (4/3*np.pi*(R2**3-R1**3))   )
else:
   vrms=np.sqrt(vrms/  (np.pi*(R2**2-R1**2))  )

print('     -> nelr= %6d ; vrms= %.14e' %(nelr,vrms/vel_unit))
print("     -> nelr= %6d ; errv= %.14e ; errp= %.14e ; errq= %.14e" %(nelr,errv,errp,errq))
print("     -> nelr= %6d ; errexx1= %.14e ; erreyy1= %.14e ; errexy1= %.14e" %(nelr,errexx1,erreyy1,errexy1))
print("     -> nelr= %6d ; errexx2= %.14e ; erreyy2= %.14e ; errexy2= %.14e" %(nelr,errexx2,erreyy2,errexy2))
print("     -> nelr= %6d ; errexx3= %.14e ; erreyy3= %.14e ; errexy3= %.14e" %(nelr,errexx3,erreyy3,errexy3))

print("compute errors (%.3fs)" % (timing.time() - start))

###############################################################################
# generate vtu files for mapping nodes and quadrature points
###############################################################################

if debug:
   mmapping=np.size(xmapping[:,0])
   vtufile=open("mapping_points_"+mapping+".vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints='%5d' NumberOfCells='%5d'> \n" %(mmapping,mmapping))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,mmapping):
       vtufile.write("%e %e %e \n" %(xmapping[i,0],ymapping[i,0],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,mmapping):
       vtufile.write("%d \n" %(iel))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,mmapping):
       vtufile.write("%d \n" %((iel+1)*1))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,mmapping):
       vtufile.write("%d \n" %1)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   vtufile=open("quadrature_points_"+str(nqperdim)+".vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints='%5d' NumberOfCells='%5d'> \n" %(nqel,nqel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for k in range(0,nqel):
       rq=qcoords_r[k]
       sq=qcoords_s[k]
       NNNV=NNN(rq,sq,mapping)
       xq=np.dot(NNNV[:],xmapping[:,0])
       yq=np.dot(NNNV[:],ymapping[:,0])
       vtufile.write("%e %e %e \n" %(xq,yq,0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nqel):
       vtufile.write("%d \n" %(iel))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nqel):
       vtufile.write("%d \n" %((iel+1)*1))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nqel):
       vtufile.write("%d \n" %1)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

###############################################################################
# plot of solution
###############################################################################
start = timing.time()

if visu==1:

   vtufile=open("solutionQ2_mapping"+mapping+".vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i]/vel_unit,v[i]/vel_unit,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal1' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(nx1[i],ny1[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(nx2[i],ny2[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal diff' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(nx1[i]-nx2[i],ny1[i]-ny2[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %viscosity_nodal[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='hull' Format='ascii'> \n")
   for i in range(0,NV):
       if hull[i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='surfaceV' Format='ascii'> \n")
   for i in range(0,NV):
       if surfaceV[i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='cmbV' Format='ascii'> \n")
   for i in range(0,NV):
       if cmbV[i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='surfaceQ1' Format='ascii'> \n")
   for i in range(0,NV):
       if surfaceQ1[i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='bc_fix(u)' Format='ascii'> \n")
   for i in range(0,NV):
       if bc_fix[2*i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='bc_fix(v)' Format='ascii'> \n")
   for i in range(0,NV):
       if bc_fix[2*i+1]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='area' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e \n" %area[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='viscosity' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e \n" %viscosity_elemental[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e \n" %density_elemental[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='vol' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e \n" %vol[iel])
   vtufile.write("</DataArray>\n")
   #
   vtufile.write("<DataArray type='Float32' Name='surface_element' Format='ascii'> \n")
   for iel in range(0,nel):
       if surface_element[iel]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #
   vtufile.write("<DataArray type='Float32' Name='cmb_element' Format='ascii'> \n")
   for iel in range(0,nel):
       if cmb_element[iel]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                   iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %23)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   ####################################

   vtufile=open("solutionQ1_mapping"+mapping+".vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,4*nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(gx(xV[i],yV[i],g0),gy(xV[i],yV[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i]/vel_unit,v[i]/vel_unit,0.))
   vtufile.write("</DataArray>\n")
   #--
   if exp==0:
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(th)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e %e %e \n" %(velocity_x(xV[i],yV[i],R1,R2,kk),velocity_y(xV[i],yV[i],R1,R2,kk),0.))
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(error)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e %e %e \n" %(u_err[i],v_err[i],0.))
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(vr[i]/vel_unit,vt[i]/vel_unit,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %rad[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta (co-latitude)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %theta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %density(xV[i],yV[i],R1,R2,kk,rho_m,g0))
   vtufile.write("</DataArray>\n")
   #--
   if exp==0:
      #--
      vtufile.write("<DataArray type='Float32' Name='exx (th)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %(sr_xx(xV[i],yV[i],R1,R2,kk)))
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='eyy (th)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %(sr_yy(xV[i],yV[i],R1,R2,kk)))
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='exy (th)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %(sr_xy(xV[i],yV[i],R1,R2,kk)))
      vtufile.write("</DataArray>\n")
   #--
   if compute_sr1:
      vtufile.write("<DataArray type='Float32' Name='sr1' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %sr1[i])
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sr2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %sr2[i])
   vtufile.write("</DataArray>\n")
   #--
   if compute_sr1:
      vtufile.write("<DataArray type='Float32' Name='exx1' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %exx1[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='eyy1' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %eyy1[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='exy1' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %exy1[i])
      vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %exx2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %eyy2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %exy2[i])
   vtufile.write("</DataArray>\n")

   if compute_sr3:
      vtufile.write("<DataArray type='Float32' Name='sr3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %sr3[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='exx3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %exx3[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='eyy3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %eyy3[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='exy3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %exy3[i])
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='err' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_rr2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='ett' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_tt2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='ert' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_rt2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %viscosity_nodal[i])
   vtufile.write("</DataArray>\n")
   #--
   if exp==0:
      vtufile.write("<DataArray type='Float32' Name='q (th)' Format='ascii'> \n")
      for i in range (0,NV):
          vtufile.write("%e \n" % pressure(xV[i],yV[i],R1,R2,kk,rho_m,g0))
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d %d %d %d\n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,4*nel):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu file (%.3fs)" % (timing.time() - start))

print("EARTH4D | nelr= %d | v_r  %.5f %.5f  v_t  %.5f %.5f v_rms: %.5f " %\
(nelr,np.min(vr)/vel_unit,np.max(vr)/vel_unit,np.min(vt)/vel_unit,np.max(vt)/vel_unit,vrms/vel_unit))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
