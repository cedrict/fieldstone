from numba import jit
import numpy as np

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
       N1s=-(625*s**5-625*s**4-250*s**3+250*s**2 +9*s-9)/768 
       N2s= (125*s**5 -75*s**4-130*s**3 +78*s**2 +5*s-3)*25/768
       N3s=-(125*s**5 -25*s**4-170*s**3 +34*s**2+45*s-9)*25/384
       N4s= (125*s**5 +25*s**4-170*s**3 -34*s**2+45*s+9)*25/384
       N5s=-(125*s**5 +75*s**4-130*s**3 -78*s**2 +5*s+3)*25/768
       N6s= (625*s**5+625*s**4-250*s**3-250*s**2 +9*s+9)/768 
       val[0]=N1r*N1s  ;val[1]=N2r*N1s  ;val[2]=N3r*N1s  ;val[3]=N4r*N1s  ;val[4]=N5r*N1s  ;val[5]=N6r*N1s  
       val[6]=N1r*N2s  ;val[7]=N2r*N2s  ;val[8]=N3r*N2s  ;val[9]=N4r*N2s  ;val[10]=N5r*N2s ;val[11]=N6r*N2s 
       val[12]=N1r*N3s ;val[13]=N2r*N3s ;val[14]=N3r*N3s ;val[15]=N4r*N3s ;val[16]=N5r*N3s ;val[17]=N6r*N3s 
       val[18]=N1r*N4s ;val[19]=N2r*N4s ;val[20]=N3r*N4s ;val[21]=N4r*N4s ;val[22]=N5r*N4s ;val[23]=N6r*N4s 
       val[24]=N1r*N5s ;val[25]=N2r*N5s ;val[26]=N3r*N5s ;val[27]=N4r*N5s ;val[28]=N5r*N5s ;val[29]=N6r*N5s 
       val[30]=N1r*N6s ;val[31]=N2r*N6s ;val[32]=N3r*N6s ;val[33]=N4r*N6s ;val[34]=N5r*N6s ;val[35]=N6r*N6s 
    if space=='Q6':
       val = np.zeros(49,dtype=np.float64)
       N1r=  (81*r**6 - 81*r**5 -  45*r**4 + 45*r**3 +  4*r**2 - 4*r)/80    
       N2r= -(27*r**6 - 18*r**5 -  30*r**4 + 20*r**3 +  3*r**2 - 2*r)*9./40 
       N3r=  (27*r**6 -  9*r**5 -  39*r**4 + 13*r**3 + 12*r**2 - 4*r)*9./16 
       N4r= -(81*r**6           - 126*r**4           + 49*r**2       -4)/4  
       N5r=  (27*r**6 +  9*r**5 -  39*r**4 - 13*r**3 + 12*r**2 + 4*r)*9./16
       N6r= -(27*r**6 + 18*r**5 -  30*r**4 - 20*r**3 +  3*r**2 + 2*r)*9./40 
       N7r=  (81*r**6 + 81*r**5 -  45*r**4 - 45*r**3 +  4*r**2 + 4*r)/80
       N1s=  (81*s**6 - 81*s**5 -  45*s**4 + 45*s**3 +  4*s**2 - 4*s)/80 
       N2s= -(27*s**6 - 18*s**5 -  30*s**4 + 20*s**3 +  3*s**2 - 2*s)*9/40
       N3s=  (27*s**6 -  9*s**5 -  39*s**4 + 13*s**3 + 12*s**2 - 4*s)*9/16
       N4s= -(81*s**6           - 126*s**4           + 49*s**2       -4)/4 
       N5s=  (27*s**6 +  9*s**5 -  39*s**4 - 13*s**3 + 12*s**2 + 4*s)*9/16
       N6s= -(27*s**6 + 18*s**5 -  30*s**4 - 20*s**3 +  3*s**2 + 2*s)*9/40 
       N7s=  (81*s**6 + 81*s**5 -  45*s**4 - 45*s**3 +  4*s**2 + 4*s)/80
       val[0]=N1r*N1s  ;val[1]=N2r*N1s  ;val[2]=N3r*N1s  ;val[3]=N4r*N1s  ;val[4]=N5r*N1s  ;val[5]=N6r*N1s  ;val[6]=N7r*N1s
       val[7]=N1r*N2s  ;val[8]=N2r*N2s  ;val[9]=N3r*N2s  ;val[10]=N4r*N2s ;val[11]=N5r*N2s ;val[12]=N6r*N2s ;val[13]=N7r*N2s
       val[14]=N1r*N3s ;val[15]=N2r*N3s ;val[16]=N3r*N3s ;val[17]=N4r*N3s ;val[18]=N5r*N3s ;val[19]=N6r*N3s ;val[20]=N7r*N3s
       val[21]=N1r*N4s ;val[22]=N2r*N4s ;val[23]=N3r*N4s ;val[24]=N4r*N4s ;val[25]=N5r*N4s ;val[26]=N6r*N4s ;val[27]=N7r*N4s
       val[28]=N1r*N5s ;val[29]=N2r*N5s ;val[30]=N3r*N5s ;val[31]=N4r*N5s ;val[32]=N5r*N5s ;val[33]=N6r*N5s ;val[34]=N7r*N5s
       val[35]=N1r*N6s ;val[36]=N2r*N6s ;val[37]=N3r*N6s ;val[38]=N4r*N6s ;val[39]=N5r*N6s ;val[40]=N6r*N6s ;val[41]=N7r*N6s
       val[42]=N1r*N7s ;val[43]=N2r*N7s ;val[44]=N3r*N7s ;val[45]=N4r*N7s ;val[46]=N5r*N7s ;val[47]=N6r*N7s ;val[48]=N7r*N7s
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
       val[0]=N1r*N1s  ;val[1]=N2r*N1s  ;val[2]=N3r*N1s  ;val[3]=N4r*N1s  ;val[4]=N5r*N1s  ;val[5]=N6r*N1s  
       val[6]=N1r*N2s  ;val[7]=N2r*N2s  ;val[8]=N3r*N2s  ;val[9]=N4r*N2s  ;val[10]=N5r*N2s ;val[11]=N6r*N2s 
       val[12]=N1r*N3s ;val[13]=N2r*N3s ;val[14]=N3r*N3s ;val[15]=N4r*N3s ;val[16]=N5r*N3s ;val[17]=N6r*N3s 
       val[18]=N1r*N4s ;val[19]=N2r*N4s ;val[20]=N3r*N4s ;val[21]=N4r*N4s ;val[22]=N5r*N4s ;val[23]=N6r*N4s 
       val[24]=N1r*N5s ;val[25]=N2r*N5s ;val[26]=N3r*N5s ;val[27]=N4r*N5s ;val[28]=N5r*N5s ;val[29]=N6r*N5s 
       val[30]=N1r*N6s ;val[31]=N2r*N6s ;val[32]=N3r*N6s ;val[33]=N4r*N6s ;val[34]=N5r*N6s ;val[35]=N6r*N6s 
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
       val[0]=N1r*N1s  ;val[1]=N2r*N1s  ;val[2]=N3r*N1s  ;val[3]=N4r*N1s  ;val[4]=N5r*N1s  ;val[5]=N6r*N1s  ;val[6]=N7r*N1s
       val[7]=N1r*N2s  ;val[8]=N2r*N2s  ;val[9]=N3r*N2s  ;val[10]=N4r*N2s ;val[11]=N5r*N2s ;val[12]=N6r*N2s ;val[13]=N7r*N2s
       val[14]=N1r*N3s ;val[15]=N2r*N3s ;val[16]=N3r*N3s ;val[17]=N4r*N3s ;val[18]=N5r*N3s ;val[19]=N6r*N3s ;val[20]=N7r*N3s
       val[21]=N1r*N4s ;val[22]=N2r*N4s ;val[23]=N3r*N4s ;val[24]=N4r*N4s ;val[25]=N5r*N4s ;val[26]=N6r*N4s ;val[27]=N7r*N4s
       val[28]=N1r*N5s ;val[29]=N2r*N5s ;val[30]=N3r*N5s ;val[31]=N4r*N5s ;val[32]=N5r*N5s ;val[33]=N6r*N5s ;val[34]=N7r*N5s
       val[35]=N1r*N6s ;val[36]=N2r*N6s ;val[37]=N3r*N6s ;val[38]=N4r*N6s ;val[39]=N5r*N6s ;val[40]=N6r*N6s ;val[41]=N7r*N6s
       val[42]=N1r*N7s ;val[43]=N2r*N7s ;val[44]=N3r*N7s ;val[45]=N4r*N7s ;val[46]=N5r*N7s ;val[47]=N6r*N7s ;val[48]=N7r*N7s
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
       N1s= (-3125*s**4+2500*s**3+750*s**2-500*s-9 )/768
       N2s= (  625*s**4- 300*s**3-390*s**2+156*s+5 )*25/768
       N3s=-(  625*s**4- 100*s**3-510*s**2+ 68*s+45)*25/384
       N4s= (  625*s**4+ 100*s**3-510*s**2- 68*s+45)*25/384 
       N5s=-(  625*s**4+ 300*s**3-390*s**2-156*s+5 )*25/768
       N6s= ( 3125*s**4+2500*s**3-750*s**2-500*s+9 )/768 
       val[0]=N1r*N1s  ;val[1]=N2r*N1s  ;val[2]=N3r*N1s  ;val[3]=N4r*N1s  ;val[4]=N5r*N1s  ;val[5]=N6r*N1s  
       val[6]=N1r*N2s  ;val[7]=N2r*N2s  ;val[8]=N3r*N2s  ;val[9]=N4r*N2s  ;val[10]=N5r*N2s ;val[11]=N6r*N2s 
       val[12]=N1r*N3s ;val[13]=N2r*N3s ;val[14]=N3r*N3s ;val[15]=N4r*N3s ;val[16]=N5r*N3s ;val[17]=N6r*N3s 
       val[18]=N1r*N4s ;val[19]=N2r*N4s ;val[20]=N3r*N4s ;val[21]=N4r*N4s ;val[22]=N5r*N4s ;val[23]=N6r*N4s 
       val[24]=N1r*N5s ;val[25]=N2r*N5s ;val[26]=N3r*N5s ;val[27]=N4r*N5s ;val[28]=N5r*N5s ;val[29]=N6r*N5s 
       val[30]=N1r*N6s ;val[31]=N2r*N6s ;val[32]=N3r*N6s ;val[33]=N4r*N6s ;val[34]=N5r*N6s ;val[35]=N6r*N6s 
   if space=='Q6':
       val = np.zeros(49,dtype=np.float64)
       N1r=  (81*r**6 - 81*r**5 -  45*r**4 + 45*r**3 +  4*r**2 - 4*r)/80 
       N2r= -(27*r**6 - 18*r**5 -  30*r**4 + 20*r**3 +  3*r**2 - 2*r)*9./40
       N3r=  (27*r**6 -  9*r**5 -  39*r**4 + 13*r**3 + 12*r**2 - 4*r)*9./16
       N4r= -(81*r**6           - 126*r**4           + 49*r**2       -4)/4 
       N5r=  (27*r**6 +  9*r**5 -  39*r**4 - 13*r**3 + 12*r**2 + 4*r)*9./16
       N6r= -(27*r**6 + 18*r**5 -  30*r**4 - 20*r**3 +  3*r**2 + 2*r)*9./40 
       N7r=  (81*r**6 + 81*r**5 -  45*r**4 - 45*r**3 +  4*r**2 + 4*r)/80
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







