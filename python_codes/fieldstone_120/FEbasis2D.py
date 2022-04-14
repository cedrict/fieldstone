import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# r is horizontal coordinate
# s is vertical coordinate
# quadrilateral reference space is [-1,1]X[-1,1]
# triangle reference space is lower left of [0,1]X[0,1]
#------------------------------------------------------------------------------
#
# P0, Q1, P1, P1+, Q2, Q2s, P2, P2+, Q3, P3, Q4, DSSY1, DSSY2, RT1, RT2
#
#------------------------------------------------------------------------------
# notes/remarks:
# DSSY1: Douglas, Santos, Sheen and Ye element with theta_1 function
# DSSY2: Douglas, Santos, Sheen and Ye element with theta_2 function
# RT1: mid point variant of Rannacher-Turek (non-conforming element)
# RT2: mid value variant of Rannacher-Turek (non-conforming element)
# Q2s: 8-node serendipity Q2
# P-1: discontinuous pressure over triangle
# Pm1: discontinuous pressure over quadrilateral (mapped)
# Pm1u: discontinuous pressure over quadrilateral (unmapped)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#  C-R       P2P1     MINI    Q2Q1,Q2Pm1    Q2s     P1-NC
#
# 2         2        2         3--6--2    3--6--2   +
# |\        |\       |\        |     |    |     |   |\
# | \       | \      | \       |     |    |     |   | \
# 5  4      5  4     |  \      7  8  5    7     5   2  1
# | 6 \     |   \    | 3 \     |     |    |     |   |   \
# |    \    |    \   |    \    |     |    |     |   |    \
# 0--3--1   0--3--1  0-----1   0--4--1    0--4--1   +--0--+
#
#------------------------------------------------------------------------------

def NNN(r,s,space,**keyword_arguments):

    if space=='Q0' or space=='P0':
       val = np.zeros(1,dtype=np.float64)
       val[0]=1

    if space=='Q1' or space=='Q-1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=0.25*(1.-r)*(1.-s)
       val[1]=0.25*(1.+r)*(1.-s)
       val[2]=0.25*(1.+r)*(1.+s)
       val[3]=0.25*(1.-r)*(1.+s)

    if space=='Q1+':
       val = np.zeros(5,dtype=np.float64)
       B=(1-r**2)*(1-s**2)*(1-r)*(1-s)
       val[0]=0.25*(1-r)*(1-s)-0.25*B
       val[1]=0.25*(1+r)*(1-s)-0.25*B
       val[2]=0.25*(1+r)*(1+s)-0.25*B
       val[3]=0.25*(1-r)*(1+s)-0.25*B
       val[4]=B

    if space=='P1' or space=='P-1' or space=='Pm1':
       val = np.zeros(3,dtype=np.float64)
       val[0]=1-r-s
       val[1]=r
       val[2]=s

    if space=='P1+':
       val = np.zeros(4,dtype=np.float64)
       val[0]=1-r-s-9*(1-r-s)*r*s
       val[1]=  r  -9*(1-r-s)*r*s
       val[2]=    s-9*(1-r-s)*r*s
       val[3]=     27*(1-r-s)*r*s

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

    if space=='Q2s':
       val=np.zeros(8,dtype=np.float64)
       val[0]=(1-r)*(1-s)*(-r-s-1)*0.25
       val[1]=(1+r)*(1-s)*(r-s-1) *0.25
       val[2]=(1+r)*(1+s)*(r+s-1) *0.25
       val[3]=(1-r)*(1+s)*(-r+s-1)*0.25
       val[4]=(1-r**2)*(1-s)*0.5
       val[5]=(1+r)*(1-s**2)*0.5
       val[6]=(1-r**2)*(1+s)*0.5
       val[7]=(1-r)*(1-s**2)*0.5

    if space=='P2':
       val = np.zeros(6,dtype=np.float64)
       val[0]= 1-3*r-3*s+2*r**2+4*r*s+2*s**2 
       val[1]= -r+2*r**2
       val[2]= -s+2*s**2
       val[3]= 4*r-4*r**2-4*r*s
       val[4]= 4*r*s 
       val[5]= 4*s-4*r*s-4*s**2

    if space=='P2+':
       val = np.zeros(7,dtype=np.float64)
       val[0]= (1-r-s)*(1-2*r-2*s+ 3*r*s)
       val[1]= r*(2*r -1 + 3*s-3*r*s-3*s**2 )
       val[2]= s*(2*s -1 + 3*r-3*r**2-3*r*s )
       val[3]= 4*(1-r-s)*r*(1-3*s)
       val[4]= 4*r*s*(-2+3*r+3*s)
       val[5]= 4*(1-r-s)*s*(1-3*r)
       val[6]= 27*(1-r-s)*r*s

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

    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[0]=0.5*(2 -11*r - 11*s + 18*r**2 + 36*r*s + 18*s**2 -9*r**3 -27*r**2*s -27*r*s**2 -9*s**3)
       val[1]=0.5*(18*r-45*r**2-45*r*s +27*r**3 +54*r**2*s+27*r*s**2  )   
       val[2]=0.5*(-9*r+36*r**2+9*r*s -27*r**3 -27*r**2*s  )   
       val[3]=0.5*(2*r-9*r**2+9*r**3  )   
       val[4]=0.5*(18*s -45*r*s-45*s**2+27*r**2*s+54*r*s**2+27*s**3  )   
       val[5]=0.5*(54*r*s-54*r**2*s-54*r*s**2   )   
       val[6]=0.5*(-9*r*s+27*r**2*s   )   
       val[7]=0.5*(-9*s+9*r*s+36*s**2-27*r*s**2-27*s**3  )   
       val[8]=0.5*(-9*r*s+27*r*s**2 ) 
       val[9]=0.5*(2*s-9*s**2+9*s**3  )   

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

    if space=='DSSY1':
       val = np.zeros(4,dtype=np.float64)
       val[3]=0.25-0.5*r+(theta1(r)-theta1(s))/(4*theta1(1))
       val[1]=0.25+0.5*r+(theta1(r)-theta1(s))/(4*theta1(1))
       val[0]=0.25-0.5*s-(theta1(r)-theta1(s))/(4*theta1(1))
       val[2]=0.25+0.5*s-(theta1(r)-theta1(s))/(4*theta1(1))

    if space=='DSSY2':
       val = np.zeros(4,dtype=np.float64)
       val[3]=0.25-0.5*r+(theta2(r)-theta2(s))/(4*theta2(1))
       val[1]=0.25+0.5*r+(theta2(r)-theta2(s))/(4*theta2(1))
       val[0]=0.25-0.5*s-(theta2(r)-theta2(s))/(4*theta2(1))
       val[2]=0.25+0.5*s-(theta2(r)-theta2(s))/(4*theta2(1))

    if space=='RT1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=0.25*(1-r**2-2*s+s**2)
       val[1]=0.25*(1+2*r+r**2-s**2)
       val[2]=0.25*(1-r**2+2*s+s**2)
       val[3]=0.25*(1-2*r+r**2-s**2)

    if space=='RT2':
       val = np.zeros(4,dtype=np.float64)
       val[0]=0.25*(1-2*s-1.5*(r**2-s**2))
       val[1]=0.25*(1+2*r+1.5*(r**2-s**2))
       val[2]=0.25*(1+2*s-1.5*(r**2-s**2))
       val[3]=0.25*(1-2*r+1.5*(r**2-s**2))

    if space=='Han':
       val = np.zeros(5,dtype=np.float64)
       phir=0.5*(5*r**4-3*r**2)
       phis=0.5*(5*s**4-3*s**2)
       val[1]=0.5*(r+phir)
       val[2]=0.5*(s+phis)
       val[3]=-0.5*(r-phir)
       val[0]=-0.5*(s-phis)
       val[4]=1-phir-phis

    if space=='P1NC':
       val = np.zeros(3,dtype=np.float64)
       val[0]=1-2*s
       val[1]=-1+2*r+2*s
       val[2]=1-2*r

    if space=='Pm1u':
       val = np.zeros(3,dtype=np.float64)
       xxP=keyword_arguments.get("xxP", None)
       yyP=keyword_arguments.get("yyP", None)
       xxq=keyword_arguments.get("xxq", None)
       yyq=keyword_arguments.get("yyq", None)
       det=xxP[1]*yyP[2]-xxP[2]*yyP[1]\
          -xxP[0]*yyP[2]+xxP[2]*yyP[0]\
          +xxP[0]*yyP[1]-xxP[1]*yyP[0]
       m11=(xxP[1]*yyP[2]-xxP[2]*yyP[1])/det
       m12=(xxP[2]*yyP[0]-xxP[0]*yyP[2])/det
       m13=(xxP[0]*yyP[1]-xxP[1]*yyP[0])/det
       m21=(yyP[1]-yyP[2])/det
       m22=(yyP[2]-yyP[0])/det
       m23=(yyP[0]-yyP[1])/det
       m31=(xxP[2]-xxP[1])/det
       m32=(xxP[0]-xxP[2])/det
       m33=(xxP[1]-xxP[0])/det
       val[0]=(m11+m21*xxq+m31*yyq)
       val[1]=(m12+m22*xxq+m32*yyq)
       val[2]=(m13+m23*xxq+m33*yyq)

    if space=='P1+P0':
       val = np.zeros(4,dtype=np.float64)
       val[0]=1-r-s
       val[1]=r
       val[2]=s
       val[3]=1

    if space=='Q1+Q0':
       val = np.zeros(5,dtype=np.float64)
       val[0]=0.25*(1.-r)*(1.-s)
       val[1]=0.25*(1.+r)*(1.-s)
       val[2]=0.25*(1.+r)*(1.+s)
       val[3]=0.25*(1.-r)*(1.+s)
       val[4]=1

    if space=='Chen':
       val = np.zeros(5,dtype=np.float64)
       val[0]=0.25*(1-r**2-2*s+s**2)
       val[1]=0.25*(1+2*r+r**2-s**2)
       val[2]=0.25*(1-r**2+2*s+s**2)
       val[3]=0.25*(1-2*r+r**2-s**2)
       val[4]=1-3/4*(r**2+s**2)

    return val

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def dNNNdr(r,s,space):

    if space=='Q0' or space=='P0':
       val = np.zeros(1,dtype=np.float64)
       val[0]=0

    if space=='Q1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=-0.25*(1.-s) 
       val[1]=+0.25*(1.-s) 
       val[2]=+0.25*(1.+s) 
       val[3]=-0.25*(1.+s) 

    if space=='Q1+':
       val = np.zeros(5,dtype=np.float64)
       dBdr=(1-s**2)*(1-s)*(-1-2*r+3*r**2)
       val[0]=-0.25*(1.-s)-0.25*dBdr
       val[1]=+0.25*(1.-s)-0.25*dBdr
       val[2]=+0.25*(1.+s)-0.25*dBdr
       val[3]=-0.25*(1.+s)-0.25*dBdr
       val[4]=dBdr

    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[0]=-1
       val[1]= 1
       val[2]= 0

    if space=='P1+':
       val = np.zeros(4,dtype=np.float64)
       val[0]= -1-9*(1-2*r-s)*s
       val[1]=  1-9*(1-2*r-s)*s
       val[2]=   -9*(1-2*r-s)*s
       val[3]=   27*(1-2*r-s)*s

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

    if space=='Q2s':
       val=np.zeros(8,dtype=np.float64)
       val[0]= -0.25*(s-1)*(2*r+s)
       val[1]= -0.25*(s-1)*(2*r-s)
       val[2]= 0.25*(s+1)*(2*r+s)
       val[3]= 0.25*(s+1)*(2*r-s)
       val[4]= r*(s-1)
       val[5]= 0.5*(1-s**2)
       val[6]= -r*(s+1)
       val[7]= -0.5*(1-s**2)

    if space=='P2':
       val=np.zeros(6,dtype=np.float64)
       val[0]= -3+4*r+4*s
       val[1]= -1+4*r
       val[2]= 0
       val[3]= 4-8*r-4*s
       val[4]= 4*s
       val[5]= -4*s

    if space=='P2+':
       val=np.zeros(7,dtype=np.float64)
       val[0]= r*(4-6*s)-3*s**2+7*s-3
       val[1]= r*(4-6*s)-3*s**2+3*s-1
       val[2]= -3*s*(2*r+s-1)
       val[3]= 4*(3*s-1)*(2*r+s-1)
       val[4]= 4*s*(6*r+3*s-2)
       val[5]= 4*s*(6*r+3*s-4)
       val[6]=-27*s*(2*r+s-1)

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

    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[0]=0.5*(-11+36*r+36*s-27*r**2-54*r*s-27*s**2)
       val[1]=0.5*(18-90*r-45*s+81*r**2+108*r*s+27*s**2)
       val[2]=0.5*(-9+72*r+9*s-81*r**2-54*r*s)
       val[3]=0.5*(2-18*r+27*r**2)
       val[4]=0.5*(-45*s+54*r*s+54*s**2)
       val[5]=0.5*(54*s-108*r*s-54*s**2)
       val[6]=0.5*(-9*s+54*r*s)
       val[7]=0.5*(9*s-27*s**2)
       val[8]=0.5*(-9*s+27*s**2)
       val[9]=0.

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
       val[ 0]=dN1dr*N1s ; val[ 1]=dN2dr*N1s ; val[ 2]=dN3dr*N1s ; val[ 3]=dN4dr*N1s ; val[ 4]=dN5dr*N1s
       val[ 5]=dN1dr*N2s ; val[ 6]=dN2dr*N2s ; val[ 7]=dN3dr*N2s ; val[ 8]=dN4dr*N2s ; val[ 9]=dN5dr*N2s
       val[10]=dN1dr*N3s ; val[11]=dN2dr*N3s ; val[12]=dN3dr*N3s ; val[13]=dN4dr*N3s ; val[14]=dN5dr*N3s
       val[15]=dN1dr*N4s ; val[16]=dN2dr*N4s ; val[17]=dN3dr*N4s ; val[18]=dN4dr*N4s ; val[19]=dN5dr*N4s
       val[20]=dN1dr*N5s ; val[21]=dN2dr*N5s ; val[22]=dN3dr*N5s ; val[23]=dN4dr*N5s ; val[24]=dN5dr*N5s

    if space=='DSSY1':
       val = np.zeros(4,dtype=np.float64)
       val[3]=-0.5+theta1p(r)/(4*theta1(1))
       val[1]=+0.5+theta1p(r)/(4*theta1(1))
       val[0]=    -theta1p(r)/(4*theta1(1))
       val[2]=    -theta1p(r)/(4*theta1(1))

    if space=='DSSY2':
       val = np.zeros(4,dtype=np.float64)
       val[3]=-0.5+theta2p(r)/(4*theta2(1))
       val[1]=+0.5+theta2p(r)/(4*theta2(1))
       val[0]=    -theta2p(r)/(4*theta2(1))
       val[2]=    -theta2p(r)/(4*theta2(1))

    if space=='RT1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=0.5*(-r)  
       val[1]=0.5*(1+r) 
       val[2]=0.5*(-r)  
       val[3]=0.5*(-1+r)

    if space=='RT2':
       val = np.zeros(4,dtype=np.float64)
       val[0]=-0.75*r     
       val[1]=0.5+0.75*r  
       val[2]=-0.75*r     
       val[3]=-0.5+0.75*r 

    if space=='Han':
       val = np.zeros(5,dtype=np.float64)
       dphidr=0.5*(20*r**3-6*r)
       val[1]=0.5*(1+dphidr)
       val[2]=0
       val[3]=-0.5*(1-dphidr)
       val[0]=0
       val[4]=-dphidr

    if space=='P1NC':
       val = np.zeros(3,dtype=np.float64)
       val[0]=0
       val[1]=2
       val[2]=-2

    if space=='Chen':
       val = np.zeros(5,dtype=np.float64)
       val[0]=0.5*(-r)  
       val[1]=0.5*(1+r) 
       val[2]=0.5*(-r)  
       val[3]=0.5*(-1+r)
       val[4]=-1.5*r


    return val 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def dNNNds(r,s,space):

    if space=='Q0' or space=='P0':
       val = np.zeros(1,dtype=np.float64)
       val[0]=0

    if space=='Q1':
       val=np.zeros(4,dtype=np.float64)
       val[0]=-0.25*(1.-r)
       val[1]=-0.25*(1.+r)
       val[2]=+0.25*(1.+r)
       val[3]=+0.25*(1.-r)

    if space=='Q1+':
       val = np.zeros(5,dtype=np.float64)
       dBds=(1-r**2)*(1-r)*(-1-2*s+3*s**2)
       val[0]=-0.25*(1.-r)-0.25*dBds
       val[1]=-0.25*(1.+r)-0.25*dBds
       val[2]=+0.25*(1.+r)-0.25*dBds
       val[3]=+0.25*(1.-r)-0.25*dBds
       val[4]=dBds

    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[0]=-1
       val[1]= 0
       val[2]= 1

    if space=='P1+':
       val=np.zeros(4,dtype=np.float64)
       val[0]=-1-9*(1-r-2*s)*r
       val[1]=  -9*(1-r-2*s)*r
       val[2]= 1-9*(1-r-2*s)*r
       val[3]=  27*(1-r-2*s)*r

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

    if space=='Q2s':
       val=np.zeros(8,dtype=np.float64)
       val[0]= -0.25*(r-1)*(r+2*s)
       val[1]= -0.25*(r+1)*(r-2*s)
       val[2]= 0.25*(r+1)*(r+2*s)
       val[3]= 0.25*(r-1)*(r-2*s)
       val[4]= -0.5*(1-r**2)
       val[5]= -(r+1)*s
       val[6]= 0.5*(1-r**2)
       val[7]= (r-1)*s

    if space=='P2':
       val=np.zeros(6,dtype=np.float64)
       val[0]= -3+4*r+4*s
       val[1]= 0
       val[2]= -1+4*s
       val[3]= -4*r
       val[4]= +4*r
       val[5]= 4-4*r-8*s

    if space=='P2+':
       val=np.zeros(7,dtype=np.float64)
       val[0]= -3*r**2+r*(7-6*s)+4*s-3
       val[1]= -3*r*(r+2*s-1)
       val[2]= -3*r**2+r*(3-6*s)+4*s-1
       val[3]= 4*r*(3*r+6*s-4)
       val[4]= 4*r*(3*r+6*s-2)
       val[5]= 4*(3*r-1)*(r+2*s-1)
       val[6]= -27*r*(r+2*s-1)

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

    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[0]=0.5*(-11+36*r+36*s-27*r**2-54*r*s-27*s**2)
       val[1]=0.5*(-45*r+54*r**2+54*r*s)
       val[2]=0.5*(9*r-27*r**2)
       val[3]=0.
       val[4]=0.5*(18-45*r-90*s+27*r**2+108*r*s+81*s**2)
       val[5]=0.5*(54*r-54*r**2-108*r*s)
       val[6]=0.5*(-9*r+27*r**2)
       val[7]=0.5*(-9+9*r+72*s-54*r*s-81*s**2)
       val[8]=0.5*(-9*r+54*r*s)
       val[9]=0.5*(2-18*s+27*s**2)

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
       val[ 0]=N1r*dN1ds ; val[ 1]=N2r*dN1ds ; val[ 2]=N3r*dN1ds ; val[ 3]=N4r*dN1ds ; val[ 4]=N5r*dN1ds
       val[ 5]=N1r*dN2ds ; val[ 6]=N2r*dN2ds ; val[ 7]=N3r*dN2ds ; val[ 8]=N4r*dN2ds ; val[ 9]=N5r*dN2ds
       val[10]=N1r*dN3ds ; val[11]=N2r*dN3ds ; val[12]=N3r*dN3ds ; val[13]=N4r*dN3ds ; val[14]=N5r*dN3ds
       val[15]=N1r*dN4ds ; val[16]=N2r*dN4ds ; val[17]=N3r*dN4ds ; val[18]=N4r*dN4ds ; val[19]=N5r*dN4ds
       val[20]=N1r*dN5ds ; val[21]=N2r*dN5ds ; val[22]=N3r*dN5ds ; val[23]=N4r*dN5ds ; val[24]=N5r*dN5ds

    if space=='DSSY1':
       val = np.zeros(4,dtype=np.float64)
       val[3]=    -theta1p(s)/(4*theta1(1))
       val[1]=    -theta1p(s)/(4*theta1(1))
       val[0]=-0.5+theta1p(s)/(4*theta1(1))
       val[2]=+0.5+theta1p(s)/(4*theta1(1))

    if space=='DSSY2':
       val = np.zeros(4,dtype=np.float64)
       val[3]=    -theta2p(s)/(4*theta2(1))
       val[1]=    -theta2p(s)/(4*theta2(1))
       val[0]=-0.5+theta2p(s)/(4*theta2(1))
       val[2]=+0.5+theta2p(s)/(4*theta2(1))

    if space=='RT1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=0.5*(-1+s)
       val[1]=0.5*(-s)
       val[2]=0.5*(1+s)
       val[3]=0.5*(-s)

    if space=='RT2':
       val = np.zeros(4,dtype=np.float64)
       val[0]=-0.5+0.75*s
       val[1]=-0.75*s
       val[2]=0.5+0.75*s
       val[3]=-0.75*s

    if space=='Han':
       val = np.zeros(5,dtype=np.float64)
       dphids=0.5*(20*s**3-6*s)
       val[1]=0
       val[2]=0.5*(1+dphids)
       val[3]=0
       val[0]=-0.5*(1-dphids)
       val[4]=-dphids

    if space=='P1NC':
       val = np.zeros(3,dtype=np.float64)
       val[0]=-2
       val[1]=2
       val[2]=0

    if space=='Chen':
       val = np.zeros(5,dtype=np.float64)
       val[0]=0.5*(-1+s)
       val[1]=0.5*(-s)
       val[2]=0.5*(1+s)
       val[3]=0.5*(-s)
       val[4]=-1.5*s

    return val 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def NNN_r(space):

    if space=='Q0':
       val = np.zeros(1,dtype=np.float64)
       val[0]=0

    if space=='P0':
       val = np.zeros(1,dtype=np.float64)
       val[0]=1/3

    if space=='Q1' or space=='Q-1':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[-1,1,1,-1]

    if space=='Q1+' or space=='Q1+Q0':
       val = np.zeros(5,dtype=np.float64)
       val[:]=[-1,1,1,-1,0]

    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[:]=[0,1,0]

    if space=='P1+' or space=='P1+P0':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[0,1,0,1/3]

    if space=='P2':
       val = np.zeros(6,dtype=np.float64)
       val[:]=[0,1,0,0.5,0.5,0]

    if space=='P2+':
       val = np.zeros(7,dtype=np.float64)
       val[:]=[0,1,0,0.5,0.5,0,1./3.]

    if space=='Q2':
       val = np.zeros(9,dtype=np.float64)
       val[:]=[-1,+1,+1,-1,0,+1,0,-1,0]

    if space=='Q2s':
       val = np.zeros(8,dtype=np.float64)
       val[:]=[-1,+1,+1,-1,0,+1,0,-1]

    if space=='Q3':
       val = np.zeros(16,dtype=np.float64)
       val[:]=[-1,-1/3,+1/3,+1,-1,-1/3,+1/3,+1,-1,-1/3,+1/3,+1,-1,-1/3,+1/3,+1]

    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[:]=[0,1/3,2/3,1,0,1/3,2/3,0,1/3,0]

    if space=='Q4':
       val = np.zeros(25,dtype=np.float64)
       val[:]=[-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1]

    if space=='DSSY1' or space=='DSSY2':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[0,1,0,-1]

    if space=='RT1' or space=='RT2':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[0,1,0,-1]

    if space=='Han' or space=='Chen':
       val = np.zeros(5,dtype=np.float64)
       val[:]=[0,1,0,-1,0]

    if space=='P1NC':
       val = np.zeros(3,dtype=np.float64)
       val[:]=[0.5,0.5,0]

    return val

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def NNN_s(space):

    if space=='Q0':
       val = np.zeros(1,dtype=np.float64)
       val[0]=0

    if space=='P0':
       val = np.zeros(1,dtype=np.float64)
       val[0]=1/3

    if space=='Q1' or space=='Q-1':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[-1,-1,1,1]

    if space=='Q1+' or space=='Q1+Q0':
       val = np.zeros(5,dtype=np.float64)
       val[:]=[-1,-1,1,1,0]

    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[:]=[0,0,1]

    if space=='P1+' or space=='P1+P0':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[0,0,1,1/3]

    if space=='P2':
       val = np.zeros(6,dtype=np.float64)
       val[:]=[0,0,1,0,0.5,0.5]

    if space=='P2+':
       val = np.zeros(7,dtype=np.float64)
       val[:]=[0,0,1,0,0.5,0.5,1/3]

    if space=='Q2':
       val = np.zeros(9,dtype=np.float64)
       val[:]=[-1,-1,+1,+1,-1,0,+1,0,0]

    if space=='Q2s':
       val = np.zeros(8,dtype=np.float64)
       val[:]=[-1,-1,+1,+1,-1,0,+1,0]

    if space=='Q3':
       val = np.zeros(16,dtype=np.float64)
       val[:]=[-1,-1,-1,-1,-1/3,-1/3,-1/3,-1/3,+1/3,+1/3,+1/3,+1/3,+1,+1,+1,+1]

    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[:]=[0,0,0,0,1/3,1/3,1/3,2/3,2/3,1]

    if space=='Q4':
       val = np.zeros(25,dtype=np.float64)
       val[:]=[-1,-1,-1,-1,-1,-0.5,-0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]

    if space=='DSSY1' or space=='DSSY2':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[-1,0,1,0]

    if space=='RT1' or space=='RT2':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[-1,0,1,0]

    if space=='Han' or space=='Chen':
       val = np.zeros(5,dtype=np.float64)
       val[:]=[-1,0,1,0,0]

    if space=='P1NC':
       val = np.zeros(3,dtype=np.float64)
       val[:]=[0,0.5,0.5]

    return val

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def NNN_m(space):

    if space=='Q0':     return 1
    if space=='P0':     return 1
    if space=='Q1':     return 4
    if space=='Q-1':    return 4
    if space=='Q1+':    return 5
    if space=='P1':     return 3
    if space=='P1NC':   return 3
    if space=='P-1':    return 3
    if space=='Pm1':    return 3
    if space=='Pm1u':   return 3
    if space=='P1+':    return 4
    if space=='P1+P0':  return 4
    if space=='Q1+Q0':  return 5
    if space=='Q2':     return 9
    if space=='Q2s':    return 8
    if space=='P2':     return 6
    if space=='P2+':    return 7
    if space=='P3':     return 10
    if space=='Q3':     return 16
    if space=='Q4':     return 25
    if space=='DSSY1':  return 4
    if space=='DSSY2':  return 4
    if space=='RT1':    return 4
    if space=='RT2':    return 4
    if space=='Han':    return 5
    if space=='Chen':   return 5

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def mapping(Vspace):
    if Vspace=='Q0' or Vspace=='Q1'  or Vspace=='Q2'  or Vspace=='Q3'  or\
       Vspace=='Q4' or Vspace=='Q1+' or Vspace=='Q2s' or Vspace=='Han' or\
       Vspace=='DSSY1' or Vspace=='DSSY2' or Vspace=='RT1' or Vspace=='RT2' or\
       Vspace=='Chen':
       return 'Q1'
    else:
       return 'P1' 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def visualise_nodes(space):
    r=NNN_r(space)
    s=NNN_s(space)
    plt.figure()
    plt.scatter(r,s,color='teal',s=80)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.25)
    plt.xlabel('r')
    plt.xlabel('s')
    plt.title(space)
    if space=='Q1' or space=='Q2' or space=='Q3' or space=='Q4' or space=='Q1+' or \
       space=='DSSY1' or space=='DSSY2' or space=='RT1' or space=='RT2' or space=='Q2s' or\
       space=='Han' or space=='Chen':
       plt.xlim([-1.1,+1.1])
       plt.ylim([-1.1,+1.1])
       plt.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],color='teal',linewidth=2)
    elif space=='P1' or space=='P2' or space=='P1+' or space=='P2+' or space=='P3' or space=='P1NC':
       plt.xlim([-0.1,+1.1])
       plt.ylim([-0.1,+1.1])
       plt.plot([0,0,1,0],[0,1,0,0],color='teal',linewidth=2)
    else:
       exit('visualise_nodes: unknown space')
    plt.savefig(space+'_nodes.pdf',bbox_inches='tight')
    print('     -> generated '+space+'_nodes.pdf')
    plt.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def visualise_basis_functions(space):
    rmin=-1
    rmax=+1
    smin=-1
    smax=+1
    zz = np.zeros((NNN_m(space),100,100),dtype=np.float64)
    rr,ss=np.meshgrid(np.arange(rmin,rmax,0.01), np.arange(smin,smax, 0.01))
    for i in range (100):
        for j in range (100):
            zz[0:NNN_m(space),i,j]=NNN(rr[i,j],ss[i,j],space)
            if space[0]=='P' and i+j>100: zz[:,i,j]=0

    plt.figure()
    #plt.colorbar()
    for k in range(NNN_m(space)):
        plt.imshow(zz[k,:,:],extent=[rmin,rmax,smin,smax], cmap=cm.RdBu ,origin='lower')
        plt.title("basis function #"+str(k), fontsize=8)
        plt.savefig(space+'_basis_function'+str(k)+'.png', bbox_inches='tight')
    plt.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# theta1 and theta2 functions for DSSY element

def theta1(x):
    return x*x-5/3*x**4

def theta1p(x):
    return 2*x-20/3*x**3

def theta2(x):
    return x*x-25/6*x**4+3.5*x**6

def theta2p(x):
    return 2*x-50/3*x**3+21*x**5

#------------------------------------------------------------------------------
