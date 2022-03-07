import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# r is horizontal coordinate
# s is vertical coordinate
# quadrilateral reference space is [-1,1]X[-1,1]
# triangle referemce space is lower left of [0,1]X[0,1]
#------------------------------------------------------------------------------
#
# P0, Q1, P1, P1+, Q2, P2, P2+, Q3, Q4
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def NNN(r,s,space):

    if space=='Q0' or space=='P0':
       val = np.zeros(1,dtype=np.float64)
       val[0]=1

    if space=='Q1':
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

    if space=='P1':
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
       val[0]=4.5*(1-r-s)*(1/3-r-s)*(2/3-r-s)
       val[1]=4.5*r*(r-1/3)*(r-2/3)
       val[2]=4.5*s*(s-1/3)*(s-2/3)
       val[3]=13.5*(1-r-s)*r*(2/3-r-s)
       val[4]=13.5*(1-r-s)*r*(r-1/3)
       val[5]=13.5*r*s*(r-1/3)
       val[6]=13.5*r*s*(r-2/3)
       val[7]=13.5*(1-r-s)*s*(s-1/3)
       val[8]=13.5*(1-r-s)*s*(2/3-r-s)
       val[9]=27*r*s*(1-r-s)

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
       val = np.zeros(3,dtype=np.float64)
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
       val=np.zeros(3,dtype=np.float64)
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

    if space=='Q1':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[-1,1,1,-1]

    if space=='Q1+':
       val = np.zeros(5,dtype=np.float64)
       val[:]=[-1,1,1,-1,0]

    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[:]=[0,1,0]

    if space=='P1+':
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
       val[:]=[-1,+1,+1,-1, 0,+1, 0,-1,0]

    if space=='Q3':
       val = np.zeros(16,dtype=np.float64)
       val[:]=[-1,-1/3,+1/3,+1,-1,-1/3,+1/3,+1,-1,-1/3,+1/3,+1,-1,-1/3,+1/3,+1]

    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[:]=[0,1,0,1/3,2/3,2/3,1/3,0,0,1/3]

    if space=='Q4':
       val = np.zeros(25,dtype=np.float64)
       val[:]=[-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1]

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

    if space=='Q1':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[-1,-1,1,1]

    if space=='Q1+':
       val = np.zeros(5,dtype=np.float64)
       val[:]=[-1,-1,1,1,0]

    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[:]=[0,0,1]

    if space=='P1+':
       val = np.zeros(4,dtype=np.float64)
       val[:]=[0,0,1,1/3]

    if space=='P2':
       val = np.zeros(6,dtype=np.float64)
       val[:]=[0,0,1,0,0.5,0.5]

    if space=='P2+':
       val = np.zeros(7,dtype=np.float64)
       val[:]=[0,0,1,0,0.5,0.5,1./3.]

    if space=='Q2':
       val = np.zeros(9,dtype=np.float64)
       val[:]=[-1,-1,+1,+1,-1,0,+1,0,0]

    if space=='Q3':
       val = np.zeros(16,dtype=np.float64)
       val[:]=[-1,-1,-1,-1,-1/3,-1/3,-1/3,-1/3,+1/3,+1/3,+1/3,+1/3,+1,+1,+1,+1]

    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[:]=[0,0,1,0,0,1/3,2/3,2/3,1/3,1/3] 

    if space=='Q4':
       val = np.zeros(25,dtype=np.float64)
       val[:]=[-1,-1,-1,-1,-1,-0.5,-0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]

    return val

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def NNN_m(space):

    if space=='Q0':  return 1
    if space=='P0':  return 1
    if space=='Q1':  return 4
    if space=='Q1+': return 5
    if space=='P1':  return 3
    if space=='P1+': return 4
    if space=='Q2':  return 9
    if space=='P2':  return 6
    if space=='P2+': return 7
    if space=='P3':  return 10
    if space=='Q3':  return 16
    if space=='Q4':  return 25

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def visualise_nodes(space):
    r=NNN_r(space)
    s=NNN_s(space)
    plt.figure()
    plt.scatter(r,s,color='teal',s=30)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.25)
    plt.xlabel('r')
    plt.xlabel('s')
    plt.title(space)
    if space=='Q1' or space=='Q2' or space=='Q3' or space=='Q4' or space=='Q1+':
       plt.xlim([-1.1,+1.1])
       plt.ylim([-1.1,+1.1])
       plt.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],color='teal',linewidth=2)
    elif space=='P1' or space=='P2' or space=='P1+' or space=='P2+' or space=='P3':
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
    zz = np.zeros((NNN_m(space),40,40),dtype=np.float64)
    rr,ss=np.meshgrid(np.arange(rmin,rmax,0.05), np.arange(smin,smax, 0.05))
    for i in range (40):
        for j in range (40):
            zz[0:NNN_m(space),i,j]=NNN(rr[i,j],ss[i,j],space)

    plt.figure()
    #plt.colorbar()
    for k in range(NNN_m(space)):
        plt.imshow(zz[k,:,:],extent=[rmin,rmax,smin,smax], cmap=cm.jet, origin='lower')
        plt.title("basis function #"+str(k), fontsize=8)
        plt.savefig(space+'_basis_function'+str(k)+'.png', bbox_inches='tight')
    plt.close()

