import numpy as np
import random

def B(r,s):
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)
    elif bubble==2:
       return (1-r**2)*(1-s**2)*(1+beta*(r+s))
    else:
       return (1-r**2)*(1-s**2)

def dBdr(r,s):
    if bubble==1:
       return (1-s**2)*(1-s)*(-1-2*r+3*r**2)
    elif bubble==2:
       return (s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))
    else:
       return (-2*r)*(1-s**2)

def dBds(r,s):
    if bubble==1:
       return (1-r**2)*(1-r)*(-1-2*s+3*s**2) 
    elif bubble==2:
       return (r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)
    else:
       return (1-r**2)*(-2*s)

def NNV(r,s):
    NV_0= 0.25*(1-r)*(1-s) - 0.25*B(r,s)
    NV_1= 0.25*(1+r)*(1-s) - 0.25*B(r,s)
    NV_2= 0.25*(1+r)*(1+s) - 0.25*B(r,s)
    NV_3= 0.25*(1-r)*(1+s) - 0.25*B(r,s)
    NV_4= B(r,s)
    return NV_0,NV_1,NV_2,NV_3,NV_4

#------------------------------------------------------------------------------
mV=5
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
xV    = np.zeros(mV,dtype=np.float64)           # shape functions V
yV    = np.zeros(mV,dtype=np.float64)           # shape functions V
bubble=1

xV[0]=1 ; yV[0]=1
xV[1]=2 ; yV[1]=1
xV[2]=2.5 ; yV[2]=2
xV[3]=0.5 ; yV[3]=2

xV[4]=0.25*(xV[0]+xV[1]+xV[2]+xV[3]) 
yV[4]=0.25*(yV[0]+yV[1]+yV[2]+yV[3]) 


for i in range(0,10):
    r=0#random.uniform(-1.,+1)
    s=0#random.uniform(-1.,+1)
    NNNV[0:mV]=NNV(r,s)
    xq=0
    yq=0
    for k in range(0,mV):
        xq+=NNNV[k]*xV[k]
        yq+=NNNV[k]*yV[k]
    print (xq,yq)




