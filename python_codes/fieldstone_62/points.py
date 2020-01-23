
import numpy as np

delta=14e3/np.sin(35./180.*np.pi)

Lx=3000.e3 
Ly=670.e3

xB=Lx/2.
yB=Ly

xA=xB-delta
yA=Ly

xD=xB-82e3/np.tan(35./180.*np.pi)  
yD=588./670.*Ly

xC=xD-delta
yC=588./670.*Ly

xE=1530./3000.*Lx
yE=560./670.*Ly

xG=xB-8e3/np.tan(35./180.*np.pi) 
yG=662./670.*Ly

xF=xA-7e3/np.tan(35./180.*np.pi) 
yF=663./670.*Ly

xI=xB-40e3/np.tan(35./180.*np.pi) 
yI=630./670.*Ly

xH=xA-39e3/np.tan(35./180.*np.pi) 
yH=631/670.*Ly

xJ=1550e3
yJ=yE

xK=1650e3
yK=yI

xL=1700e3
yL=yG

xM=1700e3
yM=Ly

xN=1320e3
yN=yC

xO=1320e3
yO=yH

xP=1320e3
yP=yF

xQ=1320e3
yQ=Ly


xS=0
yS=yP

xT=0
yT=yO

xU=0
yU=yN


xX=Lx
yX=yL

xY=Lx
yY=yK

xZ=Lx
yZ=yJ

x0=0
y0=0
x1=Lx
y1=0
x2=Lx
y2=Ly
x3=0
y3=Ly

x5=xA+(xB-xA)/3. 
y5=Ly
x6=xC+(xD-xC)/3.
y6=yC

x7=xA+(xB-xA)/3.*2 
y7=Ly
x8=xC+(xD-xC)/3.*2
y8=yC





#print(xA,yA)
#print(xB,yB)
#print(xC,yC)
#print(xD,yD)
#print(xE,yE)
#print(xF,yF)
#print(xG,yG)
#print(xH,yH)
#print(xI,yI)
#print(xJ,yJ)
#print(xK,yK)
#print(xL,yL)
#print(xM,yM)
#print(xN,yN)
#print(xO,yO)
#print(xP,yP)
#print(xQ,yQ)



