import numpy as np
import scipy as scp

TKelvin=273. # set to zero to get correct results
R=8.314
Cp=1000
rho=2750
e=1

#brace & kohlstedt 1980
#n=3.
#A=5e-6*1e-18
#Q=1.9e5

#stuwe & sandiford 1994
#n=3.
#A=2e-4*1e-18
#Q=2.5e5

#brace & kohlstedt 1980
#n=3.
#A=7e4*1e-18
#Q=5.2e5

#gleason Tullis 1995
A=1.1e-28
n=4.
Q=223e3

Trange=np.arange(400,1400,1)

###########################################################

def f(T,T0,D,t):
    TT=T+TKelvin
    x=D/TT
    TT0=T0+TKelvin
    x0=D/TT0
    return  t/C/D - ( np.exp(-x)/x -np.exp(-x0)/x0 -scp.special.expi(-x0) +scp.special.expi(-x) ) 

def feq13(T,T0,D,t):
    TT=T+TKelvin
    x=D/TT
    TT0=T0+TKelvin
    x0=D/TT0
    return  t - C*D*( np.exp(-x)/x -np.exp(-x0)/x0 +scp.special.expi(x0) -scp.special.expi(x) ) 

###########################################################

for sr in (1e-12,1e-13,1e-14,1e-15,1e-16):
    D=Q/n/R
    C=rho*Cp/sr*(sr/A)**(-1./n)
    print ('sr=',sr)
    srfile=open('sr'+str(sr)+'.ascii',"w")
    t=e/sr

    for T0 in range (400,1400,1):
        res=f(Trange,T0,D,t)
        # find the zero
        for i in range(0,len(Trange)-1):
            if res[i]*res[i+1]<0: 
               srfile.write("%e %e \n" %(Trange[i],T0))
    
        #np.savetxt('res'+str(T0)+'.ascii',np.array([Trange,res]).T)

    #end for

#end for

###########################################################

sr =1e-14
D=Q/n/R
C=rho*Cp/sr*(sr/A)**(-1./n)

for T0 in (400,500,600):
    res=feq13(Trange,T0,D,t)
    np.savetxt('feq13_'+str(T0)+'.ascii',np.array([Trange,res]).T)


