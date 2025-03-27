import numpy as np
import scipy as scp

TKelvin=273. # set to zero to get 'correct' results (see text)
R=8.314
Cp=1000
rho=2750
e=1 # desired strain value

#no melt gltu95
#A=1.1 10^(-4 pm 2) MPa 
#Q=223 pm 56
#n=4 pm 0.9

Trange=np.arange(400,1400,1)

experiment='2c'

###########################################################

def f(T,T0,D,t):
    TT=T+TKelvin
    x=D/TT
    TT0=T0+TKelvin
    x0=D/TT0
    return  t/C/D - ( np.exp(-x)/x -np.exp(-x0)/x0 -scp.special.expi(-x0) +scp.special.expi(-x) ) 

###########################################################

if experiment=='1':
   for sr in (1e-12,1e-13,1e-14,1e-15,1e-16):
       srfile=open('sr'+str(sr)+'.ascii',"w")
       t=e/sr
       for Q in range(223-56,223+56):
           for n in (4-0.9,4,4+0.9):
               print ('sr=',sr,'Q=',Q,'n=',n)
               for a in range(-2,2):
                   A=1.1*10**(-4+a-6*n)
                   D=(Q*1e3)/n/R
                   C=rho*Cp/sr*(sr/A)**(-1./n)
                   for T0 in range (400,1400,1):
                       res=f(Trange,T0,D,t)
                       # find the zero
                       for i in range(0,len(Trange)-1):
                           if res[i]*res[i+1]<0: 
                              srfile.write("%e %e \n" %(Trange[i],T0))
                       #end for
                   #end for
               #end for
           #end for
       #end for
   #end for
#end if


###########################################################

if experiment=='2a':
    sr=1e-14
    srfile=open('sr'+str(sr)+'.ascii',"w")
    t=e/sr
    a=0
    n=4

    for Q in range(223-56,223+56):
        A=1.1*10**(-4+a-6*n)
        D=(Q*1e3)/n/R
        C=rho*Cp/sr*(sr/A)**(-1./n)
        for T0 in range (400,1400,1):
            res=f(Trange,T0,D,t)
            # find the zero
            for i in range(0,len(Trange)-1):
                if res[i]*res[i+1]<0: 
                   srfile.write("%e %e \n" %(Trange[i],T0))
            #end for
        #end for
    #end for

#end for

###########################################################

if experiment=='2b':
    sr=1e-14
    srfile=open('sr'+str(sr)+'.ascii',"w")
    t=e/sr
    a=0
    Q=223

    for n in (4-0.9,4,4+0.9):
        A=1.1*10**(-4+a-6*n)
        D=(Q*1e3)/n/R
        C=rho*Cp/sr*(sr/A)**(-1./n)
        for T0 in range (400,1400,1):
            res=f(Trange,T0,D,t)
            # find the zero
            for i in range(0,len(Trange)-1):
                if res[i]*res[i+1]<0: 
                   srfile.write("%e %e \n" %(Trange[i],T0))
            #end for
        #end for
    #end for

#end for

###########################################################


if experiment=='2c':
    sr=1e-14
    srfile=open('sr'+str(sr)+'.ascii',"w")
    t=e/sr
    Q=223
    n=4

    for a in range(-2,2):
        A=1.1*10**(-4+a-6*n)
        D=(Q*1e3)/n/R
        C=rho*Cp/sr*(sr/A)**(-1./n)
        for T0 in range (400,1400,1):
            res=f(Trange,T0,D,t)
            # find the zero
            for i in range(0,len(Trange)-1):
                if res[i]*res[i+1]<0: 
                   srfile.write("%e %e \n" %(Trange[i],T0))
            #end for
        #end for
    #end for

#end for

###########################################################

