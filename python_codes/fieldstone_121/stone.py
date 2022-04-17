import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

Rgas = 8.314 # gas constant (J mol^-1 K^-1)

##################################################################

sr = 1e-15 # strain rate (s^-1)

Tmin=380   # temperature (C) 
Tmax=1300  # temperature (C)

dmin=10    # grain size (microns)
dmax=1e4   # grain size (microns)

##################################################################
# definition of newton raphson function
# f(x)=sr - (sr_dis+sr_diff+sr_gbs+sr_lowT) with x= shear stress
##################################################################

def f(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*(T+273)))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*(T+273)))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*(T+273)))* x**ngbs * gs**(-mgbs)
    sr_lowT=0
    if T<= TlowT: sr_lowT=AlowT*np.exp(-ElowT/(Rgas*(T+273)) * (1-(x/taulowT)**plowT)**qlowT)
    val=sr-sr_dis-sr_diff-sr_gbs-sr_lowT
    return val

##################################################################
# same function but with strain rate for each mechanism as output
##################################################################

def compute_sr(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*(T+273)))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*(T+273)))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*(T+273)))* x**ngbs * gs**(-mgbs)
    sr_lowT=0.
    if T<= TlowT:sr_lowT=AlowT*np.exp(-ElowT/(Rgas*(T+273)) * (1-(x/taulowT)**plowT)**qlowT)
    return sr_dis,sr_diff,sr_gbs,sr_lowT

##################################################################
# Rheological parameters (A in MPa^-n s^-1, Q in J/mol)
##################################################################

Adis = 1.1e5   ; ndis = 3.5  ;           Edis = 530e3  # Olivine, Hirth & Kohlstedt 2003 
Adiff = 10**7.6 ; ndiff = 1.0 ; mdiff=3 ; Ediff = 370e3 # Olivine, Hirth & Kohlstedt, 2003, corrected in Hansen
Agbs = 6.5e3   ; ngbs = 3.5  ; mgbs=2  ; Egbs = 400e3  # Olivine, Hirth & Kohlstedt, 2003

AlowT = 5.7e11 ; plowT=1; qlowT=2; ElowT=535e3; taulowT=8500; # Goetze, carefull define only for T<700°C!!!
TlowT=700;

#AlowT = 1e6;plowT=0.5; qlowT=2; ElowT=450e3;  taulowT=1.5e4; # Demouchy, it is not working!!!
#TlowT=800;

##################################################################
# build temperature array(s) 
##################################################################
dT=2.5
temp= np.arange(Tmin,Tmax,dT,dtype=np.float64) 
ntemp=len(temp)

ipllabel=[400,500,600,700,800,900,1000,1100,1200,1300]

##################################################################
# grain size values array
##################################################################

nd=500
d= np.linspace(np.log10(dmin),np.log10(dmax),nd,dtype=np.float64) 
d=10**d

###################################################################
# allocate arrays for the boundaries between deformation mechanisms

taubdis=np.zeros(ntemp,dtype=np.float64)
dbdis=np.zeros(ntemp,dtype=np.float64)
taubdiff=np.zeros(ntemp,dtype=np.float64)
dbdiff=np.zeros(ntemp,dtype=np.float64)
taubgbs=np.zeros(ntemp,dtype=np.float64)
dbgbs=np.zeros(ntemp,dtype=np.float64)
taublT=np.zeros(ntemp,dtype=np.float64)
dblT=np.zeros(ntemp,dtype=np.float64)

###################################################################
# pour definir les courbes contraintes/tailles de grain à différentes temperatures

tau_NR=np.zeros(nd,dtype=np.float64)

plt.title("Olivine Deformation mechanism map")
# Loop on temperature
ipl=0
ifl=0
for i in range(len(temp)):
    t=temp[i]
    if t>=ipllabel[ipl]: 
        ipl=ipl+1
        ifl=1
    # Loop on grain size
    for j in range(len(d)):
        gs=d[j]
        sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*(t+273)))
        sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*(t+273)))
        siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*(t+273)))
        siglowT=taulowT*(1-(-Rgas*(t+273)/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
        sig=min(sigdis,sigdiff,siggbs)
        if t<TlowT: sig=min(sigdis,sigdiff,siggbs,siglowT)

        # NewtonRaphson Loop
        tau_NR[j] = optimize.newton(f,sig,args=(sr,gs,t),tol=1e-3, maxiter=10,fprime=None,fprime2=None)
        #Strain rates for each deformation mechanisms
        sr_dis,sr_diff,sr_gbs,sr_lowT=compute_sr(sig,sr,gs,t)
        srmax=max(sr_dis,sr_diff,sr_gbs,sr_lowT)
        # Define the field limits
        if sr_dis < srmax :
            taubdis[i]=tau_NR[j]
            dbdis[i]=d[j]
        if sr_diff>=srmax :
            taubdiff[i]=tau_NR[j]
            dbdiff[i]=d[j]
        if sr_gbs >=srmax:
            taubgbs[i]=tau_NR[j]
            dbgbs[i]=d[j]
        if sr_lowT < srmax:
            taublT[i]=tau_NR[j]
            dblT[i]=d[j]
    # end loop on grain size
    # plotting stress/grain size curves, with label each 100°C
    if i%8==0:
       plt.plot(d,tau_NR,'k',linewidth=0.5)
    if ifl==1:
        plt.plot(d,tau_NR,linewidth=1.5,label='Temp='+str(int(t))+"°C")
        ifl=0
# end loop on temperature

# plotting limits between fields
plt.plot(dbdiff,taubdiff,'--',linewidth=1.5,label='Equil Diff')
#plt.plot(dbgbs,taubgbs,'--',linewidth=1.5,label='Equil GBS')
plt.plot(dbdis,taubdis,'--',linewidth=1.5,label='Equil Dis')
plt.plot(dblT,taublT,'--',linewidth=1.5,label='Equil Low T')
plt.xlabel('Grain size (microns)')
plt.ylabel('Stress (MPa)')
plt.yscale('log')
plt.xscale('log')
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=0)
#plt.legend()
plt.grid(True)
plt.axis([dmin,dmax, 1, 2e3])
plt.savefig('deformation_map.pdf', bbox_inches='tight')
plt.show()
