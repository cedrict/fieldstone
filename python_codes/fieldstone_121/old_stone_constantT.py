import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import math

Rgas = 8.314 # gas constant (J mol^-1 K^-1)

##################################################################

t=800

dmin=10    # grain size (microns)
dmax=1e4   # grain size (microns)

srmin=-18
srmax=-10

print("-----------------------------")
print("---------- stone 121 --------")
print("-----------------------------")
print('sr range=',srmin,srmax)
print('d range=',dmin,dmax)

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

Adis = 1.1e5    ; ndis = 3.5  ;           Edis = 530e3  # Olivine, Hirth & Kohlstedt 2003 
Adiff = 10**7.6 ; ndiff = 1.0 ; mdiff=3 ; Ediff = 370e3 # Olivine, Hirth & Kohlstedt, 2003, corrected in Hansen
Agbs = 6.5e3    ; ngbs = 3.5  ; mgbs=2  ; Egbs = 400e3  # Olivine, Hirth & Kohlstedt, 2003

AlowT = 5.7e11 ; plowT=1; qlowT=2; ElowT=535e3; taulowT=8500; # Goetze, carefull define only for T<700°C!!!
TlowT=700; #set to negative value to never trigger it

##################################################################
# build sr array(s) 
##################################################################

nsr=250

srvals= np.linspace(srmin,srmax,nsr,dtype=np.float64) 
srvals=10**srvals

#print('sr values=',srvals)
#exit()

print('strain rate nvalues=',nsr)

##################################################################
# grain size values array
##################################################################

nd=500
d= np.linspace(np.log10(dmin),np.log10(dmax),nd,dtype=np.float64) 
d=10**d

print('grain size nvalues=',nd)

##################################################################
# Storing arrays
# Create array that stores which mechanism is dominant for a 
# certain temp, tau and d. Last dimension is 4 to store temp, 
# tau, d and dominant mechanism type
##################################################################

dom_mech = np.empty((nsr,nd,5))

##################################################################
# Calculations
##################################################################

#outpuut_all = open('sr_all.ascii', 'w')
#outpuut_all.write('#gs tau mechanism \n') 

for i in range(nsr): #loop on strain rate

    sr=srvals[i]

    #outpuut = open('sr_'+str(int(t))+'.ascii', 'w')
    #outpuut.write('#gs tau T sr_dsl sr_df sr_gbs sr_lowT \n') 

    for j in range(nd): # Loop on grain size
        gs=d[j]

        #print('======================================================')
        
        # Assume all strainrate is produced by each mechanism and calculate stress
        sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*(t+273)))
        sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*(t+273)))
        siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*(t+273)))
        siglowT=taulowT*(1-(-Rgas*(t+273)/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
        
        # Select minimum of stresses as best guess for the optimization procedure 
        sig=min(sigdis,sigdiff,siggbs)
        if t<TlowT: sig=min(sigdis,sigdiff,siggbs,siglowT)

        # Newton-Raphson Loop
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
        tau_NR = optimize.newton(f,sig,args=(sr,gs,t),tol=1e-5, maxiter=100,fprime=None,fprime2=None,disp=True) #,full_output=True)

        # Strain rates for each deformation mechanisms
        # The values are not stored but the code can be adapted to do so
        computed_sr = compute_sr(tau_NR,sr,gs,t) 
        
        # Find position of largest strain rate, this is the dominant mechanism
        # Translation key: dis = 0, diff = 1, gbs = 2, lowT = 3
        max_mech = np.argmax(computed_sr)

        #outpuut.write("%e %e %e %e %e %e %e %d\n" %(gs,tau_NR,t,computed_sr[0],computed_sr[1],computed_sr[2],computed_sr[3],max_mech))
        #outpuut_all.write("%e %e %d\n" %(gs,tau_NR,max_mech))
        
        # Storing of results, along with coordinates for plotting
        dom_mech[i,j,0] = t
        dom_mech[i,j,1] = gs
        dom_mech[i,j,2] = tau_NR
        dom_mech[i,j,3] = max_mech
        dom_mech[i,j,4] = np.log10(sr) #computed_sr[max_mech]

    # end loop on grain size
# end loop on temperature

##################################################################
# same thing but now running this for a few values of strain rate 

nsr00=-srmin+srmax+1

srvals= np.linspace(srmin,srmax,nsr00,dtype=np.float64) 
srvals=10**srvals

dom_mech00 = np.empty((nsr00,nd,5))

for i in range(nsr00): #loop on strain rate
    sr=srvals[i]
    for j in range(nd): # Loop on grain size
        gs=d[j]
        sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*(t+273)))
        sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*(t+273)))
        siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*(t+273)))
        siglowT=taulowT*(1-(-Rgas*(t+273)/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
        sig=min(sigdis,sigdiff,siggbs)
        if t<TlowT: sig=min(sigdis,sigdiff,siggbs,siglowT)
        tau_NR = optimize.newton(f,sig,args=(sr,gs,t),tol=1e-5, maxiter=100,fprime=None,fprime2=None,disp=True) #,full_output=True)
        computed_sr = compute_sr(tau_NR,sr,gs,t) 
        max_mech = np.argmax(computed_sr)
        dom_mech00[i,j,0] = t
        dom_mech00[i,j,1] = gs
        dom_mech00[i,j,2] = tau_NR
        dom_mech00[i,j,3] = max_mech
        dom_mech00[i,j,4] = np.log10(sr)

    # end loop on grain size
# end loop on temperature

##################################################################
# new plotting 
##################################################################

print('gs  (m/M):',np.min(dom_mech[0:nsr,0:nd,1]),np.max(dom_mech[0:nsr,0:nd,1]))
print('tau (m/M):',np.min(dom_mech[0:nsr,0:nd,2]),np.max(dom_mech[0:nsr,0:nd,2]))

#all data
x=np.log10(np.ravel(dom_mech[0:nsr,0:nd,1]))
y=np.log10(np.ravel(dom_mech[0:nsr,0:nd,2]))
colors=np.ravel(dom_mech[0:nsr,0:nd,3])
plt.xlim(1,4)
plt.ylim(0,3.5)
cmap = plt.get_cmap('cividis', 4) 
plt.scatter(x,y,c=colors,cmap=cmap,s=5,vmax=3)

#lines
x=np.log10(np.ravel(dom_mech00[:,0:nd,1]))
y=np.log10(np.ravel(dom_mech00[:,0:nd,2]))
colors=np.ravel(dom_mech00[:,0:nd,4])
cmap = plt.get_cmap('plasma', nsr00) 
plt.scatter(x,y,s=3,c=colors,cmap=cmap) #,vmin=Tmin00-50,vmax=Tmax00+50)
plt.colorbar(ticks=(-18,-17,-16,-15,-14,-13,-12,-11,-10))

plt.text(1.5, 2.25, 'GBS', fontsize="15",c='white')
plt.text(1.5, 0.7, 'DIFF', fontsize="15",c='white')
plt.text(3, 1.8, 'DISL', fontsize="15",c='white')

#if TlowT>0:
#   plt.text(3.5, 3, 'LowT', fontsize="15")

#for i in range(ntemp00):
#    if np.log10(dom_mech00[i,-1,2])<3.5 and np.log10(dom_mech00[i,-1,2])>0 :
#       plt.text(np.log10(dom_mech00[i,-1,1])-0.35,np.log10(dom_mech00[i,-1,2])+0.035,str(i*100+Tmin00)+'C')

plt.xlabel('grain size (microns) - Log scale')
plt.ylabel('Stress (Pa) - Log scale')
plt.title("Olivine Deformation mechanism map, T="+str(t))
#plt.colorbar(ticks=(0.75,1.5,2.25))
plt.savefig('deformation_map3.png',bbox_inches='tight', dpi=200)

plt.show()
plt.close()

##################################################################
# new plotting 
##################################################################

plt.clf()

#all data
x=np.log10(np.ravel(dom_mech[0:nsr,0:nd,1]))
y=np.log10(np.ravel(dom_mech[0:nsr,0:nd,2]))
colors=np.ravel(dom_mech[0:nsr,0:nd,4])
plt.xlim(1,4)
plt.ylim(0,3.5)
cmap = plt.get_cmap('summer') #, 16) 
plt.scatter(x,y,c=colors,cmap=cmap,s=3) #,vmax=3)

#lines
#x=np.log10(np.ravel(dom_mech00[:,0:nd,1]))
#y=np.log10(np.ravel(dom_mech00[:,0:nd,2]))
#colors=np.ravel(dom_mech00[0:ntemp,0:nd,0])
#cmap = plt.get_cmap('plasma', ntemp00) 
#plt.scatter(x,y,s=3,c=colors,cmap=cmap,vmin=Tmin00-50,vmax=Tmax00+50)
#plt.colorbar(ticks=(400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600))

plt.text(1.5, 2.5, 'GBS', fontsize="15")
plt.text(1.5, 0.7, 'DIFF', fontsize="15")
plt.text(3, 1.8, 'DISL', fontsize="15")

#if TlowT>0:
#   plt.text(3.5, 3, 'LowT', fontsize="15")

#for i in range(ntemp00):
#    if np.log10(dom_mech00[i,-1,2])<3.5 and np.log10(dom_mech00[i,-1,2])>0 :
#       plt.text(np.log10(dom_mech00[i,-1,1])-0.35,np.log10(dom_mech00[i,-1,2])+0.035,str(i*100+Tmin00)+'C')

plt.xlabel('grain size (microns) - Log scale')
plt.ylabel('Stress (Pa) - Log scale')
plt.title("Olivine Deformation mech. map, T="+str(t))
#plt.colorbar() #ticks=(0.75,1.5,2.25))
plt.colorbar(ticks=(-18,-17,-16,-15,-14,-13,-12,-11,-10))
plt.savefig('deformation_map4.png',bbox_inches='tight', dpi=200)

plt.show()





exit()

