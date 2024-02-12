import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import math

print("-----------------------------")
print("---------- stone 121 --------")
print("-----------------------------")

Rgas = 8.314 # J mol^-1 K^-1

##############################################################################
# TOOLS
###############################################################################

#  definition of newton raphson function
# f(x)=sr - (sr_dis+sr_diff+sr_gbs+sr_lowT) with x= shear stress

def f(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*(T+273)))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*(T+273)))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*(T+273)))* x**ngbs * gs**(-mgbs)
    sr_lowT=0
    if T<= TlowT: sr_lowT=AlowT*np.exp(-ElowT/(Rgas*(T+273)) * (1-(x/taulowT)**plowT)**qlowT)
    val=sr-sr_dis-sr_diff-sr_gbs-sr_lowT
    return val

# same function but with strain rate for each mechanism as output

def compute_sr(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*(T+273)))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*(T+273)))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*(T+273)))* x**ngbs * gs**(-mgbs)
    sr_lowT=0.
    if T<= TlowT:sr_lowT=AlowT*np.exp(-ElowT/(Rgas*(T+273)) * (1-(x/taulowT)**plowT)**qlowT)
    return sr_dis,sr_diff,sr_gbs,sr_lowT


###############################################################################
# Rheological parameters (A in MPa^-n s^-1, Q in J/mol)
###############################################################################

Adis = 1.1e5    ; ndis = 3.5  ;           Edis = 530e3  # Olivine, Hirth & Kohlstedt 2003 
Adiff = 10**7.6 ; ndiff = 1.0 ; mdiff=3 ; Ediff = 370e3 # Olivine, Hirth & Kohlstedt, 2003, corrected in Hansen
Agbs = 6.5e3    ; ngbs = 3.5  ; mgbs=2  ; Egbs = 400e3  # Olivine, Hirth & Kohlstedt, 2003

AlowT = 5.7e11 ; plowT=1; qlowT=2; ElowT=535e3; taulowT=8500; # Goetze, carefull define only for T<700°C!!!
TlowT=700; #set to negative value to never trigger it

###############################################################################
# bounds 
###############################################################################

newton_tol=1e-5

dmin=10    # grain size (microns)
dmax=1e4   # grain size (microns)
nd=500     # number of grain size values

print('grain size range=',dmin,dmax)
print('grain size nvalues=',nd)

constant_strainrate=True
   
print('constant_strainrate=',constant_strainrate)

if constant_strainrate:
   sr=1e-15   # strain rate (s^-1)
   Tmin=380   # temperature (C) 
   Tmax=1610  # temperature (C)
   dT=2       # temperature interval  ( integer >= 1)
else:
   t=800      # temperature in Celsius
   srmin=-18  # strain rate
   srmax=-10  # strain rate
   nsr=250
   print('sr range=',srmin,srmax)

###############################################################################
# grain size values array
###############################################################################

d= np.linspace(np.log10(dmin),np.log10(dmax),nd,dtype=np.float64) 
d=10**d

###############################################################################
# build temperature or strainr rate  array(s) 
###############################################################################

if constant_strainrate:

   temp= np.arange(Tmin,Tmax,dT,dtype=np.float64) 
   ntemp=len(temp)
   print('dT,nvalues=',dT,ntemp)

else:

   srvals= np.linspace(srmin,srmax,nsr,dtype=np.float64) 
   srvals=10**srvals

   print('strain rate nvalues=',nsr)

###############################################################################
# Storing arrays
# Create array that stores which mechanism is dominant for a 
# certain temp, tau and d. Last dimension is 4 to store temp, 
# tau, d and dominant mechanism type
###############################################################################

if constant_strainrate:

   dom_mech=np.empty((ntemp,nd,5))
   lines_Tmax=math.floor(Tmax/100)*100
   lines_Tmin=math.ceil(Tmin/100)*100
   nlines=int((lines_Tmax-lines_Tmin)/100+1)
   lines_Tvals=np.linspace(lines_Tmin,lines_Tmax,nlines,dtype=np.float64) 
   lines_dom_mech=np.empty((nlines,nd,5))
   print('lines:',lines_Tvals)

else:

   dom_mech=np.empty((nsr,nd,5))
   nlines=-srmin+srmax+1
   lines_srvals=np.linspace(srmin,srmax,nlines,dtype=np.float64) 
   lines_srvals=10**lines_srvals
   lines_dom_mech=np.empty((nlines,nd,5))
   print('lines:',lines_srvals)

###############################################################################
# Calculations
###############################################################################

if constant_strainrate:

   for i in range(ntemp): # Loop on temperature
       t=temp[i]

       #outpuut = open('sr_'+str(int(t))+'.ascii', 'w')
       #outpuut.write('#gs tau T sr_dsl sr_df sr_gbs sr_lowT \n') 

       for j in range(nd): # Loop on grain size
           gs=d[j]
        
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
           tau_NR = optimize.newton(f,sig,args=(sr,gs,t),tol=newton_tol, maxiter=100,disp=True)

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
           dom_mech[i,j,4] = computed_sr[max_mech]

       # end loop on grain size
   # end loop on temperature

else:
   
   for i in range(nsr): #loop on strain rate

       sr=srvals[i]

       #outpuut = open('sr_'+str(int(t))+'.ascii', 'w')
       #outpuut.write('#gs tau T sr_dsl sr_df sr_gbs sr_lowT \n') 

       for j in range(nd): # Loop on grain size
           gs=d[j]
        
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
           tau_NR = optimize.newton(f,sig,args=(sr,gs,t),tol=newton_tol, maxiter=100,disp=True)

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

###############################################################################
# re-run for a selection of T or sr values - for line plotting purposes
###############################################################################

if constant_strainrate:

   for i in range(nlines): # Loop on temperature
       t=lines_Tvals[i]
       for j in range(nd): # Loop on grain size
           gs=d[j]
           sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*(t+273)))
           sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*(t+273)))
           siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*(t+273)))
           siglowT=taulowT*(1-(-Rgas*(t+273)/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
           sig=min(sigdis,sigdiff,siggbs)
           if t<TlowT: sig=min(sigdis,sigdiff,siggbs,siglowT)
           tau_NR = optimize.newton(f,sig,args=(sr,gs,t),tol=newton_tol, maxiter=100,disp=True)
           computed_sr = compute_sr(tau_NR,sr,gs,t) 
           max_mech = np.argmax(computed_sr)
           lines_dom_mech[i,j,0] = t
           lines_dom_mech[i,j,1] = gs
           lines_dom_mech[i,j,2] = tau_NR
           lines_dom_mech[i,j,3] = max_mech
           lines_dom_mech[i,j,4] = computed_sr[max_mech]
       # end loop on grain size
   # end loop on temperature

else:
   
   for i in range(nlines): #loop on strain rate
       sr=lines_srvals[i]
       for j in range(nd): # Loop on grain size
           gs=d[j]
           sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*(t+273)))
           sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*(t+273)))
           siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*(t+273)))
           siglowT=taulowT*(1-(-Rgas*(t+273)/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
           sig=min(sigdis,sigdiff,siggbs)
           if t<TlowT: sig=min(sigdis,sigdiff,siggbs,siglowT)
           tau_NR = optimize.newton(f,sig,args=(sr,gs,t),tol=newton_tol, maxiter=100,disp=True)
           computed_sr = compute_sr(tau_NR,sr,gs,t) 
           max_mech = np.argmax(computed_sr)
           lines_dom_mech[i,j,0] = t
           lines_dom_mech[i,j,1] = gs
           lines_dom_mech[i,j,2] = tau_NR
           lines_dom_mech[i,j,3] = max_mech
           lines_dom_mech[i,j,4] = np.log10(sr)
       # end loop on grain size
   # end loop on temperature


###############################################################################
# produce plots with matplotlib
###############################################################################

if constant_strainrate:

   x=np.log10(np.ravel(dom_mech[0:ntemp,0:nd,1]))
   y=np.log10(np.ravel(dom_mech[0:ntemp,0:nd,2]))
   colors=np.ravel(dom_mech[0:ntemp,0:nd,3])
   plt.xlim(1,4)
   plt.ylim(0,3.5)
   cmap = plt.get_cmap('cividis', 4) 
   plt.scatter(x,y,c=colors,cmap=cmap,s=5,vmax=3)

   #lines
   x=np.log10(np.ravel(lines_dom_mech[:,0:nd,1]))
   y=np.log10(np.ravel(lines_dom_mech[:,0:nd,2]))
   colors=np.ravel(lines_dom_mech[0:nlines,0:nd,0])
   cmap = plt.get_cmap('plasma', nlines) 
   plt.scatter(x,y,s=3,c=colors,cmap=cmap,vmin=lines_Tmin-50,vmax=lines_Tmax+50)
   plt.colorbar(ticks=(400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600),label='Temperature (C)')

   plt.text(1.5, 2.5, 'GBS', fontsize="15",c='white')
   plt.text(1.5, 0.7, 'DIFF', fontsize="15",c='white')
   plt.text(3, 1.8, 'DISL', fontsize="15",c='white')

   if TlowT>0: plt.text(3.5, 3, 'LowT', fontsize="15",c='white')

   #for i in range(nlines):
   #    if np.log10(lines_dom_mech[i,-1,2])<3.5 and np.log10(lines_dom_mech[i,-1,2])>0 :
   #       plt.text(np.log10(lines_dom_mech[i,-1,1])-0.35,np.log10(lines_dom_mech[i,-1,2])+0.035,str(i*100+lines_Tmin)+'C')

   plt.xlabel('grain size (microns) - Log scale')
   plt.ylabel('Stress (Pa) - Log scale')
   plt.title("Olivine Deformation mechanism map, sr="+str(sr))
   plt.savefig('constant_strainrate_deformation_map1.png',bbox_inches='tight', dpi=200)

   plt.show()
   plt.close()
 
   ##################################################################

   plt.clf()

   x=np.log10(np.ravel(dom_mech[0:ntemp,0:nd,1]))
   y=np.log10(np.ravel(dom_mech[0:ntemp,0:nd,2]))
   colors=np.ravel(dom_mech[0:ntemp,0:nd,4])
   plt.xlim(1,4)
   plt.ylim(0,3.5)
   cmap = plt.get_cmap('plasma')
   plt.scatter(x,y,c=colors,cmap=cmap,s=5) #,vmax=3)

   plt.text(1.5, 2.5, 'GBS', fontsize="15")
   plt.text(1.5, 0.7, 'DIFF', fontsize="15")
   plt.text(3, 1.8, 'DISL', fontsize="15")

   if TlowT>0: plt.text(3.5, 3, 'LowT', fontsize="15")

   plt.xlabel('grain size (microns) - Log scale')
   plt.ylabel('Stress (Pa) - Log scale')
   plt.title("Olivine Deformation mech. map, sr="+str(sr))
   plt.colorbar(label='strain rate (s^-1)') #ticks=(0.75,1.5,2.25))
   plt.savefig('constant_strainrate_deformation_map2.png',bbox_inches='tight', dpi=200)

   plt.show()

else: # constant temperature

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
   x=np.log10(np.ravel(lines_dom_mech[:,0:nd,1]))
   y=np.log10(np.ravel(lines_dom_mech[:,0:nd,2]))
   colors=np.ravel(lines_dom_mech[:,0:nd,4])
   cmap = plt.get_cmap('plasma', nlines) 
   plt.scatter(x,y,s=3,c=colors,cmap=cmap,vmin=-18.5,vmax=-9.5)
   plt.colorbar(ticks=(-18,-17,-16,-15,-14,-13,-12,-11,-10), label='strain rate (s^-1)')

   plt.text(1.5, 2.25, 'GBS', fontsize="15",c='white')
   plt.text(1.5, 0.7, 'DIFF', fontsize="15",c='white')
   plt.text(3, 1.8, 'DISL', fontsize="15",c='white')
   #if TlowT>0: plt.text(3.5, 3, 'LowT', fontsize="15")

   plt.xlabel('grain size (microns) - Log scale')
   plt.ylabel('Stress (Pa) - Log scale')
   plt.title("Olivine Deformation mechanism map, T="+str(t))
   plt.savefig('constant_temperature_deformation_map3.png',bbox_inches='tight',dpi=200)

   plt.show()
   plt.close()

   ##################################################################

   plt.clf()

   x=np.log10(np.ravel(dom_mech[0:nsr,0:nd,1]))
   y=np.log10(np.ravel(dom_mech[0:nsr,0:nd,2]))
   colors=np.ravel(dom_mech[0:nsr,0:nd,4])
   plt.xlim(1,4)
   plt.ylim(0,3.5)
   cmap = plt.get_cmap('plasma')
   plt.scatter(x,y,c=colors,cmap=cmap,s=3)

   plt.text(1.5, 2.5, 'GBS', fontsize="15")
   plt.text(1.5, 0.7, 'DIFF', fontsize="15")
   plt.text(3, 1.8, 'DISL', fontsize="15")
   #if TlowT>0: plt.text(3.5, 3, 'LowT', fontsize="15")

   plt.xlabel('grain size (microns) - Log scale')
   plt.ylabel('Stress (Pa) - Log scale')
   plt.title("Olivine Deformation mech. map, T="+str(t))
   plt.colorbar(ticks=(-18,-17,-16,-15,-14,-13,-12,-11,-10), label='strain rate (s^-1)')
   plt.savefig('constant_temperature_deformation_map4.png',bbox_inches='tight',dpi=200)


   plt.show()





exit()



