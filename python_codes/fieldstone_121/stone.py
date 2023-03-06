import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

Rgas = 8.314 # gas constant (J mol^-1 K^-1)

##################################################################

sr = 1e-16 # strain rate (s^-1)

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

##################################################################
# build temperature array(s) 
##################################################################
dT=2.5
temp= np.arange(Tmin,Tmax,dT,dtype=np.float64) 
ntemp=len(temp)

##################################################################
# grain size values array
##################################################################

nd=500
d= np.linspace(np.log10(dmin),np.log10(dmax),nd,dtype=np.float64) 
d=10**d

##################################################################
# Storing arrays
##################################################################

# Create array that stores which mechanism is dominant for a certain temp, tau and d
# Last dimension is 4 to store temp, tau, d and dominant mechanism type
dom_mech = np.empty((ntemp,nd,4))

#%%
##################################################################
# Calculations
##################################################################

# Loop on temperature
for i in range(len(temp)):
    t=temp[i]
    # Loop on grain size
    for j in range(nd):
        gs=d[j]
        
        # Assume all strainrate is produced by each mechanism and calculate stress
        sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*(t+273)))
        sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*(t+273)))
        siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*(t+273)))
        siglowT=taulowT*(1-(-Rgas*(t+273)/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
        
        # Select minimum of stresses as best guess for the optimization procedure 
        sig=min(sigdis,sigdiff,siggbs)
        if t<TlowT: sig=min(sigdis,sigdiff,siggbs,siglowT)

        # NewtonRaphson Loop
        tau_NR = optimize.newton(f,sig,args=(sr,gs,t),tol=1e-3, maxiter=10,fprime=None,fprime2=None)
        
        # Strain rates for each deformation mechanisms
        # The values are not stored but the code can be adapted to do so
        computed_sr = compute_sr(tau_NR,sr,gs,t) 
        
        # Find position of largest strain rate, this is the dominant mechanism
        # Translation key: dis = 0, diff = 1, gbs = 2, lowT = 3
        max_mech = np.argmax(computed_sr)
        
        # Storing of results, along with coordinates for plotting
        dom_mech[i,j,0] = t
        dom_mech[i,j,1] = gs
        dom_mech[i,j,2] = tau_NR
        dom_mech[i,j,3] = max_mech
    # end loop on grain size
# end loop on temperature

#%%
##################################################################
# Plotting
##################################################################

def plotDefMechMap(filledareas=True):
    # Initiate subplot
    fig, ax = plt.subplots()
    ax.set_title("Olivine Deformation mechanism map")
    
    # Plotting stress/grain size curves, with label each 100°C
    
    # Set counter for label setting
    ipl=0
    
    # Set temperatures that will be labeled
    ipllabel=[400,500,600,700,800,900,1000,1100,1200,1300]
    
    for i in range(ntemp):
        # Plot an isotherm every 20 degrees
        if i%8==0:
            ax.plot(d,dom_mech[i,:,2],'k',linewidth=0.5)
            
        # Plot a colored and labeled isotherm on predetermined levels
        if temp[i]>=ipllabel[ipl]: 
            ipl=ipl+1
            ax.plot(d,dom_mech[i,:,2],linewidth=1.5,label='T = '+str(int(temp[i]))+" °C")
    
    # Create legend for isotherms and place it to the right of the plot
    legend1 = ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    ax.add_artist(legend1)
    
    # Ravel resulting 2D array for each parameter to linear array for plotting
    dom_mech_gs_ravel = np.ravel(dom_mech[:,:,1])
    dom_mech_tau_ravel = np.ravel(dom_mech[:,:,2])
    dom_mech_type_ravel = np.ravel(dom_mech[:,:,3])
    
    if filledareas:
        # Use tricontourf for our irregular grid to create filled contour plot for areas
        fcont = ax.tricontourf(dom_mech_gs_ravel,dom_mech_tau_ravel,dom_mech_type_ravel,\
                            cmap='Pastel2',levels=range(7))
        
        # Set labels for the areas
        mech_labels = ['Disloc', 'Diff','GBS','Low T']
        
        # Create proxy artist for a custom legend
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
                 for pc in fcont.collections]
        
        # Create second legend and place it beneath the temperature legend
        legend2 = ax.legend(proxy, mech_labels, bbox_to_anchor=(1.01, 0.45), loc='upper left', borderaxespad=0)
        ax.add_artist(legend2)
        
    else:
        # Use tricontour for our irregular grid to create contour plot for boundaries
        ax.tricontour(dom_mech_gs_ravel,dom_mech_tau_ravel,dom_mech_type_ravel,colors='k',linewidths=1)
        
        # Place mechanism names in areas
        # NOTE: This is very problem specific and hard coded
        ax.text(0.75, 0.4,'Disloc',transform=ax.transAxes,fontsize=15)
        ax.text(0.2, 0.2,'Diff',transform=ax.transAxes,fontsize=15)
        ax.text(0.2, 0.7,'GBS',transform=ax.transAxes,fontsize=15)
        ax.text(0.75, 0.9,'Low T',transform=ax.transAxes,fontsize=15)
    
    
    # Further basic layout
    ax.set_xlabel('Grain size (microns)')
    ax.set_ylabel('Stress (MPa)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlim([dmin,dmax])
    ax.set_ylim([1, 2e3])
    # Save pdf
    if filledareas:
        fig.savefig('deformation_map_filledareas.pdf', bbox_inches='tight')
    else:
        fig.savefig('deformation_map_boundaries.pdf', bbox_inches='tight')

plotDefMechMap()
plotDefMechMap(filledareas=False)
