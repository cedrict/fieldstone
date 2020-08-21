import numpy as np
import matplotlib.pyplot as plt

########################################################################################
# disl + diff + visco-plastic
########################################################################################

print ("+++++ Setup +++++")

Y=20e6
eta_v=1e25
eta_m=1e20

nnz=301
neps=301
Rgas=8.314
Tkelvin=273.15

grain_size = 1.e-3
A_df     = 8.7e15/(8.e10)*(0.5e-9/grain_size)**2.5
Q_df     = 300.e3
V_df     = 5e-6

A_ds     = 2.417e-16
n_ds     = 3.5 
Q_ds     = 540.e3
V_ds     = 20e-6

Ly=660.e3
rho=3300.
gy=9.81

zA=Ly      ; TA= 20.+Tkelvin
zB=Ly-30e3 ; TB= 550.+Tkelvin
zC=Ly-90e3 ; TC=1330.+Tkelvin
zD=0.      ; TD=1380.+Tkelvin

y=np.zeros(nnz,dtype=np.float64)
p=np.zeros(nnz,dtype=np.float64)
T=np.zeros(nnz,dtype=np.float64)
srT=np.zeros(neps,dtype=np.float64)
for j in range(0,neps):
    srT[j]=-18+(-12+18)/(neps-1)*j
    for i in range(0,nnz):
        y[i]=i*Ly/(nnz-1)
        p[i]=rho*gy*(Ly-y[i])
        if y[i] > Ly-30.e3:
           T[i] = (TA-TB)/(zA-zB)*(y[i]-zB)+TB
        elif y[i] > Ly-90.e3:
           T[i] = (TB-TC)/(zB-zC)*(y[i]-zC)+TC
        else:
           T[i] = (TC-TD)/(zC-zD)*(y[i]-zD)+TD

np.savetxt('T.ascii',np.array([y,T]).T,header='# y,T')
np.savetxt('p.ascii',np.array([y,p]).T,header='# y,p')

########################################
print ("+++++ Declare all arrays +++++")

eta_df_old=np.zeros((nnz,neps),dtype=np.float64)
eta_ds_old=np.zeros((nnz,neps),dtype=np.float64)
eta_eff_old=np.zeros((nnz,neps),dtype=np.float64)
eta_vp_old=np.zeros((nnz,neps),dtype=np.float64)
tau_old=np.zeros((nnz,neps),dtype=np.float64)
sr_ds_old=np.zeros((nnz,neps),dtype=np.float64)
sr_df_old=np.zeros((nnz,neps),dtype=np.float64)
sr_v_old=np.zeros((nnz,neps),dtype=np.float64)
sr_pl_old=np.zeros((nnz,neps),dtype=np.float64)

eta_df_new=np.zeros((nnz,neps),dtype=np.float64)
eta_ds_new=np.zeros((nnz,neps),dtype=np.float64)
eta_eff_new=np.zeros((nnz,neps),dtype=np.float64)
tau_new=np.zeros((nnz,neps),dtype=np.float64)
sr_ds_new=np.zeros((nnz,neps),dtype=np.float64)
sr_df_new=np.zeros((nnz,neps),dtype=np.float64)
sr_v_new=np.zeros((nnz,neps),dtype=np.float64)
sr_pl_new=np.zeros((nnz,neps),dtype=np.float64)

eta_df_diff=np.zeros((nnz,neps),dtype=np.float64)
eta_ds_diff=np.zeros((nnz,neps),dtype=np.float64)
eta_eff_diff=np.zeros((nnz,neps),dtype=np.float64)
tau_diff=np.zeros((nnz,neps),dtype=np.float64)
sr_ds_diff=np.zeros((nnz,neps),dtype=np.float64)
sr_df_diff=np.zeros((nnz,neps),dtype=np.float64)
sr_v_diff=np.zeros((nnz,neps),dtype=np.float64)

isplast_new=np.zeros((nnz,neps),dtype=np.float64)
isplast_old=np.zeros((nnz,neps),dtype=np.float64)

#################OLD WAY###########################################################################

########################################
print ("+++++ Open files +++++")

mfile_old=open("maps_old_visc.ascii","w")
mfile_new=open("maps_new_visc.ascii","w")
mfile_diff=open("maps_diff_visc.ascii","w")

pfile_old=open("profile_old_visc.ascii","w")
pfile_new=open("profile_new_visc.ascii","w")
pfile_diff=open("profile_diff_visc.ascii","w")


########################################
print ("+++++ Only viscous dampers in series +++++")

for j in range(0,neps):
    sr_T=10**srT[j]
    for i in range(0,nnz):

        ##### OLD #####################
        eta_df_old[i,j]=0.5 / A_df * np.exp((Q_df+p[i]*V_df)/Rgas/T[i])
        eta_ds_old[i,j]=0.5 * A_ds**(-1./n_ds) * sr_T**(1./n_ds-1.) * np.exp((Q_ds+p[i]*V_ds)/n_ds/Rgas/T[i])
        eta_eff_old[i,j]=1./(1./eta_ds_old[i,j]+1./eta_df_old[i,j] + 1./eta_v)
        tau_old[i,j]=2*sr_T*eta_eff_old[i,j]

        if tau_old[i,j]>Y:
           isplast_old[i,j]=1

        sr_ds_old[i,j]=A_ds * tau_old[i,j]**n_ds * np.exp(-(Q_ds+p[i]*V_ds)/Rgas/T[i])
        sr_df_old[i,j]=A_df * tau_old[i,j]       * np.exp(-(Q_df+p[i]*V_df)/Rgas/T[i])
        sr_v_old[i,j]=0.5*tau_old[i,j]/eta_v
        mfile_old.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e %3e\n" 
                        %(sr_T,y[i],eta_df_old[i,j],eta_ds_old[i,j],eta_eff_old[i,j],
                        tau_old[i,j],sr_df_old[i,j],sr_ds_old[i,j],sr_v_old[i,j],isplast_old[i,j]))

        if np.abs(srT[j]+15)<1e-6:
           pfile_old.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e \n" 
                           %(sr_T,y[i],eta_df_old[i,j],eta_ds_old[i,j],eta_eff_old[i,j],
                           tau_old[i,j],sr_df_old[i,j],sr_ds_old[i,j],sr_v_old[i,j]))

        ##### NEW #####################
        # compute tau
        tau=1.e6       # guess at 1MPa ?
        iter=0
        func=1
        while abs(func/sr_T) > 1.e-8:
              iter=iter+1
              sr_ds=A_ds * tau**n_ds * np.exp(-(Q_ds+p[i]*V_ds)/Rgas/T[i])
              sr_df=A_df * tau       * np.exp(-(Q_df+p[i]*V_df)/Rgas/T[i])
              sr_v=0.5*tau/eta_v
              func=sr_T-sr_ds-sr_df-sr_v
              funcp=-A_ds * n_ds * tau**(n_ds-1) * np.exp(-(Q_ds+p[i]*V_ds)/Rgas/T[i])\
                    -A_df                        * np.exp(-(Q_df+p[i]*V_df)/Rgas/T[i])\
                    -0.5/eta_v
              tau=tau - func/funcp
        #end while
        tau_new[i,j]=tau
        if tau_new[i,j]>Y:
           isplast_new[i,j]=1
        sr_ds_new[i,j]=A_ds * tau_new[i,j]**n_ds * np.exp(-(Q_ds+p[i]*V_ds)/Rgas/T[i])
        sr_df_new[i,j]=A_df * tau_new[i,j]       * np.exp(-(Q_df+p[i]*V_df)/Rgas/T[i])
        sr_v_new[i,j]=0.5*tau_new[i,j]/eta_v
        eta_df_new[i,j]=0.5/A_df * np.exp((Q_df+p[i]*V_df)/Rgas/T[i])
        eta_ds_new[i,j]=0.5*A_ds**(-1./n_ds)*sr_ds_new[i,j]**(1./n_ds-1)*np.exp((Q_ds+p[i]*V_ds)/n_ds/Rgas/T[i])
        eta_eff_new[i,j]=1./(1./eta_ds_new[i,j]+1./eta_df_new[i,j] + 1./eta_v)
        mfile_new.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e %3e\n" 
                        %(sr_T,y[i],eta_df_new[i,j],eta_ds_new[i,j],eta_eff_new[i,j],
                        tau_new[i,j],sr_df_new[i,j],sr_ds_new[i,j],sr_v_new[i,j],isplast_new[i,j]))

        if np.abs(srT[j]+15)<1e-6:
           pfile_new.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e \n" 
                           %(sr_T,y[i],eta_df_new[i,j],eta_ds_new[i,j],eta_eff_new[i,j],
                           tau_new[i,j],sr_df_new[i,j],sr_ds_new[i,j],sr_v_new[i,j]))

        ##### DIFF ####################
        eta_df_diff[i,j] =eta_df_new[i,j]-eta_df_old[i,j]
        eta_ds_diff[i,j] =eta_ds_new[i,j]-eta_ds_old[i,j]
        eta_eff_diff[i,j]=(eta_eff_new[i,j]-eta_eff_old[i,j])/eta_eff_new[i,j]
        sr_ds_diff[i,j] =sr_ds_new[i,j]-sr_ds_old[i,j]
        sr_df_diff[i,j] =sr_df_new[i,j]-sr_df_old[i,j]
        sr_v_diff[i,j]  =sr_v_new[i,j]-sr_v_old[i,j]
        tau_diff[i,j]   =(tau_new[i,j]-tau_old[i,j])/tau_new[i,j]
        mfile_diff.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e \n" 
                         %(sr_T,y[i],eta_df_diff[i,j],eta_ds_diff[i,j],eta_eff_diff[i,j],
                         tau_diff[i,j],sr_df_diff[i,j],sr_ds_diff[i,j],sr_v_diff[i,j]))
        if np.abs(srT[j]+15)<1e-6:
           pfile_diff.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e \n" 
                            %(sr_T,y[i],eta_df_diff[i,j],eta_ds_diff[i,j],eta_eff_diff[i,j],
                            tau_diff[i,j],sr_df_diff[i,j],sr_ds_diff[i,j],sr_v_diff[i,j]))

    #end for
    mfile_old.write("  \n")
    mfile_new.write("  \n")
    mfile_diff.write("  \n")
#end for
mfile_old.close()
mfile_new.close()
mfile_diff.close()

####################################################################################################
# adding visco-plastic element (v + ds + df + vp)
####################################################################################################

########################################
print ("+++++ Open files +++++")

mfile_old=open("maps_old_viscpl.ascii","w")
mfile_new=open("maps_new_viscpl.ascii","w")
mfile_diff=open("maps_diff_viscpl.ascii","w")

pfile_old=open("profile_old_viscpl.ascii","w")
pfile_new=open("profile_new_viscpl.ascii","w")
pfile_diff=open("profile_diff_viscpl.ascii","w")

########################################
print ("+++++ Viscous dampers and viscoplastic element +++++")

for j in range(0,neps):
    sr_T=10**srT[j]
    for i in range(0,nnz):

        ##### OLD #####################
        # compute eta_df, eta_ds with eps_T
        # compute <eta> for viscous processes
        # compute eta_pl with eps_T

        eta_df_old[i,j]=0.5 / A_df * np.exp((Q_df+p[i]*V_df)/Rgas/T[i])
        eta_ds_old[i,j]=0.5 * A_ds**(-1./n_ds) * sr_T**(1./n_ds-1.) * np.exp((Q_ds+p[i]*V_ds)/n_ds/Rgas/T[i])
        eta_vp_old[i,j]=0.5*Y/sr_T + eta_m

        eta_visc=1./(1./eta_ds_old[i,j]+1./eta_df_old[i,j] + 1./eta_v)

        if 2.*eta_visc*sr_T < Y :
           eta_eff_old[i,j]=eta_visc
        else:
           eta_eff_old[i,j]=eta_vp_old[i,j]

        tau_old[i,j]=2*sr_T*eta_eff_old[i,j]

        sr_ds_old[i,j]=A_ds * tau_old[i,j]**n_ds * np.exp(-(Q_ds+p[i]*V_ds)/Rgas/T[i])
        sr_df_old[i,j]=A_df * tau_old[i,j]       * np.exp(-(Q_df+p[i]*V_df)/Rgas/T[i])
        sr_v_old[i,j]=0.5*tau_old[i,j]/eta_v
        mfile_old.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e %3e\n" 
                        %(sr_T,y[i],eta_df_old[i,j],eta_ds_old[i,j],eta_eff_old[i,j],
                        tau_old[i,j],sr_df_old[i,j],sr_ds_old[i,j],sr_v_old[i,j],isplast_old[i,j]))

        if np.abs(srT[j]+15)<1e-6:
           pfile_old.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e \n" 
                           %(sr_T,y[i],eta_df_old[i,j],eta_ds_old[i,j],eta_eff_old[i,j],
                           tau_old[i,j],sr_df_old[i,j],sr_ds_old[i,j],sr_v_old[i,j]))


        ##### NEW #####################
        if tau_new[i,j]>Y:
           # only if visc stresses higher than Y
           tau=0.#.e6    
           iter=0
           func=1
           print('-------------------')
           while abs(func/sr_T) > 1.e-6:
                 sr_ds=A_ds * tau**n_ds * np.exp(-(Q_ds+p[i]*V_ds)/Rgas/T[i])
                 sr_df=A_df * tau       * np.exp(-(Q_df+p[i]*V_df)/Rgas/T[i])
                 sr_v=0.5*tau/eta_v
                 func=Y+2.*(sr_T-sr_ds-sr_df-sr_v)*eta_m-tau
                 funcp=(-2.*A_ds * n_ds * tau**(n_ds-1) * np.exp(-(Q_ds+p[i]*V_ds)/Rgas/T[i])\
                        -2.*A_df                        * np.exp(-(Q_df+p[i]*V_df)/Rgas/T[i])\
                        -1./eta_v)*eta_m - 1.
                 tau-=func/funcp
                 #print(sr_T,y[i],tau/Y,func,funcp)
                 iter=iter+1
                 if iter>25: 
                    print('=====>',sr_T,y[i],tau/Y,func,funcp)
                    break
           #end while
           #print(iter)
           tau_new[i,j]=tau

           sr_ds_new[i,j]=A_ds * tau_new[i,j]**n_ds * np.exp(-(Q_ds+p[i]*V_ds)/Rgas/T[i])
           sr_df_new[i,j]=A_df * tau_new[i,j]       * np.exp(-(Q_df+p[i]*V_df)/Rgas/T[i])
           sr_v_new[i,j]=0.5*tau_new[i,j]/eta_v
           sr_pl_new[i,j]=sr_T-sr_ds-sr_df-sr_v

           eta_df_new[i,j]=0.5/A_df * np.exp((Q_df+p[i]*V_df)/Rgas/T[i])
           eta_ds_new[i,j]=0.5*A_ds**(-1./n_ds)*sr_ds_new[i,j]**(1./n_ds-1)*np.exp((Q_ds+p[i]*V_ds)/n_ds/Rgas/T[i])
           eta_pl=0.5*tau/sr_pl_new[i,j]+eta_m
           eta_eff_new[i,j]=1./(1./eta_ds_new[i,j]+1./eta_df_new[i,j] + 1./eta_v + 1./eta_pl)

        mfile_new.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e %3e\n" 
                        %(sr_T,y[i],eta_df_new[i,j],eta_ds_new[i,j],eta_eff_new[i,j],
                        tau_new[i,j],sr_df_new[i,j],sr_ds_new[i,j],sr_v_new[i,j],isplast_new[i,j]))

        if np.abs(srT[j]+15)<1e-6:
           pfile_new.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e \n" 
                           %(sr_T,y[i],eta_df_new[i,j],eta_ds_new[i,j],eta_eff_new[i,j],
                           tau_new[i,j],sr_df_new[i,j],sr_ds_new[i,j],sr_v_new[i,j]))

        ##### DIFF ####################
        eta_df_diff[i,j] =eta_df_new[i,j]-eta_df_old[i,j]
        eta_ds_diff[i,j] =eta_ds_new[i,j]-eta_ds_old[i,j]
        eta_eff_diff[i,j]=(eta_eff_new[i,j]-eta_eff_old[i,j])/eta_eff_new[i,j]
        sr_ds_diff[i,j] =sr_ds_new[i,j]-sr_ds_old[i,j]
        sr_df_diff[i,j] =sr_df_new[i,j]-sr_df_old[i,j]
        sr_v_diff[i,j]  =sr_v_new[i,j]-sr_v_old[i,j]
        tau_diff[i,j]   =(tau_new[i,j]-tau_old[i,j])/tau_new[i,j]
        mfile_diff.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e \n" 
                         %(sr_T,y[i],eta_df_diff[i,j],eta_ds_diff[i,j],eta_eff_diff[i,j],
                         tau_diff[i,j],sr_df_diff[i,j],sr_ds_diff[i,j],sr_v_diff[i,j]))
        if np.abs(srT[j]+15)<1e-6:
           pfile_diff.write(" %10e %10e %10e %10e %10e %10e %10e %10e %10e \n" 
                            %(sr_T,y[i],eta_df_diff[i,j],eta_ds_diff[i,j],eta_eff_diff[i,j],
                            tau_diff[i,j],sr_df_diff[i,j],sr_ds_diff[i,j],sr_v_diff[i,j]))

    #end for
    mfile_old.write("  \n")
    mfile_new.write("  \n")
    mfile_diff.write("  \n")
#end for
mfile_old.close()
mfile_new.close()
mfile_diff.close()


