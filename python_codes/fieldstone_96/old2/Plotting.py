#!/usr/bin/env python
##adjusted CMB and surface placement of titles -> might need to be improved upon ##


# -*- coding: utf-8 -*-
#######################################################################################################
# Making the dynamic topography profile and dynamic topography rate as a function of latitude centered#
# around tharsis.                                                                                     #
# =================================================================================================== #
# This code is used for plotting the dynamic topography and dynamic topography rate as a function of  #
# a few variables (both blob and lithosphere). It still contains some rements of plotting the dynamic #
# topography at the CMB as well but the Double_Dyntop.py code is better suited for this purpose.      # 
# =================================================================================================== #
### last date adjusted: 15/11/2022 ####################################################################
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
###############################################################################
if  sys.path[-1] != 'D:/UU/Master/Thesis/Code/FEM/February':
    sys.path.append('D:/UU/Master/Thesis/Code/FEM/February')

Spath = Spath=sys.path[-1]
print(Spath)
#radian to deg scale
r2d=180/np.pi
#meters to kilometer converter
m2km = lambda x, _: f'{x/1000:g}'
km2m =  lambda x, _: f'{x*1000:g}'


year=365.25*3600*24
dt = 50
R_outer=3389.5e3
g0 = 3.72

MinusSign = True   #sign correction for CMB
def Angle2Dist(rad,Router):
    #rad = np.pi*degrees/180  #from degrees to radians
    D = R_outer*2*np.arcsin((np.sin(-rad/2))**2+np.cos(rad)) #D in meters    
    return D

###############################################################################
## Fontsizes ##################################################################
SMALL_SIZE = 12
MEDIUM_SIZE = 16 #16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the figure title


################################################################################
## Plotting functions ##########################################################
def Check_Dir(name):
    path = Spath+"/%s" %name
    #check if an existing directory exist if not make one
    if os.path.isdir(path) == False:
       os.makedirs(path)
       print(path," directory was created")

def plot_dyntop_surf(numb,name,list_num,list_name,xlabel):
    fig1a,ax1a = plt.subplots(figsize=(10,10)) #figure(figsize=(10,10)) 

    for i in range(len(list_num)):
        num = list_num[i]
        ##surface topography##
        profile_eta0,profile_dynt0_surf=np.loadtxt(Spath+"/solutions/%s/" %(numb)+str(num)+'/dynamic_topography_surf_0000.ascii'  ,unpack=True,usecols=[0,1])
        # ## dynamic topography ## 
        if var=="b_rad" or var=="li_rad"or var=="b_dep":
            ax1a.scatter(profile_eta0*r2d,profile_dynt0_surf,label="%.2f" %(list_name[i]),alpha=0.5 ) 
        else:
            ax1a.scatter(profile_eta0*r2d, profile_dynt0_surf, label=str(num), alpha=0.5) #gravity acceleration is in m/s2 -> we want mGal (1e-5 m/s^2 )#+" timestep 0"
        
        
    ## change y-axis scale from meters to kilometers
    ax1a.yaxis.set_major_formatter(m2km)
    
    ### labels ## 
    ax1a.set_xlabel("Latitudinal degrees centered at Tharsis (degrees)")
    ax1a.grid()
    ax1a.set_ylabel("Dynamic topography (km)")
    ax1a.set_title("Dynamic topography at the surface %s" %name)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5, pad=0.5)
    ax1a.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center',transform = ax1a.transAxes)  
    ax1a.legend()
    ##plt.show()
       
    fig1a.savefig(Spath+"/%s/DynamicTopography_surf_plot_%s.png" %(name,numb))
    plt.close(fig1a)
    
def plot_dyntop_cmb(numb,name,list_num,list_name,xlabel):
    fig1b,ax1b = plt.subplots(figsize=(10,10)) #figure(figsize=(10,10)) 
        
    for i in range(len(list_num)):
        num = list_num[i]
        profile_eta0,profile_dynt0_cmb=np.loadtxt(Spath+"/solutions/%s/" %(numb) +str(num)+'/dynamic_topography_cmb_0000.ascii' ,unpack=True,usecols=[0,2])
        if MinusSign:
            profile_dynt0_cmb = -profile_dynt0_cmb
        if var=="b_rad" or var=="li_rad"or var=="b_dep":
            ax1b.scatter(profile_eta0*r2d,profile_dynt0_cmb ,label="%.2f" %(list_name[i]),alpha=0.5 )
        else:
            ax1b.scatter(profile_eta0*r2d, profile_dynt0_cmb, label=str(num), alpha=0.5) #gravity acceleration is in m/s2 -> we want mGal (1e-5 m/s^2 )#+" timestep 0" 
    
    ## change y-axis scale from meters to kilometers
    ax1b.yaxis.set_major_formatter(m2km)
    
    ### labels ## 
    ax1b.set_xlabel("Latitudinal degrees centered at Tharsis (degrees)")
    ax1b.grid()
    ax1b.set_ylabel("Dynamic topography (km)")
    ax1b.set_title("Dynamic topography at the CMB %s" %name)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5, pad=0.5)
    ax1b.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center',transform = ax1b.transAxes)  
    ax1b.legend()
    ##plt.show()
    fig1b.savefig(Spath+"/%s/DynamicTopography_cmb_plot_%s.png" %(name,numb))
    plt.close(fig1b)

def plot_dynHalfwidth_surf(numb,name,list_num,xlabel):
    list_half_surf = np.zeros(len(list_num))

    i=0
    for num in list_num:
        ##surface topography##
        profile_eta0,profile_dynt0_surf=np.loadtxt(Spath+"/solutions/%s/" %(numb)+str(num)+'/dynamic_topography_surf_0000.ascii'  ,unpack=True,usecols=[0,1])
        max_surf = max(profile_dynt0_surf,key=abs) 
        for j in range(len(profile_dynt0_surf)): #loop over dynamic topography until halfwidth is reached
            if profile_dynt0_surf[j]<=0.5*max_surf:
                list_half_surf[i] = profile_eta0[j] #save eta (angle) of halfwidth location
                break
        #end surface loop     
  
        i = i+1
    #end num loop
    ##PLOTTING
    fig2a,ax2a = plt.subplots(figsize=(10,10)) #figure(figsize=(10,10))        
    ax2a.plot(list_num,r2d*(list_half_surf),label=numb)  #latitude: *scale / Angle2Dist
    ax2a.scatter(list_num,r2d*(list_half_surf), alpha=0.5)  #latitude:  *scale / Angle2Dist
    ax2a.set_title("Surface Halfwidth of Dynamic topography %s" %name)
    ### labels ##
    ax2a.set_xlabel(xlabel)
    ax2a.grid()
    ax2a.set_ylabel("Halfwidth (degrees from Tharsis)")
    #ax2a.yaxis.set_major_formatter(m2km)
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax2a.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax2a.transAxes)  
    ## for Radius change x-scale to km ##
    if var=="b_rad":
        ax2a.xaxis.set_major_formatter(m2km)
    ## for Viscosity use log scale-90. ##
    if var=="b_eta" or var=="li_eta" or var=="ma_eta":
        ax2a.set_xscale("log")
    fig2a.savefig(Spath+"/%s/DynamicTopography_surf_Halfwidth_plot_%s.png" %(name,name))
    plt.close(fig2a)

def plot_dynHalfwidth_cmb(numb,name,list_num,xlabel):
    list_half_cmb = np.zeros(len(list_num))
    
    i=0
    for num in list_num:
        ## CMB topography##
        profile_eta0,profile_dynt0_cmb=np.loadtxt(Spath+"/solutions/%s/" %(numb) +str(num)+'/dynamic_topography_cmb_0000.ascii' ,unpack=True,usecols=[0,2])
        if MinusSign:
            profile_dynt0_cmb = -profile_dynt0_cmb
        max_cmb = max(profile_dynt0_cmb,key=abs)
        for j in range(len(profile_dynt0_cmb)):
            if profile_dynt0_cmb[j]<=0.5*max_cmb:
                list_half_cmb[i] = profile_eta0[j]
                break     
        i = i+1
    #end num loop
    
    fig2b,ax2b= plt.subplots(figsize=(10,10)) #figure(figsize=(10,10))     

    
    ax2b.plot(list_num,r2d*(list_half_cmb),label=numb)  #*scale
    ax2b.scatter(list_num,r2d*(list_half_cmb), alpha=0.5)  #*scale
    ax2b.set_title("CMB Halfwidth of Dynamic topography %s" %name)
    ### labels ##
    ax2b.set_xlabel(xlabel)
    ax2b.grid()
    ax2b.set_ylabel("Halfwidth (degrees from Tharsis)")
    #ax2a.yaxis.set_major_formatter(m2km)
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax2b.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax2b.transAxes)  
    ## for Radius change x-scale to km ##
    if var=="b_rad":
        ax2b.xaxis.set_major_formatter(m2km)
    ## for Viscosity use log scale-90. ##
    if var=="b_eta" or var=="li_eta" or var=="ma_eta":
        ax2b.set_xscale("log")
    fig2b.savefig(Spath+"/%s/DynamicTopography_cmb_Halfwidth_plot_%s.png" %(name,name))
    plt.close(fig2b)    
    
def plot_dynMax_surf(numb,name,list_num,xlabel):
    list_dynt_surf = np.zeros(len(list_num))   
    i=0
    for num in list_num:       
        ##surface topography##
        profile_eta0,profile_dynt0_surf=np.loadtxt(Spath+"/solutions/%s/" %(numb)+str(num)+'/dynamic_topography_surf_0000.ascii'  ,unpack=True,usecols=[0,1])
        
        list_dynt_surf[i] = max(profile_dynt0_surf[2:],key=abs) 
                           
        print(list_dynt_surf[i])
        i = i+1
    #print(list_dynt_surf)
    
    ## maximum dynamic topography ## 
    fig3a,ax3a = plt.subplots(figsize=(10,10))
    ax3a.plot(list_num,list_dynt_surf,label="surface")
    ax3a.scatter(list_num,list_dynt_surf, alpha=0.5)
    
    ax3a.set_ylabel("max Dynamic topography (km)", fontsize=18)
    ax3a.grid()
    ax3a.set_title("Surface Maximum Dynamic topography %s" %name)
    
    ## change y-axis scale from meters to kilometers
    ax3a.yaxis.set_major_formatter(m2km)
    ### labels ##
    ax3a.set_xlabel(xlabel, fontsize=18) 
    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax3a.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax3a.transAxes)  
    ## for Radius change x-scale to km ##
    if var=="b_rad":
        ax3a.xaxis.set_major_formatter(m2km)
    ## for Viscosity use log scale-90. ##
    if var=="b_eta" or var=="li_eta" or var=="ma_eta":
        ax3a.set_xscale("log")
    ## Saving figure ##
    fig3a.savefig(Spath+"/%s/DynamicTopography_surf_Maximum_plot_%s.png" %(name,name))
    plt.close(fig3a)
    
def plot_dynMax_cmb(numb,name,list_num,xlabel):  
    list_dynt_cmb = np.zeros(len(list_num))
    i=0
    for num in list_num:
        ## CMB topography##
        profile_eta0,profile_dynt0_cmb=np.loadtxt(Spath+"/solutions/%s/" %(numb) +str(num)+'/dynamic_topography_cmb_0000.ascii' ,unpack=True,usecols=[0,2])
        if MinusSign:
            profile_dynt0_cmb = -profile_dynt0_cmb
        list_dynt_cmb[i] = max(profile_dynt0_cmb,key=abs)

        i = i+1
    #print(list_dynt_surf)
    ## maximum dynamic topography ## 
    fig3b,ax3b = plt.subplots(figsize=(10,10))

    ax3b.plot(list_num,list_dynt_cmb,label="CMB")
    ax3b.scatter(list_num,list_dynt_cmb, alpha=0.5)
    
    ax3b.set_ylabel("max Dynamic topography (km)")
    ax3b.grid()
    ax3b.set_title("CMB Maximum Dynamic topography %s" %name)
    
    ## change y-axis scale from meters to kilometers
    ax3b.yaxis.set_major_formatter(m2km)
    ### labels ##
    ax3b.set_xlabel(xlabel) 
    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax3b.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax3b.transAxes)  
    ## for Radius change x-scale to km ##
    if var=="b_rad":
        ax3b.xaxis.set_major_formatter(m2km)
    ## for Viscosity use log scale-90. ##
    if var=="b_eta" or var=="li_eta" or var=="ma_eta":
        ax3b.set_xscale("log")
    ## Saving figure ##
    fig3b.savefig(Spath+"/%s/DynamicTopography_cmb_Maximum_plot_%s.png" %(name,name))
    plt.close(fig3b)

def plot_dynRate_surf(numb,name,list_num,xlabel):   
    fig4a,ax4a = plt.subplots(figsize=(10,10))
    for i in range(len(list_num)):
        num = list_num[i]  
        ##surface topography##
        profile_eta0,profile_dynt0=np.loadtxt(Spath+"/solutions/%s/" %(numb)+str(num)+'/dynamic_topography_surf_0000.ascii'  ,unpack=True,usecols=[0,1])
        profile_eta1,profile_dynt1=np.loadtxt(Spath+"/solutions/%s/" %(numb)+str(num)+'/dynamic_topography_surf_0001.ascii'  ,unpack=True,usecols=[0,1])
        
        ##rate##
        profile_rate = (profile_dynt1-profile_dynt0)/dt
        if var=="b_rad" or var=="li_rad"or var=="b_dep":
            ax4a.scatter(profile_eta0*r2d, profile_rate*100, label="%.2f" %(list_name[i]), alpha=0.5)
        else: 
            ax4a.scatter(profile_eta0*r2d, profile_rate*100, label=str(num), alpha=0.5) #gravity acceleration is in m/s2 -> we want mGal (1e-5 m/s^2 )#+" timestep 0"


        #ax4a.plot(profile_eta0[0:100]*r2d-90., profile_rate[0:100], alpha=0.5)
    
    ax4a.set_title("Dynamic topography rate at the surface %s" %name)    
    ax4a.set_xlabel("Latitude centered at tharsis (degrees)")
    ax4a.set_ylabel("Dynamic topography (cm/year)")
    #ax4a.ylim(top=0.08,bottom=-0.03)
    ax4a.set_xlim(right=90,left=0)
    
    ax4a.legend(loc="upper right")
    ax4a.grid()
    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax4a.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax4a.transAxes)  
    fig4a.savefig(Spath+"/%s/DynamicTop_surf_Rate_plot_%s.png" %(name,name))  
    plt.close(fig4a)    

def plot_dynRate_HM_surf(numb,name,list_num,xlabel):
    list_half_rate = np.zeros(len(list_num))
    list_max_rate = np.zeros(len(list_num))
    
    fig7a,ax7a = plt.subplots(figsize=(10,10))
    for i in range(len(list_num)):
        num = list_num[i]  
        ##surface topography##
        profile_eta0,profile_dynt0=np.loadtxt(Spath+"/solutions/%s/" %(numb)+str(num)+'/dynamic_topography_surf_0000.ascii'  ,unpack=True,usecols=[0,1])
        profile_eta1,profile_dynt1=np.loadtxt(Spath+"/solutions/%s/" %(numb)+str(num)+'/dynamic_topography_surf_0001.ascii'  ,unpack=True,usecols=[0,1])
        
        ##rate##
        profile_rate = (profile_dynt1-profile_dynt0)/dt
        max_rate = max(profile_rate,key=abs)
        
        list_max_rate[i]=max_rate
        for j in range(len(profile_rate)):
            if profile_rate[j]<=0.5*max_rate:
                list_half_rate[i] = profile_eta0[j]
                break 
    
    ax7a.plot(list_num,list_max_rate*100)
    ax7a.scatter(list_num,list_max_rate*100, alpha=0.5) #scale m to cm
    ax7a.set_ylabel("Maximum Dynamic topography rate (cm)", fontsize=18)
    ax7a.set_xlabel(xlabel,fontsize=18)
    ax7a.grid()
    ax7a.set_title("Surface Maximum Dynamic topography Rate %s" %name,fontsize=MEDIUM_SIZE)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax7a.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax7a.transAxes)  
    ##PLOTTING ##############################################################
    #fig7a,ax7a = plt.subplots(1,2,figsize=(20,10)) #figure(figsize=(10,10))        
    ## HALFWIDTH
    # ax7a[0].plot(list_num,r2d*(list_half_rate),label=numb) 
    # ax7a[0].scatter(list_num,r2d*(list_half_rate), alpha=0.5)
    # ax7a[0].set_title("Surface Halfwidth of Dynamic topography Rate %s" %name,fontsize=MEDIUM_SIZE)
    # ### labels ##
    # ax7a[0].set_xlabel(xlabel,fontsize=18)
    # ax7a[0].grid()
    # ax7a[0].set_ylabel("Halfwidth Rate (degrees from Tharsis)", fontsize=18)
    #ax7a.yaxis.set_major_formatter(m2km)
    ## MAXIMUM RATE
    # ax7a[1].plot(list_num,list_max_rate*100)
    # ax7a[1].scatter(list_num,list_max_rate*100, alpha=0.5) #scale m to cm
    # ax7a[1].set_ylabel("Maximum Dynamic topography rate (cm)", fontsize=18)
    # ax7a[1].set_xlabel(xlabel,fontsize=18)
    # ax7a[1].grid()
    # ax7a[1].set_title("Surface Maximum Dynamic topography Rate %s" %name,fontsize=MEDIUM_SIZE)
    
    ## Adding textbox ##
    # props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    # ax7a[0].text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
    #         va='center', transform = ax7a[0].transAxes)  
    # ax7a[1].text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
    #         va='center', transform = ax7a[1].transAxes) 
    
    ## for Radius change x-scale to km ##
    if var=="b_rad":
        ax7a[0].xaxis.set_major_formatter(m2km)
        ax7a[1].xaxis.set_major_formatter(m2km)
    ## for Viscosity use log scale-90. ##
    if var=="b_eta" or var=="li_eta" or var=="ma_eta":
        ax7a[0].set_xscale("log")
        ax7a[1].set_xscale("log")
        
        
    fig7a.savefig(Spath+"/%s/DynamicTopography_Rate_diagnostics_%s.png" %(name,name))
    plt.close(fig7a)        
        
def plot_dynRate_cmb(numb,name,list_num,xlabel):   
    fig4b,ax4b = plt.subplots(figsize=(10,10))
    
    for i in range(len(list_num)):
        num = list_num[i]      
        ## CMB topography##
        profile_eta0,profile_dynt0=np.loadtxt(Spath+"/solutions/%s/" %(numb) +str(num)+'/dynamic_topography_cmb_0000.ascii' ,unpack=True,usecols=[0,2]) 
        profile_eta1,profile_dynt1=np.loadtxt(Spath+"/solutions/%s/" %(numb) +str(num)+'/dynamic_topography_cmb_0001.ascii' ,unpack=True,usecols=[0,2]) 
        if MinusSign:
            profile_dynt0 = -profile_dynt0
            profile_dynt1 = -profile_dynt1
        ##rate##
        profile_rate = (profile_dynt1-profile_dynt0)/dt
        if var=="b_rad" or var=="li_rad"or var=="b_dep":
            ax4b.scatter(profile_eta0*r2d, profile_rate*100, label="%.2f" %(list_name[i]), alpha=0.5)
        else:
            ax4b.scatter(profile_eta0*r2d, profile_rate*100, label=str(num), alpha=0.5) #gravity acceleration is in m/s2 -> we want mGal (1e-5 m/s^2 )#+" timestep 0"
    

    
    ax4b.set_title("Dynamic topography rate at the CMB %s" %name)
    ax4b.set_xlabel("Latitude centered at tharsis (degrees)")
    ax4b.set_ylabel("Dynamic topography (cm/year)")
    #ax4b.ylim(top=0.08,bottom=-0.03)
    ax4b.set_xlim(right=90,left=0)
    ax4b.legend(loc="upper right")
    ax4b.grid()
    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax4b.text(0.855, 0.15, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax4b.transAxes)  
    fig4b.savefig(Spath+"/%s/DynamicTop_cmb_Rate_plot_%s.png" %(name,name))  
    plt.close(fig4b)    


def plot_grav(numb,name,list_num,xlabel,textstr):
    fig5,ax5 = plt.subplots(figsize=(10,10))
    
    #fig = plt.figure(figsize=(20,10)) 
    for i in range(len(list_num)):
        num = list_num[i]  
        # absolute gravity
        #angleM0,gvect0=np.loadtxt(Spath+'/solutions/%s/'%(numb)+str(num)+'/gravity_0000.ascii'  ,unpack=True,usecols=[4,8])
        
        ## Gravity Anomalie - Mean gravity
        #gvect_mean = sum(gvect0)/len(gvect0)
        #gvect0 = gvect0-gvect_mean
        ## Gravity Anomalie - Mass model with plume ##
        angleM0,gvect0=np.loadtxt(Spath+'/solutions/%s/'%(numb)+str(num)+'/gravityanomaly_wp_0000.ascii'  ,unpack=True,usecols=[4,8])
        ## Gravity Anomalie - Mass model no plume ##
        #angleM0,gvect0=np.loadtxt(Spath+'/solutions/%s/'%(numb)+str(num)+'/gravityanomaly_np_0000.ascii'  ,unpack=True,usecols=[4,8])
        if var=="b_rad" or var=="li_rad"or var=="b_dep":
            ax5.scatter(angleM0*r2d, (np.flip(gvect0))*1e5, label="%.2f" %(list_name[i]), alpha=0.7) #gravity acceleration is in m/s2 -> we want mGal (1e-5 m/s^2 )#+" timestep 0"
        else:
            ax5.scatter(angleM0*r2d, (np.flip(gvect0))*1e5, label=str(num), alpha=0.7) #gravity acceleration is in m/s2 -> we want mGal (1e-5 m/s^2 )#+" timestep 0"
  
    ### labels ##
    ax5.set_xlabel("Latitude degree from Tharsis")
    ax5.set_ylabel("Gravity Anomaly (mGal)") #acceleration - Mean gravity (mGal)",fontsize=14)
    ax5.grid()
    ax5.legend(loc="upper right")
    ax5.set_title("Gravity anomaly %s" %name)

    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax5.text(0.8, 0.115, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax5.transAxes)    
    fig5.savefig(Spath+"/%s/Gravity_plot_%s.png" %(name,name))
    plt.close(fig5)

def plot_gravRate(numb,name,list_num,xlabel,textstr):
    fig6,ax6 = plt.subplots(figsize=(10,10))
    for i in range(len(list_num)):
        num = list_num[i]  
        #angleM0,gvect0=np.loadtxt(Spath+'/solutions/%s/'%(numb)+str(num)+'/gravity_0000.ascii'  ,unpack=True,usecols=[4,8])
        #angleM1,gvect1=np.loadtxt(Spath+'/solutions/%s/'%(numb)+str(num)+'/gravity_0001.ascii'  ,unpack=True,usecols=[4,8])
        
        ##rate##
        #profile_rate = (gvect1-gvect0)/dt
        
        angleMd,gvectd=np.loadtxt(Spath+'/solutions/%s/'%(numb)+str(num)+'/gravity_diff_0001.ascii'  ,unpack=True,usecols=[4,8])
        if var=="b_rad" or var=="li_rad" or var=="b_dep":
            ax6.scatter(angleMd*r2d, np.flip(gvectd)/dt*1e8, label="%.2f" %(list_name[i]), alpha=0.5) #gravity difference is in m/s2 / year we want muGal/year so x1e8)
        else:
            ax6.scatter(angleMd*r2d, np.flip(gvectd)/dt*1e8, label=str(num), alpha=0.5) #gravity difference is in m/s2 / year we want muGal/year so x1e8)
        ax6.plot(angleMd*r2d, np.flip(gvectd)/dt*1e8, alpha=0.5)

    
    ax6.set_xlabel("Latitude centered at tharsis (degrees)")
    ax6.set_ylabel(r"Gravity rate ($\mu$Gal/year)")
    ax6.set_title("Gravity rate %s" %name)
    ax6.grid()
    ax6.legend(loc="upper right")
    #plt.ylim([-0.005,0.005])
    #plt.legend(loc="upper right")
    ##plt.show()
    props = dict(boxstyle='square', facecolor='white', alpha=0.5,pad=0.5)
    ax6.text(0.8, 0.115, textstr, fontsize=14, bbox=props,ha='center', 
            va='center', transform = ax6.transAxes) 
    fig6.savefig(Spath+"/%s/Gravity_Rate_plot_%s.png" %(name,name))
    plt.close(fig6)


##############################################################################
## Data ###################################################################### 
##plume variables##
#list_variables = ["b_rad",'b_dep','b_rho','l_ecc','l_rho','b_eta']
##model variables
#list_variables = ['mesh','model']
##background variables
#list_variables = ["cr_rho","li_rho","li_eta","ma_rho"] #cr_rad,li_rad
##trail
#list_variables = ['trial']
list_variables = ['b_rho']
#"model",
print(list_variables)
for var in list_variables:
    ## Plume variables ##
    if var=='b_rad':
        numb = 'R_blob'
        name = 'Plume Radius'
        xlabel = "Plume Radius (km)"
        textstr = '\n'.join((
            r'Profile=4-layer',        
            r'$\rho_p=3200$ kg/m$^3$',
            r'$z_p=2385$ km' ,
            r'$\eta_p$=1e21 Pa'))
        list_num = np.array([50e3,100e3,150e3,200e3,250e3,300e3,350e3,400e3,450e3,500e3])#np.linspace(50e3,500e3,num=9,endpoint=True)
        list_name = list_num/1000
    elif var=='b_dep':   
        list_num = np.linspace(1900e3,2600e3, num=8, endpoint=True )
        #list_num = np.array([2137.5e3,2250e3,2475e3,2587.5e3]) #reduced plot
        list_name = list_num/1000
        xlabel = "Plume depth in radius (m)"
        name = "Plume depth"
        numb= 'z_blob'
        textstr = '\n'.join((
            r'Profile=4-layer',
            r'$R_p=300$ km' ,
            r'$\rho_p=3200$ kg/m$^3$',
            r'$\eta_p$=1e21 Pa'))
    elif var=='b_rho'or var=='lb_rho':     
        xlabel ="Plume density (kg/m$^3$)"
        name = "Plume density"     
        if var=='lb_rho':
            list_num= np.linspace(3000,3500, num=6, endpoint=True )
            numb = 'rho_blob_l'
            textstr = '\n'.join((
                r'Profile=4-layer',
                r'$R_p=500$ km' ,        
                r'$z_p=2385$ km' ,
                r'$\eta_p$=1e21 Pa'))
        else:
            list_num = np.linspace(3000,4000, num=11, endpoint=True )
            numb = 'rho_blob'
            textstr = '\n'.join((
                r'Profile=4-layer',
                r'$R_p=500$ km' ,        
                r'$z_p=2385$ km' ,
                r'$\eta_p$=1e21 Pa'))
        list_name = list_num
    elif var=='b_eta':
        #list_num = [1e19,1e20,6e20,1e21,5e21,1e22,5e22,1e23,5e23,1e24,5e24,1e25]
        list_num = [1e19,1e20,1e21,1e22,1e23,1e24,1e25] #reduced plot
        list_name = list_num
        numb = "eta_blob"
        name = "Plume viscosity"
        xlabel = "Log(Viscosity blob) (log(Pa s))"
        textstr = '\n'.join((
            r'Profile=4-layer',
            r'$R_p=300$ km' ,
            r'$\rho_p=3200$ kg/m$^3$',
            r'$z_p=2385$ km'))
    elif var=='b_ecc' or var=='lb_ecc':
        list_num = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]#,0.98]
        list_name = list_num
        name = "Plume eccentricity"
        xlabel= "eccentricty"
        if var=='lb_ecc':
         name = "Large Plume eccentricity"
         numb = "eccentricity_blob_l"
         textstr = '\n'.join((
             r'Profile=4-layer',
             r'$R_p=500$ km' ,
             r'$\rho_p=3200$ kg/m$^3$',        
             r'$z_p=2385$ km',
             r'$\eta_p$=1e21 Pa'))
        else:
            numb = "eccentricity_blob"
            textstr = '\n'.join((
                r'Profile=4-layer',
                r'$R_p=200$ km' ,
                r'$\rho_p=3200$ kg/m$^3$',        
                r'$z_p=2385$ km',
                r'$\eta_p$=1e21 Pa'))
    ## Model variables ##
    elif var=='mesh':
        #list_hhh = np.linspace(30e3,80e3,num=11,endpoint=True)  
        list_num= np.array([20e3,25e3,32e3,35e3,44e3,56e3,68e3,80e3])#np.sort(np.append(list_hhh,[20e3,25e3]))
        #list_num = np.array([22e3,25e3,30e3,40e3,50e3,60e3,70e3,80e3]) #-R
        #list_num = list_num[0:8] #-L
        list_name = list_num*1/1000
        numb = "Resolution"
        name = "Mesh Resolution -L"
        xlabel = "Maximum element size at surface (km)"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
    elif var=='model':
        list_num = ['steinberger','samuelA','samuelB','4layer'] 
        list_name = list_num
        numb = "Profiles"
        name = "Profiles"
        xlabel = "Maximum length of surface element (km)"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
    ## Background variables ##
    elif var=='cr_rad':
        list_num = np.linspace(R_outer-30e3,R_outer-70e3,num=5,endpoint=True)
        list_name = R_outer - list_num/1000
        numb = " "
        name = "Crust Thickness"
        xlabel = "Depth of Crust-Lithosphere transition (km)"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
    elif var=='li_rad':
        list_num = np.linspace(R_outer-400e3,R_outer-600e3,num=11,endpoint=True)
        list_name = R_outer/1000-list_num/1000
        numb = "R_disc2"
        name = "Lithosphere Thickness"
        xlabel = "Depth of Lithosphere-Mantle transition (km)"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
    elif var=='cr_rho':
        list_num = np.linspace(2700,3300,num=7,endpoint=True)
        list_name = list_num
        numb = "rho_crust"
        name = "Crustal Density"
        xlabel = "Crustal Density (kg/m$^3$)"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
    elif var=='li_rho':
        list_num = np.linspace(3300,3500,num=11,endpoint=True)
        list_name = list_num
        xlabel = "Density of lithosphere (kg/m$^3$)" 
        numb = "rho_lith"
        name = "Lithosphere Density"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
    elif var=='li_eta':
        list_num = [1e19,1e20,6e20,1e21,5e21,1e22,5e22,1e23,5e23,1e24,5e24,1e25] #np.linspace(1E20,1E25, num=9, endpoint=True )
        #list_num = [1e19,6e20,5e22,1e25] #reduced 
        xlabel = "Viscosity of lithosphere (Pa s)"
        list_name = list_num
        numb = "eta_lith"
        name = "Lithosphere Viscosity"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
    elif var=='ma_rho':
        list_num = np.linspace(3300,3480,num=9,endpoint=False)
        list_name = list_num
        xlabel = "Mantle Density (kg/m$^3$)"
        numb = "rho_mantle"
        name = "Mantle Density"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
    elif var=="ma_eta":
        list_num = [1e19,1e20,6e20,1e21,5e21,1e22,5e22,1e23,5e23,1e24,5e24,1e25]# 
        #list_num = [1e20,6e20,1e23,1e25] #reduced 
        xlabel = "Viscosity of Mantle (Pa s)"
        list_name = list_num
        numb = "eta_mantle"
        name = "Mantle Viscosity"
        textstr = '\n'.join((
            r'Profile=4layer',
            r'$R_p=500 km$' ,
            r'$\rho_p=3200$ kg/m$^3$',        
            r'$z_p=2385 km$',
            r'$\eta_p$=1e21 Pa'))
        
    ## TRIAL ##
    elif var == "trial":
        list_num = [5]
        list_name = list_num
        xlabel = "Trial"
        numb = "trial"
        name = "trial3"
        textstr = '\n'.join(())

    print("entrees:",list_num)
    
    print(numb)
    
    ##Make / check directory
    Check_Dir(name)
    ## plot surface dynamic topography graphs
    plot_dyntop_surf(numb, name, list_num,list_name,  xlabel)
    plot_dynMax_surf(numb, name, list_num, xlabel)
    plot_dynHalfwidth_surf(numb, name, list_num,  xlabel)
    plot_dynRate_surf(numb, name, list_num,  xlabel)
    plot_dynRate_HM_surf(numb, name, list_num, xlabel)
    ## plot cmb surface dynamic topography graphs
    plot_dyntop_cmb(numb, name, list_num, list_name, xlabel)
    plot_dynMax_cmb(numb, name, list_num,  xlabel)
    plot_dynHalfwidth_cmb(numb, name, list_num,  xlabel)
    plot_dynRate_cmb(numb, name, list_num,  xlabel)
    ## plot gravity anomaly plots
    plot_grav(numb, name, list_num,  xlabel, textstr)
    plot_gravRate(numb, name, list_num,  xlabel, textstr)



