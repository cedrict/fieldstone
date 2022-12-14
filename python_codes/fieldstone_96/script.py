import numpy as np
import sys as sys

R_outer=3389.5e3

# =============================================================================
# Stone 96 - Script
# =============================================================================
## PLUME VARIABLES ##
    # R_blob       = float(sys.argv[1]) -        R_blob=200e3            #radius of blob
    # z_blob       = float(sys.argv[2]) -        z_blob=R_outer-1000e3   #starting depth
    # rho_blob     = float(sys.argv[3]) -        rho_blob=3200
    # eta_blob     = float(sys.argv[4]) -        eta_blob=1e22
    # eccentricity = float(sys.argv[5]) -        eccentricity=1
    # path         =       sys.argv[6]  -        path='solution/'
###############################################################################
## MESH ##
    # hhh          = float(sys.argv[7]) -        hhh=40e3                # element size at the surface
## Radial profile ##
    # radial_model =       sys.argv[8]  -        radial_model='4layer'

###############################################################################
## BACKGROUND ## Only radial_model = '4layer'
      # R_disc1   =sys.argv[9]  -                #R_outer-60e3   #crust
      # R_disc2   =sys.argv[10] -                #R_outer-450e3  # lith
      # rho_crust =sys.argv[11] -                #3100 #knaymeyer 2021 and Wieczorek 2022
      # rho_lith  =sys.argv[12] -                #3400  #knaymeyer 2021
      # eta_lith  =sys.argv[13] -                #1e21
      # rho_mantle=sys.argv[14] -                #3500 #knaymeyer 2021
###############################################################################

## blob radius ##
list_x = np.linspace(50e3,500e3,num=9,endpoint=True)
for x in list_x:
    sys.argv = ["stone.py",x,2385e3,3200,1e21,0,'solutions/R_blob/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,3100,3400,1e21,3500]
    print(sys.argv)
    exec(open("./stone.py").read())
    

## blob depth ##
list_x = np.linspace(1800,2700, num=9, endpoint=True )
for x in list_x:
    sys.argv = ["stone",200e3, x ,3200,1e21,0,'solutions/z_blob/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,3100,3400,1e21,3500]
    exec(open("./stone.py").read())

## blob rho ## 
list_x = np.linspace(3000,4000, num=11, endpoint=True )
for x in list_x:
    sys.argv = ["stone",200e3, 2385e3 , x,1e21,0,'solutions/rho_blob/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,3100,3400,1e21,3500]
    exec(open("./stone.py").read())

## blob eta ##
list_x = [1e19,1e20,6e20,1e21,5e21,1e22,5e22,1e23,5e23,1e24,5e24,1e25]
for x in list_x:
    sys.argv = ["stone",200e3, 2385e3 , 3200, x,0,'solutions/eta_blob/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,3100,3400,1e21,3500]
    exec(open("./stone.py").read())
    
## blob eccentricity ##
list_x = np.linspace(0.,0.9,num=11,endpoint=True)
for x in list_x:
    sys.argv = ["stone",200e3, 2385e3 , 3200, 1e21,x,'solutions/eccentricity_blob/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,3100,3400,1e21,3500]
    exec(open("./stone.py").read())

###############################################################################
# Mesh resolution ##
list_hhh = np.linspace(30e3,80e3,num=11,endpoint=True)
list_x= np.sort(np.append(list_hhh,[22e3,24e3,25e3,26e3,28e3,32e3,34e3]))
for x in list_x:
    sys.argv = ["stone.py",200e3,2385e3,3200,1e21,0,'solutions/Resolution/%s/'%(x),x,'4layer',R_outer-60e3,R_outer-450e3,3100,3400,1e21,3500]
    exec(open("./stone.py").read())

# radial_model ##
list_x = ['4layer','steinberger','samuelA','samuelB'] #
for x in list_x:
    sys.argv = ["stone.py",200e3,2385e3,3200,1e21,0,'solutions/Profiles/%s/'%(x),40e3,x,R_outer-60e3,R_outer-450e3,3100,3400,1e21,3500]
    exec(open("./stone.py").read())

###############################################################################    
 
## Crustal thickness ##
list_x = np.linspace(R_outer-30e3,R_outer-70e3,num=5,endpoint=True)
for x in list_x:
    sys.argv = ["stone.py",200e3,2385e3,3200,1e21,0,'solutions/R_disc1/%s/'%(x),40e3,'4layer',x,R_outer-450e3,3100,3400,1e21,3500]
    exec(open("./stone.py").read())   

## lithosphere thickness ##
list_x = np.linspace(R_outer-400e3,R_outer-600e3,num=11,endpoint=True)
for x in list_x:
    sys.argv = ["stone.py",200e3,2385e3,3200,1e21,0,'solutions/R_disc2/%s/'%(x),40e3,'4layer',R_outer-60e3,x,3100,3400,1e21,3500]
    exec(open("./stone.py").read()) 

## Crustal density ##
list_x = np.linspace(2700,3300,num=7,endpoint=True)
for x in list_x:
    sys.argv = ["stone.py",200e3,2385e3,3200,1e21,0,'solutions/rho_crust/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,x,3400,1e21,3500]
    exec(open("./stone.py").read()) 
    
## Lithosphere density ##
list_x = np.linspace(3300,3500,num=11,endpoint=True)
for x in list_x:
    sys.argv = ["stone.py",200e3,2385e3,3200,1e21,0,'solutions/rho_lith/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,3100,x,1e21,3500]
    exec(open("./stone.py").read()) 
    
## lithosphere viscosity ##
list_x = [1e19,1e20,6e20,1e21,5e21,1e22,5e22,1e23,5e23,1e24,5e24,1e25] #np.linspace(1E20,1E25, num=9, endpoint=True )
for x in list_x:
    sys.argv = ["stone.py",200e3,2385e3,3200,1e21,0,'solutions/eta_lith/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,3100,3400,x,3500]
    exec(open("./stone.py").read()) 
    
## Mantle density ##
list_x = np.linspace(3300,3500,num=11,endpoint=True)
for x in list_x:
    sys.argv = ["stone.py",200e3,2385e3,3200,1e21,0,'solutions/rho_mantle/%s/'%(x),40e3,'4layer',R_outer-60e3,R_outer-450e3,3100,3400,1e21,x]
    exec(open("./stone.py").read())



