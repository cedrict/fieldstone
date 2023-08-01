
#TODO: clean '' stuf around options
import numpy as np     

def set_measurement_parameters(rDEM,sDEM,site,path,ho):

   npath=0
   zpath_height=0
   Lx=0
   Ly=0
   Lz=0
   nelx=0
   nely=0
   nelz=0
   xllcorner=0
   yllcorner=0
   pathfile='nope_error'
   topofile='nope_error'
   IGRFx=0
   IGRFy=0
   IGRFz=0
   Brefx=0
   Brefy=0
   Brefz=0
   error=False

#-------------------------------------------------------------------
#IGRF on sites and average
#-------------------------------------------------------------------
   if site==1:
      IGRF_E=1560.1e-9
      IGRF_N=26887.4e-9
      IGRF_D=36243.6e-9
      Bref_inc=54.06
      Bref_dec=5.34
      Bref_int=44.89e-6
      print ('reference field and IGRF site 1')

   elif site==2:
      IGRF_E=1557.8e-9
      IGRF_N=26881e-9
      IGRF_D=36243.1e-9
      Bref_inc=52.46
      #Bref_dec=-10.8318d0
      Bref_dec=0.59
      Bref_int=44.00e-6
      print('reference field and IGRF site 2')
   elif site==3:
      IGRF_E=1565.3e-9
      IGRF_N=26804.1e-9
      IGRF_D=36407.4e-9
      Bref_inc=53.03
      Bref_dec=-1.65
      Bref_int=44.39e-6
      print('reference field and IGRF site 3')
   elif site==4:   
      IGRF_E=1563e-9
      IGRF_N=26824.3e-9
      IGRF_D=36348.6e-9
      Bref_inc=52.18
      #Bref_dec=-41.42892
      Bref_dec=-5.36
      #Bref_dec=-1.02
      Bref_int=43.36e-6
      print ('reference field and IGRF site 4')
   elif site==5:  
      IGRF_E=1557.9e-9
      IGRF_N=26881.1e-9
      IGRF_D=36243.3e-9
      Bref_inc=52.13
      Bref_dec=1.03
      Bref_int=43.85e-6
      print ('reference field and IGRF site 5')
   elif site==6:       
      IGRF_E=1562.9e-9
      IGRF_N=26824.1e-9
      IGRF_D=36348.3e-9
      Bref_inc=53.69
      Bref_dec=0.36
      Bref_int=43.85e-6
      print ('reference field and IGRF site 6')
   else:
      IGRF_E=1561.2e-9
      IGRF_N=26850.3e-9
      IGRF_D=36305.7e-9
      print('reference field is IGRF average')

   #coordinate switch from ref to pmag model
   IGRFx=IGRF_N
   IGRFy=IGRF_E
   IGRFz=IGRF_D
   
   Brefy=Bref_int*np.cos(Bref_inc*np.pi/180)*np.sin(Bref_dec*np.pi/180)
   Brefx=Bref_int*np.cos(Bref_inc*np.pi/180)*np.cos(Bref_dec*np.pi/180)
   Brefz=(Bref_int)*np.sin(Bref_inc*np.pi/180)
   
   #Brefint=np.sqrt(Brefx**2+Brefy**2+Brefz**2)
   #Brefinc=np.arctan2(Brefz,np.sqrt(Brefx**2+Brefy**2))/np.pi*180
   #Brefdec=np.arctan2(Brefy,Brefx)/np.pi*180
   #IGRFint=np.sqrt(IGRFx**2+IGRFy**2+IGRFz**2)        
   #IGRFinc=np.arctan2(IGRFz,np.sqrt(IGRFx**2+IGRFy**2))/np.pi*180
   #IGRFdec=np.arctan2(IGRFy,IGRFx)/np.pi*180
   
   #print("IGRF Int=",IGRFint)
   #print("IGRF Inc=",IGRFinc)
   #print("IGRF Dec=",IGRFdec)
   #print("Bref Int=",Brefint)
   #print("Bref Inc=",Brefinc)
   #print("Bref Dec=",Brefdec)
   
   #----------------------------------------   
   if site==1:        #age:1892,flank:SF
      if (rDEM==2 and sDEM==1):
         topofile='./DEMS/2m_utm_bili_site1.asc'
         print('reading from 2x2 DEM site 1 regular size (2000)')
         # size of domain, Lz is the added depth under the DEM
         Lx=1038*2 #2x nelx                    
         Ly=1031*2 #2x nely                    
         Lz=10
         #number of cells                      
         nelx=1038 #ncols-1                    
         nely=1031 #nrows-1                    
         nelz=5
         xllcorner=500700.44188544           
         yllcorner=4170086.9789642
      elif (rDEM==2 and sDEM==2):
         topofile='DEMS/dem2m_site1_1400.asc'
         print('reading from 2x2 DEM site 1 1400')
         Lx=738*2 #2x nelx                    
         Ly=731*2 #2x nely                    
         Lz=10
         nelx=738 #ncols-1                    
         nely=731 #nrows-1                    
         nelz=5
         xllcorner=501000.44188544           
         yllcorner=4170386.9789642  
      elif (rDEM==2 and sDEM==3):
         topofile='DEMS/dem2m_site1_1000.asc'
         print('reading from 2x2 DEM site 1 1000')
         Lx=538*2 #2x nelx                    
         Ly=531*2 #2x nely                    
         Lz=10
         nelx=538 #ncols-1                    
         nely=531 #nrows-1                    
         nelz=5
         xllcorner=501200.44188544           
         yllcorner=4170586.9789642  
      elif (rDEM==2 and sDEM==4):
         topofile='DEMS/dem2m_site1_600.asc'
         print('reading from 2x2 DEM site 1 600')
         Lx=338*2 #2x nelx                    
         Ly=331*2 #2x nely                    
         Lz=10
         nelx=338 #ncols-1                    
         nely=331 #nrows-1                    
         nelz=5
         xllcorner=501400.44188544           
         yllcorner=4170786.9789642
      elif (rDEM==2 and sDEM==5):
         topofile='DEMS/dem2m_site1_400.asc'
         print('reading from 2x2 DEM site 1 400')
         Lx=238*2 #2x nelx                       
         Ly=231*2 #2x nely                    
         Lz=10
         nelx=238 #ncols-1                    
         nely=231 #nrows-1                    
         nelz=5
         xllcorner=501500.44188544           
         yllcorner=4170886.9789642 
      elif (rDEM==2 and sDEM==6):
         topofile='DEMS/dem2m_site1_0.asc'
         print('reading from 2x2 DEM site 1 0-around paths')
         Lx=38*2 #2x nelx                    
         Ly=31*2 #2x nely                    
         Lz=10
         nelx=38 #ncols-1                    
         nely=31 #nrows-1                    
         nelz=5
         xllcorner=501700.44188544           
         yllcorner=4171086.9789642  
      elif (rDEM==2 and sDEM==7):
         topofile='DEMS/dem2m_site1_4000.asc'
         print('reading from 2x2 DEM site 1 4000 around paths')
         Lx=2038*2 #2x nelx                    
         Ly=2019*2 #2x nely                    
         Lz=10
         nelx=2038 #ncols-1                    
         nely=2019 #nrows-1                    
         nelz=5
         xllcorner=499700.44188544           
         yllcorner=4169086.9789642       
      elif (rDEM==2 and sDEM==8):
         topofile='DEMS/dem2m_site1_6000.asc'
         print('reading from 2x2 DEM site 1 6000 around paths')
         Lx=3038*2 #2x nelx                    
         Ly=3019*2 #2x nely                    
         Lz=10
         nelx=3038 #ncols-1                    
         nely=3019 #nrows-1                    
         nelz=5
         xllcorner=498700.44188544           
         yllcorner=4168086.9789642  
      elif (rDEM==5 and sDEM==1):
         topofile='DEMS/5m_site1.asc'
         print('reading from 5x5 DEM site 1 2000')
         Lx=415*5                              
         Ly=412*5                               
         Lz=10
         nelx=415                               
         nely=412                               
         nelz=5
         xllcorner=500698.81984712            
         yllcorner=4170088.3580075  
      elif (rDEM==5 and sDEM==2):
         topofile='DEMS/dem5m_site1_300.asc'
         print('reading from 5x5 DEM site 1 300')
         Lx=55*5 #2x nelx                    
         Ly=52*5 #2x nely                    
         Lz=10
         nelx=55 #ncols-1                    
         nely=52 #nrows-1                    
         nelz=5
         xllcorner=501598.81984712           
         yllcorner=4170988.3580075   
      elif (rDEM==5 and sDEM==3):
         topofile='DEMS/dem5m_site1_200.asc'
         print('reading from 5x5 DEM site 1 200')
         Lx=35*5 #2x nelx                    
         Ly=32*5 #2x nely                    
         Lz=10
         nelx=35 #ncols-1                    
         nely=32 #nrows-1                    
         nelz=5
         xllcorner=501648.81984712           
         yllcorner=4171038.3580075
      else:
         error=True
         print("unknown rDEM/sDEM combination for site 1")

   #----------------------------------------   
   elif (site==2 or site==5) : #1983,SF
      if (rDEM==2):
         topofile='DEMS/2m_utm_bili_site2_5.asc'
         print('reading from 2x2 DEM site 2 and 5')
         Lx=1050*2                             
         Ly=1039*2                             
         Lz=10
         nelx=1050                             
         nely=1039                             
         nelz=5
         xllcorner=498322.44188544           
         yllcorner=4170900.9789642     
      elif (rDEM==5 and sDEM==1): #AEH
         topofile='DEMS/5m_site2_5.asc'
         print('reading from 5x5 DEM site 2 and 5 2000')
         Lx=420*5                              
         Ly=415*5                              
         Lz=10
         nelx=420                              
         nely=415                              
         nelz=5
         xllcorner=498318.81984712           
         yllcorner=4170903.3580075     
      elif (rDEM==5 and sDEM==2): #AEH
         topofile='DEMS/dem5m_site2_5_300.asc'
         print('reading from 5x5 DEM site 2 and 5 300')
         Lx=60*5
         Ly=55*5
         Lz=10
         #number of cells
         nelx=60 #ncols-1
         nely=55 #nrows-1
         nelz=5
         xllcorner=499218.81984712
         yllcorner=4171803.3580075
      elif (rDEM==5 and sDEM==3): #AEH
         topofile='DEMS/dem5m_site2_5_200.asc'
         print('reading from 5x5 DEM site 2 and 5 200')
         Lx=40*5
         Ly=35*5
         Lz=10
         #number of cells
         nelx=40
         nely=35
         nelz=5
         xllcorner=499268.81984712
         yllcorner=4171853.3580075
      else:
         error=True
         print("unknown rDEM/sDEM combination for site 2 and/or 5")

   #----------------------------------------   
   elif (site==3) : #1923,NEF
      if (rDEM==2 and sDEM==1):
         topofile='DEMS/2m_utm_bili_site3.asc'
         print('reading from 2x2 DEM site 1 2000')
         Lx=1053*2                             
         Ly=1052*2
         Lz=10
         nelx=1053
         nely=1052
         nelz=5
         xllcorner=506076.44188544
         yllcorner=4187572.9789642
      elif (rDEM==2 and sDEM==2):
         topofile='DEMS/dem2m_site3_300.asc'
         print('reading from 2x2 DEM site 1 300')
         Lx=153*2
         Ly=152*2
         Lz=10
         #number of cells
         nelx=153 #ncols-1
         nely=152  #nrows-1
         nelz=5
         xllcorner=506976.44188544
         yllcorner=4188472.9789642
      elif (rDEM==2 and sDEM==3):
         topofile='DEMS/dem2m_site3_200.asc'
         print('reading from 2x2 DEM site 1 200')
         Lx=103*2
         Ly=107*2
         Lz=10
         #number of cells
         nelx=103
         nely=107
         nelz=5
         xllcorner=507026.44188544
         yllcorner=4188522.9789642
      elif (rDEM==5):
         error=True
         print("site 3 is not on the 5x5 DEM")
      else: 
         error=True
         print("unknown rDEM/sDEM combination for site 3")

   #----------------------------------------   
   elif (site==4): #2002,NEF
      if (rDEM==2):
         topofile='DEMS/2m_utm_bili_site4_6.asc'
         print('reading from 2x2 DEM site 4 and 6')
         Lx=1043*2
         Ly=1112*2
         Lz=10
         nelx=1043
         nely=1112
         nelz=5
         xllcorner=504334.44188544
         yllcorner=4182130.9789642
      elif (rDEM==5 and sDEM==1):
         topofile='DEMS/5m_site4_6.asc'
         print('reading from 5x5 DEM site 4 and 6 2100')
         Lx=417*5
         Ly=444*5
         Lz=10
         nelx=417
         nely=444
         nelz=5
         xllcorner=504333.81984712
         yllcorner=4182133.3580075
      elif (rDEM==5 and sDEM==2):
         topofile='DEMS/dem5m_site4_300.asc'
         print('reading from 5x5 DEM site 4 specific 300')
         Lx=44*5
         Ly=53*5
         Lz=10
         nelx=44 
         nely=53 
         nelz=5
         xllcorner=505233.81984712
         yllcorner=4183088.3580075
      elif (rDEM==5 and sDEM==3):
         topofile='DEMS/dem5m_site4_200.asc'
         print('reading from 5x5 DEM site 4 specific 200')
         Lx=24*5
         Ly=33*5
         Lz=10
         nelx=24
         nely=33
         nelz=5
         xllcorner=505283.81984712
         yllcorner=4183138.3580075
      else: 
         error=True
         print("unknown rDEM/sDEM combination for site 4")
  
  #----------------------------------------   
   elif (site==6): #2002,NEF
      if (rDEM==2):
         topofile='DEMS/2m_utm_bili_site4_6.asc'
         print('reading from 2x2 DEM site 4 and 6')
         Lx=1043*2
         Ly=1112*2
         Lz=10
         nelx=1043
         nely=1112
         nelz=5
         xllcorner=504334.44188544
         yllcorner=4182130.9789642
      elif (rDEM==5 and sDEM==1):
         topofile='DEMS/5m_site4_6.asc'
         print('reading from 5x5 DEM site 6 2100')
         Lx=417*5
         Ly=444*5
         Lz=10
         nelx=417
         nely=444
         nelz=5
         xllcorner=504333.81984712
         yllcorner=4182133.3580075
      elif (rDEM==5 and sDEM==2):
         topofile='DEMS/dem5m_site4_6_1100.asc'
         print('reading from 5x5 DEM site 6 1100')
         Lx=217*5
         Ly=244*5
         Lz=10
         #number of cells
         nelx=217
         nely=244
         nelz=5
         xllcorner=504833.81984712
         yllcorner=4182633.3580075
      elif (rDEM==5 and sDEM==3):
         topofile='DEMS/dem5m_site4_6_500.asc'
         print('reading from 5x5 DEM site 6 500')
         Lx=97*5
         Ly=124*5
         Lz=10
         #number of cells
         nelx=97
         nely=124
         nelz=5
         xllcorner=505133.81984712
         yllcorner=4182933.3580075
      elif (rDEM==5 and sDEM==4):
         topofile='DEMS/dem5m_site4_6_300.asc'
         print('reading from 5x5 DEM site 6 300')
         Lx=57*5
         Ly=84*5
         Lz=10
         #number of cells
         nelx=57
         nely=84
         nelz=5
         xllcorner=505233.81984712
         yllcorner=4183033.3580075
      elif (rDEM==5 and sDEM==5):
         topofile='DEMS/dem5m_site4_6_50r.asc'
         print('reading from 5x5 DEM site 6 50 m around')
         Lx=35*5
         Ly=60*5
         Lz=10
         #number of cells
         nelx=35
         nely=60
         nelz=5
         xllcorner=505313.81984712
         yllcorner=4183083.3580075
      elif (rDEM==5 and sDEM==6):
         topofile='DEMS/site6_20r.asc'
         print('reading from 5x5 DEM site 6 20 m around')
         Lx=23*5
         Ly=48*5
         Lz=10
         #number of cells
         nelx=23
         nely=48
         nelz=5
         xllcorner=505343.81984712
         yllcorner=4183113.3580075
      else: 
         error=True
         print("unknown rDEM/sDEM combination for site 6")
   
   #----------------------------------------   
   if (site==1 and path==1 and ho==1):
         pathfile='sites/1-1-1.txt'
         print('reading from 1-1-1')
         zpath_height=1
         npath=39
      #if (zpath_option==2 or zpath_option==1): 
      #if zpath_option==3:
      #   xmin=501682-xllcorner
      #   xmax=501764-xllcorner
      #   ymin=4171093-yllcorner
      #   ymax=4171169-yllcorner
   elif (site==1 and path==1 and ho==2): 
      pathfile='sites/1-1-2.txt'
      print('reading from 1-1-2')
      npath=39
      zpath_height=1.8         
   elif (site==1 and path==2 and ho==1) :
      pathfile='sites/1-2-1.txt'
      print('reading from 1-2-1')
      npath=27
      zpath_height=1
   elif (site==1 and path==2 and ho==2) :
      pathfile='sites/1-2-2.txt'
      print('reading from 1-2-2')
      npath=27
      zpath_height=1.8
   elif (site==1 and path==3 and ho==1) :
      pathfile='sites/1-3-1.txt'
      print('reading from 1-3-1')
      npath=36
      zpath_height=1
   elif (site==1 and path==3 and ho==2) :
      pathfile='sites/1-3-2.txt'
      print('reading from 1-3-2')
      npath=36
      zpath_height=1.8      
   elif (site==2 and path==1 and ho==1) :
      pathfile='sites/2-1-1.txt'
      print('reading from 2-1-1')
      npath=30
      zpath_height=1
   elif (site==2 and path==1 and ho==2) :
      pathfile='sites/2-1-2.txt'
      print('reading from 2-1-2')
      npath=30
      zpath_height=1.8
   elif (site==2 and path==2 and ho==1) :
      pathfile='sites/2-2-1.txt'
      print('reading from 2-2-1')
      npath=39
      zpath_height=1
   elif (site==2 and path==2 and ho==2) :
      pathfile='sites/2-2-2.txt'
      npath=39
      zpath_height=1.8
      print('reading from 2-2-2')
   elif (site==2 and path==3 and ho==1) :
      pathfile='sites/2-3-1.txt'
      print('reading from 2-3-1')
      npath=42
      zpath_height=1
   elif (site==2 and path==3 and ho==2) :
      pathfile='sites/2-3-2.txt'
      print('reading from 2-3-2')
      npath=42
      zpath_height=1.8      
   elif (site==3 and path==1 and ho==1) :
      pathfile='sites/3-1-1.txt'
      print('reading from 3-1-1')
      npath=39
      zpath_height=1
   elif (site==3 and path==1 and ho==2) :
      pathfile='sites/3-1-2.txt'
      print('reading from 3-1-2')
      npath=39
      zpath_height=1.8
   elif (site==3 and path==2 and ho==1) :
      pathfile='sites/3-2-1.txt'
      print('reading from 3-2-1')
      npath=55
      zpath_height=1
   elif (site==3 and path==2 and ho==2) :
      pathfile='sites/3-2-2.txt'
      print('reading from 3-2-2')
      npath=55
      zpath_height=1.8
   elif (site==3 and path==3 and ho==1):
      pathfile='sites/3-3-1.txt'
      print('reading from 3-3-1')
      npath=40
      zpath_height=1
   elif (site==3 and path==3 and ho==2) :
      pathfile='sites/3-3-2.txt'
      print('reading from 3-3-2')
      npath=40
      zpath_height=1.8
   elif (site==4 and path==1 and ho==1) :
      pathfile='sites/4-1-1.txt'
      print('reading from 4-1-1')
      npath=38 #AEH
      zpath_height=1
   elif (site==4 and path==1 and ho==2):
      pathfile='sites/4-1-2.txt'
      print('reading from 4-1-2')
      npath=38 #AEH
      zpath_height=1.8
   elif (site==4 and path==2 and ho==1):
      pathfile='sites/4-2-1.txt'
      print('reading from 4-2-1')
      npath=56
      zpath_height=1
   elif (site==4 and path==2 and ho==2) :
      pathfile='sites/4-2-2.txt'
      print('reading from 4-2-2')
      npath=56
      zpath_height=1.8
   elif (site==4 and path==3 and ho==1):
      pathfile='sites/4-3-1.txt'
      print('reading from 4-3-1')
      npath=54 #AEH
      zpath_height=1
   elif (site==4 and path==3 and ho==2) :
      pathfile='sites/4-3-2.txt'
      print('reading from 4-3-2')
      npath=54 #AEH
      zpath_height=1.8
   elif (site==5 and path==1 and ho==1) :
      pathfile='sites/5-1-1.txt'
      print('reading from 5-1-1')
      npath=28
      zpath_height=0.25
   elif (site==5 and path==1 and ho==2) :
      pathfile='sites/5-1-2.txt'
      print('reading from 5-1-2')
      npath=28
      zpath_height=0.75
   elif (site==5 and path==1 and ho==3) :
      pathfile='sites/5-1-3.txt'
      print('reading from 5-1-3')
      npath=28
      zpath_height=1.25
   elif (site==5 and path==1 and ho==4) :
      pathfile='sites/5-1-4.txt'
      print('reading from 5-1-4')
      npath=28 
      zpath_height=1.75
   elif (site==5 and path==2 and ho==1) :
      pathfile='sites/5-2-1.txt'
      print('reading from 5-2-1')
      npath=31 #AEH
      zpath_height=0.25
   elif (site==5 and path==2 and ho==2) :
      pathfile='sites/5-2-2.txt'
      print('reading from 5-2-2')
      npath=31 #AEH
      zpath_height=0.75
   elif (site==5 and path==2 and ho==3) :
      pathfile='sites/5-2-3.txt'
      print('reading from 5-2-3')
      npath=31 #AEH
      zpath_height=1.25
   elif (site==5 and path==2 and ho==4) :
      pathfile='sites/5-2-4.txt'
      print('reading from 5-2-4')
      npath=31 #AEH
      zpath_height=1.75
   elif (site==5 and path==3 and ho==1) :
      pathfile='sites/5-3-1.txt'
      print('reading from 5-3-1')
      npath=27 #AEH
      zpath_height=0.25
   elif (site==5 and path==3 and ho==2) :
      pathfile='sites/5-3-2.txt'
      print('reading from 5-3-2')
      npath=27  #AEH
      zpath_height=0.75
   elif (site==5 and path==3 and ho==3) :
      pathfile='sites/5-3-3.txt'
      print('reading from 5-3-3')
      npath=27  #AEH
      zpath_height=1.25
   elif (site==5 and path==3 and ho==4) :
      pathfile='sites/5-3-4.txt'
      print('reading from 5-3-4')
      npath=27  #AEH
      zpath_height=1.75
   elif (site==6) :
      pathfile='sites/6-1-1.txt'
      print('reading from 6')
      npath=147
      zpath_height=1
   else:
      error=True
      print('unknown site in set_measurement_parameters')

   return Lx,Ly,Lz,nelx,nely,nelz,xllcorner,yllcorner,npath,\
          zpath_height,pathfile,topofile,error,\
          IGRFx,IGRFy,IGRFz,Brefx,Brefy,Brefz
