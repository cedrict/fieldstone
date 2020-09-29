import numpy as np
import math 
import scipy 
from scipy import special
from scipy import interpolate

###############################################################################
# do not change
###############################################################################

nspline=21 
nlat=179
nlon=360

###############################################################################

degree=20 # 20 or 40, nothing else
use_degree=20 # less or equal to degree

#depth=24.381e3
#depth=1251164.934145
#depth=2500e3 
#depth=100e3 
#depth=2800e3 
#depth=2891e3
#depth=1500e3 # ok
#depth=600e3 #ok 
#depth=2584716.092945
depth=100e3

###############################################################################
# read dataset in data[l,2l+1,npline] array
###############################################################################
#if degree=20, then l=0,...20, so 21 values of l
#if degree=40, then l=0,...40, so 41 values of l

flm = np.empty((degree+1,2*degree+1,nspline),dtype=np.float64)  

if degree==20:
   print('using S20RTS.sph file')
   f = open('S20RTS.sph', 'r')
else:
   print('using S40RTS.sph file')
   #f = open('S40_utrecht.sph', 'r')
   f = open('S40RTS.sph', 'r')
lines = f.readlines()
f.close

counter=1 # discarding first line of file

for ispline in range(0,nspline):

    #print('     reading coeffs for spline #',ispline)

    for l in range(0,degree+1):
        # nread is the number of coefficients to be read
        # for the current value of l
        # The problem is that the .sph files structure
        # is idiotic (maximum of 11 numbers per line)
        # so that a complex algo must be put in place 
        # to read them all in properly.
        nread=2*l+1

        if nread <= 11: 
           vals=lines[counter].strip().split()
           #print('l=',l,vals)
           for i in range(0,nread):
               flm[l,i,ispline]=float(vals[i])
           counter+=1

           #print('l=',l,flm[l,0:nread,ispline])

        elif nread <= 22: 

           # read 1st line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,i,ispline]=float(vals[i])
           counter+=1

           # read 2nd line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,nread-11):
               flm[l,11+i,ispline]=float(vals[i])
           counter+=1

           #print('l=',l,flm[l,0:nread,ispline])

        elif nread <= 33:
           # read 1st line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,i,ispline]=float(vals[i])
           counter+=1

           # read 2nd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,11+i,ispline]=float(vals[i])
           counter+=1

           # read 3rd line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,nread-22):
               flm[l,22+i,ispline]=float(vals[i])
           counter+=1

           #print('l=',l,flm[l,0:nread,ispline])

        elif nread <= 44:

           # read 1st line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,i,ispline]=float(vals[i])
           counter+=1

           # read 2nd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,11+i,ispline]=float(vals[i])
           counter+=1

           # read 3rd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,22+i,ispline]=float(vals[i])
           counter+=1

           # read 4th line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,nread-33):
               flm[l,33+i,ispline]=float(vals[i])
           counter+=1

           #print('l=',l,flm[l,0:nread,ispline])

        elif nread <= 55:

           # read 1st line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,i,ispline]=float(vals[i])
           counter+=1

           # read 2nd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,11+i,ispline]=float(vals[i])
           counter+=1

           # read 3rd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,22+i,ispline]=float(vals[i])
           counter+=1

           # read 4th line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,33+i,ispline]=float(vals[i])
           counter+=1

           # read 5th line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,nread-44):
               flm[l,44+i,ispline]=float(vals[i])
           counter+=1

           #print('l=',l,flm[l,0:nread,ispline])

        elif nread <= 66:

           # read 1st line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,i,ispline]=float(vals[i])
           counter+=1

           # read 2nd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,11+i,ispline]=float(vals[i])
           counter+=1

           # read 3rd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,22+i,ispline]=float(vals[i])
           counter+=1

           # read 4th line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,33+i,ispline]=float(vals[i])
           counter+=1

           # read 5th line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,44+i,ispline]=float(vals[i])
           counter+=1

           # read 6th line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,nread-55):
               flm[l,55+i,ispline]=float(vals[i])
           counter+=1

           #print('l=',l,flm[l,0:nread,ispline])

        elif nread <= 77:

           # read 1st line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,i,ispline]=float(vals[i])
           counter+=1

           # read 2nd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,11+i,ispline]=float(vals[i])
           counter+=1

           # read 3rd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,22+i,ispline]=float(vals[i])
           counter+=1

           # read 4th line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,33+i,ispline]=float(vals[i])
           counter+=1

           # read 5th line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,44+i,ispline]=float(vals[i])
           counter+=1

           # read 6th line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,55+i,ispline]=float(vals[i])
           counter+=1

           # read 7th line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,nread-66):
               flm[l,66+i,ispline]=float(vals[i])
           counter+=1

           #print('l=',l,flm[l,0:nread,ispline])

        elif nread <= 88:

           # read 1st line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,i,ispline]=float(vals[i])
           counter+=1

           # read 2nd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,11+i,ispline]=float(vals[i])
           counter+=1

           # read 3rd line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,22+i,ispline]=float(vals[i])
           counter+=1

           # read 4th line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,33+i,ispline]=float(vals[i])
           counter+=1

           # read 5th line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,44+i,ispline]=float(vals[i])
           counter+=1

           # read 6th line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,55+i,ispline]=float(vals[i])
           counter+=1

           # read 7th line of 11 coeffs
           vals=lines[counter].strip().split()
           for i in range(0,11):
               flm[l,66+i,ispline]=float(vals[i])
           counter+=1

           # read 8th line with rest of coeffs
           vals=lines[counter].strip().split()
           for i in range(0,nread-77):
               flm[l,77+i,ispline]=float(vals[i])
           counter+=1

           #print('l=',l,flm[l,0:nread,ispline])

    #end for

#end for

print('read coefficients off .sph file')

#########################################################################################
# generating grid and connectivity for paraview output
#########################################################################################

lons = np.empty(nlat*nlon, dtype=np.float64)
lats = np.empty(nlat*nlon, dtype=np.float64)

counter = 0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        lons[counter]=    ilon*360/float(nlon)  #starts at 0 then goes to 360
        lats[counter]=-89+ilat*179/float(nlat)   #starts at -90 then goes to +90
        counter += 1
    #end for
#end for

icon =np.zeros((4,(nlat-1)*(nlon-1)),dtype=np.int32)

counter = 0
for ilat in range(0, nlat-1):
    for ilon in range(0, nlon-1):
        icon[0, counter] = ilon + ilat * (nlon)
        icon[1, counter] = ilon + 1 + ilat * (nlon)
        icon[2, counter] = ilon + 1 + (ilat + 1) * (nlon)
        icon[3, counter] = ilon + (ilat + 1) * (nlon)
        counter += 1
    #end for
#end for

print('min/max lats:',np.min(lats),np.max(lats))
print('min/max lons:',np.min(lons),np.max(lons))

print('generate grid for paraview output')

###############################################################################
# define 21 knots of the splines anc onvert them to depths
###############################################################################
#This numbering corresponds to the data in the .sph file. 
#The first pyramid of coefficients corresponds to the moho spline. 

spline_knots= np.empty(nspline,dtype=np.float64)  

spline_knots[20]=-1.00000 #cmb
spline_knots[19]=-0.78631
spline_knots[18]=-0.59207
spline_knots[17]=-0.41550
spline_knots[16]=-0.25499
spline_knots[15]=-0.10909 
spline_knots[14]=0.02353 
spline_knots[13]=0.14409 
spline_knots[12]=0.25367 
spline_knots[11]=0.35329
spline_knots[10]=0.44384 
spline_knots[9]=0.52615 
spline_knots[8]=0.60097 
spline_knots[7]=0.66899 
spline_knots[6]=0.73081
spline_knots[5]=0.78701 
spline_knots[4]=0.83810 
spline_knots[3]=0.88454 
spline_knots[2]=0.92675 
spline_knots[1]=0.96512 
spline_knots[0]=1.00000 #moho

rcmb=3480e3
rmoho=6346.619e3
spline_knots[:]=rcmb+(rmoho-rcmb)*(spline_knots[:]+1)*0.5
print('radii=',spline_knots)

spline_depths=6371e3-spline_knots
print('depths=',spline_depths)

print('compute position of spline knots')

###############################################################################
###############################################################################
phis = np.empty(nlat*nlon, dtype=np.float64)
thetas = np.empty(nlat*nlon, dtype=np.float64)
dv   = np.zeros(nlat*nlon, dtype=np.float64)  

a_coeffs   = np.zeros((use_degree+1,use_degree+1), dtype=np.float64)  
b_coeffs   = np.zeros((use_degree+1,use_degree+1), dtype=np.float64)  

counter = 0
for ilat in range(0,nlat): #179
        for ilon in range(0,nlon): #360

            if counter%180==0:
               print(counter,'/',nlat*nlon)

            phi=lons[counter]*np.pi/180.
            theta=np.pi-(90.+lats[counter])*np.pi/180.

            #puts africa in the middle of map
            #to compare with submachine
            #phi+=np.pi      
            #if phi>2*np.pi:
            #   phi-=2*np.pi

            phis[counter]=phi
            thetas[counter]=theta

            #evaluate coeffs in front of alm and blm
            #these do not depend on ispline 
            #store these for further re-use
            for l in range(0,use_degree+1):
                m=0
                x=np.cos(theta)
                Xlm=np.sqrt( (2*l+1)/4./np.pi ) * scipy.special.lpmv(m,l,x) 
                a_coeffs[l,0]=Xlm
                for m in range(1,l+1):
                    A=math.factorial(l-m)
                    B=math.factorial(l+m)
                    x=np.cos(theta)
                    Xlm=np.sqrt( (2*l+1)/4./np.pi*float(A)/float(B) ) * scipy.special.lpmv(m,l,x) 
                    a_coeffs[l,m]=Xlm*np.cos(m*phi)
                    b_coeffs[l,m]=Xlm*np.sin(m*phi)
                #end for
            #end for

            #now go through 21 spline shells and compute 21 values of dv at this depth
            shell_values=np.zeros(nspline,dtype=np.float64)  

            for ispline in range(0,nspline):
                for l in range(0,use_degree+1):
                    shell_values[ispline]+=flm[l,0,ispline]*a_coeffs[l,0]
                    for m in range(1,l+1):
                        alm=flm[l,2*m-1,ispline]
                        blm=flm[l,2*m,ispline]
                        shell_values[ispline]+=a_coeffs[l,m]*alm+b_coeffs[l,m]*blm
                    #end for
                #end for
            #end for

            #use spline coeffs to interpolate dv at right depth
            tck=interpolate.splrep(spline_depths, shell_values)
            dv[counter]=100*interpolate.splev(depth,tck)

            counter+=1

        #end for
    #end for
#end for

print (np.sum(dv)/(nlat*nlon))

np.savetxt('seismic_velocity_anomaly.ascii',np.array([lons,lats,dv,phis,thetas]).T)

print('compute seismic anomaly on grid')

#########################################################################################

S20RTS_dvRmoho = np.zeros(nlat*nlon,dtype=np.float64)  
S20RTS_dvR100  = np.zeros(nlat*nlon,dtype=np.float64)  
S20RTS_dvR600  = np.zeros(nlat*nlon,dtype=np.float64)  
S20RTS_dvR1500 = np.zeros(nlat*nlon,dtype=np.float64)  
S20RTS_dvR2800 = np.zeros(nlat*nlon,dtype=np.float64)  
S20RTS_dvRcmb  = np.zeros(nlat*nlon,dtype=np.float64)  

S40RTS_dvRmoho = np.zeros(nlat*nlon,dtype=np.float64)  
S40RTS_dvR100  = np.zeros(nlat*nlon,dtype=np.float64)  
S40RTS_dvR600  = np.zeros(nlat*nlon,dtype=np.float64)  
S40RTS_dvR1500 = np.zeros(nlat*nlon,dtype=np.float64)  
S40RTS_dvR2800 = np.zeros(nlat*nlon,dtype=np.float64)  
S40RTS_dvRcmb  = np.zeros(nlat*nlon,dtype=np.float64)  

if True:

   #----S20RTS----

   f = open('S20RTS_plotting/bin/mapS20RTS_moho.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S20RTS_dvRmoho[nlon*ilat+ilon]=vals[2]  
           counter+=1

   f = open('S20RTS_plotting/bin/mapS20RTS_100.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S20RTS_dvR100[nlon*ilat+ilon]=vals[2]  
           counter+=1

   f = open('S20RTS_plotting/bin/mapS20RTS_600.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S20RTS_dvR600[nlon*ilat+ilon]=vals[2]            
           counter+=1

   f = open('S20RTS_plotting/bin/mapS20RTS_1500.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S20RTS_dvR1500[nlon*ilat+ilon]=vals[2]            
           counter+=1

   f = open('S20RTS_plotting/bin/mapS20RTS_2800.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S20RTS_dvR2800[nlon*ilat+ilon]=vals[2]            
           counter+=1

   f = open('S20RTS_plotting/bin/mapS20RTS_cmb.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S20RTS_dvRcmb[nlon*ilat+ilon]=vals[2]            
           counter+=1



   #----S40RTS----

   f = open('S20RTS_plotting/bin/mapS40RTS_moho.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S40RTS_dvRmoho[nlon*ilat+ilon]=vals[2]  
           counter+=1

   f = open('S20RTS_plotting/bin/mapS40RTS_100.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S40RTS_dvR100[nlon*ilat+ilon]=vals[2]  
           counter+=1

   f = open('S20RTS_plotting/bin/mapS40RTS_600.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S40RTS_dvR600[nlon*ilat+ilon]=vals[2]            
           counter+=1

   f = open('S20RTS_plotting/bin/mapS40RTS_1500.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S40RTS_dvR1500[nlon*ilat+ilon]=vals[2]            
           counter+=1

   f = open('S20RTS_plotting/bin/mapS40RTS_2800.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S40RTS_dvR2800[nlon*ilat+ilon]=vals[2]            
           counter+=1

   f = open('S20RTS_plotting/bin/mapS40RTS_cmb.xyz','r')
   lines = f.readlines()
   f.close
   counter=0
   for ilon in range(0,360):
       for ilat in range(0,179):
           vals=lines[counter].strip().split()
           S40RTS_dvRcmb[nlon*ilat+ilon]=vals[2]            
           counter+=1



print('read Ritsema data')

#########################################################################################
# export map to vtu 
#########################################################################################

nel=(nlat-1)*(nlon-1)
NV=nlat*nlon

vtufile=open("map.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10f %10f %10f \n" %(lons[i],lats[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")

vtufile.write("<DataArray type='Float32' Name='lons' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (lons[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='lats' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (lats[i]))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='dv/v (%)' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (dv[i]))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='dv/v (%), S20RTS, moho' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S20RTS_dvRmoho[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S20RTS, 100km' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S20RTS_dvR100[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S20RTS, 600km' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S20RTS_dvR600[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S20RTS, 1500km' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S20RTS_dvR1500[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S20RTS, 2800km' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S20RTS_dvR2800[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S20RTS, cmb' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S20RTS_dvRcmb[i]))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='dv/v (%), S40RTS, moho' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S40RTS_dvRmoho[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S40RTS, 100km' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S40RTS_dvR100[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S40RTS, 600km' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S40RTS_dvR600[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S40RTS, 1500km' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S40RTS_dvR1500[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S40RTS, 2800km' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S40RTS_dvR2800[i]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%), S40RTS, cmb' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (S40RTS_dvRcmb[i]))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='diff' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (dv[i]-S40RTS_dvR600[i]))
vtufile.write("</DataArray>\n")

vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
   vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print('produced map.vtu')

###############################################################################
# produce sphere.vtu 
###############################################################################

radius=6371e3

vtufile=open("sphere.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10f %10f %10f \n" %(radius*np.sin(thetas[i])*np.cos(phis[i]-np.pi),\
                                        radius*np.sin(thetas[i])*np.sin(phis[i]-np.pi),\
                                        radius*np.cos(thetas[i])))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='dv/v (%)' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (dv[i]))
vtufile.write("</DataArray>\n")

vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
   vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print('produced sphere.vtu')

#########################################################################################
