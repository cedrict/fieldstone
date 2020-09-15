import numpy as np
import math 
import scipy 
from scipy import special
from scipy import interpolate

nspline=21 # do not change

degree=40 # or 40, nothing else
use_degree=40

nlat=100
nlon=200

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

###############################################################################
# define 21 knots of the splines anc onvert them to depths
###############################################################################

spline_knots= np.empty(21,dtype=np.float64)  

spline_knots[20]=-1.00000
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
spline_knots[0]=1.00000
print('raw spline knots=',spline_knots)

spline_knots+=1
spline_knots/=2
spline_knots*=2891
print('radii=',spline_knots)

spline_knots*=-1
spline_knots+=2891
print('depths=',spline_knots)

print('compute position of spline knots')

###############################################################################
# use spline functions to compute the value of flm at given depth
###############################################################################
newflm = np.empty((degree+1,2*degree+1),dtype=np.float64)  

depth=2800

for l in range(0,degree+1):  #line of the pyramid
    for m in range(0,2*l+1): #column
        #print('======',l,m)
        yyy=flm[l,m,:]
        #print(yyy)
        tck = interpolate.splrep(spline_knots, yyy, s=0)
        newflm[l,m] = interpolate.splev(depth, tck, der=0)
        #newflm[l,m] = flm[l,m,0]

print('use splines to compute coeffs at desired depth')

#########################################################################################
# generating grid and connectivity for paraview output
#########################################################################################

lons = np.empty(nlat*nlon, dtype=np.float64)
lats = np.empty(nlat*nlon, dtype=np.float64)
counter = 0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        lons[counter]=    ilon*360/float(nlon)
        lats[counter]=90-ilat*180/float(nlat)
        counter += 1
    #end for
#end for
np.savetxt('grid.ascii',np.array([lons,lats]).T)

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

print('generate grid for paraview output')

###############################################################################
# use this new flm array to compute dlnvs at location theta, phi
###############################################################################

dv   = np.zeros(nlat*nlon, dtype=np.float64)  
phis = np.empty(nlat*nlon, dtype=np.float64)
thetas = np.empty(nlat*nlon, dtype=np.float64)


counter = 0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):

        phi=ilon*2*np.pi/(nlon-1)
        theta=ilat*np.pi/(nlat-1)

        phis[counter]=phi
        thetas[counter]=theta

        val=0
        for l in range(0,use_degree+1):
            for m in range(-l,l+1):
                #print('phi=',phi,'theta=',theta,'l=',l,'m=',m,'Ylm=',Ylm.real)

                if m<0:
                   Ylm=scipy.special.sph_harm(abs(m), l, phi, theta) 
                   Y_lm=(-1)**m*Ylm.imag
                   val+=newflm[l,2*abs(m)]*Y_lm
                   #print('l=',l,m,2*abs(m))
                elif m==0:
                   #in this case Ylm is actually real since e^im\phi=1
                   Ylm=scipy.special.sph_harm(0, l, phi, theta)
                   Y_lm=Ylm.real
                   val+=newflm[l,0]*Y_lm
                else:
                   Ylm=scipy.special.sph_harm(m, l, phi, theta) 
                   Y_lm=(-1)**m*Ylm.real
                   val+=newflm[l,2*m-1]*Y_lm
                   #print('l=',l,m,2*m-1)

            #end for
        #end for

        dv[counter] = val 

        counter+=1
    #end for
#end for

np.savetxt('grid.ascii',np.array([phis,thetas,dv]).T)

###############################################################################
#benchmarking results for l=0,1,2:
###############################################################################
#sol = np.zeros(nlat*nlon, dtype=np.float64)
#if use_degree==0:
#   sol[:]=np.sqrt(1./4./np.pi)*newflm[0,0]
#if use_degree==1:
#   sol[:]=np.sqrt(1./4./np.pi)*newflm[0,0]\
#         +np.sqrt(3./8./np.pi)*np.cos(phis[:])*np.sin(thetas[:])*newflm[1,-1+1]\
#         +np.sqrt(3./4./np.pi)*np.cos(thetas[:])*newflm[1,0+1]\
#         -np.sqrt(3./8./np.pi)*np.cos(phis[:])*np.sin(thetas[:])*newflm[1,1+1]
#if use_degree==2:
#   sol[:]=np.sqrt(1./4./np.pi)*newflm[0,0]\
#         +np.sqrt(3./8./np.pi)*np.cos(phis[:])*np.sin(thetas[:])*newflm[1,-1+1]\
#         +np.sqrt(3./4./np.pi)*np.cos(thetas[:])                *newflm[1,0+1]\
#         -np.sqrt(3./8./np.pi)*np.cos(phis[:])*np.sin(thetas[:])*newflm[1,1+1]\
#         +np.sqrt(15./32./np.pi)*np.cos(2*phis[:])*(np.sin(thetas[:]))**2           *newflm[2,-2+2]\
#         +np.sqrt(15./8./np.pi) *np.cos(phis[:])*np.sin(thetas[:])*np.cos(thetas[:])*newflm[2,-1+2]\
#         +np.sqrt(5./16./np.pi) *(3*np.cos(thetas[:])**2-1)                         *newflm[2,0+2]\
#         -np.sqrt(15./8./np.pi) *np.cos(phis[:])*np.sin(thetas[:])*np.cos(thetas[:])*newflm[2,1+2]\
#         +np.sqrt(15./32./np.pi)*np.cos(2*phis[:])*(np.sin(thetas[:]))**2           *newflm[2,2+2]

#########################################################################################
# trying splines
#########################################################################################
#x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
#y = np.sin(x)
#tck = interpolate.splrep(x, y, s=0)
#print(x,y)
#print(tck)
#y= np.zeros(21,dtype=np.float64)  
#y[:]=1
#print(spline_knots)
#print(y)
#produce array of depths going from 0 to 2891km:
#depths=np.zeros(2892,dtype=np.float64)
#for i in range(0,2892):
#    depths[i]=i
#print(depths)
#tck = interpolate.splrep(spline_knots, y, s=0)
#ynew = interpolate.splev(depths, tck, der=0)
#print(ynew)
#exit()

#########################################################################################
# read spline files by maguire
#########################################################################################

#spline_functions= np.empty((2891,21),dtype=np.float64)  
#for spl_nb in range(0,21):
#    f = open('splines/spline{:02d}'.format(spl_nb+1)+'.dat')
#    lines = f.readlines()
#    f.close
#    for i in range(0,2891):
#        vals=lines[i].strip().split()
#        spline_functions[i,spl_nb]=vals[1]
    #end for
#end for
#for i in range(0,2891):
#    print(np.sum(spline_functions[i,0:21])-1)
# gives 1 with accuracy of about 1e-7


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
    vtufile.write("%10f %10f %10f \n" %(lons[i]-180.,lats[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")

vtufile.write("<DataArray type='Float32' Name='dv/v (%)' Format='ascii'> \n")
for i in range (0,NV):
    vtufile.write("%f\n" % (dv[i]*100))
vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='sol' Format='ascii'> \n")
#for i in range (0,NV):
#    vtufile.write("%f\n" % sol[i])
#vtufile.write("</DataArray>\n")

vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
   vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
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

radius=6370e3

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
    vtufile.write("%f\n" % (dv[i]*100))
vtufile.write("</DataArray>\n")

vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
   vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print('produced sphere.vtu')

