# pip install pyshtools

import pyshtools
import numpy as np
from copy import deepcopy
from os import path
from scipy.interpolate import interp1d

n_splines = 21

#########################################################################################

def read_splines(par='S40RTS'):
   '''
   Read radial spline functions. 
   Currently, the only spline functions available are those used in S40RTS
   '''

   if par=='S40RTS':
      splines_dir = path.join(path.dirname(__file__), 'data/splines')

   splines = []
   for i in range(1,n_splines+1):
      g = np.loadtxt(splines_dir+'/spline{}.dat'.format(i))
      splines.append(g)

   return splines

#########################################################################################

def read_sph(sph_file,lmin=0,lmax=40):

   #Read header
   f = open(sph_file)
   lines = f.readlines()
   sph_degree = int(lines[0].strip().split()[0])
   f.close()
   
   #Read spherical harmonic coefficients
   vals = []
   for line in lines[1:]:
      vals_list = line.strip().split()
      for val in vals_list:
         vals.append(np.float(val))

   vals = np.array(vals)
   sph_splines = []
   count = 0

   for k in range(0,n_splines):
      coeffs = np.zeros((2,sph_degree+1,sph_degree+1))
      for l in range(0,sph_degree+1):
         ind = 0
         nm = (l*2) + 1
         c_row = []
         n_m = 1. 
   
         for m in range(0,(2*l)+1):
            if m == 0:
               order = 0
               coeffs[0,l,0] = vals[count]
               count += 1
            elif np.mod(m,2) == 1:
               order = int(m * (n_m)/m)
               coeffs[0,l,order] = vals[count]
               count += 1
            elif np.mod(m,2) == 0:
               order = int(m * (n_m)/m)
               coeffs[1,l,order] = vals[count]
               count += 1
               n_m += 1

      #filter out unwanted degrees
      if lmin != 0:
         coeffs[0,0:lmin,:] = 0.0
         coeffs[1,0:lmin,:] = 0.0
      if lmax < 40:
         coeffs[0,lmax+1:,:] = 0.0
         coeffs[1,lmax+1:,:] = 0.0

      clm = pyshtools.SHCoeffs.from_array(coeffs,normalization='ortho',csphase=-1)
      sph_splines.append(deepcopy(clm))

   return sph_splines

#########################################################################################

def find_spl_vals(depth):

   splines = read_splines()
   spl_vals = []

   if depth < 0 or depth > 2891:
      raise ValueError("depth should be in range 0 - 2891 km")

   for spline in splines:
      z = spline[:,0]
      f_z = spline[:,1]
      get_val = interp1d(z,f_z)
      spl_val = get_val(depth)
      spl_vals.append(spl_val)

   return spl_vals 

#########################################################################################

lmin=0
lmax=40
depth=2800

#datafilename='SEM_WM_s'
#datafilename='GypSum_S'
datafilename='S40RTS'
#datafilename='S20RTS'
#datafilename='s362ani_m_s'
#datafilename='GAP_P4'
#datafilename='SP12RTS..EC'
#datafilename='SP12RTS..ES'
#datafilename='SP12RTS..EP'

sph_splines = read_sph('data/models/'+datafilename+'.sph',lmin,lmax)
spl_vals = find_spl_vals(depth)

nlat=360
nlon=720

#########################################################################################
# computing dv from data model at center of nlon*nlat cells
#########################################################################################

dv   = np.zeros(nlat*nlon, dtype=np.float64)  # y coordinates
counter = 0 
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        mylon=   +(ilon+0.5)*360./nlon
        mylat=-90+(ilat+0.5)*180./nlat
        for i,sph_spline in enumerate(sph_splines):
            vals = sph_spline.expand(lat=mylat,lon=mylon)
            dv[counter] += spl_vals[i] * vals * 100.0
        #end for
        counter+=1
    #end for
#end for

#########################################################################################
# generating grid and connectivity for paraview output
#########################################################################################

lons = np.empty((nlat+1)*(nlon+1), dtype=np.float64)
lats = np.empty((nlat+1)*(nlon+1), dtype=np.float64)
counter = 0 
for ilat in range(0,nlat+1):
    for ilon in range(0,nlon+1):
        lons[counter]=    ilon*360/float(nlon)
        lats[counter]=-90+ilat*180/float(nlat)
        counter += 1
    #end for
#end for
#np.savetxt('grid.ascii',np.array([lons,lats]).T)

icon =np.zeros((4,nlat*nlon),dtype=np.int32)
counter = 0 
for ilat in range(0, nlat):
    for ilon in range(0, nlon):
        icon[0, counter] = ilon + ilat * (nlon + 1)
        icon[1, counter] = ilon + 1 + ilat * (nlon + 1)
        icon[2, counter] = ilon + 1 + (ilat + 1) * (nlon + 1)
        icon[3, counter] = ilon + (ilat + 1) * (nlon + 1)
        counter += 1
    #end for
#end for

#########################################################################################
# export to vtu 
#########################################################################################

nel=nlat*nlon
NV=(nlat+1)*(nlon+1)
vtufile=open(datafilename+".vtu","w")
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
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='dV' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%f\n" % (dv[iel]))
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
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

