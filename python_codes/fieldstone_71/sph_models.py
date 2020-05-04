#!/home/romaguir/anaconda2/bin/python
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import pyshtools
from scipy.interpolate import interp1d,RegularGridInterpolator
from copy import deepcopy

n_splines = 21

def read_splines(splines_dir='./data/splines'):

   splines = []
   for i in range(1,n_splines+1):
      g = np.loadtxt(splines_dir+'/spline{}.dat'.format(i))
      splines.append(g)

   return splines

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

