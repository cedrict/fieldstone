import numpy as np
import math as math
import sys as sys
import scipy
import time as time
import random

#------------------------------------------------------------------------------

def u_th(x,y,z):
    val=(y+z)*(1-x**2)
    return val

def v_th(x,y,z):
    val=(-x+z)*(1-y**2)
    return val

def w_th(x,y,z):
    val=(-x-y)*(1-z**2)
    return val

def exx_th(x,y,z):
    val=-2*x*(y+z)
    return val

def eyy_th(x,y,z):
    val=-2*y*(-x+z)
    return val

def ezz_th(x,y,z):
    val=-2*z*(-x-z)
    return val

def exy_th(x,y,z):
    val=0.5*(y**2-x**2)
    return val

def exz_th(x,y,z):
    val=0.5*(z**2-x**2)
    return val

def eyz_th(x,y,z):
    val=0.5*(y**2-x**2)
    return val

#------------------------------------------------------------------------------

def NQ1(rq,sq,tq):
    N0=0.125*(1.-rq)*(1.-sq)*(1.-tq)
    N1=0.125*(1.+rq)*(1.-sq)*(1.-tq)
    N2=0.125*(1.+rq)*(1.+sq)*(1.-tq)
    N3=0.125*(1.-rq)*(1.+sq)*(1.-tq)
    N4=0.125*(1.-rq)*(1.-sq)*(1.+tq)
    N5=0.125*(1.+rq)*(1.-sq)*(1.+tq)
    N6=0.125*(1.+rq)*(1.+sq)*(1.+tq)
    N7=0.125*(1.-rq)*(1.+sq)*(1.+tq)
    return N0,N1,N2,N3,N4,N5,N6,N7

def dNQ1dr(rq,sq,tq):
    dNdr0=-0.125*(1.-sq)*(1.-tq) 
    dNdr1=+0.125*(1.-sq)*(1.-tq)
    dNdr2=+0.125*(1.+sq)*(1.-tq)
    dNdr3=-0.125*(1.+sq)*(1.-tq)
    dNdr4=-0.125*(1.-sq)*(1.+tq)
    dNdr5=+0.125*(1.-sq)*(1.+tq)
    dNdr6=+0.125*(1.+sq)*(1.+tq)
    dNdr7=-0.125*(1.+sq)*(1.+tq)
    return dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7

def dNQ1ds(rq,sq,tq):
    dNds0=-0.125*(1.-rq)*(1.-tq) 
    dNds1=-0.125*(1.+rq)*(1.-tq)
    dNds2=+0.125*(1.+rq)*(1.-tq)
    dNds3=+0.125*(1.-rq)*(1.-tq)
    dNds4=-0.125*(1.-rq)*(1.+tq)
    dNds5=-0.125*(1.+rq)*(1.+tq)
    dNds6=+0.125*(1.+rq)*(1.+tq)
    dNds7=+0.125*(1.-rq)*(1.+tq)
    return dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7

def dNQ1dt(rq,sq,tq):
    dNdt0=-0.125*(1.-rq)*(1.-sq) 
    dNdt1=-0.125*(1.+rq)*(1.-sq)
    dNdt2=-0.125*(1.+rq)*(1.+sq)
    dNdt3=-0.125*(1.-rq)*(1.+sq)
    dNdt4=+0.125*(1.-rq)*(1.-sq)
    dNdt5=+0.125*(1.+rq)*(1.-sq)
    dNdt6=+0.125*(1.+rq)*(1.+sq)
    dNdt7=+0.125*(1.-rq)*(1.+sq)
    return dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7

def NQ2(rq,sq,tq):
    NV_00= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    NV_01= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    NV_02= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_03= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_04= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_05= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_06= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_07= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_08= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.)
    NV_09= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_10= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_11= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_12= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_13= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_14= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_15= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_16= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_17= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_18= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_19= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_20= (1.-rq**2)     * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_21= (1.-rq**2)     * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_22= 0.5*rq*(rq+1.) * (1.-sq**2)     * (1.-tq**2)
    NV_23= (1.-rq**2)     * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_24= 0.5*rq*(rq-1.) * (1.-sq**2)     * (1.-tq**2)
    NV_25= (1.-rq**2)     * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_26= (1.-rq**2)     * (1.-sq**2)     * (1.-tq**2)
    return NV_00,NV_01,NV_02,NV_03,NV_04,NV_05,NV_06,NV_07,NV_08,\
           NV_09,NV_10,NV_11,NV_12,NV_13,NV_14,NV_15,NV_16,NV_17,\
           NV_18,NV_19,NV_20,NV_21,NV_22,NV_23,NV_24,NV_25,NV_26

def dNQ2dr(rq,sq,tq):
    dNVdr_00= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    dNVdr_01= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    dNVdr_02= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.) 
    dNVdr_03= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.) 
    dNVdr_04= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.) 
    dNVdr_05= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.) 
    dNVdr_06= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.) 
    dNVdr_07= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.) 
    dNVdr_08= (-2*rq)       * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    dNVdr_09= 0.5*(2*rq+1.) * (1.-sq**2)     * 0.5*tq*(tq-1.) 
    dNVdr_10= (-2*rq)       * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.) 
    dNVdr_11= 0.5*(2*rq-1.) * (1.-sq**2)     * 0.5*tq*(tq-1.) 
    dNVdr_12= (-2*rq)       * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.) 
    dNVdr_13= 0.5*(2*rq+1.) * (1.-sq**2)     * 0.5*tq*(tq+1.) 
    dNVdr_14= (-2*rq)       * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.) 
    dNVdr_15= 0.5*(2*rq-1.) * (1.-sq**2)     * 0.5*tq*(tq+1.) 
    dNVdr_16= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * (1.-tq**2) 
    dNVdr_17= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * (1.-tq**2) 
    dNVdr_18= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * (1.-tq**2) 
    dNVdr_19= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * (1.-tq**2) 
    dNVdr_20= (-2*rq)       * (1.-sq**2)     * 0.5*tq*(tq-1.) 
    dNVdr_21= (-2*rq)       * 0.5*sq*(sq-1.) * (1.-tq**2) 
    dNVdr_22= 0.5*(2*rq+1.) * (1.-sq**2)     * (1.-tq**2) 
    dNVdr_23= (-2*rq)       * 0.5*sq*(sq+1.) * (1.-tq**2) 
    dNVdr_24= 0.5*(2*rq-1.) * (1.-sq**2)     * (1.-tq**2) 
    dNVdr_25= (-2*rq)       * (1.-sq**2)     * 0.5*tq*(tq+1.) 
    dNVdr_26= (-2*rq)       * (1.-sq**2)     * (1.-tq**2) 
    return dNVdr_00,dNVdr_01,dNVdr_02,dNVdr_03,dNVdr_04,dNVdr_05,dNVdr_06,dNVdr_07,dNVdr_08,\
           dNVdr_09,dNVdr_10,dNVdr_11,dNVdr_12,dNVdr_13,dNVdr_14,dNVdr_15,dNVdr_16,dNVdr_17,\
           dNVdr_18,dNVdr_19,dNVdr_20,dNVdr_21,dNVdr_22,dNVdr_23,dNVdr_24,dNVdr_25,dNVdr_26

def dNQ2ds(rq,sq,tq):
    dNVds_00= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.) 
    dNVds_01= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.) 
    dNVds_02= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.) 
    dNVds_03= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.) 
    dNVds_04= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.) 
    dNVds_05= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.) 
    dNVds_06= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.) 
    dNVds_07= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.) 
    dNVds_08= (1.-rq**2)     * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.) 
    dNVds_09= 0.5*rq*(rq+1.) * (-2*sq)       * 0.5*tq*(tq-1.) 
    dNVds_10= (1.-rq**2)     * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.) 
    dNVds_11= 0.5*rq*(rq-1.) * (-2*sq)       * 0.5*tq*(tq-1.) 
    dNVds_12= (1.-rq**2)     * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.) 
    dNVds_13= 0.5*rq*(rq+1.) * (-2*sq)       * 0.5*tq*(tq+1.) 
    dNVds_14= (1.-rq**2)     * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.) 
    dNVds_15= 0.5*rq*(rq-1.) * (-2*sq)       * 0.5*tq*(tq+1.) 
    dNVds_16= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * (1.-tq**2) 
    dNVds_17= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * (1.-tq**2) 
    dNVds_18= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * (1.-tq**2) 
    dNVds_19= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * (1.-tq**2) 
    dNVds_20= (1.-rq**2)     * (-2*sq)       * 0.5*tq*(tq-1.) 
    dNVds_21= (1.-rq**2)     * 0.5*(2*sq-1.) * (1.-tq**2) 
    dNVds_22= 0.5*rq*(rq+1.) * (-2*sq)       * (1.-tq**2) 
    dNVds_23= (1.-rq**2)     * 0.5*(2*sq+1.) * (1.-tq**2) 
    dNVds_24= 0.5*rq*(rq-1.) * (-2*sq)       * (1.-tq**2) 
    dNVds_25= (1.-rq**2)     * (-2*sq)       * 0.5*tq*(tq+1.) 
    dNVds_26= (1.-rq**2)     * (-2*sq)       * (1.-tq**2) 
    return dNVds_00,dNVds_01,dNVds_02,dNVds_03,dNVds_04,dNVds_05,dNVds_06,dNVds_07,dNVds_08,\
           dNVds_09,dNVds_10,dNVds_11,dNVds_12,dNVds_13,dNVds_14,dNVds_15,dNVds_16,dNVds_17,\
           dNVds_18,dNVds_19,dNVds_20,dNVds_21,dNVds_22,dNVds_23,dNVds_24,dNVds_25,dNVds_26

def dNQ2dt(rq,sq,tq):
    dNVdt_00= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.) 
    dNVdt_01= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.) 
    dNVdt_02= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.) 
    dNVdt_03= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.) 
    dNVdt_04= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.) 
    dNVdt_05= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.) 
    dNVdt_06= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.) 
    dNVdt_07= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.) 
    dNVdt_08= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.) 
    dNVdt_09= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*(2*tq-1.) 
    dNVdt_10= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.) 
    dNVdt_11= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*(2*tq-1.) 
    dNVdt_12= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.) 
    dNVdt_13= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*(2*tq+1.) 
    dNVdt_14= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.) 
    dNVdt_15= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*(2*tq+1.) 
    dNVdt_16= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * (-2*tq) 
    dNVdt_17= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * (-2*tq) 
    dNVdt_18= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * (-2*tq) 
    dNVdt_19= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * (-2*tq) 
    dNVdt_20= (1.-rq**2)     * (1.-sq**2)     * 0.5*(2*tq-1.) 
    dNVdt_21= (1.-rq**2)     * 0.5*sq*(sq-1.) * (-2*tq) 
    dNVdt_22= 0.5*rq*(rq+1.) * (1.-sq**2)     * (-2*tq) 
    dNVdt_23= (1.-rq**2)     * 0.5*sq*(sq+1.) * (-2*tq) 
    dNVdt_24= 0.5*rq*(rq-1.) * (1.-sq**2)     * (-2*tq) 
    dNVdt_25= (1.-rq**2)     * (1.-sq**2)     * 0.5*(2*tq+1.) 
    dNVdt_26= (1.-rq**2)     * (1.-sq**2)     * (-2*tq) 
    return dNVdt_00,dNVdt_01,dNVdt_02,dNVdt_03,dNVdt_04,dNVdt_05,dNVdt_06,dNVdt_07,dNVdt_08,\
           dNVdt_09,dNVdt_10,dNVdt_11,dNVdt_12,dNVdt_13,dNVdt_14,dNVdt_15,dNVdt_16,dNVdt_17,\
           dNVdt_18,dNVdt_19,dNVdt_20,dNVdt_21,dNVdt_22,dNVdt_23,dNVdt_24,dNVdt_25,dNVdt_26

def interpolate_vel_on_pt(xm,ym,zm,x,y,z,u,v,w,icon,Lx,Ly,Lz,nelx,nely,nelz,m,Q):
    ielx=int((xm+Lx/2)/Lx*nelx)
    iely=int((ym+Ly/2)/Ly*nely)
    ielz=int((zm+Lz/2)/Ly*nelz)
    iel=nely*nelz*ielx+nelz*iely+ielz
    xx=x[icon[0:m,iel]]
    yy=y[icon[0:m,iel]]
    zz=z[icon[0:m,iel]]
    xmin=min(xx) ; xmax=max(xx)
    ymin=min(yy) ; ymax=max(yy)
    zmin=min(zz) ; zmax=max(zz)
    rm=((xm-xmin)/(xmax-xmin)-0.5)*2
    sm=((ym-ymin)/(ymax-ymin)-0.5)*2
    tm=((zm-zmin)/(zmax-zmin)-0.5)*2
    if Q==1:
       N[0:m]=NQ1(rm,sm,tm)
    if Q==2:
       N[0:m]=NQ2(rm,sm,tm)
    um=0.
    vm=0.
    wm=0.
    for k in range(0,m):
        um+=N[k]*u[icon[k,iel]]
        vm+=N[k]*v[icon[k,iel]]
        wm+=N[k]*w[icon[k,iel]]
    return um,vm,wm,rm,sm,tm,iel 

def compute_CVI_corr (u,v,w,icon,rm,sm,tm,iel,use_cvi,Q):
    if use_cvi==1 and Q==1:
       dNdr=dNQ1dr(rm,sm,tm)
       dNds=dNQ1ds(rm,sm,tm)
       dNdt=dNQ1dt(rm,sm,tm)
       # calculate jacobian matrix
       jcb=np.zeros((3,3),dtype=np.float64)
       for k in range(0,m):
           jcb[0,0] += dNdr[k]*x[icon[k,iel]]
           jcb[0,1] += dNdr[k]*y[icon[k,iel]]
           jcb[0,2] += dNdr[k]*z[icon[k,iel]]
           jcb[1,0] += dNds[k]*x[icon[k,iel]]
           jcb[1,1] += dNds[k]*y[icon[k,iel]]
           jcb[1,2] += dNds[k]*z[icon[k,iel]]
           jcb[2,0] += dNdt[k]*x[icon[k,iel]]
           jcb[2,1] += dNdt[k]*y[icon[k,iel]]
           jcb[2,2] += dNdt[k]*z[icon[k,iel]]
       jcbi = np.linalg.inv(jcb)

       a=hx/2/hz*
       e=
       g=

       u_corr=(1.-rm**2)
       v_corr=(1.-sm**2)
       w_corr=(1.-tm**2)
    else:
       u_corr=0.
       v_corr=0.
       w_corr=0.
    return u_corr,v_corr,w_corr

#------------------------------------------------------------------------------

print("------------------------------")
print("----------FIELDSTONE----------")
print("------------------------------")


Lx=2.  # x- extent of the domain 
Ly=2.  # y- extent of the domain 
Lz=2.  # z- extent of the domain 

if int(len(sys.argv) == 5):
   nelx           =int(sys.argv[1])
   nely           =int(sys.argv[2])
   nelz           = int(sys.argv[3])
   visu           =int(sys.argv[3])
   nmarker_per_dim=int(sys.argv[4])
   random_markers =int(sys.argv[5])
   CFL_nb         =float(sys.argv[6])
   RKorder        =int(sys.argv[7])
   use_cvi        =int(sys.argv[8])
   Q              =int(sys.argv[9])
else:
   nelx = 10
   nely = 10
   nelz = 10
   visu = 1
   nmarker_per_dim=4
   random_markers=1
   CFL_nb=0.5  
   RKorder=2
   use_cvi=1
   Q=1

if Q==1:
   nnx=nelx+1  # number of elements, x direction
   nny=nely+1  # number of elements, y direction
   nnz=nelz+1  # number of elements, z direction
   m=8    # number of velocity nodes making up an element
if Q==2:
   nnx=2*nelx+1  # number of elements, x direction
   nny=2*nely+1  # number of elements, y direction
   nnz=2*nelz+1  # number of elements, z direction
   m=27    # number of velocity nodes making up an element

nnp=nnx*nny*nnz  # number of nodes
nel=nelx*nely*nelz  # number of elements, total

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

nstep=250
every=1      # vtu output frequency

tijd=0.
#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("nnp=",nnp)
print("Q=",Q)
print("------------------------------")

countfile=open("markercount_stats_nelx"+str(nelx)+\
                                  '_nm'+str(nmarker_per_dim)+\
                                "_rand"+str(random_markers)+\
                                "_CFL_"+str(CFL_nb)+\
                                  "_rk"+str(RKorder)+\
                                 "_cvi"+str(use_cvi)+\
                                   "_Q"+str(Q)+".ascii","w")

######################################################################
# grid point setup
######################################################################
start = time.time()

x = np.empty(nnp,dtype=np.float64)  # x coordinates
y = np.empty(nnp,dtype=np.float64)  # y coordinates
z = np.empty(nnp,dtype=np.float64)  # z coordinates

if Q==1:
   counter=0
   for i in range(0,nnx):
       for j in range(0,nny):
           for k in range(0,nnz):
               x[counter]=i*hx-Lx/2
               y[counter]=j*hy-Ly/2
               z[counter]=k*hz-Lz/2
               counter += 1

if Q==2:
   counter=0
   for i in range(0,nnx):
       for j in range(0,nny):
           for k in range(0,nnz):
               x[counter]=i*hx/2-Lx/2
               y[counter]=j*hy/2-Ly/2
               z[counter]=k*hz/2-Lz/2
               counter += 1

np.savetxt('grid.ascii',np.array([x,y,z]).T,header='# x,y,z')

print("grid points setup: %.3f s" % (time.time() - start))

######################################################################
# connectivity
######################################################################
start = time.time()

icon=np.zeros((m,nel),dtype=np.int32)

if Q==1:
   counter = 0
   for i in range(0,nelx):
       for j in range(0,nely):
           for k in range(0,nelz):
               icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
               icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
               icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
               icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
               icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
               icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
               icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
               icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
               counter += 1

if Q==2:
   counter = 0
   for i in range(0,nelx):
       for j in range(0,nely):
           for k in range(0,nelz):
               icon[ 0,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
               icon[ 1,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
               icon[ 2,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
               icon[ 3,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
               icon[ 4,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
               icon[ 5,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
               icon[ 6,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
               icon[ 7,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
               icon[ 8,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
               icon[ 9,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
               icon[10,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
               icon[11,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
               icon[12,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
               icon[13,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
               icon[14,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
               icon[15,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
               icon[16,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
               icon[17,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
               icon[18,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
               icon[19,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
               icon[20,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
               icon[21,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
               icon[22,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
               icon[23,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
               icon[24,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
               icon[25,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
               icon[26,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
               counter += 1
   print(icon)

print("build connectivity: %.3f s" % (time.time() - start))

#################################################################
# assign nodal field values 
#################################################################
u = np.empty(nnp,dtype=np.float64)
v = np.empty(nnp,dtype=np.float64)
w = np.empty(nnp,dtype=np.float64)

for i in range(0,nnp):
    u[i]=u_th(x[i],y[i],z[i]) 
    v[i]=v_th(x[i],y[i],z[i]) 
    w[i]=w_th(x[i],y[i],z[i]) 

#################################################################

dt=CFL_nb*min(hx,hy,hz)/np.max(np.sqrt(u**2+v**2+w**2))
    
print('     -> dt= %.3e ' % dt)

#################################################################
# marker setup
#################################################################
start = time.time()

nmarker_per_element=nmarker_per_dim**3
nmarker=nel*nmarker_per_element

swarm_x=np.empty(nmarker,dtype=np.float64)  
swarm_y=np.empty(nmarker,dtype=np.float64)  
swarm_z=np.empty(nmarker,dtype=np.float64)  
swarm_u=np.zeros(nmarker,dtype=np.float64)  
swarm_v=np.zeros(nmarker,dtype=np.float64)  
swarm_w=np.zeros(nmarker,dtype=np.float64)  
swarm_u_corr=np.zeros(nmarker,dtype=np.float64)  
swarm_v_corr=np.zeros(nmarker,dtype=np.float64)  
swarm_w_corr=np.zeros(nmarker,dtype=np.float64)  
N=np.zeros(m,dtype=np.float64)
xx=np.zeros(m,dtype=np.float64)
yy=np.zeros(m,dtype=np.float64)
zz=np.zeros(m,dtype=np.float64)

if random_markers==1:
   counter=0
   for iel in range(0,nel):
       xx[0:m]=x[icon[0:m,iel]]
       yy[0:m]=y[icon[0:m,iel]]
       zz[0:m]=z[icon[0:m,iel]]
       for im in range(0,nmarker_per_element):
           # generate random numbers r,s between 0 and 1
           r=random.uniform(-1.,+1)
           s=random.uniform(-1.,+1)
           t=random.uniform(-1.,+1)
           if Q==1:
              N[0:m]=NQ1(r,s,t)
           if Q==2:
              N[0:m]=NQ2(r,s,t)
           swarm_x[counter]=sum(N[0:m]*xx[0:m])
           swarm_y[counter]=sum(N[0:m]*yy[0:m])
           swarm_z[counter]=sum(N[0:m]*zz[0:m])
           counter+=1

else:
   counter=0
   for iel in range(0,nel):
       xx[0:m]=x[icon[0:m,iel]]
       yy[0:m]=y[icon[0:m,iel]]
       zz[0:m]=z[icon[0:m,iel]]
       for k in range(0,nmarker_per_dim):
           for j in range(0,nmarker_per_dim):
               for i in range(0,nmarker_per_dim):
                   r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
                   s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
                   t=-1.+k*2./nmarker_per_dim + 1./nmarker_per_dim
                   if Q==1:
                      N[0:m]=NQ1(r,s,t)
                   if Q==2:
                      N[0:m]=NQ2(r,s,t)
                   swarm_x[counter]=sum(N[0:m]*xx[0:m])
                   swarm_y[counter]=sum(N[0:m]*yy[0:m])
                   swarm_z[counter]=sum(N[0:m]*zz[0:m])
                   counter+=1

np.savetxt('markers.ascii',np.array([swarm_x,swarm_y,swarm_z]).T,header='# x,y,z')

print("     -> swarm_x (m,M) %.4e %.4e " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4e %.4e " %(np.min(swarm_y),np.max(swarm_y)))
print("     -> swarm_z (m,M) %.4e %.4e " %(np.min(swarm_z),np.max(swarm_z)))

print("marker setup: %.3f s" % (time.time() - start))

#################################################################
# compute population stats
#################################################################

count=np.zeros(nel,dtype=np.int16)
for im in range (0,nmarker):
    ielx=int((swarm_x[im]+Lx/2)/Lx*nelx)
    iely=int((swarm_y[im]+Ly/2)/Ly*nely)
    ielz=int((swarm_z[im]+Lz/2)/Lz*nelz)
    iel=nely*nelz*ielx+nelz*iely+ielz
    count[iel]+=1

print("     -> count (m,M) %.5d %.5d " %(np.min(count),np.max(count)))
    
countfile.write(" %e %d %d %e %e\n" % (tijd, np.min(count),np.max(count),\
                                             np.min(count)/nmarker_per_dim**3,\
                                             np.max(count)/nmarker_per_dim**3 ))

#################################################################
# marker paint
#################################################################
swarm_mat=np.zeros(nmarker,dtype=np.int16)  

for i in [0,2,4,6,8,10,12,14]:
    dx=Lx/16
    for im in range (0,nmarker):
        if swarm_x[im]>-Lx/2++i*dx and swarm_x[im]<-Lx/2+(i+1)*dx:
           swarm_mat[im]+=1

for i in [0,2,4,6,8,10,12,14]:
    dy=Ly/16
    for im in range (0,nmarker):
        if swarm_y[im]>-Ly/2+i*dy and swarm_y[im]<-Ly/2+(i+1)*dy:
           swarm_mat[im]+=1

for i in [0,2,4,6,8,10,12,14]:
    dz=Lz/16
    for im in range (0,nmarker):
        if swarm_z[im]>-Lz/2+i*dz and swarm_z[im]<-Lz/2+(i+1)*dz:
           swarm_mat[im]+=1

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################

N = np.zeros(m,dtype=np.float64) # shape functions

for istep in range (0,nstep):

    print("----------------------------------")
    print("istep= ", istep)
    print("----------------------------------")
    start = time.time()

    if RKorder==1:

       for im in range(0,nmarker):

           swarm_u[im],swarm_v[im],swarm_w[im],rm,sm,tm,iel =\
           interpolate_vel_on_pt(swarm_x[im],swarm_y[im],swarm_z[im],\
                                 x,y,z,u,v,w,icon,Lx,Ly,Lz,nelx,nely,nelz,m,Q)

           swarm_u_corr[im],swarm_v_corr[im],swarm_w_corr[im]=\
           compute_CVI_corr(u,v,w,icon,rm,sm,tm,iel,use_cvi,Q)

           swarm_x[im]+=(swarm_u[im]+swarm_u_corr[im])*dt
           swarm_y[im]+=(swarm_v[im]+swarm_v_corr[im])*dt
           swarm_z[im]+=(swarm_w[im]+swarm_w_corr[im])*dt

       # end for im

    elif RKorder==2:

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           zA=swarm_z[im]
           uA,vA,wA,rm,sm,tm,iel=interpolate_vel_on_pt(xA,yA,zA,x,y,z,u,v,w,icon,\
                                                       Lx,Ly,Lz,nelx,nely,nelz,m,Q)
           uAcorr,vAcorr,wAcorr=compute_CVI_corr(u,v,w,icon,rm,sm,tm,iel,use_cvi,Q)
           uA+=uAcorr
           vA+=vAcorr
           wA+=wAcorr
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           zB=zA+wA*dt/2.
           uB,vB,wB,rm,sm,tm,iel=interpolate_vel_on_pt(xB,yB,zB,x,y,z,u,v,w,icon,\
                                                       Lx,Ly,Lz,nelx,nely,nelz,m,Q)
           uBcorr,vBcorr,wBcorr=compute_CVI_corr(u,v,w,icon,rm,sm,tm,iel,use_cvi,Q)
           uB+=uBcorr
           vB+=vBcorr
           wB+=wBcorr
           #--------------
           swarm_x[im]=xA+uB*dt
           swarm_y[im]=yA+vB*dt
           swarm_z[im]=zA+wB*dt
       # end for im


    tijd+=dt

    print("advection: %.3f s" % (time.time() - start))

    #############################
    # compute population stats
    #############################

    count=np.zeros(nel,dtype=np.int16)
    for im in range (0,nmarker):
        ielx=int((swarm_x[im]+Lx/2)/Lx*nelx)
        iely=int((swarm_y[im]+Ly/2)/Ly*nely)
        ielz=int((swarm_z[im]+Lz/2)/Lz*nelz)
        iel=nely*nelz*ielx+nelz*iely+ielz
        count[iel]+=1

    print("     -> count (m,M) %.5d %.5d " %(np.min(count),np.max(count)))

    countfile.write(" %e %d %d %e %e\n" % (tijd, np.min(count),np.max(count),\
                                                 np.min(count)/nmarker_per_dim**2,\
                                                 np.max(count)/nmarker_per_dim**2 ))

    countfile.flush()

    #############################
    # export markers to vtk file
    #############################


    if visu==1 and istep%every==0:

       #velfile=open("velocity.ascii","w")
       #for im in range(0,nmarker):
       #    if swarm_x[im]<0.5:
       #       ui,vi,pi=solcx.SolCxSolution(swarm_x[im],swarm_y[im]) 
       #       velfile.write("%e %e %e %e %e %e %e %e\n " % (swarm_x[im],swarm_y[im],\
       #                                                     swarm_u[im],swarm_u_corr[im],ui,\
       #                                                     swarm_v[im],swarm_v_corr[im],vi))
       #velfile.close()

       filename = 'markers_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[im],swarm_y[im],swarm_z[im]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_u[im],swarm_v[im],swarm_w[im]))
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (correction)' Format='ascii'> \n")
       #for im in range(0,nmarker):
       #    vtufile.write("%10e %10e %10e \n" %(swarm_u_corr[im],swarm_v_corr[im],swarm_w_corr[im]))
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='paint' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e \n" % swarm_mat[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d\n" % im )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % (im+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % 1)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()







if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%f\n" % visc[iel])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%f\n" % sr[iel])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%f %f %f %f %f %f\n" % (exx[iel], eyy[iel], ezz[iel], exy[iel], eyz[iel], exz[iel]))
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" % (u[i],v[i],w[i]) )
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %(exx_th(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %(eyy_th(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='ezz' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %(ezz_th(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %(exy_th(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exz' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %(exz_th(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyz' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %(eyz_th(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")




   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   if Q==1:
      for iel in range (0,nel):
          vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                                      icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))

   if Q==2:
      for iel in range (0,nel):
          vtufile.write("%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d\n" %\
                        (icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],icon[4,iel],\
                         icon[5,iel],icon[6,iel],icon[7,iel],icon[8,iel],icon[9,iel],\
                         icon[10,iel],icon[11,iel],icon[12,iel],icon[13,iel],icon[14,iel],\
                         icon[15,iel],icon[16,iel],icon[17,iel],icon[18,iel],icon[19,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   if Q==1:
      for iel in range (0,nel):
          vtufile.write("%d \n" %((iel+1)*8))
   if Q==2:
      for iel in range (0,nel):
          vtufile.write("%d \n" %((iel+1)*20))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   if Q==1:
      for iel in range (0,nel):
          vtufile.write("%d \n" %12)
   if Q==2:
      for iel in range (0,nel):
          vtufile.write("%d \n" %25)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu: %.3f s" % (time.time() - start))

countfile.close()
print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

