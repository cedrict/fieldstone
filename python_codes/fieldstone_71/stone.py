import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from shape_functions import NNV,NNP,dNNVdr,dNNVds

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# This code relies on the sph_models code written by Ross Ronan Maguire
# Code available (sph_models.py in this folder) at 
# https://github.com/romaguir/sph_models
# I have altered the file by only keeping the three functions we need
# and removed all plot-related functions (and unnecessary imports)
# I have also changed the path in read_splines to splines_dir='./data/splines'

from sph_models import read_splines,read_sph,find_spl_vals

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

print("-----------------------------")

#------------------------------------------------------------------------------
# reading data from Steinberger & Calderwood 2006
# first column is xi, second column is depth
# original file counted 2822 lines and 2 columns, with
# depths every km except for the first which went from 0 to 70
# I have modified it so that it is now every km from 0 to 2891 km included 
# it now counts 2892 lines

xi_stca06 = np.empty(2892,dtype=np.float64)
f = open('data/xi/xi_stca06.ascii','r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    xi_stca06[counter]=columns[0]
    counter+=1

print('     -> read stca06 ok') 

#------------------------------------------------------------------------------
# reading data from Steinberger & Calderwood 2006
# original files counts 22 lines but starts at 66km depth
# and ends at 2800km depth.
# I have then added a 0 depth line and a 2891km depth line
# by copyning the nearest line
# the file counts then 24 lines

xi_moek16 = np.empty(24,dtype=np.float64)
f = open('data/xi/xi_moek16.ascii','r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    xi_moek16[counter]=columns[4]
    counter+=1

print('     -> read moek16 ok') 

#------------------------------------------------------------------------------
# reading data from Civs12
# file is 153 lines long 
# first 51 lines are viscA, then 51 lines are viscB 
# and last 51 lines are depths, from 0 to 2900km 
# I have removed all ",&"

viscA_civs12 = np.empty(51,dtype=np.float64)
viscB_civs12 = np.empty(51,dtype=np.float64)
depths_civs12 = np.empty(51,dtype=np.float64)

f = open('data/visc/visc_civs12.ascii','r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter<51:
       viscA_civs12[counter]=columns[0]
    elif counter<102:
       viscB_civs12[counter-51]=columns[0]
    else:
       depths_civs12[counter-102]=columns[0]
    counter+=1

print('     -> read civs12 ok') 

#------------------------------------------------------------------------------
# reading data from  Steinberger & Holmes 2008
# file counts 22 lines
# first column is number between 0 and 1 (normalised radii)
# second column is viscosity
# I have added a last line R=1, eta=1e24

depths_stho08 = np.empty(23,dtype=np.float64)
visc_stho08 = np.empty(23,dtype=np.float64)
f = open('data/visc/visc_stho08.ascii','r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    depths_stho08[counter]=columns[0]
    visc_stho08[counter]=columns[1]
    counter+=1

depths_stho08[:]=6371e3*(1-depths_stho08[:])

print('     -> read stho08 ok') 

#------------------------------------------------------------------------------

def xi_coeff(depth,xi_case):
    #------------------
    # case 0: constant
    #------------------
    if xi_case == 0:
        val = 0.25 
    #------------------
    # case 1: stca06
    #------------------
    elif xi_case == 1:
        val = 0 
    #------------------
    # case 2: moek16
    #------------------
    elif xi_case == 2:
        val = 0
     
    return val

def viscosity(depth,visc_case):
    # visc_case=0: constant viscosity
    if visc_case==0:
       val=1e21  
    # visc_case=1: yoshida et al 2001
    elif visc_case==1:
       val = 3.0e20
       if depth < 150e3:
          val *= 1.e3 
       elif depth > 670.e3:
          val *= 70.
    # visc_case=2: Steinberg & Holmes
    elif visc_case==2:
       val=0
    # visc_case=3: Ciskova 2012 A 
    elif visc_case==3:
       val=0
    # visc_case=4: Ciskova 2012 B
    elif visc_case==4:
       val=0
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------stone 71-------------")
print("-----------------------------")

ndim=2   # number of dimensions
mV=9     # number of nodes making up an element
mP=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

if int(len(sys.argv) == 3):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
else:
   nelr = 24
   visu = 1

R1=3480e3
R2=6371e3

dr=(R2-R1)/nelr
nelt=12*nelr 
nel=nelr*nelt  

rho0=3000.
g0=10.

eta_ref=1e21      # scaling of G blocks
L_ref=(R1+R2)/2

eps=1.e-10

sqrt3=np.sqrt(3.)

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]

sparse=True

# case 0: constant
# case 1: stca06
# case 2: moek16
    
xi_case=0

# visc_case=0: constant viscosity
# visc_case=1: yoshida et al 2001
# visc_case=2: Steinberg & Holmes
# visc_case=3: Ciskova 2012 A 
# visc_case=4: Ciskova 2012 B

visc_case=0

#################################################################
# grid point setup
#################################################################
start = timing.time()

nnr=nelr+1
nnt=nelt
nnp=nnr*nnt  # number of nodes

xV=np.empty(nnp,dtype=np.float64)  # x coordinates
yV=np.empty(nnp,dtype=np.float64)  # y coordinates
r=np.empty(nnp,dtype=np.float64)  
theta=np.empty(nnp,dtype=np.float64) 
longitude=np.zeros(nnp,dtype=np.float64) 
latitude=np.zeros(nnp,dtype=np.float64) 

Louter=2.*math.pi*R2
Lr=R2-R1
sx = Louter/float(nelt)
sz = Lr    /float(nelr)

counter=0
for j in range(0,nnr):
    for i in range(0,nelt):
        xV[counter]=i*sx
        yV[counter]=j*sz
        counter += 1

counter=0
for j in range(0,nnr):
    for i in range(0,nnt):
        xi=xV[counter]
        yi=yV[counter]
        t=xi/Louter*2.*math.pi    
        xV[counter]=math.cos(t)*(R1+yi)
        yV[counter]=math.sin(t)*(R1+yi)
        r[counter]=R1+yi
        theta[counter]=math.atan2(yV[counter],xV[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*math.pi
        longitude[counter]=theta[counter]/2./np.pi*360.
        counter+=1

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y,u,v')

print("building coordinate arrays (%.3fs)" % (timing.time() - start))

#################################################################
# build iconQ1 array needed for vtu file
#################################################################

iconQ1 =np.zeros((4,nel),dtype=np.int32)
counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        iconQ1[0,counter] = icon2 
        iconQ1[1,counter] = icon1
        iconQ1[2,counter] = icon4
        iconQ1[3,counter] = icon3
        counter += 1
    #end for

#################################################################
# now that the grid has been built as if it was a Q1 grid, 
# we can simply use these same points to arrive at a Q2 
# connectivity array with 4 times less elements.
#################################################################

nelr=nelr//2
nelt=nelt//2
nel=nel//4

NfemV=nnp*ndofV           # Total number of degrees of V freedom 
NfemP=nelt*(nelr+1)*ndofP # Total number of degrees of P freedom
Nfem=NfemV+NfemP          # total number of dofs

print('nelr=',nelr)
print('nelr=',nelt)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV =np.zeros((mV,nel),dtype=np.int32)
iconP =np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        iconV[0,counter]=2*counter+2 +2*j*nelt
        iconV[1,counter]=2*counter   +2*j*nelt
        iconV[2,counter]=iconV[1,counter]+4*nelt
        iconV[3,counter]=iconV[1,counter]+4*nelt+2
        iconV[4,counter]=iconV[0,counter]-1
        iconV[5,counter]=iconV[1,counter]+2*nelt
        iconV[6,counter]=iconV[2,counter]+1
        iconV[7,counter]=iconV[5,counter]+2
        iconV[8,counter]=iconV[5,counter]+1
        if i==nelt-1:
           iconV[0,counter]-=2*nelt
           iconV[7,counter]-=2*nelt
           iconV[3,counter]-=2*nelt
        #print(j,i,counter,'|',iconV[0:mV,counter])
        counter += 1


iconP =np.zeros((mP,nel),dtype=np.int32)
counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        iconP[0,counter] = icon2 
        iconP[1,counter] = icon1
        iconP[2,counter] = icon4
        iconP[3,counter] = icon3
        counter += 1
    #end for


#for iel in range(0,nel):
#    print(iel,'|',iconP[:,iel])

#now that I have both connectivity arrays I can 
# easily build xP,yP

NP=NfemP
xP=np.empty(NP,dtype=np.float64)  # x coordinates
yP=np.empty(NP,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xP[iconP[0,iel]]=xV[iconV[0,iel]]
    xP[iconP[1,iel]]=xV[iconV[1,iel]]
    xP[iconP[2,iel]]=xV[iconV[2,iel]]
    xP[iconP[3,iel]]=xV[iconV[3,iel]]
    yP[iconP[0,iel]]=yV[iconV[0,iel]]
    yP[iconP[1,iel]]=yV[iconV[1,iel]]
    yP[iconP[2,iel]]=yV[iconV[2,iel]]
    yP[iconP[3,iel]]=yV[iconV[3,iel]]

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y,u,v')

print("building connectivity array (%.3fs)" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  
bc_val = np.zeros(Nfem, dtype=np.float64) 

for i in range(0,nnp):
    if r[i]/R1<1.+eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    if r[i]/R2>1.-eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

print("defining boundary conditions (%.3fs)" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

for iel in range(0,nel):
    for iq in [0,1,2]:
        for jq in [0,1,2]:
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob = np.linalg.det(jcb)
            area[iel]+=jcob*weightq
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))
print("     -> total area (anal) %.6f " %(np.pi*(R2**2-R1**2)))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# use spherical harmonics tools to assign density to nodes
#################################################################
# note that find_spl_vals expects values in kilometers
# longitudes between 0 and 360
# latitudes between -90 and 90

lmin=0
lmax=40
datafilename='S40RTS'

sph_splines = read_sph('data/models/'+datafilename+'.sph',lmin,lmax)

d_ln_vs=np.zeros(nnp,dtype=np.float64)

counter=0
for j in range(0,nnr):
    print('layer=',j)
    for i in range(0,nnt):
        #xi=xV[counter]
        #yi=yV[counter]
        #t=xi/Louter*2.*math.pi    
        #theta[counter]=math.atan2(yV[counter],xV[counter])
        if i==0:
           # find_spl_vals takes a long time and we only need it per layer/depth
           ri=r[counter]
           ri=min(ri,R2-1.)
           ri=max(ri,R1+1.)
           depth=(R2-ri)/1000.
           #print(depth,r[counter])
           spl_vals = find_spl_vals(depth)

        mylat=latitude[counter]
        mylon=longitude[counter]
        #print(mylat,mylon)

        # values returned by sph_spline are divided by sqrt(2)- WIP!!
        for i,sph_spline in enumerate(sph_splines):
            vals = sph_spline.expand(lat=mylat,lon=mylon)
            d_ln_vs[counter] += spl_vals[i] * vals / np.sqrt(2.) 

        counter+=1

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y,u,v')




#################################################################
# build FE matrix
#################################################################
start = timing.time()

if sparse:
   A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
   G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdy  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNNP= np.zeros(mP*ndofP,dtype=np.float64)   

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #end for 
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for 
            depthq=R2-np.sqrt(xq*xq+yq*yq)

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]
            #end for 

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(depthq,visc_case)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx(xq,yq,g0)*rho0
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq,yq,g0)*rho0
            #end for 

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.
            #end for 

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            NNNNP[:]+=NNNP[:]*jcob*weightq

        #end for jq
    #end for iq

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
            #end if 
        #end for 
    #end for 

    G_el*=eta_ref/L_ref
    h_el*=eta_ref/L_ref

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    if sparse:
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                if sparse:
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]
    #end for 

#end for iel

if not sparse:
   print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))

#exit()

print("build FE matrixs & rhs (%.3fs)" % (timing.time() - start))

#################################################################
# solve system
#################################################################
start = timing.time()

if not sparse:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

rhs=np.zeros(Nfem,dtype=np.float64)
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs
    
if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solving system (%.3fs)" % (timing.time() - start))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]*(eta_ref/L_ref)

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
np.savetxt('p.ascii',np.array([xP,yP,p]).T,header='# x,y,u,v')

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %.4e %.4e " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4e %.4e " %(np.min(vt),np.max(vt)))

print("reshape solution (%.3fs)" % (timing.time() - start))

#####################################################################
# compute strain rate - corners to nodes - method 2
#####################################################################
start = timing.time()

count = np.zeros(nnp,dtype=np.int16)  
q=np.zeros(nnp,dtype=np.float64)
Lxx2 = np.zeros(nnp,dtype=np.float64)  
Lxy2 = np.zeros(nnp,dtype=np.float64)  
Lyx2 = np.zeros(nnp,dtype=np.float64)  
Lyy2 = np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):
    for i in range(0,mV):
        inode=iconV[i,iel]
        rq=rVnodes[i]
        sq=sVnodes[i]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        NNNP[0:mP]=NNP(rq,sq)
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        #end for
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for
        L_xx=0.
        L_xy=0.
        L_yx=0.
        L_yy=0.
        for k in range(0,mV):
            L_xx+=dNNNVdx[k]*u[iconV[k,iel]]
            L_xy+=dNNNVdx[k]*v[iconV[k,iel]]
            L_yx+=dNNNVdy[k]*u[iconV[k,iel]]
            L_yy+=dNNNVdy[k]*v[iconV[k,iel]]
        #end for
        Lxx2[inode]+=L_xx
        Lxy2[inode]+=L_xy
        Lyx2[inode]+=L_yx
        Lyy2[inode]+=L_yy
        q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        count[inode]+=1
    #end for
#end for
Lxx2/=count
Lxy2/=count
Lyx2/=count
Lyy2/=count
q/=count

print("     -> Lxx2 (m,M) %.4e %.4e " %(np.min(Lxx2),np.max(Lxx2)))
print("     -> Lyy2 (m,M) %.4e %.4e " %(np.min(Lyy2),np.max(Lyy2)))
print("     -> Lxy2 (m,M) %.4e %.4e " %(np.min(Lxy2),np.max(Lxy2)))
print("     -> Lxy2 (m,M) %.4e %.4e " %(np.min(Lyx2),np.max(Lyx2)))

np.savetxt('q.ascii',np.array([xV,yV,q,r,theta]).T)
#np.savetxt('strainrate.ascii',np.array([xV,yV,Lxx,Lyy,Lxy,Lyx]).T)

print("compute vel gradient meth-2 (%.3fs)" % (timing.time() - start))

#################################################################
#################################################################

exx2 = np.zeros(nnp,dtype=np.float64)  
eyy2 = np.zeros(nnp,dtype=np.float64)  
exy2 = np.zeros(nnp,dtype=np.float64)  

exx2[:]=Lxx2[:]
eyy2[:]=Lyy2[:]
exy2[:]=0.5*(Lxy2[:]+Lyx2[:])

#################################################################
#################################################################
start = timing.time()

M_mat= np.zeros((nnp,nnp),dtype=np.float64)
rhsLxx=np.zeros(nnp,dtype=np.float64)
rhsLyy=np.zeros(nnp,dtype=np.float64)
rhsLxy=np.zeros(nnp,dtype=np.float64)
rhsLyx=np.zeros(nnp,dtype=np.float64)

for iel in range(0,nel):

    M_el =np.zeros((mV,mV),dtype=np.float64)
    fLxx_el=np.zeros(mV,dtype=np.float64)
    fLyy_el=np.zeros(mV,dtype=np.float64)
    fLxy_el=np.zeros(mV,dtype=np.float64)
    fLyx_el=np.zeros(mV,dtype=np.float64)
    NNNV =np.zeros((mV,1),dtype=np.float64) 

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV,0]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #end for 
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            Lxxq=0.
            Lyyq=0.
            Lxyq=0.
            Lyxq=0.
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                Lxxq+=dNNNVdx[k]*u[iconV[k,iel]]
                Lyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                Lxyq+=dNNNVdx[k]*v[iconV[k,iel]]
                Lyxq+=dNNNVdy[k]*u[iconV[k,iel]]
            #end for 

            M_el +=NNNV.dot(NNNV.T)*weightq*jcob

            fLxx_el[:]+=NNNV[:,0]*Lxxq*jcob*weightq
            fLyy_el[:]+=NNNV[:,0]*Lyyq*jcob*weightq
            fLxy_el[:]+=NNNV[:,0]*Lxyq*jcob*weightq
            fLyx_el[:]+=NNNV[:,0]*Lyxq*jcob*weightq

        #end for
    #end for

    for k1 in range(0,mV):
        m1=iconV[k1,iel]
        for k2 in range(0,mV):
            m2=iconV[k2,iel]
            M_mat[m1,m2]+=M_el[k1,k2]
        #end for
        rhsLxx[m1]+=fLxx_el[k1]
        rhsLyy[m1]+=fLyy_el[k1]
        rhsLxy[m1]+=fLxy_el[k1]
        rhsLyx[m1]+=fLyx_el[k1]
    #end for

#end for

Lxx3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxx)
Lyy3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyy)
Lxy3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxy)
Lyx3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyx)

print("     -> Lxx3 (m,M) %.4e %.4e " %(np.min(Lxx3),np.max(Lxx3)))
print("     -> Lyy3 (m,M) %.4e %.4e " %(np.min(Lyy3),np.max(Lyy3)))
print("     -> Lxy3 (m,M) %.4e %.4e " %(np.min(Lxy3),np.max(Lxy3)))
print("     -> Lxy3 (m,M) %.4e %.4e " %(np.min(Lyx3),np.max(Lyx3)))

print("compute vel gradient meth-3 (%.3fs)" % (timing.time() - start))

#################################################################
#################################################################

exx3 = np.zeros(nnp,dtype=np.float64)  
eyy3 = np.zeros(nnp,dtype=np.float64)  
exy3 = np.zeros(nnp,dtype=np.float64)  

exx3[:]=Lxx3[:]
eyy3[:]=Lyy3[:]
exy3[:]=0.5*(Lxy3[:]+Lyx3[:])

#################################################################
# normalise pressure
#################################################################
start = timing.time()

#print(np.sum(q[0:2*nelt])/(2*nelt))
#print(np.sum(q[nnp-2*nelt:nnp])/(2*nelt))
#print(np.sum(p[0:nelt])/(nelt))

poffset=np.sum(q[0:2*nelt])/(2*nelt)

q-=poffset
p-=poffset

print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))
print("     -> q (m,M) %.4e %.4e " %(np.min(q),np.max(q)))

print("normalise pressure (%.3fs)" % (timing.time() - start))

#################################################################
# export pressure at both surfaces
#################################################################
start = timing.time()



#np.savetxt('q_R1.ascii',np.array([xV[0:2*nelt],yV[0:2*nelt],q[0:2*nelt],theta[0:2*nelt]]).T)
#np.savetxt('q_R2.ascii',np.array([xV[nnp-2*nelt:nnp],\
#                                  yV[nnp-2*nelt:nnp],\
#                                   q[nnp-2*nelt:nnp],\
#                               theta[nnp-2*nelt:nnp]]).T)
#
#np.savetxt('p_R1.ascii',np.array([xP[0:nelt],yP[0:nelt],p[0:nelt]]).T)
#np.savetxt('p_R2.ascii',np.array([xP[NP-nelt:NP],yP[NP-nelt:NP],p[NP-nelt:NP]]).T)

print("export p&q on R1,R2 (%.3fs)" % (timing.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = timing.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,4*nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(gx(xV[i],yV[i],g0),gy(xV[i],yV[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(x,y)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %r[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %theta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='longitude' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %longitude[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='d_ln_vs' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %d_ln_vs[i])
   vtufile.write("</DataArray>\n")


   #--
   vtufile.write("<DataArray type='Float32' Name='exx2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %exx2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %eyy2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %exy2[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %exx3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %eyy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %exy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d %d %d %d\n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,4*nel):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu file (%.3fs)" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
