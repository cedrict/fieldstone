import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as time
from scipy.optimize import nnls
from scipy.sparse import lil_matrix
import numpy as np

#------------------------------------------------------------------------------
# These two functions were written by G. Mack

def sign(n):
    if n < 0:
        return -1.0
    return 1.0

def limiter(c_0, c_1, c_2, S = 0, L=1, h=1):
    """
    Limits a the slopes of a plane in 2D
    See latex for a complete explaination
    float- c_0, c_1, c_2 should be given by a least squares approximation.
    float - S is the lowest allowed property value
    float - L is the highest allowed property value
    float - h is the width of the cell
    """
    k_u = min(c_0 - S, L-c_0)
    k_l = (abs(c_1) + abs(c_2)) * h/2
    if k_l > k_u:
        c_max = k_u *2/h
        c_1 = sign(c_1) * min(abs(c_1), c_max)
        c_2 = sign(c_2) * min(abs(c_2), c_max)
    k_l = (abs(c_1) + abs(c_2)) * h/2
    if k_l > k_u:
        c_change = (k_l - k_u)/h # denominator is written up as 2*h/2
        c_1 = sign(c_1) * (abs(c_1) - c_change)
        c_2 = sign(c_2) * (abs(c_2) - c_change)
    return c_1, c_2

#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8

def dNNVdr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8

def dNNVds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return NP_0,NP_1,NP_2,NP_3

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#model=1: van Keken et al 1997
#model=2: layers
#model=3: stokes sphere with free surf

model=3

#avrg=1: elemental averaging - arithmetic
#avrg=2: elemental averaging - geometric
#avrg=3: elemental averaging - harmonic
#avrg=4: least square 

#avrg=-1: nodal averaging - arithmetic
#avrg=-2: nodal averaging - geometric
#avrg=-3: nddal averaging - harmonic

#------------------------------------------------------------------------------

eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

if model==1:
   Lx=9.142e3  # horizontal extent of the domain 
   Ly=1e4      # vertical extent of the domain 
   nelx=48
   nely=48
   gx=0
   gy=-10
   #material 1: eta_salt=1e19, rho_salt=2150
   #material 2: eta_sed=1e19, rho_sed=2600
   rho_mat = np.array([2150,2600],dtype=np.float64) 
   eta_mat = np.array([1e19,1.e19],dtype=np.float64) 
   salt_thickness=2000
   amplitude=200
   mass0=Lx*salt_thickness*rho_mat[0]+Lx*(Ly-salt_thickness)*rho_mat[1]
   nmarker_per_dim=10
   eta_ref=1e21      # scaling of G blocks
   year=3600.*24.*365.25
   avrg=3

if model==2:
   Lx=50e3  # horizontal extent of the domain 
   Ly=10e3  # vertical extent of the domain 
   nelx = 125
   nely = 25
   gx=0
   gy=-9.81
   #material 1: eta_salt=1e17, rho_salt=2200
   #material 2: eta_sed=1e21, #rho_sed=2600
   rho_mat = np.array([2200,2600],dtype=np.float64) 
   eta_mat = np.array([1e17,1.e20],dtype=np.float64) 
   salt_thickness=2e3
   amplitude=500
   nmarker_per_dim=4
   eta_ref=1e21      # scaling of G blocks
   year=3600.*24.*365.25
   avrg=3

if model==3:
   Lx=1
   Ly=1
   nelx=129
   nely=nelx
   gx=0
   gy=-1
   #material 1: eta_salt=1, rho_salt=1
   #material 2: eta_sph=1e3, rho_sph=2
   #material 3: eta_air=1e-3, rho_air=0
   rho_mat = np.array([1,2,0],dtype=np.float64) 
   eta_mat = np.array([1,1e3,1e-3],dtype=np.float64) 
   R_sphere=0.123456789
   x_sphere=0.5
   y_sphere=0.6
   mass0=0.79788283183
   nmarker_per_dim=5
   eta_ref=1
   year=1
   avrg=4

if int(len(sys.argv) == 2):
   avrg=int(sys.argv[1])

nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction
NV=nnx*nny  # number of nodes
NP=(nelx+1)*(nely+1)
nel=nelx*nely  # number of elements, total
nq=9*nel
NfemV=NV*ndofV               # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]
hx=Lx/nelx
hy=Ly/nely
nmarker_per_element=nmarker_per_dim**2
nmarker=nmarker_per_element*nel
pnormalise=True


nstep_change=2500000
nstep=1
CFL_nb=0.0025
rk=2

every=1

debug=False

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("avrg=",avrg)
print("------------------------------")

vrms_file=open('vrms.ascii',"w")
mass_file=open('mass.ascii',"w")
nmarker_file=open('nmarker_per_element.ascii',"w")
dt_file=open('dt.ascii',"w")
profileq_file=open('profileq.ascii',"w")
profilec_file=open('profilec.ascii',"w")

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx/2.
        y[counter]=j*hy/2.
        counter += 1
    #end for
#end for

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("grid setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
#################################################################
start = time.time()

iconV=np.zeros((mV,nel),dtype=np.int32)
iconP=np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

print("connectivity: %.3f s" % (time.time() - start))

#################################################################
# marker setup
#################################################################
start = time.time()

swarm_x=np.empty(nmarker,dtype=np.float64) 
swarm_y=np.empty(nmarker,dtype=np.float64) 
swarm_mat=np.empty(nmarker,dtype=np.int8)  
swarm_paint=np.empty(nmarker,dtype=np.float64) 
swarm_x0=np.empty(nmarker,dtype=np.float64) 
swarm_y0=np.empty(nmarker,dtype=np.float64) 
swarm_r=np.empty(nmarker,dtype=np.float64) 
swarm_s=np.empty(nmarker,dtype=np.float64) 

counter=0
for iel in range(0,nel):
    x1=x[iconV[0,iel]] ; y1=y[iconV[0,iel]]
    x2=x[iconV[1,iel]] ; y2=y[iconV[1,iel]]
    x3=x[iconV[2,iel]] ; y3=y[iconV[2,iel]]
    x4=x[iconV[3,iel]] ; y4=y[iconV[3,iel]]
    for j in range(0,nmarker_per_dim):
        for i in range(0,nmarker_per_dim):
            r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
            s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
            swarm_r[counter]=r
            swarm_s[counter]=s
            N1=0.25*(1-r)*(1-s)
            N2=0.25*(1+r)*(1-s)
            N3=0.25*(1+r)*(1+s)
            N4=0.25*(1-r)*(1+s)
            swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
            swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
            counter+=1
        #end for 
    #end for 
#end for 

swarm_x0[0:nmarker]=swarm_x[0:nmarker]
swarm_y0[0:nmarker]=swarm_y[0:nmarker]


print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("marker setup: %.3f s" % (time.time() - start))

#################################################################
# assign material id to markers 
#################################################################
start = time.time()

if model==1:
   for im in range (0,nmarker):
       if swarm_y[im]>salt_thickness+amplitude*np.cos(np.pi*swarm_x[im]/Lx):
          swarm_mat[im]=2
       else:
          swarm_mat[im]=1
       #end if
   #end for 

if model==3:
   for im in range (0,nmarker):
       swarm_mat[im]=1
       if swarm_y[im]>0.75:
          swarm_mat[im]=3
       if (swarm_x[im]-x_sphere)**2+(swarm_y[im]-y_sphere)**2<R_sphere**2:
          swarm_mat[im]=2

print("marker layout: %.3f s" % (time.time() - start))

#################################################################
# paint markers 
#################################################################
start = time.time()

for im in range (0,nmarker):
    swarm_paint[im]=(np.sin(2*np.pi*swarm_x[im]/Lx*4)*\
                     np.sin(2*np.pi*swarm_y[im]/Ly*4))
#end for 

#np.savetxt('markers.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y,mat')

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if model==1:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       if x[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       if y[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if y[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. 

if model==3:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if x[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if y[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if y[i]/Ly>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. 


print("setup: boundary conditions: %.3f s" % (time.time() - start))


#################################################################
# compute element center coordinates 
#################################################################

xc = np.zeros(nel,dtype=np.float64)
yc = np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    xc[iel]=(x[iconV[0,iel]]+x[iconV[2,iel]])/2
    yc[iel]=(y[iconV[0,iel]]+y[iconV[2,iel]])/2


#--------------------------------------------------------------------------------------------------
# time stepping loop
#--------------------------------------------------------------------------------------------------

Time=0.

for istep in range(0,nstep):

    print ('---------------------------------------')
    print ('-----------------istep= %i -------------' %istep)
    print ('---------------------------------------')

    #################################################################
    # compute elemental averagings 
    #################################################################
    start = time.time()

    eta_elemental=np.zeros(nel,dtype=np.float64) 
    rho_nodal=np.zeros(NP,dtype=np.float64) 
    eta_nodal=np.zeros(NP,dtype=np.float64) 
    nmarker_in_element=np.zeros(nel,dtype=np.int16) 
    nodal_counter=np.zeros(NP,dtype=np.float64) 
    list_of_markers_in_element=np.zeros((2*nmarker_per_element,nel),dtype=np.int32)
    eta_min=np.zeros(nel,dtype=np.float64) ; eta_min[:]=1e36
    eta_max=np.zeros(nel,dtype=np.float64) 
    rho_min=np.zeros(nel,dtype=np.float64) ; rho_min[:]=1e6
    rho_max=np.zeros(nel,dtype=np.float64) 
    

    for im in range(0,nmarker):
        #localise marker
        ielx=int(swarm_x[im]/Lx*nelx)
        iely=int(swarm_y[im]/Ly*nely)
        iel=nelx*(iely)+ielx
        if debug:
           if ielx<0:
              print ('ielx<0',ielx)
           if ielx>nelx-1:
              print ('ielx>nelx-1')
           if iely<0:
              print ('iely<0')
           if iely>nely-1:
              print ('iely>nely-1')
           if iel<0:
              print ('iel<0')
           if iel>nel-1:
              print ('iel>nel-1')
        #weighed averaging
        N1=0.25*(1-swarm_r[im])*(1-swarm_s[im])
        N2=0.25*(1+swarm_r[im])*(1-swarm_s[im])
        N3=0.25*(1+swarm_r[im])*(1+swarm_s[im])
        N4=0.25*(1-swarm_r[im])*(1+swarm_s[im])
        rho_nodal[iconP[0,iel]]+=rho_mat[swarm_mat[im]-1]*N1
        rho_nodal[iconP[1,iel]]+=rho_mat[swarm_mat[im]-1]*N2
        rho_nodal[iconP[2,iel]]+=rho_mat[swarm_mat[im]-1]*N3
        rho_nodal[iconP[3,iel]]+=rho_mat[swarm_mat[im]-1]*N4
        nodal_counter[iconP[0,iel]]+=N1
        nodal_counter[iconP[1,iel]]+=N2
        nodal_counter[iconP[2,iel]]+=N3
        nodal_counter[iconP[3,iel]]+=N4

        rho_min[iel]=min(rho_min[iel],rho_mat[swarm_mat[im]-1])
        rho_max[iel]=max(rho_max[iel],rho_mat[swarm_mat[im]-1])

        eta_min[iel]=min(eta_min[iel],eta_mat[swarm_mat[im]-1])
        eta_max[iel]=max(eta_max[iel],eta_mat[swarm_mat[im]-1])

        list_of_markers_in_element[nmarker_in_element[iel],iel]=im
        nmarker_in_element[iel]+=1

        if abs(avrg)==1 : # arithmetic
           eta_elemental[iel]     +=eta_mat[swarm_mat[im]-1]
           eta_nodal[iconP[0,iel]]+=eta_mat[swarm_mat[im]-1]*N1
           eta_nodal[iconP[1,iel]]+=eta_mat[swarm_mat[im]-1]*N2
           eta_nodal[iconP[2,iel]]+=eta_mat[swarm_mat[im]-1]*N3
           eta_nodal[iconP[3,iel]]+=eta_mat[swarm_mat[im]-1]*N4
        if abs(avrg)==2: # geometric
           eta_elemental[iel]     +=np.log10(eta_mat[swarm_mat[im]-1])
           eta_nodal[iconP[0,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N1
           eta_nodal[iconP[1,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N2
           eta_nodal[iconP[2,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N3
           eta_nodal[iconP[3,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N4
        if abs(avrg)==3: # harmonic
           eta_elemental[iel]     +=1/eta_mat[swarm_mat[im]-1]
           eta_nodal[iconP[0,iel]]+=1/eta_mat[swarm_mat[im]-1]*N1
           eta_nodal[iconP[1,iel]]+=1/eta_mat[swarm_mat[im]-1]*N2
           eta_nodal[iconP[2,iel]]+=1/eta_mat[swarm_mat[im]-1]*N3
           eta_nodal[iconP[3,iel]]+=1/eta_mat[swarm_mat[im]-1]*N4
    #end for

    rho_nodal/=nodal_counter
    if abs(avrg)==1:
       eta_nodal/=nodal_counter
       eta_elemental[:]/=nmarker_in_element[:]
    if abs(avrg)==2:
       eta_nodal[:]=10.**(eta_nodal[:]/nodal_counter[:])
       eta_elemental[:]=10.**(eta_elemental[:]/nmarker_in_element[:])
    if abs(avrg)==3:
       eta_nodal[:]=nodal_counter[:]/eta_nodal[:]
       eta_elemental[:]=nmarker_in_element[:]/eta_elemental[:]

    nmarker_file.write("%d %e %e\n" %(istep,np.min(nmarker_in_element),np.max(nmarker_in_element))) 
    nmarker_file.flush()

    if np.min(nmarker_in_element)==0:
       exit('no marker left in an element')

    print("     -> nmarker_in_elt(m,M) %.5e %.5e " %(np.min(nmarker_in_element),np.max(nmarker_in_element)))
    print("     -> rho_nodal     (m,M) %.5e %.5e " %(np.min(rho_nodal),np.max(rho_nodal)))
    print("     -> eta_elemental (m,M) %.5e %.5e " %(np.min(eta_elemental),np.max(eta_elemental)))

    print("markers onto grid: %.3f s" % (time.time() - start))

    #print(list_of_markers_in_element[:,0])
    #print(list_of_markers_in_element[:,11])
    #print(list_of_markers_in_element[:,38])

    #################################################################
    # least square process
    # for each element I loop over the markers in it and build 
    # the corresponding A matrix and b rhs. 
    # The solution, i.e. the coefficients (a,b,c for viscosity, 
    # d,e,f for density) are stored for each element.
    #################################################################
    start = time.time()

    a_ls=np.zeros(nel,dtype=np.float64) 
    b_ls=np.zeros(nel,dtype=np.float64) 
    c_ls=np.zeros(nel,dtype=np.float64) 
    d_ls=np.zeros(nel,dtype=np.float64) 
    e_ls=np.zeros(nel,dtype=np.float64) 
    f_ls=np.zeros(nel,dtype=np.float64) 

    b_ls2=np.zeros(nel,dtype=np.float64) 
    c_ls2=np.zeros(nel,dtype=np.float64) 
    e_ls2=np.zeros(nel,dtype=np.float64) 
    f_ls2=np.zeros(nel,dtype=np.float64) 

    for iel in range(0,nel):
        A_ls = np.zeros((3,3),dtype=np.float64) 
        rhs_eta_ls = np.zeros((3),dtype=np.float64) 
        rhs_rho_ls = np.zeros((3),dtype=np.float64) 
        A_ls[0,0]=nmarker_in_element[iel]
        for i in range(0,nmarker_in_element[iel]):
            im=list_of_markers_in_element[i,iel]
            xim=swarm_x[im]-xc[iel]
            yim=swarm_y[im]-yc[iel]
            A_ls[0,1]+=xim
            A_ls[0,2]+=yim
            A_ls[1,1]+=xim**2
            A_ls[1,2]+=xim*yim
            A_ls[2,2]+=yim**2
            rhs_eta_ls[0]+=eta_mat[swarm_mat[im]-1]
            rhs_eta_ls[1]+=eta_mat[swarm_mat[im]-1]*xim
            rhs_eta_ls[2]+=eta_mat[swarm_mat[im]-1]*yim
            rhs_rho_ls[0]+=rho_mat[swarm_mat[im]-1]
            rhs_rho_ls[1]+=rho_mat[swarm_mat[im]-1]*xim
            rhs_rho_ls[2]+=rho_mat[swarm_mat[im]-1]*yim
        #end for
        A_ls[1,0]=A_ls[0,1]
        A_ls[2,0]=A_ls[0,2]
        A_ls[2,1]=A_ls[1,2]
        sol=np.linalg.solve(A_ls,rhs_eta_ls)
        #sol,res=nnls(A_ls,rhs_ls,maxiter=100)
        #print(res)
        a_ls[iel]=sol[0]
        b_ls[iel]=sol[1]
        c_ls[iel]=sol[2]
        sol=np.linalg.solve(A_ls,rhs_rho_ls)
        d_ls[iel]=sol[0]
        e_ls[iel]=sol[1]
        f_ls[iel]=sol[2]
    #end for

    #apply limiter
    for iel in range(0,nel):
        b_ls2[iel],c_ls2[iel]=limiter(a_ls[iel],b_ls[iel],c_ls[iel],eta_min[iel],eta_max[iel],hx)
        e_ls2[iel],f_ls2[iel]=limiter(d_ls[iel],e_ls[iel],f_ls[iel],rho_min[iel],rho_max[iel],hx)


    print("compute ls coefficients: %.3f s" % (time.time() - start))

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = time.time()

    if pnormalise:
       A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
       rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
    else:
       A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
       rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b

    #K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
    #G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
    f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
    constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
    b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
    N_mat = np.zeros((3,ndofP*mP),dtype=np.float64)  # matrix  
    NNNV    = np.zeros(mV,dtype=np.float64)          # shape functions V
    NNNP    = np.zeros(mP,dtype=np.float64)          # shape functions P
    dNNNVdx  = np.zeros(mV,dtype=np.float64)         # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)         # shape functions derivatives
    dNNNVdr  = np.zeros(mV,dtype=np.float64)         # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)         # shape functions derivatives
    u     = np.zeros(NV,dtype=np.float64)            # x-component velocity
    v     = np.zeros(NV,dtype=np.float64)            # y-component velocity
    xq   = np.zeros(nq,dtype=np.float64)             # 
    yq   = np.zeros(nq,dtype=np.float64)             # 
    rhoq = np.zeros(nq,dtype=np.float64)             # 
    etaq = np.zeros(nq,dtype=np.float64)             # 
    c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    counter=0
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
                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)
                NNNP[0:4]=NNP(rq,sq)
                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*x[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*y[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*x[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*y[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)
                # compute dNdx & dNdy
                for k in range(0,mV):
                    xq[counter]+=NNNV[k]*x[iconV[k,iel]]
                    yq[counter]+=NNNV[k]*y[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for

                for k in range(0,mP):
                    rhoq[counter]+=NNNP[k]*rho_nodal[iconP[k,iel]]
                    etaq[counter]+=NNNP[k]*eta_nodal[iconP[k,iel]]
                #end for

                if avrg>0:
                   etaq[counter]=eta_elemental[iel]

                if avrg==4:
                   etaq[counter]=a_ls[iel]+b_ls2[iel]*(xq[counter]-xc[iel])+c_ls2[iel]*(yq[counter]-yc[iel])
                   rhoq[counter]=d_ls[iel]+e_ls2[iel]*(xq[counter]-xc[iel])+f_ls2[iel]*(yq[counter]-yc[iel])

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.        ],
                                             [0.        ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counter]*weightq*jcob

                for i in range(0,mV):
                    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*rhoq[counter]*gx
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rhoq[counter]*gy
                #end for

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=0.
                #end for

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob
                NNNNP[:]+=NNNP[:]*jcob*weightq
                counter+=1
            #end for
        #end for

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
                #end for
            #end for
        #end for

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        #K_mat[m1,m2]+=K_el[ikk,jkk]
                        A_sparse[m1,m2] += K_el[ikk,jkk]
                    #end for
                #end for
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    #G_mat[m1,m2]+=G_el[ikk,jkk]
                    A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*eta_ref/Ly
                    A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*eta_ref/Ly
                #end for
                f_rhs[m1]+=f_el[ikk]
            #end for
        #end for

        if pnormalise:
           for k2 in range(0,mP):
               m2=iconP[k2,iel]
               h_rhs[m2]+=h_el[k2]
               constr[m2]+=NNNNP[k2]
               A_sparse[Nfem,NfemV+m2]=constr[m2]
               A_sparse[NfemV+m2,Nfem]=constr[m2]
           #end for

    #end for

    #G_mat*=eta_ref/Ly
    h_rhs*=eta_ref/Ly

    #print("     -> K (m,M) %.5e %.5e " %(np.min(K_mat),np.max(K_mat)))
    #print("     -> G (m,M) %.5e %.5e " %(np.min(G_mat),np.max(G_mat)))
    print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
    print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

    #np.savetxt('rhoq.ascii',np.array([xq,yq,rhoq]).T,header='# x,y,rho')
    #np.savetxt('etaq.ascii',np.array([xq,yq,etaq]).T,header='# x,y,eta')

    print("build FE matrix: %.3f s" % (time.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    # solve system
    ######################################################################
    start = time.time()

    #if pnormalise:
    #   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
    #   a_mat[0:NfemV,0:NfemV]=K_mat
    #   a_mat[0:NfemV,NfemV:Nfem]=G_mat
    #   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
    #   a_mat[Nfem,NfemV:Nfem]=constr
    #   a_mat[NfemV:Nfem,Nfem]=constr
    #else:
    #   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
    #   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    #   a_mat[0:NfemV,0:NfemV]=K_mat
    #   a_mat[0:NfemV,NfemV:Nfem]=G_mat
    #   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
    #sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    sparse_matrix=A_sparse.tocsr()
    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (time.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = time.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*(eta_ref/Ly)

    print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))

    #np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (time.time() - start))

    ######################################################################
    # compute timestep 
    ######################################################################

    dt=CFL_nb*min(hx,hy)/max(max(abs(u)),max(abs(v)))

    print("dt= %.3e yr" %(dt/year))

    dt_file.write("%d %e \n" %(istep,dt)) ; dt_file.flush()

    #################################################################
    # compute vrms 
    #################################################################
    start = time.time()

    vrms=0.
    mass=0.
    counter=0
    for iel in range (0,nel):
        for iq in [0,1,2]:
            for jq in [0,1,2]:
                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)
                NNNP[0:4]=NNP(rq,sq)
                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*x[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*y[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*x[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*y[iconV[k,iel]]
                #end for
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)
                # compute dNdx & dNdy
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for
                uq=0.
                vq=0.
                for k in range(0,mV):
                    uq+=NNNV[k]*u[iconV[k,iel]]
                    vq+=NNNV[k]*v[iconV[k,iel]]
                #end for
                vrms+=(uq**2+vq**2)*weightq*jcob
                mass+=rhoq[counter]*weightq*jcob
                counter+=1
            #end for
        #end for
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))

    vrms_file.write("%e %e %d\n" %(Time,vrms,avrg)) ; vrms_file.flush()
    mass_file.write("%.10f %.10f %.10f %d\n" %(Time,mass,mass0,avrg)) ; mass_file.flush()

    print("     -> vrms %.5e " %vrms)
    print("     -> mass %.5e " %mass)

    print("compute vrms & mass: %.3f s" % (time.time() - start))

    ######################################################################
    # advect markers 
    ######################################################################
    start = time.time()

    if istep<nstep_change:
       sign=+1
    else:
       sign=-1

    if rk==1:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           swarm_r[im]=-1+2*(swarm_x[im]-x0)/hx
           swarm_s[im]=-1+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=um*dt*sign
           swarm_y[im]+=vm*dt*sign
       #end for

    elif rk==2:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(swarm_x[im]-x0)/hx
           s=-1+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(r,s)
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           xm=swarm_x[im]+um*dt/2*sign
           ym=swarm_y[im]+vm*dt/2*sign

           ielx=int(xm/Lx*nelx)
           iely=int(ym/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           swarm_r[im]=-1+2*(xm-x0)/hx
           swarm_s[im]=-1+2*(ym-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=um*dt*sign
           swarm_y[im]+=vm*dt*sign
       #end for

    elif rk==3:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(swarm_x[im]-x0)/hx
           s=-1+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(r,s)
           uA=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vA=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           xB=swarm_x[im]+uA*dt/2*sign
           yB=swarm_y[im]+vA*dt/2*sign

           ielx=int(xB/Lx*nelx)
           iely=int(yB/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(xB-x0)/hx
           s=-1+2*(yB-y0)/hy
           NNNV[0:mV]=NNV(r,s)
           uB=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vB=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           xC=swarm_x[im]+(2*uB-uA)*dt/2*sign
           yC=swarm_y[im]+(2*vB-vA)*dt/2*sign

           ielx=int(xC/Lx*nelx)
           iely=int(yC/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           swarm_r[im]=-1+2*(xC-x0)/hx
           swarm_s[im]=-1+2*(yC-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           uC=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vC=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=(uA+4*uB+uC)*dt/6*sign
           swarm_y[im]+=(vA+4*vB+vC)*dt/6*sign
       #end for

    #end if

    print("advect markers: %.3f s" % (time.time() - start))

    ######################################################################
    # compute strainrate 
    ######################################################################
    start = time.time()

    exx = np.zeros(nel,dtype=np.float64)
    eyy = np.zeros(nel,dtype=np.float64)
    exy = np.zeros(nel,dtype=np.float64)
    e   = np.zeros(nel,dtype=np.float64)
    rho = np.zeros(nel,dtype=np.float64)

    for iel in range(0,nel):
        rq = 0.0
        sq = 0.0
        weightq = 2.0 * 2.0
        NNNV[0:9]=NNV(rq,sq)
        dNNNVdr[0:9]=dNNVdr(rq,sq)
        dNNNVds[0:9]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*x[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*y[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*x[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*y[iconV[k,iel]]
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for
        for k in range(0,mV):
            exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
            eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
            exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
        #end for
        e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
    #end for

    print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))

    #np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

    print("compute strainrate: %.3f s" % (time.time() - start))


    #####################################################################
    # interpolate pressure (q), density (rhoV) and viscosity (etaV)
    # onto velocity grid points
    #####################################################################
    start = time.time()

    q=np.zeros(NV,dtype=np.float64)
    rhoV=np.zeros(NV,dtype=np.float64)
    etaV=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        q[iconV[0,iel]]=p[iconP[0,iel]]
        q[iconV[1,iel]]=p[iconP[1,iel]]
        q[iconV[2,iel]]=p[iconP[2,iel]]
        q[iconV[3,iel]]=p[iconP[3,iel]]
        q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
        q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
        q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
        q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
        q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])*0.25
        rhoV[iconV[0,iel]]=rho_nodal[iconP[0,iel]]
        rhoV[iconV[1,iel]]=rho_nodal[iconP[1,iel]]
        rhoV[iconV[2,iel]]=rho_nodal[iconP[2,iel]]
        rhoV[iconV[3,iel]]=rho_nodal[iconP[3,iel]]
        rhoV[iconV[4,iel]]=(rho_nodal[iconP[0,iel]]+rho_nodal[iconP[1,iel]])*0.5
        rhoV[iconV[5,iel]]=(rho_nodal[iconP[1,iel]]+rho_nodal[iconP[2,iel]])*0.5
        rhoV[iconV[6,iel]]=(rho_nodal[iconP[2,iel]]+rho_nodal[iconP[3,iel]])*0.5
        rhoV[iconV[7,iel]]=(rho_nodal[iconP[3,iel]]+rho_nodal[iconP[0,iel]])*0.5
        rhoV[iconV[8,iel]]=(rho_nodal[iconP[0,iel]]+rho_nodal[iconP[1,iel]]+\
                            rho_nodal[iconP[2,iel]]+rho_nodal[iconP[3,iel]])*0.25
        etaV[iconV[0,iel]]=eta_nodal[iconP[0,iel]]
        etaV[iconV[1,iel]]=eta_nodal[iconP[1,iel]]
        etaV[iconV[2,iel]]=eta_nodal[iconP[2,iel]]
        etaV[iconV[3,iel]]=eta_nodal[iconP[3,iel]]
        etaV[iconV[4,iel]]=(eta_nodal[iconP[0,iel]]+eta_nodal[iconP[1,iel]])*0.5
        etaV[iconV[5,iel]]=(eta_nodal[iconP[1,iel]]+eta_nodal[iconP[2,iel]])*0.5
        etaV[iconV[6,iel]]=(eta_nodal[iconP[2,iel]]+eta_nodal[iconP[3,iel]])*0.5
        etaV[iconV[7,iel]]=(eta_nodal[iconP[3,iel]]+eta_nodal[iconP[0,iel]])*0.5
        etaV[iconV[8,iel]]=(eta_nodal[iconP[0,iel]]+eta_nodal[iconP[1,iel]]+\
                            eta_nodal[iconP[2,iel]]+eta_nodal[iconP[3,iel]])*0.25

    #end for

    print("     -> press nodal (m,M) %.5e %.5e " %(np.min(q),np.max(q)))
    print("     -> rho nodal (m,M) %.5e %.5e " %(np.min(rhoV),np.max(rhoV)))

    #np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

    print("interpolate press on V nodes: %.3f s" % (time.time() - start))

    ######################################################################
    # export vertical profiles
    ######################################################################

    for iq in range(0,nq):
        if abs(xq[iq]-Lx/2)<1e-6:
           profileq_file.write("%e %e %e\n" %(yq[iq],etaq[iq],rhoq[iq]))
           profileq_file.flush()

    for iel in range(0,nel):
        if abs(xc[iel]-Lx/2)<1e-6: 
           profilec_file.write("%e %e %e %e %e\n" %(y[iconV[4,iel]],eta_elemental[iel],u[iconV[4,iel]],v[iconV[4,iel]],q[iconV[4,iel]]))
           profilec_file.write("%e %e %e %e %e\n" %(y[iconV[8,iel]],eta_elemental[iel],u[iconV[8,iel]],v[iconV[8,iel]],q[iconV[8,iel]]))
           profilec_file.write("%e %e %e %e %e\n" %(y[iconV[6,iel]],eta_elemental[iel],u[iconV[6,iel]],v[iconV[6,iel]],q[iconV[6,iel]]))
           profilec_file.flush()

    #####################################################################
    # plot of solution. The 9-node Q2 element does not exist in vtk, 
    # but the 8-node one does, i.e. type=23. 
    #####################################################################
    start = time.time()
    
    if istep%every==0:

       filename = 'fields_ls_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(4*nel,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e %10e %10e \n" %(x[iconV[0,iel]],y[iconV[0,iel]],0.))
           vtufile.write("%10e %10e %10e \n" %(x[iconV[1,iel]],y[iconV[1,iel]],0.))
           vtufile.write("%10e %10e %10e \n" %(x[iconV[2,iel]],y[iconV[2,iel]],0.))
           vtufile.write("%10e %10e %10e \n" %(x[iconV[3,iel]],y[iconV[3,iel]],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='eta_min' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (eta_min[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eta_max' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (eta_max[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eta_max-eta_min' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (eta_max[iel]-eta_min[iel]))
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='rho_min' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (rho_min[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rho_max' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (rho_max[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rho_max-rho_min' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (rho_max[iel]-rho_min[iel]))
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='a_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (a_ls[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='b_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (b_ls[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='c_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (c_ls[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='b_ls (after limiter)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (b_ls2[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='c_ls (after limiter)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (c_ls2[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='d_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (d_ls[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='e_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (e_ls[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='f_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (f_ls[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='e_ls (after limiter)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (e_ls2[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='f_ls (after limiter)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (f_ls2[iel]))
       vtufile.write("</DataArray>\n")

       vtufile.write("</CellData>\n")
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (a_ls[iel]+b_ls[iel]*(x[iconV[0,iel]]-xc[iel])+c_ls[iel]*(y[iconV[0,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (a_ls[iel]+b_ls[iel]*(x[iconV[1,iel]]-xc[iel])+c_ls[iel]*(y[iconV[1,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (a_ls[iel]+b_ls[iel]*(x[iconV[2,iel]]-xc[iel])+c_ls[iel]*(y[iconV[2,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (a_ls[iel]+b_ls[iel]*(x[iconV[3,iel]]-xc[iel])+c_ls[iel]*(y[iconV[3,iel]]-yc[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta_ls2' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (a_ls[iel]+b_ls2[iel]*(x[iconV[0,iel]]-xc[iel])+c_ls2[iel]*(y[iconV[0,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (a_ls[iel]+b_ls2[iel]*(x[iconV[1,iel]]-xc[iel])+c_ls2[iel]*(y[iconV[1,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (a_ls[iel]+b_ls2[iel]*(x[iconV[2,iel]]-xc[iel])+c_ls2[iel]*(y[iconV[2,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (a_ls[iel]+b_ls2[iel]*(x[iconV[3,iel]]-xc[iel])+c_ls2[iel]*(y[iconV[3,iel]]-yc[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (d_ls[iel]+e_ls[iel]*(x[iconV[0,iel]]-xc[iel])+f_ls[iel]*(y[iconV[0,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (d_ls[iel]+e_ls[iel]*(x[iconV[1,iel]]-xc[iel])+f_ls[iel]*(y[iconV[1,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (d_ls[iel]+e_ls[iel]*(x[iconV[2,iel]]-xc[iel])+f_ls[iel]*(y[iconV[2,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (d_ls[iel]+e_ls[iel]*(x[iconV[3,iel]]-xc[iel])+f_ls[iel]*(y[iconV[3,iel]]-yc[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho_ls2' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (d_ls[iel]+e_ls2[iel]*(x[iconV[0,iel]]-xc[iel])+f_ls2[iel]*(y[iconV[0,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (d_ls[iel]+e_ls2[iel]*(x[iconV[1,iel]]-xc[iel])+f_ls2[iel]*(y[iconV[1,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (d_ls[iel]+e_ls2[iel]*(x[iconV[2,iel]]-xc[iel])+f_ls2[iel]*(y[iconV[2,iel]]-yc[iel]) ))
           vtufile.write("%10e\n" % (d_ls[iel]+e_ls2[iel]*(x[iconV[3,iel]]-xc[iel])+f_ls2[iel]*(y[iconV[3,iel]]-yc[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(iel*4,iel*4+1,iel*4+2,iel*4+3))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %5)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()


       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % exx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % exy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % eyy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sr' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % e[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (eta_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='nmarker' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (nmarker_in_element[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %rhoV[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %etaV[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                          iconV[3,iel],iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %23)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()


       filename = 'markers_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'>\n")
       for i in range(0,nmarker):
           vtufile.write("%3e \n" %swarm_mat[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'>\n")
       for i in range(0,nmarker):
           vtufile.write("%3e \n" %swarm_paint[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='displacement' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nmarker):
           vtufile.write("%5e %5e %5e \n" %(swarm_x[i]-swarm_x0[i],swarm_y[i]-swarm_y0[i],0.))
       vtufile.write("</DataArray>\n")

       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,nmarker):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()


       filename = 'qpts_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%10e  \n" %etaq[i])
       vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Float32' Name='log10(eta)' Format='ascii'>\n")
       #for i in range(0,nq):
       #    vtufile.write("%10e  \n" %(np.log10(etaq[i])))
       #vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%10e  \n" %rhoq[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")

       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%10e %10e %10e \n" %(xq[i],yq[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,nq):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,nq):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

    print("write vtu files: %.3f s" % (time.time() - start))

    Time+=dt

#end for istep

#--------------------------------------------------------------------------------------------------
# end time stepping loop
#--------------------------------------------------------------------------------------------------

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
