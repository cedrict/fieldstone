import sys as sys
import numpy as np
import time as clock 
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix

###################################################################################################
# These two functions were written by G. Mack

def sign(n):
    if n < 0:
        return -1.0
    return 1.0

def limiter(c_0, c_1, c_2, S = 0, L=1, h=1):
    """
    Limits a the slopes of a plane in 2D
    See latex for a complete explanation
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

###################################################################################################

def basis_functions_V(r,s):
    N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N4=    (1.-r**2) * 0.5*s*(s-1.)
    N5= 0.5*r*(r+1.) *    (1.-s**2)
    N6=    (1.-r**2) * 0.5*s*(s+1.)
    N7= 0.5*r*(r-1.) *    (1.-s**2)
    N8=    (1.-r**2) *    (1.-s**2)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr4=       (-2.*r) * 0.5*s*(s-1)
    dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr6=       (-2.*r) * 0.5*s*(s+1)
    dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds5= 0.5*r*(r+1.) *       (-2.*s)
    dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds7= 0.5*r*(r-1.) *       (-2.*s)
    dNds8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###################################################################################################

#model=1: van Keken et al 1997
#model=2: absent ?!
#model=3: stokes sphere with free surf

model=3

###################################################################################################

# avrg=-1: nodal averaging - arithmetic
# avrg=-2: nodal averaging - geometric
# avrg=-3: nddal averaging - harmonic
# avrg=1: elemental averaging - arithmetic
# avrg=2: elemental averaging - geometric
# avrg=3: elemental averaging - harmonic
# avrg=4: least square 

###################################################################################################

print("*******************************")
print("********** stone 041 **********")
print("*******************************")

m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node
ndim=2

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
   nparticle_per_dim=7
   eta_ref=1e21      # scaling of G blocks
   year=3600.*24.*365.25
   avrg=3

if model==3:
   Lx=1
   Ly=1
   nelx=32
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
   nparticle_per_dim=5
   eta_ref=1
   year=1
   avrg=3
   particle_rho_projection=0 #0 nodal, 1 elemental, 2 least-squares

if int(len(sys.argv) == 3):
   avrg=int(sys.argv[1])
   nelx=int(sys.argv[2])
   nely=nelx

nel=nelx*nely  
nnx=2*nelx+1  
nny=2*nely+1  
nn_V=nnx*nny  
nn_P=(nelx+1)*(nely+1)

Nfem_V=nn_V*ndof_V      
Nfem_P=(nelx+1)*(nely+1)
Nfem=Nfem_V+Nfem_P     

nq_per_dim=3
nq=nq_per_dim**2*nel
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

nparticle_per_element=nparticle_per_dim**2
nparticle=nparticle_per_element*nel
pnormalise=False

nstep_change=2500000
nstep=1
CFL_nb=0.0025
RKorder=2

every=1

debug=False

###################################################################################################
###################################################################################################

print("model=",model)
print("nelx=",nelx)
print("nely=",nely)
print("nel=",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("avrg=",avrg)
print("particle_rho_projection",particle_rho_projection)
print("RKorder=",RKorder)
print("------------------------------")

dt_file=open('dt.ascii',"w")
vrms_file=open('vrms.ascii',"w")
mass_file=open('mass.ascii',"w")
profileq_file=open('profileq.ascii',"w")
profilec_file=open('profilec.ascii',"w")
nparticle_file=open('nparticle_per_element.ascii',"w")

###################################################################################################
# grid point setup
###################################################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter += 1
    #end for
#end for

if debug: np.savetxt('grid_V.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("grid setup: %.3f s" % (clock.time()-start))

###################################################################################################
# connectivity
###################################################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###################################################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
        icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
        icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
        icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter+=1
    #end for
#end for

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter+=1
    #end for
#end for

print("connectivity: %.3f s" % (clock.time()-start))

###################################################################################################
# particle setup: regular distribution
###################################################################################################
start=clock.time()

swarm_mat=np.zeros(nparticle,dtype=np.int8)  
swarm_x=np.zeros(nparticle,dtype=np.float64) 
swarm_y=np.zeros(nparticle,dtype=np.float64) 
swarm_r=np.zeros(nparticle,dtype=np.float64) 
swarm_s=np.zeros(nparticle,dtype=np.float64) 
swarm_x0=np.zeros(nparticle,dtype=np.float64) 
swarm_y0=np.zeros(nparticle,dtype=np.float64) 
swarm_paint=np.zeros(nparticle,dtype=np.float64) 

counter=0
for iel in range(0,nel):
    x1=x_V[icon_V[0,iel]] ; y1=y_V[icon_V[0,iel]]
    x2=x_V[icon_V[1,iel]] ; y2=y_V[icon_V[1,iel]]
    x3=x_V[icon_V[2,iel]] ; y3=y_V[icon_V[2,iel]]
    x4=x_V[icon_V[3,iel]] ; y4=y_V[icon_V[3,iel]]
    for j in range(0,nparticle_per_dim):
        for i in range(0,nparticle_per_dim):
            r=-1.+i*2./nparticle_per_dim + 1./nparticle_per_dim
            s=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
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

swarm_x0[0:nparticle]=swarm_x[0:nparticle]
swarm_y0[0:nparticle]=swarm_y[0:nparticle]

print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("particle setup: %.3f s" % (clock.time()-start))

###################################################################################################
# assign material id to particles 
###################################################################################################
start=clock.time()

if model==1:
   for im in range (0,nparticle):
       if swarm_y[im]>salt_thickness+amplitude*np.cos(np.pi*swarm_x[im]/Lx):
          swarm_mat[im]=2
       else:
          swarm_mat[im]=1

if model==3:
   for im in range (0,nparticle):
       swarm_mat[im]=1
       if swarm_y[im]>0.75: swarm_mat[im]=3
       if (swarm_x[im]-x_sphere)**2+(swarm_y[im]-y_sphere)**2<R_sphere**2: swarm_mat[im]=2

print("particle layout: %.3f s" % (clock.time()-start))

###################################################################################################
# paint particles 
###################################################################################################
start=clock.time()

for im in range (0,nparticle):
    swarm_paint[im]=(np.sin(2*np.pi*swarm_x[im]/Lx*4) * np.sin(2*np.pi*swarm_y[im]/Ly*4))

if debug: np.savetxt('particles.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y,mat')

print("particle paint: %.3f s" % (clock.time()-start))

###################################################################################################
# define boundary conditions
###################################################################################################
start=clock.time()

eps=1.e-10

bc_fix_V=np.zeros(Nfem_V,dtype=bool)
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)

if model==1:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if x_V[i]/Lx>(1-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if y_V[i]/Ly<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0. 
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
       if y_V[i]/Ly>(1-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0. 

if model==3:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if x_V[i]/Lx>(1-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if y_V[i]/Ly<eps:
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
       if y_V[i]/Ly>(1-eps):
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0. 

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###################################################################################################
# compute element center coordinates 
###################################################################################################
start=clock.time()

xc=np.zeros(nel,dtype=np.float64)
yc=np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    xc[iel]=(x_V[icon_V[0,iel]]+x_V[icon_V[2,iel]])/2
    yc[iel]=(y_V[icon_V[0,iel]]+y_V[icon_V[2,iel]])/2

print("compute element center coords: %.3f s" % (clock.time()-start))

###################################################################################################
# compute element area, test basis functions, jacobian, ...
###################################################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
#end for

print("     -> area (m,M) %.5e %.5e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))

###################################################################################################
###################################################################################################
# time stepping loop
###################################################################################################
###################################################################################################

Time=0.

for istep in range(0,nstep):

    print ('------------------------------------')
    print ('--------------istep= %i -------------' %istep)
    print ('------------------------------------')

    ###########################################################################
    # compute elemental averagings 
    ###########################################################################
    start=clock.time()

    rho_elemental=np.zeros(nel,dtype=np.float64) 
    eta_elemental=np.zeros(nel,dtype=np.float64) 
    rho_nodal=np.zeros(nn_P,dtype=np.float64) 
    eta_nodal=np.zeros(nn_P,dtype=np.float64) 
    nparticle_in_element=np.zeros(nel,dtype=np.int16) 
    nodal_counter=np.zeros(nn_P,dtype=np.float64) 
    list_of_particles_in_element=np.zeros((2*nparticle_per_element,nel),dtype=np.int32)
    eta_min=np.zeros(nel,dtype=np.float64) ; eta_min[:]=1e36
    eta_max=np.zeros(nel,dtype=np.float64) 
    rho_min=np.zeros(nel,dtype=np.float64) ; rho_min[:]=1e6
    rho_max=np.zeros(nel,dtype=np.float64) 

    for im in range(0,nparticle):
        #localise particle
        ielx=int(swarm_x[im]/Lx*nelx)
        iely=int(swarm_y[im]/Ly*nely)
        iel=nelx*(iely)+ielx
        if debug:
           if ielx<0:      print ('ielx<0',ielx)
           if ielx>nelx-1: print ('ielx>nelx-1')
           if iely<0:      print ('iely<0')
           if iely>nely-1: print ('iely>nely-1')
           if iel<0:       print ('iel<0')
           if iel>nel-1:   print ('iel>nel-1')
        # computing Q1 weights
        N0=0.25*(1-swarm_r[im])*(1-swarm_s[im])
        N1=0.25*(1+swarm_r[im])*(1-swarm_s[im])
        N2=0.25*(1+swarm_r[im])*(1+swarm_s[im])
        N3=0.25*(1-swarm_r[im])*(1+swarm_s[im])

        #arithmetic averaging for density
        rho_nodal[icon_P[0,iel]]+=rho_mat[swarm_mat[im]-1]*N0
        rho_nodal[icon_P[1,iel]]+=rho_mat[swarm_mat[im]-1]*N1
        rho_nodal[icon_P[2,iel]]+=rho_mat[swarm_mat[im]-1]*N2
        rho_nodal[icon_P[3,iel]]+=rho_mat[swarm_mat[im]-1]*N3
        nodal_counter[icon_P[0,iel]]+=N0
        nodal_counter[icon_P[1,iel]]+=N1
        nodal_counter[icon_P[2,iel]]+=N2
        nodal_counter[icon_P[3,iel]]+=N3

        rho_min[iel]=min(rho_min[iel],rho_mat[swarm_mat[im]-1])
        rho_max[iel]=max(rho_max[iel],rho_mat[swarm_mat[im]-1])

        eta_min[iel]=min(eta_min[iel],eta_mat[swarm_mat[im]-1])
        eta_max[iel]=max(eta_max[iel],eta_mat[swarm_mat[im]-1])

        list_of_particles_in_element[nparticle_in_element[iel],iel]=im
        nparticle_in_element[iel]+=1

        rho_elemental[iel]+=rho_mat[swarm_mat[im]-1]

        if abs(avrg)==1 : # arithmetic
           eta_elemental[iel]      +=eta_mat[swarm_mat[im]-1]
           eta_nodal[icon_P[0,iel]]+=eta_mat[swarm_mat[im]-1]*N0
           eta_nodal[icon_P[1,iel]]+=eta_mat[swarm_mat[im]-1]*N1
           eta_nodal[icon_P[2,iel]]+=eta_mat[swarm_mat[im]-1]*N2
           eta_nodal[icon_P[3,iel]]+=eta_mat[swarm_mat[im]-1]*N3
        if abs(avrg)==2: # geometric
           eta_elemental[iel]      +=np.log10(eta_mat[swarm_mat[im]-1])
           eta_nodal[icon_P[0,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N0
           eta_nodal[icon_P[1,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N1
           eta_nodal[icon_P[2,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N2
           eta_nodal[icon_P[3,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N3
        if abs(avrg)==3: # harmonic
           eta_elemental[iel]      +=1/eta_mat[swarm_mat[im]-1]
           eta_nodal[icon_P[0,iel]]+=1/eta_mat[swarm_mat[im]-1]*N0
           eta_nodal[icon_P[1,iel]]+=1/eta_mat[swarm_mat[im]-1]*N1
           eta_nodal[icon_P[2,iel]]+=1/eta_mat[swarm_mat[im]-1]*N2
           eta_nodal[icon_P[3,iel]]+=1/eta_mat[swarm_mat[im]-1]*N3
    #end for

    rho_nodal/=nodal_counter
    rho_elemental[:]/=nparticle_in_element[:]

    if abs(avrg)==1:
       eta_nodal/=nodal_counter
       eta_elemental[:]/=nparticle_in_element[:]
    if abs(avrg)==2:
       eta_nodal[:]=10.**(eta_nodal[:]/nodal_counter[:])
       eta_elemental[:]=10.**(eta_elemental[:]/nparticle_in_element[:])
    if abs(avrg)==3:
       eta_nodal[:]=nodal_counter[:]/eta_nodal[:]
       eta_elemental[:]=nparticle_in_element[:]/eta_elemental[:]

    nparticle_file.write("%d %e %e\n" %(istep,np.min(nparticle_in_element),np.max(nparticle_in_element))) 
    nparticle_file.flush()

    if np.min(nparticle_in_element)==0: exit('no particle left in an element')

    print("     -> nparticle_in_elt(m,M) %.5e %.5e " %(np.min(nparticle_in_element),np.max(nparticle_in_element)))
    print("     -> rho_elemental   (m,M) %.5e %.5e " %(np.min(rho_elemental),np.max(rho_elemental)))
    print("     -> rho_nodal       (m,M) %.5e %.5e " %(np.min(rho_nodal),np.max(rho_nodal)))
    print("     -> eta_elemental   (m,M) %.5e %.5e " %(np.min(eta_elemental),np.max(eta_elemental)))
    print("     -> eta_nodal       (m,M) %.5e %.5e " %(np.min(eta_nodal),np.max(eta_nodal)))

    print("particles onto grid: %.3f s" % (clock.time()-start))

    ###########################################################################
    # least square process
    # for each element I loop over the particles in it and build the corresponding 
    # A matrix and b rhs. The solution, i.e. the coefficients (a,b,c for 
    # viscosity, d,e,f for density) are stored for each element.
    ###########################################################################
    start=clock.time()

    ls_eta_a=np.zeros(nel,dtype=np.float64) 
    ls_eta_b=np.zeros(nel,dtype=np.float64) 
    ls_eta_c=np.zeros(nel,dtype=np.float64) 
    ls_rho_a=np.zeros(nel,dtype=np.float64) 
    ls_rho_b=np.zeros(nel,dtype=np.float64) 
    ls_rho_c=np.zeros(nel,dtype=np.float64) 

    ls_eta_b2=np.zeros(nel,dtype=np.float64) 
    ls_eta_c2=np.zeros(nel,dtype=np.float64) 
    ls_rho_b2=np.zeros(nel,dtype=np.float64) 
    ls_rho_c2=np.zeros(nel,dtype=np.float64) 

    for iel in range(0,nel):
        A_ls = np.zeros((3,3),dtype=np.float64) 
        rhs_etls_eta_a = np.zeros((3),dtype=np.float64) 
        rhs_rho_ls = np.zeros((3),dtype=np.float64) 
        A_ls[0,0]=nparticle_in_element[iel]
        for i in range(0,nparticle_in_element[iel]):
            im=list_of_particles_in_element[i,iel]
            xim=swarm_x[im]-xc[iel]
            yim=swarm_y[im]-yc[iel]
            A_ls[0,1]+=xim
            A_ls[0,2]+=yim
            A_ls[1,1]+=xim**2
            A_ls[1,2]+=xim*yim
            A_ls[2,2]+=yim**2
            rhs_etls_eta_a[0]+=eta_mat[swarm_mat[im]-1]
            rhs_etls_eta_a[1]+=eta_mat[swarm_mat[im]-1]*xim
            rhs_etls_eta_a[2]+=eta_mat[swarm_mat[im]-1]*yim
            rhs_rho_ls[0]+=rho_mat[swarm_mat[im]-1]
            rhs_rho_ls[1]+=rho_mat[swarm_mat[im]-1]*xim
            rhs_rho_ls[2]+=rho_mat[swarm_mat[im]-1]*yim
        #end for
        A_ls[1,0]=A_ls[0,1]
        A_ls[2,0]=A_ls[0,2]
        A_ls[2,1]=A_ls[1,2]
        sol=np.linalg.solve(A_ls,rhs_etls_eta_a)
        ls_eta_a[iel]=sol[0]
        ls_eta_b[iel]=sol[1]
        ls_eta_c[iel]=sol[2]
        sol=np.linalg.solve(A_ls,rhs_rho_ls)
        ls_rho_a[iel]=sol[0]
        ls_rho_b[iel]=sol[1]
        ls_rho_c[iel]=sol[2]
    #end for

    #apply limiter
    for iel in range(0,nel):
        ls_eta_b2[iel],ls_eta_c2[iel]=limiter(ls_eta_a[iel],ls_eta_b[iel],ls_eta_c[iel],eta_min[iel],eta_max[iel],hx)
        ls_rho_b2[iel],ls_rho_c2[iel]=limiter(ls_rho_a[iel],ls_rho_b[iel],ls_rho_c[iel],rho_min[iel],rho_max[iel],hx)

    print("compute ls coefficients: %.3f s" % (clock.time()-start))

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start=clock.time()

    if pnormalise:
       A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
       rhs   = np.zeros(Nfem+1,dtype=np.float64)
    else:
       A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
       rhs   = np.zeros(Nfem,dtype=np.float64) 

    f_rhs=np.zeros(Nfem_V,dtype=np.float64)   
    h_rhs=np.zeros(Nfem_P,dtype=np.float64)    
    constr=np.zeros(Nfem_P,dtype=np.float64)    
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64) 
    N_mat=np.zeros((3,m_P),dtype=np.float64)     
    xq=np.zeros(nq,dtype=np.float64) 
    yq=np.zeros(nq,dtype=np.float64) 
    rhoq=np.zeros(nq,dtype=np.float64)    
    etaq=np.zeros(nq,dtype=np.float64)   
    C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    counter=0
    for iel in range(0,nel):

        f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
        K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)
        NNNNP= np.zeros(m_P,dtype=np.float64)   

        for iq in [0,1,2]:
            for jq in [0,1,2]:
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                N_V=basis_functions_V(rq,sq)
                N_P=basis_functions_P(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                xq[counter]=np.dot(N_V,x_V[icon_V[:,iel]])
                yq[counter]=np.dot(N_V,y_V[icon_V[:,iel]])
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                # assign rho, eta at qpoint
                match particle_rho_projection:
                 case 0:
                  rhoq[counter]=np.dot(N_P,rho_nodal[icon_P[:,iel]])
                 case 1:
                  rhoq[counter]=rho_elemental[iel]
                 case 2:
                  rhoq[counter]=ls_rho_a[iel]+ls_rho_b2[iel]*(xq[counter]-xc[iel])+ls_rho_c2[iel]*(yq[counter]-yc[iel])

                match avrg:
                 case -1 | -2 | -3:
                  etaq[counter]=np.dot(N_P,eta_nodal[icon_P[:,iel]])
                 case 1 | 2 | 3:
                  etaq[counter]=eta_elemental[iel]
                 case 4: 
                   etaq[counter]=ls_eta_a[iel]+ls_eta_b2[iel]*(xq[counter]-xc[iel])+ls_eta_c2[iel]*(yq[counter]-yc[iel])

                # construct 3x8 B matrix
                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                # compute elemental a_mat matrix
                K_el+=B.T.dot(C.dot(B))*etaq[counter]*JxWq

                for i in range(0,m_V):
                    f_el[ndof_V*i  ]+=N_V[i]*JxWq*rhoq[counter]*gx
                    f_el[ndof_V*i+1]+=N_V[i]*JxWq*rhoq[counter]*gy
                #end for

                for i in range(0,m_P):
                    N_mat[0,i]=N_P[i]
                    N_mat[1,i]=N_P[i]
                    N_mat[2,i]=0.
                #end for

                G_el-=B.T.dot(N_mat)*JxWq
                NNNNP[:]+=N_P[:]*JxWq
                counter+=1
            #end for
        #end for

        G_el*=eta_ref/Ly

        # impose b.c. 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1=ndof_V*icon_V[k1,iel]+i1
                if bc_fix_V[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m_V*ndof_V):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   #end for
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val_V[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                   G_el[ikk,:]=0
                #end for
            #end for
        #end for

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        A_sparse[m1,m2] += K_el[ikk,jkk]
                    #end for
                #end for
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]#*eta_ref/Ly
                    A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]#*eta_ref/Ly
                #end for
                f_rhs[m1]+=f_el[ikk]
            #end for
        #end for

        if pnormalise:
           for k2 in range(0,m_P):
               m2=icon_P[k2,iel]
               h_rhs[m2]+=h_el[k2]
               constr[m2]+=NNNNP[k2]
               A_sparse[Nfem,Nfem_V+m2]=constr[m2]
               A_sparse[Nfem_V+m2,Nfem]=constr[m2]
           #end for

    #end for

    h_rhs*=eta_ref/Ly

    print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
    print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

    if debug: np.savetxt('rhoq.ascii',np.array([xq,yq,rhoq]).T,header='# x,y,rho')
    if debug: np.savetxt('etaq.ascii',np.array([xq,yq,etaq]).T,header='# x,y,eta')

    print("build FE matrix: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    rhs[0:Nfem_V]=f_rhs
    rhs[Nfem_V:Nfem]=h_rhs

    sparse_matrix=A_sparse.tocsr()
    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

    print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))

    if debug: np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (clock.time()-start))

    ###########################################################################
    # normalise pressure as post-processor
    ###########################################################################
    start=clock.time()

    if not pnormalise:
       pavrg=0
       for iel in range(nel-nelx,nel):
           pavrg+=(p[icon_P[3,iel]]+p[icon_P[2,iel]])/2*hx
       pavrg/=Lx
       p-=pavrg 

    print("normalise pressure: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute timestep 
    ###########################################################################
    start=clock.time()

    dt=CFL_nb*min(hx,hy)/max(max(abs(u)),max(abs(v)))

    print("     -> dt= %.3e yr" %(dt/year))

    dt_file.write("%d %e \n" %(istep,dt)) ; dt_file.flush()

    print("compute time step: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute vrms 
    ###########################################################################
    start=clock.time()

    vrms=0.
    mass=0.
    counter=0
    for iel in range (0,nel):
        for iq in [0,1,2]:
            for jq in [0,1,2]:
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                N_V=basis_functions_V(rq,sq)
                N_P=basis_functions_P(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                vrms+=(uq**2+vq**2)*JxWq
                mass+=rhoq[counter]*JxWq
                counter+=1
            #end for
        #end for
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))

    vrms_file.write("%e %e %d\n" %(Time,vrms,avrg)) ; vrms_file.flush()
    mass_file.write("%e %e %e %d\n" %(Time,mass,mass0,avrg)) ; mass_file.flush()

    print("     -> vrms %.5e " %vrms)
    print("     -> mass %.5e " %mass)

    print("compute vrms & mass: %.3f s" % (clock.time()-start))

    ######################################################################
    # advect particles 
    ######################################################################
    start=clock.time()

    if istep<nstep_change:
       sign=+1
    else:
       sign=-1

    if RKorder==1:
       for im in range(0,nparticle):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x_V[icon_V[0,iel]]
           y0=y_V[icon_V[0,iel]]
           swarm_r[im]=-1+2*(swarm_x[im]-x0)/hx
           swarm_s[im]=-1+2*(swarm_y[im]-y0)/hy
           N_V=basis_functions_V(swarm_r[im],swarm_s[im])
           um=np.dot(N_V,u[icon_V[:,iel]])
           vm=np.dot(N_V,v[icon_V[:,iel]])
           swarm_x[im]+=um*dt*sign
           swarm_y[im]+=vm*dt*sign
       #end for

    elif RKorder==2:
       for im in range(0,nparticle):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x_V[icon_V[0,iel]]
           y0=y_V[icon_V[0,iel]]
           r=-1+2*(swarm_x[im]-x0)/hx
           s=-1+2*(swarm_y[im]-y0)/hy
           N_V=basis_functions_V(r,s)
           um=np.dot(N_V,u[icon_V[:,iel]])
           vm=np.dot(N_V,v[icon_V[:,iel]])
           xm=swarm_x[im]+um*dt/2*sign
           ym=swarm_y[im]+vm*dt/2*sign

           ielx=int(xm/Lx*nelx)
           iely=int(ym/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x_V[icon_V[0,iel]]
           y0=y_V[icon_V[0,iel]]
           swarm_r[im]=-1+2*(xm-x0)/hx
           swarm_s[im]=-1+2*(ym-y0)/hy
           N_V=basis_functions_V(swarm_r[im],swarm_s[im])
           um=np.dot(N_V,u[icon_V[:,iel]])
           vm=np.dot(N_V,v[icon_V[:,iel]])
           swarm_x[im]+=um*dt*sign
           swarm_y[im]+=vm*dt*sign
       #end for

    elif RKorder==3:
       for im in range(0,nparticle):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x_V[icon_V[0,iel]]
           y0=y_V[icon_V[0,iel]]
           r=-1+2*(swarm_x[im]-x0)/hx
           s=-1+2*(swarm_y[im]-y0)/hy
           N_V=basis_functions_V(r,s)
           uA=np.dot(N_V,u[icon_V[:,iel]])
           vA=np.dot(N_V,v[icon_V[:,iel]])
           xB=swarm_x[im]+uA*dt/2*sign
           yB=swarm_y[im]+vA*dt/2*sign

           ielx=int(xB/Lx*nelx)
           iely=int(yB/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x_V[icon_V[0,iel]]
           y0=y_V[icon_V[0,iel]]
           r=-1+2*(xB-x0)/hx
           s=-1+2*(yB-y0)/hy
           N_V=basis_functions_V(r,s)
           uB=np.dot(N_V,u[icon_V[:,iel]])
           vB=np.dot(N_V,v[icon_V[:,iel]])
           xC=swarm_x[im]+(2*uB-uA)*dt/2*sign
           yC=swarm_y[im]+(2*vB-vA)*dt/2*sign

           ielx=int(xC/Lx*nelx)
           iely=int(yC/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x_V[icon_V[0,iel]]
           y0=y_V[icon_V[0,iel]]
           swarm_r[im]=-1+2*(xC-x0)/hx
           swarm_s[im]=-1+2*(yC-y0)/hy
           N_V=basis_functions_V(swarm_r[im],swarm_s[im])
           uC=np.dot(N_V,u[icon_V[:,iel]])
           vC=np.dot(N_V,v[icon_V[:,iel]])
           swarm_x[im]+=(uA+4*uB+uC)*dt/6*sign
           swarm_y[im]+=(vA+4*vB+vC)*dt/6*sign
       #end for

    #end if

    print("advect particles: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute elemental strainrate 
    ###########################################################################
    start=clock.time()

    e=np.zeros(nel,dtype=np.float64)
    exx=np.zeros(nel,dtype=np.float64)
    eyy=np.zeros(nel,dtype=np.float64)
    exy=np.zeros(nel,dtype=np.float64)
    rho=np.zeros(nel,dtype=np.float64)

    for iel in range(0,nel):
        rq=0.0
        sq=0.0
        N_V=basis_functions_V(rq,sq)
        N_P=basis_functions_P(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
        eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
        exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
                +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
        e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
    #end for

    print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))

    if debug: np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

    print("compute strainrate: %.3f s" % (clock.time()-start))

    ###############################################################################################
    # interpolate pressure (q), density (rhoV) and viscosity (etaV)
    # onto velocity grid points
    ###############################################################################################
    start=clock.time()

    q=np.zeros(nn_V,dtype=np.float64)
    rhoV=np.zeros(nn_V,dtype=np.float64)
    etaV=np.zeros(nn_V,dtype=np.float64)

    for iel in range(0,nel):
        q[icon_V[0,iel]]=p[icon_P[0,iel]]
        q[icon_V[1,iel]]=p[icon_P[1,iel]]
        q[icon_V[2,iel]]=p[icon_P[2,iel]]
        q[icon_V[3,iel]]=p[icon_P[3,iel]]
        q[icon_V[4,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]])*0.5
        q[icon_V[5,iel]]=(p[icon_P[1,iel]]+p[icon_P[2,iel]])*0.5
        q[icon_V[6,iel]]=(p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.5
        q[icon_V[7,iel]]=(p[icon_P[3,iel]]+p[icon_P[0,iel]])*0.5
        q[icon_V[8,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]]+p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.25
        rhoV[icon_V[0,iel]]=rho_nodal[icon_P[0,iel]]
        rhoV[icon_V[1,iel]]=rho_nodal[icon_P[1,iel]]
        rhoV[icon_V[2,iel]]=rho_nodal[icon_P[2,iel]]
        rhoV[icon_V[3,iel]]=rho_nodal[icon_P[3,iel]]
        rhoV[icon_V[4,iel]]=(rho_nodal[icon_P[0,iel]]+rho_nodal[icon_P[1,iel]])*0.5
        rhoV[icon_V[5,iel]]=(rho_nodal[icon_P[1,iel]]+rho_nodal[icon_P[2,iel]])*0.5
        rhoV[icon_V[6,iel]]=(rho_nodal[icon_P[2,iel]]+rho_nodal[icon_P[3,iel]])*0.5
        rhoV[icon_V[7,iel]]=(rho_nodal[icon_P[3,iel]]+rho_nodal[icon_P[0,iel]])*0.5
        rhoV[icon_V[8,iel]]=(rho_nodal[icon_P[0,iel]]+rho_nodal[icon_P[1,iel]]+\
                            rho_nodal[icon_P[2,iel]]+rho_nodal[icon_P[3,iel]])*0.25
        etaV[icon_V[0,iel]]=eta_nodal[icon_P[0,iel]]
        etaV[icon_V[1,iel]]=eta_nodal[icon_P[1,iel]]
        etaV[icon_V[2,iel]]=eta_nodal[icon_P[2,iel]]
        etaV[icon_V[3,iel]]=eta_nodal[icon_P[3,iel]]
        etaV[icon_V[4,iel]]=(eta_nodal[icon_P[0,iel]]+eta_nodal[icon_P[1,iel]])*0.5
        etaV[icon_V[5,iel]]=(eta_nodal[icon_P[1,iel]]+eta_nodal[icon_P[2,iel]])*0.5
        etaV[icon_V[6,iel]]=(eta_nodal[icon_P[2,iel]]+eta_nodal[icon_P[3,iel]])*0.5
        etaV[icon_V[7,iel]]=(eta_nodal[icon_P[3,iel]]+eta_nodal[icon_P[0,iel]])*0.5
        etaV[icon_V[8,iel]]=(eta_nodal[icon_P[0,iel]]+eta_nodal[icon_P[1,iel]]+\
                            eta_nodal[icon_P[2,iel]]+eta_nodal[icon_P[3,iel]])*0.25

    print("     -> press nodal (m,M) %.5e %.5e " %(np.min(q),np.max(q)))
    print("     -> rho nodal (m,M) %.5e %.5e " %(np.min(rhoV),np.max(rhoV)))

    if debug: np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

    print("interpolate press on V nodes: %.3f s" % (clock.time()-start))

    ###############################################################################################
    # export vertical profiles
    ###############################################################################################
    start=clock.time()

    for iq in range(0,nq):
        if abs(xq[iq]-Lx/2)<1e-6:
           profileq_file.write("%e %e %e\n" %(yq[iq],etaq[iq],rhoq[iq]))
           profileq_file.flush()

    for iel in range(0,nel):
        if abs(xc[iel]-Lx/2)<1e-6: 
           profilec_file.write("%e %e %e %e %e\n" %(y_V[icon_V[4,iel]],eta_elemental[iel],u[icon_V[4,iel]],v[icon_V[4,iel]],q[icon_V[4,iel]]))
           profilec_file.write("%e %e %e %e %e\n" %(y_V[icon_V[8,iel]],eta_elemental[iel],u[icon_V[8,iel]],v[icon_V[8,iel]],q[icon_V[8,iel]]))
           profilec_file.write("%e %e %e %e %e\n" %(y_V[icon_V[6,iel]],eta_elemental[iel],u[icon_V[6,iel]],v[icon_V[6,iel]],q[icon_V[6,iel]]))
           profilec_file.flush()

    print("export vertical profile: %.3f s" % (clock.time()-start))

    ###############################################################################################
    # plot of solution. The 9-node Q2 element does not exist in vtk, 
    # but the 8-node one does, i.e. type=23. 
    ###############################################################################################
    start=clock.time()
    
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
           vtufile.write("%e %e %.1e \n" %(x_V[icon_V[0,iel]],y_V[icon_V[0,iel]],0.))
           vtufile.write("%e %e %.1e \n" %(x_V[icon_V[1,iel]],y_V[icon_V[1,iel]],0.))
           vtufile.write("%e %e %.1e \n" %(x_V[icon_V[2,iel]],y_V[icon_V[2,iel]],0.))
           vtufile.write("%e %e %.1e \n" %(x_V[icon_V[3,iel]],y_V[icon_V[3,iel]],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<CellData Scalars='scalars'>\n")
       #
       vtufile.write("<DataArray type='Float32' Name='eta_min' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (eta_min[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eta_max' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (eta_max[iel]))
       vtufile.write("</DataArray>\n")
       #
       vtufile.write("<DataArray type='Float32' Name='rho_min' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (rho_min[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rho_max' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (rho_max[iel]))
       vtufile.write("</DataArray>\n")
       #
       vtufile.write("<DataArray type='Float32' Name='ls_eta_a' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_a[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_eta_b' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_b[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_eta_c' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_c[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_eta_b (after limiter)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_b2[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_eta_c (after limiter)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_c2[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_rho_a' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_a[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_rho_b' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_b[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_rho_c' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_c[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_rho_b (after limiter)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_b2[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_rho_c (after limiter)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_c2[iel]))
       vtufile.write("</DataArray>\n")
       #
       vtufile.write("</CellData>\n")
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='etls_eta_a' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" %(ls_eta_a[iel]+ls_eta_b[iel]*(x_V[icon_V[0,iel]]-xc[iel])+ls_eta_c[iel]*(y_V[icon_V[0,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_eta_a[iel]+ls_eta_b[iel]*(x_V[icon_V[1,iel]]-xc[iel])+ls_eta_c[iel]*(y_V[icon_V[1,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_eta_a[iel]+ls_eta_b[iel]*(x_V[icon_V[2,iel]]-xc[iel])+ls_eta_c[iel]*(y_V[icon_V[2,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_eta_a[iel]+ls_eta_b[iel]*(x_V[icon_V[3,iel]]-xc[iel])+ls_eta_c[iel]*(y_V[icon_V[3,iel]]-yc[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='etls_eta_a2' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" %(ls_eta_a[iel]+ls_eta_b2[iel]*(x_V[icon_V[0,iel]]-xc[iel])+ls_eta_c2[iel]*(y_V[icon_V[0,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_eta_a[iel]+ls_eta_b2[iel]*(x_V[icon_V[1,iel]]-xc[iel])+ls_eta_c2[iel]*(y_V[icon_V[1,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_eta_a[iel]+ls_eta_b2[iel]*(x_V[icon_V[2,iel]]-xc[iel])+ls_eta_c2[iel]*(y_V[icon_V[2,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_eta_a[iel]+ls_eta_b2[iel]*(x_V[icon_V[3,iel]]-xc[iel])+ls_eta_c2[iel]*(y_V[icon_V[3,iel]]-yc[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho_ls' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" %(ls_rho_a[iel]+ls_rho_b[iel]*(x_V[icon_V[0,iel]]-xc[iel])+ls_rho_c[iel]*(y_V[icon_V[0,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_rho_a[iel]+ls_rho_b[iel]*(x_V[icon_V[1,iel]]-xc[iel])+ls_rho_c[iel]*(y_V[icon_V[1,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_rho_a[iel]+ls_rho_b[iel]*(x_V[icon_V[2,iel]]-xc[iel])+ls_rho_c[iel]*(y_V[icon_V[2,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_rho_a[iel]+ls_rho_b[iel]*(x_V[icon_V[3,iel]]-xc[iel])+ls_rho_c[iel]*(y_V[icon_V[3,iel]]-yc[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho_ls2' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" %(ls_rho_a[iel]+ls_rho_b2[iel]*(x_V[icon_V[0,iel]]-xc[iel])+ls_rho_c2[iel]*(y_V[icon_V[0,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_rho_a[iel]+ls_rho_b2[iel]*(x_V[icon_V[1,iel]]-xc[iel])+ls_rho_c2[iel]*(y_V[icon_V[1,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_rho_a[iel]+ls_rho_b2[iel]*(x_V[icon_V[2,iel]]-xc[iel])+ls_rho_c2[iel]*(y_V[icon_V[2,iel]]-yc[iel]) ))
           vtufile.write("%e\n" %(ls_rho_a[iel]+ls_rho_b2[iel]*(x_V[icon_V[3,iel]]-xc[iel])+ls_rho_c2[iel]*(y_V[icon_V[3,iel]]-yc[iel]) ))
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
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % eyy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sr' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % e[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (eta_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (rho_elemental[iel]))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' Name='nparticle' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (nparticle_in_element[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(u[i]*year,v[i]*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %rhoV[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %etaV[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                          icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                          icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*9))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %28)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()


       filename = 'swarm_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nparticle,nparticle))
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%e \n" %swarm_mat[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%e \n" %swarm_paint[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='displacement' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%5e %5e %5e \n" %(swarm_x[i]-swarm_x0[i],swarm_y[i]-swarm_y0[i],0.))
       vtufile.write("</DataArray>\n")

       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%e %e %e \n" %(swarm_x[i],swarm_y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,nparticle):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,nparticle):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,nparticle):
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
       #
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%e  \n" %etaq[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%e  \n" %rhoq[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       #
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%e %e %e \n" %(xq[i],yq[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #
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

    print("write vtu files: %.3f s" % (clock.time()-start))

    Time+=dt

#end for istep

###################################################################################################
# end time stepping loop
###################################################################################################

print("*******************************")
print("********** the end ************")
print("*******************************")
