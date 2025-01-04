import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix
import time as timing
import random
from numba import jit

#------------------------------------------------------------------------------

def yLAB(x,NZ):
    return np.arctan((x-Lx/2)/NZ*4)/(np.pi/2)*75e3+475e3

#------------------------------------------------------------------------------

@jit(nopython=True)
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
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,\
                     NV_6,NV_7,NV_8],dtype=np.float64)

@jit(nopython=True)
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
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,\
                     dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

@jit(nopython=True)
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
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,\
                     dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

@jit(nopython=True)
def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

TKelvin=273.15
cm=0.01
year=3600.*24.*365.
eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
mT=9     # number of temperature nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

model=1

#...........
if model==1:
   Lx=1200e3  # horizontal extent of the domain 
   Ly=600e3   # vertical extent of the domain 
   NZ=30e3    # necking zone width
   nelx=150   # nb of elements in x direction
   nely=int(nelx*Ly/Lx)
   Tb=1400+TKelvin    # bottom temperature 
   Tt=0+TKelvin       # top temperature
   TH=0.01*1400  # amplitude temperature perturbation
   alpha=3.1e-5 # thermal expansion
   #material 0: lithosphere    eta=3e24, rho=3300
   #material 1: asthenosphere  eta=3e19, rho=3300
   nmat=2
   rho_mat=np.array([3300,3300],dtype=np.float64)
   eta_mat=np.array([3e26,3e21],dtype=np.float64) 
   rk=-1
   nparticle_per_dim=5
   particles_random=False
   eta_ref=1e21      # scaling of G blocks
   hcapa=1250
   hdiff=1e-6
   hcond=hdiff*3300*hcapa

#...........
   
nnx=2*nelx+1                  # number of V nodes, x direction
nny=2*nely+1                  # number of V nodes, y direction
NV=nnx*nny                    # total number of nodes
NP=(nelx+1)*(nely+1)          # total number of P nodes
nel=nelx*nely                 # number of elements, total
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
NfemT=NV                      # number of temperature dofs
hx=Lx/nelx                    # mesh spacing in x direction
hy=Ly/nely                    # mesh spacing in y direction

nstep=50

tfinal=10e6*year
dt_max=5e5*year

gx=0
gy=-9.8

CFL_nb=0.5

nq_per_dim=3                  # number of quad points per dimension
nq=nq_per_dim**2*nel          # number of quadrature points

nparticle_per_element=nparticle_per_dim**2
nparticle=nparticle_per_element*nel

#1: use elemental values for all q points
#2: use nodal values + Q1 shape functions to interp on q points
#3: use avrg nodal values to assign to all q points (elemental avrg)
#4: nodal rho, elemental eta
particle_projection=1

avrg=2 # type of viscosity averaging (1: arithm, 2: geom, 3: harm)

every=1 # how often vtu files are produced

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

#################################################################
#################################################################

Ranb=3300*alpha*abs(gy)*(Tb-Tt)*Ly**3/hdiff/eta_mat[1]

print("Lx",Lx)
print("Ly",Ly)
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("avrg=",avrg)
print("nparticle=",nparticle)
print("particle_projection=",particle_projection)
print("hcond=",hcond)
print("hcapa=",hcapa)
print("hdiff=",hdiff)
print("Ra nb=",Ranb)
print("------------------------------")

vel_file=open('vel_stats.ascii',"w")
vrms_file=open('vrms.ascii',"w")
nparticle_file=open('nparticle_per_element.ascii',"w")
dt_file=open('dt.ascii',"w")

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

#np.savetxt('grid_bef.ascii',np.array([xV,yV]).T,header='# x,y')

print("grid setup: %.3f s" % (timing.time() - start))

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
start = timing.time()

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

print("connectivity: %.3f s" % (timing.time() - start))

#################################################################
# compute coordinates of pressure nodes
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)  # x coordinates
yP=np.empty(NP,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xP[iconP[0,iel]]=xV[iconV[0,iel]] ; yP[iconP[0,iel]]=yV[iconV[0,iel]]
    xP[iconP[1,iel]]=xV[iconV[1,iel]] ; yP[iconP[1,iel]]=yV[iconV[1,iel]]
    xP[iconP[2,iel]]=xV[iconV[2,iel]] ; yP[iconP[2,iel]]=yV[iconV[2,iel]]
    xP[iconP[3,iel]]=xV[iconV[3,iel]] ; yP[iconP[3,iel]]=yV[iconV[3,iel]]
#end for

print("compute P nodes coords: %.3f s" % (timing.time() - start))

#################################################################
# particle setup
#################################################################
start = timing.time()

swarm_x=np.empty(nparticle,dtype=np.float64) 
swarm_y=np.empty(nparticle,dtype=np.float64) 
swarm_mat=np.empty(nparticle,dtype=np.int8)  
swarm_paint=np.empty(nparticle,dtype=np.float64) 
swarm_r=np.empty(nparticle,dtype=np.float64) 
swarm_s=np.empty(nparticle,dtype=np.float64) 
swarm_iel=np.empty(nparticle,dtype=np.int32) 

counter=0
for iel in range(0,nel):
    x1=xV[iconV[0,iel]] ; y1=yV[iconV[0,iel]]
    x2=xV[iconV[1,iel]] ; y2=yV[iconV[1,iel]]
    x3=xV[iconV[2,iel]] ; y3=yV[iconV[2,iel]]
    x4=xV[iconV[3,iel]] ; y4=yV[iconV[3,iel]]
    if particles_random:
       for j in range(0,nparticle_per_dim):
           for i in range(0,nparticle_per_dim):
               r=-1.+i*2./nparticle_per_dim + 1./nparticle_per_dim
               s=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
               r+=random.uniform(-1,+1)/nparticle_per_dim/4.
               s+=random.uniform(-1,+1)/nparticle_per_dim/4.
               swarm_r[counter]=r
               swarm_s[counter]=s
               N1=0.25*(1-r)*(1-s)
               N2=0.25*(1+r)*(1-s)
               N3=0.25*(1+r)*(1+s)
               N4=0.25*(1-r)*(1+s)
               swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
               swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
               swarm_iel[counter]=iel
               counter+=1
           #end for 
       #end for 
    else:
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
               swarm_iel[counter]=iel
               counter+=1
           #end for 
       #end for 
    #end if
#end for 

print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("particles setup: %.3f s" % (timing.time() - start))

#################################################################
# assign material id to particles  
#################################################################
start=timing.time()

if model==1:
   for im in range(0,nparticle):
       xi=swarm_x[im]
       yi=swarm_y[im]
       yL=yLAB(xi,NZ)
       if yi>yL:
          swarm_mat[im]=1 # lithosphere
       else:
          swarm_mat[im]=2 # asthenosphere
    #end for

print("particles material: %.3f s" % (timing.time() - start))

#np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y')
#exit()

#################################################################
# paint particles 
#################################################################
start=timing.time()

for im in range (0,nparticle):
    swarm_paint[im]=(np.sin(2*np.pi*swarm_x[im]/Lx*4)*\
                     np.sin(2*np.pi*swarm_y[im]/Ly*4))
#end for 

print("particles painting: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # free slip
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # free slip
#end for

bc_fixT=np.zeros(NfemT,dtype=bool)  # boundary condition, yes/no
bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if yV[i]/Ly<eps:
       bc_fixT[i] = True ; bc_valT[i]=Tb 
    if yV[i]/Ly>(1-eps):
       bc_fixT[i] = True ; bc_valT[i]=Tt 
#end for

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# layout interface markers
#################################################################
start = timing.time()

if model==1:
   nmarker=0
   #hh=1e3
   #nmarker=np.int(100e3/hh)
   #print('     -> nmarker=',nmarker)

   m_x=np.empty(nmarker,dtype=np.float64) 
   m_y=np.empty(nmarker,dtype=np.float64) 
   m_r=np.empty(nmarker,dtype=np.float64) 
   m_s=np.empty(nmarker,dtype=np.float64) 
   m_u=np.empty(nmarker,dtype=np.float64) 
   m_v=np.empty(nmarker,dtype=np.float64) 

   #counter=0
   #for i in range(0,nmarker1):
   #    m_x[counter]=200e3
   #    m_y[counter]=450e3-i*hh
   #    counter+=1

#np.savetxt('markers.ascii',np.array([m_x,m_y]).T,header='# x,y')

print("marker setup: %.3f s" % (timing.time() - start))

#################################################################
# initial temperature 
#################################################################
start = timing.time()

T=np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    yL=yLAB(xV[i],NZ)
    if yV[i]>yL:
       T[i]=Tb+(yV[i]-yL)/(Ly-yL)*(Tt-Tb)
    else:
       T[i]=Tb+TH*np.sin(np.pi*yV[i]/Ly)*np.cos(np.pi*xV[i]/2/Ly)

print("     T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

print("initial temperature: %.3f s" % (timing.time() - start))

#--------------------------------------------------------------------------------------------------
# time stepping loop
#--------------------------------------------------------------------------------------------------

Time=0.

for istep in range(0,nstep):

    print ('---------------------------------------')
    print ('-----------------istep= %i -------------' %istep)
    print ('-----------------Time= %f Myr ----------' %(Time/year/1e6))
    print ('---------------------------------------')

    #################################################################
    # all elements are rectangles of size hx,hy
    jcob=hx*hy/4
    jcbi=np.zeros((2,2),dtype=np.float64)
    jcbi[0,0]=2/hx
    jcbi[1,1]=2/hy

    #################################################################
    # locate particle
    #################################################################
    start = timing.time()

    for im in range(0,nparticle):
        ielx=int(swarm_x[im]/Lx*nelx)
        iely=int(swarm_y[im]/Ly*nely)
        swarm_iel[im]=nelx*(iely)+ielx
    #end for

    #np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_iel]).T,header='# x,y')
    #exit()

    print("locate particles: %.3f s" % (timing.time() - start))

    #################################################################
    # compute nodal averagings
    # with the tracer ratio method \cite{taki03}  
    #################################################################
    start = timing.time()

    mat_nodal=np.zeros((nmat,NP),dtype=np.float64) 
    mat_nodal_counter=np.zeros(NP,dtype=np.float64) 
    nparticle_in_element=np.zeros(nel,dtype=np.float64) 

    for im in range(0,nparticle):
        imat=swarm_mat[im]-1
        iel=swarm_iel[im]
        N1=0.25*(1-swarm_r[im])*(1-swarm_s[im])
        N2=0.25*(1+swarm_r[im])*(1-swarm_s[im])
        N3=0.25*(1+swarm_r[im])*(1+swarm_s[im])
        N4=0.25*(1-swarm_r[im])*(1+swarm_s[im])
        mat_nodal[imat,iconP[0,iel]]+=N1
        mat_nodal[imat,iconP[1,iel]]+=N2
        mat_nodal[imat,iconP[2,iel]]+=N3
        mat_nodal[imat,iconP[3,iel]]+=N4
        mat_nodal_counter[iconP[0,iel]]+=N1
        mat_nodal_counter[iconP[1,iel]]+=N2
        mat_nodal_counter[iconP[2,iel]]+=N3
        mat_nodal_counter[iconP[3,iel]]+=N4

        #if swarm_r[im]<0:
        #   if swarm_s[im]<0:
        #      mat_nodal[imat,iconP[0,iel]]+=1
        #      mat_nodal_counter[iconP[0,iel]]+=1
        #   else:
        #      mat_nodal[imat,iconP[3,iel]]+=1
        #      mat_nodal_counter[iconP[3,iel]]+=1
        #   #end if
        #else:
        #   if swarm_s[im]<0:
        #      mat_nodal[imat,iconP[1,iel]]+=1
        #      mat_nodal_counter[iconP[1,iel]]+=1
        #   else:
        #      mat_nodal[imat,iconP[2,iel]]+=1
        #      mat_nodal_counter[iconP[2,iel]]+=1
        #   #end if
        #end if
        nparticle_in_element[iel]+=1
    #end for
    mat_nodal/=mat_nodal_counter

    nparticle_file.write("%d %e %e\n" %(istep,np.min(nparticle_in_element),np.max(nparticle_in_element))) 
    nparticle_file.flush()

    if np.min(nparticle_in_element)==0:
       exit('no particle left in an element')

    #np.savetxt('mat_nodal0.ascii',np.array([xP,yP,mat_nodal[0,:]]).T)
    #np.savetxt('mat_nodal1.ascii',np.array([xP,yP,mat_nodal[1,:]]).T)
    #np.savetxt('mat_nodal2.ascii',np.array([xP,yP,mat_nodal[2,:]]).T)
    #np.savetxt('mat_nodal3.ascii',np.array([xP,yP,mat_nodal[3,:]]).T)
    #np.savetxt('mat_nodal4.ascii',np.array([xP,yP,mat_nodal[4,:]]).T)
    #np.savetxt('mat_nodal5.ascii',np.array([xP,yP,mat_nodal[5,:]]).T)
    #np.savetxt('mat_nodal6.ascii',np.array([xP,yP,mat_nodal[6,:]]).T)

    #np.savetxt('rho_nodal0.ascii',np.array([xP,yP,rho_nodal]).T,header='# x,y,rho')

    print("     -> nparticle_in_elt(m,M) %.5e %.5e " %(np.min(nparticle_in_element),np.max(nparticle_in_element)))
    print("     -> mat_nodal     (m,M) %.5e %.5e " %(np.min(mat_nodal),np.max(mat_nodal)))

    print("particles onto Q1 grid: %.3f s" % (timing.time() - start))

    ################################################################
    # compute rho and eta on P nodes
    #################################################################
    start = timing.time()

    rho_nodal=np.zeros(NP,dtype=np.float64) 
    eta_nodal=np.zeros(NP,dtype=np.float64) 

    for i in range(0,NP):
        for imat in range(0,nmat):
            rho_nodal[i]+=mat_nodal[imat,i]*rho_mat[imat]

    if avrg==1:
       for i in range(0,NP):
           for imat in range(0,nmat):
               eta_nodal[i]+=mat_nodal[imat,i]*eta_mat[imat]

    if avrg==2:
       for i in range(0,NP):
           for imat in range(0,nmat):
               eta_nodal[i]+=mat_nodal[imat,i]*np.log10(eta_mat[imat])
           eta_nodal[i]=10.**eta_nodal[i]

    if avrg==3:
       for i in range(0,NP):
           for imat in range(0,nmat):
               eta_nodal[i]+=mat_nodal[imat,i]/eta_mat[imat]
           eta_nodal[i]=1./eta_nodal[i]

    print("     -> rho_nodal     (m,M) %.5e %.5e " %(np.min(rho_nodal),np.max(rho_nodal)))
    print("     -> eta_nodal     (m,M) %.5e %.5e " %(np.min(eta_nodal),np.max(eta_nodal)))

    #np.savetxt('rho_nodal.ascii',np.array([xP,yP,rho_nodal]).T,header='# x,y,rho')
    #np.savetxt('eta_nodal.ascii',np.array([xP,yP,eta_nodal]).T,header='# x,y,eta')

    print("compute rho,eta on P nodes: %.3f s" % (timing.time() - start))

    ################################################################
    # compute elemental rho and eta 
    #################################################################
    start = timing.time()
    
    rho_elemental=np.zeros(nel,dtype=np.float64) 
    eta_elemental=np.zeros(nel,dtype=np.float64) 
    c=np.zeros(nel,dtype=np.float64)

    for im in range(0,nparticle):
        iel=swarm_iel[im]
        imat=swarm_mat[im]-1
        rho_elemental[iel]+=rho_mat[imat]
        c[iel]+=1
    rho_elemental/=c

    c=np.zeros(nel,dtype=np.float64)

    if avrg==1:
       for im in range(0,nparticle):
           iel=swarm_iel[im]
           imat=swarm_mat[im]-1
           eta_elemental[iel]+=eta_mat[imat]
           c[iel]+=1
       eta_elemental/=c

    if avrg==2:
       for im in range(0,nparticle):
           iel=swarm_iel[im]
           imat=swarm_mat[im]-1
           eta_elemental[iel]+=np.log10(eta_mat[imat])
           c[iel]+=1
       eta_elemental/=c
       eta_elemental=10.**eta_elemental

    if avrg==3:
       for im in range(0,nparticle):
           iel=swarm_iel[im]
           imat=swarm_mat[im]-1
           eta_elemental[iel]+=1./eta_mat[imat]
           c[iel]+=1
       eta_elemental=c/eta_elemental

    print("     -> rho_elemental (m,M) %.5e %.5e " %(np.min(rho_elemental),np.max(rho_elemental)))
    print("     -> eta_elemental (m,M) %.5e %.5e " %(np.min(eta_elemental),np.max(eta_elemental)))

    print("compute elemental rho,eta: %.3f s" % (timing.time() - start))

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = timing.time()

    A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
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
    Tq   = np.zeros(nq,dtype=np.float64)             # 
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
        # integrate viscous term at quadrature points
        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)
                NNNP[0:4]=NNP(rq,sq)
                # calculate jacobian matrix
                #jcb=np.zeros((2,2),dtype=np.float64)
                #for k in range(0,mV):
                #    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                #    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                #    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                #    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
                #jcob = np.linalg.det(jcb)
                #jcbi = np.linalg.inv(jcb)
                # compute dNdx & dNdy
                xq[counter]=np.dot(NNNV[0:mV],xV[iconV[0:mV,iel]])
                yq[counter]=np.dot(NNNV[0:mV],yV[iconV[0:mV,iel]])
                Tq[counter]=np.dot(NNNV[0:mV],T[iconV[0:mV,iel]])
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for

                if particle_projection==1:
                   rhoq[counter]=rho_elemental[iel]
                   etaq[counter]=eta_elemental[iel]
                elif particle_projection==2:
                   rhoq[counter]=np.dot(NNNP[0:mP],rho_nodal[iconP[0:mP,iel]])
                   etaq[counter]=np.dot(NNNP[0:mP],eta_nodal[iconP[0:mP,iel]])
                elif particle_projection==3:
                   rhoq[counter]=np.sum(rho_nodal[iconP[0:mP,iel]])*0.25
                   etaq[counter]=np.sum(eta_nodal[iconP[0:mP,iel]])*0.25
                elif particle_projection==4:
                   rhoq[counter]=np.dot(NNNP[0:mP],rho_nodal[iconP[0:mP,iel]])
                   etaq[counter]=eta_elemental[iel]
                #end if

                rhoq[counter]*=(1-alpha*Tq[counter])

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.        ],
                                             [0.        ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counter]*weightq*jcob

                for i in range(0,mV):
                    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*rhoq[counter]
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rhoq[counter]
                #end for

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                #end for

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob
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

        G_el*=eta_ref/Ly
        h_el*=eta_ref/Ly

        # assemble matrix and rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        A_sparse[m1,m2] += K_el[ikk,jkk]
                    #end for
                #end for
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                    A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                #end for
                f_rhs[m1]+=f_el[ikk]
            #end for
        #end for

        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]
        #end for

    #end for

    print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
    print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

    #np.savetxt('rhoq.ascii',np.array([xq,yq,rhoq]).T,header='# x,y,rho')
    #np.savetxt('etaq.ascii',np.array([xq,yq,etaq]).T,header='# x,y,eta')

    print("build FE matrix: %.3f s" % (timing.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = timing.time()

    rhs=np.zeros(Nfem,dtype=np.float64) 
    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("assemble rhs: %.3f s" % (timing.time() - start))

    ######################################################################
    # convert to csr
    ######################################################################
    start = timing.time()

    sparse_matrix=A_sparse.tocsr()

    print("convert from lil to csr format: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve system: %.3f s" % (timing.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = timing.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*(eta_ref/Ly)

    print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))
    
    vel_file.write("%e %e %e %e %e\n" %(Time,np.min(u/cm*year),np.max(u/cm*year),\
                                             np.min(v/cm*year),np.max(v/cm*year))) 
    vel_file.flush()

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute timestep 
    ######################################################################
    start = timing.time()

    dt=CFL_nb*min(hx,hy)/max(max(abs(u)),max(abs(v)))

    dt=min(dt_max,dt)

    print("     -> dt= %.3e yr" %(dt/year))

    dt_file.write("%d %e \n" %(istep,dt)) ; dt_file.flush()

    print("compute timestep: %.3f s" % (timing.time()-start))

    #################################################################
    # build FE system for energy equation
    #################################################################
    start = timing.time()

    A_mat=np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs=np.zeros(NfemT,dtype=np.float64)           # FE rhs 
    B_mat=np.zeros((2,mT),dtype=np.float64)        # gradient matrix B 
    N_mat=np.zeros((mT,1),dtype=np.float64)        # shape functions
    Tvect=np.zeros(mT,dtype=np.float64)          

    for iel in range (0,nel):

        b_el=np.zeros(mT,dtype=np.float64)
        a_el=np.zeros((mT,mT),dtype=np.float64)
        Ka=np.zeros((mT,mT),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((mT,mT),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,2),dtype=np.float64)

        for k in range(0,mT):
            Tvect[k]=T[iconV[k,iel]]
        #end for

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_mat[0:mT,0]=NNV(rq,sq)
                dNNNVdr[0:mT]=dNNVdr(rq,sq)
                dNNNVds[0:mT]=dNNVds(rq,sq)

                # calculate jacobian matrix
                #jcb=np.zeros((2,2),dtype=np.float64)
                #for k in range(0,mT):
                #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #end for
                #jcob = np.linalg.det(jcb)
                #jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,mT):
                    vel[0,0]+=N_mat[k,0]*u[iconV[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                    B_mat[0,k]=dNNNVdx[k]
                    B_mat[1,k]=dNNNVdy[k]
                #end for

                # compute mass matrix
                MM+=N_mat.dot(N_mat.T)*rho_elemental[iel]*hcapa*weightq*jcob

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka+=N_mat.dot(vel.dot(B_mat))*rho_elemental[iel]*hcapa*weightq*jcob

            #end for
        #end for

        a_el=MM+(Ka+Kd)*dt*0.5
        b_el=(MM-(Ka+Kd)*dt*0.5).dot(Tvect)

        # apply boundary conditions
        for k1 in range(0,mT):
            m1=iconV[k1,iel]
            if bc_fixT[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mT):
                   m2=iconV[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               #end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            #end for
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mT):
            m1=iconV[k1,iel]
            for k2 in range(0,mT):
                m2=iconV[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    #end for iel

    print("build FE matrix : %.3f s" % (timing.time() - start))

    #################################################################
    # solve linear system 
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     T (m,M) %.4f %.4f " %(np.min(T)-TKelvin,np.max(T)-TKelvin))

    print("solve system: %.3f s" % (timing.time() - start))

    #################################################################
    # compute vrms 
    #################################################################
    start = timing.time()

    vrms=0.
    mass=0.
    counterq=0
    for iel in range (0,nel):
        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)
                # calculate jacobian matrix
                #jcb=np.zeros((2,2),dtype=np.float64)
                #for k in range(0,mV):
                #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #end for
                #jcob = np.linalg.det(jcb)
                #jcbi = np.linalg.inv(jcb)
                # compute dNdx & dNdy
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for
                uq=np.dot(NNNV[0:mV],u[iconV[0:mV,iel]])
                vq=np.dot(NNNV[0:mV],v[iconV[0:mV,iel]])
                vrms+=(uq**2+vq**2)*weightq*jcob
                mass+=rhoq[counterq]*weightq*jcob
                counterq+=1
            #end for
        #end for
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))

    vrms_file.write("%e %e %e %e\n" %(Time,vrms,Time/year,vrms/cm*year)) ; vrms_file.flush()

    print("     -> vrms %.5e " %vrms)
    print("     -> mass %.5e " %mass)

    print("compute vrms & mass: %.3f s" % (timing.time() - start))

    ######################################################################
    # advect particles 
    ######################################################################
    start = timing.time()

    if rk==1:
       for im in range(0,nparticle):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           #-----
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           swarm_r[im]=-1+2*(swarm_x[im]-x0)/hx
           swarm_s[im]=-1+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
       #end for
       #np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_iel,swarm_r,swarm_s]).T)
       #exit()

    elif rk==2:
       for im in range(0,nparticle):
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
           xm=swarm_x[im]+um*dt/2
           ym=swarm_y[im]+vm*dt/2

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
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
       #end for

    elif rk==3:
       for im in range(0,nparticle):
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
           xB=swarm_x[im]+uA*dt/2
           yB=swarm_y[im]+vA*dt/2

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
           xC=swarm_x[im]+(2*uB-uA)*dt/2
           yC=swarm_y[im]+(2*vB-vA)*dt/2

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
           swarm_x[im]+=(uA+4*uB+uC)*dt/6
           swarm_y[im]+=(vA+4*vB+vC)*dt/6
       #end for
    elif rk==-1:
       print('     no particle advection')
    #end if

    print("advect particles: %.3f s" % (timing.time() - start))

    ######################################################################
    # advect markers
    ######################################################################
    start = timing.time()

    if nmarker>0:
       if rk==1:
          for im in range(0,nmarker):
              ielx=int(m_x[im]/Lx*nelx)
              iely=int(m_y[im]/Ly*nely)
              iel=nelx*(iely)+ielx
              #print (ielx,iely,iel)
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              rm=-1.+2*(m_x[im]-x0)/hx
              sm=-1.+2*(m_y[im]-y0)/hy
              m_r[im]=rm
              m_s[im]=sm
              NNNV[0:mV]=NNV(rm,sm)
              m_u[im]=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              m_v[im]=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              m_x[im]+=m_u[im]*dt
              m_y[im]+=m_v[im]*dt
          #end for

       elif rk==2:
          for im in range(0,nmarker):
              ielx=int(m_x[im]/Lx*nelx)
              iely=int(m_y[im]/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              rm=-1.+2*(m_x[im]-x0)/hx
              sm=-1.+2*(m_y[im]-y0)/hy
              NNNV[0:mV]=NNV(rm,sm)
              um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              xm=m_x[im]+um*dt/2
              ym=m_y[im]+vm*dt/2

              ielx=int(xm/Lx*nelx)
              iely=int(ym/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              m_r[im]=-1+2*(xm-x0)/hx
              m_s[im]=-1+2*(ym-y0)/hy
              NNNV[0:mV]=NNV(m_r[im],m_s[im])
              m_u[im]=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              m_v[im]=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              m_x[im]+=m_u[im]*dt
              m_y[im]+=m_v[im]*dt
          #end for

       elif rk==3:
          for im in range(0,nmarker):
              ielx=int(m_x[im]/Lx*nelx)
              iely=int(m_y[im]/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              rm=-1.+2*(m_x[im]-x0)/hx
              sm=-1.+2*(m_y[im]-y0)/hy
              NNNV[0:mV]=NNV(rm,sm)
              uA=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              vA=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              xB=m_x[im]+uA*dt/2
              yB=m_y[im]+vA*dt/2

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
              xC=m_x[im]+(2*uB-uA)*dt/2
              yC=m_y[im]+(2*vB-vA)*dt/2

              ielx=int(xC/Lx*nelx)
              iely=int(yC/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              m_r[im]=-1+2*(xC-x0)/hx
              m_s[im]=-1+2*(yC-y0)/hy
              NNNV[0:mV]=NNV(m_r[im],m_s[im])
              uC=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              vC=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              m_u[im]=uC
              m_v[im]=vC
              m_x[im]+=(uA+4*uB+uC)*dt/6
              m_y[im]+=(vA+4*vB+vC)*dt/6
          #end for
       elif rk==-1:
           print('     no marker advection')
       #end if
       filename = 'markers_{:04d}.ascii'.format(istep) 
       np.savetxt(filename,np.array([m_x,m_y,m_u,m_v]).T,header='# x,y')
    else:
       print('     no marker in domain')
    #end if

    print("advect markers: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute nodal strainrate and heat flux 
    #
    # 3--6--2
    # |  |  |
    # 7--8--5
    # |  |  |
    # 0--4--1
    #####################################################################
    start = timing.time()

    rVnodes=[-1,+1,+1,-1,0,+1,0,-1,0]
    sVnodes=[-1,-1,+1,+1,-1,0,+1,0,0]
    
    exx_n=np.zeros(NV,dtype=np.float64)  
    eyy_n=np.zeros(NV,dtype=np.float64)  
    exy_n=np.zeros(NV,dtype=np.float64)  
    count=np.zeros(NV,dtype=np.int32)  
    q=np.zeros(NV,dtype=np.float64)
    rho_Q2=np.zeros(NV,dtype=np.float64)
    eta_Q2=np.zeros(NV,dtype=np.float64)
    mats_Q2=np.zeros((nmat,NV),dtype=np.float64)
    c=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,mV):
            rq=rVnodes[i]
            sq=sVnodes[i]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)
            #jcb=np.zeros((2,2),dtype=np.float64)
            #for k in range(0,mV):
            #    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            #    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            #    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            #    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            #jcbi=np.linalg.inv(jcb)
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for
            inode=iconV[i,iel]
            exx_n[inode]+=np.dot(dNNNVdx[0:mV],u[iconV[0:mV,iel]])
            eyy_n[inode]+=np.dot(dNNNVdy[0:mV],v[iconV[0:mV,iel]])
            exy_n[inode]+=np.dot(dNNNVdy[0:mV],u[iconV[0:mV,iel]])*0.5+\
                          np.dot(dNNNVdx[0:mV],v[iconV[0:mV,iel]])*0.5
            q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
            rho_Q2[inode]+=np.dot(rho_nodal[iconP[0:mP,iel]],NNNP[0:mP])
            eta_Q2[inode]+=np.dot(eta_nodal[iconP[0:mP,iel]],NNNP[0:mP])
            for imat in range(0,nmat):
                mats_Q2[imat,inode]+=np.dot(mat_nodal[imat,iconP[0:mP,iel]],NNNP[0:mP])
            count[inode]+=1
        #end for
    #end for
    
    q/=count
    exx_n/=count
    eyy_n/=count
    exy_n/=count
    rho_Q2/=count
    eta_Q2/=count
    for imat in range(0,nmat):
        mats_Q2[imat,:]/=count

    print("     -> exx nodal (m,M) %e %e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy nodal (m,M) %e %e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy nodal (m,M) %e %e " %(np.min(exy_n),np.max(exy_n)))
    print("     -> rho nodal (m,M) %e %e " %(np.min(rho_Q2),np.max(rho_Q2)))
    print("     -> eta nodal (m,M) %e %e " %(np.min(eta_Q2),np.max(eta_Q2)))
    print("     -> press nodal (m,M) %e %e " %(np.min(q),np.max(q)))

    #np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')
    #np.savetxt('rho_Q2.ascii',np.array([x,y,rho_Q2]).T,header='# x,y,q')
    #np.savetxt('strainrate.ascii',np.array([xV,yV,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute nodal press & sr: %.3f s" % (timing.time() - start))

    #####################################################################
    # export solution to vtu files
    #####################################################################
    start = timing.time()
    
    if istep%every==0:

       filename = 'mesh_rho_eta_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e %10e %10e \n" %(xP[i],yP[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       if particle_projection > 1:
          vtufile.write("<PointData Scalars='scalars'>\n")
          vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
          for i in range(0,NP):
              vtufile.write("%10e \n" %rho_nodal[i])
          vtufile.write("</DataArray>\n")
          vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='nparticle' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (nparticle_in_element[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (eta_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       if particle_projection == 1:
          vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%10e\n" % (rho_elemental[iel]))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconP[0,iel],iconP[1,iel],iconP[2,iel],iconP[3,iel]))
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



       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (C)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %(T[i]-TKelvin))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (adim)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %((T[i]-TKelvin)/(Tb-Tt)))
       vtufile.write("</DataArray>\n")
       #--
       if particle_projection > 1:
          vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
          for i in range(0,NV):
              vtufile.write("%10e \n" %rho_Q2[i])
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
          for i in range(0,NV):
              vtufile.write("%10e \n" %eta_Q2[i])
          vtufile.write("</DataArray>\n")
          #--
       for imat in range(0,nmat):
           vtufile.write("<DataArray type='Float32' Name='mat %d' Format='ascii'> \n" %imat)
           for i in range(0,NV):
               vtufile.write("%10e \n" %mats_Q2[imat,i])
           vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
       for i in range(0,NV):
           e=np.sqrt(0.5*(exx_n[i]*exx_n[i]+eyy_n[i]*eyy_n[i])+exy_n[i]*exy_n[i])
           vtufile.write("%10e \n" %e)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                          iconV[3,iel],iconV[4,iel],iconV[5,iel],\
                                                          iconV[6,iel],iconV[7,iel],iconV[8,iel]))
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



       filename = 'particles_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nparticle,nparticle))
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%3e \n" %swarm_mat[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%3e \n" %swarm_paint[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
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

       if nmarker>0:
          filename = 'markers_{:04d}.vtu'.format(istep) 
          vtufile=open(filename,"w")
          vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
          vtufile.write("<UnstructuredGrid> \n")
          vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker-1))
          vtufile.write("<Points> \n")
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
          for i in range(0,nmarker):
              vtufile.write("%.10e %.10e %.10e \n" %(m_x[i],m_y[i],0.))
          vtufile.write("</DataArray>\n")
          vtufile.write("</Points> \n")
          vtufile.write("<PointData Scalars='scalars'>\n")
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
          for i in range(0,nmarker):
              vtufile.write("%10e %10e %10e \n" %(m_u[i],m_v[i],0.))
          vtufile.write("</DataArray>\n")
          vtufile.write("</PointData>\n")
          vtufile.write("<Cells>\n")
          vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
          for i in range(0,nmarker-1):
              vtufile.write("%d %d " % (i,i+1))
          vtufile.write("</DataArray>\n")
          vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
          for i in range(0,nmarker):
              vtufile.write("%d " % ((i+1)*2))
          vtufile.write("</DataArray>\n")
          vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
          for i in range(0,nmarker):
              vtufile.write("%d " % 3)
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

       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%3e \n" %Tq[i])
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%3e \n" %rhoq[i])
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%3e \n" %etaq[i])
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

    print("write vtu files: %.3f s" % (timing.time() - start))

    Time+=dt

    if Time>=tfinal:
       print('************************************')
       print('***** tfinal reached ')
       print('***** Time= %f Myr ' %(Time/year/1e6))
       print('************************************')
       break

#end for istep

#--------------------------------------------------------------------------------------------------
# end time stepping loop
#--------------------------------------------------------------------------------------------------

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
