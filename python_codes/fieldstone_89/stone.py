import numpy as np
import sys as sys
import time as timing
from scipy import special

#------------------------------------------------------------------------------

def compute_area_of_triangle(x1,y1,x2,y2,x3,y3):
    ABx=x2-x1
    ABy=y2-y1
    ACx=x3-x1
    ACy=y3-y1
    nz=ABx*ACy-ABy*ACx
    norm=abs(nz)
    area=0.5*norm
    return area

#------------------------------------------------------------------------------

def NQ1(r,s):
    N_0=0.25*(1.-r)*(1.-s)
    N_1=0.25*(1.+r)*(1.-s)
    N_2=0.25*(1.+r)*(1.+s)
    N_3=0.25*(1.-r)*(1.+s)
    return N_0,N_1,N_2,N_3

def dNQ1dr(r,s):
    dNdr_0=-0.25*(1.-s) 
    dNdr_1=+0.25*(1.-s) 
    dNdr_2=+0.25*(1.+s) 
    dNdr_3=-0.25*(1.+s) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNQ1ds(r,s):
    dNds_0=-0.25*(1.-r)
    dNds_1=-0.25*(1.+r)
    dNds_2=+0.25*(1.+r)
    dNds_3=+0.25*(1.-r)
    return dNds_0,dNds_1,dNds_2,dNds_3

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

#------------------------------------------------------------------------------

def interpolate_fields_on_pt(xm,ym):
    ielx=int(xm/Lx*nelx)
    iely=int(ym/Ly*nely)
    iel=nelx*(iely)+ielx
    xmin=xV[iconV[0,iel]] ; xmax=xV[iconV[2,iel]]
    ymin=yV[iconV[0,iel]] ; ymax=yV[iconV[2,iel]]
    rm=((xm-xmin)/(xmax-xmin)-0.5)*2
    sm=((ym-ymin)/(ymax-ymin)-0.5)*2
    NNNV[0:mV]=NNV(rm,sm)
    um=0.
    vm=0.
    exxm=0
    eyym=0
    exym=0
    for k in range(0,mV):
        um+=NNNV[k]*u[iconV[k,iel]]
        vm+=NNNV[k]*v[iconV[k,iel]]
        exxm+=NNNV[k]*exx_n[iconV[k,iel]]
        eyym+=NNNV[k]*eyy_n[iconV[k,iel]]
        exym+=NNNV[k]*exy_n[iconV[k,iel]]
    return um,vm,rm,sm,iel,exxm,eyym,exym 

#------------------------------------------------------------------------------

cm=0.01
eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1     # x extent of domain
Ly=1     # y extent of domain
nelx=20  # number of elements in x direction
nely=20  # number of elements in y direction 

nstep=151 # number of time steps

CFL_nb=0.1 # CFL number

every=10 # how often vtu/ascii output is generated

#1: shear band
#2: vertical extension
#3: pure shear
#4: biaxial extension 

experiment=3

oldfile=open('old.ascii',"w")
newfile=open('new.ascii',"w")
areafile=open('area.ascii',"w")

#################################################################

rVnodes=[-1,+1,1,-1, 0,1,0,-1,0]
sVnodes=[-1,-1,1,+1,-1,0,1, 0,0]

nnx=2*nelx+1                  # number of nodes, x direction
nny=2*nely+1                  # number of nodes, y direction
NV=nnx*nny                    # total number of nodes
nel=nelx*nely                 # total number of elements
hx=Lx/nelx                    # mesh size in x direction
hy=Ly/nely                    # mesh size in y direction

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("hx",hx)
print("hy",hy)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# build connectivity arrays for velocity (Q2 element) 
#################################################################
# velocity   
# 3---6---2  
# |       |  
# 7   8   5  
# |       |  
# 0---4---1  
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

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

print("setup: connectivity: %.3f s" % (timing.time() - start))

#################################################################
# define velocity field 
#################################################################
start = timing.time()

u =np.zeros(NV,dtype=np.float64)    # x-component velocity
v =np.zeros(NV,dtype=np.float64)    # y-component velocity

if experiment==1:
   for i in range(0,NV):
       u[i]= 0 
       v[i]=special.erf((xV[i]-Lx/2)*5)

if experiment==2:
   for i in range(0,NV):
       u[i]=0 
       v[i]=yV[i]

if experiment==3:
   for i in range(0,NV):
       u[i]=yV[i]
       v[i]=xV[i]

if experiment==4:
   for i in range(0,NV):
       u[i]=-xV[i]+Lx/2
       v[i]= yV[i]-Ly/2

#####################################################################
# compute nodal strainrate
#####################################################################
start = timing.time()
    
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
exx_n = np.zeros(NV,dtype=np.float64)  
eyy_n = np.zeros(NV,dtype=np.float64)  
exy_n = np.zeros(NV,dtype=np.float64)  
count = np.zeros(NV,dtype=np.int32)  

for iel in range(0,nel):
    for i in range(0,mV):
        rq=rVnodes[i]
        sq=sVnodes[i]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
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
        e_xx=0.
        e_yy=0.
        e_xy=0.
        for k in range(0,mV):
            e_xx += dNNNVdx[k]*u[iconV[k,iel]]
            e_yy += dNNNVdy[k]*v[iconV[k,iel]]
            e_xy += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
        #end for
        inode=iconV[i,iel]
        exx_n[inode]+=e_xx
        eyy_n[inode]+=e_yy
        exy_n[inode]+=e_xy
        count[inode]+=1
    #end for
#end for
    
exx_n/=count
eyy_n/=count
exy_n/=count

print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))

print("compute press & sr: %.3f s" % (timing.time() - start))

#################################################################
# marker setup. The cloud of markers is called a swarm. 
# the swarm is composed of swarm_nelx X swarm_nely cells.
#################################################################
start = timing.time()

swarm_nelx=2*nelx
swarm_nely=2*nely
swarm_nel=swarm_nelx*swarm_nely

swarm_nnx=swarm_nelx+1
swarm_nny=swarm_nely+1
swarm_nmarker=swarm_nnx*swarm_nny

swarm_hx=Lx/swarm_nnx
swarm_hy=Ly/swarm_nny

swarm_x=np.empty(swarm_nmarker,dtype=np.float64)  
swarm_y=np.empty(swarm_nmarker,dtype=np.float64)  
swarm_u=np.zeros(swarm_nmarker,dtype=np.float64)  
swarm_v=np.zeros(swarm_nmarker,dtype=np.float64)  
swarm_active=np.zeros(swarm_nmarker,dtype=np.bool) 
swarm_icon=np.zeros((4,swarm_nel),dtype=np.int32)
swarm_active_cell=np.zeros(swarm_nel,dtype=np.bool) 
swarm_exx_cell=np.zeros(swarm_nel,dtype=np.float64) 
swarm_eyy_cell=np.zeros(swarm_nel,dtype=np.float64) 
swarm_exy_cell=np.zeros(swarm_nel,dtype=np.float64) 
swarm_xc=np.zeros(swarm_nel,dtype=np.float64)  
swarm_yc=np.zeros(swarm_nel,dtype=np.float64)  
swarm_x0=np.empty(swarm_nmarker,dtype=np.float64)  
swarm_y0=np.empty(swarm_nmarker,dtype=np.float64)  

counter = 0
for j in range(0, swarm_nny):
    for i in range(0, swarm_nnx):
        swarm_x[counter]=(i+0.5)*swarm_hx
        swarm_y[counter]=(j+0.5)*swarm_hy
        counter += 1
    #end for
#end for

counter = 0
for j in range(0, swarm_nely):
    for i in range(0, swarm_nelx):
        swarm_icon[0,counter] = i + j * (swarm_nelx + 1)
        swarm_icon[1,counter] = i + 1 + j * (swarm_nelx + 1)
        swarm_icon[2,counter] = i + 1 + (j + 1) * (swarm_nelx + 1)
        swarm_icon[3,counter] = i + (j + 1) * (swarm_nelx + 1)
        counter += 1
    #end for
#end for

swarm_active[:]=True
swarm_active_cell[:]=True

swarm_x0[:]=swarm_x
swarm_y0[:]=swarm_y

print("     -> nmarker %d " % swarm_nmarker)
print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

if experiment==1:
   target_cell=int(swarm_nel/2+swarm_nelx/2) #; print(target_cell) 
if experiment==2:
   target_cell=int(swarm_nel/4+swarm_nelx/2) #; print(target_cell) 
if experiment==3:
   target_cell=int(swarm_nel/2+swarm_nelx/2) #; print(target_cell) 
if experiment==4:
   target_cell=int(swarm_nel/2+swarm_nelx/2) #; print(target_cell) 


print("markers setup: %.3f s" % (timing.time() - start))

#################################################################
# marker paint
#################################################################
start = timing.time()

swarm_mat=np.zeros(swarm_nmarker,dtype=np.int32)  

for i in [0,2]:
    dx=Lx/4
    for im in range (0,swarm_nmarker):
        if swarm_x[im]>i*dx and swarm_x[im]<(i+1)*dx:
           swarm_mat[im]+=1
    #end for
#end for

for i in [0,2,4,6]:
    dy=Ly/8
    for im in range (0,swarm_nmarker):
        if swarm_y[im]>i*dy and swarm_y[im]<(i+1)*dy:
           swarm_mat[im]+=1
    #end for
#end for

rad=Lx*0.16667
xcirc=0.5*Lx
ycirc=0.5*Ly
for im in range (0,swarm_nmarker):
    if (swarm_x[im]-xcirc)**2+(swarm_y[im]-ycirc)**2<(rad)**2:
           swarm_mat[im]=10
    if abs(swarm_x[im]-xcirc)<rad and abs(swarm_y[im]-ycirc)<rad/25:
           swarm_mat[im]=20
    if abs(swarm_x[im]-xcirc)<rad/25 and abs(swarm_y[im]-ycirc)<rad:
           swarm_mat[im]=20
    #end for
#end for

print("markers paint: %.3f s" % (timing.time() - start))

#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    dt=CFL_nb*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))

    print('dt=',dt)

    #####################################################################
    # advect markers 
    # Markers are advected by means of a simple Euler step and those
    # advected outside of the domain are labelled inactive and their 
    # strain set to zero. 
    #####################################################################
    start = timing.time()

    for im in range(0,swarm_nmarker):
           if swarm_active[im]:
              swarm_u[im],swarm_v[im],rm,sm,iel,exx_m,eyy_m,exy_m=\
                    interpolate_fields_on_pt(swarm_x[im],swarm_y[im])
              swarm_x[im]+=swarm_u[im]*dt
              swarm_y[im]+=swarm_v[im]*dt
              if swarm_x[im]<0 or swarm_x[im]>Lx or swarm_y[im]<0 or swarm_y[im]>Ly:
                 swarm_active[im]=False
           # end if active
    # end for im

    print("advect markers: %.3f s" % (timing.time() - start))

    #####################################################################
    # update strain inside swarm cells
    # If a cell has an inactive marker, it is also labelled inactive, 
    # and its strain is set to zero.
    #####################################################################

    for iel in range(0,swarm_nel):
        if not swarm_active[swarm_icon[0,iel]]:
           swarm_active_cell[iel]=False
        if not swarm_active[swarm_icon[1,iel]]:
           swarm_active_cell[iel]=False
        if not swarm_active[swarm_icon[2,iel]]:
           swarm_active_cell[iel]=False
        if not swarm_active[swarm_icon[3,iel]]:
           swarm_active_cell[iel]=False
        #end if
    #end for

    #print (swarm_nel,np.count_nonzero(swarm_active_cell))    

    for iel in range(0,swarm_nel):
        if swarm_active_cell[iel]:
           swarm_xc[iel]=0.25*np.sum(swarm_x[swarm_icon[:,iel]])
           swarm_yc[iel]=0.25*np.sum(swarm_y[swarm_icon[:,iel]])
           tempx,tempy,rm,sm,ijk,exx_m,eyy_m,exy_m=\
                       interpolate_fields_on_pt(swarm_xc[iel],swarm_yc[iel])
           swarm_exx_cell[iel]+=exx_m*dt
           swarm_eyy_cell[iel]+=eyy_m*dt
           swarm_exy_cell[iel]+=exy_m*dt
        else:
           swarm_exx_cell[iel]=0
           swarm_eyy_cell[iel]=0
           swarm_exy_cell[iel]=0
        #end if
    #end for

    print("     -> old: swarm_exx_cell (m,M) %.5e %.5e " %(np.min(swarm_exx_cell),np.max(swarm_exx_cell)))
    print("     -> old: swarm_eyy_cell (m,M) %.5e %.5e " %(np.min(swarm_eyy_cell),np.max(swarm_eyy_cell)))
    print("     -> old: swarm_exy_cell (m,M) %.5e %.5e " %(np.min(swarm_exy_cell),np.max(swarm_exy_cell)))

    #####################################################################
    # compute eigen values of integrated strain tensor
    # later well lambda 1,2 to be compared with these
    # We use here arctan2, which returns angles in rads in [-pi,pi] range
    #####################################################################
    swarm_princp_e1=np.zeros(swarm_nel,dtype=np.float64)
    swarm_princp_e2=np.zeros(swarm_nel,dtype=np.float64)
    swarm_old_angle=np.zeros(swarm_nel,dtype=np.float64)
    
    for iel in range(0,swarm_nel):
        if swarm_active_cell[iel]:
            swarm_princp_e1[iel] = 0.5*(swarm_exx_cell[iel]+swarm_eyy_cell[iel]) + \
                   np.sqrt(swarm_exy_cell[iel]**2 + 0.25*(swarm_exx_cell[iel]-swarm_eyy_cell[iel])**2) 
            swarm_princp_e2[iel] = 0.5*(swarm_exx_cell[iel]+swarm_eyy_cell[iel]) - \
                   np.sqrt(swarm_exy_cell[iel]**2 + 0.25*(swarm_exx_cell[iel]-swarm_eyy_cell[iel])**2)

            swarm_old_angle[iel]=np.arctan2(2*swarm_exy_cell[iel],\
                                 (swarm_exx_cell[iel]-swarm_eyy_cell[iel]))/2. /np.pi*180
    
    print("     -> old: swarm_exx_cell[target_cell] %.8e " % swarm_exx_cell[target_cell] )
    print("     -> old: swarm_eyy_cell[target_cell] %.8e " % swarm_eyy_cell[target_cell] )
    print("     -> old: swarm_exy_cell[target_cell] %.8e " % swarm_exy_cell[target_cell] )
    print("     -> old: swarm_princp_e1[target_cell] %.8e " % swarm_princp_e1[target_cell] )
    print("     -> old: swarm_princp_e2[target_cell] %.8e " % swarm_princp_e2[target_cell] )
    print("     -> old: swarm_old_angle[target_cell] %.8e " % swarm_old_angle[target_cell] )

    oldfile.write("%d %e %e %e %e\n" %(istep,\
                                       swarm_old_angle[target_cell],\
                                       swarm_princp_e1[target_cell],\
                                       swarm_princp_e2[target_cell],\
                                       (swarm_princp_e1[target_cell]+1)*(swarm_princp_e2[target_cell]+1)))


    #####################################################################
    # computing deformation gradient tensor F for each cell of the swarm
    # F = I + (x(t)-x0,y(t)-y0) (1/d_X 1/d_Y)^T evaluated at center of cell
    # this tensor is not symmetric
    #####################################################################
    # note that because of so far obscure reasons, the deformation 
    # tensor in the literature is stored as the transpose of what I compute
    # we follow here the conventions of this website: 
    # https://www.continuummechanics.org/deformationgradient.html
    #####################################################################

    swarm_Fxx=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Fxy=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Fyx=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Fyy=np.zeros(swarm_nel,dtype=np.float64)  

    for iel in range(0,swarm_nel):
        if swarm_active_cell[iel]:
           rq=0
           sq=0
           NNNV[0:4]=NQ1(rq,sq)
           dNNNVdr[0:4]=dNQ1dr(rq,sq)
           dNNNVds[0:4]=dNQ1ds(rq,sq)
           jcb=np.zeros((ndim,ndim),dtype=np.float64)
           for k in range(0,4):
               jcb[0,0]+=dNNNVdr[k]*swarm_x[swarm_icon[k,iel]]
               jcb[0,1]+=dNNNVdr[k]*swarm_y[swarm_icon[k,iel]]
               jcb[1,0]+=dNNNVds[k]*swarm_x[swarm_icon[k,iel]]
               jcb[1,1]+=dNNNVds[k]*swarm_y[swarm_icon[k,iel]]
           jcbi=np.linalg.inv(jcb)
           for k in range(0,4):
               dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
               dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
           for k in range(0,4):
               swarm_Fxx[iel]+=dNNNVdx[k]*(swarm_x[swarm_icon[k,iel]]-swarm_x0[swarm_icon[k,iel]])
               swarm_Fxy[iel]+=dNNNVdy[k]*(swarm_x[swarm_icon[k,iel]]-swarm_x0[swarm_icon[k,iel]])
               swarm_Fyx[iel]+=dNNNVdx[k]*(swarm_y[swarm_icon[k,iel]]-swarm_y0[swarm_icon[k,iel]])
               swarm_Fyy[iel]+=dNNNVdy[k]*(swarm_y[swarm_icon[k,iel]]-swarm_y0[swarm_icon[k,iel]])

           swarm_Fxx[iel]+=1
           swarm_Fyy[iel]+=1
        #end if
    #end for

    print("     -> Fxx (m,M) %.5e %.5e " %(np.min(swarm_Fxx),np.max(swarm_Fxx)))
    print("     -> Fxy (m,M) %.5e %.5e " %(np.min(swarm_Fxy),np.max(swarm_Fxy)))
    print("     -> Fyx (m,M) %.5e %.5e " %(np.min(swarm_Fyx),np.max(swarm_Fyx)))
    print("     -> Fyy (m,M) %.5e %.5e " %(np.min(swarm_Fyy),np.max(swarm_Fyy)))
    
    #####################################################################
    # infinitesimal strain based on cumulative displacements
    #####################################################################
    swarm_epsxx=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_epsxy=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_epsyx=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_epsyy=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_princp_eps1=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_princp_eps2=np.zeros(swarm_nel,dtype=np.float64) 
    swarm_new_angle=np.zeros(swarm_nel,dtype=np.float64)
    
    Fe=np.zeros((2,2),dtype=np.float64) # cell deformation gradient tensor 
    
    for iel in range(0,swarm_nel):
        if swarm_active_cell[iel]:
            
            # element deformation gradient
            Fe[0,0]=swarm_Fxx[iel]
            Fe[0,1]=swarm_Fxy[iel]
            Fe[1,0]=swarm_Fyx[iel]
            Fe[1,1]=swarm_Fyy[iel]
            # element infinitesimal strain, based on cumulative displacement
            eps = 0.5*(Fe+np.transpose(Fe)) - np.identity(2)
    
            # store in vector
            swarm_epsxx[iel]=eps[0,0]
            swarm_epsxy[iel]=eps[0,1]
            swarm_epsyx[iel]=eps[1,0]
            swarm_epsyy[iel]=eps[1,1]
            
            # compute eigen values
            #swarm_princp_eps1[iel] = 0.5*(swarm_epsxx[iel]+swarm_epsyy[iel]) + \
            #       np.sqrt(swarm_epsxy[iel]**2 + 0.25*(swarm_epsxx[iel]-swarm_epsyy[iel])**2) 
            #swarm_princp_eps2[iel] = 0.5*(swarm_epsxx[iel]+swarm_epsyy[iel]) - \
            #       np.sqrt(swarm_epsxy[iel]**2 + 0.25*(swarm_epsxx[iel]-swarm_epsyy[iel])**2) 

        #end if
    #end for

    #####################################################################
    # polar decomposition of F = V R
    # goal: compute rotation tensor R and symm stretch tensor V
    #####################################################################

    swarm_Rxx=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Rxy=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Ryx=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Ryy=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Vxx=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Vxy=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Vyx=np.zeros(swarm_nel,dtype=np.float64)  
    swarm_Vyy=np.zeros(swarm_nel,dtype=np.float64)  
    
    swarm_lambda1=np.zeros(swarm_nel,dtype=np.float64) # eigenvalues of V
    swarm_lambda2=np.zeros(swarm_nel,dtype=np.float64)
    swarm_lambda12=np.zeros(swarm_nel,dtype=np.float64)

    Re=np.zeros((2,2),dtype=np.float64) # cell rotation tensor 
    Ve=np.zeros((2,2),dtype=np.float64) # cell stretch tensor 
    Fe=np.zeros((2,2),dtype=np.float64) # cell deformation gradient tensor 

    for iel in range(0,swarm_nel):
        if swarm_active_cell[iel]:
            # element deformation gradient
            Fe[0,0]=swarm_Fxx[iel]
            Fe[0,1]=swarm_Fxy[iel]
            Fe[1,0]=swarm_Fyx[iel]
            Fe[1,1]=swarm_Fyy[iel]
            
            Ftransp=np.transpose(Fe)
            # step 1 Cauchy-Green deformation tensor
            C=np.matmul(Ftransp,Fe)
            
            # step 2 find eigenvalues of C

            mu1 = 0.5 * (C[0,0] + C[1,1] + np.sqrt(4*C[0,1]*C[1,0] + (C[0,0] - C[1,1])**2) )
            mu2 = 0.5 * (C[0,0] + C[1,1] - np.sqrt(4*C[0,1]*C[1,0] + (C[0,0] - C[1,1])**2) )

            # step 3 compute invariants of Cauchy Green tensor C. eq 5.1 from Hoger & Carlson
            IC = mu1 + mu2
            IIC = mu1*mu2

            # step 4 compute invariants of right stretch tensor U eq 5.2
            IU = np.sqrt(IC + 2*np.sqrt(IIC))
            IIU = np.sqrt(IIC)

            # step 5 compute U, eq 3.3
            U = (C+IIU*np.identity(2))/IU

            #  step 6 compute U inverse, eq 4.1
            Uinv = -IU/(IIU*(IIU+IC)+IIC)*(C-(IIU+IC)*np.identity(2))

            # step 7 compute R (rotation matrix)
            Re = np.matmul(Fe,Uinv)

            # step 8 compute V (left stretch tensor)
            Ve = np.matmul(Fe,np.transpose(Re)) 
            
            # save to vector
            swarm_Rxx[iel]=Re[0,0]
            swarm_Rxy[iel]=Re[0,1]
            swarm_Ryx[iel]=Re[1,0]
            swarm_Ryy[iel]=Re[1,1]
            
            swarm_Vxx[iel]=Ve[0,0]
            swarm_Vxy[iel]=Ve[0,1]
            swarm_Vyx[iel]=Ve[1,0]
            swarm_Vyy[iel]=Ve[1,1]
            
            # eigenvalues of V, which are the square roots of those of C
            swarm_lambda1[iel]=np.sqrt(mu1)
            swarm_lambda2[iel]=np.sqrt(mu2)
            swarm_lambda12[iel]=swarm_lambda1[iel]*swarm_lambda2[iel]

            swarm_new_angle[iel]=np.arctan2(2*swarm_Vxy[iel],\
                                            (swarm_Vxx[iel]-swarm_Vyy[iel]))/2. /np.pi*180
        #end if
    #end for


    #####################################################################
    # write out data for target cell
    #####################################################################

    area_triangle1=compute_area_of_triangle(swarm_x[swarm_icon[0,target_cell]],\
                                            swarm_y[swarm_icon[0,target_cell]],\
                                            swarm_x[swarm_icon[1,target_cell]],\
                                            swarm_y[swarm_icon[1,target_cell]],\
                                            swarm_x[swarm_icon[3,target_cell]],\
                                            swarm_y[swarm_icon[3,target_cell]])

    area_triangle2=compute_area_of_triangle(swarm_x[swarm_icon[1,target_cell]],\
                                            swarm_y[swarm_icon[1,target_cell]],\
                                            swarm_x[swarm_icon[2,target_cell]],\
                                            swarm_y[swarm_icon[2,target_cell]],\
                                            swarm_x[swarm_icon[3,target_cell]],\
                                            swarm_y[swarm_icon[3,target_cell]])

    area=area_triangle1+area_triangle2

    areafile.write("%d %e\n" %(istep,area/swarm_hx/swarm_hy))

    newfile.write("%d %e %e %e %e %e %e %e %e %e %e %e %e\n" %(istep,\
                                          swarm_new_angle[target_cell],\
                                          swarm_lambda1[target_cell]-1,\
                                          swarm_lambda2[target_cell]-1,\
                                          swarm_Rxx[target_cell],\
                                          swarm_Rxy[target_cell],\
                                          swarm_Ryx[target_cell],\
                                          swarm_Ryy[target_cell],\
                                          swarm_Vxx[target_cell],\
                                          swarm_Vxy[target_cell],\
                                          swarm_Vyx[target_cell],\
                                          swarm_Vyy[target_cell],\
                                          swarm_lambda12[target_cell]))

    print("     -> swarm_Rxx[target_cell] %.8e " % swarm_Rxx[target_cell] )
    print("     -> swarm_Rxy[target_cell] %.8e " % swarm_Rxy[target_cell] )
    print("     -> swarm_Ryx[target_cell] %.8e " % swarm_Ryx[target_cell] )
    print("     -> swarm_Ryy[target_cell] %.8e " % swarm_Ryy[target_cell] )

    #####################################################################
    # export swarm to vtu 
    #####################################################################

    if istep%every==0:

       filename = 'markers_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(swarm_nmarker,swarm_nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,swarm_nmarker):
          vtufile.write("%10e %10e %10e \n" %(swarm_x[im],swarm_y[im],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")

       vtufile.write("<DataArray type='Float32' Name='target' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           if iel==target_cell:
              vtufile.write("%e\n" % 1)
           else:
              vtufile.write("%e\n" % 0)
       vtufile.write("</DataArray>\n")

       #----------old----------
       vtufile.write("<DataArray type='Float32' Name='old: exx' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_exx_cell[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='old: exy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_exy_cell[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='old: eyy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_eyy_cell[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='old: e1' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_princp_e1[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='old: e2' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_princp_e2[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='old: angle' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_old_angle[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' \
                       Name='old: strain princ. 1 dir' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%10e %10e %10e \n" % (np.cos(swarm_old_angle[iel]/180*np.pi),
                                                np.sin(swarm_old_angle[iel]/180*np.pi), 0.)   )
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' \
                       Name='old: strain princ. 2 dir' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%10e %10e %10e \n" % (np.cos(swarm_old_angle[iel]/180*np.pi+np.pi/2),
                                                np.sin(swarm_old_angle[iel]/180*np.pi+np.pi/2), 0.)   )
       vtufile.write("</DataArray>\n")

       
       #----------new----------
       vtufile.write("<DataArray type='Float32' Name='new: exx' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_epsxx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='new: exy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_epsxy[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='new: eyy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_epsyy[iel])
       vtufile.write("</DataArray>\n")
       
       #vtufile.write("<DataArray type='Float32' Name='new: e1' Format='ascii'> \n")
       #for iel in range (0,swarm_nel):
       #    vtufile.write("%e\n" % swarm_princp_eps1[iel])
       #vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Float32' Name='new: e2' Format='ascii'> \n")
       #for iel in range (0,swarm_nel):
       #    vtufile.write("%e\n" % swarm_princp_eps2[iel])
       #vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='new: angle' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_new_angle[iel])
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' \
                       Name='new: strain princ. 1 dir' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%10e %10e %10e \n" % (np.cos(swarm_new_angle[iel]/180*np.pi),
                                                np.sin(swarm_new_angle[iel]/180*np.pi), 0.)   )
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' \
                       Name='new: strain princ. 2 dir' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%10e %10e %10e \n" % (np.cos(swarm_new_angle[iel]/180*np.pi+np.pi/2),
                                                np.sin(swarm_new_angle[iel]/180*np.pi+np.pi/2), 0.)   )
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='Fxx' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Fxx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Fxy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Fxy[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Fyx' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Fyx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Fyy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Fyy[iel])
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='Rxx' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Rxx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Rxy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Rxy[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Ryx' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Ryx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Ryy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Ryy[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='arccos(Rxx)' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % (np.arccos(swarm_Rxx[iel])))
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='Vxx' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Vxx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Vxy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Vxy[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Vyx' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Vyx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='Vyy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_Vyy[iel])
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='lambda1' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_lambda1[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='lambda2' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_lambda2[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='lambda1*lambda2' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % (swarm_lambda1[iel]*swarm_lambda2[iel]))
       vtufile.write("</DataArray>\n")


       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,swarm_nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_u[i],swarm_v[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'> \n")
       for im in range(0,swarm_nmarker):
           vtufile.write("%.3e \n" % swarm_mat[im])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%d %d %d %d\n" %(swarm_icon[0,iel],swarm_icon[1,iel],\
                                           swarm_icon[2,iel],swarm_icon[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,swarm_nel):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       #np.savetxt('markers_{:04d}.ascii'.format(istep),np.array([swarm_xc,swarm_exy_cell]).T)

       print("export to vtu file: %.3f s" % (timing.time() - start))



#end timestepping istep

#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 

filename = 'solution.vtu'
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
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" % exx_n[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" % eyy_n[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" % exy_n[i])
vtufile.write("</DataArray>\n")


vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*8))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %23)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

