import numpy as np
import sys as sys
import time as timing
from scipy import special

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

def interpolate_vel_on_pt(xm,ym):
    ielx=int(xm/Lx*nelx)
    iely=int(ym/Ly*nely)
    #if ielx<0:
    #   exit('ielx<0')
    #if iely<0:
    #   exit('iely<0')
    #if ielx>=nelx:
    #   exit('ielx>nelx')
    #if iely>=nely:
    #   exit('iely>nely')
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
    
    #exxm=exx[iel]
    #eyym=eyy[iel]
    #exym=exy[iel]

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

Lx=8     # x extent of domain
Ly=12    # y extent of domain
nelx=40  # number of elements in x direction
nely=60  # number of elements in y direction 

delta_u=0.5 # amplitude of velocity field

nstep=151 # number of time steps

CFL_nb=0.1 # CFL number

every=10 # how often vtu/ascii output is generated

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

for i in range(0,NV):
    u[i] = 0 
    v[i]=delta_u*special.erf(xV[i]-Lx/2)

######################################################################
# compute strainrate at center of element 
######################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
pc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives

for iel in range(0,nel):
    rq = 0.
    sq = 0.
    NNNV[0:mV]=NNV(rq,sq)
    dNNNVdr[0:mV]=dNNVdr(rq,sq)
    dNNNVds[0:mV]=dNNVds(rq,sq)
    jcb=np.zeros((ndim,ndim),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)
    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
    for k in range(0,mV):
        xc[iel] += NNNV[k]*xV[iconV[k,iel]]
        yc[iel] += NNNV[k]*yV[iconV[k,iel]]
        exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                    0.5*dNNNVdx[k]*v[iconV[k,iel]]
    #end for
#end for

print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))

print("compute press & sr: %.3f s" % (timing.time() - start))

#####################################################################
# compute nodal strainrate and heat flux 
#####################################################################
start = timing.time()
    
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

swarm_nelx=3*nelx
swarm_nely=3*nely
swarm_nel=swarm_nelx*swarm_nely

swarm_nnx=swarm_nelx+1
swarm_nny=swarm_nely+1
nmarker=swarm_nnx*swarm_nny

swarm_hx=Lx/swarm_nnx
swarm_hy=Ly/swarm_nny

swarm_x=np.empty(nmarker,dtype=np.float64)  
swarm_y=np.empty(nmarker,dtype=np.float64)  
swarm_u=np.zeros(nmarker,dtype=np.float64)  
swarm_v=np.zeros(nmarker,dtype=np.float64)  
swarm_active=np.zeros(nmarker,dtype=np.bool) 
swarm_exx=np.zeros(nmarker,dtype=np.float64)  
swarm_eyy=np.zeros(nmarker,dtype=np.float64)  
swarm_exy=np.zeros(nmarker,dtype=np.float64)  
swarm_icon=np.zeros((4,swarm_nel),dtype=np.int32)
swarm_active_cell=np.zeros(swarm_nel,dtype=np.bool) 
swarm_exy_cell=np.zeros(swarm_nel,dtype=np.float64)  
swarm_xc=np.zeros(swarm_nel,dtype=np.float64)  
swarm_yc=np.zeros(swarm_nel,dtype=np.float64)  

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

print("     -> nmarker %d " % nmarker)
print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("markers setup: %.3f s" % (timing.time() - start))

#################################################################
# marker paint
#################################################################
start = timing.time()

swarm_mat=np.zeros(nmarker,dtype=np.int32)  

for i in [0,2]:
    dx=Lx/4
    for im in range (0,nmarker):
        if swarm_x[im]>i*dx and swarm_x[im]<(i+1)*dx:
           swarm_mat[im]+=1
    #end for
#end for

for i in [0,2,4,6]:
    dy=Ly/8
    for im in range (0,nmarker):
        if swarm_y[im]>i*dy and swarm_y[im]<(i+1)*dy:
           swarm_mat[im]+=1
    #end for
#end for

rad=Lx*0.16667
xcirc=0.5*Lx
ycirc=0.5*Ly
for im in range (0,nmarker):
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
    # advect markers and compute strain component exy on the markers.
    # Markers are advected by means of a simple Euler step and those
    # advected outside of the domain are labelled inactive and their 
    # strain set to zero. 
    #####################################################################
    start = timing.time()

    for im in range(0,nmarker):
           if swarm_active[im]:
              swarm_u[im],swarm_v[im],rm,sm,iel,exx_m,eyy_m,exy_m=\
                    interpolate_vel_on_pt(swarm_x[im],swarm_y[im])
              swarm_exx[im]+=exx_m*dt
              swarm_eyy[im]+=eyy_m*dt
              swarm_exy[im]+=exy_m*dt
              swarm_x[im]+=swarm_u[im]*dt
              swarm_y[im]+=swarm_v[im]*dt
              if swarm_x[im]<0 or swarm_x[im]>Lx or swarm_y[im]<0 or swarm_y[im]>Ly:
                 swarm_active[im]=False
                 swarm_exx[im]=0
                 swarm_eyy[im]=0
                 swarm_exy[im]=0
           # end if active
    # end for im

    print("     -> swarm_exy (m,M) %.4f %.4f " %(np.min(swarm_exy),np.max(swarm_exy)))

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
                       interpolate_vel_on_pt(swarm_xc[iel],swarm_yc[iel])
           swarm_exy_cell[iel]+=exy_m*dt
        else:
           swarm_exy_cell[iel]=0
        #end if
    #end for

    #####################################################################
    # export swarm to vtu 
    #####################################################################

    if istep%every==0:

       filename = 'swarm_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%e %e %e \n" %(swarm_x[im],swarm_y[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_u[im],swarm_v[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%.3e \n" % swarm_mat[im])
       vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Float32' Name='strain xx' Format='ascii'> \n")
       #for im in range(0,nmarker):
       #    vtufile.write("%10e \n" % swarm_exx[im])
       #vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Float32' Name='strain yy' Format='ascii'> \n")
       #for im in range(0,nmarker):
       #    vtufile.write("%10e \n" % swarm_eyy[im])
       #vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='strain xy' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%.4e \n" % swarm_exy[im])
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

       filename = 'markers_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,swarm_nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nmarker):
          vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,swarm_nel):
           vtufile.write("%e\n" % swarm_exy_cell[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_u[i],swarm_v[i],0.))
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

       np.savetxt('swarm_{:04d}.ascii'.format(istep),np.array([swarm_x,swarm_exy]).T)
       np.savetxt('markers_{:04d}.ascii'.format(istep),np.array([swarm_xc,swarm_exy_cell]).T)

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
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exy[iel])
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
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
