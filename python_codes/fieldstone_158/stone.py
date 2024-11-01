import numpy as np
import matplotlib.pyplot as plt
import time as clock
import scipy.sparse as sps
import solkz
import solcx
import dh
import sys 

###############################################################################

# experiment=1: sinker
# experiment=2: solkz
# experiment=3: solcx
# experiment=4: donea&huerta 
# experiment=5: sinking block

experiment=1

debug=False
spy=False

###############################################################################

def density(x,y,experiment):
    if experiment==1:
       val=1-1
       if (x-0.5)**2+(y-0.5)**2<0.15**2:
          val=2-1
    if experiment==2:
       val=-np.sin(2*y)*np.cos(3*np.pi*x)
    if experiment==3:
       val=-np.sin(np.pi*y)*np.cos(np.pi*x)
    if experiment==4:
       val=1
    if experiment==5:
       if abs(x-0.5)<=0.0625 and abs(y-0.5)<=0.0625:
          val=1.01-1
       else:
          val=1-1
    return val

def viscosity(x,y,experiment):
    if experiment==1:
       val=1
       if (x-0.5)**2+(y-0.5)**2<0.15**2:
          val=10
    if experiment==2:
       val=np.exp(13.8155*y)
    if experiment==3:
       if x<0.5:
          val=1
       else:
          val=1e6
    if experiment==4:
       val=1
    if experiment==5:
       if abs(x-0.5)<=0.0625 and abs(y-0.5)<=0.0625:
          val=1000
       else:
          val=1
    return val

def gx(x,y,experiment):
    if experiment==1:
       val=0
    if experiment==2:
       val=0
    if experiment==3:
       val=0
    if experiment==4:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if experiment==5:
       val=0
    return val

def gy(x,y,experiment):
    if experiment==1:
       val=-1
    if experiment==2:
       val=-1
    if experiment==3:
       val=-1
    if experiment==4:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if experiment==5:
       val=-1
    return val

###############################################################################

Lx=1
Ly=1

if int(len(sys.argv) == 3): 
   nnx = int(sys.argv[1])
   nny = int(sys.argv[2])
   visu = 0 
else:
   nnx = 65 
   nny = nnx 
   visu = 1 

ncellx=nnx-1
ncelly=nny-1
ncell=ncellx*ncelly

hx=Lx/ncellx
hy=Ly/ncelly
hhx=1./hx
hhy=1./hy

Nb=nnx*nny       # background mesh
Nu=nnx*ncelly    # u-nodes
Nv=ncellx*nny    # v-nodes  
Np=ncellx*ncelly # p-nodes
N=Nu+Nv+Np       # total nb of unknowns

avrg=3

eta_ref=1
L_ref=min(hx,hy)

print('===========================')
print('-------- Stone 158 -------')
print('===========================')
print('ncell=',ncell)
print('nnx=',nnx)
print('nny=',nny)
print('Nb=',Nb)
print('Nu=',Nu)
print('Nv=',Nv)
print('Np=',Np)
print('N=',N)
print('avrg=',avrg)
print('eta_ref=',eta_ref)
print('L_ref=',L_ref)
print('===========================')

###############################################################################
# default b.c. on all sides is free slip
###############################################################################

if experiment==1:
   bottom_no_slip=True
   top_no_slip=True
   left_no_slip=True
   right_no_slip=True

elif experiment==2:
   bottom_no_slip=False
   top_no_slip=False
   left_no_slip=False
   right_no_slip=False

elif experiment==3:
   bottom_no_slip=False
   top_no_slip=False
   left_no_slip=False
   right_no_slip=False

elif experiment==4:
   bottom_no_slip=True
   top_no_slip=True
   left_no_slip=True
   right_no_slip=True

if experiment==5:
   bottom_no_slip=False
   top_no_slip=False
   left_no_slip=False
   right_no_slip=False


if bottom_no_slip:
   delta_bc_bottom=-1
else:
   delta_bc_bottom=+1

if top_no_slip:
   delta_bc_top=-1
else:
   delta_bc_top=+1

if left_no_slip:
   delta_bc_left=-1
else:
   delta_bc_left=+1

if right_no_slip:
   delta_bc_right=-1
else:
   delta_bc_right=+1

###############################################################################
# build background mesh
###############################################################################
start = clock.time()

xb=np.zeros(Nb,dtype=np.float64)  # x coordinates
yb=np.zeros(Nb,dtype=np.float64)  # y coordinates
rho=np.zeros(Nb,dtype=np.float64) # density 
eta=np.zeros(Nb,dtype=np.float64) # viscosity

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xb[counter]=i*hx
        yb[counter]=j*hy
        counter += 1

icon =np.zeros((4,ncell),dtype=np.int32)
counter = 0
for j in range(0,ncelly):
    for i in range(0,ncellx):
        icon[0,counter]=i+j*(ncellx+1)
        icon[1,counter]=i+1+j*(ncellx+1)
        icon[2,counter]=i+1+(j+1)*(ncellx+1)
        icon[3,counter]=i+(j+1)*(ncellx+1)
        counter += 1
    #end for
#end for

for i in range(0,Nb):
    rho[i]=density(xb[i],yb[i],experiment)
    eta[i]=viscosity(xb[i],yb[i],experiment)

if debug: np.savetxt('grid_b.ascii',np.array([xb,yb]).T,header='# x,y')

print("setup: b grid points: %.3f s" % (clock.time() - start))

###############################################################################
# build u mesh
###############################################################################
start = clock.time()

xu=np.zeros(Nu,dtype=np.float64)
yu=np.zeros(Nu,dtype=np.float64)
u=np.zeros(Nu,dtype=np.float64)
left=np.zeros(Nu,dtype=bool) 
right=np.zeros(Nu,dtype=bool) 

counter = 0
for j in range(0,ncelly):
    for i in range(0,nnx):
        xu[counter]=i*hx
        yu[counter]=j*hy+hy/2
        left[counter]=(i==0)
        right[counter]=(i==nnx-1)
        counter += 1

if debug: np.savetxt('grid_u.ascii',np.array([xu,yu]).T,header='# x,y')

print("setup: u grid points: %.3f s" % (clock.time() - start))

###############################################################################
# build v mesh
###############################################################################
start = clock.time()

xv=np.zeros(Nv,dtype=np.float64)
yv=np.zeros(Nv,dtype=np.float64)
v=np.zeros(Nv,dtype=np.float64)
bottom=np.zeros(Nv,dtype=bool) 
top=np.zeros(Nv,dtype=bool) 

counter = 0
for j in range(0,nny):
    for i in range(0,ncellx):
        xv[counter]=i*hx+hx/2
        yv[counter]=j*hy
        bottom[counter]=(j==0)
        top[counter]=(j==nny-1)
        counter += 1

if debug: np.savetxt('grid_v.ascii',np.array([xv,yv]).T,header='# x,y')

print("setup: v grid points: %.3f s" % (clock.time() - start))

###############################################################################
# build p mesh
###############################################################################
start = clock.time()

xp=np.zeros(Np,dtype=np.float64)
yp=np.zeros(Np,dtype=np.float64)
p=np.zeros(Np,dtype=np.float64) 

counter = 0
for j in range(0,ncelly):
    for i in range(0,ncellx):
        xp[counter]=i*hx+hx/2
        yp[counter]=j*hy+hy/2
        counter += 1

if debug: np.savetxt('grid_p.ascii',np.array([xp,yp]).T,header='# x,y')

print("setup: p grid points: %.3f s" % (clock.time() - start))

###############################################################################
# declare arrays for matrix and rhs
###############################################################################

#A=np.zeros((N,N),dtype=np.float64)
A=sps.lil_matrix((N,N),dtype=np.float64)
b=np.zeros(N,dtype=np.float64)

###############################################################################
# loop over all u nodes
###############################################################################
start = clock.time()

for i in range(0,Nu):

    if left[i]: # u node on left boundary
       A[i,i]=1
       b[i]=0

    elif right[i]: # u node on right boundary
       A[i,i]=1
       b[i]=0

    else:
       index_eta_nw=i+ncellx
       index_eta_n=index_eta_nw+1
       index_eta_ne=index_eta_n+1
       index_eta_sw=i-1
       index_eta_s=i
       index_eta_se=i+1

       match(avrg):
          case(1): # arithmetic
             eta_w=(eta[index_eta_nw]+eta[index_eta_sw]+eta[index_eta_n]+eta[index_eta_s])/4.
             eta_e=(eta[index_eta_ne]+eta[index_eta_se]+eta[index_eta_n]+eta[index_eta_s])/4.
          case(2): # geometric
             eta_w=10.**((np.log10(eta[index_eta_nw])+np.log10(eta[index_eta_sw])+np.log10(eta[index_eta_n])+np.log10(eta[index_eta_s]))/4.)
             eta_e=10.**((np.log10(eta[index_eta_ne])+np.log10(eta[index_eta_se])+np.log10(eta[index_eta_n])+np.log10(eta[index_eta_s]))/4.)
          case(3): # harmonic
             eta_w=4./(1./eta[index_eta_nw]+1./eta[index_eta_sw]+1./eta[index_eta_n]+1./eta[index_eta_s])
             eta_e=4./(1./eta[index_eta_ne]+1./eta[index_eta_se]+1./eta[index_eta_n]+1./eta[index_eta_s])
          case _: 
             exit('avrg unknown')

       eta_n=eta[index_eta_n]
       eta_s=eta[index_eta_s]

       eta_n_yy=eta_n/hy**2 ; eta_n_xy=eta_n/hx/hy
       eta_e_xx=eta_e/hx**2
       eta_w_xx=eta_w/hx**2
       eta_s_yy=eta_s/hy**2 ; eta_s_xy=eta_s/hx/hy
       
       ii=i%nnx
       jj=int((i-i%nnx)/nnx)

       index_p_w=i-jj-1
       index_p_e=index_p_w+1

       index_rho_n=i+nnx
       index_rho_s=i

       index_v_sw=i-jj-1
       index_v_se=i-jj
       index_v_nw=i-jj-1+ncellx
       index_v_ne=i-jj+ncellx

       index_u_n=i+nnx
       index_u_s=i-nnx
       index_u_w=i-1
       index_u_e=i+1

       #print(xp[index_p_w],yp[index_p_w],eta_w)
       #print(xp[index_p_e],yp[index_p_e],eta_e)
       if debug:
          print('================')
          print('u node #',i)
          print('ii,jj =',ii,jj)
          print('index_eta_nw=',index_eta_nw)
          print('index_eta_n =',index_eta_n)
          print('index_eta_ne=',index_eta_ne)
          print('index_eta_sw=',index_eta_sw)
          print('index_eta_s =',index_eta_s)
          print('index_eta_se=',index_eta_se)
          print('index_p_w =',index_p_w)
          print('index_p_e =',index_p_e)
          print('index_rho_n =',index_rho_n)
          print('index_rho_s =',index_rho_s)
          print('index_v_sw =',index_v_sw)
          print('index_v_se =',index_v_se)
          print('index_v_nw =',index_v_nw)
          print('index_v_ne =',index_v_ne)
          print('index_u_n =',index_u_n)
          print('index_u_s =',index_u_s)
          print('index_u_w =',index_u_w)
          print('index_u_e =',index_u_e)

       b[i]=-(rho[index_rho_s]+rho[index_rho_n])/2*gx(xu[i],yu[i],experiment)

       A[i,index_u_e]=2*eta_e_xx
       A[i,index_u_w]=2*eta_w_xx
       A[i,Nu+index_v_ne]=+eta_n_xy
       A[i,Nu+index_v_nw]=-eta_n_xy
       A[i,Nu+index_v_se]=-eta_s_xy
       A[i,Nu+index_v_sw]=+eta_s_xy
       A[i,Nu+Nv+index_p_e]=-hhx * eta_ref/L_ref
       A[i,Nu+Nv+index_p_w]=+hhx * eta_ref/L_ref

       if jj==0: # bottom row, ghosts nodes used
          A[i,i]=-2*eta_e_xx-2*eta_w_xx-eta_n_yy-(1-delta_bc_bottom)*eta_s_yy
          A[i,index_u_n]=eta_n_yy
       elif jj==ncelly-1: # top row, ghosts nodes used
          A[i,i]=-2*eta_e_xx-2*eta_w_xx-(1-delta_bc_top)*eta_n_yy-eta_s_yy
          A[i,index_u_s]=eta_s_yy
       else: # regular stencil
          A[i,i]=-2*eta_e_xx-2*eta_w_xx-eta_n_yy-eta_s_yy
          A[i,index_u_s]=eta_s_yy
          A[i,index_u_n]=eta_n_yy
       #end if

    #end if
#end for

if spy:
   plt.spy(A)
   plt.savefig('matrix_u.pdf', bbox_inches='tight')

print("loop over u nodes: %.3f s" % (clock.time() - start))

###############################################################################
# loop over all v nodes
###############################################################################
start = clock.time()

for i in range(0,Nv):

    if bottom[i]: # v node on bottom boundary
       A[Nu+i,Nu+i]=1
       b[Nu+i]=0

    elif top[i]: # v node on top boundary
       A[Nu+i,Nu+i]=1
       b[Nu+i]=0

    else:
       ii=i%ncellx
       jj=int((i-i%ncellx)/ncellx)

       index_eta_sw=i+jj-nnx
       index_eta_w=i+jj
       index_eta_nw=i+jj+nnx

       index_eta_se=i+jj-nnx+1
       index_eta_e=i+jj+1
       index_eta_ne=i+jj+nnx+1

       match(avrg):
          case(1): # arithmetic
             eta_n=(eta[index_eta_nw]+eta[index_eta_ne]+eta[index_eta_w]+eta[index_eta_e])/4
             eta_s=(eta[index_eta_sw]+eta[index_eta_se]+eta[index_eta_w]+eta[index_eta_e])/4
          case(2): # gemetric 
             eta_n=10.**((np.log10(eta[index_eta_nw])+np.log10(eta[index_eta_ne])+np.log10(eta[index_eta_w])+np.log10(eta[index_eta_e]))/4.)
             eta_s=10.**((np.log10(eta[index_eta_sw])+np.log10(eta[index_eta_se])+np.log10(eta[index_eta_w])+np.log10(eta[index_eta_e]))/4.)
          case(3): # harmonic
             eta_n=4./(1./eta[index_eta_nw]+1./eta[index_eta_ne]+1./eta[index_eta_w]+1./eta[index_eta_e])
             eta_s=4./(1./eta[index_eta_sw]+1./eta[index_eta_se]+1./eta[index_eta_w]+1./eta[index_eta_e])
          case _: 
             exit('avrg unknown')

       eta_e=eta[index_eta_e]
       eta_w=eta[index_eta_w]

       eta_n_yy=eta_n/hy**2
       eta_s_yy=eta_s/hy**2
       eta_e_xx=eta_e/hx**2 ; eta_e_xy=eta_e/hx/hy
       eta_w_xx=eta_w/hx**2 ; eta_w_xy=eta_w/hx/hy

       index_p_s=i-ncellx
       index_p_n=i

       index_rho_w=i+jj
       index_rho_e=i+jj+1

       index_v_w=i-1
       index_v_e=i+1
       index_v_s=i-ncellx
       index_v_n=i+ncellx

       index_u_sw=i-nnx+1 +jj-1
       index_u_se=i-nnx+1+1 +jj-1
       index_u_nw=i-nnx+1+nnx +jj-1
       index_u_ne=i-nnx+1+1 +nnx +jj-1

       #print(xp[index_p_s],yp[index_p_s],eta_s)
       #print(xp[index_p_n],yp[index_p_n],eta_n)
       if debug:
          print('================')
          print('v node #',i)
          print('ii,jj =',ii,jj)
          print('index_eta_sw=',index_eta_sw)
          print('index_eta_w =',index_eta_w)
          print('index_eta_nw=',index_eta_nw)
          print('index_eta_se=',index_eta_se)
          print('index_eta_e =',index_eta_e)
          print('index_eta_ne=',index_eta_ne)
          print('index_p_s =',index_p_s)
          print('index_p_n =',index_p_n)
          print('index_rho_w =',index_rho_w)
          print('index_rho_e =',index_rho_e)
          print('index_v_w =',index_v_w)
          print('index_v_e =',index_v_e)
          print('index_v_s =',index_v_s)
          print('index_v_n =',index_v_n)
          print('index_u_sw =',index_u_sw)
          print('index_u_se =',index_u_se)
          print('index_u_nw =',index_u_nw)
          print('index_u_ne =',index_u_ne)

       b[Nu+i]=-(rho[index_rho_e]+rho[index_rho_w])/2*gy(xv[i],yv[i],experiment)

       A[Nu+i,Nu+index_v_n]=2*eta_n_yy
       A[Nu+i,Nu+index_v_s]=2*eta_s_yy
       A[Nu+i,index_u_ne]=eta_e_xy
       A[Nu+i,index_u_se]=-eta_e_xy
       A[Nu+i,index_u_nw]=-eta_w_xy
       A[Nu+i,index_u_sw]=eta_w_xy
       A[Nu+i,Nu+Nv+index_p_n]=-hhy * eta_ref/L_ref
       A[Nu+i,Nu+Nv+index_p_s]=+hhy * eta_ref/L_ref

       if ii==0: # left column, ghosts nodes used
          A[Nu+i,Nu+i]=-eta_e_xx-(1-delta_bc_left)*eta_w_xx-2*eta_n_yy-2*eta_s_yy
          A[Nu+i,Nu+index_v_e]=eta_e_xx
       elif ii==ncellx-1: # right column, ghosts nodes used
          A[Nu+i,Nu+i]=-(1-delta_bc_right)*eta_e_xx-eta_w_xx-2*eta_n_yy-2*eta_s_yy
          A[Nu+i,Nu+index_v_w]=eta_w_xx
       else:
          A[Nu+i,Nu+i]=-eta_e_xx-eta_w_xx-2*eta_n_yy-2*eta_s_yy
          A[Nu+i,Nu+index_v_e]=eta_e_xx
          A[Nu+i,Nu+index_v_w]=eta_w_xx
       #end if

    #end if
#end for

if spy:
   plt.spy(A)
   plt.savefig('matrix_uv.pdf', bbox_inches='tight')

print("loop over v nodes: %.3f s" % (clock.time() - start))

###############################################################################
# loop over all p nodes
###############################################################################
start = clock.time()

for i in range(0,Np):

    ii=i%ncellx
    jj=int((i-i%ncellx)/ncellx)

    index_u_w=i+jj      # u node left 
    index_u_e=i+jj+1    # u node right
    index_v_s=i         # v node below
    index_v_n=i+ncellx  # v node above

    A[Nu+Nv+i,   index_u_e]= hhx * eta_ref/L_ref
    A[Nu+Nv+i,   index_u_w]=-hhx * eta_ref/L_ref
    A[Nu+Nv+i,Nu+index_v_n]= hhy * eta_ref/L_ref
    A[Nu+Nv+i,Nu+index_v_s]=-hhy * eta_ref/L_ref

    if debug:
       print('================')
       print('p node #',i)
       print('ii,jj =',ii,jj)
       print('index_u_w',index_u_w)
       print('index_u_e',index_u_e)
       print('index_v_s',index_v_s)
       print('index_v_n',index_v_n)

#end for

if spy:
   plt.spy(A)
   plt.savefig('matrix_uvp.pdf', bbox_inches='tight')

print("loop over p nodes: %.3f s" % (clock.time() - start))

###############################################################################
# solve system
###############################################################################
start = clock.time()

sol = sps.linalg.spsolve(sps.csr_matrix(A),b)

print("Solve linear system: %.5f s | N= %d " % (clock.time() - start, N))

###############################################################################
# extract u,v,p fields from solution vector 
###############################################################################
start = clock.time()

u=sol[0:Nu]
v=sol[Nu+0:Nu+Nv]
p=sol[Nu+Nv+0:Nu+Nv+Np]*eta_ref/L_ref

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

print("extract: %.5f s " % (clock.time() - start))

###############################################################################
# normalise pressure ?
###############################################################################
start = clock.time()

p_avrg=np.sum(p)/ncell

p-=p_avrg

print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

print("pressure normalise: %.5f s " % (clock.time() - start))

###############################################################################
# compute strain rate
# exx,eyy computed in middle of cells
# exy is computed on the background mesh
###############################################################################
start = clock.time()

#u[:]=1
#v[:]=1

#u[:]=xu[:]
#v[:]=yv[:]

exy=np.zeros(Nb,dtype=np.float64)  
exx=np.zeros(ncell,dtype=np.float64)  
eyy=np.zeros(ncell,dtype=np.float64)  

counter = 0
for j in range(0,ncelly):
    for i in range(0,ncellx):
        index_u_w=counter+j
        index_u_e=counter+j+1
        index_v_s=counter
        index_v_n=counter+nnx-1
        #print('cell',counter,':',index_u_w,index_u_e,index_v_s,index_v_n)
        exx[counter]=(u[index_u_e]-u[index_u_w])/hx
        eyy[counter]=(v[index_v_n]-v[index_v_s])/hy
        counter += 1

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        index_v_w=counter-j-1
        index_v_e=counter-j
        index_u_s=counter-nnx
        index_u_n=counter
        #print('node',counter,':',index_v_w,index_v_e,index_u_s,index_u_n)
        if i==0 and j>0 and j<nny-1: # left
           exy[counter]=0.5*(u[index_u_n]-u[index_u_s])/hy+\
                        0.5*(v[index_v_e]-delta_bc_left*v[index_v_e])/hx
        elif i==nnx-1 and j>0 and j<nny-1: # right
           exy[counter]=0.5*(u[index_u_n]-u[index_u_s])/hy+\
                        0.5*(delta_bc_right*v[index_v_w]-v[index_v_w])/hx
        elif j==0 and i>0 and i<nnx-1: # bottom
           exy[counter]=0.5*(u[index_u_n]-delta_bc_bottom*u[index_u_n])/hy+\
                        0.5*(v[index_v_e]-v[index_v_w])/hx
        elif j==nny-1 and i>0 and i<nnx-1: # top
           exy[counter]=0.5*(delta_bc_top*u[index_u_s]-u[index_u_s])/hy+\
                        0.5*(v[index_v_e]-v[index_v_w])/hx
        elif i>0 and i<nnx-1 and j>0  and j<nny-1:
           exy[counter]=0.5*(u[index_u_n]-u[index_u_s])/hy+\
                        0.5*(v[index_v_e]-v[index_v_w])/hx
        #end if
        counter += 1
    #end for
#end for

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

print("compute strain rate: %.5f s " % (clock.time() - start))

###############################################################################
# project u,v,p onto background mesh cell centers 
###############################################################################
start = clock.time()

error_u=np.zeros(Nu,dtype=np.float64)
error_v=np.zeros(Nv,dtype=np.float64)

for i in range(0,Nu):
    if experiment==1:
       ui=vi=pi=0 
    elif experiment==2:
       ui,vi,pi=solkz.SolKzSolution(xu[i],yu[i])
    elif experiment==3:
       ui,vi,pi=solcx.SolCxSolution(xu[i],yu[i])
    elif experiment==4:
       ui,vi,pi=dh.DHSolution(xu[i],yu[i])
    elif experiment==5:
       ui=vi=pi=0 
    error_u[i]=u[i]-ui
#end for

for i in range(0,Nv):
    if experiment==1:
       ui=vi=pi=0 
    elif experiment==2:
       ui,vi,pi=solkz.SolKzSolution(xv[i],yv[i])
    elif experiment==3:
       ui,vi,pi=solcx.SolCxSolution(xv[i],yv[i])
    elif experiment==4:
       ui,vi,pi=dh.DHSolution(xv[i],yv[i])
    elif experiment==5:
       ui=vi=pi=0 
    error_v[i]=v[i]-vi
#end for

uu=np.zeros(ncell,dtype=np.float64)
vv=np.zeros(ncell,dtype=np.float64)
vel=np.zeros(ncell,dtype=np.float64)
uth=np.zeros(ncell,dtype=np.float64)
vth=np.zeros(ncell,dtype=np.float64)
pth=np.zeros(ncell,dtype=np.float64)
error_uu=np.zeros(ncell,dtype=np.float64) # for plotting only
error_vv=np.zeros(ncell,dtype=np.float64) # for plotting only
error_p=np.zeros(ncell,dtype=np.float64)

errv=0.
errp=0.
vrms=0.
for i in range(0,ncell):

    ii=i%ncellx
    jj=int((i-i%ncellx)/ncellx)

    index_u_w=i+jj      # u node left 
    index_u_e=i+jj+1    # u node right
    index_v_s=i         # v node below
    index_v_n=i+ncellx  # v node above

    uu[i]=0.5*(u[index_u_w]+u[index_u_e])
    vv[i]=0.5*(v[index_v_n]+v[index_v_s])
    vel[i]=np.sqrt(uu[i]**2+vv[i]**2)
   
    if experiment==1:
       ui=vi=pi=0 
    elif experiment==2:
       ui,vi,pi=solkz.SolKzSolution(xp[i],yp[i])
    elif experiment==3:
       ui,vi,pi=solcx.SolCxSolution(xp[i],yp[i])
    elif experiment==4:
       ui,vi,pi=dh.DHSolution(xp[i],yp[i])
    elif experiment==5:
       ui=vi=pi=0 

    uth[i]=ui
    vth[i]=vi
    pth[i]=pi
    error_uu[i]=uu[i]-ui
    error_vv[i]=vv[i]-vi
    error_p[i]=p[i]-pi

    vrms+=(uu[i]**2+vv[i]**2)*hx*hy

    errv+=(abs(error_u[index_u_w])+abs(error_v[index_u_e]))*hx*hy/2
    errp+=abs(error_p[i])*hx*hy

print('ncellx= ',ncellx,' vrms= ',np.sqrt(vrms/Lx/Ly))
print('ncellx= ',ncellx,' errors= ',errv,errp)

print("u,v at cell center: %.5f s " % (clock.time() - start))

###############################################################################

if experiment==5:
   print(ncellx,np.min(u),np.max(u),\
                np.min(v),np.max(v),\
                np.min(p),np.max(p),\
                np.min(vel),np.max(vel),'stats')

   profile=open('profiley_'+str(ncelly)+'.ascii',"w")
   for i in range(ncell):
       if abs(xp[i]-0.5)<hx:
          profile.write("%e %e %e %e %e \n" %(xp[i],yp[i],uu[i],vv[i],p[i]))
   profile.close()

   profile=open('profilex_'+str(ncelly)+'.ascii',"w")
   for i in range(ncell):
       if abs(yp[i]-0.5)<hx:
          profile.write("%e %e %e %e %e \n" %(xp[i],yp[i],uu[i],vv[i],p[i]))
   profile.close()

###############################################################################
# export fields to paraview files
###############################################################################
start = clock.time()

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(Nb,ncell))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,Nb):
    vtufile.write("%10e %10e %10e \n" %(xb[i],yb[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e %10e %10e \n" %(uu[i],vv[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e %10e %10e \n" %(error_uu[i],error_vv[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e %10e %10e \n" %(uth[i],vth[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32'  Name='pressure' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e  \n" %(p[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32'  Name='pressure (error)' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e  \n" %(error_p[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32'  Name='pressure (th)' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e  \n" %(pth[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32'  Name='exx' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e  \n" %(exx[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32'  Name='eyy' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e  \n" %(eyy[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32'  Name='div(v)' Format='ascii'> \n")
for i in range(0,ncell):
    vtufile.write("%10e  \n" %(exx[i]+eyy[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
for i in range(0,Nb):
    vtufile.write("%10e \n" %rho[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
for i in range(0,Nb):
    vtufile.write("%10e \n" %eta[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for i in range(0,Nb):
    vtufile.write("%10e \n" %exy[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,ncell):
    vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,ncell):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,ncell):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export to vtu: %.5f s " % (clock.time() - start))

print('===========================')
print("-----------the end---------")
print('===========================')
