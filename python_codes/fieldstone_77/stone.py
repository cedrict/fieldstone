import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as time
from scipy.linalg import null_space


#------------------------------------------------------------------------------

def dpdx_th(x,y):
    return 30*(x-1./2.)**2*y**2-30*(1-x)**2*(y-1./2.)**3
    
def dpdy_th(x,y):
    return 20*(x-1./2.)**3*y + 30*(1-x)**3*(y-1./2.)**2  

def exx_th(x,y):
    return -400*(1-x)*x*(2*x-1)*(y-1)*y*(2*y-1) 

def exy_th(x,y):
    return 100*(1-x)**2*x**2*(6*y**2-6*y+1)-100*(6*x**2-6*x+1)*(1-y)**2*y**2 

def eyy_th(x,y):
    return 400*(x-1)*x*(2*x-1)*(1-y)*y*(2*y-1) 

def dexxdx(x,y):
    return 400*(6*x**2-6*x+1)*y*(2*y**2-3*y+1)

def dexydx(x,y):
    return 100*(-2*x**2*(1-x)*(6*y**2-6*y+1) + 2*x*(1-x)**2*(6*y**2-6*y+1) -6*(2*x-1)*(1-y)**2*y**2)

def dexydy(x,y):
    return 200*(6*x**2-6*x+1)*(1-y)*y**2 + 100*(1-x)**2*x**2*(12*y-6) -200*(6*x**2-6*x+1)*(1-y)**2*y

def deyydy(x,y):
    return -400*x*(2*x**2-3*x+1)*(6*y**2-6*y+1)

#------------------------------------------------------------------------------

def bx(x,y,exp):
    if experiment==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if experiment==2:
       val=0
    if experiment==3:
       val=-(1.+y-3*x**2*y**2)
    if experiment==4:
       val = dpdx_th(x,y)-2*dexxdx(x,y)-2*dexydy(x,y) 
    if experiment==5:
       val=0.
    if experiment==6:
       val=0.
    return val

def by(x,y,experiment):
    if experiment==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if experiment==2:
       if abs(x-.5)<0.125 and abs(y-0.5)<0.125:
          val=-1.001 
       else:
          val=-1. #+1
    if experiment==3:
       val=-(1.-3*x-2*x**3*y)
    if experiment==4:
       val= dpdy_th(x,y)-2*dexydx(x,y)-2*deyydy(x,y)
    if experiment==5:
       val=0
    if experiment==6:
       val=-4*np.pi**2*np.cos(np.pi*x)*np.sin(np.pi*y) 
    return val

#------------------------------------------------------------------------------

def eta(x,y,experiment):
    if experiment==1:
       val=1
    if experiment==2:
       if abs(x-.5)<0.125 and abs(y-0.5)<0.125:
          val=1.
       else:
          val=1.
    if experiment==3:
       val=1.
    if experiment==4:
       val=1.
    if experiment==5:
       val=1.
    if experiment==6:
       val=1.
    return val

#------------------------------------------------------------------------------

def velocity_x(x,y,experiment):
    if experiment==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if experiment==2:
       val=0
    if experiment==3:
       val= x + x**2 - 2*x*y + x**3 - 3*x*y**2 + x**2*y 
    if experiment==4:
       val = 200*x**2*(1-x)**2*y*(1-y)*(1-2*y) 
    if experiment==5:
       val = y*(1.-y)
    if experiment==6:
       val=np.sin(np.pi*x)*np.cos(np.pi*y)
    return val

def velocity_y(x,y,experiment):
    if experiment==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if experiment==2:
       val=0
    if experiment==3:
       val= -y - 2*x*y + y**2 - 3*x**2*y + y**3 - x*y**2
    if experiment==4:
       val = -200*x*(1-x)*(1-2*x)*y**2*(1-y)**2 
    if experiment==5:
       val = 0.
    if experiment==6:
       val=-np.cos(np.pi*x)*np.sin(np.pi*y)
    return val

def pressure(x,y,experiment):
    if experiment==1:
       val=x*(1.-x)-1./6.
    if experiment==2:
       val=0
    if experiment==3:
       val=x*y+x+y+x**3*y**2 - 4./3.
    if experiment==4:
       val= 10*( (x-1./2.)**3*y**2+(1-x)**3*(y-1./2.)**3 )
    if experiment==5:
       val=1.-2*x
    if experiment==6:
       val=2*np.pi*np.cos(np.pi*x)*np.cos(np.pi*y)
    return val

#------------------------------------------------------------------------------

def NNV(rq,sq,sft):
    if sft==1:
        NV_0=0.25*(1-rq**2-2*sq+sq**2)
        NV_1=0.25*(1+2*rq+rq**2-sq**2)
        NV_2=0.25*(1-rq**2+2*sq+sq**2)
        NV_3=0.25*(1-2*rq+rq**2-sq**2)
    if sft==2:
        NV_0=0.25*(1-2*sq-1.5*(rq**2-sq**2))
        NV_1=0.25*(1+2*rq+1.5*(rq**2-sq**2))
        NV_2=0.25*(1+2*sq-1.5*(rq**2-sq**2))
        NV_3=0.25*(1-2*rq+1.5*(rq**2-sq**2))
    return NV_0,NV_1,NV_2,NV_3

def dNNVdr(rq,sq,sft):
    if sft==1:
       dNVdr_0=0.5*(-rq)  
       dNVdr_1=0.5*(1+rq) 
       dNVdr_2=0.5*(-rq)  
       dNVdr_3=0.5*(-1+rq)
    if sft==2:
       dNVdr_0=-0.75*rq     
       dNVdr_1=0.5+0.75*rq  
       dNVdr_2=-0.75*rq     
       dNVdr_3=-0.5+0.75*rq 
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3

def dNNVds(rq,sq,sft):
    if sft==1:
       dNVds_0=0.5*(-1+sq)
       dNVds_1=0.5*(-sq)
       dNVds_2=0.5*(1+sq)
       dNVds_3=0.5*(-sq)
    if sft==2:
       dNVds_0=-0.5+0.75*sq
       dNVds_1=-0.75*sq
       dNVds_2=0.5+0.75*sq
       dNVds_3=-0.75*sq
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   sft = int(sys.argv[4])
   formulation=int(sys.argv[5])
else:
   nelx = 32
   nely = nelx 
   visu = 1
   sft=1
   formulation=1

# shape fct type 1: mid point
# shape fct type 2: mid value
# formulation=1: div-div v 
# formulation=2: laplace v

nnp=(nely+1)*nelx + nely*(nelx+1) # number of points

nel=nelx*nely  # number of elements, total

NfemV=nnp*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eps=1.e-10

# experiment 1: donea & huerta benchmark (D&H)
# experiment 2: stokes sphere
# experiment 3: DB2D benchmark
# experiment 4: volker John 3 benchmark 
# experiment 5: hor. poiseuille flow
# experiment 6: solcx?

experiment=2

pnormalise=True

nqperdim=3

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

#################################################################
# grid point setup
#################################################################
start = time.time()

xV=np.empty(nnp,dtype=np.float64)  # x coordinates
yV=np.empty(nnp,dtype=np.float64)  # y coordinates

hx=Lx/nelx
hy=Ly/nely

counter=0
for j in range(0,nely):
   # bottom line
   for i in range(0,nelx):
      xV[counter]=(i+1-0.5)*hx
      yV[counter]=(j+1-1)*hy
      counter+=1
   # middle line
   for i in range(0,nelx+1):
      xV[counter]=(i)*hx
      yV[counter]=(j+1-0.5)*hy
      counter+=1

#top line
for i in range(0,nelx):
   xV[counter]=(i+1-0.5)*hx
   yV[counter]=Ly
   counter+=1

#np.savetxt('gridpoints.ascii',np.array([xV,yV]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

iconV =np.zeros((mV,nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0, counter] = (j)*(2*nelx+1)+i+1   -1
        iconV[1, counter] = iconV[0,counter]+nelx+1 
        iconV[2, counter] = iconV[0,counter]+2*nelx+1 
        iconV[3, counter] = iconV[0,counter]+nelx
        counter += 1

#for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#     print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if experiment==1 or experiment==3 or experiment==4 or experiment==5 or experiment==6:
    for i in range(0, nnp):
        if xV[i]<eps:
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i],experiment)
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],experiment)
        #end if
        if xV[i]>(Lx-eps):
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i],experiment)
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],experiment)
        #end if
        if yV[i]<eps:
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i],experiment)
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],experiment)
        #end if
        if yV[i]>(Ly-eps):
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i],experiment)
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],experiment)
        #end if
    #end for

if experiment==2:
    for i in range(0, nnp):
        if xV[i]<eps:
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
        #end if
        if xV[i]>(Lx-eps):
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
        #end if
        if yV[i]<eps:
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
        #end if
        if yV[i]>(Ly-eps):
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
        #end if
    #end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
b_mat2= np.zeros((4,ndofV*mV),dtype=np.float64)  # gradient matrix B 
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
p     = np.zeros(nel,dtype=np.float64)          # y-component velocity
#c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

constr  = np.zeros(NfemP,dtype=np.float64) 

NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for jq in range(0,nqperdim): 
        for iq in range(0,nqperdim): 

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            # calculate shape functions
            NNNV[0:mV]=NNV(rq,sq,sft)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,sft)
            dNNNVds[0:mV]=dNNVds(rq,sq,sft)

            # calculate jacobian matrix
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
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

            # construct b_mat matrix
            if formulation==1:
               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq,experiment)*weightq*jcob

            if formulation==2:
               for i in range(0,mV):
                   b_mat2[0:4, 2*i:2*i+2] = [[dNNNVdx[i],0.        ],
                                             [dNNNVdy[i],0.        ],
                                             [0.        ,dNNNVdx[i]],
                                             [0.        ,dNNNVdy[i]]]
               K_el+=b_mat2.T.dot(b_mat2)*eta(xq,yq,experiment)*weightq*jcob


            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq,experiment)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq,experiment)
                G_el[ndofV*i  ,0]-=dNNNVdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNNNVdy[i]*jcob*weightq

        #end for iq
    #end for jq

    #print (G_el)

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
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
    h_rhs[iel]+=h_el[0]

    constr[iel]=hx*hy

print("build FE matrix: %.3f s" % (time.time() - start))

#for i in range(NfemV):
#    print(i,G_mat[i,0:NfemP])
#G2 = np.zeros((4,4),dtype=np.float64)
#G2[0,:]=G_mat[6,:] 
#G2[1,:]=G_mat[11,:] 
#G2[2,:]=G_mat[13,:] 
#G2[3,:]=G_mat[16,:] 
#print (G2)
#ns = null_space(G2)
#print(np.round(ns,decimals=4))
#print(ns.shape)


######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

if pnormalise:
   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem+1,dtype=np.float64)         # right hand side of Ax=b
else:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

a_mat[0:NfemV,0:NfemV]=K_mat
a_mat[0:NfemV,NfemV:Nfem]=G_mat
a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
a_mat[Nfem,NfemV:Nfem]=constr
a_mat[NfemV:Nfem,Nfem]=constr


rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs


#a_mat[Nfem-1,:]=0
#rhs[Nfem-1]=1
#a_mat[Nfem-1,Nfem-1]=1

print("assemble blocks: %.3f s" % (time.time() - start))

######################################################################
# solve system
######################################################################
start = time.time()

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

######################################################################
# compute elemental strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  
ue  = np.zeros(nel,dtype=np.float64)  
ve  = np.zeros(nel,dtype=np.float64)  
visc = np.zeros(nel,dtype=np.float64)  
dens = np.zeros(nel,dtype=np.float64)  

#u[:]=xV[:]
#v[:]=yV[:]

for iel in range(0,nel):

    ue[iel]=(u[iconV[0,iel]]+u[iconV[1,iel]]+u[iconV[2,iel]]+u[iconV[3,iel]])/4.
    ve[iel]=(v[iconV[0,iel]]+v[iconV[1,iel]]+v[iconV[2,iel]]+v[iconV[3,iel]])/4.

    rq = 0.0
    sq = 0.0
    wq = 2.0 * 2.0

    NNNV[0:mV]=NNV(rq,sq,sft)
    dNNNVdr[0:mV]=dNNVdr(rq,sq,sft)
    dNNNVds[0:mV]=dNNVds(rq,sq,sft)

    jcb=np.zeros((2,2),dtype=np.float64)
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

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
    visc[iel]=eta(xc[iel],yc[iel],experiment)
    dens[iel]=by(xc[iel],yc[iel],experiment)
#end for

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("     -> visc (m,M) %.4f %.4f " %(np.min(visc),np.max(visc)))
print("     -> dens (m,M) %.4f %.4f " %(np.min(dens),np.max(dens)))

#np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
#np.savetxt('velocity_el.ascii',np.array([xc,yc,ue,ve]).T,header='# xc,yc,ue,ve')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute nodal pressure
######################################################################

q=np.zeros(nnp,dtype=np.float64)  
count=np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):
    q[iconV[0,iel]]+=p[iel]
    q[iconV[1,iel]]+=p[iel]
    q[iconV[2,iel]]+=p[iel]
    q[iconV[3,iel]]+=p[iel]
    count[iconV[0,iel]]+=1
    count[iconV[1,iel]]+=1
    count[iconV[2,iel]]+=1
    count[iconV[3,iel]]+=1
#end for

q=q/count

#np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

######################################################################
# compute error and vrms
######################################################################
start = time.time()

error_u = np.empty(nnp,dtype=np.float64)
error_v = np.empty(nnp,dtype=np.float64)
error_q = np.empty(nnp,dtype=np.float64)
error_p = np.empty(nel,dtype=np.float64)

for i in range(0,nnp): 
    error_u[i]=u[i]-velocity_x(xV[i],yV[i],experiment)
    error_v[i]=v[i]-velocity_y(xV[i],yV[i],experiment)
    error_q[i]=q[i]-pressure(xV[i],yV[i],experiment)
#end for

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(xc[i],yc[i],experiment)
#end for

area=0.
vrms=0.
errv=0.
errp=0.
for iel in range (0,nel):
    for jq in range(0,nqperdim): 
        for iq in range(0,nqperdim): 
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[0:mV]=NNV(rq,sq,sft)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,sft)
            dNNNVds[0:mV]=dNNVds(rq,sq,sft)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            jcob=np.linalg.det(jcb)
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
            vrms+=(uq**2+vq**2)*weightq*jcob
            area+=weightq*jcob
            errv+=((uq-velocity_x(xq,yq,experiment))**2\
                  +(vq-velocity_y(xq,yq,experiment))**2)*weightq*jcob
            errp+=(p[iel]-pressure(xq,yq,experiment))**2*weightq*jcob
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)
vrms=np.sqrt(vrms/(Lx*Ly))

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))
print("     -> nel= %6d ; vrms = %.6es" %(nel,vrms))
print("     -> area = %.6es" %(area))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################

if visu==1:

   filename = 'solution.vtu'
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % exx[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % eyy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % exy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='u' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % ue[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='v' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % ve[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % dens[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(  velocity_x(xV[i],yV[i],experiment),  velocity_y(xV[i],yV[i],experiment) , 0. ))
   vtufile.write("</DataArray>\n")
   #-------------
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" % q[i])
   vtufile.write("</DataArray>\n")
   #-------------
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q (th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" % pressure(xV[i],yV[i],experiment))
   vtufile.write("</DataArray>\n")
   #-------------
   vtufile.write("<DataArray type='Float32'  Name='u (error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" % error_u[i])
   vtufile.write("</DataArray>\n")
   #-------------
   vtufile.write("<DataArray type='Float32'  Name='v (error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" % error_v[i])
   vtufile.write("</DataArray>\n")
   #-------------
   vtufile.write("<DataArray type='Float32'  Name='q (error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" % error_q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel]))
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

   #####################################################################
   # build Q1 visualisation grid and interp velocity on it
   #####################################################################

   nnp2=(nelx+1)*(nely+1)

   xV_2=np.empty(nnp2,dtype=np.float64)  # x coordinates
   yV_2=np.empty(nnp2,dtype=np.float64)  # y coordinates
   iconV_2 =np.zeros((mV,nel),dtype=np.int32)
   u_2 = np.zeros(nnp2,dtype=np.float64)  
   v_2 = np.zeros(nnp2,dtype=np.float64)  
   counter_2 = np.zeros(nnp2,dtype=np.float64)  

   counter=0
   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           xV_2[counter]=i*Lx/float(nelx)
           yV_2[counter]=j*Ly/float(nely)
           counter += 1
       #end for
   #end for

   counter=0
   for j in range(0,nely):
          for i in range(0,nelx):
           iconV_2[0, counter] = i + j * (nelx + 1)
           iconV_2[1, counter] = i + 1 + j * (nelx + 1)
           iconV_2[2, counter] = i + 1 + (j + 1) * (nelx + 1)
           iconV_2[3, counter] = i + (j + 1) * (nelx + 1)
           counter += 1
       #end for
   #end for

   for iel in range(0,nel):

      #----------------------------------  
      # lower left corner
      #----------------------------------  
      rq=-1 ; sq=-1
      NNNV[0:mV]=NNV(rq,sq,sft)
      uuu=0.0
      vvv=0.0
      for k in range(0,mV):
          uuu+=NNNV[k]*u[iconV[k,iel]]
          vvv+=NNNV[k]*v[iconV[k,iel]]
      u_2[iconV_2[0,iel]]+=uuu
      v_2[iconV_2[0,iel]]+=vvv
      counter_2[iconV_2[0,iel]]+=1

      #----------------------------------  
      # lower right corner
      #----------------------------------  
      rq=+1 ; sq=-1
      NNNV[0:mV]=NNV(rq,sq,sft)
      uuu=0.0
      vvv=0.0
      for k in range(0,mV):
          uuu+=NNNV[k]*u[iconV[k,iel]]
          vvv+=NNNV[k]*v[iconV[k,iel]]
      u_2[iconV_2[1,iel]]+=uuu
      v_2[iconV_2[1,iel]]+=vvv
      counter_2[iconV_2[1,iel]]+=1

      #----------------------------------  
      # upper right corner
      #----------------------------------  
      rq=+1 ; sq=+1
      NNNV[0:mV]=NNV(rq,sq,sft)
      uuu=0.0
      vvv=0.0
      for k in range(0,mV):
          uuu+=NNNV[k]*u[iconV[k,iel]]
          vvv+=NNNV[k]*v[iconV[k,iel]]
      u_2[iconV_2[2,iel]]+=uuu
      v_2[iconV_2[2,iel]]+=vvv
      counter_2[iconV_2[2,iel]]+=1

      #----------------------------------  
      # upper left corner
      #----------------------------------  
      rq=-1 ; sq=+1
      NNNV[0:mV]=NNV(rq,sq,sft)
      uuu=0.0
      vvv=0.0
      for k in range(0,mV):
          uuu+=NNNV[k]*u[iconV[k,iel]]
          vvv+=NNNV[k]*v[iconV[k,iel]]
      u_2[iconV_2[3,iel]]+=uuu
      v_2[iconV_2[3,iel]]+=vvv
      counter_2[iconV_2[3,iel]]+=1

   #end for

   u_2=u_2/counter_2
   v_2=v_2/counter_2

   np.savetxt('velocity_2.ascii',np.array([xV_2,yV_2,u_2,v_2]).T,header='# x,y,u,v')

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
