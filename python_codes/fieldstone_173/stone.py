import numpy as np
import sys as sys
import scipy
import time as clock
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix

###############################################################################

def T_analytical(x,y,experiment):
    if experiment==1: return (2*x+y)**2
    if experiment==2: return (1-x**2)*(1-y**2)
    if experiment==3: return x**2-y**2
    if experiment==4: return 2*x*y 
    if experiment==5: return x**3-3*x*y**2

def qx_analytical(x,y,experiment):
    if experiment==1: return -4*(2*x+y)
    if experiment==2: return 2*x*(1-y**2)
    if experiment==3: return -2*x
    if experiment==4: return -2*y
    if experiment==5: return -3*x**2+3*y**2

def qy_analytical(x,y,experiment):
    if experiment==1: return -2*(2*x+y)
    if experiment==2: return 2*y*(1-x**2)
    if experiment==3: return 2*y
    if experiment==4: return -2*x
    if experiment==5: return +6*x*y

def rhs_f(x,y,experiment):
    if experiment==1: return -10
    if experiment==2: return 4-2*(x**2+y**2)
    if experiment==3: return 0 
    if experiment==4: return 0
    if experiment==5: return 0

###############################################################################

def basis_functions_T(r,s,order):
    if order==1:
       N_0=0.25*(1.-r)*(1.-s)
       N_1=0.25*(1.+r)*(1.-s)
       N_2=0.25*(1.+r)*(1.+s)
       N_3=0.25*(1.-r)*(1.+s)
       return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)
    if order==2:
       N_0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       N_1=    (1.-r**2) * 0.5*s*(s-1.)
       N_2= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       N_3= 0.5*r*(r-1.) *    (1.-s**2)
       N_4=    (1.-r**2) *    (1.-s**2)
       N_5= 0.5*r*(r+1.) *    (1.-s**2)
       N_6= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       N_7=    (1.-r**2) * 0.5*s*(s+1.)
       N_8= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

def basis_functions_T_dr(r,s,order):
    if order==1:
       dNdr_0=-0.25*(1.-s)
       dNdr_1=+0.25*(1.-s)
       dNdr_2=+0.25*(1.+s)
       dNdr_3=-0.25*(1.+s)
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)
    if order==2:
       dNdr_0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       dNdr_1=       (-2.*r) * 0.5*s*(s-1)
       dNdr_2= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       dNdr_3= 0.5*(2.*r-1.) *   (1.-s**2)
       dNdr_4=       (-2.*r) *   (1.-s**2)
       dNdr_5= 0.5*(2.*r+1.) *   (1.-s**2)
       dNdr_6= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       dNdr_7=       (-2.*r) * 0.5*s*(s+1)
       dNdr_8= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

def basis_functions_T_ds(r,s,order):
    if order==1:
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.+r)
       dNds_3=+0.25*(1.-r)
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)
    if order==2:
       dNds_0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       dNds_1=    (1.-r**2) * 0.5*(2.*s-1.)
       dNds_2= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       dNds_3= 0.5*r*(r-1.) *       (-2.*s)
       dNds_4=    (1.-r**2) *       (-2.*s)
       dNds_5= 0.5*r*(r+1.) *       (-2.*s)
       dNds_6= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       dNds_7=    (1.-r**2) * 0.5*(2.*s+1.)
       dNds_8= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)

###############################################################################

eps=1.e-10 

print("*******************************")
print("********** stone 173 **********")
print("*******************************")

ndim=2       # number of space dimensions
hcond=1.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density

debug=False

if int(len(sys.argv)==4):
   experiment=int(sys.argv[1])
   order     =int(sys.argv[2])
   nelx      =int(sys.argv[3])
else:
   experiment=1
   order=1
   nelx=8

if order==1: m=4 # number of nodes making up an element
if order==2: m=9


if experiment==1:  
   nely=nelx
   Lx=1.  
   Ly=1.  
   x0=0
   y0=0

if experiment==2 or experiment==3 or experiment==4 or experiment==5:
   nely=nelx
   Lx=2.
   Ly=2.
   x0=-1
   y0=-1

if order==1: nn_B=2*(nelx+nely) # nb of nodes on boundary 

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
nn_T=nnx*nny       # number of nodes
nel=nelx*nely     # number of elements, total
Nfem_T=nn_T        # Total number of degrees of temperature freedom

###############################################################################
# define 1d quadrature points and weights
###############################################################################

if order==1:
   nq_per_dim=2
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]
   r_T=np.array([-1, 1, 1,-1],dtype=np.float64)
   s_T=np.array([-1,-1, 1, 1],dtype=np.float64)

if order==2:
   nq_per_dim=3
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

###############################################################################

print('experiment =',experiment)
print('order      =',order)
print('nnx        =',nnx)
print('nny        =',nny)
print('nn_T       =',nn_T)
print('nn_B       =',nn_B)
print('nel        =',nel)
print('Nfem_T     =',Nfem_T)
print('nq_per_dim =',nq_per_dim)
print("*******************************")

###############################################################################
# grid point setup 
###############################################################################
start=clock.time()

x=np.zeros(nn_T,dtype=np.float64)
y=np.zeros(nn_T,dtype=np.float64)

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx/order+x0
        y[counter]=j*hy/order+y0
        if i==0 and j==0: SW=counter
        if i==nnx-1 and j==0: SE=counter
        if i==0 and j==nny-1: NW=counter
        if i==nnx-1 and j==nny-1: NE=counter
        if abs(x[counter]-Lx/2)<eps and j==nny-1: North=counter
        if abs(x[counter]-Lx/2)<eps and j==0: South=counter
        if abs(y[counter]-Ly/2)<eps and i==nnx-1: East=counter
        if abs(y[counter]-Ly/2)<eps and i==0: West=counter
        counter += 1
    #end for
#end for

if debug: np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("mesh (%.3fs)" % (clock.time()-start))

###############################################################################
# stretch mesh
###############################################################################

if experiment==1:
   for i in range(0,nn_T):
       x[i]=x[i]**0.66
       y[i]=y[i]**0.66

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon=np.zeros((m,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter]= i + j * (nelx + 1)
        icon[1,counter]= i + 1 + j * (nelx + 1)
        icon[2,counter]= i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter]= i + (j + 1) * (nelx + 1)
        counter+=1

print("connectivity (%.3fs)" % (clock.time()-start))

###############################################################################
# flag nodes and elements on boundary
###############################################################################
start=clock.time()

boundary_element=np.zeros(nel,dtype=bool)  
boundary_node=np.zeros(nn_T,dtype=bool)  

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        if i==0 or i==nelx-1 or j==0 or j==nely-1: 
           boundary_element[counter]=True 
        counter+=1

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        if i==0 or i==nnx-1 or j==0 or j==nny-1: 
           boundary_node[counter]=True 
        counter+=1

print("flag boundary nodes & elements: %.3f s" % (clock.time()-start))

###############################################################################
# compute B_to_G and G_to_B arrays
###############################################################################
start=clock.time()

B_to_G=np.zeros((nn_B),dtype=np.int32) ; B_to_G[:]=-1
G_to_B=np.zeros((nn_T),dtype=np.int32) ; G_to_B[:]=-1

counter=0

for i in range(0,nnx): 
        inode=i
        G_to_B[inode]=counter
        B_to_G[counter]=inode
        counter+=1

counter-=1
for i in range(0,nny):
        inode=nnx*(i+1)-1
        G_to_B[inode]=counter
        B_to_G[counter]=inode
        counter+=1
         
counter-=1
for i in range(0,nnx): 
        inode=nn_T-i-1
        G_to_B[inode]=counter
        B_to_G[counter]=inode
        counter+=1

counter-=1
for i in range(0,nny-1):
        inode=nn_T-(i+1)*nnx
        G_to_B[inode]=counter
        B_to_G[counter]=inode
        counter+=1

print("compute B_to_G, G_to_B: %.3f s" % (clock.time()-start))

###############################################################################
# compute s coordinate along boundary
###############################################################################

s=np.zeros(nn_B,dtype=np.float64) 

for k in range(1,nn_B):
    s[k]=s[k-1]+np.sqrt( (x[B_to_G[k]]-x[B_to_G[k-1]])**2+(y[B_to_G[k]]-y[B_to_G[k-1]])**2   )

###############################################################################
# compute normal to domain at the nodes
# this is also implemented in stone 151
###############################################################################
start=clock.time()

nx=np.zeros(nn_T,dtype=np.float64) 
ny=np.zeros(nn_T,dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):
    if boundary_element[iel]: 
       for iq in range(0,nq_per_dim):
           for jq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               N_T=basis_functions_T(rq,sq,order)
               dNdr_T=basis_functions_T_dr(rq,sq,order)
               dNds_T=basis_functions_T_ds(rq,sq,order)
               jcb[0,0]=np.dot(dNdr_T,x[icon[:,iel]])
               jcb[0,1]=np.dot(dNdr_T,y[icon[:,iel]])
               jcb[1,0]=np.dot(dNds_T,x[icon[:,iel]])
               jcb[1,1]=np.dot(dNds_T,y[icon[:,iel]])
               jcbi=np.linalg.inv(jcb)
               JxWq=np.linalg.det(jcb)*weightq
               dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
               dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
               for k in range(0,m):
                   if boundary_node[icon[k,iel]]:
                      nx[icon[k,iel]]+=dNdx_T[k]*JxWq
                      ny[icon[k,iel]]+=dNdy_T[k]*JxWq
           #end for
       #end for
    #end if
#end for

for i in range(0,nn_T):
    if boundary_node[i]:
       norm=np.sqrt(nx[i]**2+ny[i]**2)
       nx[i]/=norm
       ny[i]/=norm

print("compute normal: %.3f s" % (clock.time()-start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start=clock.time()

bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

for i in range(0,nn_T):
    if (x[i]-x0)/Lx<eps:
       bc_fix_T[i]=True ; bc_val_T[i]=T_analytical(x[i],y[i],experiment)
    if (x[i]-x0)/Lx>(1-eps):
       bc_fix_T[i]=True ; bc_val_T[i]=T_analytical(x[i],y[i],experiment)
    if (y[i]-y0)/Ly<eps:
       bc_fix_T[i]=True ; bc_val_T[i]=T_analytical(x[i],y[i],experiment)
    if (y[i]-y0)/Ly>(1-eps):
       bc_fix_T[i]=True ; bc_val_T[i]=T_analytical(x[i],y[i],experiment)

print("boundary conditions (%.3fs)" % (clock.time()-start))

###############################################################################
# build FEM linear system
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem_T,Nfem_T),dtype=np.float64)
b_fem=np.zeros(Nfem_T,dtype=np.float64)    
B_mat=np.zeros((2,m),dtype=np.float64)   

for iel in range(0,nel):

        A_el=np.zeros((m,m),dtype=np.float64)
        b_el=np.zeros(m,dtype=np.float64)

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):

                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                N_T=basis_functions_T(rq,sq,order)
                dNdr_T=basis_functions_T_dr(rq,sq,order)
                dNds_T=basis_functions_T_ds(rq,sq,order)
                jcb[0,0]=np.dot(dNdr_T,x[icon[:,iel]])
                jcb[0,1]=np.dot(dNdr_T,y[icon[:,iel]])
                jcb[1,0]=np.dot(dNds_T,x[icon[:,iel]])
                jcb[1,1]=np.dot(dNds_T,y[icon[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
                dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
                xq=np.dot(N_T,x[icon[:,iel]])
                yq=np.dot(N_T,y[icon[:,iel]])
                B_mat[0,:]=dNdx_T[:]
                B_mat[1,:]=dNdy_T[:]
                A_el+=B_mat.T.dot(B_mat)*hcond*JxWq
                b_el+=N_T*rhs_f(xq,yq,experiment)*JxWq

            #end for jq
        #end for iq

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            #end if
        #end for

        # assemble matrix and right hand side
        for k1 in range(0,m):
            m1=icon[k1,iel]
            for k2 in range(0,m):
                m2=icon[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        #end for

#end for iel

print("build FEM matrix: %.3fs" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

print("solve T time: %.3f s" % (clock.time()-start))

###############################################################################
# compute heat flux - corner-to-node
###############################################################################
start=clock.time()

qx=np.zeros(nn_T,dtype=np.float64) 
qy=np.zeros(nn_T,dtype=np.float64) 
cc=np.zeros(nn_T,dtype=np.float64) 

for iel in range(0,nel):
    for k in range(0,m):
        rq=r_T[k]
        sq=s_T[k]
        inode=icon[k,iel]
        dNdr_T=basis_functions_T_dr(rq,sq,order)
        dNds_T=basis_functions_T_ds(rq,sq,order)
        jcb[0,0]=np.dot(dNdr_T,x[icon[:,iel]])
        jcb[0,1]=np.dot(dNdr_T,y[icon[:,iel]])
        jcb[1,0]=np.dot(dNds_T,x[icon[:,iel]])
        jcb[1,1]=np.dot(dNds_T,y[icon[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
        dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
        qx[inode]-=np.dot(dNdx_T,T[icon[:,iel]])
        qy[inode]-=np.dot(dNdy_T,T[icon[:,iel]])
        cc[inode]+=1
    #end for
#end for
qx/=cc
qy/=cc

qn=qx*nx+qy*ny

print("     -> qx (m,M) %.4f %.4f " %(np.min(qx),np.max(qx)))
print("     -> qy (m,M) %.4f %.4f " %(np.min(qy),np.max(qy)))
print("     -> qn (m,M) %.4f %.4f " %(np.min(qn),np.max(qn)))

print("compute heat flux: %.3f s" % (clock.time()-start))

###############################################################################

qx_anal=np.zeros(nn_T,dtype=np.float64) 
qy_anal=np.zeros(nn_T,dtype=np.float64) 
qn_anal=np.zeros(nn_T,dtype=np.float64) 
for i in range(nn_T):
    qx_anal[i]=qx_analytical(x[i],y[i],experiment)
    qy_anal[i]=qy_analytical(x[i],y[i],experiment)
    qn_anal[i]=qx_anal[i]*nx[i]+qy_anal[i]*ny[i]

qn_file=open('heat_flux_boundary_analytical.ascii',"w")
for k in range(0,nn_B):
    qn_file.write("%.4e %.4e %.4e %.4e %.4e %.4e \n" %(x[B_to_G[k]],\
                                                       y[B_to_G[k]],\
                                                       qx_anal[B_to_G[k]],\
                                                       qy_anal[B_to_G[k]],\
                                                       qn_anal[B_to_G[k]],\
                                                       s[k]))

qn_file=open('heat_flux_boundary_cornertonode.ascii',"w")
for k in range(0,nn_B):
    qn_file.write("%.4e %.4e %.4e %.4e \n" %(x[B_to_G[k]],\
                                             y[B_to_G[k]],\
                                             qn[B_to_G[k]],\
                                             s[k]))
qn_file.close()

qn_file=open('heat_flux_boundary_AB_cornertonode.ascii',"w")
for k in range(0,nnx):
    qn_file.write("%.4e %.4e %.4e \n" %(x[k],y[k],qn[k]))
qn_file.close()

###############################################################################
# compute heat flux w/ consistent boundary flux method
# internal numbering of Q1 is 
# 3--2
# |  |
# 0--1
# the order of k,l does not matter
###############################################################################
start=clock.time()

M=lil_matrix((nn_B,nn_B),dtype=np.float64) # matrix
Ml=lil_matrix((nn_B,nn_B),dtype=np.float64) # lumped matrix
rhs= np.zeros(nn_B,dtype=np.float64)   
Mel=np.zeros((2,2),dtype=np.float64) 

for iel in range(0,nel):
        if boundary_element[iel]:

           for face in range(0,4): # loop over faces
               k=face    ; inode=icon[k,iel]
               l=(k+1)%4 ; jnode=icon[l,iel]

               if boundary_node[inode] and boundary_node[jnode]: # both nodes are on boundary
                  if bc_fix_T[inode] and bc_fix_T[jnode]: # both nodes are fixed

                     # compute face length
                     h=np.sqrt( (x[icon[k,iel]]-x[icon[l,iel]])**2 + 
                                (y[icon[k,iel]]-y[icon[l,iel]])**2 )

                     #print('iel=',iel,'face=',face,'k=',k,'l=',l,'inode=',inode,'jnode=',jnode,
                     #        'h=',h,G_to_B[inode],G_to_B[jnode])

                     # compute Mel, assuming all sides are straight
                     Mel[0,0]=h/3  
                     Mel[0,1]=h/6  
                     Mel[1,0]=h/6  
                     Mel[1,1]=h/3  

                     # compute bel, integration over volume
                     bel=np.zeros(2,dtype=np.float64) 
                     for iq in range(0,nq_per_dim):
                         for jq in range(0,nq_per_dim):
                             rq=qcoords[iq]
                             sq=qcoords[jq]
                             weightq=qweights[iq]*qweights[jq]
                             # compute \int grad N . grad T
                             N_T=basis_functions_T(rq,sq,order)
                             dNdr_T=basis_functions_T_dr(rq,sq,order)
                             dNds_T=basis_functions_T_ds(rq,sq,order)
                             jcb[0,0]=np.dot(dNdr_T,x[icon[:,iel]])
                             jcb[0,1]=np.dot(dNdr_T,y[icon[:,iel]])
                             jcb[1,0]=np.dot(dNds_T,x[icon[:,iel]])
                             jcb[1,1]=np.dot(dNds_T,y[icon[:,iel]])
                             jcbi=np.linalg.inv(jcb)
                             JxWq=np.linalg.det(jcb)*weightq
                             xq=np.dot(N_T,x[icon[:,iel]])
                             yq=np.dot(N_T,y[icon[:,iel]])
                             dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
                             dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
                             dTdxq=np.dot(dNdx_T,T[icon[:,iel]])
                             dTdyq=np.dot(dNdy_T,T[icon[:,iel]])
                             bel[0]-=(dNdx_T[k]*dTdxq+dNdy_T[k]*dTdyq)*JxWq
                             bel[1]-=(dNdx_T[l]*dTdxq+dNdy_T[l]*dTdyq)*JxWq
                             # compute \int N S 
                             bel[0]+=(N_T[k]*rhs_f(xq,yq,experiment))*JxWq
                             bel[1]+=(N_T[l]*rhs_f(xq,yq,experiment))*JxWq
                         #end for jq
                     #end for iq

                     #assembly                                          
                     M[G_to_B[inode],G_to_B[inode]]+=Mel[0,0]
                     M[G_to_B[inode],G_to_B[jnode]]+=Mel[0,1]
                     M[G_to_B[jnode],G_to_B[inode]]+=Mel[1,0]
                     M[G_to_B[jnode],G_to_B[jnode]]+=Mel[1,1]
                     rhs[G_to_B[inode]]+=bel[0]
                     rhs[G_to_B[jnode]]+=bel[1]
                     Ml[G_to_B[inode],G_to_B[inode]]+=Mel[0,0]
                     Ml[G_to_B[inode],G_to_B[inode]]+=Mel[0,1]
                     Ml[G_to_B[jnode],G_to_B[jnode]]+=Mel[1,0]
                     Ml[G_to_B[jnode],G_to_B[jnode]]+=Mel[1,1]

                  #end if bc_fix                 
               #end if boundary_node   
           #end for side
        #end if boundary_element                 
#end for iel                    

# solve linear systems
qn_CBF=sps.linalg.spsolve(sps.csr_matrix(M),rhs)
qn_CBF_lumped=sps.linalg.spsolve(sps.csr_matrix(Ml),rhs)

print("     -> qn (m,M) %.4f %.4f " %(np.min(qn_CBF),np.max(qn_CBF)))

import matplotlib.pyplot as plt
plt.spy(M,markersize=1)
plt.savefig('matrix.pdf', bbox_inches='tight')

qn_file=open('heat_flux_boundary_CBF.ascii',"w")
for k in range(0,nn_B):
    qn_file.write("%.4e %.4e %.4e %.4e %.4e \n" %(x[B_to_G[k]],\
                                                  y[B_to_G[k]],\
                                                  qn_CBF[k],\
                                                  qn_CBF_lumped[k],\
                                                  s[k]))
qn_file.close()

qn_file=open('heat_flux_boundary_AB_CBF.ascii',"w")
for k in range(0,nnx):
    qn_file.write("%.4e %.4e %.4e %.4e \n" %(x[B_to_G[k]],y[B_to_G[k]],qn_CBF[k],qn_CBF_lumped[k]))
qn_file.close()

print("compute heat flux (CBF): %.3f s" % (clock.time()-start))

###############################################################################
# compute boundary heat flux: partial method
# only for bottom boundary A-B
# D--C
# |  |
# A--B  we here assume all nodes on AB are fixed.
###############################################################################
start=clock.time()

M_AB=np.zeros((nnx,nnx),dtype=np.float64)
rhs_AB= np.zeros(nnx,dtype=np.float64)   
Mel=np.zeros((2,2),dtype=np.float64) 

#compute heat flux at point A for bc: qn_A
iel=0 ; rq=-1 ; sq=-1
dNdr_T=basis_functions_T_dr(rq,sq,order)
dNds_T=basis_functions_T_ds(rq,sq,order)
jcb[0,0]=np.dot(dNdr_T,x[icon[:,iel]])
jcb[0,1]=np.dot(dNdr_T,y[icon[:,iel]])
jcb[1,0]=np.dot(dNds_T,x[icon[:,iel]])
jcb[1,1]=np.dot(dNds_T,y[icon[:,iel]])
jcbi=np.linalg.inv(jcb)
dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
dTdyq=np.dot(dNdy_T[:],T[icon[:,iel]])
qn_A=-dTdyq
print('     -> qn_A=',qn_A)

#compute heat flux at point B for bc: qn_B
iel=nelx-1 ; rq=+1 ; sq=-1

dNdr_T=basis_functions_T_dr(rq,sq,order)
dNds_T=basis_functions_T_ds(rq,sq,order)
jcb[0,0]=np.dot(dNdr_T,x[icon[:,iel]])
jcb[0,1]=np.dot(dNdr_T,y[icon[:,iel]])
jcb[1,0]=np.dot(dNds_T,x[icon[:,iel]])
jcb[1,1]=np.dot(dNds_T,y[icon[:,iel]])
jcbi=np.linalg.inv(jcb)
dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
dTdyq=np.dot(dNdy_T[:],T[icon[:,iel]])
qn_B=-dTdyq
print('     -> qn_B=',qn_B)

for iel in range(0,nelx): # loop over 
        inode=icon[0,iel] 
        jnode=icon[1,iel]

        # compute face length
        h=np.sqrt( (x[icon[0,iel]]-x[icon[1,iel]])**2 + 
                   (y[icon[0,iel]]-y[icon[1,iel]])**2 )
        Mel[0,0]=h/3  
        Mel[0,1]=h/6  
        Mel[1,0]=h/6  
        Mel[1,1]=h/3  

        # compute bel, integration over volume
        bel=np.zeros(2,dtype=np.float64) 
        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                # compute \int grad N . grad T
                N_T=basis_functions_T(rq,sq,order)
                dNdr_T=basis_functions_T_dr(rq,sq,order)
                dNds_T=basis_functions_T_ds(rq,sq,order)
                jcb[0,0]=np.dot(dNdr_T,x[icon[:,iel]])
                jcb[0,1]=np.dot(dNdr_T,y[icon[:,iel]])
                jcb[1,0]=np.dot(dNds_T,x[icon[:,iel]])
                jcb[1,1]=np.dot(dNds_T,y[icon[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
                dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
                dTdxq=np.dot(dNdx_T,T[icon[:,iel]])
                dTdyq=np.dot(dNdy_T,T[icon[:,iel]])
                xq=np.dot(N_T,x[icon[:,iel]])
                yq=np.dot(N_T,y[icon[:,iel]])
                bel[0]-=(dNdx_T[0]*dTdxq+dNdy_T[0]*dTdyq)*JxWq
                bel[1]-=(dNdx_T[1]*dTdxq+dNdy_T[1]*dTdyq)*JxWq
                # compute \int N S 
                bel[0]+=(N_T[0]*rhs_f(xq,yq,experiment))*JxWq
                bel[1]+=(N_T[1]*rhs_f(xq,yq,experiment))*JxWq
            #end for jq
        #end for iq

        M_AB[inode,inode]+=Mel[0,0]
        M_AB[inode,jnode]+=Mel[0,1]
        M_AB[jnode,inode]+=Mel[1,0]
        M_AB[jnode,jnode]+=Mel[1,1]
        rhs_AB[inode]+=bel[0]
        rhs_AB[jnode]+=bel[1]

#end for iel                    

# apply b.c.
M_AB[0    ,:]=0 ; M_AB[    0,    0]=1 ; rhs_AB[    0]=-qn_A
M_AB[nnx-1,:]=0 ; M_AB[nnx-1,nnx-1]=1 ; rhs_AB[nnx-1]=-qn_B

# solve linear systems
qn_AB=sps.linalg.spsolve(sps.csr_matrix(M_AB),rhs_AB)

print("     -> qn on AB (m,M) %.4f %.4f " %(np.min(qn_AB),np.max(qn_AB)))

qn_file=open('heat_flux_boundary_AB_partial.ascii',"w")
for k in range(0,nnx):
    qn_file.write("%.4e %.4e %.4e %.4e \n" %(x[k],y[k],qn_AB[k],s[k]))
qn_file.close()

print("compute heat flux (partial CBF): %.3f s" % (clock.time()-start))

###############################################################################
# CBF2 approch: full matrix approach
# the difficulty here is to know which area to ascribe the corner nodes...
###############################################################################
start=clock.time()

Residual=np.zeros(Nfem_T,dtype=np.float64)    
B_mat=np.zeros((2,m),dtype=np.float64)   
area=np.zeros(nn_T,dtype=np.float64)
qn_newCBF=np.zeros(nn_T,dtype=np.float64)

for iel in range(0,nel):

        A_el=np.zeros((m,m),dtype=np.float64)
        b_el=np.zeros(m,dtype=np.float64)
        R_el=np.zeros(m,dtype=np.float64)

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                N_T=basis_functions_T(rq,sq,order)
                dNdr_T=basis_functions_T_dr(rq,sq,order)
                dNds_T=basis_functions_T_ds(rq,sq,order)
                jcb[0,0]=np.dot(dNdr_T,x[icon[:,iel]])
                jcb[0,1]=np.dot(dNdr_T,y[icon[:,iel]])
                jcb[1,0]=np.dot(dNds_T,x[icon[:,iel]])
                jcb[1,1]=np.dot(dNds_T,y[icon[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
                dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
                xq=np.dot(N_T,x[icon[:,iel]])
                yq=np.dot(N_T,y[icon[:,iel]])
                B_mat[0,:]=dNdx_T[:]
                B_mat[1,:]=dNdy_T[:]
                A_el+=B_mat.T.dot(B_mat)*hcond*JxWq
                b_el+=N_T*rhs_f(xq,yq,experiment)*JxWq
            #end for jq
        #end for iq

        R_el=A_el.dot(T[icon[:,iel]])-b_el

        # assemble residual vector 
        for k1 in range(0,m):
            Residual[icon[k1,iel]]+=R_el[k1]
        #end for

#end for iel

# loop over elements that are on the boundary.
# assign to each node half the length of the face
# it belongs to. Corner nodes get two contributions.
# In this case the sum of the area array is equal
# to the perimeter of the domain.
# this could be revisited. 

iel=0
for j in range(0,nely):
    for i in range(0,nelx):
        if i==0:
           hy=y[icon[3,iel]]-y[icon[0,iel]]
           area[icon[0,iel]]+=hy/2
           area[icon[3,iel]]+=hy/2
        if i==nelx-1:
           hy=y[icon[2,iel]]-y[icon[1,iel]]
           area[icon[1,iel]]+=hy/2
           area[icon[2,iel]]+=hy/2
        if j==0:
           hx=x[icon[1,iel]]-x[icon[0,iel]]
           area[icon[0,iel]]+=hx/2
           area[icon[1,iel]]+=hx/2
        if j==nely-1: 
           hx=x[icon[2,iel]]-x[icon[3,iel]]
           area[icon[2,iel]]+=hx/2
           area[icon[3,iel]]+=hx/2
        iel+=1
    #end for
#end for

print(sum(area))

for k in range(0,nn_T):
    if boundary_node[k]:
       qn_newCBF[k]=-Residual[k]/area[k]


qn_file=open('heat_flux_boundary_CBF2.ascii',"w")
for k in range(0,nn_B):
    qn_file.write("%.4e %.4e %.4e %.4e \n" %(x[B_to_G[k]],\
                                             y[B_to_G[k]],\
                                             qn_newCBF[B_to_G[k]],\
                                             s[k]))
qn_file.close()


qn_file=open('heat_flux_boundary_AB_CBF2.ascii',"w")
for k in range(0,nnx):
    qn_file.write("%e %e %e \n" %(x[k],y[k],qn_newCBF[k]))
qn_file.close()

print("     -> qn (m,M) %.4f %.4f " %(np.min(qn_newCBF),np.max(qn_newCBF)))

print("compute heat flux (CBF2): %.3f s" % (clock.time()-start))

###############################################################################
# visualisation 
###############################################################################
start=clock.time()

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_T,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_T):
    vtufile.write("%.4e %.4e %.4e \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
T.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Residual' Format='ascii'> \n")
Residual.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='qn' Format='ascii'> \n")
qn.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
area.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='G_to_B' Format='ascii'> \n")
G_to_B.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='T (analytical)' Format='ascii'> \n")
for i in range(0,nn_T):
    vtufile.write("%.4e \n" % (T_analytical(x[i],y[i],experiment)))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='T (error)' Format='ascii'> \n")
for i in range(0,nn_T):
    vtufile.write("%.4e \n" % (T[i]-T_analytical(x[i],y[i],experiment)))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
for i in range(0,nn_T):
    vtufile.write("%.4e %.4e %.4e \n" %(qx[i],qy[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='qn (CBF)' Format='ascii'> \n")
for i in range(0,nn_T):
    if (G_to_B[i]>0):
       vtufile.write("%.4e \n" % qn_CBF[G_to_B[i]])
    else:
       vtufile.write("%.4e \n" % 0. )
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='qn (new CBF)' Format='ascii'> \n")
qn_newCBF.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux (analytical)' Format='ascii'> \n")
for i in range(0,nn_T):
    vtufile.write("%.4e %.4e %.4e \n" %(qx_anal[i],qy_anal[i],0))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux (error)' Format='ascii'> \n")
for i in range(0,nn_T):
    vtufile.write("%.4e %.4e %.4e \n" %(qx[i]-qx_anal[i],qy[i]-qy_anal[i],0))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal' Format='ascii'> \n")
for i in range(0,nn_T):
    vtufile.write("%.4e %.4e %.4e \n" %(nx[i],ny[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='boundary' Format='ascii'> \n")
for i in range(0,nn_T):
    vtufile.write("%d \n" % int(boundary_node[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Int32' Name='boundary' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" % int(boundary_element[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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

print("export to files: %.3f s" % (clock.time() - start))

print('North',nelx,qn_anal[North],qn[North],qn_CBF[G_to_B[North]],qn_CBF_lumped[G_to_B[North]],qn_newCBF[North])
print('South',nelx,qn_anal[South],qn[South],qn_CBF[G_to_B[South]],qn_CBF_lumped[G_to_B[South]],qn_newCBF[South])
print('East',nelx,qn_anal[East],qn[East],qn_CBF[G_to_B[East]],qn_CBF_lumped[G_to_B[East]],qn_newCBF[East])
print('West',nelx,qn_anal[West],qn[West],qn_CBF[G_to_B[West]],qn_CBF_lumped[G_to_B[West]],qn_newCBF[West])

print('SWcorner',nelx,qn_anal[SW],qn[SW],qn_CBF[G_to_B[SW]],qn_CBF_lumped[G_to_B[SW]],qn_newCBF[SW])
print('SEcorner',nelx,qn_anal[SE],qn[SE],qn_CBF[G_to_B[SE]],qn_CBF_lumped[G_to_B[SE]],qn_newCBF[SE])
print('NWcorner',nelx,qn_anal[NW],qn[NW],qn_CBF[G_to_B[NW]],qn_CBF_lumped[G_to_B[NW]],qn_newCBF[NW])
print('NEcorner',nelx,qn_anal[NE],qn[NE],qn_CBF[G_to_B[NE]],qn_CBF_lumped[G_to_B[NE]],qn_newCBF[NE])

print('qn_sum',nelx,np.dot(qn_anal[boundary_node],area[boundary_node]),\
                    np.dot(qn[boundary_node],area[boundary_node]),\
                    np.dot(qn_CBF,area[boundary_node]),\
                    np.dot(qn_CBF_lumped,area[boundary_node]),\
                    np.dot(qn_newCBF[boundary_node],area[boundary_node]))
    
    
print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
