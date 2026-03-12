import numpy as np
import math as math
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix,lil_matrix
import time as clock 
import random

###############################################################################

def basis_functions_V(r,s):
    N0=1-r-s-9*(1-r-s)*r*s 
    N1=  r  -9*(1-r-s)*r*s
    N2=    s-9*(1-r-s)*r*s
    N3=     27*(1-r-s)*r*s
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= -1-9*(1-2*r-s)*s 
    dNdr1=  1-9*(1-2*r-s)*s
    dNdr2=   -9*(1-2*r-s)*s
    dNdr3=   27*(1-2*r-s)*s
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= -1-9*(1-r-2*s)*r 
    dNds1=   -9*(1-r-2*s)*r
    dNds2=  1-9*(1-r-2*s)*r
    dNds3=   27*(1-r-2*s)*r
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

def basis_functions_P(r,s):
    N0=1-r-s
    N1=r
    N2=s
    return np.array([N0,N1,N2],dtype=np.float64)

def basis_functions_P_dr(r,s):
    return np.array([-1,1,0],dtype=np.float64)

def basis_functions_P_ds(r,s):
    return np.array([-1,0,1],dtype=np.float64)

###############################################################################

def basis_functions_T(r,s):
    N0=1-r-s
    N1=r
    N2=s
    return np.array([N0,N1,N2],dtype=np.float64)

def basis_functions_T_dr(r,s):
    return np.array([-1,1,0],dtype=np.float64)

def basis_functions_T_ds(r,s):
    return np.array([-1,0,1],dtype=np.float64)

###############################################################################

eps=1e-8

print("*******************************")
print("********** stone 051 **********")
print("*******************************")

ndim=2
m_V=4     # number of velocity nodes making up an element
m_P=3     # number of pressure nodes making up an element
m_T=3     # number of temperature nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

n=25

Lx=1 # horizontal extent of the domain 
Ly=1 # vertical extent of the domain 

nel=(n-1)**2
nn_V=int(n*(n+1)/2)+nel
nn_P=int(n*(n+1)/2)
nn_T=int(n*(n+1)/2)

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nn_P        # number of pressure dofs
Nfem_T=nn_T        # number of temperature dofs
Nfem=Nfem_V+Nfem_P # total number of dofs

visu=10

h=Lx/(n-1)

Ra=1e6

eta=1.

nstep=500

scramble=True
rand=False

if rand:
   deltax=h/4.
   deltay=h/4.
else:
   deltax=0.
   deltay=0.

relax=0.2


debug=False

###############################################################################
# quadrature points coordinates and weights 
###############################################################################

nq_per_el=6

qcoords_r=np.zeros(nq_per_el,dtype=np.float64)  
qcoords_s=np.zeros(nq_per_el,dtype=np.float64)  
qweights=np.zeros(nq_per_el,dtype=np.float64)  

if nq_per_el==3:
   qcoords_r[0]=1./6.; qcoords_s[0]=1./6.; qweights[0]=1./6.
   qcoords_r[1]=2./3.; qcoords_s[1]=1./6.; qweights[1]=1./6.
   qcoords_r[2]=1./6.; qcoords_s[2]=2./3.; qweights[2]=1./6.

if nq_per_el==6:
   qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
   qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
   qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
   qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
   qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
   qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 

if nq_per_el==7:
   qcoords_r[0]=0.1012865073235 ; qcoords_s[0]=0.1012865073235 ; qweights[0]=0.0629695902724 
   qcoords_r[1]=0.7974269853531 ; qcoords_s[1]=0.1012865073235 ; qweights[1]=0.0629695902724 
   qcoords_r[2]=0.1012865073235 ; qcoords_s[2]=0.7974269853531 ; qweights[2]=0.0629695902724 
   qcoords_r[3]=0.4701420641051 ; qcoords_s[3]=0.0597158717898 ; qweights[3]=0.0661970763942 
   qcoords_r[4]=0.4701420641051 ; qcoords_s[4]=0.4701420641051 ; qweights[4]=0.0661970763942 
   qcoords_r[5]=0.0597158717898 ; qcoords_s[5]=0.4701420641051 ; qweights[5]=0.0661970763942 
   qcoords_r[6]=0.3333333333333 ; qcoords_s[6]=0.3333333333333 ; qweights[6]=0.1125000000000 

###############################################################################

print('nel=',nel)
print('nn_V=',nn_V)
print('nn_T=',nn_T)
print('nn_P=',nn_P)
       
filename = 'Nusselt_bottom.ascii'
Nu_bot=open(filename,"w")

#################################################################
# grid point setup
#################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0    
for j in range(0,n):
    for i in range(0,n-j):
        if i==0 or i==n-j-1 or j==0 or j==n-1:
           x_V[counter]=i*h 
           y_V[counter]=j*h
        else:
           x_V[counter]=i*h + random.randrange(-100,100,1)/100*deltax
           y_V[counter]=j*h + random.randrange(-100,100,1)/100*deltay
        counter+=1

iel = 0
for iY in range(0,n-1):
    startLower = nn_P - int(0.5 * (n-iY) * (n-iY+1))+1
    startUpper = nn_P - int(0.5 * (n-iY-1) * (n-iY))+1
    for iX in range(0,n-iY-2):
        # add a square of two elements
        topleft     = startUpper+iX
        topright    = startUpper+iX+1
        bottomleft  = startLower+iX
        bottomright = startLower+iX+1
        if (scramble == True and iX%2 == 0 and iY%2==0):
            # switch the diagonal, in order to have more
            # mesh points with an odd number of elements adjacent to them
            icon_V[0:3,iel]   = bottomleft, bottomright, topright
            icon_V[0:3,iel+1] = bottomleft, topright, topleft
        else:
            icon_V[0:3,iel]   = bottomleft, bottomright, topleft
            icon_V[0:3,iel+1] = bottomright, topright, topleft
        iel = iel + 2
    # add the tail element
    icon_V[0:3,iel] = startLower+n-iY-2, startLower+n-iY-1, startUpper+n-iY-2
    iel = iel + 1
# add the top element
icon_V[0:3,nel-1] = nn_P-2, nn_P-1, nn_P

icon_V[0:3,0:nel] -=1

for iel in range(0,nel):
    icon_V[3,iel]=nn_P+iel

for iel in range (0,nel): #bubble nodes
    x_V[nn_P+iel]=(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])/3.
    y_V[nn_P+iel]=(y_V[icon_V[0,iel]]+y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])/3.

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconV[0,iel],"at pos.",x_V[iconV[0,iel]], y_V[iconV[0,iel]])
#    print ("node 1",iconV[1,iel],"at pos.",x_V[iconV[1,iel]], y_V[iconV[1,iel]])
#    print ("node 2",iconV[2,iel],"at pos.",x_V[iconV[2,iel]], y_V[iconV[2,iel]])
#    print ("node 3",iconV[3,iel],"at pos.",x_V[iconV[3,iel]], y_V[iconV[3,iel]])

#print("iconV (min/max): %d %d" %(np.min(iconV[0,:]),np.max(iconV[0,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[1,:]),np.max(iconV[1,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[2,:]),np.max(iconV[2,:])))
#print("iconV (min/max): %d %d" %(np.min(iconV[3,:]),np.max(iconV[3,:])))

print("grid and connectivity V: %.3f s" % (clock.time()-start))

#################################################################
# build pressure grid (nodes and icon)
#################################################################
start=clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)
x_P=np.zeros(nn_P,dtype=np.float64)     # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64)     # y coordinates

x_P[0:nn_P]=x_V[0:nn_P]
y_P[0:nn_P]=y_V[0:nn_P]

icon_P[0:m_P,0:nel]=icon_V[0:m_P,0:nel]

if debug:
   np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')
   for iel in range (0,nel):
       print ("iel=",iel)
       print ("node 0",icon_P[0,iel],"at pos.",x_P[icon_P[0,iel]], y_P[icon_P[0,iel]])
       print ("node 1",icon_P[1,iel],"at pos.",x_P[icon_P[1,iel]], y_P[icon_P[1,iel]])
       print ("node 2",icon_P[2,iel],"at pos.",x_P[icon_P[2,iel]], y_P[icon_P[2,iel]])

print("grid and connectivity P: %.3f s" % (clock.time()-start))

#################################################################
# build temperature grid (nodes and icon)
#################################################################
start=clock.time()

icon_T=np.zeros((m_T,nel),dtype=np.int32)
x_T=np.zeros(nn_T,dtype=np.float64)     # x coordinates
y_T=np.zeros(nn_T,dtype=np.float64)     # y coordinates

x_T[0:nn_T]=x_V[0:nn_T]
y_T[0:nn_T]=y_V[0:nn_T]

icon_T[0:m_T,0:nel]=icon_V[0:m_T,0:nel]

if debug: np.savetxt('gridT.ascii',np.array([x_T,y_T]).T,header='# x,y')

print("grid and connectivity T: %.3f s" % (clock.time()-start))

#################################################################
# define velocity boundary conditions
#################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0. 
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0. 
    if abs(x_V[i]+y_V[i]-1)<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0. 
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0. 
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0. 
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0. 

print("setup: boundary conditions: %.3f s" % (clock.time() - start))

#################################################################
# define temperature boundary conditions
#################################################################
start=clock.time()

bc_fix_T=np.zeros(Nfem_T,dtype=bool)  # boundary condition, yes/no
bc_val_T=np.zeros(Nfem_T,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_T):
    if x_T[i]/Lx<eps:
       bc_fix_T[i] = True ; bc_val_T[i] = 0. 
    if y_T[i]/Ly<eps:
       bc_fix_T[i] = True ; bc_val_T[i] = 2*(1.-np.cos(2*np.pi*x_T[i]))

print("setup: boundary conditions T: %.3f s" % (clock.time()-start))

#################################################################
# compute area of elements
#################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for kq in range(0,nq_per_el):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        N_V=basis_functions_V(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        JxWq=np.linalg.det(jcb)*weightq
        area[iel]+=JxWq
    #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))

#################################################################
# define temperature field
#################################################################
start=clock.time()

T=np.zeros(nn_T,dtype=np.float64) 
T_old=np.zeros(nn_T,dtype=np.float64) 

for i in range(0,nn_T):
    if y_T[i]/Ly<eps:
       T[i]=2*(1.-np.cos(2*np.pi*x_T[i]))

T_old[:]=T[:]

if debug: np.savetxt('temperature_init.ascii',np.array([x_T,y_T,T]).T,header='# x,y,T')

print("define initial temperature: %.3f s" % (clock.time()-start))

#==============================================================================
#==============================================================================
# timestepping
#==============================================================================
#==============================================================================

vrms=np.zeros(nstep,dtype=np.float64) 
avrgT=np.zeros(nstep,dtype=np.float64) 

u_old=np.zeros(nn_V,dtype=np.float64)      # x-component velocity
v_old=np.zeros(nn_V,dtype=np.float64)      # y-component velocity
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for istep in range(0,nstep):

    print("--------------------------------------------------------")
    print("----------- istep=",istep,"-----------------------------------")
    print("--------------------------------------------------------")

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start=clock.time()

    K_mat = np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
    G_mat = np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
    f_rhs = np.zeros(Nfem_V,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(Nfem_P,dtype=np.float64)         # right hand side h 
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64)  # gradient matrix 
    N_mat = np.zeros((3,m_P),dtype=np.float64) # matrix  

    for iel in range(0,nel):

        K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for kq in range (0,nq_per_el):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            N_T=basis_functions_T(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            jcbi=np.linalg.inv(jcb)
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            Tq=np.dot(N_T,T[icon_T[:,iel]])

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*eta*JxWq

            for i in range(0,m_V):
                f_el[ndof_V*i+1]+=N_V[i]*Ra*Tq*JxWq

            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]
                N_mat[2,i]=0.

            G_el-=B.T.dot(N_mat)*JxWq

        # end for kq

        # impose b.c. 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                if bc_fix_V[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m_V*ndof_V):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val_V[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                   G_el[ikk,:]=0

        # assemble matrix and right hand side
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        K_mat[m1,m2]+=K_el[ikk,jkk]
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    G_mat[m1,m2]+=G_el[ikk,jkk]
                f_rhs[m1]+=f_el[ikk]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            h_rhs[m2]+=h_el[k2]

    # end for iel

    print("build FE system V,P: %.3f s" % (clock.time()-start))


    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start=clock.time()

    A_fem = np.zeros((Nfem,Nfem),dtype=np.float64)
    A_fem[0:Nfem_V,0:Nfem_V]=K_mat
    A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
    A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T

    rhs=np.zeros(Nfem,dtype=np.float64)
    rhs[0:Nfem_V]=f_rhs
    rhs[Nfem_V:Nfem]=h_rhs

    #assign extra pressure b.c. to remove null space
    A_fem[Nfem-1,:]=0
    A_fem[:,Nfem-1]=0
    A_fem[Nfem-1,Nfem-1]=1
    rhs[Nfem-1]=0

    print("assemble blocks: %.3f s" % (clock.time()-start))

    ######################################################################
    # solve system
    ######################################################################
    start=clock.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),rhs)

    print("solve time: %.3f s" % (clock.time()-start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]

    print("     -> u (m,M) %4e %4e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %4e %4e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %4e %4e " %(np.min(p),np.max(p)))

    if debug:
       np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
       np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("split sol into u,v,p: %.3f s" % (clock.time()-start))

    #####################################################################
    # relaxation step
    #####################################################################

    u=relax*u+(1-relax)*u_old
    v=relax*v+(1-relax)*v_old

    #####################################################################
    # compute strain rate components
    #####################################################################
    start=clock.time()

    exx=np.zeros(nel,dtype=np.float64)  
    eyy=np.zeros(nel,dtype=np.float64)  
    exy=np.zeros(nel,dtype=np.float64)  
    x_e=np.zeros(nel,dtype=np.float64)  
    y_e=np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):

        rq=1./3.
        sq=1./3.
        weightq=0.5

        N_V=basis_functions_V(rq,sq)
        N_P=basis_functions_P(rq,sq)
        N_T=basis_functions_T(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
        y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
        exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
        eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
        exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
                +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    # end for iel

    print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

    if debug: np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

    print("compute el strain rate: %.3f s" % (clock.time()-start))

    #####################################################################
    # project strainrate onto velocity grid
    #####################################################################
    start=clock.time()

    r_V=[0.,1.,0.,1./3.]
    s_V=[0.,0.,1.,1./3.]

    exxn=np.zeros(nn_V,dtype=np.float64)
    eyyn=np.zeros(nn_V,dtype=np.float64)
    exyn=np.zeros(nn_V,dtype=np.float64)
    c=np.zeros(nn_V,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,m_V):
            dNdr_V=basis_functions_V_dr(r_V[i],s_V[i])
            dNds_V=basis_functions_V_ds(r_V[i],s_V[i])
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            e_xx=np.dot(dNdx_V,u[icon_V[:,iel]])
            e_yy=np.dot(dNdy_V,v[icon_V[:,iel]])
            e_xy=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
            exxn[icon_V[i,iel]]+=e_xx
            eyyn[icon_V[i,iel]]+=e_yy
            exyn[icon_V[i,iel]]+=e_xy
            c[icon_V[i,iel]]+=1.
        # end for i
    # end for iel

    exxn/=c
    eyyn/=c
    exyn/=c

    print("     -> exxn (m,M) %.4f %.4f " %(np.min(exxn),np.max(exxn)))
    print("     -> eyyn (m,M) %.4f %.4f " %(np.min(eyyn),np.max(eyyn)))
    print("     -> exyn (m,M) %.4f %.4f " %(np.min(exyn),np.max(exyn)))

    print("compute nod strain rate: %.3f s" % (clock.time()-start))

    #################################################################
    # build temperature matrix
    #################################################################
    start=clock.time()

    A_fem=np.zeros((Nfem_T,Nfem_T),dtype=np.float64) # FE matrix 
    b_fem=np.zeros(Nfem_T,dtype=np.float64)          # FE rhs 
    B=np.zeros((ndim,m_T),dtype=np.float64)
    N_mat=np.zeros((m_T,1),dtype=np.float64)       
    Tvect=np.zeros(m_T,dtype=np.float64)   

    for iel in range (0,nel):

        A_el=np.zeros((m_T,m_T),dtype=np.float64)
        b_el=np.zeros(m_T,dtype=np.float64)
        Ka=np.zeros((m_T,m_T),dtype=np.float64)
        Kd=np.zeros((m_T,m_T),dtype=np.float64)
        vel=np.zeros((1,ndim),dtype=np.float64)

        Tvect[:]=T[icon_T[:,iel]]

        for kq in range (0,nq_per_el):

            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            N_V=basis_functions_V(rq,sq)
            N_mat[:,0]=basis_functions_T(rq,sq)
            dNdr_T=basis_functions_T_dr(rq,sq)
            dNds_T=basis_functions_T_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
            jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
            jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
            jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            jcbi=np.linalg.inv(jcb)
            dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
            dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
            B[0,:]=dNdx_T[:]
            B[1,:]=dNdy_T[:]
            vel[0,0]=np.dot(N_V,u[icon_V[:,iel]])
            vel[0,1]=np.dot(N_V,v[icon_V[:,iel]])

            Kd=B.T.dot(B)*JxWq # diffusion matrix
            Ka=N_mat.dot(vel.dot(B))*JxWq # advection matrix

            A_el+=(Kd+Ka)

        # end for kq

        # apply boundary conditions
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_T):
                   m2=icon_T[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               # end for k2
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            # end if
        # end for k1

        # assemble matrix and right hand side 
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            for k2 in range(0,m_T):
                m2=icon_T[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            # end for k2
            b_fem[m1]+=b_el[k1]
        # end for k1

    # end for iel

    print("build FE system T: %.3f s" % (clock.time()-start))

    #################################################################
    # solve system
    #################################################################
    start=clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     -> T (m,M) %4f %.4f " %(np.min(T),np.max(T)))

    print("solve T time: %.3f s" % (clock.time()-start))

    #################################################################
    # relax
    #################################################################

    T=relax*T+(1-relax)*T_old

    #################################################################
    # compute vrms 
    #################################################################
    start=clock.time()

    for iel in range (0,nel):
        for kq in range (0,nq_per_el):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            N_T=basis_functions_T(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            vrms[istep]+=(uq**2+vq**2)*JxWq
        # end for kq
    # end for iel

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly))

    print("     -> vrms= %.6f" %(vrms[istep]))

    print("compute vrms: %.3fs" % (clock.time()-start))

    #################################################################
    # compute average temperature 
    #################################################################
    start=clock.time()

    for iel in range (0,nel):
        for kq in range (0,nq_per_el):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            N_mat[:,0]=basis_functions_T(rq,sq)
            dNdr_T=basis_functions_T_dr(rq,sq)
            dNds_T=basis_functions_T_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
            jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
            jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
            jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            Tq=np.dot(N_mat[:,0],T[icon_T[:,iel]])
            avrgT[istep]+=Tq*JxWq
        # end for kq
    # end for iel

    avrgT[istep]/=0.5

    print("     -> avrgT  = %.6f" %(avrgT[istep]))

    print("compute avrg T: %.3fs" % (clock.time()-start))

    #################################################################
    # compute average pressure 
    #################################################################
    start=clock.time()

    avrgp=0.
    for iel in range (0,nel):
        for kq in range (0,nq_per_el):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            N_P=basis_functions_P(rq,sq)
            dNdr_P=basis_functions_P_dr(rq,sq)
            dNds_P=basis_functions_P_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_P,x_P[icon_P[:,iel]])
            jcb[0,1]=np.dot(dNdr_P,y_P[icon_P[:,iel]])
            jcb[1,0]=np.dot(dNds_P,x_P[icon_P[:,iel]])
            jcb[1,1]=np.dot(dNds_P,y_P[icon_P[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            pq=np.dot(N_P,p[icon_P[:,iel]])
            avrgp+=pq*JxWq
        # end for kq
    # end for iel
    avrgp/=0.5

    p-=avrgp

    print("     -> p (m,M) %4e %4e " %(np.min(p),np.max(p)))

    print("compute avrg p: %.3fs" % (clock.time()-start))

    #####################################################################
    # compute nodal heat flux
    #####################################################################
    start=clock.time()

    qxn=np.zeros(nn_T,dtype=np.float64)
    qyn=np.zeros(nn_T,dtype=np.float64)
    c=np.zeros(nn_T,dtype=np.float64)

    for iel in range (0,nel):
        for i in range(0,m_T):
            N_T=basis_functions_T(rq,sq)
            dNdr_T=basis_functions_T_dr(rq,sq)
            dNds_T=basis_functions_T_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
            jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
            jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
            jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            jcbi=np.linalg.inv(jcb)
            dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
            dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
            q_x=np.dot(dNdx_T,T[icon_T[:,iel]])
            q_y=np.dot(dNdy_T,T[icon_T[:,iel]])
            qxn[icon_T[i,iel]]-=q_x #*hcond
            qyn[icon_T[i,iel]]-=q_y #*hcond
            c[icon_T[i,iel]]+=1.
        # end for i
    # end for iel

    qxn/=c
    qyn/=c

    print("compute nodal heat flux: %.3fs" % (clock.time()-start))

    #####################################################################

    if istep%visu==0:

       filename = 'temperature_hypotenuse_{:04d}.ascii'.format(istep) 
       temp_hyp=open(filename,"w")
       for i in range(0,nn_T):
           if abs(x_T[i]+y_T[i]-1)<eps:
              temp_hyp.write("%6e %6e %6e \n" %(x_T[i],y_T[i],T[i]))

       filename = 'heatflux_bottom_{:04d}.ascii'.format(istep) 
       hf_bot=open(filename,"w")
       for i in range(0,nn_T):
           if y_T[i]<eps:
              hf_bot.write("%6e %6e \n" %(x_T[i],qyn[i]))

    Nu=0.
    for iel in range(0,nel):
        if y_T[icon_T[0,iel]]<eps and y_T[icon_T[1,iel]]<eps:
           Nu+=h*(qyn[icon_T[0,iel]]+qyn[icon_T[1,iel]])/2.
    Nu_bot.write("%6e\n" %(Nu))
    Nu_bot.flush()

    #####################################################################
    # plot of solution. We export to P1. 
    #####################################################################
    start=clock.time()

    if istep%visu==0:
       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_P,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
       area.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       exx.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       eyy.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       exy.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
       divv=exx+eyy
       divv.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%10e %10e %10e \n" %(qxn[i],qyn[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
       exxn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
       eyyn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
       exyn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       p.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       T.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='fix u' Format='ascii'> \n")
       for i in range(0,nn_P):
           if bc_fix_V[i*ndof_V  ]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='fix v' Format='ascii'> \n")
       for i in range(0,nn_P):
           if bc_fix_V[i*ndof_V+1]:
              vtufile.write("%3e" %1.)
           else:
              vtufile.write("%3e" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='fix T' Format='ascii'> \n")
       for i in range(0,nn_T):
           if bc_fix_T[i]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*3))
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

       print("make vtu file: %.3fs" % (clock.time()-start))

    #####################################################################

    np.savetxt('vrms.ascii',np.array(vrms[0:istep]).T,header='# time,vrms')
    np.savetxt('avrgT.ascii',np.array(avrgT[0:istep]).T,header='# time,avrgT')

    #####################################################################

    u_old[:]=u[:]
    v_old[:]=v[:]
    T_old[:]=T[:]

#==============================================================================
#==============================================================================
# end time stepping loop
#==============================================================================
#==============================================================================

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
