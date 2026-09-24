import numpy as np
import sys as sys
import time as clock
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix

###############################################################################

def basis_functions_V(r,s):
    N0= (1-r-s)*(1-2*r-2.*s+ 3.*r*s)
    N1= r*(2*r-1+3*s-3.*r*s-3.*s**2 )
    N2= s*(2*s-1+3*r-3.*r**2-3.*r*s )
    N3= 4*(1-r-s)*r*(1-3.*s) 
    N4= 4*r*s*(-2.+3*r+3.*s)
    N5= 4*(1-r-s)*s*(1-3.*r) 
    N6= 27*(1-r-s)*r*s
    return np.array([N0,N1,N2,N3,N4,N5,N6],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= -3+4*r+7*s-6*r*s-3*s**2
    dNdr1= 4*r-1+3*s-6*r*s-3*s**2
    dNdr2= 3*s-6*r*s-3*s**2
    dNdr3= -8*r+24*r*s+4-16*s+12*s**2
    dNdr4= -8*s+24*r*s+12*s**2
    dNdr5= -16*s+24*r*s+12*s**2
    dNdr6= -54*r*s+27*s-27*s**2
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= -3+7*r+4*s-6*r*s-3*r**2
    dNds1= r*(3-3*r-6*s)
    dNds2= 4*s-1+3*r-3*r**2-6*r*s
    dNds3= -16*r+24*r*s+12*r**2
    dNds4= -8*r+12*r**2+24*r*s
    dNds5= 4-16*r-8*s+24*r*s+12*r**2
    dNds6= -54*r*s+27*r-27*r**2
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6],dtype=np.float64)

def basis_functions_P(r,s):
    N0=1-r-s
    N1=r
    N2=s
    return np.array([N0,N1,N2],dtype=np.float64)

###############################################################################

def bx(x, y):
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val

def by(x, y):
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val

def velocity_x(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def velocity_y(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pressure(x,y):
    val=x*(1.-x)-1./6.
    return val

###############################################################################

eps=1e-6

print("*******************************")
print("********** stone 046 **********")
print("*******************************")

m_V=7     # number of velocity nodes making up an element
m_P=3     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

Lx=1.
Ly=1.

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = 32
   visu = 1

nel=nelx*nely*2
nnx=2*nelx+1
nny=2*nely+1
nn_V=nnx*nny+nel
nn_P=nel*m_P

ndof_V=2

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nel*m_P     # number of pressure dofs
Nfem=Nfem_V+Nfem_P # total nb of dofs

print ('nnx   =',nnx)
print ('nny   =',nny)
print ('nn_V  =',nn_V)
print ('nn_P  =',nn_P)
print ('nel   =',nel)
print ('Nfem_V=',Nfem_V)
print ('Nfem_P=',Nfem_P)
print ('Nfem  =',Nfem)
print("-----------------------------")

debug=False

eta=1.

###############################################################################
# 6 point integration coeffs and weights 
###############################################################################

nq_per_el=6

qcoords_r=np.zeros(6,dtype=np.float64)  
qcoords_s=np.zeros(6,dtype=np.float64)  
qweights=np.zeros(6,dtype=np.float64)  

qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 

###############################################################################
# checking that all shape functions are 1 on their node and 
# zero elsewhere
#print ('node1:',NNV(0,0))
#print ('node2:',NNV(1,0))
#print ('node3:',NNV(0,1))
#print ('node4:',NNV(0.5,0))
#print ('node5:',NNV(0.5,0.5))
#print ('node6:',NNV(0,0.5))
#print ('node7:',NNV(1/3.,1/3.))

###############################################################################
# build velocity nodes coordinates and connectivity array 
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)
icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*Lx/(2*nelx) 
        y_V[counter]=j*Ly/(2*nely) 
        counter+=1

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
          # lower left triangle
          icon_V[0,counter]=(i)*2+1+(j)*2*nnx      -1  # 1 of q2
          icon_V[1,counter]=(i)*2+3+(j)*2*nnx      -1  # 3 of q2
          icon_V[2,counter]=(i)*2+1+(j)*2*nnx+nnx*2-1  # 7 of q2
          icon_V[3,counter]=(i)*2+2+(j)*2*nnx      -1  # 2 of q2
          icon_V[4,counter]=(i)*2+2+(j)*2*nnx+nnx  -1  # 5 of q2
          icon_V[5,counter]=(i)*2+1+(j)*2*nnx+nnx  -1  # 4 of q2
          icon_V[6,counter]=nnx*nny+counter
          counter=counter+1
          # upper right triangle
          icon_V[0,counter]=(i)*2+3+(j)*2*nnx+nnx*2-1  # 9 of Q2
          icon_V[1,counter]=(i)*2+1+(j)*2*nnx+nnx*2-1  # 7 of Q2
          icon_V[2,counter]=(i)*2+3+(j)*2*nnx      -1  # 3 of Q2
          icon_V[3,counter]=(i)*2+2+(j)*2*nnx+nnx*2-1  # 8 of Q2
          icon_V[4,counter]=(i)*2+2+(j)*2*nnx+nnx  -1  # 5 of Q2
          icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx  -1  # 6 of Q2
          icon_V[6,counter]=nnx*nny+counter
          counter=counter+1

for iel in range (0,nel): #bubble nodes
    x_V[nnx*nny+iel]=(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])/3.
    y_V[nnx*nny+iel]=(y_V[icon_V[0,iel]]+y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])/3.

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",icon_V[0,iel],"at pos.",xV[icon_V[0,iel]], yV[icon_V[0,iel]])
#    print ("node 1",icon_V[1,iel],"at pos.",xV[icon_V[1,iel]], yV[icon_V[1,iel]])
#    print ("node 2",icon_V[2,iel],"at pos.",xV[icon_V[2,iel]], yV[icon_V[2,iel]])
#    print ("node 3",icon_V[3,iel],"at pos.",xV[icon_V[3,iel]], yV[icon_V[3,iel]])
#    print ("node 4",icon_V[4,iel],"at pos.",xV[icon_V[4,iel]], yV[icon_V[4,iel]])
#    print ("node 5",icon_V[5,iel],"at pos.",xV[icon_V[5,iel]], yV[icon_V[5,iel]])
#    print ("node 6",icon_V[6,iel],"at pos.",xV[icon_V[6,iel]], yV[icon_V[6,iel]])

#print("icon_V (min/max): %d %d" %(np.min(icon_V[0,:]),np.max(icon_V[0,:])))
#print("icon_V (min/max): %d %d" %(np.min(icon_V[1,:]),np.max(icon_V[1,:])))
#print("icon_V (min/max): %d %d" %(np.min(icon_V[2,:]),np.max(icon_V[2,:])))
#print("icon_V (min/max): %d %d" %(np.min(icon_V[3,:]),np.max(icon_V[3,:])))
#print("icon_V (min/max): %d %d" %(np.min(icon_V[4,:]),np.max(icon_V[4,:])))
#print("icon_V (min/max): %d %d" %(np.min(icon_V[5,:]),np.max(icon_V[5,:])))

if debug: np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("grid and connectivity V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid (nodes and icon)
###############################################################################
start=clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)
x_P=np.zeros(Nfem_P,dtype=np.float64)
y_P=np.zeros(Nfem_P,dtype=np.float64)

counter=0
for iel in range(0,nel):
    x_P[counter]=x_V[icon_V[0,iel]]
    y_P[counter]=y_V[icon_V[0,iel]]
    icon_P[0,iel]=counter
    counter+=1
    x_P[counter]=x_V[icon_V[1,iel]]
    y_P[counter]=y_V[icon_V[1,iel]]
    icon_P[1,iel]=counter
    counter+=1
    x_P[counter]=x_V[icon_V[2,iel]]
    y_P[counter]=y_V[icon_V[2,iel]]
    icon_P[2,iel]=counter
    counter+=1

if debug: np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",icon_P[0,iel],"at pos.",xP[icon_P[0][iel]], yP[icon_P[0][iel]])
#    print ("node 1",icon_P[1,iel],"at pos.",xP[icon_P[1][iel]], yP[icon_P[1][iel]])
#    print ("node 2",icon_P[2,iel],"at pos.",xP[icon_P[2][iel]], yP[icon_P[2][iel]])

print("grid and connectivity P: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
    if x_V[i]>(Lx-eps):
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
    if y_V[i]<eps:
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
    if y_V[i]>(Ly-eps):
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.

print("boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
###############################################################################
start = clock.time()

jcb=np.zeros((2,2),dtype=np.float64)
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
        jcbi=np.linalg.inv(jcb)
        JxWq=np.linalg.det(jcb)*weightq
        area[iel]+=JxWq
    #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

A_fem=np.zeros((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64) 
K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
f_rhs=np.zeros(Nfem_V,dtype=np.float64)         # right hand side f 
h_rhs=np.zeros(Nfem_P,dtype=np.float64)         # right hand side h 
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)  # gradient matrix 
N_mat=np.zeros((3,m_P),dtype=np.float64)  # matrix  
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    for kq in range (0,nq_per_el):

        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
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
        xq=np.dot(N_V,x_V[icon_V[:,iel]])
        yq=np.dot(N_V,y_V[icon_V[:,iel]])
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

        # construct 3x8 b_mat matrix
        for i in range(0,m_V):
            B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                              [0.       ,dNdy_V[i]],
                              [dNdy_V[i],dNdx_V[i]]]

        # compute elemental a_mat matrix
        K_el+=B.T.dot(C.dot(B))*eta*JxWq

        # compute elemental rhs vector
        for i in range(0,m_V):
            f_el[ndof_V*i  ]+=N_V[i]*bx(xq,yq)*JxWq
            f_el[ndof_V*i+1]+=N_V[i]*by(xq,yq)*JxWq

        for i in range(0,m_P):
            N_mat[0,i]=N_P[i]
            N_mat[1,i]=N_P[i]
            N_mat[2,i]=0.

        G_el-=B.T.dot(N_mat)*JxWq

    #end for kq

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end if
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
            #end if
        #end for
    #end for

    # assemble matrices K,G and right hand sides f,h
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

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# assemble K, G, GT, f, h into A_fem and b_fem
###############################################################################
start=clock.time()

A_fem[0:Nfem_V,0:Nfem_V]=K_mat
A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
b_fem[0:Nfem_V]=f_rhs
b_fem[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

if debug: 
   np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
   np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("split solution into u,v,p: %.3f s" % (clock.time()-start))

###############################################################################
# normalise pressure
###############################################################################
start=clock.time()

pavrg=0
for iel in range(0,nel):
    for kq in range (0,nq_per_el):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        N_P=basis_functions_P(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        JxWq=np.linalg.det(jcb)*weightq
        pavrg+=N_P.dot(p[icon_P[:,iel]])*JxWq

p-=pavrg

if debug: np.savetxt('pressure_after.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental strainrate 
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  

rq = 0.33333
sq = 0.33333
for iel in range(0,nel):
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
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)
#end for

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug: np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# interpolate pressure onto velocity grid points
###############################################################################
#
#  02          #  02
#  ||\\        #  ||\\
#  || \\       #  || \\
#  ||  \\      #  ||  \\
#  05   04     #  ||   \\
#  || 06 \\    #  ||    \\
#  ||     \\   #  ||     \\
#  00==03==01  #  00======01
#
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)
temp=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    inode0=icon_V[0,iel] ; p0=p[icon_P[0,iel]]
    inode1=icon_V[1,iel] ; p1=p[icon_P[1,iel]]
    inode2=icon_V[2,iel] ; p2=p[icon_P[2,iel]]
    inode3=icon_V[3,iel]
    inode4=icon_V[4,iel]
    inode5=icon_V[5,iel]
    inode6=icon_V[6,iel]

    q[inode0]+=p0           ; temp[inode0]+=1
    q[inode1]+=p1           ; temp[inode1]+=1
    q[inode2]+=p2           ; temp[inode2]+=1
    q[inode3]+=(p0+p1)*0.5  ; temp[inode3]+=1
    q[inode4]+=(p1+p2)*0.5  ; temp[inode4]+=1
    q[inode5]+=(p0+p2)*0.5  ; temp[inode5]+=1
    q[inode6]+=(p0+p1+p2)/3 ; temp[inode6]+=1

q/=temp

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

print("interp p onto V nodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute error fields for plotting
###############################################################################
start=clock.time()

error_u=np.zeros(nn_V,dtype=np.float64)
error_v=np.zeros(nn_V,dtype=np.float64)
error_q=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V): 
    error_u[i]=u[i]-velocity_x(x_V[i],y_V[i])
    error_v[i]=v[i]-velocity_y(x_V[i],y_V[i])
    error_q[i]=q[i]-pressure(x_V[i],y_V[i])

print("compute error fields: %.3f s" % (clock.time()-start))

###############################################################################
# compute L2 errors
###############################################################################
start=clock.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for kq in range (0,nq_per_el):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        N_V=basis_functions_V(rq,sq)
        N_P=basis_functions_P(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        JxWq=np.linalg.det(jcb)*weightq
        xq=np.dot(N_V,x_V[icon_V[:,iel]])
        yq=np.dot(N_V,y_V[icon_V[:,iel]])
        uq=np.dot(N_V,u[icon_V[:,iel]])
        vq=np.dot(N_V,v[icon_V[:,iel]])
        errv+=((uq-velocity_x(xq,yq))**2+\
               (vq-velocity_y(xq,yq))**2)*JxWq
        pq=np.dot(N_P,p[icon_P[:,iel]])
        errp+=(pq-pressure(xq,yq))**2*JxWq
    # end for kq
# end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# export solution to vtu format
# the 7-node P2+ elt does not exist in vtk, but the 6-node one does: type=22. 
###############################################################################
start=clock.time()

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnx*nny,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p (el)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (q[icon_V[6,iel]]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error u' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%e \n" %error_u[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error v' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%e \n" %error_v[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error q' Format='ascii'> \n")
    for i in range(0,nnx*nny):
        vtufile.write("%e \n" %error_q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                              icon_V[3,iel],icon_V[4,iel],icon_V[5,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*6))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    print("export to vtu: %.3f s" % (clock.time()-start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
