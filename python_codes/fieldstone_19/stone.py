import numpy as np
import time as clock
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix

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

###############################################################################

def u_analytical(x,y):
    return x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)

def v_analytical(x,y):
    return -y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)

def p_analytical(x,y):
    return x*(1.-x)-1./6.

###############################################################################

def basis_functions_V(r,s):
    N1r=(-1    +r +9*r**2 - 9*r**3)/16
    N2r=(+9 -27*r -9*r**2 +27*r**3)/16
    N3r=(+9 +27*r -9*r**2 -27*r**3)/16
    N4r=(-1    -r +9*r**2 + 9*r**3)/16
    N1t=(-1    +s +9*s**2 - 9*s**3)/16
    N2t=(+9 -27*s -9*s**2 +27*s**3)/16
    N3t=(+9 +27*s -9*s**2 -27*s**3)/16
    N4t=(-1    -s +9*s**2 + 9*s**3)/16
    NV_00= N1r*N1t 
    NV_01= N2r*N1t 
    NV_02= N3r*N1t 
    NV_03= N4r*N1t 
    NV_04= N1r*N2t 
    NV_05= N2r*N2t 
    NV_06= N3r*N2t 
    NV_07= N4r*N2t 
    NV_08= N1r*N3t 
    NV_09= N2r*N3t 
    NV_10= N3r*N3t 
    NV_11= N4r*N3t 
    NV_12= N1r*N4t 
    NV_13= N2r*N4t 
    NV_14= N3r*N4t 
    NV_15= N4r*N4t 
    return np.array([NV_00,NV_01,NV_02,NV_03,\
                     NV_04,NV_05,NV_06,NV_07,\
                     NV_08,NV_09,NV_10,NV_11,\
                     NV_12,NV_13,NV_14,NV_15],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dN1rdr=( +1 +18*r -27*r**2)/16
    dN2rdr=(-27 -18*r +81*r**2)/16
    dN3rdr=(+27 -18*r -81*r**2)/16
    dN4rdr=( -1 +18*r +27*r**2)/16
    N1s=(-1    +s +9*s**2 - 9*s**3)/16
    N2s=(+9 -27*s -9*s**2 +27*s**3)/16
    N3s=(+9 +27*s -9*s**2 -27*s**3)/16
    N4s=(-1    -s +9*s**2 + 9*s**3)/16
    dNVdr_00= dN1rdr* N1s 
    dNVdr_01= dN2rdr* N1s 
    dNVdr_02= dN3rdr* N1s 
    dNVdr_03= dN4rdr* N1s 
    dNVdr_04= dN1rdr* N2s 
    dNVdr_05= dN2rdr* N2s 
    dNVdr_06= dN3rdr* N2s 
    dNVdr_07= dN4rdr* N2s 
    dNVdr_08= dN1rdr* N3s 
    dNVdr_09= dN2rdr* N3s 
    dNVdr_10= dN3rdr* N3s 
    dNVdr_11= dN4rdr* N3s 
    dNVdr_12= dN1rdr* N4s 
    dNVdr_13= dN2rdr* N4s 
    dNVdr_14= dN3rdr* N4s 
    dNVdr_15= dN4rdr* N4s 
    return np.array([dNVdr_00,dNVdr_01,dNVdr_02,dNVdr_03,\
                     dNVdr_04,dNVdr_05,dNVdr_06,dNVdr_07,\
                     dNVdr_08,dNVdr_09,dNVdr_10,dNVdr_11,\
                     dNVdr_12,dNVdr_13,dNVdr_14,dNVdr_15],dtype=np.float64)

def basis_functions_V_ds(r,s):
    N1r=(-1    +r +9*r**2 - 9*r**3)/16
    N2r=(+9 -27*r -9*r**2 +27*r**3)/16
    N3r=(+9 +27*r -9*r**2 -27*r**3)/16
    N4r=(-1    -r +9*r**2 + 9*r**3)/16
    dN1sds=( +1 +18*s -27*s**2)/16
    dN2sds=(-27 -18*s +81*s**2)/16
    dN3sds=(+27 -18*s -81*s**2)/16
    dN4sds=( -1 +18*s +27*s**2)/16
    dNVds_00= N1r*dN1sds 
    dNVds_01= N2r*dN1sds 
    dNVds_02= N3r*dN1sds 
    dNVds_03= N4r*dN1sds 
    dNVds_04= N1r*dN2sds 
    dNVds_05= N2r*dN2sds 
    dNVds_06= N3r*dN2sds 
    dNVds_07= N4r*dN2sds 
    dNVds_08= N1r*dN3sds 
    dNVds_09= N2r*dN3sds 
    dNVds_10= N3r*dN3sds 
    dNVds_11= N4r*dN3sds 
    dNVds_12= N1r*dN4sds 
    dNVds_13= N2r*dN4sds 
    dNVds_14= N3r*dN4sds 
    dNVds_15= N4r*dN4sds
    return np.array([dNVds_00,dNVds_01,dNVds_02,dNVds_03,\
                     dNVds_04,dNVds_05,dNVds_06,dNVds_07,\
                     dNVds_08,dNVds_09,dNVds_10,dNVds_11,\
                     dNVds_12,dNVds_13,dNVds_14,dNVds_15],dtype=np.float64)

def basis_functions_P(r,s):
    NP_0= 0.5*r*(r-1) * 0.5*s*(s-1)
    NP_1=    (1-r**2) * 0.5*s*(s-1)
    NP_2= 0.5*r*(r+1) * 0.5*s*(s-1)
    NP_3= 0.5*r*(r-1) *    (1-s**2)
    NP_4=    (1-r**2) *    (1-s**2)
    NP_5= 0.5*r*(r+1) *    (1-s**2)
    NP_6= 0.5*r*(r-1) * 0.5*s*(s+1)
    NP_7=    (1-r**2) * 0.5*s*(s+1)
    NP_8= 0.5*r*(r+1) * 0.5*s*(s+1)
    return np.array([NP_0,NP_1,NP_2,NP_3,NP_4,NP_5,NP_6,NP_7,NP_8],dtype=np.float64)

###############################################################################

eps=1.e-10

print("*******************************")
print("********** stone 019 **********")
print("*******************************")

m_V=16    # number of velocity nodes making up an element
m_P=9     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 8
   nely = 8
   visu = 1
    
nnx=3*nelx+1                 # number of elements, x direction
nny=3*nely+1                 # number of elements, y direction
nn_V=nnx*nny                 # number of V-nodes
nn_P=(2*nelx+1)*(2*nely+1)   # number of P-nodes 
nel=nelx*nely                # number of elements, total
Nfem_V=nn_V*ndof_V           # number of velocity dofs
Nfem_P=nn_P                  # number of pressure dofs
Nfem=Nfem_V+Nfem_P           # total number of dofs

hx=Lx/nelx
hy=Ly/nely

viscosity=1.  # dynamic viscosity

nq_per_dim=4

if nq_per_dim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nq_per_dim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

pnormalise=True

debug=False

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("hx,hy=",hx,hy)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/3.
        y_V[counter]=j*hy/3.
        counter += 1
    #end for
#end for

x_P=np.zeros(nn_P,dtype=np.float64)  # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        x_P[counter]=i*hx/2
        y_P[counter]=j*hy/2
        counter += 1
    #end for
#end for

if debug: np.savetxt('grid_V.ascii',np.array([x_V,y_V]).T,header='# x,y')
if debug: np.savetxt('grid_P.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time()-start))

#################################################################
# connectivity
#################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[ 0,counter]=(i)*3+1+(j)*3*nnx+0*nnx -1
        icon_V[ 1,counter]=(i)*3+2+(j)*3*nnx+0*nnx -1
        icon_V[ 2,counter]=(i)*3+3+(j)*3*nnx+0*nnx -1
        icon_V[ 3,counter]=(i)*3+4+(j)*3*nnx+0*nnx -1

        icon_V[ 4,counter]=(i)*3+1+(j)*3*nnx+1*nnx -1
        icon_V[ 5,counter]=(i)*3+2+(j)*3*nnx+1*nnx -1
        icon_V[ 6,counter]=(i)*3+3+(j)*3*nnx+1*nnx -1
        icon_V[ 7,counter]=(i)*3+4+(j)*3*nnx+1*nnx -1

        icon_V[ 8,counter]=(i)*3+1+(j)*3*nnx+2*nnx -1
        icon_V[ 9,counter]=(i)*3+2+(j)*3*nnx+2*nnx -1
        icon_V[10,counter]=(i)*3+3+(j)*3*nnx+2*nnx -1
        icon_V[11,counter]=(i)*3+4+(j)*3*nnx+2*nnx -1

        icon_V[12,counter]=(i)*3+1+(j)*3*nnx+3*nnx -1
        icon_V[13,counter]=(i)*3+2+(j)*3*nnx+3*nnx -1
        icon_V[14,counter]=(i)*3+3+(j)*3*nnx+3*nnx -1
        icon_V[15,counter]=(i)*3+4+(j)*3*nnx+3*nnx -1

        counter += 1
    #end for
#end for

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=(i)*2+1+(j)*2*(2*nelx+1) -1
        icon_P[1,counter]=(i)*2+2+(j)*2*(2*nelx+1) -1
        icon_P[2,counter]=(i)*2+3+(j)*2*(2*nelx+1) -1
        icon_P[3,counter]=(i)*2+1+(j)*2*(2*nelx+1)+(2*nelx+1) -1
        icon_P[4,counter]=(i)*2+2+(j)*2*(2*nelx+1)+(2*nelx+1) -1
        icon_P[5,counter]=(i)*2+3+(j)*2*(2*nelx+1)+(2*nelx+1) -1
        icon_P[6,counter]=(i)*2+1+(j)*2*(2*nelx+1)+(2*nelx+1)*2 -1
        icon_P[7,counter]=(i)*2+2+(j)*2*(2*nelx+1)+(2*nelx+1)*2 -1
        icon_P[8,counter]=(i)*2+3+(j)*2*(2*nelx+1)+(2*nelx+1)*2 -1
        counter += 1
    #end for
#end for

if debug:
   for iel in range (0,nel):
       print ("iel=",iel)
       print (icon_V[0,iel],"at pos.",x_V[icon_V[0,iel]],y_V[icon_V[0,iel]])
       print (icon_V[1,iel],"at pos.",x_V[icon_V[1,iel]],y_V[icon_V[1,iel]])
       print (icon_V[2,iel],"at pos.",x_V[icon_V[2,iel]],y_V[icon_V[2,iel]])
       print (icon_V[3,iel],"at pos.",x_V[icon_V[3,iel]],y_V[icon_V[3,iel]])
       print (icon_V[4,iel],"at pos.",x_V[icon_V[4,iel]],y_V[icon_V[4,iel]])
       print (icon_V[5,iel],"at pos.",x_V[icon_V[5,iel]],y_V[icon_V[5,iel]])
       print (icon_V[6,iel],"at pos.",x_V[icon_V[6,iel]],y_V[icon_V[6,iel]])
       print (icon_V[7,iel],"at pos.",x_V[icon_V[7,iel]],y_V[icon_V[7,iel]])
       print (icon_V[8,iel],"at pos.",x_V[icon_V[8,iel]],y_V[icon_V[8,iel]])
       print (icon_V[9,iel],"at pos.",x_V[icon_V[9,iel]],y_V[icon_V[9,iel]])
       print (icon_V[10,iel],"at pos.",x_V[icon_V[10,iel]],y_V[icon_V[10,iel]])
       print (icon_V[11,iel],"at pos.",x_V[icon_V[11,iel]],y_V[icon_V[11,iel]])
       print (icon_V[12,iel],"at pos.",x_V[icon_V[12,iel]],y_V[icon_V[12,iel]])
       print (icon_V[13,iel],"at pos.",x_V[icon_V[13,iel]],y_V[icon_V[13,iel]])
       print (icon_V[14,iel],"at pos.",x_V[icon_V[14,iel]],y_V[icon_V[14,iel]])
       print (icon_V[15,iel],"at pos.",x_V[icon_V[15,iel]],y_V[icon_V[15,iel]])
   for iel in range (0,nel):
       print ("iel=",iel)
       print (icon_P[0,iel],"at pos.",x_P[icon_P[0,iel]],y_P[icon_P[0,iel]])
       print (icon_P[1,iel],"at pos.",x_P[icon_P[1,iel]],y_P[icon_P[1,iel]])
       print (icon_P[2,iel],"at pos.",x_P[icon_P[2,iel]],y_P[icon_P[2,iel]])
       print (icon_P[3,iel],"at pos.",x_P[icon_P[3,iel]],y_P[icon_P[3,iel]])
       print (icon_P[4,iel],"at pos.",x_P[icon_P[4,iel]],y_P[icon_P[4,iel]])
       print (icon_P[5,iel],"at pos.",x_P[icon_P[5,iel]],y_P[icon_P[5,iel]])
       print (icon_P[6,iel],"at pos.",x_P[icon_P[6,iel]],y_P[icon_P[6,iel]])
       print (icon_P[7,iel],"at pos.",x_P[icon_P[7,iel]],y_P[icon_P[7,iel]])
       print (icon_P[8,iel],"at pos.",x_P[icon_P[8,iel]],y_P[icon_P[8,iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))

#################################################################
# define boundary conditions
#################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    #end if
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    #end if
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    #end if
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    #end if
#end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

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
constr= np.zeros(Nfem_P,dtype=np.float64)         # constraint matrix/vector
B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
N_mat = np.zeros((3,m_P),dtype=np.float64) # matrix  
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    h_el=np.zeros(m_P,dtype=np.float64)
    NNNP= np.zeros(m_P,dtype=np.float64)   

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):

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
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]


            # compute elemental K matrix
            K_el+=B.T.dot(C.dot(B))*viscosity*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*bx(xq,yq)*JxWq
                f_el[ndof_V*i+1]+=N_V[i]*by(xq,yq)*JxWq
            #end for

            # compute elemental G matrix
            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]
                N_mat[2,i]=0.
            G_el-=B.T.dot(N_mat)*JxWq

            NNNP[:]+=N_P[:]*JxWq

        #end for jq
    #end for iq

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
               #end for
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0
            #end if
        #end for
    #end for 

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
                #end for 
            #end for 
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                G_mat[m1,m2]+=G_el[ikk,jkk]
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNP[k2]
    #end for 

#end for iel

print("build FE matrix: %.3f s" % (clock.time()-start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start=clock.time()

if pnormalise:
   A_fem=np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   b_fem=np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
   A_fem[Nfem,Nfem_V:Nfem]=constr
   A_fem[Nfem_V:Nfem,Nfem]=constr
else:
   A_fem=np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   b_fem=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T

b_fem[0:Nfem_V]=f_rhs
b_fem[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))

######################################################################
# solve system
######################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate 
###############################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    weightq=2.*2.

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
# compute error
###############################################################################
start=clock.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):

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
            JxWq=np.linalg.det(jcb)*weightq

            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            pq=np.dot(N_P,p[icon_P[:,iel]])

            errv+=((uq-u_analytical(xq,yq))**2+(vq-v_analytical(xq,yq))**2)*JxWq
            errp+=(pq-p_analytical(xq,yq))**2*JxWq

        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %e ; errp= %e" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# interpolate pressure onto velocity grid points
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):

    #node 00 ------------------------------------
    q[icon_V[0,iel]]=p[icon_P[0,iel]]
    #node 01 ------------------------------------
    rq=-1./3.
    sq=-1.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[1,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 02 ------------------------------------
    rq=+1./3.
    sq=-1.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[2,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 03 ------------------------------------
    q[icon_V[3,iel]]=p[icon_P[2,iel]]
    #node 04 ------------------------------------
    rq=-1.
    sq=-1./3.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[4,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 05 ------------------------------------
    rq=-1./3.
    sq=-1./3.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[5,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 06 ------------------------------------
    rq=+1./3.
    sq=-1./3.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[6,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 07 ------------------------------------
    rq=+1.
    sq=-1./3.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[7,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 08 ------------------------------------
    rq=-1.
    sq=+1./3.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[8,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 09 ------------------------------------
    rq=-1./3.
    sq=+1./3.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[9,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 10 ------------------------------------
    rq=+1./3.
    sq=+1./3.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[10,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 11 ------------------------------------
    rq=+1.
    sq=+1./3.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[11,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 12 ------------------------------------
    q[icon_V[12,iel]]=p[icon_P[6,iel]]
    #node 13 ------------------------------------
    rq=-1./3.
    sq=+1.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[13,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 14 ------------------------------------
    rq=+1./3.
    sq=+1.
    N_P=basis_functions_P(rq,sq)
    q[icon_V[14,iel]]=np.dot(N_P,p[icon_P[:,iel]])
    #node 15 ------------------------------------
    q[icon_V[15,iel]]=p[icon_P[8,iel]]
#end for

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

print("project pressure on V-nodes: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
# the 16-node Q3 element does not exist in vtk so it is split into 
# 9 Q1 elements for visualisation purposes only.
###############################################################################
start=clock.time()

nel2=(nnx-1)*(nny-1)
icon_V2=np.zeros((4,nel2),dtype=np.int32)

counter = 0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        icon_V2[0,counter]=i+j*nnx
        icon_V2[1,counter]=i+1+j*nnx
        icon_V2[2,counter]=i+1+(j+1)*nnx
        icon_V2[3,counter]=i+(j+1)*nnx
        counter += 1
    #end for
#end for

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel2))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e \n" %p_analytical(x_V[i],y_V[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel2):
    vtufile.write("%d %d %d %d\n" %(icon_V2[0,iel],icon_V2[1,iel],icon_V2[2,iel],icon_V2[3,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel2):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel2):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
