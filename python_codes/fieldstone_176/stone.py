import numpy as np
import sys as sys
import time as clock
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def basis_functions_V(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-s)
    dNdr1=+0.25*(1.-s)
    dNdr2=+0.25*(1.+s)
    dNdr3=-0.25*(1.+s)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################
# bx and by are the body force components

def bx(x,y):
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val

def by(x,y):
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val

###############################################################################
# analytical solution

def u_analytical(x,y):
    return x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)

def v_analytical(x,y):
    return -y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)

def p_analytical(x,y):
    return x*(1.-x)-1./6.

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 176 **********")
print("*******************************")

m_V=4    # number of nodes making up an element
ndof_V=2 # number of degrees of freedom per node

Lx=1. # horizontal extent of the domain 
Ly=1. # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 64
   nely = 64
   visu = 1
    
nn_V=(nelx+1)*(nely+1) # total number of (velocity) nodes
nel=nelx*nely          # total number of elements
Nfem_V=nn_V*ndof_V     # total number of velocity degrees of freedom
Nfem=Nfem_V            # total number of degrees of freedom

viscosity=1. # dynamic viscosity \eta
penalty=1.e7 # penalty coefficient value

new_assembly=True
new_bc=True

debug=False

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_V[counter]=i*Lx/float(nelx)
        y_V[counter]=j*Ly/float(nely)
        counter += 1

print("Setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# build connectivity array
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx+1)
        icon_V[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("Setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions. For this benchmark: no slip. 
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.

print("Setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

jcb=np.zeros((2,2),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64) 
dofs=np.zeros(ndof_V*m_V,dtype=np.int32) 
B= np.zeros((3,ndof_V*m_V),dtype=np.float64)
H=np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

if new_assembly:
   row=[] 
   col=[]
   A_fem=[]
else:
   A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)

time_bc=0.
time_ass=0.

for iel in range(0,nel):

    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)

    for k in range(0,m_V):
        dofs[k*ndof_V  ]=icon_V[k,iel]*ndof_V
        dofs[k*ndof_V+1]=icon_V[k,iel]*ndof_V+1

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq

            # coordinates of quad points
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])

            # compute dNdx & dNdy
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            # construct 3x8 B matrix
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental matrix
            A_el+=B.T.dot(C.dot(B))*viscosity*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                b_el[2*i  ]+=N_V[i]*bx(xq,yq)*JxWq
                b_el[2*i+1]+=N_V[i]*by(xq,yq)*JxWq

        # end for
    # end for

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    weightq=2.*2.

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

    for i in range(0,m_V):
        B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                          [0.       ,dNdy_V[i]],
                          [dNdy_V[i],dNdx_V[i]]]

    A_el+=B.T.dot(H.dot(B))*penalty*JxWq

    # apply boundary conditions
    start2=clock.time()
    if new_bc:
       for i_local,idof in enumerate(dofs):
           if bc_fix_V[idof]: 
              fixt=bc_val_V[idof]
              Aref=A_el[i_local,i_local]
              for j_local,jdof in enumerate(dofs):
                  b_el[j_local]-=A_el[j_local,i_local]*fixt
              #end for
              A_el[i_local,:]=0.
              A_el[:,i_local]=0.
              A_el[i_local,i_local]=Aref
              b_el[i_local]=Aref*fixt
           #end if
       #end for
    else:
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               m1 =ndof_V*icon_V[k1,iel]+i1
               if bc_fix_V[m1]: 
                  fixt=bc_val_V[m1]
                  ikk=ndof_V*k1+i1
                  aref=A_el[ikk,ikk]
                  for jkk in range(0,m_V*ndof_V):
                      b_el[jkk]-=A_el[jkk,ikk]*fixt
                      A_el[ikk,jkk]=0.
                      A_el[jkk,ikk]=0.
                  #end for
                  A_el[ikk,ikk]=aref
                  b_el[ikk]=aref*fixt
               #end if
           #end for
       #end for
    #end if
    time_bc+=clock.time()-start2

    # assemble matrix A_fem and right hand side b_fem 
    start2=clock.time()
    if new_assembly:
       for i_local,idof in enumerate(dofs):
           for j_local,jdof in enumerate(dofs):
               row.append(idof)
               col.append(jdof)
               A_fem.append(A_el[i_local,j_local])
           #end for
           b_fem[idof]+=b_el[i_local]
       #end for
    else:
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k1          +i1
               m1 =ndof_V*icon_V[k1,iel]+i1
               for k2 in range(0,m_V):
                   for i2 in range(0,ndof_V):
                       jkk=ndof_V*k2          +i2
                       m2 =ndof_V*icon_V[k2,iel]+i2
                       A_fem[m1,m2]+=A_el[ikk,jkk]
                   #end for
               #end for
               b_fem[m1]+=b_el[ikk]
           #end for
       #end for
    #end if
    time_ass+=clock.time()-start2
#end for iel

print('     -> time bc=',time_bc,Nfem)
print('     -> time assembly=',time_ass,Nfem)

start3=clock.time()
if new_assembly:
   A_fem=sps.csr_matrix((A_fem,(row,col)),shape=(Nfem,Nfem))
else:
   A_fem=sps.csr_matrix(A_fem)

print("Convert to csr format: %.5f s | Nfem= %d " % (clock.time()-start3,Nfem))

print("Build FE matrix: %.5f s | Nfem= %d" % (clock.time()-start,Nfem))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(A_fem,b_fem)

print("Solve linear system: %.5f s | Nfem= %d " % (clock.time()-start,Nfem))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

print("split vel into u,v: %.6f s | Nfem %d " % (clock.time()-start,Nfem))

###############################################################################
# we compute the pressure and strain rate components in the middle of the elts.
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0

    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)

    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)

    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    
    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p   (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

print("compute press & sr: %.5f s | Nfem: %d" % (clock.time()-start,Nfem))

###############################################################################
# compute error in L2 norm
###############################################################################
start=clock.time()

errv=0.
errp=0.
for iel in range(0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N_V=basis_functions_V(rq,sq)
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
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            errv+=((uq-u_analytical(xq,yq))**2+(vq-v_analytical(xq,yq))**2)*JxWq
            errp+=(p[iel]-p_analytical(xq,yq))**2*JxWq
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

if debug:
   np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
   np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
   np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
