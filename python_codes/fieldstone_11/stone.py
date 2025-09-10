import numpy as np
import sys as sys
import time as clock 
import scipy.sparse as sps

###############################################################################

def density(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
       val=2.
    else:
       val=1.
    return val

def viscosity(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
       val=1.e2
    else:
       val=1.
    return val

###############################################################################

def basis_functions_V(r,s,t):
    N_0=0.125*(1-r)*(1-s)*(1-t)
    N_1=0.125*(1+r)*(1-s)*(1-t)
    N_2=0.125*(1+r)*(1+s)*(1-t)
    N_3=0.125*(1-r)*(1+s)*(1-t)
    N_4=0.125*(1-r)*(1-s)*(1+t)
    N_5=0.125*(1+r)*(1-s)*(1+t)
    N_6=0.125*(1+r)*(1+s)*(1+t)
    N_7=0.125*(1-r)*(1+s)*(1+t)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7],dtype=np.float64)

def basis_functions_V_dr(r,s,t):
    dNdr_0=-0.125*(1-s)*(1-t) 
    dNdr_1=+0.125*(1-s)*(1-t)
    dNdr_2=+0.125*(1+s)*(1-t)
    dNdr_3=-0.125*(1+s)*(1-t)
    dNdr_4=-0.125*(1-s)*(1+t)
    dNdr_5=+0.125*(1-s)*(1+t)
    dNdr_6=+0.125*(1+s)*(1+t)
    dNdr_7=-0.125*(1+s)*(1+t)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7],dtype=np.float64)

def basis_functions_V_ds(r,s,t):
    dNds_0=-0.125*(1-r)*(1-t) 
    dNds_1=-0.125*(1+r)*(1-t)
    dNds_2=+0.125*(1+r)*(1-t)
    dNds_3=+0.125*(1-r)*(1-t)
    dNds_4=-0.125*(1-r)*(1+t)
    dNds_5=-0.125*(1+r)*(1+t)
    dNds_6=+0.125*(1+r)*(1+t)
    dNds_7=+0.125*(1-r)*(1+t)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7],dtype=np.float64)

def basis_functions_V_dt(r,s,t):
    dNdt_0=-0.125*(1-r)*(1-s) 
    dNdt_1=-0.125*(1+r)*(1-s)
    dNdt_2=-0.125*(1+r)*(1+s)
    dNdt_3=-0.125*(1-r)*(1+s)
    dNdt_4=+0.125*(1-r)*(1-s)
    dNdt_5=+0.125*(1+r)*(1-s)
    dNdt_6=+0.125*(1+r)*(1+s)
    dNdt_7=+0.125*(1-r)*(1+s)
    return np.array([dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7],dtype=np.float64)

###############################################################################

print(":::::::::::::::::::::::::::::")
print("::::::::: stone 11 ::::::::::")
print(":::::::::::::::::::::::::::::")

ndim=3   # number of dimensions 
m_V=8    # number of nodes making up an element
ndof_V=3 # number of velocity degrees of freedom per node

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
else:
   nelx =14   # do not exceed ~20 
   nely =nelx
   nelz =nelx
#end if

gx=0
gy=0
gz=-1

visu=1

debug=False

pnormalise=True

nel=nelx*nely*nelz              # number of elements, total
nn_V=(nelx+1)*(nely+1)*(nelz+1) # number of velocity nodes
nn_P=nel                        # number of pressure nodes
Nfem_V=nn_V*ndof_V              # number of velocity dofs
Nfem_P=nn_P                     # number of pressure dofs
Nfem=Nfem_V+Nfem_P              # total number of dofs

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

eps=1.e-10
sqrt3=np.sqrt(3.)

###############################################################################
#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("Nfem=",Nfem)
print(":::::::::::::::::::::::::::::")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
z_V=np.zeros(nn_V,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nelx+1):
    for j in range(0,nely+1):
        for k in range(0,nelz+1):
            x_V[counter]=i*hx
            y_V[counter]=j*hy
            z_V[counter]=k*hz
            counter+=1
        #end for
    #end for
#end for

print("grid points setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

nny=nely+1
nnz=nelz+1

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon_V[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon_V[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon_V[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon_V[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon_V[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon_V[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon_V[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon_V[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("build connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem_V,dtype=bool)       # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]=0
    if x_V[i]>(Lx-eps):
       bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]=0
    if y_V[i]<eps:
       bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]=0
    if y_V[i]>(Ly-eps):
       bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]=0
    if z_V[i]<eps:
       bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]=0
    if z_V[i]>(Lz-eps):
       bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]=0 
#end for

print("define b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# note that in practice all elements are cuboids so that the jacobian 
# and derived quantities could be computed directly as fct of hx,hy,hz
###############################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
f_rhs=np.zeros(Nfem_V,dtype=np.float64)          # right hand side f 
h_rhs=np.zeros(Nfem_P,dtype=np.float64)          # right hand side h 
B=np.zeros((6,ndof_V*m_V),dtype=np.float64)      # gradient matrix B 

C=np.zeros((6,6),dtype=np.float64) 
C[0,0]=2. ; C[1,1]=2. ; C[2,2]=2.
C[3,3]=1. ; C[4,4]=1. ; C[5,5]=1.

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 8 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:
            for kq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1.*1.*1.

                # calculate shape functions
                N_V=basis_functions_V(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)

                # calculate jacobian matrix
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])

                # calculate the determinant of the jacobian
                JxWq=np.linalg.det(jcb)*weightq

                # calculate inverse of the jacobian matrix
                jcbi = np.linalg.inv(jcb)

                # compute coords of quad point
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                zq=np.dot(N_V,z_V[icon_V[:,iel]])

                # compute dNdx, dNdy, dNdz
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

                # construct 3x8 b_mat matrix
                for i in range(0,m_V):
                    B[0:6,3*i:3*i+3] = [[dNdx_V[i],0.       ,0.       ],
                                        [0.       ,dNdy_V[i],0.       ],
                                        [0.       ,0.       ,dNdz_V[i]],
                                        [dNdy_V[i],dNdx_V[i],0.       ],
                                        [dNdz_V[i],0.       ,dNdx_V[i]],
                                        [0.       ,dNdz_V[i],dNdy_V[i]]]
                #end for

                K_el+=B.T.dot(C.dot(B))*viscosity(xq,yq,zq)*JxWq

                for i in range(0,m_V):
                    f_el[ndof_V*i+0]+=N_V[i]*density(xq,yq,zq)*gx*JxWq
                    f_el[ndof_V*i+1]+=N_V[i]*density(xq,yq,zq)*gy*JxWq
                    f_el[ndof_V*i+2]+=N_V[i]*density(xq,yq,zq)*gz*JxWq
                    G_el[ndof_V*i+0,0]-=dNdx_V[i]*JxWq
                    G_el[ndof_V*i+1,0]-=dNdy_V[i]*JxWq
                    G_el[ndof_V*i+2,0]-=dNdz_V[i]*JxWq
                #end for

            #end for kq
        #end for jq
    #end for iq

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
               #end for
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0
            #end if
        #end for
    #end for

    # assemble matrix K_mat and right hand side rhs
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
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
        #end for
    #end for
    h_rhs[iel]+=h_el[0,0]

#end for iel

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start=clock.time()

if pnormalise:
   A_fem = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   b_fem = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
   A_fem[Nfem,Nfem_V:Nfem]=1
   A_fem[Nfem_V:Nfem,Nfem]=1
else:
   A_fem = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   b_fem = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
#end if

#I tried this, it makes minimum difference
#del K_mat
#del G_mat

b_fem[0:Nfem_V]=f_rhs
b_fem[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))


#import matplotlib.pyplot as plt
#plt.spy(A_fem)
#plt.savefig('matrix.pdf',bbox_inches='tight')

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

u,v,w=np.reshape(sol[0:Nfem_V],(nn_V,3)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.6f %.6f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.6f %.6f " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.6f %.6f " %(np.min(w),np.max(w)))
print("     -> p (m,M) %.6f %.6f " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

#np.savetxt('velocity.ascii',np.array([x,y,z,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate 
###############################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
zc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
ezz=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
exz=np.zeros(nel,dtype=np.float64)  
eyz=np.zeros(nel,dtype=np.float64)  
eta=np.zeros(nel,dtype=np.float64)  
rho=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    tq=0.
    wq=2.*2.*2.

    N_V=basis_functions_V(rq,sq,tq)
    dNdr_V=basis_functions_V_dr(rq,sq,tq)
    dNds_V=basis_functions_V_ds(rq,sq,tq)
    dNdt_V=basis_functions_V_dt(rq,sq,tq)

    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
    jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
    jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
    jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])

    jcbi=np.linalg.inv(jcb)

    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
    dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    zc[iel]=np.dot(N_V,z_V[icon_V[:,iel]])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    ezz[iel]=np.dot(dNdz_V[:],w[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    exz[iel]=np.dot(dNdz_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],w[icon_V[:,iel]])*0.5
    eyz[iel]=np.dot(dNdz_V[:],v[icon_V[:,iel]])*0.5\
            +np.dot(dNdy_V[:],w[icon_V[:,iel]])*0.5

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
                        +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])

    eta[iel]=viscosity(xc[iel],yc[iel],zc[iel])
    rho[iel]=density(xc[iel],yc[iel],zc[iel])

#end for iel

print("     -> exx (m,M) %.6f %.6f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.6f %.6f " %(np.min(eyy),np.max(eyy)))
print("     -> ezz (m,M) %.6f %.6f " %(np.min(ezz),np.max(ezz)))
print("     -> exy (m,M) %.6f %.6f " %(np.min(exy),np.max(exy)))
print("     -> exz (m,M) %.6f %.6f " %(np.min(exz),np.max(exz)))
print("     -> eyz (m,M) %.6f %.6f " %(np.min(eyz),np.max(eyz)))
print("     -> eta (m,M) %.6f %.6f " %(np.min(eta),np.max(eta)))
print("     -> rho (m,M) %.6f %.6f " %(np.min(rho),np.max(rho)))

if debug:
   np.savetxt('strainrate.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
   np.savetxt('p.ascii',np.array([xc,yc,zc,p]).T,header='# xc,yc,p')

print("compute strainrate: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % eta[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % rho[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % e[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f %f %f %f %f %f\n" % (exx[iel],eyy[iel],ezz[iel],exy[iel],eyz[iel],exz[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel],
                                                   icon_V[4,iel],icon_V[5,iel],icon_V[6,iel],icon_V[7,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %12)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu: %.3f s" % (clock.time()-start))

print(":::::::::::::::::::::::::::::")
print("::::::::::: the end :::::::::")
print(":::::::::::::::::::::::::::::")

###############################################################################
