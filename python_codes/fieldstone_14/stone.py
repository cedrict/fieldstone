import numpy as np
import sys as sys
import time as clock 
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def basis_functions_V(r,s):
    N_0=0.25*(1.-r)*(1.-s)
    N_1=0.25*(1.+r)*(1.-s)
    N_2=0.25*(1.+r)*(1.+s)
    N_3=0.25*(1.-r)*(1.+s)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr_0=-0.25*(1.-s) 
    dNdr_1=+0.25*(1.-s) 
    dNdr_2=+0.25*(1.+s) 
    dNdr_3=-0.25*(1.+s) 
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds_0=-0.25*(1.-r)
    dNds_1=-0.25*(1.+r)
    dNds_2=+0.25*(1.+r)
    dNds_3=+0.25*(1.-r)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)

###############################################################################
# bench=1: D&H manufactured solution
# bench=2: Burman & Hansbo manufactured solution

bench=1

###############################################################################

def bx(x, y):
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if bench==2:
       val=0
    return val

def by(x, y):
    if bench==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if bench==2:
       val=0
    return val

###############################################################################

def velocity_x(x,y):
    if bench==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if bench==2:
       val=20*x*y**3
    return val

def velocity_y(x,y):
    if bench==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if bench==2:
       val=5*x**4-5*y**4
    return val

def pressure(x,y):
    if bench==1:
       val=x*(1.-x)-1./6.
    if bench==2:
       val=60*x**2*y-20*y**3-5
    return val

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 014 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = 32
   visu = 1
    
nnx=nelx+1         # number of nodes, x direction
nny=nely+1         # number of nodes, y direction
nn_V=nnx*nny       # number of nodes, total
nel=nelx*nely      # number of elements, total
Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nel         # number of pressure dofs
Nfem=Nfem_V+Nfem_P # total number of dofs

eta=1.  # dynamic viscosity 

pnormalise=True

Gscaling=eta/(Ly/nely)

debug=False

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*Lx/float(nelx)
        y_V[counter]=j*Ly/float(nely)
        counter += 1
    #end for
#end for

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx+1)
        icon_V[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

if debug:
    for iel in range (0,nel):
        print ("iel=",iel)
        print ("node 1",icon_V[0][iel],"at pos.",x[icon_V[0][iel]], y[icon_V[0][iel]])
        print ("node 2",icon_V[1][iel],"at pos.",x[icon_V[1][iel]], y[icon_V[1][iel]])
        print ("node 3",icon_V[2][iel],"at pos.",x[icon_V[2][iel]], y[icon_V[2][iel]])
        print ("node 4",icon_V[3][iel],"at pos.",x[icon_V[3][iel]], y[icon_V[3][iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=velocity_x(x_V[i],y_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=velocity_y(x_V[i],y_V[i])
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=velocity_x(x_V[i],y_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=velocity_y(x_V[i],y_V[i])
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=velocity_x(x_V[i],y_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=velocity_y(x_V[i],y_V[i])
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=velocity_x(x_V[i],y_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=velocity_y(x_V[i],y_V[i])
#end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

if pnormalise:
   A_fem=lil_matrix((Nfem+1,Nfem+1),dtype=np.float64) # matrix A 
   b_fem=np.zeros((Nfem+1),dtype=np.float64)          # right hand side 
   A_fem[Nfem,Nfem_V:Nfem]=1
   A_fem[Nfem_V:Nfem,Nfem]=1
else:
   A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64) # matrix A 
   b_fem=np.zeros(Nfem,dtype=np.float64)          # right hand side 

jcb=np.zeros((2,2),dtype=np.float64)
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)  # gradient matrix B 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) # a
#c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64)  #b

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,1),dtype=np.float64)
    h_el=np.zeros((1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
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
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            # construct 3x8 b_mat matrix
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental a_mat matrix
            K_el+=B.T.dot(C.dot(B))*eta*JxWq

            # compute elemental rhs vector & G
            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*JxWq*bx(xq,yq)
                f_el[ndof_V*i+1]+=N_V[i]*JxWq*by(xq,yq)
                G_el[ndof_V*i  ,0]-=dNdx_V[i]*JxWq
                G_el[ndof_V*i+1,0]-=dNdy_V[i]*JxWq

        #end for jq
    #end for iq

    G_el*=Gscaling

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
               h_el[0]-=G_el[ikk,0]*bc_val_V[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            b_fem[m1]+=f_el[ikk]
            A_fem[m1,Nfem_V+iel]+=G_el[ikk,0]
            A_fem[Nfem_V+iel,m1]+=G_el[ikk,0]
        #end for
    #end for
    b_fem[Nfem_V+iel]+=h_el[0]

#end for

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]*Gscaling

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate 
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0.0
    sq=0.0
    weightq=2.0*2.0
    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    xq=np.dot(N_V,x_V[icon_V[:,iel]])
    yq=np.dot(N_V,y_V[icon_V[:,iel]])
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    x_e[iel]=N_V[:].dot(x_V[icon_V[:,iel]])
    y_e[iel]=N_V[:].dot(y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
#end for

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal pressure q
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    q[icon_V[0,iel]]+=p[iel]
    q[icon_V[1,iel]]+=p[iel]
    q[icon_V[2,iel]]+=p[iel]
    q[icon_V[3,iel]]+=p[iel]
    count[icon_V[0,iel]]+=1
    count[icon_V[1,iel]]+=1
    count[icon_V[2,iel]]+=1
    count[icon_V[3,iel]]+=1

q/=count

if debug: np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

print("project p on V nodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

error_u=np.zeros(nn_V,dtype=np.float64)
error_v=np.zeros(nn_V,dtype=np.float64)
error_q=np.zeros(nn_V,dtype=np.float64)
error_p=np.zeros(nel,dtype=np.float64)

for i in range(0,nn_V): 
    error_u[i]=u[i]-velocity_x(x_V[i],y_V[i])
    error_v[i]=v[i]-velocity_y(x_V[i],y_V[i])
    error_q[i]=q[i]-pressure(x_V[i],y_V[i])

for iel in range(0,nel): 
    error_p[iel]=p[iel]-pressure(x_e[iel],y_e[iel])

errv=0.
errp=0.
for iel in range (0,nel):
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
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*JxWq
            errp+=(p[iel]-pressure(xq,yq))**2*JxWq
        #end jq
    #end iq
#end iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution export to vtu format
###############################################################################
start=clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nel*4,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%10f %10f %10f \n" %(x_V[icon_V[i,iel]],y_V[icon_V[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%10f %10f %10f \n" %(u[icon_V[i,iel]],v[icon_V[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%10f %10f %10f \n" %(error_u[icon_V[i,iel]],error_v[icon_V[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%e \n" % q[icon_V[i,iel]])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%e \n" %p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%e \n" %error_p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%e \n" %exx[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%e \n" %eyy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%e \n" %exy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,m_V):
           vtufile.write("%e \n" %e[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   counter=0
   for iel in range(0,nel):
       vtufile.write("%d %d %d %d \n" %(counter,counter+1,counter+2,counter+3))
       counter+=4
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
   print("export to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
