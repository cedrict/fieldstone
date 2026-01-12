import numpy as np
import sys as sys
import scipy.sparse as sps
import time as clock

###############################################################################

def basis_functions_V(r,s):
    N0=0.25*(1.-rq)*(1.-sq)
    N1=0.25*(1.+rq)*(1.-sq)
    N2=0.25*(1.+rq)*(1.+sq)
    N3=0.25*(1.-rq)*(1.+sq)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-sq)
    dNdr1=+0.25*(1.-sq)
    dNdr2=+0.25*(1.+sq)
    dNdr3=-0.25*(1.+sq)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-rq)
    dNds1=-0.25*(1.+rq)
    dNds2=+0.25*(1.+rq)
    dNds3=+0.25*(1.-rq)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

def viscosity(exx,eyy,exy):
    e2=np.sqrt(0.5*(exx*exx+eyy*eyy)+exy*exy)
    e2=max(1e-8,e2)
    sigmay=1.
    val=sigmay/2./e2
    val=min(1.e3,val)
    val=max(1.e-3,val)
    #val=1.
    return val

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 008 **********")
print("*******************************")

# declare variables

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=0.5  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 128
   nely = 64
   visu = 1
    
nnx=nelx+1         # number of nodes, x direction
nny=nely+1         # number of nodes, y direction
nn_V=nnx*nny       # number of nodes
nel=nelx*nely      # number of elements, total
Nfem_V=nn_V*ndof_V # Total number of degrees of freedom
Nfem=Nfem_V

penalty=1.e7  # penalty coefficient value

rough=False
width=0.111111111
niter=250

debug=False

print('nelx=',nelx)
print('nely=',nely)
print('nel=',nel)
print('nn_V=',nn_V)
print('Nfem_V=',Nfem_V)
print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*Lx/float(nelx)
        y_V[counter]=j*Ly/float(nely)
        counter+=1

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]= i + j * (nelx + 1)
        icon_V[1,counter]= i + 1 + j * (nelx + 1)
        icon_V[2,counter]= i + 1 + (j + 1) * (nelx + 1)
        icon_V[3,counter]= i + (j + 1) * (nelx + 1)
        counter += 1

if debug:
   for iel in range (0,nel):
     print ("iel=",iel)
     print ("node 1",icon_V[0,iel],"at pos.",x_V[icon_V[0,iel]],y_V[icon_V[0,iel]])
     print ("node 2",icon_V[1,iel],"at pos.",x_V[icon_V[1,iel]],y_V[icon_V[1,iel]])
     print ("node 3",icon_V[2,iel],"at pos.",x_V[icon_V[2,iel]],y_V[icon_V[2,iel]])
     print ("node 4",icon_V[3,iel],"at pos.",x_V[icon_V[3,iel]],y_V[icon_V[3,iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]>(Ly-eps) and abs(x_V[i]-Lx/2.)<width:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = -1.
       if rough: 
          bc_fix_V[i*ndof_V]=True ; bc_val_V[i*ndof_V]= 0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

#####################################################################
###############################################################################
   
Res_file=open('residual.ascii',"w")
u_file=open('u_stats.ascii',"w")
v_file=open('v_stats.ascii',"w")
diff_file=open('diff_uv.ascii',"w")

H=np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
Res   = np.zeros(Nfem,dtype=np.float64) # non-linear residual 
sol   = np.zeros(Nfem,dtype=np.float64) # solution vector 
u     = np.zeros(nn_V,dtype=np.float64) # x-component velocity
v     = np.zeros(nn_V,dtype=np.float64) # y-component velocity
u_old = np.zeros(nn_V,dtype=np.float64) # x-component velocity
v_old = np.zeros(nn_V,dtype=np.float64) # y-component velocity

###############################################################################
###############################################################################
# non-linear iterations
###############################################################################
###############################################################################

for iter in range(0,niter):

    print("--------------------------")
    print("iter=", iter)
    print("--------------------------")

    ###########################################################################
    # build FE matrix
    ###########################################################################
    start = clock.time()

    A_fem=np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
    b_fem=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64)   # gradient matrix B 
    jcb=np.zeros((2,2),dtype=np.float64)

    for iel in range(0,nel):

        # set 2 arrays to 0 every loop
        A_el=np.zeros((m_V*ndof_V,m_V *ndof_V),dtype=np.float64)
        b_el=np.zeros(m_V *ndof_V,dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                # calculate shape functions
                N_V=basis_functions_V(rq,sq)

                # calculate shape function derivatives
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)

                # calculate jacobian matrix
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

                # calculate inverse of the jacobian matrix
                jcbi=np.linalg.inv(jcb)

                # calculate the determinant of the jacobian times weight
                JxWq=np.linalg.det(jcb)*weightq

                # compute coords of quad point
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])

                # compute dNdx & dNdy
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                # compute strain rate at quad point
                exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
                eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
                exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                     np.dot(dNdy_V,u[icon_V[:,iel]])*0.5

                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                # compute elemental a_mat matrix
                A_el+=B.T.dot(C.dot(B))*viscosity(exxq,eyyq,exyq)*JxWq

                # compute elemental rhs vector
                #for i in range(0,m_V):
                #    b_el[2*i  ]+=N_V[i]*gx*density*JxWq
                #    b_el[2*i+1]+=N_V[i]*gy*density*JxWq

            #end for
        #end for

        # integrate penalty term at 1 point
        rq=0.
        sq=0.
        weightq=2.*2.

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

        for i in range(0,m_V):
            B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                              [0.       ,dNdy_V[i]],
                              [dNdy_V[i],dNdx_V[i]]]

        # compute elemental matrix
        A_el+=B.T.dot(H.dot(B))*penalty*JxWq

        # assemble matrix A_fem and right hand side rhs
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

    #end for

    print("build FE matrix & rhs: %.3f s" % (clock.time()-start))

    ###########################################################################
    # impose boundary conditions
    ###########################################################################
    start = clock.time()

    for i in range(0,Nfem):
        if bc_fix_V[i]:
           A_ref=A_fem[i,i]
           for j in range(0,Nfem):
               b_fem[j]-=A_fem[i,j]*bc_val_V[i]
               A_fem[i,j]=0.
               A_fem[j,i]=0.
               A_fem[i,i]=A_ref
           b_fem[i]=A_ref*bc_val_V[i]
        #end if
    #end for

    print("impose b.c. (%.3fs)" % (clock.time()-start))

    ###########################################################################
    # compute non-linear residual
    ###########################################################################
    start=clock.time()

    Res=A_fem.dot(sol)-b_fem

    Res2=np.linalg.norm(Res,2)

    if iter==0:
       Res2Init=Res2

    Res_file.write("%10e \n" % (Res2/Res2Init)) ; Res_file.flush()

    print("     -> Nonlinear residual (inf. norm) %.7e" % (Res2/Res2Init))

    print("compute n.l. residual: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("solve time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol,(nn_V,2)).T

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

    u_file.write("%e %e\n" % (np.min(u),np.max(u))) ; u_file.flush()
    v_file.write("%e %e\n" % (np.min(v),np.max(v))) ; v_file.flush()

    print("split solution: %.3f s" % (clock.time()-start))

    ###########################################################################
    start=clock.time()

    udiff=np.linalg.norm(u_old-u,2)
    vdiff=np.linalg.norm(v_old-v,2)
    if iter==0:
       udiffinit=udiff
       vdiffinit=vdiff
    
    diff_file.write("%e %e\n" % (udiff/udiffinit,vdiff/vdiffinit)) ; diff_file.flush()

    u_old[:]=u[:]
    v_old[:]=v[:]

    print("compute vel difference: %.3f s" % (clock.time()-start))

#end if

###############################################################################
###############################################################################
# end of non-linear iterations
###############################################################################
###############################################################################
   
u_file.close()
v_file.close()
Res_file.close()
diff_file.close()

###############################################################################
# retrieve pressure and elemental strain rate components
# in the middle of the element (1 integration point)
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
eta=np.zeros(nel,dtype=np.float64)  

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

    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
    eta[iel]=viscosity(exx[iel],eyy[iel],exy[iel])
    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("     -> eta (m,M) %.4f %.4f " %(np.min(eta),np.max(eta)))

if debug:
   np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
   np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

###############################################################################
# computing stress tensor components
###############################################################################
start=clock.time()

sigmaxx=np.zeros(nel,dtype=np.float64)  
sigmayy=np.zeros(nel,dtype=np.float64)  
sigmaxy=np.zeros(nel,dtype=np.float64)  

sigmaxx=-p+2*eta*exx
sigmayy=-p+2*eta*eyy
sigmaxy=   2*eta*exy

###############################################################################
# smoothing pressure 
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

print("project press on V grid: %.3f s" % (clock.time()-start))

###############################################################################
# extract velocity field at domain top
###############################################################################
start=clock.time()

xtop=np.zeros(nnx,dtype=np.float64)  
utop=np.zeros(nnx,dtype=np.float64)  
vtop=np.zeros(nnx,dtype=np.float64)  
qtop=np.zeros(nnx,dtype=np.float64)  

counter=0
for i in range(0,nn_V):
    if y_V[i]>Ly-eps:
       xtop[counter]=x_V[i]
       utop[counter]=u[i]
       vtop[counter]=v[i]
       qtop[counter]=q[i]
       counter+=1
   #end if
#end for

xctop=np.zeros(nelx,dtype=np.float64)  
ptop=np.zeros(nelx,dtype=np.float64)  
exxtop=np.zeros(nelx,dtype=np.float64)  
exytop=np.zeros(nelx,dtype=np.float64)  
etop=np.zeros(nelx,dtype=np.float64)  
sigmaxxtop=np.zeros(nelx,dtype=np.float64)  
sigmayytop=np.zeros(nelx,dtype=np.float64)  
sigmaxytop=np.zeros(nelx,dtype=np.float64)  
etatop=np.zeros(nelx,dtype=np.float64)  

counter=0
for iel in range(0,nel):
    if y_V[icon_V[3,iel]]>Ly-eps:
       xctop[counter]=x_e[iel]
       ptop[counter]=p[iel]
       exxtop[counter]=exx[iel]
       exytop[counter]=exy[iel]
       etop[counter]=e[iel]
       sigmaxxtop[counter]=sigmaxx[iel]
       sigmayytop[counter]=sigmayy[iel]
       sigmaxytop[counter]=sigmaxy[iel]
       etatop[counter]=eta[iel]
       counter+=1
   #end if
#end for

np.savetxt('v_q_top.ascii',np.array([xtop,utop,vtop,qtop]).T,header='# x,y,u,v,q')
np.savetxt('p_sr_top.ascii',np.array([xctop,ptop,exxtop,exytop,etop]).T,header='# x,y,p,exx,exy,e')
np.savetxt('sigma_eta_top.ascii',np.array([xctop,sigmaxxtop,sigmayytop,sigmaxytop,etatop]).T,header='# x,y,sigmaxx,sigmaxy')

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


