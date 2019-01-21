import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def density(x,y,Lx,Ly):
    val=3150
    if (y>660e3-80e3 and y<=660e3) or (y>660e3-(80e3+250e3) and abs(x-Lx/2)<40e3):
       val=3300
    return val

def viscosity(x,y,Lx,Ly,exx,eyy,exy):
    val=1.e21
    if (y>660e3-80e3 and y<=660e3) or (y>660e3-(80e3+250e3) and abs(x-Lx/2)<40e3):
       sr=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       if sr<1e-30:
          sr=1e-30
       n_pow=4 
       val=(4.75e11)*sr**(1./n_pow -1.)
       val=max(val,1e19)
       val=min(val,1e25)
    return val

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1000e3  # horizontal extent of the domain 
Ly=660e3  # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 175
   nely = int(nelx*Ly/Lx)
   visu = 1

gx=0.
gy=-10.
eta_ref=1e21
niter=50
tol=1e-7
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemV=nnp*ndofV # number of velocity dofs
NfemP=nel*ndofP # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

year=3.154e+7
eps=1.e-10
sqrt3=np.sqrt(3.)

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnp=",nnp)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0,nnp):
    if x[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if x[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]/Ly>(1-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# allocate arrays
#################################################################

e1   =np.zeros(nnp,dtype=np.float64)  
rho  =np.zeros(nel,dtype=np.float64)    
eta  =np.zeros(nel,dtype=np.float64)   
u    =np.zeros(nnp,dtype=np.float64)  
v    =np.zeros(nnp,dtype=np.float64)   
p    =np.zeros(nel,dtype=np.float64)    
sol  =np.zeros(Nfem,dtype=np.float64) 
xi   =np.zeros(niter,dtype=np.float64) 
c_mat=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

###############################################################################
# nonlinear iterations
###############################################################################

for iter in range(0,niter):

    print("------------------------------")
    print("iter= %d" % iter) 
    print("------------------------------")

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
    b_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix B 
    N     = np.zeros(m,dtype=np.float64)             # shape functions
    dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives

    for iel in range(0, nel):

        # set arrays to 0 every loop
        f_el =np.zeros((m*ndofV),dtype=np.float64)
        K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
        G_el=np.zeros((m*ndofV,1),dtype=np.float64)
        h_el=np.zeros((1,1),dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [-1, 1]:
            for jq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.

                # calculate shape functions
                N[0:m]=NNV(rq,sq)
                dNdr[0:m]=dNNVdr(rq,sq)
                dNds[0:m]=dNNVds(rq,sq)

                # calculate jacobian matrix
                jcb = np.zeros((2, 2),dtype=float)
                for k in range(0,m):
                    jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                    jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                    jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                    jcb[1, 1] += dNds[k]*y[icon[k,iel]]
    
                # calculate the determinant of the jacobian
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                xq=0.0
                yq=0.0
                exxq=0.
                eyyq=0.
                exyq=0.
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    exxq += dNdx[k]*u[icon[k,iel]]
                    eyyq += dNdy[k]*v[icon[k,iel]]
                    exyq += 0.5*dNdy[k]*u[icon[k,iel]]+\
                            0.5*dNdx[k]*v[icon[k,iel]]

                # compute density and viscosity at qpoint
                rhoq=density(xq,yq,Lx,Ly)
                etaq=viscosity(xq,yq,Lx,Ly,exxq,eyyq,exyq)

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                             [0.     ,dNdy[i]],
                                             [dNdy[i],dNdx[i]]]

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*wq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    f_el[ndofV*i  ]+=N[i]*jcob*wq*rhoq*gx
                    f_el[ndofV*i+1]+=N[i]*jcob*wq*rhoq*gy
                    G_el[ndofV*i  ,0]-=dNdx[i]*jcob*wq
                    G_el[ndofV*i+1,0]-=dNdy[i]*jcob*wq

        # impose b.c. 
        for k1 in range(0,m):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*icon[k1,iel]+i1
                if bc_fix[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m*ndofV):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                   h_el[0]-=G_el[ikk,0]*bc_val[m1]
                   G_el[ikk,0]=0

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,m):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*icon[k1,iel]+i1
                for k2 in range(0,m):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*icon[k2,iel]+i2
                        K_mat[m1,m2]+=K_el[ikk,jkk]
                f_rhs[m1]+=f_el[ikk]
                G_mat[m1,iel]+=G_el[ikk,0]
        h_rhs[iel]+=h_el[0]

    G_mat*=eta_ref/Lx # scale G matrix

    print("build FE matrix: %.3f s" % (time.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = time.time()

    a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
    rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

    a_mat[0:NfemV,0:NfemV]=K_mat
    a_mat[0:NfemV,NfemV:Nfem]=G_mat
    a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("assemble blocks: %.3f s" % (time.time() - start))

    ######################################################################
    # convergence test 
    ######################################################################

    residual=a_mat.dot(sol)-rhs

    xi[iter]=np.linalg.norm(residual,2)/np.linalg.norm(rhs,2)

    print("     -> xi= %.4e tol= %.4e " %(xi[iter],tol))

    np.savetxt('xi.ascii',np.array([xi[0:iter]]).T,header='# xi')

    if xi[iter]<tol:
       print('*****converged*****')
       break

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
    p=sol[NfemV:Nfem]*(eta_ref/Lx)

    print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

    print("split vel into u,v: %.3f s" % (time.time() - start))

    ######################################################################
    # compute elemental strainrate, density and viscosity
    # these fields are only for visualisation purposes
    ######################################################################
    start = time.time()

    xc=np.zeros(nel,dtype=np.float64)  
    yc=np.zeros(nel,dtype=np.float64)  
    exx=np.zeros(nel,dtype=np.float64)  
    eyy=np.zeros(nel,dtype=np.float64)  
    exy=np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):
        rq = 0.0
        sq = 0.0
        N[0:m]=NNV(rq,sq)
        dNdr[0:m]=dNNVdr(rq,sq)
        dNds[0:m]=dNNVds(rq,sq)

        jcb=np.zeros((2,2),dtype=float)
        for k in range(0, m):
            jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
            jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
            jcb[1,0]+=dNds[k]*x[icon[k,iel]]
            jcb[1,1]+=dNds[k]*y[icon[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        for k in range(0, m):
            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
            xc[iel] += N[k]*x[icon[k,iel]]
            yc[iel] += N[k]*y[icon[k,iel]]
            exx[iel] += dNdx[k]*u[icon[k,iel]]
            eyy[iel] += dNdy[k]*v[icon[k,iel]]
            exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+\
                        0.5*dNdx[k]*v[icon[k,iel]]

        rho[iel]=density(xc[iel],yc[iel],Lx,Ly)
        eta[iel]=viscosity(xc[iel],yc[iel],Lx,Ly,exx[iel],eyy[iel],exy[iel])

    print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

    print("compute press & sr: %.3f s" % (time.time() - start))

#####################################################################
# line measurements 
# each line of measurements is discretised with npts points. 
# each point is localised in an element, and strainrate is computed
# at this location. density and viscosity are then computed 
# on it and stored.
#####################################################################
start = time.time()

npts=2000

xp=np.zeros(npts,dtype=np.float64)  
yp=np.zeros(npts,dtype=np.float64)  
rhop=np.zeros(npts,dtype=np.float64)  
etap=np.zeros(npts,dtype=np.float64)  
exxp=np.zeros(npts,dtype=np.float64)  
eyyp=np.zeros(npts,dtype=np.float64)  
exyp=np.zeros(npts,dtype=np.float64)  
for i in range(0,npts):
    xp[i]=i*Lx/(npts-1)
    yp[i]=550e3
    xp[i]=min(xp[i],Lx*(1-eps))
    xp[i]=max(xp[i],Lx*eps)
    ielx=int(xp[i]/Lx*nelx)
    iely=int(yp[i]/Ly*nely)
    iel=nelx*(iely)+ielx
    xmin=x[icon[0,iel]]
    xmax=x[icon[2,iel]]
    ymin=y[icon[0,iel]]
    ymax=y[icon[2,iel]]
    r=((xp[i]-xmin)/(xmax-xmin)-0.5)*2
    s=((yp[i]-ymin)/(ymax-ymin)-0.5)*2
    N[0:m]=NNV(r,s)
    dNdr[0:m]=dNNVdr(r,s)
    dNds[0:m]=dNNVds(r,s)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)
    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
        exxp[i]+=dNdx[k]*u[icon[k,iel]]
        eyyp[i]+=dNdy[k]*v[icon[k,iel]]
        exyp[i]+=0.5*dNdy[k]*u[icon[k,iel]]+\
                 0.5*dNdx[k]*v[icon[k,iel]]
    rhop[i]=density(xp[i],yp[i],Lx,Ly)
    etap[i]=viscosity(xp[i],yp[i],Lx,Ly,exxp[i],eyyp[i],exyp[i])
     
np.savetxt('horizontal.ascii',np.array([xp,yp,rhop,etap,exxp,eyyp,exyp]).T,header='# x,y,rho,eta')

xp=np.zeros(npts,dtype=np.float64)  
yp=np.zeros(npts,dtype=np.float64)  
rhop=np.zeros(npts,dtype=np.float64)  
etap=np.zeros(npts,dtype=np.float64)  
exxp=np.zeros(npts,dtype=np.float64)  
eyyp=np.zeros(npts,dtype=np.float64)  
exyp=np.zeros(npts,dtype=np.float64)  
for i in range(0,npts):
    xp[i]=Lx/2.
    yp[i]=i*Ly/(npts-1)
    yp[i]=min(yp[i],Ly*(1-eps))
    yp[i]=max(yp[i],Ly*eps)
    ielx=int(xp[i]/Lx*nelx)
    iely=int(yp[i]/Ly*nely)
    iel=nelx*(iely)+ielx
    xmin=x[icon[0,iel]]
    xmax=x[icon[2,iel]]
    ymin=y[icon[0,iel]]
    ymax=y[icon[2,iel]]
    r=((xp[i]-xmin)/(xmax-xmin)-0.5)*2
    s=((yp[i]-ymin)/(ymax-ymin)-0.5)*2
    N[0:m]=NNV(r,s)
    dNdr[0:m]=dNNVdr(r,s)
    dNds[0:m]=dNNVds(r,s)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)
    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
        exxp[i]+=dNdx[k]*u[icon[k,iel]]
        eyyp[i]+=dNdy[k]*v[icon[k,iel]]
        exyp[i]+=0.5*dNdy[k]*u[icon[k,iel]]+\
                 0.5*dNdx[k]*v[icon[k,iel]]
    rhop[i]=density(xp[i],yp[i],Lx,Ly)
    etap[i]=viscosity(xp[i],yp[i],Lx,Ly,exxp[i],eyyp[i],exyp[i])
     
np.savetxt('vertical.ascii',np.array([xp,yp,rhop,etap,exxp,eyyp,exyp]).T,header='# x,y,rho,eta')
   
print("export profiles: %.3fs" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = time.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % exx[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % eyy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % exy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%5e \n" % rho[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='viscosity' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%5e \n" % eta[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
   vtufile.write("</DataArray>\n")
   #-------------
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e \n" % q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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
   print("export to vtu | time: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

