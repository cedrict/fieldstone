import numpy as np
import sys as sys
import time 
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import random

#------------------------------------------------------------------------------
# define P1 shape functions
#------------------------------------------------------------------------------

def NNT(rq,sq):
    NV_0= (1.-rq-sq)
    NV_1= rq
    NV_2= sq
    return NV_0,NV_1,NV_2

def dNNTdr(rq,sq):
    dNVdr_0= -1. 
    dNVdr_1= +1.
    dNVdr_2=  0.
    return dNVdr_0,dNVdr_1,dNVdr_2

def dNNTds(rq,sq):
    dNVds_0= -1. 
    dNVds_1=  0.
    dNVds_2= +1.
    return dNVds_0,dNVds_1,dNVds_2

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mT=3
nqel=3
ndofT=1
ndim=2

Lx=1
Ly=1

nelx=30
nely=30
nnx=nelx+1
nny=nely+1
hx=Lx/nelx
hy=Ly/nely

nel=2*nelx*nely
NT=(nelx+1)*(nely+1)

nstep=1001
dt=2*np.pi/200

eps=1e-8

Nfem=NT*ndofT      # Total number of degrees of freedom

qcoords_r=[1./6.,2./3.,1./6.] # coordinates & weights 
qcoords_s=[1./6.,1./6.,2./3.] # of quadrature points
qweights =[1./6.,1./6.,1./6.]

#parameters for initial T field
xc = 2/3    
yc = 2/3    
Tmin = 0      
Tmax = 1      
sigma = 0.2  

hcapa = 1   
hcond = 0
rho0 = 1   

theta=0.5 # time discretisation

xi=0. # controls level of mesh randomness (between 0 and 0.5 max)

#################################################################

print ('NT       =',NT)
print ('nel      =',nel)
print ('Nfem     =',Nfem)
print("-----------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.zeros(NT,dtype=np.float64)          # x coordinates
y=np.zeros(NT,dtype=np.float64)          # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx+random.uniform(-1.,+1)*hx*xi
        y[counter]=j*hy+random.uniform(-1.,+1)*hy*xi
        if i==0:
           x[counter]=0
        if i==nnx-1:
           x[counter]=Lx
        if j==0:
           y[counter]=0
        if j==nny-1:
           y[counter]=Ly
        counter += 1 
    #end for
#end for

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((mT, nel),dtype=np.int16)

counter = 0 
for j in range(0, nely):
    for i in range(0, nelx):
        # |\
        # | \
        # |__\
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + (j + 1) * (nelx + 1)
        counter += 1
        # \--|
        #  \ |
        #   \|
        icon[0, counter] = i + 1 + j * (nelx + 1)
        icon[1, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[2, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

print("connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(NT, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(NT, dtype=np.float64)  # boundary condition, value

for i in range(0,NT):
    if x[i]<eps:
       bc_fix[i]   = True ; bc_val[i]   = 0.
    if x[i]>(Lx-eps):
       bc_fix[i]   = True ; bc_val[i]   = 0.
    if y[i]<eps:
       bc_fix[i]   = True ; bc_val[i]   = 0.
    if y[i]>(Ly-eps):
       bc_fix[i]   = True ; bc_val[i]   = 0.

print("boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# velocity field on nodes
#################################################################
start = time.time()

u=-y+Ly/2
v=x-Lx/2

print("nodal velocity: %.3f s" % (time.time() - start))

#################################################################
# initial temperature 
#################################################################
start = time.time()

Told = np.zeros(NT)

for i in range(0,NT):
    if (x[i]-xc)**2+(y[i]-yc)**2 <= sigma**2:
       Told[i]= (1/4)*(1+np.cos(np.pi*((x[i]-xc)/sigma)))*(1+np.cos(np.pi*((y[i]-yc)/sigma)))

print("initial temperature: %.3f s" % (time.time() - start))

#################################################################
# compute area of elements
#################################################################
start = time.time()

area=np.zeros(nel,dtype=np.float64) 
NNNT    = np.zeros(mT,dtype=np.float64)           # shape functions V
dNNNTdr  = np.zeros(mT,dtype=np.float64)          # shape functions derivatives
dNNNTds  = np.zeros(mT,dtype=np.float64)          # shape functions derivatives

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNT[0:mT]=NNT(rq,sq)
        dNNNTdr[0:mT]=dNNTdr(rq,sq)
        dNNNTds[0:mT]=dNNTds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mT):
            jcb[0,0] += dNNNTdr[k]*x[icon[k,iel]]
            jcb[0,1] += dNNNTdr[k]*y[icon[k,iel]]
            jcb[1,0] += dNNNTds[k]*x[icon[k,iel]]
            jcb[1,1] += dNNNTds[k]*y[icon[k,iel]]
        #end for
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (time.time() - start))

###############################################################################
###############################################################################
# time stepping loop
###############################################################################
###############################################################################
    
T_stats_file=open('T_stats.ascii',"w")

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")

    #################################################################
    # build FE matrix
    #################################################################

    dNNNTdx = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdy = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    A_mat = np.zeros((Nfem,Nfem),dtype=np.float64)     # FE matrix
    rhs   = np.zeros(Nfem,dtype=np.float64)            # FE rhs 
    B_mat = np.zeros((ndim,ndofT*mT),dtype=np.float64) # gradient matrix B 
    N_mat = np.zeros((mT,1),dtype=np.float64)          # shape functions
    Tvect = np.zeros(mT,dtype=np.float64)

    for iel in range (0,nel):

        b_el=np.zeros(mT*ndofT,dtype=np.float64)
        a_el=np.zeros((mT*ndofT,mT*ndofT),dtype=np.float64)
        Ka=np.zeros((mT,mT),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((mT,mT),dtype=np.float64)   # elemental mass matrix
        velq=np.zeros((1,ndim),dtype=np.float64)

        for kq in range(0,nqel):

            for k in range(0,mT):
                Tvect[k]=Told[icon[k,iel]]

            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            N_mat[0:mT,0]=NNT(rq,sq)
            dNNNTdr[0:mT]=dNNTdr(rq,sq)
            dNNNTds[0:mT]=dNNTds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mT):
                jcb[0,0]+=dNNNTdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNNNTdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNNNTds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNNNTds[k]*y[icon[k,iel]]
            #end for

            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)

            # compute dNdx & dNdy
            velq[0,0]=0.
            velq[0,1]=0.
            xq=0.
            yq=0.
            for k in range(0,mT):
                velq[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                velq[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                xq+=N_mat[k,0]*x[icon[k,iel]]
                yq+=N_mat[k,0]*y[icon[k,iel]]
                dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                B_mat[0,k]=dNNNTdx[k]
                B_mat[1,k]=dNNNTdy[k]
            #end for

            MM+=N_mat.dot(N_mat.T)*rho0*hcapa*weightq*jcob

            # compute diffusion matrix
            Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

            # compute advection matrix
            Ka+=N_mat.dot(velq.dot(B_mat))*rho0*hcapa*weightq*jcob

        # end for kq

        a_el=MM+(Ka+Kd)*dt*theta
        b_el=(MM -(Ka+Kd)*dt*(1-theta)).dot(Tvect)

        # apply boundary conditions

        for k1 in range(0,mT):
            m1=icon[k1,iel]
            if bc_fix[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mT):
                   m2=icon[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_val[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               #end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val[m1]
            #end if
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mT):
            m1=icon[k1,iel]
            for k2 in range(0,mT):
                m2=icon[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        # end for

    #end for iel

    #################################################################
    # solve linear system
    #################################################################

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    T_stats_file.write("%10e %10e %10e \n" %(istep*dt,np.min(T),np.max(T)))

    #################################################################

    if istep%200==0 :
       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")

       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NT,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")

       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (area[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")

       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel]))
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
    #end if

    Told[:]=T[:]

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
