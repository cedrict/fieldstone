import numpy as np
import time as clock 
import random
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix

###############################################################################
# define P1 shape functions
###############################################################################

def basis_functions_T(r,s):
    return np.array([1-r-s,r,s],dtype=np.float64)

def basis_functions_T_dr(r,s):
    return np.array([-1,+1,0],dtype=np.float64)

def basis_functions_T_ds(r,s):
    return np.array([-1,0,+1],dtype=np.float64)

###############################################################################

eps=1e-8

print("*******************************")
print("********** stone 045 **********")
print("*******************************")

m_T=3
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
nn_T=(nelx+1)*(nely+1)

nstep=201
dt=2*np.pi/200

Nfem=nn_T # Total number of degrees of freedom

nq_per_el=3
qcoords_r=[1./6.,2./3.,1./6.] # coordinates & weights 
qcoords_s=[1./6.,1./6.,2./3.] # of quadrature points
qweights =[1./6.,1./6.,1./6.]

#parameters for initial T field
xc = 2/3    
yc = 2/3    
Tmin = 0      
Tmax = 1      
sigma = 0.2  

theta=0.5 # time discretisation

xi=0 #0.25 # controls level of mesh randomness (between 0 and 0.5 max)

#set test=True if you wish to test the analytical K_a expression of appendix A
test=False

###############################################################################

print ('nn_T =',nn_T)
print ('nelx =',nelx)
print ('nely =',nely)
print ('nel  =',nel)
print ('Nfem =',Nfem)
print("-----------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_T=np.zeros(nn_T,dtype=np.float64) # x coordinates
y_T=np.zeros(nn_T,dtype=np.float64) # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_T[counter]=i*hx+random.uniform(-1.,+1)*hx*xi
        y_T[counter]=j*hy+random.uniform(-1.,+1)*hy*xi
        if i==0:     x_T[counter]=0
        if i==nnx-1: x_T[counter]=Lx
        if j==0:     y_T[counter]=0
        if j==nny-1: y_T[counter]=Ly
        counter+=1 
    #end for
#end for

print("grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_T=np.zeros((m_T,nel),dtype=np.int32)

counter=0 
for j in range(0,nely):
    for i in range(0,nelx):
        # |\
        # | \
        # |__\
        icon_T[0,counter]= i + j * (nelx + 1)
        icon_T[1,counter]= i + 1 + j * (nelx + 1)
        icon_T[2,counter]= i + (j + 1) * (nelx + 1)
        counter+=1
        # \--|
        #  \ |
        #   \|
        icon_T[0,counter]= i + 1 + j * (nelx + 1)
        icon_T[1,counter]= i + 1 + (j + 1) * (nelx + 1)
        icon_T[2,counter]= i + (j + 1) * (nelx + 1)
        counter+=1
    #end for
#end for

print("connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# velocity field on nodes
###############################################################################
start=clock.time()

u=np.zeros(nn_T,dtype=np.float64)
v=np.zeros(nn_T,dtype=np.float64)

u[:]=-y_T[:]+Ly/2
v[:]= x_T[:]-Lx/2

print("define nodal velocity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(nn_T,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(nn_T,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_T):
    if x_T[i]<eps and u[i]>0:
       bc_fix[i]   = True ; bc_val[i] = 0.
    if x_T[i]>(Lx-eps) and u[i]<0:
       bc_fix[i]   = True ; bc_val[i] = 0.
    if y_T[i]<eps and v[i]>0:
       bc_fix[i]   = True ; bc_val[i] = 0.
    if y_T[i]>(Ly-eps) and v[i]<0:
       bc_fix[i]   = True ; bc_val[i] = 0.

print("boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature 
###############################################################################
start=clock.time()

Told=np.zeros(nn_T,dtype=np.float64)

for i in range(0,nn_T):
    if (x_T[i]-xc)**2+(y_T[i]-yc)**2<=sigma**2:
       Told[i]=(1/4)*(1+np.cos(np.pi*((x_T[i]-xc)/sigma)))*(1+np.cos(np.pi*((y_T[i]-yc)/sigma)))

print("initial temperature: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
###############################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for kq in range(0,nq_per_el):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNdr_T=basis_functions_T_dr(rq,sq)
        dNds_T=basis_functions_T_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
        jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
        jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
        jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
        jcbi=np.linalg.inv(jcb)
        JxWq=np.linalg.det(jcb)*weightq
        area[iel]+=JxWq
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
# time stepping loop
###############################################################################
###############################################################################
    
T_stats_file=open('T_stats.ascii',"w")

for istep in range(0,nstep):

    start=clock.time()

    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    ###########################################################################

    A_fem=np.zeros((Nfem,Nfem),dtype=np.float64) # FE matrix
    b_fem=np.zeros(Nfem,dtype=np.float64)        # FE rhs 
    B_mat=np.zeros((ndim,m_T),dtype=np.float64)   # gradient matrix B 
    N_mat=np.zeros((m_T,1),dtype=np.float64)      # shape functions

    for iel in range (0,nel):

        b_el=np.zeros(m_T,dtype=np.float64)       # elemental rhs
        A_el=np.zeros((m_T,m_T),dtype=np.float64)  # elemental matrix
        Ka=np.zeros((m_T,m_T),dtype=np.float64)    # elemental advection matrix 
        MM=np.zeros((m_T,m_T),dtype=np.float64)    # elemental mass matrix
        velq=np.zeros((1,ndim),dtype=np.float64) # velocity at q point

        for kq in range(0,nq_per_el):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            Tvect=Told[icon_T[:,iel]]

            N_T=basis_functions_T(rq,sq)
            N_mat[0:m_T,0]=basis_functions_T(rq,sq)
            dNdr_T=basis_functions_T_dr(rq,sq)
            dNds_T=basis_functions_T_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
            jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
            jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
            jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq

            #velq[0,0]=np.dot(N_V,u[icon_V[:,iel]])
            #velq[0,1]=np.dot(N_V,v[icon_V[:,iel]])
            xq=np.dot(N_T,x_T[icon_T[:,iel]])
            yq=np.dot(N_T,y_T[icon_T[:,iel]])
            dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
            dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T

            B_mat[0,:]=dNdx_T
            B_mat[1,:]=dNdy_T

            velq[0,0]=-yq+Ly/2
            velq[0,1]= xq-Lx/2

            if test:
               velq[0,0]=np.sum(u[icon[:,iel]])/3
               velq[0,1]=np.sum(v[icon[:,iel]])/3

            MM+=N_mat.dot(N_mat.T)*JxWq
            Ka+=N_mat.dot(velq.dot(B_mat))*JxWq

        # end for kq

        if test:
           if iel<3: print(Ka)
           u0=np.sum(u[icon[:,iel]])/3
           v0=np.sum(v[icon[:,iel]])/3
           x1=x[icon[0,iel]] ; x2=x[icon[1,iel]] ; x3=x[icon[2,iel]]
           y1=y[icon[0,iel]] ; y2=y[icon[1,iel]] ; y3=y[icon[2,iel]]
           Ka=1./6.*np.array([\
                              [u0*(y2-y3)+v0*(x3-x2),u0*(y3-y1)+v0*(x1-x3), u0*(y1-y2)+v0*(x2-x1)], \
                              [u0*(y2-y3)+v0*(x3-x2),u0*(y3-y1)+v0*(x1-x3), u0*(y1-y2)+v0*(x2-x1)], \
                              [u0*(y2-y3)+v0*(x3-x2),u0*(y3-y1)+v0*(x1-x3), u0*(y1-y2)+v0*(x2-x1)]  ])
           if test and iel<3: print(Ka)
           if test and iel>2: exit()

        A_el=MM+Ka*dt*theta
        b_el=(MM -Ka*dt*(1-theta)).dot(Tvect)

        # apply boundary conditions

        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            if bc_fix[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_T):
                   m2=icon_T[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val[m1]
            #end if
        #end for

        # assemble matrix and right hand side
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            for k2 in range(0,m_T):
                m2=icon_T[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        # end for

    #end for iel

    print("build matrix: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve linear system
    ###########################################################################
    start=clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    T_stats_file.write("%e %e %e \n" %(istep*dt,np.min(T),np.max(T)))

    print("solve system: %.3f s" % (clock.time()-start))

    ###########################################################################

    if istep%20==0 :

       start=clock.time()

       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")

       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_T,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e %e %e \n" %(x_T[i],y_T[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (area[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d \n" %(icon_T[0,iel],icon_T[1,iel],icon_T[2,iel]))
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

       print("export to vtu: %.3f s" % (clock.time()-start))

    #end if

    Told[:]=T[:]

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
