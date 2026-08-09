import numpy as np
import time as clock 
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

# exp=1: simple shear
# exp=2: pure shear
# exp=3: aquarium 
# exp=4: strip load
# exp=5: strip load (hackathon 2026)

experiment=5

###############################################################################

def disp_x(x,y,rho,g,lambdaa,mu,L):
    if experiment==1: val=y
    if experiment==2: val=-2*(x-0.5)
    if experiment==3: val=0
    if experiment==4: val=0
    if experiment==5: val=0
    return val

def disp_y(x,y,rho,g,lambdaa,mu,L):
    if experiment==1: val=0
    if experiment==2: val=2*(y-0.5)
    if experiment==3: val=rho*g/(lambdaa+2*mu)*(0.5*y**2-L*y)
    if experiment==4: val=0
    if experiment==5: val=0
    return val

def pressure(x,y,rho,g,lambdaa,mu,L):
    if experiment==1: val=0.
    if experiment==2: val=0.
    if experiment==3: val=(lambdaa+2./3.*mu)/(lambdaa+2*mu)*rho*g*(L-y)
    if experiment==4: val=0
    if experiment==5: val=0
    return val

def sigma_xx(x,y,p0,a):
    xR=Lx/2+a
    xL=Lx/2-a
    theta1=np.arctan((x-xR)/(Ly-y))
    theta2=np.arctan((x-xL)/(Ly-y))
    val=p0/np.pi*(theta2-theta1-0.5*(np.sin(2*theta2)-np.sin(2*theta1)))
    return -val

def sigma_xy(x,y,p0,a):
    xR=Lx/2+a
    xL=Lx/2-a
    theta1=np.arctan((x-xR)/(Ly-y))
    theta2=np.arctan((x-xL)/(Ly-y))
    val=p0/np.pi*((np.sin(theta2))**2-(np.sin(theta1))**2)
    return val

def sigma_yy(x,y,p0,a):
    xR=Lx/2+a
    xL=Lx/2-a
    theta1=np.arctan((x-xR)/(Ly-y))
    theta2=np.arctan((x-xL)/(Ly-y))
    val=p0/np.pi*(theta2-theta1+0.5*(np.sin(2*theta2)-np.sin(2*theta1)))
    return -val

###############################################################################
###############################################################################
###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 034 **********")
print("*******************************")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 300
   nely = 100
   visu = 1

if experiment==1:
   Lx=1
   Ly=1
   gx=0
   gy=0
   rho=0
   mu=1
   nu=0.25   
   lambdaa=2*mu*nu/(1-2*nu)
   p0=0
   a=0

if experiment==2:
   Lx=1
   Ly=1
   gx=0
   gy=0
   rho=0
   mu=1
   nu=0.25   
   lambdaa=2*mu*nu/(1-2*nu)
   p0=0
   a=0

if experiment==3:
   Lx=1000.  
   Ly=1000.
   gx=0
   gy=9.81
   E=6e10 # Young's modulus
   nu=0.25 # Poisson ratio
   rho=2800
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   p0=0
   a=0

if experiment==4:
   Lx=3000.  
   Ly=2000.
   gx=0
   gy=0
   E=6e10 # Young's modulus
   nu=0.25 # Poisson ratio
   rho=2800
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   a=50
   p0=1e8

if experiment==5:
   #https://en.wikipedia.org/wiki/Bulk_modulus
   Lx=600e3
   Ly=200e3
   gx=0
   gy=0
   rho=2700
   K=1/4e-12 # bulk modulus
   mu=1e10 # shear modulus (also G)
   lambdaa=K-2*mu/3  #Lame's first parameter
   a=20e3
   p0=2.7e6 # rho*gy*1000
   nely=int(nelx*Ly/Lx)
   nelx+=1
   nu=(3*K-2*mu)/(6*K+2*mu)
   print('mu=',mu)
   print('lambdaa=',lambdaa)
   print('nelx=',nelx)
   print('nely=',nely)
   print('nu=',nu)

nnx=nelx+1 
nny=nely+1 
nn_V=nnx*nny 
nel=nelx*nely 
Nfem=nn_V*ndof 

hx=Lx/nelx
hy=Ly/nely

debug=False

print('experiment=',experiment)   
print('nel=',nel)
print('Nfem=',Nfem)
print('hx=',hx)
print('hy=',hy)
print("*******************************")

#####################################################################
# grid point setup
#####################################################################
start=clock.time()

x=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (clock.time()-start))

#####################################################################
# connectivity
#####################################################################
start=clock.time()

icon=np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (clock.time()-start))

#####################################################################
# compute element center coordinates
#####################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    for k in range(0,m):
        x_e[iel]+=x[icon[k,iel]]*0.25
        y_e[iel]+=y[icon[k,iel]]*0.25

print("setup: elt center coords: %.3f s" % (clock.time()-start))

#####################################################################
# define boundary conditions
#####################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)       # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

if experiment==1:
   for i in range(0,nn_V):
       if x[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]>(Ly-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 1
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

if experiment==2:
   for i in range(0,nn_V):
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = +1 # left
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = -1 # right
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -1 # bottom
       if y[i]>(Ly-eps):
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = +1 # top

if experiment==3 or experiment==4 or experiment==5:
   for i in range(0,nn_V):
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

#################################################################
# build FE matrix
#################################################################
start=clock.time()

A_fem = lil_matrix((Nfem,Nfem),dtype=np.float64)
b_mat = np.zeros((3,ndof*m),dtype=np.float64)    # gradient matrix B 
b_fem = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)             # shape functions
dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
jcb   = np.zeros((2,2),dtype=np.float64)
c_mat = np.array([[2*mu+lambdaa,lambdaa,     0 ],\
                  [lambdaa,     2*mu+lambdaa,0 ],\
                  [0,           0,           mu]],dtype=np.float64) 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros(m*ndof)
    A_el=np.zeros((m*ndof,m*ndof), dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]=dNdr.dot(x[icon[:,iel]])
            jcb[0,1]=dNdr.dot(y[icon[:,iel]])
            jcb[1,0]=dNds.dot(x[icon[:,iel]])
            jcb[1,1]=dNds.dot(y[icon[:,iel]])

            # calculate the determinant of the jacobian
            JxWq=np.linalg.det(jcb)*weightq

            # calculate inverse of the jacobian matrix
            jcbi=np.linalg.inv(jcb)

            xq=N.dot(x[icon[:,iel]])
            yq=N.dot(y[icon[:,iel]])

            # compute dNdx & dNdy
            dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
            dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental 
            A_el+=b_mat.T.dot(c_mat.dot(b_mat))*JxWq

            # compute elemental vector
            for i in range(0, m):
                b_el[2*i  ]-=N[i]*gx*rho*JxWq
                b_el[2*i+1]-=N[i]*gy*rho*JxWq

        #end for
    #end for

    #applying pressure b.c. 
    if (experiment==4 or experiment==5) and abs(x_e[iel]-Lx/2)<a and y_e[iel]>Ly-hy:
       b_el[2*2+1]-=0.5*p0*hx
       b_el[2*3+1]-=0.5*p0*hx

    # apply boundary conditions
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            m1 =ndof*icon[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndof*k1+i1
               aref=A_el[ikk,ikk]
               for jkk in range(0,m*ndof):
                   b_el[jkk]-=A_el[jkk,ikk]*fixt
                   A_el[ikk,jkk]=0.
                   A_el[jkk,ikk]=0.
               A_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt


    # assemble matrix and right hand side 
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
                #end for
            #end for
            b_fem[m1]+=b_el[ikk]
        #end for
    #end for

#end for iel

print("build FE matrix: %.3f s" % (clock.time()-start))

#################################################################
# solve system
#################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

#####################################################################
# put solution into separate x,y arrays
#####################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('displacement.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

#####################################################################
# retrieve pressure
#####################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
stress_xx=np.zeros(nel,dtype=np.float64)  
stress_yy=np.zeros(nel,dtype=np.float64)  
stress_xy=np.zeros(nel,dtype=np.float64)  
devstress_xx=np.zeros(nel,dtype=np.float64)  
devstress_yy=np.zeros(nel,dtype=np.float64)  
devstress_xy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb[0,0]=dNdr.dot(x[icon[:,iel]])
    jcb[0,1]=dNdr.dot(y[icon[:,iel]])
    jcb[1,0]=dNds.dot(x[icon[:,iel]])
    jcb[1,1]=dNds.dot(y[icon[:,iel]])
    jcbi=np.linalg.inv(jcb)

    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

    exx[iel]=dNdx.dot(u[icon[:,iel]])
    eyy[iel]=dNdy.dot(v[icon[:,iel]])
    exy[iel]=0.5*dNdy.dot(u[icon[:,iel]])+\
             0.5*dNdx.dot(v[icon[:,iel]])

    p[iel]=-(lambdaa+2./3.*mu)*(exx[iel]+eyy[iel])

    stress_xx[iel]=lambdaa*(exx[iel]+eyy[iel])+2*mu*exx[iel]
    stress_yy[iel]=lambdaa*(exx[iel]+eyy[iel])+2*mu*eyy[iel]
    stress_xy[iel]=2*mu*exy[iel]

    devstress_xx[iel]=stress_xx[iel]+p[iel] # not sure about sign
    devstress_yy[iel]=stress_yy[iel]+p[iel]
    devstress_xy[iel]=stress_xy[iel]

print("     -> p   (m,M) %.e %.e " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.e %.e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.e %.e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.e %.e " %(np.min(exy),np.max(exy)))
print("     -> sigma_xx (m,M) %.e %.e " %(np.min(stress_xx),np.max(stress_xx)))
print("     -> sigma_yy (m,M) %.e %.e " %(np.min(stress_yy),np.max(stress_yy)))
print("     -> sigma_xy (m,M) %.e %.e " %(np.min(stress_xy),np.max(stress_xy)))

if debug: np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')
if debug: np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time()-start))

#################################################################
# compute error
#################################################################
start=clock.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
            jcb[0,0]=dNdr.dot(x[icon[:,iel]])
            jcb[0,1]=dNdr.dot(y[icon[:,iel]])
            jcb[1,0]=dNds.dot(x[icon[:,iel]])
            jcb[1,1]=dNds.dot(y[icon[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            xq=N.dot(x[icon[:,iel]])
            yq=N.dot(y[icon[:,iel]])
            uq=N.dot(u[icon[:,iel]])
            vq=N.dot(v[icon[:,iel]])
            errv+=((uq-disp_x(xq,yq,rho,gy,lambdaa,mu,Ly))**2\
                  +(vq-disp_y(xq,yq,rho,gy,lambdaa,mu,Ly))**2)*JxWq
            errp+=(p[iel]-pressure(xq,yq,rho,gy,lambdaa,mu,Ly))**2*JxWq
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8e ; errp= %.8e" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

#####################################################################
# export data on lines 
#####################################################################
start=clock.time()

if experiment==4 or experiment==5:
   profile=open('top_profile_'+str(nelx)+'.ascii',"w")
   for i in range(0,nn_V):
       if y[i]/Ly>1-eps:
          profile.write("%e %e %e \n" %(x[i],u[i],v[i]))

   profile=open('top_profile_e_'+str(nelx)+'.ascii',"w")
   for iel in range(0,nel):
       if y_e[iel]>Ly-hy:
          profile.write("%e %e %e %e %e %e %e %e\n" %(x_e[iel],exx[iel],exy[iel],exy[iel],p[iel],
                                                      stress_xx[iel],stress_yy[iel],stress_xy[iel]))


   profile=open('top_profile_anal_'+str(nelx)+'.ascii',"w")
   for iel in range(0,nel):
       if y_e[iel]>Ly-hy:
          profile.write("%e %e %e %e \n" %(x_e[iel],\
                                           sigma_xx(x_e[iel],y_e[iel],p0,a),\
                                           sigma_yy(x_e[iel],y_e[iel],p0,a),\
                                           sigma_xy(x_e[iel],y_e[iel],p0,a)))

   profile=open('mid_profile_e_'+str(nelx)+'.ascii',"w")
   for iel in range(0,nel):
       if abs(x_e[iel]-Lx/2)<hx:
          profile.write("%e %e %e %e %e %e %e %e\n" %(y_e[iel],exx[iel],exy[iel],exy[iel],p[iel],
                                                      stress_xx[iel],stress_yy[iel],stress_xy[iel]))

   profile=open('mid_profile_anal_'+str(nelx)+'.ascii',"w")
   for iel in range(0,nel):
       if abs(x_e[iel]-Lx/2)<hx:
          profile.write("%e %e %e %e \n" %(y_e[iel],\
                                           sigma_xx(x_e[iel],y_e[iel],p0,a),\
                                           sigma_yy(x_e[iel],y_e[iel],p0,a),\
                                           sigma_xy(x_e[iel],y_e[iel],p0,a)))

   print("export profiles: %.3f s" % (clock.time()-start))

#####################################################################
# plot of solution
#####################################################################
start=clock.time()
       
if visu==1:
       vtufile=open('solution.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
          vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % pressure(x_e[iel],y_e[iel],rho,gy,lambdaa,mu,Ly))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (p[iel]-pressure(x_e[iel],y_e[iel],rho,gy,lambdaa,mu,Ly)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % eyy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (stress_xx[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (stress_yy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (stress_xy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev sigma_xx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (devstress_xx[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev sigma_yy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (devstress_yy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev sigma_xy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (devstress_xy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xx (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (sigma_xx(x_e[iel],y_e[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xy (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (sigma_xy(x_e[iel],y_e[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_yy (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (sigma_yy(x_e[iel],y_e[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xx (error)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (stress_xx[iel]-sigma_xx(x_e[iel],y_e[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xy (error)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (stress_xy[iel]-sigma_xy(x_e[iel],y_e[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_yy (error)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (stress_yy[iel]-sigma_yy(x_e[iel],y_e[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.4e %.4e %.4e \n" %(disp_x(x[i],y[i],rho,gy,lambdaa,mu,Ly),\
                                               disp_y(x[i],y[i],rho,gy,lambdaa,mu,Ly),0.))
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
   
       print("export vtu file: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
###############################################################################
###############################################################################
