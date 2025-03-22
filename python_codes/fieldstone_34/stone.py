import numpy as np
import time as time
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

# exp=1: simple shear
# exp=2: pure shear
# exp=3: aquarium 
# exp=4: strip load

experiment=4

###############################################################################

def disp_x(x,y,rho,g,lambdaa,mu,L):
    if experiment==1:
       val=y
    if experiment==2:
       val=2*(x-0.5)
    if experiment==3:
       val=0
    if experiment==4:
       val=0
    return val

def disp_y(x,y,rho,g,lambdaa,mu,L):
    if experiment==1:
       val=0
    if experiment==2:
       val=-2*(y-0.5)
    if experiment==3:
       val=rho*g/(lambdaa+2*mu)*(0.5*y**2-L*y)
    if experiment==4:
       val=0
    return val

def pressure(x,y,rho,g,lambdaa,mu,L):
    if experiment==1:
       val=0.
    if experiment==2:
       val=0.
    if experiment==3:
       val=(lambdaa+2./3.*mu)/(lambdaa+2*mu)*rho*g*(L-y)
    if experiment==4:
       val=0
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

print("-----------------------------")
print("---------- stone 34 ---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 540
   nely = 360
   visu = 1
    
nnx=nelx+1 
nny=nely+1 
NV=nnx*nny 
nel=nelx*nely 
Nfem=NV*ndof 

if experiment==1:
   Lx=1
   Ly=1
   gx=0
   gy=0
   rho=0
   mu=1
   nu=0.25   
   lambdaa=2*mu*nu/(1-2*nu)

if experiment==2:
   Lx=1
   Ly=1
   gx=0
   gy=0
   rho=0
   mu=1
   nu=0.25   
   lambdaa=2*mu*nu/(1-2*nu)

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

hx=Lx/nelx
hy=Ly/nely

debug=False

#####################################################################
# grid point setup
#####################################################################
start = time.time()

x=np.zeros(NV,dtype=np.float64)  # x coordinates
y=np.zeros(NV,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#####################################################################
# connectivity
#####################################################################
start = time.time()

icon =np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (time.time() - start))

#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    for k in range(0,m):
        xc[iel]+=x[icon[k,iel]]*0.25
        yc[iel]+=y[icon[k,iel]]*0.25

print("setup: elt center coords: %.3f s" % (time.time() - start))

#####################################################################
# define boundary conditions
#####################################################################
start = time.time()

bc_fix=np.zeros(Nfem,dtype=bool)       # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

if experiment==1:
   for i in range(0,NV):
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
   for i in range(0,NV):
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = +1
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = -1
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -1
       if y[i]>(Ly-eps):
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = +1

if experiment==3 or experiment==4:
   for i in range(0,NV):
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)
b_mat = np.zeros((3,ndof*m),dtype=np.float64)    # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
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
    a_el=np.zeros((m*ndof,m*ndof), dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

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
            jcob=np.linalg.det(jcb)

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

            # compute elemental a_mat matrix
            a_el+=b_mat.T.dot(c_mat.dot(b_mat))*wq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                b_el[2*i  ]-=N[i]*jcob*wq*gx*rho
                b_el[2*i+1]-=N[i]*jcob*wq*gy*rho

        #end for
    #end for

    #applying pressure b.c. 
    if experiment==4 and abs(xc[iel]-Lx/2)<a and yc[iel]>Ly-hy:
       b_el[2*2+1]-=0.5*p0*hx
       b_el[2*3+1]-=0.5*p0*hx

    # apply boundary conditions
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            m1 =ndof*icon[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndof*k1+i1
               aref=a_el[ikk,ikk]
               for jkk in range(0,m*ndof):
                   b_el[jkk]-=a_el[jkk,ikk]*fixt
                   a_el[ikk,jkk]=0.
                   a_el[jkk,ikk]=0.
               a_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt


    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon[k2,iel]+i2
                    a_mat[m1,m2]+=a_el[ikk,jkk]
                #end for
            #end for
            rhs[m1]+=b_el[ikk]
        #end for
    #end for

#end for iel

print("build FE matrix: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y arrays
#####################################################################
start = time.time()

u,v=np.reshape(sol,(NV,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('displacement.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure
#####################################################################
start = time.time()

p=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
stress_xx=np.zeros(nel,dtype=np.float64)  
stress_yy=np.zeros(nel,dtype=np.float64)  
stress_xy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    wq = 2.0 * 2.0

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
    jcob=np.linalg.det(jcb)
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

print("     -> p   (m,M) %.e %.e " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.e %.e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.e %.e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.e %.e " %(np.min(exy),np.max(exy)))

if debug: np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
if debug: np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

#################################################################
# compute error
#################################################################
start = time.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.
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
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)
            dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
            dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

            xq=N.dot(x[icon[:,iel]])
            yq=N.dot(y[icon[:,iel]])
            uq=N.dot(u[icon[:,iel]])
            vq=N.dot(v[icon[:,iel]])

            errv+=((uq-disp_x(xq,yq,rho,gy,lambdaa,mu,Ly))**2\
                  +(vq-disp_y(xq,yq,rho,gy,lambdaa,mu,Ly))**2)*wq*jcob
            errp+=(p[iel]-pressure(xq,yq,rho,gy,lambdaa,mu,Ly))**2*wq*jcob

        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# 
#####################################################################
start = time.time()

if experiment==4:
   profile=open('top_profile_'+str(nelx)+'.ascii',"w")
   for i in range(0,NV):
       if y[i]/Ly>1-eps:
          profile.write("%e %e %e \n" %(x[i],u[i],v[i]))

   profile=open('top_profile_e_'+str(nelx)+'.ascii',"w")
   for iel in range(0,nel):
       if yc[iel]>Ly-hy:
          profile.write("%e %e %e %e %e %e %e %e\n" %(xc[iel],exx[iel],exy[iel],exy[iel],p[iel],
                                                      stress_xx[iel],stress_yy[iel],stress_xy[iel]))


   profile=open('top_profile_anal_'+str(nelx)+'.ascii',"w")
   for iel in range(0,nel):
       if yc[iel]>Ly-hy:
          profile.write("%e %e %e %e \n" %(xc[iel],\
                                           sigma_xx(xc[iel],yc[iel],p0,a),\
                                           sigma_yy(xc[iel],yc[iel],p0,a),\
                                           sigma_xy(xc[iel],yc[iel],p0,a)))


   print("export profiles: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = time.time()
       
if visu==1:
       vtufile=open('solution.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
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
           vtufile.write("%e\n" % pressure(xc[iel],yc[iel],rho,gy,lambdaa,mu,Ly))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (p[iel]-pressure(xc[iel],yc[iel],rho,gy,lambdaa,mu,Ly)))
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
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xx (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (sigma_xx(xc[iel],yc[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xy (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (sigma_xy(xc[iel],yc[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_yy (th)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (sigma_yy(xc[iel],yc[iel],p0,a)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(0.,disp_y(x[i],y[i],rho,gy,lambdaa,mu,Ly),0.))
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
   
       print("export vtu file: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
###############################################################################
###############################################################################
