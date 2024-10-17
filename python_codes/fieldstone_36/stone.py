import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time

#------------------------------------------------------------------------------

def displacement_r(x,y,R1,R2,rho,g0,lambdaa,mu):
    r=np.sqrt(x*x+y*y)
    C1 = rho0 * g0 / (lambdaa + 2 * mu) / 3.
    k1 = (2*mu + lambdaa) * C1 * (2 * R1**2 * R2**3 - R1**3 * R2**2)
    k2 = lambdaa * C1 * (R1**2 * R2**3 - R1**3 * R2**2)
    C3 = (k1 + k2) / (( (R2**2+R1**2)*(2*mu+lambdaa) )  +  lambdaa * (R2**2-R1**2) )
    C2 = -C1 * R1 - C3 / R1**2
    val= C1*r**2 + C2*r + C3/r
    return val

def displacement_theta(x,y,R1,R2,rho,g0,lambdaa,mu):
    return 0

def displacement_x(x,y,R1,R2,rho,g0,lambdaa,mu):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    C1 = rho0 * g0 / (lambdaa + 2 * mu) / 3.
    k1 = (2*mu + lambdaa) * C1 * (2 * R1**2 * R2**3 - R1**3 * R2**2)
    k2 = lambdaa * C1 * (R1**2 * R2**3 - R1**3 * R2**2)
    C3 = (k1 + k2) / (( (R2**2+R1**2)*(2*mu+lambdaa) )  +  lambdaa * (R2**2-R1**2) )
    C2 = -C1 * R1 - C3 / R1**2
    vr= C1*r**2 + C2*r + C3/r
    val=vr*math.cos(theta)
    return val

def displacement_y(x,y,R1,R2,rho,g0,lambdaa,mu):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    C1 = rho0 * g0 / (lambdaa + 2 * mu) / 3.
    k1 = (2*mu + lambdaa) * C1 * (2 * R1**2 * R2**3 - R1**3 * R2**2)
    k2 = lambdaa * C1 * (R1**2 * R2**3 - R1**3 * R2**2)
    C3 = (k1 + k2) / (( (R2**2+R1**2)*(2*mu+lambdaa) )  +  lambdaa * (R2**2-R1**2) )
    C2 = -C1 * R1 - C3 / R1**2
    vr= C1*r**2 + C2*r + C3/r
    val=vr*math.sin(theta)
    return val

def pressure(x,y,R1,R2,rho,g0,lambdaa,mu):
    r=np.sqrt(x*x+y*y)
    C1 = rho0 * g0 / (lambdaa + 2 * mu) / 3.
    k1 = (2*mu + lambdaa) * C1 * (2 * R1**2 * R2**3 - R1**3 * R2**2)
    k2 = lambdaa * C1 * (R1**2 * R2**3 - R1**3 * R2**2)
    C3 = (k1 + k2) / (( (R2**2+R1**2)*(2*mu+lambdaa) )  +  lambdaa * (R2**2-R1**2) )
    C2 = -C1 * R1 - C3 / R1**2
    val=-(lambdaa+2*mu/3)*(3*C1*r+2*C2)
    return val

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

if int(len(sys.argv) == 3):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
else:
   nelr = 24
   visu = 1

R1=2890e3
R2=6371e3
area=math.pi*(R2**2-R1**2)

dr=(R2-R1)/nelr 
nelt=int(2.*math.pi*R2/dr)
nel=nelr*nelt  # number of elements, total
nnr=nelr+1 # number of nodes radial direction
nnt=nelt # number of nodes tangential direction
nnp=nnr*nnt  # number of nodes
Nfem=nnp*ndof  # Total number of degrees of freedom

rho0=3300.
g0=9.81
E=6e10 # Young's modulus
nu=0.49 # Poisson ratio
mu=E/2/(1+nu)
lambdaa=E*nu/(1+nu)/(1-2*nu)

print('g0=',g0)
print('rho0=',rho0)
print('mu=',mu)
print('lambda=',lambdaa)

eps=1.e-10

sqrt3=np.sqrt(3.)

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.empty(nnp,dtype=np.float64)  # x coordinates
y=np.empty(nnp,dtype=np.float64)  # y coordinates
r=np.empty(nnp,dtype=np.float64)  
theta=np.empty(nnp,dtype=np.float64) 

Louter=2.*math.pi*R2
Lr=R2-R1
sx = Louter/float(nelt)
sz = Lr    /float(nelr)

counter=0
for j in range(0,nnr):
    for i in range(0,nelt):
        x[counter]=i*sx
        y[counter]=j*sz
        counter += 1

counter=0
for j in range(0,nnr):
    for i in range(0,nnt):
        xi=x[counter]
        yi=y[counter]
        t=xi/Louter*2.*math.pi    
        x[counter]=math.cos(t)*(R1+yi)
        y[counter]=math.sin(t)*(R1+yi)
        r[counter]=R1+yi
        theta[counter]=math.atan2(y[counter],x[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*math.pi
        counter+=1

print("grid setup (%.3fs)" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        icon[0, counter] = icon2 
        icon[1, counter] = icon1
        icon[2, counter] = icon4
        icon[3, counter] = icon3
        counter += 1

print("connectivity (%.3fs)" % (time.time() - start))

#################################################################
# define boundary conditions
# actually prescribing no slip at the bottom because easier
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=bool)  
bc_val = np.zeros(Nfem, dtype=np.float64) 

for i in range(0, nnp):
    if r[i]<R1*(1+eps):
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0 
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0
    #if r[i]>R2*(1-eps):
    #   bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = displacement_x(x[i],y[i],R1,R2,kk,rho0,g0)
    #   bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = displacement_y(x[i],y[i],R1,R2,kk,rho0,g0)

print("defining boundary conditions (%.3fs)" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component displacement 
v     = np.zeros(nnp,dtype=np.float64)          # y-component displacement 
vr    = np.zeros(nnp,dtype=np.float64)          # r-component displacement 
vt    = np.zeros(nnp,dtype=np.float64)          # theta-component displacement 
c_mat = np.array([[2*mu+lambdaa,lambdaa,0],[lambdaa,2*mu+lambdaa,0],[0,0,mu]],dtype=np.float64) 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros((m * ndof), dtype=np.float64)
    a_el = np.zeros((m*ndof,m*ndof), dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

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
            jcb = np.zeros((2, 2),dtype=np.float64)
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
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            a_el += b_mat.T.dot(c_mat.dot(b_mat))*weightq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                b_el[2*i  ]+=N[i]*jcob*weightq*gx(xq,yq,g0)*rho0
                b_el[2*i+1]+=N[i]*jcob*weightq*gy(xq,yq,g0)*rho0

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
            rhs[m1]+=b_el[ikk]

print("build FE matrixs & rhs (%.3fs)" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
print("solving system (%.3fs)" % (time.time() - start))

#####################################################################
# put solution into separate x,y displacement arrays
#####################################################################
start = time.time()

u,v=np.reshape(sol,(nnp,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

np.savetxt('displacement_xy.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %.4f %.4f " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4f %.4f " %(np.min(vt),np.max(vt)))

np.savetxt('displacement_rtheta.ascii',np.array([np.sqrt(x**2+y**2),vr,vt]).T,header='#r,vr,vtheta')

print("reshape solution (%.3fs)" % (time.time() - start))

#####################################################################
# retrieve elemental pressure and strain rate
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    weightq = 2.0 * 2.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    for k in range(0, m):
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

    p[iel]=-(lambdaa+2*mu/3)*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

np.savetxt('pressure.ascii',np.array([xc,yc,p,np.sqrt(xc**2+yc**2)]).T,header='# xc,yc,p')
np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p & sr (%.3f s)" % (time.time() - start))

#################################################################
# compute error
# errors are normalised by the area !
#################################################################
start = time.time()

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
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            jcob=np.linalg.det(jcb)
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
            errv+=((uq-displacement_x(xq,yq,R1,R2,rho0,g0,lambdaa,mu))**2+\
                   (vq-displacement_y(xq,yq,R1,R2,rho0,g0,lambdaa,mu))**2)*weightq*jcob
            errp+=(p[iel]-pressure(xq,yq,R1,R2,rho0,g0,lambdaa,mu))**2*weightq*jcob

errv=np.sqrt(errv)/area
errp=np.sqrt(errp)/area

print("     -> nel= %6d ; errv= %.4e ; errp= %.4e" %(nel,errv,errp))

print("compute errors (%.3f s)" % (time.time() - start))

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
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % pressure(xc[iel],yc[iel],R1,R2,rho0,g0,lambdaa,mu))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % (p[iel]-pressure(xc[iel],yc[iel],R1,R2,rho0,g0,lambdaa,mu)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(gx(x[i],y[i],g0),gy(x[i],y[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (r,theta)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)(r,theta)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(displacement_r(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),0.,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(displacement_x(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),displacement_y(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i]-displacement_x(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),v[i]-displacement_y(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %r[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %theta[i])
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
   print("export to vtu (%.3f s)" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
