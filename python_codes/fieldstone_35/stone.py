import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# main geometry parameters
#------------------------------------------------------------------------------

R1=1.
R2=2.
kk=2

a=1
b=-2*R1-2*R2
c=R1**2+R2**2+4*R1*R2
d=-2*R1*R2**2-2*R1**2*R2
e=R1**2*R2**2

print('a=',a)
print('b=',b)
print('c=',c)
print('d=',d)
print('e=',e)

A=(kk**2-4)*(kk**2-16)
B=(kk**4-10*kk**2+9)
C=kk**2*(kk**2-4)
D=(kk**2-1)*(kk**2-1)
E=kk**2*(kk**2-4)

print('A=',A)
print('B=',B)
print('C=',C)
print('D=',D)
print('E=',E)

#------------------------------------------------------------------------------
def density(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    val=np.sin(k*theta)/k*(A*a*r**4+B*b*r**3+C*c*r**2+D*d*r+E*e)/r**3
    return val

def Psi(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    val=(a*r**4+b*r**3+c*r**2+d*r+e)*np.cos(k*theta)
    return val

def velocity_x(x,y,R1,R2,k,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    v_r=-(a*r**4+b*r**3+c*r**2+d*r+e)/r*k*np.sin(k*theta)    
    v_t=-(4*a*r**3+3*b*r**2+2*c*r+d)*np.cos(k*theta)
    val=v_r*math.cos(theta)-v_t*math.sin(theta)
    return val

def velocity_y(x,y,R1,R2,k,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    v_r=-(a*r**4+b*r**3+c*r**2+d*r+e)/r*k*np.sin(k*theta)    
    v_t=-(4*a*r**3+3*b*r**2+2*c*r+d)*np.cos(k*theta)
    val=v_r*math.sin(theta)+v_t*math.cos(theta)
    return val

def pressure(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    val=np.sin(k*theta)/k*( 2*(k**2-16)*a*r**2 +(k**2-9)*b*r +(1-k**2)*d/r -2*k**2*e/r**2 )
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
   nelr = 40
   visu = 1

dr=(R2-R1)/nelr
nelt=int(2.*math.pi*R2/dr)
nel=nelr*nelt  # number of elements, total

nnr=nelr+1
nnt=nelt
nnp=nnr*nnt  # number of nodes

g0=1.

viscosity=1.  # dynamic viscosity \mu
penalty=1.e7  # penalty coefficient value

Nfem=nnp*ndof  # Total number of degrees of freedom

eps=1.e-10

sqrt3=np.sqrt(3.)

#################################################################
# grid point setup
#################################################################

print("grid point setup")

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
    #end for
#end for

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
    #end for
#end for

#################################################################
# connectivity
#################################################################

print("connectivity")

icon=np.zeros((m,nel),dtype=np.int32)

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
    #end for
#end for

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  
bc_val = np.zeros(Nfem, dtype=np.float64) 

for i in range(0, nnp):
    if r[i]<R1+eps:
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0 #velocity_x(x[i],y[i],R1,R2,kk,g0)
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 #velocity_y(x[i],y[i],R1,R2,kk,g0)
    if r[i]>(R2-eps): 
       #bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = np.sin(theta[i]) #velocity_x(x[i],y[i],R1,R2,kk,g0)
       #bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -np.cos(theta[i]) #velocity_y(x[i],y[i],R1,R2,kk,g0)
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0 #velocity_x(x[i],y[i],R1,R2,kk,g0)
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 #velocity_y(x[i],y[i],R1,R2,kk,g0)
#end for

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
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros(m*ndof)
    a_el = np.zeros((m*ndof,m*ndof), dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

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
            a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity*wq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                b_el[2*i  ]+=N[i]*jcob*wq*gx(xq,yq,g0)*density(xq,yq,R1,R2,kk)
                b_el[2*i+1]+=N[i]*jcob*wq*gy(xq,yq,g0)*density(xq,yq,R1,R2,kk)

        #end for
    #end for

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    wq=2.*2.

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    # compute the jacobian
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)

    # compute dNdx and dNdy
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    # compute gradient matrix
    for i in range(0,m):
        b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                              [0.     ,dNdy[i]],
                              [dNdy[i],dNdx[i]]]

    # compute elemental matrix
    a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*wq*jcob

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
# put solution into separate x,y velocity arrays
#####################################################################
start = time.time()

u,v=np.reshape(sol,(nnp,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %.4f %.4f " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4f %.4f " %(np.min(vt),np.max(vt)))

print("reshape solution (%.3fs)" % (time.time() - start))

#####################################################################
# retrieve pressure
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
divv= np.zeros(nel,dtype=np.float64)  

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

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    jcob=np.linalg.det(jcb)
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

    p[iel]=-penalty*(exx[iel]+eyy[iel])
    divv[iel]=exx[iel]+eyy[iel]

#end for

print("p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

#np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p & sr | time: %.3f s" % (time.time() - start))

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
            errv+=((uq-velocity_x(xq,yq,R1,R2,kk,g0))**2+(vq-velocity_y(xq,yq,R1,R2,kk,g0))**2)*wq*jcob
            errp+=(p[iel]-pressure(xq,yq,R1,R2,kk))**2*wq*jcob
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors | time: %.3f s" % (time.time() - start))

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
       vtufile.write("%f\n" % pressure(xc[iel],yc[iel],R1,R2,kk))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (p[iel]-pressure(xc[iel],yc[iel],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (divv[iel]))
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
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (r,theta)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(velocity_x(x[i],y[i],R1,R2,kk,g0),velocity_y(x[i],y[i],R1,R2,kk,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i]-velocity_x(x[i],y[i],R1,R2,kk,g0),v[i]-velocity_y(x[i],y[i],R1,R2,kk,g0),0.))
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
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %density(x[i],y[i],R1,R2,kk))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='Psi' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %Psi(x[i],y[i],R1,R2,kk))
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
