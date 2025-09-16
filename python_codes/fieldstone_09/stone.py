import numpy as np
import math as math
import sys as sys
import scipy.sparse as sps
import time as clock
from scipy.sparse import csr_matrix,lil_matrix

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

def density(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2.+B/r**2*(1.-math.log(r))+1./r**2
    gppr=-B/r**3*(3.-2.*math.log(r))-2./r**3
    alephr=gppr - gpr/r -gr/r**2*(k**2-1.) +fr/r**2  +fpr/r
    val=k*math.sin(k*theta)*alephr + rho0 
    return val

def Psi(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    val=-r*gr*math.cos(k*theta)
    return val

def velocity_x(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    vr=k *gr * math.sin (k * theta)
    vtheta = fr *math.cos(k* theta)
    val=vr*math.cos(theta)-vtheta*math.sin(theta)
    return val

def velocity_y(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    vr=k *gr * math.sin (k * theta)
    vtheta = fr *math.cos(k* theta)
    val=vr*math.sin(theta)+vtheta*math.cos(theta)
    return val

def pressure(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    val=k*hr*math.sin(k*theta) + rho0*g0*(r-R2)
    return val

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 09 ***********")
print("*******************************")

m_V=4   # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

if int(len(sys.argv) == 3):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
else:
   nelr = 40
   visu = 1

R1=1. # inner radius
R2=2. # outer radius

dr=(R2-R1)/nelr
nelt=int(2.*math.pi*R2/dr)
nel=nelr*nelt  # number of elements, total

nnr=nelr+1
nnt=nelt
nn_V=nnr*nnt  # number of nodes

rho0=0.
kk=4
g0=1.

viscosity=1.  # dynamic viscosity \mu
penalty=1.e7  # penalty coefficient value

Nfem_V=nn_V*ndof # number of velocity dofs 
Nfem=Nfem_V # Total number of dofs

debug=False

print('nelr=',nelr)
print('nelt=',nelt)
print('nn_V=',nn_V)
print('Nfem_V=',Nfem_V)
print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start = clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
r=np.zeros(nn_V,dtype=np.float64)  
theta=np.zeros(nn_V,dtype=np.float64) 

Louter=2.*math.pi*R2
Lr=R2-R1
sx=Louter/float(nelt)
sz=Lr    /float(nelr)

counter=0
for j in range(0,nnr):
    for i in range(0,nelt):
        x_V[counter]=i*sx
        y_V[counter]=j*sz
        counter += 1

counter=0
for j in range(0,nnr):
    for i in range(0,nnt):
        xi=x_V[counter]
        yi=y_V[counter]
        t=xi/Louter*2.*math.pi    
        x_V[counter]=math.cos(t)*(R1+yi)
        y_V[counter]=math.sin(t)*(R1+yi)
        r[counter]=R1+yi
        theta[counter]=math.atan2(y_V[counter],x_V[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*math.pi
        counter+=1

print("building coordinate arrays (%.3fs)" % (clock.time() - start))

###############################################################################
# connectivity
###############################################################################
start = clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

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
        icon_V[0,counter] = icon2 
        icon_V[1,counter] = icon1
        icon_V[2,counter] = icon4
        icon_V[3,counter] = icon3
        counter += 1

print("building connectivity array (%.3fs)" % (clock.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  
bc_val_V=np.zeros(Nfem_V,dtype=np.float64) 

for i in range(0,nn_V):
    if r[i]<R1+eps:
       bc_fix_V[i*ndof  ]=True ; bc_val_V[i*ndof  ]=velocity_x(x_V[i],y_V[i],R1,R2,kk,rho0,g0)
       bc_fix_V[i*ndof+1]=True ; bc_val_V[i*ndof+1]=velocity_y(x_V[i],y_V[i],R1,R2,kk,rho0,g0)
    if r[i]>(R2-eps):
       bc_fix_V[i*ndof  ]=True ; bc_val_V[i*ndof  ]=velocity_x(x_V[i],y_V[i],R1,R2,kk,rho0,g0)
       bc_fix_V[i*ndof+1]=True ; bc_val_V[i*ndof+1]=velocity_y(x_V[i],y_V[i],R1,R2,kk,rho0,g0)

print("defining boundary conditions (%.3fs)" % (clock.time() - start))

###############################################################################
# build FE matrix
###############################################################################
start = clock.time()

jcb=np.zeros((2,2),dtype=np.float64)
A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64) 
B=np.zeros((3,ndof*m_V),dtype=np.float64) # gradient matrix B 
H=np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    A_el=np.zeros((m_V*ndof,m_V*ndof),dtype=np.float64)
    b_el=np.zeros((m_V*ndof),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
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

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental a_mat matrix
            A_el+=B.T.dot(C.dot(B))*viscosity*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                b_el[2*i  ]+=N_V[i]*gx(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)*JxWq
                b_el[2*i+1]+=N_V[i]*gy(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)*JxWq

        #end for jq
    #end for iq

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

    # apply boundary conditions
    for k1 in range(0,m_V):
        for i1 in range(0,ndof):
            m1 =ndof*icon_V[k1,iel]+i1
            if bc_fix_V[m1]: 
               fixt=bc_val_V[m1]
               ikk=ndof*k1+i1
               aref=A_el[ikk,ikk]
               for jkk in range(0,m_V*ndof):
                   b_el[jkk]-=A_el[jkk,ikk]*fixt
                   A_el[ikk,jkk]=0.
                   A_el[jkk,ikk]=0.
               A_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt

    # assemble matrix and right hand side
    for k1 in range(0,m_V):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
            b_fem[m1]+=b_el[ikk]

print("build FE matrix & rhs (%.3fs)" % (clock.time() - start))

###############################################################################
# solve system
###############################################################################
start = clock.time()

sol = sps.linalg.spsolve(A_fem.tocsr(),b_fem)

print("solving system (%.3fs)" % (clock.time() - start))

###############################################################################
# put solution into separate velocity arrays
###############################################################################
start = clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %.4f %.4f " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4f %.4f " %(np.min(vt),np.max(vt)))

print("reshape solution (%.3fs)" % (clock.time() - start))

###############################################################################
# retrieve pressure
###############################################################################
start = clock.time()

p=np.zeros(nel,dtype=np.float64)  
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0.0
    sq=0.0
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
    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute p & sr (%.3fs)" % (clock.time() - start))

###############################################################################
# compute error
###############################################################################
start = clock.time()

errv=0.
errp=0.
for iel in range(0,nel):
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
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            errv+=((uq-velocity_x(xq,yq,R1,R2,kk,rho0,g0))**2+\
                   (vq-velocity_y(xq,yq,R1,R2,kk,rho0,g0))**2)*JxWq
            errp+=(p[iel]-pressure(xq,yq,R1,R2,kk,rho0,g0))**2*JxWq
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors (%.3fs)" % (clock.time() - start))

###############################################################################
# plot of solution
###############################################################################
start = clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(x_V[i],y_V[i],0.))
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
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % exx[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % eyy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % exy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % pressure(xc[iel],yc[iel],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % (p[iel]-pressure(xc[iel],yc[iel],R1,R2,kk,rho0,g0)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(gx(x_V[i],y_V[i],g0),gy(x_V[i],y_V[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (x,y)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(velocity_x(x_V[i],y_V[i],R1,R2,kk,rho0,g0),\
                                           velocity_y(x_V[i],y_V[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i]-velocity_x(x_V[i],y_V[i],R1,R2,kk,rho0,g0),\
                                           v[i]-velocity_y(x_V[i],y_V[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (r,theta)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %r[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %theta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %density(x_V[i],y_V[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Psi' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %Psi(x_V[i],y_V[i],R1,R2,kk))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
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

   print("export to vtu file (%.3fs)" % (clock.time() - start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
