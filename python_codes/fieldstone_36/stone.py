import numpy as np
import sys as sys
import scipy.sparse as sps
import time as clock 

###############################################################################

def basis_functions_V(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-s)
    dNdr1=+0.25*(1.-s)
    dNdr2=+0.25*(1.+s)
    dNdr3=-0.25*(1.+s)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

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
    theta=np.arctan2(y,x)
    C1 = rho0 * g0 / (lambdaa + 2 * mu) / 3.
    k1 = (2*mu + lambdaa) * C1 * (2 * R1**2 * R2**3 - R1**3 * R2**2)
    k2 = lambdaa * C1 * (R1**2 * R2**3 - R1**3 * R2**2)
    C3 = (k1 + k2) / (( (R2**2+R1**2)*(2*mu+lambdaa) )  +  lambdaa * (R2**2-R1**2) )
    C2 = -C1 * R1 - C3 / R1**2
    vr= C1*r**2 + C2*r + C3/r
    val=vr*np.cos(theta)
    return val

def displacement_y(x,y,R1,R2,rho,g0,lambdaa,mu):
    r=np.sqrt(x*x+y*y)
    theta=np.arctan2(y,x)
    C1 = rho0 * g0 / (lambdaa + 2 * mu) / 3.
    k1 = (2*mu + lambdaa) * C1 * (2 * R1**2 * R2**3 - R1**3 * R2**2)
    k2 = lambdaa * C1 * (R1**2 * R2**3 - R1**3 * R2**2)
    C3 = (k1 + k2) / (( (R2**2+R1**2)*(2*mu+lambdaa) )  +  lambdaa * (R2**2-R1**2) )
    C2 = -C1 * R1 - C3 / R1**2
    vr= C1*r**2 + C2*r + C3/r
    val=vr*np.sin(theta)
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

###############################################################################

print("*******************************")
print("********** stone 036 **********")
print("*******************************")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

if int(len(sys.argv) == 3):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
else:
   nelr = 24
   visu = 1

debug=True

R1=2890e3
R2=6371e3
area=np.pi*(R2**2-R1**2)

dr=(R2-R1)/nelr 
nelt=int(2.*np.pi*R2/dr)
nel=nelr*nelt  # number of elements, total
nnr=nelr+1 # number of nodes radial direction
nnt=nelt # number of nodes tangential direction
nn_V=nnr*nnt  # number of nodes
Nfem=nn_V*ndof  # Total number of degrees of freedom

rho0=3300.
g0=9.81
E=6e10 # Young's modulus
nu=0.49 # Poisson ratio
mu=E/2/(1+nu)
lambdaa=E*nu/(1+nu)/(1-2*nu)

print('R1=',R1)
print('R2=',R2)
print('g0=',g0)
print('rho0=',rho0)
print('nu=',nu)
print('mu=',mu)
print('lambda=',lambdaa)
print('Nfem=',Nfem)
print('nnr=',nnr)
print('nnt=',nnt)
print('nn_V=',nn_V)
print("*******************************")

eps=1.e-10

sqrt3=np.sqrt(3.)

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y=np.zeros(nn_V,dtype=np.float64)  # y coordinates
r=np.zeros(nn_V,dtype=np.float64)  
theta=np.zeros(nn_V,dtype=np.float64) 

Louter=2.*np.pi*R2
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
        t=xi/Louter*2.*np.pi    
        x[counter]=np.cos(t)*(R1+yi)
        y[counter]=np.sin(t)*(R1+yi)
        r[counter]=R1+yi
        theta[counter]=np.arctan2(y[counter],x[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*np.pi
        counter+=1

print("grid setup (%.3fs)" % (clock.time()-start))

###############################################################################
# connectivity array
###############################################################################
start=clock.time()

icon=np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0,nelr):
    for i in range(0,nelt):
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

print("connectivity (%.3fs)" % (clock.time()-start))

###############################################################################
# define boundary conditions: no slip at bottom
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  
bc_val=np.zeros(Nfem,dtype=np.float64) 

for i in range(0,nn_V):
    if r[i]<R1*(1+eps):
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0 
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0

print("defining boundary conditions (%.3fs)" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

A_fem = np.zeros((Nfem,Nfem),dtype=np.float64) # matrix of Ax=b
b_fem=np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
B=np.zeros((3,ndof*m),dtype=np.float64)        # gradient matrix 
jcb=np.zeros((2,2),dtype=np.float64)

C=np.array([[2*mu+lambdaa,lambdaa     , 0],
            [lambdaa     ,2*mu+lambdaa, 0],
            [0           ,           0,mu]],dtype=np.float64) 

for iel in range(0,nel):

    A_el=np.zeros((m*ndof,m*ndof),dtype=np.float64)
    b_el=np.zeros((m*ndof),dtype=np.float64)

    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x[icon[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y[icon[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x[icon[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y[icon[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            jcbi=np.linalg.inv(jcb)
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            xq=np.dot(N_V,x[icon[:,iel]])
            yq=np.dot(N_V,y[icon[:,iel]])

            for i in range(0,m):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            A_el+=B.T.dot(C.dot(B))*JxWq

            for i in range(0,m):
                b_el[2*i  ]+=N_V[i]*gx(xq,yq,g0)*rho0*JxWq
                b_el[2*i+1]+=N_V[i]*gy(xq,yq,g0)*rho0*JxWq

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

    # assemble matrix A_fem and right hand side b_fem
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
            b_fem[m1]+=b_el[ikk]

print("build FE matrixs & rhs (%.3fs)" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solving system (%.3fs)" % (clock.time()-start))

###############################################################################
# put solution into separate x,y displacement arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

if debug: np.savetxt('displacement_xy.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %.4e %.4e " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4e %.4e " %(np.min(vt),np.max(vt)))

if debug: np.savetxt('displacement_rtheta.ascii',np.array([np.sqrt(x**2+y**2),vr,vt]).T,header='#r,vr,vtheta')

print("reshape solution (%.3fs)" % (clock.time()-start))

###############################################################################
# retrieve elemental pressure and strain tensor components 
###############################################################################
start=clock.time()

x_e = np.zeros(nel,dtype=np.float64)  
y_e = np.zeros(nel,dtype=np.float64)  
p   = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.0
    sq = 0.0
    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x[icon[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y[icon[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x[icon[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y[icon[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    x_e[iel]=np.dot(N_V,x[icon[:,iel]])
    y_e[iel]=np.dot(N_V,y[icon[:,iel]])
    exx[iel]=np.dot(dNdx_V,u[icon[:,iel]])
    eyy[iel]=np.dot(dNdy_V,v[icon[:,iel]])
    exy[iel]=np.dot(dNdx_V,v[icon[:,iel]])*0.5+\
             np.dot(dNdy_V,u[icon[:,iel]])*0.5
    p[iel]=-(lambdaa+2*mu/3)*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('pressure.ascii',np.array([x_e,y_e,p,np.sqrt(x_e**2+y_e**2)]).T,header='# x,y,p')
   np.savetxt('strain.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute p & sr (%.3f s)" % (clock.time()-start))

###############################################################################
# compute error
# errors are normalised by the area !
###############################################################################
start=clock.time()

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
            jcb[0,0]=np.dot(dNdr_V,x[icon[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y[icon[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x[icon[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y[icon[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x[icon[:,iel]])
            yq=np.dot(N_V,y[icon[:,iel]])
            uq=np.dot(N_V,u[icon[:,iel]])
            vq=np.dot(N_V,v[icon[:,iel]])
            errv+=((uq-displacement_x(xq,yq,R1,R2,rho0,g0,lambdaa,mu))**2+\
                   (vq-displacement_y(xq,yq,R1,R2,rho0,g0,lambdaa,mu))**2)*JxWq
            errp+=(p[iel]-pressure(xq,yq,R1,R2,rho0,g0,lambdaa,mu))**2*JxWq

errv=np.sqrt(errv)/area
errp=np.sqrt(errp)/area

print("     -> nel= %6d ; errv= %.4e ; errp= %.4e" %(nel,errv,errp))

print("compute errors (%.3f s)" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(x[i],y[i],0.))
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
       vtufile.write("%f\n" % pressure(x_e[iel],y_e[iel],R1,R2,rho0,g0,lambdaa,mu))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % (p[iel]-pressure(x_e[iel],y_e[iel],R1,R2,rho0,g0,lambdaa,mu)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(gx(x[i],y[i],g0),gy(x[i],y[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (r,theta)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)(r,theta)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(displacement_r(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),0.,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(displacement_x(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),displacement_y(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (error)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i]-displacement_x(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),v[i]-displacement_y(x[i],y[i],R1,R2,rho0,g0,lambdaa,mu),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %r[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
   for i in range(0,nn_V):
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
   print("export to vtu (%.3f s)" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
