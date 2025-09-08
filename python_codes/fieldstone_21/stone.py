import numpy as np
import math as math
import sys as sys
import scipy.sparse as sps
import time as clock
from scipy.sparse import csr_matrix,lil_matrix

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

def sr_xx(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=err*(math.cos(theta))**2\
       +ett*(math.sin(theta))**2\
       -2*ert*math.sin(theta)*math.cos(theta)
    return val

def sr_yy(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=err*(math.sin(theta))**2\
       +ett*(math.cos(theta))**2\
       +2*ert*math.sin(theta)*math.cos(theta)
    return val

def sr_xy(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=ert*(math.cos(theta)**2-math.sin(theta)**2)\
       +(err-ett)*math.cos(theta)*math.sin(theta)
    return val

###############################################################################

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

###############################################################################

def basis_functions_V(rq,sq):
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,\
                     NV_5,NV_6,NV_7,NV_8],dtype=np.float64)

def basis_functions_V_dr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,\
                     dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

def basis_functions_V_ds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,\
                     dNVds_5,dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

def basis_functions_P(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)

###############################################################################

print("-------------------------------")
print("---------- stone 021 ----------")
print("-------------------------------")

ndim=2    # number of dimensions
m_V=9     # number of nodes making up an element
m_P=4     # number of nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

if int(len(sys.argv) == 3):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
else:
   nelr = 16 # Q1 elements in radial direction
   visu = 1

R1=1.
R2=2.

dr=(R2-R1)/nelr # element size in r direction
nelt=12*nelr    # number of elements in the tangential direction 
nel=nelr*nelt   # total number of elements

rho0=0.
kk=4
g0=1.

viscosity=1.  # dynamic viscosity \eta

eps=1.e-10

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

r_V=np.array([-1,1,1,-1,0,1,0,-1,0],np.float64)
s_V=np.array([-1,-1,1,1,-1,0,1,0,0],np.float64)

sparse=True

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

nnr=nelr+1
nnt=nelt
nn_V=nnr*nnt  # number of velocity nodes

x_V=np.empty(nn_V,dtype=np.float64)  # x coordinates
y_V=np.empty(nn_V,dtype=np.float64)  # y coordinates
rad_V=np.empty(nn_V,dtype=np.float64)  
theta_V=np.empty(nn_V,dtype=np.float64) 

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
        rad_V[counter]=R1+yi
        theta_V[counter]=math.atan2(y_V[counter],x_V[counter])
        if theta_V[counter]<0.:
           theta_V[counter]+=2.*math.pi
        counter+=1

print("building coordinate arrays (%.3fs)" % (clock.time() - start))

###############################################################################
# now that the grid has been built as if it was a Q1 grid, 
# we can simply use these same points to arrive at a Q2 
# connectivity array with 4 times less elements.
###############################################################################

nelr=nelr//2
nelt=nelt//2
nel=nel//4
nn_P=nelt*(nelr+1) 

Nfem_V=nn_V*ndof_V # Total number of degrees of V freedom 
Nfem_P=nn_P        # Total number of degrees of P freedom
Nfem=Nfem_V+Nfem_P # total number of dofs

print('nelr=',nelr)
print('nelt=',nelt)
print('nel=',nel)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon_V[0,counter]=2*counter+2 +2*j*nelt
        icon_V[1,counter]=2*counter   +2*j*nelt
        icon_V[2,counter]=icon_V[1,counter]+4*nelt
        icon_V[3,counter]=icon_V[1,counter]+4*nelt+2
        icon_V[4,counter]=icon_V[0,counter]-1
        icon_V[5,counter]=icon_V[1,counter]+2*nelt
        icon_V[6,counter]=icon_V[2,counter]+1
        icon_V[7,counter]=icon_V[5,counter]+2
        icon_V[8,counter]=icon_V[5,counter]+1
        if i==nelt-1:
           icon_V[0,counter]-=2*nelt
           icon_V[7,counter]-=2*nelt
           icon_V[3,counter]-=2*nelt
        #print(j,i,counter,'|',icon_V[0:m_V,counter])
        counter += 1

icon_P=np.zeros((m_P,nel),dtype=np.int32)

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
        icon_P[0,counter] = icon2 
        icon_P[1,counter] = icon1
        icon_P[2,counter] = icon4
        icon_P[3,counter] = icon3
        counter += 1
    #end for


#for iel in range(0,nel):
#    print(iel,'|',icon_P[:,iel])

#now that I have both connectivity arrays I can 
# easily build xP,yP

x_P=np.empty(nn_P,dtype=np.float64)  # x coordinates
y_P=np.empty(nn_P,dtype=np.float64)  # y coordinates

x_P[icon_P[0,:]]=x_V[icon_V[0,:]]
x_P[icon_P[1,:]]=x_V[icon_V[1,:]]
x_P[icon_P[2,:]]=x_V[icon_V[2,:]]
x_P[icon_P[3,:]]=x_V[icon_V[3,:]]
y_P[icon_P[0,:]]=y_V[icon_V[0,:]]
y_P[icon_P[1,:]]=y_V[icon_V[1,:]]
y_P[icon_P[2,:]]=y_V[icon_V[2,:]]
y_P[icon_P[3,:]]=y_V[icon_V[3,:]]

print("building connectivity array (%.3fs)" % (clock.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  
bc_val=np.zeros(Nfem,dtype=np.float64) 

for i in range(0,nn_V):
    if rad_V[i]<R1+eps:
       bc_fix[i*ndof_V]  =True ; bc_val[i*ndof_V]  =velocity_x(x_V[i],y_V[i],R1,R2,kk,rho0,g0)
       bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]=velocity_y(x_V[i],y_V[i],R1,R2,kk,rho0,g0)
    if rad_V[i]>(R2-eps):
       bc_fix[i*ndof_V]  =True ; bc_val[i*ndof_V]  =velocity_x(x_V[i],y_V[i],R1,R2,kk,rho0,g0)
       bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]=velocity_y(x_V[i],y_V[i],R1,R2,kk,rho0,g0)

print("defining boundary conditions (%.3fs)" % (clock.time() - start))

###############################################################################
# compute area of elements
###############################################################################
start=clock.time()

area=np.zeros(nel,dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):
    for iq in [0,1,2]:
        for jq in [0,1,2]:
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))
print("     -> total area (anal) %.6f " %(np.pi*(R2**2-R1**2)))

print("compute elements areas: %.3f s" % (clock.time() - start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

if sparse:
   A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
   G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
f_rhs=np.zeros(Nfem_V,dtype=np.float64)            # right hand side f 
h_rhs=np.zeros(Nfem_P,dtype=np.float64)            # right hand side h 

B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
N_mat=np.zeros((3,m_P),dtype=np.float64)    # N matrix  
c_mat=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            N_P=basis_functions_P(rq,sq)
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq

            # coords of quad point
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])

            # compute dNdx & dNdy
            dNdx_V=jcbi[0,0]*dNdr_V[:]+jcbi[0,1]*dNds_V[:]
            dNdy_V=jcbi[1,0]*dNdr_V[:]+jcbi[1,1]*dNds_V[:]

            # construct B matrix 
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.     ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental a_mat matrix
            K_el+=B.T.dot(c_mat.dot(B))*viscosity*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*JxWq*gx(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)
                f_el[ndof_V*i+1]+=N_V[i]*JxWq*gy(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)
            #end for 

            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]
                N_mat[2,i]=0.
            #end for 

            G_el-=B.T.dot(N_mat)*JxWq

        #end for jq
    #end for iq

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
            #end if 
        #end for 
    #end for 

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    if sparse:
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                if sparse:
                   A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                   A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]
    #end for 

#end for iel

if not sparse:
   print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))

print("build FE matrixs & rhs (%.3fs)" % (clock.time() - start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

if not sparse:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)
   a_mat[0:Nfem_V,0:Nfem_V]=K_mat
   a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
   a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T

rhs=np.zeros(Nfem,dtype=np.float64)
rhs[0:Nfem_V]=f_rhs
rhs[Nfem_V:Nfem]=h_rhs
    
if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solving system (%.3fs)" % (clock.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

vr= np.cos(theta_V)*u+np.sin(theta_V)*v
vt=-np.sin(theta_V)*u+np.cos(theta_V)*v
    
print("     -> vr (m,M) %.4f %.4f " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4f %.4f " %(np.min(vt),np.max(vt)))

print("reshape solution (%.3fs)" % (clock.time() - start))

###############################################################################
# compute strain rate - center to nodes - method 1
###############################################################################
start=clock.time()

count=np.zeros(nn_V,dtype=np.int32)  
Lxx1=np.zeros(nn_V,dtype=np.float64)  
Lxy1=np.zeros(nn_V,dtype=np.float64)  
Lyx1=np.zeros(nn_V,dtype=np.float64)  
Lyy1=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    rq=0.
    sq=0.
    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V[:]+jcbi[0,1]*dNds_V[:]
    dNdy_V=jcbi[1,0]*dNdr_V[:]+jcbi[1,1]*dNds_V[:]
    Lxx=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    Lxy=np.dot(dNdx_V[:],v[icon_V[:,iel]])
    Lyx=np.dot(dNdy_V[:],u[icon_V[:,iel]])
    Lyy=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    for i in range(0,m_V):
        inode=icon_V[i,iel]
        Lxx1[inode]+=Lxx
        Lxy1[inode]+=Lxy
        Lyx1[inode]+=Lyx
        Lyy1[inode]+=Lyy
        count[inode]+=1
    #end for
#end for
Lxx1/=count
Lxy1/=count
Lyx1/=count
Lyy1/=count

print("     -> Lxx1 (m,M) %.4f %.4f " %(np.min(Lxx1),np.max(Lxx1)))
print("     -> Lyy1 (m,M) %.4f %.4f " %(np.min(Lyy1),np.max(Lyy1)))
print("     -> Lxy1 (m,M) %.4f %.4f " %(np.min(Lxy1),np.max(Lxy1)))
print("     -> Lxy1 (m,M) %.4f %.4f " %(np.min(Lyx1),np.max(Lyx1)))

print("compute vel gradient meth-1 (%.3fs)" % (clock.time() - start))

###############################################################################
start=clock.time()

exx1=np.zeros(nn_V,dtype=np.float64)  
eyy1=np.zeros(nn_V,dtype=np.float64)  
exy1=np.zeros(nn_V,dtype=np.float64)  

exx1[:]=Lxx1[:]
eyy1[:]=Lyy1[:]
exy1[:]=0.5*(Lxy1[:]+Lyx1[:])

print("compute strain rate meth-1 (%.3fs)" % (clock.time() - start))

###############################################################################
# compute strain rate - corners to nodes - method 2
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)
count=np.zeros(nn_V,dtype=np.int32)  
Lxx2=np.zeros(nn_V,dtype=np.float64)  
Lxy2=np.zeros(nn_V,dtype=np.float64)  
Lyx2=np.zeros(nn_V,dtype=np.float64)  
Lyy2=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    for i in range(0,m_V):
        inode=icon_V[i,iel]
        rq=r_V[i]
        sq=s_V[i]
        N_P=basis_functions_P(rq,sq)
        N_V=basis_functions_V(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V[:]+jcbi[0,1]*dNds_V[:]
        dNdy_V=jcbi[1,0]*dNdr_V[:]+jcbi[1,1]*dNds_V[:]
        Lxx=np.dot(dNdx_V[:],u[icon_V[:,iel]])
        Lxy=np.dot(dNdx_V[:],v[icon_V[:,iel]])
        Lyx=np.dot(dNdy_V[:],u[icon_V[:,iel]])
        Lyy=np.dot(dNdy_V[:],v[icon_V[:,iel]])
        Lxx2[inode]+=Lxx
        Lxy2[inode]+=Lxy
        Lyx2[inode]+=Lyx
        Lyy2[inode]+=Lyy
        q[inode]+=np.dot(p[icon_P[0:m_P,iel]],N_P[0:m_P])
        count[inode]+=1
    #end for
#end for
Lxx2/=count
Lxy2/=count
Lyx2/=count
Lyy2/=count
q/=count

print("     -> Lxx2 (m,M) %.4f %.4f " %(np.min(Lxx2),np.max(Lxx2)))
print("     -> Lyy2 (m,M) %.4f %.4f " %(np.min(Lyy2),np.max(Lyy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lxy2),np.max(Lxy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lyx2),np.max(Lyx2)))

#np.savetxt('pressure.ascii',np.array([xV,yV,q]).T)
#np.savetxt('strainrate.ascii',np.array([xV,yV,Lxx,Lyy,Lxy,Lyx]).T)

print("compute vel gradient meth-2 (%.3fs)" % (clock.time() - start))

###############################################################################
start=clock.time()

exx2=np.zeros(nn_V,dtype=np.float64)  
eyy2=np.zeros(nn_V,dtype=np.float64)  
exy2=np.zeros(nn_V,dtype=np.float64)  

exx2[:]=Lxx2[:]
eyy2[:]=Lyy2[:]
exy2[:]=0.5*(Lxy2[:]+Lyx2[:])

print("compute strain rate meth-2 (%.3fs)" % (clock.time() - start))

###############################################################################
start=clock.time()

M_mat=lil_matrix((nn_V,nn_V),dtype=np.float64)
rhsLxx=np.zeros(nn_V,dtype=np.float64)
rhsLyy=np.zeros(nn_V,dtype=np.float64)
rhsLxy=np.zeros(nn_V,dtype=np.float64)
rhsLyx=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):

    M_el=np.zeros((m_V,m_V),dtype=np.float64)
    fLxx_el=np.zeros(m_V,dtype=np.float64)
    fLyy_el=np.zeros(m_V,dtype=np.float64)
    fLxy_el=np.zeros(m_V,dtype=np.float64)
    fLyx_el=np.zeros(m_V,dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            N_P=basis_functions_P(rq,sq)
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi = np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq

            # compute dNdx & dNdy
            dNdx_V=jcbi[0,0]*dNdr_V[:]+jcbi[0,1]*dNds_V[:]
            dNdy_V=jcbi[1,0]*dNdr_V[:]+jcbi[1,1]*dNds_V[:]

            Lxx=np.dot(dNdx_V[:],u[icon_V[:,iel]])
            Lxy=np.dot(dNdx_V[:],v[icon_V[:,iel]])
            Lyx=np.dot(dNdy_V[:],u[icon_V[:,iel]])
            Lyy=np.dot(dNdy_V[:],v[icon_V[:,iel]])

            M_el +=np.outer(N_V,N_V)*JxWq

            fLxx_el[:]+=N_V[:]*Lxx*JxWq
            fLyy_el[:]+=N_V[:]*Lyy*JxWq
            fLxy_el[:]+=N_V[:]*Lxy*JxWq
            fLyx_el[:]+=N_V[:]*Lyx*JxWq

        #end for
    #end for

    for k1 in range(0,m_V):
        m1=icon_V[k1,iel]
        for k2 in range(0,m_V):
            m2=icon_V[k2,iel]
            M_mat[m1,m2]+=M_el[k1,k2]
        #end for
        rhsLxx[m1]+=fLxx_el[k1]
        rhsLyy[m1]+=fLyy_el[k1]
        rhsLxy[m1]+=fLxy_el[k1]
        rhsLyx[m1]+=fLyx_el[k1]
    #end for

#end for

Lxx3=sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxx)
Lyy3=sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyy)
Lxy3=sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxy)
Lyx3=sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyx)

print("     -> Lxx3 (m,M) %.4f %.4f " %(np.min(Lxx3),np.max(Lxx3)))
print("     -> Lyy3 (m,M) %.4f %.4f " %(np.min(Lyy3),np.max(Lyy3)))
print("     -> Lxy3 (m,M) %.4f %.4f " %(np.min(Lxy3),np.max(Lxy3)))
print("     -> Lxy3 (m,M) %.4f %.4f " %(np.min(Lyx3),np.max(Lyx3)))

print("compute vel gradient meth-3 (%.3fs)" % (clock.time() - start))

###############################################################################
start=clock.time()

exx3=np.zeros(nn_V,dtype=np.float64)  
eyy3=np.zeros(nn_V,dtype=np.float64)  
exy3=np.zeros(nn_V,dtype=np.float64)  

exx3[:]=Lxx3[:]
eyy3[:]=Lyy3[:]
exy3[:]=0.5*(Lxy3[:]+Lyx3[:])

print("compute strain rate meth-3 (%.3fs)" % (clock.time() - start))

###############################################################################
# normalise pressure (pretty rough algorithm)
###############################################################################
start=clock.time()

#print(np.sum(q[0:2*nelt])/(2*nelt))
#print(np.sum(q[nnp-2*nelt:nnp])/(2*nelt))
#print(np.sum(p[0:nelt])/(nelt))

poffset=np.sum(q[0:2*nelt])/(2*nelt)

q-=poffset
p-=poffset

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> q (m,M) %.4f %.4f " %(np.min(q),np.max(q)))

print("normalise pressure (%.3fs)" % (clock.time() - start))

###############################################################################
# export pressure at both surfaces
###############################################################################
start = clock.time()

np.savetxt('q_R1.ascii',np.array([x_V[0:2*nelt],y_V[0:2*nelt],q[0:2*nelt],theta_V[0:2*nelt]]).T)
np.savetxt('q_R2.ascii',np.array([x_V[nn_V-2*nelt:nn_V],y_V[nn_V-2*nelt:nn_V],\
                                   q[nn_V-2*nelt:nn_V],theta_V[nn_V-2*nelt:nn_V]]).T)

np.savetxt('p_R1.ascii',np.array([x_P[0:nelt],y_P[0:nelt],p[0:nelt]]).T)
np.savetxt('p_R2.ascii',np.array([x_P[nn_P-nelt:nn_P],y_P[nn_P-nelt:nn_P],p[nn_P-nelt:nn_P]]).T)

print("export p&q on R1,R2 (%.3fs)" % (clock.time() - start))

###############################################################################
# compute discretisation errors
###############################################################################
start = clock.time()

vrms=0.
errv=0.    ; errp=0.    ; errq=0.
errexx1=0. ; erreyy1=0. ; errexy1=0.
errexx2=0. ; erreyy2=0. ; errexy2=0.
errexx3=0. ; erreyy3=0. ; errexy3=0.
for iel in range (0,nel):
    for iq in [0,1,2]:
        for jq in [0,1,2]:
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_P=basis_functions_P(rq,sq)
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
            pq=np.dot(N_P,p[icon_P[:,iel]])
            qq=np.dot(N_V,q[icon_V[:,iel]])
            exx1q=N_V.dot(exx1[icon_V[:,iel]])
            eyy1q=N_V.dot(eyy1[icon_V[:,iel]])
            exy1q=N_V.dot(exy1[icon_V[:,iel]])
            exx2q=N_V.dot(exx2[icon_V[:,iel]])
            eyy2q=N_V.dot(eyy2[icon_V[:,iel]])
            exy2q=N_V.dot(exy2[icon_V[:,iel]])
            exx3q=N_V.dot(exx3[icon_V[:,iel]])
            eyy3q=N_V.dot(eyy3[icon_V[:,iel]])
            exy3q=N_V.dot(exy3[icon_V[:,iel]])

            errv+=((uq-velocity_x(xq,yq,R1,R2,kk,rho0,g0))**2+\
                   (vq-velocity_y(xq,yq,R1,R2,kk,rho0,g0))**2)*JxWq
            errp+=(pq-pressure(xq,yq,R1,R2,kk,rho0,g0))**2*JxWq
            errq+=(qq-pressure(xq,yq,R1,R2,kk,rho0,g0))**2*JxWq

            errexx1+=(exx1q-sr_xx(xq,yq,R1,R2,kk))**2*JxWq
            erreyy1+=(eyy1q-sr_yy(xq,yq,R1,R2,kk))**2*JxWq
            errexy1+=(exy1q-sr_xy(xq,yq,R1,R2,kk))**2*JxWq
            errexx2+=(exx2q-sr_xx(xq,yq,R1,R2,kk))**2*JxWq
            erreyy2+=(eyy2q-sr_yy(xq,yq,R1,R2,kk))**2*JxWq
            errexy2+=(exy2q-sr_xy(xq,yq,R1,R2,kk))**2*JxWq
            errexx3+=(exx3q-sr_xx(xq,yq,R1,R2,kk))**2*JxWq
            erreyy3+=(eyy3q-sr_yy(xq,yq,R1,R2,kk))**2*JxWq
            errexy3+=(exy3q-sr_xy(xq,yq,R1,R2,kk))**2*JxWq

            vrms+=(uq**2+vq**2)*JxWq

        # end for jq
    # end for iq
# end for iel

errv=np.sqrt(errv)       ; errp=np.sqrt(errp)       ; errq=np.sqrt(errq)
errexx1=np.sqrt(errexx1) ; erreyy1=np.sqrt(erreyy1) ; errexy1=np.sqrt(errexy1)
errexx2=np.sqrt(errexx2) ; erreyy2=np.sqrt(erreyy2) ; errexy2=np.sqrt(errexy2)
errexx3=np.sqrt(errexx3) ; erreyy3=np.sqrt(erreyy3) ; errexy3=np.sqrt(errexy3)

vrms=np.sqrt(vrms/np.pi/(R2**2-R1**2))

print('     -> nelr=',nelr,' vrms=',vrms)
print("     -> nelr= %d ; errv= %e ; errp= %e ; errq= %e" %(nelr,errv,errp,errq))
print("     -> nelr= %d ; errexx1= %e ; erreyy1= %e ; errexy1= %e" %(nelr,errexx1,erreyy1,errexy1))
print("     -> nelr= %d ; errexx2= %e ; erreyy2= %e ; errexy2= %e" %(nelr,errexx2,erreyy2,errexy2))
print("     -> nelr= %d ; errexx3= %e ; erreyy3= %e ; errexy3= %e" %(nelr,errexx3,erreyy3,errexy3))

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
   vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10f \n" %area[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(gx(x_V[i],y_V[i],g0),gy(x_V[i],y_V[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(x,y)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(th)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%13e %13e %13e \n" %(velocity_x(x_V[i],y_V[i],R1,R2,kk,rho0,g0),\
                                           velocity_y(x_V[i],y_V[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(error)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i]-velocity_x(x_V[i],y_V[i],R1,R2,kk,rho0,g0),\
                                           v[i]-velocity_y(x_V[i],y_V[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %rad_V[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %theta_V[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %density(x_V[i],y_V[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='Psi' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %Psi(x_V[i],y_V[i],R1,R2,kk))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx (th)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %(sr_xx(x_V[i],y_V[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy (th)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %(sr_yy(x_V[i],y_V[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy (th)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %(sr_xy(x_V[i],y_V[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx1' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %exx1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy1' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %eyy1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy1' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %exy1[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx2' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %exx2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy2' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %eyy2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy2' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %exy2[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %exx3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %eyy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %exy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (th)' Format='ascii'> \n")
   for i in range (0,nn_V):
       vtufile.write("%f\n" % pressure(x_V[i],y_V[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                      icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                      icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*9))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %28)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu file (%.3fs)" % (clock.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
