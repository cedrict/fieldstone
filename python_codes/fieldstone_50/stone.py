import numpy as np
import math as math
import sys as sys
import time as clock
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################
# defining Q2xQ1 basis functions and their derivatives
###############################################################################

def basis_functions_V(rq,sq):
    N_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    N_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    N_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    N_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    N_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    N_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    N_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    N_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    N_8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

def basis_functions_V_dr(rq,sq):
    dNdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNdr_8=       (-2.*rq) *    (1.-sq**2)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

def basis_functions_V_ds(rq,sq):
    dNds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNds_8=     (1.-rq**2) *       (-2.*sq)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)

def basis_functions_P(rq,sq):
    N_0=0.25*(1-rq)*(1-sq)
    N_1=0.25*(1+rq)*(1-sq)
    N_2=0.25*(1+rq)*(1+sq)
    N_3=0.25*(1-rq)*(1+sq)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

###############################################################################
# density and viscosity functions
###############################################################################

def rho(xq,yq,imat,Tq):
    if imat==1:
       rho0=2800. ; alpha=2.5e-5 ; T0=273.15
    elif imat==2:
       rho0=2900. ; alpha=2.5e-5 ; T0=273.15
    elif imat==3:
       rho0=3300. ; alpha=2.5e-5 ; T0=500+273.15
    elif imat==4:
       rho0=3300. ; alpha=2.5e-5 ; T0=500+273.15
    return rho0*(1.-alpha*(Tq-T0))

def eta(xq,yq,imat,exx,eyy,exy,p,T):
    Rgas=8.314
    eta_min=1.e19
    eta_max=1.e26

    if imat==1: # upper crust
       phi=20. ; c=20e6 ; A=8.57e-28 ; Q=223e3 ; n=4 ; V=0 ; f=1
    elif imat==2: # lower crust
       phi=20. ; c=20e6 ; A=7.13e-18 ; Q=345e3 ; n=3 ; V=0 ; f=1
    elif imat==3: # mantle
       phi=20. ; c=20e6 ; A=6.52e-16 ; Q=530e3 ; n=3.5 ; V=18e-6 ; f=1 
    else:  # seed
       phi=20. ; c=20e6 ; A=7.13e-18 ; Q=345e3 ; n=3  ; V=0 ; f=1 

    phi*=(np.pi/180)

    E2=np.sqrt( 0.5*(exx**2+eyy**2)+exy**2 )
    #print (exx,eyy,exy,imat,T,p)
    # compute effective viscous viscosity
    if E2<1e-20:
       etaeff_v=eta_max
    else:
       etaeff_v= 0.5 *f*A**(-1./n) * E2**(1./n-1.) * np.exp(max(Q+p*V,Q)/n/Rgas/T)
    # compute effective plastic viscosity
    if E2<1e-20: 
       etaeff_p=eta_max
    else:
       etaeff_p=( max(p*np.sin(phi)+c*np.cos(phi),c*np.cos(phi)) )/E2 * 0.5
    # blend the two viscosities
    etaeffq=1./(1./etaeff_p+1./etaeff_v)
    etaeffq=min(etaeffq,eta_max)
    etaeffq=max(etaeffq,eta_min)
    return etaeffq

###############################################################################
# boundary condition functions
###############################################################################

def bc_fct_left(x,y):
    return -0.25*cm/year
    
def bc_fct_right(x,y):
    return 0.25*cm/year

def bc_fct_bottom(x,y):
    return 0.125*cm/year

###############################################################################

def onePlot(variable, plotX, plotY, title, labelX, labelY, extVal, limitX, limitY, colorMap):
    im=axes[plotX][plotY].imshow(np.flipud(variable),extent=extVal,cmap=colorMap,interpolation="nearest")
    axes[plotX][plotY].set_title(title,fontsize=10, y=1.01)
    if (limitX != 0.0):
       axes[plotX][plotY].set_xlim(0,limitX)
    if (limitY != 0.0):
       axes[plotX][plotY].set_ylim(0,limitY)
    axes[plotX][plotY].set_xlabel(labelX)
    axes[plotX][plotY].set_ylabel(labelY)
    fig.colorbar(im,ax=axes[plotX][plotY])
    return

###############################################################################

print("-----------------------------")
print("---------- stone 50 ---------")
print("-----------------------------")

cm=0.01
year=365.25*24*3600
MPa=1e6

ndim=2
m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node
ndof_P=1  # number of pressure degrees of freedom 

Lx=400e3  # horizontal extent of the domain 
Ly=100e3  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx =int(sys.argv[1])
   nely =int(sys.argv[2])
   visu =int(sys.argv[3])
   niter=int(sys.argv[4])
else:
   nelx=160
   nely=40
   visu=1
   niter=40 #50
    
nn_V=(2*nelx+1)*(2*nely+1) # number of V nodes
nn_P=(nelx+1)*(nely+1)     # number of P nodes
nel=nelx*nely              # number of elements, total
Nfem_V=nn_V*ndof_V         # number of velocity dofs
Nfem_P=nn_P                # number of pressure dofs
Nfem=Nfem_V+Nfem_P         # total number of dofs

nq_per_dim=3
nqel=nq_per_dim**ndim
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

gx=0.
gy=-9.81

eta_ref=1e23 # scaling of G blocks

tol=1e-4 # nonlinear iterations convergence tolerance

gamma_eta=1   #gamma=1 -> no relax
gamma_uvp=0.1 #gamma=1 -> no relax

debug=False
   
use_matplotlib=False

###############################################################################

print("nelx=",nelx)
print("nely=",nely)
print("nel=",nel)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("hx=",hx)
print("hy=",hy)
print("---------------------------------------")

conv_file=open('conv.ascii',"w")
stats_file=open('statistics.ascii',"w")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates

counter=0    
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        x_V[counter]=i*hx/2
        y_V[counter]=j*hy/2
        counter+=1

x_P=np.empty(nn_P,dtype=np.float64) # x coordinates
y_P=np.empty(nn_P,dtype=np.float64) # y coordinates

counter=0    
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_P[counter]=i*hx
        y_P[counter]=j*hy
        counter+=1

if debug:
   np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')
   np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

nnx=2*nelx+1
nny=2*nely+1

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
        icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
        icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
        icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

eps=1.e-8

bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=bc_fct_left(x_V[i],y_V[i])
    if x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=bc_fct_right(x_V[i],y_V[i])
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=bc_fct_bottom(x_V[i],y_V[i])

print("setup: boundary conditions: %.3f s" % (clock.time() - start))

###############################################################################
# define temperature field
###############################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64) 

for i in range(0,nn_V):
    if y_V[i]>70.e3:
       T[i]=-(y_V[i]-100.e3)*(500.)/30.e3+0
    else:
       T[i]=-(y_V[i]-70.e3)*(700.)/70.e3+500

T[:]=T[:]+273.15

print("setup: initial temperature: %.3f s" % (clock.time() - start))

###############################################################################
# define material layout
###############################################################################
start=clock.time()

material=np.zeros(nel,dtype=np.int32) 

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0 ; sq=0
    N_V=basis_functions_V(rq,sq)
    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    if yc[iel]>80.e3:
       material[iel]=1
    elif yc[iel]>70.e3:
       material[iel]=2
    else:
       material[iel]=3

    # weak zone 
    if yc[iel]<68.e3 and yc[iel]>60.e3 and xc[iel]>=198.e3 and xc[iel]<=202.e3:
       material[iel]=4

print("setup: material layout: %.3f s" % (clock.time() - start))

###############################################################################
# sanity check / compute total area 
###############################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[iq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %e " %(area.sum()))
print("     -> total area (anal) %e " %(Lx*Ly))

print("compute elements areas: %.3f s" % (clock.time() - start))

###############################################################################
###############################################################################
# non linear iterations
###############################################################################
###############################################################################
u=np.zeros(nn_V,dtype=np.float64)
v=np.zeros(nn_V,dtype=np.float64)
p=np.zeros(nn_P,dtype=np.float64) 
u_mem=np.zeros(nn_V,dtype=np.float64)  
v_mem=np.zeros(nn_V,dtype=np.float64) 
p_mem=np.zeros(nn_P,dtype=np.float64) 
etaq=np.zeros(nel*nqel,dtype=np.float64)     # viscosity at quadrature points 
etaq_mem=np.zeros(nel*nqel,dtype=np.float64) 
    
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iter in range(0,niter):

    print('=======================')
    print('======= iter ',iter,'=======')
    print('=======================')

    #**************************************************************************
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #**************************************************************************
    start=clock.time()

    A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)  # FEM matrix
    b_fem=np.zeros(Nfem,dtype=np.float64)           # right hand side of Ax=b
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64)     # gradient matrix B 
    N_mat=np.zeros((3,m_P),dtype=np.float64)  # matrix  
    eta_el=np.zeros(nel,dtype=np.float64)           # elemental viscosity 
    rho_el=np.zeros(nel,dtype=np.float64)           # elemental density

    counterq=0

    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
        K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P*ndof_P),dtype=np.float64)
        h_el=np.zeros((m_P*ndof_P),dtype=np.float64)

        # integrate terms over 9 quadrature points
        for iq in [0,1,2]:
            for jq in [0,1,2]:

                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_V=basis_functions_V(rq,sq)
                N_P=basis_functions_P(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)

                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq

                dNdx_V=jcbi[0,0]*dNdr_V[:]+jcbi[0,1]*dNds_V[:]
                dNdy_V=jcbi[1,0]*dNdr_V[:]+jcbi[1,1]*dNds_V[:]

                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                pq=np.dot(N_P,p_mem[icon_P[:,iel]])

                exxq=np.dot(dNdx_V,u_mem[icon_V[:,iel]])
                eyyq=np.dot(dNdy_V,v_mem[icon_V[:,iel]])
                exyq=np.dot(dNdx_V,v_mem[icon_V[:,iel]])*0.5+\
                     np.dot(dNdy_V,u_mem[icon_V[:,iel]])*0.5

                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.     ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                etaq[counterq]=eta(xq,yq,material[iel],exxq,eyyq,exyq,pq,Tq)

                if iter>0:
                   etaq[counterq]=etaq[counterq]*gamma_eta+(1-gamma_eta)*etaq_mem[counterq]

                rhoq=rho(xq,yq,material[iel],Tq)

                eta_el[iel]+=etaq[counterq]/nqel # arithm averaging
                rho_el[iel]+=rhoq/nqel           # arithm averaging

                # compute elemental matrix
                K_el+=B.T.dot(C.dot(B))*etaq[counterq]*JxWq

                # compute elemental rhs vector
                for i in range(0,m_V):
                    f_el[ndof_V*i  ]+=N_V[i]*rhoq*gx*JxWq
                    f_el[ndof_V*i+1]+=N_V[i]*rhoq*gy*JxWq

                for i in range(0,m_P):
                    N_mat[0,i]=N_P[i]
                    N_mat[1,i]=N_P[i]
                    N_mat[2,i]=0.

                G_el-=B.T.dot(N_mat)*JxWq

                counterq+=1

            #end for jq
        #end for iq

        # impose b.c. 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                if bc_fix_V[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m_V*ndof_V):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val_V[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                   G_el[ikk,:]=0

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        A_fem[m1,m2] += K_el[ikk,jkk]
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]*eta_ref/Ly
                    A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]*eta_ref/Ly
                b_fem[m1]+=f_el[ikk]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            b_fem[Nfem_V+m2]+=h_el[k2]*eta_ref/Ly

    print("build FE matrix: %.5f s nel= %d time/elt %.5f" % (clock.time()-start,nel,(clock.time()-start)/nel))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

    print("solve time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

    if iter>0:
       u=u*gamma_uvp+u_mem*(1-gamma_uvp)
       v=v*gamma_uvp+v_mem*(1-gamma_uvp)
       p=p*gamma_uvp+p_mem*(1-gamma_uvp)

    print("     -> u (m,M) %.4f %.4f cm/yr" %(np.min(u)/cm*year,np.max(u)/cm*year))
    print("     -> v (m,M) %.4f %.4f cm/yr" %(np.min(v)/cm*year,np.max(v)/cm*year))
    print("     -> p (m,M) %.4f %.4f Mpa  " %(np.min(p)/1e6,np.max(p)/1e6))

    stats_file.write("%d %e %e %e %e %e %e %e %e\n" %(iter,\
                                                      np.min(u)/cm*year,np.max(u)/cm*year,\
                                                      np.min(v)/cm*year,np.max(v)/cm*year,\
                                                      np.min(p)/1e6,np.max(p)/1e6,\
                                                      np.sum(abs(u))/nn_V,np.sum(abs(v))/nn_V))
    stats_file.flush()

    if debug:
       np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
       np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (clock.time() - start))

    ###########################################################################

    #xi_u=np.linalg.norm(u-u_mem,2)/np.linalg.norm(u+u_mem,2)
    #xi_v=np.linalg.norm(v-v_mem,2)/np.linalg.norm(v+v_mem,2)
    #xi_p=np.linalg.norm(p-p_mem,2)/np.linalg.norm(p+p_mem,2)

    xi_u=np.linalg.norm((u-u_mem)/cm*year,2)/np.linalg.norm((u+u_mem)/cm*year,2)
    xi_v=np.linalg.norm((v-v_mem)/cm*year,2)/np.linalg.norm((v+v_mem)/cm*year,2)
    xi_p=np.linalg.norm((p-p_mem)/MPa,2)/np.linalg.norm((p+p_mem)/MPa,2)

    print("conv: u,v,p: %.6f %.6f %.6f" %(xi_u,xi_v,xi_p))
    conv_file.write("%d %e %e %e\n" %(iter,xi_u,xi_v,xi_p))
    conv_file.flush()

    if xi_u<tol and xi_v<tol and xi_p<tol:
       print('*** converged ***')
       break
    else:
       u_mem=u
       v_mem=v
       p_mem=p
       etaq_mem[:]=etaq[:]

#end for

###############################################################################
# compute elemental strainrate 
###############################################################################
start=clock.time()

exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.0 ; sq=0.0

    N_V=basis_functions_V(rq,sq)
    N_P=basis_functions_P(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)

    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)

    dNdx_V=jcbi[0,0]*dNdr_V[:]+jcbi[0,1]*dNds_V[:]
    dNdy_V=jcbi[1,0]*dNdr_V[:]+jcbi[1,1]*dNds_V[:]

    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5

    e[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time() - start))

###############################################################################
# interpolate pressure onto velocity grid points (Q1 -> Q2)
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    q[icon_V[0,iel]]=p[icon_P[0,iel]]
    q[icon_V[1,iel]]=p[icon_P[1,iel]]
    q[icon_V[2,iel]]=p[icon_P[2,iel]]
    q[icon_V[3,iel]]=p[icon_P[3,iel]]
    q[icon_V[4,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]])*0.5
    q[icon_V[5,iel]]=(p[icon_P[1,iel]]+p[icon_P[2,iel]])*0.5
    q[icon_V[6,iel]]=(p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.5
    q[icon_V[7,iel]]=(p[icon_P[3,iel]]+p[icon_P[0,iel]])*0.5
    q[icon_V[8,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]]+p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.25

#np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

print("interpolate pressure onto velocity mesh: %.3f s" % (clock.time() - start))

###############################################################################
# plot of solution
# the 9-node Q2 element does not exist in vtk, but the 8-node one does, i.e. type=23. 
###############################################################################
start=clock.time()

if visu:
   filename='solution.vtu'
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % eta_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % rho_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % exx[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % eyy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % exy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='material' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % material[iel])
   vtufile.write("</DataArray>\n")

   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity diff (cm/year)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10e %10e %10e \n" %((u[i]-u_mem[i])/cm*year,(v[i]-v_mem[i])/cm*year,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='bc_fix u' Format='ascii'> \n")
   for i in range(0,nn_V):
       if bc_fix_[i*ndof_V]:
          vtufile.write("%10e \n" % 1.)
       else:
          vtufile.write("%10e \n" % 0.)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='bc_fix v' Format='ascii'> \n")
   for i in range(0,nn_V):
       if bc_fix_T[i*ndof_V+1]:
          vtufile.write("%10e \n" % 1.)
       else:
          vtufile.write("%10e \n" % 0.)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (MPa)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10e \n" % (q[i]/MPa))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10e \n" %(T[i]-273.15))
   vtufile.write("</DataArray>\n")
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
       vtufile.write("%d \n" %((iel+1)*m_V))
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

   print("export to vtu: %.3f s" % (clock.time()-start))

   #***********************************************************************************************

   if use_matplotlib:

      import matplotlib.pyplot as plt
      fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(18,18))
      extent=(np.amin(x_V),np.amax(x_V),np.amin(y_V),np.amax(y_V))
      onePlot(u.reshape((nny,nnx)),        0, 0, "$v_x$",                 "x", "y", extent,  0,  0, 'Spectral_r')
      onePlot(v.reshape((nny,nnx)),        0, 1, "$v_y$",                 "x", "y", extent,  0,  0, 'Spectral_r')
      onePlot(q.reshape((nny,nnx)),        0, 2, "$p$",                   "x", "y", extent, Lx, Ly, 'RdGy_r')
      onePlot(exx.reshape((nely,nelx)),    1, 0, "$\dot{\epsilon}_{xx}$", "x", "y", extent, Lx, Ly, 'viridis')
      onePlot(eyy.reshape((nely,nelx)),    1, 1, "$\dot{\epsilon}_{yy}$", "x", "y", extent, Lx, Ly, 'viridis')
      onePlot(exy.reshape((nely,nelx)),    1, 2, "$\dot{\epsilon}_{xy}$", "x", "y", extent, Lx, Ly, 'viridis')
      onePlot(rho_el.reshape((nely,nelx)), 2, 0, "density",               "x", "y", extent, Lx, Ly, 'RdGy_r')
      onePlot(eta_el.reshape((nely,nelx)), 2, 1, "viscosity",             "x", "y", extent, Lx, Ly, 'RdGy_r')
      onePlot(T.reshape((nny,nnx)),        2, 2, "$T$",                   "x", "y", extent, Lx, Ly, 'RdGy_r')
      plt.subplots_adjust(hspace=0.5)
      plt.savefig('solution.pdf', bbox_inches='tight')
      plt.show()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
