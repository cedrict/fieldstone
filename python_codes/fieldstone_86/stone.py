import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt
import time as clock

eps=1.e-10
sqrt3=np.sqrt(3.)
year=365.25*24*3600

###############################################################################
# Q1 basis functions and derivatives
###############################################################################

def basis_functions_T(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_T_dr(r,s):
    dNdr0=-0.25*(1.-s) 
    dNdr1=+0.25*(1.-s) 
    dNdr2=+0.25*(1.+s) 
    dNdr3=-0.25*(1.+s) 
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_T_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

def plot_T_field(Tnew, xcoords, ycoords, nnx, nny, Title, savename, option):
    Tplot = Tnew.reshape(nny, nnx)
    xplot = xcoords.reshape(nny, nnx)
    yplot = ycoords.reshape(nny, nnx)
    plt.subplot(int(Lx/Ly),1,2)
    plt.title(Title,fontsize=7)
    plt.xlabel('x [km]',fontsize=7)
    plt.ylabel('y [km]',fontsize=7)
    a = plt.contourf(xplot/1000, yplot/1000, Tplot, 21,  cmap = plt.cm.coolwarm)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=5)
    if option==0:
       cbar.set_label('Temperature [K]',fontsize=5)
    if option==1:
       cbar.set_label('material id',fontsize=5)
       plt.grid(linestyle='--', linewidth=0.25)
    plt.savefig(savename, dpi=600,bbox_inches='tight')
    #plt.show()
    plt.close()
    return None

###############################################################################
# geometrical parameters
###############################################################################

Lx=800e3
Ly=120e3
WR=400e3
w=50e3

d_S=5e3
d_UC=20e3
d_LC=20e3
d_LM=Ly-d_UC-d_LC

beta_UC=2
beta_LC=2.2
beta_LM=2.75

###############################################################################
# material properties
###############################################################################

rho_sediments=2100
hcapa_sediments=790
hcond_sediments=2
hprod_sediments=0

rho_uppercrust=2900
hcapa_uppercrust=1100
hcond_uppercrust=2.6
hprod_uppercrust=0

rho_lowercrust=2900
hcapa_lowercrust=1100
hcond_lowercrust=2.6
hprod_lowercrust=0

rho_lithosphere=3400
hcapa_lithosphere=1260
hcond_lithosphere=3
hprod_lithosphere=0

rho_asthenosphere=3400
hcapa_asthenosphere=1260
hcond_asthenosphere=3
hprod_asthenosphere=0

###############################################################################
# temperature parameters
###############################################################################

T_surface = 0
T_sediments_base = 200
T_uppercrust_base = 310
T_moho = 550
T_lab = 1330

###############################################################################

nelx=400 #800
nely=120 #240

nstep=100

dt=2e3*year

compute_ss=False

debug=True

###############################################################################

nel=nelx*nely
nnx=nelx+1
nny=nely+1
nn_T=nnx*nny
hx=Lx/nelx
hy=Ly/nely
Nfem_T=nn_T
m_T=4

hcond=[hcond_sediments,\
       hcond_uppercrust,\
       hcond_lowercrust,\
       hcond_lithosphere,\
       hcond_asthenosphere]

hcapa=[hcapa_sediments,\
       hcapa_uppercrust,\
       hcapa_lowercrust,\
       hcapa_lithosphere,\
       hcapa_asthenosphere]

hprod=[hprod_sediments,\
       hprod_uppercrust,\
       hprod_lowercrust,\
       hprod_lithosphere,\
       hprod_asthenosphere]

rho=[rho_sediments,\
     rho_uppercrust,\
     rho_lowercrust,\
     rho_lithosphere,\
     rho_asthenosphere]

print("*******************************")
print("********** stone 086 **********")
print("*******************************")
print('nelx=',nelx)
print('nely=',nely)
print('nn_T=',nn_T)
print('Nfem_T=',Nfem_T)
print('dt (year)=',dt/year)
print('Lx=',Lx)
print('Ly=',Ly)

###############################################################################
# geometry parameters 
###############################################################################

x1=Lx/2-WR/2-w
x2=Lx/2-WR/2
x3=Lx/2+WR/2
x4=Lx/2+WR/2+w

y4=Ly-d_S
y6=y4-d_UC/beta_UC
y5=Ly-d_UC
y3=y6-d_LC/beta_LC
y2=Ly-d_UC-d_LC
y1=y3-d_LM/beta_LM

print('y1=',y1)
print('y2=',y2)
print('y3=',y3)
print('y4=',y4)
print('y5=',y5)
print('y6=',y6)

xA=x1
yA=Ly
TA=T_surface

xB=x2
yB=y4
TB=T_sediments_base

xC=x3
yC=y4
TC=T_sediments_base

xD=x4
yD=Ly
TD=T_surface

xE=x1
yE=y2
TE=T_moho

xF=x2
yF=y3
TF=T_moho

xG=x3
yG=y3
TG=T_moho

xH=x4
yH=y2
TH=T_moho

xI=x1
yI=0
TI=T_lab

xJ=x2
yJ=y1
TJ=T_lab

xK=x3
yK=y1
TK=T_lab

xL=x4
yL=0
TL=T_lab

xM=x1
yM=y5
TM=T_uppercrust_base 

xN=x2
yN=y6
TN=T_uppercrust_base 

xO=x3
yO=y6
TO=T_uppercrust_base 

xP=x4
yP=y5
TP=T_uppercrust_base 

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_T=np.zeros(nn_T,dtype=np.float64)  # x coordinates
y_T=np.zeros(nn_T,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_T[counter]=i*Lx/float(nelx)
        y_T[counter]=j*Ly/float(nely)
        counter += 1
    #end for
#end for

print("grid points: %.3f s" % (clock.time()-start))

###################################################################################################
# connectivity
###################################################################################################
start=clock.time()

icon_T=np.zeros((m_T,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_T[0,counter]=i+j*(nelx+1)
        icon_T[1,counter]=i+1+j*(nelx+1)
        icon_T[2,counter]=i+1+(j+1)*(nelx + 1)
        icon_T[3,counter]=i+(j+1)*(nelx + 1)
        counter+=1
    #end for
#end for

print("connectivity: %.3f s" % (clock.time()-start))

###################################################################################################
# initial temperature
###################################################################################################
start=clock.time()

T=np.zeros(nn_T,dtype=np.float64)

for i in range(0,nn_T):

    #-----left-----
    if x_T[i] < x1:
       if y_T[i]>y5:
          T[i]=(TA-TM)/(yA-yM)*(y_T[i]-yM)+TM
       elif y_T[i]>y2:
          T[i]=(TM-TE)/(yM-yE)*(y_T[i]-yE)+TE
       else:
          T[i]=(TE-TI)/(yE-yI)*(y_T[i]-yI)+TI

    #-----taper-----
    elif x_T[i] < x2:
       if y_T[i]>(yB-yA)/(xB-xA)*(x_T[i]-xA)+yA: #line AB
          ytop=Ly
          Ttop=T_surface
          ybot=(yB-yA)/(xB-xA)*(x_T[i]-xA)+yA
          Tbot=(TB-TA)/(xB-xA)*(x_T[i]-xA)+TA
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(y_T[i]-ybot)+Tbot
       elif y_T[i]>(yN-yM)/(xN-xM)*(x_T[i]-xM)+yM: # line MN
          ytop=(yB-yA)/(xB-xA)*(x_T[i]-xA)+yA
          Ttop=(TB-TA)/(xB-xA)*(x_T[i]-xA)+TA
          ybot=(yN-yM)/(xN-xM)*(x_T[i]-xM)+yM
          Tbot=(TN-TM)/(xN-xM)*(x_T[i]-xM)+TM
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(y_T[i]-ybot)+Tbot
       elif y_T[i]>(yF-yE)/(xF-xE)*(x_T[i]-xE)+yE: # line EF
          ytop=(yN-yM)/(xN-xM)*(x_T[i]-xM)+yM
          Ttop=(TN-TM)/(xN-xM)*(x_T[i]-xM)+TM
          ybot=(yF-yE)/(xF-xE)*(x_T[i]-xE)+yE
          Tbot=(TF-TE)/(xF-xE)*(x_T[i]-xE)+TE
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(y_T[i]-ybot)+Tbot
       elif y_T[i]>(yJ-yI)/(xJ-xI)*(x_T[i]-xI)+yI: # line IJ
          ytop=(yF-yE)/(xF-xE)*(x_T[i]-xE)+yE
          Ttop=(TF-TE)/(xF-xE)*(x_T[i]-xE)+TE
          ybot=(yJ-yI)/(xJ-xI)*(x_T[i]-xI)+yI
          Tbot=(TJ-TI)/(xJ-xI)*(x_T[i]-xI)+TI
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(y_T[i]-ybot)+Tbot
       else:
          T[i]=T_lab

    #-----middle-----
    elif x_T[i] < x3:
       if y_T[i]>y4:
          T[i]=(TA-TB)/(yA-yB)*(y_T[i]-yB)+TB
       elif y_T[i]>y6:
          T[i]=(TB-TN)/(yB-yN)*(y_T[i]-yN)+TN
       elif y_T[i]>y3:
          T[i]=(TN-TF)/(yN-yF)*(y_T[i]-yF)+TF
       elif y_T[i]>y1:
          T[i]=(TF-TJ)/(yF-yJ)*(y_T[i]-yJ)+TJ
       else:
          T[i]=(TJ-TI)/(yJ-yI)*(y_T[i]-yI)+TI

    #-----taper-----
    elif x_T[i] < x4:
       if y_T[i]>(yD-yC)/(xD-xC)*(x_T[i]-xC)+yC:
          ytop=Ly
          Ttop=T_surface
          ybot=(yD-yC)/(xD-xC)*(x_T[i]-xC)+yC
          Tbot=(TD-TC)/(xD-xC)*(x_T[i]-xC)+TC
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(y_T[i]-ybot)+Tbot

       elif y_T[i]>(yP-yO)/(xP-xO)*(x_T[i]-xO)+yO: # line OP
          ytop=(yD-yC)/(xD-xC)*(x_T[i]-xC)+yC
          Ttop=(TD-TC)/(xD-xC)*(x_T[i]-xC)+TC
          ybot=(yP-yO)/(xP-xO)*(x_T[i]-xO)+yO
          Tbot=(TP-TO)/(xP-xO)*(x_T[i]-xO)+TO
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(y_T[i]-ybot)+Tbot

       elif y_T[i]>(yH-yG)/(xH-xG)*(x_T[i]-xG)+yG: # line GH
          ytop=(yP-yO)/(xP-xO)*(x_T[i]-xO)+yO
          Ttop=(TP-TO)/(xP-xO)*(x_T[i]-xO)+TO
          ybot=(yH-yG)/(xH-xG)*(x_T[i]-xG)+yG
          Tbot=(TH-TG)/(xH-xG)*(x_T[i]-xG)+TG
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(y_T[i]-ybot)+Tbot

       elif y_T[i]>(yK-yL)/(xK-xL)*(x_T[i]-xL)+yL: # line KL
          ytop=(yH-yG)/(xH-xG)*(x_T[i]-xG)+yG
          Ttop=(TH-TG)/(xH-xG)*(x_T[i]-xG)+TG
          ybot=(yK-yL)/(xK-xL)*(x_T[i]-xL)+yL
          Tbot=(TK-TL)/(xK-xL)*(x_T[i]-xL)+TL
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(y_T[i]-ybot)+Tbot
       else:
          T[i]=T_lab

    #-----right-----
    else:
       if y_T[i]>y5:
          T[i]=(TD-TP)/(yD-yP)*(y_T[i]-yP)+TP
       elif y_T[i]>y2:
          T[i]=(TP-TH)/(yP-yH)*(y_T[i]-yH)+TH
       else:
          T[i]=(TH-TL)/(yH-yL)*(y_T[i]-yL)+TL

#end for

plot_T_field(T,x_T,y_T,nnx,nny,'Initial temperature field','T_init.pdf',0)

if debug: np.savetxt('T_init.ascii',np.array([x_T,y_T,T]).T,header='#x,y,T')

print("initial temperature: %.3f s" % (clock.time()-start))

###############################################################################
# boundary conditions 
###############################################################################
start=clock.time()
 
bc_fix_T=np.zeros(Nfem_T,dtype=bool) # boundary condition, yes/no
bc_val_T=np.zeros(Nfem_T,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_T):
    if y_T[i]/Ly<eps:
       bc_fix_T[i] = True ; bc_val_T[i] = T_lab 
    #end if
    if y_T[i]/Ly>1-eps:
       bc_fix_T[i] = True ; bc_val_T[i] = T_surface
    #end if
#end for

print("boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# material layout
###############################################################################
start=clock.time()

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
mat=np.zeros(nel,dtype=np.int16)  

for iel in range(0,nel):

    xc[iel]=x_T[icon_T[0,iel]]+hx/2
    yc[iel]=y_T[icon_T[0,iel]]+hy/2

    if yc[iel]>y5:
       mat[iel]=2
    elif yc[iel]>y2:
       mat[iel]=3
    else:
       mat[iel]=4

    #paint sediments (ABCD)
    if yc[iel]>(yB-yA)/(xB-xA)*(xc[iel]-xA)+yA and \
       yc[iel]>(yD-yC)/(xD-xC)*(xc[iel]-xC)+yC and \
       yc[iel]>y4:
       mat[iel]=1 

    #paint upper crust thinning (MNOP)
    if yc[iel]<(yN-yM)/(xN-xM)*(xc[iel]-xM)+yM and \
       yc[iel]<(yP-yO)/(xP-xO)*(xc[iel]-xO)+yO and \
       yc[iel]<y6 and xc[iel]>x1 and xc[iel]<x4:
       mat[iel]=3

    #paint lower crust thinning (EFGH)
    if yc[iel]<(yF-yE)/(xF-xE)*(xc[iel]-xE)+yE and \
       yc[iel]<(yH-yG)/(xH-xG)*(xc[iel]-xG)+yG and \
       yc[iel]<y3 and xc[iel]>x1 and xc[iel]<x4:
       mat[iel]=4

    #paint asthenosphere (IJKL)
    if yc[iel]<(yJ-yI)/(xJ-xI)*(xc[iel]-xI)+yI and \
       yc[iel]<(yK-yL)/(xK-xL)*(xc[iel]-xL)+yL and \
       yc[iel]<y1:
       mat[iel]=5

if debug: np.savetxt('mat.ascii',np.array([xc,yc,mat]).T,header='#x,y,mat')

plot_T_field(mat,xc,yc,nelx,nely,'material layout','mat.pdf',1)

print("material layout: %.3f s" % (clock.time()-start))
    
###############################################################################
###############################################################################
# BEGINNING OF TIMESTEPPING
###############################################################################
###############################################################################

for istep in range(0,nstep):

    print('-----------------------------')
    print('istep=',istep,'/',nstep)

    ###########################################################################
    # build FE matrix for Temperature 
    ###########################################################################

    start=clock.time()

    A_fem=lil_matrix((Nfem_T,Nfem_T),dtype=np.float64) # FE matrix 
    b_fem=np.zeros(Nfem_T,dtype=np.float64)         # FE b_fem 
    B_mat=np.zeros((2,m_T),dtype=np.float64)     # gradient matrix B 
    N_mat=np.zeros((m_T,1),dtype=np.float64)         # shape functions
    Tvect=np.zeros(m_T,dtype=np.float64)    
    jcb=np.zeros((2,2),dtype=np.float64)

    for iel in range (0,nel):

        A_el=np.zeros((m_T,m_T),dtype=np.float64)  # elemental FE matrix
        b_el=np.zeros(m_T,dtype=np.float64)       # elemental rhs
        Kd_el=np.zeros((m_T,m_T),dtype=np.float64) # elemental diffusion matrix 
        M_el=np.zeros((m_T,m_T),dtype=np.float64)  # elemental mass matrix 

        Tvect[:]=T[icon_T[:,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                N_T=basis_functions_T(rq,sq)
                dNdr_T=basis_functions_T_dr(rq,sq)
                dNds_T=basis_functions_T_ds(rq,sq)
                jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
                jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
                jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
                jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
                dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T

                N_mat[0:m_T,0]=N_T[:]
                B_mat[0,:]=dNdx_T[:]
                B_mat[1,:]=dNdy_T[:]

                # compute values at quadrature point
                Tq=np.dot(N_T,T[icon_T[:,iel]])
                rhoq=rho[mat[iel]-1]
                hcondq=hcond[mat[iel]-1]
                hcapaq=hcapa[mat[iel]-1]
                hprodq=hprod[mat[iel]-1]

                # compute mass matrix
                M_el+=N_mat.dot(N_mat.T)*rhoq*hcapaq*JxWq

                # compute diffusion matrix
                Kd_el+=B_mat.T.dot(B_mat)*hcondq*JxWq

                b_el+=N_mat[:,0]*rhoq*hprodq*JxWq

            # end for jq
        # end for iq

        if compute_ss:
           A_el=Kd_el
        else:
           A_el=M_el+Kd_el*dt
           b_el=M_el.dot(Tvect)

        # apply boundary conditions
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_T):
                   m2=icon_T[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            #end for
        #end for

        # assemble matrix A_fem and right hand side b_fem
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            for k2 in range(0,m_T):
                m2=icon_T[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        #end for

    # end for iel

    print("build FEM matrix T: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute (nodal) temperature gradient
    ###########################################################################
    start=clock.time()
    
    count=np.zeros(nn_T,dtype=np.int32)  
    qx_n=np.zeros(nn_T,dtype=np.float64)  
    qy_n=np.zeros(nn_T,dtype=np.float64)  
    dTdx_n=np.zeros(nn_T,dtype=np.float64)  
    dTdy_n=np.zeros(nn_T,dtype=np.float64)  
    r_T=[-1,0,+1,-1,0,+1,-1,0,+1]
    s_T=[-1,-1,-1,0,0,0,+1,+1,+1]

    for iel in range(0,nel):
        hcond_el=hcond[mat[iel]-1]
        for i in range(0,m_T):
            rq=r_T[i]
            sq=s_T[i]

            dNdr_T=basis_functions_T_dr(rq,sq)
            dNds_T=basis_functions_T_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
            jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
            jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
            jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
            dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T

            dTdxq=np.dot(dNdx_T,T[icon_T[:,iel]])
            dTdyq=np.dot(dNdy_T,T[icon_T[:,iel]])
             
            q_x=-hcond_el*dTdxq
            q_y=-hcond_el*dTdyq

            inode=icon_T[i,iel]
            qx_n[inode]+=q_x
            qy_n[inode]+=q_y
            dTdx_n[inode]+=dTdxq
            dTdy_n[inode]+=dTdyq
            count[inode]+=1
        #end for
    #end for
    
    qx_n/=count
    qy_n/=count
    dTdx_n/=count
    dTdy_n/=count

    print("     -> qx_n (m,M) %.6e %.6e " %(np.min(qx_n),np.max(qx_n)))
    print("     -> qy_n (m,M) %.6e %.6e " %(np.min(qy_n),np.max(qy_n)))
    print("     -> dTdy_n (m,M) %.6e %.6e " %(np.min(dTdy_n),np.max(dTdy_n)))

    print("compute heat flux: %.3f s" % (clock.time()-start))

    ###########################################################################
    # profiles
    ###########################################################################
    start=clock.time()

    filename = 'profile_middle_{:04d}.ascii'.format(istep) 
    profile=open(filename,"w")
    for i in range(0,nn_T):
        if abs(x_T[i]-Lx/2)/Lx<eps:
           profile.write("%e %e %e %e\n" %(y_T[i],T[i],dTdy_n[i],qy_n[i]))
    profile.close()

    filename = 'profile_left_{:04d}.ascii'.format(istep) 
    profile=open(filename,"w")
    for i in range(0,nn_T):
        if abs(x_T[i])/Lx<eps:
           profile.write("%e %e %e %e\n" %(y_T[i],T[i],dTdy_n[i],qy_n[i]))
    profile.close()

    print("export profiles: %.3f s" % (clock.time()-start))

    ###########################################################################
    # export to pdf via matplotlib
    ###########################################################################
    start=clock.time()

    if compute_ss:
       filename = 'T_final' 
       if debug: np.savetxt(filename+'.ascii',np.array([x_T,y_T,T]).T,header='#x,y,T')
       plot_T_field(T, x_T, y_T, nnx, nny, 'temperature field', filename+'.pdf',0)
    else:
       filename = 'T_{:04d}'.format(istep) 
       if debug: np.savetxt(filename+'.ascii',np.array([x_T,y_T,T]).T,header='#x,y,T')
       plot_T_field(T, x_T, y_T, nnx, nny, 'temperature field', filename+'.pdf',0)

    print("export via matplotlib: %.3f s" % (clock.time()-start))

    ###########################################################################
    # export to vtu
    ###########################################################################

    if True:
       if compute_ss:
          filename = 'solution.vtu'
       else:
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
       #--
       vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d\n" % mat[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='hcond' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % hcond[mat[iel]-1])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='hcapa' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % hcapa[mat[iel]-1])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='hprod' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % hprod[mat[iel]-1])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e \n" % (T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdx (C/km)' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e \n" % (dTdx_n[i]*1000))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdy (C/km)' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e \n" % (dTdy_n[i]*1000))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%e %e %e \n" % (qx_n[i],qy_n[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon_T[0,iel],icon_T[1,iel],icon_T[2,iel],icon_T[3,iel]))
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

    print("export solution to vtu: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
# END OF TIMESTEPPING
###############################################################################
###############################################################################

print("*******************************")
print("********** the end ************")
print("*******************************")



