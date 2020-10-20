import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt
import time

###################################################################################################
# Q1 shape functions and derivatives
###################################################################################################

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

###################################################################################################

def plot_T_field(Tnew, xcoords, ycoords, nnx, nny, Title, savename, option):
    Tplot = Tnew.reshape(nny, nnx)
    xplot = xcoords.reshape(nny, nnx)
    yplot = ycoords.reshape(nny, nnx)
    plt.subplot(Lx/Ly,1,2)
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

###################################################################################################
# geometrical parameters
###################################################################################################

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

###################################################################################################
# material properties
###################################################################################################

rho_sediments=2100
hcapa_sediments=790
hcond_sediments=2

rho_uppercrust=2900
hcapa_uppercrust=1100
hcond_uppercrust=2.6

rho_lowercrust=2900
hcapa_lowercrust=1100
hcond_lowercrust=2.6

rho_lithosphere=3400
hcapa_lithosphere=1260
hcond_lithosphere=3

rho_asthenosphere=3400
hcapa_asthenosphere=1260
hcond_asthenosphere=3

###################################################################################################
# temperature parameters
###################################################################################################

T_surface = 0
T_sediments_base = 200
T_uppercrust_base = 310
T_moho = 550
T_lab = 1330

###################################################################################################

nelx=800
nely=240

nstep=100

dt=2e3*3.154e7

compute_ss=False

###################################################################################################

nel=nelx*nely
nnx=nelx+1
nny=nely+1
NT=nnx*nny
hx=Lx/nelx
hy=Ly/nely
NfemT=NT
ndofT=1
mT=4
eps=1.e-10
sqrt3=np.sqrt(3.)

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
rho=[rho_sediments,\
     rho_uppercrust,\
     rho_lowercrust,\
     rho_lithosphere,\
     rho_asthenosphere]

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

###################################################################################################
# geometry parameters 
###################################################################################################

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



###################################################################################################
# grid point setup
###################################################################################################
start = time.time()

xT = np.empty(NT,dtype=np.float64)  # x coordinates
yT = np.empty(NT,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xT[counter]=i*Lx/float(nelx)
        yT[counter]=j*Ly/float(nely)
        counter += 1
    #end for
#end for

print("grid points: %.3f s" % (time.time() - start))

###################################################################################################
# connectivity
###################################################################################################
start = time.time()

icon=np.zeros((mT,nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter]=i+j*(nelx+1)
        icon[1,counter]=i+1+j*(nelx+1)
        icon[2,counter]=i+1+(j+1)*(nelx + 1)
        icon[3,counter]=i+(j+1)*(nelx + 1)
        counter += 1
    #end for
#end for

print("connectivity: %.3f s" % (time.time() - start))

###################################################################################################
# initial temperature
###################################################################################################
start = time.time()

T = np.empty(NT,dtype=np.float64)  # y coordinates

for i in range(0,NT):

    #-----left-----
    if xT[i] < x1:
       if yT[i]>y5:
          T[i]=(TA-TM)/(yA-yM)*(yT[i]-yM)+TM
       elif yT[i]>y2:
          T[i]=(TM-TE)/(yM-yE)*(yT[i]-yE)+TE
       else:
          T[i]=(TE-TI)/(yE-yI)*(yT[i]-yI)+TI

    #-----taper-----
    elif xT[i] < x2:
       if yT[i]>(yB-yA)/(xB-xA)*(xT[i]-xA)+yA: #line AB
          ytop=Ly
          Ttop=T_surface
          ybot=(yB-yA)/(xB-xA)*(xT[i]-xA)+yA
          Tbot=(TB-TA)/(xB-xA)*(xT[i]-xA)+TA
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(yT[i]-ybot)+Tbot
       elif yT[i]>(yN-yM)/(xN-xM)*(xT[i]-xM)+yM: # line MN
          ytop=(yB-yA)/(xB-xA)*(xT[i]-xA)+yA
          Ttop=(TB-TA)/(xB-xA)*(xT[i]-xA)+TA
          ybot=(yN-yM)/(xN-xM)*(xT[i]-xM)+yM
          Tbot=(TN-TM)/(xN-xM)*(xT[i]-xM)+TM
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(yT[i]-ybot)+Tbot
       elif yT[i]>(yF-yE)/(xF-xE)*(xT[i]-xE)+yE: # line EF
          ytop=(yN-yM)/(xN-xM)*(xT[i]-xM)+yM
          Ttop=(TN-TM)/(xN-xM)*(xT[i]-xM)+TM
          ybot=(yF-yE)/(xF-xE)*(xT[i]-xE)+yE
          Tbot=(TF-TE)/(xF-xE)*(xT[i]-xE)+TE
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(yT[i]-ybot)+Tbot
       elif yT[i]>(yJ-yI)/(xJ-xI)*(xT[i]-xI)+yI: # line IJ
          ytop=(yF-yE)/(xF-xE)*(xT[i]-xE)+yE
          Ttop=(TF-TE)/(xF-xE)*(xT[i]-xE)+TE
          ybot=(yJ-yI)/(xJ-xI)*(xT[i]-xI)+yI
          Tbot=(TJ-TI)/(xJ-xI)*(xT[i]-xI)+TI
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(yT[i]-ybot)+Tbot
       else:
          T[i]=T_lab

    #-----middle-----
    elif xT[i] < x3:
       if yT[i]>y4:
          T[i]=(TA-TB)/(yA-yB)*(yT[i]-yB)+TB
       elif yT[i]>y6:
          T[i]=(TB-TN)/(yB-yN)*(yT[i]-yN)+TN
       elif yT[i]>y3:
          T[i]=(TN-TF)/(yN-yF)*(yT[i]-yF)+TF
       elif yT[i]>y1:
          T[i]=(TF-TJ)/(yF-yJ)*(yT[i]-yJ)+TJ
       else:
          T[i]=(TJ-TI)/(yJ-yI)*(yT[i]-yI)+TI

    #-----taper-----
    elif xT[i] < x4:
       if yT[i]>(yD-yC)/(xD-xC)*(xT[i]-xC)+yC:
          ytop=Ly
          Ttop=T_surface
          ybot=(yD-yC)/(xD-xC)*(xT[i]-xC)+yC
          Tbot=(TD-TC)/(xD-xC)*(xT[i]-xC)+TC
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(yT[i]-ybot)+Tbot

       elif yT[i]>(yP-yO)/(xP-xO)*(xT[i]-xO)+yO: # line OP
          ytop=(yD-yC)/(xD-xC)*(xT[i]-xC)+yC
          Ttop=(TD-TC)/(xD-xC)*(xT[i]-xC)+TC
          ybot=(yP-yO)/(xP-xO)*(xT[i]-xO)+yO
          Tbot=(TP-TO)/(xP-xO)*(xT[i]-xO)+TO
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(yT[i]-ybot)+Tbot

       elif yT[i]>(yH-yG)/(xH-xG)*(xT[i]-xG)+yG: # line GH
          ytop=(yP-yO)/(xP-xO)*(xT[i]-xO)+yO
          Ttop=(TP-TO)/(xP-xO)*(xT[i]-xO)+TO
          ybot=(yH-yG)/(xH-xG)*(xT[i]-xG)+yG
          Tbot=(TH-TG)/(xH-xG)*(xT[i]-xG)+TG
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(yT[i]-ybot)+Tbot

       elif yT[i]>(yK-yL)/(xK-xL)*(xT[i]-xL)+yL: # line KL
          ytop=(yH-yG)/(xH-xG)*(xT[i]-xG)+yG
          Ttop=(TH-TG)/(xH-xG)*(xT[i]-xG)+TG
          ybot=(yK-yL)/(xK-xL)*(xT[i]-xL)+yL
          Tbot=(TK-TL)/(xK-xL)*(xT[i]-xL)+TL
          T[i]=(Ttop-Tbot)/(ytop-ybot)*(yT[i]-ybot)+Tbot
       else:
          T[i]=T_lab

    #-----right-----
    else:
       if yT[i]>y5:
          T[i]=(TD-TP)/(yD-yP)*(yT[i]-yP)+TP
       elif yT[i]>y2:
          T[i]=(TP-TH)/(yP-yH)*(yT[i]-yH)+TH
       else:
          T[i]=(TH-TL)/(yH-yL)*(yT[i]-yL)+TL

#end for

plot_T_field(T, xT, yT, nnx, nny, 'Initial temperature field', 'T_init.pdf',0)

np.savetxt('T_init.ascii',np.array([xT,yT,T]).T,header='#x,y,T')

print("initial temperature: %.3f s" % (time.time() - start))

###################################################################################################
# boundary conditions 
###################################################################################################
start = time.time()
 
bc_fixT=np.zeros(NfemT,dtype=np.bool) # boundary condition, yes/no
bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

for i in range(0,NT):
    if yT[i]/Ly<eps:
       bc_fixT[i] = True ; bc_valT[i] = T_lab 
    #end if
    if yT[i]/Ly>1-eps:
       bc_fixT[i] = True ; bc_valT[i] = T_surface
    #end if
#end for

print("boundary conditions: %.3f s" % (time.time() - start))

###################################################################################################
# material layout
###################################################################################################
start = time.time()

xc  = np.zeros(nel,dtype=np.float64)  
yc  = np.zeros(nel,dtype=np.float64)  
mat  = np.zeros(nel,dtype=np.int16)  

for iel in range(0,nel):

    xc[iel]=xT[icon[0,iel]]+hx/2
    yc[iel]=yT[icon[0,iel]]+hy/2

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

#np.savetxt('mat.ascii',np.array([xc,yc,mat]).T,header='#x,y,mat')

plot_T_field(mat, xc, yc, nelx, nely, 'material layout', 'mat.pdf', 1)

print("material layout: %.3f s" % (time.time() - start))
    
###################################################################################################
# BEGINNING OF TIMESTEPPING
###################################################################################################


for istep in range(0,nstep):

    print('-----------------------------')
    print('istep=',istep,'/',nstep)

    ######################################################################
    # build FE matrix for Temperature 
    ######################################################################

    start = time.time()

    N     = np.zeros(mT,dtype=np.float64)             # shape functions
    dNdx  = np.zeros(mT,dtype=np.float64)             # shape functions derivatives
    dNdy  = np.zeros(mT,dtype=np.float64)             # shape functions derivatives
    dNdr  = np.zeros(mT,dtype=np.float64)             # shape functions derivatives
    dNds  = np.zeros(mT,dtype=np.float64)             # shape functions derivatives
    A_mat = lil_matrix((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*mT),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((mT,1),dtype=np.float64)         # shape functions
    Tvect = np.zeros(mT,dtype=np.float64)    

    iiq=0
    for iel in range (0,nel):

        b_el=np.zeros(mT*ndofT,dtype=np.float64) # elemental rhs
        A_el=np.zeros((mT*ndofT,mT*ndofT),dtype=np.float64) # elemental FE matrix
        Kd_el=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
        M_el=np.zeros((mT,mT),dtype=np.float64) # elemental mass matrix 

        for k in range(0,mT):
            Tvect[k]=T[icon[k,iel]]
        #end for

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                # calculate shape functions
                N_mat[0:mT,0]=NNV(rq,sq)
                dNdr[0:mT]=dNNVdr(rq,sq)
                dNds[0:mT]=dNNVds(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mT):
                    jcb[0,0]+=dNdr[k]*xT[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*yT[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*xT[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*yT[icon[k,iel]]
                #end for
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                Tq=0.
                for k in range(0,mT):
                    Tq+=N_mat[k,0]*T[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                #end for

                rhoq=rho[mat[iel]-1]
                hcondq=hcond[mat[iel]-1]
                hcapaq=hcapa[mat[iel]-1]

                # compute mass matrix
                M_el+=N_mat.dot(N_mat.T)*rhoq*hcapaq*weightq*jcob

                # compute diffusion matrix
                Kd_el+=B_mat.T.dot(B_mat)*hcondq*weightq*jcob

                iiq+=1

            # end for jq
        # end for iq

        if compute_ss:
           A_el=Kd_el
        else:
           A_el=M_el+Kd_el*dt
           b_el=M_el.dot(Tvect)

        # apply boundary conditions
        for k1 in range(0,mT):
            m1=icon[k1,iel]
            if bc_fixT[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,mT):
                   m2=icon[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_valT[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            #end for
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mT):
            m1=icon[k1,iel]
            for k2 in range(0,mT):
                m2=icon[k2,iel]
                A_mat[m1,m2]+=A_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    # end for iel

    print("build FEM matrix T: %.3f s" % (time.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = time.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T: %.3f s" % (time.time() - start))


    #################################################################
    # compute (nodal) temperature gradient
    #################################################################
    start = time.time()
    
    count = np.zeros(NT,dtype=np.int32)  
    qx_n = np.zeros(NT,dtype=np.float64)  
    qy_n = np.zeros(NT,dtype=np.float64)  
    dTdy_n=np.zeros(NT,dtype=np.float64)  
    rVnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
    sVnodes=[-1,-1,-1,0,0,0,+1,+1,+1]

    for iel in range(0,nel):
        hcond_el=hcond[mat[iel]-1]
        for i in range(0,mT):
            rq=rVnodes[i]
            sq=sVnodes[i]

            # calculate shape functions
            N_mat[0:mT,0]=NNV(rq,sq)
            dNdr[0:mT]=dNNVdr(rq,sq)
            dNds[0:mT]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mT):
                    jcb[0,0]+=dNdr[k]*xT[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*yT[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*xT[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*yT[icon[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)

            for k in range(0,mT):
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
            #end for
            q_x=0.
            q_y=0.
            dTdy=0.
            for k in range(0,mT):
                q_x-=hcond_el*dNdx[k]*T[icon[k,iel]]
                q_y-=hcond_el*dNdy[k]*T[icon[k,iel]]
                dTdy-=dNdy[k]*T[icon[k,iel]]
            #end for
            inode=icon[i,iel]
            qx_n[inode]+=q_x
            qy_n[inode]+=q_y
            dTdy_n[inode]+=dTdy
            count[inode]+=1
        #end for
    #end for
    
    qx_n/=count
    qy_n/=count
    dTdy_n/=count

    print("     -> qx_n (m,M) %.6e %.6e " %(np.min(qx_n),np.max(qx_n)))
    print("     -> qy_n (m,M) %.6e %.6e " %(np.min(qy_n),np.max(qy_n)))
    print("     -> dTdy_n (m,M) %.6e %.6e " %(np.min(dTdy_n),np.max(dTdy_n)))

    print("compute heat flux: %.3f s" % (time.time() - start))

    #################################################################
    # profiles
    #################################################################
    start = time.time()

    filename = 'profile_middle_{:04d}.ascii'.format(istep) 
    profile=open(filename,"w")
    for i in range(0,NT):
        if abs(xT[i]-Lx/2)/Lx<eps:
           profile.write("%10e %10e %10e %10e\n" %(yT[i],T[i],dTdy_n[i],qy_n[i]))
    profile.close()

    filename = 'profile_left_{:04d}.ascii'.format(istep) 
    profile=open(filename,"w")
    for i in range(0,NT):
        if abs(xT[i])/Lx<eps:
           profile.write("%10e %10e %10e %10e\n" %(yT[i],T[i],dTdy_n[i],qy_n[i]))
    profile.close()

    print("export profiles: %.3f s" % (time.time() - start))

    #################################################################
    #################################################################
    start = time.time()

    if compute_ss:
       filename = 'T_final' 
       np.savetxt(filename+'.ascii',np.array([xT,yT,T]).T,header='#x,y,T')
       plot_T_field(T, xT, yT, nnx, nny, 'temperature field', filename+'.pdf',0)
    else:
       filename = 'T_{:04d}'.format(istep) 
       np.savetxt(filename+'.ascii',np.array([xT,yT,T]).T,header='#x,y,T')
       plot_T_field(T, xT, yT, nnx, nny, 'temperature field', filename+'.pdf',0)

    print("export via matplotlib: %.3f s" % (time.time() - start))

    #################################################################
    # export to vtu
    #################################################################

    if True:
       if compute_ss:
          filename = 'solution.vtu'
       else:
          filename = 'solution_{:04d}.vtu'.format(istep) 

       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NT,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NT):
          vtufile.write("%10e %10e %10e \n" %(xT[i],yT[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")

       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d\n" % mat[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")

       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e \n" % (T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdy (C/km)' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e \n" % (dTdy_n[i]*1000))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e %10e %10e \n" % (qx_n[i],qy_n[i],0.))
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


###################################################################################################
# END OF TIMESTEPPING
###################################################################################################

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")



