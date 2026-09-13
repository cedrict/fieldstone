import scipy.sparse as sps
from scipy.sparse.linalg import *
import time as clock
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from points import *

Myear=1e6*365.25*3600*24
km=1e3

###############################################################################
# 40Mo on left, 70Mo on right

def temp(x,y):

    Ts=0+273
    Tm=1300+273
    kappa=1e-6

    if x<1500e3:
       ta=40*Myear
       yl=82*km
    else:
       ta=80*Myear
       yl=110*km

    n=1
    t1=1./n*np.exp(-kappa*n**2*np.pi**2*ta/yl**2)*np.sin(n*np.pi*(Ly-y)/yl)
    n=2
    t2=1./n*np.exp(-kappa*n**2*np.pi**2*ta/yl**2)*np.sin(n*np.pi*(Ly-y)/yl)

    return Ts+(Tm-Ts)*((Ly-y)/yl+2/np.pi*(t1+t2))

###############################################################################

def density(imat):
    if imat==1:
       rho0  = 3200.
    elif imat==2:
       rho0   = 3250.
    elif imat==3:
       rho0  = 3250.
    elif imat==4:
       rho0  = 3250.
    elif imat==5:
       rho0  = 3250.
    elif imat==6:
       rho0  = 3250.
    elif imat==7:
       rho0  = 3250.
    else:
       rho0  = 3250.
    return rho0

def viscosity(imat):
    if imat==1:
       mu0   = 1.e20
    elif imat==2:
       mu0    = 1.e23
    elif imat==3:
       mu0   = 1.e20
    elif imat==4:
       mu0   = 1.e23
    elif imat==5:
       mu0   = 1.e23
    elif imat==6:
       mu0   = 1.e23
    elif imat==7:
       mu0   = 1.e23
    else:
       mu0   = 1.e20
    return mu0

###############################################################################

def basis_functions_V(r,s):
    N0= (1-r-s)*(1-2*r-2.*s+ 3.*r*s)
    N1= r*(2*r-1+3*s-3.*r*s-3.*s**2 )
    N2= s*(2*s-1+3*r-3.*r**2-3.*r*s )
    N3= 4*(1-r-s)*r*(1-3.*s) 
    N4= 4*r*s*(-2.+3*r+3.*s)
    N5= 4*(1-r-s)*s*(1-3.*r) 
    N6= 27*(1-r-s)*r*s
    return np.array([N0,N1,N2,N3,N4,N5,N6],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= -3+4*r+7*s-6*r*s-3*s**2
    dNdr1= 4*r-1+3*s-6*r*s-3*s**2
    dNdr2= 3*s-6*r*s-3*s**2
    dNdr3= -8*r+24*r*s+4-16*s+12*s**2
    dNdr4= -8*s+24*r*s+12*s**2
    dNdr5= -16*s+24*r*s+12*s**2
    dNdr6= -54*r*s+27*s-27*s**2
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= -3+7*r+4*s-6*r*s-3*r**2
    dNds1= r*(3-3*r-6*s)
    dNds2= 4*s-1+3*r-3*r**2-6*r*s
    dNds3= -16*r+24*r*s+12*r**2
    dNds4= -8*r+12*r**2+24*r*s
    dNds5= 4-16*r-8*s+24*r*s+12*r**2
    dNds6= -54*r*s+27*r-27*r**2
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6],dtype=np.float64)

def basis_functions_P(r,s):
    N0=1-r-s
    N1=r
    N2=s
    return np.array([N0,N1,N2],dtype=np.float64)

###############################################################################

print("*******************************")
print("********** stone 062 **********")
print("*******************************")

m_V=7   # number of velocity nodes making up an element
m_P=3   # number of pressure nodes making up an element
ndof_V=2 # number of velocity degrees of freedom per node

#read nb of elements and nb of nodes from temp file 
counter=0
file=open("temp", "r")
for line in file:
    fields = line.strip().split()
    #print(fields[0], fields[1], fields[2])
    if counter==0:
       nel=int(fields[0])
    if counter==1:
       NV0=int(fields[0])
    counter+=1

nn_V=NV0+nel

Nfem_V=nn_V*ndof_V  # number of velocity dofs
Nfem_P=nel*m_P     # number of pressure dofs
Nfem=Nfem_V+Nfem_P # total number of dofs

print ('nel=',nel)
print ('nn_V=',nn_V)
print ('Nfem_V=',Nfem_V)
print ('Nfem_P=',Nfem_P)
print ('Nfem=',Nfem)

eta_ref=1e22

nstep=1

gx=0
gy=-9.81

dt=0

cm=0.01
year=365.*3600.*24.

debug=False

###############################################################################
# 6 point integration coeffs and weights 
###############################################################################

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322/2.
nb6=0.223381589678011/2.

qcoords_r=[nb1,nb2,nb2,nb4,nb3,nb4]
qcoords_s=[nb2,nb1,nb2,nb3,nb4,nb4]
qweights =[nb5,nb5,nb5,nb6,nb6,nb6]


###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)     # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)     # y coordinates

x_V[0:NV0],y_V[0:NV0]=np.loadtxt('mesh.1.node',unpack=True,usecols=[1,2],skiprows=1)

print("     -> x_V (min/max): %.4f %.4f" %(np.min(x_V[0:NV0]),np.max(x_V[0:NV0])))
print("     -> y_V (min/max): %.4f %.4f" %(np.min(y_V[0:NV0]),np.max(y_V[0:NV0])))

if debug: np.savetxt('gridV0.ascii',np.array([x_V,y_V]).T,header='# x_V,y_V')

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
#
#  P_2^+           P_-1
#
#  02              02
#  ||\\            ||\\
#  || \\           || \\
#  ||  \\          ||  \\
#  05   04         ||   \\
#  || 06 \\        ||    \\
#  ||     \\       ||     \\
#  00==03==01      00======01
#
# note that the ordering of nodes returned by triangle is different
# than mine: https://www.cs.cmu.edu/~quake/triangle.highorder.html.
# note also that triangle returns nodes 0-5, but not 6.
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

icon_V[0,:],icon_V[1,:],icon_V[2,:],icon_V[4,:],icon_V[5,:],icon_V[3,:]=\
np.loadtxt('mesh.1.ele',unpack=True, usecols=[1,2,3,4,5,6],skiprows=1)

icon_V[0,:]-=1
icon_V[1,:]-=1
icon_V[2,:]-=1
icon_V[3,:]-=1
icon_V[4,:]-=1
icon_V[5,:]-=1

for iel in range (0,nel):
    icon_V[6,iel]=NV0+iel

print("setup: connectivity V: %.3f s" % (clock.time()-start))

for iel in range(0,nel): #bubble nodes
    x_V[NV0+iel]=(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])/3.
    y_V[NV0+iel]=(y_V[icon_V[0,iel]]+y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])/3.

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x_V,y_V')

print("read V connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid (nodes and icon)
###############################################################################
start=clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)
x_P=np.zeros(Nfem_P,dtype=np.float64)     # x coordinates
y_P=np.zeros(Nfem_P,dtype=np.float64)     # y coordinates

counter=0
for iel in range(0,nel):
    x_P[counter]=x_V[icon_V[0,iel]]
    y_P[counter]=y_V[icon_V[0,iel]]
    icon_P[0,iel]=counter
    counter+=1
    x_P[counter]=x_V[icon_V[1,iel]]
    y_P[counter]=y_V[icon_V[1,iel]]
    icon_P[1,iel]=counter
    counter+=1
    x_P[counter]=x_V[icon_V[2,iel]]
    y_P[counter]=y_V[icon_V[2,iel]]
    icon_P[2,iel]=counter
    counter+=1

if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",icon_P[0,iel],"at pos.",xP[icon_P[0][iel]], yP[icon_P[0][iel]])
#    print ("node 1",icon_P[1,iel],"at pos.",xP[icon_P[1][iel]], yP[icon_P[1][iel]])
#    print ("node 2",icon_P[2,iel],"at pos.",xP[icon_P[2][iel]], yP[icon_P[2][iel]])

print("make P connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# compute coordinates of element centers
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    x_e[iel]=(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])/3.
    y_e[iel]=(y_V[icon_V[0,iel]]+y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])/3.

print("compute elt center coords: %.3f s" % (clock.time()-start))

###############################################################################
# material layout
###############################################################################
start=clock.time()

mat=np.zeros(nel,dtype=np.int32)  
rho=np.zeros(nel,dtype=np.float64) 
eta=np.zeros(nel,dtype=np.float64) 

for iel in range(nel):
    mat[iel]=1
    if y_e[iel]>=yC and y_e[iel]>=((yA-yC)/(xA-xC)*(x_e[iel]-xC)+yC):
       mat[iel]= 6 # 40Ma

    if y_e[iel]>=yH and y_e[iel]>=((yA-yC)/(xA-xC)*(x_e[iel]-xC)+yC): 
       mat[iel]= 4 # SHB left

    if y_e[iel]>=yF and y_e[iel]>=((yA-yC)/(xA-xC)*(x_e[iel]-xC)+yC):
       mat[iel]= 2 # BOC left

    if y_e[iel]>=yE and y_e[iel]<=((yB-yD)/(xB-xD)*(x_e[iel]-xB)+yB) and\
                        y_e[iel]>=((yD-yE)/(xD-xE)*(x_e[iel]-xE)+yE): 
       mat[iel]= 7 # 70Ma

    if y_e[iel]>=yI and y_e[iel]<=((yB-yD)/(xB-xD)*(x_e[iel]-xB)+yB): 
       mat[iel]= 5 # SHB right
 
    if y_e[iel]>=yG and y_e[iel]<=((yB-yD)/(xB-xD)*(x_e[iel]-xB)+yB): 
       mat[iel]= 3 # BOC right
 
    if y_e[iel]>=yC and y_e[iel]<=((yA-yC)/(xA-xC)*(x_e[iel]-xC)+yC)  and\
                        y_e[iel]>=((yB-yD)/(xB-xD)*(x_e[iel]-xB)+yB):  
       mat[iel]= 8 # seed 

#end for

print("assign material to elements: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

v_in=-5.*cm/year

y_in=560e3
y_out=540e3

y_0=Ly
y_b=0.

v_out=-v_in * ( Ly - 0.5*(y_in+y_out)  ) / ( 0.5*(y_in+y_out))

for i in range(0,nn_V):

    #Left boundary  
    if x_V[i]/Lx<0.0000001:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] =0 

    #right boundary  
    if x_V[i]/Lx>0.9999999:
       if y_V[i]<y_out:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = v_out
       elif y_V[i]<y_in:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = (v_in-v_out)/(y_in-y_out)*(y_V[i]-y_out)+v_out
       else:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = v_in

    #bottom boundary  
    if y_V[i]/Ly<0.0000001:
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0     # y component
    #bottom boundary  
    if y_V[i]/Ly>0.9999999:
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0     # y component

print("define boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
###############################################################################
start=clock.time()

area=np.zeros(nel,dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
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

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))
print("     -> total area (anal) %.6f " %(Lx*Ly))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
# assign temperature
###############################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):

    if x_V[i]<Lx/2:
       if Ly-y_V[i]<82e3:
          T[i]=temp(x_V[i],y_V[i])
       else:
          T[i]=1440+273-0.25*y_V[i]/km
    else:
       if Ly-y_V[i]<110e3:
          T[i]=temp(x_V[i],y_V[i])
       else:
          T[i]=1440+273-0.25*y_V[i]/km

if debug: np.savetxt('temperature.ascii',np.array([x_V,y_V,T]).T,header='# x_V,y_V')

print("assign temperature to nodes: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
# TIME STEPPING
###############################################################################
###############################################################################

u=np.zeros(nn_V,dtype=np.float64)
v=np.zeros(nn_V,dtype=np.float64)

for istep in range(0,nstep):

    print("--------------------------------------------")
    print("istep= ", istep)
    print("--------------------------------------------")

    #################################################################
    # mesh evolution
    # only move nodes that are not on the prescribed boundaries 
    # element edges are maintained straight
    #################################################################
    start=clock.time()

    for i in range(0,nn_V):
        if x_V[i]/Lx>0.0000001 and x_V[i]/Lx<0.9999999 and y_V[i]/Ly>0.0000001:
           x_V[i]+=u[i]*dt
           y_V[i]+=v[i]*dt
        else:
           if not bc_fix[2*i]:   x_V[i]+=u[i]*dt
           if not bc_fix[2*i+1]: y_V[i]+=v[i]*dt

    for iel in range(0,nel):
        # node 3 is between nodes 0 and 1
        x_V[icon_V[3,iel]]=0.5*(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]]) 
        y_V[icon_V[3,iel]]=0.5*(y_V[icon_V[0,iel]]+y_V[icon_V[1,iel]]) 
        # node 4 is between nodes 1 and 2
        x_V[icon_V[4,iel]]=0.5*(x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]]) 
        y_V[icon_V[4,iel]]=0.5*(y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]]) 
        # node 5 is between nodes 0 and 2
        x_V[icon_V[5,iel]]=0.5*(x_V[icon_V[0,iel]]+x_V[icon_V[2,iel]]) 
        y_V[icon_V[5,iel]]=0.5*(y_V[icon_V[0,iel]]+y_V[icon_V[2,iel]]) 
        # recenter middle node
        x_V[icon_V[6,iel]]=(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])/3.
        y_V[icon_V[6,iel]]=(y_V[icon_V[0,iel]]+y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])/3.

    if debug: np.savetxt('gridV1.ascii',np.array([x_V,y_V]).T,header='# x_V,y_V')

    for iel in range(0,nel):
        x_P[icon_P[0,iel]]=x_V[icon_V[0,iel]]
        y_P[icon_P[0,iel]]=y_V[icon_V[0,iel]]
        x_P[icon_P[1,iel]]=x_V[icon_V[1,iel]]
        y_P[icon_P[1,iel]]=y_V[icon_V[1,iel]]
        x_P[icon_P[2,iel]]=x_V[icon_V[2,iel]]
        y_P[icon_P[2,iel]]=y_V[icon_V[2,iel]]

    print("evolve mesh: %.3f s" % (clock.time()-start))

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start=clock.time()

    A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs      = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    f_rhs    = np.zeros(Nfem_V,dtype=np.float64)        # right hand side f 
    h_rhs    = np.zeros(Nfem_P,dtype=np.float64)        # right hand side h 
    B   = np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
    N_mat    = np.zeros((3,m_P),dtype=np.float64) # matrix  

    C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    for iel in range(0,nel):

        if iel%5000==0: print('iel=',iel)

        K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for kq in range (0,nqel):

            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

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

            if JxWq<0: exit("jacobian is negative - bad triangle")

            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            etaq=viscosity(mat[iel]) ; eta[iel]=etaq
            rhoq=density(mat[iel])   ; rho[iel]=rhoq

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*etaq*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*gx*rhoq*JxWq
                f_el[ndof_V*i+1]+=N_V[i]*gy*rhoq*JxWq

            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]

            G_el-=B.T.dot(N_mat)*JxWq

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
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val[m1]
                   G_el[ikk,:]=0
                #end if
            #end for 
        #end for

        G_el*=eta_ref/Ly
        h_el*=eta_ref/Ly

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        #K_mat[m1,m2]+=K_el[ikk,jkk]
                        A_sparse[m1,m2] += K_el[ikk,jkk] 
                    #end for
                #end for
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    #G_mat[m1,m2]+=G_el[ikk,jkk]
                    A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk] 
                    A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk] 
                #end for
                f_rhs[m1]+=f_el[ikk] 
            #end for
        #end for
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            h_rhs[m2]+=h_el[k2]  
        #end for
    #end for

    rhs[0:Nfem_V]=f_rhs
    rhs[Nfem_V:Nfem]=h_rhs

    print("build FE matrix: %.3f s" % (clock.time()-start))

    ######################################################################
    # solve system
    ######################################################################
    start=clock.time()

    sparse_matrix=A_sparse.tocsr()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*eta_ref/Ly

    print("     -> u (m,M) %.6e %.6e (cm/yr)" %((np.min(u)/cm*year),(np.max(u))/cm*year))
    print("     -> v (m,M) %.6e %.6e (cm/yr)" %((np.min(v)/cm*year),(np.max(v))/cm*year))
    print("     -> p (m,M) %.6e %.6e (Pa)   " %(np.min(p),np.max(p)))

    if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

    print("solve time: %.3f s" % (clock.time()-start))

    ######################################################################
    # compute elemental strainrate in the middle
    ######################################################################
    start=clock.time()

    exx =np.zeros(nel,dtype=np.float64)  
    eyy =np.zeros(nel,dtype=np.float64)  
    exy =np.zeros(nel,dtype=np.float64)  
    sr  =np.zeros(nel,dtype=np.float64)  
  
    vrms=0.
    Trms=0.
    for iel in range(0,nel):
        for kq in range (0,nqel):
            # position & weight of quad. point
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
 
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
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            Tq=np.dot(N_V,T[icon_V[:,iel]])

            exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
            eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
            exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy_V,u[icon_V[:,iel]])*0.5

            exx[iel] += exxq*JxWq
            eyy[iel] += eyyq*JxWq
            exy[iel] += exyq*JxWq
            vrms+=(uq**2+vq**2)*JxWq
            Trms+=Tq**2*JxWq
        #end for
        exx[iel] /= area[iel] 
        eyy[iel] /= area[iel] 
        exy[iel] /= area[iel] 
        sr[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))
    Trms=np.sqrt(Trms/(Lx*Ly))

    print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))
    print("     -> vrms %.6e " %vrms )
    print("     -> Trms %.6e " %Trms )

    print("compute elemental sr: %.3f s" % (clock.time()-start))

    ######################################################################
    # compute nodal strainrate 
    #
    #  P_2^+           P_-1
    #
    #  02              02
    #  ||\\            ||\\
    #  || \\           || \\
    #  ||  \\          ||  \\
    #  05   04         ||   \\
    #  || 06 \\        ||    \\
    #  ||     \\       ||     \\
    #  00==03==01      00======01
    # 
    ######################################################################
    start=clock.time()

    exx_n=np.zeros(nn_V,dtype=np.float64)  
    eyy_n=np.zeros(nn_V,dtype=np.float64)  
    exy_n=np.zeros(nn_V,dtype=np.float64)  
    count=np.zeros(nn_V,dtype=np.int32)  

    r_V=np.array([0,1,0,0.5,0.5,  0,1./3],dtype=np.float64)
    s_V=np.array([0,0,1,  0,0.5,0.5,1./3],dtype=np.float64)

    for iel in range(0,nel):
        for kk in range(0,m_V):
            rq=r_V[kk]
            sq=s_V[kk]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            e_xx=np.dot(dNdx_V,u[icon_V[:,iel]])
            e_yy=np.dot(dNdy_V,v[icon_V[:,iel]])
            e_xy=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
            inode=icon_V[kk,iel]
            exx_n[inode]+=e_xx
            eyy_n[inode]+=e_yy
            exy_n[inode]+=e_xy
            count[inode]+=1
        #end for
    #end for

    exx_n/=count
    eyy_n/=count
    exy_n/=count
    sr_n=np.sqrt(0.5*(exx_n**2+eyy_n**2)+exy_n**2)

    print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))

    print("compute nodal sr : %.3f s" % (clock.time()-start))

    #####################################################################
    # interpolate pressure onto velocity grid points
    #####################################################################
    #
    #  02          #  02
    #  ||\\        #  ||\\
    #  || \\       #  || \\
    #  ||  \\      #  ||  \\
    #  05   04     #  ||   \\
    #  || 06 \\    #  ||    \\
    #  ||     \\   #  ||     \\
    #  00==03==01  #  00======01
    #
    #####################################################################
    start=clock.time()

    q=np.zeros(nn_V,dtype=np.float64)
    cc=np.zeros(nn_V,dtype=np.float64)
    p_el=np.zeros(nel,dtype=np.float64)

    for iel in range(0,nel):
        q[icon_V[0,iel]]+=p[icon_P[0,iel]]                        ; cc[icon_V[0,iel]]+=1.
        q[icon_V[1,iel]]+=p[icon_P[1,iel]]                        ; cc[icon_V[1,iel]]+=1.
        q[icon_V[2,iel]]+=p[icon_P[2,iel]]                        ; cc[icon_V[2,iel]]+=1.
        q[icon_V[3,iel]]+=(p[icon_P[0,iel]]+p[icon_P[1,iel]])*0.5 ; cc[icon_V[3,iel]]+=1.
        q[icon_V[4,iel]]+=(p[icon_P[1,iel]]+p[icon_P[2,iel]])*0.5 ; cc[icon_V[4,iel]]+=1.
        q[icon_V[5,iel]]+=(p[icon_P[0,iel]]+p[icon_P[2,iel]])*0.5 ; cc[icon_V[5,iel]]+=1.
        p_el[iel]=(p[icon_P[0,iel]]+p[icon_P[1,iel]]+p[icon_P[2,iel]])/3.
    #end for

    for i in range(0,nn_V):
        if cc[i] != 0:
           q[i]=q[i]/cc[i]
        #end if
    #end for

    if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

    print("project pressure to V nodes: %.3f s" % (clock.time()-start))

    #####################################################################
    # compure dev stress tensor and stress tensor 
    #####################################################################
    start=clock.time()

    tauxx = np.zeros(nel,dtype=np.float64)  
    tauyy = np.zeros(nel,dtype=np.float64)  
    tauxy = np.zeros(nel,dtype=np.float64)  
    sigmaxx = np.zeros(nel,dtype=np.float64)  
    sigmayy = np.zeros(nel,dtype=np.float64)  
    sigmaxy = np.zeros(nel,dtype=np.float64)  

    tauxx[:]=2*eta[:]*exx[:]
    tauyy[:]=2*eta[:]*eyy[:]
    tauxy[:]=2*eta[:]*exy[:]

    sigmaxx[:]=-p_el[:]+2*eta[:]*exx[:]
    sigmayy[:]=-p_el[:]+2*eta[:]*eyy[:]
    sigmaxy[:]=        +2*eta[:]*exy[:]

    print("compute element stress: %.3f s" % (clock.time()-start))

    #####################################################################
    # plot of solution
    # the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
    #####################################################################
    start=clock.time()

    filename = 'solution_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    area.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    rho.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    eta.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p (el)' Format='ascii'> \n")
    p_el.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='mat (el)' Format='ascii'> \n")
    mat.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    exx.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    eyy.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    exy.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
    tauxx.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
    tauyy.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
    tauxy.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
    sr.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
    sigmaxx.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
    sigmayy.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
    sigmaxy.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta_p(dev stress)' Format='ascii'> \n")
    for iel in range(0,nel):
        theta_p=0.5*np.arctan(2*tauxy[iel]/(tauxx[iel]-tauyy[iel]))
        vtufile.write("%e \n" % (theta_p/np.pi*180.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta_p(stress)' Format='ascii'> \n")
    for iel in range(0,nel):
        theta_p=0.5*np.arctan(2*sigmaxy[iel]/(sigmaxx[iel]-sigmayy[iel]))
        vtufile.write("%e \n" % (theta_p/np.pi*180.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_max' Format='ascii'> \n")
    for iel in range(0,nel):
        tau_max=np.sqrt( (tauxx[iel]-tauyy[iel])**2/4 +tauxy[iel]**2 )
        vtufile.write("%e \n" % tau_max)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_max' Format='ascii'> \n")
    for iel in range(0,nel):
        sigma_max=np.sqrt( (sigmaxx[iel]-sigmayy[iel])**2/4 +sigmaxy[iel]**2 )
        vtufile.write("%e \n" % sigma_max)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    exx_n.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    eyy_n.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    exy_n.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
    sr_n.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    q.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
    T.tofile(vtufile,sep=' ',format='%.4e')
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
    for i in range(0,nn_V):
        if bc_fix[i*2]:
           val=1
        else:
           val=0
        vtufile.write("%e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
    for i in range(0,nn_V):
        if bc_fix[i*2+1]:
           val=1
        else:
           val=0
        vtufile.write("%e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                              icon_V[3,iel],icon_V[4,iel],icon_V[5,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*6))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    #------------------------------------------------------

    filename = 'stress_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nel,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        vtufile.write("%e %e %e \n" %(x_e[iel],y_e[iel],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_1 (dir)' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        theta_p=0.5*np.arctan(2*tauxy[i]/(tauxx[i]-tauyy[i]))
        vtufile.write("%e %e %e \n" %( np.cos(theta_p),np.sin(theta_p),0) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_2 (dir)' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        theta_p=0.5*np.arctan(2*tauxy[i]/(tauxx[i]-tauyy[i])) + np.pi/2.
        vtufile.write("%e %e %e \n" %( np.cos(theta_p),np.sin(theta_p),0) )
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_1 (dir+mag)' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        theta_p=0.5*np.arctan(2*sigmaxy[i]/(sigmaxx[i]-sigmayy[i]))
        sigma1=(sigmaxx[iel]+sigmayy[iel])/2. + np.sqrt( (sigmaxx[iel]-sigmayy[iel])**2/4 +sigmaxy[iel]**2 ) 
        vtufile.write("%e %e %e \n" %( np.cos(theta_p)*sigma1,np.sin(theta_p)*sigma1,0.) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_2 (dir+mag)' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nel):
        theta_p=0.5*np.arctan(2*sigmaxy[i]/(sigmaxx[i]-sigmayy[i]))
        sigma2=(sigmaxx[iel]+sigmayy[iel])/2. - np.sqrt( (sigmaxx[iel]-sigmayy[iel])**2/4 +sigmaxy[iel]**2 ) 
        vtufile.write("%e %e %e \n" %( np.cos(theta_p)*sigma2,np.sin(theta_p)*sigma2,0.) )
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range (0,nel):
        vtufile.write("%d\n" % i )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range (0,nel):
        vtufile.write("%d \n" % (i+1) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range (0,nel):
        vtufile.write("%d \n" % 1)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    #------------------------------------------------------
    #filename = 'surface_{:04d}.ascii'.format(istep)
    #surffile=open(filename,"w")
    #for i in range(0,NV):
    #    if on_surf[i]:
    #       surffile.write("%e %e %e %e\n" %(xV[i],yV[i],u[i],v[i]))
    #surffile.close()
    #xsurf=xV[on_surf]
    #ysurf=yV[on_surf]
    #opla=np.argsort(xsurf)
    #np.savetxt(filename,np.array([xsurf[opla],ysurf[opla]]).T,header='# xV,yV')

    #------------------------------------------------------

    #c=np.sqrt(u**2+v**2)
    #plt.quiver(xV,yV,u,v,c,alpha=.85)
    #plt.title('Velocity field')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #filename = 'velocity_field_{:04d}.pdf'.format(istep)
    #plt.savefig(filename, bbox_inches='tight')
    ##plt.show()
    #plt.clf()

    print("write data: %.3fs" % (clock.time()-start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################

