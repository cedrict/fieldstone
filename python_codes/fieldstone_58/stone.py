import numpy as np
import scipy
import time as clock 
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def basis_functions_V(r,s):
    N0= 1.-rq-sq
    N1= rq
    N2= sq
    return np.array([N0,N1,N2],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= -1. 
    dNdr1= +1.
    dNdr2=  0.
    return np.array([dNdr0,dNdr1,dNdr2],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= -1. 
    dNds1=  0.
    dNds2= +1.
    return np.array([dNds0,dNds1,dNds2],dtype=np.float64)

###############################################################################

def sigmaxx_anal(x,y,P,R):
    if experiment==1:
       r1_2=(x-0)**2+(y-R)**2
       r2_2=(x-0)**2+(y+R)**2
       val=-2.*P/np.pi*( (R-y)*x**2/r1_2**2 +(R+y)*x**2/r2_2**2 - 1./(2*R)  )
    else:
       val=0
    return val

def sigmayy_anal(x,y,P,R):
    if experiment==1:
       r1_2=(x-0)**2+(y-R)**2
       r2_2=(x-0)**2+(y+R)**2
       val=-2.*P/np.pi*( (R-y)**3/r1_2**2 +(R+y)**3/r2_2**2 - 1./(2*R)  )
    else:
       val=0
    return val

def sigmaxy_anal(x,y,P,R):
    if experiment==1:
       r1_2=(x-0)**2+(y-R)**2
       r2_2=(x-0)**2+(y+R)**2
       val=2.*P/np.pi*( (R-y)**2*x/r1_2**2 - (R+y)**2*x/r2_2**2  )
    else:
       val=0
    return val

def sigma_1_anal(x,y,P,R):
    if experiment==1:
       sigxx=sigmaxx_anal(x,y,P,R)
       sigyy=sigmayy_anal(x,y,P,R)
       sigxy=sigmaxy_anal(x,y,P,R)
       val=0.5*(sigxx+sigyy)+np.sqrt( 0.25*(sigxx-sigyy)**2+sigxy**2 )
    else:
       val=0
    return val

def p_anal(x,y,P,R):
    if experiment==1:
       r1_2=(x-0)**2+(y-R)**2
       r2_2=(x-0)**2+(y+R)**2
       val1=-2.*P/np.pi*( (R-y)*x**2/r1_2**2 +(R+y)*x**2/r2_2**2 - 1./(2*R)  )
       val2=-2.*P/np.pi*( (R-y)**3/r1_2**2 +(R+y)**3/r2_2**2 - 1./(2*R)  )
       val=-0.5*(val1+val2)
    else:
       val=0
    return val

###############################################################################

print("*******************************")
print("********** stone 58 ***********")
print("*******************************")

nLayers=131   # must be odd !
outer_radius=1.

m_V=3
ndof=2
ndim=2

eps=1e-8

mu=1.                        #
nu=0.25                      # poisson ratio
lambdaa=2.*mu*nu/(1.-2.*nu)  #

rho=0                     # density
gx=0.                     # gravity vector x-component
gy=0.                     # gravity vector y-component

p_bc=1.

visu=1

# experiment=1: disc under vertical compression
# experiment=2: disc under compression left and bottom, no-slip top and right
experiment=1

if experiment==1: nsection=6
if experiment==2: nsection=8

nel=nsection*nLayers*nLayers               # number of elements
nn_V=1+int(nsection/2)*nLayers*(nLayers+1) # number of mesh nodes
Nfem=nn_V*ndof                             # Total number of degrees of freedom

nqel=3
qcoords_r=[1./6.,1./6.,2./3.] # coordinates & weights 
qcoords_s=[2./3.,1./6.,1./6.] # of quadrature points
qweights =[1./6.,1./6.,1./6.]

debug=False

###############################################################################
# WARNING: the Neumann b.c. are not accurate since at the moment
# they do not use the normal to the element edge!
###############################################################################

print ('nLayers  =',nLayers)
print ('nn_V     =',nn_V)
print ('nel      =',nel)
print ('Nfem     =',Nfem)
print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)      # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)      # y coordinates
outer_node = np.zeros(nn_V,dtype=bool) # on the outer hull yes/no 

# by starting at counter=1, we omit counter=0, which is 
# the center point and it automatically gets x=y=0
 
counter = 1 
for iLayer in range(1,nLayers+1):
    radius = outer_radius * float(iLayer)/float(nLayers)
    nPointsOnCircle = nsection*iLayer
    for iPoint in range (0,nPointsOnCircle):
        # Coordinates are created, starting at twelve o'clock, 
        # going in clockwise direction
        x_V[counter] = radius * np.sin(2. * np.pi * float(iPoint) / float(nPointsOnCircle))
        y_V[counter] = radius * np.cos(2. * np.pi * float(iPoint) / float(nPointsOnCircle))
        if iLayer==nLayers:
           outer_node[counter]=True
        counter += 1 
    #enddo
#enddo

if debug: np.savetxt('grid.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time()-start))

#################################################################
###############################################################################
# connectivity
###############################################################################
#################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

# first nsection triangles by hand
if nsection==4:
   icon_V[:,0] = (0,2,1)
   icon_V[:,1] = (0,3,2)
   icon_V[:,2] = (0,4,3)
   icon_V[:,3] = (0,1,4)
   iInner = 1 
   iOuter = 5 
elif nsection==6:
   icon_V[:,0] = (0,2,1)
   icon_V[:,1] = (0,3,2)
   icon_V[:,2] = (0,4,3)
   icon_V[:,3] = (0,5,4)
   icon_V[:,4] = (0,6,5)
   icon_V[:,5] = (0,1,6)
   iInner = 1 
   iOuter = 7 
elif nsection==8:
   icon_V[:,0] = (0,2,1)
   icon_V[:,1] = (0,3,2)
   icon_V[:,2] = (0,4,3)
   icon_V[:,3] = (0,5,4)
   icon_V[:,4] = (0,6,5)
   icon_V[:,5] = (0,7,6)
   icon_V[:,6] = (0,8,7)
   icon_V[:,7] = (0,1,8)
   iInner = 1 
   iOuter = 9 
else:
   exit('nsection value not supported')
#end if

storedElems = nsection 

for iLayer in range(2,nLayers+1):  
    nPointsOnCircle = nsection*iLayer   
    #print ('iLayer=',iLayer,'nPointsOnCircle=',nPointsOnCircle,'iInner=',iInner,'iOuter=',iOuter) 
    for iSection in range (1,nsection):     
        #print ('Section',iSection) 
        for iBlock in range(0,iLayer-1):
            icon_V[:,storedElems] = (iInner, iOuter +1, iOuter )
            #print ('   elt=',storedElems,'nodes:',icon_V[:,storedElems])
            storedElems = storedElems + 1    
            icon_V[:,storedElems] = (iInner, iInner+1, iOuter + 1) 
            #print ('   elt=',storedElems,'nodes:',icon_V[:,storedElems])
            storedElems = storedElems + 1 
            iInner += 1 
            iOuter += 1 
        #enddo
        icon_V[:,storedElems] = (iInner, iOuter+1, iOuter )
        #print ('   elt=',storedElems,'nodes:',icon_V[:,storedElems])
        storedElems = storedElems + 1 
        iOuter = iOuter + 1 
    #enddo

    # do the 6th and closing section. This has some extra difficulty where it is 
    # attached to the starting point

    # first do the regular blocks within the section
    #print ('Section',6) 
    for iBlock in range(0,iLayer-2): 
        icon_V[:,storedElems] = (iInner, iOuter+1, iOuter )
        #print ('   elt=',storedElems,'nodes:',icon_V[:,storedElems])
        storedElems += 1 
        icon_V[:,storedElems] = (iInner, iInner+1, iOuter + 1) 
        #print ('   elt=',storedElems,'nodes:',icon_V[:,storedElems])
        storedElems += 1 
        iInner += 1
        iOuter += 1
    #enddo

    # do the last block, which shares an inner point with the first section
    icon_V[:,storedElems] = (iInner, iOuter+1, iOuter )
    #print ('   elt=',storedElems,'nodes:',icon_V[:,storedElems])
    storedElems = storedElems + 1

    icon_V[:,storedElems] = (iInner, iInner + 1 - nsection*(iLayer-1) , iOuter+1)
    #print ('   elt=',storedElems,'nodes:',icon_V[:,storedElems])
    storedElems += 1 

    # last element, closing the layer.
    icon_V[:,storedElems] = ( iInner + 1 - nsection*(iLayer-1),iInner+1,iOuter+1 )
    #print ('   elt=',storedElems,'nodes:',icon_V[:,storedElems])
    storedElems += 1 

    iInner = iInner + 1
    iOuter = iOuter + 2
#enddo

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# rotate 90 degree mesh to expose flat surface on top and bottom
###############################################################################

if (experiment==1):
   temp=np.zeros(nn_V,dtype=np.float64)   
   temp[:]=x_V[:]
   x_V[:]=-y_V[:]
   y_V[:]=temp[:]
   if debug: np.savetxt('grid_rot.ascii',np.array([x_V,y_V]).T,header='# x,y')

###############################################################################
# define boundary conditions
# Experiment 1: The boundary conditions for this benchmark are of 
# Neumann nature. # All we need to do is to remove the rotational 
# nullspace. We do so by zeroing the horizontal displacement on 
# the vertical axis and the horizontal displacement on the vertical
# axis since the system is symmetrical with respect to both axis.
# Experiment 2: the no-slip b.c. on the top and right do not allow
# for a null space. 
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if experiment==1:
   for i in range(0,nn_V):
      if abs(x_V[i])<eps:
         bc_fix_V[i*ndof]   = True ; bc_val_V[i*ndof]   = 0.
      if abs(y_V[i])<eps:
         bc_fix_V[i*ndof+1] = True ; bc_val_V[i*ndof+1] = 0.

if experiment==2:
   for i in range(0,nn_V):
      if abs(x_V[i]-outer_radius)<eps:
         bc_fix_V[i*ndof+0] = True ; bc_val_V[i*ndof+0] = 0.
         bc_fix_V[i*ndof+1] = True ; bc_val_V[i*ndof+1] = 0.
      if abs(y_V[i]-outer_radius)<eps:
         bc_fix_V[i*ndof+0] = True ; bc_val_V[i*ndof+0] = 0.
         bc_fix_V[i*ndof+1] = True ; bc_val_V[i*ndof+1] = 0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
###############################################################################
start=clock.time()

jcb=np.zeros((2,2),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

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
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))
print("     -> total area (anal) %.6f " %(np.pi*outer_radius**2))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
B=np.zeros((3,ndof*m_V),dtype=np.float64)       # gradient matrix B 
C=np.array([[2*mu+lambdaa,lambdaa,0],[lambdaa,2*mu+lambdaa,0],[0,0,mu]],dtype=np.float64) 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    K_el=np.zeros((m_V*ndof,m_V*ndof),dtype=np.float64)
    f_el=np.zeros(m_V*ndof,dtype=np.float64)

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

        xq=np.dot(N_V,x_V[icon_V[:,iel]])
        yq=np.dot(N_V,y_V[icon_V[:,iel]])

        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

        for i in range(0,m_V):
            B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                              [0.       ,dNdy_V[i]],
                              [dNdy_V[i],dNdx_V[i]]]

        # compute elemental K_el matrix
        K_el+=B.T.dot(C.dot(B))*JxWq

        # compute elemental rhs vector
        for i in range(0,m_V):
            f_el[ndof*i  ]+=N_V[i]*gx*rho*JxWq
            f_el[ndof*i+1]+=N_V[i]*gy*rho*JxWq
        #end for

    #end for kq

    # impose dirichlet b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon_V[k1,iel]+i1
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
            #end if
        #end for 
    #end for

    # impose Neumann b.c.
    inode1=icon_V[0,iel]
    inode2=icon_V[1,iel]
    inode3=icon_V[2,iel]
    if outer_node[inode2] and outer_node[inode3]: # nodes 2&3 on boundary 
       if experiment==1:
          #top boundary
          if y_V[inode1]>0 and x_V[inode2]>0 and x_V[inode3]<0: 
             surf=1#x[inode2]-x[inode3]
             f_el[3]-=surf*p_bc*0.5
             f_el[5]-=surf*p_bc*0.5
          #end if 
          #bottom boundary
          if y_V[inode1]<0 and x_V[inode2]<0 and x_V[inode3]>0: 
             surf=1#x[inode3]-x[inode2]
             f_el[3]+=surf*p_bc*0.5
             f_el[5]+=surf*p_bc*0.5
          #end if 
       if experiment==2:
          # applying Neumann bc on left
          if abs(x_V[inode3]+outer_radius)<eps:
             surf=1
             f_el[2]+=surf*p_bc*0.5
             f_el[4]+=surf*p_bc*0.5
          #end if 
          if abs(x_V[inode2]+outer_radius)<eps:
             surf=1
             f_el[2]+=surf*p_bc*0.5
             f_el[4]+=surf*p_bc*0.5
          #end if 
          # applying Neumann bc on bottom
          if abs(y_V[inode3]+outer_radius)<eps:
             surf=1
             f_el[3]+=surf*p_bc*0.5
             f_el[5]+=surf*p_bc*0.5
          #end if 
          if abs(y_V[inode2]+outer_radius)<eps:
             surf=1
             f_el[3]+=surf*p_bc*0.5
             f_el[5]+=surf*p_bc*0.5
          #end if 
       #end if 
    #end if 

    # assemble matrix and right hand side
    for k1 in range(0,m_V):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            b_fem[m1]+=f_el[ikk]
        #end for
    #end for
#end for

print("     -> b_fem (m,M) %.4f %.4f " %(np.min(b_fem),np.max(b_fem)))

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('displacement.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split solution: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure and compute elemental strain
###############################################################################
start=clock.time()

#u[:]=x[:]**2
#v[:]=y[:]**2

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  
p=np.zeros(nel,dtype=np.float64)   
divv=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 1./3.
    sq = 1./3.

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

    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    
    divv[iel]=exx[iel]+eyy[iel]

    e[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)

    p[iel]=-(lambdaa+mu)*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))

sigmaxx=lambdaa*divv+2*mu*exx
sigmayy=lambdaa*divv+2*mu*eyy
sigmaxy=            +2*mu*exy

sigma_angle=np.arctan(2.*sigmaxy/(sigmaxx-sigmayy))/2. /np.pi*180

sigma_1=0.5*(sigmaxx+sigmayy)+np.sqrt( 0.25*(sigmaxx-sigmayy)**2+sigmaxy**2 )

if debug:
   np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
   np.savetxt('strain.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p, sr and stress: %.3f s" % (clock.time()-start))

###############################################################################
# project pressure, strain and stress onto nodes
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
counter=np.zeros(nn_V,dtype=np.float64)  
exx_nodal=np.zeros(nn_V,dtype=np.float64)  
eyy_nodal=np.zeros(nn_V,dtype=np.float64)  
exy_nodal=np.zeros(nn_V,dtype=np.float64)  
sigmaxx_nodal=np.zeros(nn_V,dtype=np.float64)  
sigmayy_nodal=np.zeros(nn_V,dtype=np.float64)  
sigmaxy_nodal=np.zeros(nn_V,dtype=np.float64)  

for iel,nodes in enumerate(icon_V.T):
    for k in range(0,m_V):
        q[nodes[k]]+=p[iel] 
        exx_nodal[nodes[k]]+=exx[iel]
        eyy_nodal[nodes[k]]+=eyy[iel]
        exy_nodal[nodes[k]]+=exy[iel]
        sigmaxx_nodal[nodes[k]]+=sigmaxx[iel]
        sigmayy_nodal[nodes[k]]+=sigmayy[iel]
        sigmaxy_nodal[nodes[k]]+=sigmaxy[iel]
        counter[nodes[k]]+=1

q/=counter
exx_nodal/=counter
eyy_nodal/=counter
exy_nodal/=counter
sigmaxx_nodal/=counter
sigmayy_nodal/=counter
sigmaxy_nodal/=counter

sigma_angle_nodal=np.arctan(2.*sigmaxy_nodal/(sigmaxx_nodal-sigmayy_nodal))/2. /np.pi*180

sigma_1_nodal=0.5*(sigmaxx_nodal+sigmayy_nodal) \
             +np.sqrt(0.25*(sigmaxx_nodal-sigmayy_nodal)**2+sigmaxy_nodal**2)
sigma_2_nodal=0.5*(sigmaxx_nodal+sigmayy_nodal) \
             -np.sqrt(0.25*(sigmaxx_nodal-sigmayy_nodal)**2+sigmaxy_nodal**2)

print("     -> q (m,M) %.4f %.4f " %(np.min(q),np.max(q)))
print("     -> exx (m,M) %.6e %.6e " %(np.min(exx_nodal),np.max(exx_nodal)))
print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy_nodal),np.max(eyy_nodal)))
print("     -> exy (m,M) %.6e %.6e " %(np.min(exy_nodal),np.max(exy_nodal)))

if debug:
   np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')
   np.savetxt('strain_nodal.ascii',np.array([x_V,y_V,exx_nodal,eyy_nodal,exy_nodal]).T)

print("compute nodal quantities: %.3f s" % (clock.time()-start))

###############################################################################
# compute root mean square displacement vrms 
###############################################################################
start=clock.time()

vrms=0.
avrg_u=0.
avrg_v=0.

for iel in range(0,nel):
    for kq in range(0,nqel):
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
        JxWq=np.linalg.det(jcb)*weightq
        uq=np.dot(N_V,u[icon_V[:,iel]])
        vq=np.dot(N_V,v[icon_V[:,iel]])
        vrms+=(uq**2+vq**2)*JxWq
        avrg_u+=uq*JxWq
        avrg_v+=vq*JxWq
    # end for kq
# end for iel

avrg_u/=np.sum(area)
avrg_v/=np.sum(area)

print("     -> vrms   = %.6e m/s" %(vrms))
print("     -> avrg u = %.6e m/s" %(avrg_u))
print("     -> avrg v = %.6e m/s" %(avrg_v))

print("compute vrms: %.3fs" % (clock.time()-start))

###############################################################################
# write out quantities on axis for (gnu)plotting 
###############################################################################
start=clock.time()

xaxis_file=open('xaxis.ascii',"w")
for i in range(0,nn_V):
    if abs(y_V[i])<eps: 
       xaxis_file.write("%8e %8e %8e %8e %8e %8e %8e %8e %8e %8e\n" %(
                         x_V[i],u[i],v[i],q[i],
                         exx_nodal[i],eyy_nodal[i],exy_nodal[i],
                         sigmaxx_nodal[i],sigmayy_nodal[i],sigmaxy_nodal[i]))

yaxis_file=open('yaxis.ascii',"w")
for i in range(0,nn_V):
    if abs(x_V[i])<eps: 
       yaxis_file.write("%8e %8e %8e %8e %8e %8e %8e %8e %8e %8e\n" %(
                         y_V[i],u[i],v[i],q[i],
                         exx_nodal[i],eyy_nodal[i],exy_nodal[i],
                         sigmaxx_nodal[i],sigmayy_nodal[i],sigmaxy_nodal[i]))

print("export data for plots: %.3fs" % (clock.time()-start))

###############################################################################
# export to vtu 
###############################################################################
start=clock.time()

if visu==1:
    vtufile=open('solution.vtu',"w")
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
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (p[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigmaxx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigmaxx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigmayy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigmayy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigmaxy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigmaxy[iel]))
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' Name='sigma_angle' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_angle[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_1' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_1[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (p_anal(xc[iel],yc[iel],p_bc,outer_radius)))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" %divv[iel]) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")

    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" %exx_nodal[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" %eyy_nodal[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" %exy_nodal[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigmaxx' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigmaxx_nodal[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigmayy' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigmayy_nodal[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigmaxy' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigmaxy_nodal[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_angle' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigma_angle_nodal[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_1' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigma_1_nodal[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_2' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigma_2_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='sigma princ. 1' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" % (sigma_1_nodal[i]*np.cos(sigma_angle_nodal[i]),
                                             sigma_1_nodal[i]*np.sin(sigma_angle_nodal[i]), 0.)   )
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='sigma princ. 2' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" % (sigma_2_nodal[i]*np.cos(sigma_angle_nodal[i]+np.pi/2),
                                             sigma_2_nodal[i]*np.sin(sigma_angle_nodal[i]+np.pi/2), 0.)   )
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='sigma princ. 1 dir' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" % (np.cos(sigma_angle_nodal[i]),
                                             np.sin(sigma_angle_nodal[i]), 0.)   )
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='sigma princ. 2 dir' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" % (np.cos(sigma_angle_nodal[i]+np.pi/2),
                                             np.sin(sigma_angle_nodal[i]+np.pi/2), 0.)   )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigmaxx (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigmaxx_anal(x_V[i],y_V[i],p_bc,outer_radius) )
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigmayy (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigmayy_anal(x_V[i],y_V[i],p_bc,outer_radius) )
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigmaxy (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigmaxy_anal(x_V[i],y_V[i],p_bc,outer_radius) )
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_1 (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" % sigma_1_anal(x_V[i],y_V[i],p_bc,outer_radius) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='outer_node' Format='ascii'> \n")
    for i in range(0,nn_V):
        if outer_node[i]:
           vtufile.write("%10e \n" % 1)
        else:
           vtufile.write("%10e \n" % 0)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='bc_fix (u)' Format='ascii'> \n")
    for i in range(0,nn_V):
        if bc_fix_V[i*ndof]:
           vtufile.write("%10e \n" % 1)
        else:
           vtufile.write("%10e \n" % 0)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='bc_fix (v)' Format='ascii'> \n")
    for i in range(0,nn_V):
        if bc_fix_V[i*ndof+1]:
           vtufile.write("%10e \n" % 1)
        else:
           vtufile.write("%10e \n" % 0)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*3))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

print("export to vtu: %.3fs" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
