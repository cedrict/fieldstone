import numpy as np
import time as clock
from scipy.sparse import lil_matrix
import scipy.sparse as sps
import sys as sys
from mesher import *
from basis_functions import *

###############################################################################

def analytical_displacement(x,y,nu,R,Ly):
    z=Ly-y
    z1=-y
    z2=-y+2*Ly
    r1_2=x**2+y**2
    r2_2=x**2+(y-2*Ly)**2
    # eq1,2+10,11 of vebo96
    ux=-(x/r1_2  + x/r2_2 ) - 2* ( (1-2*nu)*x/r2_2  - 2*x*z*z2/r2_2**2      )
    uy=-(z1/r1_2 + z2/r2_2) + 2* ( 2*(1-nu)*z2/r2_2 - z*(x**2-z2**2)/r2_2**2)

    ux*=epsilon*R**2
    uy*=epsilon*R**2 * (-1) # z -> y axis

    return ux,uy

def analytical_stress(x,y,mu,nu,R,Ly):
    z=Ly-y
    z1=-y
    z2=-y+2*Ly
    r1_2=x**2+y**2
    r2_2=x**2+(y-2*Ly)**2

    sigma_xx= (x**2-z1**2)/r1_2**2 \
            + (x**2-z2**2)/r2_2**2 \
            +2*( (x**2-z2**2)/r2_2**2 - 2*y*z2*(3*x**2-z2**2)/r2_2**3 )

    sigma_yy= -(x**2-z1**2)/r1_2**2 \
              -(x**2-z2**2)/r2_2**2 \
            +2*( (x**2-z2**2)/r2_2**2 + 2*y*z2*(3*x**2-z2**2)/r2_2**3 )

    sigma_xy= 2*x*z1/r1_2**2 \
            + 2*x*z2/r2_2**2 \
            + 4*x*y*(x**2-3*z2**2)/r2_2**3

    sigma_xx*=2*mu*epsilon*R**2
    sigma_yy*=2*mu*epsilon*R**2
    sigma_xy*=2*mu*epsilon*R**2

    return sigma_xx,sigma_yy,sigma_xy

###############################################################################

sqrt2=np.sqrt(2.)
sqrt3=np.sqrt(3.)
eps=1e-8

print("*******************************")
print("********** stone 186 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

if int(len(sys.argv) == 4):
   print(sys.argv)
   nelxA=int(sys.argv[1])
   xi=float(sys.argv[2])
   psi=float(sys.argv[3])
else:
   nelxA=32
   xi=16
   psi=8

experiment=1

if experiment==1: # GTecTon setup
   E=5e10
   nu=0.35
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   radius=2e3
   L=7e3
   epsilon=-0.55e-3
   sigma_bc=5e6

if experiment==2: # ASPECT setup
   nu=0.48
   mu=1e10
   E=2*mu*(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   radius=1e3
   L=4e3
   epsilon=0.55e-3

extent_x=xi*L
extent_y=psi*L

nelxB=int(xi*nelxA)
nelyC=int(psi*nelxA)

xSW=0          ; ySW=-L-extent_y # lower left corner
xNE=L+extent_x ; yNE=L           # upper right corner

print('     -> mu=',mu)
print('     -> E=',E)
print('     -> lambdaa=',lambdaa)
print('     -> xi=',xi)
print('     -> psi=',psi) 
print('     -> xSW,ySW=',xSW,ySW)
print('     -> xNE,yNE=',xNE,yNE)
print('     -> nelxA=',nelxA)
print('     -> nelxB=',nelxB)
print('     -> nelyC=',nelyC)

debug=False

###############################################################################
# meshing the domain
###############################################################################
start=clock.time()

x_V,y_V,icon_V,nn_V,nel,Nfem=mesher(nelxA,nelxB,nelyC,L,extent_x,extent_y,radius,m_V)

print("meshing the domain: %.3f s" % (clock.time()-start))

###############################################################################
# flag boundary elements
###############################################################################
start=clock.time()

left=np.zeros(nn_V,dtype=bool)  
right=np.zeros(nn_V,dtype=bool)  
bottom=np.zeros(nn_V,dtype=bool)  
top=np.zeros(nn_V,dtype=bool)  
circle=np.zeros(nn_V,dtype=bool)  

for i in range(0,nn_V):
    if abs(x_V[i]-xSW)/L<eps: left[i]=True  
    if abs(x_V[i]-xNE)/L<eps: right[i]=True  
    if abs(y_V[i]-ySW)/L<eps: bottom[i]=True  
    if abs(y_V[i]-yNE)/L<eps: top[i]=True  
    if np.sqrt(x_V[i]**2+y_V[i]**2)<(1+eps)*radius:  circle[i]=True  

print("flag boundary nodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute Cartesian and polar coordinates of element centers
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
r_e=np.zeros(nel,dtype=np.float64)  
theta_e=np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    for k in range(0,m_V):
        x_e[iel]+=x_V[icon_V[k,iel]]*0.25
        y_e[iel]+=y_V[icon_V[k,iel]]*0.25
    r_e[iel]=np.sqrt(x_e[iel]**2+y_e[iel]**2)
    theta_e[iel]=np.arctan2(y_e[iel],x_e[iel])
   
if debug: np.savetxt('centers.ascii',np.array([x_e,y_e,r_e,theta_e]).T)

print("compute el coords.: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if left[i]:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0
    if bottom[i]:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
    if right[i]:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
    #if circle[i]:
    #   bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = x_V[i]/100
    #   bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = y_V[i]/100

print("define boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)
jcb=np.zeros((2,2),dtype=np.float64)
C=np.array([[2*mu+lambdaa,lambdaa,0],
            [lambdaa,2*mu+lambdaa,0],
            [0,0,mu]],dtype=np.float64) 

for iel in range(0,nel):

    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)

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

            # construct 3x8 b_mat matrix
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental A_fem matrix
            A_el+=B.T.dot(C.dot(B))*JxWq

            # compute elemental rhs vector
            #for i in range(0, m):
            #    b_el[2*i  ]-=N[i]*jcob*wq*gx*rho
            #    b_el[2*i+1]-=N[i]*jcob*wq*gy*rho

        #end for
    #end for

    #if circle[icon_V[0,iel]] and circle[icon_V[1,iel]]:
    #   print('aaaa')
    #if circle[icon_V[1,iel]] and circle[icon_V[2,iel]]:
    #   print('bbbb')
    #if circle[icon_V[2,iel]] and circle[icon_V[3,iel]]:
    #   print('cccc')
    if circle[icon_V[3,iel]] and circle[icon_V[0,iel]]:
       dist=np.sqrt((x_V[icon_V[3,iel]]-x_V[icon_V[0,iel]])**2+\
                    (y_V[icon_V[3,iel]]-y_V[icon_V[0,iel]])**2 )
       xm=(x_V[icon_V[3,iel]]+x_V[icon_V[0,iel]])/2
       ym=(y_V[icon_V[3,iel]]+y_V[icon_V[0,iel]])/2
       theta=np.arctan2(ym,xm)
       theta0=np.arctan2(y_V[icon_V[0,iel]],x_V[icon_V[0,iel]])
       theta3=np.arctan2(y_V[icon_V[3,iel]],x_V[icon_V[3,iel]])
       #print(xm,ym,theta,np.cos(theta),np.sin(theta))
       b_el[0]+=sigma_bc*dist*np.cos(theta0)/2
       b_el[1]+=sigma_bc*dist*np.sin(theta0)/2
       b_el[6]+=sigma_bc*dist*np.cos(theta3)/2
       b_el[7]+=sigma_bc*dist*np.sin(theta3)/2

    # apply boundary conditions
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndof_V*k1+i1
               aref=A_el[ikk,ikk]
               for jkk in range(0,m_V*ndof_V):
                   b_el[jkk]-=A_el[jkk,ikk]*fixt
                   A_el[ikk,jkk]=0.
                   A_el[jkk,ikk]=0.
               #end for
               A_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
            #end if
        #end for
    #end for

    # assemble matrix A_fem and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
                #end for
            #end for
            b_fem[m1]+=b_el[ikk]
        #end for
    #end for

#end for

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate u,v arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))

if debug: np.savetxt('displacement.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure, stress, strain, ...
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
cc=np.zeros(nn_V,dtype=np.float64)  
e_n=np.zeros(nn_V,dtype=np.float64)  
tau_n=np.zeros(nn_V,dtype=np.float64)  
e_xx_n=np.zeros(nn_V,dtype=np.float64)  
e_yy_n=np.zeros(nn_V,dtype=np.float64)  
e_xy_n=np.zeros(nn_V,dtype=np.float64)  
tau_xx_n=np.zeros(nn_V,dtype=np.float64)  
tau_yy_n=np.zeros(nn_V,dtype=np.float64)  
tau_zz_n=np.zeros(nn_V,dtype=np.float64)  
tau_xy_n=np.zeros(nn_V,dtype=np.float64)  
sigma_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xx_n=np.zeros(nn_V,dtype=np.float64)  
sigma_yy_n=np.zeros(nn_V,dtype=np.float64)  
sigma_zz_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xy_n=np.zeros(nn_V,dtype=np.float64)  

rVnodes=[-1,+1,+1,-1]
sVnodes=[-1,-1,+1,+1]

for iel in range(0,nel):
    for i in range(0,m_V):
        rq = rVnodes[i]
        sq = sVnodes[i]
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        exx=np.dot(dNdx_V,u[icon_V[:,iel]])
        eyy=np.dot(dNdy_V,v[icon_V[:,iel]])
        exy=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5\
           +np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
        e_xx_n[icon_V[i,iel]]+=exx
        e_yy_n[icon_V[i,iel]]+=eyy
        e_xy_n[icon_V[i,iel]]+=exy
        cc[icon_V[i,iel]]+=1.
    #end for
#end for

e_xx_n[:]/=cc[:]
e_yy_n[:]/=cc[:]
e_xy_n[:]/=cc[:]

q[:]=-(lambdaa+2./3.*mu)*(e_xx_n[:]+e_yy_n[:])

e_n[:]=np.sqrt(0.5*(e_xx_n[:]**2+e_yy_n[:]**2)+e_xy_n[:]**2)

sigma_xx_n[:]=lambdaa*(e_xx_n[:]+e_yy_n[:])+2*mu*e_xx_n[:]
sigma_yy_n[:]=lambdaa*(e_xx_n[:]+e_yy_n[:])+2*mu*e_yy_n[:]
sigma_xy_n[:]=2*mu*e_xy_n[:]
sigma_zz_n[:]=nu*(sigma_xx_n[:]+sigma_yy_n[:])

sigma_n[:]=np.sqrt(0.5*(sigma_xx_n[:]**2+sigma_yy_n[:]**2+sigma_zz_n[:]**2)+sigma_xy_n[:]**2)

tau_xx_n[:]=sigma_xx_n[:]-(1+nu)/3*(sigma_xx_n[:]+sigma_yy_n[:])
tau_yy_n[:]=sigma_yy_n[:]-(1+nu)/3*(sigma_xx_n[:]+sigma_yy_n[:])
tau_zz_n[:]=sigma_zz_n[:]-(1+nu)/3*(sigma_xx_n[:]+sigma_yy_n[:])
tau_xy_n[:]=sigma_xy_n[:]

tau_n[:]=np.sqrt(0.5*(tau_xx_n[:]**2+tau_yy_n[:]**2+tau_zz_n[:]**2)+tau_xy_n[:]**2)

print("     -> exx_n (m,M) %e %e " %(np.min(e_xx_n),np.max(e_xx_n)))
print("     -> eyy_n (m,M) %e %e " %(np.min(e_yy_n),np.max(e_yy_n)))
print("     -> exy_n (m,M) %e %e " %(np.min(e_xy_n),np.max(e_xy_n)))
print("     -> q (m,M) %e %e " %(np.min(q),np.max(q)))

if debug:
   np.savetxt('pressure.ascii',np.array([x_V,y_V,q]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_V,y_V,e_xx_n,e_yy_n,e_xy_n]).T,header='# x,y,exx,eyy,exy')

print("compute press & strain: %.3f s" % (clock.time()-start))

###############################################################################
# compute analytical displacement and stress on nodes
###############################################################################

uth=np.zeros(nn_V,dtype=np.float64)  
vth=np.zeros(nn_V,dtype=np.float64)  
sigmaxxth=np.zeros(nn_V,dtype=np.float64) 
sigmayyth=np.zeros(nn_V,dtype=np.float64) 
sigmaxyth=np.zeros(nn_V,dtype=np.float64) 
 
for i in range(0,nn_V):
    uth[i],vth[i]=analytical_displacement(x_V[i],y_V[i],nu,radius,L)
    sigmaxxth[i],sigmayyth[i],sigmaxyth[i]=analytical_stress(x_V[i],y_V[i],mu,nu,radius,L)

disp=np.sqrt(u**2+v**2)
dispth=np.sqrt(uth**2+vth**2)
r=np.sqrt(x_V**2+y_V**2)
theta=np.arctan2(y_V,x_V)

###############################################################################
# export measurements on tunnel and at surface 
###############################################################################

factor_disp=epsilon*radius**2
factor_stress=2*mu*epsilon*radius**2

np.savetxt('top_solution.ascii',np.array([x_V[top],\
                                          u[top],\
                                          v[top],\
                                          sigma_xx_n[top],\
                                          sigma_yy_n[top],\
                                          sigma_xy_n[top]]).T,\
                                header='# x,u,v,xx,yy,xy',fmt='%.4e')

np.savetxt('top_analytical_solution.ascii',np.array([x_V[top],\
                                                     dispth[top]/factor_disp,\
                                                     uth[top]/factor_disp,\
                                                     vth[top]/factor_disp,\
                                                     sigmaxxth[top]/factor_stress,\
                                                     sigmayyth[top]/factor_stress,\
                                                     sigmaxyth[top]/factor_stress]).T,\
                                           header='# x,disp,u,v,xx,yy,xy',fmt='%.4e')


np.savetxt('circle_solution.ascii',np.array([theta[circle],\
                                             disp[circle],\
                                             u[circle],\
                                             v[circle],\
                                             sigma_xx_n[circle],\
                                             sigma_yy_n[circle],\
                                             sigma_xy_n[circle]]).T,\
                                   header='# x,disp,u,v,xx,yy,xy',fmt='%.4e')

np.savetxt('circle_analytical_solution.ascii',np.array([theta[circle],\
                                                        dispth[circle]/factor_disp,\
                                                        uth[circle]/factor_disp,\
                                                        vth[circle]/factor_disp,\
                                                        sigmaxxth[circle]/factor_stress,\
                                                        sigmayyth[circle]/factor_stress,\
                                                        sigmaxyth[circle]/factor_stress]).T,\
                                              header='# theta,disp,u,v,xx,yy,xy',fmt='%.4e')


###############################################################################
# plot of solution
###############################################################################
       
vtufile=open('solution.vtu',"w")
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
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")

#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(uth[i],vth[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='radius/r' Format='ascii'> \n")
rr=radius/r
rr.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
q.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
e_xx_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
e_yy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
e_xy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
e_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='circle' Format='ascii'> \n")
circle.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='left' Format='ascii'> \n")
left.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='right' Format='ascii'> \n")
right.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='bottom' Format='ascii'> \n")
bottom.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='top' Format='ascii'> \n")
top.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")

#--
vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
sigma_xx_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
sigma_yy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
sigma_xy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma' Format='ascii'> \n")
sigma_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")


vtufile.write("<DataArray type='Float32' Name='sigma_xx (th)' Format='ascii'> \n")
sigmaxxth.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_yy (th)' Format='ascii'> \n")
sigmayyth.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_xy (th)' Format='ascii'> \n")
sigmaxyth.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")


#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
r_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
theta_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
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

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################






