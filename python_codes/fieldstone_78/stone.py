import numpy as np
import sys as sys
import time as timing
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg.dsolve import linsolve
import scipy.sparse as sps
import letallec
import stenberg
import regular
import macro_A
import macro_B
import macro_QZ1 
import macro_QZ2
import macro_QZ3

###############################################################################

def bx(x, y):
    if experiment==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if experiment==2:
       val=0.
    if experiment==3:
       val=0.
    if experiment==4:
       val=0.
    return val

def by(x, y):
    if experiment==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if experiment==2:
       if np.abs(x-Lx/2)<0.125 and np.abs(y-Ly/2)<0.125:
          val=-1.02
       else:
          val= -1
    if experiment==3:
       if (x-Lx/2)**2+(y-Ly/2)**2<0.125**2:
          val=-1
       else:
          val=0
    if experiment==4:
       val=-1
    return val

def viscosity(x,y):
    if experiment==1:
       val=1
    if experiment==2:
       val=1
    if experiment==3:
       if (x-Lx/2)**2+(y-Ly/2)**2<0.125**2:
          val=1e4
       else:
          val=1
    if experiment==4:
       val=1
    return val

###############################################################################

def velocity_x(x,y):
    if experiment==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if experiment==2:
       val=0.
    if experiment==3:
       val=0.
    if experiment==4:
       val=0
    return val

def velocity_y(x,y):
    if experiment==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if experiment==2:
       val=0.
    if experiment==3:
       val=0.
    if experiment==4:
       val=0
    return val

def pressure(x,y):
    if experiment==1:
       val=x*(1.-x)-1./6.
    if experiment==2:
       val=0.
    if experiment==3:
       val=0.
    if experiment==4:
       val=0.5-y
    return val

###############################################################################

def NNV(r,s):
    NV_0= 0.25*(1-r)*(1-s) 
    NV_1= 0.25*(1+r)*(1-s) 
    NV_2= 0.25*(1+r)*(1+s) 
    NV_3= 0.25*(1-r)*(1+s) 
    return NV_0,NV_1,NV_2,NV_3

def dNNVdr(r,s):
    dNVdr_0=-0.25*(1.-s) 
    dNVdr_1=+0.25*(1.-s) 
    dNVdr_2=+0.25*(1.+s) 
    dNVdr_3=-0.25*(1.+s) 
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3

def dNNVds(r,s):
    dNVds_0=-0.25*(1.-r) 
    dNVds_1=-0.25*(1.+r) 
    dNVds_2=+0.25*(1.+r) 
    dNVds_3=+0.25*(1.-r) 
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3

###############################################################################

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   topo = int(sys.argv[4])
else:
   nelx = 3
   nely = 3
   visu = 1
   topo = 2

pnormalise=False # using int p dV=0 constrain

#exp1: mms
#exp2: block
#exp3: sphere
#exp4: aquarium
#exp5: solkz TODO

experiment=1

if topo==0: #regular
   NV=(nelx+1)*(nely+1)
   nel=nelx*nely

if topo==1: #stenberg  (S)
   NV=nely*(5*nelx+2)+2*nelx+1
   nel=5*nelx*nely

if topo==2: #le tallec (LT)
   NV=(2*nelx+1)*(2*nely+1)+nely*nelx*8
   nel=12*nelx*nely

if topo==3: # qizh07 (QZ1)
   nel=nelx*nely*12
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +9*nelx*nely

if topo==4: # qizh07 (QZ2)
   nel=nelx*nely*8
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +5*nelx*nely

if topo==5: # qizh07 (QZ3)
   nel=nelx*nely*6
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +3*nelx*nely

if topo==6: # mine (A)
   nel=nelx*nely*7
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +4*nelx*nely

if topo==7: # mine (B)
   nel=nelx*nely*5
   NV=(nelx+1)*(nely+1)+4*nelx*nely


NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

print('nelx=',nelx)
print('nely=',nely)
print('NV=',NV)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('topo=',topo)

eps=1.e-10

nqperdim=2

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

rVnodes=[-1,+1,+1,-1]
sVnodes=[-1,-1,+1,+1]

###############################################################################
# computing nodes coordinates and their connectivity
###############################################################################

if topo==0:
   xV,yV,iconV=regular.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==1:
   xV,yV,iconV=stenberg.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==2:
   xV,yV,iconV=letallec.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==3:
   xV,yV,iconV=macro_QZ1.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==4:
   xV,yV,iconV=macro_QZ2.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==5:
   xV,yV,iconV=macro_QZ3.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==6:
   xV,yV,iconV=macro_A.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==7:
   xV,yV,iconV=macro_B.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

###############################################################################
# compute coordinates of center of elements
###############################################################################

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]=0.25*np.sum(xV[iconV[:,iel]])
    yc[iel]=0.25*np.sum(yV[iconV[:,iel]])

###############################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
###############################################################################
start = timing.time()

area    =np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)  # shape functions V
dNNNVdr = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)  # shape functions derivatives

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            area[iel]+=jcob*weightq
        if area[iel]<0: 
           for k in range(0,mV):
               print (xV[iconV[k,iel]],yV[iconV[k,iel]])
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.6f " %(area.sum()))
print("     -> total area anal %.6f " %(Lx*Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0,NV):
    if xV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = timing.time()

if pnormalise:
   A_mat = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)# matrix A 
   rhs   = np.zeros((Nfem+1),dtype=np.float64)         # right hand side 
else:
   A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)# matrix A 
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side 

b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
dNNNVdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
u       = np.zeros(NV,dtype=np.float64)            # x-component velocity
v       = np.zeros(NV,dtype=np.float64)            # y-component velocity
p       = np.zeros(nel,dtype=np.float64)           # pressure field 
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0, mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.        ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xc[iel],yc[iel])*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)
                G_el[ndofV*i  ,0]-=dNNNVdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNNNVdy[i]*jcob*weightq

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    A_mat[m1,m2]+=K_el[ikk,jkk]
            rhs[m1]+=f_el[ikk]
            A_mat[m1,NfemV+iel]+=G_el[ikk,0]
            A_mat[NfemV+iel,m1]+=G_el[ikk,0]
    rhs[NfemV+iel]+=h_el[0]

    if pnormalise:
       A_mat[Nfem,NfemV+iel]=area[iel]
       A_mat[NfemV+iel,Nfem]=area[iel]

#end for iel

#pinning pressure on last element
for i in range(0,Nfem):
    A_mat[Nfem-1,i]=0
A_mat[Nfem-1,Nfem-1]=1
rhs[Nfem-1]=0    

print("build FE matrix: %.3f s" % (timing.time() - start))

###############################################################################
# solve system
###############################################################################
start = timing.time()

A_mat=A_mat.tocsr()

sol=sps.linalg.spsolve(A_mat,rhs)

print("solve time: %.3f s" % (timing.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.6f %.6f nel= %d" %(np.min(u),np.max(u),nel))
print("     -> v (m,M) %.6f %.6f nel= %d" %(np.min(v),np.max(v),nel))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (timing.time() - start))

###############################################################################
###############################################################################
start = timing.time()

avrg_p=np.sum(p[:]*area[:])/Lx/Ly

print('avrg p=',avrg_p)

p[:]-=avrg_p

#np.savetxt('p.ascii',np.array([xc,yc,p]).T)

print("     -> p (m,M) %.6f %.6f nel= %d" %(np.min(p),np.max(p),nel))

print("normalise pressure: %.3f s" % (timing.time() - start))

###############################################################################
# compute nodal pressure
###############################################################################
start = timing.time()

q=np.zeros(NV,dtype=np.float64)  
count = np.zeros(NV,dtype=np.int32) 

for iel in range(0,nel):
    for k in range(0,mV):
        inode=iconV[k,iel]
        q[inode]+=p[iel]
        count[inode]+=1

q/=count

#np.savetxt('q.ascii',np.array([xV,yV,q]).T)

print("     -> q (m,M) %.6f %.6f nel= %d" %(np.min(q),np.max(q),nel))

print("compute nodal pressure: %.3f s" % (timing.time() - start))

###############################################################################
# compute error
###############################################################################
start = timing.time()

error_u = np.zeros(NV,dtype=np.float64)
error_v = np.zeros(NV,dtype=np.float64)
error_q = np.zeros(NV,dtype=np.float64)
error_p = np.zeros(nel,dtype=np.float64)

if True: 

   for i in range(0,NV): 
       error_u[i]=u[i]-velocity_x(xV[i],yV[i])
       error_v[i]=v[i]-velocity_y(xV[i],yV[i])
       error_q[i]=q[i]-pressure(xV[i],yV[i])

   for i in range(0,nel): 
       error_p[i]=p[i]-pressure(xc[i],yc[i])

   errv=0.
   errp=0.
   errq=0.
   vrms=0.
   for iel in range (0,nel):
       for iq in range(0,nqperdim):
           for jq in range(0,nqperdim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV[0:mV]=NNV(rq,sq)
               dNNNVdr[0:mV]=dNNVdr(rq,sq)
               dNNNVds[0:mV]=dNNVds(rq,sq)
               jcb=np.zeros((2,2),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               jcob = np.linalg.det(jcb)
               xq=0.0
               yq=0.0
               uq=0.0
               vq=0.0
               qq=0.0
               for k in range(0,mV):
                   xq+=NNNV[k]*xV[iconV[k,iel]]
                   yq+=NNNV[k]*yV[iconV[k,iel]]
                   uq+=NNNV[k]*u[iconV[k,iel]]
                   vq+=NNNV[k]*v[iconV[k,iel]]
                   qq+=NNNV[k]*q[iconV[k,iel]]
               errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
               errp+=(p[iel]-pressure(xq,yq))**2*weightq*jcob
               errq+=(qq-pressure(xq,yq))**2*weightq*jcob
               vrms+=(uq**2+vq**2)*weightq*jcob
           #end for
       #end for
   #end for
   errv=np.sqrt(errv)
   errp=np.sqrt(errp)
   errq=np.sqrt(errq)
   vrms=np.sqrt(vrms/(Lx*Ly))

   print("     -> nel= %6d ; errv= %.8f ; errp= %.8f ; errq= %.8f" %(nel,errv,errp,errq))
   print("     -> nel= %6d ; vrms= %.8f " %(nel,vrms))

print("compute errors: %.3f s" % (timing.time() - start))

###############################################################################
# plot of solution
###############################################################################

if visu:
   vtufile=open('solution.vtu',"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")

   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" %(p[iel]))
   vtufile.write("</DataArray>\n")
   if experiment==1 or experiment==4:
      vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
      for iel in range(0,nel):
          vtufile.write("%10e \n" %(error_p[iel]))
      vtufile.write("</DataArray>\n")
   if experiment==3:
      vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
      for iel in range(0,nel):
           vtufile.write("%10e \n" %(viscosity(xc[iel],yc[iel])))
      vtufile.write("</DataArray>\n")


   vtufile.write("<DataArray type='Float32' Name='bx' Format='ascii'> \n")
   for iel in range(0,nel):
           vtufile.write("%10e \n" %(bx(xc[iel],yc[iel])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='by' Format='ascii'> \n")
   for iel in range(0,nel):
           vtufile.write("%10e \n" %(by(xc[iel],yc[iel])))
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
   for iel in range(0,nel):
           vtufile.write("%10e \n" %(area[iel]))
   vtufile.write("</DataArray>\n")

   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e  \n" %(q[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")

   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel]))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
