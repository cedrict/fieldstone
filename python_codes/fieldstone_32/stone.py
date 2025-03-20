
import numpy as np
import sys as sys
import scipy.sparse as sps
import random
import time as time
from scipy.sparse import csr_matrix, lil_matrix 

###############################################################################

def rho(x,y,MX,MY):
    val=-np.pi**3*(MX**4+MY**4+2*MX**2*MY**2)/MX*\
         np.cos(MX*np.pi*x)*\
         np.sin(MY*np.pi*y)
    return val

def velocity_x(x,y,MX,MY):
    val=MY*np.pi*np.sin(MX*np.pi*x)*\
              np.cos(MY*np.pi*y)
    return val

def velocity_y(x,y,MX,MY):
    val=-MX*np.pi*np.cos(MX*np.pi*x)*\
                  np.sin(MY*np.pi*y)
    return val

def pressure(x,y,MX,MY):
    val=np.pi**2*(MX**2*MY+MY**3)/MX*\
        np.cos(MX*np.pi*x)*\
        np.cos(MY*np.pi*y)
    return val

###############################################################################
eps=1.e-10
sqrt3=np.sqrt(3.)

print("-----------------------------")
print("--------- stone 32 ----------")
print("-----------------------------")

m=4      # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=2.  # horizontal extent of the domain 
Ly=2.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   MX   = int(sys.argv[4])
   MY   = int(sys.argv[5])
else:
   nelx = 64
   nely = 64
   visu = 1
   MX=2
   MY=1

eta=1
gx=0.
gy=-1.
    
nnx=nelx+1     # number of elements, x direction
nny=nely+1     # number of elements, y direction
nel=nelx*nely  # number of elements, total

NV=nnx*nny  # number of V nodes
NP=nel      # number of P nodes

NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

pnormalise=True

debug=False

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.zeros(NV,dtype=np.float64)  # x coordinates
y=np.zeros(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*Lx/float(nelx)-1.
        y[counter]=j*Ly/float(nely)-1.
        counter += 1
    #end for
#end for

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity array setup
#################################################################
start = time.time()

icon=np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0,counter]= i + j * (nelx + 1)
        icon[1,counter]= i + 1 + j * (nelx + 1)
        icon[2,counter]= i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter]= i + (j + 1) * (nelx + 1)
        counter += 1
    #end for
#end for

if debug:
   for iel in range (0,nel):
       print ("iel=",iel)
       print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
       print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
       print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
       print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions: free slip on all sides
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if x[i]<-1+eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
    #end if
    if x[i]>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
    #end if
    if y[i]<-1+eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    #end if
    if y[i]>(1.-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    #end if
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

K_mat=np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat=np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs=np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs=np.zeros(NfemP,dtype=np.float64)         # right hand side h 

b_mat = np.zeros((3,ndofV*m),dtype=np.float64)  # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
jcb   = np.zeros((2, 2),dtype=np.float64)
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m*ndofV),dtype=np.float64)
    K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
    G_el=np.zeros((m*ndofV,1),dtype=np.float64)
    h_el=0 

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]=dNdr.dot(x[icon[:,iel]])
            jcb[0,1]=dNdr.dot(y[icon[:,iel]])
            jcb[1,0]=dNds.dot(x[icon[:,iel]])
            jcb[1,1]=dNds.dot(y[icon[:,iel]])

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute coords of quad pts 
            xq=N.dot(x[icon[:,iel]])
            yq=N.dot(y[icon[:,iel]])

            # compute dNdx & dNdy
            dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
            dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]
            #end for

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*wq*jcob

            # compute elemental rhs vector
            for i in range(0,m):
                f_el[ndofV*i  ]-=N[i]*jcob*wq*rho(xq,yq,MX,MY)*gx
                f_el[ndofV*i+1]-=N[i]*jcob*wq*rho(xq,yq,MX,MY)*gy
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*wq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*wq
            #end for

        #end for jq
    #end for iq

    # impose b.c. 
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0
            #end if
        #end for
    #end for

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
        #end for
    #end for
    h_rhs[iel]+=h_el

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

if pnormalise:
   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   a_mat[Nfem,NfemV:Nfem]=1
   a_mat[NfemV:Nfem,Nfem]=1
else:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (time.time() - start))

######################################################################
# solve system
######################################################################
start = time.time()

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

######################################################################
# compute strainrate 
######################################################################
start = time.time()

xc  = np.zeros(nel,dtype=np.float64)  
yc  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
sr  = np.zeros(nel,dtype=np.float64)  
jcb = np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    wq = 2.0*2.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb[0,0]=dNdr.dot(x[icon[:,iel]])
    jcb[0,1]=dNdr.dot(y[icon[:,iel]])
    jcb[1,0]=dNds.dot(x[icon[:,iel]])
    jcb[1,1]=dNds.dot(y[icon[:,iel]])

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

    xc[iel]=N.dot(x[icon[:,iel]])
    yc[iel]=N.dot(y[icon[:,iel]])

    exx[iel]=dNdx.dot(u[icon[:,iel]])
    eyy[iel]=dNdy.dot(v[icon[:,iel]])
    exy[iel]=0.5*dNdy.dot(u[icon[:,iel]])\
            +0.5*dNdx.dot(v[icon[:,iel]])

    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

#end for iel

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute nodal pressure
######################################################################

q=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q[icon[0,iel]]+=p[iel] ; count[icon[0,iel]]+=1
    q[icon[1,iel]]+=p[iel] ; count[icon[1,iel]]+=1
    q[icon[2,iel]]+=p[iel] ; count[icon[2,iel]]+=1
    q[icon[3,iel]]+=p[iel] ; count[icon[3,iel]]+=1
#end for
q/=count

if debug: np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

######################################################################
# compute error fields and L2 errors
######################################################################
start = time.time()

error_u=np.zeros(NV,dtype=np.float64)
error_v=np.zeros(NV,dtype=np.float64)
error_q=np.zeros(NV,dtype=np.float64)
error_p=np.zeros(nel,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(x[i],y[i],MX,MY)
    error_v[i]=v[i]-velocity_y(x[i],y[i],MX,MY)
    error_q[i]=q[i]-pressure(x[i],y[i],MX,MY)

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(xc[i],yc[i],MX,MY)

errv=0.
errp=0.
errq=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
            jcb[0,0]=dNdr.dot(x[icon[:,iel]])
            jcb[0,1]=dNdr.dot(y[icon[:,iel]])
            jcb[1,0]=dNds.dot(x[icon[:,iel]])
            jcb[1,1]=dNds.dot(y[icon[:,iel]])
            jcob=np.linalg.det(jcb)
            xq=N.dot(x[icon[:,iel]])
            yq=N.dot(y[icon[:,iel]])
            uq=N.dot(u[icon[:,iel]])
            vq=N.dot(v[icon[:,iel]])
            qq=N.dot(q[icon[:,iel]])
            errv+=((uq-velocity_x(xq,yq,MX,MY))**2+(vq-velocity_y(xq,yq,MX,MY))**2)*wq*jcob
            errp+=(p[iel]-pressure(xq,yq,MX,MY))**2*wq*jcob
            errq+=(qq-pressure(xq,yq,MX,MY))**2*wq*jcob
        #end for jq
    #end for iq
#end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq=np.sqrt(errq)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f ; errq= %.8f " %(nel,errv,errp,errq))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % pressure(xc[iel],yc[iel],MX,MY))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" %error_p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % exx[iel] )
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % eyy[iel] )
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % exy[iel] )
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(velocity_x(x[i],y[i],MX,MY),velocity_y(x[i],y[i],MX,MY),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(error_u[i],error_v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %error_q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" % rho(x[i],y[i],MX,MY))
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
   print("export to vtu | time: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
