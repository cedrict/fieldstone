import numpy as np
import sys as sys
import time as clock 
import solcx as solcx
import scipy.sparse as sps

###############################################################################

def viscosity(x,y):
    if x<0.5:
       val=1.
    else:
       val=1.e6
    return val

def density(x,y):
    val=np.sin(np.pi*y)*np.cos(np.pi*x)
    return val

###############################################################################

print("-----------------------------")
print("--------- stone 05 ----------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

gx=0
gy=1

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 50 
   nely = 50
   visu = 1
    
nnx=nelx+1     # number of nodes, x direction
nny=nely+1     # number of nodes, y direction
NV=nnx*nny     # number of velocity nodes
nel=nelx*nely  # number of elements, total
Nfem=NV*ndof   # Total number of degrees of freedom

penalty=1.e10  # penalty coefficient value

eps=1.e-10
sqrt3=np.sqrt(3.)

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x=np.empty(NV,dtype=np.float64) # x coordinates
y=np.empty(NV,dtype=np.float64) # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup node coordinates: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon =np.zeros((m,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1

# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#     print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

print("setup connectivity array: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if x[i]<eps:
       bc_fix[i*ndof  ]=True ; bc_val[i*ndof  ]=0.
    if x[i]>(Lx-eps):
       bc_fix[i*ndof  ]=True ; bc_val[i*ndof  ]=0.
    if y[i]<eps:
       bc_fix[i*ndof+1]=True ; bc_val[i*ndof+1]=0.
    if y[i]>(Ly-eps):
       bc_fix[i*ndof+1]=True ; bc_val[i*ndof+1]=0.
#end for

print("define b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

A_fem = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
b_fem = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
jcb   = np.zeros((2,2),dtype=np.float64)

k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros((m*ndof),dtype=np.float64)
    a_el=np.zeros((m*ndof,m*ndof),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

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
            jcb[0,0]=dNdr[:].dot(x[icon[:,iel]])
            jcb[0,1]=dNdr[:].dot(y[icon[:,iel]])
            jcb[1,0]=dNds[:].dot(x[icon[:,iel]])
            jcb[1,1]=dNds[:].dot(y[icon[:,iel]])

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            xq=N[:].dot(x[icon[:,iel]])
            yq=N[:].dot(y[icon[:,iel]])

            # compute dNdx & dNdy
            dNdx=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
            dNdy=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

            # construct 3x8 b_mat matrix
            for i in range(0,m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]
            #end for

            # compute elemental matrix
            a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq)*wq*jcob

            # compute elemental rhs vector
            for i in range(0,m):
                b_el[2*i  ]+=N[i]*jcob*wq*density(xq,yq)*gx
                b_el[2*i+1]+=N[i]*jcob*wq*density(xq,yq)*gy
            #end for
            
        #end for
    #end for

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    wq=2.*2.

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    # compute the jacobian
    jcb[0,0]=dNdr[:].dot(x[icon[:,iel]])
    jcb[0,1]=dNdr[:].dot(y[icon[:,iel]])
    jcb[1,0]=dNds[:].dot(x[icon[:,iel]])
    jcb[1,1]=dNds[:].dot(y[icon[:,iel]])

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    # compute dNdx and dNdy
    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

    # compute gradient matrix
    for i in range(0,m):
        b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                              [0.     ,dNdy[i]],
                              [dNdy[i],dNdx[i]]]
    #end for

    # compute elemental matrix
    a_el+=b_mat.T.dot(k_mat.dot(b_mat))*penalty*wq*jcob

    # assemble matrix and right hand side
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon[k2,iel]+i2
                    A_fem[m1,m2]+=a_el[ikk,jkk]
                #end for
            #end for
            b_fem[m1]+=b_el[ikk]
        #end for
    #end for

#end for

print("build matrix: %.3f s" % (clock.time()-start))

###############################################################################
# impose boundary conditions
###############################################################################
start=clock.time()

for i in range(0, Nfem):
    if bc_fix[i]:
       A_ref=A_fem[i,i]
       for j in range(0,Nfem):
           b_fem[j]-= A_fem[i,j]*bc_val[i]
           A_fem[i,j]=0.
           A_fem[j,i]=0.
           A_fem[i,i]=A_ref
       #end for
       b_fem[i]=A_ref*bc_val[i]
    #end if
#end for

#print("A_fem (m,M) = %.4f %.4f" %(np.min(A_fem),np.max(A_fem)))
#print("b_fem (m,M) = %.6f %.6f" %(np.min(b_fem),np.max(b_fem)))

print("impose b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol = sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(NV,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("reshape sol. vector: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure
###############################################################################
start=clock.time()

p = np.zeros(nel,dtype=np.float64)  
e = np.zeros(nel,dtype=np.float64)  
xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
eta = np.zeros(nel,dtype=np.float64)  
rho = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    wq = 2.0 * 2.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb[0,0]=dNdr[:].dot(x[icon[:,iel]])
    jcb[0,1]=dNdr[:].dot(y[icon[:,iel]])
    jcb[1,0]=dNds[:].dot(x[icon[:,iel]])
    jcb[1,1]=dNds[:].dot(y[icon[:,iel]])
    jcbi=np.linalg.inv(jcb)

    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

    xc[iel]=N[:].dot(x[icon[:,iel]])
    yc[iel]=N[:].dot(y[icon[:,iel]])

    exx[iel]=dNdx[:].dot(u[icon[:,iel]])
    eyy[iel]=dNdy[:].dot(v[icon[:,iel]])
    exy[iel]=0.5*dNdy[:].dot(u[icon[:,iel]])+0.5*dNdx[:].dot(v[icon[:,iel]])

    p[iel]=-penalty*(exx[iel]+eyy[iel])
    eta[iel]=viscosity(xc[iel],yc[iel])
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
    rho[iel]=density(xc[iel],yc[iel])

#end for

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("     -> sr (m,M) %.6f %.6f " %(np.min(e),np.max(e)))

#np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute sr & press: %.3f s" % (clock.time()-start))

###############################################################################
# pressure normalisation
###############################################################################

#p[:]-=np.sum(p)/nel

#print("p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

###############################################################################
# compute error
###############################################################################
start=clock.time()

error_u = np.empty(NV,dtype=np.float64)
error_v = np.empty(NV,dtype=np.float64)
error_p = np.empty(nel,dtype=np.float64)

for i in range(0,NV):
    ui,vi,pi=solcx.SolCxSolution(x[i],y[i]) 
    error_u[i]=u[i]-ui
    error_v[i]=v[i]-vi
#end for

for i in range(0,nel): 
    ui,vi,pi=solcx.SolCxSolution(xc[i],yc[i]) 
    error_p[i]=p[i]-pi
#end for

#np.savetxt('error_pressure.ascii',np.array([xc,yc,error_p]).T,header='# xc,yc,p')

errv=0.
errp=0.
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
            jcb[0,0]=dNdr[:].dot(x[icon[:,iel]])
            jcb[0,1]=dNdr[:].dot(y[icon[:,iel]])
            jcb[1,0]=dNds[:].dot(x[icon[:,iel]])
            jcb[1,1]=dNds[:].dot(y[icon[:,iel]])
            jcob=np.linalg.det(jcb)
            xq=N[:].dot(x[icon[:,iel]])
            yq=N[:].dot(y[icon[:,iel]])
            uq=N[:].dot(u[icon[:,iel]])
            vq=N[:].dot(v[icon[:,iel]])
            ui,vi,pi=solcx.SolCxSolution(xq,yq) 
            errv+=((uq-ui)**2+(vq-vi)**2)*wq*jcob
            errp+=(p[iel]-pi)**2*wq*jcob
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute discr. errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate xx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain rate yy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain rate xy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='strain rate (effective)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (p[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='pressure (error)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (error_p[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eta[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (rho[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(error_u[i],error_v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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

print("export to vtu: %.3f s" % (clock.time()-start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
