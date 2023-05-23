import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import time as timing
from scipy.sparse import lil_matrix
from tools import *
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from basis_functions_numba import *

#------------------------------------------------------------------------------

def viscosity(x,y,Ly,eta_um,eta_c,eta_o,xA,yB,xC,xE,yE,xF,yF,yG,yI):

    if y<yI:
       val=1e21
    elif y<yG:
       val=5e22
    else:
       val=eta_um

    if x>xA and x<xC and y>yB:
       val=eta_c

    if x>xC and x<xE and y>yK:
       val=eta_o
    if x>xE and y>(yF-yE)/(xF-xE)*(x-xE)+yE:
       val=eta_o

    #val=1e22
    return val

#------------------------------------------------------------------------------

def density(x,y,yB,xL,xM,xN,rhod,rhou,rho0):

    val=rho0

    if x>xL and x<xM and y<yB:
       val=rhod

    if x>xN:
       val=rhou

    return val

#------------------------------------------------------------------------------

ndim=2
year=365.25*3600*24
cm=0.01
km=1e3
eps=1e-6

print("-----------------------------")
print("--------- stone 143 ---------")
print("-----------------------------")

CR=False

if CR:
   mV=7     # number of velocity nodes making up an element
else:
   mV=6

mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

eta_ref=1e22 # numerical parameter for FEM

Lx=6000*km
Ly=3000*km

gx=0
gy=-9.81


#------------------------------------------------------------------------------

lr=500*km
lc=2000*km

h=20*km

xA=1000*km   ; yA=Ly
xB=xA        ; yB=2800*km
xC=xA+lc     ; yC=Ly
xD=xB+lc     ; yD=yB
xE=Lx-lr     ; yE=Ly-100*km
xF=Lx-h      ; yF=Ly
xG=0         ; yG=Ly-660*km
xH=Lx        ; yH=Ly-660*km
xI=0         ; yI=350*km
xJ=Lx        ; yJ=350*km
xK=xD        ; yK=yE
xL=xA        ; yL=0
xM=xL+100*km ; yM=0
xN=Lx-100*km ; yN=0

#------------------------------------------------------------------------------

eta_um=1e20
eta_o=1e23
eta_c=1e22

Fu=1e13
Fd=1e13

Au=50e3*Ly
Ad=50e3*yB
rho_ref=3250
rho_u=3200 #rho_ref-Fu/(rho_ref*gy*Au)
rho_d=3400 #rho_ref+Fd/(rho_ref*gy*Ad)

print('rho_ref',rho_ref)
print('rho_u',rho_u)
print('rho_d',rho_d)

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

qcoords_r=np.array([nb1,nb2,nb2,nb4,nb3,nb4],dtype=np.float64)
qcoords_s=np.array([nb2,nb1,nb2,nb3,nb4,nb4],dtype=np.float64)
qweights =np.array([nb5,nb5,nb5,nb6,nb6,nb6],dtype=np.float64)

###############################################################################

if True:

   square_vertices = np.array([[0,0],[0,Ly],[Lx,Ly],[Lx,0]])
   square_edges = compute_segs(square_vertices)
   offset=4

   #segment AB 
   nptsAB=int((yA-yB)/h)
   x1=np.zeros(nptsAB) ; x1[:]=xA
   y1=np.linspace(yA,yB, nptsAB, endpoint=False)
   pointsAB = np.stack([x1,y1],axis = 1)
   segsAB = np.stack([np.arange(nptsAB-1) +offset , np.arange(nptsAB-1) + 1 +offset], axis=1) 
   offset+=nptsAB

   #segment BD
   nptsBD=int((xD-xB)/h)
   x1=np.linspace(xB,xD, nptsBD, endpoint=True)
   y1=np.zeros(nptsBD) ; y1[:]=yB
   pointsBD = np.stack([x1,y1],axis = 1)
   segsBD = np.stack([np.arange(nptsBD-1) +offset , np.arange(nptsBD-1) + 1 +offset], axis=1) 
   offset+=nptsBD

   #segment CD 
   nptsCD=int((yC-yD)/h)+1
   x1=np.zeros(nptsCD) ; x1[:]=xC
   y1=np.linspace(yC,yD, nptsCD, endpoint=False)
   pointsCD = np.stack([x1,y1],axis = 1)
   segsCD = np.stack([np.arange(nptsCD-1) +offset , np.arange(nptsCD-1) + 1 +offset], axis=1) 
   offset+=nptsCD

   #segment KE
   nptsKE=int((xE-xK)/h)
   x1=np.linspace(xK,xE, nptsKE, endpoint=False)
   y1=np.zeros(nptsKE) ; y1[:]=yK
   pointsKE = np.stack([x1,y1],axis = 1)
   segsKE = np.stack([np.arange(nptsKE-1) +offset , np.arange(nptsKE-1) + 1 +offset], axis=1) 
   offset+=nptsKE

   #segment EF
   nptsEF=int((xF-xE)/h)
   x1=np.linspace(xE,xF, nptsEF, endpoint=True)
   y1=np.linspace(yE,yF, nptsEF, endpoint=True)
   pointsEF = np.stack([x1,y1],axis = 1)
   segsEF = np.stack([np.arange(nptsEF-1) +offset , np.arange(nptsEF-1) + 1 +offset], axis=1) 
   offset+=nptsEF

   #segment GH
   #nptsGH=100
   #x1=np.linspace(xG,xH, nptsGH, endpoint=True)
   #y1=np.zeros(nptsGH) ; y1[:]=yG
   #pointsGH = np.stack([x1,y1],axis = 1)
   #segsGH = np.stack([np.arange(nptsGH-1) +offset , np.arange(nptsGH-1) + 1 +offset], axis=1) 
   #offset+=nptsGH

   #segment IJ
   #nptsIJ=100
   #x1=np.linspace(xI,xJ, nptsIJ, endpoint=True)
   #y1=np.zeros(nptsIJ) ; y1[:]=yI
   #pointsIJ = np.stack([x1,y1],axis = 1)
   #segsIJ = np.stack([np.arange(nptsIJ-1) +offset , np.arange(nptsIJ-1) + 1 +offset], axis=1) 
   #offset+=nptsIJ

   #assemble all coordinate arrays
   #points = np.vstack([square_vertices,pointsAB,pointsBD,pointsCD,pointsKE,pointsEF,pointsGH,pointsIJ])
   points = np.vstack([square_vertices,pointsAB,pointsBD,pointsCD,pointsKE,pointsEF])

   #assemble all segments arrays
   #SEGS = np.vstack([square_edges, segsAB, segsBD, segsCD, segsKE, segsEF, segsGH, segsIJ])
   SEGS = np.vstack([square_edges,segsAB,segsCD,segsKE,segsEF,])

   O1 = {'vertices' : points, 'segments' : SEGS}
   T1 = tr.triangulate(O1, 'pqa40000000000') # tr.triangulate() computes the main dictionary 

   area=compute_triangles_area(T1['vertices'], T1['triangles'])
   iconP1=T1['triangles'] ; iconP1=iconP1.T
   xP1=T1['vertices'][:,0] 
   yP1=T1['vertices'][:,1] 
   NP1=np.size(xP1)

   np.savetxt('meshP1.ascii',np.array([xP1,yP1]).T,header='# xV,zV') 

   print('NP1=',NP1)

   export_elements_to_vtu(xP1,yP1,iconP1,'meshP1.vtu',area)

   mP,nel=np.shape(iconP1)
   print('nel=',nel)

   #print(np.shape(xP1))
   #print(np.shape(yP1))
   #print(np.shape(iconP1))

   NV0,xP2,yP2,iconP2=mesh_P1_to_P2(xP1,yP1,iconP1)

   np.savetxt('meshP2.ascii',np.array([xP2,yP2]).T,header='# xV,zV') 
   print('NV0=',NV0)

   export_elements_to_vtuP2(xP2,yP2,iconP2,'meshP2.vtu')

######################################################################
# build coordinates and connectivity arrays
######################################################################
start = timing.time()

NV=NV0
NP=NP1

xV = np.zeros(NV,dtype=np.float64)  
yV = np.zeros(NV,dtype=np.float64)  
xP = np.zeros(NP,dtype=np.float64)  
yP = np.zeros(NP,dtype=np.float64)  
iconV=np.zeros((mV,nel),dtype=np.int32)
iconP=np.zeros((mP,nel),dtype=np.int32)

if not CR:
   xV[:]=xP2[:]
   yV[:]=yP2[:]
   iconV[:,:]=iconP2[:,:]
   xP[:]=xP1[:]
   yP[:]=yP1[:]
   iconP[:,:]=iconP1[:,:]

NfemV=ndofV*NV
NfemP=ndofP*NP
Nfem=NfemV+NfemP

print("build coords,icon arrays: %.3f s" % (timing.time() - start))

######################################################################
# compute element center coordinates
######################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]= (xP1[iconP1[0,iel]]+xP1[iconP1[1,iel]]+xP1[iconP1[2,iel]])/3
    yc[iel]= (yP1[iconP1[0,iel]]+yP1[iconP1[1,iel]]+yP1[iconP1[2,iel]])/3

print("     -> xc (m,M) %.6e %.6e " %(np.min(xc),np.max(xc)))
print("     -> yc (m,M) %.6e %.6e " %(np.min(yc),np.max(yc)))

print("compute element center coords: %.3f s" % (timing.time() - start))

######################################################################
# assign viscosity to elements
######################################################################
start = timing.time()

eta=np.zeros(nel,dtype=np.float64)  
rho=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    eta[iel]=viscosity(xc[iel],yc[iel],Ly,eta_um,eta_c,eta_o,xA,yB,xC,xE,yE,xF,yF,yG,yI)
    rho[iel]=density(xc[iel],yc[iel],yB,xL,xM,xN,rho_d,rho_u,rho_ref)

print("     -> rho (m,M) %.6e %.6e " %(np.min(rho),np.max(rho)))
print("     -> eta (m,M) %.6e %.6e " %(np.min(eta),np.max(eta)))

np.savetxt('viscosity.ascii',np.array([xc,yc,np.log10(eta),rho]).T) 

print("assign density,viscosity: %.3f s" % (timing.time() - start))

######################################################################
# compute area of elements
######################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV=NNV(rq,sq,CR)
        dNNNVdr=dNNVdr(rq,sq,CR)
        dNNNVds=dNNVds(rq,sq,CR)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()/Lx/Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

######################################################################
# define boundary conditions
######################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)
bc_val=np.zeros(NfemV,dtype=np.float64)

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0. 
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0. 
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. 
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. 

print("define bc: %.3f s" % (timing.time() - start))

######################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
######################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
#A_sparse = np.zeros((Nfem,Nfem),dtype=np.float64)
f_rhs = np.zeros(NfemV,dtype=np.float64)            # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)            # right hand side h 
b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)     # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64)     # matrix N 
NNNV = np.zeros(mV,dtype=np.float64)                # shape functions V
NNNP = np.zeros(mP,dtype=np.float64)                # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
u = np.zeros(NV,dtype=np.float64)                   # x-component velocity
v = np.zeros(NV,dtype=np.float64)                   # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64)

for iel in range(0,nel):

    f_el=np.zeros((mV*ndofV),dtype=np.float64)
    K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    for kq in range (0,nqel):

        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        NNNV[0:mV]=NNV(rq,sq,CR)
        dNNNVdr[0:mV]=dNNVdr(rq,sq,CR)
        dNNNVds[0:mV]=dNNVds(rq,sq,CR)

        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        jcbi = np.linalg.inv(jcb)

        #print(jcob)

        xq=0.0
        yq=0.0
        for k in range(0,mV):
            xq+=NNNV[k]*xV[iconV[k,iel]]
            yq+=NNNV[k]*yV[iconV[k,iel]]
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

        #print(xq,yq)

        # construct 3x8 b_mat matrix
        for i in range(0,mV):
            b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                     [0.        ,dNNNVdy[i]],
                                     [dNNNVdy[i],dNNNVdx[i]]]

        # compute elemental a_mat matrix
        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta[iel]*weightq*jcob

        # compute elemental rhs vector
        for i in range(0,mV):
            f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rho[iel]

        NNNP[0:mP]=NNP(rq,sq)

        for i in range(0,mP):
            N_mat[0,i]=NNNP[i]
            N_mat[1,i]=NNNP[i]
            N_mat[2,i]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

    #end for kq

    G_el*=eta_ref/Lx

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
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    A_sparse[m1,m2] += K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]

            f_rhs[m1]+=f_el[ikk]

    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
    
#end for 

print("build FE matrix: %.3f s" % (timing.time() - start))

#print(f_rhs)
#print(h_rhs)

#plt.spy(A_sparse, markersize=0.25)
#plt.savefig('matrix.png', bbox_inches='tight')
#plt.clf()

######################################################################
# solve system
######################################################################
start = timing.time()

rhs = np.zeros(Nfem,dtype=np.float64)
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

sol=sps.linalg.spsolve(sps.csr_matrix(A_sparse),rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u/cm*year),np.max(u/cm*year)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v/cm*year),np.max(v/cm*year)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("split vel into u,v: %.3f s" % (timing.time() - start))

######################################################################
# project pressure onto P2 mesh for plotting
######################################################################
start = timing.time()

q = np.zeros(NV,dtype=np.float64)            # right hand side h 

for iel in range(0,nel):
    q[iconV[0,iel]]=p[iconP[0,iel]]
    q[iconV[1,iel]]=p[iconP[1,iel]]
    q[iconV[2,iel]]=p[iconP[2,iel]]
    q[iconV[3,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    q[iconV[4,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    q[iconV[5,iel]]=(p[iconP[2,iel]]+p[iconP[0,iel]])*0.5

print("project pressure onto V grid: %.3f s" % (timing.time() - start))

#####################################################################
# plot of solution
# the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
#####################################################################
start = timing.time()

if True:
    filename = 'solution.vtu'
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (rho[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eta[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2+1]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")

    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                              iconV[3,iel],iconV[4,iel],iconV[5,iel]))
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

    print("write data: %.3fs" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
