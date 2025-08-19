import math
import numpy as np
import time as clock 
import triangle as tr
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def NNV(r,s):
    return np.array([1-r-s,r,s],dtype=np.float64)

def dNNVdr(r,s):
    return np.array([-1,1,0],dtype=np.float64)

def dNNVds(r,s):
    return np.array([-1,0,1],dtype=np.float64)

###############################################################################

E=1e5   # Young's modulus
nu=0.25 # poisson ratio
laambda=nu*E/(1+nu)/(1-2*nu)
mu=E/2/(1+nu)

###############################################################################
# analytical solution
###############################################################################

alpha=0.544483737 
omega=3*np.pi/4 
c1=-np.cos((alpha+1)*omega)/np.cos((alpha-1)*omega)
c2=2*(laambda+2*mu)/(laambda+mu) 

def displ_th(x,y):
    r=np.sqrt(x**2+y**2)
    theta=theta=math.atan2(y,x)
    ralpha=r**alpha/(2*mu)
    ur=ralpha*(-(alpha+1)*np.cos((alpha+1)*theta)+(c2-alpha-1)*c1*np.cos((alpha-1)*theta));
    ut=ralpha*( (alpha+1)*np.sin((alpha+1)*theta)+(c2+alpha-1)*c1*np.sin((alpha-1)*theta));
    uth=ur*np.cos(theta)-ut*np.sin(theta)
    vth=ur*np.sin(theta)+ut*np.cos(theta) #CHECK?
    return uth,vth

###############################################################################
# Triangle Area is calculated via Heron's formula, see wikipedia
###############################################################################

def compute_triangles_area(coords,nodesArray):
    tx = coords[:,0]
    ty = coords[:,1]
    a = np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,1]])**2)
    b = np.sqrt((tx[nodesArray[:,2]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,2]]-ty[nodesArray[:,1]])**2)
    c = np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,2]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,2]])**2)
    area = 0.5 * np.sqrt(a**2 * c**2 - (( a**2 + c**2 - b**2) / 2)**2)
    area = area.reshape(-1,1) #Transposing the 1xN matrix into Nx1 shape
    return area

###############################################################################
# main parameters
###############################################################################

print("-----------------------------")
print("---------- stone 179---------")
print("-----------------------------")

ndof=2
nqel=3

visu=1

method=0

debug=True

triangle_instructions='pqa0.05'

qcoords_r=[1./6.,1./6.,2./3.] # coordinates & weights 
qcoords_s=[2./3.,1./6.,1./6.] # of quadrature points
qweights =[1./6.,1./6.,1./6.]

###############################################################################
# build mesh
###############################################################################
start=clock.time()

pts = np.array([[-1,-1],[0,-2],[2,0],[0,2],[-1,1],[0,0]])
segments = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
O1 = {'vertices' : pts, 'segments' : segments}
T1 = tr.triangulate(O1,triangle_instructions) # tr.triangulate() computes the main dictionary 
tr.compare(plt, O1, T1) # The tr.compare() function always takes plt as its 1st argument
area=compute_triangles_area(T1['vertices'], T1['triangles'])
icon=T1['triangles'] #; icon=icon.T
x=T1['vertices'][:,0]
y=T1['vertices'][:,1] 

NV=np.size(x)
nel,m=np.shape(icon)
Nfem=NV*ndof

#print(np.sum(area))

print('m=',m)
print('NV=',NV)
print('nel=',nel)
print("-----------------------------")

print("setup: build mesh: %.3f s" % (clock.time()-start))

###############################################################################
# flag boundary nodes
###############################################################################
start=clock.time()

eps=0.0001

on_boundary=np.zeros(Nfem,dtype=bool) 

for i in range(NV):
    if abs(y[i]-(x[i]-2))<eps: on_boundary[i]=True
    if abs(y[i]-(-x[i]+2))<eps: on_boundary[i]=True
    if abs(y[i]-x[i])<eps: on_boundary[i]=True
    if abs(y[i]+x[i])<eps: on_boundary[i]=True
    if abs(y[i]+(x[i]+2))<eps: on_boundary[i]=True
    if abs(y[i]+(-x[i]-2))<eps: on_boundary[i]=True

print("setup: flag boundary nodes: %.3f s" % (clock.time()-start))
 
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if method==0:
   for i in range(0,NV):
       if on_boundary[i]: 
          ui,vi=displ_th(x[i],y[i])
          bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 1#ui
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0#vi

else:
   for i in range(0,NV):
       if on_boundary[i]: 
          ui,vi=displ_th(x[i],y[i])
          bc_fix[i   ]=True ; bc_val[i   ]=1#ui
          bc_fix[i+NV]=True ; bc_val[i+NV]=0#vi

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

#################################################################
# compute area of elements
#################################################################
start=clock.time()

area2=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV=NNV(rq,sq)
        dNNNVdr=dNNVdr(rq,sq)
        dNNNVds=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,m):
            jcb[0,0]+=dNNNVdr[k]*x[icon[iel,k]]
            jcb[0,1]+=dNNNVdr[k]*y[icon[iel,k]]
            jcb[1,0]+=dNNNVds[k]*x[icon[iel,k]]
            jcb[1,1]+=dNNNVds[k]*y[icon[iel,k]]
        #end for
        jcob = np.linalg.det(jcb)
        area2[iel]+=jcob*weightq
    #end for
#end for

print("     -> total area (meas) %.6f " %(area2.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()
    
A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)
Ael=np.zeros((ndof*m,ndof*m),dtype=np.float64)
bel=np.zeros(ndof*m,dtype=np.float64)
dNNNVdx=np.zeros(m,dtype=np.float64)  
dNNNVdy=np.zeros(m,dtype=np.float64)  
b_mat=np.zeros((3,ndof*m),dtype=np.float64) 
c_mat=np.array([[2*mu+laambda,laambda,0],[laambda,2*mu+laambda,0],[0,0,mu]],dtype=np.float64) 

for iel,nodes in enumerate(icon):
    #print(iel,nodes)

    if method==0:

       Ael=np.zeros((m*ndof,m*ndof),dtype=np.float64)
       bel=np.zeros(m*ndof)

       for kq in range (0,nqel):
           rq=qcoords_r[kq]
           sq=qcoords_s[kq]
           weightq=qweights[kq]
           NNNV=NNV(rq,sq)
           dNNNVdr=dNNVdr(rq,sq)
           dNNNVds=dNNVds(rq,sq)
           jcb=np.zeros((2,2),dtype=np.float64)
           for k in range(0,m):
               jcb[0,0]+=dNNNVdr[k]*x[nodes[k]]
               jcb[0,1]+=dNNNVdr[k]*y[nodes[k]]
               jcb[1,0]+=dNNNVds[k]*x[nodes[k]]
               jcb[1,1]+=dNNNVds[k]*y[nodes[k]]
           #end for
           jcob=np.linalg.det(jcb)
           jcbi=np.linalg.inv(jcb)

           dNNNVdx[:]=jcbi[0,0]*dNNNVdr[:]+jcbi[0,1]*dNNNVds[:]
           dNNNVdy[:]=jcbi[1,0]*dNNNVdr[:]+jcbi[1,1]*dNNNVds[:]

           for i in range(0, m):
               b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                        [0.        ,dNNNVdy[i]],
                                        [dNNNVdy[i],dNNNVdx[i]]]
           #end for

           Ael+=b_mat.T.dot(c_mat.dot(b_mat))*weightq*jcob

           # compute elemental rhs vector
           #for i in range(0, m):
           #    f_el[ndof*i  ]+=NNNV[i]*jcob*weightq*gx*rho
           #    f_el[ndof*i+1]+=NNNV[i]*jcob*weightq*gy*rho
           #end for

       #end for kq

       #print(Ael[0,0],Ael[2,2],Ael[4,4])
       #print(Ael[1,1],Ael[3,3],Ael[5,5])
       #print(Ael[1,4],Ael[3,4],Ael[5,4])

       # impose dirichlet b.c. 
       for k1 in range(0,m):
           for i1 in range(0,ndof):
               ikk=ndof*k1          +i1
               m1 =ndof*icon[iel,k1]+i1
               if bc_fix[m1]:
                  Aref=Ael[ikk,ikk] 
                  for jkk in range(0,m*ndof):
                      bel[jkk]-=Ael[jkk,ikk]*bc_val[m1]
                      Ael[ikk,jkk]=0
                      Ael[jkk,ikk]=0
                  Ael[ikk,ikk]=Aref
                  bel[ikk]=Aref*bc_val[m1]
               #end if
           #end for 
       #end for

       # assemble matrix a_mat and right hand side rhs
       for k1 in range(0,m):
           for i1 in range(0,ndof):
               ikk=ndof*k1          +i1
               m1 =ndof*icon[iel,k1]+i1
               for k2 in range(0,m):
                   for i2 in range(0,ndof):
                       jkk=ndof*k2          +i2
                       m2 =ndof*icon[iel,k2]+i2
                       A_fem[m1,m2]+=Ael[ikk,jkk]
                   #end for
               #end for
               b_fem[m1]+=bel[ikk]
           #end for
       #end for

    else:

       xvect=np.array([x[nodes[2]]-x[nodes[1]],\
                       x[nodes[0]]-x[nodes[2]],\
                       x[nodes[1]]-x[nodes[0]]],dtype=np.float64)
       yvect=np.array([y[nodes[1]]-y[nodes[2]],\
                       y[nodes[2]]-y[nodes[0]],\
                       y[nodes[0]]-y[nodes[1]]],dtype=np.float64)

       ll=laambda/(4*area[iel,0])
       mm=mu/(4*area[iel,0])

       Kxx=(ll+2*mm)*np.outer(yvect,yvect) + mm*np.outer(xvect,xvect)
       Kyy=(ll+2*mm)*np.outer(xvect,xvect) + mm*np.outer(yvect,yvect)
       Kxy=       ll*np.outer(yvect,xvect) + mm*np.outer(xvect,yvect)

       #print(Kxx[0,0],Kxx[1,1],Kxx[2,2])
       #print(Kxy[0,0],Kxy[1,1],Kxy[2,2])
       #print(Kxy[0,0],Kxy[0,1],Kxy[0,2])

       Ael[  0:m,0:m]=Kxx   ; Ael[  0:m,m:2*m]=Kxy
       Ael[m:2*m,0:m]=Kxy.T ; Ael[m:2*m,m:2*m]=Kyy

       #print(Ael[3,2],Ael[4,2],Ael[5,2])

       # impose dirichlet b.c. 
       for k1 in range(0,m):
           for i1 in range(0,ndof):
               ikk=k1          +i1*m
               m1=icon[iel,k1]+i1*NV
               if bc_fix[m1]:
                  A_ref=Ael[ikk,ikk] 
                  for jkk in range(0,m*ndof):
                      bel[jkk]-=Ael[jkk,ikk]*bc_val[m1]
                      Ael[ikk,jkk]=0
                      Ael[jkk,ikk]=0
                  Ael[ikk,ikk]=A_ref
                  bel[ikk]=A_ref*bc_val[m1]
               #end if
           #end for 
       #end for

       # assemble matrix a_mat and right hand side rhs
       for k1 in range(0,m):
           for i1 in range(0,ndof):
               ikk=k1         +i1*m
               m1=icon[iel,k1]+i1*NV
               for k2 in range(0,m):
                   for i2 in range(0,ndof):
                       jkk=k2         +i2*m
                       m2=icon[iel,k2]+i2*NV
                       A_fem[m1,m2]+=Ael[ikk,jkk]
                   #end for
               #end for
               b_fem[m1]+=bel[ikk]
           #end for
       #end for

   #end if method

#end for

print("build matrix: %.3f s" % (clock.time()-start))

if False:
   plt.spy(sps.csr_matrix(A_fem),markersize=1)
   plt.savefig('matrix.pdf', bbox_inches='tight')

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

if method==0:
   u,v=np.reshape(sol,(NV,2)).T
else:
   u=sol[0:NV]
   v=sol[NV:2*NV]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

if debug:
   np.savetxt('displacement.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split solution: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure and compute elemental strain
###############################################################################
start=clock.time()

#u[:]=x[:]**2
#v[:]=y[:]**2

e=np.zeros(nel,dtype=np.float64)  
p=np.zeros(nel,dtype=np.float64)   
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
divv=np.zeros(nel,dtype=np.float64)  

for iel,nodes in enumerate(icon):

    dNNNVdx[0]=y[nodes[1]]-y[nodes[2]]
    dNNNVdx[1]=y[nodes[2]]-y[nodes[0]]
    dNNNVdx[2]=y[nodes[0]]-y[nodes[1]]

    dNNNVdy[0]=x[nodes[2]]-x[nodes[1]]
    dNNNVdy[1]=x[nodes[0]]-x[nodes[2]]
    dNNNVdy[2]=x[nodes[1]]-x[nodes[0]]

    xc[iel]=(x[nodes[0]]+x[nodes[1]]+x[nodes[2]])/3
    yc[iel]=(y[nodes[0]]+y[nodes[1]]+y[nodes[2]])/3

    exx[iel]=dNNNVdx[:].dot(u[nodes[:]])
    eyy[iel]=dNNNVdy[:].dot(v[nodes[:]])
    exy[iel]=0.5*dNNNVdy[:].dot(u[nodes[:]])\
            +0.5*dNNNVdx[:].dot(v[nodes[:]])
#end for

divv[:]=exx[:]+eyy[:]
e[:]=np.sqrt(0.5*(exx[:]*exx[:]+eyy[:]*eyy[:])+exy[:]*exy[:])
p[:]=-(laambda+mu)*(exx[:]+eyy[:])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
   np.savetxt('strain.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time() - start))

###############################################################################
# compute elemental stress 
###############################################################################
start=clock.time()

sigma_xx = np.zeros(nel,dtype=np.float64)  
sigma_yy = np.zeros(nel,dtype=np.float64)  
sigma_xy = np.zeros(nel,dtype=np.float64)  

sigma_xx[:]=laambda*divv[:]+2*mu*exx[:]
sigma_yy[:]=laambda*divv[:]+2*mu*eyy[:]
sigma_xy[:]=               +2*mu*exy[:]

print("compute stress: %.3f s" % (clock.time() - start))

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

    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel,0]))
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
    vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_xx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_yy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigma_xy[iel]))
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" %divv[iel]) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")

    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displ.' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displ. (analytical)' Format='ascii'> \n")
    for i in range(0,NV):
        ui,vi=displ_th(x[i],y[i])
        vtufile.write("%10e %10e %10e \n" %(ui,vi,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displ. (error)' Format='ascii'> \n")
    for i in range(0,NV):
        ui,vi=displ_th(x[i],y[i])
        vtufile.write("%10e %10e %10e \n" %(u[i]-ui,v[i]-vi,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='on_boundary' Format='ascii'> \n")
    for i in range(0,NV):
        if on_boundary[i]:
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
        vtufile.write("%d %d %d \n" %(icon[iel,0],icon[iel,1],icon[iel,2]))
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

    print("export to vtu: %.3f s" % (clock.time()-start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
