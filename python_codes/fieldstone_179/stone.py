import sys 
import math
import numpy as np
import time as clock 
import triangle as tr
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def basis_functions_V(r,s):
    return np.array([1-r-s,r,s],dtype=np.float64)

def basis_functions_V_dr(r,s):
    return np.array([-1,1,0],dtype=np.float64)

def basis_functions_V_ds(r,s):
    return np.array([-1,0,1],dtype=np.float64)

###############################################################################
# elastic parameters
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
# Triangle area is calculated via Heron's formula, see wikipedia
###############################################################################

def compute_triangles_area(coords,nodesArray):
    tx=coords[:,0]
    ty=coords[:,1]
    a=np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,1]])**2+(ty[nodesArray[:,0]]-ty[nodesArray[:,1]])**2)
    b=np.sqrt((tx[nodesArray[:,2]]-tx[nodesArray[:,1]])**2+(ty[nodesArray[:,2]]-ty[nodesArray[:,1]])**2)
    c=np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,2]])**2+(ty[nodesArray[:,0]]-ty[nodesArray[:,2]])**2)
    area=0.5*np.sqrt(a**2 * c**2 - (( a**2 + c**2 - b**2) / 2)**2)
    area=area.reshape(-1,1) #Transposing the 1xN matrix into Nx1 shape
    return area

###############################################################################
# main parameters
###############################################################################

print("*******************************")
print("********** stone 179 **********")
print("*******************************")

ndof_V=2
nqel=3

visu=1

if int(len(sys.argv) == 3):
   method = int(sys.argv[1])
   sizet = sys.argv[2]
else:
   method= 1
   sizet='0.005'

debug=False

triangle_instructions='pqa'+sizet  

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
icon_V=T1['triangles'] #; icon=icon.T
x_V=T1['vertices'][:,0]
y_V=T1['vertices'][:,1] 

nn_V=np.size(x_V)
nel,m_V=np.shape(icon_V)
Nfem=nn_V*ndof_V

if debug: print(np.sum(area))

print("setup: build mesh: %.3f s | %d " % (clock.time()-start,Nfem))

###############################################################################

print('m_V=',m_V)
print('nn_V=',nn_V)
print('nel=',nel)
print('Nfem=',Nfem)
print('method=',method)
print(triangle_instructions)
print("-----------------------------")

###############################################################################
# flag boundary nodes
###############################################################################
start=clock.time()

eps=0.0001

on_boundary=np.zeros(Nfem,dtype=bool) 

for i in range(nn_V):
    if abs(y_V[i]-(x_V[i]-2))<eps: on_boundary[i]=True
    if abs(y_V[i]-(-x_V[i]+2))<eps: on_boundary[i]=True
    if abs(y_V[i]-x_V[i])<eps: on_boundary[i]=True
    if abs(y_V[i]+x_V[i])<eps: on_boundary[i]=True
    if abs(y_V[i]+(x_V[i]+2))<eps: on_boundary[i]=True
    if abs(y_V[i]+(-x_V[i]-2))<eps: on_boundary[i]=True

print("setup: flag boundary nodes: %.3f s" % (clock.time()-start))
 
###############################################################################
# boundary conditions setup
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if method==0:
   for i in range(0,nn_V):
       if on_boundary[i]: 
          ui,vi=displ_th(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=ui
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vi

else:
   for i in range(0,nn_V):
       if on_boundary[i]: 
          ui,vi=displ_th(x_V[i],y_V[i])
          bc_fix_V[i     ]=True ; bc_val_V[i     ]=ui
          bc_fix_V[i+nn_V]=True ; bc_val_V[i+nn_V]=vi

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

#################################################################
# compute area of elements
#################################################################
start=clock.time()

jcb=np.zeros((2,2),dtype=np.float64)
area2=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[iel,:]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[iel,:]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[iel,:]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[iel,:]])
        JxWq=np.linalg.det(jcb)*weightq
        area2[iel]+=JxWq
    #end for
#end for

print("     -> total area (meas) %.6f " %(area2.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()
    
#A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)
B=np.zeros((3,ndof_V*m_V),dtype=np.float64) 
dofs=np.zeros(ndof_V*m_V,dtype=np.int32) 
C=np.array([[2*mu+laambda,     laambda, 0],\
            [     laambda,2*mu+laambda, 0],\
            [           0,           0,mu]],dtype=np.float64) 

time_A_el=0
   
row=[] 
col=[]
A_fem=[]

for iel,nodes in enumerate(icon_V):

    if method==0:
       start2=clock.time()

       A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
       b_el=np.zeros(m_V*ndof_V,dtype=np.float64)

       for k in range(0,m_V):
           dofs[k*ndof_V  ]=icon_V[iel,k]*ndof_V
           dofs[k*ndof_V+1]=icon_V[iel,k]*ndof_V+1

       for kq in range (0,nqel):
           rq=qcoords_r[kq]
           sq=qcoords_s[kq]
           weightq=qweights[kq]
           N_V=basis_functions_V(rq,sq)
           dNdr_V=basis_functions_V_dr(rq,sq)
           dNds_V=basis_functions_V_ds(rq,sq)
           jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[iel,:]])
           jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[iel,:]])
           jcb[1,0]=np.dot(dNds_V,x_V[icon_V[iel,:]])
           jcb[1,1]=np.dot(dNds_V,y_V[icon_V[iel,:]])
           jcbi=np.linalg.inv(jcb)
           JxWq=np.linalg.det(jcb)*weightq
           dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
           dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

           for i in range(0,m_V):
               B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                 [0.       ,dNdy_V[i]],
                                 [dNdy_V[i],dNdx_V[i]]]

           A_el+=B.T.dot(C.dot(B))*JxWq

           # compute elemental rhs vector
           #for i in range(0,m_V):
           #    b_el[ndof_V*i  ]+=N_V[i]*gx*rho*JxWq
           #    b_el[ndof_V*i+1]+=N_V[i]*gy*rho*JxWq
           #end for

       #end for kq

       time_A_el+=clock.time()-start2

       # impose dirichlet b.c. 
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k1          +i1
               m1 =ndof_V*icon_V[iel,k1]+i1
               if bc_fix_V[m1]:
                  Aref=A_el[ikk,ikk] 
                  for jkk in range(0,m_V*ndof_V):
                      b_el[jkk]-=A_el[jkk,ikk]*bc_val_V[m1]
                      A_el[ikk,jkk]=0
                      A_el[jkk,ikk]=0
                  A_el[ikk,ikk]=Aref
                  b_el[ikk]=Aref*bc_val_V[m1]
               #end if
           #end for 
       #end for

       #if debug: print(Ael[0,0],Ael[0,1],Ael[0,2])
       #if debug: print(Ael[0,3],Ael[0,4],Ael[0,5])
       #if debug: print(bel[0],bel[2],bel[4])

       for i_local,idof in enumerate(dofs):
           for j_local,jdof in enumerate(dofs):
               row.append(idof)
               col.append(jdof)
               A_fem.append(A_el[i_local,j_local])
           #end for
           b_fem[idof]+=b_el[i_local]

    else:
       start2=clock.time()

       A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
       b_el=np.zeros(m_V*ndof_V)

       for k in range(0,m_V):
           dofs[k    ]=icon_V[iel,k]
           dofs[k+m_V]=icon_V[iel,k]+nn_V

       xvect=np.array([x_V[nodes[2]]-x_V[nodes[1]],\
                       x_V[nodes[0]]-x_V[nodes[2]],\
                       x_V[nodes[1]]-x_V[nodes[0]]],dtype=np.float64)
       yvect=np.array([y_V[nodes[1]]-y_V[nodes[2]],\
                       y_V[nodes[2]]-y_V[nodes[0]],\
                       y_V[nodes[0]]-y_V[nodes[1]]],dtype=np.float64)

       ll=laambda/(4*area[iel,0])
       mm=mu/(4*area[iel,0])

       Kxx=(ll+2*mm)*np.outer(yvect,yvect) + mm*np.outer(xvect,xvect)
       Kyy=(ll+2*mm)*np.outer(xvect,xvect) + mm*np.outer(yvect,yvect)
       Kxy=       ll*np.outer(yvect,xvect) + mm*np.outer(xvect,yvect)

       A_el[  0:m_V  ,0:m_V]=Kxx   ; A_el[    0:m_V,m_V:2*m_V]=Kxy
       A_el[m_V:2*m_V,0:m_V]=Kxy.T ; A_el[m_V:2*m_V,m_V:2*m_V]=Kyy

       time_A_el+=clock.time()-start2

       # impose dirichlet b.c. 
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=k1          +i1*m_V
               m1=icon_V[iel,k1]+i1*nn_V
               if bc_fix_V[m1]:
                  A_ref=A_el[ikk,ikk] 
                  for jkk in range(0,m_V*ndof_V):
                      b_el[jkk]-=A_el[jkk,ikk]*bc_val_V[m1]
                      A_el[ikk,jkk]=0
                      A_el[jkk,ikk]=0
                  A_el[ikk,ikk]=A_ref
                  b_el[ikk]=A_ref*bc_val_V[m1]
               #end if
           #end for 
       #end for

       #if debug: print(Ael[0,0],Ael[0,3],Ael[0,1])
       #if debug: print(Ael[0,4],Ael[0,2],Ael[0,5])
       #if debug: print(bel[0],bel[1],bel[2])

       for i_local,idof in enumerate(dofs):
           for j_local,jdof in enumerate(dofs):
               row.append(idof)
               col.append(jdof)
               A_fem.append(A_el[i_local,j_local])
           #end for
           b_fem[idof]+=b_el[i_local]
       #end for

   #end if method

#end for

A_fem=sps.csr_matrix((A_fem,(row,col)),shape=(Nfem,Nfem))
       
print("     -> A_el: %.5f s | Nfem= %d" % (time_A_el,Nfem))

print("Build matrix: %.5f s | Nfem= %d" % (clock.time()-start,Nfem))

if False:
   plt.clf()
   plt.spy(sps.csr_matrix(A_fem),markersize=1)
   plt.savefig('matrix.pdf', bbox_inches='tight')

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(A_fem,b_fem)

print("Solve time: %.5f s | Nfem= %d" % (clock.time()-start,Nfem))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

if method==0:
   u,v=np.reshape(sol,(nn_V,2)).T
else:
   u=sol[0:nn_V]
   v=sol[nn_V:2*nn_V]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

if debug:
   np.savetxt('displacement.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split solution: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure and compute elemental strain
###############################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
p=np.zeros(nel,dtype=np.float64)   
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
divv=np.zeros(nel,dtype=np.float64)  
dNdx_V=np.zeros(m_V,dtype=np.float64)
dNdy_V=np.zeros(m_V,dtype=np.float64)

for iel,nodes in enumerate(icon_V):
    dNdx_V[0]=y_V[nodes[1]]-y_V[nodes[2]]
    dNdx_V[1]=y_V[nodes[2]]-y_V[nodes[0]]
    dNdx_V[2]=y_V[nodes[0]]-y_V[nodes[1]]
    dNdy_V[0]=x_V[nodes[2]]-x_V[nodes[1]]
    dNdy_V[1]=x_V[nodes[0]]-x_V[nodes[2]]
    dNdy_V[2]=x_V[nodes[1]]-x_V[nodes[0]]
    xc[iel]=(x_V[nodes[0]]+x_V[nodes[1]]+x_V[nodes[2]])/3
    yc[iel]=(y_V[nodes[0]]+y_V[nodes[1]]+y_V[nodes[2]])/3
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[iel,:]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[iel,:]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[iel,:]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[iel,:]])*0.5
#end for

divv[:]=exx[:]+eyy[:]
e[:]=np.sqrt(0.5*(exx[:]*exx[:]+eyy[:]*eyy[:])+exy[:]*exy[:])
p[:]=-(laambda+mu)*(exx[:]+eyy[:])

sigmaxx=laambda*divv+2*mu*exx
sigmayy=laambda*divv+2*mu*eyy
sigmaxy=            +2*mu*exy

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
   np.savetxt('strain.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p, sr & stress: %.3f s" % (clock.time() - start))

###############################################################################
# compute root mean square displacement vrms 
###############################################################################
start=clock.time()

vrms=0.
avrg_u=0.
avrg_v=0.
erru=0.

for iel,nodes in enumerate(icon_V):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        N_V=basis_functions_V(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[iel,:]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[iel,:]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[iel,:]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[iel,:]])
        JxWq=np.linalg.det(jcb)*weightq
        xq=np.dot(N_V,x_V[icon_V[iel,:]])
        yq=np.dot(N_V,y_V[icon_V[iel,:]])
        uq=np.dot(N_V,u[icon_V[iel,:]])
        vq=np.dot(N_V,v[icon_V[iel,:]])
        vrms+=(uq**2+vq**2)*JxWq
        avrg_u+=uq*JxWq
        avrg_v+=vq*JxWq
        uth,vth=displ_th(xq,yq)
        erru+=((uq-uth)**2+(vq-vth)**2)*JxWq
    # end for kq
# end for iel

avrg_u/=np.sum(area)
avrg_v/=np.sum(area)
erru/=np.sum(area)
erru=np.sqrt(erru)

print("     -> vrms   = %.6e | %d" %(vrms,Nfem))
print("     -> avrg_u = %.6e " %(avrg_u))
print("     -> avrg_v = %.6e " %(avrg_v))
print("     -> err displ = %.6e | %d" %(erru,Nfem))

print("compute vrms: %.3fs" % (clock.time()-start))

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
        vtufile.write("%10e\n" % (area[iel,0]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (p[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigmaxx[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigmayy[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (sigmaxy[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" %divv[iel]) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displ.' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displ. (analytical)' Format='ascii'> \n")
    for i in range(0,nn_V):
        ui,vi=displ_th(x_V[i],y_V[i])
        vtufile.write("%10e %10e %10e \n" %(ui,vi,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displ. (error)' Format='ascii'> \n")
    for i in range(0,nn_V):
        ui,vi=displ_th(x_V[i],y_V[i])
        vtufile.write("%10e %10e %10e \n" %(u[i]-ui,v[i]-vi,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='on_boundary' Format='ascii'> \n")
    for i in range(0,nn_V):
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
        vtufile.write("%d %d %d \n" %(icon_V[iel,0],icon_V[iel,1],icon_V[iel,2]))
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

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
