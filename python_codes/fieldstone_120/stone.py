import FEbasis2D as FE
import FEquadrature as Q
import FEtools as Tools 
import numpy as np
import time as timing
import matplotlib.pyplot as plt
import sys 
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

Lx=1
Ly=1

nelx=20

ndofV=2
ndofP=1

Vspace='P2'
Pspace='P1'

visu=1

experiment='solvi'

unstructured=1
meshtype='generic'

isoparametric=True
randomize_mesh=False

#--------------------------------------------------------------------
# allowing for argument parsing through command line
#--------------------------------------------------------------------

if int(len(sys.argv) == 6):
   print("arguments:",sys.argv)
   nelx = int(sys.argv[1])
   Vspace = sys.argv[2]
   Pspace = sys.argv[3]
   experiment = sys.argv[4]  
   unstructured = int(sys.argv[5])
   #visu=0

nely=nelx

unstructured=(unstructured==1) 

if experiment=='dh'          : import mms_dh as mms
if experiment=='jolm17'      : import mms_jolm17 as mms
if experiment=='sinker'      : import mms_sinker as mms
if experiment=='sinker_open' : import mms_sinker_open as mms
if experiment=='poiseuille'  : import mms_poiseuille as mms
if experiment=='johnbook'    : import mms_johnbook as mms
if experiment=='bocg12'      : import mms_bocg12 as mms
if experiment=='solcx'       : import mms_solcx as mms
if experiment=='solkz'       : import mms_solkz as mms
if experiment=='solvi'       : import mms_solvi as mms


if experiment=='solvi'       : meshtype='solvi'

# if quadrilateral nqpts is nqperdim
# if triangle nqpts is total nb of qpoints 

nqpts=Q.nqpts_default(Vspace)

#--------------------------------------------------------------------
# mesh: node layout and connectivity
#--------------------------------------------------------------------
start = timing.time()

mV=FE.NNN_m(Vspace)
mP=FE.NNN_m(Pspace)

nqel,qcoords_r,qcoords_s,qweights=Q.quadrature(Vspace,nqpts)

if not unstructured:
   NV,nel,xV,yV,iconV=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Vspace)
   NP,nel,xP,yP,iconP=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Pspace)
else:
   nel,NV,NP,xV,yV,iconV,xP,yP,iconP=Tools.read_mesh(Vspace,Pspace,nelx,meshtype)

nq=nqel*nel
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP

print("*****************************")
print("           daSTONE           ")
print("*****************************")
print ('Vspace       =',Vspace)
print ('Pspace       =',Pspace)
print ('space1       =',FE.mapping(Vspace))
print ('nqpts        =',nqpts)
print ('nqel         =',nqel)
print ('nelx         =',nelx)
print ('nely         =',nely)
print ('NV           =',NV)
print ('NP           =',NP)
print ('nel          =',nel)
print ('NfemV        =',NfemV)
print ('NfemP        =',NfemP)
print ('Nfem         =',Nfem)
print ('experiment   =',experiment)
print ('unstructured =',unstructured)
print("*****************************")

print("mesh setup: %.3f s" % (timing.time() - start))

#--------------------------------------------------------------------
# create arrays containing analytical solution
#--------------------------------------------------------------------
start = timing.time()

uth = np.zeros(NV,dtype=np.float64)
vth = np.zeros(NV,dtype=np.float64)
pth = np.zeros(NP,dtype=np.float64)

for i in range(NV):        
    uth[i]=mms.u_th(xV[i],yV[i])
    vth[i]=mms.v_th(xV[i],yV[i])

for i in range(NP):        
    pth[i]=mms.p_th(xP[i],yP[i])

print("analytical solution: %.3f s" % (timing.time() - start))

#--------------------------------------------------------------------
# boundary conditions setup 
#--------------------------------------------------------------------
start = timing.time()

bc_fix,bc_val=Tools.bc_setup(xV,yV,uth,vth,Lx,Ly,ndofV,mms.left_bc,mms.right_bc,mms.bottom_bc,mms.top_bc)

print("bc setup: %.3f s" % (timing.time() - start))

#--------------------------------------------------------------------
# build Q1 or P1 background mesh
# all variables ending with '1' (nel1, icon1, x1, y1, ...) are those pertaining 
# to the background mesh of either Q1 or P1 elements used for mapping.
#--------------------------------------------------------------------

space1=FE.mapping(Vspace)
m1=FE.NNN_m(space1)

N1,nel1,x1,y1,icon1=Tools.cartesian_mesh(Lx,Ly,nelx,nely,space1)

if randomize_mesh:
   hx=Lx/nelx
   hy=Ly/nely
   Tools.randomize_background_mesh(x1,y1,hx,hy,N1,Lx,Ly)
   Tools.adapt_FE_mesh(x1,y1,icon1,m1,space1,xV,yV,iconV,nel,Vspace)
   Tools.adapt_FE_mesh(x1,y1,icon1,m1,space1,xP,yP,iconP,nel,Pspace)

#--------------------------------------------------------------------
# compute area of elements 
# This can be a good test because it uses the quadrature points and 
# weights as well as the shape functions (if non-isoparametric
# mapping is used). If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE matrix building process is carried out. 
#--------------------------------------------------------------------
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nqel):
        rq=qcoords_r[iq]
        sq=qcoords_s[iq]
        weightq=qweights[iq]
        NNNV=FE.NNN(rq,sq,Vspace)
        xq=NNNV.dot(xV[iconV[:,iel]]) 
        yq=NNNV.dot(yV[iconV[:,iel]]) 
        dNNNVdr=FE.dNNNdr(rq,sq,Vspace)
        dNNNVds=FE.dNNNds(rq,sq,Vspace)
        if isoparametric:
           jcob,jcbi=Tools.J(mV,dNNNVdr,dNNNVds,xV[iconV[0:mV,iel]],yV[iconV[0:mV,iel]])
        else:
           dNNN1dr=FE.dNNNdr(rq,sq,space1)
           dNNN1ds=FE.dNNNds(rq,sq,space1)
           jcob,jcbi=Tools.J(m1,dNNN1dr,dNNN1ds,x1[icon1[0:m1,iel]],y1[icon1[0:m1,iel]])
        area[iel]+=jcob*weightq
    #end for
#end for

if np.abs(area.sum()-Lx*Ly)>1e-8: exit("pb with area calculations")

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.8e " %(area.sum()))
print("     -> total area anal %.8e " %(Lx*Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

#--------------------------------------------------------------------
# build FE matrix
# |K   G|.|V|=|f|
# |G^T 0| |P| |h|
#--------------------------------------------------------------------
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
rhs = np.zeros(Nfem,dtype=np.float64) 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64)
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) 
    
xq = np.zeros(nq,dtype=np.float64)
yq = np.zeros(nq,dtype=np.float64)
uq = np.zeros(nq,dtype=np.float64)
vq = np.zeros(nq,dtype=np.float64)
pq = np.zeros(nq,dtype=np.float64)
etaq = np.zeros(nq,dtype=np.float64)
    
dNNNVdx= np.zeros(mV,dtype=np.float64)
dNNNVdy= np.zeros(mV,dtype=np.float64)

counterq=0

for iel in range(0,nel): # loop over elements

    K_el = np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el = np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    f_el = np.zeros(mV*ndofV,dtype=np.float64)
    h_el = np.zeros(mP*ndofP,dtype=np.float64)

    for iq in range(0,nqel): # loop over quadrature points inside element

        rq=qcoords_r[iq]
        sq=qcoords_s[iq]
        weightq=qweights[iq]

        NNNV=FE.NNN(rq,sq,Vspace)
        dNNNVdr=FE.dNNNdr(rq,sq,Vspace)
        dNNNVds=FE.dNNNds(rq,sq,Vspace)

        if isoparametric:
           xq[counterq]=NNNV.dot(xV[iconV[0:mV,iel]])
           yq[counterq]=NNNV.dot(yV[iconV[0:mV,iel]])
           jcob,jcbi=Tools.J(mV,dNNNVdr,dNNNVds,xV[iconV[0:mV,iel]],yV[iconV[0:mV,iel]])
        else:
           NNN1=FE.NNN(rq,sq,space1)
           dNNN1dr=FE.dNNNdr(rq,sq,space1)
           dNNN1ds=FE.dNNNds(rq,sq,space1)
           xq[counterq]=NNN1.dot(x1[icon1[0:m1,iel]])
           yq[counterq]=NNN1.dot(y1[icon1[0:m1,iel]])
           jcob,jcbi=Tools.J(m1,dNNN1dr,dNNN1ds,x1[icon1[0:m1,iel]],y1[icon1[0:m1,iel]])

        dNNNVdx[:]=jcbi[0,0]*dNNNVdr[:]+jcbi[0,1]*dNNNVds[:]
        dNNNVdy[:]=jcbi[1,0]*dNNNVdr[:]+jcbi[1,1]*dNNNVds[:]

        NNNP=FE.NNN(rq,sq,Pspace,xxP=xP[iconP[:,iel]],yyP=yP[iconP[:,iel]],xxq=xq[counterq],yyq=yq[counterq])

        for k in range(0,mV): 
            b_mat[0:3,2*k:2*k+2] = [[dNNNVdx[k],0.        ],  
                                    [0.        ,dNNNVdy[k]],
                                    [dNNNVdy[k],dNNNVdx[k]]]

        etaq[counterq]=mms.eta(xq[counterq],yq[counterq])

        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counterq]*weightq*jcob

        #print(xq[counterq],yq[counterq],mms.eta(xq[counterq],yq[counterq]))

        for k in range(0,mV): 
            f_el[2*k+0]+=NNNV[k]*jcob*weightq*mms.bx(xq[counterq],yq[counterq])
            f_el[2*k+1]+=NNNV[k]*jcob*weightq*mms.by(xq[counterq],yq[counterq])

        for k in range(0,mP):
            N_mat[0,k]=NNNP[k]
            N_mat[1,k]=NNNP[k]
            N_mat[2,k]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        counterq+=1

    #end for iq

    # apply bc
    Tools.apply_bc(K_el,G_el,f_el,h_el,bc_val,bc_fix,iconV,mV,ndofV,iel)

    # assemble
    Tools.assemble_K(K_el,A_sparse,iconV,mV,ndofV,iel)
    Tools.assemble_G(G_el,A_sparse,iconV,iconP,NfemV,mV,mP,ndofV,ndofP,iel)
    Tools.assemble_f(f_el,rhs,iconV,mV,ndofV,iel)
    Tools.assemble_h(h_el,rhs,iconP,mP,NfemV,iel)

#end for iel

print("build FE matrix: %.3f s" % (timing.time() - start))

#plt.spy(A_sparse,markersize=1)
#plt.savefig('matrix_'+Vspace+'_'+Pspace+'.pdf', bbox_inches='tight')

#------------------------------------------------------------------------------
# solve system
#------------------------------------------------------------------------------
start = timing.time()

matrix=A_sparse.tocsr()
sol=spsolve(matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# put solution into separate x,y velocity arrays
#------------------------------------------------------------------------------
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

print("split vel into u,v: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# normalise pressure 
#------------------------------------------------------------------------------
start = timing.time()

if mms.pnormalise and Pspace=='Q1+Q0':
   # normalise Q1 pressure (this will alter the Q0 
   # pressures too but these get normalised anyways after)

   avrg_p_q1=0
   for iel in range(0,nel):
       avrg_p_q1+=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])/4*area[iel]
   print('avrg_p_q1=',avrg_p_q1)
   for i in range(0,NP):
       p[i]-=avrg_p_q1

   #normalise Q0 pressure
   avrg_p_q0=0
   for iel in range(0,nel):
       avrg_p_q0+=p[iconP[4,iel]]*area[iel]
   print('avrg_p_q0=',avrg_p_q0)
   for iel in range(0,nel):
       p[iconP[4,iel]]-=avrg_p_q0

if mms.pnormalise and Pspace=='P1+P0':
   # normalise P1 pressure (this will alter the P0 
   # pressures too but these get normalised anyways after)

   avrg_p_p1=0
   for iel in range(0,nel):
       avrg_p_p1+=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3*area[iel]
   print('avrg_p_p1=',avrg_p_p1)
   for i in range(0,NP):
       p[i]-=avrg_p_p1

   #normalise P0 pressure
   avrg_p_p0=0
   for iel in range(0,nel):
       avrg_p_p0+=p[iconP[3,iel]]*area[iel]
   print('avrg_p_p0=',avrg_p_p0)
   for iel in range(0,nel):
       p[iconP[3,iel]]-=avrg_p_p0

if mms.pnormalise:
   avrg_p=0
   counterq=0
   for iel in range(0,nel):
       for iq in range(0,nqel):
           rq=qcoords_r[iq]
           sq=qcoords_s[iq]
           weightq=qweights[iq]
           if isoparametric:
              dNNNVdr=FE.dNNNdr(rq,sq,Vspace)
              dNNNVds=FE.dNNNds(rq,sq,Vspace)
              jcob,jcbi=Tools.J(mV,dNNNVdr,dNNNVds,xV[iconV[0:mV,iel]],yV[iconV[0:mV,iel]])
           else:
              dNNN1dr=FE.dNNNdr(rq,sq,space1)
              dNNN1ds=FE.dNNNds(rq,sq,space1)
              jcob,jcbi=Tools.J(m1,dNNN1dr,dNNN1ds,x1[icon1[0:m1,iel]],y1[icon1[0:m1,iel]])
           NNNP=FE.NNN(rq,sq,Pspace,xxP=xP[iconP[:,iel]],yyP=yP[iconP[:,iel]],xxq=xq[counterq],yyq=yq[counterq])
           avrg_p+=NNNP.dot(p[iconP[0:mP,iel]])*jcob*weightq
           counterq+=1
       #end if
   #end if

   print('     -> avrg_p=',avrg_p)

   p-=avrg_p/Lx/Ly

   print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
            
   print("pressure normalisation: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# compute h 
#------------------------------------------------------------------------------

if Vspace[0]=='Q':
   hmin=min(np.sqrt(area))
   hmax=max(np.sqrt(area))
   havrg=sum(np.sqrt(area))/nel
else:
   hmin=min(np.sqrt(2*area))
   hmax=max(np.sqrt(2*area))
   havrg=sum(np.sqrt(2*area))/nel

print('     -> h (m,M,avrg)=',hmin,hmax,havrg)

#------------------------------------------------------------------------------
# compute vrms and errors
#------------------------------------------------------------------------------
start = timing.time()

errdivv=0
errv=0
errp=0
vrms=0
counterq=0
for iel in range(0,nel):
    for iq in range(0,nqel):
        rq=qcoords_r[iq]
        sq=qcoords_s[iq]
        weightq=qweights[iq]
        NNNV=FE.NNN(rq,sq,Vspace)
        dNNNVdr=FE.dNNNdr(rq,sq,Vspace)
        dNNNVds=FE.dNNNds(rq,sq,Vspace)
        NNNP=FE.NNN(rq,sq,Pspace,xxP=xP[iconP[:,iel]],yyP=yP[iconP[:,iel]],xxq=xq[counterq],yyq=yq[counterq])
        if isoparametric:
           jcob,jcbi=Tools.J(mV,dNNNVdr,dNNNVds,xV[iconV[0:mV,iel]],yV[iconV[0:mV,iel]])
        else:
           dNNN1dr=FE.dNNNdr(rq,sq,space1)
           dNNN1ds=FE.dNNNds(rq,sq,space1)
           jcob,jcbi=Tools.J(m1,dNNN1dr,dNNN1ds,x1[icon1[0:m1,iel]],y1[icon1[0:m1,iel]])
        dNNNVdx[:]=jcbi[0,0]*dNNNVdr[:]+jcbi[0,1]*dNNNVds[:]
        dNNNVdy[:]=jcbi[1,0]*dNNNVdr[:]+jcbi[1,1]*dNNNVds[:]
        uq[counterq]=NNNV.dot(u[iconV[0:mV,iel]])
        vq[counterq]=NNNV.dot(v[iconV[0:mV,iel]])
        pq[counterq]=NNNP.dot(p[iconP[0:mP,iel]])
        vrms+=(uq[counterq]**2+vq[counterq]**2)*weightq*jcob
        errv+=(uq[counterq]-mms.u_th(xq[counterq],yq[counterq]))**2*weightq*jcob+\
              (vq[counterq]-mms.v_th(xq[counterq],yq[counterq]))**2*weightq*jcob
        errp+=(pq[counterq]-mms.p_th(xq[counterq],yq[counterq]))**2*weightq*jcob
        exxq=dNNNVdx.dot(u[iconV[0:mV,iel]])
        eyyq=dNNNVdy.dot(v[iconV[0:mV,iel]])
        divvq=exxq+eyyq
        errdivv+=divvq**2*weightq*jcob
        counterq+=1
    #end for iq
#end for iq

vrms=np.sqrt(vrms/(Lx*Ly))
errv=np.sqrt(errv/(Lx*Ly))
errp=np.sqrt(errp/(Lx*Ly))
errdivv=np.sqrt(errdivv/(Lx*Ly))

print("     -> nel= %6d ; vrms= %.8e | vrms_th= %.8e | %7d %7d %e" %(nel,vrms,mms.vrms_th(),NfemV,NfemP,hmin))
print("     -> nel= %6d ; errv= %.8e ; errp= %.8e ; errdivv= %.8e | %7d %7d %.8e" %(nel,errv,errp,errdivv,NfemV,NfemP,hmin))

print("compute vrms & errors: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------

if visu:

   Tools.export_elements_to_vtu(xV,yV,iconV,Vspace,'meshV.vtu',area)
   Tools.export_elements_to_vtu(xP,yP,iconP,Pspace,'meshP.vtu',area)
   if not isoparametric: Tools.export_elements_to_vtu(x1,y1,icon1,space1,'mesh1.vtu')

   Tools.export_swarm_to_ascii(xV,yV,'Vnodes.ascii')
   Tools.export_swarm_to_ascii(xP,yP,'Pnodes.ascii')
   Tools.export_swarm_to_vtu(xV,yV,'Vnodes.vtu')
   Tools.export_swarm_to_vtu(xP,yP,'Pnodes.vtu')

   Tools.export_swarm_to_ascii(xq,yq,'qpts.ascii')
   Tools.export_swarm_to_vtu(xq,yq,'qpts.vtu')
   Tools.export_swarm_vector_to_vtu(xq,yq,uq,vq,'qpts_vel.vtu')
   Tools.export_swarm_scalar_to_vtu(xq,yq,pq,'qpts_p.vtu')
   Tools.export_swarm_scalar_to_vtu(xq,yq,etaq,'qpts_eta.vtu')
   Tools.export_swarm_vector_to_ascii(xq,yq,uq,vq,'qpts_vel.ascii')
   Tools.export_swarm_scalar_to_ascii(xq,yq,pq,'qpts_p.ascii')
   Tools.export_swarm_scalar_to_ascii(xq,yq,etaq,'qpts_eta.ascii')

   Tools.export_swarm_vector_to_vtu(xV,yV,u,v,'solution_velocity.vtu')
   Tools.export_swarm_vector_to_vtu(xV,yV,uth,vth,'velocity_analytical.vtu')
   Tools.export_swarm_scalar_to_vtu(xP,yP,p,'solution_pressure.vtu')
   Tools.export_swarm_scalar_to_vtu(xP,yP,pth,'pressure_analytical.vtu')

   Tools.export_swarm_vector_to_ascii(xV,yV,u,v,'solution_velocity.ascii')
   Tools.export_swarm_vector_to_ascii(xV,yV,uth,vth,'velocity_analytical.ascii')
   Tools.export_swarm_scalar_to_ascii(xP,yP,p,'solution_pressure.ascii')
   Tools.export_swarm_scalar_to_ascii(xP,yP,pth,'pressure_analytical.ascii')

   Tools.export_connectivity_array_to_ascii(xV,yV,iconV,'iconV.ascii')
   Tools.export_connectivity_array_to_ascii(xP,yP,iconP,'iconP.ascii')

print("*****************************")
print("*****************************")
