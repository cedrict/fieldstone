import FEbasis2D as FE
import FEquadrature as Q
import FEtools as Tools 
import numpy as np
import time as timing
import matplotlib.pyplot as plt
import sys 
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy import sparse

###############################################################################
###############################################################################

Lx=1
Ly=1

ndofV=2
ndofP=1

nelx=24

mtype=0 # mesh type (how quads are divided into triangles)

Vspace='P2'
Pspace='P1'

visu=1

experiment='dh' #'RTwave' #'tesk12' #'sinker' #'lire19' #'dh'

unstructured=0

isoparametric=True
randomize_mesh=False

etastar=100
drho=0.01

ass_method=2 # assembly method

eta_ref=1

###############################################################################
# allowing for argument parsing through command line
###############################################################################

if int(len(sys.argv) == 6):
   print("arguments:",sys.argv)
   nelx = int(sys.argv[1])
   Vspace = sys.argv[2]
   Pspace = sys.argv[3]
   experiment = sys.argv[4]  
   unstructured = int(sys.argv[5])
   visu=0

if int(len(sys.argv) == 8):
   print("arguments:",sys.argv)
   nelx = int(sys.argv[1])
   Vspace = sys.argv[2]
   Pspace = sys.argv[3]
   experiment = sys.argv[4]  
   unstructured = int(sys.argv[5])
   etastar=float(sys.argv[6]) ; etastar=10.**etastar
   drho=float(sys.argv[7])
   visu=0

if Vspace=='P2' and Pspace=='P-1': mtype=3

nely=nelx

unstructured=(unstructured==1) 

if experiment=='dh'             : import mms_dh as mms
if experiment=='jolm17'         : import mms_jolm17 as mms
if experiment=='sinker'         : import mms_sinker as mms
if experiment=='sinker_reduced' : import mms_sinker_reduced as mms
if experiment=='sinker_open'    : import mms_sinker_open as mms
if experiment=='poiseuille'     : import mms_poiseuille as mms
if experiment=='johnbook'       : import mms_johnbook as mms
if experiment=='bocg12'         : import mms_bocg12 as mms
if experiment=='solcx'          : import mms_solcx as mms
if experiment=='solkz'          : import mms_solkz as mms
if experiment=='solvi'          : import mms_solvi as mms
if experiment=='RTwave'         : import mms_RTwave as mms
if experiment=='jokn16'         : import mms_jokn16 as mms
if experiment=='plin'           : import mms_plin as mms
if experiment=='lire19'         : import mms_lire19 as mms
if experiment=='tesk12'         : import mms_tesk12 as mms

# if quadrilateral nqpts is nqperdim
# if triangle nqpts is total nb of qpoints 

nqpts=Q.nqpts_default(Vspace)

###############################################################################
# mesh: node layout and connectivity
###############################################################################

start = timing.time()

mV=FE.NNN_m(Vspace)
mP=FE.NNN_m(Pspace)

nqel,qcoords_r,qcoords_s,qweights=Q.quadrature(Vspace,nqpts)

if not unstructured:
   NV,nel,xV,yV,iconV=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Vspace,mtype)
   NP,nel,xP,yP,iconP=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Pspace,mtype)
else:
   nel,NV,NP,xV,yV,iconV,xP,yP,iconP=Tools.generate_random_mesh(Lx,nelx,\
                                                 Vspace,Pspace,experiment)  

nq=nqel*nel
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP

print("*****************************")
print('Vspace       =',Vspace)
print('Pspace       =',Pspace)
print('space1       =',FE.mapping(Vspace))
print('nqpts        =',nqpts)
print('nqel         =',nqel)
print('nelx         =',nelx)
print('nely         =',nely)
print('NV           =',NV)
print('NP           =',NP)
print('nel          =',nel)
print('NfemV        =',NfemV)
print('NfemP        =',NfemP)
print('Nfem         =',Nfem)
print('experiment   =',experiment)
print('unstructured =',unstructured)
print('mtype        =',mtype)
print("*****************************")

print("mesh setup: %.3f s" % (timing.time() - start))

###############################################################################
# create arrays containing analytical solution
###############################################################################
start = timing.time()

uth = np.zeros(NV,dtype=np.float64)
vth = np.zeros(NV,dtype=np.float64)
qth = np.zeros(NV,dtype=np.float64)

for i in range(NV):        
    uth[i],vth[i],qth[i]=mms.solution(xV[i],yV[i])

print("analytical solution: %.3f s" % (timing.time() - start))

###############################################################################
# boundary conditions setup 
###############################################################################
start = timing.time()

bc_fix,bc_val=Tools.bc_setup(xV,yV,uth,vth,Lx,Ly,ndofV,mms.left_bc,mms.right_bc,mms.bottom_bc,mms.top_bc)

print("bc setup: %.3f s" % (timing.time() - start))

###############################################################################
# build Q1 or P1 background mesh
# all variables ending with '1' (nel1, icon1, x1, y1, ...) are those pertaining 
# to the background mesh of either Q1 or P1 elements used for mapping.
###############################################################################
start = timing.time()

space1=FE.mapping(Vspace)
m1=FE.NNN_m(space1)

N1,nel1,x1,y1,icon1=Tools.cartesian_mesh(Lx,Ly,nelx,nely,space1,mtype)

if randomize_mesh:
   hx=Lx/nelx
   hy=Ly/nely
   #Tools.randomize_background_mesh(x1,y1,hx,hy,N1,Lx,Ly)
   Tools.adapt_FE_mesh(x1,y1,icon1,m1,space1,xV,yV,iconV,nel,Vspace)
   Tools.adapt_FE_mesh(x1,y1,icon1,m1,space1,xP,yP,iconP,nel,Pspace)

if experiment=='RTwave':
   Tools.deform_mesh_RTwave(x1,y1,N1,Lx,Ly,nelx,nely)
   Tools.adapt_FE_mesh(x1,y1,icon1,m1,space1,xV,yV,iconV,nel,Vspace)
   Tools.adapt_FE_mesh(x1,y1,icon1,m1,space1,xP,yP,iconP,nel,Pspace)

print("make Q1/P1 mesh: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements 
# This can be a good test because it uses the quadrature points and 
# weights as well as the shape functions (if non-isoparametric
# mapping is used). If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE matrix building process is carried out. 
###############################################################################
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

###############################################################################
# compute h 
###############################################################################

if Vspace[0]=='Q':
   hmin=min(np.sqrt(area))
   hmax=max(np.sqrt(area))
   havrg=sum(np.sqrt(area))/nel
else:
   hmin=min(np.sqrt(2*area))
   hmax=max(np.sqrt(2*area))
   havrg=sum(np.sqrt(2*area))/nel

print('     -> h (m,M,avrg)= %e %e %e ' %(hmin,hmax,havrg))

###############################################################################
# compute array for assembly
###############################################################################
start = timing.time()

ndofV_el=mV*ndofV
local_to_globalV=np.zeros((ndofV_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            local_to_globalV[ikk,iel]=m1
    
print("compute local_to_global: %.3fs" % (timing.time() - start))

###############################################################################
# fill I,J arrays
###############################################################################
start = timing.time()

bignb=nel*( (mV*ndofV)**2 + 2*(mV*ndofV*mP) )

I=np.zeros(bignb,dtype=np.int32)
J=np.zeros(bignb,dtype=np.int32)
V=np.zeros(bignb,dtype=np.float64)

counter=0
for iel in range(0,nel):
    for ikk in range(ndofV_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndofV_el):
            m2=local_to_globalV[jkk,iel]
            I[counter]=m1
            J[counter]=m2
            counter+=1
        for jkk in range(0,mP):
            m2 =iconP[jkk,iel]+NfemV
            I[counter]=m1
            J[counter]=m2
            counter+=1
            I[counter]=m2
            J[counter]=m1
            counter+=1

print("fill I,J arrays: %.3fs" % (timing.time() - start))

###############################################################################
# build FE matrix
# |K   G|.|V|=|f|
# |G^T 0| |P| |h|
###############################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
rhs = np.zeros(Nfem,dtype=np.float64) 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64)
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) 
    
xq = np.zeros(nq,dtype=np.float64)
yq = np.zeros(nq,dtype=np.float64)
etaq = np.zeros(nq,dtype=np.float64)
    
dNNNVdx= np.zeros(mV,dtype=np.float64)
dNNNVdy= np.zeros(mV,dtype=np.float64)

counter=0 ; time_ass=0
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

        NNNP=FE.NNN(rq,sq,Pspace,xxP=xP[iconP[:,iel]],yyP=yP[iconP[:,iel]],\
                                 xxq=xq[counterq],yyq=yq[counterq])

        for k in range(0,mV): 
            b_mat[0:3,2*k:2*k+2] = [[dNNNVdx[k],0.        ],  
                                    [0.        ,dNNNVdy[k]],
                                    [dNNNVdy[k],dNNNVdx[k]]]

        etaq[counterq]=mms.eta(xq[counterq],yq[counterq],etastar)

        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counterq]*weightq*jcob

        #print(xq[counterq],yq[counterq],mms.eta(xq[counterq],yq[counterq]))

        for k in range(0,mV): 
            f_el[2*k+0]+=NNNV[k]*jcob*weightq*mms.bx(xq[counterq],yq[counterq],drho)
            f_el[2*k+1]+=NNNV[k]*jcob*weightq*mms.by(xq[counterq],yq[counterq],drho)

        for k in range(0,mP):
            N_mat[0,k]=NNNP[k]
            N_mat[1,k]=NNNP[k]
            N_mat[2,k]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        counterq+=1

    #end for iq

    G_el*=eta_ref/Lx

    # apply bc
    Tools.apply_bc(K_el,G_el,f_el,h_el,bc_val,bc_fix,iconV,mV,ndofV,iel)

    if ass_method==1:
       start1 = timing.time()
       Tools.assemble_K(K_el,A_sparse,iconV,mV,ndofV,iel)
       Tools.assemble_G(G_el,A_sparse,iconV,iconP,NfemV,mV,mP,ndofV,ndofP,iel)
       Tools.assemble_f(f_el,rhs,iconV,mV,ndofV,iel)
       Tools.assemble_h(h_el,rhs,iconP,mP,NfemV,iel)
       time_ass+= timing.time()-start1

    if ass_method==2:
       start1 = timing.time()
       for ikk in range(ndofV_el):
           m1=local_to_globalV[ikk,iel]
           for jkk in range(ndofV_el):
               V[counter]=K_el[ikk,jkk]
               counter+=1
           for jkk in range(0,mP):
               V[counter]=G_el[ikk,jkk]
               counter+=1
               V[counter]=G_el[ikk,jkk]
               counter+=1
           rhs[m1]+=f_el[ikk]
       for k2 in range(0,mP):
           m2=iconP[k2,iel]
           rhs[NfemV+m2]+=h_el[k2]
       time_ass+= timing.time()-start1

#end for iel

print("     -> assembly: %.3f s" % (time_ass))

print("build FE matrix: %.3f s | %d %d %e" % (timing.time() - start,Nfem,nel,havrg))

#plt.spy(A_sparse,markersize=1)
#plt.savefig('matrix_'+Vspace+'_'+Pspace+'.pdf', bbox_inches='tight')

###############################################################################
# convert to csr 
###############################################################################
start = timing.time()

if ass_method==1:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix = sparse.coo_matrix((V,(I,J)),shape=(Nfem,Nfem)).tocsr()

print("convert matrix to csr: %.3f s" % (timing.time() - start))

###############################################################################
# solve system
###############################################################################
start = timing.time()

sol=spsolve(sparse_matrix,rhs)

print("solve time: %.3f s | %d %d %e" % (timing.time() - start,Nfem,nel,havrg))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*(eta_ref/Lx)

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

print("split vel into u,v: %.3f s" % (timing.time() - start))

###############################################################################
# normalise pressure 
###############################################################################
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

###############################################################################

xc = np.zeros(nel,dtype=np.float64)
yc = np.zeros(nel,dtype=np.float64)
bxc = np.zeros(nel,dtype=np.float64)
byc = np.zeros(nel,dtype=np.float64)
etac = np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    xc[iel]=np.sum(xV[iconV[:,iel]])/mV
    yc[iel]=np.sum(yV[iconV[:,iel]])/mV
    bxc[iel]=mms.bx(xc[iel],yc[iel],drho)
    byc[iel]=mms.by(xc[iel],yc[iel],drho)
    etac[iel]=mms.eta(xc[iel],yc[iel],etastar)

###############################################################################
# compute vrms and errors
###############################################################################
start = timing.time()

uq = np.zeros(nq,dtype=np.float64)
vq = np.zeros(nq,dtype=np.float64)
pq = np.zeros(nq,dtype=np.float64)

#nqts=12
#nqel,qcoords_r,qcoords_s,qweights=Q.quadrature(Vspace,nqpts)
#nq=nqel*nel

errdivv=0
errv=0
errp=0
vrms=0
vrmss=0
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

        #if Vspace=='P1+':
        #   NNNV=FE.NNN(rq,sq,'P1')
        #   uq[counterq]=NNNV.dot(u[iconV[0:3,iel]])
        #   vq[counterq]=NNNV.dot(v[iconV[0:3,iel]])
        #else:   
        uq[counterq]=NNNV.dot(u[iconV[0:mV,iel]])
        vq[counterq]=NNNV.dot(v[iconV[0:mV,iel]])
        pq[counterq]=NNNP.dot(p[iconP[0:mP,iel]])
        vrms+=(uq[counterq]**2+vq[counterq]**2)*weightq*jcob

        if experiment=='sinker' and abs(xc[iel]-0.5)<0.125 and abs(yc[iel]-0.75)<0.125:
           vrmss+=(uq[counterq]**2+vq[counterq]**2)*weightq*jcob

        uthq,vthq,pthq=mms.solution(xq[counterq],yq[counterq])
        errv+=(uq[counterq]-uthq)**2*weightq*jcob+\
              (vq[counterq]-vthq)**2*weightq*jcob
        errp+=(pq[counterq]-pthq)**2*weightq*jcob

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
vrmss/=0.0625

print("     -> nel= %6d ; vrms= %.5e ; vrms(th)= %.5e ; %6d %6d %e" %(nel,vrms,mms.vrms(),NfemV,NfemP,havrg))
print("     -> nel= %6d ; errv= %.5e ; errp= %.5e ; errdivv= %.5e | %6d %6d %.4e %.4e" %(nel,errv,errp,errdivv,NfemV,NfemP,havrg,Lx/nelx*np.sqrt(2)))

print("compute vrms & errors: %.3f s" % (timing.time() - start))

###############################################################################

eps=1e-6

if experiment=='solvi':
   xbottom=[]
   pbottom=[]
   if Vspace+Pspace=='P1+P1' or\
      Vspace+Pspace=='P2P1'  or\
      Vspace+Pspace=='Q2Q1' :
      for iel in range(0,nel):
          for k in range(0,mP):
              inode=iconP[k,iel]
              if yP[inode]/Ly<1e-6:
                 xbottom.append(xP[inode])
                 pbottom.append(p[inode])

   if Vspace+Pspace=='Q2Pm1':
      for iel in range(0,nel):
          if yV[iconV[0,iel]]/Ly<1e-6:
             rq=-1 ; sq=0
             NNNP=FE.NNN(rq,sq,Pspace)
             xbottom.append(NNNP.dot(xP[iconP[0:mP,iel]])+eps)
             pbottom.append(NNNP.dot(p[iconP[0:mP,iel]]))
             rq=+1 ; sq=0
             NNNP=FE.NNN(rq,sq,Pspace)
             xbottom.append(NNNP.dot(xP[iconP[0:mP,iel]])-eps)
             pbottom.append(NNNP.dot(p[iconP[0:mP,iel]]))

   if Vspace+Pspace=='P2P0':
      for iel in range(0,nel):
          inode0=iconV[0,iel]
          inode1=iconV[1,iel]
          inode2=iconV[2,iel]
          if yV[inode0]/Ly<1e-6 and yV[inode1]/Ly<1e-6:
             xbottom.append(xV[inode0]+eps)
             pbottom.append(p[iel])
             xbottom.append(xV[inode1]-eps)
             pbottom.append(p[iel])
          if yV[inode1]/Ly<1e-6 and yV[inode2]/Ly<1e-6:
             xbottom.append(xV[inode1]+eps)
             pbottom.append(p[iel])
             xbottom.append(xV[inode2]-eps)
             pbottom.append(p[iel])
          if yV[inode2]/Ly<1e-6 and yV[inode0]/Ly<1e-6:
             xbottom.append(xV[inode2]+eps)
             pbottom.append(p[iel])
             xbottom.append(xV[inode0]-eps)
             pbottom.append(p[iel])

   if Vspace+Pspace=='P2+P-1':
      for iel in range(0,nel):
          inode0=iconP[0,iel]
          inode1=iconP[1,iel]
          inode2=iconP[2,iel]
          if yP[inode0]/Ly<1e-6 and yP[inode1]/Ly<1e-6:
             xbottom.append(xP[inode0]+eps)
             pbottom.append(p[inode0])
             xbottom.append(xP[inode1]-eps)
             pbottom.append(p[inode1])
          if yP[inode1]/Ly<1e-6 and yP[inode2]/Ly<1e-6:
             xbottom.append(xP[inode1]+eps)
             pbottom.append(p[inode1])
             xbottom.append(xP[inode2]-eps)
             pbottom.append(p[inode2])
          if yP[inode2]/Ly<1e-6 and yP[inode0]/Ly<1e-6:
             xbottom.append(xP[inode2]+eps)
             pbottom.append(p[inode2])
             xbottom.append(xP[inode0]-eps)
             pbottom.append(p[inode0])


   xbottom=np.array(xbottom)
   pbottom=np.array(pbottom)
   sort = np.argsort(xbottom)
   Tools.export_swarm_to_ascii(xbottom[sort],pbottom[sort],\
                               'solvi_p_profile'+Vspace+'x'+Pspace+'_'+str(nelx)+'.ascii')
            
###############################################################################

if experiment=='sinker' or experiment=='sinker_reduced' or experiment=='sinker_open':

   #for i in range(0,NV):
   #    if abs(xV[i]-0.5)<eps and abs(yV[i]-0.75)<eps:
   #       print('     -> sinker_vel',xV[i],yV[i],v[i],etastar,drho,nelx)
   print('     -> sinker_vel',vrmss,etastar,drho,nelx)

   if Vspace+Pspace=='Q2Pm1':
      for iel in range(0,nel):
          rq=-1 ; sq=-1
          NNNP=FE.NNN(rq,sq,Pspace)
          xx=NNNP.dot(xP[iconP[0:mP,iel]])
          yy=NNNP.dot(yP[iconP[0:mP,iel]])
          if abs(xx-0.5)<eps and abs(yy-0.75)<eps:
             pp=NNNP.dot(p[iconP[0:mP,iel]])
             print('     -> sinker_press',xx,yy,pp,etastar,drho,nelx)
          rq=+1 ; sq=-1
          NNNP=FE.NNN(rq,sq,Pspace)
          xx=NNNP.dot(xP[iconP[0:mP,iel]])
          yy=NNNP.dot(yP[iconP[0:mP,iel]])
          if abs(xx-0.5)<eps and abs(yy-0.75)<eps:
             pp=NNNP.dot(p[iconP[0:mP,iel]])
             print('     -> sinker_press',xx,yy,pp,etastar,drho,nelx)
          rq=-1 ; sq=+1
          NNNP=FE.NNN(rq,sq,Pspace)
          xx=NNNP.dot(xP[iconP[0:mP,iel]])
          yy=NNNP.dot(yP[iconP[0:mP,iel]])
          if abs(xx-0.5)<eps and abs(yy-0.75)<eps:
             pp=NNNP.dot(p[iconP[0:mP,iel]])
             print('     -> sinker_press',xx,yy,pp,etastar,drho,nelx)
          rq=+1 ; sq=+1
          NNNP=FE.NNN(rq,sq,Pspace)
          xx=NNNP.dot(xP[iconP[0:mP,iel]])
          yy=NNNP.dot(yP[iconP[0:mP,iel]])
          if abs(xx-0.5)<eps and abs(yy-0.75)<eps:
             pp=NNNP.dot(p[iconP[0:mP,iel]])
             print('     -> sinker_press',xx,yy,pp,etastar,drho,nelx)
   if Vspace+Pspace=='P2P0':
      for iel in range(0,nel):
          rq=0 ; sq=0
          NNNV=FE.NNN(rq,sq,Vspace)
          xx=NNNV.dot(xV[iconV[0:mV,iel]])
          yy=NNNV.dot(yV[iconV[0:mV,iel]])
          if abs(xx-0.5)<eps and abs(yy-0.75)<eps:
             print('     -> sinker_press',xx,yy,p[iel],etastar,drho,nelx)
          rq=1 ; sq=0
          NNNV=FE.NNN(rq,sq,Vspace)
          xx=NNNV.dot(xV[iconV[0:mV,iel]])
          yy=NNNV.dot(yV[iconV[0:mV,iel]])
          if abs(xx-0.5)<eps and abs(yy-0.75)<eps:
             print('     -> sinker_press',xx,yy,p[iel],etastar,drho,nelx)
          rq=0 ; sq=1
          NNNV=FE.NNN(rq,sq,Vspace)
          xx=NNNV.dot(xV[iconV[0:mV,iel]])
          yy=NNNV.dot(yV[iconV[0:mV,iel]])
          if abs(xx-0.5)<eps and abs(yy-0.75)<eps:
             print('     -> sinker_press',xx,yy,p[iel],etastar,drho,nelx)
   else:
      for i in range(0,NP):
          if abs(xP[i]-0.5)<eps and abs(yP[i]-0.75)<eps:
             print('     -> sinker_press',xP[i],yP[i],p[i],etastar,drho,nelx)

if experiment=='RTwave':
   llambda=0.5
   amplitude=0.01
   phi1=2.*np.pi*(Ly/2.)/llambda
   phi2=2.*np.pi*(Ly/2.)/llambda
   rho1=1+drho
   rho2=1
   eta1=1
   grav=1
   eta2=etastar
   c11 = (eta1*2*phi1**2)/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) \
        - (2*phi2**2)/(np.cosh(2*phi2)-1-2*phi2**2)
   d12 = (eta1*(np.sinh(2*phi1) -2*phi1))/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) \
        + (np.sinh(2*phi2)-2*phi2)/(np.cosh(2*phi2)-1-2*phi2**2)
   i21 = (eta1*phi2*(np.sinh(2*phi1)+2*phi1))/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) \
        + (phi2*(np.sinh(2*phi2)+2*phi2))/(np.cosh(2*phi2)-1-2*phi2**2) 
   j22 = (eta1*2*phi1**2*phi2)/(eta2*(np.cosh(2*phi1)-1-2*phi1**2))\
        - (2*phi2**3)/(np.cosh(2*phi2)-1-2*phi2**2)
   K=-d12/(c11*j22-d12*i21)
   val=K*(rho1-rho2)/2/eta2*(Ly/2.)*grav*amplitude

   print('     -> rt_wave',np.max(abs(v)),phi1,etastar,drho,nelx,val)
   

###############################################################################
# compute q1,q2, see Delaunay subsection. As a rule-of-thumb, in a good quality 
# mesh all triangles should have q1,q2 above about 0.4-0.5
###############################################################################
start = timing.time()

q1 = np.zeros(nel,dtype=np.float64)
q2 = np.zeros(nel,dtype=np.float64)
q3 = np.zeros(nel,dtype=np.float64)

if visu and Vspace[0]=='P':

   for iel in range(0,nel):
       a=np.sqrt((xV[iconV[0,iel]]-xV[iconV[1,iel]])**2+(yV[iconV[0,iel]]-yV[iconV[1,iel]])**2)
       b=np.sqrt((xV[iconV[0,iel]]-xV[iconV[2,iel]])**2+(yV[iconV[0,iel]]-yV[iconV[2,iel]])**2)
       c=np.sqrt((xV[iconV[1,iel]]-xV[iconV[2,iel]])**2+(yV[iconV[1,iel]]-yV[iconV[2,iel]])**2)
       q1[iel]=(b+c-a)*(c+a-b)*(a+b-c)/(a*b*c)
       q2[iel]=4*np.sqrt(3)*area[iel]/(a**2+b**2+c**2)
       q3[iel]=max(a,b,c)

   print('     -> q1 (m,M):',np.min(q1),np.max(q1))
   print('     -> q2 (m,M):',np.min(q2),np.max(q2))
   print('     -> q3 (m,M):',np.min(q3),np.max(q3))

   plt.clf()
   plt.title("q1 histogram")
   plt.hist(q1, bins=50) 
   plt.savefig('histogram_q1.pdf', bbox_inches='tight')

   plt.clf()
   plt.hist(q2, bins=50) 
   plt.title("q2 histogram")
   plt.savefig('histogram_q2.pdf', bbox_inches='tight')

   plt.clf()
   plt.hist(area, bins=50) 
   plt.title("area histogram")
   plt.savefig('histogram_area.pdf', bbox_inches='tight')

print("compute q1,q2 : %.3f s" % (timing.time() - start))

###############################################################################

#Tools.export_swarm_vector_to_ascii(xV,yV,u,v,'solution_velocity.ascii')
#Tools.export_swarm_scalar_to_ascii(xP,yP,p,'solution_pressure.ascii')
Tools.export_elements_to_vtu(xV,yV,iconV,Vspace,'meshV.vtu',area,bxc,byc,etac,q1,q2,q3)
Tools.export_V_to_vtu(NV,xV,yV,iconV,Vspace,'visu_V.vtu',u,v,Pspace,p,iconP)

if visu:
   Tools.export_elements_to_vtu(xP,yP,iconP,Pspace,'meshP.vtu',area,bxc,byc,etac,q1,q2,q3)
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
   Tools.export_swarm_vector_to_vtu(xV,yV,u-uth,v-vth,'velocity_error.vtu')
   Tools.export_swarm_scalar_to_vtu(xV,yV,qth,'pressure_analytical.vtu')
   Tools.export_swarm_scalar_to_vtu(xP,yP,p,'solution_pressure.vtu')

   Tools.export_swarm_vector_to_ascii(xV,yV,uth,vth,'velocity_analytical.ascii')

   Tools.export_connectivity_array_to_ascii(xV,yV,iconV,'iconV.ascii')
   Tools.export_connectivity_array_to_ascii(xP,yP,iconP,'iconP.ascii')

   #pth = np.zeros(NP,dtype=np.float64)
   #for i in range(NP):        
   #    pth[i]=mms.p_th(xP[i],yP[i])
   #Tools.export_swarm_scalar_to_vtu(xP,yP,pth,'pressure_analytical.vtu')
   #Tools.export_swarm_scalar_to_ascii(xP,yP,pth,'pressure_analytical.ascii')

print("*****************************")
print("*****************************")

###############################################################################
