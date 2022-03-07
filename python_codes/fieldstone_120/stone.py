import FEbasis2D as FE
import FEquadrature as Q
import FEtools as Tools 
import numpy as np
import time as timing
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

#------------------------------------------------------------------------------
# bx and by are the body force components

def bx(x, y):
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val

def by(x, y):
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val

def viscosity(x,y):
    return 1

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

Lx=1
Ly=1

nelx=32
nely=32

left_bc  ='no_slip'
right_bc ='no_slip'
bottom_bc='no_slip'
top_bc   ='no_slip'

ndofV=2
ndofP=1

Vspace='P1'
Pspace='P0'

# if quadrilateral nqpts is nqperdim
# if triangle nqpts is total nb of qpoints 

nqpts=3

#--------------------------------------------------------------------
# mesh: node layout and connectivity
#--------------------------------------------------------------------
start = timing.time()

mV=FE.NNN_m(Vspace)
mP=FE.NNN_m(Pspace)

nqel,qcoords_r,qcoords_s,qweights=Q.quadrature(Vspace,nqpts)

NV,nel,xV,yV,iconV,iconV2=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Vspace)
NP,nel,xP,yP,iconP,iconP2=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Pspace)


nq=nqel*nel
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP

print("mesh setup: %.3f s" % (timing.time() - start))

#--------------------------------------------------------------------
# boundary conditions setup 
#--------------------------------------------------------------------
start = timing.time()

bc_fix,bc_val=Tools.bc_setup(xV,yV,Lx,Ly,ndofV,left_bc,right_bc,bottom_bc,top_bc)

print("bc setup: %.3f s" % (timing.time() - start))

#--------------------------------------------------------------------
# compute area of elements 
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
#--------------------------------------------------------------------
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nqel):
        rq=qcoords_r[iq]
        sq=qcoords_s[iq]
        weightq=qweights[iq]
        NNNV=FE.NNN(rq,sq,Vspace)
        dNNNVdr=FE.dNNNdr(rq,sq,Vspace)
        dNNNVds=FE.dNNNds(rq,sq,Vspace)
        jcob,jcbi,dNNNVdx,dNNNVdy=Tools.J(mV,dNNNVdr,dNNNVds,xV[iconV[0:mV,iel]],yV[iconV[0:mV,iel]])
        area[iel]+=jcob*weightq
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.8e " %(area.sum()))
print("     -> total area anal %.8e " %(Lx*Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

#--------------------------------------------------------------------
# build FE matrix
#--------------------------------------------------------------------
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
rhs = np.zeros(Nfem,dtype=np.float64) 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64)
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) 
    
xq = np.zeros(nq,dtype=np.float64)
yq = np.zeros(nq,dtype=np.float64)

counterq=0

for iel in range(0,nel):

    K_el = np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el = np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    f_el = np.zeros(mV*ndofV,dtype=np.float64)
    h_el = np.zeros(mP*ndofP,dtype=np.float64)

    for iq in range(0,nqel):

        rq=qcoords_r[iq]
        sq=qcoords_s[iq]
        weightq=qweights[iq]

        NNNV=FE.NNN(rq,sq,Vspace)
        dNNNVdr=FE.dNNNdr(rq,sq,Vspace)
        dNNNVds=FE.dNNNds(rq,sq,Vspace)
        NNNP=FE.NNN(rq,sq,Pspace)
        xq[counterq]=NNNV.dot(xV[iconV[0:mV,iel]])
        yq[counterq]=NNNV.dot(yV[iconV[0:mV,iel]])

        jcob,jcbi,dNNNVdx,dNNNVdy=Tools.J(mV,dNNNVdr,dNNNVds,xV[iconV[0:mV,iel]],yV[iconV[0:mV,iel]])

        for k in range(0,mV): 
            b_mat[0:3,2*k:2*k+2] = [[dNNNVdx[k],0.        ],  
                                    [0.        ,dNNNVdy[k]],
                                    [dNNNVdy[k],dNNNVdx[k]]]

        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq[counterq],yq[counterq])*weightq*jcob

        for k in range(0,mV): 
            f_el[2*k+0]+=NNNV[k]*jcob*weightq*bx(xq[counterq],yq[counterq])
            f_el[2*k+1]+=NNNV[k]*jcob*weightq*by(xq[counterq],yq[counterq])

        for k in range(0,mP):
            N_mat[0,k]=NNNP[k]
            N_mat[1,k]=NNNP[k]
            N_mat[2,k]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        counterq+=1

    #end for iq

    # apply bc
    Tools.apply_bc(K_el,G_el,f_el,h_el,bc_val,bc_fix,iconV,mV,ndofV,iel)

    # assemble (missing h_el)
    Tools.assemble_K(K_el,A_sparse,iconV,mV,ndofV,iel)
    Tools.assemble_G(G_el,A_sparse,iconV,iconP,NfemV,mV,mP,ndofV,ndofP,iel)
    Tools.assemble_f(f_el,rhs,iconV,mV,ndofV,iel)

#end for iel

print("build FE matrix: %.3f s" % (timing.time() - start))


plt.spy(A_sparse)
plt.savefig('matrix_'+Vspace+'_'+Pspace+'.pdf', bbox_inches='tight')

#------------------------------------------------------------------------------
# solve system
#------------------------------------------------------------------------------
start = timing.time()

sparse_matrix=A_sparse.tocsr()
sol=spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# put solution into separate x,y velocity arrays
#------------------------------------------------------------------------------
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

print("split vel into u,v: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------

Tools.export_mesh_to_vtu(xV,yV,iconV,Vspace,'meshV.vtu')
Tools.export_mesh_to_vtu(xV,yV,iconV2,Vspace,'meshV2.vtu')
Tools.export_mesh_to_ascii(xV,yV,'meshV.ascii')
Tools.export_mesh_to_ascii(xP,yP,'meshP.ascii')
Tools.export_swarm_to_vtu(xV,yV,'meshV_nodes.vtu')
Tools.export_swarm_to_vtu(xP,yP,'meshP_nodes.vtu')

Tools.export_mesh_to_ascii(xq,yq,'meshq.ascii')
Tools.export_swarm_to_vtu(xq,yq,'meshq.vtu')

Tools.export_swarm_vector_to_vtu(xV,yV,u,v,'solution_velocity.vtu')
Tools.export_swarm_scalar_to_vtu(xP,yP,p,'solution_pressure.vtu')

Tools.export_connectivity_array_to_ascii(xV,yV,iconV,'iconV.ascii')
Tools.export_connectivity_array_to_ascii(xP,yP,iconP,'iconP.ascii')
