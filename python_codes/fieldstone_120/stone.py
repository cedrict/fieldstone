import FEbasis2D as FE
import FEquadrature as Q
import FEtools as Tools 
import numpy as np
import time as timing
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

#--------------------------------------

Lx=1
Ly=1

nelx=16
nely=16

left_bc  ='free_slip'
right_bc ='free_slip'
bottom_bc='free_slip'
top_bc   ='free_slip'

ndofV=2
ndofP=1

Vspace='Q1'
Pspace='Q0'

# if quadrilateral nqpts is nqperdim
# if triangle nqpts is total nb of qpoints 

nqpts=2

#--------------------------------------------------------------------
# mesh: node layout and connectivity
#--------------------------------------------------------------------
start = timing.time()

mV=FE.NNN_m(Vspace)
mP=FE.NNN_m(Pspace)

nq,qcoords_r,qcoords_s,qweights=Q.quadrature(Vspace,nqpts)

NV,nel,xV,yV,iconV,iconV2=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Vspace)
NP,nel,xP,yP,iconP,iconP2=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Pspace)

Tools.export_mesh_to_vtu(xV,yV,iconV,Vspace,'meshV.vtu')
Tools.export_mesh_to_vtu(xV,yV,iconV2,Vspace,'meshV2.vtu')
Tools.export_mesh_to_ascii(xV,yV,'meshV.ascii')
Tools.export_mesh_to_ascii(xP,yP,'meshP.ascii')

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
exit()
#--------------------------------------------------------------------
# build FE matrix
#--------------------------------------------------------------------
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
rhs = np.zeros(Nfem,dtype=np.float64) 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64)
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) 

for iel in range(0,nel):

    K_el = np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el = np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    b_el = np.zeros(mV*ndofV,dtype=np.float64)

    for iq in range(0,nq):

        rq=qcoords_r[iq]
        sq=qcoords_s[iq]
        weightq=qweights[iq]

        NNNV=FE.NNN(rq,sq,Vspace)
        dNNNVdr=FE.dNNNdr(rq,sq,Vspace)
        dNNNVds=FE.dNNNds(rq,sq,Vspace)
        NNNP=FE.NNN(rq,sq,Pspace)
        xq=NNNV.dot(xV[iconV[0:mV,iel]])
        yq=NNNV.dot(yV[iconV[0:mV,iel]])

        jcob,jcbi,dNNNVdx,dNNNVdy=Tools.J(mV,dNNNVdr,dNNNVds,xV[iconV[0:mV,iel]],yV[iconV[0:mV,iel]])

        for k in range(0,mV): 
            b_mat[0:3,2*k:2*k+2] = [[dNNNVdx[k],0.        ],  
                                    [0.        ,dNNNVdy[k]],
                                    [dNNNVdy[k],dNNNVdx[k]]]

        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq)*weightq*jcob

        for k in range(0,mV): 
            b_el[2*k+0]+=NNNV[k]*jcob*weightq*bx(xq,yq)
            b_el[2*k+1]+=NNNV[k]*jcob*weightq*by(xq,yq)

        for k in range(0,mP):
            N_mat[0,k]=NNNP[k]
            N_mat[1,k]=NNNP[k]
            N_mat[2,k]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

    #end for iq

    # apply bc

    # assemble 
    Tools.assemble_K(K_el,A_sparse,NfemV,mV,ndofV)
    Tools.assemble_G(G_el,A_sparse,NfemV,NfemP,mV,mP,ndofV,ndofP)


#end for iel

print("build FE matrix: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# solve system
#------------------------------------------------------------------------------
start = timing.time()

sparse_matrix=A_sparse.tocsr()
sol=spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))






Tools.export_connectivity_array_to_ascii(xV,yV,iconV,'iconV.ascii')
Tools.export_connectivity_array_to_ascii(xP,yP,iconP,'iconP.ascii')
