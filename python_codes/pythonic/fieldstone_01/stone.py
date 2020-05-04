import numpy as np
import sys as sys
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time as time
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# body force components


def body_force(x):
    bfx = ((12.-24.*x[1])*x[0]**4+(-24.+48.*x[1])*x[0]*x[0]*x[0]
           + (-48.*x[1]+72.*x[1]*x[1]-48.*x[1]*x[1]*x[1]+12.)*x[0]*x[0]
           + (-2.+24.*x[1]-72.*x[1]*x[1]+48.*x[1]*x[1]*x[1])*x[0]
           + 1.-4.*x[1]+12.*x[1]*x[1]-8.*x[1]*x[1]*x[1])

    bfy = ((8.-48.*x[1]+48.*x[1]*x[1])*x[0]*x[0]*x[0]
           + (-12.+72.*x[1]-72.*x[1]*x[1])*x[0]*x[0]
           + (4.-24.*x[1]+48.*x[1]*x[1]-48.*x[1]*x[1]*x[1]+24.*x[1]**4)*x[0]
           - 12.*x[1]*x[1]+24.*x[1]*x[1]*x[1]-12.*x[1]**4)
    return np.array([bfx, bfy])

# ------------------------------------------------------------------------------
# analytical solution


def velocity(x):
    val = np.array([x[0]*x[0]*(1.-x[0])**2
                    * (2.*x[1]-6.*x[1]*x[1]+4*x[1]*x[1]*x[1]),
                    -x[1]*x[1]*(1.-x[1])**2
                    * (2.*x[0]-6.*x[0]*x[0]+4*x[0]*x[0]*x[0])])

    return val


def pressure(x):
    val = x[0]*(1.-x[0])-1./6.
    return val

# ------------------------------------------------------------------------------


def onePlot(variable, plotX, plotY, title, labelX, labelY, extVal,
            limitX, limitY, colorMap):
    im = axes[plotX][plotY].imshow(np.flipud(variable),
                                   extent=extVal, cmap=colorMap,
                                   interpolation="nearest")
    axes[plotX][plotY].set_title(title, fontsize=10, y=1.01)

    if (limitX != 0.0):
        axes[plotX][plotY].set_xlim(0, limitX)

    if (limitY != 0.0):
        axes[plotX][plotY].set_ylim(0, limitY)

    axes[plotX][plotY].set_xlabel(labelX)
    axes[plotX][plotY].set_ylabel(labelY)
    fig.colorbar(im, ax=axes[plotX][plotY])
    return

# ------------------------------------------------------------------------------


print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

# declare variables
print("variable declaration")

m = 4     # number of nodes making up an element
ndof = 2  # number of degrees of freedom per node

Lx = 1.  # horizontal extent of the domain
Ly = 1.  # vertical extent of the domain

assert (Lx > 0.), "Lx should be positive"
assert (Ly > 0.), "Ly should be positive"

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
    nelx = int(sys.argv[1])
    nely = int(sys.argv[2])
    visu = int(sys.argv[3])
else:
    nelx = 32
    nely = 32
    visu = 1

assert (nelx > 0.), "nnx should be positive"
assert (nely > 0.), "nny should be positive"

nnx = nelx + 1  # number of elements, x direction
nny = nely + 1  # number of elements, y direction

nnp = nnx * nny  # number of nodes

nel = nelx * nely  # number of elements, total

penalty = 1.e7  # penalty coefficient value

viscosity = 1.  # dynamic viscosity \eta

Nfem = nnp * ndof  # Total number of degrees of freedom

eps = 1.e-10

sqrt3 = np.sqrt(3.)

#################################################################
# grid point setup
#################################################################
start = time.time()

xs = np.linspace(0., 1., nnx, dtype=np.float64)
ys = np.linspace(0., 1., nny, dtype=np.float64)
xv, yv = np.meshgrid(xs, ys)
x = xv.flatten()
y = yv.flatten()
xy = np.vstack((x, y)).T
print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# build connectivity array
#################################################################
start = time.time()

icon = np.zeros((m, nel), dtype=np.int16)

xis = np.linspace(0., nelx-1, nelx, dtype=np.int16)
yis = np.linspace(0., nely-1, nely, dtype=np.int16)
xiv, yiv = np.meshgrid(xis, yis)

icon = np.array([xiv + yiv * nnx,
                 (xiv + 1) + yiv * nnx,
                 (xiv + 1) + (yiv + 1) * nnx,
                 xiv + (yiv + 1) * nnx]).reshape((m, nel))

# for iel in range (0, nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#     print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
# for this benchmark: no slip.
#################################################################
start = time.time()

raw_b_inds = np.where(np.logical_or.reduce((x < eps, x > Lx-eps,
                                            y < eps, y > Ly-eps)))[0]
# the [0] index above is necessary because numpy.where returns a tuple
# with len(number of dimensions), which is here equal to one.

bc_inds = np.sort(np.hstack((raw_b_inds*ndof, raw_b_inds*ndof+1)))
bc_vals = np.array([0. for idx in bc_inds])

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# r,s are the reduced coordinates in the [-1:1]x[-1:1] ref elt
#################################################################
start = time.time()

etime = 0.

# a_mat = lil_matrix((Nfem, nfem),dtype=np.float64)

a_mat = np.zeros((Nfem, Nfem), dtype=np.float64)  # matrix of Ax=b
b_mat = np.zeros((3, ndof*m), dtype=np.float64)   # gradient matrix B
rhs = np.zeros(Nfem, dtype=np.float64)         # right hand side of Ax=b
N = np.zeros(m, dtype=np.float64)            # shape functions
dNdxy = np.zeros((2, m), dtype=np.float64)   # shape functions derivatives
dNdrs = np.zeros((2, m), dtype=np.float64)   # shape functions derivatives
u = np.zeros(nnp, dtype=np.float64)          # x-component velocity
v = np.zeros(nnp, dtype=np.float64)          # y-component velocity
k_mat = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float64)
c_mat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)

# we will integrate viscous term at 4 quadrature points
n_quad = 4
Nq = np.zeros((m, n_quad), dtype=np.float64)            # shape functions
dNdrsq = np.zeros((2, m, n_quad), dtype=np.float64)

me_null = np.zeros((m, nel), dtype=np.float64)
meq_null = np.zeros((m, nel, n_quad), dtype=np.float64)

ijq = np.array([[iq, jq] for iq in [-1, 1] for jq in [-1, 1]]).T

# position & weight of quad. points
rsq = ijq/sqrt3
weightqq = 1.*1.

Nq[0, :] = 0.25*(1.-rsq[0])*(1.-rsq[1])
Nq[1, :] = 0.25*(1.+rsq[0])*(1.-rsq[1])
Nq[2, :] = 0.25*(1.+rsq[0])*(1.+rsq[1])
Nq[3, :] = 0.25*(1.-rsq[0])*(1.+rsq[1])

dNdrsq[:, 0, :] = [-0.25*(1.-rsq[1]), -0.25*(1.-rsq[0])]
dNdrsq[:, 1, :] = [+0.25*(1.-rsq[1]), -0.25*(1.+rsq[0])]
dNdrsq[:, 2, :] = [+0.25*(1.+rsq[1]), +0.25*(1.+rsq[0])]
dNdrsq[:, 3, :] = [-0.25*(1.+rsq[1]), +0.25*(1.-rsq[0])]

jcb = np.einsum('ikq, kej->eqij', dNdrsq, xy[icon])

# calculate the determinants of the jacobians
jcob = np.linalg.det(jcb)

# calculate inverse of the jacobian matrix
jcbi = np.linalg.inv(jcb)

# compute dNdx & dNdy
xyeq = np.einsum('kq, kej->jeq', Nq, xy[icon])
dNdxyeq = np.einsum('eqij, jkq->ikeq', jcbi, dNdrsq)

# construct 3x8 b_mat matrix
b_mat = np.array([[dNdxyeq[0], meq_null],
                  [meq_null,   dNdxyeq[1]],
                  [dNdxyeq[1], dNdxyeq[0]]]).reshape(3, ndof*m, nel, n_quad,
                                                     order='F')

# compute elemental a_mat matrix
a_el = (np.einsum('jieq, jk, kleq, eq->eil', b_mat, c_mat, b_mat, jcob)
        * viscosity * weightqq)

# compute elemental rhs vector
b_el = np.einsum('iq, jeq, eq->eji', Nq, body_force(xyeq), jcob)*weightqq

# integrate penalty term at 1 point
rq = 0.
sq = 0.
weightq = 2. * 2.

N[0] = 0.25 * (1.-rq) * (1.-sq)
N[1] = 0.25 * (1.+rq) * (1.-sq)
N[2] = 0.25 * (1.+rq) * (1.+sq)
N[3] = 0.25 * (1.-rq) * (1.+sq)

dNdrs[:, 0] = [-0.25*(1.-sq), -0.25*(1.-rq)]
dNdrs[:, 1] = [+0.25*(1.-sq), -0.25*(1.+rq)]
dNdrs[:, 2] = [+0.25*(1.+sq), +0.25*(1.+rq)]
dNdrs[:, 3] = [-0.25*(1.+sq), +0.25*(1.-rq)]

# compute the jacobian
jcb = np.einsum('ik, kej->eij', dNdrs, xy[icon])

# calculate determinant of the jacobian
jcob = np.linalg.det(jcb)

# calculate the inverse of the jacobian
jcbi = np.linalg.inv(jcb)

# compute dNdx and dNdy
dNdxye = np.einsum('eij, jk->ike', jcbi, dNdrs)

# compute gradient matrix
b_mat = np.array([[dNdxye[0], me_null],
                  [me_null,   dNdxye[1]],
                  [dNdxye[1], dNdxye[0]]]).reshape(3, ndof*m, nel,
                                                   order='F')

# update elemental matrix
a_el += (np.einsum('jie, jk, kle, e->eil', b_mat, k_mat, b_mat, jcob)
         * penalty * weightq)

# assemble matrix a_mat and right hand side rhs
m_indices = ((ndof*icon).T[:, np.newaxis, :]
             + np.indices((ndof,))[0, np.newaxis, :, np.newaxis]) # iel, k1, i1

mkk_indices = m_indices.reshape(nel, ndof*m, order='F') # iel, 1kk
mm_indices = (mkk_indices[:, :, np.newaxis] +
              0*mkk_indices[:, np.newaxis, :]) # iel, 1kk, 2kk
mm_indices = (mm_indices, np.einsum('ijm -> imj', mm_indices))

np.add.at(rhs, m_indices, b_el)
np.add.at(a_mat, mm_indices, a_el)

print("build FE matrix: %.3f s" % (time.time() - start))

#################################################################
# impose boundary conditions
# for now it is done outside of the previous loop, we will see
# later in the course how it can be incorporated seamlessly in it.
#################################################################
start = time.time()

rhs -= np.einsum('ij, i', a_mat[bc_inds, :], bc_vals)
bc_diag = a_mat[bc_inds, bc_inds]  # this returns a new array, not a view.
rhs[bc_inds] = bc_diag * bc_vals

a_mat[bc_inds, :] = 0.
a_mat[:, bc_inds] = 0.
a_mat[bc_inds, bc_inds] = bc_diag

# print("a_mat (m,M) = %.4f %.4f" %(np.min(a_mat), np.max(a_mat)))
# print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs), np.max(rhs)))

print("impose b.c.: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol = spsolve(csr_matrix(a_mat), rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
start = time.time()

u, v = np.reshape(sol, (nnp, 2)).T

print("     -> u (m,M) %.4f %.4f " % (np.min(u), np.max(u)))
print("     -> v (m,M) %.4f %.4f " % (np.min(v), np.max(v)))

np.savetxt('velocity.ascii', np.array([x, y, u, v]).T, header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure
# we compute the pressure and strain rate components in the middle
# of the elements.
#####################################################################
start = time.time()

rq = 0.0
sq = 0.0
weightq = 2.0 * 2.0

N[0] = 0.25 * (1.-rq) * (1.-sq)
N[1] = 0.25 * (1.+rq) * (1.-sq)
N[2] = 0.25 * (1.+rq) * (1.+sq)
N[3] = 0.25 * (1.-rq) * (1.+sq)

dNdrs[:, 0] = [-0.25*(1.-sq), -0.25*(1.-rq)]
dNdrs[:, 1] = [+0.25*(1.-sq), -0.25*(1.+rq)]
dNdrs[:, 2] = [+0.25*(1.+sq), +0.25*(1.+rq)]
dNdrs[:, 3] = [-0.25*(1.+sq), +0.25*(1.-rq)]

# compute the jacobians
jcb = np.einsum('ik, klj->lij', dNdrs, xy[icon])

# calculate determinants of the jacobians
jcob = np.linalg.det(jcb)

# calculate the inverses of the jacobians
jcbi = np.linalg.inv(jcb)

# compute dNdx and dNdy
dNdxy = np.einsum('lij, jk->lik', jcbi, dNdrs)

xyc = np.einsum('k, kli->li', N, xy[icon])
exx = np.einsum('lk, kl->l', dNdxy[:, 0, :], u[icon])
eyy = np.einsum('lk, kl->l', dNdxy[:, 1, :], v[icon])
exy = 0.5*(np.einsum('lk, kl->l', dNdxy[:, 0, :], v[icon])
           + np.einsum('lk, kl->l', dNdxy[:, 1, :], u[icon]))

p = -penalty * (exx + eyy)


print("     -> p (m,M) %.4f %.4f " % (np.min(p), np.max(p)))
print("     -> exx (m,M) %.4f %.4f " % (np.min(exx), np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " % (np.min(eyy), np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " % (np.min(exy), np.max(exy)))

np.savetxt('pressure.ascii', np.array([xyc[:, 0], xyc[:, 1], p]).T,
           header='# xc,yc,p')

np.savetxt('strainrate.ascii', np.array([xyc[:, 0], xyc[:, 1],
                                         exx, eyy, exy]).T,
           header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

#################################################################
# compute error
#################################################################
start = time.time()

error_u = np.empty(nnp, dtype=np.float64)
error_v = np.empty(nnp, dtype=np.float64)
error_p = np.empty(nel, dtype=np.float64)

error_u = u - velocity(xy.T)[0]
error_v = v - velocity(xy.T)[1]

error_p = p - pressure(xyc.T)

print("compute nodal error for plot: %.3f s" % (time.time() - start))

#################################################################
# compute error in L2 norm
#################################################################
start = time.time()

ijq = np.array([[iq, jq] for iq in [-1, 1] for jq in [-1, 1]]).T

# position & weight of quad. points
rsq = ijq/sqrt3
weightq = 1.*1.

Nq[0, :] = 0.25*(1.-rsq[0])*(1.-rsq[1])
Nq[1, :] = 0.25*(1.+rsq[0])*(1.-rsq[1])
Nq[2, :] = 0.25*(1.+rsq[0])*(1.+rsq[1])
Nq[3, :] = 0.25*(1.-rsq[0])*(1.+rsq[1])

dNdrsq[:, 0, :] = [-0.25*(1.-rsq[1]), -0.25*(1.-rsq[0])]
dNdrsq[:, 1, :] = [+0.25*(1.-rsq[1]), -0.25*(1.+rsq[0])]
dNdrsq[:, 2, :] = [+0.25*(1.+rsq[1]), +0.25*(1.+rsq[0])]
dNdrsq[:, 3, :] = [-0.25*(1.+rsq[1]), +0.25*(1.-rsq[0])]

# calculate jacobian matrix for all the quadrature points
jcb = np.einsum('ikq, klj->lqij', dNdrsq, xy[icon])

# calculate the determinants of the jacobians
jcob = np.linalg.det(jcb)

xq = np.einsum('kq, kl->lq', Nq, x[icon])
yq = np.einsum('kq, kl->lq', Nq, y[icon])
uq = np.einsum('kq, kl->lq', Nq, u[icon])
vq = np.einsum('kq, kl->lq', Nq, v[icon])

pq = np.einsum('i, j->ij', p, np.ones(n_quad))
errv = np.sum(((uq-velocity([xq, yq])[0])**2
               + (vq-velocity([xq, yq])[1])**2) * weightq * jcob)
errp = np.sum((pq-pressure([xq, yq]))**2 * weightq * jcob)

errv = np.sqrt(errv)
errp = np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" % (nel, errv, errp))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################

u_temp = np.reshape(u, (nny, nnx))
v_temp = np.reshape(v, (nny, nnx))
p_temp = np.reshape(p, (nely, nelx))
exx_temp = np.reshape(exx, (nely, nelx))
eyy_temp = np.reshape(eyy, (nely, nelx))
exy_temp = np.reshape(exy, (nely, nelx))
error_u_temp = np.reshape(error_u, (nny, nnx))
error_v_temp = np.reshape(error_v, (nny, nnx))
error_p_temp = np.reshape(error_p, (nely, nelx))

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

uextent = (np.amin(x), np.amax(x), np.amin(y), np.amax(y))
pextent = (np.amin(xyc[:, 0]), np.amax(xyc[:, 0]),
           np.amin(xyc[:, 1]), np.amax(xyc[:, 1]))

onePlot(u_temp,       0, 0, "$v_x$",                 "x", "y",
        uextent,  0,  0, 'Spectral_r')
onePlot(v_temp,       0, 1, "$v_y$",                 "x", "y",
        uextent,  0,  0, 'Spectral_r')
onePlot(p_temp,       0, 2, "$p$",                   "x", "y",
        pextent, Lx, Ly, 'RdGy_r')
onePlot(exx_temp,     1, 0, "$\\dot{\\epsilon}_{xx}$", "x", "y",
        pextent, Lx, Ly, 'viridis')
onePlot(eyy_temp,     1, 1, "$\\dot{\\epsilon}_{yy}$", "x", "y",
        pextent, Lx, Ly, 'viridis')
onePlot(exy_temp,     1, 2, "$\\dot{\\epsilon}_{xy}$", "x", "y",
        pextent, Lx, Ly, 'viridis')
onePlot(error_u_temp, 2, 0, "$v_x-t^{th}_x$",        "x", "y",
        uextent,  0,  0, 'Spectral_r')
onePlot(error_v_temp, 2, 1, "$v_y-t^{th}_y$",        "x", "y",
        uextent,  0,  0, 'Spectral_r')
onePlot(error_p_temp, 2, 2, "$p-p^{th}$",            "x", "y",
        uextent,  0,  0, 'RdGy_r')

plt.subplots_adjust(hspace=0.5)

if visu == 1:
    plt.savefig('solution.pdf', bbox_inches='tight')
    plt.show()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
