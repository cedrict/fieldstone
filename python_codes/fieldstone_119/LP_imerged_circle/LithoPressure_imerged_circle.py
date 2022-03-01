from firedrake import *
from firedrake.petsc import PETSc


# load the mesh generated with Gmsh
mesh = Mesh('Box_imerged_circle.msh')

V = FunctionSpace(mesh, "Lagrange", 1)
q = TestFunction(V)
p = TrialFunction(V)

# ===========================================
# Boundary conditions
# ===========================================
bc = DirichletBC(V, Constant(0.0), (3,)) # Surface

# ===========================================
# Variational form
# ===========================================
# Box interior ==> 6
# Cricle interior ==> 7

gravity = Constant((0.0, -10.0))
rho_box = Constant(1.0)
rho_circle = Constant(2.0)

bform_A = inner(grad(q), grad(p)) * dx
lform_L = inner(grad(q), rho_box*gravity) * dx(6) + inner(grad(q), rho_circle*gravity) * dx(7)

# solve the variational problem
p = Function(V)
solve(bform_A == lform_L, p, bcs=bc, solver_parameters={'ksp_type': 'cg', "ksp_monitor": None})

p.rename("Pressure")
File("LP_circle.pvd").write(p)