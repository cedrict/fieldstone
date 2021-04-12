f90comp = gfortran
switch = -fbounds-check -fcheck=all -ffree-line-length-none  -Wall -Wno-unused-dummy-argument
libs = -lblas

objects = module_structures.o module_global.o module_constants.o module_timing.o\
experiment.o\
mylapack.o\
assign_values_to_qpoints.o\
basis_functions_V.o basis_functions_P.o basis_functions_T.o\
compute_elemental_matrices.o compute_dNdx_dNdy.o compute_dNdx_dNdy_dNdz.o\
output_mesh.o output_swarm.o output_solution.o output_qpoints.o\
interpolate_onto_nodes.o\
make_matrix.o\
markers_setup.o\
postprocessors.o\
prescribe_stokes_solution.o\
quadrature_setup.o\
setup_cartesian2D.o setup_cartesian3D.o\
solve_stokes.o\
elefant.o

elefant: $(objects)
	$(f90comp) -o elefant $(switch) $(objects) $(libs)

%.o: %.f90
	$(f90comp) -c $(switch) $<
%.o: %.f
	$(f90comp) -c  $<

clean:
	rm -f *.mod 
	rm -f *.o 
	rm -f elefant
	rm -f *.vtu
	rm -f opla

