f90comp = gfortran
switch =  -fbounds-check -fcheck=all -ffree-line-length-none  -Wall -Wno-unused-dummy-argument -cpp #-DUseMUMPS
libs = -lblas 

objects = module_structures.o module_global.o module_constants.o module_timing.o module_global_arrays.o\
experiment_donea_huerta.o\
mylapack.o\
assign_values_to_qpoints.o\
basis_functions_V.o basis_functions_P.o basis_functions_T.o\
compute_elemental_matrices.o compute_dNdx_dNdy.o compute_dNdx_dNdy_dNdz.o\
output_mesh.o output_swarm.o output_solution.o output_qpoints.o\
header.o\
impose_boundary_conditions.o initialise_mumps_V.o interpolate_onto_nodes.o\
locate_point.o\
make_matrix.o matrix_setup_K.o matrix_setup_GT.o matrix_setup_M.o\
postprocessors.o\
prescribe_stokes_solution.o paint_swarm.o pcg_solver_csr.o\
quadrature_setup.o\
recover_pressure_penalty.o\
setup_cartesian2D.o setup_cartesian3D.o\
set_default_values.o set_global_parameters_pair.o swarm_setup.o\
test_basis_functions.o\
solve_stokes.o spmv_kernels.o solve_KVeqf.o\
write_stats.o\
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
	rm -f OUTPUT/*.vtu
	rm -f OUTPUT/*.ascii
	rm -f opla

