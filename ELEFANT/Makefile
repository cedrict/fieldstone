
include Makefile.exp

f90comp = gfortran
switch =  -fbounds-check -fcheck=all -ffree-line-length-none  -Wall -Wno-unused-dummy-argument -cpp #-DUseMUMPS
#switch = -cpp 
#switch =  -ffree-line-length-none -O3 -cpp #-DUseMUMPS
libs = -lblas 


objects = module_mesh.o module_parameters.o module_constants.o module_timing.o module_arrays.o\
module_statistics.o module_gravity.o module_sparse.o module_materials.o module_swarm.o\
mylapack.o\
assign_values_to_qpoints.o\
compute_elemental_matrix_stokes.o compute_timestep.o\
compute_dNdx_dNdy.o compute_dNdx_dNdy_dNdz.o compute_temperature_gradient.o\
compute_dNTdx_dNTdy.o compute_dNTdx_dNTdy_dNTdz.o\
compute_elemental_rho_eta_vol.o compute_gravity.o compute_elemental_matrix_energy.o \
output_mesh.o output_swarm.o output_solution.o output_qpoints.o output_solution_python.o\
header.o\
impose_boundary_conditions_stokes.o impose_boundary_conditions_energy.o int_to_char.o\
initialise_mumps_V.o interpolate_onto_nodes.o initialise_elements.o\
locate_point.o\
make_matrix_stokes.o make_matrix_energy.o \
matrix_setup_K.o matrix_setup_GT.o matrix_setup_MV.o matrix_setup_MP.o matrix_setup_A.o\
mgmres.o mapping_setup.o\
NNN.o dNNNdr.o dNNNds.o dNNNdt.o\
output_matrix_tikz.o\
postprocessors.o project_V_onto_L.o print_parameters.o\
prescribe_stokes_solution.o paint_swarm.o pcg_solver_csr.o\
quadrature_setup.o\
recover_pressure_penalty.o read_command_line_options.o\
setup_cartesian2D.o setup_cartesian3D.o sloan.o\
set_default_values.o set_global_parameters_pair.o swarm_setup.o\
set_global_parameters_spaceV.o\
set_global_parameters_spaceP.o\
set_global_parameters_spaceT.o\
set_global_parameters_mapping.o\
test_basis_functions.o\
solve_stokes.o solve_energy.o spmv_kernels.o solve_KV_eq_f.o\
write_stats.o\
y12m.o\
elefant.o

elefant: $(objects) $(object_experiment) 
	$(f90comp) -o elefant $(switch) $(objects) experiment.o $(libs)

%.o: %.f90
	$(f90comp) -c $(switch) $<
%.o: %.f
	$(f90comp) -c  $<

clean:
	rm -f *.mod 
	rm -f *.o 
	rm -f elefant
	rm -f solu*.pdf
	rm -f OUTPUT/*.vtu
	rm -f OUTPUT/*.ascii
	rm -f OUTPUT/*.py
	rm -f OUTPUT/*.pdf
	rm -f OUTPUT/TEST/*.ascii
	rm -f OUTPUT/TEST/*.pdf
	rm -f OUTPUT/GRAVITY/*.ascii
	rm -f OUTPUT/STATS/*.ascii
	rm -f OUTPUT/STATS/*.pdf
	rm -f opla
	rm -f fort.*

cleandata:
	rm -f OUTPUT/*.vtu
	rm -f OUTPUT/*.ascii
	rm -f OUTPUT/GRAVITY/*.ascii
	rm -f opla
	rm -f fort.*
	rm statistics*
