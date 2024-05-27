
include Makefile.exp

f90comp = gfortran
switch =  -fbounds-check -fcheck=all -ffree-line-length-none  -Wall -Wno-unused-dummy-argument -cpp #-DUseMUMPS
#switch = -cpp 
#switch =  -ffree-line-length-none -O3 -cpp #-DUseMUMPS
#METIS    = /home/thieulot/work/LIBRARIES/metis-5.1.0/build/Linux-x86_64/libmetis 

MUMPS  = /home/cedrict/work/fieldstone/ELEFANT/MUMPS/MUMPS_5.6.2/lib
MUMPS2 = /home/cedrict/work/fieldstone/ELEFANT/MUMPS/MUMPS_5.6.2/libseq
PORD   = /home/cedrict/work/fieldstone/ELEFANT/MUMPS/MUMPS_5.6.2/PORD
LAPACK = /home/cedrict/work/fieldstone/ELEFANT/MUMPS/lapaack/lapack-build/lib

libs = \
-L$(MUMPS) -ldmumps -lmumps_common\
-L$(MUMPS2) -lmpiseq \
-L$(PORD) -lpord \
-L$(LAPACK) -llapack\
-lblas 

objects = module_mesh.o module_parameters.o module_constants.o module_timing.o module_arrays.o\
module_statistics.o module_gravity.o module_sparse.o module_materials.o module_swarm.o module_MUMPS.o\
module_quadrature.o module_export_vtu.o\
assemble_K.o\
assemble_S.o\
assemble_GT.o\
assemble_MP.o\
assemble_RHS.o\
assign_values_to_qpoints.o allocate_memory.o\
compute_elemental_matrix_stokes.o compute_timestep.o compute_abcd.o\
compute_dNdx_dNdy.o compute_dNdx_dNdy_dNdz.o compute_temperature_gradient.o\
compute_dNTdx_dNTdy.o compute_dNTdx_dNTdy_dNTdz.o compute_belongs.o\
compute_elemental_rho_eta_vol.o compute_gravity.o compute_elemental_matrix_energy.o \
compute_block_scaling_coefficient.o compute_elemental_strain_rate.o\
dgesl.o dgeco.o dgefa.o \
directsolver.o\
estimate_memory_use.o\
extract_K_diagonal.o\
header.o\
impose_boundary_conditions_stokes.o\
impose_boundary_conditions_energy.o\
int_to_char.o\
initialise_mumps_V.o\
initialise_elements.o\
inner_solver.o\
locate_point.o\
make_matrix_stokes.o\
make_matrix_energy.o \
mgmres.o \
NNN.o dNNNdr.o dNNNds.o dNNNdt.o\
output_mesh.o output_swarm.o output_solution.o output_qpoints.o \
output_matrix_tikz.o outer_solver_pcg.o output_matrix_for_paraview.o\
postprocessors.o process_bc.o\
prescribe_stokes_solution.o\
paint_swarm.o\
pcg_solver_csr.o\
process_inputs.o\
quadrature_setup.o\
recover_pressure_penalty.o read_command_line_options.o\
sloan.o\
set_default_values.o swarm_setup.o\
set_global_parameters_spaceV.o\
set_global_parameters_spaceP.o\
set_global_parameters_spaceT.o\
set_global_parameters_mapping.o\
setup_cartesian2D.o \
setup_cartesian3D.o \
setup_annulus.o \
setup_john.o setup_mapping.o \
setup_GT_matrix_CSR.o\
setup_A.o\
setup_GT.o\
setup_MP_and_S.o\
setup_K.o\
setup_K_matrix_FULL.o\
setup_K_matrix_MUMPS.o\
setup_K_matrix_CSR.o\
setup_K_matrix_COO.o\
setup_K_blocks_CSR.o\
setup_K_blocks_FULL.o\
test_basis_functions.o\
solve_stokes.o\
solve_stokes_SIMPLE.o\
solve_energy.o\
spmv_kernels.o \
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
	rm -f OUTPUT/ASCII/*.ascii
	rm -f OUTPUT/GRAVITY/*.ascii
	rm -f OUTPUT/STATS/*.ascii
	rm -f OUTPUT/STATS/*.pdf
	rm -f OUTPUT/MATRIX/*.vtu
	rm -f opla
	rm -f fort.*

cleandata:
	rm -f OUTPUT/*.vtu
	rm -f OUTPUT/*.ascii
	rm -f OUTPUT/GRAVITY/*.ascii
	rm -f opla
	rm -f fort.*
	rm statistics*
