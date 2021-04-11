

objects = module_structures.o\
experiment.o\
mylapack.o\
assign_values_to_qpoints.o\
basis_functions_V.o\
basis_functions_P.o\
basis_functions_T.o\
compute_elemental_matrices.o\
export_mesh.o\
export_swarm.o\
markers_setup.o\
output_solution.o\
output_qpoints.o\
postprocessors.o\
setup_cartesian2D.o\
elefant.o
f90comp = gfortran
switch = -O3 -fbounds-check -fcheck=all -ffree-line-length-none  -Wall -Wno-unused-dummy-argument
libs = -lblas

elefant: $(objects)
	$(f90comp) -o elefant $(switch) $(objects) $(libs)

#structures.mod: module_structures.o module_structures.f90
#	$(f90comp) -c $(switch) module_structures.f90

#elefant.o: elefant.f90
#	$(f90comp) -c $(switch) elefant.f90

%.o: %.f90
	$(f90comp) -c $(switch) $<

clean:
	rm -f structures.mod 
	rm -f constants.mod 
	rm -f global_parameters.mod 
	rm -f *.o 
	rm -f elefant
	rm -f *.vtu
	rm -f opla

