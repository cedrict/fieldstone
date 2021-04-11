

objects = module_structures.o\
basis_functions_V.o\
export_mesh.o\
export_swarm.o\
markers_setup.o\
output_solution.o\
output_qpoints.o\
setup_cartesian2D.o\
elefant.o
f90comp = gfortran
switch = -O3 -fbounds-check -fcheck=all -ffree-line-length-none  -Wall -Wno-unused-dummy-argument

elefant: $(objects)
	$(f90comp) -o elefant $(switch) $(objects)

structures.mod: module_structures.o module_structures.f90
	$(f90comp) -c $(switch) module_structures.f90

elefant.o: elefant.f90
	$(f90comp) -c $(switch) elefant.f90

%.o: %.f90
	$(f90comp) -c $(switch) $<

clean:
	rm structures.mod 
	rm global_parameters.mod 
	rm $(objects)
	rm elefant

