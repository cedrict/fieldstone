.SUFFIXES:.out .o .s .c .F .f .f90 .e .r .y .yr .ye .l .p .sh .csh .h

F90=gfortran 
FLAGS= -c -O3 -ffree-line-length-none
FLAGS= -c -ffree-line-length-none -Wall -fPIC -fmax-errors=1 -g -fcheck=all -fbacktrace -fbounds-check -ffpe-trap= 

OBJECTS2D =\
blas_routines.o\
linpack_d.o\
analytical_solution.o\
compute_errors.o\
output_for_paraview.o\
solve_uzawa1.o\
solve_uzawa2.o\
solve_uzawa3.o\
solve_linpack.o\
inverse_icon.o\
elemental_to_nodal.o\
compute_derivatives_errors.o\
simplefem.o 

.f.o:
	$(F90) $(FLAGS) $(INCLUDE) $*.f
.f90.o:
	$(F90) $(FLAGS) $(INCLUDE) $*.f90

code:	$(OBJECTS2D)
	$(F90) $(OPTIONS) $(OBJECTS2D) $(LIBS) -o simplefem

clean: 
	rm -f *.o
	rm -f *.dat
	rm -f fort.*
	rm -f OUT/*.dat
	rm -f OUT/*.vtu
	rm -f simplefem



