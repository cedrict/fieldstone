.SUFFIXES:.out .o .s .c .F .f .f90 .e .r .y .yr .ye .l .p .sh .csh .h

F90=gfortran 
FLAGS= -c -O3 -ffree-line-length-none
#FLAGS= -c -ffree-line-length-none -Wall -Wextra -Wimplicit-interface -fPIC -fmax-errors=1 -g -fcheck=all -fbacktrace -fbounds-check -ffpe-trap=

OBJECTS2D =\
module_structures.o\
output_for_paraview.o\
output_for_paraview_main.o\
output_for_paraview_visu.o\
blas_routines.o\
linpack_d.o\
program.o


.f.o:
	$(F90) $(FLAGS) $(INCLUDE) $*.f
.f90.o:
	$(F90) $(FLAGS) $(INCLUDE) $*.f90

code:	$(OBJECTS2D)
	$(F90) $(OPTIONS) $(OBJECTS2D) $(LIBS) -o fdm

