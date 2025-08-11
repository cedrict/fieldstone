.SUFFIXES:.out .o .s .c .F .f .f90 .e .r .y .yr .ye .l .p .sh .csh .h

default: code

include Makefile.machine

WHERE = MINE


OBJECTS=\
output_for_paraview.o\
output_profiles.o\
output_surface.o\
int_to_char.o\
simplefem.o

OOBJECTS=\
$(WHERE)/material_model.o\
$(WHERE)/define_bc.o\
$(WHERE)/temperature_layout.o\
$(WHERE)/material_layout.o

OBJECTSo=\
material_model.o\
define_bc.o\
temperature_layout.o\
material_layout.o


.f.o:
	$(F90) $(FLAGS) $(INCLUDE) $*.f
.f90.o:
	$(F90) $(FLAGS) $(INCLUDE) $*.f90

code:	$(OBJECTS) $(OOBJECTS)
	$(F90) $(OPTIONS) $(OBJECTS) $(OBJECTSo) $(LIBS) -o simplefem

