FF     = gfortran -ffixed-line-length-none -fno-automatic
CC     = gcc 

MAKE   = make
BINDIR = ../bin

all: calc colgrep depmaphj_jr distaz2latlon mdlwellln_jr mkcut_jr mkrb_jr raw2xyz_jr

calc:
	${CC} -o $(BINDIR)/calc calc.c -lm
colgrep:
	${CC} -o $(BINDIR)/colgrep colgrep.c -lm
distaz2latlon:
	${CC} -o $(BINDIR)/distaz2latlon distaz2latlon.c -lm
depmaphj_jr:
	${FF} -o $(BINDIR)/depmaphj_jr depmaphj_jr.f ../lib/libS20.a
mdlwellln_jr:
	${FF} -o $(BINDIR)/mdlwellln_jr mdlwellln_jr.f ../lib/libS20.a
mkcut_jr:
	${FF} -o $(BINDIR)/mkcut_jr mkcut_jr.f 
mkrb_jr:
	${FF} -o $(BINDIR)/mkrb_jr mkrb_jr.f 
raw2xyz_jr:
	${FF} -o $(BINDIR)/raw2xyz_jr raw2xyz_jr.f 
