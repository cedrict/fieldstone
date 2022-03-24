FC=gfortran
FFLAGS=-std=f2008 -ffree-form -g

all: quilt

enumerates.o: enumerates.f
	$(FC) $(FFLAGS) -c enumerates.f

derivedTypes.o: derivedTypes.f
	$(FC) $(FFLAGS) -c derivedTypes.f

meshData.o: meshData.f derivedTypes.o
	$(FC) $(FFLAGS) -c meshData.f

examples.o: examples.f debug.o stitcher.o patcher.o io.o
	$(FC) $(FFLAGS) -c examples.f

test.o: test.f stitcher.o patcher.o topology.o
	$(FC) $(FFLAGS) -c test.f

io.o: io.f topology.o highQ.o
	$(FC) $(FFLAGS) -c io.f

topology.o: topology.f enumerates.o derivedTypes.o meshData.o
	$(FC) $(FFLAGS) -c topology.f

highQ.o: highQ.f meshData.o enumerates.o
	$(FC) $(FFLAGS) -c highQ.f

debug.o: debug.f meshData.o
	$(FC) $(FFLAGS) -c debug.f

patcher.o: patcher.f derivedTypes.o enumerates.o meshData.o stitcher.o
	$(FC) $(FFLAGS) -c patcher.f

stitcher.o: stitcher.f derivedTypes.o meshData.o enumerates.o
	$(FC) $(FFLAGS) -c stitcher.f

quilt.o: quilt.f enumerates.o examples.o
	$(FC) $(FFLAGS) -c quilt.f

quilt: enumerates.o derivedTypes.o examples.o meshData.o test.o topology.o debug.o highQ.o io.o patcher.o stitcher.o quilt.o
	$(FC) $(FFLAGS) -o quilt enumerates.o derivedTypes.o meshData.o examples.o test.o topology.o debug.o highQ.o io.o patcher.o stitcher.o quilt.o

clean:
	rm -f *.o *.mod quilt

release:
	rm Quilt -rf
	mkdir Quilt
	cp -r makefile *.f Quilt
	zip -r quilt.zip Quilt
	rm Quilt -rf
