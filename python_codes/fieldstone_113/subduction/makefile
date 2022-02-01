all:
	rm -f *.og
	sed -i 's/\t/    /g' subductionGeo.f
	gfortran -std=f2008 -ffree-form -o subductionGeo subductionGeo.f
	chmod +x subductionGeo
	./subductionGeo
	gmsh -3 subduction.geo
