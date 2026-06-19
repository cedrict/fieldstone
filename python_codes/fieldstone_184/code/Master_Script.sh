#!/usr/bin/env bash

#Master Script to compile the code, run it and perform the plotting and statistical anaylsis 

make

mkdir -p OUT

./simplefem 

python Plot_Norms.py

python Plot_Norms_Strain.py

python Regressions.py

python Regressions_Strain.py