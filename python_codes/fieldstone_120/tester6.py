import numpy as np
import FEquadrature as Q
import FEbasis2D as FE

for space in 'Q1','Q1+','Q2','Q2s','Q3','Q4','P1','P2','P1+','P2+','P3','DSSY1','DSSY2','RT1','RT2','Han','P1NC':
    print('*****-> '+space)
    FE.visualise_nodes(space)
