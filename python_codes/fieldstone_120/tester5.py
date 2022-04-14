import FEbasis2D as FE
import FEquadrature as Q
import FEtools as Tools 
import numpy as np

Lx=3
Ly=2

nelx=3
nely=2

for Vspace in ['Q1','Q1+','Q2','Q3','Q4','Q2s','DSSY1','DSSY2','RT1','RT2','Han','Chen',\
               'P1','P1+','P1NC','P2','P2+','P3','P1+P0']:
    print('=========================================')

    NV,nel,xV,yV,iconV=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Vspace)

    Tools.export_connectivity_array_to_ascii(xV,yV,iconV,'iconV_'+Vspace+'.ascii')

    Tools.export_connectivity_array_elt1_to_ascii(xV,yV,iconV,'iconV_elt1_'+Vspace+'.ascii')

    Tools.visualise_with_tikz(xV,yV,Vspace)

    print(' '+Vspace,': NV =',NV,' nel=',nel)
    print('=========================================')
