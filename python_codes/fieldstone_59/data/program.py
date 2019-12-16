import numpy as np

sigma=0.07e6
d=5e-3

sr_glen=open("sr_glen.ascii","w")
sr_disl=open("sr_disl.ascii","w")
sr_gbs=open("sr_gbs.ascii","w")

for T in np.linspace(230,273,200):

    #glen's flow law
    if T<263:
       A=3.61e5
       n=3
       Q=60e3
    else:
       A=1.73e21
       n=3
       Q=139e3
    #end if
    sr=A * sigma**n * np.exp(-Q/8.314/T) *1e6**(-n)
    sr_glen.write("%5e %5e \n"  %(T,sr))


    #dislocation (indep of d)
    if T<262:
       A=5e5
       n=4
       Q=64.e3
    else:
       A=6.96e23
       n=4
       Q=155.e3
    #end if
    sr=A * sigma**n * np.exp(-Q/8.314/T) *1e6**(-n)
    sr_disl.write("%5e %5e \n"  %(T,sr))

    #GBS-limited creep
    if T<262:
       A=1.1e2
       n=1.8
       Q=70.e3
       p=1.4
    else:
       A=8.5e37
       n=1.8
       Q=250.e3
       p=1.4
    #end if
    sr=A * sigma**n *d**(-p) * np.exp(-Q/8.314/T) *1e6**(-n)
    sr_gbs.write("%5e %5e \n"  %(T,sr))
