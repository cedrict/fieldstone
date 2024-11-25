import numpy as np


N2_file=open('N2.ascii',"w")
N4_file=open('N4.ascii',"w")
N6_file=open('N6.ascii',"w")

n=100

for i in range(0,n):

    R=1+i/(n-1)*6

    xi2=1.-1./R

    s=1
    ratio=1/(1-xi2**(s))
    N2_1=1.+2*ratio*xi2

    s=2
    ratio=1/(1-xi2**(s))
    N2_2=1.+2*ratio*xi2

    s=3
    ratio=1/(1-xi2**(s))
    N2_3=1.+2*ratio*xi2

    s=5
    ratio=1/(1-xi2**(s))
    N2_5=1.+2*ratio*xi2

    s=100
    ratio=1/(1-xi2**(s))
    N2_100=1.+2*ratio*xi2

    N2_file.write("%e %e %e %e %e\n" % (R,N2_1,N2_2,N2_5,N2_100))
    ############################################# 
  
    s=2
    ratio=1/(1-xi2**(s))
    N4_2=N2_2+2*ratio*(1.-17./24.*ratio)*xi2**2

    s=3
    ratio=1/(1-xi2**(s))
    N4_3=N2_3+2*ratio*(1.-17./24.*ratio)*xi2**2

    s=100
    ratio=1/(1-xi2**(s))
    N4_100=N2_100+2*ratio*(1.-17./24.*ratio)*xi2**2

    N4_file.write("%e %e %e %e\n" % (R,N4_2,N4_3,N4_100))

    ############################################# 

    s=3
    ratio=1/(1-xi2**(s))
    N6_3=N4_3+2*ratio*(1-17./12*ratio+191/288*ratio**2)*xi2**3

    s=100
    ratio=1/(1-xi2**(s))
    N6_100=N4_100+2*ratio*(1-17./12*ratio+191/288*ratio**2)*xi2**3

    N6_file.write("%e %e %e\n" % (R,N6_3,N6_100))
    


