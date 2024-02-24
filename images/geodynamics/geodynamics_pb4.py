import numpy as np

M=6.39e23
R=3389.5e3
I=0.3662*M*R**2

#M=6e24
#R=6371e3
#I=0.3307*M*R**2

for Rc in [1000e3, 1100e3, 1300e3, 1500e3, 1700e3, 1800e3, 1900e3, 2000e3]:

   Delta=32*np.pi**2/45*\
   (Rc**3*(R**5-Rc**5)-Rc**5*(R**3-Rc**3))
   coeff=4*np.pi/3/Delta
   rhoc=coeff*(2/5*(R**5-Rc**5)*M-(R**3-Rc**3)*I)
   rhom=coeff*(-2/5*Rc**5*M + Rc**3*I)

   print('Rc=',Rc,'| rhoc=',rhoc,'| rhom=',rhom)
