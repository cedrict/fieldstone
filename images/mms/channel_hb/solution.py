#################################################################

year=3600.*24.*365.
cm=0.01

n=1.
Pi=-1e4
Pi2=Pi/2.
eta0=1e25 
eps0=1e-17
H=100e3
tau0=9e7
nnx=200

#################################################################

K=(eta0*eps0-tau0)/eps0**n

delta=2*eps0*eta0/abs(Pi)

x1=0.5*H-delta
x2=0.5*H+delta

#################################################################

resfile=open("res.ascii","w")

print("K=",K)
print("delta=",delta)
print('y1=',x1)
print('y2=',x2)
print(-0.5*Pi/eta0*delta,eps0)
 
#################################################################

for i in range(0,nnx):

    x=i*H/(nnx-1)

    exy1=(Pi2/K*(x-x1)+ eps0**n )**(1./n)
    exy2=Pi2/eta0*(x-H/2)
    exy3=-(-Pi2/K*(x-x2)+ eps0**n ) **(1./n)

    u1=2.*n/(n+1)*K/Pi2*(( Pi2/K*(x-x1)+ eps0**n)**(1.+1./n)  - ( -Pi2/K*x1 + eps0**n )**(1.+1./n) )
    u2=Pi2/eta0*(x**2-x*H) + 2.*n/(n+1)*K/Pi2*( eps0**(n+1)-(eps0**n-Pi2/K*x1)**(1+1./n) )- Pi2/eta0*x1*(x1-H)
    u3=2.*n/(n+1)*K/Pi2*( (-Pi2/K*(x-x2)+ eps0**n )**(1.+1./n) - ( -Pi2/K*(H-x2) + eps0**n )**(1.+1./n) )

    if x<x1:
       exy=exy1
       vel=u1
       ee=abs(exy)
       eta_hb=K+tau0/ee
    elif x<x2: 
       exy=exy2
       vel=u2
       ee=abs(exy)
       eta_hb=eta0
    else:
       exy=exy3
       vel=u3
       ee=abs(exy)
       eta_hb=K+tau0/ee
    #end if



    resfile.write("%e %e %e %e %e %e %e %e %e %e %e\n" %(x,exy1,exy2,exy3,exy,u1,u2,u3,vel,ee,eta_hb))

#end for


