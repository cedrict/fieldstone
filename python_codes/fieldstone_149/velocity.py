import numpy as np

def compute_corner_flow_velocity(x,y,l1,l2,l3,angle,v0,Lx,Ly):
    v1=-v0
    theta0=angle
    theta1=np.pi-theta0
    l4=l3*np.tan(theta0)
    A0 = (- theta0 * np.sin(theta0))/(theta0**2-np.sin(theta0)**2 ) *v0 
    B0=0
    C0=(np.sin(theta0)-theta0*np.cos(theta0))/(theta0**2-np.sin(theta0)**2 ) * v0
    D0=-A0
    A1 =1./(theta1**2-np.sin(theta1)**2 ) * \
        ( -v0*theta1*np.sin(theta1)-v1*theta1*np.cos(theta1)*(np.sin(theta1)+theta1*np.cos(theta1))\
        +v1*(np.cos(theta1)-theta1*np.sin(theta1))*theta1*np.sin(theta1) )   
    B1=0
    C1=1./(theta1**2-np.sin(theta1)**2 ) * \
       ( v0*(np.sin(theta1)-theta1*np.cos(theta1)) + v1*theta1**2*np.cos(theta1)*np.sin(theta1) \
       - v1*(np.cos(theta1)-theta1*np.sin(theta1))*(np.sin(theta1)-theta1*np.cos(theta1)) )   
    D1=-A1

    u=0.
    v=0.

    #------------------------
    # slab left 
    #------------------------
    if y>=Ly-l1 and x<=l3:
       u=v0
       v=0.

    #------------------------
    # slab 
    #------------------------
    if x>=l3 and y<=Ly+l4-x*np.tan(theta0) and y>=Ly+l4-x*np.tan(theta0)-l1:
       u=v0*np.cos(theta0)
       v=-v0*np.sin(theta0)

    #------------------------
    # overriding plate
    #------------------------
    if y>Ly+l4-x*np.tan(theta0) and y>Ly-l2:
       u=0.0
       v=0.0

    #------------------------
    # wedge
    #------------------------
    xC=l3+l2/np.tan(theta0)
    yC=Ly-l2
    if x>xC and y<yC:
       xt=x-xC 
       yt=yC-y 
       theta=np.arctan(yt/xt) 
       r=np.sqrt((xt)**2+(yt)**2)
       if theta<theta0:
          # u_r=f'(theta)
          ur = A0*np.cos(theta)-B0*np.sin(theta) +\
               C0* (np.sin(theta)+theta*np.cos(theta)) + D0 * (np.cos(theta)-theta*np.sin(theta))
          # u_theta=-f(theta)
          utheta=- ( A0*np.sin(theta) + B0*np.cos(theta) + C0*theta*np.sin(theta) + D0*theta*np.cos(theta))
          ur=-ur
          utheta=-utheta
          u=  ur*np.cos(theta)-utheta*np.sin(theta)
          v=-(ur*np.sin(theta)+utheta*np.cos(theta)) # because of reverse orientation

    #------------------------
    # under subducting plate
    #------------------------
    xD=l3
    yD=Ly-l1
    if y<yD and y<Ly+l4-x*np.tan(theta0)-l1:
       xt=xD-x 
       yt=yD-y 
       theta=np.arctan2(yt,xt) #!; write(6548,*) theta/pi*180
       r=np.sqrt((xt)**2+(yt)**2)
       #u_r=f'(theta)
       ur = A1*np.cos(theta) - B1*np.sin(theta) + C1* (np.sin(theta)+theta*np.cos(theta)) \
            + (D1-v1) * (np.cos(theta)-theta*np.sin(theta))
       #u_theta=-f(theta)
       utheta=- ( A1*np.sin(theta) + B1*np.cos(theta) + C1*theta*np.sin(theta) + (D1-v1)*theta*np.cos(theta))
       ur=-ur
       utheta=-utheta
       u=-(ur*np.cos(theta)-utheta*np.sin(theta))
       v=-(ur*np.sin(theta)+utheta*np.cos(theta)) #! because of reverse orientation

    return u,v

