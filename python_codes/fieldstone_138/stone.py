import numpy as np

epsilon=1e-20

###############################################################################

def NNN(r,s,t):
    N0=0.125*(1.-r)*(1.-s)*(1.-t)
    N1=0.125*(1.+r)*(1.-s)*(1.-t)
    N2=0.125*(1.+r)*(1.+s)*(1.-t)
    N3=0.125*(1.-r)*(1.+s)*(1.-t)
    N4=0.125*(1.-r)*(1.-s)*(1.+t)
    N5=0.125*(1.+r)*(1.-s)*(1.+t)
    N6=0.125*(1.+r)*(1.+s)*(1.+t)
    N7=0.125*(1.-r)*(1.+s)*(1.+t)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

###############################################################################

def compute_B_quadrature(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,mx,my,mz,nqdim):

    Bx=0
    By=0
    Bz=0

    for iq in [-1, 1]:
        for jq in [-1, 1]:
            for kq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/np.sqrt(3)
                sq=jq/np.sqrt(3)
                tq=kq/np.sqrt(3)
                weightq=1.*1.*1.
                JxW=weightq*hx*hy*hz/8

                N=NNN(rq,sq,tq)

                xq=N.dot(x[icon[:]])
                yq=N.dot(y[icon[:]])
                zq=N.dot(z[icon[:]])

                #print(xq,yq,zq)

                #Mxq=sum(NNN(1:gmesh%nv,iq)*gmesh%Mx(inoode(1:gmesh%nv)))
                #Myq=sum(NNN(1:gmesh%nv,iq)*gmesh%My(inoode(1:gmesh%nv)))
                #Mzq=sum(NNN(1:gmesh%nv,iq)*gmesh%Mz(inoode(1:gmesh%nv)))

                Mxq=mx
                Myq=my
                Mzq=mz

                dist2=(xmeas-xq)**2+(ymeas-yq)**2+(zmeas-zq)**2
                dist=np.sqrt(dist2)
                dist3=dist**3
                dist5=dist**5

                Mrr=Mxq*(xmeas-xq)+Myq*(ymeas-yq)+Mzq*(zmeas-zq)

                Bx+=(3.*Mrr/dist2*(xmeas-xq)-Mxq)/dist3*JxW
                By+=(3.*Mrr/dist2*(ymeas-yq)-Myq)/dist3*JxW
                Bz+=(3.*Mrr/dist2*(zmeas-zq)-Mzq)/dist3*JxW

                #Ax=Ax+(Myq*(z-zq)-Mzq*(y-yq))/dist3*JxW
                #Ay=Ay+(Mzq*(x-xq)-Mxq*(z-zq))/dist3*JxW
                #Az=Az+(Mxq*(y-yq)-Myq*(x-xq))/dist3*JxW

                #psi=psi+(Mxq*(x-xq)+ Myq*(y-yq)+ Mzq*(z-zq))/dist3*JxW

                #Thetax_x=(dist2-3.*(x-xq)*(x-xq))/dist5
                #Thetax_y=(     -3.*(x-xq)*(y-yq))/dist5
                #Thetax_z=(     -3.*(x-xq)*(z-zq))/dist5

                #Thetay_x=(     -3.*(y-yq)*(x-xq))/dist5
                #Thetay_y=(dist2-3.*(y-yq)*(y-yq))/dist5
                #Thetay_z=(     -3.*(y-yq)*(z-zq))/dist5

                #Thetaz_x=(     -3.*(z-zq)*(x-xq))/dist5
                #Thetaz_y=(     -3.*(z-zq)*(y-yq))/dist5
                #Thetaz_z=(dist2-3.*(z-zq)*(z-zq))/dist5

                #Hx=Hx-(Mxq*Thetax_x+Myq*Thetay_x+Mzq*Thetaz_x)*JxW
                #Hy=Hy-(Mxq*Thetax_y+Myq*Thetay_y+Mzq*Thetaz_y)*JxW
                #Hz=Hz-(Mxq*Thetax_z+Myq*Thetay_z+Mzq*Thetaz_z)*JxW

            #end for 
        #end for 
    #end for 

    return np.array([Bx,By,Bz])

###############################################################################
#  Subroutine plane computes the intersection (x,y,z) of a plane 
#  and a perpendicular line.  The plane is defined by three points 
#  (x1,y1,z1), (x2,y2,z2), and (x3,y3,z3).  The line passes through 
#  (x0,y0,z0).  Computation is done by a transformation and inverse 
#  transformation of coordinates systems.

def plane(x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3):
    x2n=x2-x1
    y2n=y2-y1
    z2n=z2-z1
    x0n=x0-x1
    y0n=y0-y1
    z0n=z0-z1
    x3n=x3-x1
    y3n=y3-y1
    z3n=z3-z1

    #call cross(x3n,y3n,z3n,x2n,y2n,z2n,cx,cy,cz,c)
    V3=np.array([x3n,y3n,z3n]) 
    V2=np.array([x2n,y2n,z2n]) 
    C=np.cross(V3,V2)
    cx=C[0]
    cy=C[1]
    cz=C[2]
    c=np.sqrt(cx**2+cy**2+cz**2)
    

    #call cross(x2n,y2n,z2n,cx,cy,cz,dx,dy,dz,d)
    D=np.cross(V2,C)
    dx=D[0]
    dy=D[1]
    dz=D[2]
    d=np.sqrt(dx**2+dy**2+dz**2)

    a=np.sqrt(x2n**2+y2n**2+z2n**2)
    t11=x2n/a
    t12=y2n/a
    t13=z2n/a
    t21=cx/c
    t22=cy/c
    t23=cz/c
    t31=dx/d
    t32=dy/d
    t33=dz/d
    tx0=t11*x0n+t12*y0n+t13*z0n
    tz0=t31*x0n+t32*y0n+t33*z0n
    r=t21*x0n+t22*y0n+t23*z0n
    x=t11*tx0+t31*tz0
    y=t12*tx0+t32*tz0
    z=t13*tx0+t33*tz0
    x=x+x1
    y=y+y1
    z=z+z1
    return x,y,z,r

###############################################################################
#  Subroutine LINE determines the intersection (x,y,z) of two 
#  lines.  First line is defined by points (x1,y1,z1) and 
#  (x2,y2,z2).  Second line is perpendicular to the first and 
#  passes through point (x0,y0,z0).  Distance between (x,y,z) 
#  and (x0,y0,z0) is returned as r.  Computation is done by a 
#  transformation of coordinate systems.

def line(x0, y0, z0, x1, y1, z1, x2, y2, z2):
      tx0=x0-x1
      ty0=y0-y1
      tz0=z0-z1
      tx2=x2-x1
      ty2=y2-y1
      tz2=z2-z1
      a=np.sqrt(tx2**2+ty2**2+tz2**2)

      T2 = np.array([tx2,ty2,tz2])
      T0 = np.array([tx0,ty0,tz0])
      C = np.cross(T2,T0)
      cx = C[0]
      cy = C[1]
      cz = C[2]
      c=np.sqrt(cx**2+cy**2+cz**2)

      D = np.cross(C,T2)
      dx = D[0]
      dy = D[1]
      dz = D[2]
      d=np.sqrt(dx**2+dy**2+dz**2)

      tt11=tx2/a
      tt12=ty2/a
      tt13=tz2/a
      tt21=dx/d
      tt22=dy/d
      tt23=dz/d
      tt31=cx/c
      tt32=cy/c
      tt33=cz/c
      u0=tt11*tx0+tt12*ty0+tt13*tz0
      r=tt21*tx0+tt22*ty0+tt23*tz0
      x=tt11*u0+x1
      y=tt12*u0+y1
      z=tt13*u0+z1
      v1=-u0
      v2=a-u0
      return x,y,z,v1,v2,r


###############################################################################
#  Subroutine ROT finds the sense of rotation of the vector 
#  from (ax,ay,az) to (bx,by,bz) with respect to a second 
#  vector through point (px,py,pz).  The second vector has 
#  components given by (nx,ny,nz).  Returned parameter s is 
#  1 if anticlockwise, -1 if clockwise, or 0 if colinear.

def rot(ax,ay,az,bx,by,bz,nx,ny,nz,px,py,pz):
    x=bx-ax
    y=by-ay
    z=bz-az

    #call cross(nx,ny,nz,x,y,z,cx,cy,cz,c)
    N=np.array([nx,ny,nz]) 
    V=np.array([x,y,z]) 
    C=np.cross(N,V)
    cx=C[0]
    cy=C[1]
    cz=C[2]
    c=np.sqrt(cx**2+cy**2+cz**2)

    u=px-ax
    v=py-ay
    w=pz-az
    d=u*cx+v*cy+w*cz

    if d<0:
       s=1
    elif d>0:
       s=-1
    else:
       s=0

    return s

###############################################################################
#  Subroutine FACMAG computes the magnetic field due to surface 
#  charge  on a polygonal face.  Repeated calls can build the 
#  field of an arbitrary polyhedron.  X axis is directed north, 
#  z axis vertical down.  Requires subroutines CROSS, ROT, LINE, 
#  and PLANE.  Algorithm from Bott (1963).
#  Input parameters:
#    Observation point is (x0,y0,z0).  Polygon corners defined 
#    by arrays x, y, and z of length n.  Magnetization given by 
#    mx,my,mz.  Polygon limited to 10 corners.  Distance units
#    are irrelevant but must be consistent; magnetization in A/m. 
#  Output parameters:
#    Three components of magnetic field (fx,fy,fz), in nT.
# dimension u(10),v2(10),v1(10),s(10),xk(10),yk(10),zk(10),xl(10),yl(10),zl(10),x(10),y(10),z(10)

def facmag(mx,my,mz,x0,y0,z0,x,y,z,n):

      #x,y,z of size n+1!

      fx=0.
      fy=0.
      fz=0.

      xl=np.zeros(n,dtype=np.float64)
      yl=np.zeros(n,dtype=np.float64)
      zl=np.zeros(n,dtype=np.float64)
      #------------------------
      for i in range(0,n):
         xl[i]=x[i+1]-x[i]
         yl[i]=y[i+1]-y[i]
         zl[i]=z[i+1]-z[i]
         rl=np.sqrt(xl[i]**2+yl[i]**2+zl[i]**2)
         xl[i]=xl[i]/rl
         yl[i]=yl[i]/rl
         zl[i]=zl[i]/rl
      #end for

      #call cross(xl(2),yl(2),zl(2),xl(1),yl(1),zl(1),nx,ny,nz,rn)
      L1=np.array([xl[1],yl[1],zl[1]])
      L0=np.array([xl[0],yl[0],zl[0]])
      N=np.cross(L1,L0)
      nx=N[0]
      ny=N[1]
      nz=N[2]
      rn=np.sqrt(nx**2+ny**2+nz**2)
      nx=nx/rn
      ny=ny/rn
      nz=nz/rn

      dot=mx*nx+my*ny+mz*nz

      if abs(dot)<epsilon:
         return np.array([0,0,0])

      #call plane(x0,y0,z0,x(1),y(1),z(1),x(2),y(2),z(2),x(3),y(3),z(3),px,py,pz,w)
      px,py,pz,w=plane(x0,y0,z0,x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2])

      #------------------------
      s=np.zeros(n,dtype=np.int)
      u=np.zeros(n,dtype=np.float64)
      xk=np.zeros(n,dtype=np.float64)
      yk=np.zeros(n,dtype=np.float64)
      zk=np.zeros(n,dtype=np.float64)
      v1=np.zeros(n,dtype=np.float64)
      v2=np.zeros(n,dtype=np.float64)
      for i in range(0,n):
          #call rot(x[i],y[i],z[i],x(i+1),y(i+1),z(i+1),nx,ny,nz,px,py,pz,s[i])
          s[i]=rot(x[i],y[i],z[i],x[i+1],y[i+1],z[i+1],nx,ny,nz,px,py,pz)

          if s[i]==0:
             continue

          #call line(px,py,pz,x[i],y[i],z[i],x(i+1),y(i+1),z(i+1),u1,v,w1,v1[i],v2[i],u[i])
          u1,v,w1,v1[i],v2[i],u[i]=line(px,py,pz,x[i],y[i],z[i],x[i+1],y[i+1],z[i+1])
         
          rk=np.sqrt((u1-px)**2+(v-py)**2+(w1-pz)**2)
          xk[i]=(u1-px)/rk
          yk[i]=(v-py)/rk
          zk[i]=(w1-pz)/rk
      #end for

      #------------------------
      for j in range(0,n):
          if s[j]==0:
             continue

          us=u[j]**2
          v2s=v2[j]**2
          v1s=v1[j]**2
          a2=v2[j]/u[j]
          a1=v1[j]/u[j]
          f2=np.sqrt(1.+a2*a2)
          f1=np.sqrt(1.+a1*a1)
          rho2=np.sqrt(us+v2s)
          rho1=np.sqrt(us+v1s)
          r2=np.sqrt(us+v2s+w**2)
          r1=np.sqrt(us+v1s+w**2)
          if abs(w)>epsilon:
             fu2=(a2/f2)*np.log((r2+rho2)/abs(w))-.5*np.log((r2+v2[j])/(r2-v2[j]))
             fu1=(a1/f1)*np.log((r1+rho1)/abs(w))-.5*np.log((r1+v1[j])/(r1-v1[j]))
             fv2=(1./f2)*np.log((r2+rho2)/abs(w))
             fv1=(1./f1)*np.log((r1+rho1)/abs(w))
             fw2=np.arctan2((a2*(r2-abs(w))),(r2+a2*a2*abs(w)))
             fw1=np.arctan2((a1*(r1-abs(w))),(r1+a1*a1*abs(w)))
             fu=dot*(fu2-fu1)
             fv=-dot*(fv2-fv1)
             fw=(-w*dot/abs(w))*(fw2-fw1)
          else:
             fu2=(a2/f2)*(1.+np.log((r2+rho2)/epsilon))-.5*np.log((r2+v2[j])/(r2-v2[j]))
             fu1=(a1/f1)*(1.+np.log((r1+rho1)/epsilon))-.5*np.log((r1+v1[j])/(r1-v1[j]))
             fv2=(1./f2)*(1.+np.log((r2+rho2)/epsilon))
             fv1=(1./f1)*(1.+np.log((r1+rho1)/epsilon))
             fu=dot*(fu2-fu1)
             fv=-dot*(fv2-fv1)
             fw=0.
          #end if
          fx=fx-s[j]*(fu*xk[j]+fv*xl[j]+fw*nx)
          fy=fy-s[j]*(fu*yk[j]+fv*yl[j]+fw*ny)
          fz=fz-s[j]*(fu*zk[j]+fv*zl[j]+fw*nz)
      #end for

      return np.array([fx,fy,fz],dtype=np.float64)

###############################################################################
#     z
#     |
#     4---7---y  
#    /   /
#   5---6
#     |
#     0---3---y  
#    /   /
#   1---2
#  /
# x


def compute_B_surface_integral(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,mx,my,mz):

    Bx=0
    By=0
    Bz=0

    nface=4 #square face duh :)
    xface=np.empty(nface+1)
    yface=np.empty(nface+1)
    zface=np.empty(nface+1)

    #face x=0
    xface[0]=x[icon[7]] ; yface[0]=y[icon[7]] ; zface[0]=z[icon[7]] 
    xface[1]=x[icon[3]] ; yface[1]=y[icon[3]] ; zface[1]=z[icon[3]] 
    xface[2]=x[icon[0]] ; yface[2]=y[icon[0]] ; zface[2]=z[icon[0]] 
    xface[3]=x[icon[4]] ; yface[3]=y[icon[4]] ; zface[3]=z[icon[4]] 
    xface[4]=xface[0]   ; yface[4]=yface[0]   ; zface[4]=zface[0]  
    field=facmag(mx,my,mz,xmeas,ymeas,zmeas,xface,yface,zface,nface)
    Bx+=field[0]
    By+=field[1]
    Bz+=field[2]

    #face x=1
    xface[0]=x[icon[1]] ; yface[0]=y[icon[1]] ; zface[0]=z[icon[1]] 
    xface[1]=x[icon[2]] ; yface[1]=y[icon[2]] ; zface[1]=z[icon[2]] 
    xface[2]=x[icon[6]] ; yface[2]=y[icon[6]] ; zface[2]=z[icon[6]] 
    xface[3]=x[icon[5]] ; yface[3]=y[icon[5]] ; zface[3]=z[icon[5]] 
    xface[4]=xface[0]   ; yface[4]=yface[0]   ; zface[4]=zface[0]  
    field=facmag(mx,my,mz,xmeas,ymeas,zmeas,xface,yface,zface,nface)
    Bx+=field[0]
    By+=field[1]
    Bz+=field[2]

    #face y=0
    xface[0]=x[icon[0]] ; yface[0]=y[icon[0]] ; zface[0]=z[icon[0]] 
    xface[1]=x[icon[1]] ; yface[1]=y[icon[1]] ; zface[1]=z[icon[1]] 
    xface[2]=x[icon[5]] ; yface[2]=y[icon[5]] ; zface[2]=z[icon[5]] 
    xface[3]=x[icon[4]] ; yface[3]=y[icon[4]] ; zface[3]=z[icon[4]] 
    xface[4]=xface[0]   ; yface[4]=yface[0]   ; zface[4]=zface[0]  
    field=facmag(mx,my,mz,xmeas,ymeas,zmeas,xface,yface,zface,nface)
    Bx+=field[0]
    By+=field[1]
    Bz+=field[2]

    #face y=1
    xface[0]=x[icon[2]] ; yface[0]=y[icon[2]] ; zface[0]=z[icon[2]] 
    xface[1]=x[icon[3]] ; yface[1]=y[icon[3]] ; zface[1]=z[icon[3]] 
    xface[2]=x[icon[7]] ; yface[2]=y[icon[7]] ; zface[2]=z[icon[7]] 
    xface[3]=x[icon[6]] ; yface[3]=y[icon[6]] ; zface[3]=z[icon[6]] 
    xface[4]=xface[0]   ; yface[4]=yface[0]   ; zface[4]=zface[0]  
    field=facmag(mx,my,mz,xmeas,ymeas,zmeas,xface,yface,zface,nface)
    Bx+=field[0]
    By+=field[1]
    Bz+=field[2]

    #face z=0
    xface[0]=x[icon[3]] ; yface[0]=y[icon[3]] ; zface[0]=z[icon[3]] 
    xface[1]=x[icon[2]] ; yface[1]=y[icon[2]] ; zface[1]=z[icon[2]] 
    xface[2]=x[icon[1]] ; yface[2]=y[icon[1]] ; zface[2]=z[icon[1]] 
    xface[3]=x[icon[0]] ; yface[3]=y[icon[0]] ; zface[3]=z[icon[0]] 
    xface[4]=xface[0]   ; yface[4]=yface[0]   ; zface[4]=zface[0]  
    field=facmag(mx,my,mz,xmeas,ymeas,zmeas,xface,yface,zface,nface)
    Bx+=field[0]
    By+=field[1]
    Bz+=field[2]

    #face z=1
    xface[0]=x[icon[4]] ; yface[0]=y[icon[4]] ; zface[0]=z[icon[4]] 
    xface[1]=x[icon[5]] ; yface[1]=y[icon[5]] ; zface[1]=z[icon[5]] 
    xface[2]=x[icon[6]] ; yface[2]=y[icon[6]] ; zface[2]=z[icon[6]] 
    xface[3]=x[icon[7]] ; yface[3]=y[icon[7]] ; zface[3]=z[icon[7]] 
    xface[4]=xface[0]   ; yface[4]=yface[0]   ; zface[4]=zface[0]  
    field=facmag(mx,my,mz,xmeas,ymeas,zmeas,xface,yface,zface,nface)
    Bx+=field[0]
    By+=field[1]
    Bz+=field[2]

    return np.array([Bx,By,Bz])

###############################################################################
###############################################################################
###############################################################################

hx=1
hy=1.5
hz=1.9

mx=1
my=1
mz=1

xmeas=3
ymeas=4
zmeas=5

###########################################################

m=8 # number of vertices

x = np.empty(m,dtype=np.float64) 
y = np.empty(m,dtype=np.float64) 
z = np.empty(m,dtype=np.float64) 

x[0]=0*hx ; y[0]=0*hy ; z[0]=0*hz
x[1]=1*hx ; y[1]=0*hy ; z[1]=0*hz
x[2]=1*hx ; y[2]=1*hy ; z[2]=0*hz
x[3]=0*hx ; y[3]=1*hy ; z[3]=0*hz
x[4]=0*hx ; y[4]=0*hy ; z[4]=1*hz
x[5]=1*hx ; y[5]=0*hy ; z[5]=1*hz
x[6]=1*hx ; y[6]=1*hy ; z[6]=1*hz
x[7]=0*hx ; y[7]=1*hy ; z[7]=1*hz

np.savetxt('vertices.ascii',np.array([x,y,z]).T)


#     z
#     |
#     4---7---y  
#    /   /
#   5---6
#     |
#     0---3---y  
#    /   /
#   1---2
#  /
# x


icon =np.zeros(m,dtype=np.int32)
icon[0]=0
icon[1]=1
icon[2]=2
icon[3]=3
icon[4]=4
icon[5]=5
icon[6]=6
icon[7]=7

###########################################################

nqdim=2

B=compute_B_quadrature(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,mx,my,mz,nqdim)

print('quad B:',B)


###########################################################

B=compute_B_surface_integral(xmeas,ymeas,zmeas,x,y,z,icon,hx,hy,hz,mx,my,mz)

print('Surf int B',B)

