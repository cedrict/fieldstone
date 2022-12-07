import numpy as np



#------------------------------------------------------------------------------
#subroutine rk4(y,dydx,n,x,h,yout)
#dimension dydx(0:n),y(0:n),yout(0:n)
#PARAMETER (NMAX=50)
#dimension dym(0:NMAX),dyt(0:NMAX),yt(0:NMAX)

def rk4(y,dydx,n,x,h):

    yt = np.zeros(n,dtype=np.float64)
    dym = np.zeros(n,dtype=np.float64)
    dyt = np.zeros(n,dtype=np.float64)

    for i in range(0,n):
        yt[i]=y[i]+0.5*h*dydx[i]

    #call derivs(x+0.5*h,yt,dip,curv,dyt)
    dyt,dip,curv=derivs(x+0.5*h,)

    for  i in range(0,n):
        yt[i]=y[i]+0.5*h*dyt[i]

    #call derivs(x+0.5*h,yt,dip,curv,dym)

    for i in range(0,n):
        yt[i]=y[i]+h*dym[i]
        dym[i]=dyt[i]+dym[i]

    #call derivs(x+h,yt,dip,curv,dyt)

    for i in range(0,n):
        yout[i]=y[i]+h/6*(dydx[i]+dyt[i]+2*dym[i])

    return yout





#------------------------------------------------------------------------------
#subroutine derivs(x,q,dip,curv,dqdx)
#implicit double precision (a-h,o-z)
#dimension q(2), dqdx(2)

def derivs(x,ellslab,dipend):
    dip = dipend*x*x*(3*ellslab - 2*x)/ellslab**3
    curv = - (6*dipend*(ellslab - x)*x)/ellslab**3
    return np.array([np.cos(dip),-np.sin(dip)]),dip,curv

#------------------------------------------------------------------------------
#      subroutine slab_midsurface(xmid,anglemid,curvmid)
#c  Determines coordinates xmid(i,1-2) of points
#c  on the midsurface of the slab
#      implicit real*8 (a-h,o-z)
#      include 'grid.params'
#dimension xmid(1001,2),arc(1001),dip(1001),q(2),dqdx(2),anglemid(1001),curvmid(1001),arcmid(1001)
#c Starting coordinates (measured from the trench
#c and the depth of the midsurface)

def slab_midsurface(darc,npt_slab):

    q=np.zeros(2,dtype=np.float64)
    dqdx=np.zeros(2,dtype=np.float64)
    xmid=np.zeros((1001,2),dtype=np.float64)
    arc=np.zeros(1001,dtype=np.float64)
    dip=np.zeros(1001,dtype=np.float64)
    anglemid=np.zeros(1001,dtype=np.float64)
    curvmid=np.zeros(1001,dtype=np.float64)
    arcmid=np.zeros(1001,dtype=np.float64)
 
    q[0]        = 0 
    q[1]        = 0            
    xmid[0,0]   = q[0]
    xmid[0,1]   = q[1]
    arcmid[0]   = 0
    anglemid[0] = 0
    curvmid[0]  = 0
    dip[0]      = 0

    for k in range(1,npt_slab):              #do k = 2, npt_slab
    
      dqdx,dip[k-1],curv=derivs(arcmid[k-1]) #call derivs(arcmid(k-1),q,dip(k-1),curv,dqdx)
      #call rk4(q,dqdx,2,arc(k-1),darc,q)
      xmid[k,0] = q[0]
      xmid[k,1] = q[1]
      arc[k] = arc[k-1] + darc

      dqdx,dip[k],curv=derivs(arc[k])        #call derivs(arc(k),q,dip(k),curv,dqdx)
      anglemid[k] = -dip[k]
      curvmidi[k] = curv

    return curvmid,xmid,anglemid











print('------------------------------')
print('          Worm Mesher         ')    
print('------------------------------')
      
#      Implicit Double Precision (a-h,o-z)
#      include 'grid.params'
#      common/cstr/pi,zero,one,half,dipend,ellplate,ellslab,doverh,darc
#      common/csti/nint_slab,nint_plate,nint_midsurf,
#     & npt_slab,npt_plate,npt_midsurf,nperh,mend 


#      dimension xmid(1001,2),anglemid(1001),curvmid(1001),
#     &  xout(1001,2),arcmid(1001)
#      zero = 0.0d0
#      one = 1.0d0
#      half = 0.5d0
#      pi = 4.0*datan(one)


one=1
half=0.5

#------------------------------------------------------------

ellplate = 8.0  # length of plate in units of h 
ellslab  = 4.0  # length of slab in units of h
dipend_deg = 45 # dip of the end of the slab in degrees
doverh = 0.3    # dimensionless lubrication layer thickness     
nperh  = 4      # number of elements per length h of the sheet
mend = 8        # mend (number of elements on each endpiece)

dipend = dipend_deg*np.pi/180

#------------------------------------------------------------

nint_slab = int(ellslab*nperh) 
nint_plate = int(ellplate*nperh)
nint_midsurf = int(nint_slab + nint_plate)
npt_plate = nint_plate + 1
npt_slab = nint_slab + 1
npt_midsurf = nint_plate + nint_slab + 1
npt_tot = int(2*(nint_slab + nint_plate + mend))

darc = one/nperh 

print('nint_slab',nint_slab)
print('nint_plate',nint_plate)
print('nint_midsurf',nint_midsurf)
print('npt_plate',npt_plate)
print('npt_slab',npt_slab)
print('npt_midsurf',npt_midsurf)
print('npt_tot',npt_tot)
print('darc',darc)
print('------------------------------')

xmid = np.zeros((npt_midsurf,2),dtype=np.float64)
xout = np.zeros((npt_tot,2),dtype=np.float64)


#------------------------------------------------------------

#  Coordinates of points on slab midsurface

# = slab_midsurface(xmid,anglemid,curvmid)
    
curvmid=np.array(1001,dtype=np.float64)
anglemid=np.array(1001,dtype=np.float64)
xmid=np.array((1001,2),dtype=np.float64)

curvmid,xmid,anglemid = slab_midsurface(darc,npt_slab)

np.savetxt('xmid.data',np.array([xmid[:,0],xmid[:,1]]).T)
exit()


#  Coordinates of all points on midsurface
#      call full_midsurface(xmid,anglemid,curvmid)

#  Coordinates of outer boundary of the sheet
#      call grid_outer(xout,xmid,anglemid) 

np.savetxt('xmid.data',np.array([xmid[:,0],xmid[:,1]]).T)
np.savetxt('xout.data',np.array([xout[:,0],xout[:,1]]).T)


#------------------------------------------------------------

def grid_outer(xmid,anglemid,npt_midsurf,nint_midsurf,mend):

#  subroutine grid_outer(xout,xmid,anglemid) 
#  Determines coordinates of all points on the boundary
#  of the sheet

   xout = np.array((1001,2),dtype=np.float64)

#c NB: element number increases counterclockwise from the ridge

    # Endpiece at left end   

   x1c = 0 
   x2c = - 0.5 - doverh
   dph = np.pi/mend
   orient = pi
   for i in range(0,mend): 
       ipr = i
       ph = -pi/2 + (i-1)*dph
       #call endpiece(x1c,x2c,half,ph,orient,xout(ipr,1),xout(ipr,2))
       xout[ipr,0],xout[ipr,1]=endpiece(x1c,x2c,half,ph,orient)

   #  Lower surface
   for i in range(0,npt_midsurf):
       ipr = i+mend
       xout[ipr,1] = xmid[i,1] + half*np.sin(anglemid[i])
       xout[ipr,2] = xmid[i,2] - half*np.cos(anglemid[i])

   # Endpiece at right end 
   x1c = xmid(npt_midsurf,1)
   x2c = xmid(npt_midsurf,2)
   dph = pi/mend
   orient = - dipend
   for i in range(0,mend):
       ipr = i + mend + npt_midsurf  
       ph = - pi/2 + (i - 1)*dph
       #call endpiece(x1c,x2c,half,ph,orient,xout(ipr,1),xout(ipr,2))
       xout[ipr,0],xout[ipr,1]=endpiece(x1c,x2c,half,ph,orient)

   #  Upper surface
   #for i in range(npt_midsurf-1, 1, -1):
   #    ipr = 2*(1 + mend + nint_midsurf) - i
   #    xout[ipr,0] = xmid[i,0] - half*np.sin(anglemid[i])
   #    xout[ipr,1] = xmid[i,1] + half*np.cos(anglemid[i])

   return xout

#------------------------------------------------------------

def endpiece(x1c,x2c,rad0,ph,orient):
    # Cartesian coordinates (x1,x2) of a point on the endpiece
    # ph \in [-pi/2,pi/2] = local angle within the endpiece (= 0 on symmetry axis)
    # orient \in [-pi, pi] = angle of symmetry axis relative to horizontal

    ph0 = np.pi
    theta = ph + orient
    ph2 = ph**2
    ph4 = ph**4
    ph6 = ph**6
    ph8 = ph**8
    ph02 = ph0**2
    ph04 = ph0**4
    ph06 = ph0**6
    ph08 = ph0**8
    rad = ((16*ph4 + 32*ph02 - 8*ph2*ph02 + ph04)*rad0)/(32.*ph02)
    x1 = x1c + rad*np.cos(theta)
    x2 = x2c + rad*np.sin(theta)

    return x1,x2

#------------------------------------------------------------

def full_midsurface(xmid,ellplate,doverh,darc,nint_plate,npt_slab):
    # Determines coordinates xmid[i,0-1] and inclination anglemid[i]
    # of all points on the midsurface
    # Also places the midsurface within the plate 
    # at the dimensionless depth (-1/2 - d/h)

    xmid=np.array((1001,2),dtype=np.float64)
    xmid_sv=np.array((1001,2),dtype=np.float64)
    angle_sv=np.array(1001,dtype=np.float64)
    anglemid=np.array(1001,dtype=np.float64)
    curvmid=np.array(1001,dtype=np.float64)
    arcmid=np.array(1001,dtype=np.float64)

    #Points on slab midsurface

    for i in range(0,npt_slab):
        xmid_sv[i,0] = xmid[i,0]
        xmid_sv[i,1] = xmid[i,1]
        angle_sv[i] = anglemid[i]

    for i in range(0,npt_slab):
        xmid[i+npt_plate-1,0] = ellplate      + xmid_sv[i,0]
        xmid[i+npt_plate-1,1] = -doverh - 0.5 + xmid_sv[i,1]
        anglemid[i+npt_plate-1] = angle_sv[i]

    #Points on the plate midsurface

    for i in range(0,nint_plate):
        xmid[i,1] = (i-1)*darc 
        xmid[i,2] = -doverh - 0.5
        anglemid[i] = 0 

    return xmid,anglemid







