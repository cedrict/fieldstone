      Program mkcut_jr

c-- Gives a series of points in lat lon,
c-- evenly sampled along a great circle.
c-- Jeroen Ritsema: now made with standard in and output
c-- This program is used in slice_180, for making cross-sections
c-- through S20RTS

      parameter (MAXP=5000)
      dimension pnts(2,MAXP)


      igrc=1

      read(5,*) elat,elon
      read(5,*) stlat,stlon
      read(5,*) ddeg,nlatsamp
      
c     nlatsamp = int(180./ddeg) + 1

      call cgrc(elat,elon,stlat,stlon,ddeg,pnts,nrp,igrc)

      do i=1,nlatsamp
        write(6,*) (i-1)*ddeg,pnts(2,i),pnts(1,i)
      enddo

 
      end

c -----------------------------------------------------------------
      subroutine cgrc(ths,phs,thr,phr,ddeg,pnts,nrp,igrc)

c     This subroutine calculates points every ddeg degrees along a
c     great circle path from source to receiver
c     ths = elat, phs = elon, thr = stlat, phr = stlon

      parameter (maxp=5000)
      dimension pnts(2,maxp)

      qpi=atan(1.)
      hpi=2.*qpi
      tpi=8.*qpi

c    transform to xyz coord.
      call cxyz(ths,phs,xs,ys,zs)
      call cxyz(thr,phr,xr,yr,zr)

c     write(6,*) ths,phs,xs,ys,zs
c     write(6,*) thr,phr,xr,yr,zr

c    calc. angle between two vectors
      xsixr=xs*xr+ys*yr+zs*zr
      omr=acos(xsixr)
c     write(6,*) 'omr= ',omr

c    calc. vector perpend. to xs in greatc plane
      f1=xs*xs+ys*ys+zs*zs
c     write(6,*) 'length = ',f1
      a=-xsixr
      xd=xr+a*xs
      yd=yr+a*ys
      zd=zr+a*zs
      xdixs=xd*xs+yd*ys+zd*zs
c     write(6,*) 'xdixs =',xdixs
      rl=1./(sqrt(xd*xd+yd*yd+zd*zd))
c     write(6,*) 'rl =',rl

      xb=xd*rl
      yb=yd*rl
      zb=zd*rl

      xbixr=xb*xr+yb*yr+zb*zr
      ang=acos(xbixr)
c     write(6,*) 'ang = ',ang
      If(abs(ang).gt.hpi) Then
       xb=-xb
       yb=-yb
       zb=-zb
      Endif

      if(igrc.eq.1) omr=tpi 
      if(igrc.eq.2) then
       omr=tpi-omr
       xb=-xb
       yb=-yb
       zb=-zb
      endif

c     da=tpi*float(ddeg)/360.
      da=tpi*ddeg/360.
      nrv=aint(omr/da)
      if(omr-(nrv*da).lt.0.001) then
c      write(6,*) omr,nrv,da
       nrv=nrv-1
c      write(6,*) 'nrv=nrv-1'
      endif
      Do i=1,nrv
       d=float(i)*da
       xp=cos(d)*xs+sin(d)*xb
       yp=cos(d)*ys+sin(d)*yb
       zp=cos(d)*zs+sin(d)*zb
       call ctp(xp,yp,zp,thp,php)
       pnts(1,i+1)=php
       pnts(2,i+1)=thp
      Enddo

      nrp=nrv+2
      pnts(1,1)=phs
      pnts(2,1)=ths
      if(igrc.ne.1) then
       pnts(1,nrp)=phr 
       pnts(2,nrp)=thr
      endif

      End

c ---------------------------------------------------------------

      subroutine cxyz(theta,phi,x,y,z)

      tpi=8.*atan(1.)
      ff=tpi/float(360)
      th=theta*ff
      ph=phi*ff

      z=sin(th)
      ct=cos(th)
      x=ct*cos(ph)
      y=ct*sin(ph)

      end

c -----------------------------------------------------------
      subroutine ctp(x,y,z,theta,phi)
c     this subroutine assumes the length of vector (x,y,z) to be 1.

      tpi=8.*atan(1.)
      ff=float(360)/tpi

      th=asin(z)
      ph=atan2(y,x)
      theta=th*ff
      phi=ph*ff

      end

c ---------------------------------------------------------------
