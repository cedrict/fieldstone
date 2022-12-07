      program grid
      Implicit Double Precision (a-h,o-z)
      include 'grid.params'
      dimension xmid(1001,2),anglemid(1001),curvmid(1001),
     &  xout(1001,2),arcmid(1001)
      zero = 0.0d0
      one = 1.0d0
      half = 0.5d0
      pi = 4.0*datan(one)

      open (21,file='grid.inp')
        read(21,*) ellplate ! length of plate in units of h 
        read(21,*) ellslab  ! length of slab in units of h
        read(21,*) dipend_deg ! dip of the end of the slab in degrees
        read(21,*) doverh  ! dim'less lubrication layer thickness     
        read(21,*) nperh ! number of elements per length h of the sheet
        read(21,*) mend ! number of elements on each endpiece  
      close (21)
      dipend = dipend_deg*pi/180.0

      nint_slab = ellslab*nperh 
      nint_plate = ellplate*nperh
      nint_midsurf = nint_slab + nint_plate
      npt_plate = nint_plate + 1
      npt_slab = nint_slab + 1
      npt_midsurf = nint_plate + nint_slab + 1
      npt_tot = 2*(nint_slab + nint_plate + mend)
      darc = one/nperh 

      print *,nint_slab
      print *,nint_plate
      print *,nint_midsurf
      print *,npt_plate
      print *,npt_slab
      print *,npt_midsurf
      print *,npt_tot
      print *,darc


c  Coordinates of points on slab midsurface
      call slab_midsurface(xmid,anglemid,curvmid)

      do i = 1,101
         print *,xmid(i,1),xmid(i,2)
      end do


c  Coordinates of all points on midsurface
      call full_midsurface(xmid,anglemid,curvmid)

c  Coordinates of outer boundary of the sheet
      call grid_outer(xout,xmid,anglemid) 

      open (31,file='xmid.data')
        do i = 1, npt_midsurf
          write(31,1000) xmid(i,1),xmid(i,2)
        enddo
      close (31)

      open (31,file='xout.data')
        do i = 1, npt_tot      
          write(31,1000) xout(i,1),xout(i,2)
        enddo
      close (31)

 1000 format(6(1pe13.5))
      stop     
      end

c **********************************************************************
c **********************************************************************

      subroutine grid_outer(xout,xmid,anglemid) 
c  Determines coordinates of all points on the boundary
c  of the sheet

      Implicit Double Precision (a-h,o-z)
      include 'grid.params'

      Dimension xout(1001,2),xmid(1001,2),anglemid(1001)

c NB: element number increases counterclockwise from the ridge

c Endpiece at left end   
      x1c = zero
      x2c = - half - doverh
      dph = pi/mend
      orient = pi
      do i = 1, mend 
        ipr = i
        ph = -pi/2 + (i - 1)*dph
        call endpiece(x1c,x2c,half,ph,orient,
     &                xout(ipr,1),xout(ipr,2))
      enddo

c  Lower surface
      do i = 1, npt_midsurf
        ipr = i+mend
        xout(ipr,1) = xmid(i,1) + half*dsin(anglemid(i))
        xout(ipr,2) = xmid(i,2) - half*dcos(anglemid(i))
      enddo

c Endpiece at right end 
      x1c = xmid(npt_midsurf,1)
      x2c = xmid(npt_midsurf,2)
      dph = pi/mend
      orient = - dipend
      do i = 1, mend
        ipr = i + mend + npt_midsurf  
        ph = - pi/2 + (i - 1)*dph
        call endpiece(x1c,x2c,half,ph,orient,
     &                xout(ipr,1),xout(ipr,2))
      enddo

c  Upper surface
      do i = npt_midsurf, 2, -1
        ipr = 2*(1 + mend + nint_midsurf) - i
        xout(ipr,1) = xmid(i,1) - half*dsin(anglemid(i))
        xout(ipr,2) = xmid(i,2) + half*dcos(anglemid(i))
      enddo

      Return
      End

c***********************************************************************
c***********************************************************************

      subroutine endpiece(x1c,x2c,rad0,ph,orient,x1,x2)
c  Cartesian coordinates (x1,x2) of a point on the endpiece
c  ph \in [-pi/2,pi/2] = local angle within the endpiece (= 0 on symmetry axis)
c  orient \in [-pi, pi] = angle of symmetry axis relative to horizontal
      implicit real*8 (a-h,o-z)
      include 'grid.params'
      ph0 = pi
      theta = ph + orient
      rad02 = rad0**2
      ph2 = ph**2
      ph4 = ph**4
      ph6 = ph**6
      ph8 = ph**8
      ph02 = ph0**2
      ph04 = ph0**4
      ph06 = ph0**6
      ph08 = ph0**8
      rad = ((16*ph4 + 32*ph02 - 8*ph2*ph02
     &    + ph04)*rad0)/(32.*ph02)
      x1 = x1c + rad*dcos(theta)
      x2 = x2c + rad*dsin(theta)
      return
      end

c********************************************************************
c********************************************************************

      subroutine full_midsurface(xmid,anglemid,curvmid)
c  Determines coordinates xmid(i,1-2) and inclination anglemid(i)
c  of all points on the midsurface
c  Also places the midsurface within the plate 
c  at the dimensionless depth (-1/2 - d/h)
      implicit real*8 (a-h,o-z)
      include 'grid.params'
      dimension xmid(1001,2),anglemid(1001),curvmid(1001),
     & arcmid(1001),xmid_sv(1001,2),angle_sv(1001)
c  Points on slab midsurface
      do i = 1, npt_slab       
        xmid_sv(i,1) = xmid(i,1)
        xmid_sv(i,2) = xmid(i,2)
        angle_sv(i) = anglemid(i)
      enddo
      do i = 1, npt_slab     
        xmid(i+npt_plate-1,1) = ellplate + xmid_sv(i,1)
        xmid(i+npt_plate-1,2) = -doverh - half + xmid_sv(i,2)
        anglemid(i+npt_plate-1) = angle_sv(i)
      enddo
c  Points on the plate midsurface
      do i = 1, nint_plate
        xmid(i,1) = (i-1)*darc 
        xmid(i,2) = -doverh - half
        anglemid(i) = zero
      enddo
      return
      end

c********************************************************************
c********************************************************************

      subroutine slab_midsurface(xmid,anglemid,curvmid)
c  Determines coordinates xmid(i,1-2) of points
c  on the midsurface of the slab
      implicit real*8 (a-h,o-z)
      include 'grid.params'
      dimension xmid(1001,2),arc(1001),dip(1001),q(2),dqdx(2),
     & anglemid(1001),curvmid(1001),arcmid(1001)
c Starting coordinates (measured from the trench
c and the depth of the midsurface)
      q(1) = zero      
      q(2) = zero            
      xmid(1,1) = q(1)
      xmid(1,2) = q(2)
      arcmid(1) = zero
      anglemid(1) = zero
      curvmid(1) = zero
      dip(1) = zero
      do k = 2, npt_slab
        call derivs(arcmid(k-1),q,dip(k-1),curv,dqdx)
        call rk4(q,dqdx,2,arc(k-1),darc,q)
        xmid(k,1) = q(1)
        xmid(k,2) = q(2)
        arc(k) = arc(k-1) + darc
        call derivs(arc(k),q,dip(k),curv,dqdx)
        anglemid(k) = -dip(k)
        curvmid(k) = curv
      enddo 
      return
      end

c***********************************************************************
c***********************************************************************

      subroutine derivs(x,q,dip,curv,dqdx)
      implicit double precision (a-h,o-z)
      include 'grid.params'
      dimension q(2), dqdx(2)
      dip = dipend*x*x*(3*ellslab - 2*x)/ellslab**3
      curv = - (6*dipend*(ellslab - x)*x)/ellslab**3
      dqdx(1) = dcos(dip)
      dqdx(2) = -dsin(dip)
      return
      end

c***********************************************************************
c***********************************************************************

      subroutine rk4(y,dydx,n,x,h,yout)
      implicit double precision (a-h,o-z)
      dimension dydx(0:n),y(0:n),yout(0:n)
      PARAMETER (NMAX=50)
      dimension dym(0:NMAX),dyt(0:NMAX),yt(0:NMAX)
      hh=h*0.5
      h6=h/6.
      xh=x+hh
      do 11 i=1,n
        yt(i)=y(i)+hh*dydx(i)
 11   continue
      call derivs(xh,yt,dip,curv,dyt)
      do 12 i=0,n
        yt(i)=y(i)+hh*dyt(i)
 12   continue
      call derivs(xh,yt,dip,curv,dym)
      do 13 i=0,n
        yt(i)=y(i)+h*dym(i)
        dym(i)=dyt(i)+dym(i)
 13   continue
      call derivs(x+h,yt,dip,curv,dyt)
      do 14 i=0,n
        yout(i)=y(i)+h6*(dydx(i)+dyt(i)+2.*dym(i))
 14   continue
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software *G.#Y5D]j3#Q,.
