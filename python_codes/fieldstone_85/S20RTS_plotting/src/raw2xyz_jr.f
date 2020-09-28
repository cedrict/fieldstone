      program raw2xyz_jr
c-- See raw2xyz.f
c-- input now from stdin

      parameter (MXLH=50)
      parameter (MXLENY=(MXLH+1)**2)
      parameter (MXWRK=(MXLH+1)*4)
      parameter (DTMN=0.1)
      parameter (NTMX=180/DTMN)

      character*80 rawfl,xyzfl
      dimension d0(MXLENY),d1(MXLENY,NTMX),rotv(MXLENY),dr(MXLENY)
      dimension x(MXLENY)
      dimension xlst(NTMX)
      dimension wk1(MXWRK),wk2(MXWRK),wk3(MXWRK)
      dimension emdfr(MXLH*2)
      complex ang,edf,emdf(MXLH)
      equivalence (emdf,emdfr)

33    format(a80)      
      read(5,33) rawfl
      read(5,33) xyzfl
      read(5,*) dtc
      read(5,*) fm
      read(5,*) lmn, lmx
      if(fm.eq.0) fm=1.
      if(lmx.eq.0)    lmx=MXLH
      if(lmx.gt.MXLH) lmx=MXLH
      read(5,*) rlst

      npmn=(lmn)**2+1
 
c     read in model
      open(23,file=rawfl,status='old')
      read(23,*) lmax
      write(6,*) 'model lmax =',lmax
      if(lmx.lt.lmax) lmax=lmx
      write(6,*) 'plotting up to lmax =',lmax
      np=(lmax+1)**2
      if(lmax.gt.MXLH) stop 'STOP >>>> lmax.gt.MAXLH'
      read(23,'(5e16.8)') (x(i),i=1,np)
      close(23)

c     open xyz output file
      open(23,file=xyzfl,status='unknown')

      tpi=8*atan(1.)
      dg2rad=tpi/360.

c     dtc - theta contour interval
c     ntc - nr of theta points
c     tc0 - theta degrees from poles for northern and southern most points
      ntc=2*int(90./dtc)+1
      if(mod(90.,dtc).eq.0.) ntc=ntc-2
      tc0=.5*(180.-(ntc-1)*dtc)

      xlongst=0.
      xlong1=rlst

c     loop over latitudes
      do it=1,ntc
       xlat=-90.+tc0+(float(it-1)*dtc)
       xlst(it)=xlat
      
       call ylm(xlat,xlongst,lmax,d0,wk1,wk2,wk3)

c      write(6,'(10f10.6)') (d0(i),i=1,np)

c      construct vector d1 with values of Xlm:
       call ylm02xlm (lmax,d0,d1(1,it))
c      write(6,'(10f10.6)') (d1(i),i=1,np)
      enddo

      indxl=0
      do while(xlong2.lt.((xlong1+360.)-dtc))
       xlong2=(indxl*dtc)+xlong1

c      construct rotation vector - this calculates the value
c      of a spherical harmonic ylm rotated over phi = 

       df=(xlong2-xlongst)*dg2rad
       ang=cmplx(0.,df)
       edf=cexp(ang)

c      calculate cexp(i*m*phi)
       emdf(1)=edf
       do i=2,lmax
        emdf(i)=emdf(i-1)*edf
       enddo

c      copy sines and cosines to rotv
       ind=1
       rotv(ind)=1.
       do l=1,lmax
        do m=0,l
         if(m.eq.0) then
          ind=ind+1
          rotv(ind)=1.
         else
          ind=ind+1
          rotv(ind)=emdfr(m*2-1)
          ind=ind+1
          rotv(ind)=emdfr(m*2)
         endif
        enddo
       enddo

c      write(6,'(10f10.6)') (rotv(i),i=1,np)
c      write(6,*)

       do j=1,ntc
        do i=1,np
         dr(i)=d1(i,j)*rotv(i)
        enddo

c       calculate model value
        xm=0.
        do i=npmn,np
         xm=xm+dr(i)*x(i)
        enddo

        write(23,'(2f7.2,x,e15.8)') xlong2,xlst(j),fm*100.*xm

c       call ylm(xlst(j),xlong2,lmax,d0,wk1,wk2,wk3)
c       do i=1,np
c        if((dr(i)-d0(i)).gt.1.e-6) then
c         write(6,*) xlat,xlong2,i,dr(i)-d0(i)
c        endif
c       enddo
       enddo

       indxl=indxl+1
      enddo

c     add the poles:
       call ylm(90.,0.,lmax,d0,wk1,wk2,wk3)

       xm=0.
       do i=npmn,np
        xm=xm+d0(i)*x(i)
       enddo
       write(23,'(2f7.2,x,g15.5)') 0.,90.,fm*100.*xm

       call ylm(-90.,0.,lmax,d0,wk1,wk2,wk3)

       xm=0.
       do i=npmn,np
        xm=xm+d0(i)*x(i)
       enddo
       write(23,'(2f7.2,x,g15.5)') 0.,-90.,fm*100.*xm


      end
c -----------------------------------------------------

      subroutine ylm02xlm (lmax,d0,d1)

      dimension d0(*),d1(*)

c    construct vector d1 with values of Xlm:
      ind=1
      d1(ind)=d0(ind)
      do l=1,lmax
       do m=0,l
        if(m.eq.0) then
         ind=ind+1
         d1(ind)=d0(ind)
        else
         ind=ind+1
         d1(ind)=d0(ind)
         ind=ind+1
         d1(ind)=d0(ind-1)
        endif
       enddo
      enddo

      end

c ----------------------------------------------------------------------

      subroutine ylm(xlat,xlon,lmax,y,wk1,wk2,wk3)
c
      complex temp,fac,dfac
      dimension wk1(1),wk2(1),wk3(1),y(1)
c
c     wk1,wk2,wk3 should be dimensioned at least (lmax+1)*4
c
      data radian/57.2957795/    ! 360./2pi
c
c     transform to spherical coordinates
      theta=(90.-xlat)/radian
      phi=xlon/radian
c
c    loop over l values
      ind=0
      lm1=lmax+1
      do 10 il1=1,lm1
      l=il1-1
      call legndr(theta,l,l,wk1,wk2,wk3)
c
      fac=(1.,0.)
      dfac=cexp(cmplx(0.,phi))
c
c    loop over m values
      do 20 im=1,il1
      temp=fac*cmplx(wk1(im),0.)
      ind=ind+1
      y(ind)=real(temp)
      if(im.eq.1) goto 20
      ind=ind+1
      y(ind)=aimag(temp)
   20 fac=fac*dfac   ! calculates exp(im phi)
c
   10 continue
      return
      end

c --------------------------------------------------------------------
      SUBROUTINE LEGNDR(THETA,L,M,X,XP,XCOSEC)
      DIMENSION X(*),XP(*),XCOSEC(*)
      DOUBLE PRECISION SMALL,SUM,COMPAR,CT,ST,FCT,COT,FPI,X1,X2,X3,
     1F1,F2,XM,TH,DFLOAT
      DATA FPI/12.56637062D0/
      DFLOAT(I)=FLOAT(I)
      SUM=0.D0
      LP1=L+1
      TH=THETA
      CT=DCOS(TH)
      ST=DSIN(TH)
      MP1=M+1
      FCT=DSQRT(DFLOAT(2*L+1)/FPI)
      SFL3=SQRT(FLOAT(L*(L+1)))
      COMPAR=DFLOAT(2*L+1)/FPI
      DSFL3=SFL3
      SMALL=1.D-16*COMPAR
      DO 1 I=1,MP1
      X(I)=0.
      XCOSEC(I)=0.
    1 XP(I)=0.
      IF(L.GT.1.AND.ABS(THETA).GT.1.E-5) GO TO 3
      X(1)=FCT
      IF(L.EQ.0) RETURN
      X(1)=CT*FCT
      X(2)=-ST*FCT/DSFL3
      XP(1)=-ST*FCT
      XP(2)=-.5D0*CT*FCT*DSFL3
      IF(ABS(THETA).LT.1.E-5) XCOSEC(2)=XP(2)
      IF(ABS(THETA).GE.1.E-5) XCOSEC(2)=X(2)/ST
      RETURN
    3 X1=1.D0
      X2=CT
      DO 4 I=2,L
      X3=(DFLOAT(2*I-1)*CT*X2-DFLOAT(I-1)*X1)/DFLOAT(I)
      X1=X2
    4 X2=X3
      COT=CT/ST
      COSEC=1./ST
      X3=X2*FCT
      X2=DFLOAT(L)*(X1-CT*X2)*FCT/ST
      X(1)=X3
      X(2)=X2
      SUM=X3*X3
      XP(1)=-X2
      XP(2)=DFLOAT(L*(L+1))*X3-COT*X2
      X(2)=-X(2)/SFL3
      XCOSEC(2)=X(2)*COSEC
      XP(2)=-XP(2)/SFL3
      SUM=SUM+2.D0*X(2)*X(2)
      IF(SUM-COMPAR.GT.SMALL) RETURN
      X1=X3
      X2=-X2/DSQRT(DFLOAT(L*(L+1)))
      DO 5 I=3,MP1
      K=I-1
      F1=DSQRT(DFLOAT((L+I-1)*(L-I+2)))
      F2=DSQRT(DFLOAT((L+I-2)*(L-I+3)))
      XM=K
      X3=-(2.D0*COT*(XM-1.D0)*X2+F2*X1)/F1
      SUM=SUM+2.D0*X3*X3
      IF(SUM-COMPAR.GT.SMALL.AND.I.NE.LP1) RETURN
      X(I)=X3
      XCOSEC(I)=X(I)*COSEC
      X1=X2
      XP(I)=-(F1*X2+XM*COT*X3)
    5 X2=X3
      RETURN
      END

c ---------------------------------------------------------------------------
