      program mdlwellln

      parameter (MXLH=40)
      parameter (MXLENY=(MXLH+1)**2)
      parameter (MXDEP=21)
      parameter (MXSPL=MXDEP+3)
      parameter (MXMSZ=MXSPL*MXLENY)
      parameter (MAXP=1024)

      dimension wk1(MXLENY),wk2(MXLENY),wk3(MXLENY)
      dimension d0(MXLENY)
      dimension maskl(MXLH),masks(MXSPL)
      dimension spl(MXDEP)
      dimension x(MXMSZ)
      dimension spln(MAXP,MXDEP),dep(MAXP),wl(MAXP)
      character*80 getunx,mfl,outfl,lnfl
      character*120 line1

111   format(a80)
c--   model file
      read(5,111) mfl
c--   read model
      open(21,file=mfl,status='old')
      read(21,'(a)') line1
      call rsphhead(line1,lmx,ndmn,ndmx,ndp)
c     write(6,*) lmx,ndmn,ndmx,ndp
      natd=(lmx+1)**2
      ind=(ndmn-1)*natd+1
      do i=ndmn,ndmx
        do j=0,lmx
          ind1=ind+2*j
          read(21,'(11e12.4)')(x(k),k=ind,ind1)
          ind=ind1+1
        enddo
      enddo
      close(21)

c--   lon/lat file
      read(5,111) lnfl
c--   output file
      read(5,111) outfl

c--   minimum and maximum angular order
      read(5,*) lmin,lmax
      if(lmax.lt.lmx) then
        write(6,*) 'truncating harmonic expansion at LMAX=',lmax
      else
        lmax=lmx
      endif
      noutstart=((lmin-1)+1)**2+1
      noutend=(lmax+1)**2
      nmul=noutend-noutstart+1

c--   irad = 0 --> output is depth
c--   irad = 1 --> output is radius
      read(5,*) irad

      dmin =   50
      dmax = 2750
c--   depth interval
      read(5,*) ddep
c--   number of depths
      nrs  = nint ((dmax-dmin)/ddep)+1
c     write(6,*) 'dmax,dmin,ddep,nrs',dmax,dmin,ddep,nrs
      if(nrs.gt.MAXP) stop 'nrs.gt.MAXP'

c--   Model file format = SPH
      imod=1
      iwhole=1
      idptp=1
      ncrtopo=3

      if(lmx.gt.MXLH) stop'lmx.gt.MXLH'
      if(ndmx.gt.MXSPL) stop'ndmx.gt.MXSPL'
      natd=(lmax+1)**2
      nbeg=max(ncrtopo+1,ndmn)
      ntot=ndmx-ncrtopo
c     write(6,*) lmx,nbeg,ndmx,ntot
c     write(6,*) noutstart,noutend,nmul

c--   Calculate the spline basis functions at a regular grid
      call splhsetup()

      do i=1,nrs
        dep(i)=dmin+(i-1)*ddep
        do ip=1,ntot
          call getfdp(iwhole,idptp,dep(i),ip,fp)
          spln(i,ip)=fp
        enddo
      enddo

      open(19,file=lnfl,status='old')
      open(88,file=outfl,status='unknown')

10    read(19,*,end=100) arcln,xlt,xln

c--   calculate Y_k(xlt,xln) at random depth level
      call ylm(xlt,xln,lmax,d0,wk1,wk2,wk3)

c--   calculate spline coefficients
      do i=nbeg,ndmx
        ind=(i-1)*natd+noutstart
        spl(i-ncrtopo)=sdot(nmul,d0(noutstart),1,x(ind),1)
      enddo

      do i=1,nrs
        wl(i)=0.
      enddo

      do ip=1,ntot
        call saxpy(nrs,spl(ip),spln(1,ip),1,wl,1)
      enddo

      if(irad.eq.0) then
        do i=1,nrs
          write(88,*) arcln,xln,xlt,dep(i),100.*wl(i)
        enddo
      else
        do i=1,nrs
          write(88,*) arcln,xln,xlt,6371.-dep(i),100.*wl(i)
        enddo
      endif

      goto 10

100   continue
      close(88)

      end
