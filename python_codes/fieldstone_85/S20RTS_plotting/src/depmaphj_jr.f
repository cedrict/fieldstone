      program depmaphj_jr

c-- Modified from depmaphj
c-- now without command line input
c-- Jeroen Ritsema

                                                                          
                                                                          
      parameter (MXLH=40)
      parameter (MXLENY=(MXLH+1)**2)
      parameter (MXDEP=21)
      parameter (MXNDP=200)
      parameter (MXSPL=MXDEP+3)
      parameter (MXMSZ=MXSPL*MXLENY)
                                                                          
      dimension yraw(MXLENY)
      dimension maskl(MXLH),masks(MXSPL)
      dimension x(MXMSZ)
      dimension spln(MXDEP)
      dimension dep(MXNDP),ldep(MXNDP)
      character*10 depstr(MXNDP)
      character*80 getunx,mfl,mofl,outfl,dumstr
      character*120 line1
                                                                          
33    format(a80)
c-- Get the modelname
      read(5,33) mfl
      lm = lnblnk(mfl)
        write(6,*) 'model=', mfl
        write(6,*) 'lm= ', lm
      if(mfl(lm-2:lm).eq.'sph') then
       imod=1
       iwhole=1
       idptp=1
       ncrtopo=3
      else
       stop 'unknown model format'
      endif
                                                                          
c    take directory out of model name
      ip1=lm
      do while(ip1.gt.0.and.mfl(ip1:ip1).ne.'/')
        ip1=ip1-1
      enddo
      mofl=mfl(ip1+1:lm-4)
      write(6,*) 'ip1+1=', ip1+1, ' lm-4=', lm-4
      write(6,*) 'mfl=',mfl(ip1+1:lm-4)
      lo=istlen(mofl)

c-- Read the depth from standard input
      i=0
10    i=i+1
      read(5,*,end=20) dep(i)
15    goto 10
20    continue
      ndep=i-1
                                                                          
         do i=1,ndep
            write(6,*) 'n= ',i,' dep= ',dep(i)
         enddo
                                                                          
c    read model
        write(6,*) 'y1'
      open(41,file=mfl,status='old')
        write(6,*) 'y2'
      read(41,'(a)') line1
      call rsphhead(line1,lmx,ndmn,ndmx,ndp)
      write(6,*) lmx,ndmn,ndmx,ndp
      natd=(lmx+1)**2
      ind=(ndmn-1)*natd+1
      do i=ndmn,ndmx
       do j=0,lmx
        ind1=ind+2*j
        read(41,'(11e12.4)')(x(k),k=ind,ind1)
        ind=ind1+1
       enddo
      enddo
      close(41)
                                                                          
      if(lmx.gt.MXLH) stop'lmx.gt.MXLH'
      if(ndmx.gt.MXSPL) stop'ndmx.gt.MXSPL'
      nbeg=max(ncrtopo+1,ndmn)
      ntot=ndmx-ncrtopo
                                                                          
      write(6,*) lmx,nbeg,ndmx,ntot
                                                                          
c    Calculate the spline basis functions at a regular grid
      call splhsetup()
                                                                          
      do ii=1,ndep
       do ip=1,ntot
        call getfdp(iwhole,idptp,dep(ii),ip,fp)
        write(6,*) 'ii,dep(ii),ip,fp= ',ii,dep(ii),ip,fp
        spln(ip)=fp
       enddo

       do i=1,natd
        yraw(i)=0.
       enddo
                                                                          
c      calculate map
       do i=1,ntot
        ind=(i+ncrtopo-1)*natd+1
        call saxpy(natd,spln(i),x(ind),1,yraw,1)
       enddo
                                                                          
       outfl=mofl(1:lo)//depstr(ii)(1:ldep(ii))//'.raw'
       open(14,file=outfl,status='unknown')
       write(14,'(i3,x,i6,x,f11.3)') lmx
       write(14,'(5e16.8)') (yraw(i),i=1,natd)
       close(14)
      enddo
                                                                          
                                                                          
100   continue
                                                                          
      end
