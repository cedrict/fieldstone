      program mkrb

      parameter (NCMX=30)
      character*80 chjfl
      dimension ir(NCMX),ig(NCMX),ib(NCMX)

33    format(a80)
      read(5,33) chjfl
      read(5,*) icont
      read(5,*) cmx

      cmn = -cmx
      fc  = 1.

      open(20,file=chjfl,status='old')
      read(20,*) nrc
      if(nrc.gt.NCMX) stop 'nrc.gt.NCMX'
      do i=1,nrc
       read(20,*) ir(i),ig(i),ib(i)
      enddo
      close(20)

      do i=1,nrc
       ir(i)=int(fc*ir(i))
       ig(i)=int(fc*ig(i))
       ib(i)=int(fc*ib(i))
       if(ir(i).gt.255) ir(i)=255
       if(ig(i).gt.255) ig(i)=255
       if(ib(i).gt.255) ib(i)=255
      enddo

      if(icont.eq.0) then
        dc=(cmx-cmn)/float(nrc)
        do i=1,nrc
         c1=cmn+(i-1)*dc
         c2=cmn+i*dc
         write (6,10) c1,ir(i),ig(i),ib(i),c2,ir(i),ig(i),ib(i)
        enddo
      else if(icont.eq.1) then
        dc=(cmx-cmn)/float(nrc-1)
        do i=1,nrc-1
         c1=cmn+(i-1)*dc
         c2=cmn+i*dc
         write (6,10) c1,ir(i),ig(i),ib(i),c2,ir(i+1),ig(i+1),ib(i+1)
        enddo
      else
        stop 'unknown icont'
      endif


10    format(f12.5,1x,3i5,f12.5,3i5)

      write(6,11) ir(1),ig(1),ib(1)
11    format('B ',3i5)
      write(6,12) ir(nrc),ig(nrc),ib(nrc)
12    format('F ',3i5)


      end
      

