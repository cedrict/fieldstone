!===================================================================!
!===================================================================!
!     FANTOM2                                       C.Thieulot      !
!===================================================================!
!===================================================================!

subroutine int_to_char(charac,ncharac,integ)

implicit none

character(len=*) :: charac
integer integ,ncharac

!===================================================================!

select case (ncharac)
   case (1)
      write (charac,'(i1)') integ
   case (2)
      write (charac,'(i2)') integ
      if (integ.lt.10) charac(1:1)='0'
   case (3)
      write (charac,'(i3)') integ
      if (integ.lt.100) charac(1:1)='0'
      if (integ.lt.10) charac(1:2)='00'
   case (4)
      write (charac,'(i4)') integ
      if (integ.lt.1000) charac(1:1)='0'
      if (integ.lt.100) charac(1:2)='00'
      if (integ.lt.10) charac(1:3)='000'
   case (5)
      write (charac,'(i5)') integ
      if (integ.lt.10000) charac(1:1)='0'
      if (integ.lt.1000) charac(1:2)='00'
      if (integ.lt.100) charac(1:3)='000'
      if (integ.lt.10) charac(1:4)='0000'
   case (6)
      write (charac,'(i6)') integ
      if (integ.lt.100000) charac(1:1)='0'
      if (integ.lt.10000) charac(1:2)='00'
      if (integ.lt.1000) charac(1:3)='000'
      if (integ.lt.100) charac(1:4)='0000'
      if (integ.lt.10) charac(1:5)='00000'
   case (7)
      write (charac,'(i7)') integ
      if (integ.lt.1000000) charac(1:1)='0'
      if (integ.lt.100000) charac(1:2)='00'
      if (integ.lt.10000) charac(1:3)='000'
      if (integ.lt.1000) charac(1:4)='0000'
      if (integ.lt.100) charac(1:5)='00000'
      if (integ.lt.10) charac(1:6)='000000'
   case (8)
      write (charac,'(i8)') integ
      if (integ.lt.10000000) charac(1:1)='0'
      if (integ.lt.1000000) charac(1:2)='00'
      if (integ.lt.100000) charac(1:3)='000'
      if (integ.lt.10000) charac(1:4)='0000'
      if (integ.lt.1000) charac(1:5)='00000'
      if (integ.lt.100) charac(1:6)='000000'
      if (integ.lt.10) charac(1:7)='0000000'
   case (9)
      write (charac,'(i9)') integ
      if (integ.lt.100000000) charac(1:1)='0'
      if (integ.lt.10000000) charac(1:2)='00'
      if (integ.lt.1000000) charac(1:3)='000'
      if (integ.lt.100000) charac(1:4)='0000'
      if (integ.lt.10000) charac(1:5)='00000'
      if (integ.lt.1000) charac(1:6)='000000'
      if (integ.lt.100) charac(1:7)='0000000'
      if (integ.lt.10) charac(1:8)='00000000'
   case (10)
      write (charac,'(i10)') integ
      if (integ.lt.1000000000) charac(1:1)='0'
      if (integ.lt.100000000) charac(1:2)='00'
      if (integ.lt.10000000) charac(1:3)='000'
      if (integ.lt.1000000) charac(1:4)='0000'
      if (integ.lt.100000) charac(1:5)='00000'
      if (integ.lt.10000) charac(1:6)='000000'
      if (integ.lt.1000) charac(1:7)='0000000'
      if (integ.lt.100) charac(1:8)='00000000'
      if (integ.lt.10) charac(1:9)='000000000'
   case default
      stop 'value ncharac too big'
end select

end subroutine int_to_char
