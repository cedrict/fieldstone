!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_matrix_tikz

use module_parameters
use module_mesh 
use module_sparse
use module_timing

implicit none

integer i,j,k
real ii,jj

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{output\_matrix\_tikz}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (use_T) then
open(unit=123,file='OUTPUT/TIKZ/csrA.tex')
write(123,'(a)') '\begin{tikzpicture}'
write(123,'(a,f6.1,a,f6.1,a,f6.1,a,f6.1,a)') '\draw[] (',0.1,',',0.1,') rectangle (',&
      real(NfemT)/10+0.1,',',real(NfemT)/10+0.1,');'
do i=1,NfemT
   do k=csrA%ia(i),csrA%ia(i+1)-1
      j=csrA%ja(k)
      jj=j
      ii=NfemT-i+1
      ii=ii/10
      jj=jj/10 
      write(123,'(a,f6.1,a,f6.1,a,f6.1,a,f6.1,a)') '\draw[fill=teal!20] (',ii,',',jj,') rectangle (',ii+.1,',',jj+0.1,');'
   end do
end do
write(123,'(a)') '\end{tikzpicture}'
close(123)
end if

open(unit=123,file='OUTPUT/TIKZ/csrStokes.tex')

!matrix K
write(123,'(a)') '\begin{tikzpicture}'
write(123,'(a,f6.1,a,f6.1,a,f6.1,a,f6.1,a)') '\draw[] (',0.1,',',0.1+real(NfemP)/10,') rectangle (',&
            real(NfemV)/10+0.1,',',real(NfemP+NfemV)/10+0.1,');'
do i=1,NfemV
   do k=csrK%ia(i),csrK%ia(i+1)-1
      j=csrK%ja(k)
      ii=j
      jj=NfemV-i+1 + NfemP
      ii=ii/10
      jj=jj/10 
      write(123,'(a,f6.1,a,f6.1,a,f6.1,a,f6.1,a)') '\draw[fill=teal!20] (',ii,',',jj,') rectangle (',ii+.1,',',jj+0.1,');'
   end do
end do

!matrix G
write(123,'(a,f6.1,a,f6.1,a,f6.1,a,f6.1,a)') '\draw[] (',0.1+real(NfemV)/10,',',0.1+real(NfemP)/10,&
              ') rectangle (',real(NfemV+NfemP)/10+0.1,',',real(NfemV+NfemP)/10+0.1,');'
do i=1,NfemP
   do k=csrGT%ia(i),csrGT%ia(i+1)-1
      j=csrGT%ja(k)
      jj=j + NfemP
      ii=NfemP-i+1 + NfemV
      ii=ii/10
      jj=jj/10 
      write(123,'(a,f6.1,a,f6.1,a,f6.1,a,f6.1,a)') '\draw[fill=olive!20] (',ii,',',jj,') rectangle (',ii+.1,',',jj+0.1,');'
   end do
end do

!matrix GT
write(123,'(a,f6.1,a,f6.1,a,f6.1,a,f6.1,a)') '\draw[] (',0.1,',',0.1,') rectangle (',real(NfemV)/10+0.1,',',real(NfemP)/10+0.1,');'
do i=1,NfemP
   do k=csrGT%ia(i),csrGT%ia(i+1)-1
      j=csrGT%ja(k)
      ii=j
      jj=NfemP-i+1
      ii=ii/10
      jj=jj/10 
      write(123,'(a,f6.1,a,f6.1,a,f6.1,a,f6.1,a)') '\draw[fill=olive!20] (',ii,',',jj,') rectangle (',ii+.1,',',jj+0.1,');'
   end do
end do




write(123,'(a)') '\end{tikzpicture}'
close(123)










!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') '     >> output_matrix_tikz ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
