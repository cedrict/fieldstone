!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_M

use global_parameters
use timing
use matrices, only : csrM

implicit none

integer i,jp,ip,nsees,nz,nnx,nny,i1,i2,j,j1,j2
!integer itemp(200)
!logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{matrix\_setup\_M}
!@@ See Section~\ref{ss:symmcsrss}. 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (geometry=='cartesian' .and. ndim==2) then

   csrM%n=NV

   nnx=nelx+1
   nny=nely+1
   csrM%NZ=(4*4+(2*(nnx-2)+2*(nny-2))*6+(nnx-2)*(nny-2)*9)
   csrM%nz=(csrM%nz-csrM%n)/2+csrM%n

   write(*,'(a)')     '        CSR matrix format symm' 
   write(*,'(a,i10)') '        csrM%n  =',csrM%n
   write(*,'(a,i10)') '        csrM%nz =',csrM%nz

   allocate(csrM%ia(csrM%n+1)) 
   allocate(csrM%ja(csrM%nz))   
   allocate(csrM%mat(csrM%nz))  

   nz=0
   csrM%ia(1)=1
   do j1=1,nny
      do i1=1,nnx
         ip=(j1-1)*nnx+i1 ! node number
         nsees=0
         do j2=-1,1 ! exploring neighbouring nodes
            do i2=-1,1
               i=i1+i2
               j=j1+j2
               if (i>=1 .and. i<= nnx .and. j>=1 .and. j<=nny) then ! if node exists
                  jp=(j-1)*nnx+i  ! node number of neighbour 
                  if (jp>=ip) then  ! upper diagonal
                     nz=nz+1
                     csrM%ja(nz)=jp
                     nsees=nsees+1
                  end if
               end if
            end do
         end do
         csrM%ia(ip+1)=csrM%ia(ip)+nsees
      end do
   end do

   if (debug) then
   write(*,*) '          nz=',nz
   write(*,*) '          csrM%ia (m/M)',minval(csrM%ia), maxval(csrM%ia)
   write(*,*) '          csrM%ja (m/M)',minval(csrM%ja), maxval(csrM%ja)
   write(*,*) csrM%ia
   write(*,*) csrM%ja
   end if

end if ! cartesian 2D

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> matrix_setup_M ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
