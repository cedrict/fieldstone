!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_MV

use module_mesh 
use module_parameters
use module_timing
use module_sparse, only : csrMV

implicit none

integer i,jp,ip,nsees,nz,nnx,nny,i1,i2,j,j1,j2
!integer itemp(200)
!logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{matrix\_setup\_MV}
!@@ See Section~\ref{ss:symmcsrss}. 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (geometry=='cartesian' .and. ndim==2) then

   csrMV%n=NV

   nnx=nelx+1
   nny=nely+1
   csrMV%NZ=(4*4+(2*(nnx-2)+2*(nny-2))*6+(nnx-2)*(nny-2)*9)
   csrMV%nz=(csrMV%nz-csrMV%n)/2+csrMV%n

   write(*,'(a)')     shift//'CSR matrix format symm' 
   write(*,'(a,i10)') shift//'csrMV%n  =',csrMV%n
   write(*,'(a,i10)') shift//'csrMV%nz =',csrMV%nz

   allocate(csrMV%ia(csrMV%n+1)) 
   allocate(csrMV%ja(csrMV%nz))   
   allocate(csrMV%mat(csrMV%nz))  

   nz=0
   csrMV%ia(1)=1
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
                     csrMV%ja(nz)=jp
                     nsees=nsees+1
                  end if
               end if
            end do
         end do
         csrMV%ia(ip+1)=csrMV%ia(ip)+nsees
      end do
   end do

else

   stop 'pb in matrix_setup_MV'

end if ! cartesian 2D
   
if (debug) then
write(2345,*) limit//'matrix_setup_MV'//limit
write(2345,*) 'csrMV%nz=',csrMV%nz
write(2345,*) 'csrMV%ia (m/M)',minval(csrMV%ia), maxval(csrMV%ia)
write(2345,*) 'csrMV%ja (m/M)',minval(csrMV%ja), maxval(csrMV%ja)
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'matrix_setup_MV (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
