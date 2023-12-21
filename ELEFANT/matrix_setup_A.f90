!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_A

use module_parameters
use module_arrays, only: rhs_b
use module_sparse, only: csrA
use module_timing

implicit none

integer ip,jp,i,j,k,i1,i2,j1,j2,k1,k2,nsees,nz
integer nnx,nny,nnz

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{matrix\_setup\_A}
!@@ Matrix A is the energy equation matrix.
!@@ If the geometry is Cartesian then the number of nonzeros in the matrix and its sparsity 
!@@ structures are computed in a very efficient way. 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (use_T) then

csrA%N=NfemT

if (spaceT=='__Q1') then

   if (geometry=='cartesian' .and. ndim==2) then
      nnx=nelx+1
      nny=nely+1
      csrA%NZ=4*4+(2*(nnx-2)+2*(nny-2))*6+(nnx-2)*(nny-2)*9
   end if

   if (geometry=='cartesian' .and. ndim==3) then
      nnx=nelx+1
      nny=nely+1
      nnz=nelz+1
      csrA%nz=8*8                                 & ! 8 corners with 8 neighbours
             +(nnx-2)*(nny-2)*(nnz-2)*27          & ! all the inside nodes with 27 neighbours
             +(4*(nnx-2)+4*(nny-2)+4*(nnz-2))*12  & ! the edge nodes with 12 neighbours  
             +2*(nnx-2)*(nny-2)*18                & ! 2 faces
             +2*(nnx-2)*(nnz-2)*18                & ! 2 faces
             +2*(nny-2)*(nnz-2)*18                  ! 2 faces
   end if

else

   stop 'matrix_setup_A: spaceT not supported yet'

end if

write(*,'(a,i8)') shift//'csrA%N =',csrA%N
write(*,'(a,i8)') shift//'csrA%NZ=',csrA%NZ

allocate(csrA%ia(csrA%N+1)) 
allocate(csrA%ja(csrA%NZ))   
allocate(csrA%mat(csrA%NZ))  
allocate(rhs_b(csrA%N))  

if (geometry=='cartesian' .and. ndim==2) then
   nz=0
   csrA%ia(1)=1
   do j1=1,nny
   do i1=1,nnx
      ip=(j1-1)*nnx+i1 ! print *,ip
      nsees=0
      do j2=-1,1 ! exploring neighbouring nodes
      do i2=-1,1
         i=i1+i2
         j=j1+j2
         if (i>=1 .and. i<= nnx .and. j>=1 .and. j<=nny) then ! if node exists
            nz=nz+1
            csrA%ja(nz)=(j-1)*nnx+i  
            nsees=nsees+1
         end if
      end do
      end do
      csrA%ia(ip+1)=csrA%ia(ip)+nsees
   end do
   end do
end if

if (geometry=='cartesian' .and. ndim==3) then
   nz=0
   csrA%ia(1)=1
   do k1=1,nnz
   do j1=1,nny
   do i1=1,nnx
      ip=nnx*nny*(k1-1)+(j1-1)*nnx + i1 ! node number
      nsees=0
      do k2=-1,1 ! exploring neighbouring nodes
      do j2=-1,1 ! exploring neighbouring nodes
      do i2=-1,1 ! exploring neighbouring nodes
         i=i1+i2
         j=j1+j2
         k=k1+k2
         if (i>=1 .and. i<= nnx .and. j>=1 .and. j<=nny .and. k>=1 .and. k<=nnz) then ! if node exists
            jp=nnx*nny*(k-1)+(j-1)*nnx + i ! node number
            nz=nz+1
            csrA%ja(nz)=jp
            nsees=nsees+1
         end if
      end do
      end do
      end do
      csrA%ia(ip+1)=csrA%ia(ip)+nsees
   end do
   end do
   end do
end if

!print *,csrA%ja
!print *,csrA%ia

else
   write(*,'(a)') shift//'bypassed since use_T=False'

end if ! use_T

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'matrix_setup_A (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
