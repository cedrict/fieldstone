!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_MP

use global_parameters
use timing
use matrices, only : csrMP

implicit none

integer i,jp,ip,nsees,nz,nnx,nny,nnz,i1,i2,j,j1,j2,k1,k2,k

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{matrix\_setup\_MP}
!@@ This subroutine computes the structure of the pressure mass matrix. 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (geometry=='cartesian' .and. ndim==2) then

   csrMP%n=NP

   select case(pair)
   case('q1p0')
      csrMP%NZ=NP
   case('q1q1','q2q1')
      nnx=nelx+1
      nny=nely+1
      csrMP%NZ=(4*4+(2*(nnx-2)+2*(nny-2))*6+(nnx-2)*(nny-2)*9)
      csrMP%nz=(csrMP%nz-csrMP%n)/2+csrMP%n
   case default
      stop 'matrix_setup_MP: pair not implemented'
   end select 

   write(*,'(a)')     '        CSR matrix format symm' 
   write(*,'(a,i10)') '        csrMP%n  =',csrMP%n
   write(*,'(a,i10)') '        csrMP%nz =',csrMP%nz

   allocate(csrMP%ia(csrMP%n+1)) 
   allocate(csrMP%ja(csrMP%nz))   
   allocate(csrMP%mat(csrMP%nz))  

   select case(pair)
   case('q1p0')
      csrMP%ia(1)=1
      do iel=1,nel
         csrMP%ja(iel)=iel
         csrMP%ia(iel+1)=csrMP%ia(iel)+1
      end do
   case('q1q1','q2q1')
      nz=0
      csrMP%ia(1)=1
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
                        csrMP%ja(nz)=jp
                        nsees=nsees+1
                     end if
                  end if
               end do
            end do
            csrMP%ia(ip+1)=csrMP%ia(ip)+nsees
         end do
      end do
   case default
      stop 'matrix_setup_MP: pair not implemented'
   end select 

end if ! cartesian 2D

!----------------------------------------------------------

if (geometry=='cartesian' .and. ndim==3) then

   csrMP%n=NP

   select case(pair)
   case('q1p0')
      csrMP%NZ=NP
   case('q1q1','q2q1')
      nnx=nelx+1
      nny=nely+1
      nnz=nelz+1
      csrMP%nz=8*8                                 &! 8 corners with 8 neighbours
              +(nnx-2)*(nny-2)*(nnz-2)*27          &! all the inside nodes with 27 neighbours
              +(4*(nnx-2)+4*(nny-2)+4*(nnz-2))*12  &! the edge nodes with 12 neighbours  
              +2*(nnx-2)*(nny-2)*18                &! 2 faces
              +2*(nnx-2)*(nnz-2)*18                &! 2 faces
              +2*(nny-2)*(nnz-2)*18                 ! 2 faces
      csrMP%nz=(csrMP%nz-csrMP%n)/2+csrMP%n
   case default
      stop 'matrix_setup_MP: pair not implemented'
   end select 

   write(*,'(a)')     '        CSR matrix format symm' 
   write(*,'(a,i10)') '        csrMP%n  =',csrMP%n
   write(*,'(a,i10)') '        csrMP%nz =',csrMP%nz

   allocate(csrMP%ia(csrMP%n+1)) 
   allocate(csrMP%ja(csrMP%nz))   
   allocate(csrMP%mat(csrMP%nz))  

   select case(pair)
   case('q1p0')
      csrMP%ia(1)=1
      do iel=1,nel
         csrMP%ja(iel)=iel
         csrMP%ia(iel+1)=csrMP%ia(iel)+1
      end do
   case('q1q1','q2q1')
      nz=0
      csrMP%ia(1)=1
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
               if (jp>=ip) then  ! upper diagonal
                  nz=nz+1
                  csrMP%ja(nz)=jp
                  nsees=nsees+1
               end if
            end if
         end do
         end do
         end do
         csrMP%ia(ip+1)=csrMP%ia(ip)+nsees
      end do
      end do
      end do
   case default
      stop 'matrix_setup_MP: pair not implemented'
   end select 

end if

!----------------------------------------------------------
   
if (debug) then
   write(*,*) '          nz=',nz
   write(*,*) '          csrMP%ia (m/M)',minval(csrMP%ia), maxval(csrMP%ia)
   write(*,*) '          csrMP%ja (m/M)',minval(csrMP%ja), maxval(csrMP%ja)
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> matrix_setup_MP ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
