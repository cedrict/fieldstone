!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_MP

use module_parameters, only: NP,nelx,nely,nelz,spacePressure,iproc,debug,geometry,ndim,iel,mP,nel 
use module_mesh
use module_sparse, only: csrMP
use module_arrays, only: pnode_belongs_to
use module_timing

implicit none

integer i,jp,ip,nsees,nz,nnx,nny,nnz,i1,i2,j,j1,j2,k1,k2,k,imod
real(8) :: t3,t4
logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_MP}
!@@ This subroutine computes the structure of the pressure mass matrix. 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (geometry=='XXXcartesian' .and. ndim==2) then

   csrMP%n=NP

   select case(spacePressure)
   case('__Q0','__P0')
      csrMP%NZ=NP
   case('__Q1')
      nnx=nelx+1
      nny=nely+1
      csrMP%NZ=(4*4+(2*(nnx-2)+2*(nny-2))*6+(nnx-2)*(nny-2)*9)
      csrMP%nz=(csrMP%nz-csrMP%n)/2+csrMP%n
   case default
      stop 'matrix_setup_MP: spacePressure not supported'
   end select 

   write(*,'(a)')     shift//'CSR matrix format symm' 
   write(*,'(a,i10)') shift//'csrMP%n  =',csrMP%n
   write(*,'(a,i10)') shift//'csrMP%nz =',csrMP%nz

   allocate(csrMP%ia(csrMP%n+1)) 
   allocate(csrMP%ja(csrMP%nz))   
   allocate(csrMP%mat(csrMP%nz))  

   select case(spacePressure)
   case('__Q0','__P0')
      csrMP%ia(1)=1
      do iel=1,nel
         csrMP%ja(iel)=iel
         csrMP%ia(iel+1)=csrMP%ia(iel)+1
      end do
   case('__Q1')
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
      stop 'matrix_setup_MP: spacePressure not implemented'
   end select 

!----------------------------------------------------------
elseif (geometry=='XXXcartesian' .and. ndim==3) then

   csrMP%N=NP

   select case(spacePressure)
   case('__Q0','__P0')
      csrMP%NZ=NP
   case('__Q1')
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
      stop 'matrix_setup_MP: spacePressure not implemented'
   end select 

   write(*,'(a)')     shift//'CSR matrix format symm' 
   write(*,'(a,i10)') shift//'csrMP%n  =',csrMP%n
   write(*,'(a,i10)') shift//'csrMP%nz =',csrMP%nz

   allocate(csrMP%ia(csrMP%n+1)) 
   allocate(csrMP%ja(csrMP%nz))   
   allocate(csrMP%mat(csrMP%nz))  

   select case(spacePressure)
   case('__Q0','__P0')
      csrMP%ia(1)=1
      do iel=1,nel
         csrMP%ja(iel)=iel
         csrMP%ia(iel+1)=csrMP%ia(iel)+1
      end do
   case('__Q1')
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
      stop 'matrix_setup_MP: spacePressure not implemented'
   end select 

else

   imod=NP/4
   csrMP%N=NP

   ! I could insert here typical combinations for which NZ can be computed analytically

   call cpu_time(t3)
   allocate(alreadyseen(NP))
   NZ=0
   do ip=1,NP
      if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(NP)*100.,'%'
      alreadyseen=.false.
      do k=1,pnode_belongs_to(1,ip)
         iel=pnode_belongs_to(1+k,ip)
         do i=1,mP
            jp=mesh(iel)%iconP(i)
            !print *,ip,iel,jp
            if (.not.alreadyseen(jp)) then
               alreadyseen(jp)=.true.
               NZ=NZ+1
            end if
         end do
      end do
   end do
   csrMP%NZ=NZ
   csrMP%NZ=(csrMP%NZ-csrMP%N)/2+csrMP%N
   deallocate(alreadyseen)
   call cpu_time(t4) ; write(*,'(f10.3,a)') t4-t3,'s'

   write(*,'(a)')       shift//'CSR matrix format SYMMETRIC  '
   write(*,'(a,i11,a)') shift//'csrMP%N       =',csrMP%N,' '
   write(*,'(a,i11,a)') shift//'csrMP%NZ      =',csrMP%NZ,' '

   allocate(csrMP%ia(csrMP%N+1))   
   allocate(csrMP%ja(csrMP%NZ))    
   allocate(csrMP%mat(csrMP%NZ))    

   call cpu_time(t3)
   allocate(alreadyseen(NP))
   nz=0
   csrMP%ia(1)=1
   do ip=1,NP
      if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(NP)*100.,'%'
      nsees=0
      alreadyseen=.false.
      do k=1,pnode_belongs_to(1,ip)
         iel=pnode_belongs_to(1+k,ip)
         do i=1,mP
            jp=mesh(iel)%iconP(i)
            if (.not.alreadyseen(jp)) then
               if (jp>=ip) then
                  nz=nz+1
                  csrMP%ja(nz)=jp
                  nsees=nsees+1
                  !print *,ip,jp,ip,jp
               end if
               alreadyseen(jp)=.true.
            end if
         end do
      end do    
      csrMP%ia(ip+1)=csrMP%ia(ip)+nsees    
   end do    
   deallocate(alreadyseen)    
   call cpu_time(t4) ; write(*,'(f10.3,a)') t4-t3,'s'

   write(*,'(a,i9)' ) shift//'nz=',nz
   write(*,'(a,2i9)') shift//'csrMP%ia',minval(csrMP%ia), maxval(csrMP%ia)
   write(*,'(a,2i9)') shift//'csrMP%ja',minval(csrMP%ja), maxval(csrMP%ja)

end if

!----------------------------------------------------------
   
if (debug) then
write(2345,*) limit//'setup_MP'//limit
write(2345,*) 'csrMP%nz=',csrMP%nz
write(2345,*) 'csrMP%ia (m/M)',minval(csrMP%ia), maxval(csrMP%ia)
write(2345,*) 'csrMP%ja (m/M)',minval(csrMP%ja), maxval(csrMP%ja)
write(2345,*) 'csrMP%ia ',csrMP%ia
do i=1,NP
write(2345,*) i,'th line: csrMP%ja=',csrMP%ja(csrMP%ia(i):csrMP%ia(i+1)-1)-1
end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_MP:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
