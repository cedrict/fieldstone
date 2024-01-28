!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_MP_and_S

use module_parameters, only: NP,iproc,debug,iel,mP,stokes_solve_strategy
use module_mesh
use module_sparse, only: csrMP,csrS
use module_arrays, only: pnode_belongs_to
use module_timing

implicit none

integer i,jp,ip,nsees,NZ,k
logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_MP\_and\_S}
!@@ This subroutine computes the structure of the pressure mass matrix. 
!==================================================================================================!

if (iproc==0) then

if (stokes_solve_strategy=='penalty') return

call system_clock(counti,count_rate)

!==============================================================================!

csrMP%N=NP

call cpu_time(t3)
allocate(alreadyseen(NP))
NZ=0
do ip=1,NP
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

call cpu_time(t4) ; write(*,'(a,f10.3,a)') shift//'compute NZ:',t4-t3,'s'

write(*,'(a)')       shift//'CSR matrix format SYMMETRIC  '
write(*,'(a,i11,a)') shift//'csrMP%N       =',csrMP%N,' '
write(*,'(a,i11,a)') shift//'csrMP%NZ      =',csrMP%NZ,' '

allocate(csrMP%ia(csrMP%N+1))   
allocate(csrMP%ja(csrMP%NZ))    
allocate(csrMP%mat(csrMP%NZ))    

   call cpu_time(t3)
   allocate(alreadyseen(NP))
   NZ=0
   csrMP%ia(1)=1
   do ip=1,NP
      nsees=0
      alreadyseen=.false.
      do k=1,pnode_belongs_to(1,ip)
         iel=pnode_belongs_to(1+k,ip)
         do i=1,mP
            jp=mesh(iel)%iconP(i)
            if (.not.alreadyseen(jp)) then
               if (jp>=ip) then
                  NZ=NZ+1
                  csrMP%ja(NZ)=jp
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

call cpu_time(t4) ; write(*,'(a,f10.3,a)') shift//'compute ia,ja:',t4-t3,'s'

write(*,'(a,i9)' ) shift//'NZ=',NZ
write(*,'(a,2i9)') shift//'csrMP%ia',minval(csrMP%ia), maxval(csrMP%ia)
write(*,'(a,2i9)') shift//'csrMP%ja',minval(csrMP%ja), maxval(csrMP%ja)

!----------------------------------------------------------
! and now the Schur complement matrix 
!----------------------------------------------------------

csrS%N=csrMP%N
csrS%NZ=csrMP%NZ

allocate(csrS%ia(csrS%N+1))   
allocate(csrS%ja(csrS%NZ))    
allocate(csrS%mat(csrS%NZ))    

csrS%ia=csrMP%ia
csrS%ja=csrMP%ja

!----------------------------------------------------------
   
if (debug) then
write(2345,*) limit//'setup_MP_and_S'//limit
write(2345,*) 'csrMP%NZ=',csrMP%NZ
write(2345,*) 'csrMP%ia (m/M)',minval(csrMP%ia), maxval(csrMP%ia)
write(2345,*) 'csrMP%ja (m/M)',minval(csrMP%ja), maxval(csrMP%ja)
write(2345,*) 'csrMP%ia ',csrMP%ia
do i=1,NP
write(2345,*) i,'th line: csrMP%ja=',csrMP%ja(csrMP%ia(i):csrMP%ia(i+1)-1)-1
end do
call flush(2345)
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_MP_and_S:',elapsed,' s                 |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
