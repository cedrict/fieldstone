!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine solve_stokes

use module_parameters, only: NU,NV,NW,NfemVel,ndim,mU,mV,mW,mP,iproc,iel,use_penalty,nel,outer_solver_type
use module_statistics 
use module_arrays, only: rhs_f,solVel,solP,solU,solV,solW
use module_mesh
use module_timing

implicit none

integer inode,k
real(8) :: guess(NfemVel)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{solve\_stokes}
!@@ This subroutine solves the Stokes system: if it is a saddle point problem 
!@@ it does so using the preconditioned conjugate gradient (PCG) applied 
!@@ to the Schur complement $\SSS$ !@@ (see Section~\ref{ss:schurpcg}).
!@@ If the penalty method is used 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

solVel=0
solP=0
guess=0

if (use_penalty) then

   call inner_solver(rhs_f,guess,solVel)

   SolU=SolVel(1:NU)   
   SolV=SolVel(1+NU:NU+NV)   
   SolW=SolVel(1+NU+NV:NU+NV+NW)   

   !transfer velocity onto elements
   do iel=1,nel
      do k=1,mU
         mesh(iel)%u(k)=SolU(mesh(iel)%iconU(k))
      end do
      do k=1,mV
         mesh(iel)%v(k)=SolV(mesh(iel)%iconV(k))
      end do
      do k=1,mW
         mesh(iel)%w(k)=SolW(mesh(iel)%iconW(k))
      end do
   end do

   call recover_pressure_penalty

else !-----------------------------------------------------

   select case(outer_solver_type)
   case('___pcg')
   case default
      stop 'solve_stokes: unknown outer_solver_type'
   end select 




   !transfer velocity onto elements
   do iel=1,nel
      do k=1,mU
         mesh(iel)%u(k)=SolU(mesh(iel)%iconU(k))
      end do
      do k=1,mV
         mesh(iel)%v(k)=SolV(mesh(iel)%iconV(k))
      end do
      do k=1,mW
         mesh(iel)%w(k)=SolW(mesh(iel)%iconW(k))
      end do
   end do

end if

!----------------------------------------------------------

u_max=-1d30
u_min=+1d30
v_max=-1d30
v_min=+1d30
v_max=-1d30
v_min=+1d30
do iel=1,nel
   u_max=max(maxval(mesh(iel)%u(1:mV)),u_max)
   u_min=min(minval(mesh(iel)%u(1:mV)),u_min)
   v_max=max(maxval(mesh(iel)%v(1:mV)),v_max)
   v_min=min(minval(mesh(iel)%v(1:mV)),v_min)
   w_max=max(maxval(mesh(iel)%w(1:mV)),w_max)
   w_min=min(minval(mesh(iel)%w(1:mV)),w_min)
end do

             write(*,'(a,2es12.4)') shift//'u (m,M)',u_min,u_max
             write(*,'(a,2es12.4)') shift//'v (m,M)',v_min,v_max
if (ndim==3) write(*,'(a,2es12.4)') shift//'w (m,M)',w_min,w_max
             write(*,'(a,2es12.4)') shift//'p (m,M)',p_min,p_max

write(1238,'(8es12.4)') u_min,u_max,v_min,v_max,w_min,w_max,p_min,p_max
call flush(1238)

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'solve_stokes (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
