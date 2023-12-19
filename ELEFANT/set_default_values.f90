!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_default_values

use module_parameters
use module_gravity

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{set\_default\_values}
!@@ This subroutine assigns default values to many of the global variables.
!==================================================================================================!

if (iproc==0) then

!==============================================================================!

ndim=2

CFL_nb=0.25

geometry='cartesian'
spaceV='__Q2'
spaceP='__Q1'

select case(spaceV)
case('__Q1','_Q1+')
   spaceT='__Q1'
case('__Q2')
   spaceT='__Q2'
case('__Q3')
   spaceT='__Q3'
case('__P1')
   spaceT='__P1'
case('__P2')
   spaceT='__P2'
case default
   stop 'set_default_values: spaceV/spaceT pb'
end select

mapping=spaceV !isoparametric

use_swarm=.false.
nmarker_per_dim=5 
init_marker_random=.false. 

nstep=1

solve_stokes_system=.true. 

geometry='cartesian'

inner_solver_type='__y12m'

debug=.false.

use_T=.false.

nmat=1

penalty=1e6
use_penalty=.False.

nxstripes=1
nystripes=1
nzstripes=1

nmarker=0

use_ALE=.false.

grav_pointmass=.false. 
grav_prism=.false.
plane_nnx=0
plane_nny=0
line_nnp=0

normalise_pressure=.false.

output_freq=1

write(*,'(a)') 'set_default_values '

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
