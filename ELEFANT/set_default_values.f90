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
pair='q1p0'

use_swarm=.false.
nmarker_per_dim=5 
init_marker_random=.false. 

nstep=1

solve_stokes_system=.true. 

geometry='cartesian'

use_MUMPS=.false.

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

write(*,'(a)') 'set_default_values '

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
