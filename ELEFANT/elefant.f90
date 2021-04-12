program elefant

use global_parameters
use structures

implicit none


call declare_main_parameters
call define_material_properties

!--------------------------------------

if (pair=='q1p0') then
   mV=2**ndim
   mP=1
   mT=2**ndim
   nq_per_dim=2
   nqel=nq_per_dim**ndim
   ndofV=ndim
   if (ndim==2) then
      nel=nelx*nely
      NV=(nelx+1)*(nely+1)
      NT=(nelx+1)*(nely+1)
      NP=nel
   else
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)
      NT=(nelx+1)*(nely+1)*(nelz+1)
      NP=nel
   end if
end if

if (pair=='q1q1') then
   mV=2**ndim+1
   mP=2**ndim
   mT=2**ndim
   nq_per_dim=2
   nqel=nq_per_dim**ndim
   ndofV=ndim
   if (ndim==2) then
      nel=nelx*nely
      NV=(nelx+1)*(nely+1)+nel
      NT=(nelx+1)*(nely+1)
      NP=(nelx+1)*(nely+1)
   else
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)+2*nel
      NT=(nelx+1)*(nely+1)*(nelz+1)
      NP=(nelx+1)*(nely+1)*(nelz+1)
   end if
end if

NfemV=NV*ndofV
NfemP=NP
NfemT=NT
Nq=nqel*nel
ncorners=2**ndim
if (ndim==2) ndim2=3
if (ndim==3) ndim2=6

solve_stokes_system=.true.
nstep=1

!----------------------------
write(*,*) 'Lx,Ly=',Lx,Ly
write(*,*) 'nelx,nely',nelx,nely
write(*,*) 'nel',nel
write(*,*) 'pair',pair
write(*,*) 'geometry=',geometry
write(*,*) 'nqel=',nqel
write(*,*) 'NV=',NV
write(*,*) 'NP=',NP
write(*,*) 'NT=',NT
write(*,*) 'NfemV=',NfemV
write(*,*) 'NfemP=',NfemP
write(*,*) 'NfemT=',NfemT
write(*,*) 'Nq=',Nq
write(*,*) 'ncorners=',ncorners
!----------------------------


!-------------------------------------------------

select case (geometry)
case('cartesian2D'); call setup_cartesian2D
case('cartesian3D'); call setup_cartesian3D
end select
call output_mesh
call quadrature_setup

call markers_setup
call material_layout
!call material_paint
call output_swarm

do istep=1,nstep !-----------------------------------------
                                                          !
   if (solve_stokes_system) then                          !
                                                          !
      call assign_values_to_qpoints
      call define_bcV                                     !
      call make_matrix                                    !
      call solve_stokes                                   !
      call interpolate_onto_nodes                         !

   else                                                   !

      call prescribe_stokes_solution                      !

   end if
                                                          !
end do !---------------------------------------------------

call postprocessors
call output_solution
call output_qpoints

end program
