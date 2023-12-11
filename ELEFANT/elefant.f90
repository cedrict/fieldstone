!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

program elefant

use module_parameters
use module_arrays
use module_mesh
use module_sparse
use module_materials

implicit none

open(unit=1234,file="OUTPUT/STATS/statistics.ascii")
open(unit=1235,file="OUTPUT/STATS/statistics_energy_system.ascii")
open(unit=1236,file="OUTPUT/STATS/statistics_stokes_system.ascii")
open(unit=1237,file="OUTPUT/STATS/statistics_T.ascii")
open(unit=1238,file="OUTPUT/STATS/statistics_VP.ascii")
open(unit=1239,file="OUTPUT/STATS/statistics_pmgmres.ascii")
open(unit=1240,file="OUTPUT/STATS/statistics_rheology.ascii")

call header

#ifdef UseMUMPS
print *,'with MUMPS support'
include 'mpif.h'
call mpi_init(ierr)
call mpi_comm_size (mpi_comm_world,nproc,ierr)
call mpi_comm_rank (mpi_comm_world,iproc,ierr)
call mpi_get_processor_name(procname,resultlen,ierr)
#else
print *,'no MUMPS support'
#endif

call spacer
call set_default_values
call declare_main_parameters
call read_command_line_options

!call set_global_parameters_pair ! replace by
call set_global_parameters_spaceV
call set_global_parameters_spaceP
call set_global_parameters_spaceT
call set_global_parameters_mapping

call print_parameters

!----------------------------------------------------------

if (use_penalty) then
   csrK%full_matrix_storage=.true. ! y12m solver 
else
   csrK%full_matrix_storage=.false. ! pcg_solver 
end if

ndofV=ndim
NfemV=NV*ndofV
NfemP=NP
NfemT=NT

allocate(solV(NfemV))
allocate(solP(NfemP))
allocate(rhs_f(NfemV))
allocate(rhs_h(NfemP))
allocate(materials(nmat))
allocate(Kdiag(NfemV))


call define_material_properties

call spacer
call initialise_elements
select case (geometry)
case('cartesian') 
   if (ndim==2) call setup_cartesian2D
   if (ndim==3) call setup_cartesian3D
case('spherical')
   !if (ndim==2) call setup_annulus
   !if (ndim==3) call setup_shell
end select

call output_mesh
call mapping_setup
call quadrature_setup
call test_basis_functions
call swarm_setup
call swarm_material_layout
call paint_swarm
call matrix_setup_K
stop
call matrix_setup_GT
call matrix_setup_MV
call matrix_setup_MP
call matrix_setup_A
!call output_matrix_tikz
call initial_temperature

!----------------------------------------------------------
write(*,'(a)') '..................................'
                 write(*,'(a,i10)')    '        ndim        =',ndim
                 write(*,'(a,a11)')    '        geometry    =',geometry
                 write(*,'(a,a10)')    '        pair        =',pair
                 write(*,'(a,f10.3)')  '        Lx          =',Lx
                 write(*,'(a,f10.3)')  '        Ly          =',Ly
if (ndim==3)     write(*,'(a,f10.3)')  '        Lz          =',Lz
                 write(*,'(a,i10)')    '        nelx        =',nelx
                 write(*,'(a,i10)')    '        nely        =',nely
if (ndim==3)     write(*,'(a,i10)')    '        nelz        =',nelz
                 write(*,'(a,i10)')    '        nel         =',nel
                 write(*,'(a,i10)')    '        nqel        =',nqel
                 write(*,'(a,i10)')    '        mV          =',mV
                 write(*,'(a,i10)')    '        mP          =',mP
                 write(*,'(a,i10)')    '        mT          =',mT
                 write(*,'(a,i10)')    '        NV          =',NV
                 write(*,'(a,i10)')    '        NP          =',NP
if (use_T)       write(*,'(a,i10)')    '        NT          =',NT
                 write(*,'(a,i10)')    '        NfemV       =',NfemV
                 write(*,'(a,i10)')    '        NfemP       =',NfemP
if (use_T)       write(*,'(a,i10)')    '        NfemT       =',NfemT
                 write(*,'(a,i10)')    '        Nq          =',Nq
                 write(*,'(a,l10)')    '        use_MUMPS   =',use_MUMPS
                 write(*,'(a,i10)')    '        nmat        =',nmat
                 write(*,'(a,l10)')    '        use_penalty =',use_penalty
if (use_penalty) write(*,'(a,es10.3)') '        penalty     =',penalty
if (use_ALE)     write(*,'(a,l10)')    '        use_ALE     =',penalty
!----------------------------------------------------------

do istep=1,nstep !-----------------------------------------
                                                          !
   call int_to_char(cistep,6,istep)                       !
   call spacer_istep                                      !
   call assign_values_to_qpoints                          !
   call compute_elemental_rho_eta_vol                     !
   call define_bcV                                        !
                                                          !
   if (solve_stokes_system) then                          !
      call make_matrix_stokes                             !
      call solve_stokes                                   !
      call interpolate_onto_nodes                         !
   else                                                   !
      call prescribe_stokes_solution                      !
   end if                                                 !
                                                          !
   call compute_timestep                                  !
                                                          !
   if (use_T) then                                        !
      call define_bcT                                     !
      call make_matrix_energy                             !
      call solve_energy                                   !
      call compute_temperature_gradient                   !
   end if                                                 !
                                                          !
   call compute_gravity                                   !
   call postprocessors                                    !
   call output_solution                                   !
   !call output_solution_python                            !
   call output_qpoints                                    !
   call output_swarm                                      !
   call write_stats                                       !
                                                          !
   time=time+dt                                           !
                                                          !
end do !---------------------------------------------------

!call spacer_end

call footer

end program

!==================================================================================================!
!==================================================================================================!
