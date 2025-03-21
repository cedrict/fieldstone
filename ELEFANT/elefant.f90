!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

program elefant

use module_parameters, only: geometry,nstep,dt,cistep,use_T,istep,time,ndim,solve_stokes_system,write_params
!use module_arrays
use module_mesh
!use module_sparse
!use module_materials

implicit none


open(unit=1234,file="OUTPUT/STATS/statistics.ascii")
open(unit=1235,file="OUTPUT/STATS/statistics_energy_system.ascii")
open(unit=1236,file="OUTPUT/STATS/statistics_stokes_system.ascii")
open(unit=1237,file="OUTPUT/STATS/statistics_T.ascii")
open(unit=1238,file="OUTPUT/STATS/statistics_VP.ascii")
open(unit=1239,file="OUTPUT/STATS/statistics_pmgmres.ascii")
open(unit=1240,file="OUTPUT/STATS/statistics_rheology.ascii")
open(unit=2345,file="debug.ascii")

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
call experiment_declare_main_parameters
call read_command_line_options
call process_inputs
call process_bc
call set_global_parameters_spaceV
call set_global_parameters_spaceP
call set_global_parameters_spaceT
call set_global_parameters_mapping
call allocate_memory
call experiment_define_material_properties
call initialise_elements
select case (geometry)
case('cartesian') 
   if (ndim==2) call setup_cartesian2D
   if (ndim==3) call setup_cartesian3D
case('spherical')
   if (ndim==2) call setup_annulus
   !if (ndim==3) call setup_shell
case('john')
   call setup_john
end select

call setup_mapping
call output_mesh
call quadrature_setup
call output_qpoints 
call test_basis_functions
call swarm_setup
call experiment_swarm_material_layout
call paint_swarm
call compute_belongs
call setup_K  
call setup_GT 
call setup_A  
call setup_MP_and_S 
call output_matrix_for_paraview
call experiment_initial_temperature
call estimate_memory_use
call spacer
call write_params 
call spacer

do istep=1,nstep !-----------------------------------------
                                                          !
   call int_to_char(cistep,6,istep)                       !
   call spacer_istep                                      !
   call assign_values_to_qpoints                          !
   call compute_elemental_rho_eta_vol                     !
   call compute_block_scaling_coefficient                 !
   call experiment_define_bcV                             !
                                                          !
   if (solve_stokes_system) then                          !
      call make_matrix_stokes                             !
      call extract_K_diagonal                             !
      call solve_stokes                                   !
   else                                                   !
      call prescribe_stokes_solution                      !
   end if                                                 !
                                                          !
   call compute_timestep                                  !
                                                          !
   if (use_T) then                                        !
      call experiment_define_bcT                          !
      call make_matrix_energy                             !
      call solve_energy                                   !
      call compute_temperature_gradient                   !
   end if                                                 !

   call compute_elemental_strain_rate                     !
   call compute_gravity                                   !
   call postprocessors                                    !
   call output_solution                                   !
   call output_qpoints                                    !
   call output_swarm                                      !
   call write_stats                                       !
                                                          !
   time=time+dt                                           !
                                                          !
end do !---------------------------------------------------

call spacer_end

call footer

end program

!==================================================================================================!
!==================================================================================================!
