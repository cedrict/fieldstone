!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

program elefant

use global_parameters
use global_arrays
use structures
use matrices

implicit none

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
call set_global_parameters_pair

!----------------------------------------------------------

ndofV=ndim
NfemV=NV*ndofV
NfemP=NP
NfemT=NT

!nq_per_dim=2
!nqel=nq_per_dim**ndim
!Nq=nqel*nel
ncorners=2**ndim
if (ndim==2) ndim2=3
if (ndim==3) ndim2=6
allocate(solV(NfemV))
allocate(solP(NfemP))
allocate(rhs_f(NfemV))
allocate(rhs_h(NfemP))
allocate(mat(nmat))
allocate(Kdiag(NfemV))

allocate(Cmat(ndim2,ndim2)) ; Cmat=0d0
allocate(Kmat(ndim2,ndim2)) ; Kmat=0d0
if (ndim==2) then
Cmat(1,1)=2d0
Cmat(2,2)=2d0
Cmat(3,3)=1d0
Kmat(1,1)=1d0
Kmat(1,2)=1d0
Kmat(2,1)=1d0
Kmat(2,2)=1d0
end if
if (ndim==3) then
Cmat(1,1)=2d0
Cmat(2,2)=2d0
Cmat(3,3)=2d0
Cmat(4,4)=1d0
Cmat(5,5)=1d0
Cmat(6,6)=1d0
Kmat(1,1)=1d0
Kmat(1,2)=1d0
Kmat(1,3)=1d0
Kmat(2,1)=1d0
Kmat(2,2)=1d0
Kmat(2,3)=1d0
Kmat(3,1)=1d0
Kmat(3,2)=1d0
Kmat(3,3)=1d0
end if



call define_material_properties

!----------------------------------------------------------
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
                 write(*,'(a,i10)')    '        ncorners    =',ncorners
                 write(*,'(a,l10)')    '        use_MUMPS   =',use_MUMPS
                 write(*,'(a,i10)')    '        nmat        =',nmat
                 write(*,'(a,l10)')    '        use_penalty =',use_penalty
if (use_penalty) write(*,'(a,es10.3)') '        penalty     =',penalty
if (use_ALE)     write(*,'(a,l10)')    '        use_ALE     =',penalty
!----------------------------------------------------------

call spacer
select case (geometry)
case('cartesian') 
   if (ndim==2) call setup_cartesian2D
   if (ndim==3) call setup_cartesian3D
case('spherical')
end select
call output_mesh
call quadrature_setup
call test_basis_functions
call swarm_setup
call swarm_material_layout
call paint_swarm
call matrix_setup_K
call matrix_setup_GT
call matrix_setup_M
call matrix_setup_A
call initial_temperature

do istep=1,nstep !-----------------------------------------
                                                          !
   call spacer_istep                                      !
   call assign_values_to_qpoints                          !
   call compute_elemental_rho_eta_vol                     !
                                                          !
   if (solve_stokes_system) then                          !
      call define_bcV                                     !
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
   end if                                                 !
                                                          !
   call compute_gravity                                   !
   call postprocessors                                    !
   call output_solution                                   !
   call output_qpoints                                    !
   call output_swarm                                      !
   call write_stats                                       !
                                                          !
end do !---------------------------------------------------

call spacer_end

call footer

end program

!==================================================================================================!
!==================================================================================================!
