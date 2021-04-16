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
call define_material_properties
call set_global_parameters_pair

!----------------------------------------------------------

nq_per_dim=2
nqel=nq_per_dim**ndim
ndofV=ndim
NfemV=NV*ndofV
NfemP=NP
NfemT=NT
Nq=nqel*nel
ncorners=2**ndim
if (ndim==2) ndim2=3
if (ndim==3) ndim2=6
allocate(solV(NfemV))
allocate(solP(NfemP))
allocate(mat(nmat))

!----------------------------------------------------------
             write(*,'(a,i10)')   '        ndim      =',ndim
             write(*,'(a,a11)')   '        geometry  =',geometry
             write(*,'(a,a10)')   '        pair      =',pair
             write(*,'(a,f10.3)') '        Lx        =',Lx
             write(*,'(a,f10.3)') '        Ly        =',Ly
if (ndim==3) write(*,'(a,f10.3)') '        Lz        =',Lz
             write(*,'(a,i10)')   '        nelx      =',nelx
             write(*,'(a,i10)')   '        nely      =',nely
if (ndim==3) write(*,'(a,i10)')   '        nelz      =',nelz
             write(*,'(a,i10)')   '        nel       =',nel
             write(*,'(a,i10)')   '        nqel      =',nqel
             write(*,'(a,i10)')   '        mV        =',mV
             write(*,'(a,i10)')   '        mP        =',mP
             write(*,'(a,i10)')   '        mT        =',mT
             write(*,'(a,i10)')   '        NV        =',NV
             write(*,'(a,i10)')   '        NP        =',NP
if (use_T)   write(*,'(a,i10)')   '        NT        =',NT
             write(*,'(a,i10)')   '        NfemV     =',NfemV
             write(*,'(a,i10)')   '        NfemP     =',NfemP
if (use_T)   write(*,'(a,i10)')   '        NfemT     =',NfemT
             write(*,'(a,i10)')   '        Nq        =',Nq
             write(*,'(a,i10)')   '        ncorners  =',ncorners
             write(*,'(a,l10)')   '        use_MUMPS =',use_MUMPS
             write(*,'(a,i10)')   '        nmat      =',nmat
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
call material_layout
call paint_swarm
call output_swarm
call matrix_setup_K
call matrix_setup_GT

do istep=1,nstep !-----------------------------------------
                                                          !
   call spacer_istep                                      !
                                                          !
   if (solve_stokes_system) then                          !
                                                          !
      call assign_values_to_qpoints                       !
      call define_bcV                                     !
      call make_matrix                                    !
      call solve_stokes                                   !
      call interpolate_onto_nodes                         !
                                                          !
   else                                                   !
                                                          !
      call prescribe_stokes_solution                      !
                                                          !
   end if                                                 !
                                                          !
   call postprocessors                                    !
   call output_solution                                   !
   call output_qpoints                                    !
                                                          !
end do !---------------------------------------------------

call spacer_end

call footer

end program

!==================================================================================================!
!==================================================================================================!
