program elefant

use global_parameters
use structures

implicit none

call header

#ifdef UseMUMPS
print *,'with MUMPS support'
!include 'mpif.h'
!call mpi_init(ierr)
!call mpi_comm_size (mpi_comm_world,nproc,ierr)
!call mpi_comm_rank (mpi_comm_world,iproc,ierr)
!call mpi_get_processor_name(procname,resultlen,ierr)
#else
print *,'no MUMPS support'
#endif

call spacer
call set_default_values
call declare_main_parameters
call define_material_properties

!--------------------------------------

if (pair=='q1p0') then
   mV=2**ndim
   mP=1
   mT=2**ndim
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
   mP=2**ndim
   mT=2**ndim
   if (ndim==2) then
      mV=2**ndim+1
      nel=nelx*nely
      NV=(nelx+1)*(nely+1)+nel
      NT=(nelx+1)*(nely+1)
      NP=(nelx+1)*(nely+1)
   else
      mV=2**ndim+2
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)+2*nel
      NT=(nelx+1)*(nely+1)*(nelz+1)
      NP=(nelx+1)*(nely+1)*(nelz+1)
   end if
end if

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

solve_stokes_system=.true.
nstep=1

!----------------------------
write(*,*) '          ndim     =',ndim
write(*,*) '          geometry = ',geometry
write(*,*) '          pair     = ',pair
write(*,*) '          Lx       =',Lx
write(*,*) '          Ly       =',Ly
write(*,*) '          Lz       =',Lz
write(*,*) '          nelx     =',nelx
write(*,*) '          nely     =',nely
write(*,*) '          nelz     =',nelz
write(*,*) '          nel      =',nel
write(*,*) '          nqel     =',nqel
write(*,*) '          NV       =',NV
write(*,*) '          NP       =',NP
write(*,*) '          NT       =',NT
write(*,*) '          NfemV    =',NfemV
write(*,*) '          NfemP    =',NfemP
write(*,*) '          NfemT    =',NfemT
write(*,*) '          Nq       =',Nq
write(*,*) '          ncorners =',ncorners
write(*,*) '          use_MUMPS=',use_MUMPS
!----------------------------


!-------------------------------------------------
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

call markers_setup
call material_layout
!call material_paint
call output_swarm

call matrix_setup_K
call matrix_setup_GT

do istep=1,nstep !-----------------------------------------

   call spacer_istep                                      !
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

call spacer_end
call postprocessors
call output_solution
call output_qpoints
call footer

end program
