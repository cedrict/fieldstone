!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine outer_solver_pcg

use module_parameters, only: NfemV,NfemP,iproc,debug
!use module_mesh 
!use module_constants
!use module_swarm
!use module_materials
use module_sparse, only : csrGT
use module_arrays, only: SolV, SolP, rhs_f,rhs_h
use module_timing

implicit none

integer :: k
real(8), dimension(:,:), allocatable :: rvect_k
real(8), dimension(:,:), allocatable :: pvect_k
real(8), dimension(:,:), allocatable :: zvect_k
real(8), dimension(:,:), allocatable :: dvect_k
real(8), dimension(:,:), allocatable :: ptildevect_k
real(8), dimension(:,:), allocatable :: GTV
real(8), dimension(:,:), allocatable :: GP
real(8), dimension(:), allocatable :: guess
real(8) :: rvect_0
integer, parameter :: niter=100
!==================================================================================================!
!==================================================================================================!
!@@ \subsection{outer\_solver\_pcg}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!


! declare necessary arrays
allocate(rvect_k(NfemP,1))
allocate(pvect_k(NfemP,1))
allocate(zvect_k(NfemP,1))
allocate(zvect_k(NfemP,1))
allocate(GTV(NfemP,1))
allocate(ptildevect_k(NfemV,1))
allocate(dvect_k(NfemV,1))
allocate(GP(NfemV,1)) 
allocate(guess(NfemV)) 


solP=0.d0 ! we assume that the guess pressure is zero.
solV=0.d0
guess=0.d0

call spmvT(NfemP,NfemV,csrGT%nz,SolP,GP(1:NfemV,1),csrGT%mat,csrGT%ja,csrGT%ia)  ! compute G.P_0

call inner_solver(rhs_f-GP(:,1),guess,solV)                                      ! solve K V_0 = f - G.P_0

call spmv (NfemP,NfemV,csrGT%nz,SolV,GTV(:,1),csrGT%mat,csrGT%ja,csrGT%ia)
rvect_k(:,1)=GTV(:,1)-rhs_h                                                      ! compute r_0

rvect_0=sqrt(dot_product(rvect_k(:,1),rvect_k(:,1)))                             ! compute |r_0|_2 


!   if use_precond:
!      zvect_k=sps.linalg.spsolve(M_mat,rvect_k)                 # compute z_0
!   else:
!      zvect_k=rvect_k

pvect_k=zvect_k                                                                  ! compute p_0

do k=1,niter !------------------------------------------------------+

!       ptildevect_k=G_mat.dot(pvect_k)                              # 
!       Cp=C_mat.dot(pvect_k)                                        # C . p_k
!       pCp=pvect_k.dot(Cp)                                          # p_k . C . p_k
!       dvect_k=sps.linalg.spsolve(K_mat,ptildevect_k)               #
!       alpha=(rvect_k.dot(zvect_k))/(ptildevect_k.dot(dvect_k)+pCp) #
!       solP+=alpha*pvect_k                                          #
!       solV-=alpha*dvect_k                                          #
!       rvect_kp1=rvect_k-alpha*(G_mat.T.dot(dvect_k)+Cp)            #
!       if use_precond:                                              #
!           zvect_kp1=sps.linalg.spsolve(M_mat,rvect_kp1)            #
!       else:                                                        #
!           zvect_kp1=rvect_kp1                                      #
!       beta=(zvect_kp1.dot(rvect_kp1))/(zvect_k.dot(rvect_k))       #
!       pvect_kp1=zvect_kp1+beta*pvect_k                             #
!                                                                    #
!       rvect_k=rvect_kp1                                            #
!       pvect_k=pvect_kp1                                            #
!       zvect_k=zvect_kp1                                            #
!                                                                    #
!       xi=np.linalg.norm(rvect_k)/rvect_0                           #
!       conv_file.write("%d %6e \n"  %(k,xi))                        #
!       conv_file.flush()                                            #
!       print('iter',k,'xi=',xi)                                     #
!       if xi<tol:                                                   #
!          break                                                     #

end do !------------------------------------------------------------+






!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'outer_solver_pcg'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'outer_solver_pcg (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
