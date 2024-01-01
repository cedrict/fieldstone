!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine estimate_memory_use

use module_parameters, only: iproc,debug,nel
use module_mesh 
!use module_constants
!use module_swarm
!use module_materials
!use module_arrays
use module_timing

implicit none

integer :: mem

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{estimate\_memory\_use}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

mem=0

if (allocated(mesh(1)%iconV)) mem=mem+size(mesh(1)%iconV)*4
if (allocated(mesh(1)%iconP)) mem=mem+size(mesh(1)%iconP)*4
if (allocated(mesh(1)%iconT)) mem=mem+size(mesh(1)%iconT)*4
if (allocated(mesh(1)%iconM)) mem=mem+size(mesh(1)%iconM)*4
if (allocated(mesh(1)%xV))  mem=mem+size(mesh(1)%xV)*8
if (allocated(mesh(1)%yV))  mem=mem+size(mesh(1)%yV)*8
if (allocated(mesh(1)%zV))  mem=mem+size(mesh(1)%zV)*8
if (allocated(mesh(1)%xP))  mem=mem+size(mesh(1)%xP)*8
if (allocated(mesh(1)%yP))  mem=mem+size(mesh(1)%yP)*8
if (allocated(mesh(1)%zP))  mem=mem+size(mesh(1)%zP)*8
if (allocated(mesh(1)%xT))  mem=mem+size(mesh(1)%xT)*8
if (allocated(mesh(1)%yT))  mem=mem+size(mesh(1)%yT)*8
if (allocated(mesh(1)%zT))  mem=mem+size(mesh(1)%zT)*8
if (allocated(mesh(1)%xM))  mem=mem+size(mesh(1)%xM)*8
if (allocated(mesh(1)%yM))  mem=mem+size(mesh(1)%yM)*8
if (allocated(mesh(1)%zM))  mem=mem+size(mesh(1)%zM)*8
if (allocated(mesh(1)%u))   mem=mem+size(mesh(1)%u)*8
if (allocated(mesh(1)%v))   mem=mem+size(mesh(1)%v)*8
if (allocated(mesh(1)%w))   mem=mem+size(mesh(1)%w)*8
if (allocated(mesh(1)%p))   mem=mem+size(mesh(1)%p)*8
if (allocated(mesh(1)%q))   mem=mem+size(mesh(1)%q)*8
if (allocated(mesh(1)%T))   mem=mem+size(mesh(1)%T)*8
if (allocated(mesh(1)%qx))  mem=mem+size(mesh(1)%qx)*8
if (allocated(mesh(1)%qy))  mem=mem+size(mesh(1)%qy)*8
if (allocated(mesh(1)%qz))  mem=mem+size(mesh(1)%qz)*8
if (allocated(mesh(1)%rV))      mem=mem+size(mesh(1)%rV)*8
if (allocated(mesh(1)%rP))      mem=mem+size(mesh(1)%rP)*8
if (allocated(mesh(1)%thetaV))  mem=mem+size(mesh(1)%thetaV)*8
if (allocated(mesh(1)%thetaP))  mem=mem+size(mesh(1)%thetaP)*8
if (allocated(mesh(1)%phiV))    mem=mem+size(mesh(1)%phiP)*8
if (allocated(mesh(1)%xq))      mem=mem+size(mesh(1)%xq)*8
if (allocated(mesh(1)%yq))      mem=mem+size(mesh(1)%yq)*8
if (allocated(mesh(1)%zq))      mem=mem+size(mesh(1)%zq)*8
if (allocated(mesh(1)%JxWq))    mem=mem+size(mesh(1)%JxWq)*8
if (allocated(mesh(1)%weightq)) mem=mem+size(mesh(1)%weightq)*8
if (allocated(mesh(1)%rq))      mem=mem+size(mesh(1)%rq)*8
if (allocated(mesh(1)%sq))      mem=mem+size(mesh(1)%sq)*8
if (allocated(mesh(1)%tq))      mem=mem+size(mesh(1)%tq)*8
if (allocated(mesh(1)%gxq))     mem=mem+size(mesh(1)%gxq)*8
if (allocated(mesh(1)%gyq))     mem=mem+size(mesh(1)%gyq)*8
if (allocated(mesh(1)%gzq))     mem=mem+size(mesh(1)%gzq)*8
if (allocated(mesh(1)%pq))      mem=mem+size(mesh(1)%pq)*8
if (allocated(mesh(1)%tempq))   mem=mem+size(mesh(1)%tempq)*8
if (allocated(mesh(1)%etaq))    mem=mem+size(mesh(1)%etaq)*8
if (allocated(mesh(1)%rhoq))    mem=mem+size(mesh(1)%rhoq)*8
if (allocated(mesh(1)%hcondq))  mem=mem+size(mesh(1)%hcondq)*8
if (allocated(mesh(1)%hcapaq))  mem=mem+size(mesh(1)%hcapaq)*8
if (allocated(mesh(1)%hprodq))  mem=mem+size(mesh(1)%hprodq)*8

!--------------------------
!integer :: list_of_markers(200)  

mem=mem+size(mesh(1)%list_of_markers)*4

!--------------------------
! real(8) :: exx,eyy,exy  
! real(8) :: ezz,exz,eyz 

mem=mem+6*8

!--------------------------
!integer :: ielx,iely,ielz 
!integer :: nmarker   

mem=mem+4*4

!--------------------------
!real(8) :: a_eta,b_eta,c_eta,d_eta   
!real(8) :: a_rho,b_rho,c_rho,d_rho  
!real(8) :: vol                     
!real(8) :: rho_avrg               
!real(8) :: eta_avrg              
!real(8) :: xc,yc,zc             
!real(8) :: hx,hy,hz            
!real(8) :: hr,htheta,hphi

mem=mem+19*8


!---------------------------
!logical(1) :: bnd1_elt      
!logical(1) :: bnd2_elt     
!logical(1) :: bnd3_elt    
!logical(1) :: bnd4_elt   
!logical(1) :: bnd5_elt  
!logical(1) :: bnd6_elt   
!logical(1) :: inner_elt  
!logical(1) :: outer_elt  

mem=mem+8

!  logical(1), allocatable :: bnd1_node(:)     ! flags for nodes on x=0 boundary  
!  logical(1), allocatable :: bnd2_node(:)     ! flags for nodes on x=Lx boundary  
!  logical(1), allocatable :: bnd3_node(:)     ! flags for nodes on y=0 boundary  
!  logical(1), allocatable :: bnd4_node(:)     ! flags for nodes on y=Ly boundary  
!  logical(1), allocatable :: bnd5_node(:)     ! flags for nodes on z=0 boundary  
!  logical(1), allocatable :: bnd6_node(:)     ! flags for nodes on z=Lz boundary  
!  logical(1), allocatable :: inner_node(:)    ! flags for nodes on inner boundary of annulus/shell
!  logical(1), allocatable :: outer_node(:)    ! flags for nodes on outer boundary of annulus/shell
!  logical(1), allocatable :: fix_u(:)         ! whether a given velocity dof is prescribed
!  logical(1), allocatable :: fix_v(:)         ! whether a given velocity dof is prescribed
!  logical(1), allocatable :: fix_w(:)         ! whether a given velocity dof is prescribed
!  logical(1), allocatable :: fix_T(:)         ! whether a given temperature dof is prescribed

!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1
!if (allocated(mesh(1)%))  mem=mem+size(mesh(1)%)*1

!----------------------------------------------------------

mem=mem*nel

write(*,'(a,i7,a)') shift//'mem mesh~',mem,' bytes'
write(*,'(a,i7,a)') shift//'mem mesh~',mem/1024,' kbytes'

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'estimate_memory_use:',elapsed,' s            |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
