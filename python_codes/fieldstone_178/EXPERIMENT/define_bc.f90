!==============================================!
!                                              !
! C. thieulot ; October 2018                   !
!                                              !
!==============================================!

subroutine define_bc(x,y,np,bc_fix,bc_val,Lx,Ly)

implicit none

integer, parameter :: ndof=2
real(8), parameter :: eps=1.d-10           
real(8), parameter :: year=31536000.d0
real(8), parameter :: cm=1.d-2

integer, intent(in) :: np
real(8), dimension(np) :: x,y
real(8), intent(in) :: Lx,Ly
real(8), dimension(ndof*np) :: bc_val
logical, dimension(ndof*np) :: bc_fix

integer i

!==============================================!
! This subroutine loops over all nodes of the mesh
! and for those on the boundary of the domain, 
! it prescribes the desired value for Vx and/or Vy.
!==============================================!

bc_fix=.false.

do i=1,np
   !--------------
   ! left boundary
   !--------------
   if (x(i).lt.eps*Lx) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)= 0              ! prescribing vx 
      !bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0           ! prescribing vy 
   endif
   !--------------
   ! right boundary
   !--------------
   if (x(i).gt.(Lx-eps*Lx)) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0               ! prescribing vx 
      !bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0           ! prescribing vy 
   endif
   !--------------
   ! bottom boundary
   !--------------
   if (y(i).lt.eps*Ly) then
      !bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0           ! prescribing vx 
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0               ! prescribing vy 
   endif
   !--------------
   ! top boundary
   !--------------
   !if (y(i).gt.(Ly-eps*Ly) ) then
      !bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0 ! prescribing vx 
      !bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0 ! prescribing vy 
   !endif
end do

end subroutine
