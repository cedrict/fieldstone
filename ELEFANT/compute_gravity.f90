!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_gravity

use module_parameters
use module_gravity
use module_mesh 
use module_constants
use module_timing

implicit none

integer plane_nnp,counter,i,j
real(8) dx,dy,dz
real(8), dimension(:), allocatable :: plane_x,plane_y
real(8), dimension(:), allocatable :: plane_gx,plane_gy,plane_gz,plane_U
real(8), dimension(:), allocatable :: line_x,line_y,line_z,line_r
real(8), dimension(:), allocatable :: line_gx,line_gy,line_gz,line_U,line_g

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_gravity}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==3 .and. geometry=='cartesian') then 

plane_nnp=plane_nnx*plane_nny

if (plane_nnp>0) then
   allocate(plane_x(plane_nnp))
   allocate(plane_y(plane_nnp))
   allocate(plane_gx(plane_nnp)) ; plane_gx=0
   allocate(plane_gy(plane_nnp)) ; plane_gy=0
   allocate(plane_gz(plane_nnp)) ; plane_gz=0
   allocate(plane_U(plane_nnp))  ; plane_U=0

   counter=0
   do j=1,plane_nny
   do i=1,plane_nnx
      counter=counter+1
      plane_x(counter)=(i-1)*(plane_xmax-plane_xmin)/float(plane_nnx-1)+plane_xmin
      plane_y(counter)=(j-1)*(plane_ymax-plane_ymin)/float(plane_nny-1)+plane_ymin
   end do
   end do

   open(unit=123,file="OUTPUT/GRAVITY/gravity_plane.ascii",action="write")
   write(123,'(a)') '#x, y, gx, gy, gz, U'
   do i=1,plane_nnp
   do iel=1,nel
      call gravity_at_point(plane_x(i),plane_y(i),plane_height,&
                            mesh(iel),plane_gx(i),plane_gy(i),plane_gz(i),plane_U(i))
   end do
   write(123,'(6es12.4)') plane_x(i),plane_y(i),plane_gx(i),plane_gy(i),plane_gz(i),plane_U(i)
   end do
   close(123)

   deallocate(plane_x)
   deallocate(plane_y)
   deallocate(plane_gx)
   deallocate(plane_gy)
   deallocate(plane_gz)
   deallocate(plane_U)

end if

!----------------------------------------------------------

if (line_nnp>0) then

   allocate(line_x(line_nnp))
   allocate(line_y(line_nnp))
   allocate(line_z(line_nnp))
   allocate(line_r(line_nnp))
   allocate(line_gx(line_nnp)) ; line_gx=0
   allocate(line_gy(line_nnp)) ; line_gy=0
   allocate(line_gz(line_nnp)) ; line_gz=0
   allocate(line_U(line_nnp))  ; line_U=0
   allocate(line_g(line_nnp))  ; line_g=0

   dx=(xend-xbeg)/(line_nnp-1)
   dy=(yend-ybeg)/(line_nnp-1)
   dz=(zend-zbeg)/(line_nnp-1)

   open(unit=123,file="OUTPUT/GRAVITY/gravity_line.ascii",action="write")
   write(123,'(a)') '#x, y, z, gx, gy, gz, U'
   do i=1,line_nnp
      line_x(i)=xbeg+(i-1)*dx
      line_y(i)=ybeg+(i-1)*dy
      line_z(i)=zbeg+(i-1)*dz
      line_r(i)=sqrt((line_x(i)-xbeg)**2+(line_y(i)-ybeg)**2+(line_z(i)-zbeg)**2)
      do iel=1,nel
      call gravity_at_point(line_x(i),line_y(i),line_z(i),&
                            mesh(iel),line_gx(i),line_gy(i),line_gz(i),line_U(i))
      end do
      line_g(i)=sqrt(line_gx(i)**2+line_gy(i)**2+line_gz(i)**2)
      write(123,'(9es12.4)') line_x(i),line_y(i),line_z(i),&
                             line_gx(i),line_gy(i),line_gz(i),line_U(i),&
                             line_r(i),line_g(i)
   end do
   close(123)      

   deallocate(line_x)
   deallocate(line_y)
   deallocate(line_z)
   deallocate(line_gx)
   deallocate(line_gy)
   deallocate(line_gz)
   deallocate(line_U)

end if

end if 

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'compute_gravity:',elapsed,' s                |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
! note that this subroutine updates its arguments U,gx,gy,gz and 
! does not zero them!

subroutine gravity_at_point(xM,yM,zM,elt,gx,gy,gz,U)

use module_gravity, only: grav_pointmass,grav_prism
use module_mesh, only: element
use module_constants, only: pi,pi2,eps,Ggrav

implicit none

real(8), intent(in) :: xM,yM,zM
type(element), intent(in) ::  elt
real(8), intent(inout) :: U,gx,gy,gz

integer :: i,j,k
real(8) :: r,Coeff,x(2),y(2),z(2)
real(8) :: arctan_x,arctan_y,arctan_z,log_x,log_y,log_z

!----------------------------------------------------------

if (grav_pointmass) then

   r=sqrt((xM-elt%xc)**2+(yM-elt%yc)**2+(zM-elt%zc)**2)

   Coeff=Ggrav*elt%rho_avrg*elt%vol/r
   U=U-Coeff
      
   Coeff=Coeff/r**2
   gx=gx+Coeff*(xM-elt%xc)
   gy=gy+Coeff*(yM-elt%yc)
   gz=gz+Coeff*(zM-elt%zc)

end if

!----------------------------------------------------------

if (grav_prism) then

   x(1)=elt%xV(1)-xM ; x(2)=x(1)+elt%hx
   y(1)=elt%yV(1)-yM ; y(2)=y(1)+elt%hy
   z(1)=elt%zV(1)-zM ; z(2)=z(1)+elt%hz

   do i=1,2
      do j=1,2
         do k=1,2
            r=sqrt(x(i)**2+y(j)**2+z(k)**2)

            ! There are cases where the calculations will fail, f.e. log(0) or arctan(1/0). 
            ! To stop this, exception cases are defined. They are set to the limit values, 
            ! so log(0) -> 0 and arctan(1/0) -> 1/2 pi

            if (abs(x(i))<eps*elt%hx) then
               arctan_x = pi2
            else
               arctan_x = atan((y(j)*z(k)/(x(i)*r)))
            end if

            if (abs(y(i))<eps*elt%hy) then
               arctan_y = pi2
            else
               arctan_y = atan((x(i)*z(k)/(y(j)*r)))
            end if

            if (abs(z(i))<eps*elt%hz) then
               arctan_z = pi2
            else
               arctan_z = atan((x(i)*y(j)/(z(k)*r)))
            end if
 
            ! Along with the exceptions, there is a variant of the equations that should 
            ! offer extra numerical stability. 

            if (abs(r+x(i))<eps*elt%hx) then
               log_x = 0d0
            else
               log_x = log((x(i)+r)/sqrt(y(j)**2+z(k)**2))
            end if

            if (abs(r+y(i))<eps*elt%hy) then
               log_y = 0d0
            else
               log_y = log((y(j)+r)/sqrt(x(i)**2+z(k)**2))
            end if

            if (abs(r+z(k))<eps*elt%hz) then
               log_z = 0d0
            else
               log_z = log((z(k)+r)/sqrt(x(i)**2+y(j)**2))
            end if

            Coeff=(-1)**(i+j+k)*Ggrav*elt%rho_avrg

            U=U+Coeff*(y(j)*z(k)*log_x &
                      +x(i)*y(j)*log_z &
                      +x(i)*z(k)*log_y &
                      -0.5d0*x(i)**2*arctan_x &
                      -0.5d0*y(j)**2*arctan_y &
                      -0.5d0*z(k)**2*arctan_z)

            gx=gx+Coeff*(z(k)*log_y+y(j)*log_z-x(i)*arctan_x)
            gy=gy+Coeff*(z(k)*log_x+x(i)*log_z-y(j)*arctan_y)
            gz=gz+Coeff*(x(i)*log_y+y(j)*log_x-z(k)*arctan_z)

         end do
      end do
   end do

end if

end subroutine

!==================================================================================================!
