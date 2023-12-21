!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_abcd_2D(n,x,y,rho,eta,a_rho,b_rho,c_rho,d_rho,a_eta,b_eta,c_eta,d_eta)

use module_constants, only: three

implicit none

integer, intent(in) :: n
real(8), dimension(n), intent(in) :: x,y,rho,eta
real(8), intent(out) :: a_rho,b_rho,c_rho,d_rho,a_eta,b_eta,c_eta,d_eta

real(8) rcond
real(8) A2D(3,3),B2D(3),work2D(3)
integer ipvt2D(3),job

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_abcd\_2D}
!@@ This subroutine computes the coefficients $a_\rho$ ,$b_\rho$, $c_\rho$,
!@@ and $a_\eta$, $b_\eta$, $c\_eta$ for each element so that 
!@@ inside the element the density and viscosity and linear fields given by
!@@ \[
!@@ \rho(x,y)=a_\rho + b_\rho x + c_\rho y
!@@ \qquad
!@@ \eta(x,y)=a_\eta + b_\eta x + c_\eta y
!@@ \]
!==================================================================================================!

!build matrix 
A2D(1,1)=n
A2D(1,2)=sum(x) 
A2D(1,3)=sum(y) 
A2D(2,1)=sum(x) 
A2D(2,2)=sum(x*x) 
A2D(2,3)=sum(x*y) 
A2D(3,1)=sum(y) 
A2D(3,2)=sum(y*x) 
A2D(3,3)=sum(y*y)
call DGECO (A2D, three, three, ipvt2D, rcond, work2D)

!----------------------------------------------------------
! build rhs for density and solve
B2D(1)=sum(rho)
B2D(2)=sum(x*rho)
B2D(3)=sum(y*rho)

job=0
call DGESL (A2D, three, three, ipvt2D, B2D, job)
a_rho=B2D(1)
b_rho=B2D(2)
c_rho=B2D(3)
d_rho=0d0

!----------------------------------------------------------
! build rhs for viscosity and solve
B2D(1)=sum(eta)
B2D(2)=sum(x*eta)
B2D(3)=sum(y*eta)

job=0
call DGESL (A2D, three, three, ipvt2D, B2D, job)
a_eta=B2D(1)
b_eta=B2D(2)
c_eta=B2D(3)
d_eta=0d0

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine compute_abcd_3D(n,x,y,z,rho,eta,a_rho,b_rho,c_rho,d_rho,a_eta,b_eta,c_eta,d_eta)

use module_constants, only: four

implicit none

integer, intent(in) :: n
real(8), dimension(n), intent(in) :: x,y,z,rho,eta
real(8), intent(out) :: a_rho,b_rho,c_rho,d_rho,a_eta,b_eta,c_eta,d_eta

real(8) rcond
real(8) A3D(4,4),B3D(4),work3D(4)
integer ipvt3D(4),job

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_abcd\_3D}
!@@ This subroutine computes the coefficients $a_\rho$, $b_\rho$, $c_\rho$, $d_\rho$
!@@ and $a_\eta$, $b_\eta$, $c\_eta$, $d_\eta$ for each element so that 
!@@ inside the element the density and viscosity and linear fields given by
!@@ \[
!@@ \rho(x,y)=a_\rho + b_\rho x + c_\rho y + d_\rho z
!@@ \qquad
!@@ \eta(x,y)=a_\eta + b_\eta x + c_\eta y + d_\eta z
!@@ \]
!@@
!==================================================================================================!

!build matrix 
A3D(1,1)=n      ; A3D(1,2)=sum(x)   ; A3D(1,3)=sum(y)   ; A3D(1,4)=sum(z) 
A3D(2,1)=sum(x) ; A3D(2,2)=sum(x*x) ; A3D(2,3)=sum(x*y) ; A3D(2,4)=sum(x*z) 
A3D(3,1)=sum(y) ; A3D(3,2)=sum(y*x) ; A3D(3,3)=sum(y*y) ; A3D(3,4)=sum(y*z)
A3D(4,1)=sum(z) ; A3D(4,2)=sum(z*x) ; A3D(4,3)=sum(z*y) ; A3D(4,4)=sum(z*z)
call DGECO (A3D, four, four, ipvt3D, rcond, work3D)

! build rhs for density and solve
B3D(1)=sum(rho)
B3D(2)=sum(x*rho)
B3D(3)=sum(y*rho)
B3D(4)=sum(z*rho)

job=0
call DGESL (A3D, four, four, ipvt3D, B3D, job)
a_rho=B3D(1)
b_rho=B3D(2)
c_rho=B3D(3)
d_rho=B3D(4)

! build rhs for viscosity and solve
B3D(1)=sum(eta)
B3D(2)=sum(x*eta)
B3D(3)=sum(y*eta)
B3D(3)=sum(z*eta)
job=0

call DGESL (A3D, four, four, ipvt3D, B3D, job)
a_eta=B3D(1)
b_eta=B3D(2)
c_eta=B3D(3)
d_eta=B3D(4)

! filter for over/undershoot

!mesh(iel)%b_rho=0 
!mesh(iel)%c_rho=0 
!mesh(iel)%d_rho=0 

!mesh(iel)%b_eta=0 
!mesh(iel)%c_eta=0 
!mesh(iel)%d_eta=0 

end subroutine

!==================================================================================================!
!==================================================================================================!
























        


!A2D(1,1)=mesh(iel)%nmarker
!        A2D(1,2)=sum(x(1:mesh(iel)%nmarker)) 
!        A2D(1,3)=sum(y(1:mesh(iel)%nmarker)) 
!        A2D(2,1)=sum(x(1:mesh(iel)%nmarker)) 
!        A2D(2,2)=sum(x(1:mesh(iel)%nmarker)*x(1:mesh(iel)%nmarker)) 
!        A2D(2,3)=sum(x(1:mesh(iel)%nmarker)*y(1:mesh(iel)%nmarker)) 
!        A2D(3,1)=sum(y(1:mesh(iel)%nmarker)) 
!        A2D(3,2)=sum(y(1:mesh(iel)%nmarker)*x(1:mesh(iel)%nmarker)) 
!        A2D(3,3)=sum(y(1:mesh(iel)%nmarker)*y(1:mesh(iel)%nmarker))
!        call DGECO (A2D, three, three, ipvt2D, rcond, work2D)

        ! build rhs for density and solve
!        B2D(1)=sum(rho(1:mesh(iel)%nmarker))
!        B2D(2)=sum(x(1:mesh(iel)%nmarker)*rho(1:mesh(iel)%nmarker))
!        B2D(3)=sum(y(1:mesh(iel)%nmarker)*rho(1:mesh(iel)%nmarker))

!        job=0
!        call DGESL (A2D, three, three, ipvt2D, B2D, job)
!        mesh(iel)%a_rho=B2D(1)
!        mesh(iel)%b_rho=B2D(2)
!        mesh(iel)%c_rho=B2D(3)
!        mesh(iel)%d_rho=0d0

        ! build rhs for viscosity and solve
!        B2D(1)=sum(eta(1:mesh(iel)%nmarker))
!        B2D(2)=sum(x(1:mesh(iel)%nmarker)*eta(1:mesh(iel)%nmarker))
!        B2D(3)=sum(y(1:mesh(iel)%nmarker)*eta(1:mesh(iel)%nmarker))
!        job=0
!        call DGESL (A2D, three, three, ipvt2D, B2D, job)
!        mesh(iel)%a_eta=B2D(1)
!        mesh(iel)%b_eta=B2D(2)
!        mesh(iel)%c_eta=B2D(3)
!        mesh(iel)%d_eta=0d0

        ! filter for over/undershoot

!        mesh(iel)%b_rho=0 
!        mesh(iel)%c_rho=0 
!        mesh(iel)%b_eta=0 
!        mesh(iel)%c_eta=0 

