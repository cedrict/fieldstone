module modelProperties

! model geometry
double precision, parameter :: circleRadius    = 2000.0 ! meter
double precision, parameter :: circleDepth     = 7000.0 ! meter (depth of center)

double precision, parameter :: r = circleRadius ! Verruijtian
double precision, parameter :: h = circleDepth  ! Verruijtian

! boundary condition
double precision, parameter :: stressMagnitude = 5e6 ! Pa

double precision, parameter :: t = stressMagnitude  ! Verruijtian

! material properties
double precision, parameter :: PoissonRatio    = 0.35
double precision, parameter :: YoungsModulus   = 5e10  ! so that shear Mod = 2e10
double precision, parameter :: mu              = YoungsModulus / (2d0 * (1d0+PoissonRatio ))  ! shear modulus

! derived model constants

! compute alpha
!
!  r      2 alpha                    h              h^2
! --- = -----------   ----> alpha = ---  +-  sqrt( ----- - 1 )
!  h    1 + alpha^2                  r              r^2

! + makes alpha larger than 1, which is invalid.
! - makes 0 < alpha < 1,       which is what we want
double precision, parameter :: alpha = h / r - sqrt(h**2/r**2 - 1d0)
complex, parameter          :: i = cmplx(0.0, 1.0)
complex, parameter          :: vf = -i * h * ((1.0-alpha**2)/(1.0+alpha**2))

! scaling constant
double precision, parameter :: P = alpha**2 * t * h / ( (1d0-alpha**2) * (1d0-alpha**4) )

! translate material properties to Verruijtian
double precision, parameter :: kappa = 3d0 - 4d0 * PoissonRatio                   ! plane strain
!double precision, parameter :: kappa = (3.0 - PoissonRatio)/(1.0 + PoissonRatio)  ! plane stress


end module


program analytic

use modelProperties

! computes the analytic solution of Verruijt for 
! surface displacement due to a pressurized circle in a half plane  (radial stress)

implicit none

double precision :: x, y
double precision :: dx, dy

integer, parameter :: nPoints = 20

integer :: iPoint

logical            :: eof
character(len=255) :: arg, record
integer            :: narg

! if a file argument has been provided, compute the analytical
! solution at the coordinates at the coordinates in the file.

narg = command_argument_count()

if (narg.eq.1) then
    call get_command_argument(1,arg)

    eof = .false.
    open(unit=42, file=trim(arg))

    do while (.not. eof)

        read(42,"(a)") record

        if (record(1:3).eq."eof") then
            eof = .true.
        else
            read(record,*) x, y
            call displacementAt(x, y, dx, dy)
            write(*,*) x, y, dx, dy
        end if

    end do

else
    ! no file with coordinates given, compute surface disp in [0,20000 m]
    do iPoint=1,1000
        x = (iPoint-1) * 20.0
        y = 0.01
        call displacementAt(x, y, dx, dy)
        write(*,*) x, y, dx, dy

    enddo
endif


close(42)

end program

!----------------------------------------------------------------

subroutine displacementAt(x, y, dx, dy)

use modelProperties


! Verruijt only gives the backward transformation from the complex projection 
! zeta (xi + i eta) to the R2 z (x + iy)
! as:
! z = -i * h * ((1.0-alpha**2)/(1.0+alpha**2)) * ((1.0+zeta)/(1.0-zeta))
! with
! alpha = h / r - sqrt(h**2/r**2 - 1d0)
!
! We construct the forward transformation by setting 
! vf = -i * h * ((1.0-alpha**2)/(1.0+alpha**2)) 
! so that the backward transformation simplifies to
! z = vf * ((1.0+zeta)/(1.0-zeta))
! so that elementary algebray brings us to:
! zeta = (z - vf) / (z + vf)
!
! tested with various x and y pairs



implicit none

! internal variables
double precision :: x, y
double precision :: dx, dy
double precision :: xi, eta


complex          :: z, zeta ! coordinate pairs

complex, external :: complexConjugate
complex, external :: zeta2z, z2zeta

complex          :: displacement
complex          :: phi, psi, phiaccent

! project the coordinate into the complex plane
z = cmplx (x, y)
zeta =  z2zeta(z)

! from zeta, compute the stress functions phi and psi and the derivative of phi:
phi = -2.0 * i * (1.0 + alpha**2) + &   ! a0
       2.0 * i * zeta + &               ! a1
       2.0 * i * alpha**2 / zeta        ! b1

psi = -3.0 * i * (1.0 + alpha**2) + &   ! c0
       2.0 * i * alpha**2 * zeta + &    ! c1
             i * zeta**2 + &            ! c2
       2.0 * i / zeta + &               ! d1
             i * alpha**2 / zeta**2     ! d2

! use chain differentiation!
phiaccent =  (2.0 * i - &                        ! a1'
             (2.0 * i * alpha**2) / zeta**2) / & ! b1'
             (vf * 2 / ((1-zeta)**2) )           


! from these, compute the displacement
displacement = kappa * phi - z * complexConjugate(phiaccent) - complexConjugate(psi)
displacement = -displacement * P / (2.0 * mu)

! and extract the components dx and dy
dx = real(displacement)
dy = aimag(displacement)

end subroutine

complex function z2zeta(z)
use modelProperties, only: vf
implicit none
complex z
z2zeta =  (z - vf) / (z + vf)
end function

complex function zeta2z(zeta)
use modelProperties, only: vf
implicit none
complex zeta
zeta2z = vf * ((1.0+zeta)/(1.0-zeta))
end function

complex function complexConjugate(complexNumber)
implicit none
complex complexNumber
complexConjugate = cmplx(real(complexNumber), -aimag(complexNumber))
end function
