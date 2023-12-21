!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine dNNNds(r,s,t,dNds,m,ndim,space)

use module_constants, only: aa,bb,cc,dd,ee

implicit none
integer, intent(in) :: m,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNds(m)
character(len=4), intent(in) :: space
real(8), external :: dBubbleds
real(8) Nmr,Nlr,Nrr,dNls,dNms,dNrs,Nlt,Nmt,Nrt
real(8) db1ds,db2ds

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{dNNNds}
!@@ Spaces supported so far:
!@@ 2D: Q1, Q2, Q3, Q4, Q1+
!@@ 3D: Q1, Q2, Q1++
!==================================================================================================!

if (ndim==2) then

   select case(space)
   case('__Q1')
      dNds(1)=-0.25*(1-r)
      dNds(2)=-0.25*(1+r)
      dNds(3)=+0.25*(1+r)
      dNds(4)=+0.25*(1-r)
   case('__Q2')
      dNds(1)= 0.5d0*r*(r-1.d0) * 0.5d0*(2d0*s-1d0)
      dNds(2)=      (1.d0-r**2) * 0.5d0*(2d0*s-1d0)
      dNds(3)= 0.5d0*r*(r+1.d0) * 0.5d0*(2d0*s-1d0)
      dNds(4)= 0.5d0*r*(r-1.d0) *          (-2d0*s)
      dNds(5)=      (1.d0-r**2) *          (-2d0*s)
      dNds(6)= 0.5d0*r*(r+1.d0) *          (-2d0*s)
      dNds(7)= 0.5d0*r*(r-1.d0) * 0.5d0*(2d0*s+1d0)
      dNds(8)=      (1.d0-r**2) * 0.5d0*(2d0*s+1d0)
      dNds(9)= 0.5d0*r*(r+1.d0) * 0.5d0*(2d0*s+1d0)
   case('_Q1+')
      dNds(1)=-0.25*(1-r)-0.25d0*dBubbleds(r,s)
      dNds(2)=-0.25*(1+r)-0.25d0*dBubbleds(r,s)
      dNds(3)=+0.25*(1+r)-0.25d0*dBubbleds(r,s)
      dNds(4)=+0.25*(1-r)-0.25d0*dBubbleds(r,s)
      dNds(5)=dBubbleds(r,s)      
   case('__Q3')
   case('__Q4')
   case default
      stop 'unknown 2D space in dNNNds'
   end select

else

   select case(space)
   case('__Q1')
      dNds(1)=-0.125*(1-r)*(1-t)
      dNds(2)=-0.125*(1+r)*(1-t)
      dNds(3)=+0.125*(1+r)*(1-t)
      dNds(4)=+0.125*(1-r)*(1-t)
      dNds(5)=-0.125*(1-r)*(1+t)
      dNds(6)=-0.125*(1+r)*(1+t)
      dNds(7)=+0.125*(1+r)*(1+t)
      dNds(8)=+0.125*(1-r)*(1+t)
   case('__Q2')
      Nlr=0.5d0*r*(r-1d0) ; dNls=s-0.5d0 ; Nlt=0.5d0*t*(t-1d0) 
      Nmr=(1d0-r**2)      ; dNms=-2d0*s  ; Nmt=(1d0-t**2)       
      Nrr=0.5d0*r*(r+1d0) ; dNrs=s+0.5d0 ; Nrt=0.5d0*t*(t+1d0) 
      dNds(01)= Nlr * dNls * Nlt 
      dNds(02)= Nmr * dNls * Nlt 
      dNds(03)= Nrr * dNls * Nlt 
      dNds(04)= Nlr * dNms * Nlt 
      dNds(05)= Nmr * dNms * Nlt 
      dNds(06)= Nrr * dNms * Nlt 
      dNds(07)= Nlr * dNrs * Nlt 
      dNds(08)= Nmr * dNrs * Nlt 
      dNds(09)= Nrr * dNrs * Nlt 
      dNds(10)= Nlr * dNls * Nmt 
      dNds(11)= Nmr * dNls * Nmt 
      dNds(12)= Nrr * dNls * Nmt 
      dNds(13)= Nlr * dNms * Nmt 
      dNds(14)= Nmr * dNms * Nmt 
      dNds(15)= Nrr * dNms * Nmt 
      dNds(16)= Nlr * dNrs * Nmt 
      dNds(17)= Nmr * dNrs * Nmt 
      dNds(18)= Nrr * dNrs * Nmt 
      dNds(19)= Nlr * dNls * Nrt 
      dNds(20)= Nmr * dNls * Nrt 
      dNds(21)= Nrr * dNls * Nrt 
      dNds(22)= Nlr * dNms * Nrt 
      dNds(23)= Nmr * dNms * Nrt 
      dNds(24)= Nrr * dNms * Nrt 
      dNds(25)= Nlr * dNrs * Nrt 
      dNds(26)= Nmr * dNrs * Nrt 
      dNds(27)= Nrr * dNrs * Nrt 
   case('Q1++')
      db1ds=(27d0/32d0)**3*(1-r**2)*(1-t**2)*(1-r)*(1-t)*(-1-2*s+3*s**2)
      db2ds=(27d0/32d0)**3*(1-r**2)*(1-t**2)*(1+r)*(1+t)*( 1-2*s-3*s**2)
      dNds(01)=-0.125*(1-r)*(1-t) -aa*db1ds
      dNds(02)=-0.125*(1+r)*(1-t) -aa*bb*db1ds-aa*cc*db2ds
      dNds(03)=+0.125*(1+r)*(1-t) -aa*cc*db1ds-aa*bb*db2ds
      dNds(04)=+0.125*(1-r)*(1-t) -aa*bb*db1ds-aa*cc*db2ds
      dNds(05)=-0.125*(1-r)*(1+t) -aa*bb*db1ds-aa*cc*db2ds
      dNds(06)=-0.125*(1+r)*(1+t) -aa*cc*db1ds-aa*bb*db2ds
      dNds(07)=+0.125*(1+r)*(1+t) -aa*db2ds
      dNds(08)=+0.125*(1-r)*(1+t) -aa*cc*db1ds-aa*bb*db2ds
      dNds(09)= dd*db1ds-ee*db2ds
      dNds(10)=-ee*db1ds+dd*db2ds
   case default
      stop 'unknown 3D space in dNNNds'
   end select

end if


end subroutine

!==================================================================================================!
!==================================================================================================!

function dBubbleds(r,s)
implicit none
real(8) r,s,dBubbleds
real(8), parameter :: beta=0.25d0
dBubbleds=(r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)
end function
