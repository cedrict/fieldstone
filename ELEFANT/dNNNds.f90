!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine dNNNds(r,s,t,dNds,m,ndim,space,caller)

use module_constants, only: aa,bb,cc,dd,ee

implicit none
integer, intent(in) :: m,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNds(m)
character(len=4), intent(in) :: space
integer, intent(in) :: caller
real(8), external :: dBubbleds
real(8) Nmr,Nlr,Nrr,dNls,dNms,dNrs,Nlt,Nmt,Nrt
real(8) db1ds,db2ds,N1r,N2r,N3r,N4r,dN1sds,dN2sds,dN3sds,dN4sds

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{dNNNds}
!@@ Spaces supported so far:
!@@ \begin{itemize}
!@@ \item 2D: $Q_1$, $Q_2$, $Q_3$, $Q_1^+$, $P_1$, $P_1^+$, $P_2$, $P_2^+$
!@@ \item 3D: $Q_1$, $Q_2$, $Q_1^{++}$
!@@ \end{itemize}
!==================================================================================================!

!print *,'->',caller,space

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
   case('__Q3')
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       dN1sds=( +1 +18*s -27*s**2)/16
       dN2sds=(-27 -18*s +81*s**2)/16
       dN3sds=(+27 -18*s -81*s**2)/16
       dN4sds=( -1 +18*s +27*s**2)/16
       dNds(01)=N1r*dN1sds 
       dNds(02)=N2r*dN1sds 
       dNds(03)=N3r*dN1sds 
       dNds(04)=N4r*dN1sds
       dNds(05)=N1r*dN2sds 
       dNds(06)=N2r*dN2sds 
       dNds(07)=N3r*dN2sds 
       dNds(08)=N4r*dN2sds
       dNds(09)=N1r*dN3sds 
       dNds(10)=N2r*dN3sds 
       dNds(11)=N3r*dN3sds 
       dNds(12)=N4r*dN3sds
       dNds(13)=N1r*dN4sds 
       dNds(14)=N2r*dN4sds 
       dNds(15)=N3r*dN4sds 
       dNds(16)=N4r*dN4sds
   case('_Q1+')
      dNds(1)=-0.25*(1-r)-0.25d0*dBubbleds(r,s)
      dNds(2)=-0.25*(1+r)-0.25d0*dBubbleds(r,s)
      dNds(3)=+0.25*(1+r)-0.25d0*dBubbleds(r,s)
      dNds(4)=+0.25*(1-r)-0.25d0*dBubbleds(r,s)
      dNds(5)=dBubbleds(r,s)      
   case('__P1')
      dNds(1)=-1
      dNds(2)= 0
      dNds(3)= 1
   case('_P1+')
      dNds(1)=-1-9*(1-r-2*s)*r
      dNds(2)=  -9*(1-r-2*s)*r
      dNds(3)= 1-9*(1-r-2*s)*r
      dNds(4)=  27*(1-r-2*s)*r
   case('__P2')
      dNds(1)= -3+4*r+4*s
      dNds(2)= 0
      dNds(3)= -1+4*s
      dNds(4)= -4*r
      dNds(5)= +4*r
      dNds(6)= 4-4*r-8*s
   case('_P2+')
      dNds(1)= -3*r**2+r*(7-6*s)+4*s-3
      dNds(2)= -3*r*(r+2*s-1)
      dNds(3)= -3*r**2+r*(3-6*s)+4*s-1
      dNds(4)= 4*r*(3*r+6*s-4)
      dNds(5)= 4*r*(3*r+6*s-2)
      dNds(6)= 4*(3*r-1)*(r+2*s-1)
      dNds(7)= -27*r*(r+2*s-1)
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
