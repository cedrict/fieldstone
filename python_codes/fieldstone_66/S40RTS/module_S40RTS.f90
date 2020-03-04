!==================================================================================================!
!  GGG   RRRR    AAA   PPPP   EEEEE                                                                !
! G      R   R  A   A  P   P  E                                                                    !
! G  G   RRRR   AAAAA  PPPP   EEE                                                                  !
! G   G  R   R  A   A  P      E                                                                    !
!  GGG   R   R  A   A  P      EEEEE                                                    C. Thieulot !
!==================================================================================================!

module s40rts_tomography

integer,parameter :: MXLH=40
integer,parameter :: MXLENY=(MXLH+1)**2
integer,parameter :: MXDEP=21
integer,parameter :: MXSPL=MXDEP+3
integer,parameter :: MXMSZ=MXSPL*MXLENY
integer,parameter :: MAXP=1024
real(8),parameter ::  xi_s40rts=.15

real wk1(MXLENY),wk2(MXLENY),wk3(MXLENY)
real d0(MXLENY)
real spl(MXDEP)
real x(MXMSZ)
real dum,xd


integer nbeg,nend,ipp,ind,ind1
integer lmax,natd,ndmx,nsmx,ndmn

end module
