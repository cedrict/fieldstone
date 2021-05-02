program opla
implicit none
!integer, parameter :: nz=15
!integer, parameter :: N=6
integer n,nz
integer, parameter :: iha=6!10

integer, parameter :: nn=45!200!3*nz
integer, parameter :: nn1=45!100!3*nz

double precision, dimension(nn) :: a
double precision, dimension(iha) :: b
integer, dimension(nn) :: snr
integer, dimension(nn1) :: rnr
double precision, dimension(iha) :: pivot
double precision, dimension(8) :: aflag
integer, dimension(10) :: iflag
integer ifail

integer, dimension(iha,11) :: ha

!--------------------------------

a=0
b=0
aflag=0
iflag=0

n=6
nz=15

!-------------------------------

b(1:6)=(/10,11,45,33,-22,31/)

rnr(01)=1  ; snr(01)=1  ; a(01)=10
rnr(02)=6  ; snr(02)=6  ; a(02)=6
rnr(03)=6  ; snr(03)=2  ; a(03)=-2
rnr(04)=6  ; snr(04)=1  ; a(04)=-1
rnr(05)=2  ; snr(05)=2  ; a(05)=12
rnr(06)=2  ; snr(06)=3  ; a(06)=-3
rnr(07)=2  ; snr(07)=4  ; a(07)=-1
rnr(08)=4  ; snr(08)=1  ; a(08)=-2
rnr(09)=5  ; snr(09)=1  ; a(09)=-1
rnr(10)=5  ; snr(10)=6  ; a(10)=-1
rnr(11)=5  ; snr(11)=5  ; a(11)=1
rnr(12)=5  ; snr(12)=4  ; a(12)=-5
rnr(13)=4  ; snr(13)=4  ; a(13)=10
rnr(14)=4  ; snr(14)=5  ; a(14)=-1
rnr(15)=3  ; snr(15)=3  ; a(15)=15

call y12maf(N,nz,a,snr,nn,rnr,nn1,pivot,ha,iha,aflag,iflag,b,ifail)

!write(*,*) ifail
!write(*,*) iflag

write(*,*) 'solution:',b(1:n)

write(6,*) 'the largest element in the original matrix is ',aflag(6)
write(6,*) 'the largest element found in the elimination  ',aflag(7)
write(6,*) 'the minimal(in absolute value)pivotal element ',aflag(8)
write(6,*) 'the growth factor is                          ',aflag(5)
write(6,*) 'the number of collections in the row list     ',iflag(6)
write(6,*) 'the number of collections in the column list  ',iflag(7)
write(6,*) 'the largest number of elements found in matrix',iflag(8)
end 


