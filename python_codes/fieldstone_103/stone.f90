program stone
implicit none

integer, parameter :: m=8
integer nelx,nely,nelz,nnx,nny,nnz,nel,NV,ip,iel
integer counter,i,j,k,jp,nel_new,NV_new,NV_real,nel_real
real(8) Lx,Ly,Lz
integer c_e,c_n,inode,npts,imod
real(8), dimension(:), allocatable :: x,y,z
integer, dimension(:,:), allocatable :: icon, icon_real
logical, dimension(:),   allocatable :: crnode    ! nodes flagged for refinement
integer, dimension(:), allocatable :: crtype      ! type of conformal ref of element
logical crnode_el(8)
real(8), dimension(:),   allocatable :: xnew      ! node coordinates arrays
real(8), dimension(:),   allocatable :: ynew      ! node coordinates arrays
real(8), dimension(:),   allocatable :: znew  
logical, dimension(:),   allocatable :: refinable,refinable2 
integer, dimension(:,:), allocatable :: icon_new  ! connectivity array
logical, dimension(:),   allocatable :: doubble   ! array needed for renumbering
real(8), dimension(:),   allocatable :: xreal     ! node coordinates arrays
real(8), dimension(:),   allocatable :: yreal     ! node coordinates arrays
real(8), dimension(:),   allocatable :: zreal     ! node coordinates arrays
real(8) xe(8),ye(8),ze(8)
integer, dimension(:), allocatable :: compact     ! work array needed to renumber nodes
integer, dimension(:), allocatable :: pointto     ! work array needed to renumber nodes
real(8) dx,dy,dz,x0,y0,z0,zeta,xip,yip,zip,Vel,distance

!-----------------------------------------------------------

Lx=10
Ly=8
Lz=6

nelx=20
nely=16
nelz=12

nnx=nelx+1
nny=nely+1
nnz=nelz+1

NV=nnx*nny*nnz
nel=nelx*nely*nelz

!-----------------------------------------------------------

allocate(icon(m,nel))
allocate(x(NV))            
allocate(y(NV))           
allocate(z(NV))

counter=0    
do i=0,nelx    
   do j=0,nely 
      do k=0,nelz
         counter=counter+1    
         x(counter)=dble(i)*Lx/nelx
         y(counter)=dble(j)*Ly/nely
         z(counter)=dble(k)*Lz/nelz
      end do    
   end do    
end do    

counter=0    
do i=1,nelx    
   do j=1,nely    
      do k=1,nelz    
      counter=counter+1   
      icon(1,counter)=nny*nnz*(i-1)+nnz*(j-1)+k    
      icon(2,counter)=nny*nnz*(i  )+nnz*(j-1)+k    
      icon(3,counter)=nny*nnz*(i  )+nnz*(j  )+k    
      icon(4,counter)=nny*nnz*(i-1)+nnz*(j  )+k    
      icon(5,counter)=nny*nnz*(i-1)+nnz*(j-1)+k+1    
      icon(6,counter)=nny*nnz*(i  )+nnz*(j-1)+k+1    
      icon(7,counter)=nny*nnz*(i  )+nnz*(j  )+k+1    
      icon(8,counter)=nny*nnz*(i-1)+nnz*(j  )+k+1    
      end do    
   end do    
end do    

!-----------------------------------------------------------

allocate(refinable(nel))
refinable=.true.

write(*,'(a,i6)') 'ncell before   =',nel
write(*,'(a,i6)') 'np    before   =',NV
write(*,'(a,i6)') 'ncell refinable=',count(refinable)

!------------
! flag nodes 
!------------

allocate(crnode(NV)) ; crnode=.false.

do ip=1,NV
   if (z(ip)>0.66*Lz) crnode(ip)=.true.
end do

write(*,*) 'nb of flagged nodes=',count(crnode),' i.e. ',count(crnode)/dble(NV)*100,'%'


!-------------------------------------
! establish crtype of elements
!-------------------------------------
! crtype=0 no refinement
! crtype=1 refine, transition element
! crtype=2 refine, subdivide in 3x3x3

allocate(crtype(nel))

crtype=-1

do iel=1,nel

   crnode_el(1)=crnode(icon(1,iel))
   crnode_el(2)=crnode(icon(2,iel))
   crnode_el(3)=crnode(icon(3,iel))
   crnode_el(4)=crnode(icon(4,iel))
   crnode_el(5)=crnode(icon(5,iel))
   crnode_el(6)=crnode(icon(6,iel))
   crnode_el(7)=crnode(icon(7,iel))
   crnode_el(8)=crnode(icon(8,iel))

   if (count(crnode_el)==0) crtype(iel)=0

   if (count(crnode_el)==4) then
      if (crnode_el(5) .and. crnode_el(6) .and. &
          crnode_el(7) .and. crnode_el(8) ) crtype(iel)=1
   end if

   if (count(crnode_el)==8) crtype(iel)=2

end do

print *,count(crtype==0)
print *,count(crtype==1)
print *,count(crtype==2)

!---------------------------------------------------------
! generate vtu file of original mesh with refinement flag
!---------------------------------------------------------

open(unit=123,file='OUT/grid_orig.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',NV,'" NumberOfCells="',nel,'">'
!......................
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do ip=1,NV
write(123,*) x(ip),y(ip),z(ip)
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!......................
write(123,*) '<CellData Scalars="scalars">'
write(123,*) '<DataArray type="Float32" Name="crtype" Format="ascii">'
do iel=1,nel
write(123,*) crtype(iel)
end do
write(123,*) '</DataArray>'
write(123,*) '</CellData>'
!.....................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
   write(123,*) icon(1:8,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*8,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (12,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
!.....................
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

write(6,'(a)') 'produced grid_orig.vtu'


!--------------------------------
! compute new number of elements
!--------------------------------

nel_new=0
NV_new=0

do iel=1,nel
   select case(crtype(iel))
   case(0)
      nel_new=nel_new+1
      NV_new=NV_new+8
   case(1)
      nel_new=nel_new+22
      NV_new=NV_new+48
   case(2)
      nel_new=nel_new+27
      NV_new=NV_new+64
   case default
      stop 'pb in refine_mesh_3D'
   end select
end do

write(*,'(a,i6)') 'nel_new ',nel_new
write(*,'(a,i6)') 'NV_new    ',NV_new



!--------------------------

write(*,'(a)') 'carrying out element subdivision '

allocate(xnew(NV_new))              ; xnew=0
allocate(ynew(NV_new))              ; ynew=0
allocate(znew(NV_new))              ; znew=0
allocate(icon_new(m,nel_new))   ; icon_new=1
allocate(refinable2(nel_new))      ; refinable2=.false.

c_e=0 ! elements
c_n=0 ! nodes

do iel=1,nel

   dx=(x(icon(7,iel))-x(icon(1,iel)))/3
   dy=(y(icon(7,iel))-y(icon(1,iel)))/3
   dz=(z(icon(7,iel))-z(icon(1,iel)))/3

   select case(crtype(iel))

   case(0) ! unrefined element

      c_n=c_n+1 ; xnew(c_n)=x(icon(1,iel)) ; ynew(c_n)=y(icon(1,iel)) ; znew(c_n)=z(icon(1,iel)) ! sub pt 1
      c_n=c_n+1 ; xnew(c_n)=x(icon(2,iel)) ; ynew(c_n)=y(icon(2,iel)) ; znew(c_n)=z(icon(2,iel)) ! sub pt 2
      c_n=c_n+1 ; xnew(c_n)=x(icon(3,iel)) ; ynew(c_n)=y(icon(3,iel)) ; znew(c_n)=z(icon(3,iel)) ! sub pt 3
      c_n=c_n+1 ; xnew(c_n)=x(icon(4,iel)) ; ynew(c_n)=y(icon(4,iel)) ; znew(c_n)=z(icon(4,iel)) ! sub pt 4
      c_n=c_n+1 ; xnew(c_n)=x(icon(5,iel)) ; ynew(c_n)=y(icon(5,iel)) ; znew(c_n)=z(icon(5,iel)) ! sub pt 5
      c_n=c_n+1 ; xnew(c_n)=x(icon(6,iel)) ; ynew(c_n)=y(icon(6,iel)) ; znew(c_n)=z(icon(6,iel)) ! sub pt 6
      c_n=c_n+1 ; xnew(c_n)=x(icon(7,iel)) ; ynew(c_n)=y(icon(7,iel)) ; znew(c_n)=z(icon(7,iel)) ! sub pt 7
      c_n=c_n+1 ; xnew(c_n)=x(icon(8,iel)) ; ynew(c_n)=y(icon(8,iel)) ; znew(c_n)=z(icon(8,iel)) ! sub pt 8
      ! sub elt 1
      c_e=c_e+1
      icon_new(1,c_e) = c_n-7
      icon_new(2,c_e) = c_n-6
      icon_new(3,c_e) = c_n-5
      icon_new(4,c_e) = c_n-4
      icon_new(5,c_e) = c_n-3
      icon_new(6,c_e) = c_n-2
      icon_new(7,c_e) = c_n-1
      icon_new(8,c_e) = c_n

   case(1) ! transition element

      if (.not.refinable(iel)) stop 'refine_mesh_3D: pb case 1'

      x0=x(icon(1,iel))
      y0=y(icon(1,iel))
      z0=z(icon(1,iel))

      zeta=1.5

      npts=48

      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+0*dz    ! sub pt 01
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+0*dz    ! sub pt 02
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+0*dz    ! sub pt 03
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+0*dz    ! sub pt 04
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+1*dz    ! sub pt 05
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+1*dz    ! sub pt 06
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+1*dz    ! sub pt 07
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+1*dz    ! sub pt 08
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+zeta*dz ! sub pt 09
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+zeta*dz ! sub pt 10
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+zeta*dz ! sub pt 11
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+zeta*dz ! sub pt 12
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+2*dz    ! sub pt 13
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+2*dz    ! sub pt 14
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+2*dz    ! sub pt 15
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+2*dz    ! sub pt 16
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+3*dz    ! sub pt 17
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+3*dz    ! sub pt 18
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+3*dz    ! sub pt 19
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+3*dz    ! sub pt 20
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+2*dz    ! sub pt 21
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+2*dz    ! sub pt 22
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+3*dz    ! sub pt 23
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+3*dz    ! sub pt 24
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+2*dz    ! sub pt 25
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+2*dz    ! sub pt 26
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+3*dz    ! sub pt 27
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+3*dz    ! sub pt 28
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+2*dz    ! sub pt 29
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+2*dz    ! sub pt 30
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+1*dz    ! sub pt 31
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+3*dz    ! sub pt 32
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+3*dz    ! sub pt 33
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+1*dz    ! sub pt 34
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+2*dz    ! sub pt 35
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+3*dz    ! sub pt 36
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+2*dz    ! sub pt 37
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+3*dz    ! sub pt 38
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+2*dz    ! sub pt 39
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+2*dz    ! sub pt 40
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+1*dz    ! sub pt 41
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+3*dz    ! sub pt 42
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+3*dz    ! sub pt 43
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+1*dz    ! sub pt 44
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+2*dz    ! sub pt 45
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+3*dz    ! sub pt 46
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+2*dz    ! sub pt 47
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+3*dz    ! sub pt 48

      ! sub elt 1
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+1
      icon_new(2,c_e) = c_n-npts+2
      icon_new(3,c_e) = c_n-npts+3
      icon_new(4,c_e) = c_n-npts+4
      icon_new(5,c_e) = c_n-npts+5
      icon_new(6,c_e) = c_n-npts+6
      icon_new(7,c_e) = c_n-npts+7
      icon_new(8,c_e) = c_n-npts+8

      ! sub elt 2
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+5
      icon_new(2,c_e) = c_n-npts+6
      icon_new(3,c_e) = c_n-npts+7
      icon_new(4,c_e) = c_n-npts+8
      icon_new(5,c_e) = c_n-npts+9
      icon_new(6,c_e) = c_n-npts+10
      icon_new(7,c_e) = c_n-npts+11
      icon_new(8,c_e) = c_n-npts+12

      ! sub elt 3
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+9
      icon_new(2,c_e) = c_n-npts+10
      icon_new(3,c_e) = c_n-npts+11
      icon_new(4,c_e) = c_n-npts+12
      icon_new(5,c_e) = c_n-npts+13
      icon_new(6,c_e) = c_n-npts+14
      icon_new(7,c_e) = c_n-npts+15
      icon_new(8,c_e) = c_n-npts+16

      ! sub elt 4
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+13
      icon_new(2,c_e) = c_n-npts+14
      icon_new(3,c_e) = c_n-npts+15
      icon_new(4,c_e) = c_n-npts+16
      icon_new(5,c_e) = c_n-npts+17
      icon_new(6,c_e) = c_n-npts+18
      icon_new(7,c_e) = c_n-npts+19
      icon_new(8,c_e) = c_n-npts+20

      ! sub elt 5
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+10
      icon_new(2,c_e) = c_n-npts+6
      icon_new(3,c_e) = c_n-npts+7
      icon_new(4,c_e) = c_n-npts+11
      icon_new(5,c_e) = c_n-npts+14
      icon_new(6,c_e) = c_n-npts+21
      icon_new(7,c_e) = c_n-npts+22
      icon_new(8,c_e) = c_n-npts+15

      ! sub elt 6
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+14
      icon_new(2,c_e) = c_n-npts+21
      icon_new(3,c_e) = c_n-npts+22
      icon_new(4,c_e) = c_n-npts+15
      icon_new(5,c_e) = c_n-npts+18
      icon_new(6,c_e) = c_n-npts+23
      icon_new(7,c_e) = c_n-npts+24
      icon_new(8,c_e) = c_n-npts+19

      ! sub elt 7
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+5
      icon_new(2,c_e) = c_n-npts+9
      icon_new(3,c_e) = c_n-npts+12
      icon_new(4,c_e) = c_n-npts+8
      icon_new(5,c_e) = c_n-npts+25
      icon_new(6,c_e) = c_n-npts+13
      icon_new(7,c_e) = c_n-npts+16
      icon_new(8,c_e) = c_n-npts+26

      ! sub elt 8
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+25
      icon_new(2,c_e) = c_n-npts+13
      icon_new(3,c_e) = c_n-npts+16
      icon_new(4,c_e) = c_n-npts+26
      icon_new(5,c_e) = c_n-npts+27
      icon_new(6,c_e) = c_n-npts+17
      icon_new(7,c_e) = c_n-npts+20
      icon_new(8,c_e) = c_n-npts+28

      ! sub elt 9
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+11
      icon_new(2,c_e) = c_n-npts+7
      icon_new(3,c_e) = c_n-npts+3
      icon_new(4,c_e) = c_n-npts+31
      icon_new(5,c_e) = c_n-npts+15
      icon_new(6,c_e) = c_n-npts+22
      icon_new(7,c_e) = c_n-npts+29
      icon_new(8,c_e) = c_n-npts+30

      ! sub elt 10
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+15
      icon_new(2,c_e) = c_n-npts+22
      icon_new(3,c_e) = c_n-npts+29
      icon_new(4,c_e) = c_n-npts+30
      icon_new(5,c_e) = c_n-npts+19
      icon_new(6,c_e) = c_n-npts+24
      icon_new(7,c_e) = c_n-npts+32
      icon_new(8,c_e) = c_n-npts+33

      ! sub elt 11
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+12
      icon_new(2,c_e) = c_n-npts+11
      icon_new(3,c_e) = c_n-npts+31
      icon_new(4,c_e) = c_n-npts+34
      icon_new(5,c_e) = c_n-npts+16
      icon_new(6,c_e) = c_n-npts+15
      icon_new(7,c_e) = c_n-npts+30
      icon_new(8,c_e) = c_n-npts+35

      ! sub elt 12
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+16
      icon_new(2,c_e) = c_n-npts+15
      icon_new(3,c_e) = c_n-npts+30
      icon_new(4,c_e) = c_n-npts+35
      icon_new(5,c_e) = c_n-npts+20
      icon_new(6,c_e) = c_n-npts+19
      icon_new(7,c_e) = c_n-npts+33
      icon_new(8,c_e) = c_n-npts+36

      ! sub elt 13
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+8
      icon_new(2,c_e) = c_n-npts+12
      icon_new(3,c_e) = c_n-npts+34
      icon_new(4,c_e) = c_n-npts+4
      icon_new(5,c_e) = c_n-npts+26
      icon_new(6,c_e) = c_n-npts+16
      icon_new(7,c_e) = c_n-npts+35
      icon_new(8,c_e) = c_n-npts+37

      ! sub elt 14
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+26
      icon_new(2,c_e) = c_n-npts+16
      icon_new(3,c_e) = c_n-npts+35
      icon_new(4,c_e) = c_n-npts+37
      icon_new(5,c_e) = c_n-npts+28
      icon_new(6,c_e) = c_n-npts+20
      icon_new(7,c_e) = c_n-npts+36
      icon_new(8,c_e) = c_n-npts+38

      ! sub elt 15
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+8
      icon_new(2,c_e) = c_n-npts+7
      icon_new(3,c_e) = c_n-npts+3
      icon_new(4,c_e) = c_n-npts+4
      icon_new(5,c_e) = c_n-npts+12
      icon_new(6,c_e) = c_n-npts+11
      icon_new(7,c_e) = c_n-npts+31
      icon_new(8,c_e) = c_n-npts+34

      ! sub elt 16
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+41
      icon_new(2,c_e) = c_n-npts+2
      icon_new(3,c_e) = c_n-npts+6
      icon_new(4,c_e) = c_n-npts+10
      icon_new(5,c_e) = c_n-npts+40
      icon_new(6,c_e) = c_n-npts+39
      icon_new(7,c_e) = c_n-npts+21
      icon_new(8,c_e) = c_n-npts+14

      ! sub elt 17
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+40
      icon_new(2,c_e) = c_n-npts+39
      icon_new(3,c_e) = c_n-npts+21
      icon_new(4,c_e) = c_n-npts+14
      icon_new(5,c_e) = c_n-npts+42
      icon_new(6,c_e) = c_n-npts+43
      icon_new(7,c_e) = c_n-npts+23
      icon_new(8,c_e) = c_n-npts+18

      ! sub elt 18
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+44
      icon_new(2,c_e) = c_n-npts+41
      icon_new(3,c_e) = c_n-npts+10
      icon_new(4,c_e) = c_n-npts+9
      icon_new(5,c_e) = c_n-npts+45
      icon_new(6,c_e) = c_n-npts+40
      icon_new(7,c_e) = c_n-npts+14
      icon_new(8,c_e) = c_n-npts+13

      ! sub elt 19
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+45
      icon_new(2,c_e) = c_n-npts+40
      icon_new(3,c_e) = c_n-npts+14
      icon_new(4,c_e) = c_n-npts+13
      icon_new(5,c_e) = c_n-npts+46
      icon_new(6,c_e) = c_n-npts+42
      icon_new(7,c_e) = c_n-npts+18
      icon_new(8,c_e) = c_n-npts+17

      ! sub elt 20
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+1
      icon_new(2,c_e) = c_n-npts+44
      icon_new(3,c_e) = c_n-npts+9
      icon_new(4,c_e) = c_n-npts+5
      icon_new(5,c_e) = c_n-npts+47
      icon_new(6,c_e) = c_n-npts+45
      icon_new(7,c_e) = c_n-npts+13
      icon_new(8,c_e) = c_n-npts+25

      ! sub elt 21
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+47
      icon_new(2,c_e) = c_n-npts+45
      icon_new(3,c_e) = c_n-npts+13
      icon_new(4,c_e) = c_n-npts+25
      icon_new(5,c_e) = c_n-npts+48
      icon_new(6,c_e) = c_n-npts+46
      icon_new(7,c_e) = c_n-npts+17
      icon_new(8,c_e) = c_n-npts+27

      ! sub elt 22
      c_e=c_e+1
      icon_new(1,c_e) = c_n-npts+1
      icon_new(2,c_e) = c_n-npts+2
      icon_new(3,c_e) = c_n-npts+6
      icon_new(4,c_e) = c_n-npts+5
      icon_new(5,c_e) = c_n-npts+44
      icon_new(6,c_e) = c_n-npts+41
      icon_new(7,c_e) = c_n-npts+10
      icon_new(8,c_e) = c_n-npts+9

   case(2) ! fully refined element

      if (.not.refinable(iel)) stop 'refine_mesh_3D: pb case 2'

      x0=x(icon(1,iel))
      y0=y(icon(1,iel))
      z0=z(icon(1,iel))

      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+0*dz ! sub pt 01
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+0*dz ! sub pt 02
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+0*dz ! sub pt 03
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+0*dz ! sub pt 04
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+0*dz ! sub pt 05
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+0*dz ! sub pt 06
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+0*dz ! sub pt 07
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+0*dz ! sub pt 08
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+0*dz ! sub pt 09
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+0*dz ! sub pt 10
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+0*dz ! sub pt 11
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+0*dz ! sub pt 12
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+0*dz ! sub pt 13
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+0*dz ! sub pt 14
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+0*dz ! sub pt 15
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+0*dz ! sub pt 16

      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+1*dz ! sub pt 17
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+1*dz ! sub pt 18
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+1*dz ! sub pt 19
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+1*dz ! sub pt 20
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+1*dz ! sub pt 21
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+1*dz ! sub pt 22
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+1*dz ! sub pt 23
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+1*dz ! sub pt 24
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+1*dz ! sub pt 25
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+1*dz ! sub pt 26
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+1*dz ! sub pt 27
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+1*dz ! sub pt 28
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+1*dz ! sub pt 29
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+1*dz ! sub pt 30
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+1*dz ! sub pt 31
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+1*dz ! sub pt 32

      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+2*dz ! sub pt 33
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+2*dz ! sub pt 34
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+2*dz ! sub pt 35
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+2*dz ! sub pt 36
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+2*dz ! sub pt 37
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+2*dz ! sub pt 38
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+2*dz ! sub pt 39
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+2*dz ! sub pt 40
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+2*dz ! sub pt 41
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+2*dz ! sub pt 42
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+2*dz ! sub pt 43
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+2*dz ! sub pt 44
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+2*dz ! sub pt 45
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+2*dz ! sub pt 46
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+2*dz ! sub pt 47
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+2*dz ! sub pt 48

      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+3*dz ! sub pt 49
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+3*dz ! sub pt 50
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+3*dz ! sub pt 51
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+0*dy ; znew(c_n)=z0+3*dz ! sub pt 52
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+3*dz ! sub pt 53
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+3*dz ! sub pt 54
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+3*dz ! sub pt 55
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+1*dy ; znew(c_n)=z0+3*dz ! sub pt 56
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+3*dz ! sub pt 57
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+3*dz ! sub pt 58
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+3*dz ! sub pt 59
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+2*dy ; znew(c_n)=z0+3*dz ! sub pt 60
      c_n=c_n+1 ; xnew(c_n)=x0+0*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+3*dz ! sub pt 61
      c_n=c_n+1 ; xnew(c_n)=x0+1*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+3*dz ! sub pt 62
      c_n=c_n+1 ; xnew(c_n)=x0+2*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+3*dz ! sub pt 63
      c_n=c_n+1 ; xnew(c_n)=x0+3*dx ; ynew(c_n)=y0+3*dy ; znew(c_n)=z0+3*dz ! sub pt 64

      ! sub elt 1
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+1 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+2 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+6 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+5 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 2
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+2 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+3 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+7 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+6 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 3
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+3 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+4 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+8 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+7 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 4
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+5  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+6  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+10 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+9  ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 5
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+6  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+7  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+11 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+10 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 6
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+7  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+8  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+12 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+11 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 7
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+09 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+10 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+14 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+13 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 8
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+10 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+11 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+15 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+14 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 9
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+11 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+12 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+15 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 10
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+1+16 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+2+16 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+6+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+5+16 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 11
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+2+16 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+3+16 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+7+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+6+16 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 12
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+3+16 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+4+16 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+8+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+7+16 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 13
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+5+16  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+6+16  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+10+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+9+16  ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 14
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+6+16  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+7+16  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+11+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+10+16  ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 15
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+7+16  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+8+16  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+12+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+11+16 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 16
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+09+16 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+10+16 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+14+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+13+16 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 17
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+10+16 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+11+16 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+15+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+14+16 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 18
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+11+16 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+12+16 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+16+16 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+15+16 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 19
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+1+32 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+2+32 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+6+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+5+32 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 20
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+2+32 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+3+32 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+7+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+6+32 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 21
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+3+32 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+4+32 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+8+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+7+32 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 22
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+05+32  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+06+32  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+10+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+09+32  ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 23
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+06+32  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+07+32  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+11+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+10+32  ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 24
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+07+32  ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+08+32  ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+12+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+11+32 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 25
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+09+32 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+10+32 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+14+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+13+32 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 26
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+10+32 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+11+32 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+15+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+14+32 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.
      ! sub elt 27
      c_e=c_e+1
      icon_new(1,c_e) = c_n-64+11+32 ; icon_new(5,c_e)=icon_new(1,c_e)+16  
      icon_new(2,c_e) = c_n-64+12+32 ; icon_new(6,c_e)=icon_new(2,c_e)+16  
      icon_new(3,c_e) = c_n-64+16+32 ; icon_new(7,c_e)=icon_new(3,c_e)+16  
      icon_new(4,c_e) = c_n-64+15+32 ; icon_new(8,c_e)=icon_new(4,c_e)+16  
      refinable2(c_e)=.true.

   end select

end do

open(unit=123,file='OUT/gridnodes_new.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do ip=1,NV_new
   write(123,'(3f10.5)') xnew(ip),ynew(ip),znew(ip)
end do
close(123)

!print *,c_n,c_e

!---------------------------------------
! generate vtu file of raw refined mesh 
!---------------------------------------

write(*,'(a)') 'generating OUT/grid_ref_raw.vtu  ||'

open(unit=123,file='OUT/grid_ref_raw.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',NV_new,'" NumberOfCells="',nel_new,'">'
!......................
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do ip=1,NV_new
write(123,*) xnew(ip),ynew(ip),znew(ip)
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!......................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel_new
   write(123,*) icon_new(1:8,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*8,iel=1,nel_new)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (12,iel=1,nel_new)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
!......................
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

!--------------------------------
! compute real number of points
!--------------------------------

write(*,'(a)') 'computing real number of nodes      ||'

allocate(doubble(NV_new)) ; doubble=.false.
allocate(pointto(NV_new))

do ip=1,NV_new
   pointto(ip)=ip
end do


distance=1.d-4*min(Lx,Ly,Lz)

imod=NV_new/10

counter=0
do ip=2,NV_new
   if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(NV_new)*100.,'%'
   xip=xnew(ip)
   yip=ynew(ip)
   zip=znew(ip)
   do jp=1,ip-1
      if (abs(xip-xnew(jp))<distance .and. &
          abs(yip-ynew(jp))<distance .and. &
          abs(zip-znew(jp))<distance ) then 
          doubble(ip)=.true.
          pointto(ip)=jp
          exit 
      end if
   end do
end do

write(*,'(a)') ' done   ||' 

NV_real=NV_new-count(doubble)

nel_real=nel_new

write(*,'(a,i6)') 'nel_real=',nel_real
write(*,'(a,i6)') 'NV_real=',NV_real

!print *,'pointto :',minval(pointto),maxval(pointto)

!----------------------------------------
! compact data, remove superfluous nodes
!----------------------------------------

allocate(xreal(NV_real))
allocate(yreal(NV_real))
allocate(zreal(NV_real))
allocate(icon_real(m,nel_real))

counter=0
do ip=1,NV_new
   if (.not.doubble(ip)) then
      counter=counter+1
      xreal(counter)=xnew(ip)
      yreal(counter)=ynew(ip)
      zreal(counter)=znew(ip)
   end if
end do

icon_real=icon_new

do iel=1,nel_real
   do i=1,m
      icon_real(i,iel)=pointto(icon_real(i,iel))
   end do
end do

!print *,'bef compaction:',minval(icon_real),maxval(icon_real)

allocate(compact(NV_new))

counter=0
do ip=1,NV_new
   if (.not.doubble(ip)) then
      counter=counter+1
      compact(ip)=counter
   end if
end do

!print *,'compact :',minval(compact),maxval(compact)

do iel=1,nel_real
   do i=1,m
      icon_real(i,iel)=compact(icon_real(i,iel))
   end do
end do

!print *,'aft compaction:',minval(icon_real),maxval(icon_real)

!----------------------------

open(unit=123,file='OUT/grid_ref_compacted.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',NV_real,'" NumberOfCells="',nel_real,'">'
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do ip=1,NV_real
write(123,'(3es20.7)') xreal(ip),yreal(ip),zreal(ip)
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
write(123,*) '<CellData Scalars="scalars">'
write(123,*) '<DataArray type="Float32" Name="elt. volume" Format="ascii">'
do iel=1,nel_real
   do k=1,8             
   inode=icon_real(k,iel) 
   xe(k)=xreal(inode)  
   ye(k)=yreal(inode)
   ze(k)=zreal(inode) 
   end do
   Vel  =  1!hexahedron_volume (xe,ye,ze) 
   write(123,'(es13.5)') Vel
end do
write(123,*) '</DataArray>'
write(123,*) '</CellData>'

!......................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel_real
   write(123,*) icon_real(1:8,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*8,iel=1,nel_real)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (12,iel=1,nel_real)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
!......................
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

!----------------------------

deallocate(x)
deallocate(y)
deallocate(z)
deallocate(icon)
deallocate(refinable)
deallocate(refinable2)

NV=NV_real
nel=nel_real

allocate(icon(m,nel)) 
allocate(x(NV))         
allocate(y(NV))         
allocate(z(NV))         

x=xreal
y=yreal
z=zreal
icon=icon_real

!-----------------
! release memory
!-----------------

deallocate(crnode)
deallocate(crtype)
deallocate(xnew)
deallocate(ynew)
deallocate(znew)
deallocate(icon_new)
deallocate(doubble)
deallocate(pointto)
deallocate(xreal)
deallocate(yreal)
deallocate(zreal)
deallocate(icon_real)
deallocate(compact)



end program
