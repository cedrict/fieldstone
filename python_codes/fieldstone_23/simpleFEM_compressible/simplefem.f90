!==============================================!
!                                              !
! C. thieulot ; October 2017                   !
!                                              !
!==============================================!
                                               !
program fcubed                                 !
                                               !
implicit none                                  !
                                               !
integer, parameter :: m=4                      ! number of nodes which constitute an element
integer, parameter :: ndof=2                   ! number of dofs per node
integer nnx                                    ! number of grid points in the x direction
integer nny                                    ! number of grid points in the y direction
integer np                                     ! number of grid points
integer nelx                                   ! number of elements in the x direction
integer nely                                   ! number of elements in the y direction
integer nel                                    ! number of elements
integer Nfem                                   ! size of the FEM matrix 
integer, dimension(:,:),allocatable :: icon    ! connectivity array
integer approach                               ! hel calculation method
integer niter                                  ! number of iterations 
integer i1,i2,i,j,k,iel,counter,iq,jq,iter     !
integer ik,jk,ikk,jkk,m1,m2,k1,k2              !
integer ibench,solver                          !
                                               !  
real(8) Lx,Ly                                  ! size of the numerical domain
real(8) viscosity                              ! dynamic viscosity $\mu$ of the material
real(8), dimension(:),   allocatable :: x,y    ! node coordinates arrays
real(8), dimension(:),   allocatable :: u,v    ! node velocity arrays
real(8), dimension(:),   allocatable :: Psol   ! pressure vector
real(8), dimension(:),   allocatable :: Vsol   ! velocity vector
real(8), dimension(:),   allocatable :: Rv     ! velocity residual
real(8), dimension(:),   allocatable :: Rp     ! velocity residual
real(8), dimension(:),   allocatable :: Psolmem! pressure vector
real(8), dimension(:),   allocatable :: Vsolmem! velocity vector
real(8), dimension(:),   allocatable :: rhs_f  ! right hand side
real(8), dimension(:),   allocatable :: rhs_h  ! right hand side
real(8), dimension(:,:), allocatable :: KKK,KKKmem    ! FEM matrix
real(8), dimension(:),   allocatable :: bc_val ! array containing bc values
real(8), dimension(:,:), allocatable :: G      ! gradient operator matrix
real(8), dimension(:),   allocatable :: density_nodal !density for each node
real(8), dimension(:),   allocatable :: dudx_elemental
real(8), dimension(:),   allocatable :: dudy_elemental
real(8), dimension(:),   allocatable :: dvdx_elemental
real(8), dimension(:),   allocatable :: dvdy_elemental
real(8), dimension(:),   allocatable :: dudx_nodal
real(8), dimension(:),   allocatable :: dudy_nodal
real(8), dimension(:),   allocatable :: dvdx_nodal
real(8), dimension(:),   allocatable :: dvdy_nodal
real(8), dimension(:),   allocatable :: phi_elemental
real(8), dimension(:),   allocatable :: phi_nodal
                                               !
real(8) rq,sq,wq                               ! local coordinate and weight of qpoint
real(8) xq,yq                                  ! global coordinate of qpoint
real(8) uq,vq                                  ! velocity at qpoint
real(8) exxq,eyyq,exyq                         ! strain-rate components at qpoint  
real(8) Kel(m*ndof,m*ndof)                     ! elemental FEM matrix
real(8) fel(m*ndof)                            ! elemental right hand side Vel
real(8) hel(1)                                 ! elemental right hand side Press
real(8) Gel(m*ndof,1)                          !
real(8) N(m),dNdx(m),dNdy(m),dNdr(m),dNds(m)   ! shape fcts and derivatives
real(8) jcob                                   ! determinant of jacobian matrix
real(8) jcb(2,2)                               ! jacobian matrix
real(8) jcbi(2,2)                              ! inverse of jacobian matrix
real(8) Bmat(3,ndof*m)                         ! B matrix
real(8), dimension(3,3) :: Cmat                ! C matrix
real(8) Aref                                   !
real(8) eps                                    !
real(8) offsetx,offsety                        ! coordinates of lower left corner
real(8), external :: uth,vth,pth               ! body force and analytical solution
real(8), external :: rho,drhodx,drhody,gx,gy   !
real(8), external :: dudxth,dudyth,dvdxth,dvdyth,phith
real(8) V_diff,P_diff                          !
real(8) tol                                    !
real(8) L2_err_u,L2_err_v,L2_err_p,L2_err_vel  !
real(8) L1_err_u,L1_err_v,L1_err_p             !
real(8) rhoq,drhodyq,drhodxq                   !
real(8) hx,hy                                  !
real(8) xc,yc                                  !
real(8) vrms                                   !
real(8) phi_total                              !
real(8) dudx_L1,dudx_L2,dvdx_L1,dvdx_L2,dudy_L1!
real(8) dudy_L2,dvdy_L1,dvdy_L2,phi_L1,phi_L2  !
                                               !
logical, dimension(:), allocatable :: bc_fix   ! prescribed b.c. array
                                               !
!==============================================!
!=====[setup]==================================!
!==============================================!

Lx=1.d0
Ly=1.d0

!==============================================!
!======[node number loop]======================!
!==============================================!

open(unit=888,file='OUT/discretisation_errors.dat',status='replace')
open(unit=999,file='OUT/discretisation_errors_derivatives.dat',status='replace')

do nnx= 17,17!8,48,8 
nny=nnx

write(*,'(a,i4,a,i4)') 'resolution:',nnx,' X',nny 

np=nnx*nny

nelx=nnx-1
nely=nny-1
nel=nelx*nely

hx = Lx/nnx
hy = Ly/nny
viscosity=1.d0

Nfem=np*ndof

eps=1.d-10

Cmat(1,1)=4.d0/3.d0  ; Cmat(1,2)=-2.d0/3.d0 ; Cmat(1,3)=0.d0  
Cmat(2,1)=-2.d0/3.d0 ; Cmat(2,2)=4.d0/3.d0  ; Cmat(2,3)=0.d0  
Cmat(3,1)=0.d0       ; Cmat(3,2)=0.d0       ; Cmat(3,3)=1.d0  

! Available benchmarks 
! 1 - 2D Cartesian Linear
! 2 - 2D Cartesian Sinusoidal
! 3 - 1D Cartesian Linear
! 4 - Arie van den Berg
! 5 - 1D Cartesian Sinusoidal

ibench=1
select case(ibench)
case(-1)
offsetx=0.d0
offsety=0.d0
case(1)
offsetx=1.d0
offsety=1.d0
case(2)
offsetx=0.d0
offsety=0.d0
case(3)
offsetx=1.d0
offsety=1.d0
case(4)
offsetx=20.d0
offsety=20.d0
!because a logarithm is much closer to linear away from zero
case(5)
offsetx=0.d0
offsety=0.d0
end select

niter=1

tol=1.d-8

solver=2

approach=2

if (ibench<0) approach=1 ! kill hel :)
if (ibench<0) niter=1 ! kill hel :)

!==============================================!
!===[allocate memory]==========================!
!==============================================!

allocate(x(np))
allocate(y(np))
allocate(u(np))
allocate(v(np))
allocate(icon(m,nel))
allocate(KKK(Nfem,Nfem))
allocate(KKKmem(Nfem,Nfem))
allocate(rhs_f(Nfem))
allocate(Vsol(Nfem))
allocate(Vsolmem(Nfem))
allocate(bc_fix(Nfem))
allocate(bc_val(Nfem))
allocate(Psol(nel))
allocate(Psolmem(nel))
allocate(rhs_h(nel))
allocate(G(Nfem,nel))
allocate(Rv(Nfem))
allocate(Rp(nel))
allocate(density_nodal(np))
allocate(dudx_elemental(nel))
allocate(dvdx_elemental(nel))
allocate(dudy_elemental(nel))
allocate(dvdy_elemental(nel))
allocate(dudx_nodal(np))
allocate(dvdx_nodal(np))
allocate(dudy_nodal(np))
allocate(dvdy_nodal(np))
allocate(phi_elemental(nel))
allocate(phi_nodal(np))

Vsolmem=0
Psolmem=0
Vsol=0
Psol=0
rhs_f=0
x=0
y=0
u=0
v=0
KKK=0
KKKmem=0
bc_fix=.false.
bc_val=0
rhs_h=0
G=0
Rv=0
Rp=0

!==============================================!
!===[grid points setup]========================!
!==============================================!

counter=0
do j=0,nely
   do i=0,nelx
      counter=counter+1
      x(counter)=dble(i)*Lx/dble(nelx) + offsetx
      y(counter)=dble(j)*Ly/dble(nely) + offsety
   end do
end do

open(unit=123,file='OUT/gridnodes.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,np
   write(123,'(2f10.5,i8)') x(i),y(i),i
end do
close(123)

!==============================================!
!===[connectivity]=============================!
!==============================================!

counter=0
do j=1,nely
   do i=1,nelx
      counter=counter+1
      icon(1,counter)=i+(j-1)*(nelx+1)
      icon(2,counter)=i+1+(j-1)*(nelx+1)
      icon(3,counter)=i+1+j*(nelx+1)
      icon(4,counter)=i+j*(nelx+1)
   end do
end do

open(unit=123,file='OUT/icon.dat',status='replace')
do iel=1,nel
   write(123,'(a)') '----------------------------'
   write(123,'(a,i4,a)') '---element #',iel,' -----------'
   write(123,'(a)') '----------------------------'
   write(123,'(a,i8,a,2f20.10)') '  node 1 ', icon(1,iel),' at pos. ',x(icon(1,iel)),y(icon(1,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 2 ', icon(2,iel),' at pos. ',x(icon(2,iel)),y(icon(2,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 3 ', icon(3,iel),' at pos. ',x(icon(3,iel)),y(icon(3,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 4 ', icon(4,iel),' at pos. ',x(icon(4,iel)),y(icon(4,iel))
end do
close(123)

!==============================================!
!=====[define bc]==============================!
!==============================================!

bc_fix=.false.

do i=1,np
   if (x(i)-offsetx.lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=uth(x(i),y(i),ibench)
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=vth(x(i),y(i),ibench)
   endif
   if (x(i)-offsetx.gt.(Lx-eps)) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=uth(x(i),y(i),ibench)
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=vth(x(i),y(i),ibench)
   endif
   if (y(i)-offsety.lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=uth(x(i),y(i),ibench)
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=vth(x(i),y(i),ibench)
   endif
   if (y(i)-offsety.gt.(Ly-eps) ) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=uth(x(i),y(i),ibench)
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=vth(x(i),y(i),ibench)
   endif
end do

open(unit=123,file='OUT/bc_u.dat',status='replace')
open(unit=234,file='OUT/bc_v.dat',status='replace')
do i=1,np
   if (bc_fix((i-1)*ndof+1)) write(123,'(3f20.10)') x(i),y(i),bc_val((i-1)*ndof+1) 
   if (bc_fix((i-1)*ndof+2)) write(234,'(3f20.10)') x(i),y(i),bc_val((i-1)*ndof+2) 
end do
close(123)
close(234)

!==============================================!
!=========Setup density========================!
!==============================================!

do i=1,np
   density_nodal(i) = rho(x(i),y(i),ibench)
end do

!*********************************************************!

open(unit=666,file='OUT/convergence.dat',status='replace')

do iter=1,niter ! iterations 

!==============================================!
!=====[build FE matrix]========================!
!==============================================!

KKK=0.d0
G=0.d0
rhs_f=0.d0
rhs_h=0.d0

do iel=1,nel

   Kel=0.d0
   fel=0.d0
   Gel=0.d0
   hel=0.d0

   do iq=-1,1,2
   do jq=-1,1,2

      rq=iq/sqrt(3.d0)
      sq=jq/sqrt(3.d0)
      wq=1.d0*1.d0

      N(1)=0.25d0*(1.d0-rq)*(1.d0-sq)
      N(2)=0.25d0*(1.d0+rq)*(1.d0-sq)
      N(3)=0.25d0*(1.d0+rq)*(1.d0+sq)
      N(4)=0.25d0*(1.d0-rq)*(1.d0+sq)

      dNdr(1)= - 0.25d0*(1.d0-sq)   ;   dNds(1)= - 0.25d0*(1.d0-rq)
      dNdr(2)= + 0.25d0*(1.d0-sq)   ;   dNds(2)= - 0.25d0*(1.d0+rq)
      dNdr(3)= + 0.25d0*(1.d0+sq)   ;   dNds(3)= + 0.25d0*(1.d0+rq)
      dNdr(4)= - 0.25d0*(1.d0+sq)   ;   dNds(4)= + 0.25d0*(1.d0-rq)

      jcb=0.d0
      do k=1,m
         jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
         jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
         jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
         jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      enddo

      jcob=jcb(1,1)*jcb(2,2)-jcb(2,1)*jcb(1,2)

      jcbi(1,1)=    jcb(2,2) /jcob
      jcbi(1,2)=  - jcb(1,2) /jcob
      jcbi(2,1)=  - jcb(2,1) /jcob
      jcbi(2,2)=    jcb(1,1) /jcob

      xq=0.d0
      yq=0.d0
      uq=0.d0
      vq=0.d0
      exxq=0.d0
      eyyq=0.d0
      exyq=0.d0

      drhodxq=0.d0
      drhodyq=0.d0
      rhoq=0.d0
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
         rhoq=rhoq+N(k)*density_nodal(icon(k,iel))
         dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
         dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
         exxq=exxq+ dNdx(k)*u(icon(k,iel))
         eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
         exyq=exyq+ dNdx(k)*v(icon(k,iel)) *0.5d0 &
                  + dNdy(k)*u(icon(k,iel)) *0.5d0
         drhodxq=drhodxq + dNdx(k)*density_nodal(icon(k,iel))
         drhodyq=drhodyq + dNdy(k)*density_nodal(icon(k,iel))
      end do

      do i=1,m
         i1=2*i-1
         i2=2*i
         Bmat(1,i1)=dNdx(i)   ; Bmat(1,i2)=0.d0
         Bmat(2,i1)=0.d0      ; Bmat(2,i2)=dNdy(i)
         Bmat(3,i1)=dNdy(i)   ; Bmat(3,i2)=dNdx(i)        
      end do

      Kel=Kel + matmul(transpose(Bmat),matmul(viscosity*Cmat,Bmat))*wq*jcob

      do i=1,m
         i1=2*i-1
         i2=2*i
         fel(i1)=fel(i1)+N(i)*jcob*wq*rho(xq,yq,ibench)*gx(xq,yq,ibench)
         fel(i2)=fel(i2)+N(i)*jcob*wq*rho(xq,yq,ibench)*gy(xq,yq,ibench)
         Gel(i1,1)=Gel(i1,1)-dNdx(i)*jcob*wq
         Gel(i2,1)=Gel(i2,1)-dNdy(i)*jcob*wq
      end do

      select case(approach)
      case(1)
         hel(1)=hel(1)+1.d0/rho(xq,yq,ibench)*(uq*drhodx(xq,yq,ibench) + vq*drhody(xq,yq,ibench) )*wq*jcob ! approach 1
      case(2)
         hel(1)=hel(1)+1.d0/rhoq*(uq*drhodxq+vq*drhodyq)*wq*jcob !approach 2
      end select 

   end do
   end do

   !=====================
   !=====[assemble]======
   !=====================

   do k1=1,m
      ik=icon(k1,iel)
      do i1=1,ndof
         ikk=ndof*(k1-1)+i1
         m1=ndof*(ik-1)+i1
         do k2=1,m
            jk=icon(k2,iel)
            do i2=1,ndof
               jkk=ndof*(k2-1)+i2
               m2=ndof*(jk-1)+i2
               KKK(m1,m2)=KKK(m1,m2)+Kel(ikk,jkk)
            end do
         end do
         rhs_f(m1)=rhs_f(m1)+fel(ikk)
         G(m1,iel)=G(m1,iel)+Gel(ikk,1)
      end do
   end do
   rhs_h(iel)=hel(1)

end do


!==============================================!
!=====[impose b.c.]============================!
!==============================================!

do i=1,Nfem
    if (bc_fix(i)) then 
      Aref=KKK(i,i)
      do j=1,Nfem
         rhs_f(j)=rhs_f(j)-KKK(i,j)*bc_val(i)
         KKK(i,j)=0.d0
         KKK(j,i)=0.d0
      enddo
      KKK(i,i)=Aref
      rhs_f(i)=Aref*bc_val(i)
      do j=1,nel
         rhs_h(j)=rhs_h(j)-G(i,j)*bc_val(i)
         G(i,j)=0.d0
      end do
   endif
enddo

do i=1,Nfem
do j=1,Nfem
!   if (abs(KKK(i,j))/=0.d0) &
   write(333,*) i-1,j-1,KKK(i,j)
end do
end do
do i=1,Nfem
do j=1,nel
   if (abs(G(i,j))/=0.d0) &
   write(444,*) i-1,j-1,G(i,j)
end do
end do

do i=1,Nfem
   write(555,*) i-1,rhs_f(i)
end do

do i=1,nel
   write(66,*) i-1,rhs_h(i)
end do


print *,minval(KKK),maxval(KKK)
print *,minval(G),maxval(G)
print *,minval(rhs_f),maxval(rhs_f)
print *,minval(rhs_h),maxval(rhs_h)

!==============================================!
!=====[solve system]===========================!
!==============================================!

KKKmem=KKK

!write(*,*) iter, maxval(abs(KKKmem)),maxval(abs(Vsol)),maxval(abs(G)),maxval(abs(Psol)),maxval(abs(rhs_f))


select case(solver)
case(1)
call solve_uzawa1(KKK,G,Nfem,nel,rhs_f,rhs_h,Vsol,Psol)
case(2)
call solve_uzawa2(KKK,G,Nfem,nel,rhs_f,rhs_h,Vsol,Psol)
case(3)
call solve_uzawa3(KKK,G,Nfem,nel,rhs_f,rhs_h,Vsol,Psol)
case(4)
!call solve_full(KKK,G,Nfem,nel,rhs_f,rhs_h,Vsol,Psol)
end select

do i=1,np
   u(i)=Vsol((i-1)*ndof+1)
   v(i)=Vsol((i-1)*ndof+2)
   write(678,*) x(i),y(i),u(i),v(i)
end do

print *,minval(Psol),maxval(Psol)

! normalise pressure


!if (ibench<0) then
   Psol=Psol-sum(Psol)/nel
!end if 

! compute residuals

Rv=matmul(KKKmem,Vsol)+matmul(G,Psol)-rhs_f 
Rp=matmul(transpose(G),Vsol)-rhs_h          

!==============================================!
!=====[iterations converged?]==================!
!==============================================!

V_diff=maxval(abs(Vsol-Vsolmem))/maxval(abs(Vsol))
P_diff=maxval(abs(Psol-Psolmem))/maxval(abs(Psol))

Vsolmem=Vsol
Psolmem=Psol

write(666,*) iter,V_diff,P_diff

write(*,'(a,i3,a,3es15.6,a,2es15.6,a,2es15.6)') &
'it. ',iter,&
' | <|u|>,<|v|>,<|p|> ',sum(abs(u))/np,sum(abs(v))/np,sum(abs(Psol))/nel,&
' | V_diff, P_diff ',V_diff,P_diff,'| max|Rv|, max|Rp| ',maxval(abs(Rv))/maxval(abs(rhs_f)),maxval(abs(Rp))/maxval(abs(rhs_h))

if (V_diff<tol .and. P_diff<tol) then
   print *,'-> iterations have converged'
   exit
end if

end do ! iterations

!*********************************************************!

close(666)

!==============================================!
!=======[Compute vrms and strainrates]=========!
!==============================================!

dudx_elemental=0.d0
dvdx_elemental=0.d0
dudy_elemental=0.d0
dvdy_elemental=0.d0

iel=0
do j=1,nely
do i=1,nelx
   iel=iel+1


   do iq=-1,1,2
   do jq=-1,1,2

      rq=iq/sqrt(3.d0)
      sq=jq/sqrt(3.d0)
      wq=1.d0*1.d0

      N(1)=0.25d0*(1.d0-rq)*(1.d0-sq)
      N(2)=0.25d0*(1.d0+rq)*(1.d0-sq)
      N(3)=0.25d0*(1.d0+rq)*(1.d0+sq)
      N(4)=0.25d0*(1.d0-rq)*(1.d0+sq)

      dNdr(1)= - 0.25d0*(1.d0-sq)   ;   dNds(1)= - 0.25d0*(1.d0-rq)
      dNdr(2)= + 0.25d0*(1.d0-sq)   ;   dNds(2)= - 0.25d0*(1.d0+rq)
      dNdr(3)= + 0.25d0*(1.d0+sq)   ;   dNds(3)= + 0.25d0*(1.d0+rq)
      dNdr(4)= - 0.25d0*(1.d0+sq)   ;   dNds(4)= + 0.25d0*(1.d0-rq)

      jcb=0.d0
      do k=1,m
         jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
         jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
         jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
         jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      enddo

      jcob=jcb(1,1)*jcb(2,2)-jcb(2,1)*jcb(1,2)
      uq=0.d0
      vq=0.d0
      do k=1,m
         dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
         dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
         dudx_elemental(iel) = dudx_elemental(iel) + dNdx(k)*u(icon(k,iel))*wq*jcob
         dvdx_elemental(iel) = dvdx_elemental(iel) + dNdx(k)*v(icon(k,iel))*wq*jcob
         dudy_elemental(iel) = dudy_elemental(iel) + dNdy(k)*u(icon(k,iel))*wq*jcob
         dvdy_elemental(iel) = dvdy_elemental(iel) + dNdy(k)*v(icon(k,iel))*wq*jcob
      end do
      vrms=vrms+(uq**2+vq**2)*jcob*wq
   end do
   end do

end do
end do

dudx_elemental=dudx_elemental/hx/hy
dvdx_elemental=dvdx_elemental/hx/hy
dudy_elemental=dudy_elemental/hx/hy
dvdy_elemental=dvdy_elemental/hx/hy

iel=0
do j=1,nely
do i=1,nelx
   iel=iel+1
   write(1070,*) iel,dudx_elemental(iel)
   write(1071,*) iel,dvdx_elemental(iel)
   write(1072,*) iel,dudy_elemental(iel)
   write(1073,*) iel,dvdy_elemental(iel)
end do 
end do 

call elemental_to_nodal(dudx_elemental,dudx_nodal,icon,nel,np)
call elemental_to_nodal(dvdx_elemental,dvdx_nodal,icon,nel,np)
call elemental_to_nodal(dudy_elemental,dudy_nodal,icon,nel,np)
call elemental_to_nodal(dvdy_elemental,dvdy_nodal,icon,nel,np)

vrms=sqrt(vrms/Lx/Ly)
write(*,*) "Vrms = ", vrms

! compute nodal shear heating

phi_nodal=0.d0
do i=1,np
   phi_nodal(i) =                4.d0/3.d0*viscosity*dudx_nodal(i)**2 - 2.d0/3.d0*dudx_nodal(i)*dvdy_nodal(i)
   phi_nodal(i) = phi_nodal(i) + 4.d0/3.d0*viscosity*dvdy_nodal(i)**2 - 2.d0/3.d0*dudx_nodal(i)*dvdy_nodal(i)
   phi_nodal(i) = phi_nodal(i) + viscosity*(dudy_nodal(i) + dvdx_nodal(i))**2
   ! phi_nodal(i) = 4.d0/3.d0*viscosity*dudx_nodal(i)**2
end do

! compute elemental shear heating

phi_total=0.d0
do iel=1,nel

   do iq=-1,1,2
   do jq=-1,1,2

      rq=iq/sqrt(3.d0)
      sq=jq/sqrt(3.d0)
      wq=1.d0*1.d0

      N(1)=0.25d0*(1.d0-rq)*(1.d0-sq)
      N(2)=0.25d0*(1.d0+rq)*(1.d0-sq)
      N(3)=0.25d0*(1.d0+rq)*(1.d0+sq)
      N(4)=0.25d0*(1.d0-rq)*(1.d0+sq)

      dNdr(1)= - 0.25d0*(1.d0-sq)   ;   dNds(1)= - 0.25d0*(1.d0-rq)
      dNdr(2)= + 0.25d0*(1.d0-sq)   ;   dNds(2)= - 0.25d0*(1.d0+rq)
      dNdr(3)= + 0.25d0*(1.d0+sq)   ;   dNds(3)= + 0.25d0*(1.d0+rq)
      dNdr(4)= - 0.25d0*(1.d0+sq)   ;   dNds(4)= + 0.25d0*(1.d0-rq)

      jcb=0.d0
      do k=1,m
         jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
         jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
         jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
         jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      enddo

      jcob=jcb(1,1)*jcb(2,2)-jcb(2,1)*jcb(1,2)

      jcbi(1,1)=    jcb(2,2) /jcob
      jcbi(1,2)=  - jcb(1,2) /jcob
      jcbi(2,1)=  - jcb(2,1) /jcob
      jcbi(2,2)=    jcb(1,1) /jcob

      do k=1,m
         dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
         dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
      end do

      ! v_el=0.d0
      ! T_el=0.d0
      do k=1,m

         ! phi = phi + viscosity*(dudx_nodal(icon(k,iel))**2 + dudx_nodal(icon(k,iel))*(dudx_nodal(icon(k,iel)) + dvdy_nodal(icon(k,iel))))!i=x,j=x
         ! phi = phi + viscosity*(dvdy_nodal(icon(k,iel))**2 + dvdy_nodal(icon(k,iel))*(dudx_nodal(icon(k,iel)) + dvdy_nodal(icon(k,iel))))!i=y,j=y
         ! phi = phi + viscosity*0.5*(dvdx_nodal(icon(k,iel))**2 + dvdx_nodal(icon(k,iel))*dudy_nodal(icon(k,iel)))
         ! phi = phi + viscosity*0.5*(dudy_nodal(icon(k,iel))**2 + dvdx_nodal(icon(k,iel))*dudy_nodal(icon(k,iel)))

         ! phi_elemental(iel) = phi_elemental(iel) + viscosity*(2.d0*dudx_nodal(icon(k,iel)))**2
         ! phi_elemental(iel) = phi_elemental(iel) + viscosity*2.d0*(dvdx_nodal(icon(k,iel)) + dudy_nodal(icon(k,iel)))**2
         ! phi_elemental(iel) = phi_elemental(iel) + viscosity*(2.d0*dvdy_nodal(icon(k,iel)))**2
         ! phi_elemental(iel) = phi_elemental(iel) - viscosity*4.d0/3.d0*(dudx_nodal(icon(k,iel))+dvdy_nodal(icon(k,iel)))**2

         ! phi_elemental(iel) = phi_elemental(iel) + viscosity*4.d0/3.d0*dNdx(k)*u(icon(k,iel))*dNdx(k)*u(icon(k,iel))
         ! phi_elemental(iel) = phi_elemental(iel) + viscosity*4.d0/3.d0*dNdy(k)*v(icon(k,iel))*dNdy(k)*v(icon(k,iel))
         ! phi_elemental(iel) = phi_elemental(iel) - viscosity*(dNdx(k)*v(icon(k,iel)) + dNdy(k)*u(icon(k,iel)))**2
         ! phi_elemental(iel) = phi_elemental(iel) - viscosity*4.d0/3.d0*dNdx(k)*u(icon(k,iel))*dNdy(k)*v(icon(k,iel))

          !phi_elemental(iel) = phi_elemental(iel)*N(k)*wq*jcob

         ! v_el = v_el + v(icon(k,iel))*N(k)
         ! T_el = t_el + T(icon(k,iel))*N(k)

         phi_elemental(iel) = phi_elemental(iel) + N(k)*phi_nodal(icon(k,iel))*wq*jcob/(hx*hy)
      end do
   end do
   end do
   !phi_elemental(iel) = 0.25*(phi_nodal(icon(1,iel)) + phi_nodal(icon(2,iel)) + phi_nodal(icon(3,iel)) + phi_nodal(icon(4,iel)))
end do
   
phi_total = sum(phi_elemental)*hx*hy

write(*,*) "phi_total = ", phi_total

!==============================================!
!=====[output solution to file]================!
!==============================================!

open(unit=123,file='OUT/solution_u.dat',status='replace')
open(unit=234,file='OUT/solution_v.dat',status='replace')
open(unit=345,file='OUT/solution_dudx.dat',status='replace')
open(unit=456,file='OUT/solution_dvdy.dat',status='replace')
open(unit=012,file='OUT/solution_phi_nodal.dat',status='replace')
open(unit=987,file='OUT/solution_dudy.dat',status='replace')
open(unit=988,file='OUT/solution_dvdx.dat',status='replace')
do i=1,np
   write(123,'(5f20.10)') x(i), y(i), u(i),          uth(x(i),y(i),ibench),    u(i)-uth(x(i),y(i),ibench)
   write(234,'(5f20.10)') x(i), y(i), v(i),          vth(x(i),y(i),ibench),    v(i)-vth(x(i),y(i),ibench)
   write(345,'(5f20.10)') x(i), y(i), dudx_nodal(i), dudxth(x(i),y(i),ibench), dudx_nodal(i)-dudxth(x(i),y(i),ibench)
   write(456,'(5f20.10)') x(i), y(i), dvdy_nodal(i), dvdyth(x(i),y(i),ibench), dvdy_nodal(i)-dvdyth(x(i),y(i),ibench)
   write(987,'(5f20.10)') x(i), y(i), dudy_nodal(i), dudyth(x(i),y(i),ibench), dudy_nodal(i)-dudyth(x(i),y(i),ibench)
   write(988,'(5f20.10)') x(i), y(i), dvdx_nodal(i), dvdxth(x(i),y(i),ibench), dvdx_nodal(i)-dvdxth(x(i),y(i),ibench)
   write(012,'(5f20.10)') x(i), y(i), phi_nodal(i),  phith(x(i),y(i),ibench),  phi_nodal(i)-phith(x(i),y(i),ibench)
end do
close(123)
close(234)
close(345)
close(456)
close(012)
close(987)
close(988)

open(unit=567,file='OUT/solution_dudx_el.dat',status='replace')
open(unit=678,file='OUT/solution_dudy_el.dat',status='replace')
open(unit=789,file='OUT/solution_phi.dat',status='replace')
open(unit=123,file='OUT/solution_p.dat',status='replace')
do i=1,nel
   xc=sum(x(icon(1:4,i)))*0.25d0
   yc=sum(y(icon(1:4,i)))*0.25d0
   write(567,'(5f20.10)') xc, yc, dudx_elemental(i),dudxth(xc,yc,ibench), dudx_elemental(i)-dudxth(xc,yc,ibench)
   write(678,'(5f20.10)') xc, yc, dvdy_elemental(i),dvdyth(xc,yc,ibench), dvdy_elemental(i)-dvdyth(xc,yc,ibench)
   write(789,'(5f20.10)') xc, yc, phi_elemental(i), phith(xc,yc,ibench),  phi_elemental(i)-phith(xc,yc,ibench)
   write(123,'(5f20.10)') xc, yc, Psol(i),          pth(xc,yc,ibench),    Psol(i)-pth(xc,yc,ibench)
end do
close(567)
close(678)
close(789)
close(123)

!===================================
!compute L2 norms 

call compute_errors(nel,np,x,y,u,v,Psol,icon,ibench,L2_err_u,L2_err_v,L2_err_p,L1_err_u,L1_err_v,L1_err_p)

L2_err_vel=sqrt(L2_err_u**2 + L2_err_v**2)

write(888,*) hx,L2_err_vel,L2_err_p,&
                L1_err_u,L1_err_v,L1_err_p ; call flush(888)

call compute_derivatives_errors(nel,np,x,y,dudx_nodal,dvdx_nodal,dudy_nodal,dvdy_nodal,phi_nodal,&
                                icon,ibench,&
                                dudx_L1,dudx_L2,dvdx_L1,dvdx_L2,dudy_L1,dudy_L2,dvdy_L1,dvdy_L2,&
                                phi_L1,phi_L2)

write(999,'(11es16.5)') hx,dudx_L1,dudx_L2,dvdx_L1,dvdx_L2,&
                dudy_L1,dudy_L2,dvdy_L1,dvdy_L2,&
                phi_L1,phi_L2 ; call flush(999)

!===================================!

call output_for_paraview (np,nel,x,y,u,v,Psol,icon,ibench,phi_nodal,density_nodal,Rv,Rp,dudx_nodal,dvdy_nodal)

!===================================!
!===========deallocate memory=======!
!===================================!

deallocate(x)
deallocate(y)
deallocate(u)
deallocate(v)
deallocate(icon)
deallocate(KKK)
deallocate(KKKmem)
deallocate(rhs_f)
deallocate(Vsol)
deallocate(Vsolmem)
deallocate(bc_fix)
deallocate(bc_val)
deallocate(Psol)
deallocate(Psolmem)
deallocate(rhs_h)
deallocate(G)
deallocate(Rv)
deallocate(Rp)
deallocate(density_nodal)
deallocate(dudx_elemental)
deallocate(dvdx_elemental)
deallocate(dudy_elemental)
deallocate(dvdy_elemental)
deallocate(dudx_nodal)
deallocate(dvdx_nodal)
deallocate(dudy_nodal)
deallocate(dvdy_nodal)
deallocate(phi_elemental)
deallocate(phi_nodal)

end do !end number of nodes loop

close(888)
close(999)

end program

!==============================================!
!==============================================!
!==============================================!

