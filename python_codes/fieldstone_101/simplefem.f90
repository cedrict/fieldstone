!==============================================!
                                               !
program f101                                   !
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
integer, dimension(:,:), allocatable :: icon   ! connectivity array
                                               !
integer i1,i2,i,j,k,iel,counter,iq,jq          !
integer ik,jk,ikk,jkk,m1,m2,k1,k2,job,nz       !
integer ii,ij,j1,j2,jj,inode,ip,jp,l
                                               !  
real(8) Lx,Ly                                  ! size of the numerical domain
real(8) viscosity                              ! dynamic viscosity $\eta$ of the material
real(8) density                                ! mass density $\rho$ of the material
real(8) gx,gy                                  ! gravity acceleration
real(8) penalty                                ! penalty parameter lambda
real(8), dimension(:),   allocatable :: x,y    ! node coordinates arrays
real(8), dimension(:),   allocatable :: u,v    ! node velocity arrays
real(8), dimension(:),   allocatable :: press  ! pressure 
real(8), dimension(:),   allocatable :: bc_val ! array containing bc values
                                               !
real(8), external :: b1,b2,uth,vth,pth         ! body force and analytical solution
real(8) rq,sq,wq                               ! local coordinate and weight of qpoint
real(8) xq,yq                                  ! global coordinate of qpoint
real(8) uq,vq                                  ! velocity at qpoint
real(8) exxq,eyyq,exyq                         ! strain-rate components at qpoint  
real(8) Ael(m*ndof,m*ndof)                     ! elemental FEM matrix
real(8) Bel(m*ndof)                            ! elemental right hand side
real(8) N(m),dNdx(m),dNdy(m),dNdr(m),dNds(m)   ! shape fcts and derivatives
real(8) jcob                                   ! determinant of jacobian matrix
real(8) jcb(2,2)                               ! jacobian matrix
real(8) jcbi(2,2)                              ! inverse of jacobian matrix
real(8) Bmat(3,ndof*m)                         ! B matrix
real(8), dimension(3,3) :: Kmat                ! K matrix 
real(8), dimension(3,3) :: Cmat                ! C matrix
real(8) Aref                                   !
real(8) eps,fixt                               !
real(8) vrms,errv                              !
real(8) t1,t2                                  !
                                               !
logical, dimension(:), allocatable :: bc_fix   ! prescribed b.c. array
                                               !
!==============================================!
! solver y12 arrays and variables              !
!==============================================!
                                               !
integer, dimension(:), allocatable :: ia,ja    ! 
integer, dimension(:), allocatable :: snr,rnr  !
integer, dimension(:,:), allocatable :: ha     !
integer iflag(10), ifail                       !
integer nn,nn1,y12_NZ,y12_N,nsees              !
real(8) aflag(8)                               !
real(8), dimension(:), allocatable :: y12_A    !
real(8), dimension(:), allocatable :: y12_B    !
real(8), dimension(:), allocatable :: pivot    !
                                               !
!==============================================!
!=====[setup]==================================!
!==============================================!

open(unit=777,file='timings.ascii')
open(unit=888,file='vrms.ascii')
open(unit=999,file='errors.ascii')

do nnx=8,256,8 ! resolution loop

Lx=1.d0
Ly=1.d0

!nnx=150
nny=nnx

np=nnx*nny

nelx=nnx-1
nely=nny-1

nel=nelx*nely

penalty=1.d7

viscosity=1.d0
density=1.d0

Nfem=np*ndof

eps=1.d-10

Kmat(1,1)=1.d0 ; Kmat(1,2)=1.d0 ; Kmat(1,3)=0.d0  
Kmat(2,1)=1.d0 ; Kmat(2,2)=1.d0 ; Kmat(2,3)=0.d0  
Kmat(3,1)=0.d0 ; Kmat(3,2)=0.d0 ; Kmat(3,3)=0.d0  

Cmat(1,1)=2.d0 ; Cmat(1,2)=0.d0 ; Cmat(1,3)=0.d0  
Cmat(2,1)=0.d0 ; Cmat(2,2)=2.d0 ; Cmat(2,3)=0.d0  
Cmat(3,1)=0.d0 ; Cmat(3,2)=0.d0 ; Cmat(3,3)=1.d0  

!==============================================!
!===[allocate memory]==========================!
!==============================================!

allocate(x(np))
allocate(y(np))
allocate(u(np))
allocate(v(np))
allocate(icon(m,nel))
allocate(bc_fix(Nfem))
allocate(bc_val(Nfem))
allocate(press(nel))

!==============================================!
!===[grid points setup]========================!
!==============================================!

counter=0
do j=0,nely
   do i=0,nelx
      counter=counter+1
      x(counter)=dble(i)*Lx/dble(nelx)
      y(counter)=dble(j)*Ly/dble(nely)
   end do
end do

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

!==============================================!
!=====[compute number of nonzero's]============!
!==============================================!

y12_N=Nfem
Y12_NZ=(4*4+(2*(nnx-2)+2*(nny-2))*6+(nnx-2)*(nny-2)*9)
Y12_NZ=Y12_NZ*(ndof**2)    

print *,'======================================'
print *,'nnx=',nnx
print *,'N=',Y12_N
print *,'NZ=',Y12_NZ

!==============================================!
!=====[allocate arrays for y12 solver]=========
!==============================================!

nn=15*Y12_NZ ! heuristic sizes
nn1=15*Y12_NZ

allocate(y12_A(nn))  ; y12_A=0.d0
allocate(y12_B(Nfem)) ; y12_B=0.d0
allocate(snr(nn))
allocate(rnr(nn1))
allocate(ia(Nfem+1))
allocate(ja(Y12_NZ+1))

!==============================================!
!==============================================!

nz=0
snr(1)=1
rnr(1)=1
ia(1)=1
do j1=1,nny
do i1=1,nnx
   ip=(j1-1)*nnx+i1 ! node number
   do k=1,ndof
      ii=2*(ip-1) + k ! address in the matrix
      nsees=0
      do j2=-1,1 ! exploring neighbouring nodes
      do i2=-1,1
         i=i1+i2
         j=j1+j2
         if (i>=1 .and. i<= nnx .and. j>=1 .and. j<=nny) then ! if node exists
            jp=(j-1)*nnx+i  ! node number of neighbour 
            do l=1,ndof
               jj=2*(jp-1)+l  ! address in the matrix
               nz=nz+1
               snr(nz)=ii
               rnr(nz)=jj
               ja(nz)=jj
               nsees=nsees+1
            end do
         end if
      end do
      end do
      ia(ii+1)=ia(ii)+nsees
   end do ! loop over ndofs
end do
end do

!print *,nz
!print *,minval(snr(1:nz)), maxval(snr(1:nz))
!print *,minval(rnr(1:nz)), maxval(rnr(1:nz))

!==============================================!
!=====[define bc]==============================!
!==============================================!

bc_fix=.false.

do i=1,np
   if (x(i).lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
   if (x(i).gt.(Lx-eps)) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
   if (y(i).lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
   if (y(i).gt.(Ly-eps) ) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
end do

!==============================================!
!=====[build FE matrix]========================!
!==============================================!

call cpu_time(t1)

do iel=1,nel

   Ael=0.d0
   Bel=0.d0

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
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
         dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
         dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
         exxq=exxq+ dNdx(k)*u(icon(k,iel))
         eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
         exyq=exyq+ dNdx(k)*v(icon(k,iel)) *0.5d0 &
                  + dNdy(k)*u(icon(k,iel)) *0.5d0
      end do

      !write(999,*) xq,yq,uq,vq,exxq,eyyq,exyq

      do i=1,m
         i1=2*i-1
         i2=2*i
         Bmat(1,i1)=dNdx(i) ; Bmat(1,i2)=0.d0
         Bmat(2,i1)=0.d0    ; Bmat(2,i2)=dNdy(i)
         Bmat(3,i1)=dNdy(i) ; Bmat(3,i2)=dNdx(i)
      end do

      Ael=Ael + matmul(transpose(Bmat),matmul(viscosity*Cmat,Bmat))*wq*jcob

      do i=1,m
      i1=2*i-1
      i2=2*i
      !Bel(i1)=Bel(i1)+N(i)*jcob*wq*density*gx
      !Bel(i2)=Bel(i2)+N(i)*jcob*wq*density*gy
      Bel(i1)=Bel(i1)+N(i)*jcob*wq*b1(xq,yq)
      Bel(i2)=Bel(i2)+N(i)*jcob*wq*b2(xq,yq)
      end do

   end do
   end do

   ! 1 point integration

   rq=0.d0
   sq=0.d0
   wq=2.d0*2.d0

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

   do i=1,m
      i1=2*i-1
      i2=2*i
      Bmat(1,i1)=dNdx(i) ; Bmat(1,i2)=0.d0
      Bmat(2,i1)=0.d0    ; Bmat(2,i2)=dNdy(i)
      Bmat(3,i1)=dNdy(i) ; Bmat(3,i2)=dNdx(i)
   end do

   Ael=Ael + matmul(transpose(Bmat),matmul(penalty*Kmat,Bmat))*wq*jcob

   !=====================
   !=====[impose b.c.]===
   !=====================

   do ii=1,m  
      inode=icon(ii,iel)  
      do k=1,ndof       
         ij=(inode-1)*ndof+k    
         if (bc_fix(ij)) then  
         fixt=bc_val(ij) 
         i=(ii-1)*ndof+k               
         Aref=Ael(i,i)                
         do j=1,m*ndof               
            Bel(j)=Bel(j)-Ael(j,i)*fixt 
            Ael(i,j)=0.d0              
            Ael(j,i)=0.d0             
         enddo                       
         Ael(i,i)=Aref              
         Bel(i)=Aref*fixt          
         endif    
      enddo      
   enddo        

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
               ! ikk,jkk local integer coordinates in the elemental matrix
               do k=ia(m1),ia(m1+1)-1
                  if (ja(k)==m2) then
                  Y12_A(k)=Y12_A(k)+Ael(ikk,jkk)
                  end if
               end do
            end do
         end do
         y12_B(m1)=y12_B(m1)+Bel(ikk)
      end do
   end do

end do

call cpu_time(t2)
write(*,'(a,f10.3,a)') 'time build matrix :',t2-t1,' s'

!==============================================!
!=====[solve system with Y12]==================!
!==============================================!

aflag=0
iflag=0

allocate(ha(Nfem,11))
allocate(pivot(Nfem))

call cpu_time(t1)
call y12maf(y12_N,y12_NZ,y12_A,snr,nn,rnr,nn1,pivot,ha,Nfem,aflag,iflag,y12_B,ifail)
call cpu_time(t2)

write(777,'(i6,a,f10.3,a)') Nfem,' time Y12 solver   :',t2-t1,' s'
call flush(777)

if (ifail/=0) stop 'Y12 problem'

deallocate(pivot)
deallocate(ha)

!==============================================!
!=====[transfer solution]======================!
!==============================================!

do i=1,np
   u(i)=Y12_B((i-1)*ndof+1)
   v(i)=Y12_B((i-1)*ndof+2)
end do

open(unit=123,file='OUT/solution_u.dat',status='replace')
open(unit=234,file='OUT/solution_v.dat',status='replace')
do i=1,np
   write(123,'(5f20.10)') x(i),y(i),u(i),uth(x(i),y(i)),u(i)-uth(x(i),y(i))
   write(234,'(5f20.10)') x(i),y(i),v(i),vth(x(i),y(i)),v(i)-vth(x(i),y(i))
end do
close(123)
close(234)

!==============================================!
!=====[retrieve pressure]======================!
!==============================================!

open(unit=123,file='OUT/solution_p.dat',status='replace')

do iel=1,nel

   rq=0.d0
   sq=0.d0
      
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

   xq=0.d0
   yq=0.d0
   exxq=0.d0
   eyyq=0.d0
   do k=1,m
      xq=xq+N(k)*x(icon(k,iel))
      yq=yq+N(k)*y(icon(k,iel))
      exxq=exxq+ dNdx(k)*u(icon(k,iel))
      eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
   end do

   press(iel)=-penalty*(exxq+eyyq)
   
   write(123,*) xq,yq,press(iel),pth(xq,yq)

end do

close(123)

!==============================================!
!=====[compute vrms]===========================!
!==============================================!

errv=0d0
vrms=0d0

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

      xq=0.d0
      yq=0.d0
      uq=0.d0
      vq=0.d0
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
      end do

      errv=errv+(uq-uth(xq,yq))**2*jcob*wq
      vrms=vrms+(uq**2+vq**2)*jcob*wq

   end do
   end do

end do

vrms=sqrt(vrms/Lx/Ly)
errv=sqrt(errv/Lx/Ly)

write(888,*) 'h=',Lx/nelx,'vrms=',vrms ; call flush(888)
write(999,*) 'h=',Lx/nelx,'errv=',errv ; call flush(999)

deallocate(x,y,u,v)
deallocate(icon)
deallocate(bc_fix)
deallocate(bc_val)
deallocate(press)
deallocate(y12_A)
deallocate(y12_B)
deallocate(snr)
deallocate(rnr)
deallocate(ia)
deallocate(ja)

!==============================================!

end do ! nnx

close(777)
close(888)
close(999)

end program

!==============================================!
!==============================================!
!==============================================!

function uth (x,y)
real(8) uth,x,y
uth = x**2 * (1.d0-x)**2 * (2.d0*y - 6.d0*y**2 + 4*y**3)
end function

function vth (x,y)
real(8) vth,x,y
vth = -y**2 * (1.d0-y)**2 * (2.d0*x - 6.d0*x**2 + 4*x**3)
end function

function pth (x,y)
real(8) pth,x,y
pth = x*(1.d0-x)
end function

function b1 (x,y)
real(8) b1,x,y
b1 = ( (12.d0-24.d0*y)*x**4 + (-24.d0+48.d0*y)*x**3 + (-48.d0*y+72.d0*y**2-48.d0*y**3+12.d0)*x**2 &
   + (-2.d0+24.d0*y-72.d0*y**2+48.d0*y**3)*x + 1.d0-4.d0*y+12.d0*y**2-8.d0*y**3 )
end function

function b2 (x,y)
real(8) b2,x,y
b2= ( (8.d0-48.d0*y+48.d0*y**2)*x**3 + (-12.d0+72.d0*y-72*y**2)*x**2 + &
    (4.d0-24.d0*y+48.d0*y**2-48.d0*y**3+24.d0*y**4)*x - 12.d0*y**2 + 24.d0*y**3 -12.d0*y**4)
end function



