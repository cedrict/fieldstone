!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_cartesian2D

use global_parameters
use structures
use constants 

implicit none

integer counter
integer ielx,iely,iq,jq
real(8) hx,hy,rq,sq,NNNV(mV)

call system_clock(counti,count_rate)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{\tt setup\_cartesian2D}
!@@ 
!==================================================================================================!

if (iproc==0) then

hx=Lx/nelx
hy=Ly/nely


allocate(mesh(nel))

!==========================================================
!velocity 

if (pair=='q1p0' .or. pair=='q1q1') then

   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%ielx=ielx
         mesh(counter)%iely=iely
         mesh(counter)%iconV(1)=ielx+(iely-1)*(nelx+1)    
         mesh(counter)%iconV(2)=ielx+1+(iely-1)*(nelx+1)    
         mesh(counter)%iconV(3)=ielx+1+iely*(nelx+1)    
         mesh(counter)%iconV(4)=ielx+iely*(nelx+1)    
         mesh(counter)%xV(1)=(ielx-1)*hx
         mesh(counter)%xV(2)=(ielx-1)*hx+hx
         mesh(counter)%xV(3)=(ielx-1)*hx+hx
         mesh(counter)%xV(4)=(ielx-1)*hx
         mesh(counter)%yV(1)=(iely-1)*hy
         mesh(counter)%yV(2)=(iely-1)*hy
         mesh(counter)%yV(3)=(iely-1)*hy+hy
         mesh(counter)%yV(4)=(iely-1)*hy+hy
         mesh(counter)%xc=(ielx-1)*hx+hx/2
         mesh(counter)%yc=(iely-1)*hy+hy/2
         mesh(counter)%hx=hx
         mesh(counter)%hy=hy
         if (ielx==1)    mesh(counter)%left=.true.
         if (ielx==nelx) mesh(counter)%right=.true.
         if (iely==1)    mesh(counter)%bottom=.true.
         if (iely==nely) mesh(counter)%top=.true.
      end do    
   end do    

end if

if (pair=='q1q1') then ! add bubble node
   do iel=1,nel
      mesh(iel)%xV(4)=mesh(iel)%xc
      mesh(iel)%yV(4)=mesh(iel)%yc
      mesh(counter)%iconV(5)=nel+iel
   end do
end if

!==========================================================
! pressure 

if (pair=='q1p0') then
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%iconP(1)=counter
         mesh(counter)%xP(1)=mesh(counter)%xC
         mesh(counter)%yP(1)=mesh(counter)%yC
      end do    
   end do    
end if

if (pair=='q1q1') then
   mesh(1:nel)%xP(1)=mesh(1:nel)%xV(1)
   mesh(1:nel)%xP(2)=mesh(1:nel)%xV(2)
   mesh(1:nel)%xP(3)=mesh(1:nel)%xV(3)
   mesh(1:nel)%xP(4)=mesh(1:nel)%xV(4)
   mesh(1:nel)%yP(1)=mesh(1:nel)%yV(1)
   mesh(1:nel)%yP(2)=mesh(1:nel)%yV(2)
   mesh(1:nel)%yP(3)=mesh(1:nel)%yV(3)
   mesh(1:nel)%yP(4)=mesh(1:nel)%yV(4)
   mesh(1:nel)%iconP(1)=mesh(1:nel)%iconV(1)
   mesh(1:nel)%iconP(2)=mesh(1:nel)%iconV(2)
   mesh(1:nel)%iconP(3)=mesh(1:nel)%iconV(3)
   mesh(1:nel)%iconP(4)=mesh(1:nel)%iconV(4)
end if


!==========================================================
! temperature 

mesh(1:nel)%xT(1)=mesh(1:nel)%xV(1)
mesh(1:nel)%xT(2)=mesh(1:nel)%xV(2)
mesh(1:nel)%xT(3)=mesh(1:nel)%xV(3)
mesh(1:nel)%xT(4)=mesh(1:nel)%xV(4)
mesh(1:nel)%yT(1)=mesh(1:nel)%yV(1)
mesh(1:nel)%yT(2)=mesh(1:nel)%yV(2)
mesh(1:nel)%yT(3)=mesh(1:nel)%yV(3)
mesh(1:nel)%yT(4)=mesh(1:nel)%yV(4)

mesh(1:nel)%iconT(1)=mesh(1:nel)%iconV(1)
mesh(1:nel)%iconT(2)=mesh(1:nel)%iconV(2)
mesh(1:nel)%iconT(3)=mesh(1:nel)%iconV(3)
mesh(1:nel)%iconT(4)=mesh(1:nel)%iconV(4)

!==========================================================
! quadrature points

do iel=1,nel
   allocate(mesh(iel)%xq(nqel))
   allocate(mesh(iel)%yq(nqel))
   allocate(mesh(iel)%zq(nqel))
   allocate(mesh(iel)%weightq(nqel))
   allocate(mesh(iel)%rq(nqel))
   allocate(mesh(iel)%sq(nqel))
   allocate(mesh(iel)%tq(nqel))
   allocate(mesh(iel)%gxq(nqel))
   allocate(mesh(iel)%gyq(nqel))
   allocate(mesh(iel)%gzq(nqel))
   allocate(mesh(iel)%rhoq(nqel))
   allocate(mesh(iel)%etaq(nqel))
   allocate(mesh(iel)%hcondq(nqel))
   allocate(mesh(iel)%hcapaq(nqel))
   allocate(mesh(iel)%hprodq(nqel))
end do

do iel=1,nel
   counter=0
   do iq=1,nq_per_dim
   do jq=1,nq_per_dim
      counter=counter+1
      rq=qcoords(iq)
      sq=qcoords(jq)
      call NNV(rq,sq,0,NNNV(1:mV),mV,ndim,pair)
      mesh(iel)%xq(counter)=sum(mesh(iel)%xV(1:mV)*NNNV(1:mV))
      mesh(iel)%yq(counter)=sum(mesh(iel)%yV(1:mV)*NNNV(1:mV))
      mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)
      mesh(iel)%rq(counter)=rq
      mesh(iel)%sq(counter)=sq
      mesh(iel)%tq(counter)=0
   end do
   end do
end do

!==========================================================
! flag nodes on boundaries

do iel=1,nel
   mesh(iel)%left_node(1)=(abs(mesh(iel)%xV(1)-0)<eps*Lx)
   mesh(iel)%left_node(2)=(abs(mesh(iel)%xV(2)-0)<eps*Lx)
   mesh(iel)%left_node(3)=(abs(mesh(iel)%xV(3)-0)<eps*Lx)
   mesh(iel)%left_node(4)=(abs(mesh(iel)%xV(4)-0)<eps*Lx)

   mesh(iel)%right_node(1)=(abs(mesh(iel)%xV(1)-Lx)<eps*Lx)
   mesh(iel)%right_node(2)=(abs(mesh(iel)%xV(2)-Lx)<eps*Lx)
   mesh(iel)%right_node(3)=(abs(mesh(iel)%xV(3)-Lx)<eps*Lx)
   mesh(iel)%right_node(4)=(abs(mesh(iel)%xV(4)-Lx)<eps*Lx)

   mesh(iel)%bottom_node(1)=(abs(mesh(iel)%yV(1)-0)<eps*Ly)
   mesh(iel)%bottom_node(2)=(abs(mesh(iel)%yV(2)-0)<eps*Ly)
   mesh(iel)%bottom_node(3)=(abs(mesh(iel)%yV(3)-0)<eps*Ly)
   mesh(iel)%bottom_node(4)=(abs(mesh(iel)%yV(4)-0)<eps*Ly)

   mesh(iel)%top_node(1)=(abs(mesh(iel)%yV(1)-Ly)<eps*Ly)
   mesh(iel)%top_node(2)=(abs(mesh(iel)%yV(2)-Ly)<eps*Ly)
   mesh(iel)%top_node(3)=(abs(mesh(iel)%yV(3)-Ly)<eps*Ly)
   mesh(iel)%top_node(4)=(abs(mesh(iel)%yV(4)-Ly)<eps*Ly)
end do








call export_mesh

end if ! iproc

!==================================================================================================!
!==================================================================================================!

!call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

!write(*,*) '     -> cartesian 2D setup',elapsed,'s'

end subroutine

!==================================================================================================!
!==================================================================================================!
