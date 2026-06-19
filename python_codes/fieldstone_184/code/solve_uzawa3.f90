subroutine solve_uzawa3(KKK,G,Nfem,nel,rhs_f,rhs_h,Vsol,Psol)

implicit none

integer, intent(in) :: Nfem, nel
real(8), dimension(Nfem,Nfem) :: KKK,KKKmem
real(8), dimension(Nfem,Nfem) :: G(Nfem,nel)
real(8), dimension(Nfem) :: Vsol,rhs_f,Vsolmem
real(8), dimension(nel) :: Psol,Psolmem
real(8), dimension(nel) :: rhs_h,q,d,qkp1,dkp1

real(8), dimension(Nfem) :: B,h,phi

real(8), dimension(:),   allocatable :: work   ! work array needed by the solver
integer, dimension(:), allocatable :: ipvt     ! work array needed by the solver 
integer iter,job
real(8) rcond,P_diff,V_diff

real(8) :: alpha,beta
real(8), parameter :: tol=1.d-6
integer, parameter :: niter=250

open(unit=123,file='conv_uzawa3.dat')

Psol=0.d0
Vsol=0.d0
Psolmem=0.d0
Vsolmem=0.d0
   
KKKmem=KKK
job=0
allocate(work(Nfem))
allocate(ipvt(Nfem))
call DGECO (KKK, Nfem, Nfem, ipvt, rcond, work)

!compute u1
B=rhs_f-matmul(G,Psol)
call DGESL (KKK, Nfem, Nfem, ipvt, B, job) 
Vsol=B

q=rhs_h-matmul(transpose(G),Vsol)

d=-q

do iter=1,niter

   !compute pk

   phi=matmul(G,d) 

   !compute hk=A^-1 pk

   B=phi
   call DGESL (KKK, Nfem, Nfem, ipvt, B, job) 
   h=B

   !compute alpha

   alpha=dot_product(q,q)/dot_product(phi,h) !; print *,alpha

   !update pressure

   Psol=Psol+alpha*d

   !update velocity

   Vsol=Vsol-alpha*h

   ! compute qk+1

   qkp1=rhs_h-matmul(transpose(G),Vsol)

   ! compute beta

   beta=dot_product(qkp1,qkp1)/dot_product(q,q)
 
   ! compute dkp1

   dkp1=-qkp1+beta*d

   !check for convergence

   V_diff=maxval(abs(Vsol-Vsolmem))/maxval(abs(Vsol))
   P_diff=maxval(abs(Psol-Psolmem))/maxval(abs(Psol))

   write(*,*) iter,V_diff,P_diff,maxval(abs(Vsol-Vsolmem)),maxval(abs(Psol-Psolmem))

   Psolmem=Psol
   Vsolmem=Vsol
   d=dkp1
   q=qkp1

   if (max(V_diff,P_diff)<tol) exit

end do

deallocate(ipvt)
deallocate(work)

close(123)

end subroutine



