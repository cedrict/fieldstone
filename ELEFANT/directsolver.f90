!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!
!@@ \subsection{directsolver}
!@@  I do not know where this solver comes from. It works with full matrices.
!@@ Could it be that it comes from the brown meshless methods book?

   SUBROUTINE SolverBand(ak,fp,neq,nmat)
!------------------------------------------------------------------
! Slover for banded linear equations
! input-ak,fp,neq,nmat
! output--fp
!------------------------------------------------------------------

       implicit real(8) (a-h,o-z)
       dimension ak(nmat,nEq),fp(nmat)
       real(8), allocatable :: tp(:,:)
       real(8), allocatable :: stfp(:,:)
       allocate (tp(1:neq,1:nmat))
       allocate (stfp(1:neq,1:neq))

       !write(*,'(a,f11.4,a)') '                                                  SolverBand: solver allocates ',(nmat*nEq+nmat*neq)*8./1024./1024.,' Mb'

       ep=1.d-10
       do i=1,nEq
          do j=1,nEq
             stfp(i,j)=0.
             tp(i,j)=0.
          enddo
       enddo
       do i=1,nEq
          do j=1,nEq
             stfp(i,j)=ak(i,j)
          enddo
       enddo
       ni=nEq
       Lp=0 ! half band width 
       do 20 i=1,ni
          do j=ni,i,-1
          if(stfp(i,j).ne.0.) then ! stfp[,] stifness matrix
             if(abs(j-i).gt.Lp) Lp=abs(j-i)
                go to 21
             endif
          enddo
 21    continue
       do j=1,i
          if(stfp(i,j).ne.0.) then
             if(abs(j-i).gt.Lp) Lp=abs(j-i)
                go to 20
             endif
          enddo               
 20    continue

       ilp=2*lp+1  ! band width
       nm=nEq
       if(ilp.lt.nEq) then
          call aband(stfp,fp,tp,nm,lp,ilp,nmat) ! solver for band matrix
       else
          call GaussSolver(nEq,nmat,ak,fp,ep,kkkk) ! standard solver
       endif
       deallocate (tp)
       deallocate (stfp)
   END


   SUBROUTINE ABAND(A,F,B,N,L,IL,nmat)
       implicit real(8) (a-h,o-z)
       DIMENSION A(N,N),F(N)
       DIMENSION B(N,nmat),d(n,1)
       M=1
       LP1=L+1
       DO I=1,N
          DO K=1,IL
             B(I,K)=0.
             IF(I.LE.LP1) B(I,K)=A(I,K)
             IF(I.GT.LP1.AND.I.LE.(N-L)) B(I,K)=A(I,I+K-LP1)
             IF(I.GT.(N-L).AND.(I+K-LP1).LE.N) B(I,K)=A(I,I+K-LP1)
          ENDDO
       ENDDO
       DO I=1,N
          D(I,1)=F(I)
       ENDDO
       IT=1
       IF (IL.NE.2*L+1) THEN
          IT=-1
          WRITE(*,*)'***FAIL***'
          RETURN
       END IF
       LS=L+1
       DO 100 K=1,N-1
          P=0.0
             DO I=K,LS
                IF (ABS(B(I,1)).GT.P) THEN
                   P=ABS(B(I,1))
                   IS=I
                END IF
             ENDDO
          IF (P+1.0.EQ.1.0) THEN
             IT=0
             WRITE(*,*)'***FAIL***'
             RETURN
          END IF
          DO J=1,M
             T=D(K,J)
             D(K,J)=D(IS,J)
             D(IS,J)=T
          ENDDO
          DO J=1,IL
             T=B(K,J)
             B(K,J)=B(IS,J)
             B(IS,J)=T
          ENDDO
          DO J=1,M
             D(K,J)=D(K,J)/B(K,1)
          ENDDO
          DO J=2,IL
             B(K,J)=B(K,J)/B(K,1)
          ENDDO
          DO I=K+1,LS
             T=B(I,1)
             DO J=1,M
                D(I,J)=D(I,J)-T*D(K,J)
             ENDDO
             DO J=2,IL
                B(I,J-1)=B(I,J)-T*B(K,J)
             ENDDO
             B(I,IL)=0.0
          ENDDO
          IF (LS.NE.N) LS=LS+1
 100   CONTINUE
       IF (ABS(B(N,1))+1.0.EQ.1.0) THEN
          IT=0
          WRITE(*,*)'***FAIL***'
          RETURN
       END IF
       DO J=1,M
          D(N,J)=D(N,J)/B(N,1)
       ENDDO
          JS=2
       DO 150 I=N-1,1,-1
          DO K=1,M
             DO J=2,JS
                D(I,K)=D(I,K)-B(I,J)*D(I+J-1,K)
             ENDDO
          ENDDO
          IF (JS.NE.IL) JS=JS+1
 150   CONTINUE
      
       if(it.le.0) write(*,*) "aband failed"
       DO I=1,N
          F(I)=D(I,1)
       ENDDO
    RETURN
    END

   SUBROUTINE GaussSolver(n,mk,a,b,ep,kwji)
       implicit real(8) (a-h,o-z)
       dimension a(mk,mk),b(mk)
       integer, allocatable :: m(:)
       allocate (m(2*mk))
!       write(*,'(a,f11.4,a)') '                                                  GaussSolver: solver allocates ',(2*mk)*8./1024./1024.,'Mb'
       ep=1.0e-10
       kwji=0
       do i=1,n
          m(i)=i
       enddo
       do 20 k=1,n
          p=0.0
          do 30 i=k,n
             do 30 j=k,n
             if(abs(a(i,j)).le.abs(p)) goto 30
                p=a(i,j)
                io=i
                jo=j        
  30      continue

          if(abs(p)-ep) 200,200,300
 200      kwji=1
          return
 300      if(jo.eq.k) goto 400
          do i=1,n
             t=a(i,jo)
             a(i,jo)=a(i,k)
             a(i,k)=t
          enddo 
          j=m(k)
          m(k)=m(jo)
          m(jo)=j
 400      if(io.eq.k) goto 500
          do j=k,n
             t=a(io,j)
             a(io,j)=a(k,j)
             a(k,j)=t
          enddo
          t=b(io)
          b(io)=b(k)
          b(k)=t
  500     p=1.0/p
          in=n-1
          if(k.eq.n) goto 600
          do j=k,in
             a(k,j+1)=a(k,j+1)*p
          enddo
 600      b(k)=b(k)*p
          if(k.eq.n) goto 20
          do i=k,in
             do j=k,in
                a(i+1,j+1)=a(i+1,j+1)-a(i+1,k)*a(k,j+1)
             enddo
             b(i+1)=b(I+1)-a(i+1,k)*b(k)
          enddo
 20    continue
       do i1=2,n
          i=n+1-i1
          do j=i,in
             b(i)=b(i)-a(i,j+1)*b(j+1)
          enddo
       enddo
       do k=1,n
          i=m(k)
          a(1,i)=b(k)
       enddo
       do k=1,n
          b(k)=a(1,k)
       enddo
       kwji=0
       deallocate (m)
       return
   END
 





  Subroutine GaussEqSolver_Sym(n,ma,a,b,ep,kwji)
!------------------------------------------------------------------
!  Solve sysmmetric linear equation ax=b by using Gauss elimination.
!  If kwji=1, no solution;if kwji=0,has solution
!  Input--n,ma,a(ma,n),b(n),ep,
!  Output--b,kwji
!------------------------------------------------------------------
       implicit real(8) (a-h,o-z)
       dimension a(ma,n),b(n),m(n+1)
       do 10 i=1,n
10     m(i)=i
       do 120 k=1,n
          p=0.0
          do 20 i=k,n
             do 20 j=k,n
                if(dabs(a(i,j)).gt.dabs(p)) then
                   p=a(i,j)
                   io=i
                   jo=j
                endif
20        continue
          if(dabs(p)-ep) 30,30,35
30        kwji=1
          return
35        continue
          if(jo.eq.k) go to 45
          do 40 i=1,n
             t=a(i,jo)
             a(i,jo)=a(i,k)
             a(i,k)=t
40        continue
          j=m(k)
          m(k)=m(jo)
          m(jo)=j
45        if(io.eq.k) go to 55
          do 50 j=k,n
             t=a(io,j)
             a(io,j)=a(k,j)
             a(k,j)=t
50        continue
          t=b(io)
          b(io)=b(k)
          b(k)=t
55        p=1./p
          in=n-1
          if(k.eq.n) go to 65
          do 60 j=k,in
60        a(k,j+1)=a(k,j+1)*p
65        b(k)=b(k)*p
          if(k.eq.n) go to 120
          do 80 i=k,in
             do 70 j=k,in
70              a(i+1,j+1)=a(i+1,j+1)-a(i+1,k)*a(k,j+1)
80              b(i+1)=b(i+1)-a(i+1,k)*b(k) 
120       continue
          do 130 i1=2,n
             i=n+1-i1
             do 130 j=i,in
130       b(i)=b(i)-a(i,j+1)*b(j+1)
          do 140 k=1,n
             i=m(k)
140       a(1,i)=b(k)
          do 150 k=1,n
150       b(k)=a(1,k)
          kwji=0
    return
    end 
 





