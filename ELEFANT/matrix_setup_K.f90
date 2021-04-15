!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_K

use global_parameters
use structures
use timing

implicit none

integer :: counter,idof,LELTVAR,NA_ELT,inode,nnx,nny,nnz
integer :: i1,i2,nsees,nz,ip,i,ii,j,j1,j2,jj,jp,l,k,k1,k2,kk

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{matrix\_setup\_K}
!@@
!==================================================================================================!

if (iproc==0) call system_clock(counti,count_rate)

!==============================================================================!

if (use_MUMPS) then

   Nel=ndofV*mV          ! size of an elemental matrix

   !idV%N=NfemV

   !idV%NELT=nel
   LELTVAR=nel*Nel           ! nb of elts X size of elemental matrix
   NA_ELT=nel*Nel*(Nel+1)/2  ! nb of elts X nb of nbs in elemental matrix

   !allocate(idV%A_ELT (NA_ELT)) 
   !allocate(idV%RHS   (idV%N))  

   if (iproc==0) then

      !allocate(idV%ELTPTR(idV%NELT+1)) 
      !allocate(idV%ELTVAR(LELTVAR))    

      !=====[building ELTPTR]=====

      !do iel=1,nel
      !   idV%ELTPTR(iel)=1+(iel-1)*(ndofV*mV)
      !end do
      !idV%ELTPTR(iel)=1+nel*(ndofV*mV)

      !=====[building ELTVAR]=====

      counter=0
      do iel=1,nel
         do k=1,mV
            inode=mesh(iel)%iconV(k)
            do idof=1,ndofV
               counter=counter+1
               !idV%ELTVAR(counter)=(inode-1)*ndofV+idof
            end do
         end do
      end do

   end if

else

   if (iproc==0) then

      csrK%nr=NfemV
      csrK%nc=NfemV

      if (geometry=='cartesian' .and. ndim==2) then

         nnx=nelx+1
         nny=nely+1
         csrK%NZ=(4*4+(2*(nnx-2)+2*(nny-2))*6+(nnx-2)*(nny-2)*9)
         csrK%NZ=csrK%NZ*(ndofV**2)    
         csrK%nz=(csrK%nz-csrK%nr)/2+csrK%nr

         write(*,'(a)')       '          CSR matrix format SYMMETRIC' 
         write(*,'(a,i10,a)') '          csrK%nr      =',csrK%nr,'  ' 
         write(*,'(a,i10,a)') '          csrK%nz      =',csrK%nz,'  ' 
   
         allocate(csrK%ia(csrK%nr+1))   
         allocate(csrK%ja(csrK%nz))     
         allocate(csrK%mat(csrK%nz))    
         allocate(csrK%rhs(csrK%nr))    
         
         nz=0
         csrK%ia(1)=1
         do j1=1,nny
         do i1=1,nnx
            ip=(j1-1)*nnx+i1 ! node number
            do k=1,ndofV
               ii=2*(ip-1) + k ! address in the matrix
               nsees=0
               do j2=-1,1 ! exploring neighbouring nodes
               do i2=-1,1
                  i=i1+i2
                  j=j1+j2
                  if (i>=1 .and. i<= nnx .and. j>=1 .and. j<=nny) then ! if node exists
                     jp=(j-1)*nnx+i  ! node number of neighbour 
                     do l=1,ndofV
                        jj=2*(jp-1)+l  ! address in the matrix
                        if (jj>=ii) then  ! upper diagonal
                           nz=nz+1
                           csrK%ja(nz)=jj
                           nsees=nsees+1
                        end if
                     end do
                  end if
               end do
               end do
               csrK%ia(ii+1)=csrK%ia(ii)+nsees
            end do ! loop over ndofs
         end do
         end do

         if (debug) then
         write(*,*) '          nz=',nz
         write(*,*) '          csrK%ia (m/M)',minval(csrK%ia), maxval(csrK%ia)
         write(*,*) '          csrK%ja (m/M)',minval(csrK%ja), maxval(csrK%ja)
         end if

      end if ! cartesian 2D

      if (geometry=='cartesian' .and. ndim==3) then

         stop 'not tested - pb with nodes ordering probably'

         nnx=nelx+1
         nny=nely+1
         nnz=nelz+1
         csrK%nz=8*8                                              &! 8 corners with 8 neighbours
             +(nnx-2)*(nny-2)*(nnz-2)*27           &! all the inside nodes with 27 neighbours
             +(4*(nnx-2)+4*(nny-2)+4*(nnz-2))*12   &! the edge nodes with 12 neighbours  
             +2*(nnx-2)*(nny-2)*18                      &! 2 faces
             +2*(nnx-2)*(nnz-2)*18                      &! 2 faces
             +2*(nny-2)*(nnz-2)*18                       ! 2 faces
         csrK%nz=csrK%nz*(ndofV**2)                                ! matrix expands 3fold twice 

         csrK%nz=(csrK%nz-csrK%nr)/2+csrK%nr

         write(*,'(a)')       '          CSR matrix format SYMMETRIC' 
         write(*,'(a,i10,a)') '          csrK%nr      =',csrK%nr,'  ' 
         write(*,'(a,i10,a)') '          csrK%nz      =',csrK%nz,'  ' 

         allocate(csrK%ia(csrK%nr+1))   
         allocate(csrK%ja(csrK%nz))     
         allocate(csrK%mat(csrK%nz))    
         allocate(csrK%rhs(csrK%nr))    

         nz=0
         csrK%ia(1)=1
         do i1=1,nnx
         do j1=1,nny
         do k1=1,nnz
         ip=nny*nnz*(i1-1)+(j1-1)*nnz + k1 ! node number
         do kk=1,ndofV
            ii=ndofV*(ip-1) + kk ! address in the matrix
            nsees=0
            do i2=-1,1 ! exploring neighbouring nodes
            do j2=-1,1 ! exploring neighbouring nodes
            do k2=-1,1 ! exploring neighbouring nodes
               i=i1+i2
               j=j1+j2
               k=k1+k2
               if (i>=1 .and. i<= nnx .and. j>=1 .and. j<=nny .and. k>=1 .and. k<=nnz) then ! if node exists
                  jp=nny*nnz*(i-1)+(j-1)*nnz + k ! node number
                  do l=1,ndofV
                     jj=ndofV*(jp-1)+l  ! address in the matrix
                     if (jj>=ii) then  ! upper diagonal
                        nz=nz+1
                        csrK%ja(nz)=jj
                        nsees=nsees+1
                     end if
                  end do
               end if
            end do
            end do
            end do
            csrK%ia(ii+1)=csrK%ia(ii)+nsees
         end do ! loop over ndofs
         end do
         end do
         end do

         if (debug) then
         write(*,*) '          nz=',nz
         write(*,*) '          csrK%ia (m/M)',minval(csrK%ia), maxval(csrK%ia)
         write(*,*) '          csrK%ja (m/M)',minval(csrK%ja), maxval(csrK%ja)
         end if

      end if ! cartesian 3D
   
   end if ! iproc=0

end if ! use_MUMPS

!==============================================================================!

if (iproc==0) then 

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,*) '     -> matrix_setup_K ',elapsed

end if

end subroutine

!==================================================================================================!
!==================================================================================================!
