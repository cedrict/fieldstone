!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_K

use module_parameters
use module_mesh
use module_timing
use module_sparse, only : csrK
use module_arrays, only: vdof_belongs_to

implicit none

real(8) :: t3,t4
integer :: counter,idof,LELTVAR,NA_ELT,inode,nnx,nny,nnz,imod,iV
integer :: i1,i2,nsees,nz,ip,i,ii,j,j1,j2,jj,jp,l,k,k1,k2,kk
logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{matrix\_setup\_K}
!@@
!@@ If MUMPS is not used, this subroutine allocates arrays ia, ja, and mat of csrK, 
!@@ and builds arrays ia and ja.
!@@ This subroutine allocates and fills the {\sl vdof\_belongs\_to} array.
!@@ For a given node {\sl ip},
!@@ {\sl vdof\_belongs\_to(1,ip)} is the number of elements that {\sl ip} belongs to.
!@@ Furthermore, {\sl vdof\_belongs\_to(2:9,ip)} is the actual list of elements.

! see matrix_setup_K_MUMPS.f90 in old ELEFANT
! see matrix_setup_K_SPARSKIT.f90 in old ELEFANT
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

      csrK%N=NfemV

      !----------------------------------------------------------------------------------
      if (geometry=='cartesian' .and. ndim==2 .and. spaceV=='__Q1') then

         nnx=nelx+1
         nny=nely+1
         csrK%NZ=(4*4+(2*(nnx-2)+2*(nny-2))*6+(nnx-2)*(nny-2)*9)
         csrK%NZ=csrK%NZ*(ndofV**2)    
         if (.not.csrK%full_matrix_storage) csrK%nz=(csrK%nz-csrK%n)/2+csrK%n

         write(*,'(a)')     shift//'sparse matrix format' 
         write(*,'(a,l)')   shift//'full_matrix_storage ',csrK%full_matrix_storage
         write(*,'(a,i10)') shift//'csrK%n  =',csrK%n
         write(*,'(a,i10)') shift//'csrK%nz =',csrK%nz
   
         allocate(csrK%ia(csrK%N+1))   
         allocate(csrK%ja(csrK%NZ))     
         allocate(csrK%mat(csrK%NZ))    
         if (csrK%full_matrix_storage) then
            allocate(csrK%rnr(csrK%NZ))
            allocate(csrK%snr(csrK%NZ))
            csrK%rnr(:)=0
            csrK%snr(:)=0
            csrK%rnr(1)=1
            csrK%snr(1)=1
         end if

         !for later: rnr and ja are same ? but rnr must be over allocated ?
         
         NZ=0
         csrK%ia(1)=1
         if (csrK%full_matrix_storage) csrK%rnr(1)=1
         if (csrK%full_matrix_storage) csrK%snr(1)=1
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

                        if (csrK%full_matrix_storage) then
                           nz=nz+1
                           csrK%ja(nz)=jj
                           nsees=nsees+1
                           csrK%snr(nz)=ii
                           csrK%rnr(nz)=jj
                        elseif(jj>=ii) then  ! upper diagonal
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

         !if (debug) then
         !print *,'*************************'
         !print *,'**********debug**********'
         !write(*,*) 'nz=',nz
         !write(*,*) 'csrK%ia (m/M)',minval(csrK%ia), maxval(csrK%ia)
         !write(*,*) 'csrK%ja (m/M)',minval(csrK%ja), maxval(csrK%ja)
         !print *,'csrK%ia=',csrK%ia
         !print *,'**********debug**********'
         !print *,'*************************'
         !end if

      !----------------------------------------------------------------------------------
      else if (geometry=='cartesian' .and. ndim==3 .and. spaceV=='__Q1') then

         nnx=nelx+1
         nny=nely+1
         nnz=nelz+1
         csrK%nz=8*8                                  &! 8 corners with 8 neighbours
                +(nnx-2)*(nny-2)*(nnz-2)*27           &! all the inside nodes with 27 neighbours
                +(4*(nnx-2)+4*(nny-2)+4*(nnz-2))*12   &! the edge nodes with 12 neighbours  
                +2*(nnx-2)*(nny-2)*18                 &! 2 faces
                +2*(nnx-2)*(nnz-2)*18                 &! 2 faces
                +2*(nny-2)*(nnz-2)*18                  ! 2 faces
         csrK%nz=csrK%nz*(ndofV**2)                                ! matrix expands 3fold twice 

         if (.not. csrK%full_matrix_storage) csrK%nz=(csrK%nz-csrK%n)/2+csrK%n

         write(*,'(a)')     shift//'CSR matrix format symm' 
         write(*,'(a,l)')   shift//'full_matrix_storage ',csrK%full_matrix_storage
         write(*,'(a,i10)') shift//'csrK%n  =',csrK%n
         write(*,'(a,i10)') shift//'csrK%nz =',csrK%nz

         allocate(csrK%ia(csrK%n+1))   
         allocate(csrK%ja(csrK%nz))     
         allocate(csrK%mat(csrK%nz))    
         if (csrK%full_matrix_storage) then
            allocate(csrK%rnr(csrK%NZ))
            allocate(csrK%snr(csrK%NZ))
            csrK%rnr(:)=0
            csrK%snr(:)=0
            csrK%rnr(1)=1
            csrK%snr(1)=1
         end if

         NZ=0
         csrK%ia(1)=1
         do k1=1,nnz
         do j1=1,nny
         do i1=1,nnx
         ip=nnx*nny*(k1-1)+(j1-1)*nnx + i1 ! node number
         do kk=1,ndofV
            ii=ndofV*(ip-1) + kk ! address in the matrix
            nsees=0
            do k2=-1,1 ! exploring neighbouring nodes
            do j2=-1,1 ! exploring neighbouring nodes
            do i2=-1,1 ! exploring neighbouring nodes
               i=i1+i2
               j=j1+j2
               k=k1+k2
               if (i>=1 .and. i<= nnx .and. j>=1 .and. j<=nny .and. k>=1 .and. k<=nnz) then ! if node exists
                  jp=nnx*nny*(k-1)+(j-1)*nnx + i ! node number
                  do l=1,ndofV
                     jj=ndofV*(jp-1)+l  ! address in the matrix

                     if (csrK%full_matrix_storage) then
                        nz=nz+1
                        csrK%ja(nz)=jj
                        nsees=nsees+1
                        csrK%snr(nz)=ii
                        csrK%rnr(nz)=jj
                     elseif(jj>=ii) then  ! upper diagonal
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
         print *,'*************************'
         print *,'**********debug**********'
         write(*,*) 'nz=',nz
         write(*,*) 'csrK%ia (m/M)',minval(csrK%ia), maxval(csrK%ia)
         write(*,*) 'csrK%ja (m/M)',minval(csrK%ja), maxval(csrK%ja)
         print *,'**********debug**********'
         print *,'*************************'
         end if

      !----------------------------------------------------------------------------------
      else ! use generic approach

         allocate(vdof_belongs_to(9,NV)) !9 = max nb of elements a node can belong to

         vdof_belongs_to=0
         do iel=1,nel
            !print *,'elt:',iel
            do i=1,mV
               inode=mesh(iel)%iconV(i)
               !print *,'->',inode
               vdof_belongs_to(1,inode)=vdof_belongs_to(1,inode)+1
               if (vdof_belongs_to(1,inode)>9) then
                  print *, 'matrix_setup_K: array too small'
                  stop
               end if
               vdof_belongs_to(1+vdof_belongs_to(1,inode),inode)=iel
            end do
         end do

         if (debug) then
            print *,'*************************'
            print *,'**********debug**********'
            do iV=1,NV
               print *,'node',iV,'belongs to ',vdof_belongs_to(1,iV),' elts | ',vdof_belongs_to(2:,iV)
            end do
            print *,'**********debug**********'
            print *,'*************************'
         end if

         imod=NV/10

         call cpu_time(t3)
         allocate(alreadyseen(NV*ndofV))
         NZ=0
         do ip=1,NV
            if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(NV)*100.,'%'
            alreadyseen=.false.
            do k=1,vdof_belongs_to(1,ip)
               iel=vdof_belongs_to(1+k,ip)
               do i=1,mV
                  jp=mesh(iel)%iconV(i)
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+ndofV**2
                     alreadyseen(jp)=.true.
                  end if
               end do
            end do
         end do
         csrK%NZ=NZ
         csrK%nz=(csrK%nz-csrK%N)/2+csrK%N
         deallocate(alreadyseen)
         call cpu_time(t4) ; write(*,'(f10.3,a)') t4-t3,'s'

         write(*,'(a)') '----------------------------------------------------------------------'
         write(*,'(a)')       shift//'=====[Stokes system]================||'
         write(*,'(a)')       shift//'CSR matrix format SYMMETRIC         ||'
         write(*,'(a,i11,a)') shift//'csrK%N       =',csrK%N,'            ||'
         write(*,'(a,i11,a)') shift//'csrK%nz      =',csrK%nz,'            ||'

         allocate(csrK%ia(csrK%N+1))   
         allocate(csrK%ja(csrK%NZ))     
         allocate(csrK%mat(csrK%NZ))    

         call cpu_time(t3)
         allocate(alreadyseen(NV*ndofV))
         nz=0
         csrK%ia(1)=1
         do ip=1,NV
            if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(np)*100.,'%'
            do k1=1,ndofV
               ii=ndofV*(ip-1) + k1 ! address in the matrix
               nsees=0
               alreadyseen=.false.
               do k=1,vdof_belongs_to(1,ip)
                  iel=vdof_belongs_to(1+k,ip)
                     do i=1,mV
                        jp=mesh(iel)%iconV(i)
                        if (.not.alreadyseen(jp)) then
                           do k2=1,ndofV
                              jj=ndofV*(jp-1) + k2 ! address in the matrix
                              if (jj>=ii) then
                              nz=nz+1
                              csrK%ja(nz)=jj
                              nsees=nsees+1
                              !print *,ip,jp,ii,jj
                              end if
                           end do
                           alreadyseen(jp)=.true.
                        end if
                     end do
               end do    
               csrK%ia(ii+1)=csrK%ia(ii)+nsees    
            end do ! loop over ndofs  
         end do    
         deallocate(alreadyseen)    
         call cpu_time(t4) ; write(*,'(f10.3,a)') t4-t3,'s'

         write(*,'(a,i9)' ) shift//'nz=',nz
         write(*,'(a,2i9)') shift//'csrK%ia',minval(csrK%ia), maxval(csrK%ia)
         write(*,'(a,2i9)') shift//'csrK%ja',minval(csrK%ja), maxval(csrK%ja)

         !if (debug) then
         !   print *,'*************************'
         !   print *,'**********debug**********'
         !   print *,'csrK%ia=',csrK%ia
         !   print *,'**********debug**********'
         !   print *,'*************************'
         !end if


      end if 
   
   end if ! iproc=0

end if ! use_MUMPS

!==============================================================================!

if (iproc==0) then 

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'matrix_setup_K (',elapsed,' s)'

end if

end subroutine

!==================================================================================================!
!==================================================================================================!
