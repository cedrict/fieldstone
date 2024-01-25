!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_K_matrix_CSR

use module_parameters, only: stokes_solve_strategy,NfemVel,ndim,spaceV,NU,NV,NW,mU,mV,mW,&
                             iproc,debug,geometry,nelx,nely,nelz,ndofV
use module_mesh 
use module_sparse, only : csrK
use module_arrays, only: Unode_belongs_to,Vnode_belongs_to,Wnode_belongs_to
use module_timing

implicit none

real(8) :: t3,t4
integer :: counter,idof,LELTVAR,NA_ELT,inode,nnx,nny,nnz,imod,NNel
integer :: i1,i2,nsees,nz,ip,i,ii,j,j1,j2,jj,jp,l,k,k1,k2,kk,iel
logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_K\_matrix\_CSR}
!@@ This subroutine allocates arrays ia, ja, and mat of csrK, 
!@@ and builds arrays ia and ja.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (stokes_solve_strategy=='penalty') then 
   csrK%full_matrix_storage=.true. ! y12m solver 
else
   csrK%full_matrix_storage=.false. ! pcg_solver 
end if



      csrK%N=NfemVel

      !----------------------------------------------------------------------------------
      if (geometry=='XXcartesian' .and. ndim==2 .and. spaceV=='__Q1') then

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

         if (debug) then
         write(2345,*) limit//'matrix_setup_K'//limit
         write(2345,*) 'nz=',nz
         write(2345,*) 'csrK%ia (m/M)',minval(csrK%ia), maxval(csrK%ia)
         write(2345,*) 'csrK%ja (m/M)',minval(csrK%ja), maxval(csrK%ja)
         write(2345,*) 'csrK%ia=',csrK%ia
         end if

      !----------------------------------------------------------------------------------
      else if (geometry=='XXcartesian' .and. ndim==3 .and. spaceV=='__Q1') then

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
         write(2345,*) limit//'matrix_setup_K'//limit
         write(2345,*) 'nz=',nz
         write(2345,*) 'csrK%ia (m/M)',minval(csrK%ia), maxval(csrK%ia)
         write(2345,*) 'csrK%ja (m/M)',minval(csrK%ja), maxval(csrK%ja)
         end if

      !----------------------------------------------------------------------------------
      else ! use generic approach


         call cpu_time(t3)
         allocate(alreadyseen(NU+NV+NW))
         NZ=0





         !xx,xy,xz
         imod=NU/4
         do ip=1,NU
            if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(NU)*100.,'%'
            alreadyseen=.false.
            do k=1,Unode_belongs_to(1,ip)
               iel=Unode_belongs_to(1+k,ip)
               !-------- 
               do i=1,mU
                  jp=mesh(iel)%iconU(i)
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
               !-------- 
               do i=1,mV
                  jp=mesh(iel)%iconV(i)+NU
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
               !-------- 
               do i=1,mW
                  jp=mesh(iel)%iconW(i)+NU+NV
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
            end do
         end do
         print *,NZ

         !yx,yy,yz
         imod=NV/4
         do ip=1,NV
            if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(NV)*100.,'%'
            alreadyseen=.false.
            do k=1,Vnode_belongs_to(1,ip)
               iel=Vnode_belongs_to(1+k,ip)
               !-------- 
               do i=1,mU
                  jp=mesh(iel)%iconU(i)
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
               !-------- 
               do i=1,mV
                  jp=mesh(iel)%iconV(i)+NU
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
               !-------- 
               do i=1,mW
                  jp=mesh(iel)%iconW(i)+NU+NV
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
            end do
         end do
         print *,NZ

         !zx,zy,zz
         imod=NW/4
         do ip=1,NW
            if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(NW)*100.,'%'
            alreadyseen=.false.
            do k=1,Wnode_belongs_to(1,ip)
               iel=Wnode_belongs_to(1+k,ip)
               !-------- 
               do i=1,mU
                  jp=mesh(iel)%iconU(i)
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
               !-------- 
               do i=1,mV
                  jp=mesh(iel)%iconV(i)+NU
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
               !-------- 
               do i=1,mW
                  jp=mesh(iel)%iconW(i)+NU+NV
                  if (.not.alreadyseen(jp)) then
                     NZ=NZ+1
                     alreadyseen(jp)=.true.
                  end if
               end do
            end do
         end do
         print *,NZ

         stop 'aaaa'         


         !do ip=1,NV
         !   if (mod(ip,imod)==0) write(*,'(TL10, F6.1,a)',advance='no') real(ip)/real(NV)*100.,'%'
         !   alreadyseen=.false.
         !   do k=1,vnode_belongs_to(1,ip)
         !      iel=vnode_belongs_to(1+k,ip)
         !      do i=1,mV
         !         jp=mesh(iel)%iconV(i)
         !         if (.not.alreadyseen(jp)) then
         !            NZ=NZ+ndofV**2
         !            alreadyseen(jp)=.true.
         !         end if
         !      end do
         !   end do
         !end do

         deallocate(alreadyseen)
         csrK%NZ=NZ
         csrK%NZ=(csrK%NZ-csrK%N)/2+csrK%N
         call cpu_time(t4) ; write(*,'(f10.3,a)') t4-t3,'s'

         write(*,'(a)')       shift//'CSR matrix format SYMMETRIC  '
         write(*,'(a,i11,a)') shift//'csrK%N       =',csrK%N,' '
         write(*,'(a,i11,a)') shift//'csrK%nz      =',csrK%nz,' '

         allocate(csrK%ia(csrK%N+1)) ; csrK%ia=0 
         allocate(csrK%ja(csrK%NZ))  ; csrK%ja=0 
         allocate(csrK%mat(csrK%NZ)) ; csrK%mat=0 

         call cpu_time(t3)
         allocate(alreadyseen(NU+NV+NW))
         nz=0
         csrK%ia(1)=1
         do ip=1,NU
            ii=ip ! address in the matrix
            nsees=0
            alreadyseen=.false.
            do k=1,Unode_belongs_to(1,ip)
               iel=Unode_belongs_to(1+k,ip)
               do i=1,mU
                  jp=mesh(iel)%iconU(i)
                  if (.not.alreadyseen(jp)) then
                     jj=jp
                     if (jj>=ii) then
                        nz=nz+1
                        csrK%ja(nz)=jj
                        nsees=nsees+1
                        !print *,ip,jp,ii,jj
                     end if
                     alreadyseen(jp)=.true.
                  end if
               end do
            end do    
            csrK%ia(ii+1)=csrK%ia(ii)+nsees    
         end do    
         deallocate(alreadyseen)    
         call cpu_time(t4) ; write(*,'(f10.3,a)') t4-t3,'s'




         do ip=1,NV
            do k1=1,ndofV
               ii=ndofV*(ip-1) + k1 ! address in the matrix
               nsees=0
               alreadyseen=.false.
               do k=1,vnode_belongs_to(1,ip)
                  iel=vnode_belongs_to(1+k,ip)
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

         if (debug) then
         write(2345,'(a)') limit//'matrix_setup_K'//limit
         write(2345,*) 'csrK%ia=',csrK%ia
         do i=1,NfemVel
         write(2345,*) i,'th line: csrK%ja=',csrK%ja(csrK%ia(i):csrK%ia(i+1)-1)-1
         end do
         end if


      end if 
   













if (debug) then
write(2345,*) limit//'name'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_K_matrix_CSR:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
