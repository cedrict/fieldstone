!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_belongs

use module_parameters, only: nel,iproc,NU,NV,NW,NP,iel,debug,mU,mV,mW,mP,ndim
use module_mesh 
use module_arrays, only: Unode_belongs_to,Vnode_belongs_to,Wnode_belongs_to,Pnode_belongs_to
use module_timing

implicit none

integer :: inode,i,iU,iV,iW,iP

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_belongs}
!@@ This subroutine allocates and fills the {\sl U,V,Wnode\_belongs\_to} and {\sl Pnode\_belongs\_to} 
!@@ arrays. For a given Unode {\sl ip},
!@@ {\sl Unode\_belongs\_to(1,ip)} is the number of elements that {\sl ip} belongs to.
!@@ Furthermore, {\sl Unode\_belongs\_to(2:9,ip)} is the actual list of elements.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

write(*,'(a,3i5)') shift//'NU,NV,NW=',NU,NV,NW

allocate(Unode_belongs_to(9,NU)) ; Unode_belongs_to=0
allocate(Vnode_belongs_to(9,NV)) ; Vnode_belongs_to=0
allocate(Wnode_belongs_to(9,NW)) ; Wnode_belongs_to=0

do iel=1,nel
   do i=1,mU
      inode=mesh(iel)%iconU(i)
      Unode_belongs_to(1,inode)=Unode_belongs_to(1,inode)+1
      if (Unode_belongs_to(1,inode)>9) then
         print *, 'compute_belongs: Unode_belongs_to array too small'
         stop
      end if
      Unode_belongs_to(1+Unode_belongs_to(1,inode),inode)=iel
   end do
end do

do iel=1,nel
   do i=1,mV
      inode=mesh(iel)%iconV(i)
      Vnode_belongs_to(1,inode)=Vnode_belongs_to(1,inode)+1
      if (Vnode_belongs_to(1,inode)>9) then
         print *, 'compute_belongs: Vnode_belongs_to array too small'
         stop
      end if
      Vnode_belongs_to(1+Vnode_belongs_to(1,inode),inode)=iel
   end do
end do

if (ndim>2) then
do iel=1,nel
   do i=1,mW
      inode=mesh(iel)%iconW(i)
      Wnode_belongs_to(1,inode)=Wnode_belongs_to(1,inode)+1
      if (Wnode_belongs_to(1,inode)>9) then
         print *, 'compute_belongs: Wnode_belongs_to array too small'
         stop
      end if
      Wnode_belongs_to(1+Wnode_belongs_to(1,inode),inode)=iel
   end do
end do
end if

!----------------------------------------------------------
! should it be NV long ?!?!!?

allocate(Pnode_belongs_to(9,NV)) !9 = max nb of elements a node can belong to

Pnode_belongs_to=0
do iel=1,nel
   !print *,'elt:',iel
   do i=1,mP
      inode=mesh(iel)%iconP(i)
      !print *,'->',inode
      Pnode_belongs_to(1,inode)=Pnode_belongs_to(1,inode)+1
      if (Pnode_belongs_to(1,inode)>9) then
         print *, 'compute_belongs: Pnode_belongs_to array too small'
         stop
      end if
      Pnode_belongs_to(1+Pnode_belongs_to(1,inode),inode)=iel
   end do
end do

!-----------------------------------------------------------------------------!

if (debug) then
write(2345,'(a)') limit//'compute_belongs'//limit
do iU=1,NU
write(2345,'(a,i6,a,i2,a,9i6)') 'U node',iU,' belongs to ',Unode_belongs_to(1,iU),' elts:  ',Unode_belongs_to(2:,iU)
end do
do iV=1,NV
write(2345,'(a,i6,a,i2,a,9i6)') 'V node',iV,' belongs to ',Vnode_belongs_to(1,iV),' elts:  ',Vnode_belongs_to(2:,iV)
end do
do iV=1,NW
write(2345,'(a,i6,a,i2,a,9i6)') 'W node',iW,' belongs to ',Wnode_belongs_to(1,iW),' elts:  ',Wnode_belongs_to(2:,iW)
end do
do iP=1,NP
write(2345,'(a,i6,a,i2,a,9i6)') 'P node',iP,' belongs to ',pnode_belongs_to(1,iP),' elts:  ',pnode_belongs_to(2:,iP)
end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'compute_belongs:',elapsed,' s                |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
