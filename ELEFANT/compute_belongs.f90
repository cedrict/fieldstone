!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_belongs

use module_parameters, only: nel,iproc,NV,NP,iel,debug,mV,mP
use module_mesh 
use module_arrays, only: vnode_belongs_to,pnode_belongs_to
use module_timing

implicit none

integer :: inode,i,iV,iP

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{compute\_belongs}
!@@ This subroutine allocates and fills the {\sl vnode\_belongs\_to} and {\sl pnode\_belongs\_to} 
!@@ arrays. For a given node {\sl ip},
!@@ {\sl vnode\_belongs\_to(1,ip)} is the number of elements that {\sl ip} belongs to.
!@@ Furthermore, {\sl vnode\_belongs\_to(2:9,ip)} is the actual list of elements.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

allocate(vnode_belongs_to(9,NV)) !9 = max nb of elements a node can belong to

vnode_belongs_to=0
do iel=1,nel
   !print *,'elt:',iel
   do i=1,mV
      inode=mesh(iel)%iconV(i)
      !print *,'->',inode
      vnode_belongs_to(1,inode)=vnode_belongs_to(1,inode)+1
      if (vnode_belongs_to(1,inode)>9) then
         print *, 'compute_belongs: vnode_belongs_to array too small'
         stop
      end if
      vnode_belongs_to(1+vnode_belongs_to(1,inode),inode)=iel
   end do
end do

allocate(pnode_belongs_to(9,NV)) !9 = max nb of elements a node can belong to

pnode_belongs_to=0
do iel=1,nel
   !print *,'elt:',iel
   do i=1,mP
      inode=mesh(iel)%iconP(i)
      !print *,'->',inode
      pnode_belongs_to(1,inode)=pnode_belongs_to(1,inode)+1
      if (pnode_belongs_to(1,inode)>9) then
         print *, 'compute_belongs: pnode_belongs_to array too small'
         stop
      end if
      pnode_belongs_to(1+pnode_belongs_to(1,inode),inode)=iel
   end do
end do

!-----------------------------------------------------------------------------!

if (debug) then
write(2345,'(a)') limit//'compute_belongs'//limit
do iV=1,NV
write(2345,'(a,i6,a,i2,a,9i6)') 'V node',iV,' belongs to ',vnode_belongs_to(1,iV),' elts:  ',vnode_belongs_to(2:,iV)
end do
do iP=1,NP
write(2345,'(a,i6,a,i2,a,9i6)') 'P node',iP,' belongs to ',pnode_belongs_to(1,iP),' elts:  ',pnode_belongs_to(2:,iP)
end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'compute_belongs (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
