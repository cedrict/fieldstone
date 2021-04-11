subroutine compute_belongs

use global_parameters
use structures

implicit none

integer nnelt

nnelt=11+1

allocate(vdof_belongs_to(nnelt,NV))

vdof_belongs_to=0

do iel=1,nel
   print *,'elt:',iel
   do i=1,mV
      inode=mesh(iel)%iconV(i)
      print *,'->',inode
      vdof_belongs_to(1,inode)=vdof_belongs_to(1,inode)+1
      if (vdof_belongs_to(1,inode)>nnelt) then
         print *, 'compute_belongs: pb1 with inode= ',inode
         stop
      end if
      vdof_belongs_to(1+vdof_belongs_to(1,inode),inode)=iel
   end do
end do








end subroutine
