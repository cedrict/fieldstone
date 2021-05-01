!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_solution_python

!use global_parameters
!use structures
!use constants
use timing

implicit none


!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{output\_solution\_python}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!


if (ndim==2 .and. pair=='q1p0') then 

open(unit=123,file="OUTPUT/solution_p.py",action="write")
write(123,'(a)') '#!/usr/bin/env python'
write(123,'(a)') 'import matplotlib.pyplot as plt'
write(123,'(a)') 'import numpy as np'

write(123,'(a)') 'fig, ax = plt.subplots()'
write(123,'(a)') 'ax.set_aspect("equal")'
write(123,'(a)') 'ax.set_title("nodal pressure q")'
do iel=1,nel
   write(123,'(a,es12.3,a,e12.3,a)') 'x=np.array([',mesh(iel)%xV(1),',',mesh(iel)%xV(3),'])'
   write(123,'(a,es12.3,a,e12.3,a)') 'y=np.array([',mesh(iel)%yV(1),',',mesh(iel)%yV(3),'])'
   write(123,'(a,es12.3,a,es12.3,a,es12.3,a,es12.3,a)') 'z=np.array([[',&
               mesh(iel)%q(1),',',mesh(iel)%q(2),'],[',mesh(iel)%q(4),',',mesh(iel)%q(3),']])'
   write(123,'(a,es12.3,a,es12.3,a)') 'ax.pcolormesh(x,y,z,shading="gouraud",vmin=',q_min,',vmax=',q_max,')' 
end do
write(123,'(a)') 'plt.savefig("solution_p.pdf")'
close(123)

open(unit=123,file="OUTPUT/solution_u.py",action="write")
write(123,'(a)') '#!/usr/bin/env python'
write(123,'(a)') 'import matplotlib.pyplot as plt'
write(123,'(a)') 'import numpy as np'
write(123,'(a)') 'fig, ax = plt.subplots()'
write(123,'(a)') 'ax.set_aspect("equal")'
write(123,'(a)') 'ax.set_title("velocicty v_x")'
do iel=1,nel
   write(123,'(a,es12.3,a,e12.3,a)') 'x=np.array([',mesh(iel)%xV(1),',',mesh(iel)%xV(3),'])'
   write(123,'(a,es12.3,a,e12.3,a)') 'y=np.array([',mesh(iel)%yV(1),',',mesh(iel)%yV(3),'])'
   write(123,'(a,es12.3,a,es12.3,a,es12.3,a,es12.3,a)' ) 'z=np.array([[',&
               mesh(iel)%u(1),',',mesh(iel)%u(2),'],[',mesh(iel)%u(4),',',mesh(iel)%u(3),']])'
   write(123,'(a,es12.3,a,es12.3,a)') 'ax.pcolormesh(x,y,z,shading="gouraud",vmin=',u_min,',vmax=',u_max,')' 
end do
write(123,'(a)') 'plt.savefig("solution_u.pdf")'
close(123)

open(unit=123,file="OUTPUT/solution_v.py",action="write")
write(123,'(a)') '#!/usr/bin/env python'
write(123,'(a)') 'import matplotlib.pyplot as plt'
write(123,'(a)') 'import numpy as np'
write(123,'(a)') 'fig, ax = plt.subplots()'
write(123,'(a)') 'ax.set_aspect("equal")'
write(123,'(a)') 'ax.set_title("velocity v_y")'
do iel=1,nel
   write(123,'(a,es12.3,a,e12.3,a)') 'x=np.array([',mesh(iel)%xV(1),',',mesh(iel)%xV(3),'])'
   write(123,'(a,es12.3,a,e12.3,a)') 'y=np.array([',mesh(iel)%yV(1),',',mesh(iel)%yV(3),'])'
   write(123,'(a,es12.3,a,es12.3,a,es12.3,a,es12.3,a)' ) 'z=np.array([[',&
               mesh(iel)%v(1),',',mesh(iel)%v(2),'],[',mesh(iel)%v(4),',',mesh(iel)%v(3),']])'
   write(123,'(a,es12.3,a,es12.3,a)') 'ax.pcolormesh(x,y,z,shading="gouraud",vmin=',v_min,',vmax=',v_max,')' 
end do
write(123,'(a)') 'plt.savefig("solution_v.pdf")'
close(123)

open(unit=123,file="OUTPUT/solution_rho.py",action="write")
write(123,'(a)') '#!/usr/bin/env python'
write(123,'(a)') 'import matplotlib.pyplot as plt'
write(123,'(a)') 'import numpy as np'
write(123,'(a)') 'fig, ax = plt.subplots()'
write(123,'(a)') 'ax.set_aspect("equal")'
write(123,'(a)') 'ax.set_title("density")'
write(123,'(a)') 'ax.set_xlabel("x")'
write(123,'(a)') 'ax.set_ylabel("y")'
!write(123,'(a)') 'fig.colorbar(im,ax=ax)' WTF 

do iel=1,nel
   write(123,'(a,es12.3,a,e12.3,a)') 'x=np.array([',mesh(iel)%xV(1),',',mesh(iel)%xV(3),'])'
   write(123,'(a,es12.3,a,e12.3,a)') 'y=np.array([',mesh(iel)%yV(1),',',mesh(iel)%yV(3),'])'
   do i=1,ncorners
      rho(i)=mesh(iel)%a_rho+mesh(iel)%b_rho*(mesh(iel)%xV(k)-mesh(iel)%xc)+&
                             mesh(iel)%c_rho*(mesh(iel)%yV(k)-mesh(iel)%yc)
   end do
   write(123,'(a,es12.3,a,es12.3,a,es12.3,a,es12.3,a)' ) 'z=np.array([[',rho(1),',',rho(2),'],[',rho(4),',',rho(3),']])'
   write(123,'(a,es12.3,a,es12.3,a)') 'ax.pcolormesh(x,y,z,shading="gouraud",vmin=',&
                 rhoq_min,',vmax=',rhoq_max,',edgecolors="k",linewidths=4,cmap=plt.cm.coolwarm)' 
end do
write(123,'(a)') 'plt.savefig("solution_rho.pdf")'
close(123)


call execute_command_line('python3 OUTPUT/solution_u.py',wait=.false.)
call execute_command_line('python3 OUTPUT/solution_v.py',wait=.false.)
call execute_command_line('python3 OUTPUT/solution_p.py',wait=.false.)
call execute_command_line('python3 OUTPUT/solution_rho.py',wait=.false.)

end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'output_solution_python (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
