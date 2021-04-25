!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_solution

use global_parameters
use global_measurements
use structures
use timing

implicit none

integer i,k
real(8) uth,vth,wth,pth,dum,rq,sq,tq,rho(4)
real(8) dNdx(mV),dNdy(mV),dNdz(mV),div_v,jcob

logical, parameter :: output_boundary_indicators=.false.

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{output\_solution}
!@@ This subroutine generates the {\filenamefont solution.vtu} in the {\foldernamefont OUTPUT}
!@@ folder. It also generates the basic ascii file {\filenamefont solution.ascii}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

open(unit=123,file='OUTPUT/solution.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',ncorners*nel,'" NumberOfCells="',nel,'">'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,'(3es12.4)') mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,'(a)') '<CellData Scalars="scalars">'
!-----
if (output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 1" Format="ascii">'
do iel=1,nel
if (mesh(iel)%bnd1) then
   write(123,'(i1)') 1
else
   write(123,'(i1)') 0
end if
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 2" Format="ascii">'
do iel=1,nel
if (mesh(iel)%bnd2) then
   write(123,'(i1)') 1
else
   write(123,'(i1)') 0
end if
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 3" Format="ascii">'
do iel=1,nel
if (mesh(iel)%bnd3) then
   write(123,'(i1)') 1
else
   write(123,'(i1)') 0
end if
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 4" Format="ascii">'
do iel=1,nel
if (mesh(iel)%bnd4) then
   write(123,'(i1)') 1
else
   write(123,'(i1)') 0
end if
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: 5" Format="ascii">'
do iel=1,nel
if (mesh(iel)%bnd5) then
   write(123,'(i1)') 1
else
   write(123,'(i1)') 0
end if
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: 6" Format="ascii">'
do iel=1,nel
if (mesh(iel)%bnd6) then
   write(123,'(i1)') 1
else
   write(123,'(i1)') 0
end if
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Name="hx,hy,hz" Format="ascii">'
do iel=1,nel
   write(123,'(3es12.4)') mesh(iel)%hx,mesh(iel)%hy,mesh(iel)%hz
end do
write(123,'(a)') '</DataArray>'
!-----
write(123,'(a)') '<DataArray type="Float32" Name="vol" Format="ascii">'
do iel=1,nel
   write(123,'(es12.4)') mesh(iel)%vol
end do
write(123,'(a)') '</DataArray>'
!-----
if (use_swarm) then
write(123,'(a)') '<DataArray type="Float32" Name="nmarker" Format="ascii">'
do iel=1,nel
   write(123,'(i4)') mesh(iel)%nmarker
end do
write(123,'(a)') '</DataArray>'
end if
!-----
if (use_swarm) then
write(123,*) '<DataArray type="Float32" Name="least squares: a_rho" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%a_rho
end do
write(123,*) '</DataArray>'
end if
!-----
if (use_swarm) then
write(123,*) '<DataArray type="Float32" Name="least squares: b_rho" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%b_rho
end do
write(123,*) '</DataArray>'
end if
!-----
if (use_swarm) then
write(123,*) '<DataArray type="Float32" Name="least squares: c_rho" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%c_rho
end do
write(123,*) '</DataArray>'
end if
!-----
if (use_swarm) then
write(123,*) '<DataArray type="Float32" Name="least squares: a_eta" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%a_eta
end do
write(123,*) '</DataArray>'
end if
!-----
if (use_swarm) then
write(123,*) '<DataArray type="Float32" Name="least squares: b_eta" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%b_eta
end do
write(123,*) '</DataArray>'
end if
!-----
if (use_swarm) then
write(123,*) '<DataArray type="Float32" Name="least squares: c_eta" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%c_eta
end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="div(v)" Format="ascii">'
do iel=1,nel
   rq=0d0
   sq=0d0
   tq=0d0
   if (ndim==2) then
      call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
      div_v=sum(dNdx(1:mV)*mesh(iel)%u(1:mV))&
           +sum(dNdy(1:mV)*mesh(iel)%v(1:mV))
   else
      call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNdx,dNdy,dNdz,jcob)
      div_v=sum(dNdx(1:mV)*mesh(iel)%u(1:mV))&
           +sum(dNdy(1:mV)*mesh(iel)%v(1:mV))&
           +sum(dNdz(1:mV)*mesh(iel)%w(1:mV))
   end if
   write(123,*) div_v
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '</CellData>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<PointData Scalars="scalars">'
!-----
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,'(3es12.4)') mesh(iel)%u(k),mesh(iel)%v(k),mesh(iel)%w(k)
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="pressure (p)" Format="ascii">'
do iel=1,nel
   if (pair=='q1p0') then
      do k=1,ncorners
         write(123,'(es12.4)') mesh(iel)%p(1)
      end do
   else
      do k=1,ncorners
         write(123,'(es12.4)') mesh(iel)%p(k)
      end do
   end if
end do
write(123,*) '</DataArray>'
!-----
if (use_T) then 
write(123,*) '<DataArray type="Float32" Name="temperature (T)" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,'(es12.4)') mesh(iel)%T(k)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="pressure (q)" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,'(es12.4)') mesh(iel)%q(k)
   end do
end do
write(123,*) '</DataArray>'

!-----
if (output_boundary_indicators .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: 5" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%bnd5_node(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: 6" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%bnd6_node(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 1" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%bnd1_node(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 2" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%bnd2_node(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 3" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%bnd3_node(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 4" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%bnd4_node(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="fix_u" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%fix_u(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="fix_v" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%fix_v(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
!-----
if (ndim==3) then
write(123,*) '<DataArray type="Float32" Name="fix_w" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%fix_w(k)) then
         write(123,'(i1)') 1
      else
         write(123,'(i1)') 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (use_swarm) then
write(123,*) '<DataArray type="Float32" Name="rho (LS)" Format="ascii">'
   do iel=1,nel
      do k=1,ncorners
         write(123,'(es12.4)') mesh(iel)%a_rho+&
                      mesh(iel)%b_rho*(mesh(iel)%xV(k)-mesh(iel)%xc)+&
                      mesh(iel)%c_rho*(mesh(iel)%yV(k)-mesh(iel)%yc)+&
                      mesh(iel)%d_rho*(mesh(iel)%zV(k)-mesh(iel)%zc)
         end do
   end do
write(123,*) '</DataArray>'
end if
!-----
write(123,'(a)') '<DataArray type="Float32" Name="rho (avrg)" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,'(f12.5)') mesh(iel)%rho_avrg 
   end do
end do
write(123,'(a)') '</DataArray>'
!-----
if (use_swarm) then
write(123,*) '<DataArray type="Float32" Name="eta (LS)" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,'(es12.4)') mesh(iel)%a_eta+&
                   mesh(iel)%b_eta*(mesh(iel)%xV(k)-mesh(iel)%xc)+&
                   mesh(iel)%c_eta*(mesh(iel)%yV(k)-mesh(iel)%yc)+&
                   mesh(iel)%d_eta*(mesh(iel)%zV(k)-mesh(iel)%zc)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="eta (avrg)" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,'(es12.4)') mesh(iel)%eta_avrg 
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity (analytical)" Format="ascii">'
do iel=1,nel
   do k=1,mV
      call analytical_solution(mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),&
                               uth,vth,wth,dum,dum,dum,dum,dum,dum,dum,dum)
      write(123,'(3es12.5)') uth,vth,wth
   end do
end do
write(123,*) '</DataArray>'
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity (error)" Format="ascii">'
do iel=1,nel
   do k=1,mV
      call analytical_solution(mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),&
                               uth,vth,wth,dum,dum,dum,dum,dum,dum,dum,dum)
      write(123,'(3es12.5)') mesh(iel)%u(k)-uth,mesh(iel)%v(k)-vth,mesh(iel)%w(k)-wth
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="pressure q (analytical)" Format="ascii">'
do iel=1,nel
   do k=1,mV
      call analytical_solution(mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),&
                               uth,vth,wth,pth,dum,dum,dum,dum,dum,dum,dum)
      write(123,'(es12.4)') pth
   end do
end do
write(123,*) '</DataArray>'
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="pressure q (error)" Format="ascii">'
do iel=1,nel
   do k=1,mV
      call analytical_solution(mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),&
                               uth,vth,wth,pth,dum,dum,dum,dum,dum,dum,dum)
      write(123,'(es12.4)') mesh(iel)%q(k)-pth
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="exx" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,*) mesh(iel)%exx(k)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="eyy" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(es12.5)') mesh(iel)%eyy(k)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="ezz" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(es12.5)') mesh(iel)%ezz(k)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="exy" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(es12.5)') mesh(iel)%exy(k)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="exz" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(es12.5)') mesh(iel)%exz(k)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="eyz" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(es12.5)') mesh(iel)%eyz(k)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '</PointData>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<Cells>'
!-----
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
   write(123,*) ( (iel-1)*ncorners+i-1,i=1,ncorners) 
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*ncorners,iel=1,nel)
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
if (ndim==2) write(123,*) (9,iel=1,nel)
if (ndim==3) write(123,*) (12,iel=1,nel)
write(123,*) '</DataArray>'
!-----
write(123,*) '</Cells>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

!----------------------------------------------------------

open(unit=123,file="OUTPUT/solution.ascii",action="write")
write(123,*) '#1,2,3  x,y,z '
write(123,*) '#4,5,6  u,v,w '
write(123,*) '#7      q '
write(123,*) '#8      T '

do iel=1,nel 
   do k=1,ncorners
      write(123,'(8es12.4)') mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),&
                             mesh(iel)%u(k),mesh(iel)%v(k),mesh(iel)%w(k),&
                             mesh(iel)%q(k),mesh(iel)%T(k)
   end do
end do
close(123)

!----------------------------------------------------------

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
   write(123,'(a,es12.3,a,es12.3,a,es12.3,a,es12.3,a)') 'z=np.array([[',mesh(iel)%q(1),',',mesh(iel)%q(2),'],[',mesh(iel)%q(4),',',mesh(iel)%q(3),']])'
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
   write(123,'(a,es12.3,a,es12.3,a,es12.3,a,es12.3,a)' ) 'z=np.array([[',mesh(iel)%u(1),',',mesh(iel)%u(2),'],[',mesh(iel)%u(4),',',mesh(iel)%u(3),']])'
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
   write(123,'(a,es12.3,a,es12.3,a,es12.3,a,es12.3,a)' ) 'z=np.array([[',mesh(iel)%v(1),',',mesh(iel)%v(2),'],[',mesh(iel)%v(4),',',mesh(iel)%v(3),']])'
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
   write(123,'(a,es12.3,a,es12.3,a)') 'ax.pcolormesh(x,y,z,shading="gouraud",vmin=',rhoq_min,',vmax=',rhoq_max,',edgecolors="k",linewidths=4,cmap=plt.cm.coolwarm)' 
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

write(*,'(a,f6.2,a)') '     >> output_solution                  ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
