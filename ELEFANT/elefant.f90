program elefant

use global_parameters
use structures

implicit none


ndim=2
Lx=1
Ly=1
nelx=16
nely=15
geometry='cartesian2D'
pair='q1p0'
nmarker_per_dim=8
init_marker_random=.true.

!--------------------------------------

if (pair=='q1p0') then
   mV=2**ndim
   mP=1
   mT=2**ndim
   nq_per_dim=2
   nqel=nq_per_dim**ndim
   ndofV=ndim
   if (ndim==2) then
      nel=nelx*nely
      NV=(nelx+1)*(nely+1)
      NT=(nelx+1)*(nely+1)
      NP=nel
   else
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)
      NT=(nelx+1)*(nely+1)*(nelz+1)
      NP=nel
   end if
end if

if (pair=='q1q1') then
   mV=2**ndim+1
   mP=2**ndim
   mT=2**ndim
   nq_per_dim=2
   nqel=nq_per_dim**ndim
   ndofV=ndim
   if (ndim==2) then
      nel=nelx*nely
      NV=(nelx+1)*(nely+1)+nel
      NT=(nelx+1)*(nely+1)
      NP=(nelx+1)*(nely+1)
   else
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)+2*nel
      NT=(nelx+1)*(nely+1)*(nelz+1)
      NP=(nelx+1)*(nely+1)*(nelz+1)
   end if
end if

NfemV=NV*ndofV
NfemP=NP
NfemT=NT
Nq=nqel*nel



!----------------------------
write(*,*) 'Lx,Ly=',Lx,Ly
write(*,*) 'nelx,nely',nelx,nely
write(*,*) 'nel',nel
write(*,*) 'pair',pair
write(*,*) 'geometry=',geometry
write(*,*) 'nqel=',nqel
write(*,*) 'NfemV=',NfemV
write(*,*) 'NfemP=',NfemP
write(*,*) 'NfemT=',NfemT
write(*,*) 'Nq=',Nq
!----------------------------








!-------------------------------------------------


select case (geometry)
case('cartesian2D'); call setup_cartesian2D
end select

call markers_setup
!call material_layout
!call material_paint
call export_swarm

call output_solution
call output_qpoints

end program
