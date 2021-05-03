module module_swarm

type marker
     real(8)    :: x,y,z
     real(8)    :: r,s,t 
     real(4)    :: strain
     integer(1) :: mat 
     real(4)    :: paint
     logical(1) :: active
     integer    :: iel 
     real(8)    :: eta
     real(8)    :: rho 
     real(8)    :: hcond 
     real(8)    :: hcapa
     real(8)    :: hprod
end type

type(marker), dimension(:), allocatable :: swarm

end module
