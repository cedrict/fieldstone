module structures
implicit none

type element
     integer :: iconV(10),iconT(8),iconP(8)
     integer :: ielx,iely,ielz
     integer :: nmarker
     integer :: list_of_markers(100)
     real(8) :: xV(10),yV(10),zV(10)
     real(8) :: xT(8),yT(8),zT(8)
     real(8) :: xP(8),yP(8),zP(8)
     real(8) :: xc,yc,zc
     real(8) :: hx,hy,hz
     real(8) :: u(10),v(10),w(10),p(8),T(8),q(8)
     real(8) :: a_eta,b_eta,c_eta,d_eta
     real(8) :: a_rho,b_rho,c_rho,d_rho
     logical(1) :: left,right,top,bottom,front,back
     logical(1) :: left_node(8),right_node(8),top_node(8),bottom_node(8),front_node(8),back_node(8)
     logical(1) :: fix_u(8),fix_v(8),fix_w(8),fix_T(8)
     real(8), dimension(:), allocatable :: xq,yq,zq,weightq,rq,sq,tq,gxq,gyq,gzq,pq,thetaq
     real(8), dimension(:), allocatable :: etaq,rhoq,hcondq,hcapaq,hprodq,JxWq
     real(8) :: exx(8),eyy(8),ezz(8),exy(8),exz(8),eyz(8) 
end type element
type(element), dimension(:), allocatable :: mesh

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

type material    
   real(8) rho0
   real(8) eta0 
   real(8) c,c_sw,phi,phi_sw           !
   real(8) alpha,T0                    ! thermal expansion coeff.
   real(8) hcapa                       ! heat capacity
   real(8) hcond                       ! heat conductivity
   real(8) hprod                       ! heat production coefficient
   real(8) A_diff,Q_diff,V_diff,f_diff !
   real(8) n_disl,A_disl,Q_disl,V_disl,f_disl !
   real(8) n_prls,A_prls,Q_prls,V_prls,f_prls
end type material 
type(material), dimension(:), allocatable :: mat

end module
