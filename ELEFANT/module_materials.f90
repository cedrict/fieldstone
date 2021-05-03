module module_materials

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

type(material), dimension(:), allocatable :: materials

end module
