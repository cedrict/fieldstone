module module_statistics
implicit none
real(8) :: vrms,   vrms_test
real(8) :: avrg_u, avrg_u_test
real(8) :: avrg_v, avrg_v_test
real(8) :: avrg_w, avrg_w_test
real(8) :: avrg_p, avrg_p_test
real(8) :: avrg_T, avrg_T_test
real(8) :: avrg_q, avrg_q_test
real(8) :: volume, volume_test
real(8) :: etaq_min,etaq_max
real(8) :: rhoq_min,rhoq_max
real(8) :: hcapaq_min,hcapaq_max
real(8) :: hcondq_min,hcondq_max
real(8) :: u_min,u_max
real(8) :: v_min,v_max
real(8) :: w_min,w_max
real(8) :: p_min,p_max
real(8) :: T_min,T_max
real(8) :: vol_min,vol_max
real(8) :: errv,errp,errq,errT
end module
