!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine assign_values_to_qpoints

use module_parameters 
use module_mesh
use module_swarm
use module_statistics
use module_constants
use module_timing

implicit none

integer i,im,iq,k
integer(1) idummy
real(8) x(1000),y(1000),z(1000),rho(1000),eta(1000)
real(8) pm,Tm,exxm,eyym,ezzm,exym,exzm,eyzm,NNNV(mV),NNNT(mT),NNNP(mP)
real(8) exxq,eyyq,ezzq,exyq,exzq,eyzq

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{assign\_values\_to\_qpoints.f90}
!@@ This subroutine assigns 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

etaq_min=+1d50 ; etaq_max=-1d50 
rhoq_min=+1d50 ; rhoq_max=-1d50
hcapaq_min=+1d50 ; hcapaq_max=-1d50 
hcondq_min=+1d50 ; hcondq_max=-1d50 

if (use_swarm) then 

   do iel=1,nel
  
      do i=1,mesh(iel)%nmarker

         im=mesh(iel)%list_of_markers(i)

         call NNN(swarm(im)%r,swarm(im)%s,swarm(im)%t,NNNP(1:mP),mP,ndim,spaceP)
         pm=sum(NNNP(1:mP)*mesh(iel)%p(1:mP))

         call NNN(swarm(im)%r,swarm(im)%s,swarm(im)%t,NNNT(1:mT),mT,ndim,spaceT)
         Tm=sum(NNNT(1:mT)*mesh(iel)%T(1:mT))

         call NNN(swarm(im)%r,swarm(im)%s,swarm(im)%t,NNNV(1:mV),mV,ndim,spaceV)
         !exxm=sum(NNNV(1:mV)*mesh(iel)%exx(1:mV))
         !eyym=sum(NNNV(1:mV)*mesh(iel)%eyy(1:mV))
         !ezzm=sum(NNNV(1:mV)*mesh(iel)%ezz(1:mV))
         !exym=sum(NNNV(1:mV)*mesh(iel)%exy(1:mV))
         !exzm=sum(NNNV(1:mV)*mesh(iel)%exz(1:mV))
         !eyzm=sum(NNNV(1:mV)*mesh(iel)%eyz(1:mV))

         call experiment_material_model(swarm(im)%x,swarm(im)%y,swarm(im)%z,&
                                        pm,Tm,&
                                        exxm,eyym,ezzm,exym,exzm,eyzm,&
                                        swarm(im)%mat,one,&
                                        swarm(im)%eta,&
                                        swarm(im)%rho,&
                                        swarm(im)%hcond,&
                                        swarm(im)%hcapa,&
                                        swarm(im)%hprod)

         x(i)=swarm(im)%x-mesh(iel)%xc
         y(i)=swarm(im)%y-mesh(iel)%yc
         z(i)=swarm(im)%z-mesh(iel)%zc
         rho(i)=swarm(im)%rho
         eta(i)=swarm(im)%eta

      end do 

      if (ndim==2) then 
         call compute_abcd_2D(mesh(iel)%nmarker,x,y,rho,eta,&
                           mesh(iel)%a_rho,mesh(iel)%b_rho,mesh(iel)%c_rho,mesh(iel)%d_rho,&
                           mesh(iel)%a_eta,mesh(iel)%b_eta,mesh(iel)%c_eta,mesh(iel)%d_eta)
      else
         call compute_abcd_3D(mesh(iel)%nmarker,x,y,z,rho,eta,&
                           mesh(iel)%a_rho,mesh(iel)%b_rho,mesh(iel)%c_rho,mesh(iel)%d_rho,&
                           mesh(iel)%a_eta,mesh(iel)%b_eta,mesh(iel)%c_eta,mesh(iel)%d_eta)
      end if

      !print *,mesh(iel)%a_rho,mesh(iel)%b_rho,mesh(iel)%c_rho,mesh(iel)%d_rho
      !print *,mesh(iel)%a_eta,mesh(iel)%b_eta,mesh(iel)%c_eta,mesh(iel)%d_eta

      ! project values on quadrature points

      do iq=1,nqel
         mesh(iel)%etaq(iq)=mesh(iel)%a_eta+&
                            mesh(iel)%b_eta*(mesh(iel)%xq(iq)-mesh(iel)%xc)+&
                            mesh(iel)%c_eta*(mesh(iel)%yq(iq)-mesh(iel)%yc)+&
                            mesh(iel)%d_eta*(mesh(iel)%zq(iq)-mesh(iel)%zc)
         mesh(iel)%rhoq(iq)=mesh(iel)%a_rho+&
                            mesh(iel)%b_rho*(mesh(iel)%xq(iq)-mesh(iel)%xc)+&
                            mesh(iel)%c_rho*(mesh(iel)%yq(iq)-mesh(iel)%yc)+&
                            mesh(iel)%d_rho*(mesh(iel)%zq(iq)-mesh(iel)%zc)
      end do

      do k=1,mV
         mesh(iel)%rho(k)=mesh(iel)%a_rho+&
                          mesh(iel)%b_rho*(mesh(iel)%xV(iq)-mesh(iel)%xc)+&
                          mesh(iel)%c_rho*(mesh(iel)%yV(iq)-mesh(iel)%yc)+&
                          mesh(iel)%d_rho*(mesh(iel)%zV(iq)-mesh(iel)%zc)
         mesh(iel)%eta(k)=mesh(iel)%a_eta+&
                          mesh(iel)%b_eta*(mesh(iel)%xV(iq)-mesh(iel)%xc)+&
                          mesh(iel)%c_eta*(mesh(iel)%yV(iq)-mesh(iel)%yc)+&
                          mesh(iel)%d_eta*(mesh(iel)%zV(iq)-mesh(iel)%zc)
      end do

      etaq_min=min(minval(mesh(iel)%etaq(:)),etaq_min)
      etaq_max=max(maxval(mesh(iel)%etaq(:)),etaq_max)
      rhoq_min=min(minval(mesh(iel)%rhoq(:)),rhoq_min)
      rhoq_max=max(maxval(mesh(iel)%rhoq(:)),rhoq_max)

   end do ! iel

else ! use_swarm=F

   do iel=1,nel

      do iq=1,nqel

         call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNP(1:mP),mP,ndim,spaceP)
         mesh(iel)%pq(iq)=sum(NNNP(1:mP)*mesh(iel)%p(1:mP))

         if (use_T) then
            call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNT(1:mT),mT,ndim,spaceT)
            mesh(iel)%tempq(iq)=sum(NNNT(1:mT)*mesh(iel)%T(1:mT))
         else
            mesh(iel)%tempq(iq)=0
         end if

         call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNV(1:mV),mV,ndim,spaceV)
         !exxq=sum(NNNV(1:mV)*mesh(iel)%exx(1:mV))
         !eyyq=sum(NNNV(1:mV)*mesh(iel)%eyy(1:mV))
         !ezzq=sum(NNNV(1:mV)*mesh(iel)%ezz(1:mV))
         !exyq=sum(NNNV(1:mV)*mesh(iel)%exy(1:mV))
         !exzq=sum(NNNV(1:mV)*mesh(iel)%exz(1:mV))
         !eyzq=sum(NNNV(1:mV)*mesh(iel)%eyz(1:mV))

         call experiment_material_model(mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                                        mesh(iel)%pq(iq),mesh(iel)%tempq(iq),&
                                        exxq,eyyq,ezzq,exyq,exzq,eyzq,&
                             idummy,one,&
                             mesh(iel)%etaq(iq),&
                             mesh(iel)%rhoq(iq),&
                             mesh(iel)%hcondq(iq),&
                             mesh(iel)%hcapaq(iq),&
                             mesh(iel)%hprodq(iq))

         !print *,mesh(iel)%etaq(iq),mesh(iel)%rhoq(iq)
  
      end do !nqel


      if (ndim==2) then 
         call compute_abcd_2D(nqel,mesh(iel)%xq,mesh(iel)%yq,mesh(iel)%rhoq,mesh(iel)%etaq,&
                           mesh(iel)%a_rho,mesh(iel)%b_rho,mesh(iel)%c_rho,mesh(iel)%d_rho,&
                           mesh(iel)%a_eta,mesh(iel)%b_eta,mesh(iel)%c_eta,mesh(iel)%d_eta)
      else
         call compute_abcd_3D(nqel,mesh(iel)%xq,mesh(iel)%yq,mesh(iel)%zq,mesh(iel)%rhoq,mesh(iel)%etaq,&
                           mesh(iel)%a_rho,mesh(iel)%b_rho,mesh(iel)%c_rho,mesh(iel)%d_rho,&
                           mesh(iel)%a_eta,mesh(iel)%b_eta,mesh(iel)%c_eta,mesh(iel)%d_eta)
      end if






      do k=1,mV
         mesh(iel)%rho(k)=mesh(iel)%a_rho+&
                          mesh(iel)%b_rho*(mesh(iel)%xV(k)-mesh(iel)%xc)+&
                          mesh(iel)%c_rho*(mesh(iel)%yV(k)-mesh(iel)%yc)+&
                          mesh(iel)%d_rho*(mesh(iel)%zV(k)-mesh(iel)%zc)
         mesh(iel)%eta(k)=mesh(iel)%a_eta+&
                          mesh(iel)%b_eta*(mesh(iel)%xV(k)-mesh(iel)%xc)+&
                          mesh(iel)%c_eta*(mesh(iel)%yV(k)-mesh(iel)%yc)+&
                          mesh(iel)%d_eta*(mesh(iel)%zV(k)-mesh(iel)%zc)
      end do


      etaq_min=min(minval(mesh(iel)%etaq(:)),etaq_min)
      etaq_max=max(maxval(mesh(iel)%etaq(:)),etaq_max)
      rhoq_min=min(minval(mesh(iel)%rhoq(:)),rhoq_min)
      rhoq_max=max(maxval(mesh(iel)%rhoq(:)),rhoq_max)
      hcapaq_min=min(minval(mesh(iel)%hcapaq(:)),hcapaq_min)
      hcapaq_max=max(maxval(mesh(iel)%hcapaq(:)),hcapaq_max)
      hcondq_min=min(minval(mesh(iel)%hcondq(:)),hcondq_min)
      hcondq_max=max(maxval(mesh(iel)%hcondq(:)),hcondq_max)

   end do !nel

end if

write(*,'(a,2es10.3)') shift//'rhoq (m/M):',rhoq_min,rhoq_max
write(*,'(a,2es10.3)') shift//'etaq (m/M):',etaq_min,etaq_max

write(1240,'(4es12.4)') rhoq_min,rhoq_max,etaq_min,etaq_max

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'assign_vals_to_qpoints (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
