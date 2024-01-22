!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine assign_values_to_qpoints

use module_parameters, only: mP,mT,ndim,iel,use_T,nqel,iproc,nel,&
                             spacePressure,spaceTemperature,use_swarm
use module_arrays, only: dNNNUdx,dNNNUdy,dNNNUdz,dNNNVdx,dNNNVdy,dNNNVdz,dNNNWdx,dNNNWdy,dNNNWdz,&
                         NNNT,NNNP
use module_mesh
use module_swarm
use module_statistics
use module_constants
use module_timing

implicit none

integer i,im,iq
integer(1) idummy
real(8) x(1000),y(1000),z(1000),rho(1000),eta(1000)
real(8) pm,Tm,exxm,eyym,ezzm,exym,exzm,eyzm
real(8) exxq,eyyq,ezzq,exyq,exzq,eyzq,jcob
integer, parameter :: caller_id01=501
integer, parameter :: caller_id02=502

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

         call NNN(swarm(im)%r,swarm(im)%s,swarm(im)%t,NNNP,mP,ndim,spacePressure,caller_id01)
         pm=sum(NNNP*mesh(iel)%p)

         if (use_T) then
            call NNN(swarm(im)%r,swarm(im)%s,swarm(im)%t,NNNT,mT,ndim,spaceTemperature,caller_id02)
            Tm=sum(NNNT*mesh(iel)%T)
         else
            Tm=0
         end if

         call compute_dNdx_dNdy_dNdz(swarm(im)%r,swarm(im)%s,swarm(im)%t,&
                                     dNNNUdx,dNNNUdy,dNNNUdz,&
                                     dNNNVdx,dNNNVdy,dNNNVdz,&
                                     dNNNWdx,dNNNWdy,dNNNWdz,jcob)
         exxm=sum(dNNNUdx*mesh(iel)%u)
         eyym=sum(dNNNVdy*mesh(iel)%v)
         ezzm=sum(dNNNWdz*mesh(iel)%w)
         exym=0.5d0*sum(dNNNUdy*mesh(iel)%u + dNNNVdx*mesh(iel)%v)
         exzm=0.5d0*sum(dNNNUdz*mesh(iel)%u + dNNNWdx*mesh(iel)%w)
         eyzm=0.5d0*sum(dNNNVdz*mesh(iel)%v + dNNNWdy*mesh(iel)%w)

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

      !!!!!!WHY AM I USING XC HERE>?!?!

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

      etaq_min=min(minval(mesh(iel)%etaq(:)),etaq_min)
      etaq_max=max(maxval(mesh(iel)%etaq(:)),etaq_max)
      rhoq_min=min(minval(mesh(iel)%rhoq(:)),rhoq_min)
      rhoq_max=max(maxval(mesh(iel)%rhoq(:)),rhoq_max)

   end do ! iel

else ! use_swarm=F

   do iel=1,nel
      do iq=1,nqel

         call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNP,mP,ndim,spacePressure,caller_id01)
         mesh(iel)%pq(iq)=sum(NNNP*mesh(iel)%p)

         if (use_T) then
            call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNT,mT,ndim,spaceTemperature,caller_id02)
            mesh(iel)%tempq(iq)=sum(NNNT*mesh(iel)%T)
         else
            mesh(iel)%tempq(iq)=0
         end if

         if (ndim==2) then

            call compute_dNdx_dNdy(mesh(iel)%rq(iq),mesh(iel)%sq(iq),&
                                   dNNNUdx,dNNNUdy,dNNNVdx,dNNNVdy,jcob)

            exxq=sum(dNNNUdx*mesh(iel)%u)
            eyyq=sum(dNNNVdy*mesh(iel)%v)
            ezzq=0
            exyq=0.5d0*sum(dNNNUdy*mesh(iel)%u + dNNNVdx*mesh(iel)%v)
            exzq=0
            eyzq=0

         else

            call compute_dNdx_dNdy_dNdz(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),&
                                        dNNNUdx,dNNNUdy,dNNNUdz,&
                                        dNNNVdx,dNNNVdy,dNNNVdz,&
                                        dNNNWdx,dNNNWdy,dNNNWdz,jcob)

            exxq=sum(dNNNUdx*mesh(iel)%u)
            eyyq=sum(dNNNVdy*mesh(iel)%v)
            ezzq=sum(dNNNWdz*mesh(iel)%w)
            exyq=0.5d0*sum(dNNNUdy*mesh(iel)%u + dNNNVdx*mesh(iel)%v)
            exzq=0.5d0*sum(dNNNUdz*mesh(iel)%u + dNNNWdx*mesh(iel)%w)
            eyzq=0.5d0*sum(dNNNVdz*mesh(iel)%v + dNNNWdy*mesh(iel)%w)

         end if

         mesh(iel)%JxWq(iq)=jcob*mesh(iel)%weightq(iq)

         call experiment_material_model(mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                                        mesh(iel)%pq(iq),mesh(iel)%tempq(iq),&
                                        exxq,eyyq,ezzq,exyq,exzq,eyzq,&
                                        idummy,one,&
                                        mesh(iel)%etaq(iq),&
                                        mesh(iel)%rhoq(iq),&
                                        mesh(iel)%hcondq(iq),&
                                        mesh(iel)%hcapaq(iq),&
                                        mesh(iel)%hprodq(iq))
  
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

      etaq_min=min(minval(mesh(iel)%etaq(:)),etaq_min)
      etaq_max=max(maxval(mesh(iel)%etaq(:)),etaq_max)
      rhoq_min=min(minval(mesh(iel)%rhoq(:)),rhoq_min)
      rhoq_max=max(maxval(mesh(iel)%rhoq(:)),rhoq_max)
      if (use_T) then
         hcapaq_min=min(minval(mesh(iel)%hcapaq(:)),hcapaq_min)
         hcapaq_max=max(maxval(mesh(iel)%hcapaq(:)),hcapaq_max)
         hcondq_min=min(minval(mesh(iel)%hcondq(:)),hcondq_min)
         hcondq_max=max(maxval(mesh(iel)%hcondq(:)),hcondq_max)
      end if

   end do !nel

end if

write(*,'(a,2es10.3)') shift//'rhoq (m/M):',rhoq_min,rhoq_max
write(*,'(a,2es10.3)') shift//'etaq (m/M):',etaq_min,etaq_max
if (use_T) then
write(*,'(a,2es10.3)') shift//'hcapaq (m/M):',hcapaq_min,hcapaq_max
write(*,'(a,2es10.3)') shift//'hcondq (m/M):',hcondq_min,hcondq_max
end if

write(1240,'(4es12.4)') rhoq_min,rhoq_max,etaq_min,etaq_max

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'assign_values_to_qpoints:',elapsed,' s       |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
