subroutine compute_derivatives_errors(nel,np,x,y,dudx_nodal, dvdx_nodal, dudy_nodal, dvdy_nodal, phi_nodal,&
                                      icon,ibench,dudx_L1,dudx_L2,dvdx_L1,dvdx_L2,dudy_L1,dudy_L2,dvdy_L1,dvdy_L2,phi_L1,phi_L2)

implicit none

integer, parameter :: m=4
integer nel,np,ibench
integer icon(4,nel)
real(8) x(np),y(np)
real(8) dudx_L1,dudx_L2,dvdx_L1,dvdx_L2,dudy_L1,dudy_L2,dvdy_L1,dvdy_L2,phi_L1,phi_L2
real(8) dudx_nodal(np),dvdx_nodal(np),dudy_nodal(np),dvdy_nodal(np),phi_nodal(np)
real(8), external ::  dudxth,dvdyth,dudyth,dvdxth,phith
real(8) xq,yq,jcb(2,2),phiq
integer iel,k,iq,jq
real(8) N(m),dNdr(m),dNds(m)
real(8) rq,sq,wq,jcob
real(8) dudxq,dvdxq,dudyq,dvdyq

dudx_L1=0 ; dudx_L2=0
dvdx_L1=0 ; dvdx_L2=0
dudy_L1=0 ; dudy_L2=0
dvdy_L1=0 ; dvdy_L2=0
phi_L1=0  ; phi_L2=0

do iel=1,nel

   do iq=-1,1,2
   do jq=-1,1,2

      rq=iq/sqrt(3.d0)
      sq=jq/sqrt(3.d0)
      wq=1.d0*1.d0

      N(1)=0.25d0*(1.d0-rq)*(1.d0-sq)
      N(2)=0.25d0*(1.d0+rq)*(1.d0-sq)
      N(3)=0.25d0*(1.d0+rq)*(1.d0+sq)
      N(4)=0.25d0*(1.d0-rq)*(1.d0+sq)

      dNdr(1)= - 0.25d0*(1.d0-sq)   ;   dNds(1)= - 0.25d0*(1.d0-rq)
      dNdr(2)= + 0.25d0*(1.d0-sq)   ;   dNds(2)= - 0.25d0*(1.d0+rq)
      dNdr(3)= + 0.25d0*(1.d0+sq)   ;   dNds(3)= + 0.25d0*(1.d0+rq)
      dNdr(4)= - 0.25d0*(1.d0+sq)   ;   dNds(4)= + 0.25d0*(1.d0-rq)

      jcb=0.d0
      do k=1,m
         jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
         jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
         jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
         jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      enddo

      jcob=jcb(1,1)*jcb(2,2)-jcb(2,1)*jcb(1,2)

      xq=0.d0
      yq=0.d0
      dudxq=0.d0
      dudyq=0.d0
      dvdxq=0.d0
      dvdyq=0.d0
      phiq=0.d0
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         phiq=phiq+N(k)*phi_nodal(icon(k,iel))
         dudxq=dudxq+N(k)*dudx_nodal(icon(k,iel))
         dudyq=dudyq+N(k)*dudy_nodal(icon(k,iel))
         dvdxq=dvdxq+N(k)*dvdx_nodal(icon(k,iel))
         dvdyq=dvdyq+N(k)*dvdy_nodal(icon(k,iel))
      end do

      dudx_L1 = dudx_L1 + abs(dudxq-dudxth(xq,yq,ibench))   *jcob*wq
      dudx_L2 = dudx_L2 +    (dudxq-dudxth(xq,yq,ibench))**2*jcob*wq

      dudy_L1 = dudy_L1 + abs(dudyq-dudyth(xq,yq,ibench))   *jcob*wq
      dudy_L2 = dudy_L2 +    (dudyq-dudyth(xq,yq,ibench))**2*jcob*wq

      dvdx_L1 = dvdx_L1 + abs(dvdxq-dvdxth(xq,yq,ibench))   *jcob*wq
      dvdx_L2 = dvdx_L2 +    (dvdxq-dvdxth(xq,yq,ibench))**2*jcob*wq

      dvdy_L1 = dvdy_L1 + abs(dvdyq-dvdyth(xq,yq,ibench))   *jcob*wq
      dvdy_L2 = dvdy_L2 +    (dvdyq-dvdyth(xq,yq,ibench))**2*jcob*wq

      phi_L1 = phi_L1 + abs(phiq-phith(xq,yq,ibench))*jcob*wq
      phi_L2 = phi_L2 + (phiq-phith(xq,yq,ibench))**2*jcob*wq

   end do
   end do

end do

dudx_L2 = sqrt(dudx_L2)
dvdx_L2 = sqrt(dvdx_L2)
dudy_L2 = sqrt(dudy_L2)
dvdy_L2 = sqrt(dvdy_L2)
phi_L2 = sqrt(phi_L2)

end subroutine
