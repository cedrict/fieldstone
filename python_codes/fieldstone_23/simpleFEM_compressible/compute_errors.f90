subroutine compute_errors(nel,np,x,y,u,v,p,icon,ibench,L2_err_u,L2_err_v,L2_err_p,L1_err_u,L1_err_v,L1_err_p)
implicit none
integer, parameter :: m=4       
integer nel,np,ibench
integer iel,k,iq,jq
integer icon(4,nel)
real(8) x(np),y(np),u(np),v(np),p(nel)
real(8) xq,yq,uq,vq,jcb(2,2)  
real(8) L2_err_u,L2_err_v,L2_err_p
real(8) L1_err_u,L1_err_v,L1_err_p
real(8) rq,sq,wq,vthq,uthq,pthq,pq,jcob
real(8) N(m),dNdr(m),dNds(m) 
real(8), external :: uth,vth,pth  

!==============================================!

L2_err_u=0.d0
L2_err_v=0.d0
L2_err_p=0.d0
L1_err_u=0.d0
L1_err_v=0.d0
L1_err_p=0.d0

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
      uq=0.d0
      vq=0.d0
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
      end do
      pq=p(iel)

      uthq=uth(xq,yq,ibench)
      vthq=vth(xq,yq,ibench)
      pthq=pth(xq,yq,ibench)

      L2_err_u=L2_err_u+(uq-uthq)**2*jcob*wq
      L2_err_v=L2_err_v+(vq-vthq)**2*jcob*wq
      L2_err_p=L2_err_p+(pq-pthq)**2*jcob*wq

      L1_err_u=L1_err_u+abs((uq-uthq)*jcob*wq)
      L1_err_v=L1_err_v+abs((vq-vthq)*jcob*wq)
      L1_err_p=L1_err_p+abs((pq-pthq)*jcob*wq)

   end do
   end do

end do

L2_err_u=sqrt(L2_err_u)
L2_err_v=sqrt(L2_err_v)
L2_err_p=sqrt(L2_err_p)

!write(*,'(a,3es16.6)') 'err (L_2 norm) u,v,p ',L2_err_u,L2_err_v,L2_err_p


end subroutine

