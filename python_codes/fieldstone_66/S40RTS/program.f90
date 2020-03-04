 program opla
use S40RTS_tomography
use constants
implicit none
integer i,j,k,counter,ip
integer, parameter :: nnx=360
integer, parameter :: nny=180
integer, parameter :: nnz=100
integer, parameter :: nnp=nnx*nny*nnz
real(8), dimension(nnp) ::  theta,phi,rad,dlnvs,dlnrho
real(8) hx,hy,hz,latitude,longitude,radius
real, external :: sdot,splh

!open(unit=10,file='S20RTS.sph',action='read',status='old')
open(unit=10,file='S40RTS.sph',action='read',status='old')
read(10,*) lmax,dum,nsmx
ndmx = nsmx
ndmn = 4 
natd=(lmax+1)**2
ind=(ndmn-1)*natd+1
do i=ndmn,ndmx
   do j=0,lmax
      ind1=ind+2*j
      read(10,'(11e12.4)',end=100)(x(k),k=ind,ind1)
      ind=ind1+1
   enddo
enddo
goto 200 
100  stop 'incompatible sph header'
200 continue
close(10)
if(lmax.gt.MXLH) stop 'lmax.gt.MXLH'
if(ndmx.gt.MXSPL) stop 'ndmx.gt.MXSPL'

natd=(lmax+1)**2

print *,'done reading'

!-----------------------------------------------------------------

! (x) phi-> longitude -180:180
! (y) theta -> latitude -90:90
! (z) radius -> between rcmb and rmoho

hx=360/nnx
hy=180/nny
hz=(rmoho-rcmb)/nnz

counter=0    
do i=1,nnx    
   do j=1,nny 
      do k=1,nnz
         counter=counter+1    
         phi(counter)=-180+hx/2+dble(i-1)*hx
         theta(counter)= -90+hy/2+dble(j-1)*hy
         rad(counter)=rcmb+hz/2+dble(k-1)*hz
      end do    
   end do    
end do  

theta=theta/180*pi
phi=phi/180*pi

!-----------------------------------------------------------------

do ip=1,nnp

   latitude =pi/2.d0-theta(ip)
   longitude=        phi(ip)

   latitude =latitude /pi*180. 
   longitude=longitude/pi*180. 

   radius=rad(ip)

   d0=0 ; wk1=0 ; wk2=0 ; wk3=0

   ! Calculate the spline basis functions at a regular grid

   call splhsetup()

   ! calculate Y_k(latitude,longitude) at random depth level

   call ylm(real(latitude),real(longitude),lmax,d0,wk1,wk2,wk3)

   ! calculate spline coefficients

   nbeg=max(4,ndmn)
   nend=ndmx-3
   do i=nbeg,ndmx
      ind=(i-1)*natd+1 
      spl(i-3)=sdot(natd,d0,1,x(ind),1)
   enddo

   ! xd takes value -1 at the CMB and +1 at the moho
   ! if xd is outside this range, the splh returns 0

   xd=-1.+2.*(radius-rcmb)/(rmoho-rcmb)
   do ipp=nbeg-3,nend
      dlnvs(ip)=dlnvs(ip)+spl(ipp)*splh(ipp-1,xd)      !calculation of dv
   enddo

   !write(1333,*) radius,grid%dlnv(ip) 

   !---------------------------------------
   ! The scaling factors vary over depth 

   dlnrho(ip)=xi_s40rts*dlnvs(ip)

   !call compute_PREM_density(grid%r(ip),densprem)

   !grid%drho(ip)=densprem*grid%dlnrho(ip)

   !grid%dT(ip)=-1.d0/aalpha*grid%dlnrho(ip)

   
   write(123,*) phi(ip),theta(ip),rad(ip),dlnvs(ip),dlnrho(ip)

end do


write(*,*) 'dlnvs  (m/M)',minval(dlnvs),maxval(dlnvs)
write(*,*) 'dlntho (m/M)',minval(dlnrho),maxval(dlnrho)

end program 

