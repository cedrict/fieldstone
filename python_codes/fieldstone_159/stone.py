import numpy as np
import time


make_vtu=False

c=20e6
phi=20./180.*np.pi
eps=1e5

Lx=1
Ly=1
Lz=1
nnx=600
nny=nnx
nnz=nnx
nelx=nnx-1
nely=nny-1
nelz=nnz-1
m=8
NV=nnx*nny*nnz
nel=(nnx-1)*(nny-1)*(nnz-1)

###############################################################################
# grid point setup
###############################################################################
start = time.time()

if make_vtu:

   x = np.empty(NV,dtype=np.float64)  # x coordinates
   y = np.empty(NV,dtype=np.float64)  # y coordinates
   z = np.empty(NV,dtype=np.float64)  # z coordinates

   counter=0
   for i in range(0,nnx):
       for j in range(0,nny):
           for k in range(0,nnz):
               x[counter]=i*Lx/float(nelx)
               y[counter]=j*Ly/float(nely)
               z[counter]=k*Lz/float(nelz)
               counter += 1
           #end for
       #end for
   #end for
   
   print("mesh setup: %.3f s" % (time.time() - start))

   icon =np.zeros((m, nel),dtype=np.int32)

   counter = 0
   for i in range(0,nelx):
       for j in range(0,nely):
           for k in range(0,nelz):
               icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
               icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
               icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
               icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
               icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
               icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
               icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
               icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
               counter += 1
           #end for
       #end for
   #end for

print("connectivity setup: %.3f s" % (time.time() - start))

###############################################################################
###############################################################################

xvals=np.linspace(-6e7,6e7,num=nnx)
yvals=np.linspace(-6e7,6e7,num=nny)
zvals=np.linspace(-6e7,6e7,num=nnz)

cosphi=np.cos(phi)
sinphi=np.sin(phi)

VMfile=open('VM.ascii',"w")
TRfile=open('TR.ascii',"w")
MCfile=open('MC.ascii',"w")
DPifile=open('DPi.ascii',"w")
DPcfile=open('DPc.ascii',"w")
DPmfile=open('DPm.ascii',"w")

VMfile_xy=open('VM_xy.ascii',"w")
TRfile_xy=open('TR_xy.ascii',"w")
MCfile_xy=open('MC_xy.ascii',"w")
DPifile_xy=open('DPi_xy.ascii',"w")
DPcfile_xy=open('DPc_xy.ascii',"w")
DPmfile_xy=open('DPm_xy.ascii',"w")

VMfile_plane=open('VM_plane.ascii',"w")
TRfile_plane=open('TR_plane.ascii',"w")
MCfile_plane=open('MC_plane.ascii',"w")
DPifile_plane=open('DPi_plane.ascii',"w")
DPcfile_plane=open('DPc_plane.ascii',"w")
DPmfile_plane=open('DPm_plane.ascii',"w")

ang0file=open('pi0_plane.ascii',"w")
ang6file=open('pi6_plane.ascii',"w")

sqrt3=np.sqrt(3)
onedeg=1./180.*np.pi

###############################################################################

if make_vtu:
   array_theta1=np.empty(NV,dtype=np.float64) 
   array_theta2=np.empty(NV,dtype=np.float64) 
   array_I3tau=np.empty(NV,dtype=np.float64) 
   array_sqrtI2tau=np.empty(NV,dtype=np.float64) 
   array_vM= np.empty(NV,dtype=np.float64)  
   array_TR= np.empty(NV,dtype=np.float64)  
   array_MC= np.empty(NV,dtype=np.float64)  
   array_DPc= np.empty(NV,dtype=np.float64)  
   array_DPi= np.empty(NV,dtype=np.float64)  
   array_DPm= np.empty(NV,dtype=np.float64)  

counter=0
for x in xvals:
    for y in yvals:
        for z in zvals:
            I1sig=x+y+z

            #since we divide by I2tau, we add a tiny 
            #value so as to avoid dividing by zero
            I2tau=((x-y)**2+(x-z)**2+(y-z)**2)/6.+0.001
            sqrtI2tau=np.sqrt(I2tau)
            if make_vtu:
               array_sqrtI2tau[counter]=sqrtI2tau

            I3tau=((2*x-y-z)**3+(2*y-x-z)**3+(2*z-x-y)**3)/81.
            if make_vtu:
               array_I3tau[counter]=I3tau

            #since we take arcsin of theta, we need to 
            #make sure it is within [-1,1] 
            theta=-3*np.sqrt(3)/2*I3tau/sqrtI2tau**3
            theta=min(theta,0.99999999)
            theta=max(-0.99999999,theta)
            if make_vtu:
               array_theta1[counter]=theta
            theta=np.arcsin(theta)/3.
            if make_vtu:
               array_theta2[counter]=theta

            costheta=np.cos(theta)
            sintheta=np.sin(theta)
            r=np.sqrt(x**2+y**2)

            if abs(x+y+z-1e7)<eps:
               if abs(theta)<onedeg:
                  ang0file.write("%10e %10e %10e \n" %(x,y,z))
               if abs(theta-np.pi/6.)<onedeg:
                  ang6file.write("%10e %10e %10e \n" %(x,y,z))

            #von Mises
            if make_vtu: array_vM[counter]=sqrtI2tau-c
            if abs(sqrtI2tau-c)<eps:
               VMfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  VMfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  VMfile_plane.write("%10e %10e %10e \n" %(x,y,z))

            #Tresca
            if make_vtu: array_TR[counter]=abs(2*sqrtI2tau*costheta-c)
            if abs(2*sqrtI2tau*costheta-c)<eps:
               TRfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  TRfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  TRfile_plane.write("%10e %10e %10e \n" %(x,y,z))
                
            #Mohr-Coulomb
            if make_vtu: array_MC[counter]=sqrtI2tau*(costheta-sintheta*sinphi/sqrt3)+I1sig/3.*sinphi-c*cosphi
            if abs(sqrtI2tau*(costheta-sintheta*sinphi/sqrt3)+I1sig/3.*sinphi-c*cosphi)<eps:
               MCfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  MCfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  MCfile_plane.write("%10e %10e %10e \n" %(x,y,z))

            #Drucker-Prager circumscribe 
            if make_vtu: array_DPc[counter]=sqrtI2tau-(-6*sinphi*I1sig/3 + 6*c*cosphi)/(sqrt3*(3-sinphi))
            if abs(sqrtI2tau-(-6*sinphi*I1sig/3 + 6*c*cosphi)/(sqrt3*(3-sinphi)))<eps:
               DPcfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  DPcfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  DPcfile_plane.write("%10e %10e %10e \n" %(x,y,z))

            #Drucker-Prager middle-circumscribe
            if make_vtu: array_DPm[counter]=sqrtI2tau-(-6*sinphi*I1sig/3 + 6*c*cosphi)/(sqrt3*(3+sinphi))
            if abs(sqrtI2tau-(-6*sinphi*I1sig/3 + 6*c*cosphi)/(sqrt3*(3+sinphi)))<eps:
               DPmfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  DPmfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  DPmfile_plane.write("%10e %10e %10e \n" %(x,y,z))

            #Drucker-Prager inscribe
            if make_vtu: array_DPi[counter]=sqrtI2tau-(-3*sinphi*I1sig/3 + 3*c*cosphi)/np.sqrt(9+3*sinphi**2)
            if abs(sqrtI2tau-(-3*sinphi*I1sig/3 + 3*c*cosphi)/np.sqrt(9+3*sinphi**2))<eps:
               DPifile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  DPifile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  DPifile_plane.write("%10e %10e %10e \n" %(x,y,z))
         
            counter+=1

            if counter%10000==0:
               print(counter/(nnx*nny*nnz)*100,'%')

        #end for
    #end for
#end for

###############################################################################

if make_vtu: 
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #vtufile.write("<DataArray type='Float32' Name='sqrt(I2(tau))' Format='ascii'> \n")
   #for i in range (0,NV):
   #    vtufile.write("%e\n" % array_sqrtI2tau[i])
   #vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='F(vM)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % array_vM[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='F(Tr)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % array_TR[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='F(MC)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % array_MC[i])
   vtufile.write("</DataArray>\n")
   #vtufile.write("<DataArray type='Float32' Name='theta1' Format='ascii'> \n")
   #for i in range (0,NV):
   #    vtufile.write("%e\n" % array_theta1[i])
   #vtufile.write("</DataArray>\n")
   #vtufile.write("<DataArray type='Float32' Name='theta2' Format='ascii'> \n")
   #for i in range (0,NV):
   #    vtufile.write("%e\n" % array_theta2[i])
   #vtufile.write("</DataArray>\n")
   #vtufile.write("<DataArray type='Float32' Name='I3tau' Format='ascii'> \n")
   #for i in range (0,NV):
   #    vtufile.write("%e\n" % array_I3tau[i])
   #vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='F(DP_c)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % array_DPc[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='F(DP_m)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % array_DPm[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='F(DP_i)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % array_DPi[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                       icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %12)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu: %.3f s" % (time.time() - start))


