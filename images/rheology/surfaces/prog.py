import numpy as np

xvals=np.linspace(-6e7,6e7,num=400)
yvals=np.linspace(-6e7,6e7,num=400)
zvals=np.linspace(-6e7,6e7,num=400)

c=20e6
phi=20./180.*np.pi
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

ang0file=open('pi3_plane.ascii',"w")
ang3file=open('pi3_plane.ascii',"w")
ang6file=open('pi6_plane.ascii',"w")

eps=2e5

sqrt3=np.sqrt(3)

onedeg=1./180.*np.pi

#for x in xvals:
#    for y in xvals:
#        if y>x-2*c and y<x+2*c:
#           Delta=-12*(x-y)**2 + 48*c**2
#           zmin=(2*(x+y)-np.sqrt(Delta))/4
#           zmax=(2*(x+y)+np.sqrt(Delta))/4
           #print (x,y,zmin)
           #print (x,y,zmax)

for x in xvals:
    for y in xvals:
        for z in xvals:
            I1sig=x+y+z
            I2tau=((x-y)**2+(x-z)**2+(y-z)**2)/6.
            sqrtI2tau=np.sqrt(I2tau)
            I3tau=((2*x-y-z)**3+(2*y-x-z)**3+(2*z-x-y)**3)/81.
            theta=-3*np.sqrt(3)/2*I3tau/sqrtI2tau**3
            theta=np.arcsin(theta)/3.
            costheta=np.cos(theta)
            sintheta=np.sin(theta)
            r=np.sqrt(x**2+y**2)

            if sqrtI2tau*(costheta-sintheta*sinphi/sqrt3) < -I1sig/3.*sinphi+c*cosphi:
               if abs(x+y+z-1e7)<eps:
                  if abs(theta)<onedeg:
                     ang0file.write("%10e %10e %10e \n" %(x,y,z))
                  if abs(theta-np.pi/3.)<onedeg:
                     ang3file.write("%10e %10e %10e \n" %(x,y,z))
                  if abs(theta-np.pi/6.)<onedeg:
                     ang6file.write("%10e %10e %10e \n" %(x,y,z))

            #von Mises
            if abs(sqrtI2tau-c)<eps:
               VMfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  VMfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  VMfile_plane.write("%10e %10e %10e \n" %(x,y,z))

            #Tresca
            if abs(2*sqrtI2tau*costheta-c)<eps:
               TRfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  TRfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  TRfile_plane.write("%10e %10e %10e \n" %(x,y,z))
                
            #Mohr-Coulomb
            if abs(sqrtI2tau*(costheta-sintheta*sinphi/sqrt3)+I1sig/3.*sinphi-c*cosphi)<eps:
               MCfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  MCfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  MCfile_plane.write("%10e %10e %10e \n" %(x,y,z))

            #Drucker-Prager inscribe
            if abs(sqrtI2tau-(-6*sinphi*I1sig/3 + 6*c*cosphi)/(sqrt3*(3-sinphi)))<eps:
               DPifile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  DPifile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  DPifile_plane.write("%10e %10e %10e \n" %(x,y,z))

            #Drucker-Prager circumscribe
            if abs(sqrtI2tau-(-6*sinphi*I1sig/3 + 6*c*cosphi)/(sqrt3*(3+sinphi)))<eps:
               DPcfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  DPcfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  DPcfile_plane.write("%10e %10e %10e \n" %(x,y,z))

            #Drucker-Prager circumscribe
            if abs(sqrtI2tau-(-3*sinphi*I1sig/3 + 3*c*cosphi)/np.sqrt(9+3*sinphi**2))<eps:
               DPmfile.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x-y)<eps:
                  DPmfile_xy.write("%10e %10e %10e \n" %(x,y,z))
               if abs(x+y+z-1e7)<eps:
                  DPmfile_plane.write("%10e %10e %10e \n" %(x,y,z))




