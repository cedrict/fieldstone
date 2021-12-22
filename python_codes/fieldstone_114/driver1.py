import forward as forward
import time as timing
import numpy as np

#------------------------------------------------------------------------------
Rsphere = 50e3
deltarho=-300

eta0=1e21
eta_star=1e3
rho0=3000

nrad=100
rmin=10e3
rmax=100e3

ndrho=100
drhomin=-500
drhomax=-50

#------------------------------------------------------------------------------

m=4
nelx=nrad-1
nely=ndrho-1
nel=nelx*nely
npts=nrad*ndrho

radius = np.empty(npts,dtype=np.float64)
drho   = np.empty(npts,dtype=np.float64)
icon =np.zeros((m,nel),dtype=np.int32)
misfit_g = np.empty(npts,dtype=np.float64)
misfit_v = np.empty(npts,dtype=np.float64)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter+=1

#------------------------------------------------------------------------------
# call the forward model and record the misfits
#------------------------------------------------------------------------------
start = timing.time()

counter=0
for j in range(0,ndrho):
    print(counter/npts*100,'%')
    for i in range(0,nrad):
        radius[counter]=rmin+(rmax-rmin)/(nrad-1)*i
        drho[counter]=drhomin+(drhomax-drhomin)/(nrad-1)*j
        misfit_g[counter],misfit_v[counter]=forward.compute_misfits(rho0,drho[counter],eta0,eta_star,\
                                                                    radius[counter],deltarho,Rsphere)
        counter+=1

print("completed %d measurements" %npts)
print("forward total %.3f s" %(timing.time()-start))
print("forward: %.3f s per call" % ((timing.time()-start)/npts))

#####################################################################
# export to vtu 
#####################################################################

if True:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(npts,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,npts):
       vtufile.write("%10f %10f %10f \n" %((radius[i]-rmin)/(rmax-rmin),(drho[i]-drhomin)/(drhomax-drhomin),0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='misfit gravity' Format='ascii'> \n")
   for i in range(0,npts):
       vtufile.write("%e \n" % misfit_g[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")





