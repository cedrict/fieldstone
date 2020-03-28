import numpy as np

nlon=1440
nlat=720
nfac=30

nel=(nlon-1)*(nlat-1)

rlon=np.zeros(nlon,dtype=np.float64)
rlat=np.zeros(nlat,dtype=np.float64)
height=np.zeros((nlon,nlat),dtype=np.float64)

filename='dem030_ascii.dat'
f = open(filename,'r')
i=0
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    l=len(columns)
    height[0+i*11:l+i*11,counter]=columns[0:l]
    i+=1
    if l==10:
       i=0
       counter+=1
#end for


for ilon in range(0,nlon):
    rlon[ilon]=-180.+float(2*ilon-1)*float(nfac)/240.

for ilat in range(0,nlat):
    rlat[ilat]=90.-float(2*ilat-1)*float(nfac)/240.


nelx=nlon-1
nely=nlat-1
icon=np.zeros((4,nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

if True: 
       vtufile=open('topo.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nlon*nlat,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for ilat in range(0,nlat):
           for ilon in range(0,nlon):
               vtufile.write("%10e %10e %10e \n" %(rlon[ilon],rlat[ilat],max(0,height[ilon,ilat]/2000)))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")

       vtufile.write("<DataArray type='Float32' Name='height' Format='ascii'> \n")
       for ilat in range(0,nlat):
           for ilon in range(0,nlon):
               vtufile.write("%10e \n" % max(-500,height[ilon,ilat]))
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='longitude' Format='ascii'> \n")
       for ilat in range(0,nlat):
           for ilon in range(0,nlon):
               vtufile.write("%d \n" % float(ilon))
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='latitude' Format='ascii'> \n")
       for ilat in range(0,nlat):
           for ilon in range(0,nlon):
               vtufile.write("%d \n" %float(ilat))
       vtufile.write("</DataArray>\n")


       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()
