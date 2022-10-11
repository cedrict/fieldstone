import numpy as np
import time 

###############################################################################
start = time.time()

file=open("jan76_dec20.ndk", "r")
lines = file.readlines()
file.close

print("time read file: %.3f s" % (time.time() - start))

###############################################################################
start = time.time()
         
C_nb=int(284160/5)

C_lon=np.zeros(C_nb,dtype=np.float64)
C_lat=np.zeros(C_nb,dtype=np.float64)
C_time=np.zeros(C_nb,dtype=np.float64)
C_depth=np.zeros(C_nb,dtype=np.float64)
C_hypocenter_line=np.zeros(C_nb,dtype=np.int16)
C_Mrr=np.zeros(C_nb,dtype=np.float64)
C_Mtt=np.zeros(C_nb,dtype=np.float64)
C_Mpp=np.zeros(C_nb,dtype=np.float64)
C_Mrt=np.zeros(C_nb,dtype=np.float64)
C_Mrp=np.zeros(C_nb,dtype=np.float64)
C_Mtp=np.zeros(C_nb,dtype=np.float64)

for il in range(0,C_nb):
    #-----first line-----
    vals=lines[il*5+0].strip().split()
    if vals[0]=='PDE':
       C_hypocenter_line[il]=1
    if vals[0]=='PDEW':
       C_hypocenter_line[il]=2
    if vals[0]=='MLI':
       C_hypocenter_line[il]=3
    if vals[0]=='REB':
       C_hypocenter_line[il]=4
    if vals[0]=='SWEQ':
       C_hypocenter_line[il]=5

    #-----third line-----
    vals=lines[il*5+2].strip().split()
    C_time[il]=vals[1]
    C_lat[il]=vals[3]
    C_lon[il]=vals[5]
    C_depth[il]=vals[7]

    #-----fourth line-----
    vals=lines[il*5+3].strip().split()
    #print(vals)
    #print(il*5+3,len(vals))
    #print(vals[1])
    if abs(float(vals[1]))<1e-7: 
       print(il,vals[1])
    C_Mrr[il]=float(vals[1])#*10**float(vals[0])
    C_Mtt[il]=float(vals[3])#*10**float(vals[0])
    C_Mpp[il]=float(vals[5])#*10**float(vals[0])
    C_Mrt[il]=float(vals[7])#*10**float(vals[0])
    C_Mrp[il]=float(vals[9])#*10**float(vals[0])
    C_Mtp[il]=float(vals[11])#*10**float(vals[0])

C_M=np.sqrt(0.5*(C_Mrr**2+C_Mtt**2+C_Mpp**2)+C_Mrt**2+C_Mrp**2+C_Mtp**2)

print('----------------------------------------')
print('latitude (m/M)',min(C_lat),max(C_lat))
print('longitude (m/M)',min(C_lon),max(C_lon))
print('depth (m/M)',min(C_depth),max(C_depth))
print('time (m/M)',min(C_time),max(C_time))
print('Mrr (m/M)',min(C_Mrr),max(C_Mrr))
print('Mtt (m/M)',min(C_Mtt),max(C_Mtt))
print('Mpp (m/M)',min(C_Mpp),max(C_Mpp))
print('Mrt (m/M)',min(C_Mrt),max(C_Mrt))
print('Mrp (m/M)',min(C_Mrp),max(C_Mrp))
print('Mtp (m/M)',min(C_Mtp),max(C_Mtp))
print('----------------------------------------')

#np.savetxt('data.ascii',np.array([C_time,C_lat,C_lon,C_depth]).T)
np.savetxt('data_M.ascii',np.array([C_Mrr,C_Mtt,C_Mpp,C_Mrt,C_Mrp,C_Mtp]).T)

#------------------------------------------------------------------------------

C_phi=C_lon/180*np.pi
C_theta=(90-C_lat)/180*np.pi
C_r=6371e3-C_depth*1e3

C_x=C_r*np.sin(C_theta)*np.cos(C_phi)
C_y=C_r*np.sin(C_theta)*np.sin(C_phi)
C_z=C_r*np.cos(C_theta)

#------------------------------------------------------------------------------
   
if True:
   vtufile=open("location.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(C_nb,C_nb))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10f %10f %10f \n" %(C_x[i],C_y[i],C_z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='hypocenter_line' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10e \n" %(C_hypocenter_line[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Mrr' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10e \n" %(C_Mrr[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Mtt' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10e \n" %(C_Mtt[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Mpp' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10e \n" %(C_Mpp[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Mrt' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10e \n" %(C_Mrt[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Mrp' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10e \n" %(C_Mrp[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Mtp' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10e \n" %(C_Mtp[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='M (effective)' Format='ascii'> \n")
   for i in range(0,C_nb):
       vtufile.write("%10e \n" %(C_M[i]))
   vtufile.write("</DataArray>\n")



   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range (0,C_nb):
       vtufile.write("%d \n" % i)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range (0,C_nb):
       vtufile.write("%d \n" %(i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,C_nb):
       vtufile.write("%d \n" % 1)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

