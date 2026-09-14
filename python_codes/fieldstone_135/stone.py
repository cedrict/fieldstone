import csv
import numpy as np

slabid=np.zeros(1500000)

###############################################################################
# 'slab2/jap_09-21_input.csv',\ it seems to have disappeared from database ?!

count_total=0

rows = []
slab_id=0
for name in 'slab2/alu_06-23_input.csv', 'slab2/cal_06-23_input.csv', 'slab2/cam_06-23_input.csv',\
            'slab2/car_06-23_input.csv', 'slab2/cas_06-23_input.csv', 'slab2/cot_06-23_input.csv',\
            'slab2/hal_06-23_input.csv', 'slab2/hel_06-23_input.csv', 'slab2/him_06-23_input.csv',\
            'slab2/hin_06-23_input.csv', 'slab2/izu_06-23_input.csv',\
            'slab2/ker_06-23_input.csv', 'slab2/kur_06-23_input.csv', 'slab2/mak_06-23_input.csv',\
            'slab2/man_06-23_input.csv', 'slab2/mue_06-23_input.csv', 'slab2/pam_06-23_input.csv',\
            'slab2/phi_06-23_input.csv', 'slab2/png_06-23_input.csv', 'slab2/puy_06-23_input.csv',\
            'slab2/ryu_06-23_input.csv', 'slab2/sam_06-23_input.csv', 'slab2/sco_06-23_input.csv',\
            'slab2/sol_06-23_input.csv', 'slab2/sul_06-23_input.csv', 'slab2/sum_06-23_input.csv',\
            'slab2/van_06-23_input.csv':

    with open(name) as csv_file:
         csv_reader = csv.reader(csv_file, delimiter=',')
         #print(csv_file)
         count=0
         for row in csv_reader:
             if count>0: # bypass header
                rows.append(row)
                slabid[count_total]=slab_id
                count_total+=1 
             count+=1 

    print(name+' | nb of points:',count)
    
    slab_id+=1

###############################################################################

N=count_total
x=np.zeros(N)
y=np.zeros(N)
z=np.zeros(N)
lat=np.zeros(N)
lon=np.zeros(N)
depth=np.zeros(N)

for i in range(0,N):
    lat[i]=float(rows[i][0])
    lon[i]=float(rows[i][1])
    depth[i]=float(rows[i][2])

print('lat',min(lat),max(lat))
print('lon',min(lon),max(lon))
print('depth',min(depth),max(depth))

print('total number of points:',N)

###############################################################################

C_phi=lon/180*np.pi
C_theta=(90-lat)/180*np.pi
C_r=6371e3-depth*1e3

C_x=C_r*np.sin(C_theta)*np.cos(C_phi)
C_y=C_r*np.sin(C_theta)*np.sin(C_phi)
C_z=C_r*np.cos(C_theta)

vtufile=open("slab2.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,N))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10f %10f %10f \n" %(C_x[i],C_y[i],C_z[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData>\n")
#--
vtufile.write("<DataArray type='Float32' Name='depth' Format='ascii'> \n")
depth.tofile(vtufile,sep=' ',format='%.4e')
#for i in range(0,N):
#    vtufile.write("%10e \n" %(depth[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='id' Format='ascii'> \n")
slabid.tofile(vtufile,sep=' ',format='%.4e')
#for i in range(0,N):
#    vtufile.write("%10e \n" %(slabid[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for i in range (0,N):
    vtufile.write("%d \n" % i)
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for i in range (0,N):
    vtufile.write("%d \n" %(i+1))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,N):
    vtufile.write("%d \n" % 1)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

###############################################################################
