import numpy as np

#------------------------------------------------------------------------------
   
def make_vtu_file(N,x,y,z,depth,name):

   vtufile=open(name,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,N))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,N):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='depth' Format='ascii'> \n")
   for i in range(0,N):
       vtufile.write("%10e \n" %(depth[i]))
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='id' Format='ascii'> \n")
   #for i in range(0,N):
   #    vtufile.write("%10e \n" %(slabid[i]))
   #vtufile.write("</DataArray>\n")
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

#------------------------------------------------------------------------------


print('**********************')
print('***** stone 136 ******')
print('**********************')

N=100000
lon=np.zeros(N)
lat=np.zeros(N)
depth=np.zeros(N)

lon_all=[]
lat_all=[]
depth_all=[]
N=0

for i in range(0,25):
    if i==0:
       name='slabs/Contours/aleutians.slb'
       n_layers=7
       vtuname='aleutians.vtu'
    if i==1:
       name='slabs/Contours/assam.slb'
       n_layers=4
       vtuname='assam.vtu'
    if i==2:
       name='slabs/Contours/camerica.slb'
       n_layers=7
       vtuname='camerica.vtu'
    if i==3:
       name='slabs/Contours/caribbean.slb'
       n_layers=5
       vtuname='caribbean.vtu'
    if i==4:
       name='slabs/Contours/ephilippines.slb'
       n_layers=6
       vtuname='ephilippines.vtu'
    if i==5:
       name='slabs/Contours/halmahera.slb'
       n_layers=6
       vtuname='halmahera.vtu'
    if i==6:
       name='slabs/Contours/hellas.slb'
       n_layers=5
       vtuname='hellas.vtu'
    if i==7:
       name='slabs/Contours/hindu1.slb'
       n_layers=7
       vtuname='hindu1.vtu'
    if i==8:
       name='slabs/Contours/hindu2.slb'
       n_layers=5
       vtuname='hindu2.vtu'
    if i==9:
       name='slabs/Contours/indonesia.slb'
       n_layers=15
       vtuname='indonesia.vtu'
    if i==10:
       name='slabs/Contours/italia.slb'
       n_layers=8
       vtuname='italia.vtu'
    if i==11:
       name='slabs/Contours/luzon.slb'
       n_layers=4
       vtuname='luzon.vtu'
    if i==12:
       name='slabs/Contours/marjapkur.slb'
       n_layers=15
       vtuname='marjapkur.vtu'
    if i==13:
       name='slabs/Contours/mindanao.slb'
       n_layers=3
       vtuname='mindanao.vtu'
    if i==14:
       name='slabs/Contours/molucca.slb'
       n_layers=14
       vtuname='molucca.vtu'
    if i==15:
       name='slabs/Contours/nbritain.slb'
       n_layers=13
       vtuname='nbritain.vtu'
    if i==16:
       name='slabs/Contours/ryukyus.slb'
       n_layers=7
       vtuname='ryukyus.vtu'
    if i==17:
       name='slabs/Contours/samerica.slb'
       n_layers=14
       vtuname='samerica.vtu'
    if i==18:
       name='slabs/Contours/solomons.slb'
       n_layers=12
       vtuname='solomons.vtu'
    if i==19:
       name='slabs/Contours/ssandwich.slb'
       n_layers=6
       vtuname='ssandwich.vtu'
    if i==20:
       name='slabs/Contours/sulawesi.slb'
       n_layers=5
       vtuname='sulawesi.vtu'
    if i==21:
       name='slabs/Contours/tonga.slb'
       n_layers=15
       vtuname='tonga.vtu'
    if i==22:
       name='slabs/Contours/vanuatu.slb'
       n_layers=9
       vtuname='vanuatu.vtu'
    if i==23:
       name='slabs/Contours/wphilippines.slb'
       n_layers=7
       vtuname='wphilippines.vtu'

    print(name)
    file=open(name, "r") 
    lines = file.readlines() 
    file.close

    counter=0
    for i in range(1,n_layers+1):
        npts=int(lines[counter+i]) ; print('     layer ',i,'counts',npts,' points')
        for il in range(counter+i+1,counter+i+npts+1):
            vals=lines[il].strip().split()
            lon[counter]=float(vals[0])
            lat[counter]=float(vals[1])
            depth[counter]=float(vals[2])
            lon_all.append(float(vals[0]))
            lat_all.append(float(vals[1]))
            depth_all.append(float(vals[2]))
            counter+=1

    phi=lon/180*np.pi
    theta=(90-lat)/180*np.pi
    r=(6371-depth)*1000

    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)

    make_vtu_file(counter,x,y,z,depth,vtuname)

    N+=counter

lon_all=np.array(lon_all)
lat_all=np.array(lat_all)
depth_all=np.array(depth_all)

phi=lon_all/180.*np.pi
theta=(90.-lat_all)/180.*np.pi
r=(6371.-depth_all)*1000.

x=r*np.sin(theta)*np.cos(phi)
y=r*np.sin(theta)*np.sin(phi)
z=r*np.cos(theta)

make_vtu_file(N,x,y,z,depth_all,'all.vtu')

#------------------------------------------------------------------------------





