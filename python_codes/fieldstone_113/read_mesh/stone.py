import numpy as np

#------------------------------------------------------------------------------

f = open('cube86.mesh', 'r')
#f = open('cyl248.mesh', 'r')
#f = open('p01.mesh', 'r')
#f = open('example1.mesh', 'r')
#f = open('part.mesh', 'r')
lines = f.readlines()

#print(lines)
nlines=np.size(lines)
print('mesh file counts ',nlines,' lines')

for i in range(0,nlines):
    line=lines[i].strip()
    columns=line.split()
    if np.size(columns)>0 and columns[0]=='Vertices':
       nextline=lines[i+1].strip()
       print('mesh counts ',nextline, 'vertices')
       NV=int(nextline)
       vline=i+2
    if np.size(columns)>0 and columns[0]=='Triangles':
       nextline=lines[i+1].strip()
       print('mesh counts ',nextline, 'triangles')
    if np.size(columns)>0 and columns[0]=='Tetrahedra':
       nextline=lines[i+1].strip()
       print('mesh counts ',nextline, 'tetrahedra')
       nel=int(nextline)
       tline=i+2

#------------------------------------------------------------------------------

m=4

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
z = np.empty(NV,dtype=np.float64)  # z coordinates

icon =np.zeros((m,nel),dtype=np.int32) # connectivity array

counter=0
for i in range(vline,vline+NV):
    line=lines[i].strip()
    columns=line.split()
    x[counter]=float(columns[0])
    y[counter]=float(columns[1])
    z[counter]=float(columns[2])
    counter+=1
    
counter=0
for i in range(tline,tline+nel):
    line=lines[i].strip()
    columns=line.split()
    icon[0,counter]=int(columns[0])-1
    icon[1,counter]=int(columns[1])-1
    icon[2,counter]=int(columns[2])-1
    icon[3,counter]=int(columns[3])-1
    counter+=1

#------------------------------------------------------------------------------

if True: 
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
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
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
       vtufile.write("%d \n" %10)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

