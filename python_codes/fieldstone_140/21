import numpy as np
import matplotlib.pyplot as plt
import triangle as tr

#------------------------------------------------------------------------------

def export_elements_to_vtu(x,y,icon,filename,area):
    N=np.size(x)
    m,nel=np.shape(icon)

    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#------------------------------------------------------------------------------

def compute_segs(InputCoords):
    segs = np.stack([np.arange(len(InputCoords)),np.arange(len(InputCoords))+1],axis=1)%len(InputCoords)
    return segs

#------------------------------------------------------------------------------

def compute_triangles_area(coords,nodesArray):
    
    tx = coords[:,0]
    ty = coords[:,1]
    
    # Triangle Area is calculated via Heron's formula, see wikipedia
    
    a = np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,1]])**2)
    b = np.sqrt((tx[nodesArray[:,2]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,2]]-ty[nodesArray[:,1]])**2)
    c = np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,2]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,2]])**2)
    
    area = 0.5 * np.sqrt(a**2 * c**2 - (( a**2 + c**2 - b**2) / 2)**2)
    area = area.reshape(-1,1) #Transposing the 1xN matrix into Nx1 shape
    
    return area

###############################################################################
###############################################################################

L = 10


#------------------------------------------------------------------------------
# build mesh
#------------------------------------------------------------------------------
 
square_vertices = np.array([[0,0],[0,L],[L,L],[L,0]])
square_edges = compute_segs(square_vertices)

O1 = {'vertices' : square_vertices, 'segments' : square_edges}
T1 = tr.triangulate(O1, 'pqa0.7') # tr.triangulate() computes the main dictionary 

tr.compare(plt, O1, T1) # The tr.compare() function always takes plt as its 1st argument
#plt.savefig('ex1.pdf', bbox_inches='tight')
#plt.show()

#print('vertices:',T1['vertices'])
#print('segments:',T1['segments'])
#print('triangles connectivity:',T1['triangles']) # this is icon!
#print('vertices on hull:',T1['vertex_markers'])
#print('segments on hull:',T1['segment_markers'])

area=compute_triangles_area(T1['vertices'], T1['triangles'])
icon=T1['triangles'] ; icon=icon.T
m,nel=np.shape(icon)
x=T1['vertices'][:,0] 
y=T1['vertices'][:,1] 
export_elements_to_vtu(x,y,icon,'example1.vtu',area)

print(area,np.sum(area))

#------------------------------------------------------------------------------
# boundary conditions
#------------------------------------------------------------------------------

for i in range(0,N):
     


#------------------------------------------------------------------------------
# solve system
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# export to vtu
#------------------------------------------------------------------------------





