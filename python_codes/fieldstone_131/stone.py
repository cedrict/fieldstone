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

#------------------------------------------------------------------------------

def compute_triangles_center_coordinates(coords,nodesArray):

    tx = coords[:,0]
    ty = coords[:,1]

    xc = (tx[nodesArray[:,0]] + tx[nodesArray[:,1]] + tx[nodesArray[:,2]]) / 3 
    yc = (ty[nodesArray[:,0]] + ty[nodesArray[:,1]] + ty[nodesArray[:,2]]) / 3 

    center = np.stack([xc, yc],axis=1)

    return center 

###############################################################################
###############################################################################

ex1=True
ex2=False
ex3=False
ex4=False
ex5=False

#------------------------------------------------------------------------------
# first example: a square of size L. 

if ex1:

   L = 10 
   square_vertices = np.array([[0,0],[0,L],[L,L],[L,0]])
   square_edges = compute_segs(square_vertices)

   print(square_edges)

   O1 = {'vertices' : square_vertices, 'segments' : square_edges}
   T1 = tr.triangulate(O1, 'pqa10') # tr.triangulate() computes the main dictionary 

   tr.compare(plt, O1, T1) # The tr.compare() function always takes plt as its 1st argument
   #plt.savefig('ex1.pdf', bbox_inches='tight')
   #plt.show()

   print('vertices:',T1['vertices'])
   print('segments:',T1['segments'])
   print('triangles connectivity:',T1['triangles']) # this is icon!
   print('vertices on hull:',T1['vertex_markers'])
   print('segments on hull:',T1['segment_markers'])

   area=compute_triangles_area(T1['vertices'], T1['triangles'])
   icon=T1['triangles'] ; icon=icon.T
   x=T1['vertices'][:,0] 
   y=T1['vertices'][:,1] 
   export_elements_to_vtu(x,y,icon,'example1.vtu',area)

#------------------------------------------------------------------------------
# second example
#------------------------------------------------------------------------------

if ex2: 

   ### Martian Surface

   npoints10 = 50 #amount of nodes
   R10 = 3.39 #.e6 meters. Distance Martian center to surface
   thetas10 = np.linspace(-np.pi*0.5, 0.5*np.pi, npoints10, endpoint=True)
   points10 = np.stack([np.cos(thetas10), np.sin(thetas10)], axis=1) * R10 
   segs10 = np.stack([np.arange(npoints10-1), np.arange(npoints10-1) + 1], axis=1) % npoints10

   ### Martian Moho ### coded as 11

   npoints11 = 40 #amount of nodes
   R11 = 2.99 #.e6 meters. Distance Martian center to moho
   thetas11 = np.linspace(-np.pi*0.5, 0.5*np.pi, npoints11, endpoint=True)
   points11 = np.stack([np.cos(thetas11), np.sin(thetas11)], axis = 1) * R11
   segs11 = np.stack([np.arange(npoints11-1), np.arange(npoints11-1)+1], axis=1) % npoints11

   ### Martian CMB ### coded as 12

   npoints12 = 30
   R12 = 1.83 #.e6 meters. Distance Martian center to CMB
   theta12 = np.linspace(-np.pi*0.5, np.pi*0.5, npoints12, endpoint=True)
   points12 = np.stack([np.cos(theta12), np.sin(theta12)], axis=1) * R12
   segs12 = np.stack([np.arange(npoints12-1), np.arange(npoints12-1)+1], axis=1) % npoints12

   # Stacks the nodes of all three half-circles into one matrix
   points1 = np.vstack([points10, points11, points12])

   # Combines all segment matrices and adds segments that connect 
   # The code below is a bit messy. The goal here is to add segments that
   # connect the Surface half-circle to the Moho half-circle and the
   # Moho half-circle to the CMB half-circle. 
   # e.g. Firstly, the Surface half-circle consists of 50 points and is computed.
   # Secondly, the Moho half-circle consists of 40 points and is computed.
   # So, we need to connect index 0 and 49 to 50 and 89 respectively to the 'segments' variable
   segs1 = np.vstack([segs10, segs11 + 1 + segs10.shape[0]])
   SEGS1 = np.vstack([segs1, segs12 + 2 + segs1.shape[0]])
   SEGS2 = np.stack([[segs10.shape[0], 0], [segs10.shape[0]+segs11.shape[0]+1, segs10.shape[0]+1]], axis=-1)
   SEGS3 = np.vstack([SEGS1, SEGS2])
   SEGS4 = np.stack([[segs10.shape[0] + segs11.shape[0] + segs12.shape[0] + 2, segs10.shape[0] + 1], [segs10.shape[0]+segs11.shape[0]+1, segs10.shape[0]+segs11.shape[0]+2]], axis=1)
   SEGS5 = np.vstack([SEGS3, SEGS4])

   # Prepare data
   A1 = {'vertices' : points1, 'segments' : SEGS5, 'holes' : [[0,0]]}

   # q ensures a minimum angle of 20 degrees in all triangles. 
   # p ensures the segments close off the area in which the hole should be.
   # a in combination with the number after it enables a maximum area constraint of {number} square meters
   t1 = tr.triangulate(A1, 'qpa0.1')

   area=compute_triangles_area(t1['vertices'], t1['triangles'])
   icon=t1['triangles'] ; icon=icon.T
   x=t1['vertices'][:,0] 
   y=t1['vertices'][:,1] 
   export_elements_to_vtu(x,y,icon,'example2.vtu',area)


#------------------------------------------------------------------------------
# third example: filling a circle with triangles
#------------------------------------------------------------------------------

if ex3: 

   N = 32
   theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
   pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
   A = dict(vertices=pts)
   t1 = tr.triangulate(A, 'qa0.001')
   #tr.compare(plt, A, B)
   #plt.show()

   area=compute_triangles_area(t1['vertices'], t1['triangles'])
   icon=t1['triangles'] ; icon=icon.T
   x=t1['vertices'][:,0] 
   y=t1['vertices'][:,1] 
   export_elements_to_vtu(x,y,icon,'example3.vtu',area)

#------------------------------------------------------------------------------
# fourth example: filling an annulus with triangles
#------------------------------------------------------------------------------

if ex4:

   def circle(N, R):
       i = np.arange(N) #where N is the number of points on the circle
       theta = i * 2 * np.pi / N
       pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R #and R is the radius of the circle
       seg = np.stack([i, i + 1], axis=1) % N
       return pts, seg

   #make two circles
   pts0, seg0 = circle(30, 1.4)
   pts1, seg1 = circle(16, 0.6)

   pts = np.vstack([pts0, pts1]) #making 1 vector out of the location of all nodes
   seg = np.vstack([seg0, seg1 + seg0.shape[0]]) #making 1 vector out of all segments 

   A = dict(vertices=pts, segments=seg, holes=[[0, 0]])
   t1 = tr.triangulate(A, 'qpa0.025')

   area=compute_triangles_area(t1['vertices'], t1['triangles'])
   icon=t1['triangles'] ; icon=icon.T
   x=t1['vertices'][:,0] 
   y=t1['vertices'][:,1] 
   export_elements_to_vtu(x,y,icon,'example4.vtu',area)

#------------------------------------------------------------------------------
# fifth example: filling an annulus with triangles
#------------------------------------------------------------------------------

if ex5:

   def halfcircle(N, R):   #a function that returns half a sphere with radius R and N# of nodes on the border
       i = np.arange(N)
       #angle of the position of the node on the sphere
       theta = np.linspace(0,np.pi,num=N,endpoint=True)
       #sphere is rotated by half pi
       pts = np.stack([np.cos(theta-np.pi/2),np.sin(theta-np.pi/2)], axis=1)*R 
       #each segment runs from one node to the next and it must be a closed shape
       seg = np.stack([i, i+1], axis=1) %N 
       return pts , seg

   #make a sphere of diameter 1,4 and with 15 nodes 
   pts, seg = halfcircle(15, 1.4) 

   A = dict(vertices=pts, segments=seg)  #make a dictionary of the points and segments
   t1 = tr.triangulate(A, 'qpa0.05')       #make a triangle mesh of the half sphere where the minimum area is 0.05

   area=compute_triangles_area(t1['vertices'], t1['triangles'])
   icon=t1['triangles'] ; icon=icon.T
   x=t1['vertices'][:,0] 
   y=t1['vertices'][:,1] 
   export_elements_to_vtu(x,y,icon,'example5.vtu',area)


