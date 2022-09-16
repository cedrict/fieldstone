import numpy as np
import triangle as tr

#------------------------------------------------------------------------------

def export_swarm_to_vtu(x,y,filename):
       N=np.size(x)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,N))
       #--
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,N):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,N):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

#------------------------------------------------------------------------------

def export_elements_to_vtuP1(x,y,icon,filename):
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

def export_elements_to_vtuP2(x,y,icon,filename):
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
        vtufile.write("%f %f %f \n" %(x[i],y[i],0.))
        #print(x[i],y[i],0.)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],icon[4,iel],icon[5,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#------------------------------------------------------------------------------
# convert P1 mesh into P2 mesh
#------------------------------------------------------------------------------

def mesh_P1_to_P2(x,y,icon):
    NV=np.size(x)
    m,nel=np.shape(icon)

    NVme=3*nel   # mid-edge nodes
    xme = np.empty(NVme,dtype=np.float64)  # x coordinates
    yme = np.empty(NVme,dtype=np.float64)  # y coordinates
    for iel in range(0,nel):
        xme[iel*3+0]=0.5*(x[icon[0,iel]]+x[icon[1,iel]]) #first edge
        yme[iel*3+0]=0.5*(y[icon[0,iel]]+y[icon[1,iel]])
        xme[iel*3+1]=0.5*(x[icon[1,iel]]+x[icon[2,iel]]) #second edge
        yme[iel*3+1]=0.5*(y[icon[1,iel]]+y[icon[2,iel]])
        xme[iel*3+2]=0.5*(x[icon[2,iel]]+x[icon[0,iel]]) #third edge
        yme[iel*3+2]=0.5*(y[icon[2,iel]]+y[icon[0,iel]])

    eps=1e-6*min(max(x)-min(x),max(y)-min(y))

    double = np.zeros(NVme, dtype=np.bool)  # default is false
    for i in range(0,NVme):
        if not double[i]:
           for j in range(0,NVme):
               if j!=i:
                  if abs(xme[i]-xme[j])<eps and abs(yme[i]-yme[j])<eps:
                     double[j]=True
    

    NVme_new=NVme-sum(double)
    #print('real nb of mid edges pts',NVme_new)

    NVnew=NV+NVme_new
    #print('total number of P2 nodes',NVnew)

    xV = np.zeros(NVnew,dtype=np.float64)  # x coordinates
    yV = np.zeros(NVnew,dtype=np.float64)  # y coordinates
    iconV =np.zeros((6,nel),dtype=np.int32)

    xV[0:NV]=x[0:NV]
    yV[0:NV]=y[0:NV]
    xV[NV:NVnew]=xme[np.invert(double)]
    yV[NV:NVnew]=yme[np.invert(double)]

    counter=0
    for iel in range(0,nel):
        iconV[0,iel]=icon[0,iel]
        iconV[1,iel]=icon[1,iel]
        iconV[2,iel]=icon[2,iel]

        xmid=xme[3*iel] #first edge
        ymid=yme[3*iel]
        for i in range(NV,NVnew):
            if abs(xmid-xV[i])<eps and abs(ymid-yV[i])<eps:
               iconV[3,iel]=i
               break

        xmid=xme[3*iel+1] #second edge
        ymid=yme[3*iel+1]
        for i in range(NV,NVnew):
            if abs(xmid-xV[i])<eps and abs(ymid-yV[i])<eps:
               iconV[4,iel]=i
               break

        xmid=xme[3*iel+2] #third edge
        ymid=yme[3*iel+2]
        for i in range(NV,NVnew):
            if abs(xmid-xV[i])<eps and abs(ymid-yV[i])<eps:
               iconV[5,iel]=i
               break

    return NVnew,xV,yV,iconV

#------------------------------------------------------------------------------
# define original P1 mesh
#------------------------------------------------------------------------------

example=False

if example:

   NV=6 # nb of vertices
   nel=4 # nb of triangles/elements
   m=3 # P1 triangles have 3 vertices

   icon =np.zeros((m, nel),dtype=np.int32)
   x=np.array([-2 ,0,-1,1,2,1.5],dtype=np.float64) 
   y=np.array([0.5,0,2,3,0,-1],dtype=np.float64) 

   icon[0,0]=0  ; icon[1,0]=1  ; icon[2,0]=2 # elt 0
   icon[0,1]=1  ; icon[1,1]=3  ; icon[2,1]=2 # elt 1
   icon[0,2]=1  ; icon[1,2]=4  ; icon[2,2]=3 # elt 2
   icon[0,3]=1  ; icon[1,3]=5  ; icon[2,3]=4 # elt 3

else:

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
   t1 = tr.triangulate(A, 'qpa0.25')

   icon=t1['triangles'] ; icon=icon.T
   x=t1['vertices'][:,0] 
   y=t1['vertices'][:,1] 

   NV=np.shape(x)[0]
   m,nel=np.shape(icon)

print('NV=',NV)

export_elements_to_vtuP1(x,y,icon,'meshP1.vtu')

np.savetxt('meshP1.ascii',np.array([x,y]).T,header='# x,y')

#------------------------------------------------------------------------------
# create all mid edge nodes (with duplicates)
#------------------------------------------------------------------------------

NVme=3*nel   # mid-edge nodes

xme = np.empty(NVme,dtype=np.float64)  # x coordinates
yme = np.empty(NVme,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xme[iel*3+0]=0.5*(x[icon[0,iel]]+x[icon[1,iel]]) #first edge
    yme[iel*3+0]=0.5*(y[icon[0,iel]]+y[icon[1,iel]])
    xme[iel*3+1]=0.5*(x[icon[1,iel]]+x[icon[2,iel]]) #second edge
    yme[iel*3+1]=0.5*(y[icon[1,iel]]+y[icon[2,iel]])
    xme[iel*3+2]=0.5*(x[icon[2,iel]]+x[icon[0,iel]]) #third edge
    yme[iel*3+2]=0.5*(y[icon[2,iel]]+y[icon[0,iel]])

export_swarm_to_vtu(xme,yme,'midedge_points.vtu')

np.savetxt('midedge_points.ascii',np.array([xme,yme]).T,header='# x,y')

#------------------------------------------------------------------------------
# compute number of duplicates, pointer array and merge
#------------------------------------------------------------------------------

eps=1e-6

double = np.zeros(NVme, dtype=np.bool)  # default is false

for i in range(0,NVme):
    if not double[i]:
       for j in range(0,NVme):
           if j!=i:
              if abs(xme[i]-xme[j])<eps and abs(yme[i]-yme[j])<eps:
                 double[j]=True

#print('double=',double)

NVme_new=NVme-sum(double)
print('real nb of mid edges pts',NVme_new)

NVnew=NV+NVme_new
print('total number of P2 nodes',NVnew)

xV = np.zeros(NVnew,dtype=np.float64)  # x coordinates
yV = np.zeros(NVnew,dtype=np.float64)  # y coordinates
iconV =np.zeros((6,nel),dtype=np.int32)

xV[0:NV]=x[0:NV]
yV[0:NV]=y[0:NV]
xV[NV:NVnew]=xme[np.invert(double)]
yV[NV:NVnew]=yme[np.invert(double)]

counter=0
for iel in range(0,nel):
    iconV[0,iel]=icon[0,iel]
    iconV[1,iel]=icon[1,iel]
    iconV[2,iel]=icon[2,iel]

    xmid=xme[3*iel] #first edge
    ymid=yme[3*iel]
    for i in range(NV,NVnew):
        if abs(xmid-xV[i])<eps and abs(ymid-yV[i])<eps:
           iconV[3,iel]=i
           break

    xmid=xme[3*iel+1] #second edge
    ymid=yme[3*iel+1]
    for i in range(NV,NVnew):
        if abs(xmid-xV[i])<eps and abs(ymid-yV[i])<eps:
           iconV[4,iel]=i
           break

    xmid=xme[3*iel+2] #third edge
    ymid=yme[3*iel+2]
    for i in range(NV,NVnew):
        if abs(xmid-xV[i])<eps and abs(ymid-yV[i])<eps:
           iconV[5,iel]=i
           break

#for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",iconV[0,iel],"at pos.",xV[iconV[0,iel]],' | ',yV[iconV[0,iel]])
#     print ("node 2",iconV[1,iel],"at pos.",xV[iconV[1,iel]],' | ',yV[iconV[1,iel]])
#     print ("node 3",iconV[2,iel],"at pos.",xV[iconV[2,iel]],' | ',yV[iconV[2,iel]])
#     print ("node 4",iconV[3,iel],"at pos.",xV[iconV[3,iel]],' | ',yV[iconV[3,iel]])
#     print ("node 5",iconV[4,iel],"at pos.",xV[iconV[4,iel]],' | ',yV[iconV[4,iel]])
#     print ("node 6",iconV[5,iel],"at pos.",xV[iconV[5,iel]],' | ',yV[iconV[5,iel]])

np.savetxt('meshP2.ascii',np.array([xV,yV]).T,header='# x,y')

export_elements_to_vtuP2(xV,yV,iconV,'meshP2.vtu')

#-------------------------------------------------------------------



NV,xV,yV,iconV=mesh_P1_to_P2(x,y,icon)


export_elements_to_vtuP2(xV,yV,iconV,'meshP2bis.vtu')



