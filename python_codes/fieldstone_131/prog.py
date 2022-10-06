import numpy as np
import triangle as tr

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

    #create mid-edge nodes
    NVme=3*nel   
    xme = np.empty(NVme,dtype=np.float64) 
    yme = np.empty(NVme,dtype=np.float64) 
    for iel in range(0,nel):
        xme[iel*3+0]=0.5*(x[icon[0,iel]]+x[icon[1,iel]]) #first edge
        yme[iel*3+0]=0.5*(y[icon[0,iel]]+y[icon[1,iel]])
        xme[iel*3+1]=0.5*(x[icon[1,iel]]+x[icon[2,iel]]) #second edge
        yme[iel*3+1]=0.5*(y[icon[1,iel]]+y[icon[2,iel]])
        xme[iel*3+2]=0.5*(x[icon[2,iel]]+x[icon[0,iel]]) #third edge
        yme[iel*3+2]=0.5*(y[icon[2,iel]]+y[icon[0,iel]])

    eps=1e-6*min(max(x)-min(x),max(y)-min(y))

    #find out which nodes are present twice
    double = np.zeros(NVme, dtype=np.bool)  # default is false
    for i in range(0,NVme):
        if not double[i]:
           for j in range(0,NVme):
               if j!=i:
                  if abs(xme[i]-xme[j])<eps and abs(yme[i]-yme[j])<eps:
                     double[j]=True

    #compute real nb of mid-edges nodes
    NVme_new=NVme-sum(double)

    #compute total nb of nodes
    NVnew=NV+NVme_new

    xV = np.zeros(NVnew,dtype=np.float64)  
    yV = np.zeros(NVnew,dtype=np.float64)  
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

        #first edge
        for i in range(NV,NVnew):
            if abs(xme[3*iel]-xV[i])<eps and abs(yme[3*iel]-yV[i])<eps:
               iconV[3,iel]=i
               break

        #second edge
        for i in range(NV,NVnew):
            if abs(xme[3*iel+1]-xV[i])<eps and abs(yme[3*iel+1]-yV[i])<eps:
               iconV[4,iel]=i
               break

        #third edge
        for i in range(NV,NVnew):
            if abs(xme[3*iel+2]-xV[i])<eps and abs(yme[3*iel+2]-yV[i])<eps:
               iconV[5,iel]=i
               break

    return NVnew,xV,yV,iconV

#------------------------------------------------------------------------------
# define original P1 mesh
#------------------------------------------------------------------------------

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

export_elements_to_vtuP1(x,y,icon,'meshP1.vtu')

np.savetxt('meshP1.ascii',np.array([x,y]).T,header='# x,y')

NV,xV,yV,iconV=mesh_P1_to_P2(x,y,icon)

export_elements_to_vtuP2(xV,yV,iconV,'meshP2.vtu')

