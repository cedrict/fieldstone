import numpy as np
import time as timing



##################################################################################################

def export_elements_to_vtu(x,y,icon,filename):
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

##################################################################################################

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

##################################################################################################
# Make Triangle mesh
##################################################################################################
def make_P1_mesh(nnt,nnr,np_blob,R_outer,R_disc1,R_disc2,R_disc3,R_inner,R_blob,z_blob):
    ###############################################################################
    #############  Defining the nodes and vertices ################################
    ###############################################################################
    start = timing.time()

    #------------------------------------------------------------------------------
    # inner boundary counterclockwise
    #------------------------------------------------------------------------------
    theta = np.linspace(-np.pi*0.5, 0.5*np.pi,nnt, endpoint=False)          #half inner sphere in the x-positive domain
    pts_ib = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R_inner     #nnt-points on inner boundary
    seg_ib = np.stack([np.arange(nnt), np.arange(nnt) + 1], axis=1)         #vertices on innerboundary (and the last vertices to the upper wall)
    for i in range(0,nnt):                                                  #first point must be exactly on the y-axis
        if i==0:
           pts_ib[i,0]=0

    #------------------------------------------------------------------------------
    # top vertical (left) wall 
    #------------------------------------------------------------------------------
    topw_z = np.linspace(R_inner,R_outer,nnr,endpoint=False)                #vertical boundary wall from inner to outer boundary sphere
    pts_topw = np.stack([np.zeros(nnr),topw_z],axis=1)                      #nnr-points on vertical wall
    seg_topw = np.stack([nnt+np.arange(nnr),nnt+np.arange(nnr)+1], axis=1)  #vertices on vertical wall

    #------------------------------------------------------------------------------
    # outer boundary clockwise
    #------------------------------------------------------------------------------
    theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=False)            #half outer sphere in the x-positive domain
    pts_ob = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_outer        #nnt-points on outer boundary
    seg_ob = np.stack([nnr+nnt+np.arange(nnt), nnr+nnt+np.arange(nnt)+1], axis=1) #vertices on outerboundary
    for i in range(0,nnt):                                                  #first point must be exactly on the y-axis
        if i==0:
           pts_ob[i,0]=0

    #------------------------------------------------------------------------------
    # bottom vertical wall
    #------------------------------------------------------------------------------
    botw_z = np.linspace(-R_outer,-R_inner,nnr,endpoint=False)              #vertical boundary wall from outer to inner boundary sphere
    pts_botw = np.stack([np.zeros(nnr),botw_z],axis=1)                      #nnr-points on vertical wall
    seg_botw = np.stack([2*nnt+nnr+np.arange(nnr),2*nnt+nnr+np.arange(nnr)+1], axis=1) #vertices on bottem vertical wall
    seg_botw[-1,1]=0                                                        #stitch last point to first point with last vertice

    #------------------------------------------------------------------------------
    # blob 
    #------------------------------------------------------------------------------
    theta_bl = np.linspace(-np.pi/2,np.pi/2,num=np_blob,endpoint=True,dtype=np.float64) #half-sphere in the x and y positive domain
    pts_bl = np.stack([R_blob*np.cos(theta_bl),z_blob+R_blob*np.sin(theta_bl)], axis=1) #points on blob outersurface 
    seg_bl = np.stack([2*nnt+2*nnr+np.arange(np_blob-1), 2*nnt+2*nnr+np.arange(np_blob-1)+1], axis=1) #vertices on outersurface blob
    for i in range(0,np_blob):                                              #first and last point must be exactly on the y-axis.
        if i==0 or i==np_blob-1:
           pts_bl[i,0]=0

    #------------------------------------------------------------------------------
    # discontinuity #1
    #------------------------------------------------------------------------------
    theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain
    pts_mo1 = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_disc1        #nnt-points on outer boundary
    seg_mo1 = np.stack([2*nnt+2*nnr+np_blob+np.arange(nnt-1), 2*nnt+2*nnr+np_blob+np.arange(nnt-1)+1], axis=1) #vertices on disc1
    for i in range(0,nnt):                                                 #first and last point must be exactly on the y-axis
        if i==0 or i==nnt-1:
           pts_mo1[i,0]=0    

    #------------------------------------------------------------------------------
    # discontinuity #2
    #------------------------------------------------------------------------------
    theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain
    pts_mo2 = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_disc2        #nnt-points on outer boundary
    seg_mo2 = np.stack([3*nnt+2*nnr+np_blob+np.arange(nnt-1), 3*nnt+2*nnr+np_blob+np.arange(nnt-1)+1], axis=1) #vertices on disc2
    for i in range(0,nnt):                                                 #first and last point must be exactly on the y-axis
        if i==0 or i==nnt-1:
           pts_mo2[i,0]=0    

    #------------------------------------------------------------------------------
    # discontinuity #3
    #------------------------------------------------------------------------------
    theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain
    pts_mo3 = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_disc3        #nnt-points on outer boundary
    seg_mo3 = np.stack([4*nnt+2*nnr+np_blob+np.arange(nnt-1), 4*nnt+2*nnr+np_blob+np.arange(nnt-1)+1], axis=1) #vertices on disc3
    for i in range(0,nnt):                                                 #first and last point must be exactly on the y-axis
        if i==0 or i==nnt-1:
           pts_mo3[i,0]=0    

    # Stacking the nodes and vertices 

    seg = np.vstack([seg_ib,seg_topw,seg_ob,seg_botw,seg_bl,seg_mo1,seg_mo2,seg_mo3])
    pts = np.vstack([pts_ib,pts_topw,pts_ob,pts_botw,pts_bl,pts_mo1,pts_mo2,pts_mo3]) 

    #put all segments and nodes in a dictionary

    dict_nodes = dict(vertices=pts, segments=seg,holes=[[0,0]]) #no core so we add a hole at x=0,y=0

    print("generate nodes: %.3f s" % (timing.time() - start))

    


##################################################################################################
# convert P1 mesh into P2 mesh
##################################################################################################

def mesh_P1_to_P2(NV,nel,x,y,icon):
    #NV=np.size(x)
    #m,nel=np.shape(icon)

    #create mid-edge nodes
    start = timing.time()
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
    print("...P1->P2 (a): %.3f s" % (timing.time() - start))

    eps=1e-6*min(max(x)-min(x),max(y)-min(y))

    #find out which nodes are present twice
    start = timing.time()
    double = np.zeros(NVme, dtype=np.bool)  # default is false
    for i in range(0,NVme):
        if not double[i]:
           for j in range(0,NVme):
               if j!=i:
                  if abs(xme[i]-xme[j])<eps and abs(yme[i]-yme[j])<eps:
                     double[j]=True
    print("...P1->P2 (b): %.3f s" % (timing.time() - start))

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

    start = timing.time()
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
    print("...P1->P2 (c): %.3f s" % (timing.time() - start))

    return NVnew,xV,yV,iconV
