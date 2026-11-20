import numpy as np

def export_borehole(nx,nz,x,y,z,theta,radius,Lx,Ly,Lz,u,v,w,sigmaeff):

    eps=1e-8*Lz
    m=4
    nelx=4*nx
    nelz=nz
    nel=nelx*nelz
    nn=(nelx+1)*(nelz+1) 
    LLx=2*np.pi
    nwall=np.size(x)

    #theta is between -pi (excluded) to +pi included
    #print(np.min(theta),np.max(theta))
    #print(np.min(xx),np.max(xx))
    #print(np.min(u),np.max(u))
    #print(np.min(v),np.max(v))
    #print(np.min(w),np.max(w))

    #######################################################
    #build wall mesh in rectangle [0:2pi,0:Lz]

    xx=np.zeros(nn,dtype=np.float64)
    zz=np.zeros(nn,dtype=np.float64)
    icon=np.zeros((m,nel),dtype=np.int32)

    counter=0
    for j in range(0,nelz+1):
        for i in range(0,nelx+1):
            xx[counter]=-np.pi+i*LLx/float(nelx)
            zz[counter]=j*Lz/float(nelz)
            counter += 1

    counter = 0
    for j in range(0,nelz):
        for i in range(0,nelx):
            icon[0,counter]=i+j*(nelx+1)
            icon[1,counter]=i+1+j*(nelx+1)
            icon[2,counter]=i+1+(j+1)*(nelx+1)
            icon[3,counter]=i+(j+1)*(nelx+1)
            counter += 1

    #######################################################
    # find correspondance between local and global nodes

    number=np.zeros(nn,dtype=np.int32)
    for i in range(0,nn):
        for k in range(nwall):
            if abs(xx[i]-theta[k])<eps and abs(zz[i]-z[k])<eps:
               #print(xx[i]/np.pi*180.,'found it',i,'->',k)
               number[i]=k
               break
            if abs(xx[i]+2*np.pi-theta[k])<eps and abs(zz[i]-z[k])<eps:
               #print(xx[i]/np.pi*180.,'found it',i,'->',k)
               number[i]=k
               break

    #######################################################
    # make data arrays for visualisation

    uu=np.zeros(nn,dtype=np.float64)
    vv=np.zeros(nn,dtype=np.float64)
    ww=np.zeros(nn,dtype=np.float64)
    ss=np.zeros(nn,dtype=np.float64)

    counter=0
    for j in range(0,nelz+1):
        for i in range(0,nelx+1):
            uu[counter]=u[number[counter]]
            vv[counter]=v[number[counter]]
            ww[counter]=w[number[counter]]
            ss[counter]=sigmaeff[number[counter]]

            counter += 1

    #######################################################

    vtufile=open('borehole_2d.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn):
        vtufile.write("%e %e %e \n" %(xx[i]/np.pi*180,zz[i]*1000,0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
    for i in range(0,nn):
        vtufile.write("%e %e %e \n" %(uu[i],vv[i],ww[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma (eff)' Format='ascii'> \n")
    for i in range(0,nn):
        vtufile.write("%e  \n" %(ss[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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

    #######################################################

    xxx=radius*np.cos(xx)+Lx/2
    yyy=radius*np.sin(xx)+Ly/2

    vtufile=open('borehole_3d.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn):
        vtufile.write("%e %e %e \n" %(xxx[i],yyy[i],zz[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
    for i in range(0,nn):
        vtufile.write("%e %e %e \n" %(uu[i],vv[i],ww[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='sigma (eff)' Format='ascii'> \n")
    for i in range(0,nn):
        vtufile.write("%e  \n" %(ss[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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

###############################################################################
