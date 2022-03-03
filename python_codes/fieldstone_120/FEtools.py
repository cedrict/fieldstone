import numpy as np

#------------------------------------------------------------------------------

def cartesian_mesh(Lx,Ly,nelx,nely,element):

    hx=Lx/nelx
    hy=Ly/nely
    nel=nelx*nely

    if element=='Q0' or element=='P0':
       N=nelx*nely
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=(i+0.5)*hx
               y[counter]=(j+0.5)*hy
               counter += 1
       icon =np.zeros((1,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter]=counter
               counter += 1
       icon2 =np.zeros((1,nel),dtype=np.int32)
       icon2[:]=icon[:]

    if element=='Q1':
       N=(nelx+1)*(nely+1)
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely+1):
           for i in range(0,nelx+1):
               x[counter]=i*hx
               y[counter]=j*hy
               counter += 1
       icon =np.zeros((4,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter]=i+j*(nelx+1)
               icon[1,counter]=i+1+j*(nelx+1)
               icon[2,counter]=i+1+(j+1)*(nelx+1)
               icon[3,counter]=i+(j+1)*(nelx+1)
               counter += 1
       icon2 =np.zeros((4,nel),dtype=np.int32)
       icon2[:]=icon[:]

    if element=='Q2':
       N=(2*nelx+1)*(2*nely+1)
       nnx=2*nelx+1
       nny=2*nely+1
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nny):
           for i in range(0,nnx):
               x[counter]=i*hx/2.
               y[counter]=j*hy/2.
               counter += 1
           #end for
       #end for
       icon=np.zeros((9,nel),dtype=np.int32)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter]=(i)*2+1+(j)*2*nnx -1
               icon[1,counter]=(i)*2+3+(j)*2*nnx -1
               icon[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
               icon[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
               icon[4,counter]=(i)*2+2+(j)*2*nnx -1
               icon[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
               icon[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
               icon[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
               icon[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
               counter += 1
           #end for
       #end for
       icon2 =np.zeros((4,4*nel),dtype=np.int32)
       counter = 0 
       for j in range(0,2*nely):
           for i in range(0,2*nelx):
               icon2[0,counter]=i+j*(2*nelx+1)
               icon2[1,counter]=i+1+j*(2*nelx+1)
               icon2[2,counter]=i+1+(j+1)*(2*nelx+1)
               icon2[3,counter]=i+(j+1)*(2*nelx+1)
               counter += 1



    if element=='Q3':
       exit('Q3 not implemented')

    if element=='Q4':
       exit('Q4 not implemented')

    return N,nel,x,y,icon,icon2

#------------------------------------------------------------------------------

def export_mesh_to_ascii(x,y,filename):
    np.savetxt(filename,np.array([x,y]).T,header='# x,y')

#------------------------------------------------------------------------------

def export_connectivity_array_to_ascii(x,y,icon,filename):
    m,nel=np.shape(icon)
    iconfile=open(filename,"w")
    for iel in range (0,nel):
        iconfile.write('--------'+str(iel)+'-------\n')
        for k in range(0,m):
            iconfile.write("node "+str(k)+' | '+str(icon[k,iel])+" at pos. "+str(x[icon[k,iel]])+','+str(y[icon[k,iel]])+'\n')

#------------------------------------------------------------------------------

def export_mesh_to_vtu(x,y,icon,element,filename):
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
        vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()


def bc_setup(x,y,Lx,Ly,ndof,left,right,bottom,top):
    eps=1e-8
    N=np.size(x)
    Nfem=2*N
    bc_fix = np.zeros(Nfem, dtype=np.bool)     # boundary condition, yes/no
    bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value
    for i in range(0,N):

        if x[i]/Lx<eps:
           if left=='free_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
           if left=='no_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

        if x[i]/Lx>(1-eps):
           if right=='free_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
           if right=='no_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

        if y[i]/Ly<eps:
           if bottom=='free_slip':
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if bottom=='no_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

        if y[i]/Ly>(1-eps):
           if top=='free_slip':
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if top=='no_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

    return bc_fix,bc_val

#------------------------------------------------------------------------------

def J(m,dNdr,dNds,x,y):
    jcb = np.zeros((2,2),dtype=np.float64)
    jcb[0,0] = dNdr.dot(x)
    jcb[0,1] = dNdr.dot(y)
    jcb[1,0] = dNds.dot(x)
    jcb[1,1] = dNds.dot(y)
    jcbi=np.linalg.inv(jcb)
    jcob=np.linalg.det(jcb)
    dNdx= np.zeros(m,dtype=np.float64)
    dNdy= np.zeros(m,dtype=np.float64)
    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]
    return jcob,jcbi,dNdx,dNdy

#------------------------------------------------------------------------------
