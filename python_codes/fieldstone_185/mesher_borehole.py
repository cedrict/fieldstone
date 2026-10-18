import numpy as np

###############################################################################

def basis_functions_2d(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)


def mesher_borehole(Lx,Ly,Lz,nelx,nely,nelz,radius):

    hz=Lz/nelz

    sqrt2=np.sqrt(2.)
    eps=1e-8
    distance=eps*Lx

    debug=True

    ###############################################################################
    # The mesh in the xy plane is composed of eight blocks. Each is first built 
    # and warped to fit the contour of the hole and they are subsequently merged.
    #
    #  G-----E-----C
    #  |   3 | 2   |
    #  |     F     |
    #  | 4  H D  1 |
    #  I---J   A---B
    #  | 5  L P  8 |
    #  |     N     |
    #  |   6 | 7   |
    #  K-----M-----O
    #
    ###############################################################################

    # Lx and Ly are the sides of the domain above. However in what follows 
    # we will need their half values, i.e. Lx2, Ly2

    Lx2=Lx/2
    Ly2=Ly/2
 
    nelx2=int(nelx/2)
    nely2=int(nely/2)

    b_nelx=nelx2
    b_nely=nely2
    b_nnx=nelx2+1
    b_nny=nely2+1
    b_NV=b_nnx*b_nny
    b_nel=b_nelx*b_nely

    b1_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
    b1_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
    b2_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
    b2_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
    b3_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
    b3_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
    b4_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
    b4_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
    b5_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
    b5_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
    b6_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
    b6_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
    b7_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
    b7_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
    b8_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
    b8_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates

    counter = 0
    for j in range(0,b_nny):
        for i in range(0,b_nnx):
            b1_x[counter]=i*Lx2/float(b_nelx)
            b1_y[counter]=j*Ly2/float(b_nely)
            counter += 1

    b2_x[:]=b1_x[:]
    b2_y[:]=b1_y[:]

    b1_icon=np.zeros((4,b_nel),dtype=np.int32)
    b2_icon=np.zeros((4,b_nel),dtype=np.int32)
    b3_icon=np.zeros((4,b_nel),dtype=np.int32)
    b4_icon=np.zeros((4,b_nel),dtype=np.int32)
    b5_icon=np.zeros((4,b_nel),dtype=np.int32)
    b6_icon=np.zeros((4,b_nel),dtype=np.int32)
    b7_icon=np.zeros((4,b_nel),dtype=np.int32)
    b8_icon=np.zeros((4,b_nel),dtype=np.int32)

    counter = 0
    for j in range(0,b_nely):
        for i in range(0,b_nelx):
            b1_icon[0, counter] = i + j * (b_nelx + 1)
            b1_icon[1, counter] = i + 1 + j * (b_nelx + 1)
            b1_icon[2, counter] = i + 1 + (j + 1) * (b_nelx + 1)
            b1_icon[3, counter] = i + (j + 1) * (b_nelx + 1)
            counter += 1

    b2_icon[:,:]=b1_icon[:,:]
    b3_icon[:,:]=b1_icon[:,:]
    b4_icon[:,:]=b1_icon[:,:]
    b5_icon[:,:]=b1_icon[:,:]
    b6_icon[:,:]=b1_icon[:,:]
    b7_icon[:,:]=b1_icon[:,:]
    b8_icon[:,:]=b1_icon[:,:]

    ###############################################################################
    # map block 1
    ###############################################################################

    xA=radius
    yA=0

    xB=Lx2
    yB=0

    xC=Lx2
    yC=Ly2

    xD=radius/sqrt2
    yD=radius/sqrt2

    # map to quadrilateral ABCD
    counter=0
    for j in range(0,b_nny):
        for i in range(0,b_nnx):
            r=2*(b1_x[counter]/Lx2-0.5)
            s=2*(b1_y[counter]/Ly2-0.5)
            N_V=basis_functions_2d(r,s)
            b1_x[counter]=N_V[0]*xA+N_V[1]*xB+N_V[2]*xC+N_V[3]*xD
            b1_y[counter]=N_V[0]*yA+N_V[1]*yB+N_V[2]*yC+N_V[3]*yD
            counter+=1
       #end for
    #end for

    # bend AD side
    counter=0
    for j in range(0,b_nny):
        for i in range(0,b_nnx):
            if i==0:
               angle=np.arctan(b1_y[counter]/b1_x[counter])
               b1_x[counter]=radius*np.cos(angle)
               b1_y[counter]=radius*np.sin(angle)
            counter+=1
       #end for
    #end for

    # recompute position of inside nodes
    counter=0
    for j in range(0,b_nny):
        for i in range(0,b_nnx):
            if i!=0 and i!=b_nnx-1 and j!=0 and j!=b_nny-1:
               inode_beg=j*b_nnx
               inode_end=j*b_nnx+b_nnx-1
               b1_x[counter]=float(i)/b_nelx*(b1_x[inode_end]-b1_x[inode_beg]) + b1_x[inode_beg]
               b1_y[counter]=float(i)/b_nelx*(b1_y[inode_end]-b1_y[inode_beg]) + b1_y[inode_beg]
            counter+=1
       #end for
    #end for

    ###############################################################################
    # map block 2
    ###############################################################################

    xC=Lx2
    yC=Ly2

    xD=radius/sqrt2
    yD=radius/sqrt2

    xF=0
    yF=radius

    xE=0
    yE=Lx2

    # map to quadrilateral FDCE
    counter=0
    for j in range(0,b_nny):
        for i in range(0,b_nnx):
            r=2*(b2_x[counter]/Lx2-0.5)
            s=2*(b2_y[counter]/Ly2-0.5)
            N_V=basis_functions_2d(r,s)
            b2_x[counter]=N_V[0]*xD+N_V[1]*xC+N_V[2]*xE+N_V[3]*xF
            b2_y[counter]=N_V[0]*yD+N_V[1]*yC+N_V[2]*yE+N_V[3]*yF
            counter+=1
       #end for
    #end for

    #bend FD side
    counter=0
    for j in range(0,b_nny):
        for i in range(0,b_nnx):
            if i==0 and j<b_nny-1:
               angle=np.arctan(b2_y[counter]/b2_x[counter])
               b2_x[counter]=radius*np.cos(angle)
               b2_y[counter]=radius*np.sin(angle)
            counter+=1
       #end for
    #end for

    # recompute position of inside nodes
    counter=0
    for j in range(0,b_nny):
        for i in range(0,b_nnx):
            if i!=0 and i!=b_nnx-1 and j!=0 and j!=b_nny-1:
               inode_beg=j*b_nnx
               inode_end=j*b_nnx+b_nnx-1
               b2_x[counter]=float(i)/b_nelx*(b2_x[inode_end]-b2_x[inode_beg]) + b2_x[inode_beg]
               b2_y[counter]=float(i)/b_nelx*(b2_y[inode_end]-b2_y[inode_beg]) + b2_y[inode_beg]
            counter+=1
       #end for
    #end for

    ###############################################################################
    # make block 3 - it is a rotated block 1
    # make block 4 - it is a rotated block 2
    # make block 5 - it is a rotated block 3
    # make block 6 - it is a rotated block 4
    # make block 7 - it is a rotated block 5
    # make block 8 - it is a rotated block 6
    ###############################################################################

    b3_x[:]=-b1_y[:]
    b3_y[:]= b1_x[:]

    b4_x[:]=-b2_y[:]
    b4_y[:]= b2_x[:]

    b5_x[:]=-b3_y[:]
    b5_y[:]= b3_x[:]

    b6_x[:]=-b4_y[:]
    b6_y[:]= b4_x[:]

    b7_x[:]=-b5_y[:]
    b7_y[:]= b5_x[:]

    b8_x[:]=-b6_y[:]
    b8_y[:]= b6_x[:]

    if debug:
       np.savetxt('temp1.ascii',np.array([b1_x,b1_y]).T,header='# x,y')
       np.savetxt('temp2.ascii',np.array([b2_x,b2_y]).T,header='# x,y')
       np.savetxt('temp3.ascii',np.array([b3_x,b3_y]).T,header='# x,y')
       np.savetxt('temp4.ascii',np.array([b4_x,b4_y]).T,header='# x,y')
       np.savetxt('temp5.ascii',np.array([b5_x,b5_y]).T,header='# x,y')
       np.savetxt('temp6.ascii',np.array([b6_x,b6_y]).T,header='# x,y')
       np.savetxt('temp7.ascii',np.array([b7_x,b7_y]).T,header='# x,y')
       np.savetxt('temp8.ascii',np.array([b8_x,b8_y]).T,header='# x,y')

    ###############################################################################
    # merge blocks
    ###############################################################################

    nblock=8

    tempx=np.zeros(nblock*b_NV,dtype=np.float64)  # x coordinates
    tempy=np.zeros(nblock*b_NV,dtype=np.float64)  # y coordinates

    tempx[0*b_NV:1*b_NV]=b1_x[:]
    tempx[1*b_NV:2*b_NV]=b2_x[:]
    tempx[2*b_NV:3*b_NV]=b3_x[:]
    tempx[3*b_NV:4*b_NV]=b4_x[:]
    tempx[4*b_NV:5*b_NV]=b5_x[:]
    tempx[5*b_NV:6*b_NV]=b6_x[:]
    tempx[6*b_NV:7*b_NV]=b7_x[:]
    tempx[7*b_NV:8*b_NV]=b8_x[:]

    tempy[0*b_NV:1*b_NV]=b1_y[:]
    tempy[1*b_NV:2*b_NV]=b2_y[:]
    tempy[2*b_NV:3*b_NV]=b3_y[:]
    tempy[3*b_NV:4*b_NV]=b4_y[:]
    tempy[4*b_NV:5*b_NV]=b5_y[:]
    tempy[5*b_NV:6*b_NV]=b6_y[:]
    tempy[6*b_NV:7*b_NV]=b7_y[:]
    tempy[7*b_NV:8*b_NV]=b8_y[:]

    doubble=np.zeros(nblock*b_NV,dtype=bool)  # boundary condition, yes/no
    pointto=np.zeros(nblock*b_NV,dtype=np.int32)

    for i in range(0,nblock*b_NV):
        pointto[i]=i

    counter=0
    for i in range(1,nblock*b_NV):
       gxi=tempx[i]
       gyi=tempy[i]
       for j in range(0,i):
          if abs(gxi-tempx[j])<distance and abs(gyi-tempy[j])<distance:
              doubble[i]=True
              pointto[i]=j
              break
          #end if
       #end do
    #end do

    nn_V=nblock*b_NV-sum(doubble)
    nel=nblock*b_nel

    print('     -> doubles=',sum(doubble))
    print('     -> nn_V=',nn_V)
    print('     -> nel=',nel)

    x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
    y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
    icon_V=np.zeros((4,nel),dtype=np.int32)

    counter=0
    for i in range(0,nblock*b_NV):
        if not doubble[i]: 
           x_V[counter]=tempx[i]+Lx2
           y_V[counter]=tempy[i]+Ly2
           counter+=1

    m_V=4

    icon_V[0:m_V,0*b_nel:1*b_nel]=b1_icon[0:m_V,0:b_nel]+0*b_NV
    icon_V[0:m_V,1*b_nel:2*b_nel]=b2_icon[0:m_V,0:b_nel]+1*b_NV
    icon_V[0:m_V,2*b_nel:3*b_nel]=b2_icon[0:m_V,0:b_nel]+2*b_NV
    icon_V[0:m_V,3*b_nel:4*b_nel]=b2_icon[0:m_V,0:b_nel]+3*b_NV
    icon_V[0:m_V,4*b_nel:5*b_nel]=b2_icon[0:m_V,0:b_nel]+4*b_NV
    icon_V[0:m_V,5*b_nel:6*b_nel]=b2_icon[0:m_V,0:b_nel]+5*b_NV
    icon_V[0:m_V,6*b_nel:7*b_nel]=b2_icon[0:m_V,0:b_nel]+6*b_NV
    icon_V[0:m_V,7*b_nel:8*b_nel]=b2_icon[0:m_V,0:b_nel]+7*b_NV

    for iel in range(0,nel):
        for i in range(0,m_V):
            icon_V[i,iel]=pointto[icon_V[i,iel]]

    compact=np.zeros(nblock*b_NV,dtype=np.int32)

    counter=0
    for i in range(0,nblock*b_NV):
        if not doubble[i]:
           compact[i]=counter
           counter=counter+1

    for iel in range(0,nel):
       for i in range(0,m_V):
          icon_V[i,iel]=compact[icon_V[i,iel]]

    if debug: np.savetxt('mesh.ascii',np.array([x_V,y_V]).T,header='# x,y')

    #Lx*=2
    #Ly*=2

    ###############################################################################

    vtufile=open('mesh2d.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
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
    # extrude 2d mesh to make 3d volume
    ###############################################################################

    x_2d=np.copy(x_V)
    y_2d=np.copy(y_V)
    icon_2d=np.copy(icon_V)
    nn_2d=nn_V
    nel_2d=nel

    m_V=8
    nel=nel_2d*nelz
    nn_V=nn_2d*(nelz+1)

    print('     -> nn_V=',nn_V)
    print('     -> nel=',nel)
    print('     -> hz=',hz)

    x_V=np.zeros(nn_V,dtype=np.float64) 
    y_V=np.zeros(nn_V,dtype=np.float64) 
    z_V=np.zeros(nn_V,dtype=np.float64) 
    icon_V=np.zeros((m_V,nel),dtype=np.int32)

    for i in range(0,nelz+1):
        x_V[i*nn_2d:(i+1)*nn_2d]=x_2d[:]
        y_V[i*nn_2d:(i+1)*nn_2d]=y_2d[:]
        z_V[i*nn_2d:(i+1)*nn_2d]=i*hz

    if debug: np.savetxt('nodes.ascii',np.array([x_V,y_V,z_V]).T,header='# x,y,z')

    counter=0
    for i in range(0,nel_2d):
        for k in range(0,nelz):
            icon_V[0:4,counter]=icon_2d[0:4,i]+nn_2d*k
            icon_V[4:8,counter]=icon_2d[0:4,i]+nn_2d*(k+1)
            counter+=1

    return x_V,y_V,z_V,icon_V,nn_V,nel


