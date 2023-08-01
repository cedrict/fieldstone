import numpy as np

###############################################################################
# inspired by stone 69 - also see GHOST code 
# this function merges two meshes that potentially have different numbers
# of points and elements. It will be fine as long as the side they have in 
# common counts the same number of points at the same location
###############################################################################

def merge_two_blocks(x1,y1,icon1,hull1,x2,y2,icon2,hull2):

    debug=False

    nnp1=np.size(x1)
    nnp2=np.size(x2)
    m,nel1=np.shape(icon1)
    m,nel2=np.shape(icon2)

    tempx=np.zeros(nnp1+nnp2,dtype=np.float64)
    tempy=np.zeros(nnp1+nnp2,dtype=np.float64)
    temphull=np.zeros(nnp1+nnp2,dtype=bool)

    tempx[0:nnp1]=x1[:] 
    tempy[0:nnp1]=y1[:]
    temphull[0:nnp1]=hull1[:]

    tempx[0+nnp1:nnp2+nnp1]=x2[:]
    tempy[0+nnp1:nnp2+nnp1]=y2[:]
    temphull[0+nnp1:nnp2+nnp1]=hull2[:]

    if debug:
       np.savetxt('temp.ascii',np.array([tempx,tempy]).T)

    doubble=np.zeros(nnp1+nnp2,dtype=bool)
    pointto=np.zeros(nnp1+nnp2,dtype=np.int32)

    for i in range(0,nnp1+nnp2):
        pointto[i]=i

    distance=max(max(x1)-min(x1),max(y1)-min(y1))*1e-6
    if debug:
       print(distance)

    counter=0
    for ip in range(1,nnp1+nnp2):
        if temphull[ip]:
           gxip=tempx[ip]
           gyip=tempy[ip]
           for jp in range(0,ip-1):
               if temphull[jp]:
                  if np.abs(gxip-tempx[jp])<distance and \
                     np.abs(gyip-tempy[jp])<distance :
                     doubble[ip]=True
                     pointto[ip]=jp
                     break
                  #end if
               #end if
           #end do
        #end if
    #end for

    together_nnp=nnp1+nnp2-np.count_nonzero(doubble)

    together_nel=nel1+nel2

    #print('count(doubble)=',np.count_nonzero(doubble))
    print('new: nnp=',together_nnp,' ; nel=',together_nel)

    together_x=np.zeros(together_nnp,dtype=np.float64)
    together_y=np.zeros(together_nnp,dtype=np.float64)
    together_icon=np.zeros((m,together_nel),dtype=np.int32)
    together_hull=np.zeros(together_nnp,dtype=bool)

    counter=0
    for ip in range(0,nnp1+nnp2):
        if not doubble[ip]:
           together_x[counter]=tempx[ip]
           together_y[counter]=tempy[ip]
           together_hull[counter]=temphull[ip]
           counter+=1
        #end if
    #end for

    if debug:
       np.savetxt('together_xy.ascii',np.array([together_x,together_y]).T)

    #----- merging icons -----

    ib=1 ; together_icon[0:m,0:nel1]=icon1[:,:]
    ib=2 ; together_icon[0:m,nel1:nel1+nel2]=icon2[:,:]+nnp1

    for iel in range(0,together_nel):
        for i in range(0,m):
            together_icon[i,iel]=pointto[together_icon[i,iel]]
        #end for
    #end for

    compact=np.zeros(nnp1+nnp2,dtype=np.int32)

    counter=0
    for ip in range(0,nnp1+nnp2):
        if not doubble[ip]:
           compact[ip]=counter
           counter+=1
        #end if
    #end for

    for iel in range(0,together_nel):
        for i in range(0,m):
            together_icon[i,iel]=compact[together_icon[i,iel]]
        #end for
    #end for

    return together_x,together_y,together_icon,together_hull

###############################################################################

def export_to_vtu(name,x,y,icon,hull):

    nnp=np.size(x)
    m,nel=np.shape(icon)

    vtufile=open(name,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='hull' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e  \n" % hull[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
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

###############################################################################
