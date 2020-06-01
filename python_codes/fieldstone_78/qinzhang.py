import numpy as np

def mesher(Lx,Ly,nelx,nely,nel,NV,mV):

    xV = np.zeros(NV,dtype=np.float64)  # x coordinates
    yV = np.zeros(NV,dtype=np.float64)  # y coordinates
    iconV =np.zeros((mV,nel),dtype=np.int32)

    sx=Lx/nelx
    sy=Ly/nely

    counter=0    
    for j in range(0,nely+1):
        for i in range(0,nelx+1):    
            xV[counter]=i*sx
            yV[counter]=j*sy
            counter=counter+1    
        #end do 
    #end do   
    np1=counter

    for j in range(0,nely+1):    
        for i in range(1,nelx+1):
          xV[counter]=(i-0.5)*sx
          yV[counter]=j*sy
          counter=counter+1    
       #end do
    #end do
    np2=counter
    
    for j in range(1,nely+1):    
        for i in range(0,nelx+1):    
            xV[counter]=i*sx
            yV[counter]=(j-0.5)*sy
            counter=counter+1    
        #end do
    #end do
    np3=counter

    delta=1./3.

    for j in range(0,nely):    
          for i in range(0,nelx):    
              xV[counter]=(i+0.5)*sx
              yV[counter]=(j+0.5)*sy
              counter=counter+1    

              xV[counter]=(i+0.5)*sx
              yV[counter]=(j+0.5-delta)*sy
              counter=counter+1               #11

              xV[counter]=(i+0.5+delta/2)*sx
              yV[counter]=(j+0.5-delta/2)*sy
              counter=counter+1               #12

              xV[counter]=(i+0.5+delta)*sx
              yV[counter]=(j+0.5)*sy
              counter=counter+1               #13

              xV[counter]=(i+0.5+delta/2)*sx
              yV[counter]=(j+0.5+delta/2)*sy
              counter=counter+1             #14

              xV[counter]=(i+0.5)*sx
              yV[counter]=(j+0.5+delta)*sy
              counter=counter+1             #15

              xV[counter]=(i+0.5-delta/2)*sx
              yV[counter]=(j+0.5+delta/2)*sy
              counter=counter+1             #16

              xV[counter]=(i+0.5-delta)*sx
              yV[counter]=(j+0.5)*sy
              counter=counter+1             #17

              xV[counter]=(i+0.5-delta/2)*sx
              yV[counter]=(j+0.5-delta/2)*sy
              counter=counter+1             #10

        #end do
    #end do

    #np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y,u,v')

    counter=0
    for irowf in range(0,nely):
        for icolf in range(0,nelx):
            i=icolf+1
            j=irowf+1

            inode1=i+(j-1)*(nelx+1)
            inode2=i+1+(j-1)*(nelx+1)
            inode3=i+1+j*(nelx+1)
            inode4=i+j*(nelx+1)
            inode5=np1+i+nelx*(j-1)
            inode6=np2+i+1+(nelx+1)*(j-1)
            inode7=np1+i+nelx*(j)
            inode8=np2+i+(nelx+1)*(j-1)
            inode9 =np3+(i+(j-1)*nelx-1)*9+1
            inode10=np3+(i+(j-1)*nelx-1)*9+2
            inode11=np3+(i+(j-1)*nelx-1)*9+3
            inode12=np3+(i+(j-1)*nelx-1)*9+4
            inode13=np3+(i+(j-1)*nelx-1)*9+5
            inode14=np3+(i+(j-1)*nelx-1)*9+6
            inode15=np3+(i+(j-1)*nelx-1)*9+7
            inode16=np3+(i+(j-1)*nelx-1)*9+8
            inode17=np3+(i+(j-1)*nelx-1)*9+9

            #sub element 1
            iconV[0,counter]=inode1   -1
            iconV[1,counter]=inode5   -1
            iconV[2,counter]=inode10   -1
            iconV[3,counter]=inode17   -1
            counter=counter+1

            #sub element 2
            iconV[0,counter]=inode17   -1
            iconV[1,counter]=inode10   -1
            iconV[2,counter]=inode11   -1
            iconV[3,counter]=inode9   -1
            counter=counter+1

            #sub element 3
            iconV[0,counter]=inode5   -1
            iconV[1,counter]=inode2   -1
            iconV[2,counter]=inode11   -1
            iconV[3,counter]=inode10   -1
            counter=counter+1

            #sub element 4
            iconV[0,counter]=inode11   -1
            iconV[1,counter]=inode2   -1
            iconV[2,counter]=inode6   -1
            iconV[3,counter]=inode12   -1
            counter=counter+1

            #sub element 5
            iconV[0,counter]=inode9  -1
            iconV[1,counter]=inode11  -1
            iconV[2,counter]=inode12  -1
            iconV[3,counter]=inode13  -1
            counter=counter+1

            #sub element 6
            iconV[0,counter]=inode12  -1
            iconV[1,counter]=inode6  -1
            iconV[2,counter]=inode3  -1
            iconV[3,counter]=inode13  -1
            counter=counter+1

            #sub element 7
            iconV[0,counter]=inode14  -1
            iconV[1,counter]=inode13  -1
            iconV[2,counter]=inode3  -1
            iconV[3,counter]=inode7  -1
            counter=counter+1

            #sub element 8
            iconV[0,counter]=inode15  -1
            iconV[1,counter]=inode9  -1
            iconV[2,counter]=inode13  -1
            iconV[3,counter]=inode14  -1
            counter=counter+1

            #sub element 9
            iconV[0,counter]=inode4  -1
            iconV[1,counter]=inode15  -1
            iconV[2,counter]=inode14  -1
            iconV[3,counter]=inode7  -1
            counter=counter+1

            #sub element 10
            iconV[0,counter]=inode8  -1
            iconV[1,counter]=inode16  -1
            iconV[2,counter]=inode15  -1
            iconV[3,counter]=inode4  -1
            counter=counter+1

            #sub element 11
            iconV[0,counter]=inode16  -1
            iconV[1,counter]=inode17  -1
            iconV[2,counter]=inode9  -1
            iconV[3,counter]=inode15  -1
            counter=counter+1

            #sub element 12
            iconV[0,counter]=inode1  -1
            iconV[1,counter]=inode17  -1
            iconV[2,counter]=inode16  -1
            iconV[3,counter]=inode8  -1
            counter=counter+1

        #end for
    #end for

    return xV,yV,iconV
