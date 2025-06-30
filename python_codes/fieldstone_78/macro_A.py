import numpy as np

def mesher(Lx,Ly,nelx,nely,nel,NV,mV):

    xV = np.zeros(NV,dtype=np.float64)  # x coordinates
    yV = np.zeros(NV,dtype=np.float64)  # y coordinates
    iconV =np.zeros((mV,nel),dtype=np.int32)

    # nodes 1,2,3,4
    counter=0    
    for j in range(0,nely+1):
        for i in range(0,nelx+1):
            xV[counter]=i*Lx/nelx 
            yV[counter]=j*Ly/nely
            counter+=1    

    np1=counter

    hx=Lx/nelx
    hy=Ly/nely
    eps=hx/2/np.sqrt(5)

    for j in range(0,nely+1):
        for i in range (0,nelx):
            xV[counter]=((i+1)-0.5)*hx
            yV[counter]=(j)*hy
            counter=counter+1    
    np2=counter
    
    for j in range(0,nely):
        for i in range (0,nelx+1):
            xV[counter]=(i)*hx
            yV[counter]=((j+1)-0.5)*hy
            counter=counter+1    
    np3=counter

    for j in range(0,nely):
        for i in range (0,nelx):
            xV[counter]=(i+0.3)*hx
            yV[counter]=(j+0.3)*hy
            counter=counter+1    
            xV[counter]=(i+0.7)*hx
            yV[counter]=(j+0.3)*hy
            counter=counter+1    
            xV[counter]=(i+0.7)*hx
            yV[counter]=(j+0.7)*hy
            counter=counter+1    
            xV[counter]=(i+0.3)*hx
            yV[counter]=(j+0.7)*hy
            counter=counter+1    


    #np.savetxt('velocity.ascii',np.array([xV,yV]).T)


    ###############################################################################
    # computing grid connectivity 
    ###############################################################################

    counter=0
    for j in range(0,nely):
        for i in range(0,nelx):

            inode1=i+(j)*(nelx+1)
            inode2=i+1+(j)*(nelx+1)
            inode3=i+1+(j+1)*(nelx+1)
            inode4=i+(j+1)*(nelx+1)

            inode5=np1+i+nelx*(j)
            inode6=np2+i+1+(nelx+1)*(j)
            inode7=np1+i+nelx*(j+1)
            inode8=np2+i+(nelx+1)*(j)

            inode9 =np3+(i+(j)*nelx)*4+0
            inode10=np3+(i+(j)*nelx)*4+1
            inode11=np3+(i+(j)*nelx)*4+2
            inode12=np3+(i+(j)*nelx)*4+3

            #print('---------------')
            #print(inode1+1,inode2+1,inode3+1,inode4+1,inode5+1,inode6+1,inode7+1,inode8+1)
            #print(inode9+1,inode10+1,inode11+1,inode12+1)

            #sub element 1
            iconV[0,counter]=inode1
            iconV[1,counter]=inode5
            iconV[2,counter]=inode10
            iconV[3,counter]=inode9
            counter=counter+1

            #sub element 2
            iconV[0,counter]=inode5
            iconV[1,counter]=inode2
            iconV[2,counter]=inode6
            iconV[3,counter]=inode10
            counter=counter+1

            #sub element 3
            iconV[0,counter]=inode10
            iconV[1,counter]=inode6
            iconV[2,counter]=inode3
            iconV[3,counter]=inode11
            counter=counter+1

            #sub element 4
            iconV[0,counter]=inode12
            iconV[1,counter]=inode11
            iconV[2,counter]=inode3
            iconV[3,counter]=inode7
            counter=counter+1

            #sub element 5
            iconV[0,counter]=inode8
            iconV[1,counter]=inode12
            iconV[2,counter]=inode7
            iconV[3,counter]=inode4
            counter=counter+1

            #sub element 6
            iconV[0,counter]=inode1
            iconV[1,counter]=inode9
            iconV[2,counter]=inode12
            iconV[3,counter]=inode8
            counter=counter+1

            #sub element 7
            iconV[0,counter]=inode9
            iconV[1,counter]=inode10
            iconV[2,counter]=inode11
            iconV[3,counter]=inode12
            counter=counter+1

    return xV,yV,iconV
