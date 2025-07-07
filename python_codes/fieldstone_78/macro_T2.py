import numpy as np

def mesher(Lx,Ly,nelx,nely,nel,NV,mV):

    xV = np.zeros(NV,dtype=np.float64)  # x coordinates
    yV = np.zeros(NV,dtype=np.float64)  # y coordinates
    iconV =np.zeros((mV,nel),dtype=np.int32)

    
# 4--------3  |--------|
# |\      /|  |\   3  /|
# | \8--7/ |  | \----/ |
# |  |  |  |  |4|  5 |2|
# | /5--6\ |  | /----\ |
# |/      \|  |/  1   \|
# 1--------2  |--------|


    # nodes 1,2,3,4
    counter=0    
    for j in range(0,nely+1):
        for i in range(0,nelx+1):
            xV[counter]=i*Lx/nelx 
            yV[counter]=j*Ly/nely
            counter+=1    

    hx=Lx/nelx
    hy=Ly/nely
    eps=hx/2/np.sqrt(5)

    # nodes 5,6,7,8
    for j in range(0,nely):
        for i in range (0,nelx):
            inode1=i+j*(nelx+1)
            x1=xV[inode1]+hx/2
            y1=yV[inode1]+hy/2
            xV[counter]=x1-eps
            yV[counter]=y1-eps
            counter=counter+1    
            xV[counter]=x1+eps
            yV[counter]=y1-eps
            counter=counter+1    
            xV[counter]=x1+eps
            yV[counter]=y1+eps
            counter=counter+1    
            xV[counter]=x1-eps
            yV[counter]=y1+eps
            counter=counter+1    

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
            inode5=(nelx+1)*(nely+1)+(nelx*(j)+i)*4+1-1
            inode6=(nelx+1)*(nely+1)+(nelx*(j)+i)*4+2-1
            inode7=(nelx+1)*(nely+1)+(nelx*(j)+i)*4+3-1
            inode8=(nelx+1)*(nely+1)+(nelx*(j)+i)*4+4-1

            #sub element 1
            iconV[0,counter]=inode1
            iconV[1,counter]=inode2
            iconV[2,counter]=inode6
            iconV[3,counter]=inode5
            counter=counter+1
            #sub element 2
            iconV[0,counter]=inode6
            iconV[1,counter]=inode2
            iconV[2,counter]=inode3
            iconV[3,counter]=inode7
            counter=counter+1
            #sub element 3
            iconV[0,counter]=inode8
            iconV[1,counter]=inode7
            iconV[2,counter]=inode3
            iconV[3,counter]=inode4
            counter=counter+1
            #sub element 4
            iconV[0,counter]=inode1
            iconV[1,counter]=inode5
            iconV[2,counter]=inode8
            iconV[3,counter]=inode4
            counter=counter+1
            #sub element 5
            iconV[0,counter]=inode5
            iconV[1,counter]=inode6
            iconV[2,counter]=inode7
            iconV[3,counter]=inode8
            counter=counter+1

    return xV,yV,iconV
