import numpy as np

def mesher(Lx,Ly,nelx,nely,nel,NV,mV):

    xV = np.zeros(NV,dtype=np.float64)  # x coordinates
    yV = np.zeros(NV,dtype=np.float64)  # y coordinates
    iconV =np.zeros((mV,nel),dtype=np.int32)

    counter = 0
    for j in range(0,nely+1):
        for i in range(0,nelx+1):
            xV[counter]=i*Lx/float(nelx)
            yV[counter]=j*Ly/float(nely)
            counter += 1

    counter = 0
    for j in range(0, nely):
        for i in range(0, nelx):
            iconV[0, counter] = i + j * (nelx + 1)
            iconV[1, counter] = i + 1 + j * (nelx + 1)
            iconV[2, counter] = i + 1 + (j + 1) * (nelx + 1)
            iconV[3, counter] = i + (j + 1) * (nelx + 1)
            counter += 1

    return xV,yV,iconV
