import numpy as np

def mesher_cube(Lx,Ly,Lz,nelx,nely,nelz):

    nnx=nelx+1 
    nny=nely+1 
    nnz=nelz+1 
    nel=nelx*nely*nelz 
    nn_V=nnx*nny*nnz

    x_V=np.zeros(nn_V,dtype=np.float64)
    y_V=np.zeros(nn_V,dtype=np.float64)
    z_V=np.zeros(nn_V,dtype=np.float64) 

    counter=0
    for i in range(0,nelx+1):
        for j in range(0,nely+1):
            for k in range(0,nelz+1):
                x_V[counter]=i*Lx/float(nelx)
                y_V[counter]=j*Ly/float(nely)
                z_V[counter]=k*Lz/float(nelz)
                counter += 1
            #end for
        #end for
    #end for

    icon_V=np.zeros((8,nel),dtype=np.int32)

    counter = 0
    for i in range(0,nelx):
        for j in range(0,nely):
            for k in range(0,nelz):
                icon_V[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
                icon_V[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
                icon_V[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
                icon_V[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
                icon_V[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
                icon_V[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
                icon_V[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
                icon_V[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
                counter += 1
            #end for
        #end for
    #end for

    return x_V,y_V,z_V,icon_V,nn_V,nel
