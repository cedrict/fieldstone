import numpy as np
from basisQ1 import *
from compute_volume_hexahedron import *

def compute_gravity_hexahedron_mascons2(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,nm_per_dim):

    nmascons=nm_per_dim**3

    xx=np.array([pt_1[0],pt_2[0],pt_3[0],pt_4[0],
                 pt_5[0],pt_6[0],pt_7[0],pt_8[0]],dtype=np.float64)

    yy=np.array([pt_1[1],pt_2[1],pt_3[1],pt_4[1],
                 pt_5[1],pt_6[1],pt_7[1],pt_8[1]],dtype=np.float64)

    zz=np.array([pt_1[2],pt_2[2],pt_3[2],pt_4[2],
                 pt_5[2],pt_6[2],pt_7[2],pt_8[2]],dtype=np.float64)

    #--------------------------------------------------------------------------
    # generate r,s,t coordinates for mascons
    # and compute their real coordinates inside the hex
    #--------------------------------------------------------------------------

    npts_per_dim=nm_per_dim+1
    npts=npts_per_dim**3

    xpts = np.zeros(npts,dtype=np.float64)  # x coordinates
    ypts = np.zeros(npts,dtype=np.float64)  # y coordinates
    zpts = np.zeros(npts,dtype=np.float64)  # z coordinates

    counter=0
    for i in range(0,npts_per_dim):
        for j in range(0,npts_per_dim):
            for k in range(0,npts_per_dim):
                r=-1+i*2/(npts_per_dim-1)
                s=-1+j*2/(npts_per_dim-1)
                t=-1+k*2/(npts_per_dim-1)
                N=NNN(r,s,t)
                xpts[counter]=N.dot(xx)
                ypts[counter]=N.dot(yy)
                zpts[counter]=N.dot(zz)
                counter+=1    
            #end for
        #end for
    #end for

    #np.savetxt('mascons_hexa.ascii',np.array([xx,yy,zz]).T)
    #np.savetxt('mascons_pts.ascii',np.array([xpts,ypts,zpts]).T)

    #--------------------------------------------------------------------------
    # compute connectivity of the submesh
    #--------------------------------------------------------------------------

    icon=np.zeros((8,nmascons),dtype=np.int32)
    nnx=nm_per_dim+1
    nny=nm_per_dim+1
    nnz=nm_per_dim+1
    nelx=nnx-1
    nely=nny-1
    nelz=nnz-1
    counter=0 
    for i in range(0,nelx):
        for j in range(0,nely):
            for k in range(0,nelz):
                icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
                icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
                icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
                icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
                icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
                icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
                icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
                icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
                counter += 1
            #end for
        #end for
    #end for

    #--------------------------------------------------------------------------
    # compute coordinates of mascons
    #--------------------------------------------------------------------------

    xm = np.zeros(nmascons,dtype=np.float64)  # x coordinates
    ym = np.zeros(nmascons,dtype=np.float64)  # y coordinates
    zm = np.zeros(nmascons,dtype=np.float64)  # z coordinates
    mascon = np.zeros(nmascons,dtype=np.float64)  

    for ic in range(0,nmascons):
        xm[ic]=np.sum(xpts[icon[:,ic]])/8        
        ym[ic]=np.sum(ypts[icon[:,ic]])/8        
        zm[ic]=np.sum(zpts[icon[:,ic]])/8
        volume=hexahedron_volume(xpts[icon[:,ic]],ypts[icon[:,ic]],zpts[icon[:,ic]])
        mascon[ic]=volume*rho0

    #np.savetxt('mascons2.ascii',np.array([xm,ym,zm]).T)

    #------------------------------------------------------------------------------

    grav=np.zeros(3,dtype=np.float64)
    U=0
    for i in range(0,nmascons):
        dist=np.sqrt((xm[i]-pt_M[0])**2+(ym[i]-pt_M[1])**2+(zm[i]-pt_M[2])**2)
        grav[0]+=mascon[i]*Ggrav/dist**3*(xm[i]-pt_M[0])
        grav[1]+=mascon[i]*Ggrav/dist**3*(ym[i]-pt_M[1])
        grav[2]+=mascon[i]*Ggrav/dist**3*(zm[i]-pt_M[2])
        U+=mascon[i]*Ggrav/dist

    return grav,U

