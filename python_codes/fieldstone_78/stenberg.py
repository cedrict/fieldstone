import numpy as np

def mesher(Lx,Ly,nelx,nely,nel,NV,mV):

    xV = np.zeros(NV,dtype=np.float64)  # x coordinates
    yV = np.zeros(NV,dtype=np.float64)  # y coordinates
    iconV =np.zeros((mV,nel),dtype=np.int32)

    dx=Lx/nelx
    dy=Ly/nely
    dx2=Lx/nelx*0.3
    dy2=Ly/2./nely

    counter=0
    for irow in range(0,nely):
        #-------------
        # first line
        for icol in range(0,nelx):
            xV[counter]=icol*dx
            yV[counter]=0 + irow*dy
            counter=counter+1
            xV[counter]=icol*dx+dx/2
            yV[counter]=0+ irow*dy
            counter=counter+1
        #end do
        xV[counter]=Lx
        yV[counter]=0+ irow*dy
        counter=counter+1
        #-------------
        # second line
        for icol in range(0,nelx):
            xV[counter]=icol*dx
            yV[counter]=irow*dy+dy2
            counter=counter+1
            xV[counter]=icol*dx+dx2
            yV[counter]=irow*dy+dy2
            counter=counter+1
            xV[counter]=icol*dx+(dx-dx2)
            yV[counter]=irow*dy+dy2
            counter=counter+1
        #end do
        xV[counter]=Lx
        yV[counter]=irow*dy+dy2
        counter=counter+1
    #end for

    # top line

    for icol in range(0,nelx):
       xV[counter]=icol*dx
       yV[counter]=Ly
       counter=counter+1
       xV[counter]=icol*dx+dx/2.
       yV[counter]=Ly
       counter=counter+1
    #end for
    xV[counter]=Lx
    yV[counter]=Ly
    counter=counter+1

    ###############################################################################
    # computing grid connectivity 
    ###############################################################################

    counter=0
    for irowf in range(0,nely):
        for icolf in range(0,nelx):
            icol=icolf+1
            irow=irowf+1

            #elt1
            iconV[0,counter]=(2*icol-1)                + (irow-1)*(5*nelx+2)  -1
            iconV[1,counter]=(2*icol)                  + (irow-1)*(5*nelx+2)  -1
            iconV[2,counter]=(2*nelx+1)+(icol-1)*3+2 + (irow-1)*(5*nelx+2)  -1
            iconV[3,counter]=(2*nelx+1)+(icol-1)*3+1 + (irow-1)*(5*nelx+2)  -1
            counter=counter+1

            #elt2
            iconV[0,counter]=(2*icol)   + (irow-1)*(5*nelx+2) -1
            iconV[1,counter]=(2*icol+1)   + (irow-1)*(5*nelx+2) -1
            iconV[2,counter]=(2*nelx+1)+(icol-1)*3+4   + (irow-1)*(5*nelx+2) -1
            iconV[3,counter]=(2*nelx+1)+(icol-1)*3+3   + (irow-1)*(5*nelx+2) -1
            counter=counter+1

            #elt3
            iconV[0,counter]=(2*icol)                   + (irow-1)*(5*nelx+2) -1
            iconV[1,counter]=(2*nelx+1)+(icol-1)*3+3  + (irow-1)*(5*nelx+2) -1
            iconV[2,counter]=(2*icol)                   + (irow)*(5*nelx+2) -1
            iconV[3,counter]=(2*nelx+1)+(icol-1)*3+2  + (irow-1)*(5*nelx+2) -1
            counter=counter+1

            #elt4
            iconV[0,counter]=(2*nelx+1)+(icol-1)*3+1     + (irow-1)*(5*nelx+2) -1
            iconV[1,counter]=(2*nelx+1)+(icol-1)*3+2     + (irow-1)*(5*nelx+2) -1
            iconV[2,counter]=(2*icol) + irow*(5*nelx+2)   -1
            iconV[3,counter]=(2*icol) + irow*(5*nelx+2)-1 -1
            counter=counter+1

            #elt5
            iconV[0,counter]=(2*nelx+1)+(icol-1)*3+3 + (irow-1)*(5*nelx+2) -1
            iconV[1,counter]=(2*nelx+1)+(icol-1)*3+4 + (irow-1)*(5*nelx+2) -1
            iconV[2,counter]=(2*icol) + irow*(5*nelx+2)+1 -1
            iconV[3,counter]=(2*icol) + irow*(5*nelx+2) -1
            counter=counter+1

         #end do
    #end do

    return xV,yV,iconV
