import numpy as np

def mesher(Lx,Ly,nelx,nely,nel,NV,mV):

    xV = np.zeros(NV,dtype=np.float64)  # x coordinates
    yV = np.zeros(NV,dtype=np.float64)  # y coordinates
    iconV =np.zeros((mV,nel),dtype=np.int32)

    dx=Lx/nelx
    dy=Ly/nely
    
    dx2=Lx/nelx/2
    dy2=Ly/nely/2

    dx4=Lx/nelx/4
    dy4=Ly/nely/4

    eps=dx/2/np.sqrt(3)

    counter=0
    for irow in range(0,nely):

       #-------------
       # first line
       for icol in range(0,nelx):
          xV[counter]=icol*dx
          yV[counter]=0 + irow*dy
          counter+=1
          xV[counter]=icol*dx+dx2
          yV[counter]=0 + irow*dy
          counter=counter+1
       #end for
       xV[counter]=Lx
       yV[counter]=0 + irow*dy
       counter=counter+1

       #-------------
       # second line
       for icol in range(0,nelx):
          xV[counter]=icol*dx    + dx2- eps
          yV[counter]=dy2-eps + irow*dy
          counter=counter+1
          xV[counter]=icol*dx+dx2
          yV[counter]=dy2-eps + irow*dy
          counter=counter+1
          xV[counter]=icol*dx+dx2 + eps
          yV[counter]=dy2-eps + irow*dy
          counter=counter+1
       #end for

       #-------------
       # third line
       for icol in range(0,nelx):
          xV[counter]=icol*dx
          yV[counter]=dy2 + irow*dy
          counter=counter+1
          xV[counter]=icol*dx+dx2 - eps
          yV[counter]=dy2 + irow*dy
          counter=counter+1
          xV[counter]=icol*dx+dx2
          yV[counter]=dy2 + irow*dy
          counter=counter+1
          xV[counter]=icol*dx+dx2 +eps
          yV[counter]=dy2 + irow*dy
          counter=counter+1
       #end for
       xV[counter]=Lx
       yV[counter]=dy2 + irow*dy
       counter=counter+1

       #-------------
       # fourth line
       for icol in range(0,nelx):
          xV[counter]=icol*dx+dx2-eps
          yV[counter]=dy2+eps + irow*dy
          counter=counter+1
          xV[counter]=icol*dx+dx2
          yV[counter]=dy2+eps + irow*dy
          counter=counter+1
          xV[counter]=icol*dx+dx2 + eps
          yV[counter]=dy2+eps + irow*dy
          counter=counter+1
       #end for

    #end for

    #top line
    for icol in range(0,nelx):
       xV[counter]=icol*dx
       yV[counter]=Ly
       counter=counter+1
       xV[counter]=icol*dx+dx2
       yV[counter]=Ly
       counter=counter+1
    #end do
    xV[counter]=Lx
    yV[counter]=Ly
    counter=counter+1

    #np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y,u,v')

    ###############################################################################
    # computing  connectivity 
    ###############################################################################

    counter=0
    for irowf in range(0,nely):
        for icolf in range(0,nelx):
            icol=icolf+1
            irow=irowf+1

            #elt1
            iconV[0,counter]=(2*icol-1)                         + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(icol-1)*3+1            + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+1 + (icol-1)*4 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt2
            iconV[0,counter]=(2*icol-1)              + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*icol)                + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(icol-1)*3+2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(icol-1)*3+1 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt3
            iconV[0,counter]=(2*icol)                + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*icol)+1              + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(icol-1)*3+3 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(icol-1)*3+2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt4
            iconV[0,counter]=(2*icol)+1                             + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4 + 3 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4 + 2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(icol-1)*3+3                + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt5
            iconV[0,counter]=(2*nelx+1)+(icol-1)*3+1                 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(icol-1)*3+2                 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4 + 1  + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4      + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt6
            iconV[0,counter]=(2*nelx+1)+(icol-1)*3+2                  + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(icol-1)*3+2 +1               + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4 + 2   + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4 + 1   + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt7
            iconV[0,counter]=(2*nelx+1)+(3*nelx)+1 + (icol-1)*4                    + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4                    + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+1 + (icol-1)*3         + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+(3*nelx)+1+ (icol-1)*2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt8
            iconV[0,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+1 + (icol-1)*3         + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+2 + (icol-1)*3         + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+(3*nelx)+2+ (icol-1)*2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+(3*nelx)+1+ (icol-1)*2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt9
            iconV[0,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+2 + (icol-1)*3         + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+3 + (icol-1)*3         + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+(3*nelx)+3+ (icol-1)*2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+(3*nelx)+2+ (icol-1)*2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt10
            iconV[0,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4 + 2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4 + 3 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+(3*nelx)+3+ (icol-1)*2 + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+3 + (icol-1)*3         + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt11
            iconV[0,counter]=(2*nelx+1)+(3*nelx)+2 + (icol-1)*4              + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(3*nelx)+3 + (icol-1)*4              + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+2 + (icol-1)*3   + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+1 + (icol-1)*3   + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

            #elt12
            iconV[0,counter]=(2*nelx+1)+(3*nelx)+3 + (icol-1)*4              + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[1,counter]=(2*nelx+1)+(3*nelx)+4 + (icol-1)*4              + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[2,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+3 + (icol-1)*3   + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            iconV[3,counter]=(2*nelx+1)+(3*nelx)+(4*nelx+1)+2 + (icol-1)*3   + (irow-1)*((2*nelx+1)+(3*nelx)+ (4*nelx+1) + (3*nelx))    -1
            counter+=1

         #end for
    #end for

    return xV,yV,iconV


