import numpy as np
import random 
import FEbasis2D as FE

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def cartesian_mesh(Lx,Ly,nelx,nely,space):

    mtype=0 #make it an argument later

    hx=Lx/nelx
    hy=Ly/nely
    nel=nelx*nely

    #---------------------------------
    if space=='Q0':
       N=nelx*nely
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=(i+0.5)*hx
               y[counter]=(j+0.5)*hy
               counter += 1
       icon =np.zeros((1,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter]=counter
               counter += 1
           #end for
       #end for

    #---------------------------------
    elif space=='Q1':
       N=(nelx+1)*(nely+1)
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely+1):
           for i in range(0,nelx+1):
               x[counter]=i*hx
               y[counter]=j*hy
               counter += 1
       icon =np.zeros((4,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter]=i+j*(nelx+1)
               icon[1,counter]=i+1+j*(nelx+1)
               icon[2,counter]=i+1+(j+1)*(nelx+1)
               icon[3,counter]=i+(j+1)*(nelx+1)
               counter += 1
           #end for
       #end for

    #-----------------------------------
    elif space=='Q1+' or space=='Q1+Q0':
       N=(nelx+1)*(nely+1)+nel
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nely+1):
           for i in range(0,nelx+1):
               x[counter]=i*hx
               y[counter]=j*hy
               counter += 1
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=i*hx+1/2.*hx
               y[counter]=j*hy+1/2.*hy
               counter += 1
       icon=np.zeros((5,nel),dtype=np.int32)
       counter = 0
       for j in range(0, nely):
           for i in range(0, nelx):
               icon[0,counter]=i+j*(nelx+1)
               icon[1,counter]=i+1+j*(nelx+1)
               icon[2,counter]=i+1+(j+1)*(nelx+1)
               icon[3,counter]=i+(j+1)*(nelx+1)
               icon[4,counter]=(nelx+1)*(nely+1)+counter
               counter += 1
           #end for
       #end for

    #---------------------------------
    elif space=='Q2':
       N=(2*nelx+1)*(2*nely+1)
       nnx=2*nelx+1
       nny=2*nely+1
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nny):
           for i in range(0,nnx):
               x[counter]=i*hx/2.
               y[counter]=j*hy/2.
               counter += 1
           #end for
       #end for
       icon=np.zeros((9,nel),dtype=np.int32)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter]=(i)*2+1+(j)*2*nnx -1
               icon[1,counter]=(i)*2+3+(j)*2*nnx -1
               icon[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
               icon[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
               icon[4,counter]=(i)*2+2+(j)*2*nnx -1
               icon[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
               icon[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
               icon[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
               icon[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
               counter += 1
           #end for
       #end for

    #---------------------------------
    elif space=='Q2s':
       N=(nelx+1)*(nely+1)+nelx*(nely+1)+(nelx+1)*nely
       x=np.empty(N,dtype=np.float64) 
       y=np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nely+1):
           for i in range(0,nelx+1):
               x[counter]=i*hx
               y[counter]=j*hy
               counter += 1
       for j in range(nely):
           for i in range(0,nelx):
               x[counter]=i*hx+hx/2
               y[counter]=j*hy
               counter+=1
           for i in range(0,nelx+1):
               x[counter]=i*hx
               y[counter]=j*hy+hy/2
               counter+=1
       for i in range(0,nelx):
           x[counter]=i*hx+hx/2
           y[counter]=nely*hy
           counter+=1
       #end for
       icon=np.zeros((8,nel),dtype=np.int32)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter] = i + j * (nelx + 1)
               icon[1,counter] = i + 1 + j * (nelx + 1)
               icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
               icon[3,counter] = i + (j + 1) * (nelx + 1)
               icon[4,counter] = (nelx+1)*(nely+1)+i +(2*nelx+1)*j
               icon[5,counter] = (nelx+1)*(nely+1)+i +(2*nelx+1)*j + (nelx+1)
               icon[6,counter] = (nelx+1)*(nely+1)+i +(2*nelx+1)*(j+1)
               icon[7,counter] = (nelx+1)*(nely+1)+i +(2*nelx+1)*j + (nelx+1)-1
               counter += 1
           #end for
       #end for

    #---------------------------------
    elif space=='Q3':
       N=(3*nelx+1)*(3*nely+1)
       nnx=3*nelx+1
       nny=3*nely+1
       x=np.empty(N,dtype=np.float64) 
       y=np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nny):
           for i in range(0,nnx):
               x[counter]=i*hx/3
               y[counter]=j*hy/3
               counter += 1
           #end for
       #end for
       icon=np.zeros((16,nel),dtype=np.int32)
       counter=0
       for j in range(0,nely):
           for i in range(0,nelx):
               counter2=0
               for k in range(0,4):
                   for l in range(0,4):
                       icon[counter2,counter]=i*3+l+j*3*nnx+nnx*k
                       counter2+=1
               counter += 1 
           #end for
       #end for

    #---------------------------------
    elif space=='Q4':
       N=(4*nelx+1)*(4*nely+1)
       nnx=4*nelx+1
       nny=4*nely+1
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nny):
           for i in range(0,nnx):
               x[counter]=i*hx/4
               y[counter]=j*hy/4
               counter += 1
           #end for
       #end for
       icon=np.zeros((25,nel),dtype=np.int32)
       counter=0
       for j in range(0,nely):
           for i in range(0,nelx):
               counter2=0
               for k in range(0,5):
                   for l in range(0,5):
                       icon[counter2,counter]=i*4+l+j*4*nnx+nnx*k
                       counter2+=1
               counter += 1 
           #end for
       #end for

    #---------------------------------
    #
    # +-----+    +-----+
    # |\  B |    | D  /|
    # |  \  | or |  /  |
    # | A  \|    |/  C |
    # +-----+    +-----+

    elif space=='P0' and mtype==0:
       nel*=2
       N=nel
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2) : 
                  #C
                  x[counter]=(i+2/3)*hx 
                  y[counter]=(j+1/3)*hy
                  counter += 1
                  #D
                  x[counter]=(i+1/3)*hx 
                  y[counter]=(j+2/3)*hy
                  counter += 1
               else:
                  #A
                  x[counter]=(i+1/3)*hx 
                  y[counter]=(j+1/3)*hy
                  counter += 1
                  #B
                  x[counter]=(i+2/3)*hx 
                  y[counter]=(j+2/3)*hy
                  counter += 1
               #end if
           #end for
       #end for
       icon =np.zeros((1,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter]=counter
               counter += 1
               icon[0,counter]=counter
               counter += 1
           #end for
       #end for

    #---------------------------------
    #
    # 3-------2
    # |\  C  /|
    # |  \ /  |
    # | D 4 B |
    # |  / \  |
    # |/  A  \|
    # 0-------1

    elif space=='P0' and mtype==2:
       nel*=4
       N=nel
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=(i+1/2)*hx #A
               y[counter]=(j+1/6)*hy
               counter += 1
               x[counter]=(i+5/6)*hx #B
               y[counter]=(j+1/2)*hy
               counter += 1
               x[counter]=(i+1/2)*hx #C
               y[counter]=(j+5/6)*hy
               counter += 1
               x[counter]=(i+1/6)*hx #D
               y[counter]=(j+1/2)*hy
               counter += 1
       icon =np.zeros((1,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter]=counter
               counter += 1
               icon[0,counter]=counter
               counter += 1
               icon[0,counter]=counter
               counter += 1
               icon[0,counter]=counter
               counter += 1


    #---------------------------------
    #    A        B
    #  3---2    3---2
    #  |\ B|    |D /|
    #  | \ | or | / |
    #  |A \|    |/ C|
    #  0---1    0---1

    elif space=='P1' and mtype==0:
       nel*=2
       N=(nelx+1)*(nely+1)
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely+1):
           for i in range(0,nelx+1):
               x[counter]=i*hx
               y[counter]=j*hy
               counter += 1
       icon =np.zeros((3,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               inode0=i+j*(nelx+1)
               inode1=i+1+j*(nelx+1)
               inode2=i+1+(j+1)*(nelx+1)
               inode3=i+(j+1)*(nelx+1)
               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2):
                  #C
                  icon[0,counter]=inode1 
                  icon[1,counter]=inode2
                  icon[2,counter]=inode0
                  counter += 1
                  #D
                  icon[0,counter]=inode3 
                  icon[1,counter]=inode0
                  icon[2,counter]=inode2
                  counter += 1
               else: 
                  #A
                  icon[0,counter]=inode0 
                  icon[1,counter]=inode1
                  icon[2,counter]=inode3
                  counter += 1
                  #B
                  icon[0,counter]=inode2 
                  icon[1,counter]=inode3
                  icon[2,counter]=inode1
                  counter += 1
            #end if
        #end for
    #end for

    #---------------------------------
    # disc space for Q2Pm1 element
    elif space=='Pm1' or space=='Pm1u': 
       N=3*nel
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       icon=np.zeros((3,nel),dtype=np.int32)

       iel=0
       counter=0
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=i*hx+hx/2
               y[counter]=j*hy+hy/2
               icon[0,iel]=counter
               counter+=1
               x[counter]=i*hx+hx/2+hx/2
               y[counter]=j*hy+hy/2
               icon[1,iel]=counter
               counter+=1
               x[counter]=i*hx+hx/2
               y[counter]=j*hy+hy/2+hy/2
               icon[2,iel]=counter
               counter+=1
               iel+=1
           #end for
       #end for

    elif space=='Q-1': 
       N=4*nel
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       icon=np.zeros((4,nel),dtype=np.int32)

       iel=0
       counter=0
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=i*hx
               y[counter]=j*hy
               icon[0,iel]=counter
               counter+=1
               x[counter]=i*hx+hx
               y[counter]=j*hy
               icon[1,iel]=counter
               counter+=1
               x[counter]=i*hx+hx
               y[counter]=j*hy+hy
               icon[2,iel]=counter
               counter+=1
               x[counter]=i*hx
               y[counter]=j*hy+hy
               icon[3,iel]=counter
               counter+=1
               iel+=1
           #end for
       #end for


    #---------------------------------
    # disc P1 space for P2P-1 element
    #
    # +-----+    +-----+
    # |\  B |    | D  /|
    # |  \  | or |  /  |
    # | A  \|    |/  C |
    # +-----+    +-----+

    elif space=='P-1' and mtype==0:
       nel*=2
       N=3*nel
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       icon=np.zeros((3,nel),dtype=np.int32)
       iel=0
       counter=0
       for j in range(0,nely):
           for i in range(0,nelx):
               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2):
                  # C
                  x[counter]=i*hx+hx
                  y[counter]=j*hy
                  icon[0,iel]=counter
                  counter+=1
                  x[counter]=i*hx+hx
                  y[counter]=j*hy+hy
                  icon[1,iel]=counter
                  counter+=1
                  x[counter]=i*hx
                  y[counter]=j*hy
                  icon[2,iel]=counter
                  counter+=1
                  iel+=1
                  # D
                  x[counter]=i*hx
                  y[counter]=j*hy+hy
                  icon[0,iel]=counter
                  counter+=1
                  x[counter]=i*hx
                  y[counter]=j*hy
                  icon[1,iel]=counter
                  counter+=1
                  x[counter]=i*hx+hx
                  y[counter]=j*hy+hy
                  icon[2,iel]=counter
                  counter+=1
                  iel+=1
               else:
                  # A
                  x[counter]=i*hx
                  y[counter]=j*hy
                  icon[0,iel]=counter
                  counter+=1
                  x[counter]=i*hx+hx
                  y[counter]=j*hy
                  icon[1,iel]=counter
                  counter+=1
                  x[counter]=i*hx
                  y[counter]=j*hy+hy
                  icon[2,iel]=counter
                  counter+=1
                  iel+=1
                  # B
                  x[counter]=i*hx+hx
                  y[counter]=j*hy+hy
                  icon[0,iel]=counter
                  counter+=1
                  x[counter]=i*hx
                  y[counter]=j*hy+hy
                  icon[1,iel]=counter
                  counter+=1
                  x[counter]=i*hx+hx
                  y[counter]=j*hy
                  icon[2,iel]=counter
                  counter+=1
                  iel+=1
               #end if
           #end for
       #end for

    #---------------------------------
    elif space=='P1' and mtype==2:
       N=(nelx+1)*(nely+1)+nel
       nel*=4
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely+1):
           for i in range(0,nelx+1):
               x[counter]=i*hx
               y[counter]=j*hy
               counter += 1
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=(i+0.5)*hx
               y[counter]=(j+0.5)*hy
               counter += 1
       icon =np.zeros((3,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               inode0=i+j*(nelx+1)     
               inode1=i+1+j*(nelx+1)
               inode2=i+1+(j+1)*(nelx+1)
               inode3=i+(j+1)*(nelx+1)
               inode4=(nelx+1)*(nely+1)+nelx*j+i
               # triangle A
               icon[0,counter]=inode0
               icon[1,counter]=inode1
               icon[2,counter]=inode4
               counter += 1
               # triangle B
               icon[0,counter]=inode1
               icon[1,counter]=inode2
               icon[2,counter]=inode4
               counter += 1
               # triangle C
               icon[0,counter]=inode2
               icon[1,counter]=inode3
               icon[2,counter]=inode4
               counter += 1
               # triangle D
               icon[0,counter]=inode3
               icon[1,counter]=inode0
               icon[2,counter]=inode4
               counter += 1
           #end for
       #end for



    #---------------------------------
    #    A        B
    #  3---2    3---2
    #  |\  |    |  /|
    #  | \ | or | / |
    #  |  \|    |/  |
    #  0---1    0---1

    elif (space=='P1+' or space=='P1+P0') and mtype==0:
       nel*=2
       N=(nelx+1)*(nely+1)+nel
       nnx=nelx+1
       nny=nely+1
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0 
       for j in range(0,nely+1):
           for i in range(0,nelx+1):
               x[counter]=i*hx
               y[counter]=j*hy
               counter += 1
       icon =np.zeros((4,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               inode0=i+j*(nelx+1)
               inode1=i+1+j*(nelx+1)
               inode2=i+1+(j+1)*(nelx+1)
               inode3=i+(j+1)*(nelx+1)
               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2):
                  #C 
                  icon[0,counter]=inode1
                  icon[1,counter]=inode2
                  icon[2,counter]=inode0
                  icon[3,counter]=counter+nnx*nny
                  counter += 1
                  #D 
                  icon[0,counter]=inode3
                  icon[1,counter]=inode0
                  icon[2,counter]=inode2
                  icon[3,counter]=counter+nnx*nny
                  counter += 1
               else: 
                  #A
                  icon[0,counter]=inode0
                  icon[1,counter]=inode1
                  icon[2,counter]=inode3
                  icon[3,counter]=counter+nnx*nny
                  counter += 1
                  #B
                  icon[0,counter]=inode2
                  icon[1,counter]=inode3
                  icon[2,counter]=inode1
                  icon[3,counter]=counter+nnx*nny
                  counter += 1
               #end if
           #end for
       #end for
       for iel in range (0,nel): #bubble nodes
           x[nnx*nny+iel]=(x[icon[0,iel]]+x[icon[1,iel]]+x[icon[2,iel]])/3.
           y[nnx*nny+iel]=(y[icon[0,iel]]+y[icon[1,iel]]+y[icon[2,iel]])/3.

    #---------------------------------
    #
    # 3--6--2    3--6--2
    # |\  B |    | D  /|
    # 7  8  5 or 7  8  5
    # | A  \|    |/  C |
    # 0--4--1    0--4--1

    elif space=='P2':
       nel*=2
       N=(2*nelx+1)*(2*nely+1)
       nnx=2*nelx+1
       nny=2*nely+1
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nny):
           for i in range(0,nnx):
               x[counter]=i*hx/2.
               y[counter]=j*hy/2.
               counter += 1
           #end for
       #end for
       icon=np.zeros((6,nel),dtype=np.int32)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               inode0=(i)*2+1+(j)*2*nnx -1       #0
               inode1=(i)*2+3+(j)*2*nnx -1       #1
               inode2=(i)*2+3+(j)*2*nnx+nnx*2 -1 #2
               inode3=(i)*2+1+(j)*2*nnx+nnx*2 -1 #3
               inode4=(i)*2+2+(j)*2*nnx -1       #4
               inode5=(i)*2+3+(j)*2*nnx+nnx -1   #5
               inode6=(i)*2+2+(j)*2*nnx+nnx*2 -1 #6
               inode7=(i)*2+1+(j)*2*nnx+nnx -1   #7
               inode8=(i)*2+2+(j)*2*nnx+nnx -1   #8
               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2):
                  #C
                  icon[0,counter]=inode1
                  icon[1,counter]=inode2
                  icon[2,counter]=inode0
                  icon[3,counter]=inode5
                  icon[4,counter]=inode8
                  icon[5,counter]=inode4
                  counter += 1
                  #D
                  icon[0,counter]=inode3
                  icon[1,counter]=inode0
                  icon[2,counter]=inode2
                  icon[3,counter]=inode7
                  icon[4,counter]=inode8
                  icon[5,counter]=inode6
                  counter += 1
               else:
                  #A
                  icon[0,counter]=inode0
                  icon[1,counter]=inode1
                  icon[2,counter]=inode3
                  icon[3,counter]=inode4
                  icon[4,counter]=inode8
                  icon[5,counter]=inode7
                  counter += 1
                  #B
                  icon[0,counter]=inode2
                  icon[1,counter]=inode3
                  icon[2,counter]=inode1
                  icon[3,counter]=inode6
                  icon[4,counter]=inode8
                  icon[5,counter]=inode5
                  counter += 1
              #end if
           #end for
       #end for

    #-----------------
    # 2         
    # |\        
    # | \       
    # 5  4      
    # | 6 \     
    # |    \    
    # 0--3--1   

    elif space=='P2+':
       nel*=2
       nnx=2*nelx+1
       nny=2*nely+1
       N=nnx*nny+nel
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64)
       counter=0
       for j in range(0,nny):
           for i in range(0,nnx):
               x[counter]=i*hx/2.
               y[counter]=j*hy/2.
               counter += 1
           #end for
       #end for
       icon=np.zeros((7,nel),dtype=np.int32)
       counter=0
       for j in range(0,nely):
           for i in range(0,nelx):
               inode0=(i)*2+1+(j)*2*nnx -1       #0
               inode1=(i)*2+3+(j)*2*nnx -1       #1
               inode2=(i)*2+3+(j)*2*nnx+nnx*2 -1 #2
               inode3=(i)*2+1+(j)*2*nnx+nnx*2 -1 #3
               inode4=(i)*2+2+(j)*2*nnx -1       #4
               inode5=(i)*2+3+(j)*2*nnx+nnx -1   #5
               inode6=(i)*2+2+(j)*2*nnx+nnx*2 -1 #6
               inode7=(i)*2+1+(j)*2*nnx+nnx -1   #7
               inode8=(i)*2+2+(j)*2*nnx+nnx -1   #8

               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2) : 
                  #C
                  icon[0,counter]=inode1
                  icon[1,counter]=inode2
                  icon[2,counter]=inode0
                  icon[3,counter]=inode5
                  icon[4,counter]=inode8
                  icon[5,counter]=inode4
                  icon[6,counter]=nnx*nny+counter
                  counter += 1
                  #D
                  icon[0,counter]=inode3
                  icon[1,counter]=inode0
                  icon[2,counter]=inode2
                  icon[3,counter]=inode7
                  icon[4,counter]=inode8
                  icon[5,counter]=inode6
                  icon[6,counter]=nnx*nny+counter
                  counter += 1

               else:
                  #A
                  icon[0,counter]=(i)*2+1+(j)*2*nnx -1       #0
                  icon[1,counter]=(i)*2+3+(j)*2*nnx -1       #1
                  icon[2,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1 #3
                  icon[3,counter]=(i)*2+2+(j)*2*nnx -1       #4
                  icon[4,counter]=(i)*2+2+(j)*2*nnx+nnx -1   #8
                  icon[5,counter]=(i)*2+1+(j)*2*nnx+nnx -1   #7
                  icon[6,counter]=nnx*nny+counter
                  counter += 1
                  #B
                  icon[0,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1 #2
                  icon[1,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1 #3
                  icon[2,counter]=(i)*2+3+(j)*2*nnx -1       #1
                  icon[3,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1 #6
                  icon[4,counter]=(i)*2+2+(j)*2*nnx+nnx -1   #8
                  icon[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1   #5
                  icon[6,counter]=nnx*nny+counter
                  counter += 1
               #end if
           #end for
       #end for
       for iel in range (0,nel): #bubble nodes
           x[nnx*nny+iel]=(x[icon[0,iel]]+x[icon[1,iel]]+x[icon[2,iel]])/3.
           y[nnx*nny+iel]=(y[icon[0,iel]]+y[icon[1,iel]]+y[icon[2,iel]])/3.
       #end for

    #-----------------
    elif space=='P3':

       #              
       # 12=13==14==15     9        
       # |   |   |   |     | \      
       # 8===9==10==11     7   8    
       # |   |   |   |     |    \    
       # 4===5===6===7     4  5  6  
       # |   |   |   |     |       \ 
       # 0===1===2===3     0==1==2==3

       nel*=2
       N=(3*nelx+1)*(3*nely+1)
       nnx=3*nelx+1
       nny=3*nely+1
       x=np.empty(N,dtype=np.float64) 
       y=np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nny):
           for i in range(0,nnx):
               x[counter]=i*hx/3
               y[counter]=j*hy/3
               counter += 1
           #end for
       #end for
       icon=np.zeros((10,nel),dtype=np.int32)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               
               inode=np.zeros(16,dtype=np.int32)
               counter2=0
               for k in range(0,4):
                   for l in range(0,4):
                       inode[counter2]=i*3+l+j*3*nnx+nnx*k
                       counter2+=1
                   # end for
               # end for

               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2) : 
                  #C
                  icon[0,counter]=inode[3]
                  icon[1,counter]=inode[7]
                  icon[2,counter]=inode[11]
                  icon[3,counter]=inode[15]
                  icon[4,counter]=inode[2]
                  icon[5,counter]=inode[6]
                  icon[6,counter]=inode[10]
                  icon[7,counter]=inode[1]
                  icon[8,counter]=inode[5]
                  icon[9,counter]=inode[0]
                  counter += 1
                  #D
                  icon[0,counter]=inode[12]
                  icon[1,counter]=inode[8]
                  icon[2,counter]=inode[4]
                  icon[3,counter]=inode[0]
                  icon[4,counter]=inode[13]
                  icon[5,counter]=inode[9]
                  icon[6,counter]=inode[5]
                  icon[7,counter]=inode[14]
                  icon[8,counter]=inode[10]
                  icon[9,counter]=inode[15]
                  counter += 1
               else:
                  #A
                  icon[0,counter]=inode[0]
                  icon[1,counter]=inode[1]
                  icon[2,counter]=inode[2]
                  icon[3,counter]=inode[3]
                  icon[4,counter]=inode[4]
                  icon[5,counter]=inode[5]
                  icon[6,counter]=inode[6]
                  icon[7,counter]=inode[8]
                  icon[8,counter]=inode[9]
                  icon[9,counter]=inode[12]
                  counter += 1
                  #B
                  icon[0,counter]=inode[15]
                  icon[1,counter]=inode[14]
                  icon[2,counter]=inode[13]
                  icon[3,counter]=inode[12]
                  icon[4,counter]=inode[11]
                  icon[5,counter]=inode[10]
                  icon[6,counter]=inode[9]
                  icon[7,counter]=inode[7]
                  icon[8,counter]=inode[6]
                  icon[9,counter]=inode[3]
                  counter += 1
              #end if
           #end for
       #end for

    #-----------------
    elif space=='P4':

       # 20==21==22==23==24    14
       # |   |   |   |   |     |  \      
       # 15==16==17==18==19    12  13        
       # |   |   |   |   |     |    \      
       # 10==11==12==13==14    9  10  11      
       # |   |   |   |   |     |       \    
       # 5===6===7===8===9     5  6  7  8  
       # |   |   |   |   |     |          \ 
       # 0===1===2===3===4     0==1==2==3==4

       # +-----+    +-----+
       # |\  B |    | D  /|
       # |  \  | or |  /  |
       # | A  \|    |/  C |
       # +-----+    +-----+

       nel*=2
       N=(4*nelx+1)*(4*nely+1)
       nnx=4*nelx+1
       nny=4*nely+1
       x=np.empty(N,dtype=np.float64) 
       y=np.empty(N,dtype=np.float64)
       counter = 0
       for j in range(0,nny):
           for i in range(0,nnx):
               x[counter]=i*hx/4
               y[counter]=j*hy/4
               counter += 1
           #end for
       #end for
       icon=np.zeros((15,nel),dtype=np.int32)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               
               inode=np.zeros(25,dtype=np.int32)
               counter2=0
               for k in range(0,5):
                   for l in range(0,5):
                       inode[counter2]=i*4+l+j*4*nnx+nnx*k
                       counter2+=1
                   # end for
               # end for

               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2) : 
                  #C
                  icon[0,counter]=inode[4]
                  icon[1,counter]=inode[9]
                  icon[2,counter]=inode[14]
                  icon[3,counter]=inode[19]
                  icon[4,counter]=inode[24]
                  icon[5,counter]=inode[3]
                  icon[6,counter]=inode[8]
                  icon[7,counter]=inode[13]
                  icon[8,counter]=inode[18]
                  icon[9,counter]=inode[2]
                  icon[10,counter]=inode[7]
                  icon[11,counter]=inode[12]
                  icon[12,counter]=inode[1]
                  icon[13,counter]=inode[6]
                  icon[14,counter]=inode[0]
                  counter += 1
                  #D
                  icon[0,counter]=inode[20]
                  icon[1,counter]=inode[15]
                  icon[2,counter]=inode[10]
                  icon[3,counter]=inode[5]
                  icon[4,counter]=inode[0]
                  icon[5,counter]=inode[21]
                  icon[6,counter]=inode[16]
                  icon[7,counter]=inode[11]
                  icon[8,counter]=inode[6]
                  icon[9,counter]=inode[22]
                  icon[10,counter]=inode[17]
                  icon[11,counter]=inode[12]
                  icon[12,counter]=inode[23]
                  icon[13,counter]=inode[18]
                  icon[14,counter]=inode[24]
                  counter += 1
               else:
                  #A
                  icon[0,counter]=inode[0]
                  icon[1,counter]=inode[1]
                  icon[2,counter]=inode[2]
                  icon[3,counter]=inode[3]
                  icon[4,counter]=inode[4]
                  icon[5,counter]=inode[5]
                  icon[6,counter]=inode[6]
                  icon[7,counter]=inode[7]
                  icon[8,counter]=inode[8]
                  icon[9,counter]=inode[10]
                  icon[10,counter]=inode[11]
                  icon[11,counter]=inode[12]
                  icon[12,counter]=inode[15]
                  icon[13,counter]=inode[16]
                  icon[14,counter]=inode[20]
                  counter += 1
                  #B
                  icon[0,counter]=inode[24]
                  icon[1,counter]=inode[23]
                  icon[2,counter]=inode[22]
                  icon[3,counter]=inode[21]
                  icon[4,counter]=inode[20]
                  icon[5,counter]=inode[19]
                  icon[6,counter]=inode[18]
                  icon[7,counter]=inode[17]
                  icon[8,counter]=inode[16]
                  icon[9,counter]=inode[14]
                  icon[10,counter]=inode[13]
                  icon[11,counter]=inode[12]
                  icon[12,counter]=inode[9]
                  icon[13,counter]=inode[8]
                  icon[14,counter]=inode[4]
                  counter += 1
              #end if
           #end for
       #end for

    #---------------------------------
    elif space=='DSSY1' or space=='DSSY2' or\
         space=='RT1' or space=='RT2':
       N=(nely+1)*nelx + nely*(nelx+1)
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64) 
       counter=0
       for j in range(0,nely):
           # bottom line
           for i in range(0,nelx):
               x[counter]=(i+1-0.5)*hx
               y[counter]=(j+1-1)*hy
               counter+=1
           # middle line
           for i in range(0,nelx+1):
               x[counter]=(i)*hx
               y[counter]=(j+1-0.5)*hy
               counter+=1
       #top line
       for i in range(0,nelx):
           x[counter]=(i+1-0.5)*hx
           y[counter]=Ly
           counter+=1
       icon =np.zeros((4,nel),dtype=np.int32)
       counter = 0 
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0,counter] = (j)*(2*nelx+1)+i+1   -1   
               icon[1,counter] = icon[0,counter]+nelx+1
               icon[2,counter] = icon[0,counter]+2*nelx+1
               icon[3,counter] = icon[0,counter]+nelx
               counter += 1
           #end for
       #end for

    #---------------------------------
    elif space=='Han' or space=='Chen':
       N=(nely+1)*nelx + nely*(nelx+1)+nel
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64) 
       counter=0
       for j in range(0,nely):
           # bottom line
           for i in range(0,nelx):
               x[counter]=(i+1-0.5)*hx
               y[counter]=(j+1-1)*hy
               counter+=1
           #end for
           # middle line
           for i in range(0,nelx+1):
               x[counter]=(i)*hx
               y[counter]=(j+1-0.5)*hy
               counter+=1
           #end for
       #end for
       #top line
       for i in range(0,nelx):
           x[counter]=(i+1-0.5)*hx
           y[counter]=Ly
           counter+=1
       #center nodes
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=(i+0.5)*hx
               y[counter]=(j+0.5)*hx
               counter+=1
           #end for
       #end for

       icon =np.zeros((5,nel),dtype=np.int32)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               icon[0, counter] = (j)*(2*nelx+1)+i+1   -1
               icon[1, counter] = icon[0,counter]+nelx+1
               icon[2, counter] = icon[0,counter]+2*nelx+1
               icon[3, counter] = icon[0,counter]+nelx
               icon[4, counter] = (nely+1)*nelx + nely*(nelx+1) + counter
               counter += 1
           #end for
       #end for

    #---------------------------------
    elif space=='P1NC':
       N=(nely+1)*nelx + nely*(nelx+1)+nel
       nel*=2
       x = np.empty(N,dtype=np.float64) 
       y = np.empty(N,dtype=np.float64) 
       counter=0
       for j in range(0,nely):
           # bottom line
           for i in range(0,nelx):
               x[counter]=(i+1-0.5)*hx
               y[counter]=(j+1-1)*hy
               counter+=1
           # middle line
           for i in range(0,nelx+1):
               x[counter]=(i)*hx
               y[counter]=(j+1-0.5)*hy
               counter+=1
       #top line
       for i in range(0,nelx):
           x[counter]=(i+1-0.5)*hx
           y[counter]=Ly
           counter+=1
       #center nodes
       for j in range(0,nely):
           for i in range(0,nelx):
               x[counter]=(i+0.5)*hx
               y[counter]=(j+0.5)*hx
               counter+=1

       icon =np.zeros((3,nel),dtype=np.int32)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               node0= (j)*(2*nelx+1)+i+1 -1
               node1= node0+nelx+1
               node2= node0+2*nelx+1
               node3= node0+nelx
               node4= (nely+1)*nelx + nely*(nelx+1) + j*nelx+i
               if (i<nelx/2 and j<nely/2) or\
                  (i>=nelx/2 and j>=nely/2) : 
                  #C
                  icon[0, counter] = node1
                  icon[1, counter] = node4
                  icon[2, counter] = node0
                  counter += 1
                  #D
                  icon[0, counter] = node3
                  icon[1, counter] = node4
                  icon[2, counter] = node2
                  counter += 1
               else:
                  #A
                  icon[0, counter] = node0
                  icon[1, counter] = node4
                  icon[2, counter] = node3
                  counter += 1
                  #B
                  icon[0, counter] = node2
                  icon[1, counter] = node4
                  icon[2, counter] = node1
                  counter += 1
               #end if
           #end for
       #end for

    else:
       exit("FEtools:cartesian_mesh: space unknown ")

    return N,nel,x,y,icon

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def read_meshXX(Vspace,Pspace):

    NV=393
    nel=180

    f_vel = open('square_lvl1_P2.mesh', 'r') 
    lines_vel = f_vel.readlines()
    nlines=np.size(lines_vel)
    print('P2 mesh file counts ',nlines,' lines')
    for i in range(0,nlines):
          line=lines_vel[i].strip()
          columns=line.split()
          if np.size(columns)>0 and columns[0]=='Vertices':
             nextline=lines_vel[i+1].strip()
             print('mesh counts ',nextline, 'vertices')
             NV=int(nextline)
             vline_vel=i+2
          if np.size(columns)>0 and columns[0]=='Triangles':
             nextline=lines_vel[i+1].strip()
             print('mesh counts ',nextline, 'triangles')
             nel=int(nextline)
             tline_vel=i+2
    #end for
    f_press = open('square_lvl1_P1.mesh', 'r') 
    lines_press = f_press.readlines()
    nlines=np.size(lines_press)
    print('P1 mesh file counts ',nlines,' lines')
    for i in range(0,nlines):
          line=lines_press[i].strip()
          columns=line.split()
          if np.size(columns)>0 and columns[0]=='Vertices':
             nextline=lines_press[i+1].strip()
             print('mesh counts ',nextline, 'vertices')
             NP=int(nextline)
             vline_press=i+2
          if np.size(columns)>0 and columns[0]=='Triangles':
             nextline=lines_press[i+1].strip()
             print('mesh counts ',nextline, 'triangles')
             nel=int(nextline)
             tline_press=i+2
    #end for

    xV=np.zeros(NV,dtype=np.float64)
    yV=np.zeros(NV,dtype=np.float64)
    counter=0
    for i in range(vline_vel,vline_vel+NV):
          line=lines_vel[i].strip()
          columns=line.split()
          xV[counter]=float(columns[0])
          yV[counter]=float(columns[1])
          counter+=1
    #end for
    np.savetxt('mesh.ascii',np.array([xV,yV]).T)

    iconV=np.zeros((6,nel),dtype=np.int32)
    counter=0
    for i in range(tline_vel,tline_vel+nel):
             line=lines_vel[i].strip()
             columns=line.split()
             iconV[0,counter]=int(columns[0])-1
             iconV[1,counter]=int(columns[1])-1
             iconV[2,counter]=int(columns[2])-1
             iconV[3,counter]=int(columns[3])-1
             iconV[4,counter]=int(columns[4])-1
             iconV[5,counter]=int(columns[5])-1
             counter+=1


    xP=np.zeros(NP,dtype=np.float64)
    yP=np.zeros(NP,dtype=np.float64)
    counter=0
    for i in range(vline_press,vline_press+NP):
             line=lines_press[i].strip()
             columns=line.split()
             xP[counter]=float(columns[0])
             yP[counter]=float(columns[1])
             counter+=1
    np.savetxt('meshP.ascii',np.array([xP,yP]).T)

    iconP=np.zeros((3,nel),dtype=np.int32)
    counter=0
    for i in range(tline_press,tline_press+nel):
             line=lines_press[i].strip()
             columns=line.split()
             iconP[0,counter]=int(columns[0])-1
             iconP[1,counter]=int(columns[1])-1
             iconP[2,counter]=int(columns[2])-1
             counter+=1


    return nel,NV,NP,xV,yV,iconV,xP,yP,iconP




#------------------------------------------------------------------------------

def read_mesh(Vspace,Pspace,nelx,meshtype):

    counter=0
    file=open("meshes/"+meshtype+"/"+str(nelx)+"/meshinfo", "r") 
    for line in file:
        fields = line.strip().split()
        #print(fields[0], fields[1], fields[2])
        if counter==0:
           nel=int(fields[0])
        if counter==1:
           N_P2=int(fields[0])
        counter+=1
    print('nel', nel) 
    print('N_P2', N_P2)

    x_P2=np.zeros(N_P2,dtype=np.float64)
    y_P2=np.zeros(N_P2,dtype=np.float64)
    x_P2[0:N_P2],y_P2[0:N_P2]=np.loadtxt("meshes/"+meshtype+"/"+str(nelx)+"/mesh.1.node",unpack=True,usecols=[1,2],skiprows=1)
    icon_P2=np.zeros((6,nel),dtype=np.int32)
    icon_P2[0,:],icon_P2[1,:],icon_P2[2,:],icon_P2[4,:],icon_P2[5,:],icon_P2[3,:]=\
    np.loadtxt("meshes/"+meshtype+"/"+str(nelx)+"/mesh.1.ele",unpack=True, usecols=[1,2,3,4,5,6],skiprows=1)
    icon_P2[0,:]-=1
    icon_P2[1,:]-=1
    icon_P2[2,:]-=1
    icon_P2[3,:]-=1
    icon_P2[4,:]-=1
    icon_P2[5,:]-=1
    P1bool=np.zeros(N_P2,dtype=np.bool) 
    for iel in range(0,nel):
        P1bool[icon_P2[0,iel]]=True
        P1bool[icon_P2[1,iel]]=True
        P1bool[icon_P2[2,iel]]=True

    N_P1=np.count_nonzero(P1bool)
    icon_P1=np.zeros((3,nel),dtype=np.int32)
    icon_P1[0,:]=icon_P2[0,:]
    icon_P1[1,:]=icon_P2[1,:]
    icon_P1[2,:]=icon_P2[2,:]
    x_P1=np.zeros(N_P1,dtype=np.float64)
    y_P1=np.zeros(N_P1,dtype=np.float64)
    for iel in range(0,nel):
        x_P1[icon_P1[0,iel]]=x_P2[icon_P1[0,iel]]
        x_P1[icon_P1[1,iel]]=x_P2[icon_P1[1,iel]]
        x_P1[icon_P1[2,iel]]=x_P2[icon_P1[2,iel]]
        y_P1[icon_P1[0,iel]]=y_P2[icon_P1[0,iel]]
        y_P1[icon_P1[1,iel]]=y_P2[icon_P1[1,iel]]
        y_P1[icon_P1[2,iel]]=y_P2[icon_P1[2,iel]]

    if Vspace=='P1+':
       NV=N_P1+nel
       xV=np.zeros(NV,dtype=np.float64)
       yV=np.zeros(NV,dtype=np.float64)
       xV[0:N_P1]=x_P1[0:N_P1]
       yV[0:N_P1]=y_P1[0:N_P1]
       iconV=np.zeros((4,nel),dtype=np.int32)
       iconV[0,:]=icon_P1[0,:]
       iconV[1,:]=icon_P1[1,:]
       iconV[2,:]=icon_P1[2,:]
       for iel in range (0,nel): #bubble nodes
           xV[N_P1+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
           yV[N_P1+iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.
           iconV[3,iel]=N_P1+iel

    if Vspace=='P2':
       NV=N_P2
       xV=np.zeros(NV,dtype=np.float64)
       yV=np.zeros(NV,dtype=np.float64)
       xV[:]=x_P2[:]
       yV[:]=y_P2[:]
       iconV=np.zeros((6,nel),dtype=np.int32)
       iconV[0,:]=icon_P2[0,:]
       iconV[1,:]=icon_P2[1,:]
       iconV[2,:]=icon_P2[2,:]
       iconV[3,:]=icon_P2[3,:]
       iconV[4,:]=icon_P2[4,:]
       iconV[5,:]=icon_P2[5,:]

    if Vspace=='P2+':
       NV=N_P2+nel
       xV=np.zeros(NV,dtype=np.float64)
       yV=np.zeros(NV,dtype=np.float64)
       xV[0:N_P2]=x_P2[0:N_P2]
       yV[0:N_P2]=y_P2[0:N_P2]
       iconV=np.zeros((7,nel),dtype=np.int32)
       iconV[0,:]=icon_P2[0,:]
       iconV[1,:]=icon_P2[1,:]
       iconV[2,:]=icon_P2[2,:]
       iconV[3,:]=icon_P2[3,:]
       iconV[4,:]=icon_P2[4,:]
       iconV[5,:]=icon_P2[5,:]
       for iel in range (0,nel): #bubble nodes
           xV[N_P2+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
           yV[N_P2+iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.
           iconV[6,iel]=N_P2+iel

    if Pspace=='P0':
       NP=nel
       xP=np.zeros(NP,dtype=np.float64)
       yP=np.zeros(NP,dtype=np.float64)
       iconP=np.zeros((1,nel),dtype=np.int32)
       for iel in range(0,nel):
           xP[iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
           yP[iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.
           iconP[0,iel]=iel

    if Pspace=='P1':
       NP=N_P1
       xP=np.zeros(NP,dtype=np.float64)
       yP=np.zeros(NP,dtype=np.float64)
       xP[:]=x_P1[:]
       yP[:]=y_P1[:]
       iconP=np.zeros((3,nel),dtype=np.int32)
       iconP[0,:]=icon_P1[0,:]
       iconP[1,:]=icon_P1[1,:]
       iconP[2,:]=icon_P1[2,:]

    if Pspace=='P-1': # and Vspace=='P2+' !
       NP=3*nel
       xP=np.zeros(NP,dtype=np.float64)
       yP=np.zeros(NP,dtype=np.float64)
       iconP=np.zeros((3,nel),dtype=np.int32)
       counter=0
       for iel in range(0,nel):
           xP[counter]=xV[iconV[0,iel]]
           yP[counter]=yV[iconV[0,iel]]
           iconP[0,iel]=counter
           counter+=1
           xP[counter]=xV[iconV[1,iel]]
           yP[counter]=yV[iconV[1,iel]]
           iconP[1,iel]=counter
           counter+=1
           xP[counter]=xV[iconV[2,iel]]
           yP[counter]=yV[iconV[2,iel]]
           iconP[2,iel]=counter
           counter+=1

    return nel,NV,NP,xV,yV,iconV,xP,yP,iconP

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def randomize_background_mesh(x1,y1,hx,hy,N1,Lx,Ly):
    alpha=0.1
    for i in range(0,N1):
        psi=random.uniform(-1.,+1)
        phi=random.uniform(-1.,+1)
        if x1[i]>0 and x1[i]<Lx and y1[i]>0 and y1[i]<Ly:
           x1[i]+=hx*psi*alpha
           y1[i]+=hy*phi*alpha

def deform_mesh_RT(x1,y1,N1,Lx,Ly,nelx,nely):
    hx=Lx/nelx
    hy=Ly/nely
    eps=1e-8
    llambda=0.5
    amplitude=0.01
    for i in range(0,N1):
        if abs(y1[i]-Ly/2.)/Ly<eps:
           y1[i]+=amplitude*np.cos(2*np.pi*x1[i]/llambda)

    counter=0
    for j in range(0,nely+1):
        for i in range(0,nelx+1):
            ya=0.5+amplitude*np.cos(2*np.pi*x1[counter]/llambda)
            if j<(nely+1)/2:
               dy=ya/(nely/2)
               y1[counter]=j*dy
            else:
               dy=(Ly-ya)/(nely/2)
               y1[counter]=ya+(j-nely/2)*dy
            counter+=1




def adapt_FE_mesh(x1,y1,icon1,m1,space1,x,y,icon,nel,space):
    r=FE.NNN_r(space)
    s=FE.NNN_s(space)
    m=FE.NNN_m(space)
    for iel in range(0,nel):
        for i in range(0,m):
            NNN1=FE.NNN(r[i],s[i],space1)
            x[icon[i,iel]]=NNN1.dot(x1[icon1[0:m1,iel]])
            y[icon[i,iel]]=NNN1.dot(y1[icon1[0:m1,iel]])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def export_swarm_to_ascii(x,y,filename):
    np.savetxt(filename,np.array([x,y]).T,header='# x,y')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def export_swarm_scalar_to_ascii(x,y,f,filename):
    np.savetxt(filename,np.array([x,y,f]).T,header='# x,y,field')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def export_swarm_vector_to_ascii(x,y,u,v,filename):
    np.savetxt(filename,np.array([x,y,u,v]).T,header='# x,y,vx,vy')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def export_connectivity_array_to_ascii(x,y,icon,filename):
    m,nel=np.shape(icon)
    iconfile=open(filename,"w")
    for iel in range (0,nel):
        iconfile.write('--------elt:'+str(iel)+'-------\n')
        for k in range(0,m):
            iconfile.write("node "+str(k)+' | '+str(icon[k,iel])+" at pos. "+str(x[icon[k,iel]])+','+str(y[icon[k,iel]])+'\n')

def export_connectivity_array_elt1_to_ascii(x,y,icon,filename):
    m,nel=np.shape(icon)
    iconfile=open(filename,"w")
    iel=0
    iconfile.write('--------elt:'+str(iel)+'-------\n')
    for k in range(0,m):
        iconfile.write("node "+str(k)+' | '+str(icon[k,iel])+" at pos. "+str(x[icon[k,iel]])+','+str(y[icon[k,iel]])+'\n')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def export_elements_to_vtu(x,y,icon,space,filename,area,bx,by,eta):
    N=np.size(x)
    m,nel=np.shape(icon)
 
    if space=='P0' or space=='Q0' or space=='P-1' or space=='Pm1' or space=='Pm1u':
       return

    if space=='Q1' or space=='Q1+' or space=='Q2' or space=='Q2s' or \
       space=='RT1' or space=='RT2' or space=='DSSY1' or space=='DSSY2' or\
       space=='Q1+Q0' :
       node0=0 ; node1=1 ; node2=2 ; node3=3
       m=4
    if space=='Q3':
       node0=0
       node1=3
       node2=12
       node3=15
       m=4
    if space=='Q4':
       node0=0
       node1=4
       node2=20
       node3=24
       m=4
    if space=='P1' or space=='P1+' or space=='P2' or space=='P2+' or space=='P1NC':
       node0=0
       node1=1
       node2=2
       m=3       
    if space=='P3':
       node0=0
       node1=3
       node2=9
       m=3       
    if space=='P4':
       node0=0
       node1=4
       node2=14
       m=3       

    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='bx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (bx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='by' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (by[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eta[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        if m==3:
           vtufile.write("%d %d %d \n" %(icon[node0,iel],icon[node1,iel],icon[node2,iel]))
        if m==4:
           vtufile.write("%d %d %d %d \n" %(icon[node0,iel],icon[node1,iel],icon[node2,iel],icon[node3,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        if m==3:
           vtufile.write("%d \n" %5)
        else:
           vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#------------------------------------------------------------------------------

def export_V_to_vtu(NV,xV,yV,iconV,Vspace,filename,u,v,Pspace,p,iconP):
    mV,nel=np.shape(iconV)

    if Vspace=='Q2': m=4
    if Vspace=='P2': m=3
    if Vspace=='P2+': m=3
    if Vspace=='P1+': m=3

    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nel*m,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for iel in range(0,nel):
        for k in range(0,m):
            vtufile.write("%10e %10e %10e \n" %(xV[iconV[k,iel]],yV[iconV[k,iel]],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")

    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for iel in range(0,nel):
        for k in range(0,m):
            vtufile.write("%10e %10e %10e \n" %(u[iconV[k,iel]],v[iconV[k,iel]],0.))
    vtufile.write("</DataArray>\n")

    if Pspace=='P0':
       vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e \n" % p[iconP[0,iel]])
           vtufile.write("%10e \n" % p[iconP[0,iel]])
           vtufile.write("%10e \n" % p[iconP[0,iel]])
       vtufile.write("</DataArray>\n")

    if Pspace=='P1' or Pspace=='P-1' or Pspace=='Q1':
       vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
       for iel in range(0,nel):
           for k in range(0,m):
               vtufile.write("%10e \n" % p[iconP[k,iel]])
       vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        if m==3:
           vtufile.write("%d %d %d \n" %(iel*m,iel*m+1,iel*m+2))
        if m==4:
           vtufile.write("%d %d %d %d \n" %(iel*m,iel*m+1,iel*m+2,iel*m+3))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        if m==3:
           vtufile.write("%d \n" %5)
        else:
           vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def export_swarm_to_vtu(x,y,filename):
       N=np.size(x)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,N))
       #--
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,N):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,N):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def export_swarm_vector_to_vtu(x,y,vx,vy,filename):
       N=np.size(x)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,N))
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,N):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e %10e %10e \n" %(vx[i],vy[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")

       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,N):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def export_swarm_scalar_to_vtu(x,y,scalar,filename):
       N=np.size(x)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,N))
       #--
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,N):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='field' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e \n" %(scalar[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,N):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def bc_setup(x,y,u,v,Lx,Ly,ndof,left,right,bottom,top):
    eps=1e-8
    N=np.size(x)
    Nfem=2*N
    bc_fix = np.zeros(Nfem, dtype=np.bool)     # boundary condition, yes/no
    bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value
    for i in range(0,N):

        #left
        if x[i]/Lx<eps:
           if left=='free_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
           if left=='no_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if left=='v_zero':
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if left=='analytical':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = u[i]
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = v[i]

        #right
        if x[i]/Lx>(1-eps):
           if right=='free_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
           if right=='no_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if right=='v_zero':
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if right=='analytical':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = u[i]
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = v[i]

        #bottom
        if y[i]/Ly<eps:
           if bottom=='free_slip':
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if bottom=='no_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if bottom=='mone':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = -1.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if bottom=='analytical':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = u[i]
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = v[i]

        #top
        if y[i]/Ly>(1-eps):
           if top=='free_slip':
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if top=='no_slip':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 0.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if top=='one':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = 1.
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
           if top=='analytical':
              bc_fix[i*ndof+0] = True ; bc_val[i*ndof+0] = u[i]
              bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = v[i]

    return bc_fix,bc_val

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def J(m,dNdr,dNds,x,y):
    jcb=np.zeros((2,2),dtype=np.float64)
    jcb[0,0]=dNdr.dot(x)
    jcb[0,1]=dNdr.dot(y)
    jcb[1,0]=dNds.dot(x)
    jcb[1,1]=dNds.dot(y)
    jcbi=np.linalg.inv(jcb)
    jcob=np.linalg.det(jcb)
    if jcob<0: exit('jcob<0')
    return jcob,jcbi

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def assemble_K(K_el,A_sparse,iconV,mV,ndofV,iel):

    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1+i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2+i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    A_sparse[m1,m2] += K_el[ikk,jkk]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def assemble_G(G_el,A_sparse,iconV,iconP,NfemV,mV,mP,ndofV,ndofP,iel):

    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1+i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mP):
                m2 =iconP[k2,iel]
                A_sparse[m1,NfemV+m2]+=G_el[ikk,k2]
                A_sparse[NfemV+m2,m1]+=G_el[ikk,k2]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def assemble_f(f_el,rhs,iconV,mV,ndofV,iel):
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1+i1
            m1 =ndofV*iconV[k1,iel]+i1
            rhs[m1]+=f_el[ikk]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def assemble_h(h_el,rhs,iconP,mP,NfemV,iel):
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        rhs[NfemV+m2]+=h_el[k2]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def apply_bc(K_el,G_el,f_el,h_el,bc_val,bc_fix,iconV,mV,ndofV,iel):
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1+i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def visualise_with_tikz(x,y,space):
    N=np.size(x)
    tikzfile=open('3x2_'+space+'.tex',"w")
    scale=2
    offset=0.15

    tikzfile.write("\\begin{center} \n")
    tikzfile.write("\\begin{tikzpicture} \n")

    tikzfile.write("\\node[violet] at (3,4.5) {"+space+"}; \n")

    tikzfile.write("\\draw[thick] (0,0) -- (6,0) -- (6,4) -- (0,4) -- cycle; \n")
    tikzfile.write("\\draw[thick] (0,2) -- (6,2) ; \n")
    tikzfile.write("\\draw[thick] (2,0) -- (2,4) ; \n")
    tikzfile.write("\\draw[thick] (4,0) -- (4,4) ; \n")

    if space=='P1' or space=='P2' or space=='P3' or space=='P1+' or\
       space=='P1NC' or space=='P2+' or space=='P1+P0' or space=='P4':
       tikzfile.write("\\draw[thick] (0,0) -- (2,2) -- (0,4) ; \n")
       tikzfile.write("\\draw[thick] (2,0) -- (4,2) -- (2,4) ; \n")
       tikzfile.write("\\draw[thick] (6,0) -- (4,2) -- (6,4) ; \n")

    for i in range(0,N):
         tikzfile.write("\\draw[black,fill=teal] ( %f , %f)     circle (2pt); \n" %(x[i]*scale,y[i]*scale)) 
         tikzfile.write("\\node[] at ( %f, %f ) {\\tiny %d }; \n" %(x[i]*scale-offset,y[i]*scale-offset,i))

    tikzfile.write("\\end{tikzpicture} \n")
    tikzfile.write("\\end{center} \n")
    tikzfile.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
