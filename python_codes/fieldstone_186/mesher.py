import numpy as np
from basis_functions import *

###############################################################################
# The mesh is composed of eight blocks. Each is first built and warped
# to fit the contour of the hole and they are subsequently merged.
# all blocks have the same topology and number of elements.
#
#   E-----C-------------G
#   | 2   |             |
#   F     |      3      |
#    D  1 |             |
#     A---B-------------H
#    P  8 |             |
#   N     |      4      |
#   | 7   |             |
#   M-----O-------------I
#   |     |             |
#   |  6  |      5      |
#   |     |             |
#   L-----K-------------J
#
###############################################################################

sqrt2=np.sqrt(2.)

def mesher(nelxA,nelxB,nelyC,L,extent_x,extent_y,radius,m):

    debug=False

    verbose=False
 
    if verbose: print('-----------------')

    nelyA=nelxA
    nelyB=nelyA
    nelxC=nelxA
    nelxD=nelxB
    nelyD=nelyC   

    xA=radius       ; yA=0
    xB=L            ; yB=0
    xC=L            ; yC=L
    xD=radius/sqrt2 ; yD=radius/sqrt2
    xE=0            ; yE=L          
    xF=0            ; yF=radius    
    xG=L+extent_x   ; yG=L        
    xH=L+extent_x   ; yH=0       
    xI=L+extent_x   ; yI=-L     
    xJ=L+extent_x   ; yJ=-L-extent_y    
    xK=L            ; yK=-L-extent_y   
    xL=0            ; yL=-L-extent_y 
    xM=0            ; yM=-L         
    xN=0            ; yN=-radius   
    xO=L            ; yO=-L       
    xP=radius/sqrt2 ; yP=-radius/sqrt2 

    if verbose:
       print('xA,yA',xA,yA)
       print('xB,yB',xB,yB)
       print('xC,yC',xC,yC)
       print('xD,yD',xD,yD)
       print('xE,yE',xE,yE)
       print('xF,yF',xF,yF)
       print('xG,yG',xG,yG)
       print('xH,yH',xH,yH)
       print('xI,yI',xI,yI)
       print('xJ,yJ',xJ,yJ)
       print('xK,yK',xK,yK)
       print('xL,yL',xL,yL)
       print('xM,yM',xM,yM)
       print('xN,yN',xN,yN)
       print('xO,yO',xO,yO)
       print('xP,yP',xP,yP)

    ###########################################################################

    # block #1
    b1_nelx=nelxA
    b1_nely=nelyA
    b1_nnx=nelxA+1
    b1_nny=nelyA+1
    b1_NV=b1_nnx*b1_nny
    b1_nel=b1_nelx*b1_nely

    # block #2
    b2_nelx=nelxA
    b2_nely=nelyA
    b2_nnx=nelxA+1
    b2_nny=nelyA+1
    b2_NV=b2_nnx*b2_nny
    b2_nel=b2_nelx*b2_nely

    # block #3
    b3_nelx=nelxB
    b3_nely=nelyB
    b3_nnx=nelxB+1
    b3_nny=nelyB+1
    b3_NV=b3_nnx*b3_nny
    b3_nel=b3_nelx*b3_nely

    # block #4
    b4_nelx=nelxB
    b4_nely=nelyB
    b4_nnx=nelxB+1
    b4_nny=nelyB+1
    b4_NV=b4_nnx*b4_nny
    b4_nel=b4_nelx*b4_nely

    # block #5
    b5_nelx=nelxD
    b5_nely=nelyD
    b5_nnx=nelxD+1
    b5_nny=nelyD+1
    b5_NV=b5_nnx*b5_nny
    b5_nel=b5_nelx*b5_nely

    # block #6
    b6_nelx=nelxC
    b6_nely=nelyC
    b6_nnx=nelxC+1
    b6_nny=nelyC+1
    b6_NV=b6_nnx*b6_nny
    b6_nel=b6_nelx*b6_nely

    # block #7
    b7_nelx=nelxA
    b7_nely=nelyA
    b7_nnx=nelxA+1
    b7_nny=nelyA+1
    b7_NV=b7_nnx*b7_nny
    b7_nel=b7_nelx*b7_nely

    # block #8
    b8_nelx=nelxA
    b8_nely=nelyA
    b8_nnx=nelxA+1
    b8_nny=nelyA+1
    b8_NV=b8_nnx*b8_nny
    b8_nel=b8_nelx*b8_nely

    if verbose: 
       print('-----------------')
       print('block 1',b1_nel,' elements, ',b1_NV,'nodes | ABCD')
       print('block 2',b2_nel,' elements, ',b2_NV,'nodes | DCEF')
       print('block 3',b3_nel,' elements, ',b3_NV,'nodes | BHGC')
       print('block 4',b4_nel,' elements, ',b4_NV,'nodes | OIHB')
       print('block 5',b5_nel,' elements, ',b5_NV,'nodes | KJIO')
       print('block 6',b6_nel,' elements, ',b6_NV,'nodes | LKOM')
       print('block 7',b7_nel,' elements, ',b7_NV,'nodes | MOPN')
       print('block 8',b8_nel,' elements, ',b8_NV,'nodes | POBA')

    ###########################################################################
    # coordinates arrays
    ###########################################################################

    b1_x=np.zeros(b1_NV,dtype=np.float64)  # x coordinates
    b1_y=np.zeros(b1_NV,dtype=np.float64)  # y coordinates
    b2_x=np.zeros(b2_NV,dtype=np.float64)  # x coordinates
    b2_y=np.zeros(b2_NV,dtype=np.float64)  # y coordinates
    b3_x=np.zeros(b3_NV,dtype=np.float64)  # x coordinates
    b3_y=np.zeros(b3_NV,dtype=np.float64)  # y coordinates
    b4_x=np.zeros(b4_NV,dtype=np.float64)  # x coordinates
    b4_y=np.zeros(b4_NV,dtype=np.float64)  # y coordinates
    b5_x=np.zeros(b5_NV,dtype=np.float64)  # x coordinates
    b5_y=np.zeros(b5_NV,dtype=np.float64)  # y coordinates
    b6_x=np.zeros(b6_NV,dtype=np.float64)  # x coordinates
    b6_y=np.zeros(b6_NV,dtype=np.float64)  # y coordinates
    b7_x=np.zeros(b7_NV,dtype=np.float64)  # x coordinates
    b7_y=np.zeros(b7_NV,dtype=np.float64)  # y coordinates
    b8_x=np.zeros(b8_NV,dtype=np.float64)  # x coordinates
    b8_y=np.zeros(b8_NV,dtype=np.float64)  # y coordinates

    b1_hull=np.zeros(b1_NV,dtype=bool) 
    b2_hull=np.zeros(b2_NV,dtype=bool) 
    b3_hull=np.zeros(b3_NV,dtype=bool) 
    b4_hull=np.zeros(b4_NV,dtype=bool) 
    b5_hull=np.zeros(b5_NV,dtype=bool) 
    b6_hull=np.zeros(b6_NV,dtype=bool) 
    b7_hull=np.zeros(b7_NV,dtype=bool) 
    b8_hull=np.zeros(b8_NV,dtype=bool) 

    counter=0
    for j in range(0,b1_nny):
        for i in range(0,b1_nnx):
            b1_x[counter]=i/float(b1_nelx)
            b1_y[counter]=j/float(b1_nely)
            if i==0 or j==0 or i==b1_nnx-1 or j==b1_nny-1: b1_hull[counter]=True
            counter+=1
        #end for
    #end for

    counter=0
    for j in range(0,b2_nny):
        for i in range(0,b2_nnx):
            b2_x[counter]=i/float(b2_nelx)
            b2_y[counter]=j/float(b2_nely)
            if i==0 or j==0 or i==b2_nnx-1 or j==b2_nny-1: b2_hull[counter]=True
            counter+=1
        #end for
    #end for

    counter=0
    for j in range(0,b3_nny):
        for i in range(0,b3_nnx):
            b3_x[counter]=i/float(b3_nelx)
            b3_y[counter]=j/float(b3_nely)
            if i==0 or j==0 or i==b3_nnx-1 or j==b3_nny-1: b3_hull[counter]=True
            counter+=1
        #end for
    #end for

    counter=0
    for j in range(0,b4_nny):
        for i in range(0,b4_nnx):
            b4_x[counter]=i/float(b4_nelx)
            b4_y[counter]=j/float(b4_nely)
            if i==0 or j==0 or i==b4_nnx-1 or j==b4_nny-1: b4_hull[counter]=True
            counter+=1
        #end for
    #end for

    counter=0
    for j in range(0,b5_nny):
        for i in range(0,b5_nnx):
            b5_x[counter]=i/float(b5_nelx)
            b5_y[counter]=j/float(b5_nely)
            if i==0 or j==0 or i==b5_nnx-1 or j==b5_nny-1: b5_hull[counter]=True
            counter+=1
        #end for
    #end for

    counter=0
    for j in range(0,b6_nny):
        for i in range(0,b6_nnx):
            b6_x[counter]=i/float(b6_nelx)
            b6_y[counter]=j/float(b6_nely)
            if i==0 or j==0 or i==b6_nnx-1 or j==b6_nny-1: b6_hull[counter]=True
            counter+=1
        #end for
    #end for

    counter=0
    for j in range(0,b7_nny):
        for i in range(0,b7_nnx):
            b7_x[counter]=i/float(b7_nelx)
            b7_y[counter]=j/float(b7_nely)
            if i==0 or j==0 or i==b7_nnx-1 or j==b7_nny-1: b7_hull[counter]=True
            counter+=1
        #end for
    #end for

    counter=0
    for j in range(0,b8_nny):
        for i in range(0,b8_nnx):
            b8_x[counter]=i/float(b8_nelx)
            b8_y[counter]=j/float(b8_nely)
            if i==0 or j==0 or i==b8_nnx-1 or j==b8_nny-1: b8_hull[counter]=True
            counter+=1
        #end for
    #end for

    if debug:
       np.savetxt('temp1.ascii',np.array([b1_x,b1_y]).T,header='# x,y')
       np.savetxt('temp2.ascii',np.array([b2_x,b2_y]).T,header='# x,y')
       np.savetxt('temp3.ascii',np.array([b3_x,b3_y]).T,header='# x,y')
       np.savetxt('temp4.ascii',np.array([b4_x,b4_y]).T,header='# x,y')
       np.savetxt('temp5.ascii',np.array([b5_x,b5_y]).T,header='# x,y')
       np.savetxt('temp6.ascii',np.array([b6_x,b6_y]).T,header='# x,y')
       np.savetxt('temp7.ascii',np.array([b7_x,b7_y]).T,header='# x,y')
       np.savetxt('temp8.ascii',np.array([b8_x,b8_y]).T,header='# x,y')

    ###########################################################################
    # connectivity arrays
    ###########################################################################

    b1_icon=np.zeros((m,b1_nel),dtype=np.int32)
    b2_icon=np.zeros((m,b2_nel),dtype=np.int32)
    b3_icon=np.zeros((m,b3_nel),dtype=np.int32)
    b4_icon=np.zeros((m,b4_nel),dtype=np.int32)
    b5_icon=np.zeros((m,b5_nel),dtype=np.int32)
    b6_icon=np.zeros((m,b6_nel),dtype=np.int32)
    b7_icon=np.zeros((m,b7_nel),dtype=np.int32)
    b8_icon=np.zeros((m,b8_nel),dtype=np.int32)

    counter=0
    for j in range(0,b1_nely):
        for i in range(0,b1_nelx):
            b1_icon[0,counter]=i+j*(b1_nelx+1)
            b1_icon[1,counter]=i+1+j*(b1_nelx+1)
            b1_icon[2,counter]=i+1+(j+1)*(b1_nelx+1)
            b1_icon[3,counter]=i+(j+1)*(b1_nelx+1)
            counter+=1

    counter=0
    for j in range(0,b2_nely):
        for i in range(0,b2_nelx):
            b2_icon[0,counter]=i+j*(b2_nelx+1)
            b2_icon[1,counter]=i+1+j*(b2_nelx+1)
            b2_icon[2,counter]=i+1+(j+1)*(b2_nelx+1)
            b2_icon[3,counter]=i+(j+1)*(b2_nelx+1)
            counter+=1

    counter=0
    for j in range(0,b3_nely):
        for i in range(0,b3_nelx):
            b3_icon[0,counter]=i+j*(b3_nelx+1)
            b3_icon[1,counter]=i+1+j*(b3_nelx+1)
            b3_icon[2,counter]=i+1+(j+1)*(b3_nelx+1)
            b3_icon[3,counter]=i+(j+1)*(b3_nelx+1)
            counter+=1

    counter=0
    for j in range(0,b4_nely):
        for i in range(0,b4_nelx):
            b4_icon[0,counter]=i+j*(b4_nelx+1)
            b4_icon[1,counter]=i+1+j*(b4_nelx+1)
            b4_icon[2,counter]=i+1+(j+1)*(b4_nelx+1)
            b4_icon[3,counter]=i+(j+1)*(b4_nelx+1)
            counter+=1

    counter=0
    for j in range(0,b5_nely):
        for i in range(0,b5_nelx):
            b5_icon[0,counter]=i+j*(b5_nelx+1)
            b5_icon[1,counter]=i+1+j*(b5_nelx+1)
            b5_icon[2,counter]=i+1+(j+1)*(b5_nelx+1)
            b5_icon[3,counter]=i+(j+1)*(b5_nelx+1)
            counter+=1

    counter=0
    for j in range(0,b6_nely):
        for i in range(0,b6_nelx):
            b6_icon[0,counter]=i+j*(b6_nelx+1)
            b6_icon[1,counter]=i+1+j*(b6_nelx+1)
            b6_icon[2,counter]=i+1+(j+1)*(b6_nelx+1)
            b6_icon[3,counter]=i+(j+1)*(b6_nelx+1)
            counter+=1

    counter=0
    for j in range(0,b7_nely):
        for i in range(0,b7_nelx):
            b7_icon[0,counter]=i+j*(b7_nelx+1)
            b7_icon[1,counter]=i+1+j*(b7_nelx+1)
            b7_icon[2,counter]=i+1+(j+1)*(b7_nelx+1)
            b7_icon[3,counter]=i+(j+1)*(b7_nelx+1)
            counter+=1

    counter=0
    for j in range(0,b8_nely):
        for i in range(0,b8_nelx):
            b8_icon[0,counter]=i+j*(b8_nelx+1)
            b8_icon[1,counter]=i+1+j*(b8_nelx+1)
            b8_icon[2,counter]=i+1+(j+1)*(b8_nelx+1)
            b8_icon[3,counter]=i+(j+1)*(b8_nelx+1)
            counter+=1

    ###########################################################################
    # mapping block 1 to quadrilateral ABCD
    ###########################################################################

    counter=0
    for j in range(0,b1_nny):
        for i in range(0,b1_nnx):
            r=2*(b1_x[counter]-0.5)
            s=2*(b1_y[counter]-0.5)
            N_V=basis_functions_V(r,s)
            b1_x[counter]=N_V[0]*xA+N_V[1]*xB+N_V[2]*xC+N_V[3]*xD
            b1_y[counter]=N_V[0]*yA+N_V[1]*yB+N_V[2]*yC+N_V[3]*yD
            counter+=1
       #end for
    #end for

    # bend AD side
    counter=0
    for j in range(0,b1_nny):
        for i in range(0,b1_nnx):
            if i==0:
               angle=np.arctan(b1_y[counter]/b1_x[counter])
               b1_x[counter]=radius*np.cos(angle)
               b1_y[counter]=radius*np.sin(angle)
            counter+=1
       #end for
    #end for

    # recompute position of inside nodes
    counter=0
    for j in range(0,b1_nny):
        for i in range(0,b1_nnx):
            if i!=0 and i!=b1_nnx-1 and j!=0 and j!=b1_nny-1:
               inode_beg=j*b1_nnx
               inode_end=j*b1_nnx+b1_nnx-1
               b1_x[counter]=float(i)/b1_nelx*(b1_x[inode_end]-b1_x[inode_beg]) + b1_x[inode_beg]
               b1_y[counter]=float(i)/b1_nelx*(b1_y[inode_end]-b1_y[inode_beg]) + b1_y[inode_beg]
            counter+=1
       #end for
    #end for

    if verbose: print('mapped block 1')

    ###########################################################################
    # mapping block 2 to quadrilateral FDCE
    ###########################################################################
    
    counter=0
    for j in range(0,b2_nny):
        for i in range(0,b2_nnx):
            r=2*(b2_x[counter]-0.5)
            s=2*(b2_y[counter]-0.5)
            N_V=basis_functions_V(r,s)
            b2_x[counter]=N_V[0]*xD+N_V[1]*xC+N_V[2]*xE+N_V[3]*xF
            b2_y[counter]=N_V[0]*yD+N_V[1]*yC+N_V[2]*yE+N_V[3]*yF
            counter+=1
       #end for
    #end for

    #bend FD side
    counter=0
    for j in range(0,b2_nny):
        for i in range(0,b2_nnx):
            if i==0 and j<b2_nny-1:
               angle=np.arctan(b2_y[counter]/b2_x[counter])
               b2_x[counter]=radius*np.cos(angle)
               b2_y[counter]=radius*np.sin(angle)
            counter+=1
       #end for
    #end for

    # recompute position of inside nodes
    counter=0
    for j in range(0,b2_nny):
        for i in range(0,b2_nnx):
            if i!=0 and i!=b2_nnx-1 and j!=0 and j!=b2_nny-1:
               inode_beg=j*b2_nnx
               inode_end=j*b2_nnx+b2_nnx-1
               b2_x[counter]=float(i)/b2_nelx*(b2_x[inode_end]-b2_x[inode_beg]) + b2_x[inode_beg]
               b2_y[counter]=float(i)/b2_nelx*(b2_y[inode_end]-b2_y[inode_beg]) + b2_y[inode_beg]
            counter+=1
       #end for
    #end for

    if verbose: print('mapped block 2')

    ###########################################################################
    # mapping block 3 to quadrilateral BHGC 
    ###########################################################################

    counter=0
    for j in range(0,b3_nny):
        for i in range(0,b3_nnx):
            r=2*(b3_x[counter]-0.5)
            s=2*(b3_y[counter]-0.5)
            N_V=basis_functions_V(r,s)
            b3_x[counter]=N_V[0]*xB+N_V[1]*xH+N_V[2]*xG+N_V[3]*xC
            b3_y[counter]=N_V[0]*yB+N_V[1]*yH+N_V[2]*yG+N_V[3]*yC
            counter+=1
       #end for
    #end for

    if verbose: print('mapped block 3')

    ###########################################################################
    # mapping block 4 to quadrilateral OIHB 
    ###########################################################################

    counter=0
    for j in range(0,b4_nny):
        for i in range(0,b4_nnx):
            r=2*(b4_x[counter]-0.5)
            s=2*(b4_y[counter]-0.5)
            N_V=basis_functions_V(r,s)
            b4_x[counter]=N_V[0]*xO+N_V[1]*xI+N_V[2]*xH+N_V[3]*xB
            b4_y[counter]=N_V[0]*yO+N_V[1]*yI+N_V[2]*yH+N_V[3]*yB
            counter+=1
       #end for
    #end for

    if verbose: print('mapped block 4')

    ###########################################################################
    # mapping block 5 to block KJIO
    ###########################################################################

    counter=0
    for j in range(0,b5_nny):
        for i in range(0,b5_nnx):
            r=2*(b5_x[counter]-0.5)
            s=2*(b5_y[counter]-0.5)
            N_V=basis_functions_V(r,s)
            b5_x[counter]=N_V[0]*xK+N_V[1]*xJ+N_V[2]*xI+N_V[3]*xO
            b5_y[counter]=N_V[0]*yK+N_V[1]*yJ+N_V[2]*yI+N_V[3]*yO
            counter+=1
       #end for
    #end for

    if verbose: print('mapped block 5')

    ###########################################################################
    # mapping block 6 to quadrilateral LKOM
    ###########################################################################

    counter=0
    for j in range(0,b6_nny):
        for i in range(0,b6_nnx):
            r=2*(b6_x[counter]-0.5)
            s=2*(b6_y[counter]-0.5)
            N_V=basis_functions_V(r,s)
            b6_x[counter]=N_V[0]*xL+N_V[1]*xK+N_V[2]*xO+N_V[3]*xM
            b6_y[counter]=N_V[0]*yL+N_V[1]*yK+N_V[2]*yO+N_V[3]*yM
            counter+=1
       #end for
    #end for

    if verbose: print('mapped block 6')

    ###########################################################################
    # mapping block 7 to NMOP quadrilateral
    ###########################################################################

    counter=0
    for j in range(0,b7_nny):
        for i in range(0,b7_nnx):
            r=2*(b7_x[counter]-0.5)
            s=2*(b7_y[counter]-0.5)
            N_V=basis_functions_V(r,s)
            b7_x[counter]=N_V[0]*xN+N_V[1]*xM+N_V[2]*xO+N_V[3]*xP
            b7_y[counter]=N_V[0]*yN+N_V[1]*yM+N_V[2]*yO+N_V[3]*yP
            counter+=1
       #end for
    #end for

    # bend NP side
    counter=0
    for j in range(0,b7_nny):
        for i in range(0,b7_nnx):
            if i==0 and j>0:
               angle=np.arctan(b7_y[counter]/b7_x[counter])
               b7_x[counter]=radius*np.cos(angle)
               b7_y[counter]=radius*np.sin(angle)
            counter+=1
       #end for
    #end for

    # recompute position of inside nodes
    counter=0
    for j in range(0,b7_nny):
        for i in range(0,b7_nnx):
            if i!=0 and i!=b7_nnx-1 and j!=0 and j!=b7_nny-1:
               inode_beg=j*b7_nnx
               inode_end=j*b7_nnx+b7_nnx-1
               b7_x[counter]=float(i)/b7_nelx*(b7_x[inode_end]-b7_x[inode_beg]) + b7_x[inode_beg]
               b7_y[counter]=float(i)/b7_nelx*(b7_y[inode_end]-b7_y[inode_beg]) + b7_y[inode_beg]
            counter+=1
       #end for
    #end for

    if verbose: print('mapped block 7')

    ###########################################################################
    # mapping block 8 to quadrilateral POBA
    ###########################################################################

    counter=0
    for j in range(0,b8_nny):
        for i in range(0,b8_nnx):
            r=2*(b8_x[counter]-0.5)
            s=2*(b8_y[counter]-0.5)
            N_V=basis_functions_V(r,s)
            b8_x[counter]=N_V[0]*xP+N_V[1]*xO+N_V[2]*xB+N_V[3]*xA
            b8_y[counter]=N_V[0]*yP+N_V[1]*yO+N_V[2]*yB+N_V[3]*yA
            counter+=1
       #end for
    #end for

    # bend PA side
    counter=0
    for j in range(0,b8_nny):
        for i in range(0,b8_nnx):
            if i==0:
               angle=np.arctan(b8_y[counter]/b8_x[counter])
               b8_x[counter]=radius*np.cos(angle)
               b8_y[counter]=radius*np.sin(angle)
            counter+=1
       #end for
    #end for

    # recompute position of inside nodes
    counter=0
    for j in range(0,b8_nny):
        for i in range(0,b8_nnx):
            if i!=0 and i!=b8_nnx-1 and j!=0 and j!=b8_nny-1:
               inode_beg=j*b8_nnx
               inode_end=j*b8_nnx+b8_nnx-1
               b8_x[counter]=float(i)/b8_nelx*(b8_x[inode_end]-b8_x[inode_beg]) + b8_x[inode_beg]
               b8_y[counter]=float(i)/b8_nelx*(b8_y[inode_end]-b8_y[inode_beg]) + b8_y[inode_beg]
            counter+=1
       #end for
    #end for

    if verbose: print('mapped block 8')

    ###########################################################################

    if debug:
       np.savetxt('temp11.ascii',np.array([b1_x,b1_y]).T,header='# x,y')
       np.savetxt('temp22.ascii',np.array([b2_x,b2_y]).T,header='# x,y')
       np.savetxt('temp33.ascii',np.array([b3_x,b3_y]).T,header='# x,y')
       np.savetxt('temp44.ascii',np.array([b4_x,b4_y]).T,header='# x,y')
       np.savetxt('temp55.ascii',np.array([b5_x,b5_y]).T,header='# x,y')
       np.savetxt('temp66.ascii',np.array([b6_x,b6_y]).T,header='# x,y')
       np.savetxt('temp77.ascii',np.array([b7_x,b7_y]).T,header='# x,y')
       np.savetxt('temp88.ascii',np.array([b8_x,b8_y]).T,header='# x,y')

    ###########################################################################
    # merging blocks 
    ###########################################################################

    NNN=b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV+b8_NV

    tempx=np.zeros(NNN,dtype=np.float64)
    tempy=np.zeros(NNN,dtype=np.float64)
    temphull=np.zeros(NNN,dtype=bool)

    tempx[0:b1_NV]=b1_x[:]
    tempx[b1_NV:b1_NV+b2_NV]=b2_x[:]
    tempx[b1_NV+b2_NV:b1_NV+b2_NV+b3_NV]=b3_x[:]
    tempx[b1_NV+b2_NV+b3_NV:b1_NV+b2_NV+b3_NV+b4_NV]=b4_x[:]
    tempx[b1_NV+b2_NV+b3_NV+b4_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV]=b5_x[:]
    tempx[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV]=b6_x[:]
    tempx[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV]=b7_x[:]
    tempx[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV+b8_NV]=b8_x[:]

    tempy[0:b1_NV]=b1_y[:]
    tempy[b1_NV:b1_NV+b2_NV]=b2_y[:]
    tempy[b1_NV+b2_NV:b1_NV+b2_NV+b3_NV]=b3_y[:]
    tempy[b1_NV+b2_NV+b3_NV:b1_NV+b2_NV+b3_NV+b4_NV]=b4_y[:]
    tempy[b1_NV+b2_NV+b3_NV+b4_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV]=b5_y[:]
    tempy[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV]=b6_y[:]
    tempy[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV]=b7_y[:]
    tempy[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV+b8_NV]=b8_y[:]

    temphull[0:b1_NV]=b1_hull[:]
    temphull[b1_NV:b1_NV+b2_NV]=b2_hull[:]
    temphull[b1_NV+b2_NV:b1_NV+b2_NV+b3_NV]=b3_hull[:]
    temphull[b1_NV+b2_NV+b3_NV:b1_NV+b2_NV+b3_NV+b4_NV]=b4_hull[:]
    temphull[b1_NV+b2_NV+b3_NV+b4_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV]=b5_hull[:]
    temphull[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV]=b6_hull[:]
    temphull[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV]=b7_hull[:]
    temphull[b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV:b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV+b8_NV]=b8_hull[:]

    doubble=np.zeros(NNN,dtype=bool)  
    pointto=np.zeros(NNN,dtype=np.int32)

    for i in range(0,NNN):
        pointto[i]=i

    distance=1e-6*L

    counter=0
    for i in range(1,NNN):
        if temphull[i]:
           gxi=tempx[i]
           gyi=tempy[i]
           for j in range(0,i):
              if abs(gxi-tempx[j])<distance and abs(gyi-tempy[j])<distance:
                  doubble[i]=True
                  pointto[i]=j
                  break
              #end if
           #end do
        #end if
    #end do

    nn_V=NNN-sum(doubble)
    nel=b1_nel+b2_nel+b3_nel+b4_nel+b5_nel+b6_nel+b7_nel+b8_nel

    print('     -> doubles=',sum(doubble))
    print('     -> nn_V=',nn_V)
    print('     -> nel=',nel)

    x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
    y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
    icon_V=np.zeros((m,nel),dtype=np.int32)

    counter=0
    for i in range(0,NNN):
        if not doubble[i]: 
           x_V[counter]=tempx[i]#+Lx
           y_V[counter]=tempy[i]#+Ly
           counter+=1

    icon_V[0:m,0:b1_nel]\
          =b1_icon[0:m,0:b1_nel]
    icon_V[0:m,b1_nel:b1_nel+b2_nel]\
          =b2_icon[0:m,0:b2_nel]+b1_NV
    icon_V[0:m,b1_nel+b2_nel:b1_nel+b2_nel+b3_nel]\
          =b3_icon[0:m,0:b3_nel]+b1_NV+b2_NV
    icon_V[0:m,b1_nel+b2_nel+b3_nel:b1_nel+b2_nel+b3_nel+b4_nel]\
          =b4_icon[0:m,0:b4_nel]+b1_NV+b2_NV+b3_NV
    icon_V[0:m,b1_nel+b2_nel+b3_nel+b4_nel:b1_nel+b2_nel+b3_nel+b4_nel+b5_nel]\
          =b5_icon[0:m,0:b5_nel]+b1_NV+b2_NV+b3_NV+b4_NV
    icon_V[0:m,b1_nel+b2_nel+b3_nel+b4_nel+b5_nel:b1_nel+b2_nel+b3_nel+b4_nel+b5_nel+b6_nel]\
          =b6_icon[0:m,0:b6_nel]+b1_NV+b2_NV+b3_NV+b4_NV+b5_NV
    icon_V[0:m,b1_nel+b2_nel+b3_nel+b4_nel+b5_nel+b6_nel:b1_nel+b2_nel+b3_nel+b4_nel+b5_nel+b6_nel+b7_nel]\
          =b7_icon[0:m,0:b7_nel]+b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV
    icon_V[0:m,b1_nel+b2_nel+b3_nel+b4_nel+b5_nel+b6_nel+b7_nel:b1_nel+b2_nel+b3_nel+b4_nel+b5_nel+b6_nel+b7_nel+b8_nel]\
          =b8_icon[0:m,0:b8_nel]+b1_NV+b2_NV+b3_NV+b4_NV+b5_NV+b6_NV+b7_NV

    for iel in range(0,nel):
        for i in range(0,m):
            icon_V[i,iel]=pointto[icon_V[i,iel]]

    compact=np.zeros(NNN,dtype=np.int32)

    counter=0
    for i in range(0,NNN):
        if not doubble[i]:
           compact[i]=counter
           counter=counter+1

    for iel in range(0,nel):
        for i in range(0,m):
          icon_V[i,iel]=compact[icon_V[i,iel]]

    if debug: np.savetxt('mesh.ascii',np.array([x_V,y_V]).T,header='# x,y')

    Nfem=nn_V*2

    return x_V,y_V,icon_V,nn_V,nel,Nfem


