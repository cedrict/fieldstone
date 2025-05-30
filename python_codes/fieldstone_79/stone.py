import numpy as np
import random 
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import sys

experiment = 1

###############################################################################

def Tbc(x,y):
    #----------------
    if experiment==1:
       #Tleft
       if x<1e-6:
         val=x+y 
       #Tright
       if abs(x-Lx)<1e-6:
         val=x+y 
       #Tbottom
       if y<1e-6:
         val=x+y 
       #Ttop
       if abs(y-Ly)<1e-6:
         val=x+y
    #----------------
    if experiment==2:
       #Tleft
       if x<1e-6:
         val=y*(Ly-y)+x*(Lx-x) 
       #Tright
       if abs(x-Lx)<1e-6:
         val=y*(Ly-y)+x*(Lx-x) 
       #Tbottom
       if y<1e-6:
         val=y*(Ly-y)+x*(Lx-x) 
       #Ttop
       if abs(y-Ly)<1e-6:
         val=y*(Ly-y)+x*(Lx-x) 

    return val

def T_analytical(x,y):
    if experiment==1:
       return x+y
    if experiment==2:
       return 0 

def qx_analytical(x,y):
    if experiment==1:
       return -1
    if experiment==2:
       return 0

def qy_analytical(x,y):
    if experiment==1:
       return -1
    if experiment==2:
       return 0

###############################################################################

# allowing for argument parsing through command line
if int(len(sys.argv) == 3):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
else:
   nelx=20
   nely=20

Lx=1.
Ly=1.


m=3
nedge=m

if m==3:
   nel=nelx*nely*2 # two triangles per square
if m==4:
   nel=nelx*nely   # square elements

NT=nel*m       # total number of T nodes 
NQ=nel*m       # total number of q nodes 

hx=Lx/nelx
hy=Ly/nely

C12_inside=0.5 # do not change
C11=5   #????

niter=100

print('m=',m)
print('nel=',nel)
print('NT=',NT)

visu=True

debug=False

tol=1e-9
    
visualise_all=False

#########################
#physical pb 

Tinit =0

hcond=1

T_stats_file=open('T_stats.ascii',"w")
qx_stats_file=open('qx_stats.ascii',"w")
qy_stats_file=open('qy_stats.ascii',"w")
residual_T_stats_file=open('residual_T_stats.ascii',"w")
residual_qx_stats_file=open('residual_qx_stats.ascii',"w")
residual_qy_stats_file=open('residual_qy_stats.ascii',"w")

###############################################################################
# fill icon array
###############################################################################
icon =np.zeros((m,nel),dtype=np.int32)

if debug:
   print('filling icon array')

if m==3:
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           icon[0, counter] = m*counter+0
           icon[1, counter] = m*counter+1
           icon[2, counter] = m*counter+2
           if debug:
              print(counter,icon[:,counter]+1)
           counter += 1
           icon[0, counter] = m*counter+0
           icon[1, counter] = m*counter+1
           icon[2, counter] = m*counter+2
           if debug:
              print(counter,icon[:,counter]+1)
           counter += 1
       #end for
   #end for

if m==4:
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           icon[0, counter] = m*counter+0
           icon[1, counter] = m*counter+1
           icon[2, counter] = m*counter+2
           icon[3, counter] = m*counter+3
           counter += 1
       #end for
   #end for

###############################################################################
# compute coordinates of nodes
###############################################################################
xT = np.empty(NT,dtype=np.float64)  # x coordinates
yT = np.empty(NT,dtype=np.float64)  # y coordinates

if m==3:
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           xT[counter]=i*hx
           yT[counter]=j*hy
           #print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           xT[counter]=i*hx+hx
           yT[counter]=j*hy
           #print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           xT[counter]=i*hx
           yT[counter]=j*hy+hy
           #print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           #-----------------
           xT[counter]=i*hx+hx
           yT[counter]=j*hy
           #print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           xT[counter]=i*hx+hx
           yT[counter]=j*hy+hy
           #print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           xT[counter]=i*hx
           yT[counter]=j*hy+hy
           #print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
       #end for
   #end for

if m==4:
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           xT[counter]=i*hx
           yT[counter]=j*hy
           counter+=1
           xT[counter]=i*hx+hx
           yT[counter]=j*hy
           counter+=1
           xT[counter]=i*hx+hx
           yT[counter]=j*hy+hy
           counter+=1
           xT[counter]=i*hx
           yT[counter]=j*hy+hy
           counter+=1
       #end for
   #end for

np.savetxt('grid.ascii',np.array([xT,yT]).T,header='# x,y')

###############################################################################
# filling T,qx,qy arrays with random values
###############################################################################

residual_T  = np.zeros(NT,dtype=np.float64)  
residual_qx = np.zeros(NT,dtype=np.float64)  
residual_qy = np.zeros(NT,dtype=np.float64)  

T = np.zeros(NT,dtype=np.float64)  
qx = np.zeros(NQ,dtype=np.float64) 
qy = np.zeros(NQ,dtype=np.float64) 
area = np.empty(nel,dtype=np.float64) 

T[:]=Tinit

#T[:]=xT[:]
#qx[:]=1
#qy[:]=0

#T[:]=yT[:]
#qx[:]=0
#qy[:]=1

#T  = np.random.rand(NT)
#qx  = np.random.rand(NQ)
#qy  = np.random.rand(NQ)

###############################################################################
# computing edges normals and lengths, mid-point coordinates, element area
# note that I am cheating for triangles: I do not automatically compute 
# the normal vector but use the current regular layout information!
# Also quadrilaterals are assumed to be rectangles.

#  +---4---+   
#  |       |
#  1       2  edgeX_boundary_indicator information
#  |       |
#  +---3---+
#
###############################################################################

if m==3:
   edge1_nx = np.zeros(nel,dtype=np.float64) # x component of normal to edge1
   edge1_ny = np.zeros(nel,dtype=np.float64) # y component of normal to edge1
   edge2_nx = np.zeros(nel,dtype=np.float64) # x component of normal to edge2
   edge2_ny = np.zeros(nel,dtype=np.float64) # y component of normal to edge2
   edge3_nx = np.zeros(nel,dtype=np.float64) # x component of normal to edge3
   edge3_ny = np.zeros(nel,dtype=np.float64) # y component of normal to edge3
   edge1_xc = np.zeros(nel,dtype=np.float64) # x coord of edge1 midpoint
   edge1_yc = np.zeros(nel,dtype=np.float64) # y coord of edge1 midpoint
   edge2_xc = np.zeros(nel,dtype=np.float64) # x coord of edge2 midpoint
   edge2_yc = np.zeros(nel,dtype=np.float64) # y coord of edge2 midpoint
   edge3_xc = np.zeros(nel,dtype=np.float64) # x coord of edge3 midpoint
   edge3_yc = np.zeros(nel,dtype=np.float64) # y coord of edge3 midpoint
   edge1_L = np.zeros(nel,dtype=np.float64) # length of edge 1 
   edge2_L = np.zeros(nel,dtype=np.float64) # length of edge 2 
   edge3_L = np.zeros(nel,dtype=np.float64) # length of edge 3 
   edge1_boundary_indicator = np.zeros(nel,dtype=np.int32) # if edge1 on a boundary then contains its boundary indicator (1-4)
   edge2_boundary_indicator = np.zeros(nel,dtype=np.int32) # is edge2 on a boundary then contains its boundary indicator (1-4)
   edge3_boundary_indicator = np.zeros(nel,dtype=np.int32) # is edge3 on a boundary then contains its boundary indicator (1-4)
   edge1_neighb = np.zeros(nel,dtype=np.int32) # neighbour element on the other side of edge 1 
   edge2_neighb = np.zeros(nel,dtype=np.int32) # neighbour element on the other side of edge 2 
   edge3_neighb = np.zeros(nel,dtype=np.int32) # neighbour element on the other side of edge 3 
   edge1_neighb.fill(-1)
   edge2_neighb.fill(-1)
   edge3_neighb.fill(-1)
   edge1_neighbedge = np.zeros(nel,dtype=np.int32) # neighbour element edge number on the other side of edge 1 
   edge2_neighbedge = np.zeros(nel,dtype=np.int32) # neighbour element edge number on the other side of edge 2 
   edge3_neighbedge = np.zeros(nel,dtype=np.int32) # neighbour element edge number on the other side of edge 3 

   iel=0
   for j in range(0,nely):
       for i in range(0,nelx):
           #lower left triangle
           x1=xT[icon[0,iel]]
           x2=xT[icon[1,iel]]
           x3=xT[icon[2,iel]]
           y1=yT[icon[0,iel]]
           y2=yT[icon[1,iel]]
           y3=yT[icon[2,iel]]
           edge1_nx[iel]=0
           edge1_ny[iel]=-1
           edge2_nx[iel]=1/np.sqrt(2)
           edge2_ny[iel]=1/np.sqrt(2)
           edge3_nx[iel]=-1
           edge3_ny[iel]=0
           edge1_xc[iel]=(x1+x2)/2.
           edge1_yc[iel]=(y1+y2)/2.
           edge2_xc[iel]=(x2+x3)/2.
           edge2_yc[iel]=(y2+y3)/2.
           edge3_xc[iel]=(x3+x1)/2.
           edge3_yc[iel]=(y3+y1)/2.
           edge1_L[iel]=np.sqrt((x1-x2)**2+(y1-y2)**2)
           edge2_L[iel]=np.sqrt((x3-x2)**2+(y3-y2)**2)
           edge3_L[iel]=np.sqrt((x1-x3)**2+(y1-y3)**2)
           area[iel]=0.5*((x1-x3)*(y2-y3)-(x2-x3)*(y1-y3))
           if edge1_xc[iel]<1e-6:
              edge1_boundary_indicator[iel]=1
           if edge2_xc[iel]<1e-6:
              edge2_boundary_indicator[iel]=1
           if edge3_xc[iel]<1e-6:
              edge3_boundary_indicator[iel]=1

           if abs(edge1_xc[iel]-Lx)<1e-6:
              edge1_boundary_indicator[iel]=2
           if abs(edge2_xc[iel]-Lx)<1e-6:
              edge2_boundary_indicator[iel]=2
           if abs(edge3_xc[iel]-Lx)<1e-6:
              edge3_boundary_indicator[iel]=2

           if edge1_yc[iel]<1e-6:
              edge1_boundary_indicator[iel]=3
           if edge2_yc[iel]<1e-6:
              edge2_boundary_indicator[iel]=3
           if edge3_yc[iel]<1e-6:
              edge3_boundary_indicator[iel]=3

           if abs(edge1_yc[iel]-Ly)<1e-6:
              edge1_boundary_indicator[iel]=4
           if abs(edge2_yc[iel]-Ly)<1e-6:
              edge2_boundary_indicator[iel]=4
           if abs(edge3_yc[iel]-Ly)<1e-6:
              edge3_boundary_indicator[iel]=4

           if edge1_boundary_indicator[iel]==0: 
              edge1_neighb[iel]=iel-(2*nelx-1)
           if edge2_boundary_indicator[iel]==0:
              edge2_neighb[iel]=iel+1
           if edge3_boundary_indicator[iel]==0: 
              edge3_neighb[iel]=iel-1
           #print('iel',iel,'has neighbours',edge1_neighb[iel],edge2_neighb[iel],edge3_neighb[iel])
           iel+=1

           #upper right triangle
           x1=xT[icon[0,iel]]
           x2=xT[icon[1,iel]]
           x3=xT[icon[2,iel]]
           y1=yT[icon[0,iel]]
           y2=yT[icon[1,iel]]
           y3=yT[icon[2,iel]]
           edge1_nx[iel]=1
           edge1_ny[iel]=0
           edge2_nx[iel]=0
           edge2_ny[iel]=1
           edge3_nx[iel]=-1/np.sqrt(2)
           edge3_ny[iel]=-1/np.sqrt(2)
           edge1_xc[iel]=(x1+x2)/2.
           edge1_yc[iel]=(y1+y2)/2.
           edge2_xc[iel]=(x2+x3)/2.
           edge2_yc[iel]=(y2+y3)/2.
           edge3_xc[iel]=(x3+x1)/2.
           edge3_yc[iel]=(y3+y1)/2.
           edge1_L[iel]=np.sqrt((x1-x2)**2+(y1-y2)**2)
           edge2_L[iel]=np.sqrt((x3-x2)**2+(y3-y2)**2)
           edge3_L[iel]=np.sqrt((x1-x3)**2+(y1-y3)**2)
           area[iel]=0.5*((x1-x3)*(y2-y3)-(x2-x3)*(y1-y3))

           if edge1_xc[iel]<1e-6:
              edge1_boundary_indicator[iel]=1
           if edge2_xc[iel]<1e-6:
              edge2_boundary_indicator[iel]=1
           if edge3_xc[iel]<1e-6:
              edge3_boundary_indicator[iel]=1

           if abs(edge1_xc[iel]-Lx)<1e-6:
              edge1_boundary_indicator[iel]=2
           if abs(edge2_xc[iel]-Lx)<1e-6:
              edge2_boundary_indicator[iel]=2
           if abs(edge3_xc[iel]-Lx)<1e-6:
              edge3_boundary_indicator[iel]=2

           if edge1_yc[iel]<1e-6:
              edge1_boundary_indicator[iel]=3
           if edge2_yc[iel]<1e-6:
              edge2_boundary_indicator[iel]=3
           if edge3_yc[iel]<1e-6:
              edge3_boundary_indicator[iel]=3

           if abs(edge1_yc[iel]-Ly)<1e-6:
              edge1_boundary_indicator[iel]=4
           if abs(edge2_yc[iel]-Ly)<1e-6:
              edge2_boundary_indicator[iel]=4
           if abs(edge3_yc[iel]-Ly)<1e-6:
              edge3_boundary_indicator[iel]=4

           if edge1_boundary_indicator[iel]==0: 
              edge1_neighb[iel]=iel+1
           if edge2_boundary_indicator[iel]==0: 
              edge2_neighb[iel]=iel+(2*nelx-1)
           if edge3_boundary_indicator[iel]==0: 
              edge3_neighb[iel]=iel-1
           #print('iel',iel,'has neighbours',edge1_neighb[iel],edge2_neighb[iel],edge3_neighb[iel])

           iel+=1
       #end for
   #end for

   #np.savetxt('edge1.ascii',np.array([edge1_xc,edge1_yc,edge1_nx/10,edge1_ny/10,edge1_boundary_indicator]).T)
   #np.savetxt('edge2.ascii',np.array([edge2_xc,edge2_yc,edge2_nx/10,edge2_ny/10,edge2_boundary_indicator]).T)
   #np.savetxt('edge3.ascii',np.array([edge3_xc,edge3_yc,edge3_nx/10,edge3_ny/10,edge3_boundary_indicator]).T)

   #in what follows local numbering is 0,1,2 inside triangle
   for iel in range(0,nel):
       x0=xT[icon[0,iel]]
       x1=xT[icon[1,iel]]
       x2=xT[icon[2,iel]]
       y0=yT[icon[0,iel]]
       y1=yT[icon[1,iel]]
       y2=yT[icon[2,iel]]
       #--------------------edge 1 (nodes 0&1)-------------------------
       if edge1_neighb[iel]!=-1: # if there is element on other side
          jel=edge1_neighb[iel] # identity of neighbour 
          # case1: is it edge 1 of neighbour element?
          if abs(x0-xT[icon[1,jel]])<1e-6 and abs(y0-yT[icon[1,jel]])<1e-6 and\
             abs(x1-xT[icon[0,jel]])<1e-6 and abs(y1-yT[icon[0,jel]])<1e-6:
             edge1_neighbedge[iel] = 1 
             #print ('elt',iel,'looking at neighbour',jel,'through edge1')
             #print('corresponding edge of neighbour is edge',edge1_neighbedge[iel])
          # case2: is it edge 2 of neighbour element?
          if abs(x0-xT[icon[2,jel]])<1e-6 and abs(y0-yT[icon[2,jel]])<1e-6 and\
             abs(x1-xT[icon[1,jel]])<1e-6 and abs(y1-yT[icon[1,jel]])<1e-6:
             edge1_neighbedge[iel] = 2 
             #print('elt',iel,'looking at neighbour',jel,'through edge1')
             #print('corresponding edge of neighbour is edge',edge1_neighbedge[iel])
          # case3: is it edge 3 of neighbour element?
          if abs(x0-xT[icon[0,jel]])<1e-6 and abs(y0-yT[icon[0,jel]])<1e-6 and\
             abs(x1-xT[icon[2,jel]])<1e-6 and abs(y1-yT[icon[2,jel]])<1e-6:
             edge1_neighbedge[iel] = 3 
             #print ('elt',iel,'looking at neighbour',jel,'through edge1')
             #print('corresponding edge of neighbour is edge',edge1_neighbedge[iel])
        # end if

       #--------------------edge 2 (nodes 1&2)-------------------------
       if edge2_neighb[iel]!=-1: # if there is element on other side
          jel=edge2_neighb[iel] # identity of neighbour 
          # case1: is it edge 1 of neighbour element?
          if abs(x1-xT[icon[1,jel]])<1e-6 and abs(y1-yT[icon[1,jel]])<1e-6 and\
             abs(x2-xT[icon[0,jel]])<1e-6 and abs(y2-yT[icon[0,jel]])<1e-6:
             edge2_neighbedge[iel] = 1 
             #print ('elt',iel,'looking at neighbour',jel,'through edge2')
             #print('corresponding edge of neighbour is edge',edge2_neighbedge[iel])
          # case2: is it edge 2 of neighbour element?
          if abs(x1-xT[icon[2,jel]])<1e-6 and abs(y1-yT[icon[2,jel]])<1e-6 and\
             abs(x2-xT[icon[1,jel]])<1e-6 and abs(y2-yT[icon[1,jel]])<1e-6:
             edge2_neighbedge[iel] = 2 
             #print('elt',iel,'looking at neighbour',jel,'through edge2')
             #print('corresponding edge of neighbour is edge',edge2_neighbedge[iel])
          # case3: is it edge 3 of neighbour element?
          if abs(x1-xT[icon[0,jel]])<1e-6 and abs(y1-yT[icon[0,jel]])<1e-6 and\
             abs(x2-xT[icon[2,jel]])<1e-6 and abs(y2-yT[icon[2,jel]])<1e-6:
             edge2_neighbedge[iel] = 3 
             #print ('elt',iel,'looking at neighbour',jel,'through edge2')
             #print('corresponding edge of neighbour is edge',edge2_neighbedge[iel])
        # end if

       #--------------------edge 3 (nodes 2&0)-------------------------
       if edge3_neighb[iel]!=-1: # if there is element on other side
          jel=edge3_neighb[iel] # identity of neighbour 
          # case1: is it edge 1 of neighbour element?
          if abs(x2-xT[icon[1,jel]])<1e-6 and abs(y2-yT[icon[1,jel]])<1e-6 and\
             abs(x0-xT[icon[0,jel]])<1e-6 and abs(y0-yT[icon[0,jel]])<1e-6:
             edge3_neighbedge[iel] = 1 
             #print ('elt',iel,'looking at neighbour',jel,'through edge3')
             #print('corresponding edge of neighbour is edge',edge3_neighbedge[iel])
          # case2: is it edge 2 of neighbour element?
          if abs(x2-xT[icon[2,jel]])<1e-6 and abs(y2-yT[icon[2,jel]])<1e-6 and\
             abs(x0-xT[icon[1,jel]])<1e-6 and abs(y0-yT[icon[1,jel]])<1e-6:
             edge3_neighbedge[iel] = 2 
             #print('elt',iel,'looking at neighbour',jel,'through edge3')
             #print('corresponding edge of neighbour is edge',edge3_neighbedge[iel])
          # case3: is it edge 3 of neighbour element?
          if abs(x2-xT[icon[0,jel]])<1e-6 and abs(y2-yT[icon[0,jel]])<1e-6 and\
             abs(x0-xT[icon[2,jel]])<1e-6 and abs(y0-yT[icon[2,jel]])<1e-6:
             edge3_neighbedge[iel] = 3 
             #print ('elt',iel,'looking at neighbour',jel,'through edge3')
             #print('corresponding edge of neighbour is edge',edge3_neighbedge[iel])
       # end if

   #end for iel            
            
if m==4:
   edge1_nx = np.zeros(nel) # x component of normal to edge1
   edge1_ny = np.zeros(nel) # y component of normal to edge1
   edge2_nx = np.zeros(nel) # x component of normal to edge2
   edge2_ny = np.zeros(nel) # y component of normal to edge2
   edge3_nx = np.zeros(nel) # x component of normal to edge3
   edge3_ny = np.zeros(nel) # y component of normal to edge3
   edge4_nx = np.zeros(nel) # x component of normal to edge4
   edge4_ny = np.zeros(nel) # y component of normal to edge4
   edge1_xc = np.zeros(nel) # x coord of edge1 midpoint
   edge1_yc = np.zeros(nel) # y coord of edge1 midpoint
   edge2_xc = np.zeros(nel) # x coord of edge2 midpoint
   edge2_yc = np.zeros(nel) # y coord of edge2 midpoint
   edge3_xc = np.zeros(nel) # x coord of edge3 midpoint
   edge3_yc = np.zeros(nel) # y coord of edge3 midpoint
   edge4_xc = np.zeros(nel) # x coord of edge4 midpoint
   edge4_yc = np.zeros(nel) # y coord of edge4 midpoint
   edge1_boundary_indicator = np.zeros(nel,dtype=np.int32) # if edge1 on a boundary then contains its boundary indicator (1-4)
   edge2_boundary_indicator = np.zeros(nel,dtype=np.int32) # is edge2 on a boundary then contains its boundary indicator (1-4)
   edge3_boundary_indicator = np.zeros(nel,dtype=np.int32) # is edge3 on a boundary then contains its boundary indicator (1-4)
   edge4_boundary_indicator = np.zeros(nel,dtype=np.int32) # is edge4 on a boundary then contains its boundary indicator (1-4)
   edge1_neighb = np.zeros(nel,dtype=np.int32) # neighbour element on the other side of edge 1 
   edge2_neighb = np.zeros(nel,dtype=np.int32) # neighbour element on the other side of edge 2 
   edge3_neighb = np.zeros(nel,dtype=np.int32) # neighbour element on the other side of edge 3 
   edge4_neighb = np.zeros(nel,dtype=np.int32) # neighbour element on the other side of edge 4 
   edge1_neighb.fill(-1)
   edge2_neighb.fill(-1)
   edge3_neighb.fill(-1)
   edge4_neighb.fill(-1)

   for iel in range(0,nel):
       edge1_nx[iel]=0
       edge1_ny[iel]=-1
       edge2_nx[iel]=1
       edge2_ny[iel]=0
       edge3_nx[iel]=0
       edge3_ny[iel]=1
       edge4_nx[iel]=-1
       edge4_ny[iel]=0
       edge1_xc[iel]=(xT[icon[0,iel]]+xT[icon[1,iel]])/2.
       edge1_yc[iel]=(yT[icon[0,iel]]+yT[icon[1,iel]])/2.
       edge2_xc[iel]=(xT[icon[1,iel]]+xT[icon[2,iel]])/2.
       edge2_yc[iel]=(yT[icon[1,iel]]+yT[icon[2,iel]])/2.
       edge3_xc[iel]=(xT[icon[2,iel]]+xT[icon[3,iel]])/2.
       edge3_yc[iel]=(yT[icon[2,iel]]+yT[icon[3,iel]])/2.
       edge4_xc[iel]=(xT[icon[3,iel]]+xT[icon[0,iel]])/2.
       edge4_yc[iel]=(yT[icon[3,iel]]+yT[icon[0,iel]])/2.
       edge1_L=hx
       edge2_L=hy
       edge3_L=hx
       edge4_L=hy
       area[iel]=hx*hy
       if edge1_xc[iel]<1e-6:
           edge1_boundary_indicator[iel]=1
       if edge2_xc[iel]<1e-6:
           edge2_boundary_indicator[iel]=1
       if edge3_xc[iel]<1e-6:
           edge3_boundary_indicator[iel]=1       
       if edge4_xc[iel]<1e-6:
           edge4_boundary_indicator[iel]=1
           
       if abs(edge1_xc[iel]-Lx)<1e-6:
           edge1_boundary_indicator[iel]=2
       if abs(edge2_xc[iel]-Lx)<1e-6:
           edge2_boundary_indicator[iel]=2
       if abs(edge3_xc[iel]-Lx)<1e-6:
           edge3_boundary_indicator[iel]=2       
       if abs(edge4_xc[iel]-Lx)<1e-6:
           edge4_boundary_indicator[iel]=2
           
       if edge1_yc[iel]<1e-6:
           edge1_boundary_indicator[iel]=3
       if edge2_yc[iel]<1e-6:
           edge2_boundary_indicator[iel]=3
       if edge3_yc[iel]<1e-6:
           edge3_boundary_indicator[iel]=3       
       if edge4_yc[iel]<1e-6:
           edge4_boundary_indicator[iel]=3
                
       if abs(edge1_yc[iel]-Ly)<1e-6:
           edge1_boundary_indicator[iel]=4
       if abs(edge2_yc[iel]-Ly)<1e-6:
           edge2_boundary_indicator[iel]=4
       if abs(edge3_yc[iel]-Ly)<1e-6:
           edge3_boundary_indicator[iel]=4       
       if abs(edge4_yc[iel]-Ly)<1e-6:
           edge4_boundary_indicator[iel]=4
           
       if edge1_boundary_indicator[iel]==0: 
          edge1_neighb[iel]=iel-nelx
       if edge2_boundary_indicator[iel]==0: 
          edge2_neighb[iel]=iel+1
       if edge3_boundary_indicator[iel]==0: 
          edge3_neighb[iel]=iel+nelx
       if edge4_boundary_indicator[iel]==0: 
          edge4_neighb[iel]=iel-1
   #end for

   np.savetxt('edge1.ascii',np.array([edge1_xc,edge1_yc,edge1_nx/10,edge1_ny/10]).T)
   np.savetxt('edge2.ascii',np.array([edge2_xc,edge2_yc,edge2_nx/10,edge2_ny/10]).T)
   np.savetxt('edge3.ascii',np.array([edge3_xc,edge3_yc,edge3_nx/10,edge3_ny/10]).T)
   np.savetxt('edge4.ascii',np.array([edge4_xc,edge4_yc,edge4_nx/10,edge4_ny/10]).T)

#end if

#==============================================================================
#==============================================================================
#==============================================================================
# steady state loop iterations / sweeping through elements 
#==============================================================================
#==============================================================================
#==============================================================================

for iter in range(0,niter):

    ###############################################################################
    # loop over elements
    # forward then backward element sweep
    ###############################################################################

    if iter%2==0:
       start=0
       end=nel
       step=1
    else:
       start=nel-1
       end=-1
       step=-1

    for iel in range(start,end,step):
    #for iel in range(0,nel):
    
        #**********************************************************************

        if m==3:
           x1=xT[icon[0,iel]]
           x2=xT[icon[1,iel]]
           x3=xT[icon[2,iel]]
           y1=yT[icon[0,iel]]
           y2=yT[icon[1,iel]]
           y3=yT[icon[2,iel]]
           #volume terms (E,H,J)
           E=area[iel]/12*np.array([[2,1,1],[1,2,1],[1,1,2]],dtype=np.float64) 
           Jx= 1/6*np.array([[y2-y3,y2-y3,y2-y3],[y3-y1,y3-y1,y3-y1],[y1-y2,y1-y2,y1-y2]],dtype=np.float64) 
           Jy= 1/6*np.array([[x3-x2,x3-x2,x3-x2],[x1-x3,x1-x3,x1-x3],[x2-x1,x2-x1,x2-x1]],dtype=np.float64) 
           Hx=hcond*Jx
           Hy=hcond*Jy
   
           #precomputed C matrices
           C1=edge1_L[iel]/6*np.array([[2,1,0],[1,2,0],[0,0,0]],dtype=np.float64) 
           C2=edge2_L[iel]/6*np.array([[0,0,0],[0,2,1],[0,1,2]],dtype=np.float64) 
           C3=edge3_L[iel]/6*np.array([[2,0,1],[0,0,0],[1,0,2]],dtype=np.float64) 
     
           #edge matrices
           if edge1_boundary_indicator[iel]!=0: # eq 4.6e
              C12=-0.5
           else:
              C12=C12_inside
           edge1_Hx=-(0.5+C12)*edge1_nx[iel]*C1 # eq 4.14e
           edge1_Hy=-(0.5+C12)*edge1_ny[iel]*C1
           edge1_JBx=edge1_Hx # eq 4.14e
           edge1_JBy=edge1_Hy
           edge1_HBx=-(0.5-C12)*edge1_nx[iel]*C1 # eq 4.15e
           edge1_HBy=-(0.5-C12)*edge1_ny[iel]*C1
           edge1_Jx=edge1_HBx # eq 4.15e
           edge1_Jy=edge1_HBy
           edge1_GT=C11*C1 # eq 4.16e
           edge1_GTB=-C11*C1 # eq 4.16e
    
           if edge2_boundary_indicator[iel]!=0:
              C12=-0.5
           else:
              C12=C12_inside

           edge2_Hx=-(0.5+C12)*edge2_nx[iel]*C2
           edge2_Hy=-(0.5+C12)*edge2_ny[iel]*C2
           edge2_JBx=edge2_Hx
           edge2_JBy=edge2_Hy
           edge2_HBx=-(0.5-C12)*edge2_nx[iel]*C2
           edge2_HBy=-(0.5-C12)*edge2_ny[iel]*C2
           edge2_Jx=edge2_HBx
           edge2_Jy=edge2_HBy
           edge2_GT=C11*C2
           edge2_GTB=-C11*C2
    
           if edge3_boundary_indicator[iel]!=0:
              C12=-0.5
           else:
              C12=C12_inside
           edge3_Hx=-(0.5+C12)*edge3_nx[iel]*C3
           edge3_Hy=-(0.5+C12)*edge3_ny[iel]*C3
           edge3_JBx=edge3_Hx
           edge3_JBy=edge3_Hy
           edge3_HBx=-(0.5-C12)*edge3_nx[iel]*C3
           edge3_HBy=-(0.5-C12)*edge3_ny[iel]*C3
           edge3_Jx=edge3_HBx
           edge3_Jy=edge3_HBy
           edge3_GT=C11*C3
           edge3_GTB=-C11*C3
           #hcond ?!!

           #build A_{\partial\Omega_e}
           A_pOmegae=np.zeros((3*m,3*m),dtype=np.float64)
           A_pOmegae[0:m    ,2*m:3*m]=edge1_Hx[:,:]+ edge2_Hx[:,:]+ edge3_Hx[:,:] 
           A_pOmegae[m:2*m  ,2*m:3*m]=edge1_Hy[:,:]+ edge2_Hy[:,:]+ edge3_Hy[:,:] 
           A_pOmegae[2*m:3*m,0:m    ]=edge1_Jx[:,:]+ edge2_Jx[:,:]+ edge3_Jx[:,:] 
           A_pOmegae[2*m:3*m,m:2*m  ]=edge1_Jy[:,:]+ edge2_Jy[:,:]+ edge3_Jy[:,:] 
           A_pOmegae[2*m:3*m,2*m:3*m]=edge1_GT+edge2_GT+edge3_GT
           
           #edge1 contribution: computing qx_edge1,qy_edge1,T_edge1 vectors of length m
           if edge1_boundary_indicator[iel]!=0:
              T_edge1=[Tbc(xT[icon[0,iel]],yT[icon[0,iel]]),Tbc(xT[icon[1,iel]],yT[icon[1,iel]]),0]
              qx_edge1=[0,0,0] # irrelevant if C22=0!
              qy_edge1=[0,0,0]
           else:
               #edge 1 is nodes 0 -> 1
               jel=edge1_neighb[iel] # identity of neighbour 
               ed=edge1_neighbedge[iel] 
               if ed==1:
                  nodeA=icon[0,jel]
                  nodeB=icon[1,jel]
               if ed==2:
                  nodeA=icon[1,jel]
                  nodeB=icon[2,jel]
               if ed==3:
                  nodeA=icon[2,jel]
                  nodeB=icon[0,jel]
               T_edge1 =[ T[nodeB], T[nodeA],0]
               qx_edge1=[qx[nodeB],qx[nodeA],0]
               qy_edge1=[qy[nodeB],qy[nodeA],0]
        
           #edge2 contribution: computing qx_edge2,qy_edge2,T_edge2 vectors of length m
           if edge2_boundary_indicator[iel]!=0:
              T_edge2=[0,Tbc(xT[icon[1,iel]],yT[icon[1,iel]]),Tbc(xT[icon[2,iel]],yT[icon[2,iel]])]
              qx_edge2=[0,0,0]
              qy_edge2=[0,0,0]
           else:
               #edge 2 is nodes 1 -> 2
               jel=edge2_neighb[iel] # identity of neighbour 
               ed=edge2_neighbedge[iel] 
               if ed==1:
                  nodeA=icon[0,jel]
                  nodeB=icon[1,jel]
               if ed==2:
                  nodeA=icon[1,jel]
                  nodeB=icon[2,jel]
               if ed==3:
                  nodeA=icon[2,jel]
                  nodeB=icon[0,jel]
               T_edge2 =[ 0,  T[nodeB], T[nodeA] ]
               qx_edge2=[ 0, qx[nodeB],qx[nodeA] ]
               qy_edge2=[ 0, qy[nodeB],qy[nodeA] ]
        
           #edge3 contribution: computing qx_edge3,qy_edge3,T_edge3 vectors of length m
           if edge3_boundary_indicator[iel]!=0:
              T_edge3=[Tbc(xT[icon[0,iel]],yT[icon[0,iel]]),0,Tbc(xT[icon[2,iel]],yT[icon[2,iel]])]
              qx_edge3=[0,0,0]
              qy_edge3=[0,0,0]
           else:
               #edge 3 is nodes 2 -> 0
               jel=edge3_neighb[iel] # identity of neighbour 
               ed=edge3_neighbedge[iel] 
               if ed==1:
                  nodeA=icon[0,jel]
                  nodeB=icon[1,jel]
               if ed==2:
                  nodeA=icon[1,jel]
                  nodeB=icon[2,jel]
               if ed==3:
                  nodeA=icon[2,jel]
                  nodeB=icon[0,jel]
               T_edge3 =[ T[nodeA],0, T[nodeB] ]
               qx_edge3=[qx[nodeA],0,qx[nodeB] ]
               qy_edge3=[qy[nodeA],0,qy[nodeB] ]

           bel=np.zeros(3*m,dtype=np.float64)
           bel[0  :m  ]=edge1_HBx.dot(T_edge1)+edge2_HBx.dot(T_edge2)+edge3_HBx.dot(T_edge3)
           bel[m  :2*m]=edge1_HBy.dot(T_edge1)+edge2_HBy.dot(T_edge2)+edge3_HBy.dot(T_edge3)
           bel[2*m:3*m]=edge1_JBx.dot(qx_edge1)+edge1_JBy.dot(qy_edge1)+edge1_GTB.dot(T_edge1)+\
                        edge2_JBx.dot(qx_edge2)+edge2_JBy.dot(qy_edge2)+edge2_GTB.dot(T_edge2)+\
                        edge3_JBx.dot(qx_edge3)+edge3_JBy.dot(qy_edge3)+edge3_GTB.dot(T_edge3)

        #end if m=3    

        #**********************************************************************

        if m==4:
           #volume terms (E,H,J)
           E=hx*hy/9*np.array([[1,0.5,0.25,0.5],[0.5,1,0.5,0.25],[0.25,0.5,1,0.5],[0.5,0.25,0.5,1]])
           Jx=hy/12*np.array([[-2,-2,-1,-1],[2,2,1,1],[1,1,2,2],[-1,-1,-2,-2]])
           Jy=hx/12*np.array([[-2,-1,-1,-2],[-1,-2,-2,-1],[1,2,2,1],[2,1,1,2]])
           Hx=hcond*Jx
           Hy=hcond*Jy
           #precomputed C matrices
           C1=hx/6*np.array([[2,1,0,0],[1,2,0,0],[0,0,0,0],[0,0,0,0]])
           C2=hy/6*np.array([[0,0,0,0],[0,2,1,0],[0,1,2,0],[0,0,0,0]])
           C3=hx/6*np.array([[0,0,0,0],[0,0,0,0],[0,0,2,1],[0,0,1,2]])
           C4=hy/6*np.array([[2,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,2]])
           
           if edge1_boundary_indicator[iel]!=0: # eq 4.6e
              C12=-0.5
           else:
              C12=C12_inside
           edge1_Hx=-(0.5+C12)*edge1_nx[iel]*C1 # eq 4.14e
           edge1_Hy=-(0.5+C12)*edge1_ny[iel]*C1
           edge1_JBx=edge1_Hx # eq 4.14e
           edge1_JBy=edge1_Hy
           edge1_HBx=-(0.5-C12)*edge1_nx[iel]*C1 # eq 4.15e
           edge1_HBy=-(0.5-C12)*edge1_ny[iel]*C1
           edge1_Jx=edge1_HBx # eq 4.15e
           edge1_Jy=edge1_HBy
           edge1_GT=C11*C1 # eq 4.16e
           edge1_GTB=-C11*C1 # eq 4.16e
           
           if edge2_boundary_indicator[iel]!=0:
              C12=-0.5
           else:
              C12=C12_inside
           edge2_Hx=-(0.5+C12)*edge2_nx[iel]*C2
           edge2_Hy=-(0.5+C12)*edge2_ny[iel]*C2
           edge2_JBx=edge2_Hx
           edge2_JBy=edge2_Hy
           edge2_HBx=-(0.5-C12)*edge2_nx[iel]*C2
           edge2_HBy=-(0.5-C12)*edge2_ny[iel]*C2
           edge2_Jx=edge2_HBx
           edge2_Jy=edge2_HBy
           edge2_GT=C11*C2
           edge2_GTB=-C11*C2

           if edge3_boundary_indicator[iel]!=0:
              C12=-0.5
           else:
              C12=C12_inside
           edge3_Hx=-(0.5+C12)*edge3_nx[iel]*C3
           edge3_Hy=-(0.5+C12)*edge3_ny[iel]*C3
           edge3_JBx=edge3_Hx
           edge3_JBy=edge3_Hy
           edge3_HBx=-(0.5-C12)*edge3_nx[iel]*C3
           edge3_HBy=-(0.5-C12)*edge3_ny[iel]*C3
           edge3_Jx=edge3_HBx
           edge3_Jy=edge3_HBy
           edge3_GT=C11*C3
           edge3_GTB=-C11*C3     
           
           if edge4_boundary_indicator[iel]!=0:
              C12=-0.5
           else:
              C12=C12_inside
           edge4_Hx=-(0.5+C12)*edge4_nx[iel]*C4
           edge4_Hy=-(0.5+C12)*edge4_ny[iel]*C4
           edge4_JBx=edge4_Hx
           edge4_JBy=edge4_Hy
           edge4_HBx=-(0.5-C12)*edge4_nx[iel]*C4
           edge4_HBy=-(0.5-C12)*edge4_ny[iel]*C4
           edge4_Jx=edge4_HBx
           edge4_Jy=edge4_HBy
           edge4_GT=C11*C4
           edge4_GTB=-C11*C4

           A_pOmegae=np.zeros((3*m,3*m),dtype=np.float64)
           A_pOmegae[0:m    ,2*m:3*m]=edge1_Hx[:,:]+ edge2_Hx[:,:]+ edge3_Hx[:,:] + edge4_Hx[:,:] 
           A_pOmegae[m:2*m  ,2*m:3*m]=edge1_Hy[:,:]+ edge2_Hy[:,:]+ edge3_Hy[:,:] + edge4_Hy[:,:] 
           A_pOmegae[2*m:3*m,0:m    ]=edge1_Jx[:,:]+ edge2_Jx[:,:]+ edge3_Jx[:,:] + edge4_Jx[:,:] 
           A_pOmegae[2*m:3*m,m:2*m  ]=edge1_Jy[:,:]+ edge2_Jy[:,:]+ edge3_Jy[:,:] + edge4_Jy[:,:] 
           A_pOmegae[2*m:3*m,2*m:3*m]=edge1_GT+edge2_GT+edge3_GT+edge4_GT
            
           if edge1_boundary_indicator[iel]!=0: # must be on bottom boundary
              T_edge1=[Tbc(xT[icon[0,iel]],yT[icon[0,iel]]),Tbc(xT[icon[1,iel]],yT[icon[1,iel]]),0,0]
              qx_edge1=[0,0,0,0]
              qy_edge1=[0,0,0,0]
           else:
              #edge 1 is nodes 0 -> 1 | neighbour jel shares his edge 3 with iel
              jel=edge1_neighb[iel] # identity of neighbour 
              nodeA=icon[2,jel]
              nodeB=icon[3,jel]
              T_edge1 =[ T[nodeB], T[nodeA],0,0]
              qx_edge1=[qx[nodeB],qx[nodeA],0,0]
              qy_edge1=[qy[nodeB],qy[nodeA],0,0]
        
           if edge2_boundary_indicator[iel]!=0: # must be on right boundary
              T_edge2=[0,Tbc(xT[icon[1,iel]],yT[icon[1,iel]]),Tbc(xT[icon[2,iel]],yT[icon[2,iel]]),0]
              qx_edge2=[0,0,0,0]
              qy_edge2=[0,0,0,0]
           else:
              #edge 2 is nodes 1 -> 2 | neighbour jel shares his edge 4 with iel
              jel=edge2_neighb[iel] # identity of neighbour 
              nodeA=icon[3,jel]
              nodeB=icon[0,jel]
              T_edge2 =[ 0,  T[nodeB], T[nodeA],0 ]
              qx_edge2=[ 0, qx[nodeB],qx[nodeA],0 ]
              qy_edge2=[ 0, qy[nodeB],qy[nodeA],0 ]
        
           if edge3_boundary_indicator[iel]!=0: # must be on top boundary
              T_edge3=[0,0,Tbc(xT[icon[2,iel]],yT[icon[2,iel]]),Tbc(xT[icon[3,iel]],yT[icon[3,iel]])]
              qx_edge3=[0,0,0,0]
              qy_edge3=[0,0,0,0]
           else:
              #edge 3 is nodes 2 -> 3 | neighbour jel shares his edge 1 with iel
               jel=edge3_neighb[iel] # identity of neighbour 
               nodeA=icon[0,jel]
               nodeB=icon[1,jel]
               T_edge3 =[0,0, T[nodeB], T[nodeA] ]
               qx_edge3=[0,0,qx[nodeB],qx[nodeA] ]
               qy_edge3=[0,0,qy[nodeB],qy[nodeA] ]
               
           if edge4_boundary_indicator[iel]!=0: # must be on left boundary 
              T_edge4=[Tbc(xT[icon[0,iel]],yT[icon[0,iel]]),0,0,Tbc(xT[icon[3,iel]],yT[icon[3,iel]])]
              qx_edge4=[0,0,0,0]
              qy_edge4=[0,0,0,0]
           else:
              #edge 4 is nodes 2 -> 0 | neighbour jel shares his edge 2 with iel
              jel=edge4_neighb[iel] # identity of neighbour 
              nodeA=icon[1,jel]
              nodeB=icon[2,jel]
              T_edge4 =[ T[nodeA],0,0, T[nodeB] ]
              qx_edge4=[qx[nodeA],0,0,qx[nodeB] ]
              qy_edge4=[qy[nodeA],0,0,qy[nodeB] ]

           bel=np.zeros(3*m,dtype=np.float64)
           bel[0  :m  ]=edge1_HBx.dot(T_edge1)+edge2_HBx.dot(T_edge2)+edge3_HBx.dot(T_edge3)+edge4_HBx.dot(T_edge4)
           bel[m  :2*m]=edge1_HBy.dot(T_edge1)+edge2_HBy.dot(T_edge2)+edge3_HBy.dot(T_edge3)+edge4_HBy.dot(T_edge4)
           bel[2*m:3*m]=edge1_JBx.dot(qx_edge1)+edge1_JBy.dot(qy_edge1)+edge1_GTB.dot(T_edge1)+\
                        edge2_JBx.dot(qx_edge2)+edge2_JBy.dot(qy_edge2)+edge2_GTB.dot(T_edge2)+\
                        edge3_JBx.dot(qx_edge3)+edge3_JBy.dot(qy_edge3)+edge3_GTB.dot(T_edge3)+\
                        edge4_JBx.dot(qx_edge4)+edge4_JBy.dot(qy_edge4)+edge4_GTB.dot(T_edge4)

        #end if m=4

        #**********************************************************************

        #build A_{\Omega_e} (independent of m)
        A_Omegae=np.zeros((3*m,3*m),dtype=np.float64)
        A_Omegae[0:m    ,0:m    ]=E[:,:] 
        A_Omegae[m:2*m  ,m:2*m  ]=E[:,:] 
        A_Omegae[0:m    ,2*m:3*m]=Hx[:,:] 
        A_Omegae[m:2*m  ,2*m:3*m]=Hy[:,:] 
        A_Omegae[2*m:3*m,0:m    ]=Jx[:,:] 
        A_Omegae[2*m:3*m,m:2*m  ]=Jy[:,:] 

        #temporary residual debug
        #constant temp
        #T_el =[1,1,1,1]
        #qx_el =[0,0,0,0]
        #qy_el =[0,0,0,0]
        #linear x temp
        #T_el =[xT[icon[0,iel]],xT[icon[1,iel]],xT[icon[2,iel]],xT[icon[3,iel]]]
        #qx_el =[1,1,1,1]
        #qy_el =[0,0,0,0]
        #linear y temp
        #T_el =[yT[icon[0,iel]],yT[icon[1,iel]],yT[icon[2,iel]],yT[icon[3,iel]]]
        #qx_el =[0,0,0,0]
        #qy_el =[1,1,1,1]

        T_el  =T[icon[0:m,iel]]
        qx_el =qx[icon[0:m,iel]]
        qy_el =qy[icon[0:m,iel]]

        #compute residual qx
        if m==3:    
           res_qx=E.dot(qx_el)\
                 +Hx.dot(T_el)+edge1_Hx.dot(T_el)+edge2_Hx.dot(T_el)+edge3_Hx.dot(T_el)\
                 +edge1_HBx.dot(T_edge1)+edge2_HBx.dot(T_edge2)+edge3_HBx.dot(T_edge3) 
        if m==4:    
           res_qx=E.dot(qx_el)\
                 +Hx.dot(T_el)+edge1_Hx.dot(T_el)+edge2_Hx.dot(T_el)+edge3_Hx.dot(T_el)+edge4_Hx.dot(T_el)\
                 +edge1_HBx.dot(T_edge1)+edge2_HBx.dot(T_edge2)+edge3_HBx.dot(T_edge3) +edge4_HBx.dot(T_edge4) 
        if debug:
           print('residual qx->',res_qx)
    
        #compute residual qy
        if m==3:    
           res_qy=E.dot(qy_el)\
                 +Hy.dot(T_el)+edge1_Hy.dot(T_el)+edge2_Hy.dot(T_el)+edge3_Hy.dot(T_el)\
                 +edge1_HBy.dot(T_edge1)+edge2_HBy.dot(T_edge2)+edge3_HBy.dot(T_edge3) 
        if m==4:    
           res_qy=E.dot(qy_el)\
                 +Hy.dot(T_el)+edge1_Hy.dot(T_el)+edge2_Hy.dot(T_el)+edge3_Hy.dot(T_el)+edge4_Hy.dot(T_el)\
                 +edge1_HBy.dot(T_edge1)+edge2_HBy.dot(T_edge2)+edge3_HBy.dot(T_edge3) +edge4_HBy.dot(T_edge4) 
        if debug:
           print('residual qy->',res_qy)

        #compute residual T
        if m==3:    
           res_T=Jx.dot(qx_el)+Jy.dot(qy_el)\
                +edge1_Jx.dot(qx_el)+edge2_Jx.dot(qx_el)+edge3_Jx.dot(qx_el)\
                +edge1_Jy.dot(qy_el)+edge2_Jy.dot(qy_el)+edge3_Jy.dot(qy_el)\
                +edge1_GT.dot(T_el)+edge2_GT.dot(T_el)+edge3_GT.dot(T_el)\
                +edge1_JBx.dot(qx_edge1)+edge2_JBx.dot(qx_edge2)+edge3_JBx.dot(qx_edge3)\
                +edge1_JBy.dot(qy_edge1)+edge2_JBy.dot(qy_edge2)+edge3_JBy.dot(qy_edge3)\
                +edge1_GTB.dot(T_edge1)+edge2_GTB.dot(T_edge2)+edge3_GTB.dot(T_edge3)
        if m==4:    
           res_T=Jx.dot(qx_el)+Jy.dot(qy_el)\
                +edge1_Jx.dot(qx_el)+edge2_Jx.dot(qx_el)+edge3_Jx.dot(qx_el)+edge4_Jx.dot(qx_el)\
                +edge1_Jy.dot(qy_el)+edge2_Jy.dot(qy_el)+edge3_Jy.dot(qy_el)+edge4_Jy.dot(qy_el)\
                +edge1_GT.dot(T_el)+edge2_GT.dot(T_el)+edge3_GT.dot(T_el)+edge4_GT.dot(T_el)\
                +edge1_JBx.dot(qx_edge1)+edge2_JBx.dot(qx_edge2)+edge3_JBx.dot(qx_edge3)+edge4_JBx.dot(qx_edge4)\
                +edge1_JBy.dot(qy_edge1)+edge2_JBy.dot(qy_edge2)+edge3_JBy.dot(qy_edge3)+edge4_JBy.dot(qy_edge4)\
                +edge1_GTB.dot(T_edge1)+edge2_GTB.dot(T_edge2)+edge3_GTB.dot(T_edge3)+edge4_GTB.dot(T_edge4)
        if debug:
           print('residual T->',res_T)

        # solve system for element iel    
    
        sol = sps.linalg.spsolve(sps.csr_matrix(A_Omegae+A_pOmegae),-bel)
        if debug:
           print(iel,'qx=',sol[0:m])
           print(iel,'qy=',sol[m:2*m])
           print(iel,'T =',sol[2*m:3*m],yT[icon[:,iel]])

        #export elemental quantities (solution & residual) to global arrays
        qx[icon[0:m,iel]]=sol[0:m]
        qy[icon[0:m,iel]]=sol[m+0:m+m]
        T[icon[0:m,iel]]=sol[2*m+0:2*m+m]
        residual_qx[icon[0:m,iel]]=res_qx[0:m]
        residual_qy[icon[0:m,iel]]=res_qy[0:m]
        residual_T [icon[0:m,iel]]=res_T [0:m]

    #end for iel

    print("iter= %3d : T,qx,qy (m/M)= %.3e %.3e | %.3e %.3e | %.3e %.3e | max(resT)= %.3e (tol= %.1e)"\
             %(iter,min(T),max(T),min(qx),max(qx),min(qy),max(qy),np.max(np.abs(residual_T)),tol))

    T_stats_file.write("%d %e %e \n" %(iter,np.min(T),np.max(T))) ; T_stats_file.flush()
    qx_stats_file.write("%d %e %e \n" %(iter,np.min(qx),np.max(qy))) ; qx_stats_file.flush()
    qy_stats_file.write("%d %e %e \n" %(iter,np.min(qx),np.max(qy))) ; qy_stats_file.flush()
    residual_T_stats_file.write("%d %e %e \n" %(iter,np.min(residual_T),np.max(residual_T))) ; residual_T_stats_file.flush()
    residual_qx_stats_file.write("%d %e %e \n" %(iter,np.min(residual_qx),np.max(residual_qy))) ; residual_qx_stats_file.flush()
    residual_qy_stats_file.write("%d %e %e \n" %(iter,np.min(residual_qx),np.max(residual_qy))) ; residual_qy_stats_file.flush()

    convergence = np.max(np.abs(residual_T))<tol and np.max(np.abs(residual_qx))<tol and np.max(np.abs(residual_qy))<tol

    ###############################################################################
    # plot of solution
    ###############################################################################

    if visualise_all or convergence:

       if visualise_all:
          filename = 'solution_{:04d}.vtu'.format(iter) 
       else:
          filename = 'solution.vtu'.format(iter) 

       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NT,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %(xT[i],yT[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
 
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
       for iel in range(0,nel):
               vtufile.write("%10e \n" %(area[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='q' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e %10e %10e \n" %(qx[i],qy[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32'  Name='T' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e  \n" %(T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32'  Name='qx' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e  \n" %(qx[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32'  Name='qy' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e  \n" %(qy[i]))
       vtufile.write("</DataArray>\n")
 
       vtufile.write("<DataArray type='Float32'  Name='T (res)' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e  \n" %(residual_T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32'  Name='qx (res)' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e  \n" %(residual_qx[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32'  Name='qy (res)' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10e  \n" %(residual_qy[i]))
       vtufile.write("</DataArray>\n")
 
       vtufile.write("</PointData>\n")

       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if m==3:
          for iel in range (0,nel):
              vtufile.write("%d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel]))
       if m==4:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       if m==3:
          for iel in range (0,nel):
              vtufile.write("%d \n" %5) 
       if m==4:
          for iel in range (0,nel):
              vtufile.write("%d \n" %9) 
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

    if convergence:
       break

#end for iter

np.savetxt('T.ascii',np.array([xT,yT,T]).T,header='# x,y,T')
np.savetxt('qx.ascii',np.array([xT,yT,qx]).T,header='# x,y,qx')
np.savetxt('qy.ascii',np.array([xT,yT,qy]).T,header='# x,y,qy')


exit()
###################################################################3
#compute errors
###################################################################3

N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
errT=0.
errqx=0.
errqy=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/np.sqrt(3)
            sq=jq/np.sqrt(3)
            weightq=1.*1.
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*xT[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*yT[icon[k,iel]]
                jcb[1,0]+=dNds[k]*xT[icon[k,iel]]
                jcb[1,1]+=dNds[k]*yT[icon[k,iel]]
            jcob=np.linalg.det(jcb)
            xq=0.0
            yq=0.0
            Tq=0.0
            qxq=0.0
            qyq=0.0
            for k in range(0,m):
                xq+=N[k]*xT[icon[k,iel]]
                yq+=N[k]*yT[icon[k,iel]]
                Tq+=N[k]*T[icon[k,iel]]
                qxq+=N[k]*qx[icon[k,iel]]
                qyq+=N[k]*qy[icon[k,iel]]
            errT+=(Tq-T_analytical(xq,yq))**2*weightq*jcob
            errqx+=(qxq-qx_analytical(xq,yq))**2*weightq*jcob
            errqy+=(qyq-qy_analytical(xq,yq))**2*weightq*jcob

errT=np.sqrt(errT)
errqx=np.sqrt(errqx)
errqy=np.sqrt(errqy)

print("nel= %6d ; errT= %.8e ; errqx,errqy= %.8e %.8e" %(nel,errT,errqx,errqy))



print("-----------------------------")
print("------------the end----------")
print("-----------------------------")


