import numpy as np
import random 
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix

###############################################################################

Lx=1.
Ly=1.

nelx=4
nely=3

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


C12=0.5  #????
C11=50   #????

print('m=',m)
print('nel=',nel)
print('NT=',NT)

visu=True


#########################
#physical pb 

Tleft =1
Ttop =1
Tbottom =1
Tright=1
Tinit =0
hcond=1

###############################################################################
# fill icon array
###############################################################################
icon =np.zeros((m,nel),dtype=np.int32)

if m==3:
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           icon[0, counter] = m*counter+0
           icon[1, counter] = m*counter+1
           icon[2, counter] = m*counter+2
           print(counter,icon[:,counter]+1)
           counter += 1
           icon[0, counter] = m*counter+0
           icon[1, counter] = m*counter+1
           icon[2, counter] = m*counter+2
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
   #print(icon)

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
           print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           xT[counter]=i*hx+hx
           yT[counter]=j*hy
           print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           xT[counter]=i*hx
           yT[counter]=j*hy+hy
           print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           #-----------------
           xT[counter]=i*hx+hx
           yT[counter]=j*hy
           print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           xT[counter]=i*hx+hx
           yT[counter]=j*hy+hy
           print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
           xT[counter]=i*hx
           yT[counter]=j*hy+hy
           print('node=',counter+1,'|',xT[counter],yT[counter])
           counter+=1
       #end for
   #end for
#print (xT,yT)
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
   print(xT,yT)

np.savetxt('grid.ascii',np.array([xT,yT]).T,header='# x,y')

###############################################################################
# filling T,qx,qy arrays with random values
###############################################################################

T = np.empty(NT,dtype=np.float64)  
qx = np.empty(NQ,dtype=np.float64) 
qy = np.empty(NQ,dtype=np.float64) 
area = np.empty(nel,dtype=np.float64) 

T[:]=Tinit

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
           edge2_nx[iel]=np.sqrt(2)
           edge2_ny[iel]=np.sqrt(2)
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
           print('iel',iel,'has neighbours',edge1_neighb[iel],edge2_neighb[iel],edge3_neighb[iel])
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
           edge3_nx[iel]=-np.sqrt(2)
           edge3_ny[iel]=-np.sqrt(2)
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
           print('iel',iel,'has neighbours',edge1_neighb[iel],edge2_neighb[iel],edge3_neighb[iel])

           iel+=1
       #end for
   #end for

   np.savetxt('edge1.ascii',np.array([edge1_xc,edge1_yc,edge1_nx/10,edge1_ny/10,edge1_boundary_indicator]).T)
   np.savetxt('edge2.ascii',np.array([edge2_xc,edge2_yc,edge2_nx/10,edge2_ny/10,edge2_boundary_indicator]).T)
   np.savetxt('edge3.ascii',np.array([edge3_xc,edge3_yc,edge3_nx/10,edge3_ny/10,edge3_boundary_indicator]).T)

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
             print ('elt',iel,'looking at neighbour',jel,'through edge1')
             print('corresponding edge of neighbour is edge',edge1_neighbedge[iel])
          # case2: is it edge 2 of neighbour element?
          if abs(x0-xT[icon[2,jel]])<1e-6 and abs(y0-yT[icon[2,jel]])<1e-6 and\
             abs(x1-xT[icon[1,jel]])<1e-6 and abs(y1-yT[icon[1,jel]])<1e-6:
             edge1_neighbedge[iel] = 2 
             print('elt',iel,'looking at neighbour',jel,'through edge1')
             print('corresponding edge of neighbour is edge',edge1_neighbedge[iel])
          # case3: is it edge 3 of neighbour element?
          if abs(x0-xT[icon[0,jel]])<1e-6 and abs(y0-yT[icon[0,jel]])<1e-6 and\
             abs(x1-xT[icon[2,jel]])<1e-6 and abs(y1-yT[icon[2,jel]])<1e-6:
             edge1_neighbedge[iel] = 3 
             print ('elt',iel,'looking at neighbour',jel,'through edge1')
             print('corresponding edge of neighbour is edge',edge1_neighbedge[iel])
        # end if

       #--------------------edge 2 (nodes 1&2)-------------------------
       if edge2_neighb[iel]!=-1: # if there is element on other side
          jel=edge2_neighb[iel] # identity of neighbour 
          # case1: is it edge 1 of neighbour element?
          if abs(x1-xT[icon[1,jel]])<1e-6 and abs(y1-yT[icon[1,jel]])<1e-6 and\
             abs(x2-xT[icon[0,jel]])<1e-6 and abs(y2-yT[icon[0,jel]])<1e-6:
             edge2_neighbedge[iel] = 1 
             print ('elt',iel,'looking at neighbour',jel,'through edge2')
             print('corresponding edge of neighbour is edge',edge2_neighbedge[iel])
          # case2: is it edge 2 of neighbour element?
          if abs(x1-xT[icon[2,jel]])<1e-6 and abs(y1-yT[icon[2,jel]])<1e-6 and\
             abs(x2-xT[icon[1,jel]])<1e-6 and abs(y2-yT[icon[1,jel]])<1e-6:
             edge2_neighbedge[iel] = 2 
             print('elt',iel,'looking at neighbour',jel,'through edge2')
             print('corresponding edge of neighbour is edge',edge2_neighbedge[iel])
          # case3: is it edge 3 of neighbour element?
          if abs(x1-xT[icon[0,jel]])<1e-6 and abs(y1-yT[icon[0,jel]])<1e-6 and\
             abs(x2-xT[icon[2,jel]])<1e-6 and abs(y2-yT[icon[2,jel]])<1e-6:
             edge2_neighbedge[iel] = 3 
             print ('elt',iel,'looking at neighbour',jel,'through edge2')
             print('corresponding edge of neighbour is edge',edge2_neighbedge[iel])
        # end if

       #--------------------edge 3 (nodes 2&0)-------------------------
       if edge3_neighb[iel]!=-1: # if there is element on other side
          jel=edge3_neighb[iel] # identity of neighbour 
          # case1: is it edge 1 of neighbour element?
          if abs(x2-xT[icon[1,jel]])<1e-6 and abs(y2-yT[icon[1,jel]])<1e-6 and\
             abs(x0-xT[icon[0,jel]])<1e-6 and abs(y0-yT[icon[0,jel]])<1e-6:
             edge3_neighbedge[iel] = 1 
             print ('elt',iel,'looking at neighbour',jel,'through edge3')
             print('corresponding edge of neighbour is edge',edge3_neighbedge[iel])
          # case2: is it edge 2 of neighbour element?
          if abs(x2-xT[icon[2,jel]])<1e-6 and abs(y2-yT[icon[2,jel]])<1e-6 and\
             abs(x0-xT[icon[1,jel]])<1e-6 and abs(y0-yT[icon[1,jel]])<1e-6:
             edge3_neighbedge[iel] = 2 
             print('elt',iel,'looking at neighbour',jel,'through edge3')
             print('corresponding edge of neighbour is edge',edge3_neighbedge[iel])
          # case3: is it edge 3 of neighbour element?
          if abs(x2-xT[icon[0,jel]])<1e-6 and abs(y2-yT[icon[0,jel]])<1e-6 and\
             abs(x0-xT[icon[2,jel]])<1e-6 and abs(y0-yT[icon[2,jel]])<1e-6:
             edge3_neighbedge[iel] = 3 
             print ('elt',iel,'looking at neighbour',jel,'through edge3')
             print('corresponding edge of neighbour is edge',edge3_neighbedge[iel])
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
       area[iel]=hx*hy
   #end for

   np.savetxt('edge1.ascii',np.array([edge1_xc,edge1_yc,edge1_nx/10,edge1_ny/10]).T)
   np.savetxt('edge2.ascii',np.array([edge2_xc,edge2_yc,edge2_nx/10,edge2_ny/10]).T)
   np.savetxt('edge3.ascii',np.array([edge3_xc,edge3_yc,edge3_nx/10,edge3_ny/10]).T)
   np.savetxt('edge4.ascii',np.array([edge4_xc,edge4_yc,edge4_nx/10,edge4_ny/10]).T)

   #TODO fill neighbour arrays and edges ...

#end if

###############################################################################

for iel in range(0,nel):





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
       edge1_Hx=-(0.5+C12)*edge1_nx[iel]*C1 # eq 4.14e
       edge1_Hy=-(0.5+C12)*edge1_ny[iel]*C1
       edge2_Hx=-(0.5+C12)*edge2_nx[iel]*C2
       edge2_Hy=-(0.5+C12)*edge2_ny[iel]*C2
       edge3_Hx=-(0.5+C12)*edge3_nx[iel]*C3
       edge3_Hy=-(0.5+C12)*edge3_ny[iel]*C3

       edge1_JBx=edge1_Hx # eq 4.14e
       edge1_JBy=edge1_Hy
       edge2_JBx=edge2_Hx
       edge2_JBy=edge2_Hy
       edge3_JBx=edge3_Hx
       edge3_JBy=edge3_Hy

       edge1_HBx=-(0.5-C12)*edge1_nx[iel]*C1 # eq 4.15e
       edge1_HBy=-(0.5-C12)*edge1_ny[iel]*C1
       edge2_HBx=-(0.5-C12)*edge2_nx[iel]*C2
       edge2_HBy=-(0.5-C12)*edge2_ny[iel]*C2
       edge3_HBx=-(0.5-C12)*edge3_nx[iel]*C3
       edge3_HBy=-(0.5-C12)*edge3_ny[iel]*C3

       edge1_Jx=edge1_HBx # eq 4.15e
       edge1_Jy=edge1_HBy
       edge2_Jx=edge2_HBx
       edge2_Jy=edge2_HBy
       edge3_Jx=edge3_HBx
       edge3_Jy=edge3_HBy

       edge1_GT=C11*C1 # eq 4.16e
       edge2_GT=C11*C2
       edge3_GT=C11*C3

       edge1_GTB=-C11*C1 # eq 4.16e
       edge2_GTB=-C11*C2
       edge3_GTB=-C11*C3

    #end if m

    if m==4:
       x1=xT[icon[0,iel]]
       x2=xT[icon[1,iel]]
       x3=xT[icon[2,iel]]
       x4=yT[icon[3,iel]]
       y1=yT[icon[0,iel]]
       y2=yT[icon[1,iel]]
       y3=yT[icon[2,iel]]
       y4=yT[icon[3,iel]]
       #volume terms (E,H,J)
       E=hx*hy/9*np.array([[1,0.5,0.25,0.5],[0.5,1,0.5,0.25],[0.25,0.5,1,0.5],[0.5,0.25,0.5,1]])
       Jx=hy/12*np.array([[-2,2,1,-1],[-2,2,1,-1],[-1,1,2,-2],[-1,1,2,-2]])
       Jy=hx/12*np.array([[-2,-1,1,2],[-1,-2,2,1],[-1,-2,2,1],[-2,-1,1,2]])
       Hx=hcond*Jx
       Hy=hcond*Jy

       #precomputed C matrices
       C1=hx/6*np.array([[2,1,0,0],[1,2,0,0],[0,0,0,0],[0,0,0,0]])
       C2=hy/6*np.array([[0,0,0,0],[0,2,1,0],[0,1,2,0],[0,0,0,0]])
       C3=hx/6*np.array([[0,0,0,0],[0,0,0,0],[0,0,2,1],[0,0,1,2]])
       C4=hy/6*np.array([[2,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,2]])

    #end if m

    #build A_{\Omega_e}
    A_Omegae=np.zeros((3*m,3*m),dtype=np.float64)
    A_Omegae[0:m    ,0:m    ]=E[:,:] 
    A_Omegae[m:2*m  ,m:2*m  ]=E[:,:] 
    A_Omegae[0:m    ,2*m:3*m]=Hx[:,:] 
    A_Omegae[m:2*m  ,2*m:3*m]=Hy[:,:] 
    A_Omegae[2*m:3*m,0:m    ]=Jx[:,:] 
    A_Omegae[2*m:3*m,m:2*m  ]=Jy[:,:] 

    #build A_{\partial\Omega_e}
    A_pOmegae=np.zeros((3*m,3*m),dtype=np.float64)
    A_pOmegae[0:m    ,2*m:3*m]=edge1_Hx[:,:]+ edge2_Hx[:,:]+ edge3_Hx[:,:] 
    A_pOmegae[m:2*m  ,2*m:3*m]=edge1_Hy[:,:]+ edge2_Hy[:,:]+ edge3_Hy[:,:] 
    A_pOmegae[2*m:3*m,0:m    ]=edge1_Jx[:,:]+ edge2_Jx[:,:]+ edge3_Jx[:,:] 
    A_pOmegae[2*m:3*m,m:2*m  ]=edge1_Jy[:,:]+ edge2_Jy[:,:]+ edge3_Jy[:,:] 
    A_pOmegae[2*m:3*m,2*m:3*m]=edge1_GT+edge2_GT+edge3_GT

    #edge1 contribution: computing qx_edge1,qy_edge1,T_edge1 vectors of length m
    if edge1_boundary_indicator[iel]!=0:
       if edge1_boundary_indicator[iel]==1:  
          T_edge1=[Tleft,Tleft,Tleft]
          qx_edge1=[0,0,0]
          qy_edge1=[0,0,0]
       if edge1_boundary_indicator[iel]==2:  
          T_edge1=[Tright,Tright,Tright]
          qx_edge1=[0,0,0]
          qy_edge1=[0,0,0]
       if edge1_boundary_indicator[iel]==3:  
          T_edge1=[Tbottom,Tbottom,Tbottom]
          qx_edge1=[0,0,0]
          qy_edge1=[0,0,0]
       if edge1_boundary_indicator[iel]==4:  
          T_edge1=[Ttop,Ttop,Ttop]
          qx_edge1=[0,0,0]
          qy_edge1=[0,0,0]
    else:
       #edge 1 is nodes 0 -> 1
       jel=edge1_neighb[iel] # identity of neighbour 
       ed=edge1_neighbedge[iel] 
       if ed==1:
          nodeA=icon[0,jel]
          nodeB=icon[1,jel]
          nodeC=icon[2,jel] # not shared on edge
       if ed==2:
          nodeA=icon[1,jel]
          nodeB=icon[2,jel]
          nodeC=icon[0,jel] # not shared on edge
       if ed==3:
          nodeA=icon[2,jel]
          nodeB=icon[0,jel]
          nodeC=icon[1,jel] # not shared on edge
       T_edge1 =[ T[nodeA], T[nodeB], T[nodeC] ]
       qx_edge1=[qx[nodeA],qx[nodeB],qx[nodeC] ]
       qy_edge1=[qx[nodeA],qy[nodeB],qy[nodeC] ]


    #edge2 contribution: computing qx_edge2,qy_edge2,T_edge2 vectors of length m
    if edge2_boundary_indicator[iel]!=0:
       if edge2_boundary_indicator[iel]==1:  
          T_edge2=[Tleft,Tleft,Tleft]
          qx_edge2=[0,0,0]
          qy_edge2=[0,0,0]
       if edge2_boundary_indicator[iel]==2:  
          T_edge2=[Tright,Tright,Tright]
          qx_edge2=[0,0,0]
          qy_edge2=[0,0,0]
       if edge2_boundary_indicator[iel]==3:  
          T_edge2=[Tbottom,Tbottom,Tbottom]
          qx_edge2=[0,0,0]
          qy_edge2=[0,0,0]
       if edge2_boundary_indicator[iel]==4:  
          T_edge2=[Ttop,Ttop,Ttop]
          qx_edge2=[0,0,0]
          qy_edge2=[0,0,0]
    else:
       #edge 2 is nodes 1 -> 2
       jel=edge2_neighb[iel] # identity of neighbour 
       ed=edge2_neighbedge[iel] 
       if ed==1:
          nodeA=icon[0,jel]
          nodeB=icon[1,jel]
          nodeC=icon[2,jel] # not shared on edge
       if ed==2:
          nodeA=icon[1,jel]
          nodeB=icon[2,jel]
          nodeC=icon[0,jel] # not shared on edge
       if ed==3:
          nodeA=icon[2,jel]
          nodeB=icon[0,jel]
          nodeC=icon[1,jel] # not shared on edge
       T_edge2 =[ T[nodeA], T[nodeB], T[nodeC] ]
       qx_edge2=[qx[nodeA],qx[nodeB],qx[nodeC] ]
       qy_edge2=[qx[nodeA],qy[nodeB],qy[nodeC] ]


    #edge3 contribution: computing qx_edge3,qy_edge3,T_edge3 vectors of length m
    if edge3_boundary_indicator[iel]!=0:
       if edge3_boundary_indicator[iel]==1:  
          T_edge3=[Tleft,Tleft,Tleft]
          qx_edge3=[0,0,0]
          qy_edge3=[0,0,0]
       if edge3_boundary_indicator[iel]==2:  
          T_edge3=[Tright,Tright,Tright]
          qx_edge3=[0,0,0]
          qy_edge3=[0,0,0]
       if edge3_boundary_indicator[iel]==3:  
          T_edge3=[Tbottom,Tbottom,Tbottom]
          qx_edge3=[0,0,0]
          qy_edge3=[0,0,0]
       if edge3_boundary_indicator[iel]==4:  
          T_edge3=[Ttop,Ttop,Ttop]
          qx_edge3=[0,0,0]
          qy_edge3=[0,0,0]
    else:
       #edge 3 is nodes 2 -> 0
       jel=edge3_neighb[iel] # identity of neighbour 
       ed=edge3_neighbedge[iel] 
       if ed==1:
          nodeA=icon[0,jel]
          nodeB=icon[1,jel]
          nodeC=icon[2,jel] # not shared on edge
       if ed==2:
          nodeA=icon[1,jel]
          nodeB=icon[2,jel]
          nodeC=icon[0,jel] # not shared on edge
       if ed==3:
          nodeA=icon[2,jel]
          nodeB=icon[0,jel]
          nodeC=icon[1,jel] # not shared on edge
       T_edge3 =[ T[nodeA], T[nodeB], T[nodeC] ]
       qx_edge3=[qx[nodeA],qx[nodeB],qx[nodeC] ]
       qy_edge3=[qx[nodeA],qy[nodeB],qy[nodeC] ]


    bel=np.zeros(3*m,dtype=np.float64)
    bel[0  :m  ]=edge1_HBx.dot(qx_edge1)+edge2_HBx.dot(qx_edge2)+edge3_HBx.dot(qx_edge3)
    bel[m  :2*m]=edge1_HBy.dot(qy_edge1)+edge2_HBy.dot(qy_edge2)+edge3_HBy.dot(qy_edge3)
    bel[2*m:3*m]=edge1_JBx.dot(qx_edge1)+edge1_JBy.dot(qy_edge1)+edge1_GTB.dot(T_edge1)+\
                 edge2_JBx.dot(qx_edge2)+edge2_JBy.dot(qy_edge2)+edge2_GTB.dot(T_edge2)+\
                 edge3_JBx.dot(qx_edge3)+edge3_JBy.dot(qy_edge3)+edge3_GTB.dot(T_edge3)

    rhs=-bel[:]

    sol = sps.linalg.spsolve(A_Omegae+A_pOmegae,rhs)
    print(sol)

#end for iel


###############################################################################
# plot of solution
###############################################################################

if visu:
   vtufile=open('solution.vtu',"w")
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
   #vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   #for iel in range(0,nel):
   #    vtufile.write("%10e \n" %(p[iel]))
   #vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
   for iel in range(0,nel):
           vtufile.write("%10e \n" %(area[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
   #for i in range(0,NV):
   #    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
   #vtufile.write("</DataArray>\n")
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")















