import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import time as timing

from tools import *
from basis_functions_numba import *


#------------------------------------------------------------------------------

def viscosity(x,y,Ly,eta_um,eta_c,eta_o,xA,yB,xC,xE,yE,xF,yF,yG,yI):

    if y<yI:
       val=1e21
    elif y<yG:
       val=5e22
    else:
       val=eta_um

    if x>xA and x<xC and y>yB:
       val=eta_c

    if x>xC and x<xE and y>yK:
       val=eta_o
    if x>xE and y>(yF-yE)/(xF-xE)*(x-xE)+yE:
       val=eta_o

    return val

#------------------------------------------------------------------------------

year=365.25*3600*24
cm=0.01
km=1e3

print("-----------------------------")
print("--------- stone 143 ---------")
print("-----------------------------")

CR=False

if CR:
   mV=7     # number of velocity nodes making up an element
else:
   mV=6

mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

eta_ref=1e22 # numerical parameter for FEM

Lx=6000*km
Ly=3000*km










#note that lengths are in km for now

h=20*km

xA=1000*km
yA=Ly

xB=1000*km
yB=2800*km

xC=xA+2000*km
yC=Ly

xD=xB+2000*km
yD=yB

xE=6000*km-500*km #Lx-lr
yE=3000*km-100*km

xF=6000*km-h
yF=3000*km

xG=0
yG=3000*km-660*km

xH=6000*km
yH=3000*km-660*km

xI=0
yI=350*km

xJ=6000*km
yJ=350*km

xK=xD
yK=yE

eta_um=1e20
eta_o=1e23
eta_c=1e22


if True:

   square_vertices = np.array([[0,0],[0,Ly],[Lx,Ly],[Lx,0]])
   square_edges = compute_segs(square_vertices)
   offset=4

   #segment AB 
   nptsAB=int((yA-yB)/h)
   print(nptsAB)
   x1=np.zeros(nptsAB) ; x1[:]=xA
   y1=np.linspace(yA,yB, nptsAB, endpoint=True)
   pointsAB = np.stack([x1,y1],axis = 1)
   segsAB = np.stack([np.arange(nptsAB-1) +offset , np.arange(nptsAB-1) + 1 +offset], axis=1) 
   offset+=nptsAB

   #segment BD
   nptsBD=int((xD-xB)/h)
   print(nptsBD)
   x1=np.linspace(xB,xD, nptsBD, endpoint=True)
   y1=np.zeros(nptsBD) ; y1[:]=yB
   pointsBD = np.stack([x1,y1],axis = 1)
   segsBD = np.stack([np.arange(nptsBD-1) +offset , np.arange(nptsBD-1) + 1 +offset], axis=1) 
   offset+=nptsBD

   #segment CD 
   nptsCD=int((yC-yD)/h)
   x1=np.zeros(nptsCD) ; x1[:]=xC
   y1=np.linspace(yC,yD, nptsCD, endpoint=True)
   pointsCD = np.stack([x1,y1],axis = 1)
   segsCD = np.stack([np.arange(nptsCD-1) +offset , np.arange(nptsCD-1) + 1 +offset], axis=1) 
   offset+=nptsCD

   #segment KE
   nptsKE=int((xE-xK)/h)
   x1=np.linspace(xK,xE, nptsKE, endpoint=True)
   y1=np.zeros(nptsKE) ; y1[:]=yK
   pointsKE = np.stack([x1,y1],axis = 1)
   segsKE = np.stack([np.arange(nptsKE-1) +offset , np.arange(nptsKE-1) + 1 +offset], axis=1) 
   offset+=nptsKE

   #segment EF
   nptsEF=int((xF-xE)/h)
   x1=np.linspace(xE,xF, nptsEF, endpoint=True)
   y1=np.linspace(yE,yF, nptsEF, endpoint=True)
   pointsEF = np.stack([x1,y1],axis = 1)
   segsEF = np.stack([np.arange(nptsEF-1) +offset , np.arange(nptsEF-1) + 1 +offset], axis=1) 
   offset+=nptsEF

   #segment GH
   nptsGH=100
   x1=np.linspace(xG,xH, nptsGH, endpoint=True)
   y1=np.zeros(nptsGH) ; y1[:]=yG
   pointsGH = np.stack([x1,y1],axis = 1)
   segsGH = np.stack([np.arange(nptsGH-1) +offset , np.arange(nptsGH-1) + 1 +offset], axis=1) 
   offset+=nptsGH

   #segment IJ
   nptsIJ=100
   x1=np.linspace(xI,xJ, nptsIJ, endpoint=True)
   y1=np.zeros(nptsIJ) ; y1[:]=yI
   pointsIJ = np.stack([x1,y1],axis = 1)
   segsIJ = np.stack([np.arange(nptsIJ-1) +offset , np.arange(nptsIJ-1) + 1 +offset], axis=1) 
   offset+=nptsIJ

   #assemble all coordinate arrays
   points = np.vstack([square_vertices,pointsAB,pointsBD,pointsCD,pointsKE,pointsEF,pointsGH,pointsIJ])

   #assemble all segments arrays
   SEGS = np.vstack([square_edges, segsAB, segsBD, segsCD, segsKE, segsEF, segsGH, segsIJ])

   O1 = {'vertices' : points, 'segments' : SEGS}
   T1 = tr.triangulate(O1, 'pqa30000000000') # tr.triangulate() computes the main dictionary 

   area=compute_triangles_area(T1['vertices'], T1['triangles'])
   iconP1=T1['triangles'] ; iconP1=iconP1.T
   xP1=T1['vertices'][:,0] 
   yP1=T1['vertices'][:,1] 
   NP1=np.size(xP1)

   np.savetxt('meshP1.ascii',np.array([xP1,yP1]).T,header='# xV,zV') 


   print('NP1=',NP1)

   export_elements_to_vtu(xP1,yP1,iconP1,'meshP1.vtu',area)

   mP,nel=np.shape(iconP1)
   print('nel=',nel)

   print(np.shape(xP1))
   print(np.shape(yP1))
   print(np.shape(iconP1))

   NV0,xP2,yP2,iconP2=mesh_P1_to_P2(xP1,yP1,iconP1)

   np.savetxt('meshP2.ascii',np.array([xP2,yP2]).T,header='# xV,zV') 
   print('NV0=',NV0)

   export_elements_to_vtuP2(xP2,yP2,iconP2,'meshP2.vtu')

######################################################################
# compute element center coordinates
######################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]= (xP1[iconP1[0,iel]]+xP1[iconP1[1,iel]]+xP1[iconP1[2,iel]])/3
    yc[iel]= (yP1[iconP1[0,iel]]+yP1[iconP1[1,iel]]+yP1[iconP1[2,iel]])/3

print("     -> xc (m,M) %.6e %.6e " %(np.min(xc),np.max(xc)))
print("     -> yc (m,M) %.6e %.6e " %(np.min(yc),np.max(yc)))

print("compute element center coords: %.3f s" % (timing.time() - start))


######################################################################
# assign viscosity to elements
######################################################################

eta=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    eta[iel]=viscosity(xc[iel],yc[iel],Ly,eta_um,eta_c,eta_o,xA,yB,xC,xE,yE,xF,yF,yG,yI)


np.savetxt('viscosity.ascii',np.array([xc,yc,np.log10(eta)]).T,header='# xV,zV') 





