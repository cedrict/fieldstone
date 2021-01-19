import numpy as np
import numpy.ma as ma
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
from scipy.sparse import lil_matrix
import os

#------------------------------------------------------------------------------

def NNV(rq,sq):
    if element==1:
       NV_0=1-rq-sq-9*(1-rq-sq)*rq*sq 
       NV_1=  rq   -9*(1-rq-sq)*rq*sq
       NV_2=     sq-9*(1-rq-sq)*rq*sq
       NV_3=       27*(1-rq-sq)*rq*sq
       return NV_0,NV_1,NV_2,NV_3
    if element==2:
       NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
       NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
       NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
       NV_3= 4.*(1.-rq-sq)*rq*(1.-3.*sq) 
       NV_4= 4.*rq*sq*(-2.+3.*rq+3.*sq)
       NV_5= 4.*(1.-rq-sq)*sq*(1.-3.*rq) 
       NV_6= 27*(1.-rq-sq)*rq*sq
       return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6

def dNNVdr(rq,sq):
    if element==1:
       dNdr_0= -1-9*(1-2*rq-sq)*sq 
       dNdr_1=  1-9*(1-2*rq-sq)*sq
       dNdr_2=   -9*(1-2*rq-sq)*sq
       dNdr_3=   27*(1-2*rq-sq)*sq
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3
    if element==2:
       dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
       dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
       dNVdr_2= 3*sq-6*rq*sq-3*sq**2
       dNVdr_3= -8*rq+24*rq*sq+4-16*sq+12*sq**2
       dNVdr_4= -8*sq+24*rq*sq+12*sq**2
       dNVdr_5= -16*sq+24*rq*sq+12*sq**2
       dNVdr_6= -54*rq*sq+27*sq-27*sq**2
       return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6

def dNNVds(rq,sq):
    if element==1:
       dNds_0= -1-9*(1-rq-2*sq)*rq
       dNds_1=   -9*(1-rq-2*sq)*rq
       dNds_2=  1-9*(1-rq-2*sq)*rq
       dNds_3=   27*(1-rq-2*sq)*rq
       return dNds_0,dNds_1,dNds_2,dNds_3
    if element==2:
       dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
       dNVds_1= rq*(3-3*rq-6*sq)
       dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
       dNVds_3= -16*rq+24*rq*sq+12*rq**2
       dNVds_4= -8*rq+12*rq**2+24*rq*sq
       dNVds_5= 4-16*rq-8*sq+24*rq*sq+12*rq**2
       dNVds_6= -54*rq*sq+27*rq-27*rq**2
       return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6

def NNP(rq,sq):
    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return NP_0,NP_1,NP_2

#------------------------------------------------------------------------------

print("---------------------------------------")
print("---------------fieldstone--------------")
print("---------------------------------------")

# 1: MINI element
# 2: Crouzeix-Raviart element

element=1

if element==1: # MINI elements
   mV=4     # number of velocity nodes making up an element

if element==2: # Crouzeix-Raviart elements
   mV=7     # number of velocity nodes making up an element

mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

eta_ref=1

#------------------------------------------------------------------------------
# 6 point integration coeffs and weights
# for both elements 

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322/2.
nb6=0.223381589678011/2.

qcoords_r=[nb1,nb2,nb2,nb4,nb3,nb4]
qcoords_s=[nb2,nb1,nb2,nb3,nb4,nb4]
qweights =[nb5,nb5,nb5,nb6,nb6,nb6]

#------------------------------------------------------------------------------
# physical parameters (do not change)
#------------------------------------------------------------------------------

gx=0
gy=-10
Lx=0.9142
Ly=1

#------------------------------------------------------------------------------
# numerical parameters
#------------------------------------------------------------------------------

every=1
nstep=2000
CFL=0.25
end_time=1500
np_surf=100          # initial nb of P1 nodes on the interface 
dist0=Lx/(np_surf-1) # rough estimate of avrg distance between interface nodes
stretch_factor=1.5   # if distance between 2 nodes exceeds stretch_factor*dist0 then add node 

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################
benchfile=open('benchmark.ascii',"w")

model_time=0

for istep in range(0,nstep):

    print("--------------------------------------------")
    print("istep= ", istep, '; time=',model_time)
    print("--------------------------------------------")


    #################################################################
    # make mesh
    #################################################################
    start = timing.time()

    nodesfile=open('mesh.poly',"w")

    if istep==0: # initialisation: I create the interface 

       nodesfile.write("%5d %5d %3d %3d\n" %(4+np_surf,2,0,1))
       counter=0
       #lower left corner
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,0.,0.,0))
       counter+=1
       #lower right corner
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,Lx,0.,0))
       counter+=1
       #upper right corner
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,Lx,Ly,0))
       counter+=1
       #upper left corner
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,0,Ly,0))
       counter+=1

       for i in range (0,np_surf):
           x=i*Lx/(np_surf-1)
           nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,x,0.2+0.02*np.cos(np.pi*x/Lx), 0))
           counter+=1

       np_added=0

    else: # I write out the current interface

       #compute new number of points on interface
       np_added=0
       for i in range (4,np_surf+3):
           #compute distance with next neighbour
           dist=np.sqrt((xV[i]-xV[i+1])**2+(yV[i]-yV[i+1])**2)
           if dist >= stretch_factor*dist0:
              np_added+=1

       nodesfile.write("%5d %5d %3d %3d\n" %(4+np_surf+np_added,2,0,1))
       counter=0
       #lower left corner
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,0.,0.,0))
       counter+=1
       #lower right corner
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,Lx,0.,0))
       counter+=1
       #upper right corner
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,Lx,Ly,0))
       counter+=1
       #upper left corner
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,0,Ly,0))
       counter+=1

       np_added=0
       for i in range (4,np_surf+3):
           #compute distance with next neighbour
           dist=np.sqrt((xV[i]-xV[i+1])**2+(yV[i]-yV[i+1])**2)
           if dist<stretch_factor*dist0:
              #no node added
              nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,xV[i],yV[i], 0))
              counter+=1
           else:
              np_added+=1
              #write current node 
              nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,xV[i],yV[i], 0))
              counter+=1
              #write add node in the middle
              dx=(xV[i+1]-xV[i])/2
              dy=(yV[i+1]-yV[i])/2
              nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,xV[i]+dx,yV[i]+dy, 0))
              counter+=1

       #add last node
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,xV[np_surf+3],yV[np_surf+3], 0))

    print ('     -> old value of np_surf',np_surf)
    print ('     -> nb of added points to interface',np_added)
    np_surf+=np_added
    print ('     -> new value of np_surf',np_surf)

    #write out connectivity of interface 
    nodesfile.write("%5d %5d \n" %(np_surf-1,0))
    counter=4
    counter_segment=0
    for i in range (0,np_surf):
        counter+=1
        if i<np_surf-1:
           counter_segment+=1
           nodesfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

    #write number of holes
    nodesfile.write("%5d \n" %(0))

    nodesfile.close()

    # call triangle mesher
    # -p: Triangulates a Planar Straight Line Graph (.poly file)
    # -c: Encloses the convex hull with segments 
    # -j: Jettison unused vertices from output .node file
    # -q: Quality mesh generation.  A minimum angle may be specified
    #-o2: Generates second-order subparametric elements.
    # -a: Applies a maximum triangle area constraint.

    if element==1:
       os.system("../../../../triangle/triangle  -j -q20 -a0.001 -pc mesh.poly") # default

    if element==2:
       #os.system("../../../../triangle/triangle  -j -q -a0.001 -o2 -pc mesh.poly") # default
       os.system("../../../../triangle/triangle  -j -q20 -a0.001 -o2 -pc mesh.poly") # default

    #read nb of elements and nb of nodes 
    os.system(" head -1 mesh.1.ele > temp ")
    os.system(" head -1 mesh.1.node >> temp ")
    counter=0
    file=open("temp", "r")
    for line in file:
        fields = line.strip().split()
        #print(fields[0], fields[1], fields[2])
        if counter==0:
           nel=int(fields[0])
        if counter==1:
           NV0=int(fields[0])
        counter+=1

    NV=NV0+nel # adding bubbles
 
    NfemV=NV*ndofV     # number of velocity dofs

    if element==1:
       NfemP=NV0*ndofP   # number of pressure dofs
    if element==2:
       NfemP=nel*3*ndofP   # number of pressure dofs

    Nfem=NfemV+NfemP    # total number of dofs
 
    print ('     -> nel', nel)
    print ('     -> NV0', NV0)
    print ('     -> NfemV', NfemV)
    print ('     -> NfemP', NfemP)
    print ('     -> Nfem ', Nfem)

    print("make mesh with Triangle: %.3f s" % (timing.time() - start))

    #################################################################
    # grid point setup
    #################################################################
    start = timing.time()

    xV=np.zeros(NV,dtype=np.float64)     # x coordinates
    yV=np.zeros(NV,dtype=np.float64)     # y coordinates

    xV[0:NV0],yV[0:NV0]=np.loadtxt('mesh.1.node',unpack=True,usecols=[1,2],skiprows=1)

    print("     -> xV (min/max): %.4f %.4f" %(np.min(xV[0:NV0]),np.max(xV[0:NV0])))
    print("     -> yV (min/max): %.4f %.4f" %(np.min(yV[0:NV0]),np.max(yV[0:NV0])))

    #np.savetxt('gridV0.ascii',np.array([xV,yV]).T,header='# xV,yV')

    print("read in mesh from file: %.3f s" % (timing.time() - start))

    #################################################################
    # connectivity
    #################################################################
    #
    #  P_2^+           P_1^+
    #
    #  02              02
    #  ||\\            ||\\
    #  || \\           || \\
    #  ||  \\          ||  \\
    #  05   04         ||   \\
    #  || 06 \\        || 03 \\
    #  ||     \\       ||     \\
    #  00==03==01      00======01
    #
    # note that the ordering of nodes returned by triangle is different
    # than mine: https://www.cs.cmu.edu/~quake/triangle.highorder.html.
    # note also that triangle returns nodes 0-5, but not 6.
    #################################################################
    start = timing.time()

    iconV=np.zeros((mV,nel),dtype=np.int32)

    if element==1:
       iconV[0,:],iconV[1,:],iconV[2,:]=\
       np.loadtxt('mesh.1.ele',unpack=True, usecols=[1,2,3],skiprows=1)
       iconV[0,:]-=1
       iconV[1,:]-=1
       iconV[2,:]-=1
       for iel in range (0,nel):
           iconV[3,iel]=NV0+iel

    if element==2:
       iconV[0,:],iconV[1,:],iconV[2,:],iconV[4,:],iconV[5,:],iconV[3,:]=\
       np.loadtxt('mesh.1.ele',unpack=True, usecols=[1,2,3,4,5,6],skiprows=1)
       iconV[0,:]-=1
       iconV[1,:]-=1
       iconV[2,:]-=1
       iconV[3,:]-=1
       iconV[4,:]-=1
       iconV[5,:]-=1
       for iel in range (0,nel):
           iconV[6,iel]=NV0+iel

    for iel in range (0,nel): #bubble nodes
        xV[NV0+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
        yV[NV0+iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.

    #np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# xV,yV')

    print("read in connectivity from file: %.3f s" % (timing.time() - start))

    #################################################################
    # build pressure grid (nodes and icon)
    #################################################################
    start = timing.time()

    iconP=np.zeros((mP,nel),dtype=np.int32)
    xP=np.empty(NfemP,dtype=np.float64)     # x coordinates
    yP=np.empty(NfemP,dtype=np.float64)     # y coordinates

    if element==1:
       iconP[0,:]=iconV[0,:]
       iconP[1,:]=iconV[1,:]
       iconP[2,:]=iconV[2,:]
       xP[0:NV0]=xV[0:NV0]
       yP[0:NV0]=yV[0:NV0]

    if element==2:
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
 
    #np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')
 
    #for iel in range (0,nel):
    #    print ("iel=",iel)
    #    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], yP[iconP[0][iel]])
    #    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], yP[iconP[1][iel]])
    #    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], yP[iconP[2][iel]])
 
    print("build pressure connectivity P: %.3f s" % (timing.time() - start))

    #################################################################
    # flag nodes and edges on interface
    #################################################################
    start = timing.time()

    interface=np.zeros(NV,dtype=np.bool) 
    edge_on_interface=np.zeros((3,nel),dtype=np.bool) 

    for i in range(4,np_surf+4):
        interface[i]=True

    for iel in range(0,nel):
        if (interface[iconV[0,iel]] and interface[iconV[1,iel]]):
           edge_on_interface[0,iel]=True
        if (interface[iconV[1,iel]] and interface[iconV[2,iel]]):
           edge_on_interface[1,iel]=True
        if (interface[iconV[2,iel]] and interface[iconV[0,iel]]):
           edge_on_interface[2,iel]=True

    if element==2:
       for iel in range(0,nel):
           if (interface[iconV[0,iel]] and interface[iconV[1,iel]]):
              interface[iconV[3,iel]]=True
           if (interface[iconV[1,iel]] and interface[iconV[2,iel]]):
              interface[iconV[4,iel]]=True
           if (interface[iconV[2,iel]] and interface[iconV[0,iel]]):
              interface[iconV[5,iel]]=True

    print("flag interface nodes & edges: %.3f s" % (timing.time() - start))

    #################################################################
    # assigning material properties to elements
    #################################################################
    start = timing.time()

    rho=np.zeros(nel,dtype=np.float64) 
    eta=np.zeros(nel,dtype=np.float64) 
    mat=np.zeros(nel,dtype=np.int16) 
    matnod=np.zeros(NV,dtype=np.int16) 
    edge=np.ones((3,nel),dtype=np.int16) 

    mat[:]=1
    matnod[:]=1
    rho[:]=1010
    eta[:]=100
 
    if istep==0:
       if element==1:
          centernode=3
       if element==2:
          centernode=6
       for iel in range(0,nel):
           x_c=xV[iconV[centernode,iel]]
           y_c=yV[iconV[centernode,iel]]
           if y_c<0.2+0.02*np.cos(x_c*np.pi/Lx):
              mat[iel]=2
              rho[iel]=1000
              eta[iel]=100
              matnod[iconV[0:mV,iel]]=2
              edge[0:3,iel]=2

    else:
       mat[:]=1
       matnod[:]=1
       rho[:]=1010
       eta[:]=100

       #first layer of triangles below interface
       for iel in range(0,nel):
           #if two consecutive nodes on the interface...
           if (interface[iconV[0,iel]] and interface[iconV[1,iel]] and iconV[1,iel]+1-iconV[0,iel]==0) or \
              (interface[iconV[1,iel]] and interface[iconV[2,iel]] and iconV[2,iel]+1-iconV[1,iel]==0) or \
              (interface[iconV[2,iel]] and interface[iconV[0,iel]] and iconV[0,iel]+1-iconV[2,iel]==0) :
              mat[iel]=2
              rho[iel]=1000
              eta[iel]=100
              matnod[iconV[0:mV,iel]]=2
              edge[0:3,iel]=2

       #recursively loop over all triangles
       #if a triangle has an edge (ie its node in the middle) that is not on the 
       #interface but that is flagged mat=2, then the triangle is mat=2 
       #I could make it more elegant by detecting when the number does 
       #not change, but it is super fast, so good enough for now.
       for k in range(0,15):

          #update/propagate material nodal information to edges
          #I need to exclude edges on interface to avoid bleeding on the wrong side of interface
          for iel in range(0,nel):
              if matnod[iconV[0,iel]]==2 and matnod[iconV[1,iel]]==2 and not edge_on_interface[0,iel]:
                 edge[0,iel]=2
              if matnod[iconV[1,iel]]==2 and matnod[iconV[2,iel]]==2 and not edge_on_interface[1,iel]:
                 edge[1,iel]=2
              if matnod[iconV[2,iel]]==2 and matnod[iconV[0,iel]]==2 and not edge_on_interface[2,iel]:
                 edge[2,iel]=2

          for iel in range(0,nel):
              if (edge[0,iel]==2 and not edge_on_interface[0,iel]) or \
                 (edge[1,iel]==2 and not edge_on_interface[1,iel]) or \
                 (edge[2,iel]==2 and not edge_on_interface[2,iel]) :
                 mat[iel]=2
                 rho[iel]=1000
                 eta[iel]=100
                 matnod[iconV[0:mV,iel]]=2
                 edge[0:3,iel]=2

          print('     it=',k,'nb triangles of mat 2=',np.count_nonzero(mat==2)) 
    #end if
 
    print("assign material to triangles: %.3f s" % (timing.time() - start))

    #################################################################
    # define boundary conditions
    #################################################################
    start = timing.time()

    bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
    bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

    for i in range(0, NV):
        #Left boundary  
        if xV[i]/Lx<0.0000001:
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
        #right boundary  
        if xV[i]/Lx>0.9999999:
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
        #bottom boundary  
        if yV[i]/Lx<1e-6:
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0   
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
        #top boundary  
        if yV[i]/Ly>0.9999999:
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0  
           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
 
    print("define boundary conditions: %.3f s" % (timing.time() - start))

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = timing.time()

    A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    NNNV     = np.zeros(mV,dtype=np.float64)           # shape functions V
    dNNNVdr  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    rhs      = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    f_rhs    = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs    = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
    b_mat    = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat    = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
    NNNV     = np.zeros(mV,dtype=np.float64)           # shape functions V
    NNNP     = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNNNVdx  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVdr  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives

    c_mat    = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)
        NNNP= np.zeros(mP*ndofP,dtype=np.float64)   

        for kq in range (0,nqel):

            # position & weight of quad. point
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            #if jcob<0:
            #   exit("jacobian is negative - bad triangle")

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            # compute etaq, rhoq
            etaq=eta[iel]
            rhoq=rho[iel]
            #print (etaq,rhoq)

            # construct b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*rhoq
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rhoq

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        # impose b.c. 
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
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
                #end if
            #end for 
        #end for

        G_el*=eta_ref/Ly
        h_el*=eta_ref/Ly

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        #K_mat[m1,m2]+=K_el[ikk,jkk]
                        A_sparse[m1,m2] += K_el[ikk,jkk] 
                    #end for
                #end for
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    #G_mat[m1,m2]+=G_el[ikk,jkk]
                    A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk] 
                    A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk] 
                #end for
                f_rhs[m1]+=f_el[ikk] 
            #end for
        #end for
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]  
        #end for
    #end for

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("build FE matrix: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    sol = np.zeros(Nfem,dtype=np.float64) 

    sparse_matrix=A_sparse.tocsr()

    #print(sparse_matrix.min(),sparse_matrix.max())

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*eta_ref/Ly

    print("     -> u (m,M) %.6e %.6e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.6e %.6e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.6e %.6e " %(np.min(p),np.max(p)))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

    print("solve time: %.3f s" % (timing.time() - start))

    #################################################################
    # compute area of elements
    #################################################################
    start = timing.time()

    area = np.zeros(nel,dtype=np.float64) 

    avrg_p=0
    vrms=0
    vrms_a=0
    vrms_f=0
    vrms_s=0
    vol_f=0
    vol_s=0
    for iel in range(0,nel):
        for kq in range (0,nqel):
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            area[iel]+=jcob*weightq

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
 
            vrms+=(uq**2+vq**2)*jcob*weightq 

            if mat[iel]==1:
               vrms_f+=(uq**2+vq**2)*jcob*weightq 
               vol_f+=jcob*weightq
               
            if mat[iel]==2:
               vrms_s+=(uq**2+vq**2)*jcob*weightq 
               vol_s+=jcob*weightq

            pq=0.0
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            avrg_p+=pq*jcob*weightq

        #end for
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))
    vrms_f=np.sqrt(vrms_f/vol_f)
    vrms_s=np.sqrt(vrms_s/vol_s)

    avrg_p/=(Lx*Ly)
    p-=avrg_p

    print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
    print("     -> total area (meas) %.6f " %(area.sum()))
    print("     -> total area (anal) %.6f " %(Lx*Ly))
    print("     -> vrms   = %e " %(vrms))
    print("     -> vrms_f = %e " %(vrms_f))
    print("     -> vrms_s = %e " %(vrms_s))
    print("     -> avrg_p = %e " %(avrg_p))

    print("compute area & vrms: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute time stepping
    ######################################################################
    start = timing.time()

    dt=(CFL*np.min(np.sqrt(area))/np.max(np.sqrt(u**2+v**2)))

    dt=min(dt,1)

    print("     -> dt=", dt)

    print("compute time step: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute elemental strainrate 
    ######################################################################
    start = timing.time()

    xc = np.zeros(nel,dtype=np.float64)  
    yc = np.zeros(nel,dtype=np.float64)  
    exx = np.zeros(nel,dtype=np.float64)  
    eyy = np.zeros(nel,dtype=np.float64)  
    exy = np.zeros(nel,dtype=np.float64)  
    e   = np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):
        rq = 0.0
        sq = 0.0
        weightq = 2.0 * 2.0
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        for k in range(0,mV):
            xc[iel] += NNNV[k]*xV[iconV[k,iel]]
            yc[iel] += NNNV[k]*yV[iconV[k,iel]]
            exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
            eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
            exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
        e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

    print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))

    print("compute sr and stress: %.3f s" % (timing.time() - start))

    #####################################################################
    # interpolate pressure onto velocity grid points
    #####################################################################
    #
    #  02          #  02
    #  ||\\        #  ||\\
    #  || \\       #  || \\
    #  ||  \\      #  ||  \\
    #  05   04     #  ||   \\
    #  || 06 \\    #  ||    \\
    #  ||     \\   #  ||     \\
    #  00==03==01  #  00======01
    #
    #####################################################################

    q=np.zeros(NV,dtype=np.float64)
    p_el=np.zeros(nel,dtype=np.float64)

    if element==1:
       q[0:NfemP]=p[0:NfemP]
       for iel in range(0,nel):
           p_el[iel]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.

    if element==2:
       cc=np.zeros(NV,dtype=np.float64)
       for iel in range(0,nel):
           q[iconV[0,iel]]+=p[iconP[0,iel]]
           cc[iconV[0,iel]]+=1.
           q[iconV[1,iel]]+=p[iconP[1,iel]]
           cc[iconV[1,iel]]+=1.
           q[iconV[2,iel]]+=p[iconP[2,iel]]
           cc[iconV[2,iel]]+=1.
           q[iconV[3,iel]]+=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
           cc[iconV[3,iel]]+=1.
           q[iconV[4,iel]]+=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
           cc[iconV[4,iel]]+=1.
           q[iconV[5,iel]]+=(p[iconP[0,iel]]+p[iconP[2,iel]])*0.5
           cc[iconV[5,iel]]+=1.
           p_el[iel]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.

       for i in range(0,NV):
           if cc[i] != 0:
              q[i]=q[i]/cc[i]

    #####################################################################
    # interpolate strain rate onto velocity grid points
    #####################################################################
    #
    #  02          #  02
    #  ||\\        #  ||\\
    #  || \\       #  || \\
    #  ||  \\      #  ||  \\
    #  05   04     #  ||   \\
    #  || 06 \\    #  ||    \\
    #  ||     \\   #  ||     \\
    #  00==03==01  #  00======01
    #
    #####################################################################
    # crude method - could be improved!

    exx_n=np.zeros(NV,dtype=np.float64)
    eyy_n=np.zeros(NV,dtype=np.float64)
    exy_n=np.zeros(NV,dtype=np.float64)
    cc=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        for k in range(0,mV):
            exx_n[iconV[k,iel]]+=exx[iel]
            eyy_n[iconV[k,iel]]+=eyy[iel]
            exy_n[iconV[k,iel]]+=exy[iel]
            cc[iconV[k,iel]]+=1.

    for i in range(0,NV):
        if cc[i] != 0:
           exx_n[i]/=cc[i]
           eyy_n[i]/=cc[i]
           exy_n[i]/=cc[i]

    #####################################################################
    # compure dev stress tensor and stress tensor 
    #####################################################################

    tauxx = np.zeros(nel,dtype=np.float64)  
    tauyy = np.zeros(nel,dtype=np.float64)  
    tauxy = np.zeros(nel,dtype=np.float64)  
    sigmaxx = np.zeros(nel,dtype=np.float64)  
    sigmayy = np.zeros(nel,dtype=np.float64)  
    sigmaxy = np.zeros(nel,dtype=np.float64)  
    
    tauxx[:]=2*eta[:]*exx[:]
    tauyy[:]=2*eta[:]*eyy[:]
    tauxy[:]=2*eta[:]*exy[:]

    sigmaxx[:]=-p_el[:]+2*eta[:]*exx[:]
    sigmayy[:]=-p_el[:]+2*eta[:]*eyy[:]
    sigmaxy[:]=        +2*eta[:]*exy[:]


    #####################################################################
    # carry out measurements for benchmark
    #####################################################################
    start = timing.time()

    avrg_rho=0.
    avrg_eta=0.
    for iel in range(0,nel):
        avrg_rho+=rho[iel]*area[iel]
        avrg_eta+=eta[iel]*area[iel]
    avrg_rho/=(Lx*Ly)
    avrg_eta/=(Lx*Ly)

    vel=np.sqrt(u**2+v**2)

    length=0.
    for i in range(4,3+np_surf):
        length+=np.sqrt((xV[i+1]-xV[i])**2+(yV[i+1]-yV[i])**2)
    print('     -> length interface=',length)

    benchfile.write("%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n" %( \
    nel,Nfem,model_time,\
    np.min(u),np.max(u),\
    np.min(v),np.max(v),\
    np.min(vel),np.max(vel),\
    np.min(p),np.max(p),
    vrms,vrms_f,vrms_s,\
    avrg_rho,avrg_eta,\
    vol_f,vol_s,dt,np_surf,
    yV[np_surf+3],length))
    benchfile.flush()

    print("export measurements: %.3f s" % (timing.time() - start))

    #####################################################################
    # plot of solution
    # the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
    #####################################################################
    start = timing.time()

    if istep%every==0:

       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (area[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%7e\n" % (rho[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%7e\n" % (eta[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d\n" % (mat[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p (el)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%7e\n" % (p_el[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (eyy[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")

       #--
       #vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    vtufile.write("%10e\n" % tauxx[iel])
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    vtufile.write("%10e\n" % tauyy[iel])
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    vtufile.write("%10e\n" % tauxy[iel])
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (e[iel]))
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    vtufile.write("%10e\n" % sigmaxx[iel])
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    vtufile.write("%10e\n" % sigmayy[iel])
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
       #for iel in range (0,nel):
       #    vtufile.write("%10e\n" % sigmaxy[iel])
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='theta_p(dev stress)' Format='ascii'> \n")
       #for iel in range(0,nel):
       #    theta_p=0.5*np.arctan(2*tauxy[iel]/(tauxx[iel]-tauyy[iel]))
       #    vtufile.write("%10e \n" % (theta_p/np.pi*180.))
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='theta_p(stress)' Format='ascii'> \n")
       #for iel in range(0,nel):
       #    theta_p=0.5*np.arctan(2*sigmaxy[iel]/(sigmaxx[iel]-sigmayy[iel]))
       #    vtufile.write("%10e \n" % (theta_p/np.pi*180.))
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='tau_max' Format='ascii'> \n")
       #for iel in range(0,nel):
       #    tau_max=np.sqrt( (tauxx[iel]-tauyy[iel])**2/4 +tauxy[iel]**2 )
       #    vtufile.write("%10e \n" % tau_max)
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='sigma_max' Format='ascii'> \n")
       #for iel in range(0,nel):
       #    sigma_max=np.sqrt( (sigmaxx[iel]-sigmayy[iel])**2/4 +sigmaxy[iel]**2 )
       #    vtufile.write("%10e \n" % sigma_max)
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p (nod)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %matnod[i])
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" % exx_n[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" % eyy_n[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" % exy_n[i])
       vtufile.write("</DataArray>\n")




       #--
       vtufile.write("<DataArray type='Float32' Name='interface' Format='ascii'> \n")
       for i in range(0,NV):
           if interface[i]:
              vtufile.write("%10e \n" % 1)
           else:
              vtufile.write("%10e \n" % 0)
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
       #for i in range(0,NV):
       #    if bc_fix[i*2]:
       #       val=1
       #    else:
       #       val=0
       #    vtufile.write("%10e \n" %val)
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
       #for i in range(0,NV):
       #    if bc_fix[i*2+1]:
       #       val=1
       #    else:
       #       val=0
       #    vtufile.write("%10e \n" %val)
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if element==1:
          for iel in range (0,nel):
              vtufile.write("%d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel]))
       if element==2:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                    iconV[3,iel],iconV[4,iel],iconV[5,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       if element==1:
          for iel in range (0,nel):
              vtufile.write("%d \n" %((iel+1)*3))
       if element==2:
          for iel in range (0,nel):
              vtufile.write("%d \n" %((iel+1)*6))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       if element==1:
          for iel in range (0,nel):
              vtufile.write("%d \n" %5)
       if element==2:
          for iel in range (0,nel):
              vtufile.write("%d \n" %22)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()


       filename = 'interface_{:04d}.ascii'.format(istep)
       surffile=open(filename,"w")
       surffile.write("#time= %e \n" %model_time)
       for i in range(4,4+np_surf):
           surffile.write("%e %e %e %e\n" %(xV[i],yV[i],u[i],v[i]))
       surffile.close()


    print("write data: %.3fs" % (timing.time() - start))

    #################################################################
    # mesh evolution
    #################################################################

    for i in range(0,NV):
        xV[i]+=u[i]*dt
        yV[i]+=v[i]*dt

    if element==2:
       for iel in range(0,nel):
           # node 3 is between nodes 0 and 1
           xV[iconV[3,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[1,iel]]) 
           yV[iconV[3,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[1,iel]]) 
           # node 4 is between nodes 1 and 2
           xV[iconV[4,iel]]=0.5*(xV[iconV[1,iel]]+xV[iconV[2,iel]]) 
           yV[iconV[4,iel]]=0.5*(yV[iconV[1,iel]]+yV[iconV[2,iel]]) 
           # node 5 is between nodes 0 and 2
           xV[iconV[5,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[2,iel]]) 
           yV[iconV[5,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[2,iel]]) 
           # recenter middle node
           xV[iconV[6,iel]]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
           yV[iconV[6,iel]]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.

    for iel in range(0,nel):
        xP[iconP[0,iel]]=xV[iconV[0,iel]]
        yP[iconP[0,iel]]=yV[iconV[0,iel]]
        xP[iconP[1,iel]]=xV[iconV[1,iel]]
        yP[iconP[1,iel]]=yV[iconV[1,iel]]
        xP[iconP[2,iel]]=xV[iconV[2,iel]]
        yP[iconP[2,iel]]=yV[iconV[2,iel]]

    #####################################################################

    model_time+=dt

    #####################################################################

    if model_time>=end_time: 
       print("***********************************")
       print("*********end time reached**********")
       print("***********************************")
       break

    #####################################################################
