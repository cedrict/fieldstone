import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from numpy import random
from numpy import linalg as LA

xx=0
yy=1
zz=2

Ggrav = 1 #6.67e-11

#--------------------------------------------------------------------

def export_vector_to_vtu(pt,vec,name):
    vtufile=open(name+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %2d ' NumberOfCells=' %2d '> \n" %(1,1))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(pt[0],pt[1],pt[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vector' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(vec[0],vec[1],vec[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    vtufile.write("%d\n" %(0))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    vtufile.write("%d \n" %(1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    vtufile.write("%d \n" %1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

def export_tetrahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,name):
    vtufile=open(name+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(4,1))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(pt_1[0],pt_1[1],pt_1[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_2[0],pt_2[1],pt_2[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_3[0],pt_3[1],pt_3[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_4[0],pt_4[1],pt_4[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    vtufile.write("%d %d %d %d\n" %(0,1,2,3))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    vtufile.write("%d \n" %(4))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    vtufile.write("%d \n" %10)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#--------------------------------------------------------------------

def compute_face_edge_normal(vec_nface,pt_start,pt_end,pt_midface,name):
    pt_mid=0.5*(pt_start+pt_end)
    vec_l=pt_end-pt_start
    vec_n=np.empty(3)
    det=vec_nface[yy]*vec_l[zz]-vec_nface[zz]*vec_l[yy]
    vec_n[xx]=1
    vec_n[yy]=(-vec_l[zz]*vec_nface[xx]+vec_nface[zz]*vec_l[xx] )/det
    vec_n[zz]=(+vec_l[yy]*vec_nface[xx]-vec_nface[yy]*vec_l[xx] )/det
    vec_n/=LA.norm(vec_n,2)
    #print('in',np.dot(vec_n,vec_nface))
    #print('in',np.dot(vec_n,vec_l))
    vec_n*=np.sign(np.dot(vec_n,pt_mid-pt_midface))
    export_vector_to_vtu(pt_mid,vec_n,name)
    return vec_n

#--------------------------------------------------------------------

def compute_gravity_tetrahedron_pointmass(pt_1,pt_2,pt_3,pt_4,pt_M,rho0):

    #compute volume of tet

    vec_l1 = pt_2-pt_1
    vec_l2 = pt_3-pt_1
    vec_l3 = pt_4-pt_1
    volume = np.abs(np.dot(np.cross(vec_l3,vec_l2),vec_l1))/6

    #print('volume=',volume, 8/9/np.sqrt(3)) 

    #compute center of mass N

    pt_N = (pt_1+pt_2+pt_3+pt_4)*0.25

    #print('pt_N',pt_N)

    # compute distance \vec{MN}

    vec_rN=pt_N-pt_M

    dist = LA.norm(vec_rN,2)

    #print(dist)

    vec_g = Ggrav*rho0*volume/dist**3*vec_rN

    U=Ggrav*rho0*volume/dist

    return vec_g,U


#--------------------------------------------------------------------

def compute_gravity_tetrahedron(pt_1,pt_2,pt_3,pt_4,pt_M,rho0):
    # pt1,2,3,4,M: arrays containing x,y,z coordinates

    vec_gf=np.zeros(3,dtype=np.float64)
    vec_ge=np.zeros(3,dtype=np.float64)

    #---step1---
    # order 12 or 21 does not matter much:
    # these are used to compute the normal to faces.
    # and later only the norm is used.

    vec_l1 = pt_2-pt_1
    vec_l2 = pt_3-pt_1
    vec_l3 = pt_4-pt_1
    vec_l4 = pt_3-pt_2
    vec_l5 = pt_4-pt_2
    vec_l6 = pt_4-pt_3

    l1 = LA.norm(vec_l1,2)
    l2 = LA.norm(vec_l2,2)
    l3 = LA.norm(vec_l3,2)
    l4 = LA.norm(vec_l4,2)
    l5 = LA.norm(vec_l5,2)
    l6 = LA.norm(vec_l6,2)

    #print(l1,l2,l3,l4,l5,l6,np.sqrt(8/3))

    #---step2---

    vec_r1 = pt_1-pt_M 
    vec_r2 = pt_2-pt_M 
    vec_r3 = pt_3-pt_M 
    vec_r4 = pt_4-pt_M 

    r1=LA.norm(vec_r1,2)
    r2=LA.norm(vec_r2,2)
    r3=LA.norm(vec_r3,2)
    r4=LA.norm(vec_r4,2)
 
    #---step3---compute normal vectors to faces

    pt_midA = (pt_1+pt_2+pt_3)/3
    pt_midB = (pt_1+pt_3+pt_4)/3
    pt_midC = (pt_1+pt_4+pt_2)/3
    pt_midD = (pt_2+pt_4+pt_3)/3

    vec_nA = np.cross(vec_l1,vec_l2)/(l1*l2)
    vec_nB = np.cross(vec_l2,vec_l3)/(l2*l3)
    vec_nC = np.cross(vec_l3,vec_l1)/(l3*l1)
    vec_nD = np.cross(vec_l5,vec_l4)/(l5*l4)

    export_vector_to_vtu(pt_midA,vec_nA,'faceA_normal')
    export_vector_to_vtu(pt_midB,vec_nB,'faceB_normal')
    export_vector_to_vtu(pt_midC,vec_nC,'faceC_normal')
    export_vector_to_vtu(pt_midD,vec_nD,'faceD_normal')

    #---step4---

    mat_FA = np.outer(vec_nA,vec_nA)
    mat_FB = np.outer(vec_nB,vec_nB)
    mat_FC = np.outer(vec_nC,vec_nC)
    mat_FD = np.outer(vec_nD,vec_nD)

    #---step5---

    wA = np.dot(vec_r1,np.cross(vec_r2,vec_r3))/\
         (r1*r2*r3+r1*np.dot(vec_r2,vec_r3)+r2*np.dot(vec_r3,vec_r1)+r3*np.dot(vec_r1,vec_r2))
    wA = 2*np.arctan(wA)

    wB = np.dot(vec_r1,np.cross(vec_r3,vec_r4))/\
         (r1*r3*r4+r1*np.dot(vec_r3,vec_r4)+r3*np.dot(vec_r4,vec_r1)+r4*np.dot(vec_r1,vec_r3))
    wB = 2*np.arctan(wB)

    wC = np.dot(vec_r1,np.cross(vec_r4,vec_r2))/\
         (r1*r4*r2+r1*np.dot(vec_r4,vec_r2)+r4*np.dot(vec_r2,vec_r1)+r2*np.dot(vec_r1,vec_r4))
    wC = 2*np.arctan(wC)

    wD = np.dot(vec_r2,np.cross(vec_r4,vec_r3))/\
         (r2*r4*r3+r2*np.dot(vec_r4,vec_r3)+r4*np.dot(vec_r3,vec_r2)+r3*np.dot(vec_r2,vec_r4))
    wD = 2*np.arctan(wD)

    #use maths.atan?

    #print('w->',wA,wB,wC,wD)

    #---step6---

    vec_rA = pt_midA-pt_M
    vec_rB = pt_midB-pt_M
    vec_rC = pt_midC-pt_M
    vec_rD = pt_midD-pt_M

    #print(vec_rA,vec_rB,vec_rC,vec_rD)

    #---step7---

    vec_gf=wA*np.dot(mat_FA,vec_rA) +\
           wB*np.dot(mat_FB,vec_rB) +\
           wC*np.dot(mat_FC,vec_rC) +\
           wD*np.dot(mat_FD,vec_rD) 
    vec_gf*=Ggrav*rho0

    print('vec_gf',vec_gf)

    #---step 8---

    vec_nA12 = compute_face_edge_normal(vec_nA,pt_1,pt_2,pt_midA,'faceA_n12')
    vec_nA23 = compute_face_edge_normal(vec_nA,pt_2,pt_3,pt_midA,'faceA_n23')
    vec_nA13 = compute_face_edge_normal(vec_nA,pt_3,pt_1,pt_midA,'faceA_n31')

    vec_nB13 = compute_face_edge_normal(vec_nB,pt_1,pt_3,pt_midB,'faceB_n13')
    vec_nB34 = compute_face_edge_normal(vec_nB,pt_3,pt_4,pt_midB,'faceB_n34')
    vec_nB14 = compute_face_edge_normal(vec_nB,pt_4,pt_1,pt_midB,'faceB_n41')

    vec_nC14 = compute_face_edge_normal(vec_nC,pt_1,pt_4,pt_midC,'faceC_n14')
    vec_nC24 = compute_face_edge_normal(vec_nC,pt_4,pt_2,pt_midC,'faceC_n42')
    vec_nC12 = compute_face_edge_normal(vec_nC,pt_2,pt_1,pt_midC,'faceC_n21')

    vec_nD24 = compute_face_edge_normal(vec_nD,pt_2,pt_4,pt_midD,'faceD_n24')
    vec_nD34 = compute_face_edge_normal(vec_nD,pt_4,pt_3,pt_midD,'faceD_n43')
    vec_nD23 = compute_face_edge_normal(vec_nD,pt_3,pt_2,pt_midD,'faceD_n32')

    #print(LA.norm(vec_nB14,2))

    #---step 9---

    mat_E12=np.outer(vec_nA,vec_nA12)+np.outer(vec_nC,vec_nC12) #edge 12(1) belongs to A & C
    mat_E13=np.outer(vec_nA,vec_nA13)+np.outer(vec_nB,vec_nB13) #edge 13(2) belongs to A & B
    mat_E14=np.outer(vec_nB,vec_nB14)+np.outer(vec_nC,vec_nC14) #edge 14(3) belongs to B & C
    mat_E23=np.outer(vec_nA,vec_nA23)+np.outer(vec_nD,vec_nD23) #edge 23(4) belongs to A & D
    mat_E24=np.outer(vec_nC,vec_nC24)+np.outer(vec_nD,vec_nD24) #edge 24(5) belongs to C & D
    mat_E34=np.outer(vec_nB,vec_nB34)+np.outer(vec_nD,vec_nD34) #edge 34(6) belongs to B & D

    #---step 10---

    L12=np.log((r1+r2+l1)/(r1+r2-l1))
    L13=np.log((r1+r3+l2)/(r1+r3-l2))
    L14=np.log((r1+r4+l3)/(r1+r4-l3))
    L23=np.log((r2+r3+l4)/(r2+r3-l4))
    L24=np.log((r2+r4+l5)/(r2+r4-l5))
    L34=np.log((r3+r4+l6)/(r3+r4-l6))

    #print(L12,L13,L14,L23,L24,L34)

    #---step 11---vector from M to any point on edge

    vec_r12 = pt_1-pt_M
    vec_r13 = pt_1-pt_M
    vec_r14 = pt_1-pt_M
    vec_r23 = pt_2-pt_M  # replace by mid-edge?
    vec_r24 = pt_2-pt_M
    vec_r34 = pt_3-pt_M

    #---step 12---

    vec_ge=L12*np.dot(mat_E12,vec_r12)+\
           L13*np.dot(mat_E13,vec_r13)+\
           L14*np.dot(mat_E14,vec_r14)+\
           L23*np.dot(mat_E23,vec_r23)+\
           L24*np.dot(mat_E24,vec_r24)+\
           L34*np.dot(mat_E34,vec_r34)
    vec_ge*=Ggrav*rho0

    print('vec_ge',vec_ge)

    #---step 13---

    vec_g = -vec_ge+vec_gf

    U_e=L12*np.dot(vec_r12,np.dot(mat_E12,vec_r12))+\
        L13*np.dot(vec_r13,np.dot(mat_E13,vec_r13))+\
        L14*np.dot(vec_r14,np.dot(mat_E14,vec_r14))+\
        L23*np.dot(vec_r23,np.dot(mat_E23,vec_r23))+\
        L24*np.dot(vec_r24,np.dot(mat_E24,vec_r24))+\
        L34*np.dot(vec_r34,np.dot(mat_E34,vec_r34))
    U_f=wA*np.dot(vec_rA,np.dot(mat_FA,vec_rA))+ \
        wB*np.dot(vec_rB,np.dot(mat_FB,vec_rB))+ \
        wC*np.dot(vec_rC,np.dot(mat_FC,vec_rC))+ \
        wD*np.dot(vec_rD,np.dot(mat_FD,vec_rD)) 
    print(U_e,U_f)
    U=U_e-U_f
    U*=0.5*Ggrav*rho0

    export_tetrahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,'tetra')

    return vec_g,U


#--------------------------------------------------------------------

rho0=1

# coordinates of point M (x,y,z)
pt_meas = np.array([0,0,10])

#vertices of a tetrahedron with edge length 2, centered at the origin
#https://en.wikipedia.org/wiki/Tetrahedron
#edges are sqrt(8/3) long
pt_one=np.array([0,0,1],dtype=np.float64)
pt_two=np.array([np.sqrt(8/9),0,-1/3],dtype=np.float64)
pt_three=np.array([-np.sqrt(2/9),np.sqrt(2/3),-1/3],dtype=np.float64)
pt_four=np.array([-np.sqrt(2/9),-np.sqrt(2/3),-1/3],dtype=np.float64)



#right angles
#pt_one=np.array([0,0,0])
#pt_two=np.array([1,0,0])
#pt_three=np.array([0,1,0])
#pt_four=np.array([0,0,1])

myfile=open('pt_M.ascii',"w")
myfile.write("%e %e %e\n" % (pt_meas[0],pt_meas[1],pt_meas[2]) )
myfile.close() 
myfile=open('pt_tetrahedron.ascii',"w")
myfile.write("%e %e %e\n" % (pt_one[0],pt_one[1],pt_one[2]) )
myfile.write("%e %e %e\n" % (pt_two[0],pt_two[1],pt_two[2]) )
myfile.write("%e %e %e\n" % (pt_three[0],pt_three[1],pt_three[2]) )
myfile.write("%e %e %e\n" % (pt_four[0],pt_four[1],pt_four[2]) )
myfile.close() 

#---------------------------------------------------------------------

vec_g,U = compute_gravity_tetrahedron(pt_one,pt_two,pt_three,pt_four,pt_meas,rho0)

print('vec_g   ',vec_g,'norm:',LA.norm(vec_g,2),'| U=',U)

export_vector_to_vtu(pt_meas,vec_g,'g')

#---------------------------------------------------------------------

vec_g_pm,U_pm = compute_gravity_tetrahedron_pointmass(pt_one,pt_two,pt_three,pt_four,pt_meas,rho0)

print('vec_g_pm',vec_g_pm,'norm:',LA.norm(vec_g_pm,2),'| U=',U_pm)
    
export_vector_to_vtu(pt_meas,vec_g_pm,'g_pm')

#--------------------------------------------------------------------


