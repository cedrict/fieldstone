import numpy as np
from numpy import linalg as LA
import time as time


#----------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------

def export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,name):
    vtufile=open(name+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8,1))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(pt_1[0],pt_1[1],pt_1[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_2[0],pt_2[1],pt_2[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_3[0],pt_3[1],pt_3[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_4[0],pt_4[1],pt_4[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_5[0],pt_5[1],pt_5[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_6[0],pt_6[1],pt_6[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_7[0],pt_7[1],pt_7[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_8[0],pt_8[1],pt_8[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(0,1,2,3,4,5,6,7))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    vtufile.write("%d \n" %(8))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    vtufile.write("%d \n" %12)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#----------------------------------------------------------------------------------------

def export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,name):
    vtufile=open(name+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8,12))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(pt_1[0],pt_1[1],pt_1[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_2[0],pt_2[1],pt_2[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_3[0],pt_3[1],pt_3[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_4[0],pt_4[1],pt_4[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_5[0],pt_5[1],pt_5[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_6[0],pt_6[1],pt_6[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_7[0],pt_7[1],pt_7[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_8[0],pt_8[1],pt_8[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    vtufile.write("%d %d %d \n" %( 4, 0, 5))
    vtufile.write("%d %d %d \n" %( 1, 5, 0))
    vtufile.write("%d %d %d \n" %( 5, 1, 6))
    vtufile.write("%d %d %d \n" %( 2, 6, 1))
    vtufile.write("%d %d %d \n" %( 2, 3, 6))
    vtufile.write("%d %d %d \n" %( 7, 6, 3))
    vtufile.write("%d %d %d \n" %( 7, 3, 4))
    vtufile.write("%d %d %d \n" %( 0, 4, 3))
    vtufile.write("%d %d %d \n" %( 6, 7, 5))
    vtufile.write("%d %d %d \n" %( 4, 5, 7))
    vtufile.write("%d %d %d \n" %( 3, 2, 0))
    vtufile.write("%d %d %d \n" %( 1, 0, 2))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,12):
        vtufile.write("%d \n" %((iel+1)*3))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,12):
        vtufile.write("%d \n" %5)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#------------------------------------------------------------------------------

def compute_face_edge_normal(vec_nface,pt_start,pt_end,pt_midface,name):
    #pt_start,pt_end are the vertices making the edge
    pt_mid=0.5*(pt_start+pt_end)
    vec_l=pt_end-pt_start
    #vec_n=np.empty(3)
    #det=vec_nface[yy]*vec_l[zz]-vec_nface[zz]*vec_l[yy]
    #vec_n[xx]=1
    #vec_n[yy]=(-vec_l[zz]*vec_nface[xx]+vec_nface[zz]*vec_l[xx] )/det
    #vec_n[zz]=(+vec_l[yy]*vec_nface[xx]-vec_nface[yy]*vec_l[xx] )/det
    vec_n=np.cross(vec_nface,vec_l)
    vec_n/=LA.norm(vec_n,2)
    #print('in',np.dot(vec_n,vec_nface))
    #print('in',np.dot(vec_n,vec_l))
    vec_n*=np.sign(np.dot(vec_n,pt_mid-pt_midface))
    #export_vector_to_vtu(pt_mid,vec_n,name)
    return vec_n

#------------------------------------------------------------------------------

def NNN(r,s,t):
    N0=0.125*(1.-r)*(1.-s)*(1.-t)
    N1=0.125*(1.+r)*(1.-s)*(1.-t)
    N2=0.125*(1.+r)*(1.+s)*(1.-t)
    N3=0.125*(1.-r)*(1.+s)*(1.-t)
    N4=0.125*(1.-r)*(1.-s)*(1.+t)
    N5=0.125*(1.+r)*(1.-s)*(1.+t)
    N6=0.125*(1.+r)*(1.+s)*(1.+t)
    N7=0.125*(1.-r)*(1.+s)*(1.+t)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

def dNNNdr(r,s,t):
    dNdr0=-0.125*(1.-s)*(1.-t)
    dNdr1=+0.125*(1.-s)*(1.-t)
    dNdr2=+0.125*(1.+s)*(1.-t)
    dNdr3=-0.125*(1.+s)*(1.-t)
    dNdr4=-0.125*(1.-s)*(1.+t)
    dNdr5=+0.125*(1.-s)*(1.+t)
    dNdr6=+0.125*(1.+s)*(1.+t)
    dNdr7=-0.125*(1.+s)*(1.+t)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)

def dNNNds(r,s,t):
    dNds0=-0.125*(1.-r)*(1.-t)
    dNds1=-0.125*(1.+r)*(1.-t)
    dNds2=+0.125*(1.+r)*(1.-t)
    dNds3=+0.125*(1.-r)*(1.-t)
    dNds4=-0.125*(1.-r)*(1.+t)
    dNds5=-0.125*(1.+r)*(1.+t)
    dNds6=+0.125*(1.+r)*(1.+t)
    dNds7=+0.125*(1.-r)*(1.+t)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)

def dNNNdt(r,s,t):
    dNdt0=-0.125*(1.-r)*(1.-s)
    dNdt1=-0.125*(1.+r)*(1.-s)
    dNdt2=-0.125*(1.+r)*(1.+s)
    dNdt3=-0.125*(1.-r)*(1.+s)
    dNdt4=+0.125*(1.-r)*(1.-s)
    dNdt5=+0.125*(1.+r)*(1.-s)
    dNdt6=+0.125*(1.+r)*(1.+s)
    dNdt7=+0.125*(1.-r)*(1.+s)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7],dtype=np.float64)

#------------------------------------------------------------------------------

def compute_gravity_hexahedron_faces(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0):
    # pt1,2,3,4,5,6,7,8,M: arrays containing x,y,z coordinates

    #---step1------------------------------------------------------------------
    # order 12 or 21 does not matter much:
    # these are used to compute the normal to faces.
    # and later only the norm is used.

    vec_l01=np.zeros(3,dtype=np.float64) ; vec_l01 = pt_5-pt_6 ; l01 = LA.norm(vec_l01,2) 
    vec_l02=np.zeros(3,dtype=np.float64) ; vec_l02 = pt_6-pt_7 ; l02 = LA.norm(vec_l02,2) 
    vec_l03=np.zeros(3,dtype=np.float64) ; vec_l03 = pt_7-pt_8 ; l03 = LA.norm(vec_l03,2)
    vec_l04=np.zeros(3,dtype=np.float64) ; vec_l04 = pt_8-pt_5 ; l04 = LA.norm(vec_l04,2) 
    vec_l05=np.zeros(3,dtype=np.float64) ; vec_l05 = pt_1-pt_2 ; l05 = LA.norm(vec_l05,2) 
    vec_l06=np.zeros(3,dtype=np.float64) ; vec_l06 = pt_2-pt_3 ; l06 = LA.norm(vec_l06,2)
    vec_l07=np.zeros(3,dtype=np.float64) ; vec_l07 = pt_3-pt_4 ; l07 = LA.norm(vec_l07,2) 
    vec_l08=np.zeros(3,dtype=np.float64) ; vec_l08 = pt_4-pt_1 ; l08 = LA.norm(vec_l08,2) 
    vec_l09=np.zeros(3,dtype=np.float64) ; vec_l09 = pt_1-pt_5 ; l09 = LA.norm(vec_l09,2)
    vec_l10=np.zeros(3,dtype=np.float64) ; vec_l10 = pt_2-pt_6 ; l10 = LA.norm(vec_l10,2) 
    vec_l11=np.zeros(3,dtype=np.float64) ; vec_l11 = pt_3-pt_7 ; l11 = LA.norm(vec_l11,2) 
    vec_l12=np.zeros(3,dtype=np.float64) ; vec_l12 = pt_4-pt_8 ; l12 = LA.norm(vec_l12,2)
    vec_l13=np.zeros(3,dtype=np.float64) ; vec_l13 = pt_1-pt_6 ; l13 = LA.norm(vec_l13,2) 
    vec_l14=np.zeros(3,dtype=np.float64) ; vec_l14 = pt_2-pt_7 ; l14 = LA.norm(vec_l14,2) 
    vec_l15=np.zeros(3,dtype=np.float64) ; vec_l15 = pt_4-pt_7 ; l15 = LA.norm(vec_l15,2)
    vec_l16=np.zeros(3,dtype=np.float64) ; vec_l16 = pt_4-pt_5 ; l16 = LA.norm(vec_l16,2) 
    vec_l17=np.zeros(3,dtype=np.float64) ; vec_l17 = pt_6-pt_8 ; l17 = LA.norm(vec_l17,2) 
    vec_l18=np.zeros(3,dtype=np.float64) ; vec_l18 = pt_1-pt_3 ; l18 = LA.norm(vec_l18,2)

    #---step2------------------------------------------------------------------

    vec_r01=np.zeros(3,dtype=np.float64) ; vec_r01 = pt_1-pt_M ; r01=LA.norm(vec_r01,2)
    vec_r02=np.zeros(3,dtype=np.float64) ; vec_r02 = pt_2-pt_M ; r02=LA.norm(vec_r02,2)
    vec_r03=np.zeros(3,dtype=np.float64) ; vec_r03 = pt_3-pt_M ; r03=LA.norm(vec_r03,2)
    vec_r04=np.zeros(3,dtype=np.float64) ; vec_r04 = pt_4-pt_M ; r04=LA.norm(vec_r04,2)
    vec_r05=np.zeros(3,dtype=np.float64) ; vec_r05 = pt_5-pt_M ; r05=LA.norm(vec_r05,2)
    vec_r06=np.zeros(3,dtype=np.float64) ; vec_r06 = pt_6-pt_M ; r06=LA.norm(vec_r06,2)
    vec_r07=np.zeros(3,dtype=np.float64) ; vec_r07 = pt_7-pt_M ; r07=LA.norm(vec_r07,2)
    vec_r08=np.zeros(3,dtype=np.float64) ; vec_r08 = pt_8-pt_M ; r08=LA.norm(vec_r08,2)

    #export_vector_to_vtu(pt_M,vec_r01,'vec_r01')
    #export_vector_to_vtu(pt_M,vec_r02,'vec_r02')
    #export_vector_to_vtu(pt_M,vec_r03,'vec_r03')
    #export_vector_to_vtu(pt_M,vec_r04,'vec_r04')
    #export_vector_to_vtu(pt_M,vec_r05,'vec_r05')
    #export_vector_to_vtu(pt_M,vec_r06,'vec_r06')
    #export_vector_to_vtu(pt_M,vec_r07,'vec_r07')
    #export_vector_to_vtu(pt_M,vec_r08,'vec_r08')

    #---step3---compute normal vectors to faces -------------------------------

    pt_midA=np.zeros(3,dtype=np.float64) ; pt_midA = (pt_5+pt_1+pt_6)/3
    pt_midB=np.zeros(3,dtype=np.float64) ; pt_midB = (pt_2+pt_6+pt_1)/3
    pt_midC=np.zeros(3,dtype=np.float64) ; pt_midC = (pt_6+pt_2+pt_7)/3 
    pt_midD=np.zeros(3,dtype=np.float64) ; pt_midD = (pt_3+pt_7+pt_2)/3
    pt_midE=np.zeros(3,dtype=np.float64) ; pt_midE = (pt_3+pt_4+pt_7)/3
    pt_midF=np.zeros(3,dtype=np.float64) ; pt_midF = (pt_8+pt_7+pt_4)/3
    pt_midG=np.zeros(3,dtype=np.float64) ; pt_midG = (pt_8+pt_4+pt_5)/3
    pt_midH=np.zeros(3,dtype=np.float64) ; pt_midH = (pt_1+pt_5+pt_4)/3
    pt_midI=np.zeros(3,dtype=np.float64) ; pt_midI = (pt_7+pt_8+pt_6)/3
    pt_midJ=np.zeros(3,dtype=np.float64) ; pt_midJ = (pt_5+pt_6+pt_8)/3
    pt_midK=np.zeros(3,dtype=np.float64) ; pt_midK = (pt_4+pt_3+pt_1)/3
    pt_midL=np.zeros(3,dtype=np.float64) ; pt_midL = (pt_2+pt_1+pt_3)/3

    vec_nA=np.zeros(3,dtype=np.float64) ; vec_nA = np.cross(-vec_l09, vec_l01) ; vec_nA/=LA.norm(vec_nA,2)
    vec_nB=np.zeros(3,dtype=np.float64) ; vec_nB = np.cross( vec_l10,-vec_l05) ; vec_nB/=LA.norm(vec_nB,2)
    vec_nC=np.zeros(3,dtype=np.float64) ; vec_nC = np.cross(-vec_l10, vec_l02) ; vec_nC/=LA.norm(vec_nC,2)
    vec_nD=np.zeros(3,dtype=np.float64) ; vec_nD = np.cross( vec_l11,-vec_l06) ; vec_nD/=LA.norm(vec_nD,2)
    vec_nE=np.zeros(3,dtype=np.float64) ; vec_nE = np.cross( vec_l07, vec_l11) ; vec_nE/=LA.norm(vec_nE,2)
    vec_nF=np.zeros(3,dtype=np.float64) ; vec_nF = np.cross(-vec_l03,-vec_l12) ; vec_nF/=LA.norm(vec_nF,2)
    vec_nG=np.zeros(3,dtype=np.float64) ; vec_nG = np.cross(-vec_l12, vec_l04) ; vec_nG/=LA.norm(vec_nG,2)
    vec_nH=np.zeros(3,dtype=np.float64) ; vec_nH = np.cross( vec_l09,-vec_l08) ; vec_nH/=LA.norm(vec_nH,2)
    vec_nI=np.zeros(3,dtype=np.float64) ; vec_nI = np.cross( vec_l03,-vec_l02) ; vec_nI/=LA.norm(vec_nI,2)
    vec_nJ=np.zeros(3,dtype=np.float64) ; vec_nJ = np.cross( vec_l01,-vec_l04) ; vec_nJ/=LA.norm(vec_nJ,2)
    vec_nK=np.zeros(3,dtype=np.float64) ; vec_nK = np.cross(-vec_l07, vec_l08) ; vec_nK/=LA.norm(vec_nK,2)
    vec_nL=np.zeros(3,dtype=np.float64) ; vec_nL = np.cross(-vec_l05, vec_l06) ; vec_nL/=LA.norm(vec_nL,2)

    #export_vector_to_vtu(pt_midA,vec_nA,'normal_faceA')
    #export_vector_to_vtu(pt_midB,vec_nB,'normal_faceB')
    #export_vector_to_vtu(pt_midC,vec_nC,'normal_faceC')
    #export_vector_to_vtu(pt_midD,vec_nD,'normal_faceD')
    #export_vector_to_vtu(pt_midE,vec_nE,'normal_faceE')
    #export_vector_to_vtu(pt_midF,vec_nF,'normal_faceF')
    #export_vector_to_vtu(pt_midG,vec_nG,'normal_faceG')
    #export_vector_to_vtu(pt_midH,vec_nH,'normal_faceH')
    #export_vector_to_vtu(pt_midI,vec_nI,'normal_faceI')
    #export_vector_to_vtu(pt_midJ,vec_nJ,'normal_faceJ')
    #export_vector_to_vtu(pt_midK,vec_nK,'normal_faceK')
    #export_vector_to_vtu(pt_midL,vec_nL,'normal_faceL')

    #all checked

    #---step4------------------------------------------------------------------

    mat_FaceA=np.zeros((3,3),dtype=np.float64) ; mat_FaceA=np.outer(vec_nA,vec_nA)
    mat_FaceB=np.zeros((3,3),dtype=np.float64) ; mat_FaceB=np.outer(vec_nB,vec_nB)
    mat_FaceC=np.zeros((3,3),dtype=np.float64) ; mat_FaceC=np.outer(vec_nC,vec_nC)
    mat_FaceD=np.zeros((3,3),dtype=np.float64) ; mat_FaceD=np.outer(vec_nD,vec_nD)
    mat_FaceE=np.zeros((3,3),dtype=np.float64) ; mat_FaceE=np.outer(vec_nE,vec_nE)
    mat_FaceF=np.zeros((3,3),dtype=np.float64) ; mat_FaceF=np.outer(vec_nF,vec_nF)
    mat_FaceG=np.zeros((3,3),dtype=np.float64) ; mat_FaceG=np.outer(vec_nG,vec_nG)
    mat_FaceH=np.zeros((3,3),dtype=np.float64) ; mat_FaceH=np.outer(vec_nH,vec_nH)
    mat_FaceI=np.zeros((3,3),dtype=np.float64) ; mat_FaceI=np.outer(vec_nI,vec_nI)
    mat_FaceJ=np.zeros((3,3),dtype=np.float64) ; mat_FaceJ=np.outer(vec_nJ,vec_nJ)
    mat_FaceK=np.zeros((3,3),dtype=np.float64) ; mat_FaceK=np.outer(vec_nK,vec_nK)
    mat_FaceL=np.zeros((3,3),dtype=np.float64) ; mat_FaceL=np.outer(vec_nL,vec_nL)

    #---step5---solid angles---------------------------------------------------

    # A: 516
    num=np.dot(vec_r05,np.cross(vec_r01,vec_r06))
    denom=(r05*r01*r06+r05*np.dot(vec_r01,vec_r06)
                      +r01*np.dot(vec_r06,vec_r05)
                      +r06*np.dot(vec_r05,vec_r01))
    wA=2*np.arctan2(num,denom)

    # B: 261
    num=np.dot(vec_r02,np.cross(vec_r06,vec_r01))
    denom=(r02*r06*r01+r02*np.dot(vec_r06,vec_r01)
                      +r06*np.dot(vec_r01,vec_r02)
                      +r01*np.dot(vec_r02,vec_r06))
    wB=2*np.arctan2(num,denom)

    # C: 627
    num=np.dot(vec_r06,np.cross(vec_r02,vec_r07))
    denom=(r06*r02*r07+r06*np.dot(vec_r02,vec_r07)
                      +r02*np.dot(vec_r07,vec_r06)
                      +r07*np.dot(vec_r06,vec_r02))
    wC=2*np.arctan2(num,denom)

    # D: 372
    num=np.dot(vec_r03,np.cross(vec_r07,vec_r02))
    denom=(r03*r07*r02+r03*np.dot(vec_r07,vec_r02)
                      +r07*np.dot(vec_r02,vec_r03)
                      +r02*np.dot(vec_r03,vec_r07))
    wD=2*np.arctan2(num,denom)

    # E: 347
    num=np.dot(vec_r03,np.cross(vec_r04,vec_r07))
    denom=(r03*r04*r07+r03*np.dot(vec_r04,vec_r07)
                      +r04*np.dot(vec_r07,vec_r03)
                      +r07*np.dot(vec_r03,vec_r04))
    wE=2*np.arctan2(num,denom)

    # F: 874
    num=np.dot(vec_r08,np.cross(vec_r07,vec_r04))
    denom=(r08*r07*r04+r08*np.dot(vec_r07,vec_r04)
                      +r07*np.dot(vec_r04,vec_r08)
                      +r04*np.dot(vec_r08,vec_r07))
    wF=2*np.arctan2(num,denom)

    # G: 845
    num=np.dot(vec_r08,np.cross(vec_r04,vec_r05))
    denom=(r08*r04*r05+r08*np.dot(vec_r04,vec_r05)
                      +r04*np.dot(vec_r05,vec_r08)
                      +r05*np.dot(vec_r08,vec_r04))
    wG=2*np.arctan2(num,denom)

    # H: 154
    num=np.dot(vec_r01,np.cross(vec_r05,vec_r04))
    denom=(r01*r05*r04+r01*np.dot(vec_r05,vec_r04)
                      +r05*np.dot(vec_r04,vec_r01)
                      +r04*np.dot(vec_r01,vec_r05))
    wH=2*np.arctan2(num,denom)

    # I: 786
    num=np.dot(vec_r07,np.cross(vec_r08,vec_r06))
    denom=(r07*r08*r06+r07*np.dot(vec_r08,vec_r06)
                      +r08*np.dot(vec_r06,vec_r07)
                      +r06*np.dot(vec_r07,vec_r08))
    wI=2*np.arctan2(num,denom)

    # J: 568
    num=np.dot(vec_r05,np.cross(vec_r06,vec_r08))
    denom=(r05*r06*r08+r05*np.dot(vec_r06,vec_r08)
                      +r06*np.dot(vec_r08,vec_r05)
                      +r08*np.dot(vec_r05,vec_r06))
    wJ=2*np.arctan2(num,denom)

    # K: 431
    num=np.dot(vec_r04,np.cross(vec_r03,vec_r01))
    denom=(r04*r03*r01+r04*np.dot(vec_r03,vec_r01)
                      +r03*np.dot(vec_r01,vec_r04)
                      +r01*np.dot(vec_r04,vec_r03))
    wK=2*np.arctan2(num,denom)

    # L: 213
    num=np.dot(vec_r02,np.cross(vec_r01,vec_r03))
    denom=(r02*r01*r03+r02*np.dot(vec_r01,vec_r03)
                      +r01*np.dot(vec_r03,vec_r02)
                      +r03*np.dot(vec_r02,vec_r01))
    wL=2*np.arctan2(num,denom)

    #---step6------------------------------------------------------------------

    vec_rA = pt_midA-pt_M  #export_vector_to_vtu(pt_M,vec_rA,'vec_rA')
    vec_rB = pt_midB-pt_M  #export_vector_to_vtu(pt_M,vec_rB,'vec_rB')
    vec_rC = pt_midC-pt_M  #export_vector_to_vtu(pt_M,vec_rC,'vec_rC')
    vec_rD = pt_midD-pt_M  #export_vector_to_vtu(pt_M,vec_rD,'vec_rD')
    vec_rE = pt_midE-pt_M  #export_vector_to_vtu(pt_M,vec_rE,'vec_rE')
    vec_rF = pt_midF-pt_M  #export_vector_to_vtu(pt_M,vec_rF,'vec_rF')
    vec_rG = pt_midG-pt_M  #export_vector_to_vtu(pt_M,vec_rG,'vec_rG')
    vec_rH = pt_midH-pt_M  #export_vector_to_vtu(pt_M,vec_rH,'vec_rH')
    vec_rI = pt_midI-pt_M  #export_vector_to_vtu(pt_M,vec_rI,'vec_rI')
    vec_rJ = pt_midJ-pt_M  #export_vector_to_vtu(pt_M,vec_rJ,'vec_rJ')
    vec_rK = pt_midK-pt_M  #export_vector_to_vtu(pt_M,vec_rK,'vec_rK')
    vec_rL = pt_midL-pt_M  #export_vector_to_vtu(pt_M,vec_rL,'vec_rL')

    #---step7------------------------------------------------------------------

    vec_gf=np.zeros(3,dtype=np.float64)
    vec_gf=wA*np.dot(mat_FaceA,vec_rA) +\
           wB*np.dot(mat_FaceB,vec_rB) +\
           wC*np.dot(mat_FaceC,vec_rC) +\
           wD*np.dot(mat_FaceD,vec_rD) +\
           wE*np.dot(mat_FaceE,vec_rE) +\
           wF*np.dot(mat_FaceF,vec_rF) +\
           wG*np.dot(mat_FaceG,vec_rG) +\
           wH*np.dot(mat_FaceH,vec_rH) +\
           wI*np.dot(mat_FaceI,vec_rI) +\
           wJ*np.dot(mat_FaceJ,vec_rJ) +\
           wK*np.dot(mat_FaceK,vec_rK) +\
           wL*np.dot(mat_FaceL,vec_rL) 
    vec_gf*=Ggrav*rho0

    #---step 8-----------------------------------------------------------------
    # note that in what follows the resulting vectors are perpendicular to an 
    # edge and a face, so that for example vec_nA51 would be the same as vec_na15.
    # I just used a numbering for each set of three vectors following the nodes
    # of a given face. 

    # A: 516
    vec_nA51=np.zeros(3,dtype=np.float64) ; vec_nA51=compute_face_edge_normal(vec_nA,pt_5,pt_1,pt_midA,'faceA_n51')
    vec_nA16=np.zeros(3,dtype=np.float64) ; vec_nA16=compute_face_edge_normal(vec_nA,pt_1,pt_6,pt_midA,'faceA_n16')
    vec_nA65=np.zeros(3,dtype=np.float64) ; vec_nA65=compute_face_edge_normal(vec_nA,pt_6,pt_5,pt_midA,'faceA_n65')

    # B: 261
    vec_nB26=np.zeros(3,dtype=np.float64) ; vec_nB26=compute_face_edge_normal(vec_nB,pt_2,pt_6,pt_midB,'faceB_n26')
    vec_nB61=np.zeros(3,dtype=np.float64) ; vec_nB61=compute_face_edge_normal(vec_nB,pt_6,pt_1,pt_midB,'faceB_n61')
    vec_nB12=np.zeros(3,dtype=np.float64) ; vec_nB12=compute_face_edge_normal(vec_nB,pt_1,pt_2,pt_midB,'faceB_n12')

    # C: 627
    vec_nC62=np.zeros(3,dtype=np.float64) ; vec_nC62=compute_face_edge_normal(vec_nC,pt_6,pt_2,pt_midC,'faceC_n62')
    vec_nC27=np.zeros(3,dtype=np.float64) ; vec_nC27=compute_face_edge_normal(vec_nC,pt_2,pt_7,pt_midC,'faceC_n27')
    vec_nC76=np.zeros(3,dtype=np.float64) ; vec_nC76=compute_face_edge_normal(vec_nC,pt_7,pt_6,pt_midC,'faceC_n76')

    # D: 372
    vec_nD37=np.zeros(3,dtype=np.float64) ; vec_nD37=compute_face_edge_normal(vec_nD,pt_3,pt_7,pt_midD,'faceD_n37')
    vec_nD72=np.zeros(3,dtype=np.float64) ; vec_nD72=compute_face_edge_normal(vec_nD,pt_7,pt_2,pt_midD,'faceD_n72')
    vec_nD23=np.zeros(3,dtype=np.float64) ; vec_nD23=compute_face_edge_normal(vec_nD,pt_2,pt_3,pt_midD,'faceD_n23')

    # E: 347
    vec_nE34=np.zeros(3,dtype=np.float64) ; vec_nE34=compute_face_edge_normal(vec_nE,pt_3,pt_4,pt_midE,'faceE_n34')
    vec_nE47=np.zeros(3,dtype=np.float64) ; vec_nE47=compute_face_edge_normal(vec_nE,pt_4,pt_7,pt_midE,'faceE_n47')
    vec_nE73=np.zeros(3,dtype=np.float64) ; vec_nE73=compute_face_edge_normal(vec_nE,pt_7,pt_3,pt_midE,'faceE_n73')

    # F: 874
    vec_nF87=np.zeros(3,dtype=np.float64) ; vec_nF87=compute_face_edge_normal(vec_nF,pt_8,pt_7,pt_midF,'faceF_n87')
    vec_nF74=np.zeros(3,dtype=np.float64) ; vec_nF74=compute_face_edge_normal(vec_nF,pt_7,pt_4,pt_midF,'faceF_n74')
    vec_nF48=np.zeros(3,dtype=np.float64) ; vec_nF48=compute_face_edge_normal(vec_nF,pt_4,pt_8,pt_midF,'faceF_n48')

    # G: 845
    vec_nG84=np.zeros(3,dtype=np.float64) ; vec_nG84=compute_face_edge_normal(vec_nG,pt_8,pt_4,pt_midG,'faceG_n84')
    vec_nG45=np.zeros(3,dtype=np.float64) ; vec_nG45=compute_face_edge_normal(vec_nG,pt_4,pt_5,pt_midG,'faceG_n45')
    vec_nG58=np.zeros(3,dtype=np.float64) ; vec_nG58=compute_face_edge_normal(vec_nG,pt_5,pt_8,pt_midG,'faceG_n58')

    # H: 154
    vec_nH15=np.zeros(3,dtype=np.float64) ; vec_nH15=compute_face_edge_normal(vec_nH,pt_1,pt_5,pt_midH,'faceH_n15')
    vec_nH54=np.zeros(3,dtype=np.float64) ; vec_nH54=compute_face_edge_normal(vec_nH,pt_5,pt_4,pt_midH,'faceH_n54')
    vec_nH41=np.zeros(3,dtype=np.float64) ; vec_nH41=compute_face_edge_normal(vec_nH,pt_4,pt_1,pt_midH,'faceH_n41')

    # I: 786
    vec_nI78=np.zeros(3,dtype=np.float64) ; vec_nI78=compute_face_edge_normal(vec_nI,pt_7,pt_8,pt_midI,'faceI_n78')
    vec_nI86=np.zeros(3,dtype=np.float64) ; vec_nI86=compute_face_edge_normal(vec_nI,pt_8,pt_6,pt_midI,'faceI_n86')
    vec_nI67=np.zeros(3,dtype=np.float64) ; vec_nI67=compute_face_edge_normal(vec_nI,pt_6,pt_7,pt_midI,'faceI_n67')

    # J: 568
    vec_nJ56=np.zeros(3,dtype=np.float64) ; vec_nJ56=compute_face_edge_normal(vec_nJ,pt_5,pt_6,pt_midJ,'faceJ_n56')
    vec_nJ68=np.zeros(3,dtype=np.float64) ; vec_nJ68=compute_face_edge_normal(vec_nJ,pt_6,pt_8,pt_midJ,'faceJ_n68')
    vec_nJ85=np.zeros(3,dtype=np.float64) ; vec_nJ85=compute_face_edge_normal(vec_nJ,pt_8,pt_5,pt_midJ,'faceJ_n85')

    # K: 431
    vec_nK43=np.zeros(3,dtype=np.float64) ; vec_nK43=compute_face_edge_normal(vec_nK,pt_4,pt_3,pt_midK,'faceK_n43')
    vec_nK31=np.zeros(3,dtype=np.float64) ; vec_nK31=compute_face_edge_normal(vec_nK,pt_3,pt_1,pt_midK,'faceK_n31')
    vec_nK14=np.zeros(3,dtype=np.float64) ; vec_nK14=compute_face_edge_normal(vec_nK,pt_1,pt_4,pt_midK,'faceK_n14')

    # L: 213
    vec_nL21=np.zeros(3,dtype=np.float64) ; vec_nL21=compute_face_edge_normal(vec_nL,pt_2,pt_1,pt_midL,'faceL_n21')
    vec_nL13=np.zeros(3,dtype=np.float64) ; vec_nL13=compute_face_edge_normal(vec_nL,pt_1,pt_3,pt_midL,'faceL_n13')
    vec_nL32=np.zeros(3,dtype=np.float64) ; vec_nL32=compute_face_edge_normal(vec_nL,pt_3,pt_2,pt_midL,'faceL_n32')

    # checked these ok 

    #---step 9-----------------------------------------------------------------
    # one can check these matrices are indeed symmetric as 
    # explained in section 2.1.8

    #mat_E12=np.zeros((3,3),dtype=np.float64)
    #mat_E13=np.zeros((3,3),dtype=np.float64)
    #mat_E14=np.zeros((3,3),dtype=np.float64)
    #mat_E23=np.zeros((3,3),dtype=np.float64)
    #mat_E24=np.zeros((3,3),dtype=np.float64)
    #mat_E34=np.zeros((3,3),dtype=np.float64)

    mat_E01=np.outer(vec_nJ,vec_nJ56)+np.outer(vec_nA,vec_nA65) #edge 5-6 ( 1) belongs to J & A
    mat_E02=np.outer(vec_nI,vec_nI67)+np.outer(vec_nC,vec_nC76) #edge 6-7 ( 2) belongs to I & C
    mat_E03=np.outer(vec_nI,vec_nI78)+np.outer(vec_nF,vec_nF87) #edge 7-8 ( 3) belongs to I & F
    mat_E04=np.outer(vec_nJ,vec_nJ85)+np.outer(vec_nG,vec_nG58) #edge 8-5 ( 4) belongs to J & G
    mat_E05=np.outer(vec_nB,vec_nB12)+np.outer(vec_nL,vec_nL21) #edge 1-2 ( 5) belongs to B & L
    mat_E06=np.outer(vec_nD,vec_nD23)+np.outer(vec_nL,vec_nL32) #edge 2-3 ( 6) belongs to D & L
    mat_E07=np.outer(vec_nE,vec_nE34)+np.outer(vec_nK,vec_nK43) #edge 3-4 ( 7) belongs to E & K 
    mat_E08=np.outer(vec_nH,vec_nH41)+np.outer(vec_nK,vec_nK14) #edge 4-1 ( 8) belongs to H & K
    mat_E09=np.outer(vec_nH,vec_nH15)+np.outer(vec_nA,vec_nA51) #edge 1-5 ( 9) belongs to H & A
    mat_E10=np.outer(vec_nB,vec_nB26)+np.outer(vec_nC,vec_nC62) #edge 2-6 (10) belongs to B & C
    mat_E11=np.outer(vec_nD,vec_nD37)+np.outer(vec_nE,vec_nE73) #edge 3-7 (11) belongs to D & E
    mat_E12=np.outer(vec_nF,vec_nF48)+np.outer(vec_nG,vec_nG84) #edge 4-8 (12) belongs to F & G
    mat_E13=np.outer(vec_nA,vec_nA16)+np.outer(vec_nB,vec_nB61) #edge 1-6 (13) belongs to A & B
    mat_E14=np.outer(vec_nC,vec_nC27)+np.outer(vec_nD,vec_nD72) #edge 2-7 (14) belongs to C & D
    mat_E15=np.outer(vec_nE,vec_nE47)+np.outer(vec_nF,vec_nF74) #edge 4-7 (15) belongs to E & F
    mat_E16=np.outer(vec_nG,vec_nG45)+np.outer(vec_nH,vec_nH54) #edge 4-5 (16) belongs to G & H
    mat_E17=np.outer(vec_nJ,vec_nJ68)+np.outer(vec_nI,vec_nI86) #edge 6-8 (17) belongs to J & I
    mat_E18=np.outer(vec_nL,vec_nL13)+np.outer(vec_nK,vec_nK31) #edge 1-3 (18) belongs to L & K

    #---step 10---potential of each edge---------------------------------------

    L01=np.log((r05+r06+l01)/(r05+r06-l01)) #56
    L02=np.log((r06+r07+l02)/(r06+r07-l02)) #67
    L03=np.log((r07+r08+l03)/(r07+r08-l03)) #78
    L04=np.log((r08+r05+l04)/(r08+r05-l04)) #85
    L05=np.log((r01+r02+l05)/(r01+r02-l05)) #12
    L06=np.log((r02+r03+l06)/(r02+r03-l06)) #23
    L07=np.log((r03+r04+l07)/(r03+r04-l07)) #34
    L08=np.log((r04+r01+l08)/(r04+r01-l08)) #41
    L09=np.log((r01+r05+l09)/(r01+r05-l09)) #15
    L10=np.log((r02+r06+l10)/(r02+r06-l10)) #26
    L11=np.log((r03+r07+l11)/(r03+r07-l11)) #37
    L12=np.log((r04+r08+l12)/(r04+r08-l12)) #48
    L13=np.log((r01+r06+l13)/(r01+r06-l13)) #16
    L14=np.log((r02+r07+l14)/(r02+r07-l14)) #27
    L15=np.log((r04+r07+l15)/(r04+r07-l15)) #47
    L16=np.log((r04+r05+l16)/(r04+r05-l16)) #45
    L17=np.log((r06+r08+l17)/(r06+r08-l17)) #68
    L18=np.log((r01+r03+l18)/(r01+r03-l18)) #13

    #if np.abs(r1+r2-l1)/l1<1e-10: L12=0
    #if np.abs(r1+r3-l2)/l2<1e-10: L13=0
    #if np.abs(r1+r4-l3)/l3<1e-10: L14=0
    #if np.abs(r2+r3-l4)/l4<1e-10: L23=0
    #if np.abs(r2+r4-l5)/l5<1e-10: L24=0
    #if np.abs(r3+r4-l6)/l6<1e-10: L34=0

    #---step 11---vector from M to any point on edge---------------------------
    # we choose the middle of the edge

    vec_r01=np.zeros(3,dtype=np.float64) ; vec_r01 = 0.5*(pt_5+pt_6)-pt_M #56 
    vec_r02=np.zeros(3,dtype=np.float64) ; vec_r02 = 0.5*(pt_6+pt_7)-pt_M #67
    vec_r03=np.zeros(3,dtype=np.float64) ; vec_r03 = 0.5*(pt_7+pt_8)-pt_M #78
    vec_r04=np.zeros(3,dtype=np.float64) ; vec_r04 = 0.5*(pt_8+pt_5)-pt_M #85
    vec_r05=np.zeros(3,dtype=np.float64) ; vec_r05 = 0.5*(pt_1+pt_2)-pt_M #12
    vec_r06=np.zeros(3,dtype=np.float64) ; vec_r06 = 0.5*(pt_2+pt_3)-pt_M #23
    vec_r07=np.zeros(3,dtype=np.float64) ; vec_r07 = 0.5*(pt_3+pt_4)-pt_M #34
    vec_r08=np.zeros(3,dtype=np.float64) ; vec_r08 = 0.5*(pt_4+pt_1)-pt_M #41
    vec_r09=np.zeros(3,dtype=np.float64) ; vec_r09 = 0.5*(pt_1+pt_5)-pt_M #15
    vec_r10=np.zeros(3,dtype=np.float64) ; vec_r10 = 0.5*(pt_2+pt_6)-pt_M #26
    vec_r11=np.zeros(3,dtype=np.float64) ; vec_r11 = 0.5*(pt_3+pt_7)-pt_M #37
    vec_r12=np.zeros(3,dtype=np.float64) ; vec_r12 = 0.5*(pt_4+pt_8)-pt_M #48
    vec_r13=np.zeros(3,dtype=np.float64) ; vec_r13 = 0.5*(pt_1+pt_6)-pt_M #16
    vec_r14=np.zeros(3,dtype=np.float64) ; vec_r14 = 0.5*(pt_2+pt_7)-pt_M #27
    vec_r15=np.zeros(3,dtype=np.float64) ; vec_r15 = 0.5*(pt_4+pt_7)-pt_M #47
    vec_r16=np.zeros(3,dtype=np.float64) ; vec_r16 = 0.5*(pt_4+pt_5)-pt_M #45
    vec_r17=np.zeros(3,dtype=np.float64) ; vec_r17 = 0.5*(pt_6+pt_8)-pt_M #68
    vec_r18=np.zeros(3,dtype=np.float64) ; vec_r18 = 0.5*(pt_1+pt_3)-pt_M #13

    #export_vector_to_vtu(pt_M,vec_r01,'vec_r01')
    #export_vector_to_vtu(pt_M,vec_r02,'vec_r02')
    #export_vector_to_vtu(pt_M,vec_r03,'vec_r03')
    #export_vector_to_vtu(pt_M,vec_r04,'vec_r04')
    #export_vector_to_vtu(pt_M,vec_r05,'vec_r05')
    #export_vector_to_vtu(pt_M,vec_r06,'vec_r06')

    #---step 12----------------------------------------------------------------

    vec_ge=np.zeros(3,dtype=np.float64)
    vec_ge=L01*np.dot(mat_E01,vec_r01)+\
           L02*np.dot(mat_E02,vec_r02)+\
           L03*np.dot(mat_E03,vec_r03)+\
           L04*np.dot(mat_E04,vec_r04)+\
           L05*np.dot(mat_E05,vec_r05)+\
           L06*np.dot(mat_E06,vec_r06)+\
           L07*np.dot(mat_E07,vec_r07)+\
           L08*np.dot(mat_E08,vec_r08)+\
           L09*np.dot(mat_E09,vec_r09)+\
           L10*np.dot(mat_E10,vec_r10)+\
           L11*np.dot(mat_E11,vec_r11)+\
           L12*np.dot(mat_E12,vec_r12)+\
           L13*np.dot(mat_E13,vec_r13)+\
           L14*np.dot(mat_E14,vec_r14)+\
           L15*np.dot(mat_E15,vec_r15)+\
           L16*np.dot(mat_E16,vec_r16)+\
           L17*np.dot(mat_E17,vec_r17)+\
           L18*np.dot(mat_E18,vec_r18)
    vec_ge*=Ggrav*rho0

    #---step 13----------------------------------------------------------------

    U_e=L01*np.dot(vec_r01,np.dot(mat_E01,vec_r01))+\
        L02*np.dot(vec_r02,np.dot(mat_E02,vec_r02))+\
        L03*np.dot(vec_r03,np.dot(mat_E03,vec_r03))+\
        L04*np.dot(vec_r04,np.dot(mat_E04,vec_r04))+\
        L05*np.dot(vec_r05,np.dot(mat_E05,vec_r05))+\
        L06*np.dot(vec_r06,np.dot(mat_E06,vec_r06))+\
        L07*np.dot(vec_r07,np.dot(mat_E07,vec_r07))+\
        L08*np.dot(vec_r08,np.dot(mat_E08,vec_r08))+\
        L09*np.dot(vec_r09,np.dot(mat_E09,vec_r09))+\
        L10*np.dot(vec_r10,np.dot(mat_E10,vec_r10))+\
        L11*np.dot(vec_r11,np.dot(mat_E11,vec_r11))+\
        L12*np.dot(vec_r12,np.dot(mat_E12,vec_r12))+\
        L13*np.dot(vec_r13,np.dot(mat_E13,vec_r13))+\
        L14*np.dot(vec_r14,np.dot(mat_E14,vec_r14))+\
        L15*np.dot(vec_r15,np.dot(mat_E15,vec_r15))+\
        L16*np.dot(vec_r16,np.dot(mat_E16,vec_r16))+\
        L17*np.dot(vec_r17,np.dot(mat_E17,vec_r17))+\
        L18*np.dot(vec_r18,np.dot(mat_E18,vec_r18))
    U_e*=0.5*Ggrav*rho0

    U_f=wA*np.dot(vec_rA,np.dot(mat_FaceA,vec_rA))+ \
        wB*np.dot(vec_rB,np.dot(mat_FaceB,vec_rB))+ \
        wC*np.dot(vec_rC,np.dot(mat_FaceC,vec_rC))+ \
        wD*np.dot(vec_rD,np.dot(mat_FaceD,vec_rD))+ \
        wE*np.dot(vec_rE,np.dot(mat_FaceE,vec_rE))+ \
        wF*np.dot(vec_rF,np.dot(mat_FaceF,vec_rF))+ \
        wG*np.dot(vec_rG,np.dot(mat_FaceG,vec_rG))+ \
        wH*np.dot(vec_rH,np.dot(mat_FaceH,vec_rH))+ \
        wI*np.dot(vec_rI,np.dot(mat_FaceI,vec_rI))+ \
        wJ*np.dot(vec_rJ,np.dot(mat_FaceJ,vec_rJ))+ \
        wK*np.dot(vec_rK,np.dot(mat_FaceK,vec_rK))+ \
        wL*np.dot(vec_rL,np.dot(mat_FaceL,vec_rL))
    U_f*=0.5*Ggrav*rho0

    U=U_e-U_f

    vec_g =-vec_ge+vec_gf

    #print('vec_ge',vec_ge[xx],vec_ge[yy],vec_ge[zz])
    #print('vec_gf',vec_gf[xx],vec_gf[yy],vec_gf[zz])
    #print('U_e',U_e)
    #print('U_f',U_f)
    #print('U  ',U)

    return vec_g,U

#----------------------------------------------------------------------------------------

def compute_gravity_hexahedron_mascons(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,nm_per_dim):

    nmascons=nm_per_dim**3

    x=np.array([pt_1[0],pt_2[0],pt_3[0],pt_4[0],
                pt_5[0],pt_6[0],pt_7[0],pt_8[0]],dtype=np.float64)

    y=np.array([pt_1[1],pt_2[1],pt_3[1],pt_4[1],
                pt_5[1],pt_6[1],pt_7[1],pt_8[1]],dtype=np.float64)

    z=np.array([pt_1[2],pt_2[2],pt_3[2],pt_4[2],
                pt_5[2],pt_6[2],pt_7[2],pt_8[2]],dtype=np.float64)


    #--------------------------------------------------------------------------
    # generate r,s,t coordinates for mascons
    # and compute their real coordinates inside the hex
    #--------------------------------------------------------------------------

    xm = np.zeros(nmascons,dtype=np.float64)  # x coordinates
    ym = np.zeros(nmascons,dtype=np.float64)  # y coordinates
    zm = np.zeros(nmascons,dtype=np.float64)  # z coordinates

    counter=0
    for i in range(0,nm_per_dim):
        for j in range(0,nm_per_dim):
            for k in range(0,nm_per_dim):
                r=-1+1/nm_per_dim+i*2/nm_per_dim
                s=-1+1/nm_per_dim+j*2/nm_per_dim
                t=-1+1/nm_per_dim+k*2/nm_per_dim
                #print(r,s,t)
                N=NNN(r,s,t)
                xm[counter]=N.dot(x)
                ym[counter]=N.dot(y)
                zm[counter]=N.dot(z)
                counter+=1    
            #end for
        #end for
    #end for

    #np.savetxt('mascons.ascii',np.array([xm,ym,zm]).T)

    #--------------------------------------------------------------------------
    # compute volume of hexahedron
    #--------------------------------------------------------------------------

    sqrt3=np.sqrt(3.)
    jcb= np.zeros((3,3),dtype=np.float64)

    volume=0
    for iq in [-1,1]:
        for jq in [-1,1]:
            for kq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1.*1.*1.
                N=NNN(rq,sq,tq)
                dNdr=dNNNdr(rq,sq,tq)
                dNds=dNNNds(rq,sq,tq)
                dNdt=dNNNdt(rq,sq,tq)
                jcb[0,0]=dNdr.dot(x) ; jcb[0,1]=dNdr.dot(y) ; jcb[0,2]=dNdr.dot(z)
                jcb[1,0]=dNds.dot(x) ; jcb[1,1]=dNds.dot(y) ; jcb[1,2]=dNds.dot(z)
                jcb[2,0]=dNdt.dot(x) ; jcb[2,1]=dNdt.dot(y) ; jcb[2,2]=dNdt.dot(z)
                jcob = np.linalg.det(jcb)
                volume+=1*jcob*weightq
            #end for
        #end for
    #end for

    #print(volume)

    #------------------------------------------------------------------------------

    mascon=volume*rho0/nmascons

    grav=np.zeros(3,dtype=np.float64)
    for i in range(0,nmascons):
        dist=np.sqrt((xm[i]-pt_M[0])**2+(ym[i]-pt_M[1])**2+(zm[i]-pt_M[2])**2)
        grav[0]+=mascon*Ggrav/dist**3*(xm[i]-pt_M[0])
        grav[1]+=mascon*Ggrav/dist**3*(ym[i]-pt_M[1])
        grav[2]+=mascon*Ggrav/dist**3*(zm[i]-pt_M[2])

    U=0

    return grav,U

#----------------------------------------------------------------------------------------

def compute_gravity_hexahedron_quadrature(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,nq_per_dim):
    # pt1,2,3,4,5,6,7,8,M: arrays containing x,y,z coordinates

    x=np.array([pt_1[0],pt_2[0],pt_3[0],pt_4[0],
                pt_5[0],pt_6[0],pt_7[0],pt_8[0]],dtype=np.float64)

    y=np.array([pt_1[1],pt_2[1],pt_3[1],pt_4[1],
                pt_5[1],pt_6[1],pt_7[1],pt_8[1]],dtype=np.float64)

    z=np.array([pt_1[2],pt_2[2],pt_3[2],pt_4[2],
                pt_5[2],pt_6[2],pt_7[2],pt_8[2]],dtype=np.float64)

    grav=np.zeros(3,dtype=np.float64)

    if nq_per_dim==1:
       qcoords=[0.]
       qweights=[2.]
    elif nq_per_dim==2:
       qcoords=[-np.sqrt(1./3.),np.sqrt(1./3.)]
       qweights=[1.,1.]
    elif nq_per_dim==3:
       qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
       qweights=[5./9.,8./9.,5./9.]
    elif nq_per_dim==4:
       qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
       qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
       qw4a=(18-np.sqrt(30.))/36.
       qw4b=(18+np.sqrt(30.))/36.
       qcoords=[-qc4a,-qc4b,qc4b,qc4a]
       qweights=[qw4a,qw4b,qw4b,qw4a]
    elif nq_per_dim==5:
       qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
       qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
       qc5c=0.
       qw5a=(322.-13.*np.sqrt(70.))/900.
       qw5b=(322.+13.*np.sqrt(70.))/900.
       qw5c=128./225.
       qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
       qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]
    elif nq_per_dim==6:
       qcoords=[-0.932469514203152,\
                -0.661209386466265,\
                -0.238619186083197,\
                +0.238619186083197,\
                +0.661209386466265,\
                +0.932469514203152]
       qweights=[0.171324492379170,\
                 0.360761573048139,\
                 0.467913934572691,\
                 0.467913934572691,\
                 0.360761573048139,\
                 0.171324492379170]
    elif nq_per_dim==7:
       qcoords=[-0.949107912342759,\
                -0.741531185599394,\
                -0.405845151377397,\
                0.,\
                +0.405845151377397,\
                +0.741531185599394,\
                +0.949107912342759]
       qweights=[0.129484966168870,\
                 0.279705391489277,\
                 0.381830050505119,\
                 0.417959183673469,\
                 0.381830050505119,\
                 0.279705391489277,\
                 0.129484966168870]
    elif nq_per_dim==8:
       qcoords=[-0.960289856497536,\
                -0.796666477413627,\
                -0.525532409916329,\
                -0.183434642495650,\
                +0.183434642495650,\
                +0.525532409916329,\
                +0.796666477413627,\
                +0.960289856497536]
       qweights=[0.101228536290376,\
                 0.222381034453374,\
                 0.313706645877887,\
                 0.362683783378362,\
                 0.362683783378362,\
                 0.313706645877887,\
                 0.222381034453374,\
                 0.101228536290376]
    else:
       return grav,0

    jcb=np.zeros((3,3),dtype=np.float64)
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]
                N=NNN(rq,sq,tq)
                dNdr=dNNNdr(rq,sq,tq)
                dNds=dNNNds(rq,sq,tq)
                dNdt=dNNNdt(rq,sq,tq)
                jcb[0,0]=dNdr.dot(x) ; jcb[0,1]=dNdr.dot(y) ; jcb[0,2]=dNdr.dot(z)
                jcb[1,0]=dNds.dot(x) ; jcb[1,1]=dNds.dot(y) ; jcb[1,2]=dNds.dot(z)
                jcb[2,0]=dNdt.dot(x) ; jcb[2,1]=dNdt.dot(y) ; jcb[2,2]=dNdt.dot(z)
                jcob = np.linalg.det(jcb)

                xq=N.dot(x)
                yq=N.dot(y)
                zq=N.dot(z)

                dist=np.sqrt((xq-pt_M[0])**2+(yq-pt_M[1])**2+(zq-pt_M[2])**2)
                grav[0]+=rho0*Ggrav/dist**3*(xq-pt_M[0])*weightq*jcob
                grav[1]+=rho0*Ggrav/dist**3*(yq-pt_M[1])*weightq*jcob
                grav[2]+=rho0*Ggrav/dist**3*(zq-pt_M[2])*weightq*jcob

            #end for
        #end for
    #end for


    U=0
    return grav,U



#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

test=2


if test==1:

   Ggrav=1
    
   mascons_file=open('mascons.ascii',"w")
   faces_file=open('faces.ascii',"w")
   quadrature_file=open('quadrature.ascii',"w")

   pt_1=np.array([0,0,0],dtype=np.float64)
   pt_2=np.array([1,0,0],dtype=np.float64)
   pt_3=np.array([1,1,0],dtype=np.float64)
   pt_4=np.array([0,1,0],dtype=np.float64)
   pt_5=np.array([0,0,1],dtype=np.float64)
   pt_6=np.array([1,0,1],dtype=np.float64)
   pt_7=np.array([1,1,1],dtype=np.float64)
   pt_8=np.array([0,1,1],dtype=np.float64)

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces')

   rho0=1
   pt_M=np.array([2,0.5,0.5],dtype=np.float64)

   for n_per_dim in range(1,48):
       
       start = time.time()
       g,U=compute_gravity_hexahedron_mascons(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,n_per_dim)
       mascons_file.write("%d %e %e %e %e %e \n" %(n_per_dim,g[0],g[1],g[2],U,time.time() - start))

       start = time.time()
       g,U=compute_gravity_hexahedron_faces(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
       faces_file.write("%d %e %e %e %e %e \n" %(n_per_dim,g[0],g[1],g[2],U,time.time() - start))

       start = time.time()
       g,U=compute_gravity_hexahedron_quadrature(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,n_per_dim)
       if max(abs(g))>1e-10:
          quadrature_file.write("%d %e %e %e %e %e \n" %(n_per_dim,g[0],g[1],g[2],U,time.time() - start))
 
#----------------------------------------------------------------------------------------

if test==2:

   Ggrav=6.67e-11
    
   mascons_file=open('mascons.ascii',"w")
   faces_file=open('faces.ascii',"w")
   quadrature_file=open('quadrature.ascii',"w")

   pt_1=np.array([0  ,-0.1,0.2],dtype=np.float64)
   pt_2=np.array([0.2,-0.1,0.2],dtype=np.float64)
   pt_3=np.array([0.2, 0.1,0.2],dtype=np.float64)
   pt_4=np.array([0  , 0.1,0.2],dtype=np.float64)
   pt_5=np.array([-0.2,-0.1,0.9],dtype=np.float64)
   pt_6=np.array([0   ,-0.1,0.9],dtype=np.float64)
   pt_7=np.array([0   , 0.1,0.9],dtype=np.float64)
   pt_8=np.array([-0.2, 0.1,0.9],dtype=np.float64)

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces')

   rho0=-170
   pt_M=np.array([0,0,2],dtype=np.float64)

   g,U=compute_gravity_hexahedron_mascons(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
   print(g)
   g,U=compute_gravity_hexahedron_faces(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
   print(g)
   g,U=compute_gravity_hexahedron_quadrature(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,5)
   print(g)



   Lx=2
   Ly=2
   nnx=21 ; nelx=nnx-1
   nny=21 ; nely=nny-1
   NV=nnx*nny
   x=np.empty(NV,dtype=np.float64)
   y=np.empty(NV,dtype=np.float64)
   z=np.empty(NV,dtype=np.float64)
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           x[counter]=i*Lx/float(nnx-1)-Lx/2
           y[counter]=j*Ly/float(nny-1)-Ly/2
           counter += 1

   z[:]=10

   icon=np.zeros((4,nelx*nely),dtype=np.int32)
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon[0, counter] = i + j * (nelx + 1)
           icon[1, counter] = i + 1 + j * (nelx + 1)
           icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
           icon[3, counter] = i + (j + 1) * (nelx + 1)
           counter += 1

   grid_gx=np.empty(NV,dtype=np.float64)
   grid_gy=np.empty(NV,dtype=np.float64)
   grid_gz=np.empty(NV,dtype=np.float64)

   for i in range(0,NV):
       pt_M[0]=x[i]
       pt_M[1]=y[i]
       pt_M[2]=z[i]
       g,U=compute_gravity_hexahedron_mascons(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
       grid_gx[i]=g[0]

   #expor t vtu









