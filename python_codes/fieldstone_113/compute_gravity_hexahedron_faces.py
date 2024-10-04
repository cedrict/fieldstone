import numpy as np
from numpy import linalg as LA


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

def compute_gravity_hexahedron_faces(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0):
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
