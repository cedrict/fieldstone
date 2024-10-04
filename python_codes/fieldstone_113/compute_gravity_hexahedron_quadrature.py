import numpy as np
from basisQ1 import *

def compute_gravity_hexahedron_quadrature(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,nq_per_dim):
    # pt1,2,3,4,5,6,7,8,M: arrays containing x,y,z coordinates

    xx=np.array([pt_1[0],pt_2[0],pt_3[0],pt_4[0],
                 pt_5[0],pt_6[0],pt_7[0],pt_8[0]],dtype=np.float64)

    yy=np.array([pt_1[1],pt_2[1],pt_3[1],pt_4[1],
                 pt_5[1],pt_6[1],pt_7[1],pt_8[1]],dtype=np.float64)

    zz=np.array([pt_1[2],pt_2[2],pt_3[2],pt_4[2],
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
    volume=0
    U=0
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
                jcb[0,0]=dNdr.dot(xx) ; jcb[0,1]=dNdr.dot(yy) ; jcb[0,2]=dNdr.dot(zz)
                jcb[1,0]=dNds.dot(xx) ; jcb[1,1]=dNds.dot(yy) ; jcb[1,2]=dNds.dot(zz)
                jcb[2,0]=dNdt.dot(xx) ; jcb[2,1]=dNdt.dot(yy) ; jcb[2,2]=dNdt.dot(zz)
                jcob = np.linalg.det(jcb)
                xq=N.dot(xx)
                yq=N.dot(yy)
                zq=N.dot(zz)
                dist=np.sqrt((xq-pt_M[0])**2+(yq-pt_M[1])**2+(zq-pt_M[2])**2)
                grav[0]+=rho0*Ggrav/dist**3*(xq-pt_M[0])*weightq*jcob
                grav[1]+=rho0*Ggrav/dist**3*(yq-pt_M[1])*weightq*jcob
                grav[2]+=rho0*Ggrav/dist**3*(zq-pt_M[2])*weightq*jcob
                U+=rho0*Ggrav/dist*weightq*jcob
                volume+=weightq*jcob
            #end for
        #end for
    #end for

    #print('quadrature:',volume)

    return grav,U

