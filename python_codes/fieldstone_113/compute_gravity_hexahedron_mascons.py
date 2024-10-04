import numpy as np
from basisQ1 import *

def compute_gravity_hexahedron_mascons(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,nm_per_dim):

    nmascons=nm_per_dim**3

    xx=np.array([pt_1[0],pt_2[0],pt_3[0],pt_4[0],
                 pt_5[0],pt_6[0],pt_7[0],pt_8[0]],dtype=np.float64)

    yy=np.array([pt_1[1],pt_2[1],pt_3[1],pt_4[1],
                 pt_5[1],pt_6[1],pt_7[1],pt_8[1]],dtype=np.float64)

    zz=np.array([pt_1[2],pt_2[2],pt_3[2],pt_4[2],
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
                #r=-1+i*2/(nm_per_dim-1)
                #s=-1+j*2/(nm_per_dim-1)
                #t=-1+k*2/(nm_per_dim-1)
                N=NNN(r,s,t)
                xm[counter]=N.dot(xx)
                ym[counter]=N.dot(yy)
                zm[counter]=N.dot(zz)
                counter+=1    
            #end for
        #end for
    #end for

    #np.savetxt('mascons1.ascii',np.array([xx,yy,zz]).T)
    #np.savetxt('mascons2.ascii',np.array([xm,ym,zm]).T)

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
                jcb[0,0]=dNdr.dot(xx) ; jcb[0,1]=dNdr.dot(yy) ; jcb[0,2]=dNdr.dot(zz)
                jcb[1,0]=dNds.dot(xx) ; jcb[1,1]=dNds.dot(yy) ; jcb[1,2]=dNds.dot(zz)
                jcb[2,0]=dNdt.dot(xx) ; jcb[2,1]=dNdt.dot(yy) ; jcb[2,2]=dNdt.dot(zz)
                jcob = np.linalg.det(jcb)
                volume+=jcob*weightq
            #end for
        #end for
    #end for
    #print('mascons:',volume)

    #------------------------------------------------------------------------------

    mascon=volume*rho0/nmascons

    grav=np.zeros(3,dtype=np.float64)
    U=0
    for i in range(0,nmascons):
        dist=np.sqrt((xm[i]-pt_M[0])**2+(ym[i]-pt_M[1])**2+(zm[i]-pt_M[2])**2)
        grav[0]+=mascon*Ggrav/dist**3*(xm[i]-pt_M[0])
        grav[1]+=mascon*Ggrav/dist**3*(ym[i]-pt_M[1])
        grav[2]+=mascon*Ggrav/dist**3*(zm[i]-pt_M[2])
        U+=mascon*Ggrav/dist

    return grav,U

