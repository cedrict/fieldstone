from basis_functions import *
import numpy as np

@jit(nopython=True)
def compute_strain_rate2(nel,mV,NV,iconV,mapping,xmapping,ymapping,u,v):

    count= np.zeros(NV,dtype=np.int32)  
    exx2 = np.zeros(NV,dtype=np.float64)  
    eyy2 = np.zeros(NV,dtype=np.float64)  
    exy2 = np.zeros(NV,dtype=np.float64)  
    jcb=np.zeros((2,2),dtype=np.float64)
    dNNNVdx=np.zeros(mV,dtype=np.float64)
    dNNNVdy=np.zeros(mV,dtype=np.float64)

    rVnodes=[-1,1,1,-1,0,1,0,-1,0]
    sVnodes=[-1,-1,1,1,-1,0,1,0,0]

    for iel in range(0,nel):
        for i in range(0,mV):
            rq=rVnodes[i]
            sq=sVnodes[i]

            dNNNVdr=dNNNdr(rq,sq,mapping)
            dNNNVds=dNNNds(rq,sq,mapping)
            jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
            jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
            jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
            jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
            jcbi=np.linalg.inv(jcb)

            #basis functions
            NNNV=NNN(rq,sq,'Q2')
            dNNNVdr=dNNNdr(rq,sq,'Q2')
            dNNNVds=dNNNds(rq,sq,'Q2')

            # compute dNdx & dNdy
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for 

            e_xx=np.dot(dNNNVdx[0:mV],u[iconV[0:mV,iel]])
            e_xy=np.dot(dNNNVdx[0:mV],v[iconV[0:mV,iel]])*0.5+\
                 np.dot(dNNNVdy[0:mV],u[iconV[0:mV,iel]])*0.5
            e_yy=np.dot(dNNNVdy[0:mV],v[iconV[0:mV,iel]])

            inode=iconV[i,iel]
            exx2[inode]+=e_xx
            exy2[inode]+=e_xy
            eyy2[inode]+=e_yy
            count[inode]+=1
        #end for
    #end for
    exx2/=count
    exy2/=count
    eyy2/=count

    return exx2,eyy2,exy2
