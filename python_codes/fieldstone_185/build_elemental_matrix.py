import numpy as np
import numba

###############################################################################

@numba.njit
def build_elemental_matrix(m_V,nq_per_el,ndof_V_el,D,dNdr_V,dNds_V,dNdt_V,nodes,x_V,y_V,z_V):

    jcb=np.zeros((3,3),dtype=np.float64)
    A_el=np.zeros((ndof_V_el,ndof_V_el),dtype=np.float64)
    B=np.zeros((6,ndof_V_el),dtype=np.float64)

    for k in range(0,nq_per_el):
        jcb[0,0]=np.dot(dNdr_V[k,:],x_V[nodes])
        jcb[0,1]=np.dot(dNdr_V[k,:],y_V[nodes])
        jcb[0,2]=np.dot(dNdr_V[k,:],z_V[nodes])
        jcb[1,0]=np.dot(dNds_V[k,:],x_V[nodes])
        jcb[1,1]=np.dot(dNds_V[k,:],y_V[nodes])
        jcb[1,2]=np.dot(dNds_V[k,:],z_V[nodes])
        jcb[2,0]=np.dot(dNdt_V[k,:],x_V[nodes])
        jcb[2,1]=np.dot(dNdt_V[k,:],y_V[nodes])
        jcb[2,2]=np.dot(dNdt_V[k,:],z_V[nodes])
        jcbi=np.linalg.inv(jcb)
        JxWq=np.linalg.det(jcb) #*weightq(=1)
        dNdx_V=jcbi[0,0]*dNdr_V[k,:]+jcbi[0,1]*dNds_V[k,:]+jcbi[0,2]*dNdt_V[k,:]
        dNdy_V=jcbi[1,0]*dNdr_V[k,:]+jcbi[1,1]*dNds_V[k,:]+jcbi[1,2]*dNdt_V[k,:]
        dNdz_V=jcbi[2,0]*dNdr_V[k,:]+jcbi[2,1]*dNds_V[k,:]+jcbi[2,2]*dNdt_V[k,:]
        for i in range(0,m_V):
            B[0,3*i  ]=dNdx_V[i]
            B[1,3*i+1]=dNdy_V[i]
            B[2,3*i+2]=dNdz_V[i]
            B[3,3*i  ]=dNdy_V[i] ; B[3,3*i+1]=dNdx_V[i]
            B[4,3*i  ]=dNdz_V[i] ; B[4,3*i+2]=dNdx_V[i]
            B[5,3*i+1]=dNdz_V[i] ; B[5,3*i+2]=dNdy_V[i]
        # end for i
        A_el+=B.T.dot(D.dot(B))*JxWq
    # end for k

    return A_el
