import numba
from basis_functions import *

###############################################################################

@numba.njit
def compute_strain_tensor_components(rq,sq,tq,x_V,y_V,z_V,nodes,u,v,w):

    jcb=np.zeros((3,3),dtype=np.float64)
    dNdr__V=basis_functions_V_dr(rq,sq,tq)
    dNds__V=basis_functions_V_ds(rq,sq,tq)
    dNdt__V=basis_functions_V_dt(rq,sq,tq)
    jcb[0,0]=np.dot(dNdr__V,x_V[nodes])
    jcb[0,1]=np.dot(dNdr__V,y_V[nodes])
    jcb[0,2]=np.dot(dNdr__V,z_V[nodes])
    jcb[1,0]=np.dot(dNds__V,x_V[nodes])
    jcb[1,1]=np.dot(dNds__V,y_V[nodes])
    jcb[1,2]=np.dot(dNds__V,z_V[nodes])
    jcb[2,0]=np.dot(dNdt__V,x_V[nodes])
    jcb[2,1]=np.dot(dNdt__V,y_V[nodes])
    jcb[2,2]=np.dot(dNdt__V,z_V[nodes])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr__V+jcbi[0,1]*dNds__V+jcbi[0,2]*dNdt__V
    dNdy_V=jcbi[1,0]*dNdr__V+jcbi[1,1]*dNds__V+jcbi[1,2]*dNdt__V
    dNdz_V=jcbi[2,0]*dNdr__V+jcbi[2,1]*dNds__V+jcbi[2,2]*dNdt__V
    exx=np.dot(dNdx_V,u[nodes])
    eyy=np.dot(dNdy_V,v[nodes])
    ezz=np.dot(dNdz_V,w[nodes])
    exy=np.dot(dNdx_V,v[nodes])*0.5\
       +np.dot(dNdy_V,u[nodes])*0.5
    exz=np.dot(dNdx_V,w[nodes])*0.5\
       +np.dot(dNdz_V,u[nodes])*0.5
    eyz=np.dot(dNdy_V,w[nodes])*0.5\
       +np.dot(dNdz_V,v[nodes])*0.5

    return exx,eyy,ezz,exy,exz,eyz
