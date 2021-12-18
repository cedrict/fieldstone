import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from numpy import linalg as LA

def compute_misfits(rho0,drho,eta0,eta_star,radius,deltarho,Rsphere):

    radius2=radius**2
 
    m=4     # number of nodes making up an element
    ndofV=2  # number of degrees of freedom per node

    nelx = 64
    nely = 64

    Lx=500e3 
    Ly=500e3
    
    nnx=nelx+1  # number of elements, x direction
    nny=nely+1  # number of elements, y direction

    NV=nnx*nny  # number of nodes

    nel=nelx*nely  # number of elements, total

    penalty=1e25  # penalty coefficient value

    Nfem=NV*ndofV  # Total number of degrees of freedom

    eps=1.e-10

    gx=0.  # gravity vector, x component
    gy=-9.81  # gravity vector, y component

    sqrt3=np.sqrt(3.)

    hx=Lx/nelx
    hy=Ly/nely

    Ggrav=6.67e-11

    #################################################################
    # grid point setup
    #################################################################

    x = np.empty(NV,dtype=np.float64)  # x coordinates
    y = np.empty(NV,dtype=np.float64)  # y coordinates

    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            counter += 1

    #################################################################
    # connectivity
    #################################################################

    icon =np.zeros((m,nel),dtype=np.int32)
    counter = 0
    for j in range(0,nely):
        for i in range(0,nelx):
            icon[0, counter] = i + j * (nelx + 1)
            icon[1, counter] = i + 1 + j * (nelx + 1)
            icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
            icon[3, counter] = i + (j + 1) * (nelx + 1)
            counter += 1

    #################################################################
    # define boundary conditions
    # free slip on left and top, no slip on bottom and right
    #################################################################

    bc_fix=np.zeros(Nfem,dtype=np.bool)    # boundary condition, yes/no
    bc_val=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

    for i in range(0,NV):
        if x[i]/Lx<eps:
           bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = 0.
        if x[i]/Lx>1-eps:
           bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = 0.
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
        if y[i]/Ly<eps:
           bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = 0.
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
        if y[i]/Ly>1-eps:
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

    #################################################################
    # build FE matrix
    #################################################################

    elt_in_sphere=np.zeros(Nfem,dtype=np.bool)  
    a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
    b_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix B 
    rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    N     = np.zeros(m,dtype=np.float64)            # shape functions
    dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    u     = np.zeros(NV,dtype=np.float64)          # x-component velocity
    v     = np.zeros(NV,dtype=np.float64)          # y-component velocity
    k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
    c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    for iel in range(0, nel):

        # set 2 arrays to 0 every loop
        b_el = np.zeros(m*ndofV)
        a_el = np.zeros((m*ndofV,m*ndofV), dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [-1, 1]:
            for jq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.

                # calculate shape functions
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)

                # calculate shape function derivatives
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

                # calculate jacobian matrix
                #jcb = np.zeros((2, 2),dtype=np.float64)
                #for k in range(0,m):
                #    jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                #    jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                #    jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                #    jcb[1, 1] += dNds[k]*y[icon[k,iel]]
                #jcob = np.linalg.det(jcb)
                #jcbi = np.linalg.inv(jcb)

                jcb = np.array([[hx/2,0],[0,hy/2]],dtype=np.float64) 
                jcob = hx*hy/4
                jcbi = np.array([[2/hx,0],[0,2/hy]],dtype=np.float64) 

                # compute dNdx & dNdy
                xq=0.0
                yq=0.0
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                             [0.     ,dNdy[i]],
                                             [dNdy[i],dNdx[i]]]

                if xq**2+(yq-0.5*Ly)**2<radius2:
                   etaq=eta0*eta_star
                   rhoq=rho0+drho
                   elt_in_sphere[iel]=True
                else:
                   rhoq=rho0
                   etaq=eta0

                # compute elemental a_mat matrix
                a_el += b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    b_el[2*i+1]+=N[i]*jcob*weightq*rhoq*gy

            #end for jq
        #end for iq

        # integrate penalty term at 1 point
        rq=0.
        sq=0.
        weightq=4

        N[0]=0.25#(1.-rq)*(1.-sq)
        N[1]=0.25#(1.+rq)*(1.-sq)
        N[2]=0.25#(1.+rq)*(1.+sq)
        N[3]=0.25#(1.-rq)*(1.+sq)

        dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
        dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
        dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
        dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

        # compute the jacobian
        #jcb=np.zeros((2,2),dtype=np.float64)
        #for k in range(0, m):
        #    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        #    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        #    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        #    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
        #jcob = np.linalg.det(jcb)
        #jcbi = np.linalg.inv(jcb)

        jcb = np.array([[hx/2,0],[0,hy/2]],dtype=np.float64) 
        jcob = hx*hy/4
        jcbi = np.array([[2/hx,0],[0,2/hy]],dtype=np.float64) 

        # compute dNdx and dNdy
        for k in range(0,m):
            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

        # compute gradient matrix
        for i in range(0,m):
            b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                                  [0.     ,dNdy[i]],
                                  [dNdy[i],dNdx[i]]]

        # compute elemental matrix
        a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*weightq*jcob

        # apply boundary conditions
        for k1 in range(0,m):
                for i1 in range(0,ndofV):
                    m1 =ndofV*icon[k1,iel]+i1
                    if bc_fix[m1]: 
                       fixt=bc_val[m1]
                       ikk=ndofV*k1+i1
                       aref=a_el[ikk,ikk]
                       for jkk in range(0,m*ndofV):
                           b_el[jkk]-=a_el[jkk,ikk]*fixt
                           a_el[ikk,jkk]=0.
                           a_el[jkk,ikk]=0.
                       #end for
                       a_el[ikk,ikk]=aref
                       b_el[ikk]=aref*fixt
                    #end if
                #end for
        #end for
    
        # assemble matrix a_mat and right hand side rhs
        for k1 in range(0,m):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*icon[k1,iel]+i1
                for k2 in range(0,m):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*icon[k2,iel]+i2
                        a_mat[m1,m2]+=a_el[ikk,jkk]
                    #end for i2
                #end for k2
                rhs[m1]+=b_el[ikk]
            #end for i1
        #end for k1

    #end for iel

    #################################################################
    # solve system
    #################################################################

    sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

    #####################################################################
    # put solution into separate x,y velocity arrays
    #####################################################################

    u,v=np.reshape(sol,(NV,2)).T

    #print("u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
    #print("v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

    #np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    #np.savetxt('velocity.ascii',np.array([x[0:nnx],u[NV-nnx:NV]]).T,header='# x,u')

    #####################################################################
    # compute gravity
    # because half the disc is missing we on the fly mirror it
    #####################################################################

    volume=np.pi*Rsphere**2

    gravy = np.zeros(nnx,dtype=np.float64) 
    gravy_th = np.zeros(nnx,dtype=np.float64) 

    for iel in range(0, nel):
        if elt_in_sphere[iel]:
           for iq in [-1, 1]:
               for jq in [-1, 1]:
                   rq=iq/sqrt3
                   sq=jq/sqrt3
                   weightq=1.*1.
                   N[0]=0.25*(1.-rq)*(1.-sq)
                   N[1]=0.25*(1.+rq)*(1.-sq)
                   N[2]=0.25*(1.+rq)*(1.+sq)
                   N[3]=0.25*(1.-rq)*(1.+sq)
                   xq=0.0
                   yq=0.0
                   for k in range(0,m):
                       xq+=N[k]*x[icon[k,iel]]
                       yq+=N[k]*y[icon[k,iel]]
                   jcob=hx*hy/4
                   for i in range(0,nnx):
                       dist2=(x[i]-xq)**2+(Ly-yq)**2
                       gravy[i]+=Ggrav*drho*weightq*jcob/dist2*(Ly-yq)
                       dist2=(x[i]+xq)**2+(Ly-yq)**2
                       gravy[i]+=Ggrav*drho*weightq*jcob/dist2*(Ly-yq)

    for i in range(0,nnx):
        gravy_th[i]=Ggrav*volume*deltarho/(x[i]**2+(y[i]-Ly/2)**2)*(Ly-Ly/2)

    #np.savetxt('gravity.ascii',np.array([x[0:nnx],gravy,gravy_th]).T)

    misfit_grav=LA.norm(gravy-gravy_th,2)

    #print('misfit gravity=',misfit_grav)

    #####################################################################
    # export to vtu 
    #####################################################################

    if False:
       vtufile=open("solution.vtu","w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='sphere' Format='ascii'> \n")
       for iel in range (0,nel):
           if elt_in_sphere[iel]:
              vtufile.write("%f\n" % 1.)
           else:
              vtufile.write("%f\n" % 0.)
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

    misfit_vel=0.

    return  misfit_grav,misfit_vel
