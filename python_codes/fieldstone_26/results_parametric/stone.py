import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as time
import matplotlib.pyplot as plt 

#------------------------------------------------------------------------------

def density(x,y,Lx,Ly):
    val=3150
    if (y>660e3-80e3 and y<=660e3) or (y>660e3-(80e3+250e3) and abs(x-Lx/2)<40e3):
       val=3300
    #val-=3150
    return val

#------------------------------------------------------------------------------

def viscosity(x,y,Lx,Ly,exx,eyy,exy):
    val=1.e21
    sr=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
    if (y>660e3-80e3 and y<=660e3) or (y>660e3-(80e3+250e3) and abs(x-Lx/2)<40e3):
       if sr<1e-30:
          sr=1e-30
       n_pow=4 
       val=(4.75e11)*sr**(1./n_pow -1.)
    else:
       sr=max(sr,1e-30)
       n_pow=3 
       if case=='2a' or case=='2b':
          val=(4.54e10)*sr**(1./n_pow -1.)
    val=max(val,1e19)
    val=min(val,1e25)
    return val

#------------------------------------------------------------------------------

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------
start_time = time.time() #---------------------------------------------------------> Start total runtime
#------------------------------------------------------------------------------

#sys.stdout = open(r'C:\Users\thoma\Documents\uni\2022-2023\Honours\URP\report.txt', 'w')
#sys.stdout = open(r'report.txt', 'w')

matplotlib = True #----------------------------------------------------------------> matplotlib if statement

print("-----------------------------")
print("---------- stone 26 ---------")
print("-----------------------------")

#------------------------------------------------------------------------------
#start the loopies
#------------------------------------------------------------------------------


#define loop parameters
tol_loop = [1e-3, 1e-4, 1e-5, 1e-7, 1e-8]
case_loop = ['1a']
res_loop = [50, 100, 150, 200, 250, 300, 400]
loopnr = 0

#controls
nonconverged=[]

    
for it in tol_loop:

    vrmsfile=open("vrms_tol_%s.ascii"%(it),"w")
    ptA_file=open("ptA_tol_%s.ascii"%(it),"w")
    ptB_file=open("ptB_tol_%s.ascii"%(it),"w")
    ptC_file=open("ptC_tol_%s.ascii"%(it),"w")

    for ic in case_loop:
        for ir in res_loop: 
                    
                    loopnr += 1
                
            
                    cm=0.01
                    year=3.154e+7
                    eps=1.e-10
                    sqrt3=np.sqrt(3.)
                    
                    print("-----------------------------")
                    print("-------------------------------case %s, res nelx = %s-----LOOP %s----------\n" % (ic, ir, loopnr))
                    print("-----------------------------")
                    
                    mV=4     # number of nodes making up a Q1 element
                    mP=1     # number of nodes making up a Q0 element
                    ndofV=2  # number of velocity degrees of freedom per node
                    ndofP=1  # number of pressure degrees of freedom 
                    
                    Lx=1000e3  # horizontal extent of the domain 
                    Ly=660e3  # vertical extent of the domain 
                    
                    if int(len(sys.argv) == 4):
                       nelx = int(sys.argv[1])
                       nely = int(sys.argv[2])
                       visu = int(sys.argv[3])
                    else:
                       nelx = ir  #-------------------------------------------------------------------> RES [nelx] here
                       nely = int(nelx*Ly/Lx)
                       visu = 1
                    
                    gy=-10.
                    eta_ref=1e21
                    niter=150 
                    tol=it   #-------------------------------------------------------------------> TOL [tol] here
                    
                    case = ic #-------------------------------------------------------------------> CASE [case] here
                        
                    nnx=nelx+1  # number of elements, x direction
                    nny=nely+1  # number of elements, y direction
                    NV=nnx*nny  # number of nodes
                    nel=nelx*nely  # number of elements, total
                    NfemV=NV*ndofV # number of velocity dofs
                    NfemP=nel*ndofP # number of pressure dofs
                    Nfem=NfemV+NfemP # total number of dofs
                    
                    #################################################################
                    
                    print("nelx",nelx)
                    print("nely",nely)
                    print("nel",nel)
                    print("nnx=",nnx)
                    print("nny=",nny)
                    print("NV=",NV)
                    print("NfemV=",NfemV)
                    print("NfemP=",NfemP)
                    print("Nfem=",Nfem)
                    print("------------------------------")
                    
                    #################################################################
                    # grid point setup
                    #################################################################
                    start = time.time()
                    
                    x = np.empty(NV, dtype=np.float64)  # x coordinates
                    y = np.empty(NV, dtype=np.float64)  # y coordinates
                    
                    counter = 0
                    for j in range(0, nny):
                        for i in range(0, nnx):
                            x[counter]=i*Lx/float(nelx)
                            y[counter]=j*Ly/float(nely)
                            counter += 1
                        #end for
                    #end for
                    
                    print("setup: grid points: %.3f s" % (time.time() - start))
                    
                    #################################################################
                    # connectivity
                    #################################################################
                    start = time.time()
                    
                    icon =np.zeros((mV,nel),dtype=np.int32)
                    
                    counter = 0
                    for j in range(0,nely):
                        for i in range(0,nelx):
                            icon[0,counter]=i+j*(nelx+1)
                            icon[1,counter]=i+1+j*(nelx+1)
                            icon[2,counter]=i+1+(j+1)*(nelx+1)
                            icon[3,counter]=i+(j+1)*(nelx+1)
                            counter += 1
                        #end for
                    #end for
                    
                    print("setup: connectivity: %.3f s" % (time.time() - start))
                    
                    #################################################################
                    # define boundary conditions
                    #################################################################
                    start = time.time()
                    
                    bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
                    bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
                    
                    for i in range(0,NV):
                        if x[i]/Lx<eps:
                           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
                           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
                        #end if
                        if x[i]/Lx>(1-eps):
                           bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
                           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
                        #end if
                        if y[i]/Ly<eps:
                           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
                        #end if
                        if case=='1a' or case=='2a':
                           if y[i]/Ly>(1-eps):
                              bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
                           #end if
                    #end for
                    
                    print("setup: boundary conditions: %.3f s" % (time.time() - start))
                    
                    #################################################################
                    # allocate arrays
                    #################################################################
                    
                    u    =np.zeros(NV,dtype=np.float64)  
                    v    =np.zeros(NV,dtype=np.float64)   
                    p    =np.zeros(nel,dtype=np.float64)    
                    sol  =np.zeros(Nfem,dtype=np.float64) 
                    xi   =np.zeros(niter,dtype=np.float64) 
                    c_mat=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
                    
                    ###############################################################################
                    # nonlinear iterations
                    ###############################################################################
    
                    for iiter in range(0,niter):
    
                        print("------------------------------")
                        print("iter= %d" % iiter) 
                        print("------------------------------")
    
                        #################################################################
                        # build FE matrix
                        # [ K G ][u]=[f]
                        # [GT 0 ][p] [h]
                        #################################################################
                        start = time.time()
    
                        A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
                        f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
                        h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
                        b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
                        NNNV  = np.zeros(mV,dtype=np.float64)            # shape functions
                        dNdx  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
                        dNdy  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
                        dNdr  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
                        dNds  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    
                        for iel in range(0, nel):
    
                            # set arrays to 0 every loop
                            f_el =np.zeros((mV*ndofV),dtype=np.float64)
                            K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
                            G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
                            h_el=np.zeros((1,1),dtype=np.float64)
    
                            # integrate viscous term at 4 quadrature points
                            for iq in [-1,1]:
                                for jq in [-1,1]:
    
                                    # position & weight of quad. point
                                    rq=iq/sqrt3
                                    sq=jq/sqrt3
                                    weightq=1.*1.
    
                                    # calculate shape functions
                                    NNNV[0:mV]=NNV(rq,sq)
                                    dNdr[0:mV]=dNNVdr(rq,sq)
                                    dNds[0:mV]=dNNVds(rq,sq)
    
                                    # calculate jacobian matrix
                                    jcb = np.zeros((2,2),dtype=np.float64)
                                    for k in range(0,mV):
                                        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                                        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                                        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                                        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                                    #end for 
                                    jcob = np.linalg.det(jcb)
                                    jcbi = np.linalg.inv(jcb)
    
                                    # compute dNdx & dNdy
                                    xq=0.0
                                    yq=0.0
                                    exxq=0.
                                    eyyq=0.
                                    exyq=0.
                                    for k in range(0,mV):
                                        xq+=NNNV[k]*x[icon[k,iel]]
                                        yq+=NNNV[k]*y[icon[k,iel]]
                                        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                                        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                                        exxq += dNdx[k]*u[icon[k,iel]]
                                        eyyq += dNdy[k]*v[icon[k,iel]]
                                        exyq += 0.5*dNdy[k]*u[icon[k,iel]]+\
                                                0.5*dNdx[k]*v[icon[k,iel]]
                                    #end for 
    
                                    # compute density and viscosity at qpoint
                                    rhoq=density(xq,yq,Lx,Ly)
                                    etaq=viscosity(xq,yq,Lx,Ly,exxq,eyyq,exyq)
    
                                    # construct 3x8 b_mat matrix
                                    for i in range(0,mV):
                                        b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                                                 [0.     ,dNdy[i]],
                                                                 [dNdy[i],dNdx[i]]]
                                    #end for 
    
                                    # compute elemental matrix
                                    K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob
    
                                    # compute elemental rhs vector
                                    for i in range(0,mV):
                                        f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rhoq*gy
                                        G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                                        G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq
                                    #end for 
    
                                #end for jq
                            #end for iq
    
                            # impose b.c. 
                            for k1 in range(0,mV):
                                for i1 in range(0,ndofV):
                                    ikk=ndofV*k1          +i1
                                    m1 =ndofV*icon[k1,iel]+i1
                                    if bc_fix[m1]:
                                       K_ref=K_el[ikk,ikk] 
                                       for jkk in range(0,mV*ndofV):
                                           f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                                           K_el[ikk,jkk]=0
                                           K_el[jkk,ikk]=0
                                       #end for 
                                       K_el[ikk,ikk]=K_ref
                                       f_el[ikk]=K_ref*bc_val[m1]
                                       h_el[0]-=G_el[ikk,0]*bc_val[m1]
                                       G_el[ikk,0]=0
                                    #end if
                                #end for 
                            #end for 
    
                            # assemble matrix and rhs 
                            for k1 in range(0,mV):
                                for i1 in range(0,ndofV):
                                    ikk=ndofV*k1          +i1
                                    m1 =ndofV*icon[k1,iel]+i1
                                    for k2 in range(0,mV):
                                        for i2 in range(0,ndofV):
                                            jkk=ndofV*k2          +i2
                                            m2 =ndofV*icon[k2,iel]+i2
                                            A_sparse[m1,m2]+=K_el[ikk,jkk]
                                        #end for 
                                    #end for 
                                    for k2 in range(0,mP):
                                        jkk=k2
                                        m2 = iel#iconP[k2,iel]
                                        A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*eta_ref/Lx
                                        A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*eta_ref/Lx
                                    #end for
                                    f_rhs[m1]+=f_el[ikk]
                                #end for 
                            #end for 
                            for k2 in range(0,mP):
                                m2= iel#iconP[k2,iel]
                                h_rhs[m2]+=h_el[k2]
    
                        #end for iel
    
                        print("build FE matrix: %.3f s" % (time.time() - start))
    
                        ######################################################################
                        # assemble K, G, GT, f, h into A and rhs
                        ######################################################################
                        start = time.time()
    
                        rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
                        rhs[0:NfemV]=f_rhs
                        rhs[NfemV:Nfem]=h_rhs
    
                        print("assemble blocks: %.3f s" % (time.time() - start))
    
                        ######################################################################
                        # convergence test 
                        ######################################################################
                        start = time.time()
    
                        residual=A_sparse.dot(sol)-rhs
    
                        xi[iiter]=np.linalg.norm(residual,2)/np.linalg.norm(rhs,2)
    
                        print("     -> xi= %.4e tol= %.4e " %(xi[iiter],tol))
    
                        xifile="xi_case%s_nelx_%s_tol_%s.ascii"  % (ic, ir, tol)
                        np.savetxt(xifile,np.array([xi[0:iiter]]).T,header='# xi')
    
                        if xi[iiter]<tol:
                           print('     *****converged*****')
                           break
    
                        print("compute residual: %.3f s" % (time.time() - start))
    
                        ######################################################################
                        # solve system
                        ######################################################################
                        start = time.time()
    
                        sol=sps.linalg.spsolve(sps.csr_matrix(A_sparse),rhs)
    
                        print("solve time: %.3f s" % (time.time() - start))
    
                        ######################################################################
                        # put solution into separate x,y velocity arrays
                        ######################################################################
                        start = time.time()
    
                        u,v=np.reshape(sol[0:NfemV],(NV,2)).T
                        p=sol[NfemV:Nfem]*(eta_ref/Lx)
    
                        print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
                        print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
    
                        print("split vel into u,v: %.3f s" % (time.time() - start))
    
                        ######################################################################
                        # compute elemental strainrate, density and viscosity
                        # these fields are only for visualisation purposes
                        ######################################################################
                        start = time.time()
                    
                        xc=np.zeros(nel,dtype=np.float64)  
                        yc=np.zeros(nel,dtype=np.float64)  
                        exx=np.zeros(nel,dtype=np.float64)  
                        eyy=np.zeros(nel,dtype=np.float64)  
                        exy=np.zeros(nel,dtype=np.float64)  
                        rho=np.zeros(nel,dtype=np.float64)    
                        eta=np.zeros(nel,dtype=np.float64)   
                    
                        for iel in range(0,nel):
                            rq = 0.0
                            sq = 0.0
                            NNNV[0:mV]=NNV(rq,sq)
                            dNdr[0:mV]=dNNVdr(rq,sq)
                            dNds[0:mV]=dNNVds(rq,sq)
                    
                            jcb=np.zeros((2,2),dtype=np.float64)
                            for k in range(0,mV):
                                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                            #end for
                            jcbi=np.linalg.inv(jcb)
                    
                            for k in range(0,mV):
                                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                                xc[iel]+=NNNV[k]*x[icon[k,iel]]
                                yc[iel]+=NNNV[k]*y[icon[k,iel]]
                                exx[iel]+=dNdx[k]*u[icon[k,iel]]
                                eyy[iel]+=dNdy[k]*v[icon[k,iel]]
                                exy[iel]+=0.5*(dNdy[k]*u[icon[k,iel]]+dNdx[k]*v[icon[k,iel]])
                            #end for
                    
                            rho[iel]=density(xc[iel],yc[iel],Lx,Ly)
                            eta[iel]=viscosity(xc[iel],yc[iel],Lx,Ly,exx[iel],eyy[iel],exy[iel])
                    
                        #end for
                    
                        print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
                        print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
                        print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
                    
                        print("compute press & sr: %.3f s" % (time.time() - start))
                    
                    #end for iter

                    if xi[iiter]>tol: 
                        nonconverged.append('niter=%s, tol=%s, case=%s, nelx=%s, xi=%s | ' % (niter, it, ic, ir, xi[iiter]))
                    
                    #####################################################################
                    # compute pressure and strain rate onto Q1 grid
                    #####################################################################
                    
                    rVnodes=[-1,+1,+1,-1]
                    sVnodes=[-1,-1,+1,+1]
                    
                    q=np.zeros(NV,dtype=np.float64)  
                    exxn=np.zeros(NV,dtype=np.float64)  
                    exyn=np.zeros(NV,dtype=np.float64)  
                    eyyn=np.zeros(NV,dtype=np.float64)  
                    count=np.zeros(NV,dtype=np.float64)  
                    
                    for iel in range(0,nel):
                        for i in range(0,mV):
                            rq=rVnodes[i]
                            sq=sVnodes[i]
                            NNNV[0:mV]=NNV(rq,sq)
                            dNdr[0:mV]=dNNVdr(rq,sq)
                            dNds[0:mV]=dNNVds(rq,sq)
                            jcb=np.zeros((2,2),dtype=np.float64)
                            for k in range(0,mV):
                                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                            #end for
                            jcbi=np.linalg.inv(jcb)
                            for k in range(0,mV):
                                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                                exxn[icon[i,iel]]+=dNdx[k]*u[icon[k,iel]]
                                eyyn[icon[i,iel]]+=dNdy[k]*v[icon[k,iel]]
                                exyn[icon[i,iel]]+=0.5*(dNdy[k]*u[icon[k,iel]]+dNdx[k]*v[icon[k,iel]])
                            #end for
                        #end for
                    
                        q[icon[0,iel]]+=p[iel]
                        q[icon[1,iel]]+=p[iel]
                        q[icon[2,iel]]+=p[iel]
                        q[icon[3,iel]]+=p[iel]
                    
                        count[icon[0,iel]]+=1
                        count[icon[1,iel]]+=1
                        count[icon[2,iel]]+=1
                        count[icon[3,iel]]+=1
                    #end for
                    
                    q/=count
                    exxn/=count
                    exyn/=count
                    eyyn/=count
                        
                    print("     -> exxn (m,M) %.4e %.4e " %(np.min(exxn),np.max(exxn)))
                    print("     -> eyyn (m,M) %.4e %.4e " %(np.min(eyyn),np.max(eyyn)))
                    print("     -> exyn (m,M) %.4e %.4e " %(np.min(exyn),np.max(exyn)))

                    #####################################################################
                    #compute vrms
                    #####################################################################
                    start = time.time()

                    vrms=0.
                    for iel in range(0,nel):
                        # integrate viscous term at 4 quadrature points
                        for iq in [-1,1]:
                            for jq in [-1,1]:
                                rq=iq/sqrt3
                                sq=jq/sqrt3
                                weightq=1.
                                NNNV[0:mV]=NNV(rq,sq)
                                dNdr[0:mV]=dNNVdr(rq,sq)
                                dNds[0:mV]=dNNVds(rq,sq)
                                jcb=np.zeros((2,2),dtype=np.float64)
                                for k in range(0,mV):
                                    jcb[0,0] += dNdr[k]*x[icon[k,iel]]
                                    jcb[0,1] += dNdr[k]*y[icon[k,iel]]
                                    jcb[1,0] += dNds[k]*x[icon[k,iel]]
                                    jcb[1,1] += dNds[k]*y[icon[k,iel]]
                                jcob=np.linalg.det(jcb)
                                uq=NNNV[0:mV].dot(u[icon[0:mV,iel]])
                                vq=NNNV[0:mV].dot(v[icon[0:mV,iel]])
                                vrms+=(uq**2+vq**2)*jcob*weightq 
                            #end for jq
                        #end for iq
                    #end for iel
                    vrms=np.sqrt(vrms/Lx/Ly)

                    print("     -> nel= %6d ; vrms (cm/year)= %e " %(nel,vrms/cm*year))

                    vrmsfile.write("%s %s %s %e\n" %(it,ic,ir,vrms/cm*year))
                    vrmsfile.flush()

                    print("compute vrms: %.3f s" % (time.time() - start))
                    
                    #####################################################################
                    # line measurements 
                    # each line of measurements is discretised with npts points. 
                    # each point is localised in an element, and strainrate is computed
                    # at this location. density and viscosity are then computed 
                    # on it and stored.
                    #####################################################################
                    start = time.time()
                    
                    npts=2000
                    
                    xp=np.zeros(npts,dtype=np.float64)  
                    yp=np.zeros(npts,dtype=np.float64)  
                    rhop=np.zeros(npts,dtype=np.float64)  
                    etap=np.zeros(npts,dtype=np.float64)  
                    exxp=np.zeros(npts,dtype=np.float64)  
                    eyyp=np.zeros(npts,dtype=np.float64)  
                    exyp=np.zeros(npts,dtype=np.float64)  
                    exxp2=np.zeros(npts,dtype=np.float64)  
                    eyyp2=np.zeros(npts,dtype=np.float64)  
                    exyp2=np.zeros(npts,dtype=np.float64)  
                    
                    for i in range(0,npts):
                        xp[i]=i*Lx/(npts-1)
                        yp[i]=550e3
                        xp[i]=min(xp[i],Lx*(1-eps))
                        xp[i]=max(xp[i],Lx*eps)
                        ielx=int(xp[i]/Lx*nelx)
                        iely=int(yp[i]/Ly*nely)
                        iel=nelx*(iely)+ielx
                        xmin=x[icon[0,iel]] ; xmax=x[icon[2,iel]]
                        ymin=y[icon[0,iel]] ; ymax=y[icon[2,iel]]
                        r=((xp[i]-xmin)/(xmax-xmin)-0.5)*2
                        s=((yp[i]-ymin)/(ymax-ymin)-0.5)*2
                        NNNV[0:mV]=NNV(r,s)
                        dNdr[0:mV]=dNNVdr(r,s)
                        dNds[0:mV]=dNNVds(r,s)
                        jcb=np.zeros((2,2),dtype=np.float64)
                        for k in range(0,mV):
                            jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                            jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                            jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                            jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                        #end for
                        jcbi=np.linalg.inv(jcb)
                        for k in range(0,mV):
                            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                            exxp[i]+=dNdx[k]*u[icon[k,iel]]
                            eyyp[i]+=dNdy[k]*v[icon[k,iel]]
                            exyp[i]+=0.5*(dNdy[k]*u[icon[k,iel]]+dNdx[k]*v[icon[k,iel]])
                            exxp2[i]+=NNNV[k]*exxn[icon[k,iel]]
                            exyp2[i]+=NNNV[k]*exyn[icon[k,iel]]
                            eyyp2[i]+=NNNV[k]*eyyn[icon[k,iel]]
                        #end for
                        rhop[i]=density(xp[i],yp[i],Lx,Ly)
                        etap[i]=viscosity(xp[i],yp[i],Lx,Ly,exxp[i],eyyp[i],exyp[i])
                    #end for
                    
                    #####################################################################
                    # plot x nodal data with matplotlib 
                    #####################################################################
                    
                    if matplotlib:
                        
                        ###########################################
                        
                        #x-axis plots
                        fig, axs = plt.subplots(2, 2)
                        fig.suptitle('x-axis plots at nelx:%s for case:%s' % (ir, ic))
                        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
                        fig.set_figheight(12)
                        fig.set_figwidth(15)
                        
                        #x,exx
                        axs[0,0].plot(xp*1e-3,exxp2)
                        axs[0,0].set_title("exx nodal")
                        axs[0,0].set_xlabel("x [km]")
                        axs[0,0].set_ylabel("exx")
                        
                        #x,eyy
                        axs[0,1].plot(xp*1e-3,eyyp2)
                        axs[0,1].set_title("eyy nodal")
                        axs[0,1].set_xlabel("x [km]")
                        axs[0,1].set_ylabel("eyy")
                    
                        #x,exy
                        axs[1,0].plot(xp*1e-3,exyp2)
                        axs[1,0].set_title("exy nodal")
                        axs[1,0].set_xlabel("x [km]")
                        axs[1,0].set_ylabel("exy")
                    
                        #x,etap
                        axs[1,1].plot(xp*1e-3, etap)
                        axs[1,1].set_title("viscocity")
                        axs[1,1].set_xlabel("x [km]")
                        axs[1,1].set_ylabel("etap")
                        axs[1,1].set_yscale('log')
                        
                        #plt.savefig('x-axisplot_tol=e'+str(int(np.log(it))))
                        
                        pltfile="x-axisplot_case%s_nelx_%s_tol_1e%s"  % (ic, ir, str(int(np.log10(tol))))
                        plt.savefig(pltfile)
                    
                    ############################################### 
                    
                    np.savetxt('horizontal_case_%s_nelx_%s_tol_%s.ascii' % (ic, ir, tol),np.array([xp,yp,rhop,etap,exxp,eyyp,exyp,exxp2,eyyp2,exyp2]).T,header='# x,y,rho,eta')
                    
                    xp=np.zeros(npts,dtype=np.float64)  
                    yp=np.zeros(npts,dtype=np.float64)  
                    rhop=np.zeros(npts,dtype=np.float64)  
                    etap=np.zeros(npts,dtype=np.float64)  
                    exxp=np.zeros(npts,dtype=np.float64)  
                    eyyp=np.zeros(npts,dtype=np.float64)  
                    exyp=np.zeros(npts,dtype=np.float64)  
                    exxp2=np.zeros(npts,dtype=np.float64)  
                    eyyp2=np.zeros(npts,dtype=np.float64)  
                    exyp2=np.zeros(npts,dtype=np.float64)  
                    
                    for i in range(0,npts):
                        xp[i]=Lx/2.
                        yp[i]=i*Ly/(npts-1)
                        yp[i]=min(yp[i],Ly*(1-eps))
                        yp[i]=max(yp[i],Ly*eps)
                        ielx=int(xp[i]/Lx*nelx)
                        iely=int(yp[i]/Ly*nely)
                        iel=nelx*(iely)+ielx
                        xmin=x[icon[0,iel]] ; xmax=x[icon[2,iel]]
                        ymin=y[icon[0,iel]] ; ymax=y[icon[2,iel]]
                        r=((xp[i]-xmin)/(xmax-xmin)-0.5)*2
                        s=((yp[i]-ymin)/(ymax-ymin)-0.5)*2
                        NNNV[0:mV]=NNV(r,s)
                        dNdr[0:mV]=dNNVdr(r,s)
                        dNds[0:mV]=dNNVds(r,s)
                        jcb=np.zeros((2,2),dtype=np.float64)
                        for k in range(0,mV):
                            jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                            jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                            jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                            jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                        #end for
                        jcbi=np.linalg.inv(jcb)
                        for k in range(0,mV):
                            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                            exxp[i]+=dNdx[k]*u[icon[k,iel]]
                            eyyp[i]+=dNdy[k]*v[icon[k,iel]]
                            exyp[i]+=0.5*(dNdy[k]*u[icon[k,iel]]+dNdx[k]*v[icon[k,iel]])
                            exxp2[i]+=NNNV[k]*exxn[icon[k,iel]]
                            exyp2[i]+=NNNV[k]*exyn[icon[k,iel]]
                            eyyp2[i]+=NNNV[k]*eyyn[icon[k,iel]]
                        #end for
                        rhop[i]=density(xp[i],yp[i],Lx,Ly)
                        etap[i]=viscosity(xp[i],yp[i],Lx,Ly,exxp[i],eyyp[i],exyp[i])
                    #end for
                         
                    np.savetxt('vertical_case%s_nelx_%s_tol_%s.ascii' % (ic, ir,tol),np.array([xp,yp,rhop,etap,exxp,eyyp,exyp,exxp2,eyyp2,exyp2]).T,header='# x,y,rho,eta')
                     
                    #####################################################################
                    # plot y nodal data with matplotlib 
                    #####################################################################
                    
                    if matplotlib:
                        
                        ###########################################
                        #y-axis plots
                        fig, axs = plt.subplots(2, 2)
                        fig.suptitle('y-axis plots at nelx:%s for case:%s' % (ir, ic))
                        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
                        fig.set_figheight(12)
                        fig.set_figwidth(15)
                        
                        #x,exx
                        axs[0,0].plot(yp*1e-3,exxp2)
                        axs[0,0].set_title("exx nodal")
                        axs[0,0].set_xlabel("y [km]")
                        axs[0,0].set_ylabel("exx")
                        
                        #x,eyy
                        axs[0,1].plot(yp*1e-3,eyyp2)
                        axs[0,1].set_title("eyy nodal")
                        axs[0,1].set_xlabel("y [km]")
                        axs[0,1].set_ylabel("eyy")
                    
                        #x,exy
                        axs[1,0].plot(yp*1e-3,exyp2)
                        axs[1,0].set_title("exy nodal")
                        axs[1,0].set_xlabel("y [km]")
                        axs[1,0].set_ylabel("exy")
                    
                        #x,etap
                        axs[1,1].plot(yp*1e-3, etap)
                        axs[1,1].set_title("viscosity")
                        axs[1,1].set_xlabel("y [km]")
                        axs[1,1].set_ylabel("etap")
                        axs[1,1].set_yscale('log')  

                        pltfile="y-axisplot_case%s_nelx_%s_tol_1e%s"  % (ic, ir, str(int(np.log10(tol))))
                        plt.savefig(pltfile)
                        
                    ################################################################################
                    
                    print("export profiles: %.3fs" % (time.time() - start))

                    #####################################################################
                    # export values at key points
                    #####################################################################
                    start = time.time()

                    npts=3
                    xp=np.array([500e3,500e3,750e3],dtype=np.float64) 
                    yp=np.array([440e3,620e3,620e3],dtype=np.float64) 
                    rhop=np.zeros(npts,dtype=np.float64)  
                    etap=np.zeros(npts,dtype=np.float64)  
                    exxp=np.zeros(npts,dtype=np.float64)  
                    eyyp=np.zeros(npts,dtype=np.float64)  
                    exyp=np.zeros(npts,dtype=np.float64)  
                    exxp2=np.zeros(npts,dtype=np.float64)  
                    eyyp2=np.zeros(npts,dtype=np.float64)  
                    exyp2=np.zeros(npts,dtype=np.float64)  

                    for i in range(0,npts):
                        ielx=int(xp[i]/Lx*nelx)
                        iely=int(yp[i]/Ly*nely)
                        iel=nelx*(iely)+ielx
                        xmin=x[icon[0,iel]] ; xmax=x[icon[2,iel]]
                        ymin=y[icon[0,iel]] ; ymax=y[icon[2,iel]]
                        r=((xp[i]-xmin)/(xmax-xmin)-0.5)*2
                        s=((yp[i]-ymin)/(ymax-ymin)-0.5)*2
                        NNNV[0:mV]=NNV(r,s)
                        dNdr[0:mV]=dNNVdr(r,s)
                        dNds[0:mV]=dNNVds(r,s)
                        jcb=np.zeros((2,2),dtype=np.float64)
                        for k in range(0,mV):
                            jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                            jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                            jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                            jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                        #end for
                        jcbi=np.linalg.inv(jcb)
                        for k in range(0,mV):
                            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                            exxp[i]+=dNdx[k]*u[icon[k,iel]]
                            eyyp[i]+=dNdy[k]*v[icon[k,iel]]
                            exyp[i]+=0.5*(dNdy[k]*u[icon[k,iel]]+dNdx[k]*v[icon[k,iel]])
                            exxp2[i]+=NNNV[k]*exxn[icon[k,iel]]
                            exyp2[i]+=NNNV[k]*exyn[icon[k,iel]]
                            eyyp2[i]+=NNNV[k]*eyyn[icon[k,iel]]
                        #end for
                        rhop[i]=density(xp[i],yp[i],Lx,Ly)
                        etap[i]=viscosity(xp[i],yp[i],Lx,Ly,exxp[i],eyyp[i],exyp[i])
                    #end for
     
                    ptA_file.write("%e %e %e %s \n" %(xp[0],yp[0],etap[0],nelx))
                    ptB_file.write("%e %e %e %s \n" %(xp[1],yp[1],etap[1],nelx))
                    ptC_file.write("%e %e %e %s \n" %(xp[2],yp[2],etap[2],nelx))
                    ptA_file.flush()
                    ptB_file.flush()
                    ptC_file.flush()

                    #np.savetxt('three_points.ascii',np.array([xp/1000,yp/1000,rhop,etap,\
                    #           exxp,eyyp,exyp,exxp2,eyyp2,exyp2]).T,\
                    #           header='# x,y,rho,eta,exx,eyy,exy,exx,eyy,exp',fmt='%.6e')

                    print("export values at key points: %.3fs" % (time.time() - start))
                    
                    #####################################################################
                    # plot of solution
                    #####################################################################
                    start = time.time()
                    
                    if visu==1:
                       vtufile=open("solution_case%s_nelx=%s_tol=%s.vtu"  % (ic, ir, tol),"w")
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
                       #--
                       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
                       for iel in range (0,nel):
                           vtufile.write("%10e\n" % p[iel])
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx' Format='ascii'> \n" )
                       for iel in range (0,nel):
                           vtufile.write("%10e\n" % exx[iel])
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy' Format='ascii'> \n" )
                       for iel in range (0,nel):
                           vtufile.write("%10e\n" % eyy[iel])
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy' Format='ascii'> \n" )
                       for iel in range (0,nel):
                           vtufile.write("%10e\n" % exy[iel])
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
                       for iel in range(0,nel):
                           vtufile.write("%5e \n" % rho[iel])
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='viscosity' Format='ascii'> \n" )
                       for iel in range(0,nel):
                           vtufile.write("%5e \n" % eta[iel])
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("</CellData>\n")
                       #####
                       vtufile.write("<PointData Scalars='scalars'>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n" )
                       for i in range(0,NV):
                           vtufile.write("%10e %10e %10e \n" %(u[i]*year/cm,v[i]*year/cm,0.))
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n" )
                       for i in range(0,NV):
                           vtufile.write("%10e \n" % exxn[i] )
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n" )
                       for i in range(0,NV):
                           vtufile.write("%10e \n" % eyyn[i] )
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n" )
                       for i in range(0,NV):
                           vtufile.write("%10e \n" % exyn[i] )
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n" )
                       for i in range(0,NV):
                           vtufile.write("%10e \n" % q[i] )
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("</PointData>\n")
                       #####
                       vtufile.write("<Cells>\n")
                       #--
                       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n") 
                       for iel in range (0,nel):
                           vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n" )
                       for iel in range (0,nel):
                           vtufile.write("%d \n" %((iel+1)*4))
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n" )
                       for iel in range (0,nel):
                           vtufile.write("%d \n" %9)
                       vtufile.write("</DataArray>\n")
                       #--
                       vtufile.write("</Cells>\n")
                       #####
                       vtufile.write("</Piece>\n")
                       vtufile.write("</UnstructuredGrid>\n")
                       vtufile.write("</VTKFile>\n")
                       vtufile.close()
                       print("export to vtu | time: %.3f s" % (time.time() - start))
                       
                       print("-----------------------------")
                       print("-----------end loop---------")
                       print("-----------------------------")

                    #end if 

        #for ir in res_loop: 

    #for ic in case_loop:

#for it in tol_loop:

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

#####################################################################
# report runtime
#####################################################################

print ("at resolution nelx=%s" %nelx)
print("total runtime in s: %s seconds " % (time.time() - start_time)) #--------------> End total runtime
#for future convenience:
minutes = ((time.time() - start_time))/60
print("total runtime in m: %s minutes" %minutes)
hours = minutes/60
print("total runtime in h: %s hours" %hours)

#####################################################################
# report loop controls
#####################################################################

if len(nonconverged)>0:
    print('%s uncompleted runs:' % len(nonconverged))
    print(nonconverged)
elif len(nonconverged)==0:
    print('all runs completed')
