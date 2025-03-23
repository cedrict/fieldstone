import numpy as np
import time as timing
import sys as sys
import scipy.sparse as sps
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def NN(rq,sq):
    N0=0.25*(1.-rq)*(1.-sq)
    N1=0.25*(1.+rq)*(1.-sq)
    N2=0.25*(1.+rq)*(1.+sq)
    N3=0.25*(1.-rq)*(1.+sq)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def dNNdr(rq,sq):
    dNdr0=-0.25*(1.-sq)
    dNdr1=+0.25*(1.-sq)
    dNdr2=+0.25*(1.+sq)
    dNdr3=-0.25*(1.+sq)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def dNNds(rq,sq):
    dNds0=-0.25*(1.-rq)
    dNds1=-0.25*(1.+rq)
    dNds2=+0.25*(1.+rq)
    dNds3=+0.25*(1.-rq)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

sqrt3=np.sqrt(3.)
eps=1.e-10 

print("-----------------------------")
print("--------- stone 130 ---------")
print("-----------------------------")

m=4        # number of nodes making up an element
ndof=1     # number of degrees of freedom per node
Lx=5       # horizontal extent of the domain 
Ly=5       # vertical extent of the domain 
d=20       # diffusivity of species B
a=0.05     # growth rate of species A
b=1        # growth rate of species B
gamma=600  # kinetics
nstep=2000 # maximum number of timestep   
dt=1e-4    # time step

nelx=64
nely=nelx

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=nelx+1      # number of elements, x direction
nny=nely+1      # number of elements, y direction
NP=nnx*nny      # number of nodes
nel=nelx*nely   # number of elements, total
Nfem=2*NP*ndof  # Total number of degrees of freedom

niter=10
tol=1e-6

stats_AB_file=open('stats_AB.ascii',"w")
conv_AB_file=open('conv_AB.ascii',"w")

debug=False

#####################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('nelx=',nelx)
print('nely=',nely)
print('nel=',nel)
print('Nfem=',Nfem)
print('hx=',hx)
print('hy=',hy)
print("-----------------------------")

#####################################################################
# grid point setup 
#####################################################################
start = timing.time()

x=np.zeros(NP,dtype=np.float64)  # x coordinates
y=np.zeros(NP,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1
    #end for
#end for
        
print("setup: node coordinates %.3f s" % (timing.time() - start))

#####################################################################
# connectivity
#####################################################################
start = timing.time()

icon=np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1
    #end for
#end for

print("setup: connectivity array %.3f s" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################
start = timing.time()

A = np.zeros(NP,dtype=np.float64)
B = np.zeros(NP,dtype=np.float64)
Amem = np.zeros(NP,dtype=np.float64)
Bmem = np.zeros(NP,dtype=np.float64)

for i in range(0,NP):
    A[i]=a+b+random.uniform(-0.01,0.01)
    B[i]=b/(a+b)**2+random.uniform(-0.01,0.01)
#end for

if debug: 
   np.savetxt('A_init.ascii',np.array([x,y,A]).T,header='# x,y,A')
   np.savetxt('B_init.ascii',np.array([x,y,B]).T,header='# x,y,B')

print("setup: initial conditions %.3f s" % (timing.time() - start))

#####################################################################
# create necessary arrays 
#####################################################################

dNdx  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)   # shape functions derivatives
Avect = np.zeros(m,dtype=np.float64)   
Bvect = np.zeros(m,dtype=np.float64)   

#==============================================================================
# time stepping loop
#==============================================================================

time=0.

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    for iiter in range(0,niter): # Picard iterations loop

        print("    iter= ", iiter)

        #################################################################
        # build FE matrix
        #################################################################
        start = timing.time()

        A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64) # FE matrix 
        rhs   = np.zeros(Nfem,dtype=np.float64)          # FE rhs 
        B_mat=np.zeros((2,m),dtype=np.float64)           # gradient matrix B 
        N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
        jcb=np.zeros((2,2),dtype=np.float64)

        for iel in range (0,nel):

            a_el_A=np.zeros((m,m),dtype=np.float64)
            a_el_B=np.zeros((m,m),dtype=np.float64)
            b_el_A=np.zeros(m,dtype=np.float64)
            b_el_B=np.zeros(m,dtype=np.float64)
            Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
            MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 

            Avect=A[icon[:,iel]]
            Bvect=B[icon[:,iel]]

            for iq in [-1,1]:
                for jq in [-1,1]:

                    # position & weight of quad. point
                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    weightq=1.*1.

                    #evaluate basis fcts & derivatives
                    N=NN(rq,sq)
                    dNdr=dNNdr(rq,sq)
                    dNds=dNNds(rq,sq)
                    N_mat[:,0]=N[:]

                    # calculate jacobian matrix & inverse
                    jcb[0,0]=dNdr.dot(x[icon[:,iel]])
                    jcb[0,1]=dNdr.dot(y[icon[:,iel]])
                    jcb[1,0]=dNds.dot(x[icon[:,iel]])
                    jcb[1,1]=dNds.dot(y[icon[:,iel]])
                    jcob=np.linalg.det(jcb)
                    jcbi=np.linalg.inv(jcb)

                    Aq=N.dot(Amem[icon[:,iel]])
                    Bq=N.dot(Bmem[icon[:,iel]])

                    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
                    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

                    B_mat[0,:]=dNdx[:]
                    B_mat[1,:]=dNdy[:]

                    # compute mass matrix
                    MM=N_mat.dot(N_mat.T)*weightq*jcob

                    # compute diffusion matrix
                    Kd=B_mat.T.dot(B_mat)*weightq*jcob

                    a_el_A+=MM*(1+gamma*dt)+Kd*dt
                    b_el_A+=MM.dot(Avect) + gamma*N*(a+Aq**2*Bq)*weightq*jcob*dt

                    a_el_B+=MM+d*Kd*dt
                    b_el_B+=MM.dot(Bvect) + gamma*N*(b-Aq**2*Bq)*weightq*jcob*dt

                #end for
            #end for

            # assemble matrix A_mat and right hand side rhs
            for k1 in range(0,m):
                m1=icon[k1,iel]
                for k2 in range(0,m):
                    m2=icon[k2,iel]
                    A_mat[m1,m2]+=a_el_A[k1,k2]
                    A_mat[m1+NP,m2+NP]+=a_el_B[k1,k2]
                #end for
                rhs[m1]+=b_el_A[k1]
                rhs[m1+NP]+=b_el_B[k1]
            #end for
            #for k1 in range(0,m):
            #    m1=icon[k1,iel]+NP
            #    for k2 in range(0,m):
            #        m2=icon[k2,iel]+NP
            #        A_mat[m1,m2]+=a_el_B[k1,k2]
            #    #end for
            #    rhs[m1]+=b_el_B[k1]
            #end for

        #end for iel

        print("    building matrix and rhs: %.3f s" % (timing.time() - start))

        #export matrix nonzero structure
        #plt.spy(A_mat, markersize=2.5)
        #plt.savefig('matrix.png', bbox_inches='tight')
        #plt.clf()

        #################################################################
        # solve system
        #################################################################
        start = timing.time()

        sol = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

        Asol=sol[0:NP]
        Bsol=sol[NP:Nfem]

        print("     -> A (m,M) %.4f %.4f " %(np.min(Asol),np.max(Asol)))
        print("     -> B (m,M) %.4f %.4f " %(np.min(Bsol),np.max(Bsol)))

        print("    solve time: %.3f s" % (timing.time() - start))

        #################################################################
        # assess nl convergence 
        #################################################################

        chi_A=LA.norm(Asol-Amem,2) # A convergence indicator
        chi_B=LA.norm(Bsol-Bmem,2) # B convergence indicator

        print('     -> convergence A,B: %.3e %.3e | tol= %.2e' %(chi_A,chi_B,tol))

        conv_AB_file.write("%f %10e %10e %10e\n" %(istep+iiter/100,chi_A,chi_B,tol))
        conv_AB_file.flush()

        if chi_A<tol and chi_B<tol:
           A[:]=Asol[:]
           B[:]=Bsol[:]
           print('     ***converged***')
           break

        Amem[:]=Asol[:]
        Bmem[:]=Bsol[:]

    #end for iter

    stats_AB_file.write("%e %e %e %e %e\n" % (istep*dt,np.min(A),np.max(A),np.min(B),np.max(B)))
    stats_AB_file.flush()

    #################################################################
    # compute averages
    #################################################################
    start = timing.time()

    avrg_A=0.
    avrg_B=0.
    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.
                N=NN(rq,sq)
                dNdr=dNNdr(rq,sq)
                dNds=dNNds(rq,sq)
                jcb[0,0]=dNdr.dot(x[icon[:,iel]])
                jcb[0,1]=dNdr.dot(y[icon[:,iel]])
                jcb[1,0]=dNds.dot(x[icon[:,iel]])
                jcb[1,1]=dNds.dot(y[icon[:,iel]])
                jcob=np.linalg.det(jcb)
                Aq=N.dot(A[icon[:,iel]])
                Bq=N.dot(B[icon[:,iel]])
                avrg_A+=Aq*weightq*jcob
                avrg_B+=Bq*weightq*jcob

    avrg_A=np.sqrt(avrg_A/Lx/Ly)
    avrg_B=np.sqrt(avrg_B/Lx/Ly)

    print("     -> avrg_A= %.8f ; avrg_B= %.8f" %(avrg_A,avrg_B))

    print("compute averages: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute gradients
    #####################################################################
    start = timing.time()

    dAdx=np.zeros(NP,dtype=np.float64)
    dAdy=np.zeros(NP,dtype=np.float64)
    dBdx=np.zeros(NP,dtype=np.float64)
    dBdy=np.zeros(NP,dtype=np.float64)
    count=np.zeros(NP,dtype=np.float64)

    rVnodes=[-1,1,1,-1]
    sVnodes=[-1,-1,1,1]

    for iel in range (0,nel):
        for i in range(0,m):
            inode=icon[i,iel]
            rq=rVnodes[i]
            sq=sVnodes[i]
            dNdr=dNNdr(rq,sq)
            dNds=dNNds(rq,sq)
            jcb[0,0]=dNdr.dot(x[icon[:,iel]])
            jcb[0,1]=dNdr.dot(y[icon[:,iel]])
            jcb[1,0]=dNds.dot(x[icon[:,iel]])
            jcb[1,1]=dNds.dot(y[icon[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
            dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]
            dAdx[inode]+=dNdx.dot(A[icon[:,iel]])
            dAdy[inode]+=dNdy.dot(A[icon[:,iel]])
            dBdx[inode]+=dNdx.dot(B[icon[:,iel]])
            dBdy[inode]+=dNdy.dot(B[icon[:,iel]])
            count[inode]+=1
        #end for
    #end for
    dAdx/=count
    dAdy/=count
    dBdx/=count
    dBdy/=count

    print("     -> dAdx (m,M) %.4f %.4f " %(np.min(dAdx),np.max(dAdx)))
    print("     -> dAdy (m,M) %.4f %.4f " %(np.min(dAdy),np.max(dAdy)))
    print("     -> dBdx (m,M) %.4f %.4f " %(np.min(dBdx),np.max(dBdx)))
    print("     -> dBdy (m,M) %.4f %.4f " %(np.min(dBdy),np.max(dBdy)))

    print("compute gradients: %.3f s" % (timing.time() - start))

    #####################################################################
    # export to vtu 
    #####################################################################
    start = timing.time()

    filename = 'solution_{:04d}.vtu'.format(istep) 
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NP):
        vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    #vtufile.write("<CellData Scalars='scalars'>\n")
    #vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='A' Format='ascii'> \n")
    for i in range(0,NP):
        vtufile.write("%e \n" % A[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='B' Format='ascii'> \n")
    for i in range(0,NP):
        vtufile.write("%e \n" % B[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='A^2B' Format='ascii'> \n")
    for i in range(0,NP):
        vtufile.write("%e \n" % (A[i]*A[i]*B[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='grad(A)' Format='ascii'> \n")
    for i in range(0,NP):
        vtufile.write("%10f %10f %10f \n" %(dAdx[i],dAdy[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='grad(B)' Format='ascii'> \n")
    for i in range(0,NP):
        vtufile.write("%10f %10f %10f \n" %(dBdx[i],dBdy[i],0.))
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

    print("export to vtu: %.3f s" % (timing.time() - start))

    time+=dt
    
#end for

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
