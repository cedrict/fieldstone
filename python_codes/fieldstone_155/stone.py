import numpy as np
import time
import scipy.sparse as sps
from numpy import linalg as LA

print("-----------------------------")
print("---------- stone 155 --------")
print("-----------------------------")

###############################################################################

Lx=1
Ly=1
Ra=1e4
niter=1000
nnx=33
nny=33
tol=1e-6
beta=0.0005

###############################################################################

N=nnx*nny

hx=Lx/(nnx-1)
hy=Ly/(nny-1)

###############################################################################

print('Ra= ',Ra)
print('Lx= ',Lx)
print('Ly= ',Ly)
print('hx= ',hx)
print('hy= ',hy)
print('nnx=',nnx)
print('nny=',nny)
print('N=  ',N)

convfile=open('conv.ascii',"w")
statsfile=open('stats.ascii',"w")
avrgfile=open('avrgs.ascii',"w")

###############################################################################
# mesh nodes layout 
###############################################################################
start = time.time()

x=np.zeros(N,dtype=np.float64)
y=np.zeros(N,dtype=np.float64)

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter+=1
    #end for
#end for

print("Nodes setup: %.5f s" % (time.time() - start))

###############################################################################
# initial temperature
###############################################################################
start = time.time()

T=np.zeros(N,dtype=np.float64)
T_mem=np.zeros(N,dtype=np.float64)
dTdx=np.zeros(N,dtype=np.float64)
dTdy=np.zeros(N,dtype=np.float64)

for i in range(0,N):
    T[i]=1.-y[i]-0.01*np.cos(np.pi*x[i])*np.sin(np.pi*y[i])
    dTdx[i]=0.01*np.pi*np.sin(np.pi*x[i])*np.sin(np.pi*y[i])
#end for

T_mem[:]=T[:]

print("Initial temperature: %.5f s" % (time.time() - start))

###############################################################################
# build matrix
# MA: used for steps 1 and 2
# MB: used for steps 4
###############################################################################
start = time.time()

MA=np.zeros((N,N),dtype=np.float64)
MB=np.zeros((N,N),dtype=np.float64)

for j in range(0,nny):
    for i in range(0,nnx):
        k=j*nnx+i
        kW=j*nnx+(i-1)
        kE=j*nnx+(i+1)
        kN=(j+1)*nnx+i
        kS=(j-1)*nnx+i
        #---------------------------------------
        if i==0 or i==nnx-1 or j==0 or j==nny-1:
           MA[k,k]=1
        else:
           MA[k,k]=-2/hx**2-2/hy**2
           MA[k,kW]=1/hx**2
           MA[k,kE]=1/hx**2
           MA[k,kN]=1/hy**2
           MA[k,kS]=1/hy**2
        #end if
        #---------------------------------------
        if i==0 and j>0 and j<nny-1:
           MB[k,k]=1
           MB[k,kE]=-1
        elif i==nnx-1 and j>0 and j<nny-1:
           MB[k,k]=1
           MB[k,kW]=-1
        elif j==0 or j==nny-1:
           MB[k,k]=1
        else:
           MB[k,k]=-2/hx**2-2/hy**2
           MB[k,kW]=1/hx**2
           MB[k,kE]=1/hx**2
           MB[k,kN]=1/hy**2
           MB[k,kS]=1/hy**2
        #end if
        #---------------------------------------
    #end for
#end for

MA=sps.csr_matrix(MA)
MB=sps.csr_matrix(MB)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
u_mem=np.zeros(N,dtype=np.float64)
v_mem=np.zeros(N,dtype=np.float64)
psi_mem=np.zeros(N,dtype=np.float64)
omega_mem=np.zeros(N,dtype=np.float64)

for iter in range(0,niter):

    print('*************iter=',iter,'***************')

    ###########################################################################
    # step 1: solve vorticity equation
    ###########################################################################
    start = time.time()
    b=np.zeros(N,dtype=np.float64)

    b[:]=Ra*dTdx[:]

    for j in range(0,nny):
        for i in range(0,nnx):
            k=j*nnx+i
            if i==0 or i==nnx-1 or j==0 or j==nny-1:
               b[k]=0

    omega=sps.linalg.spsolve(MA,b)

    print("     -> omega (m,M) %.4f %.4f " %(np.min(omega),np.max(omega)))
   
    omega[:]=beta*omega[:]+(1-beta)*omega_mem[:] #relaxation

    print("Solve linear system omega: %.5f s" % (time.time() - start))

    ###########################################################################
    #step 2: solve stream function equation
    ###########################################################################
    start = time.time()

    psi=sps.linalg.spsolve(MA,-omega)

    print("     -> psi (m,M) %.4f %.4f " %(np.min(psi),np.max(psi)))

    psi[:]=beta*psi[:]+(1-beta)*psi_mem[:] #relaxation 

    print("Solve linear system psi: %.5f s" % (time.time() - start))

    ###########################################################################
    #step 3:  compute velocity field
    ###########################################################################
    start = time.time()

    u=np.zeros(N,dtype=np.float64)
    v=np.zeros(N,dtype=np.float64)

    for j in range(0,nny):
        for i in range(0,nnx):
            k=j*nnx+i
            kW=j*nnx+(i-1)
            kE=j*nnx+(i+1)
            kN=(j+1)*nnx+i
            kS=(j-1)*nnx+i
            #--------------------------------
            if i==0: #left
               v[k]=-(psi[kE]-psi[k])/hx
            elif i==nnx-1: #right   
               v[k]=-(psi[k]-psi[kW])/hx
            elif j==0: #bottom
               u[k]=(psi[kN]-psi[k])/hy
            elif j==nny-1: #top
               u[k]=(psi[k]-psi[kS])/hy
            else:
               u[k]= (psi[kN]-psi[kS])/(2*hy)
               v[k]=-(psi[kE]-psi[kW])/(2*hx)
            #end if
            #--------------------------------
        #end for
    #end for

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

    print("Compute velocity: %.5f s" % (time.time() - start))

    #u[:]=0
    #v[:]=0

    ###########################################################################
    #step 4: solve temperature equation
    ###########################################################################
    start = time.time()

    b=np.zeros(N,dtype=np.float64)

    b[:]=u[:]*dTdx[:]+v[:]*dTdy[:]

    for j in range(0,nny):
        for i in range(0,nnx):
            k=j*nnx+i
            if j==0:
               b[k]=1
            elif j==nny-1:
               b[k]=0

    T=sps.linalg.spsolve(MB,b)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    T[:]=beta*T[:]+(1-beta)*T_mem[:] #relaxation

    print("Solve linear system T: %.5f s" % (time.time() - start))

    ###########################################################################
    #step 5: compute temperature gradient
    ###########################################################################
    start = time.time()

    for j in range(0,nny):
        for i in range(0,nnx):
            k=j*nnx+i
            kW=j*nnx+(i-1)
            kE=j*nnx+(i+1)
            kN=(j+1)*nnx+i
            kS=(j-1)*nnx+i
            #-------------------------------
            if i==0 and j==0:               #SW corner
               dTdx[k]=(T[kE]-T[k])/hx
               dTdy[k]=(T[kN]-T[k])/hy
            elif i==0 and j==nny-1:         #NW corner
               dTdx[k]=(T[kE]-T[k])/hx
               dTdy[k]=(T[k]-T[kS])/hy
            elif i==nnx-1 and j==0:         #SE corner
               dTdx[k]=(T[k]-T[kW])/hx
               dTdy[k]=(T[kN]-T[k])/hy
            elif i==nnx-1 and j==nny-1:     #NE corner
               dTdx[k]=(T[k]-T[kW])/hx
               dTdy[k]=(T[k]-T[kS])/hy
            elif i==0:                      #left
               dTdx[k]=(T[kE]-T[k] )/hx
               dTdy[k]=(T[kN]-T[kS])/(2*hy)
            elif i==nnx-1:                  #right   
               dTdx[k]=(T[k]-T[kW] )/hx
               dTdy[k]=(T[kN]-T[kS])/(2*hy)
            elif j==0:                      #bottom
               dTdx[k]=(T[kE]-T[kW])/(2*hx)
               dTdy[k]= (T[kN]-T[k])/hy
            elif j==nny-1:                  #top
               dTdx[k]=(T[kE]-T[kW])/(2*hx)
               dTdy[k]= (T[k]-T[kS])/hy
            else:                           #interior
               dTdx[k]=(T[kE]-T[kW])/(2*hx)
               dTdy[k]=(T[kN]-T[kS])/(2*hy)
            #end if
            #-------------------------------
        #end for
    #end for

    print("     -> dTdx (m,M) %.4f %.4f " %(np.min(dTdx),np.max(dTdx)))
    print("     -> dTdy (m,M) %.4f %.4f " %(np.min(dTdy),np.max(dTdy)))

    print("Compute velocity: %.5f s" % (time.time() - start))

    np.savetxt('T_gradient.ascii',np.array([x,y,dTdx,dTdy]).T)

    #######################################################################

    statsfile.write("%e %e %e %e %e %e %e %e %e %f %f\n" %(iter,\
                                                     min(u),max(u),\
                                                     min(v),max(v),\
                                                     min(psi),max(psi),\
                                                     min(omega),max(omega),\
                                                     min(T),max(T)))

    avrgfile.write("%f %f %f %f %f %f\n" %(iter,\
                                           np.mean(abs(u)),
                                           np.mean(abs(v)),
                                           np.mean(psi),
                                           np.mean(omega),
                                           np.mean(T))) 

    #######################################################################

    xi_u=LA.norm(u-u_mem,2)#/LA.norm(u,2)
    xi_v=LA.norm(v-v_mem,2)#/LA.norm(v,2)
    xi_T=LA.norm(T-T_mem,2)#/LA.norm(v,2)
    xi_psi=LA.norm(psi-psi_mem,2)#/LA.norm(psi,2)
    xi_omega=LA.norm(omega-omega_mem,2)#/LA.norm(omega,2)

    convfile.write("%f %f %f %f %f %f\n" %(iter,xi_u,xi_v,xi_psi,xi_omega,xi_T))
    print("     -> xi_u= %.5e " %(xi_u))
    print("     -> xi_v= %.5e " %(xi_v)) 
    print("     -> xi_psi= %.5e " %(xi_psi))
    print("     -> xi_omega= %.5e " %(xi_omega)) 

    u_mem[:]=u[:]
    v_mem[:]=v[:]
    psi_mem[:]=psi[:]
    omega_mem[:]=omega[:]
    T_mem[:]=T[:]

    if xi_u<tol and xi_v<tol and xi_psi<tol and xi_omega<tol:
       print('**********')
       print('converged!')
       print('**********')
       break

#end for iter

###############################################################################
# export fields to vtu
###############################################################################
start = time.time()

m=4
nelx=nnx-1
nely=nny-1
nel=nelx*nely

icon =np.zeros((m,nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
#vtufile.write("<CellData Scalars='scalars'>\n")
#--
#vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
#for iel in range (0,nel):
#    vtufile.write("%10e\n" % p[iel])
#vtufile.write("</DataArray>\n")
#--
#vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='T gradient' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" %(dTdx[i],dTdy[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='omega' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %omega[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='psi' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %psi[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %T[i])
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
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*m))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
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

print("Export to vtu: %.5f s" % (time.time() - start))

#np.savetxt('omega.ascii',np.array([x,y,omega]).T)
#np.savetxt('psi.ascii',np.array([x,y,psi]).T)
#np.savetxt('velocity.ascii',np.array([x,y,u/cm*year,v/cm*year]).T)

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
