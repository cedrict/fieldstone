import numpy as np
import time
import scipy.sparse as sps
from numpy import linalg as LA

###############################################################################

def u_th(x,y):
    return y*(1-y)

def v_th(x,y):
    return 0

def psi_th(x,y):
    return y**2/2-y**3/3

def omega_th(x,y):
    return -1+2*y

###############################################################################

Re=400                 # Reynolds Number
Lx=5                   # Length of domain
Ly=1                   # Height of domain
nny=21                 # Number of points in x direction 
nnx=101                # Number of points in y direction
hx=Lx/(nnx-1)          # Step size in length
hy=Ly/(nny-1)          # Step size in height
dt=0.002               # Time step size
tfinal=5               # Simulation end time
nstep=int(tfinal/dt)   # Number of time steps
u0=1                   # Inlet uniform velocity
niter=250
tol=1e-8

nstep=1

beta = 0.33 # 1=no relaxation

#physics='ssNS'
physics='Stokes'

###############################################################################

N=nnx*nny

print('Re= ',Re)
print('Lx= ',Lx)
print('Ly= ',Ly)
print('hx= ',hx)
print('hy= ',hy)
print('nnx=',nnx)
print('nny=',nny)
print('N=  ',N)

###############################################################################

statsfile=open('stats.ascii',"w")
convfile=open('conv.ascii',"w")
errfile=open('errors.ascii',"w")

###############################################################################
# mesh nodes layout 
###############################################################################
start = time.time()

x=np.zeros(N,dtype=np.float64)
y=np.zeros(N,dtype=np.float64)
left=np.zeros(N,dtype=bool)
right=np.zeros(N,dtype=bool)
bottom=np.zeros(N,dtype=bool)
top=np.zeros(N,dtype=bool)

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        if i==0     : left[counter]=True
        if i==nnx-1 : right[counter]=True
        if j==0     and i>0 and i<nnx-1: bottom[counter]=True
        if j==nny-1 and i>0 and i<nnx-1: top[counter]=True
        counter+=1

print("Nodes setup: %.5f s" % (time.time() - start))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
omega=np.zeros(N,dtype=np.float64)
psi=np.zeros(N,dtype=np.float64)
umem=np.zeros(N,dtype=np.float64)
vmem=np.zeros(N,dtype=np.float64)
psimem=np.zeros(N,dtype=np.float64)
omegamem=np.zeros(N,dtype=np.float64)

for istep in range(0,nstep):

    for iter in range(0,niter):

        print('********* iter=',iter,'*************')

        #######################################################################
        # step 1: solve Delta Psi = -omega
        #######################################################################
        start = time.time()

        A=np.zeros((N,N),dtype=np.float64)
        b=np.zeros(N,dtype=np.float64)

        for j in range(0,nny):
            for i in range(0,nnx):

                k=j*nnx+i
                kW=j*nnx+(i-1)
                kE=j*nnx+(i+1)
                kN=(j+1)*nnx+i
                kS=(j-1)*nnx+i

                if left[i]:
                   A[k,k]=1
                   #b[k]=u0*y[k]
                   b[k]=y[k]**2/2-y[k]**3/3
                elif bottom[k]:
                   A[k,k]=1
                   b[k]=0
                elif top[k]:
                   A[k,k]=1
                   b[k]=1/6  #u0*Ly
                elif right[k]:
                   A[k,k]=1
                   A[k,kW]=-1
                   b[k]=0
                else: 
                   A[k,k]=-2/hx**2-2/hy**2
                   A[k,kW]=1/hx**2
                   A[k,kE]=1/hx**2
                   A[k,kN]=1/hy**2
                   A[k,kS]=1/hy**2
                   b[k]=-omega[k]
                #end if
            #end for
        #end for

        print("Build matrix: %.5f s" % (time.time() - start))

        #######################################################################
        start = time.time()

        psi=sps.linalg.spsolve(sps.csr_matrix(A),b)

        print("     -> psi (m,M) %.4f %.4f " %(np.min(psi),np.max(psi)))

        print("Solve linear system psi: %.5f s" % (time.time() - start))

        #relaxation 
        psi[:]=beta*psi[:]+(1-beta)*psimem[:]

        #######################################################################
        # step 2: compute velocity field 
        #######################################################################
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
                if i==0: #left
                   u[k]=y[k]*(Ly-y[k]) #u0
                   v[k]=0
                elif i==nnx-1 and j>0 and j<nny-1: #right   
                   u[k]=(psi[kN]-psi[kS])/(2*hy)
                   v[k]=0
                elif j==0: #bottom
                   u[k]=0
                   v[k]=0
                elif j==nny-1: #top
                   u[k]=0
                   v[k]=0
                else:
                   u[k]= (psi[kN]-psi[kS])/(2*hy)
                   v[k]=-(psi[kE]-psi[kW])/(2*hx)
                #end if
            #end for
        #end for

        print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
        print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

        vel_max = np.max(np.sqrt(u**2+v**2)) 
        Pe = (vel_max*dt)/(min(hx,hy)**2) # Peclet number

        print('     -> Pe=',Pe)

        #######################################################################
        # step 3: solve vorticity advection-diffusion eq 
        #######################################################################
        start = time.time()

        A=np.zeros((N,N),dtype=np.float64)
        b=np.zeros(N,dtype=np.float64)

        for j in range(0,nny):
            for i in range(0,nnx):

                k=j*nnx+i
                kW=j*nnx+(i-1)
                kE=j*nnx+(i+1)
                kN=(j+1)*nnx+i
                kS=(j-1)*nnx+i

                if left[i]:
                   A[k,k]=1
                   b[k]=-1+2*y[k]
                elif bottom[k]:
                   A[k,k]=1
                   b[k]=2/hy**2*(psi[k]-psi[kN])
                elif top[k]:
                   A[k,k]=1
                   b[k]=2/hy**2*(psi[k]-psi[kS])
                elif right[k]:
                   A[k,k]=1
                   A[k,kW]=-1
                   b[k]=0
                else: 
                   #time dependent N-S
                   #A[k,k]=1-2*dt/hx**2/Re-2*dt/hy**2/Re
                   #A[k,kW]=u[k]*dt/2/hx-dt/hx**2/Re
                   #A[k,kS]=v[k]*dt/2/hy-dt/hy**2/Re
                   #A[k,kE]=-u[k]*dt/2/hx-dt/hx**2/Re
                   #A[k,kN]=-v[k]*dt/2/hy-dt/hy**2/Re
                   #b[k]=0 #omega[k]

                   #steady state N-S
                   if physics=='ssNS':
                      A[k,k]=2/hx**2/Re+2/hy**2/Re
                      A[k,kW]=-u[k]/2/hx-1/hx**2/Re
                      A[k,kS]=-v[k]/2/hy-1/hy**2/Re
                      A[k,kE]=u[k]/2/hx-1/hx**2/Re
                      A[k,kN]=v[k]/2/hy-1/hy**2/Re

                   #Stokes 
                   if physics=='Stokes':
                      A[k,k]=2/hx**2+2/hy**2
                      A[k,kW]=-1/hx**2
                      A[k,kE]=-1/hx**2
                      A[k,kS]=-1/hy**2
                      A[k,kN]=-1/hy**2

                #end if
            #end for
        #end for

        print("Build matrix: %.5f s" % (time.time() - start))

        #######################################################################
        start = time.time()

        omega=sps.linalg.spsolve(sps.csr_matrix(A),b)

        print("     -> omega (m,M) %.4f %.4f " %(np.min(omega),np.max(omega)))

        print("Solve linear system omega: %.5f s" % (time.time() - start))

        #relaxation 
        omega[:]=beta*omega[:]+(1-beta)*omegamem[:]

        #######################################################################

        statsfile.write("%e %e %e %e %e %e %e %e %e\n" %(istep+iter/100,\
                                                         min(u),max(u),\
                                                         min(v),max(v),\
                                                         min(psi),max(psi),\
                                                         min(omega),max(omega)))

        #######################################################################
        psi_analytical=np.zeros(N,dtype=np.float64)
        omega_analytical=np.zeros(N,dtype=np.float64)
   
        for i in range(0,N):

            psi_analytical[i]=psi_th(x[i],y[i])
            omega_analytical[i]=omega_th(x[i],y[i])

        err_psi=LA.norm(psi-psi_analytical,2)
        err_omega=LA.norm(omega-omega_analytical,2)

        errfile.write("%e %e %e\n" %(istep+iter/100,err_psi,err_omega))

        #######################################################################

        xi_u=LA.norm(u-umem,2)/LA.norm(u,2)
        xi_v=LA.norm(v-vmem,2)/LA.norm(v,2)
        xi_psi=LA.norm(psi-psimem,2)/LA.norm(psi,2)
        xi_omega=LA.norm(omega-omegamem,2)/LA.norm(omega,2)

        convfile.write("%e %e %e %e %e\n" %(istep+iter/100,xi_u,xi_v,xi_psi,xi_omega))
        print("     -> xi_u= %.5e " %(xi_u))
        print("     -> xi_v= %.5e " %(xi_v)) 
        print("     -> xi_psi= %.5e " %(xi_psi))
        print("     -> xi_omega= %.5e " %(xi_omega)) 

        umem[:]=u[:]
        vmem[:]=v[:]
        psimem[:]=psi[:]
        omegamem[:]=omega[:]

        if xi_u<tol and xi_v<tol and xi_psi<tol and xi_omega<tol:
           print('**********')
           print('converged!')
           print('**********')
           break

    #end for iter

#end for istep

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
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" %(u_th(x[i],y[i]),v_th(x[i],y[i]),0))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (err)' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" %(u[i]-u_th(x[i],y[i]),v[i]-v_th(x[i],y[i]),0))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='omega' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %omega[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='omega (th)' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %omega_th(x[i],y[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='omega (err)' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" % (omega[i]-omega_th(x[i],y[i])))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='psi' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %psi[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='psi (th)' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" % psi_th(x[i],y[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='psi (err)' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" % (psi[i]-psi_th(x[i],y[i])))
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
#np.savetxt('psi.ascii',np.array([x,y,Psi]).T)
#np.savetxt('velocity.ascii',np.array([x,y,u/cm*year,v/cm*year]).T)

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

