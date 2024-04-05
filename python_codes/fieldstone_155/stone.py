import numpy as np
import time 
import sys as sys
import scipy.sparse as sps
from numpy import linalg as LA

print("-----------------------------")
print("---------- stone 155 --------")
print("-----------------------------")

###############################################################################

# allowing for argument parsing through command line
if int(len(sys.argv) == 3):
   nnx = int(sys.argv[1])
   Ra = int(sys.argv[2])
else:
   nnx=33
   Ra=806

Lx=1
Ly=1
nstep=1000
nny=nnx
tol=1e-7
CFL_nb=0.5
every=100
dt_min=1e-5

steady_state=True

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
vrmsfile=open('vrms.ascii',"w")
Nufile=open('Nu.ascii',"w")

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
# mesh connectivity (from paraview export)
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
        icon[0,counter]=i+j*(nelx+1)
        icon[1,counter]=i+1+j*(nelx+1)
        icon[2,counter]=i+1+(j+1)*(nelx+1)
        icon[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("connectivity setup: %.5f s" % (time.time() - start))

###############################################################################
# initial temperature
###############################################################################
start = time.time()

T=np.zeros(N,dtype=np.float64)
T_mem=np.zeros(N,dtype=np.float64)
dTdx=np.zeros(N,dtype=np.float64)
dTdy=np.zeros(N,dtype=np.float64)
dTdx0=np.zeros(N,dtype=np.float64)
dTdy0=np.zeros(N,dtype=np.float64)

amplitude=0.01

for i in range(0,N):
    T[i]=1.-y[i]-amplitude*np.cos(np.pi*x[i])*np.sin(np.pi*y[i])
    dTdx[i]=amplitude*np.pi*np.sin(np.pi*x[i])*np.sin(np.pi*y[i])
    dTdy[i]=-1-amplitude*np.pi*np.cos(np.pi*x[i])*np.cos(np.pi*y[i])
#end for

T_mem[:]=T[:]
dTdx0[:]=dTdx[:]
dTdy0[:]=dTdy[:]

print("Initial temperature: %.5f s" % (time.time() - start))

###############################################################################
# build matrix
# MA: used for steps 1 and 2
###############################################################################
start = time.time()

MA=np.zeros((N,N),dtype=np.float64)

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
    #end for
#end for

MA=sps.csr_matrix(MA)
    
print("Compute MA matrix: %.5f s" % (time.time() - start))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

total_time=0
u_mem=np.zeros(N,dtype=np.float64)
v_mem=np.zeros(N,dtype=np.float64)
psi_mem=np.zeros(N,dtype=np.float64)
omega_mem=np.zeros(N,dtype=np.float64)

for istep in range(0,nstep):

    print('*************istep=',istep,'***************')

    ###########################################################################
    # step 1: solve vorticity equation
    ###########################################################################
    start = time.time()
    b=np.zeros(N,dtype=np.float64)

    b[:]=-Ra*dTdx[:]

    for j in range(0,nny):
        for i in range(0,nnx):
            k=j*nnx+i
            if i==0 or i==nnx-1 or j==0 or j==nny-1:
               b[k]=0

    omega=sps.linalg.spsolve(MA,b)

    print("     -> omega (m,M) %.4f %.4f " %(np.min(omega),np.max(omega)))

    print("Solve linear system omega: %.5f s" % (time.time() - start))

    ###########################################################################
    #step 2: solve stream function equation
    ###########################################################################
    start = time.time()

    psi=sps.linalg.spsolve(MA,-omega)

    print("     -> psi (m,M) %.4f %.4f " %(np.min(psi),np.max(psi)))

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
    # compute time step
    ###########################################################################

    if steady_state:

       dt=1

    else:

       dt=CFL_nb*min(hx,hy)/np.max(np.sqrt(u**2+v**2))

       if istep<10:
          dt=dt_min
       elif istep<50 and dt>dt_mem:
          dt=min(1.25*dt_mem,dt)

       dt_mem=dt

       print('     -> dt  = %.6f' %dt)

    total_time+=dt

    ###########################################################################
    #step 4: solve temperature equation
    ###########################################################################
    start = time.time()

    MB=np.zeros((N,N),dtype=np.float64)
    b=np.zeros(N,dtype=np.float64)

    for j in range(0,nny):
        for i in range(0,nnx):
            k=j*nnx+i
            kW=j*nnx+(i-1)
            kE=j*nnx+(i+1)
            kN=(j+1)*nnx+i
            kS=(j-1)*nnx+i
            #---------------------------------------
            if i==0 and j>0 and j<nny-1:            # left
               MB[k,k]=1
               MB[k,kE]=-1
               b[k]=0
            elif i==nnx-1 and j>0 and j<nny-1:      # right
               MB[k,k]=1
               MB[k,kW]=-1
               b[k]=0
            elif j==0:                              # bottom
               MB[k,k]=1
               b[k]=1
            elif j==nny-1:                          # top
               MB[k,k]=1
               b[k]=0
            else:                                   #internal

               if steady_state:
                  MB[k,k]=2*dt/hx**2+2*dt/hy**2
                  MB[k,kW]=-u[k]*dt/2/hx-dt/hx**2
                  MB[k,kS]=-v[k]*dt/2/hy-dt/hy**2
                  MB[k,kE]= u[k]*dt/2/hx-dt/hx**2
                  MB[k,kN]= v[k]*dt/2/hy-dt/hy**2

               #fully implicit
               #MB[k,k]=1+2*dt/hx**2+2*dt/hy**2
               #MB[k,kW]=-u[k]*dt/2/hx-dt/hx**2
               #MB[k,kS]=-v[k]*dt/2/hy-dt/hy**2
               #MB[k,kE]= u[k]*dt/2/hx-dt/hx**2
               #MB[k,kN]= v[k]*dt/2/hy-dt/hy**2
               #b[k]=T[k]
               #crank-nicolson
               #MB[k,k]=1+dt/hx**2+dt/hy**2
               #MB[k,kW]=-u[k]*dt/4/hx-dt/2/hx**2
               #MB[k,kS]=-v[k]*dt/4/hy-dt/2/hy**2
               #MB[k,kE]= u[k]*dt/4/hx-dt/2/hx**2
               #MB[k,kN]= v[k]*dt/4/hy-dt/2/hy**2
               #b[k]=(1-dt/hx**2-dt/hy**2)*T[k]\
               #    +( u[k]*dt/4/hx+dt/2/hx**2)*T[kW]\
               #    +( v[k]*dt/4/hy+dt/2/hy**2)*T[kS]\
               #    +(-u[k]*dt/4/hx+dt/2/hx**2)*T[kE]\
               #    +(-v[k]*dt/4/hy+dt/2/hy**2)*T[kN]

            #end if
            #---------------------------------------
        #end for
    #end for

    MB=sps.csr_matrix(MB)

    T=sps.linalg.spsolve(MB,b)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

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

    #np.savetxt('T_gradient.ascii',np.array([x,y,dTdx,dTdy]).T)

    #######################################################################
    # compute vrms
    #######################################################################

    vrms=0
    for iel in range(0,nel):
        uc=(u[icon[0,iel]]+u[icon[1,iel]]+u[icon[2,iel]]+u[icon[3,iel]])*0.25
        vc=(v[icon[0,iel]]+v[icon[1,iel]]+v[icon[2,iel]]+v[icon[3,iel]])*0.25
        vrms+=(uc**2+vc**2)*hx*hy

    vrms=np.sqrt(vrms)

    print("     -> vrms %.4f " %(vrms))

    vrmsfile.write("%e %e \n" %(total_time,vrms)) ; vrmsfile.flush()

    #######################################################################
    # compute Nu
    #######################################################################

    Nu=0
    for iel in range(0,nel):
        if y[icon[3,iel]]>0.9999: #top cell
           Nu-=(dTdy[icon[3,iel]]+dTdy[icon[2,iel]])/2*hx
    
    print("     -> Nu %.4f " %(Nu))

    Nufile.write("%e %e \n" %(total_time,Nu))
    Nufile.flush()

    #######################################################################

    statsfile.write("%e %e %e %e %e %e %e %e %e %f %f\n" %(total_time,\
                                                     min(u),max(u),\
                                                     min(v),max(v),\
                                                     min(psi),max(psi),\
                                                     min(omega),max(omega),\
                                                     min(T),max(T)))

    avrgfile.write("%e %e %e %e %e %e\n" %(total_time,\
                                           np.mean(abs(u)),
                                           np.mean(abs(v)),
                                           np.mean(psi),
                                           np.mean(omega),
                                           np.mean(T)-0.5)) 

    avrgfile.flush()
    statsfile.flush()

    ###########################################################################
    # export fields to vtu
    ###########################################################################
    start = time.time()

    if istep%every==0:
       filename = 'solution_{:04d}.vtu'.format(istep)
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
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='T gradient' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e %10e %10e \n" %(dTdx[i],dTdy[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='T gradient (t=0)' Format='ascii'> \n")
       for i in range(0,N):
           vtufile.write("%10e %10e %10e \n" %(dTdx0[i],dTdy0[i],0.))
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

    #######################################################################

    xi_u=LA.norm(u-u_mem,2)/LA.norm(u,2)
    xi_v=LA.norm(v-v_mem,2)/LA.norm(v,2)
    xi_T=LA.norm(T-T_mem,2)/LA.norm(v,2)
    xi_psi=LA.norm(psi-psi_mem,2)/LA.norm(psi,2)
    xi_omega=LA.norm(omega-omega_mem,2)/LA.norm(omega,2)

    convfile.write("%e %e %e %e %e %e\n" %(total_time,xi_u,xi_v,xi_psi,xi_omega,xi_T))
    convfile.flush()
    print("     -> xi_u= %.5e " %(xi_u))
    print("     -> xi_v= %.5e " %(xi_v)) 
    print("     -> xi_psi= %.5e " %(xi_psi))
    print("     -> xi_omega= %.5e " %(xi_omega)) 

    u_mem[:]=u[:]
    v_mem[:]=v[:]
    psi_mem[:]=psi[:]
    omega_mem[:]=omega[:]
    T_mem[:]=T[:]

    if (xi_u<tol and xi_v<tol and xi_psi<tol and xi_omega<tol) or vrms<1e-2:
       print('**********')
       print('converged!')
       print('**********')
       break

#end for istep
       
print('sssss:',Ra,nelx,Nu,vrms)

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
