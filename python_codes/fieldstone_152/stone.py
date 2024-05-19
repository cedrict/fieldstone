import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import math 

###########################################################    
# Fonction vitesse

def compute_corner_flow_velocity(x,y,l1,l2,l3,angle,v0,Lx,Ly):
    v1=-v0
    theta0=angle
    theta1=np.pi-theta0
    l4=l3*np.tan(theta0)
    A0 = (- theta0 * np.sin(theta0))/(theta0**2-np.sin(theta0)**2 ) *v0 
    B0=0
    C0=(np.sin(theta0)-theta0*np.cos(theta0))/(theta0**2-np.sin(theta0)**2 ) * v0
    D0=-A0
    A1 =1./(theta1**2-np.sin(theta1)**2 ) * \
        ( -v0*theta1*np.sin(theta1)-v1*theta1*np.cos(theta1)*(np.sin(theta1)+theta1*np.cos(theta1))\
        +v1*(np.cos(theta1)-theta1*np.sin(theta1))*theta1*np.sin(theta1) )   
    B1=0
    C1=1./(theta1**2-np.sin(theta1)**2 ) * \
       ( v0*(np.sin(theta1)-theta1*np.cos(theta1)) + v1*theta1**2*np.cos(theta1)*np.sin(theta1) \
       - v1*(np.cos(theta1)-theta1*np.sin(theta1))*(np.sin(theta1)-theta1*np.cos(theta1)) )   
    D1=-A1

    u=0.
    v=0.

    #------------------------
    # slab left 
    #------------------------
    if y>=Ly-l1 and x<=l3:
       u=v0
       v=0.

    #------------------------
    # slab 
    #------------------------
    if x>=l3 and y<=Ly+l4-x*np.tan(theta0) and y>=Ly+l4-x*np.tan(theta0)-l1:
       u=v0*np.cos(theta0)
       v=-v0*np.sin(theta0)

    #------------------------
    # overriding plate
    #------------------------
    if y>Ly+l4-x*np.tan(theta0) and y>Ly-l2:
       u=0.0
       v=0.0

    #------------------------
    # wedge
    #------------------------
    xC=l3+l2/np.tan(theta0)
    yC=Ly-l2
    if x>xC and y<yC:
       xt=x-xC 
       yt=yC-y 
       theta=np.arctan(yt/xt) 
       r=np.sqrt((xt)**2+(yt)**2)
       if theta<theta0:
          # u_r=f'(theta)
          ur = A0*np.cos(theta)-B0*np.sin(theta) +\
               C0* (np.sin(theta)+theta*np.cos(theta)) + D0 * (np.cos(theta)-theta*np.sin(theta))
          # u_theta=-f(theta)
          utheta=- ( A0*np.sin(theta) + B0*np.cos(theta) + C0*theta*np.sin(theta) + D0*theta*np.cos(theta))
          ur=-ur
          utheta=-utheta
          u=  ur*np.cos(theta)-utheta*np.sin(theta)
          v=-(ur*np.sin(theta)+utheta*np.cos(theta)) # because of reverse orientation

    #------------------------
    # under subducting plate
    #------------------------
    xD=l3
    yD=Ly-l1
    if y<yD and y<Ly+l4-x*np.tan(theta0)-l1:
       xt=xD-x 
       yt=yD-y 
       theta=np.arctan2(yt,xt) #!; write(6548,*) theta/pi*180
       #r=np.sqrt((xt)**2+(yt)**2)
       #u_r=f'(theta)
       ur = A1*np.cos(theta) - B1*np.sin(theta) + C1* (np.sin(theta)+theta*np.cos(theta)) \
            + (D1-v1) * (np.cos(theta)-theta*np.sin(theta))
       #u_theta=-f(theta)
       utheta=- ( A1*np.sin(theta) + B1*np.cos(theta) + C1*theta*np.sin(theta) + (D1-v1)*theta*np.cos(theta))
       ur=-ur
       utheta=-utheta
       u=-(ur*np.cos(theta)-utheta*np.sin(theta))
       v=-(ur*np.sin(theta)+utheta*np.cos(theta)) #! because of reverse orientation

    return u,v

###########################################################    
# Définition des fonctions 

def plot_field(x,y,T0, T1, field,filename,nfig): 
    plt.figure(nfig).clear()
    plt.figure(nfig)    
    plt.title(filename)
    plt.xlabel('Distance x (km)')
    plt.ylabel('Distance y (km)')
    plt.scatter(x,y,c=field,cmap='coolwarm',s=1) 
    plt.clim(T0, T1)
    plt.colorbar()
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()

###########################################################    
# Données initiales 
kappa = 1e-6 
year=3600*24*365.25 
Lx = 660e3 
Ly = 600e3 
nnx = 67*3 
nny = 61*3 
wp = 1e4 
N = nnx*nny 
dx=Lx/(nnx-1) 
dy=Ly/(nny-1)
xmin=0
xmax= Lx
ymin=0
ymax=Ly
hcond=3    #heat conductivity
hcapa=1250 #heat capacity
rho=3300   #density
l1=1000.e3 #m 
l2=50.e3 #m ##épaisseur riding plate 
l3=0.e3 #m ##?? 
vslab=5e-2/year #vitesse de plongement du slab
angle=45./180.*np.pi #rad ##angle de plongement du slab (et vitesse)
limplate = 50e3 #m ##limite de la plaque sup 

###########################################################    
# build mesh coordinates

x=np.zeros(N, dtype=np.float64)
y=np.zeros(N, dtype=np.float64)
for j in range(0,nny):
    for i in range(0,nnx):
        x[j*nnx+i]=xmin+i*dx
        y[j*nnx+i]=ymin+j*dy

###########################################################
#Temperatures 

Ts = 273 #°K 
To = 1573 #°K
grad = (To-Ts)/50e3 #°K/m 

###########################################################
# Calculs vitesses en tout point 

u = np.zeros(N)
v = np.zeros(N)
for i in range(0,N):
    u[i],v[i]=compute_corner_flow_velocity(x[i],y[i],l1,l2,l3,angle,vslab,Lx,Ly)

plot_field(x*1e-3, y*1e-3, max(u) , min(u), u, 'Vitesse_u.pdf', 1)
plot_field(x*1e-3, y*1e-3, max(v) , min(v), v, 'Vitesse_v.pdf', 2)

###########################################################
# Variables temporelles 

C = 0.5 #nombre de courant
dt = C*min(dx,dy)/max(max(u),max(v)) #s
t50 = 50e6*year 

###########################################################
# Conditions aux limites 

cl_bl = np.zeros(N, dtype= bool) #vecteur booléen pour les conditions limites 
cl_val= np.zeros(N) #vecteur de valeur pour les conditions limites 

tolx=1e-6*Lx #CED
toly=1e-6*Ly #CED

for i in range(N): 
    if x[i] < tolx : #slab inflow
        cl_bl[i] = True 
        cl_val[i] = Ts + (To-Ts)*math.erf((ymax-y[i])/(2*math.sqrt(kappa*t50)))
    elif abs(y[i]-ymax)<toly : #fixed surface
        cl_bl[i] = True 
        cl_val[i] = Ts 
    elif abs(x[i]-xmax)<tolx and y[i] >= ymax-limplate : #gradient 
        cl_bl[i] = True
        cl_val[i] = Ts + (ymax-y[i])*grad 
    elif abs(x[i]-xmax)<tolx and u[i]< 0 : #mantle inflow 
        cl_bl[i] = True
        cl_val[i] = To 
    
plot_field(x*1e-3, y*1e-3, 0 , To, cl_val, 'Conditions_limites.pdf', 3)

###########################################################
#Matrice et vecteur 

sx = kappa/(dx**2)
sy = kappa/(dy**2) 
A = np.zeros((N,N),dtype=np.float64)
b = np.zeros(N,dtype=np.float64)

for j in range(0,nny) :
    for i in range(0,nnx) :
        k=j*nnx+i
        cu = u[k]/(2*dx)
        cv = v[k]/(2*dy)
        if cl_bl[k] : #Conditions limites 
           A[k,k] = 1 
           b[k] = cl_val[k]
        else:
           if i==nnx-1: #right boundary node
              k_left=j*nnx+i-1
              k_leftleft=j*nnx+i-2
              A[k,k]+= -sy+2*cu
              A[k,k_left]+= 2*sy-2*cu
              A[k,k_leftleft]+= -sy
           else: #inside node
              k_left=j*nnx+i-1
              k_right=j*nnx+i+1
              A[k,k]+=2*sx  
              A[k,k_left]+=-cu-sx
              A[k,k_right]+=cu-sx

           if j==0: #bottom boundary node
              k_top=(j+1)*nnx+i
              k_toptop=(j+2)*nnx+i
              A[k,k]+=-2*cv-sy
              A[k,k_top]+=2*cv+2*sy
              A[k,k_toptop]+=-sy
           else: #inside node
              k_bot=(j-1)*nnx+i
              k_top=(j+1)*nnx+i
              A[k,k]+=2*sy
              A[k,k_bot]+=-cv-sy
              A[k,k_top]+=cv-sy
           #end if
        #end if
    #end for
#end for

T = sps.linalg.spsolve(sps.csr_matrix(A), b)

plot_field(x, y, Ts, To, T, 'stationnaire.pdf', 6)

###########################################################
### Température du slab à 60 km , Approximation 

ybench = ymax - 60*1e3 #m
xbench = 60*1e3 #m #45° 
eps = 1e3 #m
for i in range(N):
    if abs(x[i]-xbench)<eps and abs(y[i]-ybench)<eps: 
        print(T[i])

###########################################################
# export to vtu

if True:

    m=4
    nelx=nnx-1
    nely=nny-1
    nel=nelx*nely

    icon =np.zeros((m, nel),dtype=np.int32)
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
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '>\n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='T (K)' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10f \n" % T[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel (cm/year)' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%e %e %e \n" % (u[i]*year/0.01,v[i]*year/0.01,0))
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
        vtufile.write("%d \n" %((iel+1)*4))
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
        
