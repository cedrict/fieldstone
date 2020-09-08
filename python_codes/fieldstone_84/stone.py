import numpy as np
import time as time
import sys as sys

G = 6.6738480e-11 # gravitational constant [m^3 s^-2 kg^-1]

def grav_calc(prism,obs_point,method,rho,hx,hy,hz,xx,yy,zz):
    #The gravity calculations of a single cell on one observation point according to a chosen method. 
    # The cell always has a constant density. 
    # For the point mass method all of the mass of the cell is assumed to be concentrated in the centre of the cell. 
    # For the prism method the gravitational response is integrated over the volume of the prism. 
    # The first is more time efficient, while the second is more accurate. 
    # The coordinates of the cell must be given as the corner with the smallest coordinates. 
    # INPUT:
    #    prism     --> the coordinates of the cell used (3,) [m]
    #    obs_point --> the coordinates of the observation point used (3,) [m]
    #    method    --> which method of calculation to use, can be 'prism' or 'point'
    #    rho       --> the density of the cell [kg/m^3]
    #    hx,hy,hz  --> dimensions of the cell in the x,y,z-direction [m]
    # OPTIONAL:
    #    use_num_stab --> whether or not to use the extra numerical stability gotten from Heck and Seitz, 2007
    # RESULTS:
    #    U         --> the gravitational potential [J/kg]
    #    g         --> the gravity vector (3,) [m/s^2]
    #    T         --> the gravity gradient tensor (3,3) [s^-2]

    use_num_stab=False

    # Converting the given corner of the cell to the observation coordinate system
    coords = prism - obs_point
    
    U_calc = 0.
    g_calc = np.zeros(3,dtype=np.float64)
    T_calc = np.zeros([3,3],dtype=np.float64)
    
    if method == "point":

        if rho==0:
           return U_calc, g_calc, T_calc

        # The coordinates needed are at the point mass in the centre of the cell
        x = coords[0] + 0.5*hx
        y = coords[1] + 0.5*hy
        z = coords[2] + 0.5*hz
    
        # calculating the distance between the observation point and the point mass
        r = np.sqrt(x**2+y**2+z**2)
        # The volume of the cell
        dV = hx*hy*hz

        # calculating the different values
        U_calc = -G * rho * dV / r

        g_calc[0] = G * rho * dV * x / r**3
        g_calc[1] = G * rho * dV * y / r**3
        g_calc[2] = G * rho * dV * z / r**3

        T_calc[0,0] = -G*rho*dV*(3*x**2-r**2)/r**5
        T_calc[1,1] = -G*rho*dV*(3*y**2-r**2)/r**5
        T_calc[2,2] = -G*rho*dV*(3*z**2-r**2)/r**5
        T_calc[0,1] = -G*rho*dV*3*x*y/r**5
        T_calc[1,0] = T_calc[0,1]
        T_calc[0,2] = -G*rho*dV*3*x*z/r**5
        T_calc[2,0] = T_calc[0,2]
        T_calc[1,2] = -G*rho*dV*3*y*z/r**5
        T_calc[2,1] = T_calc[1,2]
    
    elif method == "prism":

        if rho==0:
           return U_calc, g_calc, T_calc

        # The coordinates needed are the boundary coordinates of the prism. If the the prism 
        # would range from 1 to 5 on the x axis, 2 to 4 on the y axis and 0 to 6 on the z axis 
        # the two points would be (1,2,0) and (5,4,6) and the arrays x=[1,5], y=[2,4] and z=[0,6].
        x = [coords[0],coords[0]+hx]
        y = [coords[1],coords[1]+hy]
        z = [coords[2],coords[2]+hz]

        # The triple summation is implemented as a triple loop
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # The distance between the observation point and the integration point is calculated
                    r = np.sqrt(x[i]**2+y[j]**2+z[k]**2)

                    # There are cases where the calculations will fail, f.e. log(0) or arctan(1/0). 
                    # To stop this, exception cases are defined. They are set to the limit values, 
                    # so log(0) -> 0 and arctan(1/0) -> 1/2 pi
                    if x[i] == 0:
                        arctan_x = - 0.5*np.pi
                    else:
                        arctan_x = - np.arctan((y[j]*z[k]/(x[i]*r)))
                    if y[j] == 0:
                        arctan_y = - 0.5*np.pi
                    else:
                        arctan_y = - np.arctan((x[i]*z[k]/(y[j]*r)))
                    if z[k] == 0:
                        arctan_z = - 0.5*np.pi
                    else:
                        arctan_z = - np.arctan((x[i]*y[j]/(z[k]*r)))

                    # Along with the exceptions, there is a variant of the equations that should 
                    # offer extra numerical stability. 
                    if r+x[i] == 0:
                        log_x = 0
                    else:
                        if use_num_stab == False:
                            log_x = np.log(x[i]+r)
                        else:
                            log_x = np.log((x[i]+r)/(np.sqrt(y[j]**2+z[k]**2)))
                    if r+y[j] == 0:
                        log_y = 0
                    else:
                        if use_num_stab == False:
                            log_y = np.log(y[j]+r)
                        else:
                            log_y = np.log((y[j]+r)/(np.sqrt(x[i]**2+z[k]**2)))
                    if r+z[k] == 0:
                        log_z = 0
                    else:
                        if use_num_stab == False:
                            log_z = np.log(z[k]+r)
                        else:
                            log_z = np.log((z[k]+r)/(np.sqrt(x[i]**2+y[j]**2)))

                    U_calc += (-1)**(i+j+k) * (y[j]*z[k]*log_x + x[i]*y[j]*log_z  + x[i]*z[k]*log_y\
                             + x[i]**2/2*arctan_x + y[j]**2/2*arctan_y + z[k]**2/2*arctan_z)

                    g_calc[0] += (-1)**(i+j+k) * (z[k]*log_y + y[j]*log_z + x[i]*arctan_x)
                    g_calc[1] += (-1)**(i+j+k) * (z[k]*log_x + x[i]*log_z + y[j]*arctan_y)
                    g_calc[2] += (-1)**(i+j+k) * (x[i]*log_y + y[j]*log_x + z[k]*arctan_z)

                    T_calc[0,0] += (-1)**(i+j+k) * arctan_x
                    T_calc[1,1] += (-1)**(i+j+k) * arctan_y
                    T_calc[2,2] += (-1)**(i+j+k) * arctan_z
                    T_calc[0,1] += (-1)**(i+j+k) * log_z
                    T_calc[0,2] += (-1)**(i+j+k) * log_y
                    T_calc[1,2] += (-1)**(i+j+k) * log_x

                # end for
            # end for
        # end for

        # T is symmetric
        T_calc[1,0] = T_calc[0,1]
        T_calc[2,0] = T_calc[0,2]
        T_calc[2,1] = T_calc[1,2]
        # When the summations are complete, everything is multiplied with G and rho to get the final values
        U_calc *= G*rho
        g_calc *= G*rho
        T_calc *= G*rho

    #end if
    elif method == "quadrature":

        if rho==0:
           return U_calc, g_calc, T_calc

        N = np.zeros(8,dtype=np.float64)
        dNdx = np.zeros(8,dtype=np.float64)             # shape functions derivatives
        dNdy = np.zeros(8,dtype=np.float64)             # shape functions derivatives
        dNdz = np.zeros(8,dtype=np.float64)             # shape functions derivatives
        dNdr = np.zeros(8,dtype=np.float64)             # shape functions derivatives
        dNds = np.zeros(8,dtype=np.float64)             # shape functions derivatives
        dNdt = np.zeros(8,dtype=np.float64)             # shape functions derivatives

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                for kq in range(0,nqperdim):

                    rq=qcoords[iq]
                    sq=qcoords[jq]
                    tq=qcoords[kq]
                    weightq=qweights[iq]*qweights[jq]*qweights[kq]

                    N[0]=0.125*(1.-rq)*(1.-sq)*(1.-tq)
                    N[1]=0.125*(1.+rq)*(1.-sq)*(1.-tq)
                    N[2]=0.125*(1.+rq)*(1.+sq)*(1.-tq)
                    N[3]=0.125*(1.-rq)*(1.+sq)*(1.-tq)
                    N[4]=0.125*(1.-rq)*(1.-sq)*(1.+tq)
                    N[5]=0.125*(1.+rq)*(1.-sq)*(1.+tq)
                    N[6]=0.125*(1.+rq)*(1.+sq)*(1.+tq)
                    N[7]=0.125*(1.-rq)*(1.+sq)*(1.+tq)

                    dNdr[0]=-0.125*(1.-sq)*(1.-tq)
                    dNdr[1]=+0.125*(1.-sq)*(1.-tq)
                    dNdr[2]=+0.125*(1.+sq)*(1.-tq)
                    dNdr[3]=-0.125*(1.+sq)*(1.-tq)
                    dNdr[4]=-0.125*(1.-sq)*(1.+tq)
                    dNdr[5]=+0.125*(1.-sq)*(1.+tq)
                    dNdr[6]=+0.125*(1.+sq)*(1.+tq)
                    dNdr[7]=-0.125*(1.+sq)*(1.+tq)

                    dNds[0]=-0.125*(1.-rq)*(1.-tq)
                    dNds[1]=-0.125*(1.+rq)*(1.-tq)
                    dNds[2]=+0.125*(1.+rq)*(1.-tq)
                    dNds[3]=+0.125*(1.-rq)*(1.-tq)
                    dNds[4]=-0.125*(1.-rq)*(1.+tq)
                    dNds[5]=-0.125*(1.+rq)*(1.+tq)
                    dNds[6]=+0.125*(1.+rq)*(1.+tq)
                    dNds[7]=+0.125*(1.-rq)*(1.+tq)

                    dNdt[0]=-0.125*(1.-rq)*(1.-sq)
                    dNdt[1]=-0.125*(1.+rq)*(1.-sq)
                    dNdt[2]=-0.125*(1.+rq)*(1.+sq)
                    dNdt[3]=-0.125*(1.-rq)*(1.+sq)
                    dNdt[4]=+0.125*(1.-rq)*(1.-sq)
                    dNdt[5]=+0.125*(1.+rq)*(1.-sq)
                    dNdt[6]=+0.125*(1.+rq)*(1.+sq)
                    dNdt[7]=+0.125*(1.-rq)*(1.+sq)

                    jcb=np.zeros((3,3),dtype=np.float64)
                    for k in range(0,m):
                        jcb[0,0] += dNdr[k]*xx[k]
                        jcb[0,1] += dNdr[k]*yy[k]
                        jcb[0,2] += dNdr[k]*zz[k]
                        jcb[1,0] += dNds[k]*xx[k]
                        jcb[1,1] += dNds[k]*yy[k]
                        jcb[1,2] += dNds[k]*zz[k]
                        jcb[2,0] += dNdt[k]*xx[k]
                        jcb[2,1] += dNdt[k]*yy[k]
                        jcb[2,2] += dNdt[k]*zz[k]
                    #end for 
                    jcob = np.linalg.det(jcb)
                    JxW=jcob*weightq

                    xq=0.0
                    yq=0.0
                    zq=0.0
                    for k in range(0,8):
                        xq+=N[k]*xx[k]
                        yq+=N[k]*yy[k]
                        zq+=N[k]*zz[k]

                    dx = obs_point[0]-xq
                    dy = obs_point[1]-yq
                    dz = obs_point[2]-zq
                    dx2 = dx**2
                    dy2 = dy**2
                    dz2 = dz**2
                    dist2=dx2+dy2+dz2 
                    dist=np.sqrt(dist2) 

                    KK=JxW/dist**3
                    g_calc[0]+=KK*dx
                    g_calc[1]+=KK*dy
                    g_calc[2]+=KK*dz
                    U_calc-=JxW/dist

                    KK=JxW/dist**5
                    T_calc[0,0]+=KK*(3.*dx2-dist2)
                    T_calc[1,1]+=KK*(3.*dy2-dist2)
                    T_calc[2,2]+=KK*(3.*dz2-dist2)
                    T_calc[0,1]+=KK*3.*dx*dy
                    T_calc[0,2]+=KK*3.*dx*dz
                    T_calc[1,2]+=KK*3.*dy*dz
                    T_calc[1,0]=T_calc[0,1]
                    T_calc[2,0]=T_calc[0,2]
                    T_calc[2,1]=T_calc[1,2]

                # end for
            # end for
        # end for
        U_calc *= G*rho
        g_calc *= G*rho
        T_calc *= G*rho
    
    # The results are returned
    return U_calc, g_calc, T_calc

##############################################################################################################
##############################################################################################################


#################################################################
# experiment setup parameters
#################################################################

buried_object = 'sphere'

rho0=0 

if buried_object == 'cube':
   Lx=1.e3  
   Ly=1.e3  
   Lz=1.e3  
   xc_object=0.5*Lx
   yc_object=0.5*Ly
   zc_object=0.75*Lz
   cube_size=Lx/8
   rho_cube=100


if buried_object == 'sphere':
   Lx=1.e3  
   Ly=1.e3  
   Lz=1.e3  
   xc_object=0.5*Lx
   yc_object=0.5*Ly
   zc_object=0.5*Lz
   radius_sphere=Lx/2
   rho_sphere=100

#################################################################
# gravity calculations parameters
#################################################################

method='point'
#method='prism'
#method='quadrature'

nqperdim=2

compute_gravity_on_plane=False
nnx_m=15
nny_m=15
z_plane=Lz+100

compute_gravity_on_line=False
nnp_line=133
x_begin=xc_object
y_begin=yc_object
z_begin=zc_object
x_end=1.11e3
y_end=2.22e3
z_end=5.55e3

compute_gravity_at_single_point=True
xpt=123
ypt=234
zpt=345

#################################################################

if int(len(sys.argv) == 2):
   nelx = int(sys.argv[1])
else:
   nelx = 32

nely=nelx
nelz=nelx

nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

NV=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

m=8 # number of corners per element/cell

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

#################################################################

print('-------------------------------')
print('nelx= ',nelx)
print('method= ',method)
print('buried_object= ',buried_object)
print('compute_gravity_on_plane=',compute_gravity_on_plane)
print('compute_gravity_on_line=',compute_gravity_on_line)
print('compute_gravity_at_single_point=',compute_gravity_at_single_point)
print('-------------------------------')

#################################################################

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

if nqperdim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

if nqperdim==6:
   qcoords=[-0.932469514203152,\
            -0.661209386466265,\
            -0.238619186083197,\
            +0.238619186083197,\
            +0.661209386466265,\
            +0.932469514203152]
   qweights=[0.171324492379170,\
             0.360761573048139,\
             0.467913934572691,\
             0.467913934572691,\
             0.360761573048139,\
             0.171324492379170]

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
z = np.empty(NV,dtype=np.float64)  # z coordinates

counter=0
for i in range(0, nnx):
    for j in range(0, nny):
        for k in range(0, nnz):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            z[counter]=k*Lz/float(nelz)
            counter += 1
        #end for
    #end for
#end for
   
print("mesh setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)

counter = 0
for i in range(0, nelx):
    for j in range(0, nely):
        for k in range(0, nelz):
            icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("connectivity setup: %.3f s" % (time.time() - start))

#################################################################
# assign density to cells
#################################################################
start = time.time()

rho = np.zeros(nel,dtype=np.float64)  # elemental density

if buried_object == 'cube':

   for iel in range(0,nel):
       xc=x[icon[0,iel]]+hx/2
       yc=y[icon[0,iel]]+hy/2
       zc=z[icon[0,iel]]+hz/2
       if abs(xc-xc_object)<cube_size/2 and \
          abs(yc-yc_object)<cube_size/2 and \
          abs(zc-zc_object)<cube_size/2:
          rho[iel]=rho_cube
       else:
          rho[iel]=rho0
       #end if
   #end for
#end if

elif buried_object == 'sphere':

   for iel in range(0,nel):
       xc=x[icon[0,iel]]+hx/2
       yc=y[icon[0,iel]]+hy/2
       zc=z[icon[0,iel]]+hz/2
       if (xc-xc_object)**2+(yc-yc_object)**2+(zc-zc_object)**2<radius_sphere**2:
          rho[iel]=rho_sphere
       else:
          rho[iel]=rho0
       #end if
   #end for

else:

   exit("buried_object does not exist")

print("assign density: %.3f s" % (time.time() - start))

#################################################################
# compute mass of the system 
#################################################################

Mass=np.sum(rho)*hx*hy*hz

if buried_object=='sphere':
   Massth=4/3*np.pi*radius_sphere**3*rho_sphere

if buried_object=='cube':
   Massth=cube_size**3*rho_cube

print("     -> Total mass %d %e " %(nel,Mass))
print("     -> Total mass error %d %e " %(nel,(Mass-Massth)/Massth))

#################################################################
# measurement plane grid point setup
#################################################################

if compute_gravity_on_plane:

   nnp_m=nnx_m*nny_m
   nelx_m=nnx_m-1
   nely_m=nny_m-1
   nel_m=nelx_m*nely_m
   Lx_m=Lx
   Ly_m=Ly

   x_m = np.zeros(nnp_m,dtype=np.float64)  # x coordinates
   y_m = np.zeros(nnp_m,dtype=np.float64)  # y coordinates
   z_m = np.zeros(nnp_m,dtype=np.float64)  # y coordinates

   counter = 0
   for j in range(0, nny_m):
       for i in range(0, nnx_m):
           x_m[counter]=i*Lx_m/float(nnx_m-1)
           y_m[counter]=j*Ly_m/float(nny_m-1)
           z_m[counter]=z_plane
           counter += 1

   icon_m =np.zeros((4,nel_m),dtype=np.int32)
   counter = 0
   for j in range(0, nely_m):
       for i in range(0, nelx_m):
           icon_m[0, counter] = i + j * (nelx_m + 1)
           icon_m[1, counter] = i + 1 + j * (nelx_m + 1)
           icon_m[2, counter] = i + 1 + (j + 1) * (nelx_m + 1)
           icon_m[3, counter] = i + (j + 1) * (nelx_m + 1)
           counter += 1

   meas_point = np.zeros(3,dtype=np.float64)
   ll_corner = np.zeros(3,dtype=np.float64)
    
   U = np.zeros(nnp_m,dtype=np.float64)
   gx = np.zeros(nnp_m,dtype=np.float64)
   gy = np.zeros(nnp_m,dtype=np.float64)
   gz = np.zeros(nnp_m,dtype=np.float64)
   Txx = np.zeros(nnp_m,dtype=np.float64)
   Tyy = np.zeros(nnp_m,dtype=np.float64)
   Tzz = np.zeros(nnp_m,dtype=np.float64)
   Txy = np.zeros(nnp_m,dtype=np.float64)
   Txz = np.zeros(nnp_m,dtype=np.float64)
   Tyz = np.zeros(nnp_m,dtype=np.float64)

   start = time.time()

   for i in range(0,nnp_m):
       if i%10==0:
          print('point=',i,'out of ',nnp_m)
       meas_point[0]=x_m[i]
       meas_point[1]=y_m[i]
       meas_point[2]=z_m[i]
       for iel in range(0,nel):
          ll_corner[0]=x[icon[0,iel]]
          ll_corner[1]=y[icon[0,iel]]
          ll_corner[2]=z[icon[0,iel]]
          UU,g,T=grav_calc(ll_corner,meas_point,method,rho[iel],hx,hy,hz)
          U[i]+=UU
          gx[i]+=g[0]
          gy[i]+=g[1]
          gz[i]+=g[2]
          Txx[i]+=T[0,0]
          Tyy[i]+=T[1,1]
          Tzz[i]+=T[2,2]
          Txy[i]+=T[0,1]
          Txz[i]+=T[0,2]
          Tyz[i]+=T[1,2]
       #end for
   #end for

   print("compute gravity on plane: %.3f s" % (time.time() - start))

   # export measurement grid
   gnorm=np.sqrt(gx**2+gy**2+gz**2)

   filename = 'gravity.vtu'
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp_m,nel_m))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e %10e %10e \n" %(x_m[i],y_m[i],z_m[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='U' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e \n" %(U[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='Txx' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e \n" %(Txx[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='Tyy' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e \n" %(Tyy[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='Tzz' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e \n" %(Tzz[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='Txy' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e \n" %(Txy[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='Txz' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e \n" %(Txz[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='Tyz' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e \n" %(Tyz[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity vector g' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e %10e %10e \n" %(gx[i],gy[i],gz[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='|g|' Format='ascii'> \n")
   for i in range(0,nnp_m):
       vtufile.write("%10e \n" %(gnorm[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel_m):
       vtufile.write("%d %d %d %d\n" %(icon_m[0,iel],icon_m[1,iel],icon_m[2,iel],icon_m[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel_m):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel_m):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

#################################################################
# line gravity 
#################################################################

if compute_gravity_on_line:

   x_line = np.zeros(nnp_line,dtype=np.float64)  
   y_line = np.zeros(nnp_line,dtype=np.float64)  
   z_line = np.zeros(nnp_line,dtype=np.float64)  
   r_line = np.zeros(nnp_line,dtype=np.float64)

   meas_point = np.zeros(3,dtype=np.float64)
   ll_corner = np.zeros(3,dtype=np.float64)
    
   U = np.zeros(nnp_line,dtype=np.float64)
   gx = np.zeros(nnp_line,dtype=np.float64)
   gy = np.zeros(nnp_line,dtype=np.float64)
   gz = np.zeros(nnp_line,dtype=np.float64)
   Txx = np.zeros(nnp_line,dtype=np.float64)
   Tyy = np.zeros(nnp_line,dtype=np.float64)
   Tzz = np.zeros(nnp_line,dtype=np.float64)
   Txy = np.zeros(nnp_line,dtype=np.float64)
   Txz = np.zeros(nnp_line,dtype=np.float64)
   Tyz = np.zeros(nnp_line,dtype=np.float64)

   start = time.time()

   for i in range(0,nnp_line):
       if i%10==0:
          print('point=',i,'out of ',nnp_line)
       meas_point[0]=x_begin+(x_end-x_begin)/(nnp_line-1)*i
       meas_point[1]=y_begin+(y_end-y_begin)/(nnp_line-1)*i
       meas_point[2]=z_begin+(z_end-z_begin)/(nnp_line-1)*i
       x_line[i]=x_begin+(x_end-x_begin)/(nnp_line-1)*i
       y_line[i]=y_begin+(y_end-y_begin)/(nnp_line-1)*i
       z_line[i]=z_begin+(z_end-z_begin)/(nnp_line-1)*i
       for iel in range(0,nel):
          ll_corner[0]=x[icon[0,iel]]
          ll_corner[1]=y[icon[0,iel]]
          ll_corner[2]=z[icon[0,iel]]
          UU,g,T=grav_calc(ll_corner,meas_point,method,rho[iel],hx,hy,hz)
          U[i]+=UU
          gx[i]+=g[0]
          gy[i]+=g[1]
          gz[i]+=g[2]
          Txx[i]+=T[0,0]
          Tyy[i]+=T[1,1]
          Tzz[i]+=T[2,2]
          Txy[i]+=T[0,1]
          Txz[i]+=T[0,2]
          Tyz[i]+=T[1,2]
       #end for
   #end for

   print("compute gravity on line: %.3f s" % (time.time() - start))

   for i in range(0,nnp_line):
       r_line[i]=np.sqrt((x_line[i]-xc_object)**2+(y_line[i]-yc_object)**2+(z_line[i]-zc_object)**2)

   # compute analytical solution 
   U_th = np.zeros(nnp_line,dtype=np.float64)
   g_th = np.zeros(nnp_line,dtype=np.float64)

   if buried_object=='sphere':
      for i in range(0,nnp_line):
          if r_line[i]>radius_sphere:
             g_th[i]=G*Massth/r_line[i]**2
             U_th[i]=-G*Massth/r_line[i]
          else:
             g_th[i]=G*4/3*np.pi*r_line[i]*rho_sphere
             U_th[i]=-2*np.pi*G*rho_sphere*(radius_sphere**2-r_line[i]**2/3)


   gnorm=np.sqrt(gx**2+gy**2+gz**2)
   np.savetxt('gravity_on_line.ascii',np.array([r_line,U,gx,gy,gz,gnorm,Txx,Tyy,Tzz,Txy,Txz,Tyz,U_th,g_th]).T)

   vtufile=open("line.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp_line,nnp_line-1))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp_line):
       vtufile.write("%10f %10f %10f \n" %(x_line[i],y_line[i],z_line[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nnp_line-1):
       vtufile.write("%d %d\n" %(iel,iel+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nnp_line-1):
       vtufile.write("%d \n" %((iel+1)*2))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nnp_line-1):
       vtufile.write("%d \n" %3)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

#################################################################

if compute_gravity_at_single_point:

   U = np.zeros(1,dtype=np.float64)
   gx = np.zeros(1,dtype=np.float64)
   gy = np.zeros(1,dtype=np.float64)
   gz = np.zeros(1,dtype=np.float64)
   Txx = np.zeros(1,dtype=np.float64)
   Tyy = np.zeros(1,dtype=np.float64)
   Tzz = np.zeros(1,dtype=np.float64)
   Txy = np.zeros(1,dtype=np.float64)
   Txz = np.zeros(1,dtype=np.float64)
   Tyz = np.zeros(1,dtype=np.float64)

   meas_point = np.zeros(3,dtype=np.float64)
   ll_corner = np.zeros(3,dtype=np.float64)

   start = time.time()

   meas_point[0]=xpt+xc_object
   meas_point[1]=ypt+yc_object
   meas_point[2]=zpt+zc_object
   for iel in range(0,nel):
       ll_corner[0]=x[icon[0,iel]]
       ll_corner[1]=y[icon[0,iel]]
       ll_corner[2]=z[icon[0,iel]]
       UU,g,T=grav_calc(ll_corner,meas_point,method,rho[iel],hx,hy,hz,x[icon[:,iel]],y[icon[:,iel]],z[icon[:,iel]])
       U[0]+=UU
       gx[0]+=g[0]
       gy[0]+=g[1]
       gz[0]+=g[2]
       Txx[0]+=T[0,0]
       Tyy[0]+=T[1,1]
       Tzz[0]+=T[2,2]
       Txy[0]+=T[0,1]
       Txz[0]+=T[0,2]
       Tyz[0]+=T[1,2]

   if buried_object=='sphere':
      if xpt**2+ypt**2+zpt**2>radius_sphere**2:
         g_th=G*Massth/(xpt**2+ypt**2+zpt**2)
         U_th=-G*Massth/np.sqrt(xpt**2+ypt**2+zpt**2)
      else:
         g_th=G*4/3*np.pi* np.sqrt(xpt**2+ypt**2+zpt**2) *rho_sphere
         U_th=-2*np.pi*G*rho_sphere*(radius_sphere**2-(xpt**2+ypt**2+zpt**2)/3)
      
      print("     -> U_at_pt %d %10e %10e " %(nel,U,U_th))
      print("     -> grav_at_pt %d %.10e %.10e" %(nelx,np.sqrt(gx[0]**2+gy[0]**2+gz[0]**2),g_th))

   if buried_object=='cube':
      ll_corner[0]=xc_object-cube_size/2
      ll_corner[1]=yc_object-cube_size/2
      ll_corner[2]=zc_object-cube_size/2
      U_th,g_th,T=grav_calc(ll_corner,meas_point,method,rho_cube,cube_size,cube_size,cube_size)
      print("     -> U_at_pt %d %e %e " %(nel,U[0],U_th))
      print("     -> grav_at_pt %d %e %e %e %e %e %e" %(nelx,gx[0],gy[0],gz[0],g_th[0],g_th[1],g_th[2]))

   print("compute gravity at single pt: %.3f s" % (time.time() - start))

#################################################################
# export volume
#################################################################
start = time.time()

if True: 
   vtufile=open("volume.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % rho[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                                   icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %12)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

print("export to vtu: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
