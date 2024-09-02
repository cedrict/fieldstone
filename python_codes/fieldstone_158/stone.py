import numpy as np

experiment=1

###############################################################################

def density(x,y,experiment):
    if experiment==1:
       val=1
       if (x-0.5)**2+(y-0.5)**2<0.023:
          val=2
    return val

def viscosity(x,y,experiment):
    if experiment==1:
       val=1
    return val

###############################################################################

Lx=1
Ly=1

gx=0
gy=-1

nnx=5
nny=4

hx=Lx/(nnx-1)
hy=Ly/(nny-1)

ncellx=nnx-1
ncelly=nny-1
ncell=ncellx*ncelly

Nb=nnx*nny       # background mesh
Nu=nnx*ncelly    # u-nodes
Nv=ncellx*nny    # v-nodes  
Np=ncellx*ncelly # p-nodes
N=Nu+Nv+Np       # total nb of unknowns

print('===============')
print('Nb=',Nb)
print('Nu=',Nu)
print('Nv=',Nv)
print('Np=',Np)
print('N=',N)
print('ncell=',ncell)
print('===============')

hhx=1/hx
hhy=1/hy

###############################################################################
# build background mesh
###############################################################################

xb=np.zeros(Nb,dtype=np.float64)  # x coordinates
yb=np.zeros(Nb,dtype=np.float64)  # y coordinates
rho=np.zeros(Nb,dtype=np.float64) # density 
eta=np.zeros(Nb,dtype=np.float64) # viscosity

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xb[counter]=i*hx
        yb[counter]=j*hy
        counter += 1

icon =np.zeros((4,ncell),dtype=np.int32)
counter = 0
for j in range(0,ncelly):
    for i in range(0,ncellx):
        icon[0,counter]=i+j*(ncellx+1)
        icon[1,counter]=i+1+j*(ncellx+1)
        icon[2,counter]=i+1+(j+1)*(ncellx+1)
        icon[3,counter]=i+(j+1)*(ncellx+1)
        counter += 1
    #end for
#end for

for i in range(0,Nb):
    rho[i]=density(xb[i],yb[i],experiment)
    eta[i]=viscosity(xb[i],yb[i],experiment)

np.savetxt('grid_b.ascii',np.array([xb,yb]).T,header='# x,y')

###############################################################################
# build u mesh
###############################################################################

xu=np.zeros(Nu,dtype=np.float64)
yu=np.zeros(Nu,dtype=np.float64)
u=np.zeros(Nu,dtype=np.float64)
left=np.zeros(Nu,dtype=bool) 
right=np.zeros(Nu,dtype=bool) 

counter = 0
for j in range(0,ncelly):
    for i in range(0,nnx):
        xu[counter]=i*hx
        yu[counter]=j*hy+hy/2
        left[counter]=(i==0)
        right[counter]=(i==nnx-1)
        counter += 1

np.savetxt('grid_u.ascii',np.array([xu,yu]).T,header='# x,y')

###############################################################################
# build v mesh
###############################################################################

xv=np.zeros(Nv,dtype=np.float64)
yv=np.zeros(Nv,dtype=np.float64)
v=np.zeros(Nv,dtype=np.float64)
bottom=np.zeros(Nv,dtype=bool) 
top=np.zeros(Nv,dtype=bool) 

counter = 0
for j in range(0,nny):
    for i in range(0,ncellx):
        xv[counter]=i*hx+hx/2
        yv[counter]=j*hy
        bottom[counter]=(j==0)
        top[counter]=(j==nny-1)
        counter += 1

np.savetxt('grid_v.ascii',np.array([xv,yv]).T,header='# x,y')

###############################################################################
# build p mesh
###############################################################################

xp=np.zeros(Np,dtype=np.float64)
yp=np.zeros(Np,dtype=np.float64)
p=np.zeros(Np,dtype=np.float64) 

counter = 0
for j in range(0,ncelly):
    for i in range(0,ncellx):
        xp[counter]=i*hx+hx/2
        yp[counter]=j*hy+hy/2
        counter += 1

np.savetxt('grid_p.ascii',np.array([xp,yp]).T,header='# x,y')

###############################################################################

A=np.zeros((N,N),dtype=np.float64)
b=np.zeros(N,dtype=np.float64)

###############################################################################
# loop over all u nodes
###############################################################################

for i in range(0,Nu):

    if left[i]: # u node on left boundary

       1

    elif right[i]: # u node on right boundary

       2

    else:
       print('================')
       print('u node #',i)

       index_eta_nw=i+ncellx
       index_eta_n=index_eta_nw+1
       index_eta_ne=index_eta_n+1
       index_eta_sw=i-1
       index_eta_s=i
       index_eta_se=i+1

       print('index_eta_nw=',index_eta_nw)
       print('index_eta_n =',index_eta_n)
       print('index_eta_ne=',index_eta_ne)
       print('index_eta_sw=',index_eta_sw)
       print('index_eta_s =',index_eta_s)
       print('index_eta_se=',index_eta_se)

       eta_w=(eta[index_eta_nw]+eta[index_eta_sw]+eta[index_eta_n]+eta[index_eta_s])/4
       eta_e=(eta[index_eta_ne]+eta[index_eta_se]+eta[index_eta_n]+eta[index_eta_s])/4
       
       ii=i%nnx
       jj=int((i-i%nnx)/nnx)

       index_p_w=i-jj-1
       index_p_e=index_p_w+1
       print('index_p_w =',index_p_w)
       print('index_p_e =',index_p_e)

       index_rho_n=i+nnx
       index_rho_s=i
       print('index_rho_n =',index_rho_n)
       print('index_rho_s =',index_rho_s)

       index_v_sw=i-jj-1
       index_v_se=i-jj
       index_v_nw=i-jj-1+ncellx
       index_v_ne=i-jj+ncellx
       print('index_v_sw =',index_v_sw)
       print('index_v_se =',index_v_se)
       print('index_v_nw =',index_v_nw)
       print('index_v_ne =',index_v_ne)

       index_u_n=i+nnx
       index_u_s=i-nnx
       index_u_w=i-1
       index_u_e=i+1
       print('index_u_n =',index_u_n)
       print('index_u_s =',index_u_s)
       print('index_u_w =',index_u_w)
       print('index_u_e =',index_u_e)

       b[i]=(rho[index_rho_s]+rho[index_rho_n])/2*gx

    

###############################################################################
# loop over all v nodes
###############################################################################



for i in range(0,Nv):

    if bottom[i]: # v node on bottom boundary
       1
    elif top[i]: # v node on top boundary
       2
    else:

       print('================')
       print('v node #',i)
       ii=i%ncellx
       jj=int((i-i%ncellx)/ncellx)

       index_eta_sw=i+jj-nnx
       index_eta_se=i+jj-nnx+1
       index_eta_w=i+jj
       index_eta_e=i+jj+1
       index_eta_nw=i+jj+nnx
       index_eta_ne=i+jj+nnx+1

       print('index_eta_sw=',index_eta_sw)
       print('index_eta_se=',index_eta_se)
       print('index_eta_w =',index_eta_w)
       print('index_eta_e =',index_eta_e)
       print('index_eta_nw=',index_eta_nw)
       print('index_eta_ne=',index_eta_ne)

       eta_n=(eta[index_eta_nw]+eta[index_eta_ne]+eta[index_eta_w]+eta[index_eta_e])/4
       eta_s=(eta[index_eta_sw]+eta[index_eta_se]+eta[index_eta_w]+eta[index_eta_e])/4

       index_p_s=i-ncellx
       index_p_n=i
       print('index_p_s =',index_p_s)
       print('index_p_n =',index_p_n)

       index_rho_w=i+jj
       index_rho_e=i+jj+1
       print('index_rho_w =',index_rho_w)
       print('index_rho_e =',index_rho_e)

       index_v_w=i-1
       index_v_e=i+1
       index_v_s=i-ncellx
       index_v_n=i+ncellx
       print('index_v_w =',index_v_w)
       print('index_v_e =',index_v_e)
       print('index_v_s =',index_v_s)
       print('index_v_n =',index_v_n)

       index_u_sw=i-jj-1
       index_u_se=i-jj
       index_u_nw=i-jj-1+nnx
       index_u_ne=i-jj+nnx
       print('index_u_sw =',index_u_sw)
       print('index_u_se =',index_u_se)
       print('index_u_nw =',index_u_nw)
       print('index_u_ne =',index_u_ne)


       #b[Nu+i]=(rho[index_rho_e]+rho[index_rho_w])/2*gy



###############################################################################
# loop over all p nodes
###############################################################################

for i in range(0,Np):

    print('================')
    print('p node #',i)
    ii=i%ncellx
    jj=int((i-i%ncellx)/ncellx)

    index_u_w=i+jj    # u node left 
    index_u_e=i+jj+1  # u node right
    index_v_s=i       # v node below
    index_v_n=i+ncellx  # v node above

    print('index_u_w',index_u_w)
    print('index_u_e',index_u_e)
    print('index_v_s',index_v_s)
    print('index_v_n',index_v_n)

    A[Nu+Nv+i,   index_u_e]= hhx
    A[Nu+Nv+i,   index_u_w]=-hhx
    A[Nu+Nv+i,Nu+index_v_n]= hhy
    A[Nu+Nv+i,Nu+index_v_s]=-hhy

# apparently i need to do something at all corners?!

exit()

#------------------------


filename = 'solution_b.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(Nb,ncell))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,Nb):
    vtufile.write("%10e %10e %10e \n" %(xb[i],yb[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
#vtufile.write("<CellData Scalars='scalars'>\n")
#--
#vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
#vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
#for i in range(0,NV):
#    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
#vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
for i in range(0,Nb):
    vtufile.write("%10e \n" %rho[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
for i in range(0,Nb):
    vtufile.write("%10e \n" %eta[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,ncell):
    vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,ncell):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,ncell):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
