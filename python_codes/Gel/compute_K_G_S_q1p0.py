import numpy as np

Lx=1
Ly=1
nelx = 7
nely = 7
hx=Lx/nelx
hy=Ly/nely
nnx=nelx+1       # number of elements, x direction
nny=nely+1       # number of elements, y direction
NV=nnx*nny       # number of nodes
nel=nelx*nely    # number of elements, total
mV=4
ndofV=2
sqrt3=np.sqrt(3)
eta=1

#--------------------------------------------------------------------
x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates
counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

#--------------------------------------------------------------------
icon =np.zeros((mV,nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1
    #end for

#--------------------------------------------------------------------
b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
N     = np.zeros(mV,dtype=np.float64)            # shape functions
dNdx  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
S     = np.zeros(nel,dtype=np.float64)           # pressure field 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) # a

np.set_printoptions(formatter={'float': "{: 5.4f}".format})

for iel in range(0, nel):

    # set arrays to 0 every loop
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    K_L  =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

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
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3,2*i:2*i+2] = [[dNdx[i],0.     ],
                                        [0.     ,dNdy[i]],
                                        [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq

        #end for jq
    #end for iq


print (jcob/hx/hy)
print(G_el)

print('---------------------------------------------------------------')
print('-------MATRIX--------------------------------------------------')
print('---------------------------------------------------------------')

print(K_el)

print('-----------------------------------------------')

K_th =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)

#diagonal terms
K_th[0,0]=(2*hy/hx+hx/hy)/3
K_th[1,1]=(hy/hx+2*hx/hy)/3
K_th[2,2]=K_th[0,0]
K_th[3,3]=K_th[1,1]
K_th[4,4]=K_th[0,0]
K_th[5,5]=K_th[1,1]
K_th[6,6]=K_th[0,0]
K_th[7,7]=K_th[1,1]

K_th[0,1]=0.25                 ; K_th[1,0]=K_th[0,1]
K_th[0,2]=(-2*hy/hx+hx/2/hy)/3 ; K_th[2,0]=K_th[0,2]
K_th[0,3]=-0.25                ; K_th[3,0]=K_th[0,3]
K_th[0,4]=(-hy/hx-hx/2/hy)/3   ; K_th[4,0]=K_th[0,4]
K_th[0,5]=-0.25                ; K_th[5,0]=K_th[0,5]
K_th[0,6]=(hy/hx -hx/hy)/3     ; K_th[6,0]=K_th[0,6]
K_th[0,7]=0.25                 ; K_th[7,0]=K_th[0,7]

K_th[1,2]=0.25                 ; K_th[2,1]=K_th[1,2]
K_th[1,3]= -K_th[0,6]          ; K_th[3,1]=K_th[1,3]
K_th[1,4]=-0.25                ; K_th[4,1]=K_th[1,4]
K_th[1,5]=(-hx/hy-hy/2/hx)/3   ; K_th[5,1]=K_th[1,5]
K_th[1,6]=-0.25                ; K_th[6,1]=K_th[1,6]
K_th[1,7]=(-2*hx/hy+hy/2/hx)/3 ; K_th[7,1]=K_th[1,7]

K_th[2,3]=-0.25                ; K_th[3,2]=K_th[2,3]
K_th[2,4]= K_th[0,6]           ; K_th[4,2]=K_th[2,4]
K_th[2,5]=-0.25                ; K_th[5,2]=K_th[2,5]
K_th[2,6]= K_th[0,4]           ; K_th[6,2]=K_th[2,6]
K_th[2,7]=0.25                 ; K_th[7,2]=K_th[2,7]

K_th[3,4]=0.25                 ; K_th[4,3]=K_th[3,4]
K_th[3,5]=K_th[1,7]            ; K_th[5,3]=K_th[3,5]
K_th[3,6]=0.25                 ; K_th[6,3]=K_th[3,6]
K_th[3,7]=K_th[1,5]            ; K_th[7,3]=K_th[3,7]

K_th[4,5]=0.25                 ; K_th[5,4]=K_th[4,5]
K_th[4,6]=K_th[0,2]            ; K_th[6,4]=K_th[4,6]
K_th[4,7]=-0.25                ; K_th[7,4]=K_th[4,7]

K_th[5,6]=0.25                 ; K_th[6,5]=K_th[5,6]
K_th[5,7]= -K_th[0,6]          ; K_th[7,5]=K_th[5,7]

K_th[6,7]=-0.25                ; K_th[7,6]=K_th[6,7]





print(K_th)

print('------------difference----------')
print(K_el-K_th)


print('---------------------------------------------------------------')
print('-------LUMPED MATRIX-------------------------------------------')
print('---------------------------------------------------------------')
for i in range(0,8):
    for j in range(0,8):
        K_L[i,i]+=abs(K_el[i,j])

print(K_L)

print('-----------------------------------------------')

K_L =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)

K_L[0,0]=abs(K_th[0,0])+abs(K_th[0,2])+abs(K_th[0,4])+abs(K_th[0,6]) +1 
K_L[1,1]=abs(K_th[1,1])+abs(K_th[1,3])+abs(K_th[1,5])+abs(K_th[1,7]) +1 
K_L[2,2]=abs(K_th[2,0])+abs(K_th[2,2])+abs(K_th[2,4])+abs(K_th[2,6]) +1 
K_L[3,3]=abs(K_th[3,1])+abs(K_th[3,3])+abs(K_th[3,5])+abs(K_th[3,7]) +1 
K_L[4,4]=abs(K_th[4,0])+abs(K_th[4,2])+abs(K_th[4,4])+abs(K_th[4,6]) +1 
K_L[5,5]=abs(K_th[5,1])+abs(K_th[5,3])+abs(K_th[5,5])+abs(K_th[5,7]) +1 
K_L[6,6]=abs(K_th[6,0])+abs(K_th[6,2])+abs(K_th[6,4])+abs(K_th[6,6]) +1 
K_L[7,7]=abs(K_th[7,1])+abs(K_th[7,3])+abs(K_th[7,5])+abs(K_th[7,7]) +1 

print(K_L)

print('---------------------------------------------------------------')
print('---------------------------------------------------------------')

print(G_el)

G_th=np.zeros((mV*ndofV,1),dtype=np.float64)
G_th[0]=+hy/2
G_th[1]=+hx/2
G_th[2]=-hy/2
G_th[3]=+hx/2
G_th[4]=-hy/2
G_th[5]=-hx/2
G_th[6]=+hy/2
G_th[7]=-hx/2

print(G_th)

print(G_el-G_th)



K_L=np.linalg.inv(K_L)
S=G_el.T.dot(K_L.dot(G_el))

print(S)

print(2/3*hx*hy)
