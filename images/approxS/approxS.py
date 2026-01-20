import numpy as np

Lx=1
Ly=1

m_V=9
m_P=4
ndof_V=2
nelx=3
nely=2
nnx=2*nelx+1           # number of V nodes, x direction
nny=2*nely+1           # number of V nodes, y direction
nn_V=nnx*nny           # number of V nodes
nn_P=(nelx+1)*(nely+1) # number of P nodes
nel=nelx*nely          # total number of elements
Nfem_V=nn_V*ndof_V     # number of velocity dofs
Nfem_P=nn_P            # number of pressure dofs
Nfem=Nfem_V+Nfem_P     # total number of dofs

hx=Lx/nelx
hy=Ly/nely

#print(nn_V,nn_P)

###############################################################################
# grid point setup
###############################################################################

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter += 1
    #end for
#end for



###############################################################################
# connectivity
###############################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###############################################################################

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
        icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
        icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
        icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

#######################################################################

nb_of_Vnodes_seen_by_Pnode=np.zeros(nn_P,dtype=np.int32)
list_of_Vnodes_seen_by_Pnode=np.zeros((nn_P,25),dtype=np.int32)

counter=0
for jp in range(0,nely+1):
    for ip in range(0,nelx+1): # loop over pressure nodes
        iv=2*ip
        jv=2*jp
        kv=nnx*jv+iv
        print('P-node',ip,jp,counter,' -> V-node',iv,jv,kv)

        Nv=0
        for n in (-2,-1,0,1,2):
            for m in (-2,-1,0,1,2):
                iiv=iv+m
                jjv=jv+n
                if iiv>=0 and iiv<nnx and jjv>=0 and jjv<nny: #if V-node exists
                   kkv=nnx*jjv+iiv
                   #print('m=',m,'n=',n,'iiv=',iiv,'jjv=',jjv,'kkv=',kkv)
                   list_of_Vnodes_seen_by_Pnode[counter,Nv]=kkv
                   Nv+=1
                #end if
            #end for
        #end for
        nb_of_Vnodes_seen_by_Pnode[counter]=Nv
        counter += 1
    #end for
#end for

print('-------------------------------------')

#print(nb_of_Vnodes_seen_by_Pnode)

for i in range(0,nn_P):
    print('Pnode',i,'sees Vnodes:',list_of_Vnodes_seen_by_Pnode[i,:nb_of_Vnodes_seen_by_Pnode[i]])


print('-------------------------------------')

for i in range(0,nn_P):
    for j in range(0,nn_P):
        #print(list_of_Vnodes_seen_by_Pnode[i,:nb_of_Vnodes_seen_by_Pnode[i]])
        #print(list_of_Vnodes_seen_by_Pnode[j,:nb_of_Vnodes_seen_by_Pnode[j]])

        print(i,j,list(set(list_of_Vnodes_seen_by_Pnode[i,:nb_of_Vnodes_seen_by_Pnode[i]]) &\
                       set(list_of_Vnodes_seen_by_Pnode[j,:nb_of_Vnodes_seen_by_Pnode[j]])))

  
