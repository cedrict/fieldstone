using Printf
using SparseArrays
using LinearAlgebra
using Formatting

#------------------------------------------------------------------------------

function NNN(r,s,t)
N0=0.125*(1-r)*(1-s)*(1-t)
N1=0.125*(1+r)*(1-s)*(1-t)
N2=0.125*(1+r)*(1+s)*(1-t)
N3=0.125*(1-r)*(1+s)*(1-t)
N4=0.125*(1-r)*(1-s)*(1+t)
N5=0.125*(1+r)*(1-s)*(1+t)
N6=0.125*(1+r)*(1+s)*(1+t)
N7=0.125*(1-r)*(1+s)*(1+t)
return Float64[N0,N1,N2,N3,N4,N5,N6,N7]
end

function dNNNdr(r,s,t)
dNdr0=-0.125*(1-s)*(1-t) 
dNdr1=+0.125*(1-s)*(1-t)
dNdr2=+0.125*(1+s)*(1-t)
dNdr3=-0.125*(1+s)*(1-t)
dNdr4=-0.125*(1-s)*(1+t)
dNdr5=+0.125*(1-s)*(1+t)
dNdr6=+0.125*(1+s)*(1+t)
dNdr7=-0.125*(1+s)*(1+t)
return Float64[dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7]
end

function dNNNds(r,s,t)
dNds0=-0.125*(1-r)*(1-t) 
dNds1=-0.125*(1+r)*(1-t)
dNds2=+0.125*(1+r)*(1-t)
dNds3=+0.125*(1-r)*(1-t)
dNds4=-0.125*(1-r)*(1+t)
dNds5=-0.125*(1+r)*(1+t)
dNds6=+0.125*(1+r)*(1+t)
dNds7=+0.125*(1-r)*(1+t)
return Float64[dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7]
end

function dNNNdt(r,s,t)
dNdt0=-0.125*(1-r)*(1-s) 
dNdt1=-0.125*(1+r)*(1-s)
dNdt2=-0.125*(1+r)*(1+s)
dNdt3=-0.125*(1-r)*(1+s)
dNdt4=+0.125*(1-r)*(1-s)
dNdt5=+0.125*(1+r)*(1-s)
dNdt6=+0.125*(1+r)*(1+s)
dNdt7=+0.125*(1-r)*(1+s)
return Float64[dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7]
end

#------------------------------------------------------------------------------

function bx(x,y,z)
if experiment==1 || experiment==2 || experiment==3 || experiment==4
   val=0
end 
if experiment==5
   val=4*(2*y-1)*(2*z-1)
end 
return val
end

function by(x,y,z)
if experiment==1 || experiment==2 || experiment==3 || experiment==4
   val=0
end 
if experiment==5
   val=4*(2*x-1)*(2*z-1)
end 
return val
end

function bz(x,y,z)
if experiment==1 || experiment==2 || experiment==3 || experiment==4
   if (x-0.5)^2+(y-0.5)^2+(z-0.5)^2<0.123456789^2
      val=1.01*gz
   else
      val=1*gz
   end 
end 
if experiment==5
   val=-2*(2*x-1)*(2*y-1) 
end 
return val
end

#------------------------------------------------------------------------------

function viscosity(x,y,z)
if experiment==1 || experiment==2 || experiment==3 || experiment==4
   if (x-0.5)^2+(y-0.5)^2+(z-0.5)^2<0.123456789^2
      val=1e3
   else
      val=1.
   end 
end 
if experiment==5
   val=1.
end 
return val
end

#------------------------------------------------------------------------------

function uth(x,y,z)
val=x*(1-x)*(1-2*y)*(1-2*z)
return val
end

function vth(x,y,z)
val=(1-2*x)*y*(1-y)*(1-2*z)
return val
end

function wth(x,y,z)
val=-2*(1-2*x)*(1-2*y)*z*(1-z)
return val
end

function pth(x,y,z)
val=(2*x-1)*(2*y-1)*(2*z-1)
return val
end

#------------------------------------------------------------------------------

experiment=5

println("-----------------------------")
println("--------- stone 10 ----------")
println("-----------------------------") ; flush(stdout)

m=8     # number of nodes making up an element
ndofV=3  # number of degrees of freedom per node

if length(ARGS) == 2
   nelx = eval(parse(Int64,ARGS[1]))
   visu = eval(parse(Int64,ARGS[2]))
else
   nelx = 24
   visu = 1
end 

if experiment==1
   quarter=false
elseif experiment==2
   quarter=false
elseif experiment==3
   quarter=false
elseif experiment==4
   quarter=true
elseif experiment==5
   quarter=false
end

if quarter
   nely=nelx
   nelz=2*nelx
   Lx=0.5 
   Ly=0.5
   Lz=1.
else
   nely=nelx
   nelz=nelx
   Lx=1.
   Ly=1.
   Lz=1.
end

FS=true
NS=false
OT=false
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

NV=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

penalty=1.e6  # penalty coefficient value

Nfem=NV*ndofV  # Total number of degrees of freedom

eps=1.e-10

gz=-1.  # gravity vector, z component

sqrt3=sqrt(3.)

#################################################################

println("Lx=",Lx)
println("Ly=",Ly)
println("Lz=",Lz)
println("nelx=",nelx)
println("nely=",nely)
println("nelz=",nelz)
println("nel=",nel)
println("Nfem=",Nfem)
println("-----------------------------")

#################################################################
# grid point setup
#################################################################
time1 =  @elapsed begin

x=zeros(Float64,NV)
y=zeros(Float64,NV)
z=zeros(Float64,NV)

counter=0
for i = 1:nnx
    for j = 1:nny
        for k = 1:nnz
            global(counter) += 1
            x[counter]=(i-1)*Lx/nelx
            y[counter]=(j-1)*Ly/nely
            z[counter]=(k-1)*Lz/nelz
        end 
    end 
end 

end ; println("mesh setup:",time1," s")

#################################################################
# connectivity
#################################################################
time1 =  @elapsed begin

icon = zeros(Int32,m,nel)

counter=0
for i=1:nelx
    for j=1:nely
        for k=1:nelz
            global(counter)+=1
            icon[1,counter]=nny*nnz*(i-1)+nnz*(j-1)+k
            icon[2,counter]=nny*nnz*(i  )+nnz*(j-1)+k
            icon[3,counter]=nny*nnz*(i  )+nnz*(j  )+k
            icon[4,counter]=nny*nnz*(i-1)+nnz*(j  )+k
            icon[5,counter]=nny*nnz*(i-1)+nnz*(j-1)+k+1
            icon[6,counter]=nny*nnz*(i  )+nnz*(j-1)+k+1
            icon[7,counter]=nny*nnz*(i  )+nnz*(j  )+k+1
            icon[8,counter]=nny*nnz*(i-1)+nnz*(j  )+k+1
        end
    end
end

end ; println("connectivity array ", time1 , " s")

#################################################################
# define boundary conditions
#################################################################
time1 = @elapsed begin

bc_fix = zeros(Bool,Nfem)
bc_val = zeros(Float64,Nfem)

if experiment==1 || experiment==2 || experiment==3 || experiment==4

   if FS || OT
      for i=1:NV
          if x[i]<eps
             bc_fix[(i-1)*ndofV+1]=true ; bc_val[(i-1)*ndofV+1]= 0.
          end
          if x[i]>(Lx-eps)
             bc_fix[(i-1)*ndofV+1]=true ; bc_val[(i-1)*ndofV+1]= 0.
          end
          if y[i]<eps
             bc_fix[(i-1)*ndofV+2]=true ; bc_val[(i-1)*ndofV+2]= 0.
          end
          if y[i]>(Ly-eps)
             bc_fix[(i-1)*ndofV+2]=true ; bc_val[(i-1)*ndofV+2]= 0.
          end
          if z[i]<eps
             bc_fix[(i-1)*ndofV+3]=true ; bc_val[(i-1)*ndofV+3]= 0.
          end
          if not OT && z[i]>(Lz-eps)
             bc_fix[(i-1)*ndofV+3]=true ; bc_val[(i-1)*ndofV+3]= 0.
          end
      end
   end

   if NS
      for i=1:NV
          if x[i]<eps
             bc_fix[i*ndofV+0]=true ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=true ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=true ; bc_val[i*ndofV+2]= 0.
          end
          if x[i]>(1-eps)
             bc_fix[i*ndofV+0]=true ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=true ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=true ; bc_val[i*ndofV+2]= 0.
          end
          if y[i]<eps
             bc_fix[i*ndofV+0]=true ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=true ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=true ; bc_val[i*ndofV+2]= 0.
          end
          if y[i]>(1-eps)
             bc_fix[i*ndofV+0]=true ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=true ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=true ; bc_val[i*ndofV+2]= 0.
          end
          if z[i]<eps
             bc_fix[i*ndofV+0]=true ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=true ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=true ; bc_val[i*ndofV+2]= 0.
          end
          if z[i]>(Lz-eps)
             bc_fix[i*ndofV+0]=true ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=true ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=true ; bc_val[i*ndofV+2]= 0.
          end
          if quarter && x[i]>(0.5-eps)
             bc_fix[i*ndofV+0]=true ; bc_val[i*ndofV+0]= 0.
          end
          if quarter && y[i]>(0.5-eps)
             bc_fix[i*ndofV+1]=true ; bc_val[i*ndofV+1]= 0.
          end
      end
   end

end

if experiment==5
   for i=1:NV
       if x[i]<eps
          bc_fix[(i-1)*ndofV+1]=true ; bc_val[(i-1)*ndofV+1]= uth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+2]=true ; bc_val[(i-1)*ndofV+2]= vth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+3]=true ; bc_val[(i-1)*ndofV+3]= wth(x[i],y[i],z[i])
       end 
       if x[i]>(1-eps)
          bc_fix[(i-1)*ndofV+1]=true ; bc_val[(i-1)*ndofV+1]= uth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+2]=true ; bc_val[(i-1)*ndofV+2]= vth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+3]=true ; bc_val[(i-1)*ndofV+3]= wth(x[i],y[i],z[i])
       end 
       if y[i]<eps
          bc_fix[(i-1)*ndofV+1]=true ; bc_val[(i-1)*ndofV+1]= uth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+2]=true ; bc_val[(i-1)*ndofV+2]= vth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+3]=true ; bc_val[(i-1)*ndofV+3]= wth(x[i],y[i],z[i])
       end 
       if y[i]>(1-eps)
          bc_fix[(i-1)*ndofV+1]=true ; bc_val[(i-1)*ndofV+1]= uth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+2]=true ; bc_val[(i-1)*ndofV+2]= vth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+3]=true ; bc_val[(i-1)*ndofV+3]= wth(x[i],y[i],z[i])
       end 
       if z[i]<eps
          bc_fix[(i-1)*ndofV+1]=true ; bc_val[(i-1)*ndofV+1]= uth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+2]=true ; bc_val[(i-1)*ndofV+2]= vth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+3]=true ; bc_val[(i-1)*ndofV+3]= wth(x[i],y[i],z[i])
       end 
       if z[i]>(Lz-eps)
          bc_fix[(i-1)*ndofV+1]=true ; bc_val[(i-1)*ndofV+1]= uth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+2]=true ; bc_val[(i-1)*ndofV+2]= vth(x[i],y[i],z[i])
          bc_fix[(i-1)*ndofV+3]=true ; bc_val[(i-1)*ndofV+3]= wth(x[i],y[i],z[i])
       end 
   end 
end 

end ; println("boundary conditions array ",time1 ," s") ; flush(stdout)

#################################################################
# build FE matrix
#################################################################
time1 = @elapsed begin

a_mat = spzeros(Nfem,Nfem)
b_mat = zeros(Float64,(6,ndofV*m))
rhs   = zeros(Float64,Nfem)
N     = zeros(Float64,m)
dNdx  = zeros(Float64,m)
dNdy  = zeros(Float64,m)
dNdz  = zeros(Float64,m)
dNdr  = zeros(Float64,m)
dNds  = zeros(Float64,m)
dNdt  = zeros(Float64,m)
u     = zeros(Float64,NV)
v     = zeros(Float64,NV)
w     = zeros(Float64,NV)
b_mat = zeros(Float64,(6,ndofV*m))
jcb   = zeros(Float64,(3,3))

k_mat = Float64[1 1 1 0 0 0 ; 
                1 1 1 0 0 0 ;
                1 1 1 0 0 0 ;
                0 0 0 0 0 0 ;
                0 0 0 0 0 0 ;
                0 0 0 0 0 0 ]

c_mat = Float64[2 0 0 0 0 0 ; 
                0 2 0 0 0 0 ; 
                0 0 2 0 0 0 ; 
                0 0 0 1 0 0 ; 
                0 0 0 0 1 0 ;
                0 0 0 0 0 1 ]


for iel = 1:nel

    #println(iel,"/",nel)
    #time2 = @elapsed begin

    a_el= zeros(Float64,m*ndofV,m*ndofV)
    b_el= zeros(Float64,m*ndofV)

    # integrate viscous term at 4 quadrature points
    for iq=-1:2:1
        for jq=-1:2:1
            for kq=-1:2:1

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1*1*1

                # calculate shape functions
                N=NNN(rq,sq,tq)
                dNdr=dNNNdr(rq,sq,tq)
                dNds=dNNNds(rq,sq,tq)
                dNdt=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                jcb[1,1]=dot(dNdr,x[icon[:,iel]])
                jcb[1,2]=dot(dNdr,y[icon[:,iel]])
                jcb[1,3]=dot(dNdr,z[icon[:,iel]])
                jcb[2,1]=dot(dNds,x[icon[:,iel]])
                jcb[2,2]=dot(dNds,y[icon[:,iel]])
                jcb[2,3]=dot(dNds,z[icon[:,iel]])
                jcb[3,1]=dot(dNdt,x[icon[:,iel]])
                jcb[3,2]=dot(dNdt,y[icon[:,iel]])
                jcb[3,3]=dot(dNdt,z[icon[:,iel]])

                # calculate the determinant of the jacobian
                jcob = det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi = inv(jcb)

                # compute coordinates of quadrature points
                xq=dot(N,x[icon[:,iel]])
                yq=dot(N,y[icon[:,iel]])
                zq=dot(N,z[icon[:,iel]])

                # compute dNdx, dNdy, dNdz
                for k=1:m
                    dNdx[k]=jcbi[1,1]*dNdr[k]+jcbi[1,2]*dNds[k]+jcbi[1,3]*dNdt[k]
                    dNdy[k]=jcbi[2,1]*dNdr[k]+jcbi[2,2]*dNds[k]+jcbi[2,3]*dNdt[k]
                    dNdz[k]=jcbi[3,1]*dNdr[k]+jcbi[3,2]*dNds[k]+jcbi[3,3]*dNdt[k]
                end

                # construct 3x8 b_mat matrix
                for i=1:m
                    b_mat[1:6, 3*(i-1)+1:3*i] = Float64[dNdx[i] 0.      0.     ;
                                                        0.      dNdy[i] 0.     ;
                                                        0.      0.      dNdz[i];
                                                        dNdy[i] dNdx[i] 0.     ;
                                                        dNdz[i] 0.      dNdx[i];
                                                        0.      dNdz[i] dNdy[i]]
                end

                # compute elemental matrix
                a_el += transpose(b_mat) * (c_mat * b_mat) * viscosity(xq,yq,zq)*weightq*jcob

                # compute elemental rhs vector
                for i=1:m
                    b_el[ndofV*(i-1)+1]+=N[i]*jcob*weightq*bx(xq,yq,zq)
                    b_el[ndofV*(i-1)+2]+=N[i]*jcob*weightq*by(xq,yq,zq)
                    b_el[ndofV*(i-1)+3]+=N[i]*jcob*weightq*bz(xq,yq,zq)
                end 

            end #kq 
        end #jq  
    end #iq  

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    tq=0.
    weightq=2*2*2

    # calculate shape functions
    N=NNN(rq,sq,tq)
    dNdr=dNNNdr(rq,sq,tq)
    dNds=dNNNds(rq,sq,tq)
    dNdt=dNNNdt(rq,sq,tq)

    # calculate jacobian matrix
    jcb[1,1]=dot(dNdr,x[icon[:,iel]])
    jcb[1,2]=dot(dNdr,y[icon[:,iel]])
    jcb[1,3]=dot(dNdr,z[icon[:,iel]])
    jcb[2,1]=dot(dNds,x[icon[:,iel]])
    jcb[2,2]=dot(dNds,y[icon[:,iel]])
    jcb[2,3]=dot(dNds,z[icon[:,iel]])
    jcb[3,1]=dot(dNdt,x[icon[:,iel]])
    jcb[3,2]=dot(dNdt,y[icon[:,iel]])
    jcb[3,3]=dot(dNdt,z[icon[:,iel]])
    jcob=det(jcb)
    jcbi=inv(jcb)

    # compute dNdx, dNdy, dNdz
    for k=1:m
        dNdx[k]=jcbi[1,1]*dNdr[k]+jcbi[1,2]*dNds[k]+jcbi[1,3]*dNdt[k]
        dNdy[k]=jcbi[2,1]*dNdr[k]+jcbi[2,2]*dNds[k]+jcbi[2,3]*dNdt[k]
        dNdz[k]=jcbi[3,1]*dNdr[k]+jcbi[3,2]*dNds[k]+jcbi[3,3]*dNdt[k]
    end

    # compute gradient matrix
    for i=1:m
        b_mat[1:6, 3*(i-1)+1:3*i] = Float64[dNdx[i] 0.      0.     ;
                                            0.      dNdy[i] 0.     ;
                                            0.      0.      dNdz[i];
                                            dNdy[i] dNdx[i] 0.     ;
                                            dNdz[i] 0.      dNdx[i];
                                            0.      dNdz[i] dNdy[i]]
    end

    # compute elemental matrix
    a_el += transpose(b_mat) * (k_mat * b_mat) * penalty*weightq*jcob

    # apply boundary conditions
    for k1 = 1:m 
        for i1 = 1:ndofV
            m1 =ndofV*(icon[k1,iel]-1)+i1
            if bc_fix[m1]
               fixt=bc_val[m1]
               ikk=ndofV*(k1-1)+i1
               aref=a_el[ikk,ikk]
               for jkk = 1:(m*ndofV)
                   b_el[jkk]-=a_el[jkk,ikk]*fixt
                   a_el[ikk,jkk]=0.
                   a_el[jkk,ikk]=0.
               end
               a_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
           end
       end
   end 

   # assemble matrix a_mat and right hand side rhs
   for k1 = 1:m 
        for i1 = 1:ndofV
            ikk=ndofV*(k1-1)+i1
            m1 =ndofV*(icon[k1,iel]-1)+i1
            for k2 = 1:m 
                for i2 = 1:ndofV
                    jkk=ndofV*(k2-1)      +i2 
                    m2 =ndofV*(icon[k2,iel]-1)+i2
                    a_mat[m1,m2]+=a_el[ikk,jkk]
                end
            end
            rhs[m1]+=b_el[ikk]
        end
    end

    #end ; println(time2, " s")

end #iel

end ; println("build FE matrix ",time1 ," s | nel= ", nel) ; flush(stdout)

#################################################################
# solve system
#################################################################
time1 = @elapsed begin

#@time sol= cholesky(a_mat)\rhs
@time sol= lu(a_mat)\rhs

end ; println("solve linear system ", time1, " s | Nfem= ",Nfem)

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
time1 = @elapsed begin

sol1 =(reshape(sol,(3,NV)))

u= sol1[1,:]
v= sol1[2,:]
w= sol1[3,:]

println("     -> u (m,M): ", minimum(u)," , ", maximum(u))
println("     -> v (m,M): ", minimum(v)," , ", maximum(v))
println("     -> w (m,M): ", minimum(w)," , ", maximum(w))

end ; println("Split solution into u,v ", time1, " s")

#####################################################################
# retrieve pressure
#####################################################################
time1 = @elapsed begin

xc=zeros(Float64,nel)
yc=zeros(Float64,nel)
zc=zeros(Float64,nel)
p=zeros(Float64,nel)
exx=zeros(Float64,nel)
eyy=zeros(Float64,nel)
ezz=zeros(Float64,nel)
exy=zeros(Float64,nel)
exz=zeros(Float64,nel)
eyz=zeros(Float64,nel)

for iel =  1:nel

    rq=0.
    sq=0.
    tq=0.
    weightq=2*2*2

    # calculate shape functions
    N=NNN(rq,sq,tq)
    dNdr=dNNNdr(rq,sq,tq)
    dNds=dNNNds(rq,sq,tq)
    dNdt=dNNNdt(rq,sq,tq)

    # calculate jacobian matrix
    jcb[1,1]=dot(dNdr,x[icon[:,iel]])
    jcb[1,2]=dot(dNdr,y[icon[:,iel]])
    jcb[1,3]=dot(dNdr,z[icon[:,iel]])
    jcb[2,1]=dot(dNds,x[icon[:,iel]])
    jcb[2,2]=dot(dNds,y[icon[:,iel]])
    jcb[2,3]=dot(dNds,z[icon[:,iel]])
    jcb[3,1]=dot(dNdt,x[icon[:,iel]])
    jcb[3,2]=dot(dNdt,y[icon[:,iel]])
    jcb[3,3]=dot(dNdt,z[icon[:,iel]])
    jcob=det(jcb)
    jcbi=inv(jcb)

    # compute dNdx, dNdy, dNdz
    for k=1:m
        dNdx[k]=jcbi[1,1]*dNdr[k]+jcbi[1,2]*dNds[k]+jcbi[1,3]*dNdt[k]
        dNdy[k]=jcbi[2,1]*dNdr[k]+jcbi[2,2]*dNds[k]+jcbi[2,3]*dNdt[k]
        dNdz[k]=jcbi[3,1]*dNdr[k]+jcbi[3,2]*dNds[k]+jcbi[3,3]*dNdt[k]
    end

    xc[iel]=dot(N,x[icon[:,iel]])
    yc[iel]=dot(N,y[icon[:,iel]])
    zc[iel]=dot(N,z[icon[:,iel]])
    exx[iel]=dot(dNdx,u[icon[:,iel]])
    eyy[iel]=dot(dNdy,v[icon[:,iel]])
    ezz[iel]=dot(dNdz,w[icon[:,iel]])
    exy[iel]=0.5*dot(dNdx,v[icon[:,iel]])+0.5*dot(dNdy,u[icon[:,iel]])
    exz[iel]=0.5*dot(dNdx,w[icon[:,iel]])+0.5*dot(dNdz,u[icon[:,iel]])
    eyz[iel]=0.5*dot(dNdy,w[icon[:,iel]])+0.5*dot(dNdz,v[icon[:,iel]])
    p[iel]=-penalty*(exx[iel]+eyy[iel]+ezz[iel])

end 

println("     -> exx (m,M): ", minimum(exx)," , ", maximum(exx))
println("     -> eyy (m,M): ", minimum(eyy)," , ", maximum(eyy))
println("     -> ezz (m,M): ", minimum(ezz)," , ", maximum(ezz))
println("     -> exy (m,M): ", minimum(exy)," , ", maximum(exy))
println("     -> exz (m,M): ", minimum(exz)," , ", maximum(exz))
println("     -> eyz (m,M): ", minimum(eyz)," , ", maximum(eyz))
println("     -> p   (m,M): ", minimum(  p)," , ", maximum(  p))

end ; println("compute pressure ", time1, " s")

#####################################################################
# compute vrms and errors
#####################################################################
time1 = @elapsed begin

errv=0.
errp=0.
vrms=0.

for iel = 1:nel

    # integrate viscous term at 4 quadrature points
    for iq=-1:2:1
        for jq=-1:2:1
            for kq=-1:2:1

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1*1*1

                # calculate shape functions
                N=NNN(rq,sq,tq)
                dNdr=dNNNdr(rq,sq,tq)
                dNds=dNNNds(rq,sq,tq)
                dNdt=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                jcb[1,1]=dot(dNdr,x[icon[:,iel]])
                jcb[1,2]=dot(dNdr,y[icon[:,iel]])
                jcb[1,3]=dot(dNdr,z[icon[:,iel]])
                jcb[2,1]=dot(dNds,x[icon[:,iel]])
                jcb[2,2]=dot(dNds,y[icon[:,iel]])
                jcb[2,3]=dot(dNds,z[icon[:,iel]])
                jcb[3,1]=dot(dNdt,x[icon[:,iel]])
                jcb[3,2]=dot(dNdt,y[icon[:,iel]])
                jcb[3,3]=dot(dNdt,z[icon[:,iel]])
                jcob = det(jcb)

                xq=dot(N,x[icon[:,iel]])
                yq=dot(N,y[icon[:,iel]])
                zq=dot(N,z[icon[:,iel]])
                uq=dot(N,u[icon[:,iel]])
                vq=dot(N,v[icon[:,iel]])
                wq=dot(N,w[icon[:,iel]])

                global(errv)+=((uq-uth(xq,yq,zq))^2+
                               (vq-vth(xq,yq,zq))^2+
                               (wq-wth(xq,yq,zq))^2)*weightq*jcob

                global(errp)+=(p[iel]-pth(xq,yq,zq))^2*weightq*jcob

                global(vrms)+=(uq^2+vq^2+wq^2)*jcob*weightq

            end #kq 
        end #jq  
    end #iq  

end #iel

errv=sqrt(errv)
errp=sqrt(errp)
vrms=sqrt(vrms/Lx/Ly/Lz)

println("     -> nel= ", nel," vrms= ", vrms)

println("     -> nel= ", nel," errv= ", errv," errp= ", errp)

end ; println("compute errors ", time1, " s")

###########################################################
# export to vtu
###########################################################
time1 = @elapsed begin

if visu==1
   file=open("solution.vtu","w") 
   write(file, "<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   write(file, "<UnstructuredGrid> \n")
   @printf(file,"<Piece NumberOfPoints='  %d ' NumberOfCells='  %d '> \n",NV,nel)
   #--
   write(file,"<Points> \n")
   write(file,"<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i = 1:NV
   @printf(file,"%f %f %f \n",x[i],y[i],z[i])
   end
   write(file,"</DataArray> \n")
   write(file,"</Points> \n")
   #--
   write(file,"<PointData Scalars='scalars'>\n")
   write(file,"<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i = 1:NV
   @printf(file,"%f %f %f \n" ,u[i],v[i],w[i])
   end
   write(file,"</DataArray>\n")
   write(file,"</PointData>\n")
   #--
   write(file,"<CellData Scalars='scalars'>\n")
   write(file,"<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%f \n" ,p[iel])
   end
   write(file,"</DataArray> \n")
   #
   write(file,"<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%f \n" ,exx[iel])
   end
   write(file,"</DataArray> \n")
   write(file,"<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%f \n" ,eyy[iel])
   end
   write(file,"</DataArray> \n")
   write(file,"<DataArray type='Float32' Name='ezz' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%f \n" ,ezz[iel])
   end
   write(file,"</DataArray> \n")

   write(file,"<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%f \n" ,exy[iel])
   end
   write(file,"</DataArray> \n")

   write(file,"<DataArray type='Float32' Name='exz' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%f \n" ,exz[iel])
   end
   write(file,"</DataArray> \n")

   write(file,"<DataArray type='Float32' Name='eyz' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%f \n" ,eyz[iel])
   end
   write(file,"</DataArray> \n")

   write(file,"</CellData> \n")

   #####
   write(file,"<Cells>\n")
   #--
   write(file,"<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%d %d %d %d %d %d %d %d \n",icon[1,iel]-1,icon[2,iel]-1,
                                             icon[3,iel]-1,icon[4,iel]-1,
                                             icon[5,iel]-1,icon[6,iel]-1,
                                             icon[7,iel]-1,icon[8,iel]-1)
   end
   write(file,"</DataArray>\n")
   #--
   write(file,"<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel = 1:nel
   @printf(file,"%d \n" ,iel*8)
   end
   write(file,"</DataArray>\n")
   #--
   write(file,"<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel = 1:nel
   @printf(file,"%d \n" ,12)
   end
   write(file,"</DataArray>\n")
   #--
   write(file,"</Cells>\n")
   #####
   write(file,"</Piece>\n")
   write(file,"</UnstructuredGrid>\n")
   write(file,"</VTKFile>\n")
   close(file)

end #if visu

end ; println("export to vtu ", time1, " s")

println("-----------------------------")






