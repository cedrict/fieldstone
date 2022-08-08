#!/usr/bin/env julia
using Printf
using Statistics
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using Formatting

###########################################################
# Functions for body force components
###########################################################

function bx(x,y)
   val=Float64(((12-24*y)*x^4+(-24+48*y)*x^3 +
               (-48*y+72*y^2-48*y^3+12)*x^2 +
               (-2+24*y-72*y^2+48*y^3)*x +
               1-4*y+12*y^2-8*y^3))
   return val
end

function by(x, y)
   val=Float64(((8-48*y+48*y^2)*x^3+
               (-12+72*y-72*y^2)*x^2+
               (4-24*y+48*y^2-48*y^3+24*y^4)*x -
               12*y^2+24*y^3-12*y^4))
   return val
end

###########################################################
# analytical solution
###########################################################

function velocity_x(x,y)
   val=x^2*(1-x)^2*(2*y-6*y^2+4*y^3)
   return val
end

function velocity_y(x,y)
   val=-y^2*(1-y)^2*(2*x-6*x^2+4*x^3)
   return val
end

function pressure(x,y)
   val=x*(1-x)-1/6
   return val
end

###########################################################
# declare variables
###########################################################

println("-----------------------------")
println("---------- stone 01 ---------")
println("-----------------------------")
flush(stdout)

println("variable declaration")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain
Ly=1.  # vertical extent of the domain

if length(ARGS) == 3
    nelx = eval(parse(Int64,ARGS[1]))
    nely = eval(parse(Int64,ARGS[2]))
    visu = eval(parse(Int64,ARGS[3]))
else
    nelx = 160
    nely = 160
    visu = 1
end

@assert(Lx>0. , "Lx should be positive" )
@assert(Ly>0. , "Ly should be positive" )
@assert(nelx>0. , "nnx should be positive" )
@assert(nely>0. , "nny should be positive" )

nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

penalty=1.e7  # penalty coefficient value

viscosity=1.  # dynamic viscosity \eta

Nfem=nnp*ndof  # Total number of degrees of freedom

NV=nnx*nny

epsilon=1e-10

sqrt3=sqrt(3.)

###########################################################

println("nelx    = ", nelx)
println("nely    = ", nely)
println("Lx      = ", Lx)
println("Ly      = ", Ly)
println("penalty = ", penalty)
println("-----------------------------")

###########################################################
# grid point setup
###########################################################
time1 =  @elapsed begin

x=zeros(Float64,nnp)
y=zeros(Float64,nnp)

counter = 1
for j = 0:nely
    for i = 0: nelx
        x[counter]=Float64(i*Lx/(nelx))
        y[counter]=Float64(j*Ly/(nely))
        global(counter) += 1
    end
end

end ; println("Setup: grid points                 ", time1 ," s")

###########################################################
# build connectivity array
###########################################################
time1 =  @elapsed begin

icon = zeros(Int32,m,nel)
counter = 1
for j = 1:nely
    for i = 1:nelx
        icon[1,counter]= (i)+ (j-1) * (nelx +1)
        icon[2,counter] = i + 1 + (j-1) * (nelx + 1)
        icon[3,counter] = i + 1 + (j) * (nelx + 1)
        icon[4,counter] = i + (j ) * (nelx + 1)
        global(counter) += 1
    end
end

end ; println("Setup: connectivity array          ", time1 , " s")

###########################################################
# define boundary conditions. for this benchmark: no slip.
###########################################################
time1 = @elapsed begin

bc_fix = zeros(Bool,Nfem)
bc_val = zeros(Float64,Nfem)

for i = 1:nnp
    if x[i]<epsilon
       bc_fix[(i-1)*ndof+1] = true ; bc_val[(i-1)*ndof+1] = 0.
       bc_fix[(i-1)*ndof+2] = true ; bc_val[(i-1)*ndof+2] = 0.
    end
    if x[i]> (Lx-epsilon)
       bc_fix[(i-1)*ndof+1] = true ; bc_val[(i-1)*ndof+1] = 0.
       bc_fix[(i-1)*ndof+2] = true ; bc_val[(i-1)*ndof+2] = 0.
    end
    if y[i]<epsilon
       bc_fix[(i-1)*ndof+1] = true ; bc_val[(i-1)*ndof+1] = 0.
       bc_fix[(i-1)*ndof+2] = true ; bc_val[(i-1)*ndof+2] = 0.
    end
    if y[i]> (Ly-epsilon)
       bc_fix[(i-1)*ndof+1] = true ; bc_val[(i-1)*ndof+1] = 0.
       bc_fix[(i-1)*ndof+2] = true ; bc_val[(i-1)*ndof+2] = 0.
    end
end

end ; println("Setup: boundary conditions array   ",time1 ," s")

#################################################################
# build FE matrix
# r,s are the reduced coordinates in the [-1:1]x[-1:1] ref elt
#################################################################
time1 = @elapsed begin

a_mat = spzeros(Nfem,Nfem)
#a_mat = zeros(Float64,(Nfem,Nfem) )
b_mat = zeros(Float64,(3,ndof*m))
rhs   = zeros(Float64,Nfem)
N     = zeros(Float64,m)
dNdx  = zeros(Float64,m)
dNdy  = zeros(Float64,m)
dNdr  = zeros(Float64,m)
dNds  = zeros(Float64,m)
u     = zeros(Float64,nnp)
v     = zeros(Float64,nnp)
k_mat = Float64[1 1 0 ; 1 1 0 ; 0 0 0]
c_mat = Float64[2 0 0 ; 0 2 0 ; 0 0 1]

for iel = 1:nel

    b_el= zeros(Float64,m*ndof)
    a_el= zeros(Float64,m*ndof, m*ndof)

    for iq = -1:2:1
        for jq = -1:2:1

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1. *1.

            # calculate shape functions
            N[1]=0.25*(1-rq)*(1-sq)
            N[2]=0.25*(1+rq)*(1-sq)
            N[3]=0.25*(1+rq)*(1+sq)
            N[4]=0.25*(1-rq)*(1+sq)

            # calculate shape function derivatives
            dNdr[1]=-0.25*(1-sq) ; dNds[1]=-0.25*(1-rq)
            dNdr[2]=+0.25*(1-sq) ; dNds[2]=-0.25*(1+rq)
            dNdr[3]=+0.25*(1+sq) ; dNds[3]=+0.25*(1+rq)
            dNdr[4]=-0.25*(1+sq) ; dNds[4]=+0.25*(1-rq)

            # calculate jacobian matrix
            jcb = zeros(Float64,(2,2))
            for k = 1:m
                jcb[1,1] += dNdr[k]*x[icon[k,iel]]
                jcb[1,2] += dNdr[k]*y[icon[k,iel]]
                jcb[2,1] += dNds[k]*x[icon[k,iel]]
                jcb[2,2] += dNds[k]*y[icon[k,iel]]
            end

            # calculate the determinant of the jacobian
            jcob = det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k = 1:m
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[1,1]*dNdr[k]+jcbi[1,2]*dNds[k]
                dNdy[k]=jcbi[2,1]*dNdr[k]+jcbi[2,2]*dNds[k]
            end

            # construct 3x8 b_mat matrix
            for i = 1:m
                i1=2*i-1
                i2=2*i
                b_mat[1,i1]=dNdx[i] ; b_mat[1,i2]=0.
                b_mat[2,i1]=0.      ; b_mat[2,i2]=dNdy[i]
                b_mat[3,i1]=dNdy[i] ; b_mat[3,i2]=dNdx[i]
            end

            # compute elemental a_mat matrix
            a_el+=transpose(b_mat) *  (c_mat * b_mat) .* viscosity*weightq*jcob

            # compute elemental rhs vector
            for i = 1:m
                b_el[2*i-1] += N[i]*jcob*weightq*bx(xq,yq)
                b_el[2*i]   += N[i]*jcob*weightq*by(xq,yq)
            end

        end # jq
    end # iq

    rq=0.
    sq=0.
    weightq=2. *2.

    # calculate shape functions
    N[1]=0.25 * (1. -rq) * (1. -sq)
    N[2]=0.25 * (1. +rq) * (1. -sq)
    N[3]=0.25 * (1. +rq) * (1. +sq)
    N[4]=0.25 * (1. -rq) * (1. +sq)

    # calculate shape function derivatives
    dNdr[1]=-0.25*(1. -sq) ; dNds[1]=-0.25*(1. -rq)
    dNdr[2]=+0.25*(1. -sq) ; dNds[2]=-0.25*(1. +rq)
    dNdr[3]=+0.25*(1. +sq) ; dNds[3]=+0.25*(1. +rq)
    dNdr[4]=-0.25*(1. +sq) ; dNds[4]=+0.25*(1. -rq)

    # calculate jacobian matrix
    jcb = zeros(Float64,(2,2))
    for k = 1:m
        jcb[1, 1] += dNdr[k]*x[icon[k,iel]]
        jcb[1, 2] += dNdr[k]*y[icon[k,iel]]
        jcb[2, 1] += dNds[k]*x[icon[k,iel]]
        jcb[2, 2] += dNds[k]*y[icon[k,iel]]
    end
    jcob = det(jcb)
    jcbi = inv(jcb)

    # compute dNdx & dNdy
    for k = 1:m
        dNdx[k]=jcbi[1,1]*dNdr[k]+jcbi[1,2]*dNds[k]
        dNdy[k]=jcbi[2,1]*dNdr[k]+jcbi[2,2]*dNds[k]
    end

    # construct 3x8 b_mat matrix
    for i = 1:m
        i1=2*i-1
        i2=2*i
        b_mat[1,i1]=dNdx[i] ; b_mat[1,i2]=0.
        b_mat[2,i1]=0.      ; b_mat[2,i2]=dNdy[i]
        b_mat[3,i1]=dNdy[i] ; b_mat[3,i2]=dNdx[i]
    end

    # compute elemental a_mat matrix
    a_el +=transpose(b_mat)*(k_mat*b_mat)*penalty*weightq*jcob

    for k1 = 1:m
        for i1 = 1:ndof
            m1 =ndof*(icon[k1,iel]-1)+i1
            if bc_fix[m1]
               fixt=bc_val[m1]
               ikk=ndof*(k1-1)+i1
               aref=a_el[ikk,ikk]
               for jkk = 1:(m*ndof)
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
        for i1 = 1:ndof
            ikk=ndof*(k1-1)+i1
            m1 =ndof*(icon[k1,iel]-1)+i1
            for k2 = 1:m
                for i2 = 1:ndof
                    jkk=ndof*(k2-1)      +i2
                    m2 =ndof*(icon[k2,iel]-1)+i2
                    a_mat[m1,m2]+=a_el[ikk,jkk]
                end
            end
            rhs[m1]+=b_el[ikk]
        end
    end

end # nel

end ; println("Build FE matrix                    ", time1, " s | Nfem= ", Nfem)

#################################################################
# impose boundary conditions
# for now it is done outside of the previous loop, we will see
# later in the course how it can be incorporated seamlessly in it.
#################################################################
time = @elapsed begin

for i = 1:Nfem
    if bc_fix[i]
       a_matref = a_mat[i,i]
       for j = 1:Nfem
           rhs[j] -=a_mat[i,j] * bc_val[i]
           a_mat[i,j] = 0
           a_mat[j,i] = 0
       end
       a_mat[i,i] = a_matref
       rhs[i]=a_matref*bc_val[i]
    end
end

# println("a_mat (m,M) =" , minimum(a_mat),",",maximum(a_mat))
# println("rhs   (m,M) =", minimum(rhs),maximum(rhs))

end ; println("Impose boundary conditions         ", time, " s")

#################################################################
# solve system
#################################################################
time1 = @elapsed begin

#@time Sps_amat=sparse(a_mat)
#@time sol= Sps_amat\rhs
#@time sol= cholesky(Sps_amat)\rhs
#@time sol= ldlt(Sps_amat)\rhs
#@time sol= lu(Sps_amat)\rhs

#@time sol= cholesky(a_mat)\rhs
@time sol= lu(a_mat)\rhs

end ; println("Solve linear system                ", time1, " s | Nfem= ",Nfem)

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
time1 = @elapsed begin

sol1 =(reshape(sol,(2,nnp)))

u= sol1[1,:]
v= sol1[2,:]

# println("u = ", minimum(u)," , ",maximum(u))
# println("v = ", minimum(v)," , ",maximum(v))

end ; println("Split solution into u,v            ", time1, " s")

###########################################################
# retrieve pressure and strain rate components 
# in the middle of the elements.
###########################################################
time1 = @elapsed begin

xc=zeros(Float64,nel)
yc=zeros(Float64,nel)
p=zeros(Float64,nel)
exx=zeros(Float64,nel)
eyy=zeros(Float64,nel)
exy=zeros(Float64,nel)

for iel =  1:nel

    rq=0.
    sq=0.
    weightq=2*2

    # calculate shape functions
    N[1]=0.25*(1-rq)*(1-sq)
    N[2]=0.25*(1+rq)*(1-sq)
    N[3]=0.25*(1+rq)*(1+sq)
    N[4]=0.25*(1-rq)*(1+sq)

    # calculate shape function derivatives
    dNdr[1]=-0.25*(1-sq) ; dNds[1]=-0.25*(1-rq)
    dNdr[2]=+0.25*(1-sq) ; dNds[2]=-0.25*(1+rq)
    dNdr[3]=+0.25*(1+sq) ; dNds[3]=+0.25*(1+rq)
    dNdr[4]=-0.25*(1+sq) ; dNds[4]=+0.25*(1-rq)

    # calculate jacobian matrix
    jcb = zeros(Float64,(2,2))
    for k = 1:m
        jcb[1, 1] += dNdr[k]*x[icon[k,iel]]
        jcb[1, 2] += dNdr[k]*y[icon[k,iel]]
        jcb[2, 1] += dNds[k]*x[icon[k,iel]]
        jcb[2, 2] += dNds[k]*y[icon[k,iel]]
    end

    # calculate the determinant of the jacobian
    jcob = det(jcb)
    # calculate inverse of the jacobian matrix
    jcbi = inv(jcb)

    # compute dNdx & dNdy
    for k = 1:m
        dNdx[k]=jcbi[1,1]*dNdr[k]+jcbi[1,2]*dNds[k]
        dNdy[k]=jcbi[2,1]*dNdr[k]+jcbi[2,2]*dNds[k]
    end

    for k = 1:m
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]] + 0.5 * dNdx[k]*v[icon[k,iel]]
    end

    p[iel] = -penalty*(exx[iel]+eyy[iel])

end

# println("p (m,M)", minimum(p)," , ", maximum(p))
# println("exx (m,M)", minimum(exx)," , ", maximum(exx))
# println("eyy (m,M)", minimum(eyy)," , ", maximum(eyy))
# println("exy (m,M)", minimum(exy)," , ", maximum(exy))

end ; println("compute pressure                   ", time1, " s")

##################################################################
## compute error
##################################################################
time1 = @elapsed begin

error_u = zeros(Float64,nnp)
error_v = zeros(Float64,nnp)
error_p = zeros(Float64,nel)

for i = 1:nnp
    error_u[i]=u[i]-velocity_x(x[i],y[i])
    error_v[i]=v[i]-velocity_y(x[i],y[i])
end

for i = 1:nel
    error_p[i]=p[i]-pressure(xc[i],yc[i])
end

end ; println("compute nodal error for plot       ", time1, " s")

#################################################################
# compute error in L2 norm
#################################################################
time1 = @elapsed begin

errv=0.
errp=0.

for iel = 1:nel
    for iq = -1:2:1
        for jq = -1:2:1

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1. *1.
            # calculate shape functions
            N[1]=0.25 * (1. -rq) * (1. -sq)
            N[2]=0.25 * (1. +rq) * (1. -sq)
            N[3]=0.25 * (1. +rq) * (1. +sq)
            N[4]=0.25 * (1. -rq) * (1. +sq)

            # calculate shape function derivatives
            dNdr[1]=-0.25*(1. -sq) ; dNds[1]=-0.25*(1. -rq)
            dNdr[2]=+0.25*(1. -sq) ; dNds[2]=-0.25*(1. +rq)
            dNdr[3]=+0.25*(1. +sq) ; dNds[3]=+0.25*(1. +rq)
            dNdr[4]=-0.25*(1. +sq) ; dNds[4]=+0.25*(1. -rq)

            # calculate jacobian matrix
            jcb = zeros(Float64, (2, 2))
            for k = 1:m
                jcb[1, 1] += dNdr[k]*x[icon[k,iel]]
                jcb[1, 2] += dNdr[k]*y[icon[k,iel]]
                jcb[2, 1] += dNds[k]*x[icon[k,iel]]
                jcb[2, 2] += dNds[k]*y[icon[k,iel]]
            end

            # calculate the determinant of the jacobian
            jcob = det(jcb)

            xq=0.
            yq=0.
            uq=0.
            vq=0.

            for k = 1:m
                xq += N[k]*x[icon[k,iel]]
                yq += N[k]*y[icon[k,iel]]
                uq += N[k]*u[icon[k,iel]]
                vq += N[k]*v[icon[k,iel]]
            end

            global(errv) += ((uq-velocity_x(xq,yq))^2+(vq-velocity_y(xq,yq))^2)*weightq*jcob
            global(errp) += (p[iel]-pressure(xq,yq))^2*weightq*jcob
        end
    end
end

errv=sqrt(errv)
errp=sqrt(errp)


println("nel = ", nel,"     errv = ", errv,"     errp = ", errp)

end ; println("compute errors                     ", time1, " s")

###########################################################
## naive depth averaging
###########################################################
time1 = @elapsed begin

avrg_u_profile = zeros(Float64,nny)
avrg_v_profile = zeros(Float64,nny)
avrg_vel_profile = zeros(Float64,nny)
avrg_y_profile = zeros(Float64,nny)

counter = 1
for j = 1:nny
    for i = 1:nnx
        avrg_y_profile[j] += y[counter]/nnx
        avrg_u_profile[j] += u[counter]/nnx
        avrg_v_profile[j] += v[counter]/nnx
        avrg_v_profile[j] += sqrt(u[counter]^2+v[counter]^2)/nnx
        global(counter)+=1
    end
end

end ; println("naive depth averaging              ", time1, " s")

###########################################################
# export to vtu
###########################################################
time1 = @elapsed begin

if visu == 1
   file=open("solution.vtu","w") #do file
   write(file, "<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   write(file, "<UnstructuredGrid> \n")
   printfmt(file,"<Piece NumberOfPoints='  {:d} ' NumberOfCells='   {:d} '> \n",NV,nel)
   write(file,"<Points> \n")
   write(file,"<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i = 1:NV
       printfmt(file,"{:10f} {:10f} {:10f} \n",x[i],y[i],0.)
   end
   write(file,"</DataArray> \n")
   write(file,"</Points> \n")

   write(file,"<PointData Scalars='scalars'>\n")
   write(file,"<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i = 1:NV
       printfmt(file,"{:10f} {:10f} {:10f} \n" ,u[i],v[i],0.)
   end
   write(file,"</DataArray>\n")

   write(file,"<DataArray type='Float32' NumberOfComponents='3' Name='velocity error' Format='ascii'> \n")
   for i = 1:NV
       printfmt(file,"{:10f} {:10f} {:10f} \n" ,error_u[i],error_v[i],0.)
   end
   write(file,"</DataArray>\n")
   write(file,"</PointData>\n")

   #--
   write(file,"<CellData Scalars='scalars'>\n")
   write(file,"<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
   for i = 1:nel
       printfmt(file,"{:10f} \n" ,p[i])
   end
   write(file,"</DataArray> \n")

   write(file,"<DataArray type='Float32' Name='pressure error' Format='ascii'> \n")
   for i = 1:nel
       printfmt(file,"{:10f} \n" ,error_p[i])
   end
   write(file,"</DataArray> \n")
   #
   write(file,"<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for i = 1:nel
       printfmt(file,"{:10f} \n" ,exx[i])
   end
   write(file,"</DataArray> \n")

   write(file,"<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for i = 1:nel
       printfmt(file,"{:10f} \n" ,eyy[i])
   end
   write(file,"</DataArray> \n")

   write(file,"<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for i = 1:nel
       printfmt(file,"{:10f} \n" ,exy[i])
   end
   write(file,"</DataArray> \n")
   write(file,"</CellData> \n")

   #####
   write(file,"<Cells>\n")
   #--
   write(file,"<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel = 1:nel
       printfmt(file,"{:d} {:d} {:d} {:d} \n" ,icon[1,iel]-1,icon[2,iel]-1,icon[3,iel]-1,icon[4,iel]-1)
   end
   write(file,"</DataArray>\n")

   write(file,"<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel = 1:nel
       printfmt(file,"{:d} \n" ,((iel)*4))
   end
   write(file,"</DataArray>\n")

   write(file,"<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel = 1:nel
       printfmt(file,"{:d} \n" ,9)
   end
   write(file,"</DataArray>\n")

   write(file,"</Cells>\n")
   #####
   write(file,"</Piece>\n")
   write(file,"</UnstructuredGrid>\n")
   write(file,"</VTKFile>\n")
   close(file)

end # if

end ; println("export to vtu                      ", time1, " s")

println("-----------------------------")
