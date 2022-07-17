#!/usr/bin/env julia
using Printf
using Statistics
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using Formatting

#------------------------------------------------------------------------------

function rho(rho0,alpha,T,T0)
    val=rho0*(1. -alpha*(T-T0))
    return val
end

function eta(T)
    val=1.
    return val
end
#------------------------------------------------------------------------------

println("-----------------------------")
println("--------fieldstone 03--------")
println("-----------------------------")
totaltime =  @elapsed begin

sqrt3=sqrt(3.)
epsilon=1.e-10

ndim=2       # number of space dimensions
m=4          # number of nodes making up an element
ndofV=2      # number of degrees of freedom per node
ndofT=1      # number of degrees of freedom per node
Lx=1.        # horizontal extent of the domain
Ly=1.        # vertical extent of the domain
Ra=1e6       # Rayleigh number
alpha=1e-2   # thermal expansion coefficient
hcond=1.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density
T0=0         # reference temperature
CFL=1       # CFL number
gy=-Ra/alpha # vertical component of gravity vector
penalty=1.e7 # penalty coefficient value
nstep=15000 # maximum number of timestep
tol=1.e-6

visu=1
if length(ARGS) == 2
    nelx = eval(parse(Int64,ARGS[1]))
    nely = eval(parse(Int64,ARGS[2]))
else
    nelx = 48
    nely = 48
end

hx=Lx/float(nelx)
hy=Ly/float(nely)

nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

NfemV=nnp*ndofV  # Total number of degrees of velocity freedom
NfemT=nnp*ndofT  # Total number of degrees of temperature freedom

#################################################################
# grid point setup
#################################################################
global(time1) =  @elapsed begin
x=zeros(Float64,nnp)
y=zeros(Float64,nnp)

global(counter) = 0
for j = 0:nely
    for i = 0: nelx
        global(counter) += 1
        x[counter]=Float64(i*Lx/(nelx))
        y[counter]=Float64(j*Ly/(nely))

    end
end
end  # end timing
# time_elapsed = time()- start_time
println("Setup: grid points                 ", time1 ,"   sec")

##################################################################
## build connectivity array
##################################################################

global(time1) =  @elapsed begin

icon = zeros(Int32,m,nel)
counter = 0
for j = 1:nely
    for i = 1:nelx
        global(counter) += 1
        icon[1, counter]= (i)+ (j-1) * (nelx +1)
        icon[2, counter] = i + 1 + (j-1) * (nelx + 1)
        icon[3, counter] = i + 1 + (j) * (nelx + 1)
        icon[4, counter] = i + (j ) * (nelx + 1)

    end
end

end # end timing
println("Setup: connectivity array          ", time1 , "   sec")

#####################################################################
# define velocity boundary conditions
#####################################################################

global(time1) = @elapsed begin

bc_fixV = zeros(Bool,NfemV)
bc_valV = zeros(Float64,NfemV)
for i = 1:nnp
    if x[i] < epsilon
        bc_fixV[(i-1)*ndofV+1]     = true ; bc_valV[(i-1)*ndofV+1]    = 0.
    end
    if x[i] > (Lx-epsilon)
        bc_fixV[(i-1)*ndofV+1]     = true ; bc_valV[(i-1)*ndofV+1]    = 0.
    end
    if y[i] < epsilon
        bc_fixV[(i-1)*ndofV+2]     = true ; bc_valV[(i-1)*ndofV+2]    = 0.
    end
    if y[i] > (Ly-epsilon)
        bc_fixV[(i-1)*ndofV+2]     = true ; bc_valV[(i-1)*ndofV+2]    = 0.
    end
end
end # end timing
println("Setup: defining temperature boundary conditions array   ",time1 ,"   sec")

#####################################################################
# define temperature boundary conditions
#####################################################################

global(time1) = @elapsed begin

bc_fixT = zeros(Bool,NfemT)
bc_valT = zeros(Float64,NfemT)
for i = 1:nnp
    if y[i]<epsilon
        bc_fixT[i]     = true ; bc_valT[i]    = 1.
    end
    if y[i]> (Ly-epsilon)
        bc_fixT[i]     = true ; bc_valT[i]    = 0.
    end
end

end # end timing
println("Setup: defining temperature boundary conditions array   ",time1 ,"   sec")

#####################################################################
# initial temperature
#####################################################################

T = zeros(Float64,nnp)
T_prev = zeros(Float64,nnp)

for i = 1:nnp
    T[i]=1. -y[i]-0.01*cos(pi*x[i])*sin(pi*y[i])
end

T_prev[:]=T[:]

#####################################################################
# create necessary arrays
#####################################################################

N     = zeros(Float64, m)    # shape functions
dNdx  = zeros(Float64, m)    # shape functions derivatives
dNdy  = zeros(Float64, m)    # shape functions derivatives
dNdr  = zeros(Float64, m)    # shape functions derivatives
dNds  = zeros(Float64, m)    # shape functions derivatives
u     = zeros(Float64, nnp)  # x-component velocity
v     = zeros(Float64, nnp)  # y-component velocity
u_prev= zeros(Float64, nnp)  # x-component velocity
v_prev= zeros(Float64, nnp)  # y-component velocity
Tvect = zeros(Float64, 4)
k_mat = Float64[1 1 0 ; 1 1 0 ; 0 0 0]
c_mat = Float64[2 0 0 ; 0 2 0 ; 0 0 1]

#####################################################################
# time stepping loop
#####################################################################


timedt=0

for istep = 1:nstep
    println("-----------------------------")
    println("istep= ", istep)
    println("-----------------------------")
    timeloop = @elapsed begin
    global(time1) = @elapsed begin

    global(a_mat) = zeros(NfemV , NfemV )
    global(b_mat) = zeros(Float64,(3, ndofV*m))
    global(rhs)   = zeros(Float64, NfemV)

    for iel =1:nel

        global(b_el)= zeros(Float64,m*ndofV)
        global(a_el)= zeros(Float64,m*ndofV, m*ndofV)

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
                # calculate inverse of the jacobian matrix
                jcbi = inv(jcb)

                # compute dNdx & dNdy
                xq=0.0
                yq=0.0
                Tq=0.0

                # compute dNdx & dNdy
                for k = 1:m
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
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
                a_el +=transpose(b_mat) *  (c_mat * b_mat) .* eta(Tq)*weightq*jcob
                # compute elemental rhs vector
                for i = 1:m
                    b_el[2*i] += N[i]*jcob*weightq*rho(rho0,alpha,Tq,T0)*gy
                end
            end
        end
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
        jcb = zeros(Float64, (2, 2))

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
        # construct 3x8 b_mat matrix
        for i = 1:m
            i1=2*i-1
            i2=2*i

            b_mat[1,i1]=dNdx[i] ; b_mat[1,i2]=0.
            b_mat[2,i1]=0.      ; b_mat[2,i2]=dNdy[i]
            b_mat[3,i1]=dNdy[i] ; b_mat[3,i2]=dNdx[i]
        end

        # compute elemental a_mat matrix
        a_el +=transpose(b_mat) *  (k_mat * b_mat) * penalty*weightq*jcob

        for k1 = 1:m
            for i1 = 1:ndofV
                m1 =ndofV*(icon[k1,iel]-1)+i1
                if bc_fixV[m1]
                   fixt=bc_valV[m1]
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
    end # end iel


    end # timing
    println("","Build Stokes matrix and rhs                  ", time1, "  sec")

    #################################################################
    # solve system
    #################################################################

    global(time1) = @elapsed begin
    Sps_amat=sparse(a_mat)
    global(sol)= Sps_amat\rhs
    end

    println("Solve v time                           ", time1, "  sec")

    #####################################################################
    # put solution into separate x,y velocity arrays
    #####################################################################
    global(time1) = @elapsed begin

    global(sol1) =(reshape(sol,(2,nnp)))

    global(u)= sol1[1,:]
    global(v)= sol1[2,:]

    println("u = ", minimum(u)," , ",maximum(u))
    println("v = ", minimum(v)," , ",maximum(v))

    end
    println("Split solution into u,v            ", time1, "  sec")

    #################################################################
    # compute timestep
    #################################################################

    dt1=CFL*(Lx/nelx)/(maximum(sqrt.(u .^ 2+v .^ 2)))

    dt2=CFL*(Lx/nelx) .^ 2/(hcond/hcapa/rho0)

    dt=minimum([dt1,dt2])

    global(timedt) += dt

    printfmtln("dt1= {:.6f}", dt1)
    printfmtln("dt2= {:.6f}", dt2)
    printfmtln("dt= {:.6f}", dt)

    #################################################################
    # build temperature matrix
    #################################################################
    global(time1) = @elapsed begin

    global(a_mat) = zeros(Float64,(NfemT , NfemT) )
    global(b_mat) = zeros(Float64,(2, ndofT*m))
    global(N_mat) = zeros(Float64,(m,1))
    global(rhs)   = zeros(Float64, NfemT)

    for iel =1:nel

        global(b_el)= zeros(Float64,m*ndofT)
        global(a_el)= zeros(Float64,m*ndofT, m*ndofT)
        Ka  = zeros(Float64,(m,m))
        Kd  = zeros(Float64,(m,m))
        MM  = zeros(Float64,(m,m))
        vel = zeros(Float64,(1,ndim))

        for k = 1:m
            Tvect[k]= T[icon[k,iel]]
        end

        for iq = -1:2:1
            for jq = -1:2:1

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1. *1.
                # calculate shape functions
                N_mat[1,1]=0.25 * (1. -rq) * (1. -sq)
                N_mat[2,1]=0.25 * (1. +rq) * (1. -sq)
                N_mat[3,1]=0.25 * (1. +rq) * (1. +sq)
                N_mat[4,1]=0.25 * (1. -rq) * (1. +sq)

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
                # calculate inverse of the jacobian matrix
                jcbi = inv(jcb)

                vel[1,1] = 0.
                vel[1,2] = 0.

                for k = 1:m
                    vel[1,1] += N_mat[k,1]*u[icon[k,iel]]
                    vel[1,2] += N_mat[k,1]*v[icon[k,iel]]
                    dNdx[k]=jcbi[1,1]*dNdr[k]+jcbi[1,2]*dNds[k]
                    dNdy[k]=jcbi[2,1]*dNdr[k]+jcbi[2,2]*dNds[k]
                    b_mat[1,k]=dNdx[k]
                    b_mat[2,k]=dNdy[k]
                end

                # compute mass matrix
                MM= N_mat * transpose(N_mat)*rho0*hcapa*weightq*jcob

                # compute diffusion matrix
                Kd=(transpose(b_mat) * b_mat) *hcond*weightq*jcob

                # compute advection matrix
                Ka=N_mat * (vel * b_mat) * rho0 *hcapa *weightq *jcob

                a_el+=MM+(Ka+Kd)*dt

                b_el+=MM * Tvect
            end
        end

        for k1 = 1:m
            m1=icon[k1,iel]
            if bc_fixT[m1]
                Aref=a_el[k1,k1]
                for k2 = 1:m
                    m2= icon[k2,iel]
                    b_el[k2] -=a_el[k2,k1]*bc_valT[m1]
                    a_el[k1,k2]=0
                    a_el[k2,k1]=0
                end
                a_el[k1,k1]=Aref
                b_el[k1]=Aref*bc_valT[m1]
            end
        end
        for k1 = 1:m
            m1=icon[k1,iel]
            for k2 = 1:m
                m2=icon[k2,iel]
                a_mat[m1,m2]+=a_el[k1,k2]
            end
            rhs[m1] += b_el[k1]
        end
    end



    end

    println("building temperature matrix and rhs          ", time1, "  sec")

    #################################################################
    # solve system
    #################################################################
    global(time1) = @elapsed begin
    global(Sps_amat)=sparse(a_mat)

    global(T) = Sps_amat \ rhs
    end
    println("Solving for temperature                     ", time1, "  sec")

#     #################################################################
#     # compute vrms
#     #################################################################
#
    global(time1) = @elapsed begin

    vrms=0.
    Tavrg=0.

    for iel =  1:nel
        for iq = -1:2:1
            for jq = -1:2:1

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq= 1. *1.
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

                uq=0.
                vq=0.
                Tq=0.
                for k = 1:m
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
                end
                vrms+=(uq^2+vq^2)*weightq*jcob
                Tavrg+=Tq*weightq*jcob
            end
        end
    end

    vrms=sqrt(vrms/(Lx*Ly))
    Tavrg /= (Lx*Ly)
    end
    printfmtln("time= {:.6f}  <vrms>= {:.6f}",timedt,vrms)
    println("compute vrms                           ", time1, "  sec")

    #################################################################
    # compute Nusselt number at top
    #################################################################

    Nusselt=0

    for iel =  1:nel

        global(qy)=0
        rq=0.
        sq=0.
        weightq=2. *2.

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
        # calculate inverse of the jacobian matrix
        jcbi = inv(jcb)

        # compute dNdx & dNdy
        for k = 1:m
            dNdy[k]=jcbi[2,1]*dNdr[k]+jcbi[2,2]*dNds[k]
        end

        for k = 1:m
            qy += -hcond*dNdy[k]*T[icon[k,iel]]
        end

        if y[icon[4,iel]]>Ly-epsilon
            Nusselt += qy*hx
        end
    end

    printfmtln("time= {:.6f}  Nusselt= {:.6f}",timedt,Nusselt)

    T_diff=sum(abs.(T-T_prev))/nnp
    u_diff=sum(abs.(u-u_prev))/nnp
    v_diff=sum(abs.(v-v_prev))/nnp

    printfmtln("time= {:.6f}  <T_diff>= {:.6f}",timedt,T_diff)
    printfmtln("time= {:.6f}  <u_diff>= {:.6f}",timedt,u_diff)
    printfmtln("time= {:.6f}  <v_diff>= {:.6f}",timedt,v_diff)

    println("T conv " , T_diff<tol*Tavrg)
    println("u conv " , u_diff<tol*vrms)
    println("v conv " , v_diff<tol*vrms)

    if T_diff<tol*Tavrg && u_diff<tol*vrms && v_diff<tol*vrms
        print("convergence reached")
        break
    end

    T_prev[:]=T[:]
    u_prev[:]=u[:]
    v_prev[:]=v[:]

    end # end loop timing
    println("total time of one timestep                ", timeloop, "  sec")
end
####################################################################
# end time stepping loop
####################################################################

#####################################################################
# retrieve pressure
#####################################################################

xc=zeros(Float64,nel)
yc=zeros(Float64,nel)
p=zeros(Float64,nel)
qx=zeros(Float64,nel)
qy=zeros(Float64,nel)
exx=zeros(Float64,nel)
eyy=zeros(Float64,nel)
exy=zeros(Float64,nel)
dens=zeros(Float64,nel)

for iel =  1:nel

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
    jcb = zeros(Float64, (2, 2))

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

    Tc=0.
    for k = 1:m
        Tc += N[k]*T[icon[k,iel]]
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        qx[iel] += -hcond*dNdx[k]*T[icon[k,iel]]
        qy[iel] += -hcond*dNdy[k]*T[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]] + 0.5 * dNdx[k]*v[icon[k,iel]]
    end

    p[iel] = -penalty*(exx[iel]+eyy[iel])
    dens[iel]=rho(rho0,alpha,Tc,T0)
end

println("p (m,M)", minimum(p)," , ", maximum(p))
println("exx (m,M)", minimum(exx)," , ", maximum(exx))
println("eyy (m,M)", minimum(eyy)," , ", maximum(eyy))
println("exy (m,M)", minimum(exy)," , ", maximum(exy))
println("dens (m,M)", minimum(dens)," , ", maximum(dens))
println("qx (m,M)", minimum(qx)," , ", maximum(qx))
println("qy (m,M)", minimum(qy)," , ", maximum(qy))

end
println("total time                ", totaltime, "  sec")


#####################################################################
# plot of solution
#####################################################################

filename = "Stone03.vtu"
open(filename,"w") do file
    write(file, "<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    write(file, "<UnstructuredGrid> \n")
    printfmt(file,"<Piece NumberOfPoints='  {:d} ' NumberOfCells='   {:d} '> \n",nnp,nel)
    write(file,"<Points> \n")
    write(file,"<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i = 1:nnp
        printfmt(file,"{:10f} {:10f} {:10f} \n",x[i],y[i],0.)
    end
    write(file,"</DataArray> \n")
    write(file,"</Points> \n")

    write(file,"<PointData Scalars='scalars'>\n")
    write(file,"<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i = 1:nnp
        printfmt(file,"{:10f} {:10f} {:10f} \n" ,u[i],v[i],0.)
    end
    write(file,"</DataArray>\n")

    write(file,"<DataArray type='Float32' Name='temperature' Format='ascii'> \n")
    for i = 1:nnp
        printfmt(file,"{:10f} \n" ,T[i])
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

    write(file,"<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for i = 1:nel
        printfmt(file,"{:10f} \n" ,dens[i])
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

    write(file,"<DataArray type='Float32' Name='qx' Format='ascii'> \n")
    for i = 1:nel
        printfmt(file,"{:10f} \n" ,qx[i])
    end
    write(file,"</DataArray> \n")

    write(file,"<DataArray type='Float32' Name='qy' Format='ascii'> \n")
    for i = 1:nel
        printfmt(file,"{:10f} \n" ,qy[i])
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

end
