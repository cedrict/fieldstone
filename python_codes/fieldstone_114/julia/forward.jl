#!/usr/bin/env julia
using Printf
using Statistics
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using Formatting
using DelimitedFiles

function compute_misfits(rho0,drho,depth,eta0,eta_star,radius,deltarho,Rsphere,truedepth,nelx,nely)
    writetofile=false
    radius2=radius^2
    m=4
    ndofV=2      # number of degrees of freedom per node

    Lx=500e3
    Ly=500e3

    nnx=nelx+1  # number of elements, x direction
    nny=nely+1  # number of elements, y direction

    nnp=nnx*nny  # number of nodes

    nel=nelx*nely  # number of elements, total

    penalty=1e25

    Nfem=nnp*ndofV  # Total number of degrees of velocity freedom

    eps=1.e-10

    gx=0.  # gravity vector, x component
    gy=-9.81  # gravity vector, y component

    sqrt3=sqrt(3.)

    hx=Lx/nelx
    hy=Ly/nely

    Ggrav=6.67e-11

    vrms_ref=8.988643046707479e-10
    vrmssurf_ref=7.41160061087621e-10
    absusurf_ref=6.600967198590761e-10

    solves_stokes=true
    #################################################################
    # grid point setup
    #################################################################
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

    ##################################################################
    ## build connectivity array
    ##################################################################

    icon = zeros(Int32,m,nel)
    global(counter) = 0
    for j = 1:nely
        for i = 1:nelx
            global(counter) += 1
            icon[1, counter]= (i)+ (j-1) * (nelx +1)
            icon[2, counter] = i + 1 + (j-1) * (nelx + 1)
            icon[3, counter] = i + 1 + (j) * (nelx + 1)
            icon[4, counter] = i + (j ) * (nelx + 1)

        end
    end
    #################################################################
    # element center coordinates
    #################################################################

    xc = zeros(Float64,nel)
    yc = zeros(Float64,nel)
    elt_in_sphere=zeros(Bool,Nfem)

    for iel = 1:nel
        xc[iel]=0.5*(x[icon[1,iel]]+x[icon[3,iel]])
        yc[iel]=0.5*(y[icon[1,iel]]+y[icon[3,iel]])
        if (xc[iel])^2+(yc[iel]-depth)^2<radius2
          elt_in_sphere[iel]=true
        end
    end
     #####################################################################
     # define velocity boundary conditions
     #####################################################################
    if solves_stokes

         bc_fix = zeros(Bool,Nfem)
         bc_val = zeros(Float64,Nfem)
         for i = 1:nnp
                if x[i] < eps
                    bc_fix[(i-1)*ndofV+1]     = true ; bc_val[(i-1)*ndofV+1]    = 0.
                end
                if x[i] > (Lx-eps)
                    bc_fix[(i-1)*ndofV+1]     = true ; bc_val[(i-1)*ndofV+1]    = 0.
                end
                if y[i] < eps
                    bc_fix[(i-1)*ndofV+2]     = true ; bc_val[(i-1)*ndofV+2]    = 0.
                end
                if y[i] > (Ly-eps)
                    bc_fix[(i-1)*ndofV+2]     = true ; bc_val[(i-1)*ndofV+2]    = 0.
                end
         end
         #####################################################################
         # create necessary arrays
         #####################################################################
         a_mat = spzeros(Nfem , Nfem)
         b_mat = zeros(Float64,(3, ndofV*m))
         rhs   = zeros(Float64, Nfem)
         N     = zeros(Float64, m)    # shape functions
         dNdx  = zeros(Float64, m)    # shape functions derivatives
         dNdy  = zeros(Float64, m)    # shape functions derivatives
         dNdr  = zeros(Float64, m)    # shape functions derivatives
         dNds  = zeros(Float64, m)    # shape functions derivatives
         u     = zeros(Float64, nnp)  # x-component velocity
         v     = zeros(Float64, nnp)  # y-component velocity
         Tvect = zeros(Float64, 4)
         k_mat = Float64[1 1 0 ; 1 1 0 ; 0 0 0]
         c_mat = Float64[2 0 0 ; 0 2 0 ; 0 0 1]
         for iel = 1:nel

             b_el= zeros(Float64,m*ndofV)
             a_el= zeros(Float64,m*ndofV, m*ndofV)

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

                     # compute dNdx & dNdy
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
                     if elt_in_sphere[iel]
                         etaq=eta0*eta_star
                         rhoq=rho0+drho
                      else
                         rhoq=rho0
                         etaq=eta0
                     end

                     a_el +=transpose(b_mat) *  (c_mat * b_mat) * etaq*weightq*jcob

                     for i=1:m
                         b_el[2*i] += N[i]*jcob*weightq*rhoq*gy
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

         end

         ####################################################################
         #Solve system
         ####################################################################

         Sps_amat=sparse(a_mat)
         sol= Sps_amat\rhs
         sol1 =(reshape(sol,(2,nnp)))

         #####################################################################
         # put solution into separate x,y velocity arrays
         #####################################################################

         u= sol1[1,:]
         v= sol1[2,:]

         # if writetofile
         #     open("Data/velocity.ascii", "w") do io
         #         header=["x" "y" "u" "v"]
         #         writedlm(io,[header ; x y u v])
         #     end
         # end
         ####################################################################
         #compute vrms
         ####################################################################
         (vrms)=0.
         (vrmssurf)=0.
         (absu)=0.

         for iel = 1:nel
             for iq = -1:2:1
                 if (y[icon[3,iel]]>(Ly-eps) && y[icon[4,iel]]>(Ly-eps))
                     rq=iq/sqrt3
                     weightq=1.

                     N[3]=0.5*(1+rq)
                     N[4]=0.5*(1-rq)

                     dNdr[3]=0.5
                     dNdr[4]=-0.5

                     jcb1D=0
                     jcb1D += dNdr[3]*x[icon[3,iel]]
                     jcb1D += dNdr[4]*x[icon[4,iel]]

                     uq=0.
                     uq+=N[3]*u[icon[3,iel]]
                     uq+=N[4]*u[icon[4,iel]]

                     (vrmssurf) +=(uq^2)*jcb1D*weightq
                     (absu) += (uq)*jcb1D*weightq
                 end
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
                        end
                        (vrms)+=(uq^2+vq^2)*weightq*jcob
                    end
            end
         end
         vrms=sqrt(vrms/(Lx*Ly))
         vrmssurf=sqrt(vrmssurf/Lx)
         absu=absu/Lx
         misfit_vrms=abs(vrms-vrms_ref)
         misfit_vrmssurf=abs(vrmssurf-vrmssurf_ref)
         misfit_absu=abs(absu-absusurf_ref)

    end

    #####################################################################
    # compute gravity
    # because half the disc is missing we on the fly mirror it
    #####################################################################

    N=zeros(Float64,m)
    volumesphere=pi*Rsphere^2
    gravy=zeros(Float64,nnx)
    gravy_th=zeros(Float64,nnx)

    for iel=1:nel
        if elt_in_sphere[iel]
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

                     xq=0.0
                     yq=0.0
                     for k = 1:m
                         xq+=N[k]*x[icon[k,iel]]
                         yq+=N[k]*y[icon[k,iel]]
                     end

                     for i = 1:nnx
                         dist2=(x[i]-xq)^2+(Ly-yq)^2
                         gravy[i] += Ggrav*drho*weightq*jcob/dist2*(Ly-yq)
                         dist2=(x[i]+xq)^2+(Ly-yq)^2
                         gravy[i] += Ggrav*drho*weightq*jcob/dist2*(Ly-yq)
                     end
                 end
             end
        end
    end

    for i =1:nnx
         gravy_th[i]=Ggrav*volumesphere*deltarho/(x[i]^2+(Ly-truedepth)^2)*(Ly-truedepth)
    end
    misfit_grav=LinearAlgebra.norm(gravy[:]-gravy_th[:],2)

    # if writetofile
    #     open("Data/gravity.ascii", "w") do io
    #         writedlm(io,[x[1:nnx] gravy gravy_th])
    #     end
    # end

    #####################################################################
    # export to vtu
    #####################################################################

    filename = "solution1000.vtu"
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

        write(file,"<CellData Scalars='scalars'>\n")
        write(file,"<DataArray type='Float32' Name='sphere' Format='ascii'> \n")
        for iel = 1:nel
            if elt_in_sphere[iel]
                printfmt(file,"{:} \n" ,1.)
            else
                printfmt(file,"{:} \n" ,0.)
            end
        end
        write(file,"</DataArray>\n")
        write(file,"</CellData>\n")

        # write(file,"<PointData Scalars='scalars'>\n")
        # write(file,"<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
        # for i = 1:nnp
        #     printfmt(file,"{:} {:} {:} \n" ,u[i],v[i],0.)
        # end
        # write(file,"</DataArray>\n")
        # write(file,"</PointData>\n")

        #######
        write(file,"<Cells>\n")
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
     return misfit_grav,misfit_vrms,misfit_vrmssurf,misfit_absu
 end
