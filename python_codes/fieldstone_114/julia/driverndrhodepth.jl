#!/usr/bin/env julia
using Printf
using Statistics
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using Formatting
#using PlotlyJS
using Polynomials
using PyPlot
using PyCall

mat = pyimport("matplotlib")
timing1=false
if timing1==true
    include("forwardtiming.jl")
else
    include("forward.jl")
end

cm=0.01
year=365.25*3600*24

###############################################################################

Rsphere = 50e3
deltarho=-300
truedepth=250e3

println("Give number of x cells for forward model ")
nelxforward=readline()
println("Give number of y cells for forward model ")
nelyforward=readline()

if length(nelxforward) ==0 || length(nelyforward) ==0
    println("standard 16x16")
    nelxforward=16
    nelyforward=16
else
    nelxforward=tryparse(Int32,nelxforward)
    nelyforward=parse(Int32,nelyforward)

    if nelxforward <= 2  || nelyforward <= 2
        println("standard 16x16")
        nelxforward=16
        nelyforward=16
    end
end

##############################################################################


println("Give number of x points for the depth")
ndepth=readline()
println("Give number of y points for the density difference")
ndrho=readline()
if length(ndrho) ==0 || length(ndepth) ==0
    println("standard 8x8")
    ndepth=8
    ndrho=8
else
    ndrho=parse(Int32,ndrho)
    ndepth=parse(Int32,ndepth)
    if ndrho <= 2  || ndepth <= 2
        println("standard 8x8")
        ndepth=8
        ndrho=8
    end
end

###############################################################################

eta0=1*10^(21.)
eta_star=1e3
rho0=3000

drhomin=-500
drhomax=-50

depthmin=100e3
depthmax=400e3

###############################################################################

m=4
nelx=ndepth-1
nely=ndrho-1
nel=nelx*nely
nnp=ndrho*ndepth

x=zeros(Float64,nnp)
y=zeros(Float64,nnp)

radius = zeros(Float64,nnp)
depth = zeros(Float64,nnp)
drho   = zeros(Float64,nnp)
icon = zeros(Int32,(m,nel))
misfit_g = zeros(Float64,nnp)
misfit_vrms = zeros(Float64,nnp)
misfit_vrmssurf = zeros(Float64,nnp)
misfit_absu = zeros(Float64,nnp)

y=LinRange(depthmin,depthmax,ndepth)
x=LinRange(drhomin,drhomax,ndrho)

counter = 0
for j = 1:nely
    for i = 1:nelx
        global(counter) += 1
        icon[1, counter]= (i)+ (j - 1) * (nelx +1)
        icon[2, counter] = i + 1 + (j - 1) * (nelx + 1)
        icon[3, counter] = i + 1 + (j) * (nelx + 1)
        icon[4, counter] = i + (j ) * (nelx + 1)
    end
end

###############################################################################
# call the forward model and record the misfits
###############################################################################

misfit_grav_min=1e30
misfit_vrms_min=1e30
misfit_vrmssurf_min=1e30
misfit_absu_min=1e30

global(time2) =  @elapsed begin
    time1=0
    counter1=1
    for j = 1:ndrho
        for i = 1:ndepth
            depth[counter1]=depthmin+(depthmax-depthmin)/(ndepth-1)*(i-1)
            drho[counter1]=drhomin+(drhomax-drhomin)/(ndrho-1)*(j-1)
            radius[counter1]=Rsphere

            global(time1) = @elapsed begin
            misfit_g[counter1],misfit_vrms[counter1],misfit_vrmssurf[counter1],misfit_absu[counter1]=
                    compute_misfits(rho0,drho[counter1],depth[counter1],eta0,eta_star,radius[counter1],deltarho,Rsphere,truedepth,nelxforward,nelyforward)

            end     #timing

            printfmtln("{:3.2f}% - xi_grav= '{:.4e}' -xi_vrms '{:.4e}' cm/yr -xi_vrms_surface '{:.4e}' -xi_absolute_v_surface '{:.4e}' time= '{:.4e}'s ",
                                counter1/ndepth/ndrho*100,misfit_g[counter1],misfit_vrms[counter1]/cm*year,misfit_vrmssurf[counter1]/cm*year,misfit_absu[counter1]/cm*year,time1)

            if counter1==2
                proj_time=time1*ndepth*ndrho
                printfmtln("projected time= '{:}' sec",proj_time)
            end
            if misfit_g[counter1]<misfit_grav_min
                global(misfit_grav_min)=misfit_g[counter1]
                global(min_grav_misfit)=[drho[counter1],depth[counter1]]
            end

            if misfit_vrms[counter1]<misfit_vrms_min
                global(misfit_vrms_min)=misfit_vrms[counter1]
                global(min_vrms_misfit)=[drho[counter1],depth[counter1]]
            end

            if misfit_vrmssurf[counter1]<misfit_vrmssurf_min
                global(misfit_vrmssurf_min)=misfit_vrmssurf[counter1]
                global(min_vrmssurf_misfit)=[drho[counter1],depth[counter1]]
            end
            if misfit_absu[counter1]<misfit_absu_min
                global(misfit_absu_min)=misfit_absu[counter1]
                global(min_absu_misfit)=[drho[counter1],depth[counter1]]
            end
            global(counter1) += 1
        end
    end
end

###############################################################################

printfmtln("completed {:} measurements",nnp)
println("total time for forward calculations ", time2 ,"   sec")
println("time for forward calculations per call", time2/nnp ,"   sec")

println("real radius = ",Rsphere)
println("real density difference = ",deltarho)
println("real depth =", truedepth)

println("location of lowest gravity misfit  ",min_grav_misfit)
println("location of lowest vrms misfit ",min_vrms_misfit)
println("location of lowest vrms at surface misfit ",min_vrmssurf_misfit)
println("location of lowest average v at surface misfit ",min_absu_misfit)

###############################################################################
misfit_g_line = zeros(Float64,0)
misfit_vrms_line = zeros(Float64,0)
drho_gravity = zeros(Float64,0)
depth_gravity = zeros(Float64,0)
drho_vrms = zeros(Float64,0)
depth_vrms = zeros(Float64,0)
truemin=zeros(Float64,nnp)
average_gravity=sum(misfit_g)/nnp

gravity_threshold=1.0*10^-3.25
vrms_threshold= 1.0*10^-10.5

for i = 1:nnp
    if misfit_g[i]<gravity_threshold
        append!(misfit_g_line,misfit_g[i])
        append!(drho_gravity,drho[i])
        append!(depth_gravity,depth[i])
        # println(misfit_g[i],radius[i],depth[i])

    end
    if misfit_vrms[i]<vrms_threshold
        append!(misfit_vrms_line,misfit_vrms[i])
        append!(drho_vrms,drho[i])
        append!(depth_vrms,depth[i])
        # println(misfit_vrms[i],radius[i],depth[i])

    end
    truemin[i]=log10(abs(truedepth-depth[i])*abs(deltarho-drho[i]))
    misfit_g[i]=log10(misfit_g[i])
    misfit_vrms[i]=log10(misfit_vrms[i]*cm/year)
    misfit_vrmssurf[i]=log10(misfit_vrmssurf[i]*cm/year)
    misfit_absu[i]=log10(misfit_absu[i]*cm/year)

end

truemin_temp=transpose(reshape(truemin,(ndepth,ndrho)))
misfit_g_temp=transpose(reshape(misfit_g,(ndepth,ndrho)))
misfit_vrms_temp=transpose(reshape(misfit_vrms,(ndepth,ndrho)))
misfit_vrmssurf_temp=transpose(reshape(misfit_vrmssurf,(ndepth,ndrho)))
misfit_absu_temp=transpose(reshape(misfit_absu,(ndepth,ndrho)))

###############################################################################

if (size(drho_gravity)<(4,))
    println("Not enough gravity data for a fit line")
elseif (size(drho_vrms)<(4,))
    println("Not enough vrms data for a fit line")
else
    line_gravity=fit(depth_gravity, drho_gravity, 4)
    line_vrms=fit(depth_vrms,drho_vrms, 4)
    xgravityvrms=roots(line_gravity-line_vrms)
    ygravityvrms=zeros(Float64,size(xgravityvrms)[1])
    println(xgravityvrms)


    for i =1:size(xgravityvrms)[1]
        if imag.(xgravityvrms[i]) == 0.0
            if real.(xgravityvrms[i])<depthmax && real.(xgravityvrms[i])>depthmin
                ygravityvrms[i]=line_vrms(real.(xgravityvrms[i]))
                if ygravityvrms[i]<drhomax && ygravityvrms[i]>drhomin
                    println("Point where the fit lines intersect  ", real.(xgravityvrms[i]),"  ", real.(ygravityvrms[i]))
                else
                    println("No overlap between the two fit functions within the expected range")
                end
            end
        end
    end
end
###############################################################################

uextent=(depthmin,depthmax,drhomin,drhomax)
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")
fig,axes = subplots(nrows=2,ncols=3,figsize=(35,14))

###############################################################################

ax = axes[1,1]
im = ax.imshow(truemin_temp,origin="lower",extent=uextent,aspect="auto",cmap="Spectral_r",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("depth")
ax.set_ylabel("δp")
ax.set_title("true minimum")
fig.colorbar(im,ax=axes[1,1])

###############################################################################

ax=axes[1,2]
if (size(drho_vrms)<(3,)) || (size(drho_gravity)<(4,))
else
    ax.plot(np.unique(depth), np.poly1d(np.polyfit(depth_gravity,drho_gravity,  4))(np.unique(depth)),label="best fit for gravity")
end

im = ax.imshow(misfit_g_temp,origin="lower",extent=uextent,aspect="auto",cmap="RdGy",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("depth")
ax.set_ylabel("δp")
ax.set_title("gravity misfit")
cbar = fig.colorbar(im,ax=axes[1,2])
cbar.set_label("gravity misfit")

###############################################################################

ax=axes[1,3]
ax.set_xlim(depthmin,depthmax)
ax.set_ylim(drhomin,drhomax)
ax.set_xlabel("depth")
ax.set_ylabel("δp")
ax.set_title("gravity & vrms ")
if (size(drho_vrms)<(4,)) || (size(drho_gravity)<(4,))
else
    for i =1:4
        if imag.(xgravityvrms[i]) == 0.0
            global(im) = ax.scatter(real.(xgravityvrms[i]),ygravityvrms[i])
            println(real.(xgravityvrms[i]),"  ",ygravityvrms[i])
        end
    end
    ax.plot(np.unique(depth), np.poly1d(np.polyfit(depth_gravity, drho_gravity,  4))(np.unique(depth)),label="best fit for gravity")
    ax.plot(np.unique(depth), np.poly1d(np.polyfit(depth_vrms, drho_vrms,  4))(np.unique(depth)),label="best fit for vrms")
end
im = ax.imshow((misfit_vrms_temp),origin="lower",extent=uextent,aspect="auto",cmap="Spectral_r",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
cbar = fig.colorbar(im,ax=axes[1,3])
cbar.set_label("vrms misfit (cm/year)")
ax.legend(loc="lower right")

###############################################################################

ax=axes[2 ,1]
im = ax.imshow((misfit_vrms_temp),origin="lower",extent=uextent,aspect="auto",cmap="Spectral_r",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("depth")
ax.set_ylabel("δp")
ax.set_title("vrms misfit")
cbar = fig.colorbar(im,ax=axes[2,1])
cbar.set_label("vrms misfit (cm/year)")

###############################################################################

ax=axes[2 ,2]
im = ax.imshow(misfit_vrmssurf_temp,origin="lower",extent=uextent,aspect="auto",cmap="Spectral_r",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("depth")
ax.set_ylabel("δp")
ax.set_title("vrms at the surface misfit")
cbar = fig.colorbar(im,ax=axes[2,2])
cbar.set_label("surface vrms misfit (cm/year)")

###############################################################################

ax=axes[2 ,3]
im = ax.imshow(misfit_absu_temp,origin="lower",extent=uextent,aspect="auto",cmap="Spectral_r",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("drho")
ax.set_ylabel("depth")
ax.set_title("average velocity at surface misfit")
cbar = fig.colorbar(im,ax=axes[2,3])
cbar.set_label("absolute u misfit (cm/year)")

###############################################################################

s1=string("ndrhodepth/Figures/","ndepth x ndrho ",ndepth,"x",ndrho,".png")
plt.savefig(s1)
