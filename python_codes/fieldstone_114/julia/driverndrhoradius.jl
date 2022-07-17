#!/usr/bin/env julia
using Printf
using Statistics
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using Formatting
using PlotlyJS
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

println("Give number of x points for the radius")
nrad=readline()
println("Give number of y points for the density difference")
ndrho=readline()

if length(nrad) ==0 || length(ndrho) ==0
    println("standard 8x8")
    nrad=8
    ndrho=8
else
    nrad=parse(Int32,nrad)
    ndrho=parse(Int32,ndrho)
    if nrad <= 2  || ndrho <= 2
        println("standard 8x8")
        nrad=8
        ndrho=8
    end
end

###############################################################################

eta0=1*10^(20.5)
eta_star=1e3
rho0=3000

rmin=10e3
rmax=100e3

drhomin=-500
drhomax=-50

###############################################################################

m=4
nelx=nrad-1
nely=ndrho-1
nel=nelx*nely
nnp=nrad*ndrho

x=zeros(Float64,nnp)
y=zeros(Float64,nnp)

radius = zeros(Float64,nnp)
drho   = zeros(Float64,nnp)
depth = zeros(Float64,nnp)
icon =zeros(Int32,(m,nel))
misfit_g = zeros(Float64,nnp)
misfit_vrms = zeros(Float64,nnp)
misfit_vrmssurf = zeros(Float64,nnp)
misfit_absu = zeros(Float64,nnp)

y=LinRange(drhomin,drhomax,ndrho)
x=LinRange(rmin,rmax,nrad)

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
        for i = 1:nrad
            radius[counter1]=rmin+(rmax-rmin)/(nrad-1)*(i-1)
            drho[counter1]=drhomin+(drhomax-drhomin)/(ndrho-1)*(j-1)
            depth[counter1]=truedepth
            global(time1) = @elapsed begin

            misfit_g[counter1],misfit_vrms[counter1],misfit_vrmssurf[counter1],misfit_absu[counter1]=
                        compute_misfits(rho0,drho[counter1],depth[counter1],eta0,eta_star,radius[counter1],deltarho,Rsphere,truedepth,nelxforward,nelyforward)

            end     #timing

            printfmtln("{:3.2f}% - xi_grav= '{:.4e}' -xi_vrms '{:.4e}' cm/yr -xi_vrms_surface '{:.4e}' -xi_absolute_v_surface '{:.4e}' time= '{:.4e}'s ",
                                counter1/ndrho/nrad*100,misfit_g[counter1],misfit_vrms[counter1]/cm*year,misfit_vrmssurf[counter1]/cm*year,misfit_absu[counter1]/cm*year,time1)

            if counter1==2
                proj_time=time1*ndrho*nrad
                printfmtln("projected time= '{:}' sec",proj_time)
            end
            if misfit_g[counter1]<misfit_grav_min
                global(misfit_grav_min)=misfit_g[counter1]
                global(min_grav_misfit)=[radius[counter1],drho[counter1]]
            end

            if misfit_vrms[counter1]<misfit_vrms_min
                global(misfit_vrms_min)=misfit_vrms[counter1]
                global(min_vrms_misfit)=[radius[counter1],drho[counter1]]
            end

            if misfit_vrmssurf[counter1]<misfit_vrmssurf_min
                global(misfit_vrmssurf_min)=misfit_vrmssurf[counter1]
                global(min_vrmssurf_misfit)=[radius[counter1],drho[counter1]]
            end
            if misfit_absu[counter1]<misfit_absu_min
                global(misfit_absu_min)=misfit_absu[counter1]
                global(min_absu_misfit)=[radius[counter1],drho[counter1]]
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
radius_gravity = zeros(Float64,0)
drho_gravity = zeros(Float64,0)
radius_vrms = zeros(Float64,0)
drho_vrms = zeros(Float64,0)
truemin=zeros(Float64,nnp)
average_gravity=sum(misfit_g)/nnp

gravity_threshold=1.0*10^-3.5
vrms_threshold= 1.0*10^-10.

for i = 1:nnp
    if misfit_g[i]<gravity_threshold
        append!(misfit_g_line,misfit_g[i])
        append!(radius_gravity,radius[i])
        append!(drho_gravity,drho[i])
        # println(misfit_g[i],radius[i],drho[i])

    end
    if misfit_vrms[i]<vrms_threshold
        append!(misfit_vrms_line,misfit_vrms[i])
        append!(radius_vrms,radius[i])
        append!(drho_vrms,drho[i])
        # println(misfit_vrms[i],radius[i],drho[i])

    end
    truemin[i]=log10(abs(deltarho-drho[i])*abs(Rsphere-radius[i]))
    misfit_g[i]=log10(misfit_g[i])
    misfit_vrms[i]=log10(misfit_vrms[i]*cm/year)
    misfit_vrmssurf[i]=log10(misfit_vrmssurf[i]*cm/year)
    misfit_absu[i]=log10(misfit_absu[i]*cm/year)

end
truemin_temp=transpose(reshape(truemin,(nrad,ndrho)))
misfit_g_temp=transpose(reshape(misfit_g,(nrad,ndrho)))
misfit_vrms_temp=transpose(reshape(misfit_vrms,(nrad,ndrho)))
misfit_vrmssurf_temp=transpose(reshape(misfit_vrmssurf,(nrad,ndrho)))
misfit_absu_temp=transpose(reshape(misfit_absu,(nrad,ndrho)))


###############################################################################

if (size(radius_gravity) <(4,))
    println("Not enough gravity data for a fit line")
elseif (size(radius_vrms)<(4,))
    println("Not enough vrms data for a fit line")
else

    line_gravity=fit(radius_gravity, drho_gravity, 4)
    line_vrms=fit(radius_vrms, drho_vrms, 4)
    xgravityvrms=roots(line_gravity-line_vrms)
    ygravityvrms=zeros(Float64,size(xgravityvrms)[1])

    for i =1:size(xgravityvrms)[1]
        if imag.(xgravityvrms[i]) == 0.0
            if real.(xgravityvrms[i])<rmax && real.(xgravityvrms[i])>rmin
                ygravityvrms[i]=line_vrms(real.(xgravityvrms[i]))
                if ygravityvrms[i]<drhomax && ygravityvrms[i]>drhomin
                    println("Point where the fit lines intersect  ", real.(xgravityvrms[i]),"  ", ygravityvrms[i])
                else
                    println("No overlap between the two fit functions within the expected range")
                end
            end
        end
    end
end
###############################################################################

uextent=(rmin,rmax,drhomin,drhomax)
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")
fig,axes = subplots(nrows=2,ncols=3,figsize=(35,14))

###############################################################################

ax = axes[1,1]
im = ax.imshow(truemin_temp,origin="lower",extent=uextent,aspect="auto",cmap="Spectral_r",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("radius")
ax.set_ylabel("δp")
ax.set_title("true minimum")
fig.colorbar(im,ax=axes[1,1])

###############################################################################

ax=axes[1,2]
if (size(radius_vrms)<(4,)) || (size(radius_gravity)<(3,))
else
    ax.plot(np.unique(radius), np.poly1d(np.polyfit(radius_gravity, drho_gravity, 4))(np.unique(radius)),label="best fit for gravity")
end
im = ax.imshow(misfit_g_temp,origin="lower",extent=uextent,aspect="auto",cmap="RdGy",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("radius")
ax.set_ylabel("δp")
ax.set_title("gravity misfit")
cbar = fig.colorbar(im,ax=axes[1,2])
cbar.set_label("gravity misfit")

###############################################################################

ax=axes[1,3]
ax.set_xlim(rmin,rmax)
ax.set_ylim(drhomin,drhomax)
ax.set_xlabel("radius")
ax.set_ylabel("δp")
ax.set_title("gravity & vrms ")
if (size(radius_vrms)<(4,)) || (size(radius_gravity)<(4,))
else
    for i =1:4
        if imag.(xgravityvrms[i]) == 0.0
            global(im) = ax.scatter(real.(xgravityvrms[i]),ygravityvrms[i])
        end
    end
    ax.plot(np.unique(radius), np.poly1d(np.polyfit(radius_gravity, drho_gravity, 4))(np.unique(radius)),label="best fit for gravity")
    ax.plot(np.unique(radius), np.poly1d(np.polyfit(radius_vrms, drho_vrms, 4))(np.unique(radius)),label="best fit for vrms")
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
ax.set_xlabel("radius")
ax.set_ylabel("δp")
ax.set_title("vrms misfit")
cbar = fig.colorbar(im,ax=axes[2,1])
cbar.set_label("vrms misfit (cm/year)")

###############################################################################

ax=axes[2 ,2]
im = ax.imshow(misfit_vrmssurf_temp,origin="lower",extent=uextent,aspect="auto",cmap="Spectral_r",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("radius")
ax.set_ylabel("δp")
ax.set_title("vrms at the surface misfit")
cbar = fig.colorbar(im,ax=axes[2,2])
cbar.set_label("surface vrms misfit (cm/year)")

###############################################################################

ax=axes[2 ,3]
im = ax.imshow(misfit_absu_temp,origin="lower",extent=uextent,aspect="auto",cmap="Spectral_r",interpolation="nearest")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax.set_xlabel("radius")
ax.set_ylabel("δp")
ax.set_title("average velocity at surface misfit")
cbar = fig.colorbar(im,ax=axes[2,3])
cbar.set_label("absolute u misfit (cm/year)")

###############################################################################

s1=string("ndrhoradius/Figures/","visco20",ndrho,"x",nrad,".png")
plt.savefig(s1)
print(s1)
