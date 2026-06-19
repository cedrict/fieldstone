import numpy as np
from scipy import special

npts = 1000  # number of points

Rsurf = 6370e3
Rcmb = Rsurf - 1000e3
rho0 = 3300
g = 9.8
eta0 = 1e21
R = 8.314
alpha = 3e-5
kappa = 1e-6
hcapa = 1250
hcond = kappa * rho0 * hcapa
Tsurf = 273
DeltaT = 1350
year = 365.25 * 24 * 3600
Estar = np.log(1000)

print(Estar)
print(hcond)

# Tm = 0.9 * DeltaT + 273
Tm = DeltaT + 273

dr = (Rsurf - Rcmb) / (npts - 1)  # spacing between points

###############################################################################
# function borrowed from MEEUUW


def initial_temperature_hsc(r, Rcmb, Rsurf, Tcmb, Tsurf, age_cmb, age_surf, Tm, kappa):
    val = (
        Tsurf
        + (Tm - Tsurf) * special.erf((Rsurf - r) / 2 / np.sqrt(age_surf * kappa))
        - (Tcmb - Tm) * (-1 + special.erf((r - Rcmb) / 2 / np.sqrt(age_cmb * kappa)))
    )
    return val


###############################################################################
# compute coordinates: equidistant points between Rcmb and Rsurf
###############################################################################

r = np.zeros(npts, dtype=np.float64)

for i in range(0, npts):
    r[i] = Rcmb + dr * i

# np.savetxt('r.ascii',np.array([r]).T)

###############################################################################
###############################################################################
###############################################################################

for age in (50e6, 80e6, 100e6, 200e6):
    print("age=", age)

    age_cmb = age * year
    age_surf = age * year

    Tcmb = Tsurf + DeltaT

    ###############################################################################
    # assign initial temperature
    ###############################################################################

    T = np.zeros(npts, dtype=np.float64)

    for i in range(0, npts):
        T[i] = initial_temperature_hsc(
            r[i], Rcmb, Rsurf, Tcmb, Tsurf, age_cmb, age_surf, Tm, kappa
        )

    np.savetxt("T_" + str(int(age / 1e6)) + ".ascii", np.array([r, T]).T)

    ###############################################################################
    # compute viscosity
    ###############################################################################

    eta = np.zeros(npts, dtype=np.float64)

    for i in range(0, npts):
        Tstar = (T[i] - 273) / DeltaT
        eta[i] = eta0 * np.exp(Estar * (1 - Tstar))

    np.savetxt("eta_" + str(int(age / 1e6)) + ".ascii", np.array([r, np.log10(eta)]).T)

    ###############################################################################
    # compute new viscosity
    ###############################################################################

    for Q in (100e3,200e3,300e3,400e3):

        eta = np.zeros(npts, dtype=np.float64)

        A=1/(2*eta0)*np.exp(Q/R/Tm)

        for i in range(0, npts):
            eta[i] = 0.5/A * np.exp(Q/R/T[i])

        np.savetxt("eta2_" + str(int(age / 1e6))+'_' + str(int(Q/1000))+ ".ascii", np.array([r, np.log10(eta)]).T)

# end for
