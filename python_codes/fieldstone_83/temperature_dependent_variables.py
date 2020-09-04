import numpy as np

# ========== temperature dependent variables ==========

#thermal_parameters = 2 # 1 = constant according to Parsons & Slater 1977; 2 = T-dependent with k from (simplified) Hofmeister 1999 as posed by McKenzie et al., 2005; 3 = T-dependent with k from Xu et al., 2004 

#option_k:  # 0 = constant according to Parsons & Slater 1977; 1 = T-dependent with k from (simplified) Hofmeister 1999 as posed by McKenzie et al., 2005; 2 = T-dependent with k from Xu et al., 2004 

#option_C_p:  #  0 = constant according to Parsons & Slater 1977; 1 = 100% forsterite Berman 1988; 2 = 100% fayalite Berman 1988; 3 = 89% forsterite & 11% fayalite Berman 1988; 4 = 100% forsterite Berman & Aranovich 1996; 5 = 100% fayalite Berman & Aranovich 1996; 6 = 89% forsterite & 11% fayalite Berman & Aranovich 1996;

#option_rho:  # 0 = constant according to Parsons & Slater 1977; 1 = T-dependent based on McKenzie et al., 2005


kelvin=273

def heat_conductivity(T,p,imat,option_k):
    if option_k == 0:
        # constant variables 
        k = 3.138 
    elif option_k == 1:
        # (simplified) conductivity of Hofmeister 1999 as posed by McKenzie et al., 2005 
        # constants 
        b = 5.3
        c = 0.0015
        #d = np.array([1.753e-2,-1.0365e-4,2.2451e-7,-3.4071e-11])
        d = [1.753e-2,-1.0365e-4,2.2451e-7,-3.4071e-11]

        # heat transport component
        k_h_heat_transport = b / (1 + c * (T-kelvin))
        
        # radiative component of the conductivity
        k_h_radiative = 0.
        for i in range(0,4):
            k_h_radiative = k_h_radiative + d[i] * (T)**i
            
        # calculate k 
        k = k_h_heat_transport + k_h_radiative
        
    elif option_k == 2: 
        # Xu et al., 2004
        # constants 
        k298 = 4.08 
        n    = 0.406 
        # calculate k 
        k = k298 * (298 / (T + kelvin))**n
    return k  

def heat_capacity(T,p,imat,option_C_p):
    if option_C_p == 0: 
        # constant vriables 
        C_p = 1171.52 
    elif option_C_p > 0 and option_C_p < 7:
        
        molecular_mass_fo = 140.691
        molecular_mass_fa = 203.771

        if option_C_p > 0 and option_C_p < 4:
            # Berman 1988 
            # forsterite 
            k_0_fo = 238.64    
            k_1_fo = -20.013e2 
            k_3_fo = -11.624e7 
        
            # fayalite 
            k_0_fa = 248.93    
            k_1_fa = -19.239e2 
            k_3_fa = -13.910e7 
        elif option_C_p > 3 and option_C_p < 7:
            # Berman & Aranovich 1996 
            # forsterite 
            k_0_fo = 233.18
            k_1_fo = -18.016e2
            k_3_fo = -26.794e7
        
            # fayalite 
            k_0_fa = 252.
            k_1_fa = -20.137e2
            k_3_fa = -6.219e7
    
        if option_C_p == 1 or option_C_p == 4: 
            # forsterite 
            molar_fraction_fa = 0.
        if option_C_p == 2 or option_C_p == 5: 
            # fayalite 
            molar_fraction_fa = 1.
        if option_C_p == 3 or option_C_p == 6: 
            # molar fraction of fayalite is 0.11 
            molar_fraction_fa = 0.11 
            
        # calculate C_p 
        C_p_fo = (k_0_fo + k_1_fo * np.power(T,-0.5) + k_3_fo * np.power(T,-3.)) * (1000./molecular_mass_fo)
        C_p_fa = (k_0_fa + k_1_fa * np.power(T,-0.5) + k_3_fa * np.power(T,-3.)) * (1000./molecular_mass_fa)
     
        C_p       = (1 - molar_fraction_fa) * C_p_fo + molar_fraction_fa * C_p_fa
        
    return C_p

def density(T,p,imat,option_rho):
    if option_rho == 0:
        # constant variables 
        rho = 3330. 
    elif option_rho == 1:
        # constants 
        rho_0   = 3330
        alpha_0 = 2.832e-5
        alpha_1 = 3.79e-8 
        # calculate rho
        rho     = rho_0 * np.exp( - ( alpha_0 * (T - kelvin) + (alpha_1/2.) * ( np.power(T,2) - kelvin**2 ) ) )
    return rho
