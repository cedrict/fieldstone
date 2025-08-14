import numpy as np

def NNV(rq,sq,tq):
    NV_00= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    NV_01= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    NV_02= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_03= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_04= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_05= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_06= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_07= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_08= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.)
    NV_09= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_10= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_11= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_12= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_13= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_14= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_15= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_16= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_17= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_18= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_19= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_20= (1.-rq**2)     * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_21= (1.-rq**2)     * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_22= 0.5*rq*(rq+1.) * (1.-sq**2)     * (1.-tq**2)
    NV_23= (1.-rq**2)     * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_24= 0.5*rq*(rq-1.) * (1.-sq**2)     * (1.-tq**2)
    NV_25= (1.-rq**2)     * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_26= (1.-rq**2)     * (1.-sq**2)     * (1.-tq**2)
    return np.array([NV_00,NV_01,NV_02,NV_03,NV_04,NV_05,NV_06,\
                     NV_07,NV_08,NV_09,NV_10,NV_11,NV_12,NV_13,\
                     NV_14,NV_15,NV_16,NV_17,NV_18,NV_19,NV_20,\
                     NV_21,NV_22,NV_23,NV_24,NV_25,NV_26],dtype=np.float64)


def dNNVdr(rq,sq,tq):
    dNVdr_00= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    dNVdr_01= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    dNVdr_02= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.) 
    dNVdr_03= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.) 
    dNVdr_04= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.) 
    dNVdr_05= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.) 
    dNVdr_06= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.) 
    dNVdr_07= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.) 
    dNVdr_08= (-2*rq)       * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    dNVdr_09= 0.5*(2*rq+1.) * (1.-sq**2)     * 0.5*tq*(tq-1.) 
    dNVdr_10= (-2*rq)       * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.) 
    dNVdr_11= 0.5*(2*rq-1.) * (1.-sq**2)     * 0.5*tq*(tq-1.) 
    dNVdr_12= (-2*rq)       * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.) 
    dNVdr_13= 0.5*(2*rq+1.) * (1.-sq**2)     * 0.5*tq*(tq+1.) 
    dNVdr_14= (-2*rq)       * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.) 
    dNVdr_15= 0.5*(2*rq-1.) * (1.-sq**2)     * 0.5*tq*(tq+1.) 
    dNVdr_16= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * (1.-tq**2) 
    dNVdr_17= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * (1.-tq**2) 
    dNVdr_18= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * (1.-tq**2) 
    dNVdr_19= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * (1.-tq**2) 
    dNVdr_20= (-2*rq)       * (1.-sq**2)     * 0.5*tq*(tq-1.) 
    dNVdr_21= (-2*rq)       * 0.5*sq*(sq-1.) * (1.-tq**2) 
    dNVdr_22= 0.5*(2*rq+1.) * (1.-sq**2)     * (1.-tq**2) 
    dNVdr_23= (-2*rq)       * 0.5*sq*(sq+1.) * (1.-tq**2) 
    dNVdr_24= 0.5*(2*rq-1.) * (1.-sq**2)     * (1.-tq**2) 
    dNVdr_25= (-2*rq)       * (1.-sq**2)     * 0.5*tq*(tq+1.) 
    dNVdr_26= (-2*rq)       * (1.-sq**2)     * (1.-tq**2) 
    return np.array([dNVdr_00,dNVdr_01,dNVdr_02,dNVdr_03,dNVdr_04,\
                     dNVdr_05,dNVdr_06,dNVdr_07,dNVdr_08,dNVdr_09,\
                     dNVdr_10,dNVdr_11,dNVdr_12,dNVdr_13,dNVdr_14,\
                     dNVdr_15,dNVdr_16,dNVdr_17,dNVdr_18,dNVdr_19,\
                     dNVdr_20,dNVdr_21,dNVdr_22,dNVdr_23,dNVdr_24,\
                     dNVdr_25,dNVdr_26],dtype=np.float64)

def dNNVds(rq,sq,tq):
    dNVds_00= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.) 
    dNVds_01= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.) 
    dNVds_02= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.) 
    dNVds_03= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.) 
    dNVds_04= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.) 
    dNVds_05= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.) 
    dNVds_06= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.) 
    dNVds_07= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.) 
    dNVds_08= (1.-rq**2)     * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.) 
    dNVds_09= 0.5*rq*(rq+1.) * (-2*sq)       * 0.5*tq*(tq-1.) 
    dNVds_10= (1.-rq**2)     * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.) 
    dNVds_11= 0.5*rq*(rq-1.) * (-2*sq)       * 0.5*tq*(tq-1.) 
    dNVds_12= (1.-rq**2)     * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.) 
    dNVds_13= 0.5*rq*(rq+1.) * (-2*sq)       * 0.5*tq*(tq+1.) 
    dNVds_14= (1.-rq**2)     * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.) 
    dNVds_15= 0.5*rq*(rq-1.) * (-2*sq)       * 0.5*tq*(tq+1.) 
    dNVds_16= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * (1.-tq**2) 
    dNVds_17= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * (1.-tq**2) 
    dNVds_18= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * (1.-tq**2) 
    dNVds_19= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * (1.-tq**2) 
    dNVds_20= (1.-rq**2)     * (-2*sq)       * 0.5*tq*(tq-1.) 
    dNVds_21= (1.-rq**2)     * 0.5*(2*sq-1.) * (1.-tq**2) 
    dNVds_22= 0.5*rq*(rq+1.) * (-2*sq)       * (1.-tq**2) 
    dNVds_23= (1.-rq**2)     * 0.5*(2*sq+1.) * (1.-tq**2) 
    dNVds_24= 0.5*rq*(rq-1.) * (-2*sq)       * (1.-tq**2) 
    dNVds_25= (1.-rq**2)     * (-2*sq)       * 0.5*tq*(tq+1.) 
    dNVds_26= (1.-rq**2)     * (-2*sq)       * (1.-tq**2) 
    return np.array([dNVds_00,dNVds_01,dNVds_02,dNVds_03,dNVds_04,\
                     dNVds_05,dNVds_06,dNVds_07,dNVds_08,dNVds_09,\
                     dNVds_10,dNVds_11,dNVds_12,dNVds_13,dNVds_14,\
                     dNVds_15,dNVds_16,dNVds_17,dNVds_18,dNVds_19,\
                     dNVds_20,dNVds_21,dNVds_22,dNVds_23,dNVds_24,\
                     dNVds_25,dNVds_26],dtype=np.float64)



def dNNVdt(rq,sq,tq):
    dNVdt_00= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.) 
    dNVdt_01= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.) 
    dNVdt_02= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.) 
    dNVdt_03= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.) 
    dNVdt_04= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.) 
    dNVdt_05= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.) 
    dNVdt_06= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.) 
    dNVdt_07= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.) 
    dNVdt_08= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.) 
    dNVdt_09= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*(2*tq-1.) 
    dNVdt_10= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.) 
    dNVdt_11= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*(2*tq-1.) 
    dNVdt_12= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.) 
    dNVdt_13= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*(2*tq+1.) 
    dNVdt_14= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.) 
    dNVdt_15= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*(2*tq+1.) 
    dNVdt_16= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * (-2*tq) 
    dNVdt_17= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * (-2*tq) 
    dNVdt_18= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * (-2*tq) 
    dNVdt_19= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * (-2*tq) 
    dNVdt_20= (1.-rq**2)     * (1.-sq**2)     * 0.5*(2*tq-1.) 
    dNVdt_21= (1.-rq**2)     * 0.5*sq*(sq-1.) * (-2*tq) 
    dNVdt_22= 0.5*rq*(rq+1.) * (1.-sq**2)     * (-2*tq) 
    dNVdt_23= (1.-rq**2)     * 0.5*sq*(sq+1.) * (-2*tq) 
    dNVdt_24= 0.5*rq*(rq-1.) * (1.-sq**2)     * (-2*tq) 
    dNVdt_25= (1.-rq**2)     * (1.-sq**2)     * 0.5*(2*tq+1.) 
    dNVdt_26= (1.-rq**2)     * (1.-sq**2)     * (-2*tq) 
    return np.array([dNVdt_00,dNVdt_01,dNVdt_02,dNVdt_03,dNVdt_04,\
                     dNVdt_05,dNVdt_06,dNVdt_07,dNVdt_08,dNVdt_09,\
                     dNVdt_10,dNVdt_11,dNVdt_12,dNVdt_13,dNVdt_14,\
                     dNVdt_15,dNVdt_16,dNVdt_17,dNVdt_18,dNVdt_19,\
                     dNVdt_20,dNVdt_21,dNVdt_22,dNVdt_23,dNVdt_24,\
                     dNVdt_25,dNVdt_26],dtype=np.float64)

def NNP(rq,sq,tq):
    NP_0=0.125*(1-rq)*(1-sq)*(1-tq)
    NP_1=0.125*(1+rq)*(1-sq)*(1-tq)
    NP_2=0.125*(1+rq)*(1+sq)*(1-tq)
    NP_3=0.125*(1-rq)*(1+sq)*(1-tq)
    NP_4=0.125*(1-rq)*(1-sq)*(1+tq)
    NP_5=0.125*(1+rq)*(1-sq)*(1+tq)
    NP_6=0.125*(1+rq)*(1+sq)*(1+tq)
    NP_7=0.125*(1-rq)*(1+sq)*(1+tq)
    return np.array([NP_0,NP_1,NP_2,NP_3,NP_4,NP_5,NP_6,NP_7],dtype=np.float64)





