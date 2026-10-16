import numpy as np

def compute_principal_stresses(sxx,syy,szz,sxy,sxz,syz):
    
    s1=0
    v1x=0
    v1y=0
    v1z=0
    s2=0
    v2x=0
    v2y=0
    v2z=0
    s3=0
    v3x=0
    v3y=0
    v3z=0

    return s1,v1x,v1y,v1z, s2,v2x,v2y,v2z, s3,v3x,v3y,v3z
