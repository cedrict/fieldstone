import numpy as np

def NNN(r,s,t):
    N0=0.125*(1.-r)*(1.-s)*(1.-t)
    N1=0.125*(1.+r)*(1.-s)*(1.-t)
    N2=0.125*(1.+r)*(1.+s)*(1.-t)
    N3=0.125*(1.-r)*(1.+s)*(1.-t)
    N4=0.125*(1.-r)*(1.-s)*(1.+t)
    N5=0.125*(1.+r)*(1.-s)*(1.+t)
    N6=0.125*(1.+r)*(1.+s)*(1.+t)
    N7=0.125*(1.-r)*(1.+s)*(1.+t)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

def dNNNdr(r,s,t):
    dNdr0=-0.125*(1.-s)*(1.-t)
    dNdr1=+0.125*(1.-s)*(1.-t)
    dNdr2=+0.125*(1.+s)*(1.-t)
    dNdr3=-0.125*(1.+s)*(1.-t)
    dNdr4=-0.125*(1.-s)*(1.+t)
    dNdr5=+0.125*(1.-s)*(1.+t)
    dNdr6=+0.125*(1.+s)*(1.+t)
    dNdr7=-0.125*(1.+s)*(1.+t)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)

def dNNNds(r,s,t):
    dNds0=-0.125*(1.-r)*(1.-t)
    dNds1=-0.125*(1.+r)*(1.-t)
    dNds2=+0.125*(1.+r)*(1.-t)
    dNds3=+0.125*(1.-r)*(1.-t)
    dNds4=-0.125*(1.-r)*(1.+t)
    dNds5=-0.125*(1.+r)*(1.+t)
    dNds6=+0.125*(1.+r)*(1.+t)
    dNds7=+0.125*(1.-r)*(1.+t)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)

def dNNNdt(r,s,t):
    dNdt0=-0.125*(1.-r)*(1.-s)
    dNdt1=-0.125*(1.+r)*(1.-s)
    dNdt2=-0.125*(1.+r)*(1.+s)
    dNdt3=-0.125*(1.-r)*(1.+s)
    dNdt4=+0.125*(1.-r)*(1.-s)
    dNdt5=+0.125*(1.+r)*(1.-s)
    dNdt6=+0.125*(1.+r)*(1.+s)
    dNdt7=+0.125*(1.-r)*(1.+s)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7],dtype=np.float64)







