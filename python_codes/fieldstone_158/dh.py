
def DHSolution(x,y):
    u=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    v=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    p=x*(1.-x)-1./6.
    return u,v,p
