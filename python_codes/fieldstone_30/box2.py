#this is in fact the Donea & Huerta velocity field

def Solution(x,y):

    u = 2*x**2*y*(2*y-1)*(y-1)*(x-1)**2 

    v = -2*x*y**2*(y-1)**2*(2*x-1)*(x-1)

    return u,v,0
