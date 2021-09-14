
def Solution(x,y):

    xx=(x-0.5)*2
    yy=(y-0.5)*2

    u = (2-xx**2-xx**4)*(-yy-2*yy**3)

    v = (2-yy**2-yy**4)*(xx+2*xx**3)

    return u,v,0
