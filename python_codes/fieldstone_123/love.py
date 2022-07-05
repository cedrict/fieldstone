import numpy as np

pi=3.14159265358979323846264338327950288

#**************************************************
# Names corresponding to Becker and Bevis (GJI,2004)
#**************************************************       
# original fortran functions written by Lukas van der Wiel
#**************************************************       

def rj0(j,x,y,yy,z,a):
    if j==1 :
        val = np.sqrt( (a-x)*(a-x) + (yy-y)*(yy-y) + (z*z) )
    elif j==2:
        val = np.sqrt( (a+x)*(a+x) + (yy-y)*(yy-y) + (z*z) )
    return val

#**************************************************       

def r0j(j,x,xx,y,z,b):
    if j==1:
        val = np.sqrt( (xx-x)*(xx-x) + (b-y)*(b-y) + (z*z) )
    elif j==2:
        val = np.sqrt( (xx-x)*(xx-x) + (b+y)*(b+y) + (z*z) )
    return val

#**************************************************       

def betaj0(j,x,z,a):
    if j==1:
        val = np.sqrt( (a-x)*(a-x) + (z*z) )
    elif j==2:
        val = np.sqrt( (a+x)*(a+x) + (z*z) )
    return val


#**************************************************       

def beta0j(j,y,z,b):
    if j==1:
        val = np.sqrt( (b-y)*(b-y) + (z*z) )
    elif j==2:
        val = np.sqrt( (b+y)*(b+y) + (z*z) )
    return val

#**************************************************       

def PSIj0(j,x,y,yy,z,a):
    if j==1:
        val = (yy-y) / (rj0(j,x,y,yy,z,a) + betaj0(j,x,z,a))
    elif j==2:
        val = (yy-y) / (rj0(j,x,y,yy,z,a) + betaj0(j,x,z,a))
    return val


#**************************************************       

def PSIj0(j,x,y,yy,z,a):
    if j==1:
        val = (yy-y) / (rj0(j,x,y,yy,z,a) + betaj0(j,x,z,a))
    elif j==2:
        val = (yy-y) / (rj0(j,x,y,yy,z,a) + betaj0(j,x,z,a))
    return val

#**************************************************       

def PSI0j(j,x,xx,y,z,b):

    if j==1:
        val = (xx-x) / (r0j(j,x,xx,y,z,b) + beta0j(j,y,z,b))
    elif j==2:
        val = (xx-x) / (r0j(j,x,xx,y,z,b) + beta0j(j,y,z,b))
    return val

#**************************************************       

def Jj (j,x,y,yy,z,a):

    term1 = (yy-y) * ( np.log(z+rj0(j,x,y,yy,z,a)) -1.0)

    if z==0:
        term2 = 0
    else:
        term2 = z * np.log( (1.0 + PSIj0(j,x,y,yy,z,a)) / (1.0 - PSIj0(j,x,y,yy,z,a))  )

    if j==1:
        ax = a - x 
    elif j==2:
        ax = a + x 
    ax = abs(ax)

    term3 = 2 * ax * np.arctan(  ax * PSIj0(j,x,y,yy,z,a) / (z + betaj0(j,x,z,a))  )

    val = term1 + term2 + term3

    return val

#**************************************************       

def Kj (j,x,xx,y,z,b):

    term1 = (xx-x) * (  np.log(z+r0j(j,x,xx,y,z,b)) -1.0)

    term2 = z * np.log( (1.0 + PSI0j(j,x,xx,y,z,b)) / (1.0 - PSI0j(j,x,xx,y,z,b)) )

    if j==1:
        by = b - y
    elif j==2:
        by = b + y

    by = abs(by)

    term3 = 2 * by * np.arctan((by * PSI0j(j,x,xx,y,z,b)) / (z + beta0j(j,y,z,b))  )

    val = term1 + term2 + term3

    return val

#**************************************************       

def Lj (j,x,y,yy,z,a):

    if j==1:
        ax = a - x
    elif j==2:
        ax = - a - x

    term1 = (yy-y) * (np.log(ax + rj0(j,x,y,yy,z,a)) -1.0)

    term2 = ax * np.log((1.0 + PSIj0(j,x,y,yy,z,a))/ (1.0 - PSIj0(j,x,y,yy,z,a)))

    #! when z = 0, the denominator of the atan fraction is 0.
    #! so that the fraction is a 0/0 limit.
    #! Emprically defined that term3 goes to 0, for z->0

    if z==0:
        term3 = 0.
    else:
        term3 = 2 * z * np.arctan ((z * PSIj0(j,x,y,yy,z,a))/ ( ax+betaj0(j,x,z,a)))

    val = term1 + term2 + term3

    return val

#**************************************************       
# displacement in x direction
#**************************************************       

def u_Love(x,y,z,a,b,pressure,lambdaa,mu):
    upper = uterm(x,y, b,z,a,pressure,lambdaa,mu)
    lower = uterm(x,y,-b,z,a,pressure,lambdaa,mu)
    val = upper - lower
    return val

def uterm(x,y,yy,z,a,pressure,lambdaa,mu):
    if z==0:
        val= -pressure/(4*pi) * ((1/(lambdaa + mu)) * (Jj(2,x,y,yy,z,a) - Jj(1,x,y,yy,z,a)) )
    else:
        val= -pressure/(4*pi) * ((1/(lambdaa + mu)) * (Jj(2,x,y,yy,z,a) - Jj(1,x,y,yy,z,a)) + \
              (z/mu) * np.log(  (yy-y+rj0(2,x,y,yy,z,a)) / (yy-y+rj0(1,x,y,yy,z,a))   )   )
    return val

#**************************************************       
# displacement in y direction
#**************************************************       

def v_Love(x,y,z,a,b,pressure,lambdaa,mu):
    upper = vterm(x, a,y,z,b,pressure,lambdaa,mu)
    lower = vterm(x,-a,y,z,b,pressure,lambdaa,mu)
    val = upper - lower
    return val

def vterm(x,xx,y,z,b,pressure,lambdaa,mu):
    val = -pressure/(4*pi)*( (1/(lambdaa + mu)) * (Kj(2,x,xx,y,z,b) - Kj(1,x,xx,y,z,b)) +  \
                           (z/mu) * np.log(  (xx-x+r0j(2,x,xx,y,z,b)) / (xx-x+r0j(1,x,xx,y,z,b)) ))
    return val

#**************************************************       
# displacement in z direction
#**************************************************       

def w_Love(x,y,z,a,b,pressure,lambdaa,mu):
    upper = wterm(x,y, b,z,a,pressure,lambdaa,mu)
    lower = wterm(x,y,-b,z,a,pressure,lambdaa,mu)
    val = upper - lower
    return val

#**************************************************       

def wterm(x,y,yy,z,a,pressure,lambdaa,mu):
    if z==0:
        val=(pressure/(4*pi*mu))*( (lambdaa + 2*mu)/(lambdaa + mu)*(Lj(1,x,y,yy,z,a) - Lj(2,x,y,yy,z,a)) )
    else:
        val=(pressure/(4*pi*mu))*( (lambdaa + 2*mu)/(lambdaa + mu)*(Lj(1,x,y,yy,z,a) - Lj(2,x,y,yy,z,a)) + \
             z*(np.arctan((a-x)*(yy-y)/(z*rj0(1,x,y,yy,z,a))) + np.arctan((a+x)*(yy-y)/(z*rj0(2,x,y,yy,z,a)))))
    return val


