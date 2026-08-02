import numpy as np

###############################################################################
# 1) smooth zone 1d
# 2) narrow zone 1d
# 3) two narrow zones 2d
# 4) rotated zone 2d (2x1 domain)
# 5) rotated zone 2d (2x3 domain)
# 6) pseudo1d bench (W.Bangerth idea)
# 7) disk radial dilation
# 8) rotated zone 2d (3x2 domain) - comparison with Alexandr' model
# 9) disk radial dilation + ring

experiment=5

###############################################################################

if experiment==4: theta=np.pi/6
if experiment==5: theta=np.pi/20
if experiment==7: radius=0.16
if experiment==8: theta=0.3
if experiment==9: R1=0.16 ; R2=0.6 ; R3=0.8
w3x=8.
w3y=12.
w4=12.

def ud(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return 0.5*(x-Lx/2 -Lx/2/np.pi*np.sin(2*np.pi*x/Lx) )
    #----------------
    if experiment==2:
       delta=Lx/8
       if x<=Lx/2-delta: 
          return -delta/2
       elif x<=Lx/2+delta:
          return 0.5*(x-Lx/2+delta/np.pi*np.sin(np.pi*(x-Lx/2)/delta))
       else: 
          return delta/2
    #----------------
    if experiment==3: 
       delta=Lx/w3x
       if x<=Lx/2-delta: 
          return -delta/2
       elif x<=Lx/2+delta:
          return 0.5*(x-Lx/2+delta/np.pi*np.sin(np.pi*(x-Lx/2)/delta))
       else: 
          return delta/2
    #----------------
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if xp<=Lx/2-delta4: 
          return -delta4/2*np.cos(theta)
       elif xp<=Lx/2+delta4:
          return 0.5*(xp-Lx/2+delta4/np.pi*np.sin(np.pi*(xp-Lx/2)/delta4))*np.cos(theta)
       else: 
          return delta4/2*np.cos(theta)
    #----------------
    if experiment==6:
       return x
    #----------------
    if experiment==7:
       r2= (x-Lx/2)**2+(y-Ly/2)**2
       if r2<radius**2:
          return (x-Lx/2)/3*np.sqrt(r2) 
       else:
          return radius**3/3/r2 * (x-Lx/2)
    #----------------
    if experiment==9:
       r2= (x-Lx/2)**2+(y-Ly/2)**2
       if r2<R1**2:
          return (x-Lx/2)/3*np.sqrt(r2) 
       elif r2<R2**2:
          return R1**3/3/r2 * (x-Lx/2)
       elif r2<R3**2:
          return (np.sqrt(r2)-R3)/(R2-R3)*R1**3/3/R2*(x-Lx/2)/np.sqrt(r2) 
       else:
          return 0


def dud_dx(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return (1-np.cos(2*np.pi*x/Lx))/2
    #----------------
    if experiment==2:
       delta=Lx/8
       if abs(x-Lx/2)<=delta: 
          return 0.5*(1+np.cos(np.pi*(x-Lx/2)/delta))
       else:
          return 0
    #----------------
    if experiment==3:
       delta=Lx/w3x
       if abs(x-Lx/2)<=delta: 
          return 0.5*(1+np.cos(np.pi*(x-Lx/2)/delta))
       else:
          return 0
    #----------------
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return 0.5*(1+np.cos(np.pi*(xp-Lx/2)/delta4))*np.cos(theta)*np.cos(theta)
       else:
          return 0
    #----------------
    if experiment==6:
       return 1
    #----------------
    if experiment==7: 
       r2= (x-Lx/2)**2+(y-Ly/2)**2
       if r2<radius**2:
          return np.sqrt(r2)
       else:
          return 0
    #----------------
    if experiment==9: 
       r2= (x-Lx/2)**2+(y-Ly/2)**2
       if r2<R1**2:
          return np.sqrt(r2)
       elif r2<R2**2:
          return 0 
       elif r2<R3**2:
          return (2-R3/np.sqrt(r2))/(R2-R3)*R1**3/3/R2
       else:
          return 0



def d2ud_dx2(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return np.pi/Lx*np.sin(2*np.pi*x/Lx)
    #----------------
    if experiment==2:
       delta=Lx/8
       if abs(x-Lx/2)<=delta: 
          return -0.5*np.pi/delta*np.sin(np.pi*(x-Lx/2)/delta)
       else:
          return 0
    #----------------
    if experiment==3:
       delta=Lx/w3x
       if abs(x-Lx/2)<=delta: 
          return -0.5*np.pi/delta*np.sin(np.pi*(x-Lx/2)/delta)
       else:
          return 0
    #----------------
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.cos(theta)*np.cos(theta)*np.cos(theta)
       else:
          return 0
    if experiment==6: 
       return 0 
    if experiment==7: 
       return 0 
    if experiment==9: 
       return 0 




def d2ud_dy2(x,y,Lx,Ly):
    if experiment==1: return 0
    if experiment==2: return 0
    if experiment==3: return 0
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.sin(theta)*np.sin(theta)*np.cos(theta)
       else:
          return 0
    if experiment==6: 
       return 0 
    if experiment==7: 
       return 0 
    if experiment==9: 
       return 0 


def d2ud_dxdy(x,y,Lx,Ly):
    if experiment==1: return 0
    if experiment==2: return 0
    if experiment==3: return 0
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.sin(theta)*np.cos(theta)*np.cos(theta)
       else:
          return 0
    if experiment==6: 
       return 0 
    if experiment==7: 
       return 0 
    if experiment==9: 
       return 0 





def vd(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return 0
    #----------------
    if experiment==2: 
       return 0
    #----------------
    if experiment==3: 
       delta=Ly/w3y
       if y<=Ly/2-delta: 
          return -delta/2
       elif y<=Ly/2+delta:
          return 0.5*(y-Ly/2+delta/np.pi*np.sin(np.pi*(y-Ly/2)/delta))
       else: 
          return delta/2
    #----------------
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if xp<=Lx/2-delta4: 
          return -delta4/2*np.sin(theta)
       elif xp<=Lx/2+delta4:
          return 0.5*(xp-Lx/2+delta4/np.pi*np.sin(np.pi*(xp-Lx/2)/delta4))*np.sin(theta)
       else: 
          return delta4/2*np.sin(theta)
    if experiment==6: 
       return 0 
    #----------------
    if experiment==7: 
       r2= (x-Lx/2)**2+(y-Ly/2)**2
       if r2<radius**2:
          return (y-Ly/2)/3*np.sqrt(r2) 
       else:
          return radius**3/3/r2 * (y-Ly/2)

    #----------------
    if experiment==9: 
       r2= (x-Lx/2)**2+(y-Ly/2)**2
       if r2<R1**2:
          return (y-Ly/2)/3*np.sqrt(r2) 
       elif r2<R2**2:
          return R1**3/3/r2 * (y-Ly/2)
       elif r2<R3**2:
          return (np.sqrt(r2)-R3)/(R2-R3)*R1**3/3/R2*(y-Ly/2)/np.sqrt(r2) 
       else:
          return 0
          



def dvd_dy(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return 0
    #----------------
    if experiment==2: 
       return 0
    #----------------
    if experiment==3:
       delta=Ly/w3y
       if abs(y-Ly/2)<=delta: 
          return 0.5*(1+np.cos(np.pi*(y-Ly/2)/delta))
       else:
          return 0
    #----------------
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return 0.5*(1+np.cos(np.pi*(xp-Lx/2)/delta4))*np.sin(theta)*np.sin(theta)
       else:
          return 0
    if experiment==6: 
       return 0 
    #----------------
    if experiment==7: 
       return 0
    if experiment==9: 
       return 0 


def d2vd_dy2(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return 0
    #----------------
    if experiment==2: 
       return 0
    #----------------
    if experiment==3:
       delta=Ly/w3y
       if abs(y-Ly/2)<=delta: 
          return -0.5*np.pi/delta*np.sin(np.pi*(y-Ly/2)/delta)
       else:
          return 0
    #----------------
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.sin(theta)*np.sin(theta)*np.sin(theta)
       else:
          return 0
    if experiment==6: 
       return 0 
    if experiment==7: 
       return 0
    if experiment==9: 
       return 0 

def d2vd_dx2(x,y,Lx,Ly):
    if experiment==1: return 0
    if experiment==2: return 0
    if experiment==3: return 0
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.cos(theta)*np.cos(theta)*np.sin(theta)
       else:
          return 0
    if experiment==6: 
       return 0 
    if experiment==7: 
       return 0 
    if experiment==9: 
       return 0 

def d2vd_dxdy(x,y,Lx,Ly):
    if experiment==1: return 0
    if experiment==2: return 0
    if experiment==3: return 0
    if experiment==4 or experiment==5 or experiment==8: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.sin(theta)*np.cos(theta)*np.sin(theta)
       else:
          return 0
    if experiment==6: 
       return 0 
    if experiment==7: 
       return 0 
    if experiment==9: 
       return 0 






