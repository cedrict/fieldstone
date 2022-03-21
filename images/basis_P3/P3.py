import numpy as np
from numpy.linalg import inv

def d0(r,s):
    return 1

def d1(r,s):
    return r

def d2(r,s):
    return s

def d3(r,s):
    return r*r

def d4(r,s):
    return r*s

def d5(r,s):
    return s*s

def d6(r,s):
    return r*r*r

def d7(r,s):
    return r*r*s

def d8(r,s):
    return r*s*s

def d9(r,s):
    return s*s*s

matrix=np.zeros((10,10),dtype=np.float64)

r=0 ; s=0
matrix[0,0]=d0(r,s)
matrix[0,1]=d1(r,s)
matrix[0,2]=d2(r,s)
matrix[0,3]=d3(r,s)
matrix[0,4]=d4(r,s)
matrix[0,5]=d5(r,s)
matrix[0,6]=d6(r,s)
matrix[0,7]=d7(r,s)
matrix[0,8]=d8(r,s)
matrix[0,9]=d9(r,s)

r=1/3 ; s=0
matrix[1,0]=d0(r,s)
matrix[1,1]=d1(r,s)
matrix[1,2]=d2(r,s)
matrix[1,3]=d3(r,s)
matrix[1,4]=d4(r,s)
matrix[1,5]=d5(r,s)
matrix[1,6]=d6(r,s)
matrix[1,7]=d7(r,s)
matrix[1,8]=d8(r,s)
matrix[1,9]=d9(r,s)

r=2/3 ; s=0
matrix[2,0]=d0(r,s)
matrix[2,1]=d1(r,s)
matrix[2,2]=d2(r,s)
matrix[2,3]=d3(r,s)
matrix[2,4]=d4(r,s)
matrix[2,5]=d5(r,s)
matrix[2,6]=d6(r,s)
matrix[2,7]=d7(r,s)
matrix[2,8]=d8(r,s)
matrix[2,9]=d9(r,s)

r=1 ; s=0
matrix[3,0]=d0(r,s)
matrix[3,1]=d1(r,s)
matrix[3,2]=d2(r,s)
matrix[3,3]=d3(r,s)
matrix[3,4]=d4(r,s)
matrix[3,5]=d5(r,s)
matrix[3,6]=d6(r,s)
matrix[3,7]=d7(r,s)
matrix[3,8]=d8(r,s)
matrix[3,9]=d9(r,s)

r=0 ; s=1/3
matrix[4,0]=d0(r,s)
matrix[4,1]=d1(r,s)
matrix[4,2]=d2(r,s)
matrix[4,3]=d3(r,s)
matrix[4,4]=d4(r,s)
matrix[4,5]=d5(r,s)
matrix[4,6]=d6(r,s)
matrix[4,7]=d7(r,s)
matrix[4,8]=d8(r,s)
matrix[4,9]=d9(r,s)

r=1/3 ; s=1/3
matrix[5,0]=d0(r,s)
matrix[5,1]=d1(r,s)
matrix[5,2]=d2(r,s)
matrix[5,3]=d3(r,s)
matrix[5,4]=d4(r,s)
matrix[5,5]=d5(r,s)
matrix[5,6]=d6(r,s)
matrix[5,7]=d7(r,s)
matrix[5,8]=d8(r,s)
matrix[5,9]=d9(r,s)

r=2/3 ; s=1/3
matrix[6,0]=d0(r,s)
matrix[6,1]=d1(r,s)
matrix[6,2]=d2(r,s)
matrix[6,3]=d3(r,s)
matrix[6,4]=d4(r,s)
matrix[6,5]=d5(r,s)
matrix[6,6]=d6(r,s)
matrix[6,7]=d7(r,s)
matrix[6,8]=d8(r,s)
matrix[6,9]=d9(r,s)

r=0 ; s=2/3
matrix[7,0]=d0(r,s)
matrix[7,1]=d1(r,s)
matrix[7,2]=d2(r,s)
matrix[7,3]=d3(r,s)
matrix[7,4]=d4(r,s)
matrix[7,5]=d5(r,s)
matrix[7,6]=d6(r,s)
matrix[7,7]=d7(r,s)
matrix[7,8]=d8(r,s)
matrix[7,9]=d9(r,s)

r=1/3 ; s=2/3
matrix[8,0]=d0(r,s)
matrix[8,1]=d1(r,s)
matrix[8,2]=d2(r,s)
matrix[8,3]=d3(r,s)
matrix[8,4]=d4(r,s)
matrix[8,5]=d5(r,s)
matrix[8,6]=d6(r,s)
matrix[8,7]=d7(r,s)
matrix[8,8]=d8(r,s)
matrix[8,9]=d9(r,s)

r=0 ; s=1
matrix[9,0]=d0(r,s)
matrix[9,1]=d1(r,s)
matrix[9,2]=d2(r,s)
matrix[9,3]=d3(r,s)
matrix[9,4]=d4(r,s)
matrix[9,5]=d5(r,s)
matrix[9,6]=d6(r,s)
matrix[9,7]=d7(r,s)
matrix[9,8]=d8(r,s)
matrix[9,9]=d9(r,s)

print('***********************************************')
print('* matrix *')
print('***********************************************')
print(matrix)

I = inv(matrix) * 2 

print('***********************************************')
print('* 2X inverse of matrix*')
print('***********************************************')
print(np.round(I))
print('***********************************************')

def NNN(r,s):
    val = np.zeros(10,dtype=np.float64)
    val[0]=0.5*( 2 -11*r - 11*s + 18*r**2 + 36*r*s + 18*s**2 -9*r**3 -27*r**2*s -27*r*s**2 -9*s**3)
    val[1]=0.5*( 18*r-45*r**2-45*r*s +27*r**3 +54*r**2*s+27*r*s**2  ) 
    val[2]=0.5*( -9*r+36*r**2+9*r*s -27*r**3 -27*r**2*s  ) 
    val[3]=0.5*( 2*r-9*r**2+9*r**3  ) 
    val[4]=0.5*( 18*s -45*r*s-45*s**2+27*r**2*s+54*r*s**2+27*s**3  ) 
    val[5]=0.5*( 54*r*s-54*r**2*s-54*r*s**2   ) 
    val[6]=0.5*( -9*r*s+27*r**2*s   ) 
    val[7]=0.5*( -9*s+9*r*s+36*s**2-27*r*s**2-27*s**3  ) 
    val[8]=0.5*( -9*r*s+27*r*s**2 ) 
    val[9]=0.5*( 2*s-9*s**2+9*s**3  ) 
    return val

print('0***********************************************')
r=0 ; s=0
print(NNN(r,s))
print('1***********************************************')
r=1/3 ; s=0
print(NNN(r,s))
print('2***********************************************')
r=2/3 ; s=0
print(NNN(r,s))
print('3***********************************************')
r=1 ; s=0
print(NNN(r,s))
print('4***********************************************')
r=0 ; s=1/3
print(NNN(r,s))
print('5***********************************************')
r=1/3 ; s=1/3
print(NNN(r,s))
print('6***********************************************')
r=2/3 ; s=1/3
print(NNN(r,s))
print('7***********************************************')
r=0 ; s=2/3
print(NNN(r,s))
print('8***********************************************')
r=1/3 ; s=2/3
print(NNN(r,s))
print('9***********************************************')
r=0 ; s=1
print(NNN(r,s))



