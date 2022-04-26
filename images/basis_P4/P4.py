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

def d10(r,s):
    return r**4

def d11(r,s):
    return r**3*s

def d12(r,s):
    return r**2*s**2

def d13(r,s):
    return r*s**3

def d14(r,s):
    return s**4


matrix=np.zeros((15,15),dtype=np.float64)

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
matrix[0,10]=d10(r,s)
matrix[0,11]=d11(r,s)
matrix[0,12]=d12(r,s)
matrix[0,13]=d13(r,s)
matrix[0,14]=d14(r,s)

r=0.25 ; s=0
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
matrix[1,10]=d10(r,s)
matrix[1,11]=d11(r,s)
matrix[1,12]=d12(r,s)
matrix[1,13]=d13(r,s)
matrix[1,14]=d14(r,s)

r=0.5 ; s=0
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
matrix[2,10]=d10(r,s)
matrix[2,11]=d11(r,s)
matrix[2,12]=d12(r,s)
matrix[2,13]=d13(r,s)
matrix[2,14]=d14(r,s)

r=0.75 ; s=0
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
matrix[3,10]=d10(r,s)
matrix[3,11]=d11(r,s)
matrix[3,12]=d12(r,s)
matrix[3,13]=d13(r,s)
matrix[3,14]=d14(r,s)

r=1 ; s=0
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
matrix[4,10]=d10(r,s)
matrix[4,11]=d11(r,s)
matrix[4,12]=d12(r,s)
matrix[4,13]=d13(r,s)
matrix[4,14]=d14(r,s)

r=0 ; s=0.25
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
matrix[5,10]=d10(r,s)
matrix[5,11]=d11(r,s)
matrix[5,12]=d12(r,s)
matrix[5,13]=d13(r,s)
matrix[5,14]=d14(r,s)




r=0.25 ; s=0.25
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
matrix[6,10]=d10(r,s)
matrix[6,11]=d11(r,s)
matrix[6,12]=d12(r,s)
matrix[6,13]=d13(r,s)
matrix[6,14]=d14(r,s)

r=0.5 ; s=0.25
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
matrix[7,10]=d10(r,s)
matrix[7,11]=d11(r,s)
matrix[7,12]=d12(r,s)
matrix[7,13]=d13(r,s)
matrix[7,14]=d14(r,s)

r=0.75 ; s=0.25
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
matrix[8,10]=d10(r,s)
matrix[8,11]=d11(r,s)
matrix[8,12]=d12(r,s)
matrix[8,13]=d13(r,s)
matrix[8,14]=d14(r,s)

r=0 ; s=0.5
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
matrix[9,10]=d10(r,s)
matrix[9,11]=d11(r,s)
matrix[9,12]=d12(r,s)
matrix[9,13]=d13(r,s)
matrix[9,14]=d14(r,s)




r=0.25 ; s=0.5
matrix[10,0]=d0(r,s)
matrix[10,1]=d1(r,s)
matrix[10,2]=d2(r,s)
matrix[10,3]=d3(r,s)
matrix[10,4]=d4(r,s)
matrix[10,5]=d5(r,s)
matrix[10,6]=d6(r,s)
matrix[10,7]=d7(r,s)
matrix[10,8]=d8(r,s)
matrix[10,9]=d9(r,s)
matrix[10,10]=d10(r,s)
matrix[10,11]=d11(r,s)
matrix[10,12]=d12(r,s)
matrix[10,13]=d13(r,s)
matrix[10,14]=d14(r,s)

r=0.5 ; s=0.5
matrix[11,0]=d0(r,s)
matrix[11,1]=d1(r,s)
matrix[11,2]=d2(r,s)
matrix[11,3]=d3(r,s)
matrix[11,4]=d4(r,s)
matrix[11,5]=d5(r,s)
matrix[11,6]=d6(r,s)
matrix[11,7]=d7(r,s)
matrix[11,8]=d8(r,s)
matrix[11,9]=d9(r,s)
matrix[11,10]=d10(r,s)
matrix[11,11]=d11(r,s)
matrix[11,12]=d12(r,s)
matrix[11,13]=d13(r,s)
matrix[11,14]=d14(r,s)


r=0 ; s=0.75
matrix[12,0]=d0(r,s)
matrix[12,1]=d1(r,s)
matrix[12,2]=d2(r,s)
matrix[12,3]=d3(r,s)
matrix[12,4]=d4(r,s)
matrix[12,5]=d5(r,s)
matrix[12,6]=d6(r,s)
matrix[12,7]=d7(r,s)
matrix[12,8]=d8(r,s)
matrix[12,9]=d9(r,s)
matrix[12,10]=d10(r,s)
matrix[12,11]=d11(r,s)
matrix[12,12]=d12(r,s)
matrix[12,13]=d13(r,s)
matrix[12,14]=d14(r,s)


r=0.25 ; s=0.75
matrix[13,0]=d0(r,s)
matrix[13,1]=d1(r,s)
matrix[13,2]=d2(r,s)
matrix[13,3]=d3(r,s)
matrix[13,4]=d4(r,s)
matrix[13,5]=d5(r,s)
matrix[13,6]=d6(r,s)
matrix[13,7]=d7(r,s)
matrix[13,8]=d8(r,s)
matrix[13,9]=d9(r,s)
matrix[13,10]=d10(r,s)
matrix[13,11]=d11(r,s)
matrix[13,12]=d12(r,s)
matrix[13,13]=d13(r,s)
matrix[13,14]=d14(r,s)


r=0 ; s=1
matrix[14,0]=d0(r,s)
matrix[14,1]=d1(r,s)
matrix[14,2]=d2(r,s)
matrix[14,3]=d3(r,s)
matrix[14,4]=d4(r,s)
matrix[14,5]=d5(r,s)
matrix[14,6]=d6(r,s)
matrix[14,7]=d7(r,s)
matrix[14,8]=d8(r,s)
matrix[14,9]=d9(r,s)
matrix[14,10]=d10(r,s)
matrix[14,11]=d11(r,s)
matrix[14,12]=d12(r,s)
matrix[14,13]=d13(r,s)
matrix[14,14]=d14(r,s)





print('***********************************************')
print('* matrix *')
print('***********************************************')
print(matrix)
print('***********************************************')
print(matrix*256)


I = inv(matrix)*3 

print('***********************************************')
print('* 3X inverse of matrix*')
print('***********************************************')

print(I)
print(np.round(I))
print('***********************************************')




def NNN(r,s):
    val = np.zeros(15,dtype=np.float64)
    val[ 0]=(3-25*r-25*s+70*r**2+140*r*s+70*s**2 -80*r**3-240*r**2*s-240*r*s**2-80*s**3\
             +32*r**4 + 128*r**3*s + 192*r**2*s**2 + 128*r*s**3 + 32*s**4 )/3
    val[ 1]=(48*r -208*r**2-208*r*s +288*r**3+576*r**2*s+288*r*s**2-128*r**4-\
             384*r**3*s-384*r**2*s**2-128*r*s**3)/3
    val[ 2]=(-36*r +228*r**2+84*r*s -384*r**3-432*r**2*s-48*r*s**2+192*r**4+384*r**3*s+192*r**2*s**2)/3
    val[ 3]=(16*r -112*r**2-16*r*s +224*r**3+96*r**2*s-128*r**4-128*r**3*s)/3
    val[ 4]=(-3*r+22*r**2 -48*r**3+32*r**4)/3
    val[ 5]=(48*s -208*r*s-208*s**2 +288*r**2*s+576*r*s**2+288*s**3-128*r**3*s\
             -384*r**2*s**2-384*r*s**3-128*s**4)/3
    val[ 6]=(288*r*s -672*r**2*s-672*r*s**2+384*r**3*s+768*r**2*s**2+384*r*s**3)/3
    val[ 7]=(-96*r*s +480*r**2*s+96*r*s**2-384*r**3*s-384*r**2*s**2)/3
    val[ 8]=(16*r*s -96*r**2*s+128*r**3*s )/3
    val[ 9]=(-36*s+84*r*s+228*s**2 -48*r**2*s-432*r*s**2-384*s**3 +192*r**2*s**2+384*r*s**3+192*s**4)/3
    val[10]=(-96*r*s+96*r**2*s+480*r*s**2-384*r**2*s**2-384*r*s**3)/3
    val[11]=(12*r*s-48*r**2*s-48*r*s**2+192*r**2*s**2)/3
    val[12]=(16*s-16*r*s-112*s**2+96*r*s**2+224*s**3-128*r*s**3-128*s**4)/3
    val[13]=(16*r*s-96*r*s**2+128*r*s**3)/3
    val[14]=(-3*s+22*s**2-48*s**3+32*s**4)/3
    return val

print('1***********************************************')
r=0 ; s=0
print(NNN(r,s))
print('2***********************************************')
r=0.25 ; s=0
print(NNN(r,s))
print('3***********************************************')
r=0.5 ; s=0
print(NNN(r,s))
print('4***********************************************')
r=0.75 ; s=0
print(NNN(r,s))
print('5***********************************************')
r=1 ; s=0
print(NNN(r,s))

print('6***********************************************')
r=0 ; s=0.25
print(NNN(r,s))
print('7***********************************************')
r=0.25 ; s=0.25
print(NNN(r,s))
print('8***********************************************')
r=0.5 ; s=0.25
print(NNN(r,s))
print('9***********************************************')
r=0.75 ; s=0.25
print(NNN(r,s))


print('10***********************************************')
r=0 ; s=0.5
print(NNN(r,s))
print('11***********************************************')
r=0.25 ; s=0.5
print(NNN(r,s))
print('12***********************************************')
r=0.5 ; s=0.5
print(NNN(r,s))

print('13***********************************************')
r=0 ; s=0.75
print(NNN(r,s))
print('14***********************************************')
r=0.25 ; s=0.75
print(NNN(r,s))

print('14***********************************************')
r=0 ; s=1
print(NNN(r,s))



