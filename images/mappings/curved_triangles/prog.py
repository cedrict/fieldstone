import numpy as np
import random

###############################################################################

def N(r,s,space):

    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[0]=1-r-s
       val[1]=r
       val[2]=s

    if space=='P2':
       val = np.zeros(6,dtype=np.float64)
       val[0]= 1-3*r-3*s+2*r**2+4*r*s+2*s**2 
       val[1]= -r+2*r**2
       val[2]= -s+2*s**2
       val[3]= 4*r-4*r**2-4*r*s
       val[4]= 4*r*s 
       val[5]= 4*s-4*r*s-4*s**2

    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[0]=0.5*(2 -11*r - 11*s + 18*r**2 + 36*r*s + 18*s**2 -9*r**3 -27*r**2*s -27*r*s**2 -9*s**3)
       val[1]=0.5*(18*r-45*r**2-45*r*s +27*r**3 +54*r**2*s+27*r*s**2  )
       val[2]=0.5*(-9*r+36*r**2+9*r*s -27*r**3 -27*r**2*s  )
       val[3]=0.5*(2*r-9*r**2+9*r**3  )
       val[4]=0.5*(18*s -45*r*s-45*s**2+27*r**2*s+54*r*s**2+27*s**3  )
       val[5]=0.5*(54*r*s-54*r**2*s-54*r*s**2   )
       val[6]=0.5*(-9*r*s+27*r**2*s   )
       val[7]=0.5*(-9*s+9*r*s+36*s**2-27*r*s**2-27*s**3  )
       val[8]=0.5*(-9*r*s+27*r*s**2 )
       val[9]=0.5*(2*s-9*s**2+9*s**3  )

    if space=='P4':
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

###############################################################################

def dNdr(r,s,space):
    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[0]=-1
       val[1]= 1
       val[2]= 0
    if space=='P2':
       val=np.zeros(6,dtype=np.float64)
       val[0]= -3+4*r+4*s
       val[1]= -1+4*r
       val[2]= 0
       val[3]= 4-8*r-4*s
       val[4]= 4*s
       val[5]= -4*s
    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[0]=0.5*(-11+36*r+36*s-27*r**2-54*r*s-27*s**2)
       val[1]=0.5*(18-90*r-45*s+81*r**2+108*r*s+27*s**2)
       val[2]=0.5*(-9+72*r+9*s-81*r**2-54*r*s)
       val[3]=0.5*(2-18*r+27*r**2)
       val[4]=0.5*(-45*s+54*r*s+54*s**2)
       val[5]=0.5*(54*s-108*r*s-54*s**2)
       val[6]=0.5*(-9*s+54*r*s)
       val[7]=0.5*(9*s-27*s**2)
       val[8]=0.5*(-9*s+27*s**2)
       val[9]=0.
    if space=='P4':
       val = np.zeros(15,dtype=np.float64)
       frac13=1./3.
       val[ 0]=frac13*(-25+140*r+140*s-240*r**2-480*r*s-240*s**2+128*r**3+384*r**2*s+384*r*s**2+128*s**3)
       val[ 1]=frac13*(48-416*r-208*s+864*r**2+1152*r*s+288*s**2-512*r**3-1152*r**2*s-768*r*s**2-128*s**3)
       val[ 2]=frac13*(-36+456*r+84*s-1152*r**2-864*r*s-48*s**2+768*r**3+1152*r**2*s+384*r*s**2)
       val[ 3]=frac13*(16-224*r-16*s+672*r**2+192*r*s-512*r**3-384*r**2*s)
       val[ 4]=frac13*(-3+44*r-144*r**2+128*r**3)
       val[ 5]=frac13*(-208*s+576*r*s+576*s**2-384*r**2*s-768*r*s**2-384*s**3)
       val[ 6]=frac13*(288*s-1344*r*s-672*s**2+1152*r**2*s+1536*r*s**2+384*s**3)
       val[ 7]=frac13*(-96*s+960*r*s+96*s**2-1152*r**2*s-768*r*s**2)
       val[ 8]=frac13*(16*s-192*r*s+384*r**2*s)
       val[ 9]=frac13*(84*s-96*r*s-432*s**2+384*r*s**2+384*s**3)
       val[10]=frac13*(-96*s+192*r*s+480*s**2-768*r*s**2-384*s**3)
       val[11]=frac13*(12*s-96*r*s-48*s**2+384*r*s**2)
       val[12]=frac13*(-16*s+96*s**2-128*s**3)
       val[13]=frac13*(16*s-96*s**2+128*s**3)
       val[14]=0
    return val

###############################################################################

def dNds(r,s,space):
    if space=='P1':
       val = np.zeros(3,dtype=np.float64)
       val[0]=-1
       val[1]= 0
       val[2]= 1
    if space=='P2':
       val=np.zeros(6,dtype=np.float64)
       val[0]= -3+4*r+4*s
       val[1]= 0
       val[2]= -1+4*s
       val[3]= -4*r
       val[4]= +4*r
       val[5]= 4-4*r-8*s
    if space=='P3':
       val = np.zeros(10,dtype=np.float64)
       val[0]=0.5*(-11+36*r+36*s-27*r**2-54*r*s-27*s**2)
       val[1]=0.5*(-45*r+54*r**2+54*r*s)
       val[2]=0.5*(9*r-27*r**2)
       val[3]=0.
       val[4]=0.5*(18-45*r-90*s+27*r**2+108*r*s+81*s**2)
       val[5]=0.5*(54*r-54*r**2-108*r*s)
       val[6]=0.5*(-9*r+27*r**2)
       val[7]=0.5*(-9+9*r+72*s-54*r*s-81*s**2)
       val[8]=0.5*(-9*r+54*r*s)
       val[9]=0.5*(2-18*s+27*s**2)
    if space=='P4':
       val = np.zeros(15,dtype=np.float64)
       frac13=1./3.
       val[ 0]=frac13*(-25+140*r+140*s-240*r**2-480*r*s-240*s**2+128*r**3+384*r**2*s+384*r*s**2+128*s**3)
       val[ 1]=frac13*(-208*r+576*r**2+576*r*s-384*r**3-768*r**2*s-384*r*s**2)
       val[ 2]=frac13*(84*r-432*r**2-96*r*s+384*r**3+384*r**2*s)
       val[ 3]=frac13*(-16*r+96*r**2-128*r**3)
       val[ 4]=0
       val[ 5]=frac13*(48-208*r-416*s+288*r**2+1152*r*s+864*s**2-128*r**3-768*r**2*s-1152*r*s**2-512*s**3)
       val[ 6]=frac13*(288*r-672*r**2-1344*r*s+384*r**3+1536*r**2*s+1152*r*s**2)
       val[ 7]=frac13*(-96*r+480*r**2+192*r*s-384*r**3-768*r**2*s)
       val[ 8]=frac13*(16*r-96*r**2+128*r**3)
       val[ 9]=frac13*(-36+84*r+456*s-48*r**2-864*r*s-1152*s**2+384*r**2*s+1152*r*s**2+768*s**3)
       val[10]=frac13*(-96*r+96*r**2+960*r*s-768*r**2*s-1152*r*s**2)
       val[11]=frac13*(12*r-48*r**2-96*r*s+384*r**2*s)
       val[12]=frac13*(16-16*r-224*s+192*r*s+672*s**2-384*r*s**2-512*s**3)
       val[13]=frac13*(16*r-192*r*s+384*r*s**2)
       val[14]=frac13*(-3+44*s-144*s**2+128*s**3)
    return val

###############################################################################

thetaA=31./180.*np.pi
thetaB=23./180.*np.pi
thetaC=52./180.*np.pi
Rin=1.
Rout=1.5

#r,theta coords of vertices
radP1=[Rin,Rout,Rout]
thetaP1=[thetaA,thetaB,thetaC]
xP1=radP1*np.cos(thetaP1)
yP1=radP1*np.sin(thetaP1)

npts=10000
rr=np.zeros(npts,dtype=np.float64)   
ss=np.zeros(npts,dtype=np.float64)   
xx=np.zeros(npts,dtype=np.float64)   
yy=np.zeros(npts,dtype=np.float64)   

volume=False

###############################################################################
###############################################################################
###############################################################################
print('**********P1*********')
m=3
theta=[thetaA,thetaB,thetaC]
r=[Rin,Rout,Rout]
x=r*np.cos(theta)
y=r*np.sin(theta)

np.savetxt('xy_P1.ascii',np.array([x,y]).T)

for i in range(0,npts):
    # compute random r,s coordinates
    if volume:
       rr[i]=random.uniform(0,+1)
       ss[i]=random.uniform(0,1-rr[i])
    else:
       rr[i]=i/(npts-1)
       ss[i]=1-rr[i]
    # compute basis function values at r,s
    NN=N(rr[i],ss[i],'P1')
    # compute x,y coordinates
    xx[i]=NN.dot(x)
    yy[i]=NN.dot(y)

#np.savetxt('rs1.ascii',np.array([rr,ss]).T)

if volume:
   np.savetxt('xy1_volume.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
else:
   np.savetxt('xy1_line.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

#######################################

for nqel in (3,4,6,7):
    qcoords_r=np.zeros(nqel,dtype=np.float64)   
    qcoords_s=np.zeros(nqel,dtype=np.float64)   
    qweights =np.zeros(nqel,dtype=np.float64)   
    if nqel==3: #quadratic 2nd order - confirmed
       qcoords_r[0]=1/6 ; qcoords_s[0]=1/6 ; qweights[0]=1./3/2
       qcoords_r[1]=2/3 ; qcoords_s[1]=1/6 ; qweights[1]=1./3/2
       qcoords_r[2]=1/6 ; qcoords_s[2]=2/3 ; qweights[2]=1./3/2
    elif nqel==4: #cubic 3rd order - confirmed 
       qcoords_r[0]=1/3 ; qcoords_s[0]=1/3 ; qweights[0]=-27./48/2
       qcoords_r[1]=1/5 ; qcoords_s[1]=3/5 ; qweights[1]= 25./48/2
       qcoords_r[2]=1/5 ; qcoords_s[2]=1/5 ; qweights[2]= 25./48/2
       qcoords_r[3]=3/5 ; qcoords_s[3]=1/5 ; qweights[3]= 25./48/2
    elif nqel==6: #4th order - confirmed
       qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
       qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
       qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
       qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
       qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
       qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 
    elif nqel==7: #5th order - confirmed
       qcoords_r[0]=0.1012865073235 ; qcoords_s[0]=0.1012865073235 ; qweights[0]=0.0629695902724 
       qcoords_r[1]=0.7974269853531 ; qcoords_s[1]=0.1012865073235 ; qweights[1]=0.0629695902724 
       qcoords_r[2]=0.1012865073235 ; qcoords_s[2]=0.7974269853531 ; qweights[2]=0.0629695902724 
       qcoords_r[3]=0.4701420641051 ; qcoords_s[3]=0.0597158717898 ; qweights[3]=0.0661970763942 
       qcoords_r[4]=0.4701420641051 ; qcoords_s[4]=0.4701420641051 ; qweights[4]=0.0661970763942 
       qcoords_r[5]=0.0597158717898 ; qcoords_s[5]=0.4701420641051 ; qweights[5]=0.0661970763942 
       qcoords_r[6]=0.3333333333333 ; qcoords_s[6]=0.3333333333333 ; qweights[6]=0.1125000000000 

    area=0
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNNNVdr=dNdr(rq,sq,'P1')
        dNNNVds=dNds(rq,sq,'P1')
        jcb=np.zeros((2,2),dtype=np.float64)
        jcb[0,0]=np.dot(dNNNVdr[:],x[:])
        jcb[0,1]=np.dot(dNNNVdr[:],y[:])
        jcb[1,0]=np.dot(dNNNVds[:],x[:])
        jcb[1,1]=np.dot(dNNNVds[:],y[:])
        jcobq=np.linalg.det(jcb)
        area+=jcobq*weightq

    print('nqel=',nqel,'area=',area)

###############################################################################
###############################################################################
###############################################################################
print('**********P2*********')
m=6

theta=[thetaA, thetaB, thetaC, 0, (thetaB+thetaC)/2, 0.]   
r=[Rin, Rout, Rout, 0, Rout, 0] 

x=r*np.cos(theta)
y=r*np.sin(theta)

x[3]=(x[0]+x[1])/2 ; y[3]=(y[0]+y[1])/2
x[5]=(x[0]+x[2])/2 ; y[5]=(y[0]+y[2])/2

np.savetxt('xy_P2.ascii',np.array([x,y]).T)

for i in range(0,npts):
    # compute random r,s coordinates
    if volume:
       rr[i]=random.uniform(0,+1)
       ss[i]=random.uniform(0,1-rr[i])
    else:
       rr[i]=i/(npts-1)
       ss[i]=1-rr[i]
    # compute basis function values at r,s
    NN=N(rr[i],ss[i],'P2')
    # compute x,y coordinates
    xx[i]=NN.dot(x)
    yy[i]=NN.dot(y)

#np.savetxt('rs2.ascii',np.array([rr,ss]).T)

if volume:
   np.savetxt('xy2_volume.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
else:
   np.savetxt('xy2_line.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

#######################################

for nqel in (3,4,6,7):
    qcoords_r=np.zeros(nqel,dtype=np.float64)   
    qcoords_s=np.zeros(nqel,dtype=np.float64)   
    qweights =np.zeros(nqel,dtype=np.float64)   
    if nqel==3: #quadratic 2nd order - confirmed
       qcoords_r[0]=1/6 ; qcoords_s[0]=1/6 ; qweights[0]=1./3/2
       qcoords_r[1]=2/3 ; qcoords_s[1]=1/6 ; qweights[1]=1./3/2
       qcoords_r[2]=1/6 ; qcoords_s[2]=2/3 ; qweights[2]=1./3/2
    elif nqel==4: #cubic 3rd order - confirmed 
       qcoords_r[0]=1/3 ; qcoords_s[0]=1/3 ; qweights[0]=-27./48/2
       qcoords_r[1]=1/5 ; qcoords_s[1]=3/5 ; qweights[1]= 25./48/2
       qcoords_r[2]=1/5 ; qcoords_s[2]=1/5 ; qweights[2]= 25./48/2
       qcoords_r[3]=3/5 ; qcoords_s[3]=1/5 ; qweights[3]= 25./48/2
    elif nqel==6: #4th order - confirmed
       qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
       qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
       qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
       qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
       qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
       qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 
    elif nqel==7: #5th order - confirmed
       qcoords_r[0]=0.1012865073235 ; qcoords_s[0]=0.1012865073235 ; qweights[0]=0.0629695902724 
       qcoords_r[1]=0.7974269853531 ; qcoords_s[1]=0.1012865073235 ; qweights[1]=0.0629695902724 
       qcoords_r[2]=0.1012865073235 ; qcoords_s[2]=0.7974269853531 ; qweights[2]=0.0629695902724 
       qcoords_r[3]=0.4701420641051 ; qcoords_s[3]=0.0597158717898 ; qweights[3]=0.0661970763942 
       qcoords_r[4]=0.4701420641051 ; qcoords_s[4]=0.4701420641051 ; qweights[4]=0.0661970763942 
       qcoords_r[5]=0.0597158717898 ; qcoords_s[5]=0.4701420641051 ; qweights[5]=0.0661970763942 
       qcoords_r[6]=0.3333333333333 ; qcoords_s[6]=0.3333333333333 ; qweights[6]=0.1125000000000 

    area=0
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNNNVdr=dNdr(rq,sq,'P2')
        dNNNVds=dNds(rq,sq,'P2')
        jcb=np.zeros((2,2),dtype=np.float64)
        jcb[0,0]=np.dot(dNNNVdr[:],x[:])
        jcb[0,1]=np.dot(dNNNVdr[:],y[:])
        jcb[1,0]=np.dot(dNNNVds[:],x[:])
        jcb[1,1]=np.dot(dNNNVds[:],y[:])
        jcobq=np.linalg.det(jcb)
        area+=jcobq*weightq

    print('nqel=',nqel,'area=',area)


###############################################################################
###############################################################################
###############################################################################
print('**********P3*********')
m=10

#location of nodes in ref cell
rnodesP3=[0, 1/3, 2/3, 1,   0, 1/3, 2/3,   0, 1/3, 0] 
snodesP3=[0,   0,   0, 0, 1/3, 1/3, 1/3, 2/3, 2/3, 1]

x=np.zeros(m,dtype=np.float64)
y=np.zeros(m,dtype=np.float64)
for k in range(0,m):
    NN=N(rnodesP3[k],snodesP3[k],'P1')
    x[k]=NN.dot(xP1)
    y[k]=NN.dot(yP1)

#correct positions of nodes on circle
x[6]=Rout*np.cos( (2*thetaB+thetaC)/3  )
y[6]=Rout*np.sin( (2*thetaB+thetaC)/3  )
x[8]=Rout*np.cos( (thetaB+2*thetaC)/3  )
y[8]=Rout*np.sin( (thetaB+2*thetaC)/3  )


np.savetxt('xy_P3.ascii',np.array([x,y]).T)

for i in range(0,npts):
    # compute random r,s coordinates
    if volume:
       rr[i]=random.uniform(0,+1)
       ss[i]=random.uniform(0,1-rr[i])
    else:
       rr[i]=i/(npts-1)
       ss[i]=1-rr[i]
    # compute basis function values at r,s
    NN=N(rr[i],ss[i],'P3')
    # compute x,y coordinates
    xx[i]=NN.dot(x)
    yy[i]=NN.dot(y)

#np.savetxt('rs2.ascii',np.array([rr,ss]).T)

if volume:
   np.savetxt('xy3_volume.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
else:
   np.savetxt('xy3_line.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

#######################################

for nqel in (3,4,6,7):
    qcoords_r=np.zeros(nqel,dtype=np.float64)   
    qcoords_s=np.zeros(nqel,dtype=np.float64)   
    qweights =np.zeros(nqel,dtype=np.float64)   
    if nqel==3: #quadratic 2nd order - confirmed
       qcoords_r[0]=1/6 ; qcoords_s[0]=1/6 ; qweights[0]=1./3/2
       qcoords_r[1]=2/3 ; qcoords_s[1]=1/6 ; qweights[1]=1./3/2
       qcoords_r[2]=1/6 ; qcoords_s[2]=2/3 ; qweights[2]=1./3/2
    elif nqel==4: #cubic 3rd order - confirmed 
       qcoords_r[0]=1/3 ; qcoords_s[0]=1/3 ; qweights[0]=-27./48/2
       qcoords_r[1]=1/5 ; qcoords_s[1]=3/5 ; qweights[1]= 25./48/2
       qcoords_r[2]=1/5 ; qcoords_s[2]=1/5 ; qweights[2]= 25./48/2
       qcoords_r[3]=3/5 ; qcoords_s[3]=1/5 ; qweights[3]= 25./48/2
    elif nqel==6: #4th order - confirmed
       qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
       qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
       qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
       qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
       qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
       qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 
    elif nqel==7: #5th order - confirmed
       qcoords_r[0]=0.1012865073235 ; qcoords_s[0]=0.1012865073235 ; qweights[0]=0.0629695902724 
       qcoords_r[1]=0.7974269853531 ; qcoords_s[1]=0.1012865073235 ; qweights[1]=0.0629695902724 
       qcoords_r[2]=0.1012865073235 ; qcoords_s[2]=0.7974269853531 ; qweights[2]=0.0629695902724 
       qcoords_r[3]=0.4701420641051 ; qcoords_s[3]=0.0597158717898 ; qweights[3]=0.0661970763942 
       qcoords_r[4]=0.4701420641051 ; qcoords_s[4]=0.4701420641051 ; qweights[4]=0.0661970763942 
       qcoords_r[5]=0.0597158717898 ; qcoords_s[5]=0.4701420641051 ; qweights[5]=0.0661970763942 
       qcoords_r[6]=0.3333333333333 ; qcoords_s[6]=0.3333333333333 ; qweights[6]=0.1125000000000 

    area=0
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNNNVdr=dNdr(rq,sq,'P3')
        dNNNVds=dNds(rq,sq,'P3')
        jcb=np.zeros((2,2),dtype=np.float64)
        jcb[0,0]=np.dot(dNNNVdr[:],x[:])
        jcb[0,1]=np.dot(dNNNVdr[:],y[:])
        jcb[1,0]=np.dot(dNNNVds[:],x[:])
        jcb[1,1]=np.dot(dNNNVds[:],y[:])
        jcobq=np.linalg.det(jcb)
        area+=jcobq*weightq

    print('nqel=',nqel,'area=',area)

###############################################################################
###############################################################################
###############################################################################
print('**********P4*********')
m=15

#location of nodes in ref cell
rnodesP4=[0, 1/4, 1/2, 3/4, 1,   0, 1/4, 1/2, 3/4, 0,   1/4, 1/2, 0,   1/4, 0]
snodesP4=[0,   0,   0,   0, 0, 1/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 1]

x=np.zeros(m,dtype=np.float64)
y=np.zeros(m,dtype=np.float64)
for k in range(0,m):
    NN=N(rnodesP4[k],snodesP4[k],'P1')
    x[k]=NN.dot(xP1)
    y[k]=NN.dot(yP1)

#correct positions:
x[ 8]=Rout*np.cos( (3*thetaB+1*thetaC)/4  )
y[ 8]=Rout*np.sin( (3*thetaB+1*thetaC)/4  )
x[11]=Rout*np.cos( (2*thetaB+2*thetaC)/4  )
y[11]=Rout*np.sin( (2*thetaB+2*thetaC)/4  )
x[13]=Rout*np.cos( (1*thetaB+3*thetaC)/4  )
y[13]=Rout*np.sin( (1*thetaB+3*thetaC)/4  )

np.savetxt('xy_P4.ascii',np.array([x,y]).T)

for i in range(0,npts):
    # compute random r,s coordinates
    if volume:
       rr[i]=random.uniform(0,+1)
       ss[i]=random.uniform(0,1-rr[i])
    else:
       rr[i]=i/(npts-1)
       ss[i]=1-rr[i]
    # compute basis function values at r,s
    NN=N(rr[i],ss[i],'P4')
    # compute x,y coordinates
    xx[i]=NN.dot(x)
    yy[i]=NN.dot(y)

#np.savetxt('rs2.ascii',np.array([rr,ss]).T)

if volume:
   np.savetxt('xy4_volume.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
else:
   np.savetxt('xy4_line.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

#######################################

for nqel in (3,4,6,7):
    qcoords_r=np.zeros(nqel,dtype=np.float64)   
    qcoords_s=np.zeros(nqel,dtype=np.float64)   
    qweights =np.zeros(nqel,dtype=np.float64)   
    if nqel==3: #quadratic 2nd order - confirmed
       qcoords_r[0]=1/6 ; qcoords_s[0]=1/6 ; qweights[0]=1./3/2
       qcoords_r[1]=2/3 ; qcoords_s[1]=1/6 ; qweights[1]=1./3/2
       qcoords_r[2]=1/6 ; qcoords_s[2]=2/3 ; qweights[2]=1./3/2
    elif nqel==4: #cubic 3rd order - confirmed 
       qcoords_r[0]=1/3 ; qcoords_s[0]=1/3 ; qweights[0]=-27./48/2
       qcoords_r[1]=1/5 ; qcoords_s[1]=3/5 ; qweights[1]= 25./48/2
       qcoords_r[2]=1/5 ; qcoords_s[2]=1/5 ; qweights[2]= 25./48/2
       qcoords_r[3]=3/5 ; qcoords_s[3]=1/5 ; qweights[3]= 25./48/2
    elif nqel==6: #4th order - confirmed
       qcoords_r[0]=0.091576213509771 ; qcoords_s[0]=0.091576213509771 ; qweights[0]=0.109951743655322/2.0 
       qcoords_r[1]=0.816847572980459 ; qcoords_s[1]=0.091576213509771 ; qweights[1]=0.109951743655322/2.0 
       qcoords_r[2]=0.091576213509771 ; qcoords_s[2]=0.816847572980459 ; qweights[2]=0.109951743655322/2.0 
       qcoords_r[3]=0.445948490915965 ; qcoords_s[3]=0.445948490915965 ; qweights[3]=0.223381589678011/2.0 
       qcoords_r[4]=0.108103018168070 ; qcoords_s[4]=0.445948490915965 ; qweights[4]=0.223381589678011/2.0 
       qcoords_r[5]=0.445948490915965 ; qcoords_s[5]=0.108103018168070 ; qweights[5]=0.223381589678011/2.0 
    elif nqel==7: #5th order - confirmed
       qcoords_r[0]=0.1012865073235 ; qcoords_s[0]=0.1012865073235 ; qweights[0]=0.0629695902724 
       qcoords_r[1]=0.7974269853531 ; qcoords_s[1]=0.1012865073235 ; qweights[1]=0.0629695902724 
       qcoords_r[2]=0.1012865073235 ; qcoords_s[2]=0.7974269853531 ; qweights[2]=0.0629695902724 
       qcoords_r[3]=0.4701420641051 ; qcoords_s[3]=0.0597158717898 ; qweights[3]=0.0661970763942 
       qcoords_r[4]=0.4701420641051 ; qcoords_s[4]=0.4701420641051 ; qweights[4]=0.0661970763942 
       qcoords_r[5]=0.0597158717898 ; qcoords_s[5]=0.4701420641051 ; qweights[5]=0.0661970763942 
       qcoords_r[6]=0.3333333333333 ; qcoords_s[6]=0.3333333333333 ; qweights[6]=0.1125000000000 

    area=0
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNNNVdr=dNdr(rq,sq,'P4')
        dNNNVds=dNds(rq,sq,'P4')
        jcb=np.zeros((2,2),dtype=np.float64)
        jcb[0,0]=np.dot(dNNNVdr[:],x[:])
        jcb[0,1]=np.dot(dNNNVdr[:],y[:])
        jcb[1,0]=np.dot(dNNNVds[:],x[:])
        jcb[1,1]=np.dot(dNNNVds[:],y[:])
        jcobq=np.linalg.det(jcb)
        area+=jcobq*weightq

    print('nqel=',nqel,'area=',area)

###############################################################################
###############################################################################
###############################################################################
