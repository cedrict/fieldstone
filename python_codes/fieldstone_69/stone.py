import numpy as np
import time as time

#------------------------------------------------------------------------------

def laypts3(x1,y1,z1,x2,y2,z2,x3,y3,z3,n,x,y,z):
    for ip in range(0,n):
        N1=1-x[ip]-y[ip]
        N2=x[ip]
        N3=y[ip]
        x[ip]=x1*N1+x2*N2+x3*N3
        y[ip]=y1*N1+y2*N2+y3*N3
        z[ip]=z1*N1+z2*N2+z3*N3

def laypts4(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x,y,z,level):
    counter=0
    for j in range(0,level+1):
        for i in range(0,level+1):
            #equidistant
            r=-1.+2./level*i
            s=-1.+2./level*j
            #equiangular
            if equiangular:
               x0=-np.pi/4+i*2*np.pi/4/level
               y0=-np.pi/4+j*2*np.pi/4/level
               r=np.tan(x0)
               s=np.tan(y0)
            #end if
            #print (r,s)
            N1=0.25*(1.-r)*(1.-s)
            N2=0.25*(1.+r)*(1.-s)
            N3=0.25*(1.+r)*(1.+s)
            N4=0.25*(1.-r)*(1.+s)
            x[counter]=x1*N1+x2*N2+x3*N3+x4*N4
            y[counter]=y1*N1+y2*N2+y3*N3+y4*N4
            z[counter]=z1*N1+z2*N2+z3*N3+z4*N4
            counter+=1
        #end for
    #end for

#------------------------------------------------------------------------------

def project_on_sphere(radius,n,x,y,z):
    for ip in range(0,n):
        r=np.sqrt(x[ip]**2+y[ip]**2+z[ip]**2)
        theta=np.arctan2(y[ip],x[ip])
        phi=np.arccos(z[ip]/r)
        x[ip]=radius*np.cos(theta)*np.sin(phi)
        y[ip]=radius*np.sin(theta)*np.sin(phi)
        z[ip]=radius*np.cos(phi)

def project_on_sphere_1pt(radius,x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arctan2(y,x)
    phi=np.arccos(z/r)
    x=radius*np.cos(theta)*np.sin(phi)
    y=radius*np.sin(theta)*np.sin(phi)
    z=radius*np.cos(phi)
    return x,y,z 

#------------------------------------------------------------------------------

nblock=20
level=150
equiangular=True
radius=6371e3
vertical_exaggeration=20
debug=False

print('nblock=',nblock)
print('level=',level)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if nblock==6: 
   block_nel=level**2
   block_np=(level+1)**2
   block_m=4
elif nblock==12: 
   block_nel=level**2
   block_np=(level+1)**2
   block_m=4
elif nblock==20: 
   block_nel=level**2
   block_np=int((level+1)*(level+2)/2)
   block_m=3
else:
   exit("nblock must be 6, 12, or 20")

print('np per block',block_np)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
start = time.time()

block01_x=np.zeros(block_np,dtype=np.float64)
block02_x=np.zeros(block_np,dtype=np.float64)
block03_x=np.zeros(block_np,dtype=np.float64)
block04_x=np.zeros(block_np,dtype=np.float64)
block05_x=np.zeros(block_np,dtype=np.float64)
block06_x=np.zeros(block_np,dtype=np.float64)
if nblock==12 or nblock==20:
   block07_x=np.zeros(block_np,dtype=np.float64)
   block08_x=np.zeros(block_np,dtype=np.float64)
   block09_x=np.zeros(block_np,dtype=np.float64)
   block10_x=np.zeros(block_np,dtype=np.float64)
   block11_x=np.zeros(block_np,dtype=np.float64)
   block12_x=np.zeros(block_np,dtype=np.float64)
if nblock==20:
   block13_x=np.zeros(block_np,dtype=np.float64)
   block14_x=np.zeros(block_np,dtype=np.float64)
   block15_x=np.zeros(block_np,dtype=np.float64)
   block16_x=np.zeros(block_np,dtype=np.float64)
   block17_x=np.zeros(block_np,dtype=np.float64)
   block18_x=np.zeros(block_np,dtype=np.float64)
   block19_x=np.zeros(block_np,dtype=np.float64)
   block20_x=np.zeros(block_np,dtype=np.float64)

block01_y=np.zeros(block_np,dtype=np.float64)
block02_y=np.zeros(block_np,dtype=np.float64)
block03_y=np.zeros(block_np,dtype=np.float64)
block04_y=np.zeros(block_np,dtype=np.float64)
block05_y=np.zeros(block_np,dtype=np.float64)
block06_y=np.zeros(block_np,dtype=np.float64)
if nblock==12 or nblock==20:
   block07_y=np.zeros(block_np,dtype=np.float64)
   block08_y=np.zeros(block_np,dtype=np.float64)
   block09_y=np.zeros(block_np,dtype=np.float64)
   block10_y=np.zeros(block_np,dtype=np.float64)
   block11_y=np.zeros(block_np,dtype=np.float64)
   block12_y=np.zeros(block_np,dtype=np.float64)
if nblock==20:
   block13_y=np.zeros(block_np,dtype=np.float64)
   block14_y=np.zeros(block_np,dtype=np.float64)
   block15_y=np.zeros(block_np,dtype=np.float64)
   block16_y=np.zeros(block_np,dtype=np.float64)
   block17_y=np.zeros(block_np,dtype=np.float64)
   block18_y=np.zeros(block_np,dtype=np.float64)
   block19_y=np.zeros(block_np,dtype=np.float64)
   block20_y=np.zeros(block_np,dtype=np.float64)

block01_z=np.zeros(block_np,dtype=np.float64)
block02_z=np.zeros(block_np,dtype=np.float64)
block03_z=np.zeros(block_np,dtype=np.float64)
block04_z=np.zeros(block_np,dtype=np.float64)
block05_z=np.zeros(block_np,dtype=np.float64)
block06_z=np.zeros(block_np,dtype=np.float64)
if nblock==12 or nblock==20:
   block07_z=np.zeros(block_np,dtype=np.float64)
   block08_z=np.zeros(block_np,dtype=np.float64)
   block09_z=np.zeros(block_np,dtype=np.float64)
   block10_z=np.zeros(block_np,dtype=np.float64)
   block11_z=np.zeros(block_np,dtype=np.float64)
   block12_z=np.zeros(block_np,dtype=np.float64)
if nblock==20:
   block13_z=np.zeros(block_np,dtype=np.float64)
   block14_z=np.zeros(block_np,dtype=np.float64)
   block15_z=np.zeros(block_np,dtype=np.float64)
   block16_z=np.zeros(block_np,dtype=np.float64)
   block17_z=np.zeros(block_np,dtype=np.float64)
   block18_z=np.zeros(block_np,dtype=np.float64)
   block19_z=np.zeros(block_np,dtype=np.float64)
   block20_z=np.zeros(block_np,dtype=np.float64)

block01_icon=np.zeros((block_m,block_nel),dtype=np.int32)
block02_icon=np.zeros((block_m,block_nel),dtype=np.int32)
block03_icon=np.zeros((block_m,block_nel),dtype=np.int32)
block04_icon=np.zeros((block_m,block_nel),dtype=np.int32)
block05_icon=np.zeros((block_m,block_nel),dtype=np.int32)
block06_icon=np.zeros((block_m,block_nel),dtype=np.int32)
if nblock==12 or nblock==20:
   block07_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block08_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block09_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block10_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block11_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block12_icon=np.zeros((block_m,block_nel),dtype=np.int32)
if nblock==20:
   block13_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block14_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block15_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block16_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block17_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block18_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block19_icon=np.zeros((block_m,block_nel),dtype=np.int32)
   block20_icon=np.zeros((block_m,block_nel),dtype=np.int32)

block01_hull=np.zeros(block_np,dtype=np.bool)
block02_hull=np.zeros(block_np,dtype=np.bool)
block03_hull=np.zeros(block_np,dtype=np.bool)
block04_hull=np.zeros(block_np,dtype=np.bool)
block05_hull=np.zeros(block_np,dtype=np.bool)
block06_hull=np.zeros(block_np,dtype=np.bool)
if nblock==12 or nblock==20:
   block07_hull=np.zeros(block_np,dtype=np.bool)
   block08_hull=np.zeros(block_np,dtype=np.bool)
   block09_hull=np.zeros(block_np,dtype=np.bool)
   block10_hull=np.zeros(block_np,dtype=np.bool)
   block11_hull=np.zeros(block_np,dtype=np.bool)
   block12_hull=np.zeros(block_np,dtype=np.bool)
if nblock==20:
   block13_hull=np.zeros(block_np,dtype=np.bool)
   block14_hull=np.zeros(block_np,dtype=np.bool)
   block15_hull=np.zeros(block_np,dtype=np.bool)
   block16_hull=np.zeros(block_np,dtype=np.bool)
   block17_hull=np.zeros(block_np,dtype=np.bool)
   block18_hull=np.zeros(block_np,dtype=np.bool)
   block19_hull=np.zeros(block_np,dtype=np.bool)
   block20_hull=np.zeros(block_np,dtype=np.bool)

print("book memory: %.3f s" % (time.time() - start))

#------------------------------------------------------------------------------
# block_node_layout
#------------------------------------------------------------------------------
start = time.time()

if nblock==6 or nblock==12: #------------------------------
   #build icon
   counter=0
   for j in range(0,level):
       for i in range(0,level):
           block01_icon[0,counter] = i + j * (level + 1)
           block01_icon[1,counter] = i + 1 + j * (level + 1)
           block01_icon[2,counter] = i + 1 + (j + 1) * (level + 1)
           block01_icon[3,counter] = i + (j + 1) * (level + 1)
           counter+=1
       #end for
   #end for
#end if

if nblock==20: #-------------------------------------------
   #build x,y,z
   nplevel=level+1
   counter=0
   for il in range(0,level+1):
      for i in range(0,nplevel):
         block01_x[counter]=float(i)/level
         block01_y[counter]=float(il)/level
         block01_z[counter]=0.
         counter+=1
      #end for
      nplevel=nplevel-1
   #end for

   #build icon
   counter=0
   offset=0
   nplevel=level
   for il in range(0,level):
       for i in range(0,nplevel-1):
           block01_icon[0,counter]= i            +offset
           block01_icon[1,counter]= i+1          +offset
           block01_icon[2,counter]= i+(nplevel+1)+offset
           counter+=1
           block01_icon[0,counter]= i+1            +offset
           block01_icon[1,counter]= i+(nplevel+1)+1+offset
           block01_icon[2,counter]= i+(nplevel+1)  +offset
           counter+=1
       #end for
       block01_icon[0,counter]= nplevel            +offset-1
       block01_icon[1,counter]= nplevel+1          +offset-1
       block01_icon[2,counter]= nplevel+(nplevel+1)+offset-1
       counter=counter+1
       offset=offset+nplevel+1
       nplevel=nplevel-1
   #end for

   #test
   #for iel in range(0,block_nel):
   #    print(block01_icon[:,iel])

#end if

block02_icon[:,:]=block01_icon[:,:]
block03_icon[:,:]=block01_icon[:,:]
block04_icon[:,:]=block01_icon[:,:]
block05_icon[:,:]=block01_icon[:,:]
block06_icon[:,:]=block01_icon[:,:]
if nblock==12 or nblock==20:
   block07_icon[:,:]=block01_icon[:,:]
   block08_icon[:,:]=block01_icon[:,:]
   block09_icon[:,:]=block01_icon[:,:]
   block10_icon[:,:]=block01_icon[:,:]
   block11_icon[:,:]=block01_icon[:,:]
   block12_icon[:,:]=block01_icon[:,:]
if nblock==20:
   block13_icon[:,:]=block01_icon[:,:]
   block14_icon[:,:]=block01_icon[:,:]
   block15_icon[:,:]=block01_icon[:,:]
   block16_icon[:,:]=block01_icon[:,:]
   block17_icon[:,:]=block01_icon[:,:]
   block18_icon[:,:]=block01_icon[:,:]
   block19_icon[:,:]=block01_icon[:,:]
   block20_icon[:,:]=block01_icon[:,:]
   block02_x[:]=block01_x[:] ; block02_y[:]=block01_y[:] ; block02_z[:]=block01_z[:] 
   block03_x[:]=block01_x[:] ; block03_y[:]=block01_y[:] ; block03_z[:]=block01_z[:] 
   block04_x[:]=block01_x[:] ; block04_y[:]=block01_y[:] ; block04_z[:]=block01_z[:] 
   block05_x[:]=block01_x[:] ; block05_y[:]=block01_y[:] ; block05_z[:]=block01_z[:] 
   block06_x[:]=block01_x[:] ; block06_y[:]=block01_y[:] ; block06_z[:]=block01_z[:] 
   block07_x[:]=block01_x[:] ; block07_y[:]=block01_y[:] ; block07_z[:]=block01_z[:] 
   block08_x[:]=block01_x[:] ; block08_y[:]=block01_y[:] ; block08_z[:]=block01_z[:] 
   block09_x[:]=block01_x[:] ; block09_y[:]=block01_y[:] ; block09_z[:]=block01_z[:] 
   block10_x[:]=block01_x[:] ; block10_y[:]=block01_y[:] ; block10_z[:]=block01_z[:] 
   block11_x[:]=block01_x[:] ; block11_y[:]=block01_y[:] ; block11_z[:]=block01_z[:] 
   block12_x[:]=block01_x[:] ; block12_y[:]=block01_y[:] ; block12_z[:]=block01_z[:] 
   block13_x[:]=block01_x[:] ; block13_y[:]=block01_y[:] ; block13_z[:]=block01_z[:] 
   block14_x[:]=block01_x[:] ; block14_y[:]=block01_y[:] ; block14_z[:]=block01_z[:] 
   block15_x[:]=block01_x[:] ; block15_y[:]=block01_y[:] ; block15_z[:]=block01_z[:] 
   block16_x[:]=block01_x[:] ; block16_y[:]=block01_y[:] ; block16_z[:]=block01_z[:] 
   block17_x[:]=block01_x[:] ; block17_y[:]=block01_y[:] ; block17_z[:]=block01_z[:] 
   block18_x[:]=block01_x[:] ; block18_y[:]=block01_y[:] ; block18_z[:]=block01_z[:] 
   block19_x[:]=block01_x[:] ; block19_y[:]=block01_y[:] ; block19_z[:]=block01_z[:] 
   block20_x[:]=block01_x[:] ; block20_y[:]=block01_y[:] ; block20_z[:]=block01_z[:] 

print("block node layout: %.3f s" % (time.time() - start))

#------------------------------------------------------------------------------
# map_blocks
#------------------------------------------------------------------------------
start = time.time()

if nblock==6:

   alpha_min=-np.pi/4.
   alpha_extent=np.pi/2.
   beta_min=-np.pi/4.
   beta_extent=np.pi/2.
   d_alpha=(alpha_extent)/level
   d_beta =(beta_extent)/level

   counter=0    
   for j in range(0,level+1):
       for i in range(0,level+1):
           #equidistant
           X=-1.+2./level*i
           Y=-1.+2./level*j
           #equiangular
           if equiangular:
              alphaa=alpha_min+d_alpha*i
              betaa =beta_min +d_beta *j
              X=np.tan(alphaa)
              Y=np.tan(betaa)
           #end if
           delta=1+X**2+Y**2
           block01_x[counter]= radius/np.sqrt(delta)
           block01_y[counter]= radius/np.sqrt(delta)*X
           block01_z[counter]= radius/np.sqrt(delta)*Y
           block02_x[counter]=-radius/np.sqrt(delta)*X
           block02_y[counter]= radius/np.sqrt(delta)
           block02_z[counter]= radius/np.sqrt(delta)*Y
           block05_x[counter]=-radius/np.sqrt(delta)*Y
           block05_y[counter]= radius/np.sqrt(delta)*X
           block05_z[counter]= radius/np.sqrt(delta)
           if i==0 or i==level or j==0 or j==level:
              block01_hull[counter]=True
              block02_hull[counter]=True
              block03_hull[counter]=True
              block04_hull[counter]=True
              block05_hull[counter]=True
              block06_hull[counter]=True
           #end if
           counter=counter+1   
       #end for
   #end for

   # block3 is rotation of block 1
   # by 180 degrees around axis Z
   block03_x=-block01_x
   block03_y=-block01_y
   block03_z= block01_z

   # block4 is rotation of block 2
   # by 180 degrees around axis Z
   block04_x=-block02_x
   block04_y=-block02_y
   block04_z= block02_z

   # block6 is rotation of block 5
   # by 180 degrees around axis Y
   block06_x=-block05_x
   block06_y= block05_y
   block06_z=-block05_z

if nblock==12: #-------------------------------------------

   counter=0    
   for j in range(0,level+1):
       for i in range(0,level+1):
           if i==0 or i==level or j==0 or j==level:
              block01_hull[counter]=True
              block02_hull[counter]=True
              block03_hull[counter]=True
              block04_hull[counter]=True
              block05_hull[counter]=True
              block06_hull[counter]=True
              block07_hull[counter]=True
              block08_hull[counter]=True
              block09_hull[counter]=True
              block10_hull[counter]=True
              block11_hull[counter]=True
              block12_hull[counter]=True
           #end if
           counter+=1
       #end for
   #end for


   #----------------------
   # four corners

   xA=-1.
   yA= 0.
   zA=-1./np.sqrt(2.)

   xB=+1.
   yB= 0. 
   zB=-1./np.sqrt(2.)

   xC= 0. 
   yC=-1.
   zC= 1./np.sqrt(2.)

   xD= 0. 
   yD=+1.
   zD= 1./np.sqrt(2.)

   #----------------------
   # middles of faces

   xM=(xA+xB+xC)/3.
   yM=(yA+yB+yC)/3.
   zM=(zA+zB+zC)/3.

   xN=(xA+xD+xC)/3.
   yN=(yA+yD+yC)/3.
   zN=(zA+zD+zC)/3.

   xP=(xA+xD+xB)/3.
   yP=(yA+yD+yB)/3.
   zP=(zA+zD+zB)/3.

   xQ=(xC+xD+xB)/3.
   yQ=(yC+yD+yB)/3.
   zQ=(zC+zD+zB)/3.

   #----------------------
   # middle of edges

   xF=(xB+xC)/2.
   yF=(yB+yC)/2.
   zF=(zB+zC)/2.

   xG=(xA+xC)/2.
   yG=(yA+yC)/2.
   zG=(zA+zC)/2.

   xE=(xB+xA)/2.
   yE=(yB+yA)/2.
   zE=(zB+zA)/2.

   xH=(xD+xC)/2.
   yH=(yD+yC)/2.
   zH=(zD+zC)/2.

   xJ=(xD+xA)/2.
   yJ=(yD+yA)/2.
   zJ=(zD+zA)/2.

   xK=(xD+xB)/2.
   yK=(yD+yB)/2.
   zK=(zD+zB)/2.

   xA,yA,zA=project_on_sphere_1pt(radius,xA,yA,zA)
   xB,yB,zB=project_on_sphere_1pt(radius,xB,yB,zB)
   xC,yC,zC=project_on_sphere_1pt(radius,xC,yC,zC)
   xD,yD,zD=project_on_sphere_1pt(radius,xD,yD,zD)
   xE,yE,zE=project_on_sphere_1pt(radius,xE,yE,zE)
   xF,yF,zF=project_on_sphere_1pt(radius,xF,yF,zF)
   xG,yG,zG=project_on_sphere_1pt(radius,xG,yG,zG)
   xH,yH,zH=project_on_sphere_1pt(radius,xH,yH,zH)
   xJ,yJ,zJ=project_on_sphere_1pt(radius,xJ,yJ,zJ)
   xK,yK,zK=project_on_sphere_1pt(radius,xK,yK,zK)
   xM,yM,zM=project_on_sphere_1pt(radius,xM,yM,zM)
   xN,yN,zN=project_on_sphere_1pt(radius,xN,yN,zN)
   xP,yP,zP=project_on_sphere_1pt(radius,xP,yP,zP)
   xQ,yQ,zQ=project_on_sphere_1pt(radius,xQ,yQ,zQ)

   laypts4(xM,yM,zM,xG,yG,zG,xA,yA,zA,xE,yE,zE,block01_x,block01_y,block01_z,level) #01 MGAE
   laypts4(xF,yF,zF,xM,yM,zM,xE,yE,zE,xB,yB,zB,block02_x,block02_y,block02_z,level) #02 MEBF 
   laypts4(xC,yC,zC,xG,yG,zG,xM,yM,zM,xF,yF,zF,block03_x,block03_y,block03_z,level) #03 MFCG
   laypts4(xG,yG,zG,xN,yN,zN,xJ,yJ,zJ,xA,yA,zA,block04_x,block04_y,block04_z,level) #04 NJAG
   laypts4(xC,yC,zC,xH,yH,zH,xN,yN,zN,xG,yG,zG,block05_x,block05_y,block05_z,level) #05 NGCH
   laypts4(xH,yH,zH,xD,yD,zD,xJ,yJ,zJ,xN,yN,zN,block06_x,block06_y,block06_z,level) #06 NHDJ
   laypts4(xA,yA,zA,xJ,yJ,zJ,xP,yP,zP,xE,yE,zE,block07_x,block07_y,block07_z,level) #07 PEAJ
   laypts4(xJ,yJ,zJ,xD,yD,zD,xK,yK,zK,xP,yP,zP,block08_x,block08_y,block08_z,level) #08 PJDK
   laypts4(xP,yP,zP,xK,yK,zK,xB,yB,zB,xE,yE,zE,block09_x,block09_y,block09_z,level) #09 PKBE
   laypts4(xQ,yQ,zQ,xK,yK,zK,xD,yD,zD,xH,yH,zH,block10_x,block10_y,block10_z,level) #10 QKDH
   laypts4(xQ,yQ,zQ,xH,yH,zH,xC,yC,zC,xF,yF,zF,block11_x,block11_y,block11_z,level) #11 QHCF
   laypts4(xQ,yQ,zQ,xF,yF,zF,xB,yB,zB,xK,yK,zK,block12_x,block12_y,block12_z,level) #12 QFBK

   project_on_sphere(radius,block_np,block01_x,block01_y,block01_z)
   project_on_sphere(radius,block_np,block02_x,block02_y,block02_z)
   project_on_sphere(radius,block_np,block03_x,block03_y,block03_z)
   project_on_sphere(radius,block_np,block04_x,block04_y,block04_z)
   project_on_sphere(radius,block_np,block05_x,block05_y,block05_z)
   project_on_sphere(radius,block_np,block06_x,block06_y,block06_z)
   project_on_sphere(radius,block_np,block07_x,block07_y,block07_z)
   project_on_sphere(radius,block_np,block08_x,block08_y,block08_z)
   project_on_sphere(radius,block_np,block09_x,block09_y,block09_z)
   project_on_sphere(radius,block_np,block10_x,block10_y,block10_z)
   project_on_sphere(radius,block_np,block11_x,block11_y,block11_z)
   project_on_sphere(radius,block_np,block12_x,block12_y,block12_z)

if nblock==20: #--------------------------------------------

   for i in range(0,block_np):
       if block01_x[i]<1e-6:
          block01_hull[i]=True
       if block01_y[i]<1e-6:
          block01_hull[i]=True
       if abs(1-block01_x[i]-block01_y[i])<1e-6:
          block01_hull[i]=True
   #end for
   block02_hull=block01_hull
   block03_hull=block01_hull
   block04_hull=block01_hull
   block05_hull=block01_hull
   block06_hull=block01_hull
   block07_hull=block01_hull
   block08_hull=block01_hull
   block09_hull=block01_hull
   block10_hull=block01_hull
   block11_hull=block01_hull
   block12_hull=block01_hull
   block13_hull=block01_hull
   block14_hull=block01_hull
   block15_hull=block01_hull
   block16_hull=block01_hull
   block17_hull=block01_hull
   block18_hull=block01_hull
   block19_hull=block01_hull
   block20_hull=block01_hull

   t=(1+np.sqrt(5.))/2.
   tt=1./np.sqrt(1+t**2)

   xA=+t*tt ; yA=+1*tt ; zA=0     ; thetaA=np.arccos(zA) ; phiA=np.arctan2(yA,xA)
   xB=-t*tt ; yB=+1*tt ; zB=0     ; thetaB=np.arccos(zB) ; phiB=np.arctan2(yB,xB)
   xC=+t*tt ; yC=-1*tt ; zC=0     ; thetaC=np.arccos(zC) ; phiC=np.arctan2(yC,xC)
   xD=-t*tt ; yD=-1*tt ; zD=0     ; thetaD=np.arccos(zD) ; phiD=np.arctan2(yD,xD)
   xE=+1*tt ; yE=0     ; zE=+t*tt ; thetaE=np.arccos(zE) ; phiE=np.arctan2(yE,xE)
   xF=+1*tt ; yF=0     ; zF=-t*tt ; thetaF=np.arccos(zF) ; phiF=np.arctan2(yF,xF)
   xG=-1*tt ; yG=0     ; zG=+t*tt ; thetaG=np.arccos(zG) ; phiG=np.arctan2(yG,xG)
   xH=-1*tt ; yH=0     ; zH=-t*tt ; thetaH=np.arccos(zH) ; phiH=np.arctan2(yH,xH)
   xI=0     ; yI=+t*tt ; zI=+1*tt ; thetaI=np.arccos(zI) ; phiI=np.arctan2(yI,xI)
   xJ=0     ; yJ=-t*tt ; zJ=+1*tt ; thetaJ=np.arccos(zJ) ; phiJ=np.arctan2(yJ,xJ)
   xK=0     ; yK=+t*tt ; zK=-1*tt ; thetaK=np.arccos(zK) ; phiK=np.arctan2(yK,xK)
   xL=0     ; yL=-t*tt ; zL=-1*tt ; thetaL=np.arccos(zL) ; phiL=np.arctan2(yL,xL)

   #A B C D E F G H I J K  L 
   #0 1 2 3 4 5 6 7 8 9 10 11

   # T00 0-8-4  : AIE
   # T01 0-5-10 : AFK
   # T02 2-4-9  : CEJ
   # T03 2-11-5 : CLF
   # T04 1-6-8  : BGI
   # T05 1-10-7 : BKH
   # T06 3-9-6  : DJG
   # T07 3-7-11 : DHL
   # T08 0-10-8 : AKI
   # T09 1-8-10 : BIK
   # T10 2-9-11 : CJL
   # T11 3-11-9 : DLJ
   # T12 4-2-0  : ECA
   # T13 5-0-2  : FAC
   # T14 6-1-3  : GBD
   # T15 7-3-1  : HDB
   # T16 8-6-4  : IGE
   # T17 9-4-6  : JEG
   # T18 10-5-7 : KFH
   # T19 11-7-5 : LHF

   laypts3(xA,yA,zA,xI,yI,zI,xE,yE,zE,block_np,block01_x,block01_y,block01_z) #T00
   laypts3(xA,yA,zA,xF,yF,zF,xK,yK,zK,block_np,block02_x,block02_y,block02_z) #T01
   laypts3(xC,yC,zC,xE,yE,zE,xJ,yJ,zJ,block_np,block03_x,block03_y,block03_z) #T02
   laypts3(xC,yC,zC,xL,yL,zL,xF,yF,zF,block_np,block04_x,block04_y,block04_z) #T03
   laypts3(xB,yB,zB,xG,yG,zG,xI,yI,zI,block_np,block05_x,block05_y,block05_z) #T04
   laypts3(xB,yB,zB,xK,yK,zK,xH,yH,zH,block_np,block06_x,block06_y,block06_z) #T05
   laypts3(xD,yD,zD,xJ,yJ,zJ,xG,yG,zG,block_np,block07_x,block07_y,block07_z) #T06
   laypts3(xD,yD,zD,xH,yH,zH,xL,yL,zL,block_np,block08_x,block08_y,block08_z) #T07
   laypts3(xA,yA,zA,xK,yK,zK,xI,yI,zI,block_np,block09_x,block09_y,block09_z) #T08
   laypts3(xB,yB,zB,xI,yI,zI,xK,yK,zK,block_np,block10_x,block10_y,block10_z) #T09
   laypts3(xC,yC,zC,xJ,yJ,zJ,xL,yL,zL,block_np,block11_x,block11_y,block11_z) #T10
   laypts3(xD,yD,zD,xL,yL,zL,xJ,yJ,zJ,block_np,block12_x,block12_y,block12_z) #T11
   laypts3(xE,yE,zE,xC,yC,zC,xA,yA,zA,block_np,block13_x,block13_y,block13_z) #T12
   laypts3(xF,yF,zF,xA,yA,zA,xC,yC,zC,block_np,block14_x,block14_y,block14_z) #T13
   laypts3(xG,yG,zG,xB,yB,zB,xD,yD,zD,block_np,block15_x,block15_y,block15_z) #T14
   laypts3(xH,yH,zH,xD,yD,zD,xB,yB,zB,block_np,block16_x,block16_y,block16_z) #T15
   laypts3(xI,yI,zI,xG,yG,zG,xE,yE,zE,block_np,block17_x,block17_y,block17_z) #T16
   laypts3(xJ,yJ,zJ,xE,yE,zE,xG,yG,zG,block_np,block18_x,block18_y,block18_z) #T17
   laypts3(xK,yK,zK,xF,yF,zF,xH,yH,zH,block_np,block19_x,block19_y,block19_z) #T18
   laypts3(xL,yL,zL,xH,yH,zH,xF,yF,zF,block_np,block20_x,block20_y,block20_z) #T19

   project_on_sphere(radius,block_np,block01_x,block01_y,block01_z)
   project_on_sphere(radius,block_np,block02_x,block02_y,block02_z)
   project_on_sphere(radius,block_np,block03_x,block03_y,block03_z)
   project_on_sphere(radius,block_np,block04_x,block04_y,block04_z)
   project_on_sphere(radius,block_np,block05_x,block05_y,block05_z)
   project_on_sphere(radius,block_np,block06_x,block06_y,block06_z)
   project_on_sphere(radius,block_np,block07_x,block07_y,block07_z)
   project_on_sphere(radius,block_np,block08_x,block08_y,block08_z)
   project_on_sphere(radius,block_np,block09_x,block09_y,block09_z)
   project_on_sphere(radius,block_np,block10_x,block10_y,block10_z)
   project_on_sphere(radius,block_np,block11_x,block11_y,block11_z)
   project_on_sphere(radius,block_np,block12_x,block12_y,block12_z)
   project_on_sphere(radius,block_np,block13_x,block13_y,block13_z)
   project_on_sphere(radius,block_np,block14_x,block14_y,block14_z)
   project_on_sphere(radius,block_np,block15_x,block15_y,block15_z)
   project_on_sphere(radius,block_np,block16_x,block16_y,block16_z)
   project_on_sphere(radius,block_np,block17_x,block17_y,block17_z)
   project_on_sphere(radius,block_np,block18_x,block18_y,block18_z)
   project_on_sphere(radius,block_np,block19_x,block19_y,block19_z)
   project_on_sphere(radius,block_np,block20_x,block20_y,block20_z)

#end if nblock

if debug:
   np.savetxt('block01.ascii',np.array([block01_x,block01_y,block01_z,block01_hull]).T)
   np.savetxt('block02.ascii',np.array([block02_x,block02_y,block02_z]).T)
   np.savetxt('block03.ascii',np.array([block03_x,block03_y,block03_z]).T)
   np.savetxt('block04.ascii',np.array([block04_x,block04_y,block04_z]).T)
   np.savetxt('block05.ascii',np.array([block05_x,block05_y,block05_z]).T)
   np.savetxt('block06.ascii',np.array([block06_x,block06_y,block06_z]).T)
   if nblock==12 or nblock==20:
      np.savetxt('block07.ascii',np.array([block07_x,block07_y,block07_z]).T)
      np.savetxt('block08.ascii',np.array([block08_x,block08_y,block08_z]).T)
      np.savetxt('block09.ascii',np.array([block09_x,block09_y,block09_z]).T)
      np.savetxt('block10.ascii',np.array([block10_x,block10_y,block10_z]).T)
      np.savetxt('block11.ascii',np.array([block11_x,block11_y,block11_z]).T)
      np.savetxt('block12.ascii',np.array([block12_x,block12_y,block12_z]).T)
   if nblock==20:
      np.savetxt('block13.ascii',np.array([block13_x,block13_y,block13_z]).T)
      np.savetxt('block14.ascii',np.array([block14_x,block14_y,block14_z]).T)
      np.savetxt('block15.ascii',np.array([block15_x,block15_y,block15_z]).T)
      np.savetxt('block16.ascii',np.array([block16_x,block16_y,block16_z]).T)
      np.savetxt('block17.ascii',np.array([block17_x,block17_y,block17_z]).T)
      np.savetxt('block18.ascii',np.array([block18_x,block18_y,block18_z]).T)
      np.savetxt('block19.ascii',np.array([block19_x,block19_y,block19_z]).T)
      np.savetxt('block20.ascii',np.array([block20_x,block20_y,block20_z]).T)

print("map blocks: %.3f s" % (time.time() - start))

#------------------------------------------------------------------------------
# merge blocks
#------------------------------------------------------------------------------
start = time.time()

nnp=block_np
nnel=block_nel

tempx=np.zeros(nblock*nnp,dtype=np.float64)
tempy=np.zeros(nblock*nnp,dtype=np.float64)
tempz=np.zeros(nblock*nnp,dtype=np.float64)
sides=np.zeros(nblock*nnp,dtype=np.bool)

ib=1
tempx[(ib-1)*nnp:ib*nnp]=block01_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block01_y[:]
tempz[(ib-1)*nnp:ib*nnp]=block01_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block01_hull[:]
ib=2
tempx[(ib-1)*nnp:ib*nnp]=block02_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block02_y[:]
tempz[(ib-1)*nnp:ib*nnp]=block02_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block02_hull[:]
ib=3
tempx[(ib-1)*nnp:ib*nnp]=block03_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block03_y[:]
tempz[(ib-1)*nnp:ib*nnp]=block03_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block03_hull[:]
ib=4
tempx[(ib-1)*nnp:ib*nnp]=block04_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block04_y[:]
tempz[(ib-1)*nnp:ib*nnp]=block04_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block04_hull[:]
ib=5
tempx[(ib-1)*nnp:ib*nnp]=block05_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block05_y[:]
tempz[(ib-1)*nnp:ib*nnp]=block05_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block05_hull[:]
ib=6
tempx[(ib-1)*nnp:ib*nnp]=block06_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block06_y[:]
tempz[(ib-1)*nnp:ib*nnp]=block06_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block06_hull[:]
if nblock==12 or nblock==20:
   ib=7
   tempx[(ib-1)*nnp:ib*nnp]=block07_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block07_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block07_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block07_hull[:]
   ib=8
   tempx[(ib-1)*nnp:ib*nnp]=block08_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block08_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block08_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block08_hull[:]
   ib=9
   tempx[(ib-1)*nnp:ib*nnp]=block09_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block09_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block09_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block09_hull[:]
   ib=10
   tempx[(ib-1)*nnp:ib*nnp]=block10_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block10_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block10_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block10_hull[:]
   ib=11
   tempx[(ib-1)*nnp:ib*nnp]=block11_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block11_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block11_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block11_hull[:]
   ib=12
   tempx[(ib-1)*nnp:ib*nnp]=block12_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block12_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block12_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block12_hull[:]
if nblock==20:
   ib=13
   tempx[(ib-1)*nnp:ib*nnp]=block13_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block13_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block13_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block13_hull[:]
   ib=14
   tempx[(ib-1)*nnp:ib*nnp]=block14_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block14_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block14_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block14_hull[:]
   ib=15
   tempx[(ib-1)*nnp:ib*nnp]=block15_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block15_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block15_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block15_hull[:]
   ib=16
   tempx[(ib-1)*nnp:ib*nnp]=block16_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block16_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block16_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block16_hull[:]
   ib=17
   tempx[(ib-1)*nnp:ib*nnp]=block17_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block17_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block17_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block17_hull[:]
   ib=18
   tempx[(ib-1)*nnp:ib*nnp]=block18_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block18_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block18_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block18_hull[:]
   ib=19
   tempx[(ib-1)*nnp:ib*nnp]=block19_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block19_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block19_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block19_hull[:]
   ib=20
   tempx[(ib-1)*nnp:ib*nnp]=block20_x[:] ; tempy[(ib-1)*nnp:ib*nnp]=block20_y[:]
   tempz[(ib-1)*nnp:ib*nnp]=block20_z[:] ; sides[(ib-1)*nnp:ib*nnp]=block20_hull[:]

if debug:
   np.savetxt('temp.ascii',np.array([tempx,tempy,tempz,sides]).T)

doubble=np.zeros(nblock*nnp,dtype=np.bool)
pointto=np.zeros(nblock*nnp,dtype=np.int32)

for i in range(0,nblock*nnp):
    pointto[i]=i

distance=1e-6*radius

counter=0
for ip in range(1,nblock*nnp):
    if sides[ip]:
       gxip=tempx[ip]
       gyip=tempy[ip]
       gzip=tempz[ip]
       for jp in range(0,ip-1):
           if sides[jp]:
              if np.abs(gxip-tempx[jp])<distance and \
                 np.abs(gyip-tempy[jp])<distance and \
                 np.abs(gzip-tempz[jp])<distance : 
                 doubble[ip]=True
                 pointto[ip]=jp
                 break
              #end if
           #end if
       #end do
    #end if
#end for

shell_np=nblock*nnp-np.count_nonzero(doubble)

shell_nel=nblock*block_nel

print('count(doubble)=',np.count_nonzero(doubble))
print('shell_np',shell_np)
print('shell_nel',shell_nel)

shell_m=block_m

shell_x=np.zeros(shell_np,dtype=np.float64)
shell_y=np.zeros(shell_np,dtype=np.float64)
shell_z=np.zeros(shell_np,dtype=np.float64)
shell_icon=np.zeros((shell_m,shell_nel),dtype=np.int32)

counter=0
for ip in range(0,nblock*nnp):
    if not doubble[ip]: 
       shell_x[counter]=tempx[ip]
       shell_y[counter]=tempy[ip]
       shell_z[counter]=tempz[ip]
       counter+=1
    #end if
#end for

if debug:
   np.savetxt('shell_xyz.ascii',np.array([shell_x,shell_y,shell_z]).T)

ib=1 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block01_icon[:,:]+(ib-1)*block_np
ib=2 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block02_icon[:,:]+(ib-1)*block_np
ib=3 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block03_icon[:,:]+(ib-1)*block_np
ib=4 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block04_icon[:,:]+(ib-1)*block_np
ib=5 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block05_icon[:,:]+(ib-1)*block_np
ib=6 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block06_icon[:,:]+(ib-1)*block_np
if nblock==12 or nblock==20:
   ib=7  ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block07_icon[:,:]+(ib-1)*block_np
   ib=8  ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block08_icon[:,:]+(ib-1)*block_np
   ib=9  ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block09_icon[:,:]+(ib-1)*block_np
   ib=10 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block10_icon[:,:]+(ib-1)*block_np
   ib=11 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block11_icon[:,:]+(ib-1)*block_np
   ib=12 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block12_icon[:,:]+(ib-1)*block_np
if nblock==20:
   ib=13 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block13_icon[:,:]+(ib-1)*block_np
   ib=14 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block14_icon[:,:]+(ib-1)*block_np
   ib=15 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block15_icon[:,:]+(ib-1)*block_np
   ib=16 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block16_icon[:,:]+(ib-1)*block_np
   ib=17 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block17_icon[:,:]+(ib-1)*block_np
   ib=18 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block18_icon[:,:]+(ib-1)*block_np
   ib=19 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block19_icon[:,:]+(ib-1)*block_np
   ib=20 ; shell_icon[0:block_m,(ib-1)*block_nel:ib*block_nel]=block20_icon[:,:]+(ib-1)*block_np


for iel in range(0,shell_nel):
    for i in range(0,shell_m):
        shell_icon[i,iel]=pointto[shell_icon[i,iel]]
    #end for
#end for

compact=np.zeros(nblock*block_np,dtype=np.int32)

counter=0
for ip in range(0,nblock*block_np):
    if not doubble[ip]: 
       compact[ip]=counter
       counter+=1
    #end if
#end for

for iel in range(0,shell_nel):
    for i in range(0,shell_m):
        shell_icon[i,iel]=compact[shell_icon[i,iel]]
    #end for
#end for

print("merge blocks: %.3f s" % (time.time() - start))


#------------------------------------------------------------------------------
# compute r,theta,phi

shell_r=np.zeros(shell_np,dtype=np.float64)
shell_theta=np.zeros(shell_np,dtype=np.float64)
shell_phi=np.zeros(shell_np,dtype=np.float64)

for i in range(0,shell_np):
    shell_r[i]=np.sqrt(shell_x[i]**2+shell_y[i]**2+shell_z[i]**2)
    shell_theta[i]=np.arccos(shell_z[i]/shell_r[i])
    shell_phi[i]=np.arctan2(shell_y[i],shell_x[i])

#------------------------------------------------------------------------------
# read in topography and displace nodes

nlon=1440
nlat=720
nfac=30

nel=(nlon-1)*(nlat-1)

rlon=np.zeros(nlon,dtype=np.float64)
rlat=np.zeros(nlat,dtype=np.float64)
height=np.zeros((nlon,nlat),dtype=np.float64)

filename='./topo/dem030_ascii.dat'
f = open(filename,'r')
i=0
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    l=len(columns)
    height[0+i*11:l+i*11,counter]=columns[0:l]
    i+=1
    if l==10:
       i=0
       counter+=1
#end for

for ilon in range(0,nlon):
    rlon[ilon]=-180.+float(2*ilon-1)*float(nfac)/240.

for ilat in range(0,nlat):
    rlat[ilat]=90.-float(2*ilat-1)*float(nfac)/240.

for i in range(0,shell_np):
    lon=shell_phi[i]           ; lon=lon/np.pi*180.
    lat=np.pi/2-shell_theta[i] ; lat=lat/np.pi*180.

    #print(lon,lat)

    for ilon in range(0,nlon):
        if abs(lon-rlon[ilon])<=float(nfac)/240.:
           break
    for ilat in range(0,nlat):
        if abs(lat-rlat[ilat])<=float(nfac)/240.:
           break

    if height[ilon,ilat]>0: 
       shell_r[i]+=height[ilon,ilat]*vertical_exaggeration
 
    shell_x[i]=shell_r[i]*np.cos(shell_phi[i])*np.sin(shell_theta[i])
    shell_y[i]=shell_r[i]*np.sin(shell_phi[i])*np.sin(shell_theta[i])
    shell_z[i]=shell_r[i]*np.cos(shell_theta[i])

#end for






#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
start = time.time()

vtufile=open('shell.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(shell_np,shell_nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,shell_np):
    vtufile.write("%10e %10e %10e \n" %(shell_x[i],shell_y[i],shell_z[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####

vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
for i in range(0,shell_np):
    vtufile.write("%10e \n" %shell_r[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
for i in range(0,shell_np):
    vtufile.write("%10e \n" %shell_theta[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='phi' Format='ascii'> \n")
for i in range(0,shell_np):
    vtufile.write("%10e \n" %shell_phi[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='topo' Format='ascii'> \n")
for i in range(0,shell_np):
    vtufile.write("%10e \n" %(shell_r[i]-radius))
vtufile.write("</DataArray>\n")


vtufile.write("</PointData>\n")

#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,shell_nel):
    if shell_m==3:
       vtufile.write("%d %d %d \n" %(shell_icon[0,iel],shell_icon[1,iel],shell_icon[2,iel]))
    if shell_m==4:
       vtufile.write("%d %d %d %d \n" %(shell_icon[0,iel],shell_icon[1,iel],shell_icon[2,iel],shell_icon[3,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,shell_nel):
    if shell_m==3:
       vtufile.write("%d \n" %((iel+1)*3))
    if shell_m==4:
       vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,shell_nel):
    if shell_m==3:
       vtufile.write("%d \n" %5)
    if shell_m==4:
       vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export to vtu: %.3f s" % (time.time() - start))
