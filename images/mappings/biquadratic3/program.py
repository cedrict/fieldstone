import random
import numpy as np
import matplotlib.pyplot as plt
import sys

###############################################################################
# function and its spatial derivatives

def fct(x,y):
    return x**3+y**3

def dfctdx(x,y):
    return 3*x**2

def dfctdy(x,y):
    return 3*y**2

###############################################################################
###############################################################################

npts=500
element=4

if int(len(sys.argv) == 3): 
   xi = int(sys.argv[1])
   nqperdim = int(sys.argv[2])
   print('read xi=',xi)
   print('read nqperdim=',nqperdim)
else:
   xi = 0.1
   nqperdim=3

###############################################################################
###############################################################################

if nqperdim==2:
   coords=[-1/np.sqrt(3),1/np.sqrt(3)]
   weights=[1,1]

if nqperdim==3:
   coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   weights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   coords=[-qc4a,-qc4b,qc4b,qc4a]
   weights=[qw4a,qw4b,qw4b,qw4a]

if nqperdim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   coords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   weights=[qw5a,qw5b,qw5c,qw5b,qw5a]

if nqperdim==6:
   coords=[-0.932469514203152,\
           -0.661209386466265,\
           -0.238619186083197,\
           +0.238619186083197,\
           +0.661209386466265,\
           +0.932469514203152]
   weights=[0.171324492379170,\
            0.360761573048139,\
            0.467913934572691,\
            0.467913934572691,\
            0.360761573048139,\
            0.171324492379170]

if nqperdim==7:
   coords=[-0.949107912342759,\
           -0.741531185599394,\
           -0.405845151377397,\
            0.000000000000000,\
            0.405845151377397,\
            0.741531185599394,\
            0.949107912342759]
   weights=[0.129484966168870,\
            0.279705391489277,\
            0.381830050505119,\
            0.417959183673469,\
            0.381830050505119,\
            0.279705391489277,\
            0.129484966168870]

if nqperdim==10:
   coords=[-0.973906528517172,\
            -0.865063366688985,\
            -0.679409568299024,\
            -0.433395394129247,\
            -0.148874338981631,\
             0.148874338981631,\
             0.433395394129247,\
             0.679409568299024,\
             0.865063366688985,\
             0.973906528517172]
   weights=[0.066671344308688,\
             0.149451349150581,\
             0.219086362515982,\
             0.269266719309996,\
             0.295524224714753,\
             0.295524224714753,\
             0.269266719309996,\
             0.219086362515982,\
             0.149451349150581,\
             0.066671344308688]

nqel=nqperdim**2
qcoords_r=np.empty(nqel,dtype=np.float64)
qcoords_s=np.empty(nqel,dtype=np.float64)
qweights=np.empty(nqel,dtype=np.float64)

counterq=0
for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):
        qcoords_r[counterq]=coords[iq]
        qcoords_s[counterq]=coords[jq]
        qweights[counterq]=weights[iq]*weights[jq]
        counterq+=1
    #end for
#end for

###############################################################################
# Q2 basis functions

def NNV(rq,sq):
    N_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    N_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    N_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    N_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    N_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    N_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    N_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    N_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    N_8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

def dNNVdr(rq,sq):
    dNdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNdr_8=       (-2.*rq) *    (1.-sq**2)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,\
                     dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

def dNNVds(rq,sq):
    dNds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNds_8=     (1.-rq**2) *       (-2.*sq)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,\
                     dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)

###############################################################################

if element==0: # reference element [-1,1]x[-1,1]

   x1=-1              ; y1=-1
   x2=1               ; y2=-1
   x3=1               ; y3=1
   x4=-1              ; y4=1
   x5=0.5*(x1+x2)     ; y5=0.5*(y1+y2)
   x6=0.5*(x2+x3)     ; y6=0.5*(y2+y3)
   x7=0.5*(x3+x4)     ; y7=0.5*(y3+y4)
   x8=0.5*(x1+x4)     ; y8=0.5*(y1+y4)

   ncenter=3

if element==1: # quadrilateral element 

   x1=-1              ; y1=-2
   x2=3               ; y2=-1
   x3=2               ; y3=2
   x4=-3              ; y4=1
   x5=0.5*(x1+x2)     ; y5=0.5*(y1+y2)
   x6=0.5*(x2+x3)     ; y6=0.5*(y2+y3)
   x7=0.5*(x3+x4)     ; y7=0.5*(y3+y4)
   x8=0.5*(x1+x4)     ; y8=0.5*(y1+y4)

   ncenter=3

if element==2: # quadrilateral element with 2 curved edges 

   x1=-1 ; y1=-2
   x2=3  ; y2=-1
   x3=2  ; y3=2
   x4=-3 ; y4=1
   x5=0.5*(x1+x2) ; y5=0.5*(y1+y2) -0.25
   x6=0.5*(x2+x3) ; y6=0.5*(y2+y3)
   x7=0.5*(x3+x4) ; y7=0.5*(y3+y4) +0.5
   x8=0.5*(x1+x4) ; y8=0.5*(y1+y4)

   ncenter=3

if element==3: # quadrilateral element with 4 curved edges

   x1=-1 ; y1=-2
   x2=3  ; y2=-1
   x3=2  ; y3=2
   x4=-3 ; y4=1
   x5=0.5*(x1+x2)     ; y5=0.5*(y1+y2) -0.5
   x6=0.5*(x2+x3)+0.4 ; y6=0.5*(y2+y3) +0.1
   x7=0.5*(x3+x4)     ; y7=0.5*(y3+y4) +0.5
   x8=0.5*(x1+x4)-0.2 ; y8=0.5*(y1+y4) -0.1

   ncenter=3

if element==4: # annulus element

   R1=1
   R2=2
   dtheta=np.pi/4
   x1=R1*np.cos(np.pi/2-0)               ; y1=R1*np.sin(np.pi/2-0) 
   x5=R1*np.cos(np.pi/2-dtheta/2)        ; y5=R1*np.sin(np.pi/2-dtheta/2) 
   x2=R1*np.cos(np.pi/2-dtheta)          ; y2=R1*np.sin(np.pi/2-dtheta) 
   x8=0.5*(R1+R2)*np.cos(np.pi/2-0)      ; y8=0.5*(R1+R2)*np.sin(np.pi/2-0)  
   x6=0.5*(R1+R2)*np.cos(np.pi/2-dtheta) ; y6=0.5*(R1+R2)*np.sin(np.pi/2-dtheta) 
   x4=R2*np.cos(np.pi/2-0)               ; y4=R2*np.sin(np.pi/2-0)   
   x7=R2*np.cos(np.pi/2-dtheta/2)        ; y7=R2*np.sin(np.pi/2-dtheta/2)   
   x3=R2*np.cos(np.pi/2-dtheta)          ; y3=R2*np.sin(np.pi/2-dtheta)   

   ncenter=6

   area_th=0.5*(R2**2-R1**2)*dtheta
   xcg=(R2**3-R1**2)/3*(np.sin(np.pi/2)-np.sin(np.pi/4))/area_th
   ycg=(R2**3-R1**2)/3*(-np.cos(np.pi/2)+np.cos(np.pi/4))/area_th
   print('center of mass coords (x,y):',xcg,ycg)
   print('center of mass coords (r,theta)',np.sqrt(xcg**2+ycg**2),np.arctan2(ycg,xcg)/np.pi*180)



vtufile=open('element'+str(element)+'.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8,1))

vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
vtufile.write("%e %e %e \n" %(x1,y1,0.))
vtufile.write("%e %e %e \n" %(x2,y2,0.))
vtufile.write("%e %e %e \n" %(x3,y3,0.))
vtufile.write("%e %e %e \n" %(x4,y4,0.))
vtufile.write("%e %e %e \n" %(x5,y5,0.))
vtufile.write("%e %e %e \n" %(x6,y6,0.))
vtufile.write("%e %e %e \n" %(x7,y7,0.))
vtufile.write("%e %e %e \n" %(x8,y8,0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")

vtufile.write("<Cells>\n")
#--  
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
vtufile.write("%d %d %d %d %d %d %d %d\n" %(0, 1, 2 , 3 , 4 , 5 , 6 , 7))
vtufile.write("</DataArray>\n")
#--  
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
vtufile.write("%d \n" %(8))
vtufile.write("</DataArray>\n")
#--  
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
vtufile.write("%d \n" % 23) 
vtufile.write("</DataArray>\n")
#--  
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()



###############################################################################

xq=np.zeros((nqel,ncenter),dtype=np.float64)   
yq=np.zeros((nqel,ncenter),dtype=np.float64)   

for center in range(0,ncenter):

    print('============================================================')

    if center==0:
       x9=(x1+x2+x3+x4)/4 
       y9=(y1+y2+y3+y4)/4
       rad=np.sqrt(x9**2+y9**2)
    if center==1:
       x9=(x1+x2+x3+x4+x5+x6+x7+x8)/8. 
       y9=(y1+y2+y3+y4+y5+y6+y7+y8)/8.
       rad=np.sqrt(x9**2+y9**2)
    if center==2:
       x9=(x1+x2+x3+x4+3*x5+3*x6+3*x7+3*x8)/16. 
       y9=(y1+y2+y3+y4+3*y5+3*y6+3*y7+3*y8)/16.
       rad=np.sqrt(x9**2+y9**2)
    if center==3:
       x9=0.5*(R1+R2)*np.cos(np.pi/2-dtheta/2) 
       y9=0.5*(R1+R2)*np.sin(np.pi/2-dtheta/2)
       rad=np.sqrt(x9**2+y9**2)
    if center==4:
       rad=np.sqrt(xcg**2+ycg**2)
       x9=xcg
       y9=ycg
    if center==5:
       rad=0.5*(R1+R2)+xi/100*(R2-R1)/2
       x9=rad*np.cos(np.pi/2-dtheta/2) 
       y9=rad*np.sin(np.pi/2-dtheta/2)

    xV = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9],dtype=np.float64)
    yV = np.array([y1,y2,y3,y4,y5,y6,y7,y8,y9],dtype=np.float64)
    fV = fct(xV,yV)

    np.savetxt('nodes'+str(center)+'.ascii',np.array([xV,yV]).T)

    r=np.zeros(npts,dtype=np.float64)   
    s=np.zeros(npts,dtype=np.float64)   
    xQ2=np.zeros(npts,dtype=np.float64)   
    yQ2=np.zeros(npts,dtype=np.float64)   
    xx=np.zeros(npts,dtype=np.float64)   
    yy=np.zeros(npts,dtype=np.float64)   
    err_pos2=np.zeros(npts,dtype=np.float64)   
    err_dfdx=np.zeros(npts,dtype=np.float64)   
    err_dfdy=np.zeros(npts,dtype=np.float64)   
    err_gradf=np.zeros(npts,dtype=np.float64)   
    jcb=np.zeros((2,2),dtype=np.float64)
    dNNNVdx=np.zeros(9,dtype=np.float64)
    dNNNVdy=np.zeros(9,dtype=np.float64)
    jcob=np.zeros(npts,dtype=np.float64)

    for i in range(0,npts):
        r[i]=random.uniform(-1.,+1)
        s[i]=random.uniform(-1.,+1)
        NNNV=NNV(r[i],s[i])
        dNNNVdr=dNNVdr(r[i],s[i])
        dNNNVds=dNNVds(r[i],s[i])
        xQ2[i]=np.dot(NNNV,xV)
        yQ2[i]=np.dot(NNNV,yV)
        jcb[0,0]=np.dot(dNNNVdr[:],xV[:])
        jcb[0,1]=np.dot(dNNNVdr[:],yV[:])
        jcb[1,0]=np.dot(dNNNVds[:],xV[:])
        jcb[1,1]=np.dot(dNNNVds[:],yV[:])
        
        jcob[i] = np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)
        for k in range(0,9):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

        err_dfdx[i]=np.dot(dNNNVdx[:],fV[:])-dfctdx(xQ2[i],yQ2[i])
        err_dfdy[i]=np.dot(dNNNVdy[:],fV[:])-dfctdy(xQ2[i],yQ2[i])
        err_gradf[i]=err_dfdx[i]**2+err_dfdy[i]**2

        xx[i]=abs(np.dot(NNNV[:],xV[:]**2)-xQ2[i]**2)
        yy[i]=abs(np.dot(NNNV[:],yV[:]**2)-yQ2[i]**2)
        err_pos2=xx[i]+yy[i]

    #end for

    #np.savetxt('rs.ascii',np.array([r,s]).T)

    np.savetxt('points'+str(center)+'.ascii',np.array([xQ2,yQ2,xx,yy,err_dfdx,err_dfdy,jcob]).T)

    print('avrg err_pos2 on pts',np.sum(err_pos2)/npts,' ',rad,'c'+str(center))
    print('avrg err_grad on pts',np.sum(err_gradf)/npts,' ',rad,'c'+str(center))

    ###########################################################################

    vtufile=open('points'+str(center)+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(npts,npts))
    #
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,npts):
        vtufile.write("%10e %10e %10e \n" %(xQ2[i],yQ2[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='jcob' Format='ascii'>\n")
    for i in range(0,npts):
        vtufile.write("%e \n" % jcob[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='error on x^2' Format='ascii'>\n")
    for i in range(0,npts):
        vtufile.write("%e \n" % xx[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='error on y^2' Format='ascii'>\n")
    for i in range(0,npts):
        vtufile.write("%e \n" % yy[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='error on dfdx' Format='ascii'>\n")
    for i in range(0,npts):
        vtufile.write("%e \n" % err_dfdx[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='error on dfdy' Format='ascii'>\n")
    for i in range(0,npts):
        vtufile.write("%e \n" % err_dfdy[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    #
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range(0,npts):
        vtufile.write("%d " % i)
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range(0,npts):
        vtufile.write("%d " % (i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range(0,npts):
        vtufile.write("%d " % 1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    #
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    ###############################################################################

    #fig = plt.figure()
    #plt.grid()
    #plt.scatter(r, s, s=1)
    #ax = fig.gca()
    #plt.xlim([-1,1])
    #plt.ylim([-1,1])
    #ax.set_aspect('equal', adjustable='box')
    #plt.savefig('plt_rs.pdf', bbox_inches='tight')
    #plt.show()

    #fig = plt.figure()
    #plt.grid()
    #plt.scatter(xV, yV, c='red',s=25)
    #ax = fig.gca()
    #plt.savefig('nodes'+str(center)+'.pdf', bbox_inches='tight')
    #plt.show()

    #fig = plt.figure()
    #plt.grid()
    #plt.scatter(xQ2, yQ2, s=0.5)
    #plt.scatter(xV, yV, s=25)
    #ax = fig.gca()
    #plt.savefig('points_'+str(center)+'.png', bbox_inches='tight')
    #plt.show()

    #fig = plt.figure()
    #sc = plt.scatter(xQ2, yQ2, c=jcob, s=2)
    #ax = fig.gca()
    #plt.colorbar(sc)
    #plt.grid()
    #plt.savefig('jcob'+str(center)+'.pdf', bbox_inches='tight')
    #plt.show()

    ###############################################################################

    jcb=np.zeros((2,2),dtype=np.float64)

    err_posx2=0
    err_posy2=0
    err_dfdx=0
    err_dfdy=0
    err_gradf=0
    area=0

    cq=0
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords_r[cq]
            sq=qcoords_s[cq]
            weightq=qweights[cq]

            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)

            xq[cq,center]=np.dot(NNNV,xV)
            yq[cq,center]=np.dot(NNNV,yV)

            jcb[0,0]=np.dot(dNNNVdr[:],xV[:])
            jcb[0,1]=np.dot(dNNNVdr[:],yV[:])
            jcb[1,0]=np.dot(dNNNVds[:],xV[:])
            jcb[1,1]=np.dot(dNNNVds[:],yV[:])

            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)

            area+=jcob*weightq

            for k in range(0,9):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            err_posx2+=(np.dot(NNNV[:],xV[:]**2)-xq[cq,center]**2) *jcob*weightq 
            err_posy2+=(np.dot(NNNV[:],yV[:]**2)-yq[cq,center]**2) *jcob*weightq 
            err_dfdx+=(np.dot(dNNNVdx[:],fV[:])-dfctdx(xq[cq,center],yq[cq,center])) *jcob*weightq 
            err_dfdy+=(np.dot(dNNNVdy[:],fV[:])-dfctdy(xq[cq,center],yq[cq,center])) *jcob*weightq 
            err_gradf+=err_dfdx**2+err_dfdy**2

            cq+=1
        #end for
    #end for

    print('center=',center,': x9,y9=',x9,y9,'| rad 9,', rad  )
    print('int err_pos2=',abs(err_posx2)+abs(err_posy2),' ',rad,'c'+str(center))
    print('int err_grad=',err_gradf,' ',rad,'c'+str(center))

    if element==4: 
       print('area=',area,' area th=',area_th,' area error',area-area_th,rad,'c'+str(center))

#end for
    
###############################################################################
    
print('============================================================')

fig = plt.figure()
for i in range(0,ncenter):
    plt.scatter(xq[:,i], yq[:,i], s=10)
ax = fig.gca()
plt.grid()
plt.savefig('quads.pdf', bbox_inches='tight')
#plt.show()

for i in range(0,ncenter):
    np.savetxt('quads'+str(i)+'.ascii',np.array([xq[:,i],yq[:,i]]).T)



