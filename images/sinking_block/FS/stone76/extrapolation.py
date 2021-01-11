import numpy as np

f = open('vrms_nq3','r')
lines = f.readlines()
f.close

h  = np.zeros(6,dtype=np.float64)
vrms = np.zeros(6,dtype=np.float64)
maxvel = np.zeros(6,dtype=np.float64)

for i in range(0,6):
    vals=lines[i].strip().split()
    h[i]=1/np.sqrt(float(vals[2]))
    vrms[i]=float(vals[5])
    maxvel[i]=float(vals[8])

f = open('extrapolation','w')
for i in range(0,4):
    vrms_star=(vrms[i]*vrms[i+2]-vrms[i+1]**2)/(vrms[i]-2*vrms[i+1]+vrms[i+2])
    r_vrms=np.log2( (vrms[i+1]-vrms_star)/(vrms[i+2]-vrms_star)  )
    maxvel_star=(maxvel[i]*maxvel[i+2]-maxvel[i+1]**2)/(maxvel[i]-2*maxvel[i+1]+maxvel[i+2])
    r_maxvel=np.log2( (maxvel[i+1]-maxvel_star)/(maxvel[i+2]-maxvel_star)  )
    f.write("%10e %10e %10e %10e %10e \n" %(h[i],vrms_star,r_vrms,maxvel_star,r_maxvel))

