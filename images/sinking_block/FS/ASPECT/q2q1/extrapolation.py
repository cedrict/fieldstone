import numpy as np

f = open('statistics_none_5678910','r')
lines = f.readlines()
f.close

h  = np.zeros(6,dtype=np.float64)
vrms = np.zeros(6,dtype=np.float64)
maxvel = np.zeros(6,dtype=np.float64)

for i in range(21,27):
    vals=lines[i].strip().split()
    h[i-21]=1/np.sqrt(float(vals[3]))
    vrms[i-21]=float(vals[13])
    maxvel[i-21]=float(vals[14])

f = open('extrapolation_none_5678910','w')
for i in range(0,4):
    vrms_star=(vrms[i]*vrms[i+2]-vrms[i+1]**2)/(vrms[i]-2*vrms[i+1]+vrms[i+2])
    r_vrms=np.log2( (vrms[i+1]-vrms_star)/(vrms[i+2]-vrms_star)  )
    #print(vrms_star,r)
    maxvel_star=(maxvel[i]*maxvel[i+2]-maxvel[i+1]**2)/(maxvel[i]-2*maxvel[i+1]+maxvel[i+2])
    r_maxvel=np.log2( (maxvel[i+1]-maxvel_star)/(maxvel[i+2]-maxvel_star)  )
    f.write("%10e %10e %10e %10e %10e \n" %(h[i],vrms_star,r_vrms,maxvel_star,r_maxvel))

#################################################################################

f = open('statistics_arithmetic_5678910','r')
lines = f.readlines()
f.close

h  = np.zeros(6,dtype=np.float64)
vrms = np.zeros(6,dtype=np.float64)
maxvel = np.zeros(6,dtype=np.float64)

for i in range(21,27):
    vals=lines[i].strip().split()
    h[i-21]=1/np.sqrt(float(vals[3]))
    vrms[i-21]=float(vals[13])
    maxvel[i-21]=float(vals[14])

f = open('extrapolation_arithmetic_5678910','w')
for i in range(0,4):
    vrms_star=(vrms[i]*vrms[i+2]-vrms[i+1]**2)/(vrms[i]-2*vrms[i+1]+vrms[i+2])
    r_vrms=np.log2( (vrms[i+1]-vrms_star)/(vrms[i+2]-vrms_star)  )
    maxvel_star=(maxvel[i]*maxvel[i+2]-maxvel[i+1]**2)/(maxvel[i]-2*maxvel[i+1]+maxvel[i+2])
    r_maxvel=np.log2( (maxvel[i+1]-maxvel_star)/(maxvel[i+2]-maxvel_star)  )
    f.write("%10e %10e %10e %10e %10e \n" %(h[i],vrms_star,r_vrms,maxvel_star,r_maxvel))

#################################################################################

f = open('statistics_geometric_5678910','r')
lines = f.readlines()
f.close

h  = np.zeros(6,dtype=np.float64)
vrms = np.zeros(6,dtype=np.float64)
maxvel = np.zeros(6,dtype=np.float64)

for i in range(21,27):
    vals=lines[i].strip().split()
    h[i-21]=1/np.sqrt(float(vals[3]))
    vrms[i-21]=float(vals[13])
    maxvel[i-21]=float(vals[14])

f = open('extrapolation_geometric_5678910','w')
for i in range(0,4):
    vrms_star=(vrms[i]*vrms[i+2]-vrms[i+1]**2)/(vrms[i]-2*vrms[i+1]+vrms[i+2])
    r_vrms=np.log2( (vrms[i+1]-vrms_star)/(vrms[i+2]-vrms_star)  )
    maxvel_star=(maxvel[i]*maxvel[i+2]-maxvel[i+1]**2)/(maxvel[i]-2*maxvel[i+1]+maxvel[i+2])
    r_maxvel=np.log2( (maxvel[i+1]-maxvel_star)/(maxvel[i+2]-maxvel_star)  )
    #print(vrms_star,r)
    f.write("%10e %10e %10e %10e %10e \n" %(h[i],vrms_star,r_vrms,maxvel_star,r_maxvel))

#################################################################################

f = open('statistics_harmonic_5678910','r')
lines = f.readlines()
f.close

h  = np.zeros(6,dtype=np.float64)
vrms = np.zeros(6,dtype=np.float64)
maxvel = np.zeros(6,dtype=np.float64)

for i in range(21,27):
    vals=lines[i].strip().split()
    h[i-21]=1/np.sqrt(float(vals[3]))
    vrms[i-21]=float(vals[13])
    maxvel[i-21]=float(vals[14])

f = open('extrapolation_harmonic_5678910','w')
for i in range(0,4):
    vrms_star=(vrms[i]*vrms[i+2]-vrms[i+1]**2)/(vrms[i]-2*vrms[i+1]+vrms[i+2])
    r_vrms=np.log2( (vrms[i+1]-vrms_star)/(vrms[i+2]-vrms_star)  )
    maxvel_star=(maxvel[i]*maxvel[i+2]-maxvel[i+1]**2)/(maxvel[i]-2*maxvel[i+1]+maxvel[i+2])
    r_maxvel=np.log2( (maxvel[i+1]-maxvel_star)/(maxvel[i+2]-maxvel_star)  )
    #print(vrms_star,r)
    f.write("%10e %10e %10e %10e %10e \n" %(h[i],vrms_star,r_vrms,maxvel_star,r_maxvel))

#################################################################################

f = open('statistics_q1_5678910','r')
lines = f.readlines()
f.close

h  = np.zeros(6,dtype=np.float64)
vrms = np.zeros(6,dtype=np.float64)
maxvel = np.zeros(6,dtype=np.float64)

for i in range(21,27):
    vals=lines[i].strip().split()
    h[i-21]=1/np.sqrt(float(vals[3]))
    vrms[i-21]=float(vals[13])
    maxvel[i-21]=float(vals[14])

f = open('extrapolation_q1_5678910','w')
for i in range(0,4):
    vrms_star=(vrms[i]*vrms[i+2]-vrms[i+1]**2)/(vrms[i]-2*vrms[i+1]+vrms[i+2])
    r_vrms=np.log2( (vrms[i+1]-vrms_star)/(vrms[i+2]-vrms_star)  )
    maxvel_star=(maxvel[i]*maxvel[i+2]-maxvel[i+1]**2)/(maxvel[i]-2*maxvel[i+1]+maxvel[i+2])
    r_maxvel=np.log2( (maxvel[i+1]-maxvel_star)/(maxvel[i+2]-maxvel_star)  )
    #print(vrms_star,r)
    f.write("%10e %10e %10e %10e %10e \n" %(h[i],vrms_star,r_vrms,maxvel_star,r_maxvel))

#################################################################################







