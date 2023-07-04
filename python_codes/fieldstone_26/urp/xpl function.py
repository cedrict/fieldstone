#imports 
import numpy as np
import re
import matplotlib.pyplot as plt

#######################
# dumps and functions #
#######################

#dumps for extrapolated values
extr_50=[] #nelx = 50, 100, 200
extr_100=[] #nelx = 100, 200, 400

#empty array for strings from opla
vrms_str = []
nelx_str = []
h=[]
log_h=[]


#functions for the extrapolation 

def xpl_res50 (X, case="vrms", tol=10):
    
    Xe=(X[0]*X[3]-X[1]**2)/(X[0]-2*X[1]+X[3])
    r=np.log2((X[1]-Xe)/(X[3]-Xe))
    extr_50.append([Xe, r, tol])
    
    print("for tolerance", tol, "and with nelx=50,100,200")
    print (case, "extrapolation =", Xe)
    print("with rate=", r)
    print("  ")
    
    return (Xe)

def xpl_res100 (X, case="vrms", tol=10):
    
    Xe=(X[1]*X[6]-X[3]**2)/(X[1]-2*X[3]+X[6])
    r=np.log2((X[3]-Xe)/(X[6]-Xe))
    extr_100.append([Xe, r, tol])
    
    print("for tolerance", tol, "and with nelx=100,200,400")
    print (case, "extrapolation =", Xe)
    print("with rate=", r)
    print("  ")
    
    return (Xe)


######################
# from opla to array #
######################

#find the right values from opla
with open('C:/Users/thoma/Documents/uni/2022-2023/Honours/URP/opla', 'r') as file:
    for line in file: 
        if line.startswith("     -> nel=")==True:
            value = re.findall(r'vrms \(cm/year\)= (\d.+)', line)
            vrms_str.append(value)
        elif line.startswith("-------------------------------case 1a")==True:
            nelx = re.findall(r'nelx = (\d+)', line)
            nelx_str.append(nelx)

#convert to float and integer values
vrms_flt = [float(item[0]) for item in vrms_str]
nelx_int = [float(item[0]) for item in nelx_str]

#nelx to h and log(h)
for i in range(len(nelx_int)):
    h.append(1000/nelx_int[i])
    log_h.append(np.log10(1000/nelx_int[i]))
            
    
###################
# seperate arrays #
###################

#split vrms for tol
vrms_tol03 = np.array(vrms_flt[:7])
vrms_tol04 = np.array(vrms_flt[7:14])
vrms_tol05 = np.array(vrms_flt[14:21])
vrms_tol07 = np.array(vrms_flt[21:28])
vrms_tol08 = np.array(vrms_flt[28:])

#split h, repeats with tol
h = np.array(h[:7])

#split log(h), repeats with tol
log_h = np.array(log_h[:7])

#split nelx, repeats with tol
nelx_int = np.array(nelx_int[:7])


############################
# extrapolation from array #
############################

xpl_res50(vrms_tol03, "vrms", 1e-3)
xpl_res50(vrms_tol04, "vrms", 1e-4)
xpl_res50(vrms_tol05, "vrms", 1e-5)
xpl_res50(vrms_tol07, "vrms", 1e-7)
xpl_res50(vrms_tol08, "vrms", 1e-8)

xpl_res100(vrms_tol03, "vrms", 1e-3)
xpl_res100(vrms_tol04, "vrms", 1e-4)
xpl_res100(vrms_tol05, "vrms", 1e-5)
xpl_res100(vrms_tol07, "vrms", 1e-7)
xpl_res100(vrms_tol08, "vrms", 1e-8)


####################
# calculate log(e) #
####################

log_e_tol03=np.log10(abs(vrms_tol03-extr_100[4][0]))
log_e_tol04=np.log10(abs(vrms_tol04-extr_100[4][0]))
log_e_tol05=np.log10(abs(vrms_tol05-extr_100[4][0]))
log_e_tol07=np.log10(abs(vrms_tol07-extr_100[4][0]))
log_e_tol08=np.log10(abs(vrms_tol08-extr_100[4][0]))


#################
# least squares #
#################

ab_tol03=np.linalg.lstsq(np.vstack([log_h, np.ones(len(log_h))]).T,log_e_tol03, rcond=None)[0]
ab_tol04=np.linalg.lstsq(np.vstack([log_h, np.ones(len(log_h))]).T,log_e_tol04, rcond=None)[0]
ab_tol08=np.linalg.lstsq(np.vstack([log_h, np.ones(len(log_h))]).T,log_e_tol08, rcond=None)[0]

lin_tol03=log_h*ab_tol03[0]+ab_tol03[1]
lin_tol04=log_h*ab_tol04[0]+ab_tol04[1]
lin_tol08=log_h*ab_tol08[0]+ab_tol08[1]


##############
# some plots #            
##############

#vrms to log(h)
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(log_h, vrms_tol03, marker='o', label='tol=1e-3', color='red')
ax.plot(log_h, vrms_tol04, marker='o', label='tol=1e-4', color='blue')
ax.plot(log_h, vrms_tol05, marker='o', label='tol=1e-5', color='green')
ax.plot(log_h, vrms_tol07, marker='o', label='tol=1e-7', color='orange')
ax.plot(log_h, vrms_tol08, marker='o', label='tol=1e-8', color='brown')

ax.axhline(y=extr_100[4][0], color='black', linestyle='--', label='vrms*')

plt.ylim(0.046, 0.058)
ax.set_xlabel('log(h) [km]')
ax.set_ylabel('vrms [cm/yr]')
ax.set_title('vrms vs log(h)')

ax.legend()
plt.show()


#vrms to nelx
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(nelx_int, vrms_tol03, marker='o', label='tol=1e-3', color='red')
ax.plot(nelx_int, vrms_tol04, marker='o', label='tol=1e-4', color='blue')
ax.plot(nelx_int, vrms_tol05, marker='o', label='tol=1e-5', color='green')
ax.plot(nelx_int, vrms_tol07, marker='o', label='tol=1e-7', color='orange')
ax.plot(nelx_int, vrms_tol08, marker='o', label='tol=1e-8', color='brown')

ax.axhline(y=extr_100[4][0], color='black', linestyle='--', label='vrms*')

plt.ylim(0.046, 0.058)
ax.set_xlabel('nelx')
ax.set_ylabel('vrms [cm/yr]')
ax.set_title('vrms vs nelx')

ax.legend()
plt.show()

#errors and rates
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(log_h, log_e_tol03, marker='o', label='tol=1e-3', color='red')
ax.plot(log_h, log_e_tol04, marker='o', label='tol=1e-4', color='blue')
ax.plot(log_h, log_e_tol05, marker='o', label='tol=1e-5', color='green')
ax.plot(log_h, log_e_tol07, marker='o', label='tol=1e-7', color='orange')
ax.plot(log_h, log_e_tol08, marker='o', label='tol=1e-8', color='brown')

ax.plot(log_h, lin_tol03, label='slope=%s' %round(ab_tol03[0], 4), color='pink')
ax.plot(log_h, lin_tol04, label='slope=%s' %round(ab_tol04[0], 4), color='lightblue')
ax.plot(log_h, lin_tol08, label='slope=%s' %round(ab_tol08[0], 4), color='grey')

ax.set_xlabel('log(h)')
ax.set_ylabel('error = log10(abs(vrms-vrms*))')
ax.set_title('vrms error vs log(h)')

ax.legend()
plt.show()

#just slopes

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(log_h, lin_tol03, label='tol=1e-3 with slope=%s' %round(ab_tol03[0], 4), color='pink')
ax.plot(log_h, lin_tol04, label='tol=1e-4 with slope=%s' %round(ab_tol04[0], 4), color='lightblue')
ax.plot(log_h, lin_tol08, label='tol=1e-5 with slope=%s' %round(ab_tol08[0], 4), color='grey')

ax.set_xlabel('log(h)')
ax.set_ylabel('error = log10(abs(vrms-vrms*))')
ax.set_title('linear trend of the vrms error vs log(h)')

ax.legend()
plt.show()