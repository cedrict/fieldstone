# Scipt to plot the L1 and L2 norms 
# of the strains and of $\phi$ with grid spacing
#for the 4 benchmarks used in this version of simplefem


import numpy as np
import matplotlib.pyplot as plt


data1 = np.loadtxt("OUT/discretisation_errors_derivatives_1.dat")
h = data1[:,0]
dxu_L1_1 = data1[:,1]
dxu_L2_1 = data1[:,2]
dxv_L1_1 = data1[:,3]
dxv_L2_1 = data1[:,4]
dyu_L1_1 = data1[:,5]
dyu_L2_1 = data1[:,6]
dyv_L1_1 = data1[:,7]
dyv_L2_1 = data1[:,8]
phi_L1_1 = data1[:,9]
phi_L2_1 = data1[:,10]



data2 = np.loadtxt("OUT/discretisation_errors_derivatives_2.dat")

dxu_L1_2 = data2[:,1]
dxu_L2_2 = data2[:,2]
dxv_L1_2 = data2[:,3]
dxv_L2_2 = data2[:,4]
dyu_L1_2 = data2[:,5]
dyu_L2_2 = data2[:,6]
dyv_L1_2 = data2[:,7]
dyv_L2_2 = data2[:,8]
phi_L1_2 = data2[:,9]
phi_L2_2 = data2[:,10]




data3 = np.loadtxt("OUT/discretisation_errors_derivatives_3.dat")

dxu_L1_3 = data3[:,1]
dxu_L2_3 = data3[:,2]
dxv_L1_3 = data3[:,3]
dxv_L2_3 = data3[:,4]
dyu_L1_3 = data3[:,5]
dyu_L2_3 = data3[:,6]
dyv_L1_3 = data3[:,7]
dyv_L2_3 = data3[:,8]
phi_L1_3 = data3[:,9]
phi_L2_3 = data3[:,10]




data5 = np.loadtxt("OUT/discretisation_errors_derivatives_4.dat")

dxu_L1_5 = data5[:,1]
dxu_L2_5 = data5[:,2]
dxv_L1_5 = data5[:,3]
dxv_L2_5 = data5[:,4]
dyu_L1_5 = data5[:,5]
dyu_L2_5 = data5[:,6]
dyv_L1_5 = data5[:,7]
dyv_L2_5 = data5[:,8]
phi_L1_5 = data5[:,9]
phi_L2_5 = data5[:,10]

x = np.log(h)


#Plot the L1 Norms

f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))

ax1.plot(x,np.log(dxu_L1_1),"-+",label="Benchmark 1")
ax1.plot(x,np.log(dxu_L1_2),"-+",label="Benchmark 2")
ax1.plot(x,np.log(dxu_L1_3),"-+",label="Benchmark 3")
ax1.plot(x,np.log(dxu_L1_5),"-+",label="Benchmark 4")
ax1.legend()
ax1.set_xlabel("$\log_{10}(h)$")
ax1.set_ylabel("log($e_{xx}$ $L_1$)")

ax2.plot(x,np.log(dyv_L1_1),"-+",label="Benchmark 1")
ax2.plot(x,np.log(dyv_L1_2),"-+",label="Benchmark 2")
ax2.plot(x,np.log(dyv_L1_3),"-+",label="Benchmark 3")
ax2.plot(x,np.log(dyv_L1_5),"-+",label="Benchmark 4")
ax2.legend()
ax2.set_xlabel("$\log_{10}(h)$")
ax2.set_ylabel("log($e_{yy}$ $L_1$)")

ax3.plot(x,np.log(dxv_L1_1),"-+",label="Benchmark 1")
ax3.plot(x,np.log(dxv_L1_2),"-+",label="Benchmark 2")
ax3.plot(x,np.log(dxv_L1_3),"-+",label="Benchmark 3")
ax3.plot(x,np.log(dxv_L1_5),"-+",label="Benchmark 4")
ax3.legend()
ax3.set_xlabel("$\log_{10}(h)$")
ax3.set_ylabel("log($e_{xy}$ $L_1$)")

ax4.plot(x,np.log(phi_L1_1),"-+",label="Benchmark 1")
ax4.plot(x,np.log(phi_L1_2),"-+",label="Benchmark 2")
ax4.plot(x,np.log(phi_L1_3),"-+",label="Benchmark 3")
ax4.plot(x,np.log(phi_L1_5),"-+",label="Benchmark 4")
ax4.legend()
ax4.set_xlabel("$\log_{10}(h)$")
ax4.set_ylabel("log($\phi$ $L_1$)")

f.suptitle("L1 Norms for components of the strain rate tensor and $\phi$, for Benchmarks 1-4",fontsize=20,y=0.925)


f.savefig("Strain_Norms_L1.pdf")

#Plot the L2 Norms

f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))

ax1.plot(x,np.log(dxu_L2_1),"-+",label="Benchmark 1")
ax1.plot(x,np.log(dxu_L2_2),"-+",label="Benchmark 2")
ax1.plot(x,np.log(dxu_L2_3),"-+",label="Benchmark 3")
ax1.plot(x,np.log(dxu_L2_5),"-+",label="Benchmark 4")
ax1.legend()
ax1.set_xlabel("$\log_{10}(h)$")
ax1.set_ylabel("log($e_{xx}$ $L_2$)")

ax2.plot(x,np.log(dyv_L2_1),"-+",label="Benchmark 1")
ax2.plot(x,np.log(dyv_L2_2),"-+",label="Benchmark 2")
ax2.plot(x,np.log(dyv_L2_3),"-+",label="Benchmark 3")
ax2.plot(x,np.log(dyv_L2_5),"-+",label="Benchmark 4")
ax2.legend()
ax2.set_xlabel("$\log_{10}(h)$")
ax2.set_ylabel("log($e_{yy}$ $L_2$)")

ax3.plot(x,np.log(dxv_L2_1),"-+",label="Benchmark 1")
ax3.plot(x,np.log(dxv_L2_2),"-+",label="Benchmark 2")
ax3.plot(x,np.log(dxv_L2_3),"-+",label="Benchmark 3")
ax3.plot(x,np.log(dxv_L2_5),"-+",label="Benchmark 4")
ax3.legend()
ax3.set_xlabel("$\log_{10}(h)$")
ax3.set_ylabel("log($e_{xy}$ $L_2$)")

ax4.plot(x,np.log(phi_L2_1),"-+",label="Benchmark 1")
ax4.plot(x,np.log(phi_L2_2),"-+",label="Benchmark 2")
ax4.plot(x,np.log(phi_L2_3),"-+",label="Benchmark 3")
ax4.plot(x,np.log(phi_L2_5),"-+",label="Benchmark 4")
ax4.legend()
ax4.set_xlabel("$\log_{10}(h)$")
ax4.set_ylabel("log($\phi$ $L_2$)")

f.suptitle("L2 Norms for components of the strain rate tensor and $\phi$, for Benchmarks 1-4",fontsize=20,y=0.925)

f.savefig("Strain_Norms_L2.pdf")