#This script plots the L1 and L2 norms of velocity and pressure
#for the 4 benchmarks featured in this version of simplefem

import numpy as np
import matplotlib.pyplot as plt


#Load all the data needed
data1=np.loadtxt("OUT/discretisation_errors_1.dat")
u1L2 = data1[:,1]
v1L2 = data1[:,2]
p1L2 = data1[:,3]
u1L1 = np.abs(data1[:,4])
v1L1 = np.abs(data1[:,5])
p1L1 = np.abs(data1[:,6])
uv1L1 = np.sqrt(u1L1**2+v1L1**2)
uv1L2 = np.sqrt(u1L2**2+v1L2**2)


data2=np.loadtxt("OUT/discretisation_errors_2.dat")
u2L2 = data2[:,1]
v2L2 = data2[:,2]
p2L2 = data2[:,3]
u2L1 = np.abs(data2[:,4])
v2L1 = np.abs(data2[:,5])
p2L1 = np.abs(data2[:,6])
uv2L1 = np.sqrt(u2L1**2+v2L1**2)
uv2L2 = np.sqrt(u2L2**2+v2L2**2)


data3=np.loadtxt("OUT/discretisation_errors_3.dat")
u3L2 = data3[:,1]
v3L2 = data3[:,2]
p3L2 = data3[:,3]
u3L1 = np.abs(data3[:,4])
v3L1 = np.abs(data3[:,5])
p3L1 = np.abs(data3[:,6])
uv3L1 = np.sqrt(u3L1**2+v3L1**2)
uv3L2 = np.sqrt(u3L2**2+v3L2**2)

data5=np.loadtxt("OUT/discretisation_errors_4.dat")
u4L2 = data5[:,1]
v4L2 = data5[:,2]
p4L2 = data5[:,3]
u4L1 = np.abs(data5[:,4])
v4L1 = np.abs(data5[:,5])
p4L1 = np.abs(data5[:,6])
uv4L1 = np.sqrt(u4L1**2+v4L1**2)
uv4L2 = np.sqrt(u4L2**2+v4L2**2)



h = data1[:,0]
x = np.log(h)



#Plot the L1 Norms
f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))
f.suptitle("L1 Norms for u, v, p and uv, for Benchmarks 1-4", fontsize=20,y=0.99)

f.tight_layout(w_pad=3,pad=3)
ax1.plot(x,np.log(u1L1),"-+",label="Benchmark 1")
ax1.plot(x,np.log(u2L1),"-+",label="Benchmark 2")
ax1.plot(x,np.log(u3L1),"-+",label="Benchmark 3")
ax1.plot(x,np.log(u4L1),"-+",label="Benchmark 4")
ax1.plot(x,2*x-2,label="Theoretical Convergence")
ax1.legend()
ax1.set_xlabel("$\log_{10}(h)$")
ax1.set_ylabel("log($||e_u||_1$)")

ax2.plot(x,np.log(v1L1),"-+",label="Benchmark 1")
ax2.plot(x,np.log(v2L1),"-+",label="Benchmark 2")
ax2.plot(x,np.log(v3L1),"-+",label="Benchmark 3")
ax2.plot(x,np.log(v4L1),"-+",label="Benchmark 4")
ax2.plot(x,2*x-3.5,label="Theoretical Convergence")
ax2.legend()
ax1.set_xlabel("$\log_{10}(h)$")
ax2.set_ylabel("log($||e_v||_1$)")

ax3.plot(x,np.log(p1L1),"-+",label="Benchmark 1")
ax3.plot(x,np.log(p2L1),"-+",label="Benchmark 2")
ax3.plot(x,np.log(p3L1),"-+",label="Benchmark 3")
ax3.plot(x,np.log(p4L1),"-+",label="Benchmark 4")
ax3.plot(x,x,label="Theoretical Convergence")
ax3.legend()
ax1.set_xlabel("$\log_{10}(h)$")
ax3.set_ylabel("log($||e_p||_1$)")

ax4.plot(x,np.log(uv1L1),"-+",label="Benchmark 1")
ax4.plot(x,np.log(uv2L1),"-+",label="Benchmark 2")
ax4.plot(x,np.log(uv3L1),"-+",label="Benchmark 3")
ax4.plot(x,np.log(uv4L1),"-+",label="Benchmark 4")
ax4.plot(x,2*x-2,label="Theoretical Convergence")
ax4.legend()
ax4.set_xlabel("$\log_{10}(h)$")
ax4.set_ylabel("log($||e_{uv}||_1$)")

f.savefig("L1_norms.pdf")


#Plot the L2 Norms

f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))
f.suptitle("L2 Norms for u, v, p and uv, for Benchmarks 1-4", fontsize=20,y=0.99)

f.tight_layout(w_pad=3,pad=3)
ax1.plot(x,np.log(u1L2),"-+",label="Benchmark 1")
ax1.plot(x,np.log(u2L2),"-+",label="Benchmark 2")
ax1.plot(x,np.log(u3L2),"-+",label="Benchmark 3")
ax1.plot(x,np.log(u4L2),"-+",label="Benchmark 4")
ax1.plot(x,2*x-2,label="Theoretical Convergence")
ax1.legend()
ax1.set_xlabel("$\log_{10}(h)$")
ax1.set_ylabel("log($||e_u||_2)$")

ax2.plot(x,np.log(v1L2),"-+",label="Benchmark 1")
ax2.plot(x,np.log(v2L2),"-+",label="Benchmark 2")
ax2.plot(x,np.log(v3L2),"-+",label="Benchmark 3")
ax2.plot(x,np.log(v4L2),"-+",label="Benchmark 4")
ax2.plot(x,2*x-3,label="Theoretical Convergence")
ax2.legend()
ax2.set_xlabel("$\log_{10}(h)$")
ax2.set_ylabel("log($||e_v||_2$)")

ax3.plot(x,np.log(p1L2),"-+",label="Benchmark 1")
ax3.plot(x,np.log(p2L2),"-+",label="Benchmark 2")
ax3.plot(x,np.log(p3L2),"-+",label="Benchmark 3")
ax3.plot(x,np.log(p4L2),"-+",label="Benchmark 4")
ax3.plot(x,x,label="Theoretical Convergence")
ax3.legend()
ax3.set_xlabel("$\log_{10}(h)$")
ax3.set_ylabel("log($||e_p||_2$)")

ax4.plot(x,np.log(uv1L2),"-+",label="Benchmark 1")
ax4.plot(x,np.log(uv2L2),"-+",label="Benchmark 2")
ax4.plot(x,np.log(uv3L2),"-+",label="Benchmark 3")
ax4.plot(x,np.log(uv4L2),"-+",label="Benchmark 4")
ax4.plot(x,2*x-1.5,label="Theoretical Convergence")
ax4.legend()
ax4.set_xlabel("$\log_{10}(h)$")
ax4.set_ylabel("log($||e_{uv}||_2$)")

f.savefig("L2_norms.pdf")

