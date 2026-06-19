#Script to carry out regression analysis on the L1 and L2
#norms of the velocity and pressure against grid spacing
#for the 4 benchmarks used in this version of simplefem





import numpy as np
import matplotlib.pyplot as plt 
import io
from scipy import stats


def clean_data(data,nnx_minimum):
    #Removes data for nnx < nnx_minimum
    #To allow regression only on data
    #on larger values of nnx
    while True:
        if (1/data[0,0] < nnx_minimum+1):
            data = np.delete(data,0,0)
        else:
            break
nnx_minimum=32

data1=np.loadtxt("OUT/discretisation_errors_1.dat")
clean_data(data1,nnx_minimum)
u1L2 = data1[:,1]
v1L2 = data1[:,2]
p1L2 = data1[:,3]
u1L1 = np.abs(data1[:,4])
v1L1 = np.abs(data1[:,5])
p1L1 = np.abs(data1[:,6])

uv1L1 = np.sqrt(u1L1**2+v1L1**2)
uv1L2 = np.sqrt(u1L2**2+v1L2**2)


data2=np.loadtxt("OUT/discretisation_errors_2.dat")
clean_data(data2,nnx_minimum)
u2L2 = data2[:,1]
v2L2 = data2[:,2]
p2L2 = data2[:,3]
u2L1 = np.abs(data2[:,4])
v2L1 = np.abs(data2[:,5])
p2L1 = np.abs(data2[:,6])

uv2L1 = np.sqrt(u2L1**2+v2L1**2)
uv2L2 = np.sqrt(u2L2**2+v2L2**2)



data3=np.loadtxt("OUT/discretisation_errors_3.dat")
clean_data(data3,nnx_minimum)
u3L2 = data3[:,1]
v3L2 = data3[:,2]
p3L2 = data3[:,3]
u3L1 = np.abs(data3[:,4])
v3L1 = np.abs(data3[:,5])
p3L1 = np.abs(data3[:,6])

uv3L1 = np.sqrt(u3L1**2+v3L1**2)
uv3L2 = np.sqrt(u3L2**2+v3L2**2)



data5=np.loadtxt("OUT/discretisation_errors_4.dat")
clean_data(data5,nnx_minimum)
u5L2 = data5[:,1]
v5L2 = data5[:,2]
p5L2 = data5[:,3]
u5L1 = np.abs(data5[:,4])
v5L1 = np.abs(data5[:,5])
p5L1 = np.abs(data5[:,6])

uv5L1 = np.sqrt(u5L1**2+v5L1**2)
uv5L2 = np.sqrt(u5L2**2+v5L2**2)



nnx = data1[:,0]
h=nnx
def get_regression(h,y):
    y=np.abs(y)
    x=np.abs(h)
    #return np.linalg.lstsq(np.vstack([np.log(h), np.ones(len(np.log(h)))]).T,np.log(y))[0][0]
    return stats.linregress(np.log(x),np.log(y))[0]

r_u1=['r_{u1}']
r_u1.append(get_regression(h,u1L1))
r_u1.append(get_regression(h,u2L1))
r_u1.append(get_regression(h,u3L1))
r_u1.append(get_regression(h,u5L1))


r_u2=['r_{u2}']
r_u2.append(get_regression(h,u1L2))
r_u2.append(get_regression(h,u2L2))
r_u2.append(get_regression(h,u3L2))
r_u2.append(get_regression(h,u5L2))


r_v1=['r_{v1}']
r_v1.append(get_regression(h,v1L1))
r_v1.append(get_regression(h,v2L1))
r_v1.append(get_regression(h,v3L1))
r_v1.append(get_regression(h,v5L1))


r_v2=['r_{v1}']
r_v2.append(get_regression(h,v1L2))
r_v2.append(get_regression(h,v2L2))
r_v2.append(get_regression(h,v3L2))
r_v2.append(get_regression(h,v5L2))
r_v2



r_p1=['r_{p1}']
r_p1.append(get_regression(h,p1L1))
r_p1.append(get_regression(h,p2L1))
r_p1.append(get_regression(h,p3L1))
r_p1.append(get_regression(h,p5L1))



r_p2=['r_{p2}']
r_p2.append(get_regression(h,p1L2))
r_p2.append(get_regression(h,p2L2))
r_p2.append(get_regression(h,p3L2))
r_p2.append(get_regression(h,p5L2))


r_uv1=['r_{uv1}']
r_uv1.append(get_regression(h,uv1L1))
r_uv1.append(get_regression(h,uv2L1))
r_uv1.append(get_regression(h,uv3L1))
r_uv1.append(get_regression(h,uv5L1))


r_uv2=['r_{uv2}']
r_uv2.append(get_regression(h,uv1L2))
r_uv2.append(get_regression(h,uv2L2))
r_uv2.append(get_regression(h,uv3L2))
r_uv2.append(get_regression(h,uv5L2))

ratesArray=[r_u1,r_u2,r_v1,r_v2,r_p1,r_p2,r_uv1,r_uv2]

file=io.open("convergence_rates.txt",'w')
file.write('\\begin{tabular}{| l | c c c c |} \n \hline \n')
file.write(' & Benchmark 1 & Benchmark 2 & Benchmark 3 & Benchmark 5 \\\ \hline \n')
for sublist in ratesArray:
    for i in range(len(sublist)):
        if (i==0):
            file.write('$'+sublist[i]+'$')
        else:
            file.write('{:1.4f}'.format(sublist[i]))
        if(i < len(sublist)-1):
            file.write(" & ")
    file.write(' \\\ \hline \n')
file.write(' \n \end{tabular}')
file.close()

