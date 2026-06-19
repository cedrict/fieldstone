#Script to carry out regression analysis on the L1 and L2
#norms of strain and viscous dissipation against grid spacing
#for the 4 benchmarks used in this version of simplefem

import numpy as np
import matplotlib.pyplot as plt
import io
from scipy import stats


data1 = np.loadtxt("OUT/discretisation_errors_derivatives_1.dat")
nnx = data1[:,0]
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

x = nnx


# And now for the regression analysis
# 

dxu_list_1 = [dxu_L1_1,dxu_L1_2,dxu_L1_3,dxu_L1_5]
dxu_list_2 = [dxu_L2_1,dxu_L2_2,dxu_L2_3,dxu_L2_5]
dyv_list_1 = [dyv_L1_1,dyv_L1_2,dyv_L1_3,dyv_L1_5]
dyv_list_2 = [dyv_L2_1,dyv_L2_2,dyv_L2_3,dyv_L2_5]
dxv_list_1 = [dxv_L1_1,dxv_L1_2,dxv_L1_3,dxv_L1_5]
dxv_list_2 = [dxv_L2_1,dxv_L2_2,dxv_L2_3,dxv_L2_5]
dyu_list_1 = [dyu_L1_1,dyu_L1_2,dyu_L1_3,dyu_L1_5]
dyu_list_2 = [dyu_L2_1,dyu_L2_2,dyu_L2_3,dyu_L2_5]
phi_list_1 = [phi_L1_1,phi_L1_2,phi_L1_3,phi_L1_5]
phi_list_2 = [phi_L2_1,phi_L2_2,phi_L2_3,phi_L2_5]

def get_regression(h,y):
    y=np.abs(y)
    x=np.abs(h)
    #return np.linalg.lstsq(np.vstack([np.log(h), np.ones(len(np.log(h)))]).T,np.log(y))[0][0]
    return stats.linregress(np.log(x),np.log(y))[0]


dxu_regression_list_1=['r_{\partial_x u_1}']
for dxu in dxu_list_1:
    dxu_regression_list_1.append(get_regression(x,dxu))
dxu_regression_list_2=['r_{\partial_x u_2}']
for dxu in dxu_list_2:
    dxu_regression_list_2.append(get_regression(x,dxu))


dyv_regression_list_1=['r_{\partial_y v_1}']
for dyv in dyv_list_1:
    dyv_regression_list_1.append(get_regression(x,dyv))
dyv_regression_list_2=['r_{\partial_y v_2}']
for dyv in dyv_list_2:
    dyv_regression_list_2.append(get_regression(x,dyv))





dxv_regression_list_1=['r_{\partial_x v_1}']
for dxv in dxv_list_1:
    dxv_regression_list_1.append(get_regression(x,dxv))
dxv_regression_list_2=['r_{\partial_x v_2}']
for dxv in dxv_list_2:
    dxv_regression_list_2.append(get_regression(x,dxv))


dyu_regression_list_1=['r_{\partial_y u_1}']
for dyu in dyu_list_1:
    dyu_regression_list_1.append(get_regression(x,dyu))
dyu_regression_list_2=['r_{\partial_y u_2}']
for dyu in dyu_list_2:
    dyu_regression_list_2.append(get_regression(x,dyu))

phi_regression_list_1=['r_{\phi_1}']
for phi in phi_list_1:
    phi_regression_list_1.append(get_regression(x,phi))
phi_regression_list_2=['r_{\phi_2}']
for phi in phi_list_2:
    phi_regression_list_2.append(get_regression(x,phi))

ratesArray=[]
ratesArray.append(dxu_regression_list_1)
ratesArray.append(dxu_regression_list_1)
ratesArray.append(dyu_regression_list_1)
ratesArray.append(dyu_regression_list_1)
ratesArray.append(dyv_regression_list_1)
ratesArray.append(dyv_regression_list_1)
ratesArray.append(dxv_regression_list_1)
ratesArray.append(dxv_regression_list_1)
ratesArray.append(phi_regression_list_1)
ratesArray.append(phi_regression_list_1)


file=io.open("convergence_rates_strain.txt",'w')
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


