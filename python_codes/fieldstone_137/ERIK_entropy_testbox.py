from cmath import sqrt
import numpy as np
import math as math
import numpy.linalg as LA
from matplotlib import pyplot as plt

mx = 10
my = 10
M=mx*my
C = 2
nmarker=10000

S_entropy = np.zeros((800,6),float)

for i in range(0,800): 
    particles_0 = np.zeros((nmarker,9),float) 
    c = np.zeros((C),float)
    P_c = np.zeros((C),float) 
    bins = np.zeros((M,C),float)
    bins2 = np.zeros((M,C),float)
    bins3= np.zeros((M),float)
    bins4 = np.zeros((M,C),float)
    S_entropy_array = np.zeros((M),float)  

    #for j in range(0,1):
      #file = np.loadtxt(''+folder+'/particles/particles-'+eerste+'.'+tweede+'.gnuplot', ndmin=2)
      #length_file = len(file)
      # <x> <y> <id> <pos> <pos> <vel> <vel> <init 1_mantle> <init 3_continent> <p> <T> <init pos> <init pos> 
      #for k in range(length_file):
          #ID = int(file[k][2])
          particles_0[ID][0] = file[k][2] # id 1
          particles_0[ID][1] = file[k][0] # x 2
          particles_0[ID][2] = file[k][1] # y 3
          particles_0[ID][3] = math.sqrt(file[k][0]**2 + file[k][1]**2) / 1000 #radius_end
          particles_0[ID][4] = file[k][9] # ori x 5
          particles_0[ID][5] = file[k][10] # ori y 6
          particles_0[ID][6] = math.sqrt(file[k][9]**2 + file[k][10]**2) / 1000 #radius start

    #painting
    #for k in range (0,nmarker):
    #        if particles_0[k][4] <= 0.5:
    #           particles_0[k][7] = 1
    #        else:
    #           particles_0[k][7] = 2





    for j in range (0,nmarker):
        l_x = 1.0 / mx
        for l in range(0,my):
            cx =  l * l_x
            cx2 = ((l+1) * l_x)
            for k in range(0,mx):
                cy = k * l_x
                cy2 = (k+1) * l_x
                counter = k + (mx*l)
                if particles_0[j,1] >= cx and particles_0[j,1] <= cx2 and\
                   particles_0[j,2] >= cy and particles_0[j,2] <= cy2:
                    particles_0[j,8] = counter
                    for m in range(0,C):
                        if particles_0[j,7] == (m+1):
                            bins[counter,m] +=   1

    for j in range(0,M):
        for k in range(0,C):
            c[k] += bins[j][k] 

    total = sum(c)
    sum_Njc_Pc =0
    for k in range(0,C):
        P_c[k] = c[k] / M 

    for j in range(0,M):
        for k in range(0,C):
            sum_Njc_Pc += bins[j][k] / P_c[k]  

    for j in range(0,M):
        for k in range(0,C):
            #   Ni,c / Pc (i: 1 tot bins, c: 1 tot species) voor Pj,c
            bins2[j][k] = bins[j][k] / P_c[k] / sum_Njc_Pc
            #   Nj,c / Pc (c: 1 tot species gesommeerd -> voor Pj
            bins3[j] +=  bins[j][k] / P_c[k]
        for k in range(0,C):
            #   Nj,c / Pc ) / all part/all classes -> voor Pc,j
            bins4[j][k] = bins[j][k] / P_c[k] / bins3[j] 
        bins3[j] = bins3[j] / sum_Njc_Pc

    for j in range(0,M):
        for k in range(0,C):         
            if bins2[j][k] != 0:
               S_entropy[i][0] -= bins2[j][k] * math.log(bins2[j][k]) ## S full
        if bins3[j] !=0:
           S_entropy[i][1] -= bins3[j] * math.log(bins3[j])   ## S(location)    

    for j in range(0,M):
        for k in range(0,C):
            if bins4[j][k] !=0:
               ## Sj (species)
               S_entropy_array[j] -=  bins4[j][k] * math.log(bins4[j][k])

    for j in range(0,M):
        S_entropy[i][2] +=  S_entropy_array[j] * bins3[j] ## S_location(species)

    S_entropy[i][3] = S_entropy[i][0] / math.log(M)
    S_entropy[i][4] = S_entropy[i][1] / math.log(C)
    S_entropy[i][5] = S_entropy[i][2] / math.log(C)




    np.savetxt('mixing.txt',particles_0)
    np.savetxt('part_per_bin.txt',bins)
    np.savetxt('Prob_part_per_bin.txt',bins2)
    np.savetxt('Prob_number_part_per_bin.txt',bins3)
    np.savetxt('Prob_classpart_per_bin.txt',bins4)
    np.savetxt('Entropy.txt',S_entropy)

