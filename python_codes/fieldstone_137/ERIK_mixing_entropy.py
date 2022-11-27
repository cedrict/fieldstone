from cmath import sqrt
from stringprep import c22_specials
from tkinter import N
import numpy as np
import math as math
import numpy.linalg as LA
from matplotlib import pyplot as plt

folder = 'modelK_vol12_pref1e18'
output_folder = 'model_K'
number_of_bins_degree = 40
number_of_bins_radial = 10
number_of_c = 2
particles_0 = np.zeros((100000,9),float)

pie = math.pi
S_entropy = np.zeros((1001,6),float)
for i in range(1000,1001,5):  
    c = np.zeros((number_of_c),float)
    P_c = np.zeros((number_of_c),float) 
    bins = np.zeros((number_of_bins_radial*number_of_bins_degree,number_of_c),float)
    bins2 = np.zeros((number_of_bins_radial*number_of_bins_degree,number_of_c),float)
    bins3= np.zeros((number_of_bins_radial*number_of_bins_degree),float)
    bins4 = np.zeros((number_of_bins_radial*number_of_bins_degree,number_of_c),float)
    S_entropy_array = np.zeros((number_of_bins_radial*number_of_bins_degree),float)  
    eerste = str(i)
    eerste = eerste.zfill(5)
    iterat =0
    print(i)
    for j in range(0,192):
      tweede = str(j)
      tweede = tweede.zfill(4)
      print(j)
      z = np.zeros((2,8),float) 
      file = np.loadtxt('../../../../../../../../../eejit/lab/2022_januari/sinkrate/final/'+folder+'/particles/particles-'+eerste+'.'+tweede+'.gnuplot', ndmin=2)
      length_file = len(file)
      #print(file)
      # <x> <y> <id> <position> <position> <velocity> <velocity> <initial 1_mantle> <initial 3_continent> <p> <T> <initial position> <initial position> 
      for k in range(length_file):
          ID = int(file[k][2])
          particles_0[ID][0] = file[k][2] # id 1
          
          particles_0[ID][1] = file[k][0] # x 2
          particles_0[ID][2] = file[k][1] # y 3
          particles_0[ID][3] = math.sqrt(file[k][0]**2 + file[k][1]**2) / 1000 #radius_end
          
          particles_0[ID][4] = file[k][11] # ori x 5
          particles_0[ID][5] = file[k][12] # ori y 6
          particles_0[ID][6] = math.sqrt(file[k][11]**2 + file[k][12]**2) / 1000 #radius start
    if i == 1000:
        middle_rad = 4926
        for k in range (0,100000):
            if particles_0[k][6] <= middle_rad:
                if particles_0[k][4]<= 0:
                    if particles_0[k][5] <= 0:
                        particles_0[k][7] = 1
                    else:    
                        particles_0[k][7] = 1
                else:
                    if particles_0[k][5] <= 0:
                        particles_0[k][7] = 1
                    else:    
                        particles_0[k][7] = 1
            else:
                if particles_0[k][4] <= 0:
                    if particles_0[k][5] <= 0:
                        particles_0[k][7] = 2
                    else:    
                        particles_0[k][7] = 2
                else:
                    if particles_0[k][5] <= 0:
                        particles_0[k][7] = 2
                    else:    
                        particles_0[k][7] = 2

    for j in range (0,100000):
        angle = np.arctan2(particles_0[j][2],particles_0[j][1])
        #print(i,angle)
        iterat = 0
        for l in range(0,number_of_bins_radial):
            radius = 3481 + (289 * l)
            radius2 = 3481 + (289 *( l+1))
            for k in range(0,number_of_bins_degree):
                bin_angle = -1 * pie + ((2*pie/number_of_bins_degree ) *k)
                bin_angle2 = -1 * pie + (((2*pie)/number_of_bins_degree ) * (k+1))
                #print(iterat,bin_angle,bin_angle2, radius, radius2)
                iterat = k + (number_of_bins_degree * l)
                if particles_0[j][3] >= radius and particles_0[j][3] <= radius2 and angle >= bin_angle and angle <= bin_angle2:
                    particles_0[j][8] = iterat
                    for m in range(0,number_of_c):
                        if particles_0[j][7] == (m+1):
                            bins[iterat][m] = bins[iterat][m] + 1
    for j in range(0,number_of_bins_radial*number_of_bins_degree):
        for k in range(0,number_of_c):
            c[k] = bins[j][k] + c[k]
    print(c)
    total = sum(c)
    sum_Njc_Pc =0
    for k in range(0,number_of_c):
        P_c[k] = c[k] / (number_of_bins_radial*number_of_bins_degree)

    for j in range(0,number_of_bins_radial*number_of_bins_degree):
        for k in range(0,number_of_c):
            sum_Njc_Pc = (bins[j][k] / P_c[k] ) + sum_Njc_Pc
    print(sum_Njc_Pc)

    for j in range(0,number_of_bins_radial*number_of_bins_degree):
        for k in range(0,number_of_c):
            #   Ni,c / Pc (i: 1 tot bins, c: 1 tot species) voor Pj,c
            bins2[j][k] = (bins[j][k] / P_c[k]) / sum_Njc_Pc
            #   Nj,c / Pc (c: 1 tot species gesommeerd -> voor Pj
            bins3[j] = bins3[j] + ( bins[j][k] / P_c[k])
        for k in range(0,number_of_c):
            #   Nj,c / Pc ) / all part/all classes -> voor Pc,j
            bins4[j][k] = (bins[j][k] / P_c[k] )/ bins3[j] 
         
        bins3[j] = bins3[j] / sum_Njc_Pc

#print(a,b,c,d, total, totalP)

    for j in range(0,number_of_bins_radial*number_of_bins_degree):
        if bins3[j] ==0:
            bins3[j] = 0.0000000001
        for k in range(0,number_of_c):
            if bins2[j][k] == 0:
                bins2[j][k] = 0.0000000001
            if bins4[j][k] ==0:
                bins4[j][k] = 0.0000000001
            ## S full
            S_entropy[i][0] = (-1 * (bins2[j][k] * math.log(bins2[j][k]))) + S_entropy[i][0]
        ## S(location)    
        S_entropy[i][1] = (-1 * (bins3[j] * math.log(bins3[j])) )+ S_entropy[i][1] 

    for j in range(0,number_of_bins_degree*number_of_bins_radial):
        for k in range(0,number_of_c):
            ## Sj (species)
            S_entropy_array[j] =( -1 * (bins4[j][k] * math.log(bins4[j][k]))) + S_entropy_array[j]

    for j in range(0,number_of_bins_degree*number_of_bins_radial):
        ## S_location(species)
            S_entropy[i][2] = ( S_entropy_array[j] * bins3[j]) + S_entropy[i][2]

    S_entropy[i][3] = S_entropy[i][0] / (math.log(number_of_bins_radial*number_of_bins_degree))
    S_entropy[i][4] = S_entropy[i][1] / (math.log(number_of_c))
    S_entropy[i][5] = S_entropy[i][2] / (math.log(number_of_c))


    np.savetxt(''+output_folder+'/part/mixing_'+eerste+'.txt',particles_0)
    np.savetxt(''+output_folder+'/part/part_per_bin_'+eerste+'.txt',bins)
    np.savetxt(''+output_folder+'/part/Prob_part_per_bin'+eerste+'.txt',bins2)
    np.savetxt(''+output_folder+'/part/Prob_number_part_per_bin'+eerste+'.txt',bins3)
    np.savetxt(''+output_folder+'/part/Prob_classpart_per_bin'+eerste+'.txt',bins4)
    np.savetxt(''+output_folder+'/part/Entropy.txt',S_entropy)