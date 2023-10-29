###############################################################################
#eta_model=0 # constant
#eta_model=1 # civs12 a
#eta_model=2 # civs12 b
#eta_model=3 # stho08
#eta_model=4 # yohk01
###############################################################################

from numba import jit
import numpy as np

@jit(nopython=True)
def viscosity(x,y,R1,R2,eta_m,eta_model):
       
    depth=R2-np.sqrt(x**2+y**2)

    #print(x,y,eta_m,eta_model)
    val=eta_m
    return val
  
    #--------------------------------------
    if eta_model==0:
       val=eta_m

    #--------------------------------------
    elif eta_model==1: #civs12a
       cell_index=49
       for kk in range(0,50):
           if depth<depths_civs12[kk+1]:
              cell_index=kk
              break
           #end if
       #end for
       val=(depth-depths_civs12[cell_index])/(depths_civs12[cell_index+1]-depths_civs12[cell_index])\
          *(viscA_civs12[cell_index+1]-viscA_civs12[cell_index])+viscA_civs12[cell_index]
       val=10**val

    #--------------------------------------
    elif eta_model==2: #civs12b
       cell_index=49
       for kk in range(0,50):
           if depth<depths_civs12[kk+1]:
              cell_index=kk
              break
           #end if
       #end for
       val=(depth-depths_civs12[cell_index])/(depths_civs12[cell_index+1]-depths_civs12[cell_index])\
          *(viscB_civs12[cell_index+1]-viscB_civs12[cell_index])+viscB_civs12[cell_index]
       val=10**val

    #--------------------------------------
    elif eta_model==3: #stho08
       cell_index=21
       for kk in range(0,22):
           if depth<depths_stho08[kk+1]:
              cell_index=kk
              break
           #end if
       #end for
       val=(depth-depths_stho08[cell_index])/(depths_stho08[cell_index+1]-depths_stho08[cell_index])\
          *(visc_stho08[cell_index+1]-visc_stho08[cell_index])+visc_stho08[cell_index]

    #--------------------------------------
    elif eta_model==4: #yohk01
       val = 3e21
       if depth < 150e3:
          val *= 1e3 
       elif depth > 670e3:
          val *= 70

    #--------------------------------------
    else:
       exit('unknown eta_model')

    return val       

###############################################################################
