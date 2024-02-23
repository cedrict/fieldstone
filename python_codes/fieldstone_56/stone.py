import random
import numpy as np
from scipy.ndimage import gaussian_filter

###################################################################################################

def generate_fractal_map(size, roughness, sigma, seed=None):
    """
    Generates a square fractal map using the diamond-square algorithm, followed by Gaussian filtering.

    The diamond-square algorithm is a fractal method used to generate random landscapes or textures.
    The algorithm works by dividing the map into squares and diamonds, then randomly displacing the midpoint
    of each to create roughness. Gaussian filtering is applied at the end to smooth out the map for a more
    natural look.

    :param size: Determines the size of the map which will be `(2^size) + 1`.
    :type size: scalar(int)
    :param roughness: The initial roughness factor for the diamond-square algorithm.
    :type roughness: scalar(float)
    :param sigma: Standard deviation for Gaussian filter, affecting the smoothness.
    :type sigma: scalar(float)
    :param seed: Optional seed for the random number generator.
    :type seed: (optional(scalar(int)))

    :return: **dem** *(array_like(float))* - A 2D numpy array representing the fractal map.
    """

    # Print out the parameters for the map generation
    print(f"size art DEM (2^n+1 nodes): {2**size+1}")
    print(f"roughness art DEM: {roughness}")
    print(f"filter sigma: {sigma}")
    def diamond_square(map_, stepsize, roughness):
        """
        Perform the diamond square procedure on a map to generate terrain elevation.

        The function takes a 2D numpy array representing an elevation map, a step size, and a roughness factor.
        It updates the map in place, adding randomness to create a fractal pattern.

        Parameters:
        :param map_: The elevation map, a 2D numpy array that is modified in place.
        :type map_: array_like(float)
        :param stepsize: The size of the square or diamond step in the algorithm. It determines the distance between points that will be updated.
        :type stepsize: scalar(int)
        :param roughness: A roughness factor determining the range of random values added to the midpoints.
        :type roughness: scalar(float)

        :return: None: The function modifies the input array `map_` in place and has no return value.
        """

        # Calculate half of the stepsize, which will be used to offset coordinates
        half_step = int(stepsize / 2)

        # Perform the diamond step
        for y in range(half_step, len(map_), stepsize):
            for x in range(half_step, len(map_[0]), stepsize):
                # Calculate the average of the corners of the square
                square_averages = np.mean(
                    [
                        map_[y - half_step][x - half_step],
                        map_[y - half_step][x + half_step],
                        map_[y + half_step][x - half_step],
                        map_[y + half_step][x + half_step],
                    ]
                )
                # Set the middle of the square to average plus some randomness based on roughness
                map_[y][x] = square_averages + random.uniform(-roughness, roughness)

        # Perform the square step
        for y in range(0, len(map_), half_step):
            for x in range((y + half_step) % stepsize, len(map_[0]), stepsize):
                # Calculate the average of the points of the diamond
                #  (considering wrap-around for edges)
                diamond_averages = np.mean(
                    [
                        map_[y][x],
                        map_[(y - half_step) % len(map_)][x],
                        map_[(y + half_step) % len(map_)][x],
                        map_[y][(x - half_step) % len(map_[0])],
                        map_[y][(x + half_step) % len(map_[0])],
                    ]
                )

                # Only update the point if it is within the bounds of the map
                if y < len(map_) and x < len(map_[0]):
                    # Set the diamond point to the average plus some randomness based on roughness
                    map_[y][x] = diamond_averages + random.uniform(-roughness, roughness)

    # Set the random seed if provided for reproducibility
    if seed is not None:
        random.seed(seed)
        print(f"random seed: {seed}")

    # Calculate the actual size of the map
    size = 2 ** size + 1

    # Initialize the map with zeros and set the corner values with initial roughness
    map_ = [[0] * size for _ in range(size)]
    map_[0][0] = map_[0][size-1] = map_[size-1][0] = map_[size-1][size-1] \
        = random.uniform(-roughness, roughness)

    # Set the initial step size to the size of the map minus one
    stepsize = size - 1

    # A flag for determining how to decrease roughness, can be toggled for different effects
    linear = False

    # Loop through the diamond-square steps to create terrain until the step size is small enough
    while stepsize >= 2:
        # Apply the diamond-square algorithm with current step size and roughness

        diamond_square(map_, stepsize, roughness)
        # Decrease the step size for the next iteration
        stepsize = int(stepsize / 2)

        # Decrease roughness as the features become smaller, this can be linear or exponential
        if linear:
           roughness *= 0.5
        else:
           roughness *= np.exp(-1 / stepsize)

    # Once the fractal generation is complete, apply a Gaussian filter to smooth the map
    map_ = gaussian_filter(map_, sigma=sigma)

    return map_

###################################################################################################

print("-----------------------------")
print("---------- stone 56 ---------")
print("-----------------------------")

size=8
roughness=100
sigma=10
L=1e3


mapp = generate_fractal_map(size, roughness, sigma, seed=None)

print(np.shape(mapp))

Nx=2**size+1
Ny=2**size+1
N=Nx*Ny
nelx=Nx-1
nely=Ny-1
nel=nelx*nely
hx=L/nelx
hy=L/nely

#for i in range(0,Nx):
#    for j in range(0,Ny):
#        print(i,j,mapp[i,j])


print('domain is',L,'x',L)
print('Nx,Ny,N=',Nx,Ny,N)
print('nelx,nely,nel=',nelx,nely,nel)
print('hx,hy=',hx,hy)
print("-----------------------------")

###################################################################################################

x = np.empty(N,dtype=np.float64)  # x coordinates
y = np.empty(N,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,Ny):
    for i in range(0,Nx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1
    #end for
#end for

icon =np.zeros((4,nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter+=1
    #end for
#end for

###################################################################################################

if True: 
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   counter = 0
   for j in range(0,Ny):
       for i in range(0,Nx):
           vtufile.write("%10f %10f %10f \n" %(x[counter],y[counter],mapp[j,i]))
           counter+=1
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='topo' Format='ascii'> \n")
   for j in range(0,Ny):
       for i in range(0,Nx):
           vtufile.write("%10f \n" %mapp[j,i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

###################################################################################################
