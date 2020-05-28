# making a .geo file for GMSH, so that we can produce a mesh file (with .msh extension)

import numpy as np

## ----- read in desired subduction interface geometry -----

a = np.loadtxt('geometry_van_keken.txt')
slab_x = a[:,0] # x-coordinates of the slab interface (convention: positive x-axis to the right)
slab_y = a[:,1] # y-coordinates of the slab interface (convention: positive y-axis is down)

## ----- INPUT (setting up the model) -----

filename = 'vankeken.geo'

# converting km to m
slab_x = slab_x*1e3
slab_y = -slab_y*1e3 # convert the convention of the y-axis to y-axis up (as used in  fieldstone)

# model setup dimensions
min_x           = np.amin(slab_x)
max_x           = 660e3
min_y           = np.amin(slab_y)

width_channel           = 5e3   # width of the subduction channel
lithosphere_thickness   = 50e3    # thickness of the overriding plate lithosphere
# to avoid any meshing problems in the top left corner of the model where the subduction geometry can be quite shallow, we impose a higher resolution there. Here we define to which depth this resolution is applied
depth_high_res_trench   = 1.5e3   

# resolution 
lc_normal       = 40e3              # default resolution defined at far-away points (e.g., lower left corner)
lc_top          = 20e3              # resolution at the surface
lc_interface    = width_channel     # resolution in the subduction channel
lc_litho        = 15e3              # resolution of points at the bottom of the overriding lithosphere
lc_left_corner  = lc_interface      # resolution at the top left corner 

## ----- start on .geo file -----

id_file  = open(filename,'w+')

nr_interface_points = np.amax(slab_x.shape)

# write resolutions to file
id_file.write('// Resolutions \n')
id_file.write('lc_normal     = %s; \n' %lc_normal)
id_file.write('lc_top        = %s; \n' %lc_top)
id_file.write('lc_interface  = %s; \n' %lc_interface)
id_file.write('lc_litho      = %s; \n' %lc_litho)
id_file.write('lc_left_corner= %s; \n' %lc_left_corner)
id_file.write('\n')

## ----- make all the points in the domain -----

point_number = 0
line_number  = 0

# make all the points
id_file.write('// Subduction interface points \n')
for i in range(0,nr_interface_points):
    point_number = point_number+1
    if slab_y[i] > -depth_high_res_trench:
        id_file.write('Point(%d) = {%s,%s,%s,lc_left_corner}; \n' %(point_number,slab_x[i],slab_y[i],0.0))
    else:
        id_file.write('Point(%d) = {%s,%s,%s,lc_interface}; \n' %(point_number,slab_x[i],slab_y[i],0.0))

id_file.write('\n')

id_file.write('// Anchor points for domain \n')
for i in range(0,4):
    point_number = point_number + 1
    if i == 0:
    # bottom left corner
        id_file.write('Point(%d) = {%s,%s,%s,lc_normal}; \n' %(point_number,slab_x[0],slab_y[-1],0.0))
    elif i==1:
    # top right corner 
        id_file.write('Point(%d) = {%s,%s,%s,lc_top}; \n' %(point_number,max_x,slab_y[0],0.0))
    elif i==2:
    # overriding plate lithosphere thickness 
        id_file.write('Point(%d) = {%s,%s,%s,lc_litho}; \n' %(point_number,max_x,-lithosphere_thickness,0.0))
    elif i==3:
    # bottom right corner 
        id_file.write('Point(%d) = {%s,%s,%s,lc_normal}; \n' %(point_number,max_x,slab_y[-1],0.0))

id_file.write('\n')

## ----- create the points for the 'subduction channel', which we need in order for the numerics to work -----

channel_x = np.empty([np.amax(slab_x.shape),1])
channel_y = np.empty([np.amax(slab_x.shape),1])
remove_index = []

counter = 0
for i in range(1,nr_interface_points-1):
    angle = np.degrees(np.arccos((slab_x[i+1]-slab_x[i-1]) / (np.sqrt( np.square(slab_x[i+1]-slab_x[i-1]) + np.square(slab_y[i+1]-slab_y[i-1]) ))))
    channel_x[i] = slab_x[i] + width_channel*np.sin(np.radians(angle))
    channel_y[i] = slab_y[i] + width_channel*np.cos(np.radians(angle))
    if angle < 45:
        if channel_y[i] > 0:
            remove_index.append(i)
            counter = counter+1

# fix x-coordinate of point at y = 0 
if channel_y[1] < 0: 
    channel_x[0] = slab_x[0] + width_channel*np.sin(np.radians(angle))
    remove_index.append(0)
    shallow_point = 1
else:
    shallow_point = np.amax(remove_index)

shift_x1 = ((slab_x[shallow_point] - slab_x[shallow_point-1]) * (channel_y[shallow_point])) / (slab_y[shallow_point] - slab_y[shallow_point-1])

channel_x[0] = channel_x[shallow_point] - shift_x1

if remove_index[0] > 0:
    channel_x[remove_index] = []
    channel_y[remove_index] = []

# fix x-coordinate of point at y = y_max 
channel_y[-1] = min_y

shift_x2 = ((slab_x[-1] - slab_x[-2]) * (channel_y[-1] - channel_y[-2])) / (slab_y[-1] - slab_y[-2]) 
channel_x[-1] = channel_x[-2] + shift_x2

nr_channel_points = np.amax(channel_x.shape)

# transform the lists into arrays, so that we can more easily write them to file 
channel_x = np.array([ elem for singleList in channel_x for elem in singleList])
channel_y = np.array([ elem for singleList in channel_y for elem in singleList])

# check if there is a node that coincides with the thickness of the overriding plate 

for i in range (0,nr_channel_points-1):
    if channel_y[i] == -lithosphere_thickness:
        extra_node = 0
    elif channel_y[i] > -lithosphere_thickness and channel_y[i+1] < -lithosphere_thickness:
        extra_node = 1
        index_extra_node = i+1

if extra_node == 1:
# add a node for a flat base of the overriding plate
    channel_x = np.insert(channel_x,index_extra_node,channel_x[index_extra_node-2] + ( (channel_x[index_extra_node] - channel_x[index_extra_node-2]) * (lithosphere_thickness - np.abs(channel_y[index_extra_node-2])) ) / (np.abs(channel_y[index_extra_node]) - np.abs(channel_y[index_extra_node-2])))
    channel_y = np.insert(channel_y,index_extra_node,-lithosphere_thickness)   

## ----- write points channel interface to file -----

nr_channel_points = np.amax(channel_x.shape)

# make all the points
id_file.write('// Write channel interface points \n')
for i in range(0,nr_channel_points):
    point_number = point_number+1
    if channel_y[i] > -depth_high_res_trench:
        if i == 0:
            id_file.write('Point(%d) = {%s,%s,%s,lc_left_corner}; \n' %(point_number,channel_x[i],0.0,0.0))
        else: 
            id_file.write('Point(%d) = {%s,%s,%s,lc_left_corner}; \n' %(point_number,channel_x[i],channel_y[i],0.0))
    else:
        id_file.write('Point(%d) = {%s,%s,%s,lc_interface}; \n' %(point_number,channel_x[i],channel_y[i],0.0))

id_file.write('\n')

## ----- make the lines of the box -----

nr_lines_domain_box = 8

id_file.write('// Make lines for left box domain \n')
for i in range(0,nr_lines_domain_box):
    line_number = line_number + 1
    if i == 0:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,1,nr_interface_points+5))
    elif i == 1:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,nr_interface_points+5,nr_interface_points+2))
    elif i==4:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,nr_interface_points+4,nr_interface_points+nr_channel_points+4))
    elif i==5:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,nr_interface_points+nr_channel_points+4,nr_interface_points))
    elif i==6:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,nr_interface_points,nr_interface_points+1))
    elif i==7:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,nr_interface_points+1,1))
    elif i < 4:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,nr_interface_points+line_number-1,nr_interface_points+line_number))
 
id_file.write('\n')


## ----- add megathrust geometry lines -----

nr_lines_interface = nr_interface_points-1
start_v_lines = []

counter_v = 0
# make all the lines    
id_file.write('// Make lines for interface \n')
for i in range(0,nr_lines_interface):
    line_number = line_number + 1
    if i<nr_lines_interface:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,line_number-8,line_number-7))
        counter_v = counter_v + 1
        start_v_lines.append(line_number)
 
id_file.write('\n') 

## ----- add channel interface lines -----

nr_lines_channel = nr_channel_points
start_c_lines = []

counter_c = 0
# make all the lines    
id_file.write('// Make lines for channel interface \n')
for i in range(0,nr_lines_channel-1):
    line_number = line_number + 1
    if i<nr_lines_channel:
        id_file.write('Line(%d) = {%d,%d}; \n' %(line_number,line_number-3,line_number-2))
        counter_c = counter_c + 1
        start_c_lines.append(line_number)
        if line_number - 3 == index_extra_node + 8 + nr_interface_points-1:
            last_line_channel_overriding_plate = line_number-3 

## ----- find out where to connect lithosphere point -----

for i in range(0,nr_channel_points-1):
   if np.abs(channel_y[i-1]) <= np.abs(lithosphere_thickness) and np.abs(channel_y[i]) > np.abs(lithosphere_thickness):
      index_lithosphere_connection = i 

id_file.write('Line(%d) = {%d,%d}; \n' %(line_number+1,index_lithosphere_connection+nr_interface_points+4,nr_interface_points+3))
id_file.write('\n') 

## ----- add a flag to know where the exact interface is where the velocities should be imposed -----
# 101

# transform lists into arrays 
start_v_lines = np.array(start_v_lines)
start_c_lines = np.array(start_c_lines)

# find number of lines that should have a velocity imposed 
nr_prescribed_v = np.amax(start_v_lines.shape)
id_file.write('// Add flag for prescribed velocities \n')
for i in range(0,nr_prescribed_v):
    nr_line = start_v_lines[i]
    if i == 0:
        id_file.write('Physical Line(101) = {%d' %nr_line)
    elif i == nr_prescribed_v-1:
        id_file.write(',%d};' %nr_line)
    else:
        id_file.write(',%d' %nr_line)

id_file.write('\n')
id_file.write('\n')

## ----- add a flag to know where the subduction interface is that connects to the overriding plate and a flag for the rest of this line -----
# 102 # 103

# find number of lines that should have a velocity imposed 
nr_prescribed_c = np.amax(start_c_lines.shape)
id_file.write('// Add flags for subduction channel \n')
for i in range(0,nr_prescribed_c):
    nr_line = start_c_lines[i]
    if i == 0:
        id_file.write('Physical Line(102) = {%d' %nr_line)
    elif start_c_lines[i] == last_line_channel_overriding_plate:
        id_file.write(',%d};' %nr_line)
    elif start_c_lines[i] == last_line_channel_overriding_plate + 1:
        id_file.write('\n')
        id_file.write('\n')
        id_file.write('Physical Line(103) = {%d' %nr_line)
    elif i == nr_prescribed_c-1:
        id_file.write(',%d};' %nr_line)
    else:
        id_file.write(',%d' %nr_line)

id_file.write('\n')
id_file.write('\n')

## ----- add a flag to know where the base of the overriding lithosphere is ----- 
# 104

line_nr_litho = nr_lines_interface + nr_lines_domain_box + nr_lines_channel
id_file.write('// Add flag for bottom overriding plate \n')
id_file.write('Physical Line(104) = {%d};' %line_nr_litho)
id_file.write('\n')
id_file.write('\n')

## ----- add a flag for the top of the model ----- 
# 111

lines_top    = 2
lines_right  = 4
lines_bottom = 7
lines_left   = 8

counter = 1
id_file.write('// Add flag for top boundary \n')
for i in range(0,lines_top):
    if i == 0:
        id_file.write('Physical Line(111) = {%d' %counter)
    elif i == lines_top-1:
        id_file.write(',%d};' %counter)
    else:
        id_file.write(',%d' %counter)
 
    counter = counter + 1

id_file.write('\n')
id_file.write('\n')

## ----- add a flag for the right hand side of the model -----
# 112

id_file.write('// Add flag for right boundary \n')
for i in range(lines_top,lines_right):
    if i == lines_top:
        id_file.write('Physical Line(112) = {%d' %counter)
    elif i == lines_right-1:
        id_file.write(',%d};' %counter)
    else:
        id_file.write(',%d' %counter)
    counter = counter + 1
 
id_file.write('\n')
id_file.write('\n')

## ----- add a flag for the bottom of the model -----
# 113

id_file.write('// Add flag for bottom boundary \n')
for i in range(lines_right,lines_bottom):
    if i == lines_right:
        id_file.write('Physical Line(113) = {%d' %counter)
    elif i == lines_bottom-1:
        id_file.write(',%d};' %counter)
    else:
        id_file.write(',%d' %counter)
    counter = counter + 1
 
id_file.write('\n')
id_file.write('\n')

## ----- add a flag for the left hand side of the model -----
# 114

id_file.write('// Add flag for left boundary \n')
for i in range(lines_bottom,lines_left):
    if lines_bottom+1 == lines_left:
        id_file.write('Physical Line(114) = {%d};' %counter)
    elif i == lines_bottom:
        id_file.write('Physical Line(114) = {%d' %counter)
    elif i == lines_left-1:
        id_file.write(',%d};' %counter)
    else:
        id_file.write(',%d' %counter)
counter = counter + 1

id_file.write('\n')
id_file.write('\n')

## ----- make line loop: left - interface ----- 

nr_lines = nr_lines_interface + 1

for qr in range(0,nr_lines+1):
    if qr == 0:
        ish = 8
        id_file.write('Line Loop(10) = {%d' %-ish)
    elif qr == 1:
        ish = 7
        id_file.write(',%d' %-ish)
    elif qr < nr_lines:
        ish = nr_interface_points+8-qr+1
        id_file.write(',%d' %-ish)
    elif qr == nr_lines:
        ish = nr_interface_points+8-qr+1
        id_file.write(',%d};' %-ish)
 
id_file.write(' \n')
id_file.write(' \n')

## ----- make line loop: subduction channel -----

nr_lines = nr_lines_interface + nr_lines_channel

counter = 0
for qr in range(0,nr_lines+1):
    if qr == 0:
        ish = qr+9
        id_file.write('Line Loop(20) = {%d' %ish)
    elif qr < nr_lines_interface:
        ish = qr+9
        id_file.write(',%d' %ish)
    elif qr == nr_lines_interface :
        ish = 6
        id_file.write(',%d' %-ish)
    elif qr < nr_lines:
        ish = nr_interface_points+nr_channel_points+6 - counter
        id_file.write(',%d' %-ish)
        counter = counter + 1
    elif qr == nr_lines:
        ish = 1
        id_file.write(',%d};' %-ish)

id_file.write(' \n')

## ----- make line loop: subduction channel - overriding plate -----

id_file.write(' \n')

nr_lines = index_lithosphere_connection + 2

counter = 0
for qr in range(0,nr_lines+1):
    if qr == 0:
        ish = nr_interface_points + 8 + qr
        id_file.write('Line Loop(30) = {%d' %ish)
    elif qr < index_lithosphere_connection-1:
        ish = nr_interface_points + 8 + qr
        id_file.write(',%d' %ish)
    elif qr == index_lithosphere_connection-1:
        ish = nr_interface_points + nr_channel_points +7
        id_file.write(',%d' %ish)
    elif qr <= index_lithosphere_connection:
        ish = 3
        counter = counter + 1
        id_file.write(',%d' %-ish)
    elif qr == nr_lines:
        ish = 2
        id_file.write(',%d};' %-ish)

id_file.write(' \n')

## ----- make line loop: mantle wedge -----

id_file.write(' \n')

nr_lines = nr_channel_points - index_lithosphere_connection + 3

counter = 0
for qr in range(0,nr_lines):
    if qr == 0:
        ish = index_lithosphere_connection + nr_interface_points + 7 + qr
        id_file.write('Line Loop(40) = {%d' %ish)
    elif qr <= nr_channel_points - index_lithosphere_connection-1:
        ish = index_lithosphere_connection + nr_interface_points + 7 + qr
        id_file.write(',%d' %ish)
    elif qr < nr_lines-1:
        ish = 5 - counter 
        id_file.write(',%d' %-ish)
        counter = counter + 1
    elif qr == nr_lines-1:
        ish = nr_lines_channel + nr_lines_interface + 8 
        id_file.write(',%d};' %-ish)

id_file.write(' \n')

## ----- make plane surface -----

id_file.write(' \n')
id_file.write('// Make plane surface \n')
counter = 0
for i in range(10,50,10):
    counter = counter + 1
    id_file.write('Plane Surface(%d) = {%d}; \n' %(counter*1000000,i))
 
id_file.write(' \n')

# make physical surface
nr_surfaces = 4
id_file.write('// Make physical surface \n')
for i in range(1,nr_surfaces+1):
    if i == 1:
        id_file.write('Physical Surface(0) = {%d' %(i*1000000))
    elif i == nr_surfaces:
        id_file.write(',%d};' %(i*1000000))
    else:
        id_file.write(',%d' %(i*1000000))

id_file.write('\n')
id_file.write('\n')

# add this line to ensure that the subduction interface is only one element wide 
id_file.write('MeshAlgorithm Surface{2000000} = 3;')

id_file.close()
