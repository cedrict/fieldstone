set term pdf
set grid
set key left
set title 'axisymmetric geometry'
set xlabel '{/Symbol q}'

set output 'dynamic_topography.pdf'
plot[-pi/2:pi/2][]\
'dynamic_topography_surface.00000' u (atan2($3,sqrt($1**2+$2**2))):4 pt 1 ps .12 t 'ASPECT (4+3)',\
'dynamic_topography_surface.5_2' u (atan2($3,sqrt($1**2+$2**2))):4   pt 1 ps .12 t 'ASPECT (5+2)',\
'd_t_R2.ascii' u 4:($3) w l lw .5 t 'dyn. topo. (nodal)',\
0 lt -1 lw 1 

set ytics 5e-5
set output 'dynamic_topography_zoom.pdf'
plot[pi/4:pi/2][]\
'dynamic_topography_surface.00000' u (atan2($3,sqrt($1**2+$2**2))):4 pt 1 ps .12 t 'ASPECT (4+3)',\
'dynamic_topography_surface.5_2' u (atan2($3,sqrt($1**2+$2**2))):4   pt 1 ps .12 t 'ASPECT (5+2)',\
'd_t_R2.ascii' u 4:($3) w l lw .5 t 'dyn. topo. (nodal)',\
0 lt -1 lw 1 

