set term pdf
set grid
set key left
set xlabel 'theta'
set output 'dynamic_topography.pdf'
set xtics pi/4

plot[0:pi][]\
'd_t_R2.ascii'  u ($1):2 w p lw .5 ps .4 t 'dyn. topo.',\
'dynamic_topography_surface.00000' u (atan2($2,$1)-pi/2):3 pt 1 ps .3 t 'ASPECT',\
0 lt -1 lw 1 

