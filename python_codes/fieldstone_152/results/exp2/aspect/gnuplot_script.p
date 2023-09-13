set term pdf
set grid
set key left
set xlabel 'theta'
set output 'dynamic_topography.pdf'
plot[0:pi/2][]\
'err_R2.ascii' u 4:3 ps .2 t '{/Symbol e}_{rr}',\
'qqq_R2.ascii' u 4:3 ps .2 t 'p',\
'd_t_R2.ascii' u 4:3 w lp lw .5 ps .3 t 'dyn. topo.',\
'dynamic_topography_surface.00000' u (atan2($2,$1)):3 pt 1 ps .4 t 'ASPECT',\
0 lt -1 lw 1 
