set term pdf font "Times,12pt"
set output 'results.pdf'
set grid
set xlabel 'x'
set ylabel 'cost'
set log x
set title 'my title above the plot'


set output 'results.pdf'
plot 'results.dat' u 1:6 w lp lw 2 t 'velocity', 1e7/x lw 3 t 'fit' , 6e-11 t 'threshold'


set output 'results_a.pdf'
plot 'results.dat' u 1:6 w l lw 2 t 'velocity',
set output 'results_b.pdf'
plot 'results.dat' u 1:6 w p lw 2 t 'velocity',
set output 'results_c.pdf'
plot 'results.dat' u 1:6 w lp lw 2 t 'velocity',


set key outside
set output 'results_d.pdf'
plot 'results.dat' u 1:6 w lp lw 2 t 'velocity', 1e7/x lw 3 t 'fit' , 6e-11 t 'threshold'

set key inside 
set key bottom left 
set output 'results_e.pdf'
plot 'results.dat' u 1:6 w lp lw 2 t 'velocity', 1e7/x lw 3 t 'fit' , 6e-11 t 'threshold'


reset 
set key outside
set output 'linetypes.pdf'
plot[][]\
x+0  w l lt 0 t 'linetype 1',\
x+1  w l lt 1 t 'linetype 2',\
x+2  w l lt 2 t 'linetype 3',\
x+3  w l lt 3 t 'linetype 4',\
x+4  w l lt 4 t 'linetype 5',\
x+5  w l lt 5 t 'linetype 6',\
x+6  w l lt 6 t 'linetype 7',\
x+7  w l lt 7 t 'linetype 8',\
x+8  w l lt 8 t 'linetype 9',\
x+9  w l lt 9 t 'linetype 10',\
x+10 w l lt 10 t 'linetype 11',\
x+11 w l lt 11 t 'linetype 12'

set output 'dashtypes.pdf'
plot[][]\
x+1  w l lt 1  dt 1 t 'linetype 2',\
x+2  w l lt 2  dt 2 t 'linetype 3',\
x+3  w l lt 3  dt 3 t 'linetype 4',\
x+4  w l lt 4  dt 4 t 'linetype 5',\
x+5  w l lt 5  dt 5 t 'linetype 6',\
x+6  w l lt 6  dt 6 t 'linetype 7',\
x+7  w l lt 7  dt 7 t 'linetype 8',\
x+8  w l lt 8  dt 8 t 'linetype 9',\
x+9  w l lt 9  dt 9 t 'linetype 10',\
x+10 w l lt 10 dt 10 t 'linetype 11',\
x+11 w l lt 11 dt 11 t 'linetype 12'

set output 'pointtypes.pdf'
plot[][]\
x+0  w p pt 0  ps .5 t 'pointtype 0',\
x+1  w p pt 1  ps .5 t 'pointtype 1',\
x+2  w p pt 2  ps .5 t 'pointtype 2',\
x+3  w p pt 3  ps .5 t 'pointtype 3',\
x+4  w p pt 4  ps .5 t 'pointtype 4',\
x+5  w p pt 5  ps .5 t 'pointtype 5',\
x+6  w p pt 6  ps .5 t 'pointtype 6',\
x+7  w p pt 7  ps .5 t 'pointtype 7',\
x+8  w p pt 8  ps .5 t 'pointtype 8',\
x+9  w p pt 9  ps .5 t 'pointtype 9',\
x+10 w p pt 10 ps .5 t 'pointtype 10',\
x+11 w p pt 11 ps .5 t 'pointtype 11',\
x+12 w p pt 12 ps .5 t 'pointtype 12',\
x+13 w p pt 12 ps .5 t 'pointtype 13',\
x+14 w p pt 12 ps .5 t 'pointtype 14'

#############################################################

set output 'velocity_1.pdf'
set xlabel 'x'
set ylabel 'y'
set xtics 0.125
set ytics 0.333333333333
set grid
set size square
set title 'a) without scaling'
plot[0:1][0:1]\
'velocity.dat' u 1:2:3:4 w vectors lt -1 notitle 


set output 'velocity_2.pdf'
set xlabel 'x'
set ylabel 'y'
set xtics 0.125
set ytics 0.333333333333
set grid
set size square
set title 'b) with scaling'
plot[0:1][0:1]\
'velocity.dat' u 1:2:($3*4):($4*4) w vectors lt -1 notitle 












