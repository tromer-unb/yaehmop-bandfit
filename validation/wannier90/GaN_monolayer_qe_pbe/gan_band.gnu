set style data dots
set nokey
set xrange [0: 3.08688]
set yrange [-19.60149 :  8.57368]
set arrow from  1.30462, -19.60149 to  1.30462,   8.57368 nohead
set arrow from  1.95693, -19.60149 to  1.95693,   8.57368 nohead
set xtics ("G"  0.00000,"K"  1.30462,"M"  1.95693,"G"  3.08688)
 plot "gan_band.dat"
