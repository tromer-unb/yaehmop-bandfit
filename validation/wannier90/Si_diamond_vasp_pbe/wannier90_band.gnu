set style data dots
set nokey
set xrange [0: 5.19242]
set yrange [ -7.49961 : 75.92053]
set arrow from  1.15712,  -7.49961 to  1.15712,  75.92053 nohead
set arrow from  1.56623,  -7.49961 to  1.56623,  75.92053 nohead
set arrow from  2.79355,  -7.49961 to  2.79355,  75.92053 nohead
set arrow from  3.79564,  -7.49961 to  3.79564,  75.92053 nohead
set arrow from  4.61385,  -7.49961 to  4.61385,  75.92053 nohead
set xtics ("G"  0.00000,"X"  1.15712,"U|K"  1.56623,"G"  2.79355,"L"  3.79564,"W"  4.61385,"X"  5.19242)
 plot "wannier90_band.dat"
