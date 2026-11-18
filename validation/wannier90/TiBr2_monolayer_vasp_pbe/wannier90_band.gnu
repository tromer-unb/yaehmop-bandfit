set style data dots
set nokey
set xrange [0: 3.67542]
set yrange [-58.47135 : 14.12439]
set arrow from  1.22515, -58.47135 to  1.22515,  14.12439 nohead
set arrow from  2.45019, -58.47135 to  2.45019,  14.12439 nohead
set xtics ("G"  0.00000,"K"  1.22515,"K'"  2.45019,"G"  3.67542)
 plot "wannier90_band.dat"
