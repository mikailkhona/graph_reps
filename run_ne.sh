#/bin/bash

niter=15000
N=31

for Ne in 8 24;
do 
    p=0.1
    python3 train.py $niter $N $p $Ne 0 spoke & 
    pid1=$!
    python3 train.py $niter $N $p $Ne 1 spoke &
    pid2=$!
    wait $pid1
    wait $pid2
done



