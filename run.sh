#/bin/bash

niter=10000
Ne=256

for i in "20 0.2" "50 0.1" "75 0.1"; 
do 
    set -- $i
    N=$1
    p=$2
    python3 train.py $niter $N $p $Ne 0 bernoulli & 
    pid1=$!
    python3 train.py $niter $N $p $Ne 1 bernoulli &
    pid2=$!
    wait $pid1
    wait $pid2
done

for N in 11 31 51; 
do 
    p=0.1
    python3 train.py $niter $N $p $Ne 0 spoke & 
    pid1=$!
    python3 train.py $niter $N $p $Ne 1 spoke &
    pid2=$!
    wait $pid1
    wait $pid2
done

for N in 11 31 51; 
do 
    p=0.1
    python3 train.py $niter $N $p $Ne 0 ring & 
    pid1=$!
    python3 train.py $niter $N $p $Ne 1 ring &
    pid2=$!
    wait $pid1
    wait $pid2
done



