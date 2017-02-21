#!/bin/bash

for (( N=700; N<=320000; N+=200 ))
do

nvprof --print-gpu-trace --csv ./mem_mem 2 0 $N 2> ./profile_results/trace_$N.csv

nvprof --metrics all --csv ./mem_mem 2 0 $N 2> ./profile_results/metrics_$N.csv

done
