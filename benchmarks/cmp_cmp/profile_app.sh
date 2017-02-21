#!/bin/bash

for (( N=5000; N<=20000; N+=100 ))
do

nvprof --print-gpu-trace --csv ./cmp_cmp 2 0 $N 2> ./profile_results/trace_$N.csv

nvprof --metrics all --csv ./cmp_cmp 2 0 $N 2> ./profile_results/metrics_$N.csv

done
