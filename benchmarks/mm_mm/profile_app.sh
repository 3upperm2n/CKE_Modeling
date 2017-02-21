#!/bin/bash

for (( N=192; N<=960; N+=32 ))
do

nvprof --print-gpu-trace --csv ./mm -device=0 -wA=$N -hA=$N -wB=$N -hB=$N 2> ./profile_results/trace_$N.csv

nvprof --metrics all --csv ./mm  -device=0 -wA=$N -hA=$N -wB=$N -hB=$N 2> ./profile_results/metrics_$N.csv

done
