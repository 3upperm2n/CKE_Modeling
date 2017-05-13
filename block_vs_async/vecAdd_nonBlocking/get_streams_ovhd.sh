#!/bin/bash

# read commandline
for (( i=2; i<=6; i++ )) 
do
	./01_gpuTrace.sh "$@" $i 0
	./03_print_streamLaunch.py 
done

