#!/bin/bash

nvprof --print-gpu-summary "$@" 

nvprof --print-gpu-trace "$@" 

nvprof --print-gpu-trace  --csv  "$@" 2> nvprof_gputrace.csv
