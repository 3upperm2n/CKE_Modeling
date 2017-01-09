#!/bin/bash

nvprof --print-gpu-trace  --csv  "$@" 2> nvprof_gputrace.csv
