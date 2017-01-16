#!/usr/bin/env python

import pandas as pd
import numpy as np

trace_file = "nvprof_gputrace.csv"

## skip the 1st three rows
df_trace = pd.read_csv(trace_file, skiprows=3)

rows, cols = df_trace.shape
# print rows
# print cols

start_unit = df_trace['Start'].iloc[0]
# print start_unit

duration_unit = df_trace['Duration'].iloc[0]
# print duration_unit


start_coef =  1.0

if start_unit == 's':
    start_coef = 1e3
    
if start_unit == 'us':
    start_coef = 1e-3
    
# print start_coef


duration_coef =  1.0

if duration_unit == 's':
    duration_coef = 1e3
    
if duration_unit == 'us':
    duration_coef = 1e-3
    
# print duration_coef

# read the row 1 for the starting time
start_time = df_trace['Start'].iloc[1]
# print type(start_time)
start_time = float(start_time) * start_coef
# print start_time


# read the last row  for the starting time
end_time = df_trace['Start'].iloc[rows-1]
end_dur_time = df_trace['Duration'].iloc[rows-1]
# print type(end_time)
# print type(end_dur_time)
# print end_time
# print end_dur_time
end_time = float(end_time) * start_coef
end_dur_time = float(end_dur_time) * duration_coef

end_time = end_time + end_dur_time

# print end_time

#---------------------------------
# total runtime
#---------------------------------
runtime = end_time - start_time
print "runtime : " + str(runtime) + " ms"
