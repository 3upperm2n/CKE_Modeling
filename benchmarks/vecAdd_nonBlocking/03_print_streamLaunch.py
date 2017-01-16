#!/usr/bin/env python

import pandas as pd
import numpy as np
import operator
from collections import OrderedDict

#-----------------
# read input csv
#-----------------
trace_file = "nvprof_gputrace.csv"

# There are max 17 columns in the output csv
max_cols = ["Start","Duration","Grid X","Grid Y","Grid Z","Block X","Block Y","Block Z","Registers Per Thread","Static SMem","Dynamic SMem","Size","Throughput","Device","Context","Stream","Name"]

df_trace = pd.read_csv(trace_file, names=max_cols, engine='python')

rows_to_skip = 0
## find out the number of rows to skip
for index, row in df_trace.iterrows():
    if row['Start'] == 'Start':
        rows_to_skip = index
        break
# read the input csv again 
df_trace = pd.read_csv(trace_file, skiprows=rows_to_skip)


#----------
# use ms as the timing unit
#----------
rows, cols = df_trace.shape

start_unit = df_trace['Start'].iloc[0]
# print start_unit

duration_unit = df_trace['Duration'].iloc[0]
# print duration_unit

start_coef =  1.0

if start_unit == 's':
    start_coef = 1e3

if start_unit == 'us':
    start_coef = 1e-3

duration_coef =  1.0

if duration_unit == 's':
    duration_coef = 1e3
if duration_unit == 'us':
    duration_coef = 1e-3


#----------
# stream starting time 
#----------

# remove the 1st row
df = df_trace.drop(df_trace.index[[0]])

streams_ls = list(df_trace['Stream'].unique())

# remove nan
streams_ls = [x for x in streams_ls if str(x) != 'nan']
#print streams_ls

# init the dict with large values
streams_start = dict()
for x in streams_ls:
    streams_start[str(x)] = 999999999999999.0

for index, row in df.iterrows():
#    print row['Start']
#    print row['Stream']
    start_ms = start_coef * float(row['Start'])
    sid = row['Stream']
    if streams_start[str(sid)] > start_ms:
        streams_start[str(sid)] = start_ms

## streams_start time in ms
#print streams_start


# sort stream starting time  in asc order
sorted_streams_start = sorted(streams_start.items(), key=operator.itemgetter(1))
print sorted_streams_start

#----------
# compute the stream lauching overheads 
#----------
od = OrderedDict(sorted_streams_start)

prev_stream = -1
curr_stream = -1

prev_start_ms = -1
curr_start_ms = -1

streams_lnch_ovhd = []

for key, value in od.iteritems():
    if prev_stream == -1:
        prev_stream = key
        prev_start_ms = value
        print str(prev_stream) + ' 0 (ms)'

    curr_stream = key
    curr_start_ms = value
    
    if curr_stream != prev_stream:
        ovhd = curr_start_ms - prev_start_ms
        print str(curr_stream) + ' ' + str(ovhd) + ' (ms)'
        streams_lnch_ovhd.append(ovhd)

        #update prev
        prev_stream = curr_stream
        prev_start_ms = curr_start_ms

#---- compute the avg/min/max ----
print 'min : ' + str(min(streams_lnch_ovhd)) + ' ms'
print 'max : ' + str(max(streams_lnch_ovhd)) + ' ms'
print 'avg : ' + str(sum(streams_lnch_ovhd) / float(len(streams_lnch_ovhd))) + ' ms'

