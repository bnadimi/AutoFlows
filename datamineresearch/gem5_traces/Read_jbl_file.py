# For reading a .jbl trace file you need to install joblib library
#   pip install joblib
# The file may contain several traces, when loading the file the traces will be loaded into an array 
# which each index of array is an array of message numbers
# Each message number is associated with a message in the message definition file.

import joblib

def read_trace_file(trace_file):
    traces = 0
    if '.jbl' in trace_file:
        file = joblib.load(trace_file)
        for i in file: # there might be several traces in a single file
            # print("\n****************** Trace number :", traces, "***********************\n")
            #file[i] includes all the messages in a single trace
            traces = traces + 1
    print(traces)

trace_file_path = "gem5_traces/totalSliced.jbl"
read_trace_file(trace_file_path)



def read_trace_file_txt(trace_file, output_file):
    traces = 0
    if '.jbl' in trace_file:
        file = joblib.load(trace_file)
        with open(output_file, 'w') as f:
            for i in file: # there might be several traces in a single file
                f.write("\n****************** Trace number : {} ***********************\n\n".format(traces))
                #file[i] includes all the messages in a single trace
                traces += 1
                f.write(str(file[i]) + "\n\n")
                if traces == 50:
                    break

trace_file_path = "gem5_traces/unsliced-RubelPrintFormat.jbl"
output_file_path = "output.txt"
read_trace_file_txt(trace_file_path, output_file_path)

