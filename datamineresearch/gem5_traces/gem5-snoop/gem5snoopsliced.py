import os
import joblib

# Read in the msg file and extract the pairings in the data flow.
# 1 and 2 are a pair if: src1 = dest2, dest1=src2. cmd1=cmd2. if type1 is resp, type2 must be req and vice versa.
def extract_groups_from_msg_file(file_path):
    group_indices = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue  # Ignore comments
            elif line:
                parts = [part.strip() for part in line.split(':')]
                if len(parts) == 4:
                    index, src, dest, type_ = parts
                    key = tuple(sorted((src, dest)))  # Sorting to consider src, dest and dest, src as the same
                    if key not in group_indices:
                        group_indices[key] = []
                    group_indices[key].append(int(index))

    groups = set()  # Avoid duplicates
    for key, indices in group_indices.items():
        src, dest = key
        group = (src, dest, tuple(sorted(indices)))  # Include the indices
        groups.add(group)

    return sorted(groups)  # Return sorted groups


# Read in the trace file from a jbl file
def read_trace_file(trace_file):
    traces = joblib.load(trace_file)
    # traces = [['0', '9'], ['1', '2', '3', '4', '5', '6', '7', '8'], ['0', '9'],['10', '19']] 
    return traces


# Extract the sequences and output into filenames based on the src/dest

def extract_sequences(traces, groups, folder_name_prefix, start=0, end=100):
    parent_folder = f"gem5-snoop"
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Combine all sequences into one list
    all_sequences = []

    for i, trace in enumerate(traces, start=1):
        if(i<start):
            continue
        elif(i==end):
            break
        group_sequences = {group: [] for group in groups}

        # Iterate through the trace list
        for num in traces[trace]:
            # Check if the number belongs to any group
            for group in groups:
                src, dest, indices = group
                if int(num) in indices:  # Convert num to int and check if it's in group indices
                    group_sequences[group].append(int(num))

        # Append each group's sequence to all_sequences
        all_sequences.append("[")
        for group, sequence in group_sequences.items():
            if sequence:  # Check if sequence is not empty
                src, dest = group[0], group[1]
                group_name = f"{src}-{dest}"
                all_sequences.append(f"trace-{i} -- {group_name}\n")
                all_sequences.append(" ".join(map(str, sequence)) + "\n")
        all_sequences.append("]\n\n")
    # Write all sequences to a single file
    filename = os.path.join(parent_folder, f"{folder_name_prefix}-all-sequences.txt")
    with open(filename, "w") as file:
        file.writelines(all_sequences)






# Main code
if __name__ == "__main__":
    msg_file_path = "gem5_traces/defSnoop-RubelPrintFormat.msg"
    trace_file_path = "gem5_traces/totalSliced.jbl"

    groups = extract_groups_from_msg_file(msg_file_path)
    traces = read_trace_file(trace_file_path)

    extract_sequences(traces, groups, folder_name_prefix="sliced 1-1000", start=1, end=1001)
    extract_sequences(traces, groups, folder_name_prefix="sliced 1001-2000", start=1001, end=2001)

