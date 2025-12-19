import torch
from torch_geometric.data import Data
from collections import Counter
import os
import itertools
import joblib

"""
Reads messages from a specified file and organizes them into sections.
    Parameters:
    - file_path: Path to the file containing the messages.
    
    Returns:
    - messages: A list of messages, each represented as a tuple (index, src, dest).
    - sections: A list of sections, each containing a list of message strings.
    
"""
def read_messages_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip().split('#')
    sections = [section.strip().split('\n') for section in content if section.strip()]
    messages = []
    for section in sections:
        for line in section:
            if line:
                parts = line.split(':')
                index, src, dest = int(parts[0].strip()), parts[1].strip(), parts[2].strip()
                messages.append((index, src, dest))
    return messages, sections


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





#Given a file, add all causal pairs to a list
def find_causal_pairs(file_path):
    pairs = set()  # Using a set to automatically handle duplicate pairs

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace
            if line:  # Check if the line is not empty
                try:
                    first, second = map(int, line.split('_'))
                    pairs.add((first, second))
                except ValueError as e:
                    print(f"Skipping line due to error: {e}")

    return pairs



"""
Constructs a causality graph from the given messages.
    
    Parameters:
    - messages: A list of messages, each represented as a tuple (index, src, dest).
    - sections: A list of sections from the file, used to determine initial and terminating nodes.
    
    Returns:
    - A Data object representing the graph, sets of initial and terminating nodes, and a dictionary of successors.
    """

def construct_causality_graph(messages, sections):
    edge_list = []
    initial_nodes = set(int(line.split(':')[0].strip()) for line in sections[0] if line)
    terminating_nodes = set(int(line.split(':')[0].strip()) for line in sections[-1] if line)
    successors_dict = {}  # Adjusted to hold more detailed info
    
    for index, src, dest in messages:
        if index not in successors_dict:
            successors_dict[index] = {'src': src, 'dest': dest, 'successors': []}
        for _, s, d in messages:
            if dest == s and index not in terminating_nodes and _ not in initial_nodes:
                edge_list.append((index, _))
                successors_dict[index]['successors'].append(_)
                if _ not in successors_dict:
                    successors_dict[_] = {'src': s, 'dest': d, 'successors': []}

    edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long)
    return Data(edge_index=edge_index), initial_nodes, terminating_nodes, successors_dict



"""
Reads a trace from a file and returns it as a list of integers.

Parameters:
- file_path: Path to the file containing the trace.

Returns:
- A list of integers representing the trace.
"""

def read_trace_from_file(file_path):
    with open(file_path, 'r') as file:
        trace = file.read().strip().split()
    return [int(index) for index in trace]

def read_trace_file(trace_file):
    traces = joblib.load(trace_file)
    return [int(index) for index in traces[0]]

"""
Finds binary patterns within a trace for a given group.
    Parameters:
    - trace: List of integers representing the trace.
    - successors_dict: Dictionary mapping each index to its successors.
    - group_name: Name of the group to find patterns for.
    - groups: List of groups to search within.
    
    Returns:
    - pair_acceptance_ratios: List of tuples representing pairs and their acceptance ratios.
"""

def find_binary_pattern(trace, successors_dict, file):
    
    print("initial trace", trace)

    # Generate routes for the given group name
    causal_pairs = find_causal_pairs(file)
    print(causal_pairs)

    # List to track acceptance ratios for each pair
    pair_acceptance_ratios = []

    while causal_pairs:
        used_numbers = set()
        remaining_trace = trace    # Copy the trace for each pass
        pairs_to_remove = []

        for pair_index, pair in enumerate(causal_pairs):
            print(f"\nTrying pair {pair_index + 1}/{len(causal_pairs)}: {pair}")

            # Skip pair if it has already used numbers
            if pair[0] in used_numbers or pair[1] in used_numbers:
                continue

            updated_remaining_trace = remove_binary_pattern(pair, remaining_trace)
           
            remaining_count = Counter(updated_remaining_trace)
            original_count = Counter(remaining_trace)
            orphans = remaining_count[pair[0]] + remaining_count[pair[1]]
            original = original_count[pair[0]] + original_count[pair[1]]

            # Calculate acceptance ratio for the pair
            if(original!=0):
                acceptance_ratio = 1 - (orphans / original)
                print("Acceptance ratio:", acceptance_ratio)
            else:
                acceptance_ratio = 0
                print("pair",pair, "dne")

            # Append the pair and its acceptance ratio to the list
            pair_acceptance_ratios.append((pair, acceptance_ratio))

            # Update used numbers and remaining trace
            used_numbers.update(pair)
           

            # Mark pair for removal
            pairs_to_remove.append(pair)

        # Remove used pairs from the list
        causal_pairs = [pair for pair in causal_pairs if pair not in pairs_to_remove]

    print(pair_acceptance_ratios, len(pair_acceptance_ratios))
    return pair_acceptance_ratios


"""
Computes common causal pairs from traces in a folder and writes results to an output folder.
    Parameters:
    - folder_path: Path to the folder containing trace files.
    - output_folder: Path to the output folder to write results.
    - successors_dict: Dictionary mapping each index to its successors.
    - groups: List of groups to search within.
"""
def compute_common_causal_pairs(trace_file, output_folder, successors_dict, file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    
    trace = read_trace_file(trace_file)
       

    # Find pairs for the extracted group name
    causal_pairs_info = find_binary_pattern(trace, successors_dict, file)

    # Filter and sort pairs based on acceptance ratio
    # filtered_pairs = [(pair, ratio) for pair, ratio in causal_pairs_info if ratio > 0.5]
    filtered_pairs = [(pair, ratio) for pair, ratio in causal_pairs_info]
    sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)

    # Write results to a separate file
    output_file_path = os.path.join(output_folder, f"componentPatterns.txt")
    number = 1
    with open(output_file_path, 'w') as f:
        f.write(f"{'-' * 25}\n")
        for pair, acceptance_ratio in sorted_pairs:
            f.write(f"{number}. {pair}, Acceptance Ratio: {acceptance_ratio}\n")
            number += 1

        f.write(f"{'-' * 25}\n")
        print("Processing completed for file:", file_path)

    print(f"Results written to {output_folder}")



"""
Removes a binary pattern from a trace.
    Parameters:
    - pair: A tuple representing the binary pattern to remove.
    - trace: List of integers representing the trace.
    
    Returns:
    - new_trace: A new list representing the trace with the binary pattern removed.
"""

def remove_binary_pattern(pair, trace):
    # Find all occurrences of the binary pattern and mark them
    marked_trace = [-1] * len(trace)
    for i in range(len(trace)):
        if trace[i] == pair[0]:
            marked_trace[i] = 1
        elif trace[i] == pair[1]:
            marked_trace[i] = 2

    # Remove pairs from the trace
    new_trace = []
    i = 0
    pair = -1
    toremove = set()
    firsthalf = []
   

    while i < len(trace):
        #add index for first part of potential pair
        if marked_trace[i] == 1:
            firsthalf.append(i) 
            i+=1
            continue
        if marked_trace[i] == 2 and len(firsthalf)!=0:
            #if the index is before this
            if(firsthalf[0] < i): 
                remove = firsthalf.pop(0) #pop and track the index 
                toremove.add(remove) #the index of the first part of the pair
                toremove.add(i) #the index of the second part of the pair    
        i+=1

    

    new_trace = [trace[i] for i in range(len(trace)) if i not in toremove]
    return new_trace




if __name__ == "__main__":
    file_path = "gem5_traces/gem5-snoop/defSnoop-RubelPrintFormat.msg"
    messages, sections = read_messages_from_file(file_path)
    data, initial_nodes, terminating_nodes, successors_dict = construct_causality_graph(messages, sections)
    
    groups = extract_groups_from_msg_file(file_path)
    for g in groups:
        print(g)




    ########### testing removal of pattern from trace
    # trace = read_trace_from_file('gem5_traces/gem5-snoop/unslicedtrace-1 copy/unsliced-cpu0-icache0.txt')
    # pair = (0,9)
    # result = remove_binary_pattern(pair,trace)
    # print(result)


    trace_file_path = "gem5_traces/gem5-snoop/snoopunsliced-RubelPrintFormat.jbl"
    output_folder = "gem5_traces/gem5-snoop/unslicedtrace-1-binarypatterns"
    file = "gem5_traces/gem5-snoop/localPatterns.txt"
    compute_common_causal_pairs(trace_file_path, output_folder, successors_dict, file)
    read_trace_file(trace_file_path)



