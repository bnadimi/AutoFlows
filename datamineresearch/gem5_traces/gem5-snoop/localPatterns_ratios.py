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





#Given a group, find all the possible causal pairings.
def find_causal_pairs(indices, successors_dict):
    causal_pairs = set()
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            if isCausal(indices[i], indices[j], successors_dict):
                causal_pairs.add((indices[i], indices[j]))
            if isCausal(indices[j], indices[i], successors_dict):
                causal_pairs.add((indices[j], indices[i]))
    return causal_pairs


def findgroup(group_names, groups):
    group_names = group_names.split('-')
    group_indices = None
    for group in groups:
        if group_names[0] in group[0] and group_names[1] in group[1]:
            group_indices = group[2]
            break
    if group_indices is None:
        return []  # Group not found
    return group_indices

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
Gets the successors of a node in the graph.

Parameters:
- node_index: The index of the node
- successors_dict: A dictionary mapping each node to list of successors.

Returns:
- A list of tuples, each representing a successor with its index, source, and destination.
"""


def get_successors(node_index, successors_dict):
    node_info = successors_dict.get(node_index)
    if node_info:
        successors_info = [(succ, successors_dict[succ]['src'], successors_dict[succ]['dest']) for succ in node_info['successors']]
        return successors_info
    return []

""" Generate all causal pairs for a specific index, considering its successors. """
def get_causal_pairs_for_index(node_index, successors_dict):
    causal_pairs = []
    successors_info = get_successors(node_index, successors_dict)
    for succ, src, dest in successors_info:
        causal_pairs.append((node_index, succ))
    return causal_pairs
"""
Prints detailed path information for the successors of the specified node.
    
    Parameters:
    - node_index: The index of the node to query.
    - successors_dict: A dictionary mapping each node to its details and list of successors.
"""

def print_successor_info(node_index, successors_dict):

    if node_index in successors_dict:
        node_info = successors_dict[node_index]
        print(f"Node {node_index} (Source: {node_info['src']}, Destination: {node_info['dest']}) has successors:")
        for succ in node_info['successors']:
            succ_info = successors_dict.get(succ, {})
            print(f"  Successor: {succ}, Path: {succ_info.get('src', 'N/A')} -> {succ_info.get('dest', 'N/A')}")
    else:
        print(f"Node {node_index} has no successors or does not exist in the graph.")


"""
Checks if there is a causal connection between two nodes in the graph.

Parameters:
- node_index1: The index of the first node.
- node_index2: The index of the second node.
- successors_dict: A dictionary mapping each node to its details and list of successors.

Returns:
- True if there is a causal connection, False otherwise.
"""
def isCausal(node_index1, node_index2, successors_dict):
    successors_of_node1 = successors_dict.get(node_index1, {}).get('successors', [])
    successors_of_node2 = successors_dict.get(node_index2, {}).get('successors', [])
    
    if (node_index2 in successors_of_node1) or (node_index1 in successors_of_node2):
        return True
    return False

"""
Prints whether two nodes are causally connected or not.

Parameters:
- node_index1: The index of the first node.
- node_index2: The index of the second node.
- successors_dict: A dictionary mapping each node to its details and list of successors.
"""
def print_causality_info(node_index1, node_index2, successors_dict):
    successors_of_node1 = successors_dict.get(node_index1, {}).get('successors', [])
    successors_of_node2 = successors_dict.get(node_index2, {}).get('successors', [])
    
    if node_index1 in successors_of_node2:
        print(f"Node {node_index2} is causal to Node {node_index1}.")
    elif node_index2 in successors_of_node1:
        print(f"Node {node_index1} is causal to Node {node_index2}.")
    else:
        print(f"Nodes {node_index1} and {node_index2} are not connected.")



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

def find_binary_pattern(trace, successors_dict, group_name, groups):
    group_indices = list(set(trace))
    print("initial trace", trace)

    # Generate routes for the given group name
    causal_pairs = find_causal_pairs(group_indices, successors_dict)
    print("Generated pairs for group:", group_name)
    print(causal_pairs)

    # List to track acceptance ratios for each pair
    pair_acceptance_ratios = []

    while causal_pairs:
        used_numbers = set()
        remaining_trace = trace[:]  # Copy the trace for each pass
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
            acceptance_ratio = 1 - (orphans / original)
            print("Acceptance ratio:", acceptance_ratio)

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
def compute_common_causal_pairs(folder_path, output_folder, successors_dict, groups):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            trace = read_trace_from_file(file_path)
            print("\nProcessing file:", file_path)

            # Extract group name from the file name
            group_name = "-".join(file_name.split("-")[-2:]).replace(".txt", "")
            print("\nGroup:", group_name)
            group_indices = findgroup(group_name, groups)

            # Find pairs for the extracted group name
            causal_pairs_info = find_binary_pattern(trace, successors_dict, group_name, groups)

            # Filter and sort pairs based on acceptance ratio
            filtered_pairs = [(pair, ratio) for pair, ratio in causal_pairs_info if ratio > 0.5]
            sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)

            # Write results to a separate file for each group
            output_file_path = os.path.join(output_folder, f"{group_name}.txt")
            with open(output_file_path, 'w') as f:
                f.write(f"{'-' * 25}\n")
                f.write(f"File: {file_name},\n Group: {group_name} Indices: {group_indices}\n")

                for pair, acceptance_ratio in sorted_pairs:
                    f.write(f"BinaryPatterns: {pair}, Acceptance Ratio: {acceptance_ratio}\n")

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


    folder_path = "gem5_traces/gem5-snoop/unslicedtrace-1 copy (testing)"
    output_folder = "gem5_traces/gem5-snoop/unslicedtrace-1-binarypatterns"
    compute_common_causal_pairs(folder_path, output_folder, successors_dict, groups)


