import torch
from torch_geometric.data import Data
from collections import Counter
import os
import itertools

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


#Read in the msg file and extract the pairings in the data flow.
#1 and 2 are a pair if: src1 = dest2, dest1=src2. cmd1=cmd2. if type1 is resp, type2 must be req and vice versa.
def extract_groups_from_msg_file(file_path):
    group_indices = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Ignore lines that start with #
            if line.startswith('#'):
                continue  
            elif line:
                parts = [part.strip() for part in line.split(':')]
                if len(parts) == 5:
                    index, src, dest, cmd, type_ = parts
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
    print(causal_pairs)
    return causal_pairs


#Use memoization to store the possible routes of pairs to explore
def generate_routes(indices, pairs):
    routes = []
    used_indices = set()
    memo = {}

    def backtrack(current_route):
        if len(current_route) == (len(indices)/2):
            routes.append(current_route)
            return

        last_index = current_route[-1][1] if current_route else None

        for pair in pairs:
            if pair[0] == last_index or pair[1] == last_index:
                continue

            if pair[0] in used_indices or pair[1] in used_indices:
                continue

            used_indices.add(pair[0])
            used_indices.add(pair[1])

            new_route = tuple(sorted(current_route + [pair]))
            if new_route not in memo:
                memo[new_route] = True
                backtrack(current_route + [pair])

            used_indices.remove(pair[0])
            used_indices.remove(pair[1])

    backtrack([])
    return routes


def generate_routes_for_group(group_name, groups, successors_dict):
    group_names = group_name.split('-')
    group_indices = None
    for group in groups:
        if group_names[0] in group[0] and group_names[1] in group[1]:
            group_indices = group[2]
            break
    if group_indices is None:
        return []  # Group not found

    causal_pairs = find_causal_pairs(group_indices, successors_dict)
    possible_routes = generate_routes(group_indices, causal_pairs)
    return possible_routes


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
Finds causal pairs within a trace using backtracking.

Parameters:
- trace: The list representing the trace.
- successors_dict: A dictionary mapping each node to its details and list of successors.

Returns:
- List of causal pairs found in the trace.
"""


"""
Finds causal pairs within a trace using backtracking.

Parameters:
- trace: The list representing the trace.
- successors_dict: A dictionary mapping each node to its details and list of successors.

Returns:
- List of causal pairs found in the trace.
"""
def find_binary_pattern(trace, successors_dict):
    causal_pairs = []
    print("initial trace", trace)


    """
    Recursively backtrack to find causal pairs in the trace.

    Parameters:
    - start_index: The starting index for searching in the trace.
    - remaining_trace: The remaining trace to be explored, after removing all the found causal pairs.

    Returns:
    - True if causal pairs are found and resolve the trace, False otherwise.
    """
    def backtrack(start_index, remaining_trace):

        #when there is nothing left remaining, it successfully resolved the trace (found all pairs)
        if not remaining_trace:
            print("Successfully resolved the trace.")
            return True
        
        #check initial nodes (first ones in trace)
        for i in range(start_index, len(remaining_trace)):
            initial_node = remaining_trace[i]
            print(f"Trying initial node: {initial_node}")

            #For all nodes that follow. check, if one is causal, find the pair. Remove all instances of pair from trace and proceed.
            # Skip duplicates or pairs already seen

            printed_end_nodes = set()  # To track printed end nodes
            for j in range(i + 1, len(remaining_trace)):
                end_node = remaining_trace[j]
                if end_node != initial_node and end_node not in printed_end_nodes:
                    print(f"Checking end node: {end_node}")
                    printed_end_nodes.add(end_node)
               
                if isCausal(initial_node, end_node, successors_dict):
                    print(f"Found causal pair: ({initial_node}, {end_node})")
                    causal_pairs.append((initial_node, end_node))
                    updated_remaining_trace = [node for node in remaining_trace if node not in causal_pairs[-1]]

                    #track how many pairs removed
                    removed_pairs_count = (len(remaining_trace) - len(updated_remaining_trace)) // 2
                    counter.update([(initial_node, end_node)] * removed_pairs_count)

                    print(f"Remaining trace after removal: {updated_remaining_trace}")

                    #recursively call the backtrack function with the remaining trace. so the initial will be remainingtrace[i]
                    if backtrack(i, updated_remaining_trace):
                        return True
                    #if it fails, remove the causal pair and try with another
                    causal_pairs.pop()
                    counter.subtract([(initial_node, end_node)] * removed_pairs_count)
                    print("Backtracking...try with another pair")

            # Backtrack if no causal pair found for the current initial node
            print("Backtracking to previous initial node.")
            return False
        
        #if no causal pairs are found, so it's unsuccessful
        print("No pairs found...")
        return False

    #start the backtracking
    counter = Counter()
    backtrack(0, trace)
    return causal_pairs, counter




"""
Computes common causal pairs from multiple trace files in a folder.

Parameters:
- folder_path: Path to the folder containing trace files.
- output_file: Path to the output file to write the results.
- successors_dict: A dictionary mapping each node to its details and list of successors.
"""
def compute_common_causal_pairs(folder_path, output_file, successors_dict):
    total_counter = Counter()

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            trace = read_trace_from_file(file_path)
            print("Getting pairs for:", file_path)
            causal_pairs, counter = find_binary_pattern(trace, successors_dict)
            total_counter.update(counter)

    with open(output_file, 'w') as f:
        for pair, count in total_counter.most_common():
            f.write(f"{pair[0]} {pair[1]} : {count}\n")

    print(f"Most common causal pairs written to {output_file}")

# def compute_common_causal_pairs_from_file(trace_file, successors_dict):
#     total_counter = Counter()

#     trace = read_trace_from_file(trace_file)
#     causal_pairs = find_causal_pairs(trace, successors_dict)
#     total_counter.update(causal_pairs)

#     with open(output_file, 'w') as f:
#         for pair, count in total_counter.most_common():
#             print(f"{pair[0]} {pair[1]} : {count}\n")

    



if __name__ == "__main__":
    file_path = "synthetic_traces/newLarge.msg"  
    messages, sections = read_messages_from_file(file_path)
    data, initial_nodes, terminating_nodes, successors_dict = construct_causality_graph(messages, sections)
    
    groups = extract_groups_from_msg_file(file_path)
    # for g in groups:
    #     print(g)

    group_name = 'cache0-cpu0'
    possible_routes = generate_routes_for_group(group_name, groups, successors_dict)
    # print(possible_routes)

    # folder_path = "synthetic_traces/traces/trace-small-5"  
    # output_file = "synthetic_traces/traces/trace-small-5-common_subsequences.txt"
    # trace_file = "synthetic_traces/traces/trace-small-5/trace-small-5-cache1-membus.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict)
    # # compute_common_causal_pairs_from_file(trace_file, successors_dict)

    # folder_path = "synthetic_traces/traces/trace-small-10"  
    # output_file = "synthetic_traces/traces/trace-small-10-common_subsequences.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict)

    folder_path = "synthetic_traces/traces/trace-small-20"  
    output_file = "synthetic_traces/traces/trace-small-20-common_subsequences.txt"
    compute_common_causal_pairs(folder_path, output_file, successors_dict)


    # folder_path = "synthetic_traces/traces/trace-large-5"  
    # output_file = "synthetic_traces/traces/trace-large-5-common_subsequences.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict)

    # folder_path = "synthetic_traces/traces/trace-large-10"  
    # output_file = "synthetic_traces/traces/trace-large-10-common_subsequences.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict)

    # folder_path = "synthetic_traces/traces/trace-large-20"  
    # output_file = "synthetic_traces/traces/trace-large-20-common_subsequences.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict)

    

