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





def find_binary_pattern(trace, successors_dict, group_name, groups):
    causal_pairs = []
    print("initial trace", trace)

    # Generate routes for the given group name
    group_routes = generate_routes_for_group(group_name, groups, successors_dict)
    print("Generated routes for group:", group_name)
    print(group_routes)

    # Track acceptance ratios for each route
    route_acceptance_ratios = {}

    # Recursively backtrack to find causal pairs in the trace using each route
    for route_index, route in enumerate(group_routes):
       

        potential_start = []
        for pair in route:
            potential_start.append(pair[0])


        print(f"\nTrying route {route_index + 1}/{len(group_routes)}: {route}")
        #skip route 
        if trace[0] not in potential_start:
            print(f"skipping this route, because this trace has different initial node. sliced trace starts with {trace[0]}")
            continue
        
        uniquenumbers = set(trace)
    
        remaining_trace = trace  # Copy the trace for each route
        orphaned_nodes = len(remaining_trace)
        
        #keep track of what pairs were used
        used_pairs = []

        # Try resolving the trace using the current route
        for pair in route:
            
            if any(num not in uniquenumbers for num in pair):
                continue
            print(f"Trying pair: {pair}")
            # print(remaining_trace)
            updated_remaining_trace = remove_binary_pattern(pair,remaining_trace)
            orphaned_nodes -= (len(remaining_trace) - len(updated_remaining_trace))
            remaining_trace = updated_remaining_trace
            used_pairs.append(pair)
            print(f"trace after removal: {remaining_trace}")

        # Calculate acceptance ratio for the route
        acceptance_ratio = 1- (orphaned_nodes / len(trace))
        print("Acceptance ratio:", acceptance_ratio)

        route_acceptance_ratios[route_index] = (used_pairs, acceptance_ratio)
    return route_acceptance_ratios

def compute_common_causal_pairs(folder_path, output_file, successors_dict, groups):
    with open(output_file, 'w') as f:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)
                trace = read_trace_from_file(file_path)
                print("\nProcessing file:", file_path)
                
                # Extract group name from the file name
                group_name = file_name.split("-")[-2] + "-" + file_name.split("-")[-1].replace(".txt", "")
                print("\nGroup:", group_name)
                group_indices = findgroup(group_name, groups)
                
                # Find pairs for the extracted group name
                causal_pairs_info = find_binary_pattern(trace, successors_dict, group_name, groups)
                sorted_pairs = sorted(causal_pairs_info.items(), key=lambda x: x[1][1], reverse=True)
                top_pairs_with_ratio_1 = set()
                top_pairs_without_ratio_1 = set()
                for route_index, (route, acceptance_ratio) in sorted_pairs:
                    if acceptance_ratio == 1:
                        top_pairs_with_ratio_1.add((tuple(route), acceptance_ratio))
                    elif acceptance_ratio > .8 and len(top_pairs_without_ratio_1) < 5:
                        top_pairs_without_ratio_1.add((tuple(route), acceptance_ratio))
        
                top_pairs_without_ratio_1 = sorted(top_pairs_without_ratio_1, key=lambda x: x[1], reverse=True)
                f.write(f"-"*25)
                f.write("\n")
                f.write(f"File: {file_name},\n Group: {group_name} Indices: {group_indices}\n")
                
                # Write pairs with acceptance ratio 1
                for route, acceptance_ratio in top_pairs_with_ratio_1:
                    f.write(f"BinaryPatterns: {route}, Acceptance Ratio: {acceptance_ratio}\n")
                # Write top 5 pairs without ratio 1
                for route, acceptance_ratio in top_pairs_without_ratio_1:
                    f.write(f"BinaryPatterns: {route}, Acceptance Ratio: {acceptance_ratio}\n")
                f.write(f"-"*25)
                print("Processing completed for file:", file_path)

    print(f"Results written to {output_file}")



def remove_binary_pattern(pair, trace):
    # Find all occurrences of the binary pattern and mark them
    marked_trace = [0] * len(trace)
    for i in range(len(trace)):
        if trace[i] == pair[0]:
            marked_trace[i] = 1
        elif trace[i] == pair[1]:
            marked_trace[i] = 2

    # Remove pairs from the trace
    new_trace = []
    i = 0
    pair = -1
    toremove = []
    firsthalf = []
    secondhalf = []
    remove = 0

    while i < len(trace):
        if marked_trace[i] == 1:
            firsthalf.append(i) #add index for first part of potential pair
            i+=1
            continue
        if marked_trace[i] == 2 and len(firsthalf)!=0:
            if(firsthalf[0] < i): #if the index is before this
                remove = firsthalf.pop(0) #remove first thing in popped
            toremove.append(remove) #the index of the first part of the pair
            toremove.append(i) #the index of the second part of the pair

        else:
            new_trace.append(trace[i]) #unpaired index, part of pattern but not pair.
        i+=1
    new_trace = [trace[i] for i in range(len(trace)) if i not in toremove]
    return new_trace




if __name__ == "__main__":
    file_path = "synthetic_traces/newLarge.msg"  
    messages, sections = read_messages_from_file(file_path)
    data, initial_nodes, terminating_nodes, successors_dict = construct_causality_graph(messages, sections)
    
    groups = extract_groups_from_msg_file(file_path)
    for g in groups:
        print(g)


    ########### testing generating routes
    # group_name = 'audio-membus'
    # possible_routes = generate_routes_for_group(group_name, groups, successors_dict)
    # print(possible_routes)

    ########### testing removal of pattern from trace
    # trace = [0, 0, 25, 2, 2, 25, 0, 25, 2, 2, 0, 25, 0, 25, 0, 25, 2, 2, 2, 2]
    # pair = (25,2)
    # result = remove_binary_pattern(pair,trace)
    # print(result)


    # folder_path = "synthetic_traces/traces/trace-small-5"  
    # output_file = "synthetic_traces/traces/trace-small-5-binarypatterns.txt"
    # trace_file = "synthetic_traces/traces/trace-small-5/trace-small-5-cache1-membus.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict, groups)
    # # compute_common_causal_pairs_from_file(trace_file, successors_dict)

    # folder_path = "synthetic_traces/traces/trace-small-10"  
    # output_file = "synthetic_traces/traces/trace-small-10-binarypatterns.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict, groups)

    # folder_path = "synthetic_traces/traces/trace-small-20"  
    # output_file = "synthetic_traces/traces/trace-small-20-binarypatterns.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict, groups)
        
    # folder_path = "synthetic_traces/traces/TESTING"  
    # output_file = "synthetic_traces/traces/TEST-binarypatterns.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict, groups)


    # folder_path = "synthetic_traces/traces/trace-large-5"  
    # output_file = "synthetic_traces/traces/trace-large-5-binarypatterns.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict, groups)

    # folder_path = "synthetic_traces/traces/trace-large-10"  
    # output_file = "synthetic_traces/traces/trace-large-10-binarypatterns.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict, groups)

    # folder_path = "synthetic_traces/traces/trace-large-20"  
    # output_file = "synthetic_traces/traces/trace-large-20-binarypatterns.txt"
    # compute_common_causal_pairs(folder_path, output_file, successors_dict, groups)

    

