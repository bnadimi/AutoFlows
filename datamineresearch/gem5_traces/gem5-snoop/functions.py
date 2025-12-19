import torch
from torch_geometric.data import Data
import numpy as np
from collections import Counter
import os
import joblib
import pickle
import itertools
from itertools import zip_longest
from multiprocessing import Pool, cpu_count


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

"""
Remove occurrences of a specified pattern from a trace list.

Args:
- trace (list): The trace list from which the pattern occurrences should be removed.
- pattern (list): The pattern (sequence of numbers) to be removed from the trace.

Returns:
- list: A new trace list with the specified pattern occurrences removed.

Notes:
- hash table-based indexing and pointer manipulation for efficiency
"""

def remove_pattern_from_trace(trace, pattern):

    # Initialize buckets. Every number in pattern has a bucket to track indices
    buckets = {num: [] for num in pattern}
    
    # Fill buckets with indices
    for index, num in enumerate(trace):
        if num in buckets:
            buckets[num].append(index)
    # print("buckets", buckets)

    to_remove = set()    
    currentindices = []

    #pointer array for pattern indices in buckets (instead of pop)
    pointers = [0] * len(pattern)

    #iterate through first bucket, or first number in pattern 
    for index in buckets[pattern[0]]:

        currentindices = [index]
        # print("current indices: ", currentindices)
        valid_pattern = True

        #check subsequent buckets
        for j in range(1, len(pattern)):
            # print("checking bucket of", pattern[j], " in ", pattern)

            # Find the smallest index in the current bucket that is larger than the last index in currentindices, and greater than any already used
            while pointers[j] < len(buckets[pattern[j]]) and buckets[pattern[j]][pointers[j]] <= currentindices[-1]:
                # print("traversing bucket.", buckets[pattern[j]][pointers[j]], "which is index ", pointers[j])
                pointers[j] += 1

            #once it finds an index that is greater than the current one, add to the patern
            if pointers[j] < len(buckets[pattern[j]]):
                # print("index found from bucket.", buckets[pattern[j]][pointers[j]])
                currentindices.append(buckets[pattern[j]][pointers[j]])
                pointers[j] += 1
            else:
                valid_pattern = False
                break 

        # If currentindices match the pattern length, add to to_remove set
        if valid_pattern and len(currentindices) == len(pattern):
            # print("valid pattern found. indices:", currentindices)
            to_remove.update(currentindices)
       
    # Remove the pattern indices from the trace
    new_trace = [trace[i] for i in range(len(trace)) if i not in to_remove]
    return new_trace

"""
    Find patterns in the trace and calculate acceptance ratios for each pattern.

    Args:
    - traces (list): List of traces, each list being a list of integers.
    - patterns (list): A list of patterns (each a list of numbers) to find in the trace.

    Returns:
    - pair_acceptance_ratios: List of tuples representing pairs and their average acceptance ratios.

"""

def find_acceptance_ratios(traces, patterns):

    
    # List to track acceptance ratios for each pattern across all traces
    pattern_acceptance_ratios = {tuple(pattern): [] for pattern in patterns}

    for trace_index, trace in enumerate(traces):
        
        print(f"Processing Trace {trace_index + 1}/{len(traces)}")

        #track count of every number in original trace
        original_count = Counter(trace)
        num_patterns = len(patterns)
        # Initialize bitmap for processed patterns (all bits initially 0)
        processed_bitmap = [0] * num_patterns
        # Iterate through all patterns using the pointer index
        current_index = 0


        while current_index < num_patterns:
            # Reset the set of used numbers
            used_numbers = set()
            # Restore original trace for each pass
            remaining_trace = trace[:]
            # Track skip count to limit unnecessary skipping
            pattern_skip_count = 0
        

            for pointer_index in range(current_index, num_patterns):

                # Check if the pattern is already processed
                if processed_bitmap[pointer_index] == 1:
                    continue
                pattern = patterns[pointer_index]

                # Skip pattern if it has already used numbers
                if any(num in used_numbers for num in pattern):
                    pattern_skip_count += 1
                    if pattern_skip_count > 15:
                        break
                    continue

                print(f"\nTrying pattern {pointer_index + 1}/{num_patterns}: {pattern}")

                updated_remaining_trace = remove_pattern_from_trace(remaining_trace, pattern)
                remaining_count = Counter(updated_remaining_trace)
                
                orphans = sum(remaining_count[num] for num in pattern)
                original = sum(original_count[num] for num in pattern)

                remaining_trace = updated_remaining_trace

                # Calculate acceptance ratio for the pair
                if original != 0:
                    acceptance_ratio = 1 - (orphans / original)
                    print("Acceptance ratio:", acceptance_ratio)
                else:
                    acceptance_ratio = 0
                    print("pattern", pattern, "does not exist in trace")

                # Append the pattern and its acceptance ratio to the list
                pattern_acceptance_ratios[tuple(pattern)].append(acceptance_ratio)
                # Mark the pattern as processed
                processed_bitmap[pointer_index] = 1
                # Update used numbers
                used_numbers.update(pattern)


            # Increment current_index to move to the next batch of patterns
            while current_index < num_patterns and processed_bitmap[current_index] == 1:
                current_index += 1

        # Calculate the average acceptance ratio across all traces
        average_acceptance_ratios = [(list(pattern), np.mean(ratios)) for pattern, ratios in pattern_acceptance_ratios.items()]
            
    return average_acceptance_ratios


"""
Rank patterns based on their acceptance ratios.

Args:
- patternsRankings: List of all patterns ARs in decreasing order.

Returns:
- allPatternsRanked: List of all patterns ranked in decreasing order.

"""
def ranking_patterns(patternsRankings):
    sorted_patterns = sorted(patternsRankings, key=lambda x: x[1], reverse=True)
    return sorted_patterns


"""
Writes the ranking of patterns to a txt file.
Argument:
-sorted_patterns: list of patterns with their respective average ratios in sorted order, descending
-output_filename: choose the file name where results will be written to 

"""
def output_ranked_patterns_to_file(sorted_patterns, output_filename="synthetic_traces/multipleTraces/ranked_patterns.txt"):
    with open(output_filename, 'w') as file:
        file.write(f"{'-' * 25}\n")
        for idx, (pattern, acceptance_ratio) in enumerate(sorted_patterns, 1):
            file.write(f"{idx}. Pattern: {pattern}, Acceptance Ratio: {acceptance_ratio:.4f}\n")
        file.write(f"{'-' * 25}\n")
    print(f"Ranked patterns written to {output_filename}")



"""
    Find patterns in the trace and calculate acceptance ratios for each pattern. 

    Args:
    - trace (list): List of integers representing the trace.
    - patterns (list): A list of patterns (each a list of numbers) to find in the trace.

    Returns:
    - pair_acceptance_ratios: List of tuples representing pairs and their acceptance ratios.

"""
def find_acceptance_ratios_singletrace(trace, patterns):

    # List to track acceptance ratios for each pair
    pattern_acceptance_ratios = []
    #track count of every number in original trace
    original_count = Counter(trace)
    # Number of patterns
    num_patterns = len(patterns)
    # Initialize bitmap for processed patterns (all bits initially 0)
    processed_bitmap = [0] * num_patterns
    # Iterate through all patterns using the pointer index
    current_index = 0

    while current_index < num_patterns:
        # Reset the set of used numbers
        used_numbers = set()
        # Restore original trace for each pass
        remaining_trace = trace[:]
        # Track skip count to limit unnecessary skipping
        pattern_skip_count = 0
       

        for pointer_index in range(current_index, num_patterns):

            # Check if the pattern is already processed
            if processed_bitmap[pointer_index] == 1:
                continue
            pattern = patterns[pointer_index]

            # Skip pattern if it has already used numbers
            if any(num in used_numbers for num in pattern):
                pattern_skip_count += 1
                if pattern_skip_count > 15:
                    break
                continue

            print(f"\nTrying pattern {pointer_index + 1}/{num_patterns}: {pattern}")

            updated_remaining_trace = remove_pattern_from_trace(remaining_trace, pattern)
            remaining_count = Counter(updated_remaining_trace)
            
            orphans = sum(remaining_count[num] for num in pattern)
            original = sum(original_count[num] for num in pattern)

            remaining_trace = updated_remaining_trace

            # Calculate acceptance ratio for the pair
            if original != 0:
                acceptance_ratio = 1 - (orphans / original)
                print("Acceptance ratio:", acceptance_ratio)
            else:
                acceptance_ratio = 0
                print("pattern", pattern, "does not exist in trace")

            # Append the pattern and its acceptance ratio to the list
            pattern_acceptance_ratios.append((pattern, acceptance_ratio))
            # Mark the pattern as processed
            processed_bitmap[pointer_index] = 1
            # Update used numbers
            used_numbers.update(pattern)


        # Increment current_index to move to the next batch of patterns
        while current_index < num_patterns and processed_bitmap[current_index] == 1:
            current_index += 1
            
    print(f"Processed {len(pattern_acceptance_ratios)} patterns out of {num_patterns}.")
    return pattern_acceptance_ratios

"""
Computes acceptance ratios of each pattern from a list of patterns on a trace and writes results to an output folder. Splits the pattern list into chunks 
to use a multithreading approach.
    Parameters:
    - trace: pass in a trace file
    - output_folder: Path to the output folder to write results.
    - patterns: pass in the list of patterns to get ratios from 
"""
def ranking_patterns_multithread(trace, output_folder, patterns):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine the number of processes to use
    num_processes = min(cpu_count(), len(patterns))
    
    # Split patterns into chunks for each process
    chunk_size = len(patterns) // num_processes
    patterns_chunks = [patterns[i:i + chunk_size] for i in range(0, len(patterns), chunk_size)]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(find_acceptance_ratios, [(trace, chunk) for chunk in patterns_chunks])

    # Combine results from all processes
    patterns_info = [item for sublist in results for item in sublist]

    # Filter and sort pairs based on acceptance ratio
    filtered_patterns = [(pattern, ratio) for pattern, ratio in patterns_info]
    sorted_patterns = sorted(filtered_patterns, key=lambda x: x[1], reverse=True)

    # Write results to a separate file
    output_file_path = os.path.join(output_folder, f"Patterns_AcceptanceRatios.txt")
    number = 1
    with open(output_file_path, 'w') as f:
        f.write(f"{'-' * 25}\n")
        for pattern, acceptance_ratio in sorted_patterns:
            f.write(f"{number}. {pattern}, Acceptance Ratio: {acceptance_ratio}\n")
            number += 1

        f.write(f"{'-' * 25}\n")
        print("Processing completed for file:", output_file_path)

    print(f"Results written to {output_folder}")

"""
Computes acceptance ratios of each pattern from a list of patterns on a trace and writes results to an output folder.
    Parameters:
    - trace: pass in a trace file
    - output_folder: Path to the output folder to write results.
    - patterns: pass in the list of patterns to get ratios from 
"""
def ranking_patterns_singlethread(trace, output_folder, patterns):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Find the acceptance ratio for the list of patterns 
    patterns_info = find_acceptance_ratios(trace, patterns)

    # Filter and sort pairs based on acceptance ratio
    # filtered_pairs = [(pair, ratio) for pair, ratio in causal_pairs_info if ratio > 0.5]
    filtered_patterns = [(pattern, ratio) for pattern, ratio in patterns_info]
    sorted_patterns = sorted(filtered_patterns, key=lambda x: x[1], reverse=True)

    # Write results to a separate file
    output_file_path = os.path.join(output_folder, f"Patterns_AcceptanceRatios.txt")
    number = 1
    with open(output_file_path, 'w') as f:
        f.write(f"{'-' * 25}\n")
        for pattern, acceptance_ratio in sorted_patterns:
            f.write(f"{number}. {pattern}, Acceptance Ratio: {acceptance_ratio}\n")
            number += 1

        f.write(f"{'-' * 25}\n")
        print("Processing completed for file:", file_path)

    print(f"Results written to {output_folder}")






# Function to extract patterns
def extract_patterns_from_rows(rows):
    # Create a list of lists to store patterns
    extracted_patterns = []

    # Use zip_longest to handle rows with different lengths
    for patterns in zip_longest(*rows, fillvalue=None):
        # Filter out None values
        valid_patterns = [pattern for pattern in patterns if pattern is not None]
        if valid_patterns:
            extracted_patterns.append(valid_patterns)

    return extracted_patterns




if __name__ == "__main__":
    file_path = "gem5_traces/gem5-snoop/defSnoop-RubelPrintFormat.msg"
    messages, sections = read_messages_from_file(file_path)
    data, initial_nodes, terminating_nodes, successors_dict = construct_causality_graph(messages, sections)
    

    """
    See the indices from the message file
    """
    # groups = extract_groups_from_msg_file(file_path)
    # for g in groups:
    #     print(g)


    """
    testing removal of pattern from trace
    """
    # trace = read_trace_from_file('gem5_traces/gem5-snoop/unslicedtrace-1 copy (testing)/test.txt')
    # pattern = [1,2,3,4]
    # result = remove_pattern_from_trace(trace, pattern)
    # print(result)

    
    with open('gem5_traces/gem5-snoop/allPatterns.data', 'rb') as f:
       allPatterns = pickle.load(f)

    # Extract patterns
    patterns_by_column = extract_patterns_from_rows(allPatterns)

    # Combine all pattern lists into one giant list
    allPatterns = [pattern for sublist in patterns_by_column for pattern in sublist]


    """
    Testing with single trace (single or multithread)
    """

    # trace = read_trace_from_file('gem5_traces/gem5-snoop/reducedTrace-TCAD-snoop.txt')
    # output_folder = "gem5_traces/gem5-snoop/unslicedtrace-1-patterns"
    # file = "gem5_traces/gem5-snoop/localPatterns.txt"

    # # Iterate through the extracted patterns single or multithread
    # ranking_patterns_singlethread(trace, output_folder, allPatterns)
    # ranking_patterns_multithread(trace, output_folder, allPatterns)

    
    """
    Testing the AR function
    """

    trace1 = read_trace_from_file('gem5_traces/gem5-snoop/reducedTrace-TCAD-snoop.txt')
    trace2 = read_trace_from_file('gem5_traces/gem5-snoop/reducedTrace-TCAD-threads.txt')
    traces = [trace1,trace2]
    allPatterns = allPatterns[:10]
    acceptance_ratios = find_acceptance_ratios(traces, allPatterns)
    ranked_patterns = ranking_patterns(acceptance_ratios)

    outputfile = "gem5_traces/gem5-snoop/unslicedtrace-1-patterns/ranked_patterns.txt"
    output_ranked_patterns_to_file(ranked_patterns, outputfile)







    

    





