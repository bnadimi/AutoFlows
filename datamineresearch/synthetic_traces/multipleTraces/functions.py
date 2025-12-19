import os
from collections import Counter
import numpy as np
from scipy.stats import zscore

#Read in the msg file and extract the pairings in the data flow.
#1 and 2 are a pair if: src1 = dest2, dest1=src2. cmd1=cmd2. if type1 is resp, type2 must be req and vice versa.
def extract_groups_from_msg_file(file_path):
    group_indices = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue  # Ignore comments
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




# Read in the trace file into a list of traces
def read_trace_file(file_path):
    traces = []
    with open(file_path, 'r') as file:
        trace = []
        for line in file:
            parts = line.strip().split()
            if not parts:  # If line is empty
                if trace:
                    traces.append(trace)
                    trace = []  # Start a new trace
                continue
            for part in parts:
                if part == '-1':
                    continue
                elif part == '-2':
                    if trace:
                        traces.append(trace)
                        trace = []  # Start a new trace
                else:
                    trace.append(int(part))
        if trace:  # If there's any remaining trace
            traces.append(trace)
    return traces




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
- allPatterns: List of all patterns available in the causality graph.
- patternsRankings: List of all patterns ARs in decreasing order.

Returns:
- allPatternsRanked: List of all patterns ranked in decreasing order.

"""
def ranking_patterns(allPatterns, patternsRankings):
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



if __name__ == "__main__":

    file_path = "synthetic_traces/newLarge.msg"
    # groups = extract_groups_from_msg_file(file_path)
    # for g in groups:
    #     print(g)


    file_path = "synthetic_traces/multipleTraces/RubelMultiTrace.txt"
    traces = read_trace_file(file_path)


    patterns = [[48,49], [58, 59], [39, 41], [5, 36], [16, 20], [8, 12],[17, 19],[7, 11],[2, 26], [0, 25],[10, 27], [21, 30],[9, 18],[13, 24],[21, 24],[30, 13],[3, 32],[1, 29],[22, 31],[14, 28],[54, 55],[38, 40],[4, 35],[56, 57],[15, 23],[33, 34],[46, 47],[50, 51],[42, 43],[52, 53],[6, 37],[44, 45], [58,49,48], [58, 39], [25, 41], [25, 36], [56, 29], [18, 12],[7, 9],[11,2, 26], [26, 25],[11, 18]]

    acceptance_ratios = find_acceptance_ratios(traces, patterns)
    ranked_patterns = ranking_patterns(patterns, acceptance_ratios)

    print("Acceptance Ratios:", acceptance_ratios)
    print("Ranked Patterns:",ranked_patterns)

    output_ranked_patterns_to_file(ranked_patterns)

    

