# This script is used to slice the interface of the trace file

import networkx as nx
import time
from src.graph.graph import Graph
from src.logging import *
from src.evaluation.newEvaluationMethod import newEvaluationMethod
import functions
from datetime import timedelta
import os 
from src.evaluation.newEvaluationMethodOptimized import newEvaluationMethodOptimized
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

start_time = time.time()

G = nx.DiGraph()

print('Path Mining Tool Demo by USF SEES Lab')
print()

def checkAR (trace_file, selected_paths, initialNodes, terminalNodes, resultFileName, preFound):
    ev = newEvaluationMethodOptimized(trace_file, selected_paths, initialNodes, terminalNodes, resultFileName, preFound)
    res1, res2, notAccepted, notUsedPaths = ev.Evaluate()
    return None


def minimal_edges_cover_all_nodes(myArray: Dict) -> List[str]:
    """
    Find a minimal edge cover of the given graph, prioritizing solutions with the
    fewest edges (true minimal cover). If multiple covers with the same minimum size exist,
    return the one with the highest total (fconf + bconf) score.
    Brute-force for small graphs; fallback to greedy for large instances (>20 edges).
    """
    import itertools

    nodes: List[str] = [str(n) for n in myArray["list_of_messages"]]
    node_set: Set[str] = set(nodes)
    edges_info = []
    edge_set = set()
    # Prepare edges and mapping for lookup
    for anInst in myArray["info"]:
        e = anInst["id"]
        if not isinstance(e, str) or "_" not in e:
            raise ValueError(f"Edge must be 'a_b', got: {e!r}")
        a, b = e.split("_", 1)
        if a not in node_set or b not in node_set:
            raise ValueError(f"Edge {e!r} references node(s) not in node list")
        score = float(anInst["fconf"]) + float(anInst["bconf"])
        edges_info.append((e, a, b, score))
        edge_set.add(e)

    if not edges_info:
        raise ValueError("No edges to cover nodes")

    # Adjacency for fast checking
    adj = {node: set() for node in nodes}
    for e, a, b, _ in edges_info:
        adj[a].add(e)
        adj[b].add(e)

    # Fast precheck: make sure every node has some incident edge
    for n in nodes:
        if not adj[n]:
            print(f"ERROR: Node {n!r} is isolated (not covered by any edge).")
            # input("Press Enter to continue after reviewing the error...")

    # Brute force for small cases
    MAX_BRUTE_EDGES = 20  # Safe up to 20 edges; otherwise fallback to greedy
    edge_list = [x[0] for x in edges_info]
    eid2ab = {e: (a, b, score) for e, a, b, score in edges_info}

    min_size = len(nodes)  # maximal cover size can't be larger than that
    minimal_covers = []
    for k in range(1, min(len(edge_list)+1, min_size+1)):
        found_this_k = False
        for comb in itertools.combinations(edge_list, k):
            covered = set()
            total_score = 0.0
            for eid in comb:
                a, b, score = eid2ab[eid]
                covered.add(a)
                covered.add(b)
                total_score += score
            if covered == node_set:
                minimal_covers.append((k, -total_score, comb))  # negative for descending order of score
                found_this_k = True
        if found_this_k:
            # Found at least one cover with minimal size k, break per minimal requirement
            break
    if minimal_covers:
        # Sort by (minimal #edges, highest score descending), return the best
        minimal_covers.sort()
        # Return as list of edge IDs as in input order
        best_cover = minimal_covers[0][2]
        # Optionally, sort result edges as in the previous code (by highest score and then as in original)
        orig_order = {eid: i for i, (eid, *_rest) in enumerate(edges_info)}
        return sorted(list(best_cover), key=lambda e: orig_order[e])

    # If too large for brute-force, fallback to greedy
    # Greedy algorithm: always select the edge that covers the largest number of uncovered nodes,
    # breaking ties with highest score, as close as possible to minimum edge cover.
    left_nodes = set(nodes)
    edges_remain = set(e for e, a, b, score in edges_info)
    chosen = set()
    edge_scores = {e: score for e, a, b, score in edges_info}

    # Prepare bidirectional map edge->nodes and node->edges
    e2nodes = {e: set([a, b]) for e, a, b, s in edges_info}
    node2e = {n: set() for n in nodes}
    for e, ab in e2nodes.items():
        for n in ab:
            node2e[n].add(e)

    while left_nodes:
        # Find the edge covering the most uncovered nodes; break ties by score
        candidates = []
        for e in edges_remain:
            gain = len(e2nodes[e] & left_nodes)
            if gain == 0:
                continue
            candidates.append((gain, edge_scores[e], e))
        if not candidates:
            break  # Should not happen if all nodes covered
        # maximize: gain, then score
        candidates.sort(reverse=True)
        _, _, picked = candidates[0]
        chosen.add(picked)
        left_nodes -= e2nodes[picked]
        edges_remain.remove(picked)

    # As above, sort by input order
    orig_order = {eid: i for i, (eid, *_rest) in enumerate(edges_info)}
    return sorted(list(chosen), key=lambda e: orig_order[e])


if __name__ == '__main__':

    max_pat_len = 8
    max_solutions = 10
    def_f = ""
    trace_f = ""

    # Uncomment corresponding lines to genearte solutions for different traces

    # For gem5 traces

    # Full system (FS) simulation traces
    # def_f = './traces/gem5_traces/fs/definition/fs_def.msg'
    # def_f = './traces/gem5_traces/fs/definition/def-FS-RublePrintFormat.msg'
    # fs unsliced
    # trace_f = ['./traces/gem5_traces/fs/unsliced/unsliced0.jbl']
    # trace_f = ['./traces/gem5_traces/fs/unsliced/fs_boot_unsliced.txt']
    # fs packet id sliced
    # trace_f = ['./traces/gem5_traces/fs/packet_sliced/packet_sliced.jbl']
    # fs memory address sliced
    # trace_f = ['./traces/gem5_traces/fs/addr_sliced/address_sliced_no_duplicates.jbl']

    # Snoop (SE) traces
    # def_f = './traces/gem5_traces/snoop/definition/paterson_def.msg'
    # snoop unsliced
    # trace_f = ['./traces/gem5_traces/snoop/unsliced/paterson_unsliced.txt']
    # snoop packet id sliced
    # trace_f = ['./traces/gem5_traces/snoop/packet_sliced/packet_sliced.jbl']
    # snoop memory address sliced
    # trace_f = ['./traces/gem5_traces/snoop/addr_sliced/address_sliced.jbl']

    # Threads (SE) traces
    # def_f = './traces/gem5_traces/threads/definition/threads_def.msg' # This definition file doesn't include any initial or terminal nodes
    # def_f = './traces/gem5_traces/threads/definition/definition.txt'  
    # def_f = './traces/gem5_traces/threads/definition/renamedDefinitionFile.msg'  
    # def_f = './traces/gem5_traces/threads/definition/myDefinition.txt'  
    # def_f = './traces/gem5_traces/threads/definition/newDefinition.txt' 
    # def_f = './traces/gem5_traces/threads/definition/generatedByMeDef.msg'  
    # def_f = './traces/gem5_traces/threads/definition/TestDef.msg'  # Threads Final Definition file generated By Bardia
    # threads unsliced
    # trace_f = ['./traces/gem5_traces/threads/unsliced/unsliced.txt']
    # trace_f = ['./traces/gem5_traces/threads/unsliced/testTrace.txt']   # Threads Final Trace file generated By Bardia
    # trace_f = ['./traces/gem5_traces/threads/unsliced/generatedByMeThreadsTraceFile.txt']
    # threads packet id sliced
    # trace_f = ['./traces/gem5_traces/threads/packet_sliced/packet_sliced.jbl']   
    # snoop memory address sliced
    # trace_f = ['./traces/gem5_traces/threads/addr_sliced/address_sliced.jbl']
    # trace_f = ['./traces/gem5_traces/threads/addr_sliced/address_sliced_compact.jbl']


    # For synthetic traces
    def_f = './traces/synthetic/newLarge.msg'
    # def_f = './traces/synthetic/large.msg'
    # def_f = './traces/synthetic/medium.msg'
    # def_f = './traces/synthetic/small.msg'
    # def_f = './traces/synthetic/Test/testDefinition.msg'

    # small traces
    # trace_f = ['./traces/synthetic/trace-small-5.txt']
    # trace_f = ['./traces/synthetic/trace-small-10.txt']
    # trace_f = ['./traces/synthetic/trace-small-20.txt']
    # trace_f = ['./traces/synthetic/trace-small-5-New.txt']
    # trace_f = ['./traces/synthetic/trace-small-test.txt']
    # trace_f = ['./traces/synthetic/trace-small-test2.txt']
    # trace_f = ['./traces/synthetic/trace-small-test3.txt']

    # large traces
    # trace_f = ['./traces/synthetic/trace-large-5.txt']
    # trace_f = ['./traces/synthetic/trace-large-10.txt']
    # trace_f = ['./traces/synthetic/trace-large-20.txt']
    trace_f = ['./traces/synthetic/new-trace-large-20.txt']

    # trace_f = ['./traces/synthetic/testCode.txt']
    # trace_f = ['./traces/synthetic/Test/testTrace.txt']

    # For Testing 
    # def_f = './traces/ForTest/testDefinitionFile.txt'
    # trace_f = ['./traces/ForTest/testTrace.txt']

    # def_f = './traces/ForTest/L3cacheTestDefinitionFile.txt'
    # trace_f = ['./traces/ForTest/L3cacheTestTrace.txt']

    # def_f   = './traces/fromTCAD/gem5/snoop/definition/defSnoop-RubelPrintFormat.msg'
    # trace_f = ['./traces/fromTCAD/gem5/snoop/unsliced/unsliced-RubelPrintFormat.jbl']
    # def_f   = './traces/fromTCAD/gem5/snoop/definition/defSnoop-RubelPrintFormat.msg'
    # trace_f = ['/home/bardia/GitHub/AutoFlows/interface_sliced_traces/unsliced-RubelPrintFormat/interface_sliced_v3_dcache0_l2bus.txt']

    # traceType = "synthetic"
    # traceType = "gem5"
    filters_filename = None
    rank_filename    = None

    graph = Graph()
    graph.set_max_height(max_pat_len)
    graph.set_max_solutions(max_solutions)

    
    if "gem5" in def_f:
        ####################### For Threads
        graph.window = False
    else:
        ####################### For Large20
        graph.window = False
    graph.window_size = 50

    if (graph.window and (graph.window <= 0)):
        print("Winodw size must > 0")
        exit()
    if(graph.window):
        print("Added window slicing...window size: ", graph.window_size)
        print()

    log('Reading the message definition file %s... \n' % def_f)
    if def_f=="":
        exit()
    graph.read_message_file(def_f)
    log('Done\n\n')
    print("Interfaces = ", graph.interfaces)

    traces = None
    log('Reading the trace file(s) %s... ' % trace_f)
    graph.read_trace_file_list(trace_f)
    log('Trace reading and processing status: Done\n\n')
    print(f"Length of trace from inside interfaceSlicing.py = {len(graph.trace_tokens)}")
    print(graph.interfaces)
    # exit()


    # Prepare output files for each interface, and a mapping from message to the correct file
    interface_files = []
    message_to_file = {}
    
    ##############################################################################################################################################  Slicing traces based on the interfaces ################################################################
    prev_cwd = os.getcwd()
    # Extract the main trace file's name without extension or directories
    if isinstance(trace_f, list) and len(trace_f) > 0:
        trace_path = trace_f[0]
    elif isinstance(trace_f, str):
        trace_path = trace_f
    else:
        trace_path = "not_found"

    base_name = os.path.basename(trace_path)
    folder_name = os.path.splitext(base_name)[0]

    # Ensure the central output folder exists
    main_folder = "interface_sliced_traces"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    # Make a subfolder per trace, e.g. interface_sliced_traces/[trace name]/
    output_folder = os.path.join(main_folder, folder_name)
    if folder_name and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Change the working directory to the intended subfolder for subsequent output
    os.chdir(output_folder)
    # exit()

    # First, open one file for each interface and build message -> file mapping
    for i, interface in enumerate(graph.interfaces):
        print(f"Length of interface {interface['pairs']} = {len(interface['list'])}")
        filename = f"interface_sliced_v3_{interface['pairs'][0]}_{interface['pairs'][1]}.txt"
        f = open(filename, "w")
        interface_files.append(f)
        for message in interface['list']:
            # map message to its interface file handler
            message_to_file[message] = f

    # Iterate over trace_tokens only once. Route each message to its interface file (if any)
    for message in graph.trace_tokens:
        output_file = message_to_file.get(message)
        if output_file is not None:
            output_file.write(str(message) + " ")

    # After writing all messages, write a newline and close all files
    for output_file in interface_files:
        output_file.write("\n")
        output_file.close()
    ##############################################################################################################################################  Slicing traces based on the interfaces end ############################################################

    sliced_traces = []
    traces_list = []
    for i, interface in enumerate(graph.interfaces):
        filename = os.path.join(os.getcwd(), f"interface_sliced_v3_{interface['pairs'][0]}_{interface['pairs'][1]}.txt")
        traces_list.append(filename)
        try:
            file = open(filename, 'r')
        except IOError as e:
            print("Couldn't open file (%s)." % e)
        slicedTrace = file.readline()
        numbers_list = [int(x) for x in slicedTrace.strip().split() if x]
        sliced_traces.append(numbers_list)   
        file.close() 




    ############################################################################################################################################## By Bardia Start for interface slicing 
    all_permutations = []
    # print(f"The number of interfaces is {len(graph.interfaces)}")
    # print(f"The interfaces are {graph.interfaces}")
    for interface in graph.interfaces:
        perms = []
        for index in interface['list']:
            for next_index in interface['list']:
                if index != next_index:
                    # print(f"{index}_{next_index}")
                    perms.append(f"{index}_{next_index}")
        all_permutations.append(perms)
    print(f"The number of permutations is {len(all_permutations)}")
    # print(f"The permutations are {all_permutations}")
    # exit()

    # import itertools
    # for i, perms in enumerate(all_permutations):
    #     print(f"Len Binary Patterns: {len(perms)}")
    #     # Generate all possible non-empty combinations of perms
    #     # Eg: if perms = ['a', 'b'], combinations are ['a'], ['b'], ['a','b']
    #     combinations = []
    #     for r in range(1, len(perms) + 1):
    #         for combo in itertools.combinations(perms, r):
    #             # Join permutation ids with something (e.g. '_' or just keep as tuple/list)
    #             # Here, keeping as tuple - you can join as string if needed
    #             combinations.append(list(combo))
    #     print(f"Number of possible non-empty combinations: {len(combinations)}")
    #     # Optionally, print the combinations, capped at small length for readability
    #     if len(combinations) < 20:
    #         for combo in combinations:
    #             print(combo)
    #     else:
    #         print(f"(combinations list omitted, too large)")



    os.chdir(prev_cwd)

    interface_slices_info = []
    for i, aTrace in enumerate(traces_list):

        graph = Graph()
        graph.set_max_height(max_pat_len)
        graph.set_max_solutions(max_solutions)

        graph.window = False
        graph.window_size = 50

        graph.read_message_file(def_f)
        print(aTrace)
        log('Reading the trace file(s) %s... ' % aTrace)
        graph.read_trace_file_list([aTrace])
        log('Trace reading and processing status: Done\n\n')
        print(f"Len of graph trace tokens: {len(graph.trace_tokens)}")
        print("--"*100)
        current_interface = {}
        current = []
        current_interface['pairs'] = graph.interfaces[i]['pairs']
        current_interface['list_of_messages'] = graph.interfaces[i]['list']
        # print(graph.interfaces)
        for anEdge in graph.edges:
            if anEdge in all_permutations[i]:
                edge = graph.edges.get(anEdge)
                anInstance = {'id': anEdge, 'support': edge.support, 'fconf': (edge.forward_conf), 'bconf': (edge.backward_conf), 'hconf': (edge.mean_conf)}
                # anInstance['pairs'] = graph.interfaces[i]['pairs']
                # anInstance['list_of_messages'] = graph.interfaces[i]['list']
                id    = "{0:<10}".format(str(anEdge))
                sup   = "{0:<6}".format(str(edge.support))
                fconf = "{0:<6}".format(str(round(edge.forward_conf, 2)))
                bconf = "{0:<6}".format(str(round(edge.backward_conf, 2)))
                hconf = "{0:<6}".format(str(round(edge.mean_conf, 2)))
                print(id, ' ', fconf, ' ', bconf, ' ', hconf)
                # current_interface.append(anInstance)
                current.append(anInstance)
        current_interface['info'] = current
        current_interface['trace'] = sliced_traces[i]
        interface_slices_info.append(current_interface)
        print("--"*100)

    # with ope, "w") as f:
    all_binary_list = []
    for i, trace in enumerate(interface_slices_info):
        for edge in trace['info']:
            # print(edge['id'])
            all_binary_list.append(edge['id'])
    # exit()

    after_pruning = []
    for i in range(len(interface_slices_info)):
        anInterface = interface_slices_info[i]
        # print(graph.interfaces[i])
        print(f"pairs: {anInterface['pairs']}, List: {anInterface['list_of_messages']}")
        mean_fconf = 0
        mean_bconf = 0
        mean_hconf = 0
        for instance in anInterface['info']:
            id    = "{0:<10}".format(str(instance['id']))
            sup   = "{0:<6}".format(str(instance['support']))
            fconf = "{0:<6}".format(str(round(instance['fconf'], 2)))
            bconf = "{0:<6}".format(str(round(instance['bconf'], 2)))
            hconf = "{0:<6}".format(str(round(instance['hconf'], 2)))
            mean_fconf += instance['fconf']
            mean_bconf += instance['bconf']
            mean_hconf += instance['hconf']
            print(f"ID: {instance['id']} {instance['fconf']} {instance['bconf']} {instance['hconf']}")
        mean_fconf /= len(anInterface['info'])
        mean_bconf /= len(anInterface['info'])
        mean_hconf /= len(anInterface['info'])
        print(f"The mean fconf is {mean_fconf}")
        print(f"The mean bconf is {mean_bconf}")
        print(f"The mean hconf is {mean_hconf}")

        after = {}
        after['pairs']            = interface_slices_info[i]['pairs']
        after['list_of_messages'] = interface_slices_info[i]['list_of_messages']
        binary_patterns = []
        for instance in anInterface['info']:
            if instance['fconf'] >= mean_fconf and instance['bconf'] >= mean_bconf and instance['hconf'] >= mean_hconf:
                binary_patterns.append(instance)
        after['info']  = binary_patterns
        after['trace'] = interface_slices_info[i]['trace']
        after_pruning.append(after)


            
        print(f"Len of sliced_trace[{i}] = {len(anInterface['trace'])}")
        print("")


    print("After Pruning")
    for i in range(len(after_pruning)):
        anInterface = after_pruning[i]
        # print(graph.interfaces[i])
        print(f"pairs: {anInterface['pairs']}, List: {anInterface['list_of_messages']}")
        mean_fconf = 0
        mean_bconf = 0
        mean_hconf = 0
        for instance in anInterface['info']:
            id    = "{0:<10}".format(str(instance['id']))
            sup   = "{0:<6}".format(str(instance['support']))
            fconf = "{0:<6}".format(str(round(instance['fconf'], 2)))
            bconf = "{0:<6}".format(str(round(instance['bconf'], 2)))
            hconf = "{0:<6}".format(str(round(instance['hconf'], 2)))
            mean_fconf += instance['fconf']
            mean_bconf += instance['bconf']
            mean_hconf += instance['hconf']
            print(f"ID: {instance['id']} {instance['fconf']} {instance['bconf']} {instance['hconf']}")
        # print(f"Len info = {len(anInterface['info'])}, info = {anInterface['info']}, Interface = {anInterface}")
        mean_fconf /= len(anInterface['info'])
        mean_bconf /= len(anInterface['info'])
        mean_hconf /= len(anInterface['info'])
        # print(f"The mean fconf is {mean_fconf}")
        # print(f"The mean bconf is {mean_bconf}")
        # print(f"The mean hconf is {mean_hconf}")
        print("")
    print("After pruning end")

    total_messages_number = 0
    for i in sliced_traces:
        print(f"Len of sliced_trace[] = {len(i)}")
        total_messages_number += len(i)
    # print(sliced_traces[0][-3])
    print(f"Total number of messages in the trace file: {total_messages_number}")




    print("--"*100)
    # print(after_pruning[0])

    selected_binary_patterns_list = []
    binary_patterns_filename = output_folder.split("/")[-1] + "_minimal_binary_patterns.txt"
    with open(binary_patterns_filename, "w") as f:
        # for i, instance in enumerate(after_pruning):
        for i, instance in enumerate(tqdm(interface_slices_info, desc="Computing minimal binary patterns")):
            patterns = minimal_edges_cover_all_nodes(instance)
            print(f"Minimal binary patterns: {patterns}")
            for pattern in patterns:
                selected_binary_patterns_list.append(pattern)
                f.write(f"{pattern}\n")
        f.write("\n")
    f.close()
    # print(minimal_edges_cover_all_nodes(after_pruning[0]))
    # print(output_folder.split('/')[-1])
    # for i, edge in enumerate(graph.edges):
    #     print(f"id: {edge}, type: {type(edge)}")

    print(f"All: {all_binary_list}")
    print(f"Selected: {selected_binary_patterns_list}")

    # Create a list of not selected binary patterns based on all_binary_list and selected_binary_patterns_list
    
    not_selected_binary_patterns_filename = output_folder.split("/")[-1] + "_not_selected_binary_patterns.txt"
    not_selected_binary_patterns_list = []
    with open(not_selected_binary_patterns_filename, "w") as f:
        for pattern in all_binary_list:
            if pattern not in selected_binary_patterns_list:
                not_selected_binary_patterns_list.append(pattern)
                f.write(f"{pattern}\n")
        f.write("\n")
    f.close()
    print(f"Not selected: {not_selected_binary_patterns_list}")





    exit()



    # for edge in graph.edges:
    #     print(f"The edge is {edge}")
    present_permutations = []
    for permutation in all_permutations:
        if permutation in graph.edges:
            edge = graph.edges.get(permutation)
            # print(f"The permutation {permutation} is in the edges")
            anInstance = {'id': permutation, 'support': edge.support, 'fconf': (edge.forward_conf), 'bconf': (edge.backward_conf), 'hconf': (edge.mean_conf)}
            present_permutations.append(anInstance)
        # else:
        #     print(f"The permutation {permutation} is not in the edges")
    print(f"The number of present permutations is {len(present_permutations)}")
    # print(f"The present permutations are {present_permutations}")
    mean_fconf = 0
    mean_bconf = 0
    mean_hconf = 0
    for instance in present_permutations:
        id    = "{0:<10}".format(str(instance['id']))
        sup   = "{0:<6}".format(str(instance['support']))
        fconf = "{0:<6}".format(str(round(instance['fconf'], 2)))
        bconf = "{0:<6}".format(str(round(instance['bconf'], 2)))
        hconf = "{0:<6}".format(str(round(instance['hconf'], 2)))
        mean_fconf += instance['fconf']
        mean_bconf += instance['bconf']
        mean_hconf += instance['hconf']
        print(f"{id} {fconf} {bconf} {hconf}")
    mean_fconf /= len(present_permutations)
    mean_bconf /= len(present_permutations)
    mean_hconf /= len(present_permutations)
    print(f"The mean fconf is {mean_fconf}")
    print(f"The mean bconf is {mean_bconf}")
    print(f"The mean hconf is {mean_hconf}")
    print(f"Length of present_permutations is {len(present_permutations)}")
    # Remove instances from present_permutations that are less than mean fconf and bconf
    present_permutations = [
        instance for instance in present_permutations
        if instance['fconf'] >= mean_fconf and instance['bconf'] >= mean_bconf and instance['hconf'] >= mean_hconf
    ]
    print(f"The number of present permutations after filtering is {len(present_permutations)}")
    print(f"The present permutations after filtering:")
    for instance in present_permutations:
        id    = "{0:<10}".format(str(instance['id']))
        sup   = "{0:<6}".format(str(instance['support']))
        fconf = "{0:<6}".format(str(round(instance['fconf'], 2)))
        bconf = "{0:<6}".format(str(round(instance['bconf'], 2)))
        hconf = "{0:<6}".format(str(round(instance['hconf'], 2)))
        print(f"{id} {fconf} {bconf} {hconf}")
    ############################################################################################################################################## By Bardia end

