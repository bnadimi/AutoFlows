# This script is used to find the local patterns in the trace file

import os
import networkx as nx

from src.graph.graph import Graph
import time
from src.logging import *
import math
from src.evaluation.newEvaluationMethod import newEvaluationMethod
from src.evaluation.newEvaluationMethodOptimized import newEvaluationMethodOptimized
from src.evaluation.maxMatchedPaths import maxMatchedPaths
from src.evaluation.backTrackingEvaluation import backTrackingEvaluation
from src.evaluation.linkedListDS import Node
from src.evaluation.linkedListDS import SLinkedList

from datetime import timedelta

from numpy.random import seed
from numpy.random import randint

from copy import copy, deepcopy
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

def permutation(lst):
    if len(lst) == 0:
        return []
 
    if len(lst) == 1:
        return [lst]
 
    l = []
    for i in range(len(lst)):
       m = lst[i]
       remLst = lst[:i] + lst[i+1:]

       for p in permutation(remLst):
           l.append([m] + p)
    return l
 

# def bruteForce(inputPaths, currentIndex):
#     if currentIndex == 

#     if len(inputPaths) == 0:
#         return []
#     if len(inputPaths) == 1 and inputPaths:
#         return permutation(inputPaths)

def pruningGraph(inputGraph, graph, traceType, not_selected_BP_graph_intSliced, interface_slicing=True):

    if interface_slicing:
        edges_to_remove = []
        for edge in not_selected_BP_graph_intSliced.edges():
            if inputGraph.has_edge(*edge):
                edges_to_remove.append(edge)
        # print(f"Edges to remove: {edges_to_remove}")
        inputGraph.remove_edges_from(edges_to_remove)


    # not_selected_graph = nx.DiGraph()
    # # Read the file "new-trace-large-20_not_selected_binary_patterns" and print its contents
    # try:
    #     with open("new-trace-large-20_not_selected_binary_patterns.txt", "r") as f:
    #         not_selected_patterns = [line.strip() for line in f if line.strip()]
    #         print("Contents of new-trace-large-20_not_selected_binary_patterns:")
    #         for pattern in not_selected_patterns:
    #             src_node = int(pattern.split("_")[0])
    #             dest_node = int(pattern.split("_")[1])
    #             not_selected_graph.add_edge(src_node, dest_node)
    #             # print(pattern)
    # except FileNotFoundError:
    #     print("File 'new-trace-large-20_not_selected_binary_patterns' not found.")
    # # for anEdge in inputGraph.edges():
    # #     print(anEdge)
    # # print("Not selected binary patterns")
    # # for anEdge in not_selected_graph.edges():
    # #     print(anEdge)

    # # INSERT_YOUR_CODE
    # # Remove the edges from inputGraph that are present in not_selected_graph
    # edges_to_remove = []
    # for edge in not_selected_graph.edges():
    #     if inputGraph.has_edge(*edge):
    #         edges_to_remove.append(edge)
    # # print(f"Edges to remove: {edges_to_remove}")
    # inputGraph.remove_edges_from(edges_to_remove)
    # # exit()

    all_edge_supports = []
    for edge in inputGraph.edges:
        src, dest = edge
        all_edge_supports.append(graph.get_edge(src, dest).get_support())

    all_edge_supports.sort()
    mid = math.floor(len(all_edge_supports)*35/51)
    print ("Threshold position is ", mid, " out of ", len(all_edge_supports), "variables!")
    edgeSupportThreshold = all_edge_supports[mid]

    if traceType == "synthetic":
        ############################ For Large-20 
        fConfThreshold = 0.5 #0.95
        bConfThreshold = 0.5 #0.95
        mConfThreshold = 1

        pruningOption = ['forward', 'backward']
    else:
        ############################ For threads 
        fConfThreshold = 0
        bConfThreshold = 0
        mConfThreshold = 0

        pruningOption = ['forward', 'mean']

    print ("Threshold value is ", edgeSupportThreshold, "!\n")

    
    ########################## Pruning the causality graph based on edge support, forward confidence, backward confidence, and mean confidence. ###################
    pruned_graph = nx.DiGraph()
    for edge in inputGraph.edges:
        src, dest = edge

        if str(src) in graph.root_nodes or str(dest) in graph.terminal_nodes:
            pruned_graph.add_edge(src, dest)

        ############## trimming based on edge support 
        if 'edgeSupport' in pruningOption:
            if graph.get_edge(src, dest).get_support() >= edgeSupportThreshold:
                pruned_graph.add_edge(src, dest)

        ############## trimming based on edge forward support 
        if 'forward' in pruningOption:
            if graph.get_edge(src, dest).get_fconf() >= fConfThreshold:
                pruned_graph.add_edge(src, dest)

        ############## trimming based on edge backward support 
        if 'backward' in pruningOption:
            if graph.get_edge(src, dest).get_bconf() >= bConfThreshold:
                pruned_graph.add_edge(src, dest)

        ############## trimming based on edge mean support 
        if 'mean' in pruningOption:
            if graph.get_edge(src, dest).get_hconf() >= mConfThreshold:
                pruned_graph.add_edge(src, dest)

    print ("Before trimming : ", inputGraph.number_of_edges())
    print ("After trimming : ", pruned_graph.number_of_edges())
    for e in pruned_graph.edges:
        print(e)
    ########################## Pruning the causality graph based on edge support, forward confidence, backward confidence, and mean confidence. ####### end #######

    return pruned_graph


def pathPoolFinder(graph, causalityGraph, initialNodes, correspondingTerminalNodes):

    pathsPool = [[] for x in range(graph.maxInitials+1)]  # = []
    numOfFoundPaths = 0

    ##################################### Finding paths' pool without graph slicing (Tested) ###########
    # for anInitialNode in initialNodes:
    #     if correspondingTerminalNodes[anInitialNode]:
    #         print("Initial node =", anInitialNode, "and Terminal node =", correspondingTerminalNodes[anInitialNode][0])
    #         for path in nx.all_simple_paths(causalityGraph, source=anInitialNode, target=correspondingTerminalNodes[anInitialNode][0]):
    #             pathsPool[anInitialNode].append(path)
    #             numOfFoundPaths += 1 #len(path)
    #         print ("Number of all paths found till now = ", numOfFoundPaths, "\n")  

    # return pathsPool
    ##################################### Finding paths' pool without graph slicing (Tested) ### end ###

    ####################################### Finding paths' pool with graph slicing (Reverse graph check) ###########
    reversedCausalityGraph = nx.reverse(causalityGraph) 
    for anInitialNode in initialNodes:
        print("##########################################################################")
        subGraph = nx.DiGraph()
        print(correspondingTerminalNodes[anInitialNode])
        for aTerminalNode in correspondingTerminalNodes[anInitialNode]:
            # if correspondingTerminalNodes[anInitialNode]:
            if aTerminalNode:
                # print("Initial node =", anInitialNode, "and Terminal node =", correspondingTerminalNodes[anInitialNode][0])
                print("Initial node =", anInitialNode, "and Terminal node =", aTerminalNode)
            
                activeNodes = [anInitialNode]
                testSubGraph = nx.DiGraph()
                visited = [False for i in range(155)]
                while len(activeNodes)!=0:
                    node = activeNodes[0]
                    for n in causalityGraph.neighbors(node):
                        testSubGraph.add_edge(node, n)
                        if visited[n] == False:
                            activeNodes.append(n)
                        visited[n] = True
                    activeNodes.pop(0)
                ########################################## test for reversed graph    
                # activeNodes = [correspondingTerminalNodes[anInitialNode][0]]
                activeNodes = [aTerminalNode]
                testSubGraphReversed = nx.DiGraph()
                visited = [False for i in range(155)]
                while len(activeNodes)!=0:
                    node = activeNodes[0]
                    for n in reversedCausalityGraph.neighbors(node):
                        testSubGraphReversed.add_edge(node, n)
                        if visited[n] == False:
                            activeNodes.append(n)
                        visited[n] = True
                    activeNodes.pop(0)
                testSubGraphReversed = nx.reverse(testSubGraphReversed)

                print("------------------------------------------")
                for e1 in testSubGraph.edges:
                    s1, d1 = e1
                    for e2 in testSubGraphReversed.edges:
                        s2, d2 = e2
                        if s1 == s2 and d1 == d2:
                            subGraph.add_edge(s1, d1)

                print(subGraph)
                # for e in subGraph.edges:
                #     print(e)
                # for path in nx.all_simple_paths(subGraph, source=anInitialNode, target=correspondingTerminalNodes[anInitialNode][0], cutoff=9):
                for path in nx.all_simple_paths(subGraph, source=anInitialNode, target=aTerminalNode, cutoff=9):
                    pathsPool[anInitialNode].append(path)
                    numOfFoundPaths += 1 #len(path)
                print ("Number of all paths found till now = ", numOfFoundPaths, "\n")

    # for ee in pathsPool:
    #     for eee in ee:
    #         print(eee)
    # exit()
    return pathsPool
    ####################################### Finding paths' pool with graph slicing (Reverse graph check) ### end ###

def modelSelector(pathPool, graph):

    selected_paths   = [[] for x in range(graph.maxInitials+1)] 

    ############################ Sorting paths by the length and selecting the longest path from each initial node ###################
    for path in pathPool:
        if path:
            # Chosing the smallest path
            path.sort(key=len, reverse=False)  
            selected_paths[path[0][0]].append(path[0])
            path.pop(0)

            # Sorting longest to shortest and selecting ()
            # path.sort(key=len, reverse=True) 
            # i = 0
            # while i < len(path):
            #     if len(path[i]) <= 8:
            #         # print("Test", path[0][0])
            #         selected_paths[path[0][0]].append(path[i])
            #         break
            #     else:
            #         i += 1
            # path.sort(key=len, reverse=False) 

            # applying a threshold on the paths sizes
            # path.sort(key=len, reverse=True) 
            # i = 0
            # while i < len(path):
            # # for i in range(len(path)):
            #     if len(path[i]) > 10:
            #         path.pop(i)
            #         # del path[i]
            #     else:
            #         i += 1

            # selected_paths[path[0][0]].append(path[-1])
    # print(all_paths_sorted)
    # exit()
    ############################ Sorting paths by the length and selecting the longest path from each initial node ####### end #######
    coverage_test = graph.all_messages
    coverage_test.sort()
    print ("\n1- First coverage array = ", coverage_test)
    print ("\tLength = ", len(coverage_test))

    print ("Selected path with first paths from each initial node = ", selected_paths)
    for paths in selected_paths:
        for aPath in paths:
            for message in aPath:
                if (message in coverage_test):
                    coverage_test.remove(message)
    
    print ("\n2- coverage array after selecting the first paths = ", coverage_test)
    print ("\tlength = ", len(coverage_test))

    # sizeOfAllPaths = 0
    # for aPath in pathPool:
    #     sizeOfAllPaths += len(aPath)
    # print ("Length of remaining paths in the array = ", sizeOfAllPaths)

    ########################################### satistying coverage requirement ###############
    # index = 0
    # while(coverage_test):
    #     index += 1
    #     if pathPool:
    #         for pathClasses in pathPool:
    #             if pathClasses:
    #                 # print("Len Coverage =", len(coverage_test))
    #                 selected_paths[pathClasses[0][0]].append(pathClasses[0])
    #                 for messages in pathClasses[0]:
    #                     if (messages in coverage_test):
    #                         coverage_test.remove(messages)
    #                 pathClasses.pop(0)
                            
    #                 if not coverage_test:
    #                     break;
    #             else:
    #                 pathPool.remove(pathClasses)
    #     else:
    #         print ("coverage array on exit= ", coverage_test)
    #         break;
    ########################################### satistying coverage requirement ##### end #####
    ########################################### satistying coverage requirement ###############
    coverageImprovement = len(coverage_test)
    index = 0
    chooseIndex = 0
    totalLimitCounter = 0
    while(coverage_test and totalLimitCounter < 1000000):
        index += 1
        if pathPool:
            for pathClasses in pathPool:
                chooseIndex = 0
                if pathClasses:
                    while chooseIndex < len(pathClasses):
                        totalLimitCounter += 1
                        # print("Len Coverage =", len(coverage_test))
                        # exit()
                        deletedMessages = []
                        for messages in pathClasses[chooseIndex]:
                            if (messages in coverage_test):
                                coverage_test.remove(messages)
                                deletedMessages.append(messages)
                        if len(coverage_test) < coverageImprovement:
                            print("Len coverage =", len(coverage_test), "- coverageImprovement =", coverageImprovement)
                            print("coverage_test: ", coverage_test)
                            # selected_paths[pathClasses[0][0]].append(pathClasses[0])
                            selected_paths[pathClasses[0][0]].append(pathClasses[chooseIndex])
                            # pathClasses.pop(0)
                            pathClasses.pop(chooseIndex)
                            coverageImprovement = len(coverage_test)
                        else:
                            for messages in deletedMessages:
                                if (messages not in coverage_test):
                                    coverage_test.append(messages)
                            chooseIndex += 1
                            # if chooseIndex >= len(pathClasses):
                            #     chooseIndex = 0

                            # temp = pathClasses[0]
                            # pathClasses[0] = pathClasses[-1]
                            # pathClasses[-1] = pathClasses[0]
                            coverage_test.sort()
                            
                        if not coverage_test:
                            break;
                else:
                    pathPool.remove(pathClasses)
        else:
            print ("coverage array on exit= ", coverage_test)
            break;
    ########################################### satistying coverage requirement ##### end #####

    for path in selected_paths:
        if path:
            path.sort(key=len, reverse=True)

    print ("\n3- Final coverage array = ", coverage_test)
    print ("\tLength = ", len(coverage_test))
    
    selectedPathLength = 0
    for eachPathCase in selected_paths:
        selectedPathLength += len(eachPathCase)
    print ("selected path length = ", selectedPathLength)
    print ("selected path = ", selected_paths)

    return selected_paths

def modelrefinement(trace_file, pathPool, selected_paths, initialNodes, terminalNodes, preFound):

    # list1 = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    # list2 = deepcopy(list1)
    # print(list1)
    # print(list2)
    # list2[0].append(10)
    # print(list2)
    # list2 = []
    # list2 = deepcopy(list1)
    # print(list1)
    # print(list2)
    # exit()

    ############################################################ First Evaluation Phase
    evaluationStartTime = time.time()
    resultFileName = "test.txt"
    # ev = newEvaluationMethod(trace_file, selected_paths, initialNodes, terminalNodes, resultFileName)
    # res1, res2 = ev.Evaluate()
    ev = newEvaluationMethodOptimized(trace_file, selected_paths, initialNodes, terminalNodes, resultFileName, preFound)
    res1, res2, notAccepted, notUsedPaths = ev.Evaluate()
    # ev = maxMatchedPaths(trace_file, selected_paths, initialNodes, terminalNodes, resultFileName)
    # res1, res2 = ev.Evaluate()
    # ev = backTrackingEvaluation(trace_file, selected_paths, initialNodes, terminalNodes, resultFileName)
    # res1, res2 = ev.Evaluate()
    for apath in notUsedPaths:
        for i in range(len(selected_paths[apath[0]])):
            if selected_paths[apath[0]][i] == apath:
                del selected_paths[apath[0]][i]
                break

        # selected_paths[apath[0]].pop(apath)

    modelSize = 0
    for apath in selected_paths:
        if apath:
            modelSize += len(apath)
            # print(apath)
    # exit()
    print("Model Size =", modelSize)

    ############################################################ Model Refinement Phase
    maxNotAccepted = max(notAccepted)
    maxNotAcceptedIndex = notAccepted.index(maxNotAccepted)
    print("Max = ", maxNotAccepted, "And index = ", maxNotAcceptedIndex)

    testCounter = 0
    refinedModel = deepcopy(selected_paths) #selected_paths.copy()
    for i in range(55):
        for pathClass in pathPool:
            if pathClass:
                for path in pathClass:
                    if (maxNotAcceptedIndex in path) and (path not in refinedModel[path[0]]):
                        refinedModel[path[0]].append(path)
                        refinedModel[path[0]].sort(key=len, reverse=True)
                        
                        ev = newEvaluationMethodOptimized(trace_file, refinedModel, initialNodes, terminalNodes, resultFileName, preFound)
                        testRes1, testRes2, notAccepted, notUsedPaths = ev.Evaluate()


                        print("i =", i,"Initial =", path[0], "Counter =", testCounter, "- First result =", res1, "- Test reslt =", testRes1, "looking for:", maxNotAcceptedIndex, end="\r")
                        testCounter += 1
                        if testRes1 > res1:
                            print("Found =", path)
                            selected_paths = []
                            selected_paths = deepcopy(refinedModel) #refinedModel.copy()
                            res1 = testRes1

                            maxNotAccepted = max(notAccepted)
                            maxNotAcceptedIndex = notAccepted.index(maxNotAccepted)
                            print("Max = ", maxNotAccepted, "And index = ", maxNotAcceptedIndex)
                            break
                            # exit()
                        else:
                            refinedModel = []
                            refinedModel = deepcopy(selected_paths) #selected_paths.copy()
                        if testCounter > 10:
                            break
        if testCounter > 1:
            break
                        # print("path = ", path)
                        # print("Initial node = ", path[0])
                        # exit()
                        # selected_paths[]
    totalNotAccepted = 0
    print("Not Accepted")
    for i in range(len(notAccepted)):
        if notAccepted[i]:
            print (i, " : ", notAccepted[i])
            totalNotAccepted += notAccepted[i]
    print ("Not accepted in total = ", totalNotAccepted, "\n")

    finalModelSize = 0
    for apath in selected_paths:
        if apath:
            print(apath)
            finalModelSize += len(apath)
    print("Final Model Size =", finalModelSize)
    # print("Exited by me!")
    # exit()
    # print("Final Model = ", selected_paths)

    evaluationTime = time.time() - evaluationStartTime
    msg = "\nEvaluation phase took: %s secs (Wall clock time)" % timedelta(milliseconds=round(evaluationTime*1000))
    print(msg)

    return res1


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


def interface_slicing(trace_files, definition_file, graph, pruning=False):
    # Prepare output files for each interface, and a mapping from message to the correct file
    interface_files = []
    message_to_file = {}
    
    ##############################################################################################################################################  Slicing traces based on the interfaces ################################################################
    prev_cwd = os.getcwd()
    # Extract the main trace file's name without extension or directories
    if isinstance(trace_files, list) and len(trace_files) > 0:
        trace_path = trace_files[0]
    elif isinstance(trace_files, str):
        trace_path = trace_files
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

    all_permutations = []
    for interface in graph.interfaces:
        perms = []
        for index in interface['list']:
            for next_index in interface['list']:
                if index != next_index:
                    perms.append(f"{index}_{next_index}")
        all_permutations.append(perms)

    os.chdir(prev_cwd)

    interface_slices_info = []
    for i, aTrace in enumerate(traces_list):

        graph = Graph()
        # graph.set_max_height(max_pat_len)
        # graph.set_max_solutions(max_solutions)

        graph.window = False
        graph.window_size = 50

        graph.read_message_file(definition_file)
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
        # print(f"pairs: {anInterface['pairs']}, List: {anInterface['list_of_messages']}")
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
            # print(f"ID: {instance['id']} {instance['fconf']} {instance['bconf']} {instance['hconf']}")
        mean_fconf /= len(anInterface['info'])
        mean_bconf /= len(anInterface['info'])
        mean_hconf /= len(anInterface['info'])
        # print(f"The mean fconf is {mean_fconf}")
        # print(f"The mean bconf is {mean_bconf}")
        # print(f"The mean hconf is {mean_hconf}")

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


            
        # print(f"Len of sliced_trace[{i}] = {len(anInterface['trace'])}")
        # print("")


    # # print("After Pruning")
    # for i in range(len(after_pruning)):
    #     anInterface = after_pruning[i]
    #     # print(graph.interfaces[i])
    #     # print(f"pairs: {anInterface['pairs']}, List: {anInterface['list_of_messages']}")
    #     mean_fconf = 0
    #     mean_bconf = 0
    #     mean_hconf = 0
    #     for instance in anInterface['info']:
    #         id    = "{0:<10}".format(str(instance['id']))
    #         sup   = "{0:<6}".format(str(instance['support']))
    #         fconf = "{0:<6}".format(str(round(instance['fconf'], 2)))
    #         bconf = "{0:<6}".format(str(round(instance['bconf'], 2)))
    #         hconf = "{0:<6}".format(str(round(instance['hconf'], 2)))
    #         mean_fconf += instance['fconf']
    #         mean_bconf += instance['bconf']
    #         mean_hconf += instance['hconf']
    #         # print(f"ID: {instance['id']} {instance['fconf']} {instance['bconf']} {instance['hconf']}")
    #     # print(f"Len info = {len(anInterface['info'])}, info = {anInterface['info']}, Interface = {anInterface}")
    #     mean_fconf /= len(anInterface['info'])
    #     mean_bconf /= len(anInterface['info'])
    #     mean_hconf /= len(anInterface['info'])
    #     # print(f"The mean fconf is {mean_fconf}")
    #     # print(f"The mean bconf is {mean_bconf}")
    #     # print(f"The mean hconf is {mean_hconf}")
    #     # print("")
    # # print("After pruning end")

    total_messages_number = 0
    for i in sliced_traces:
        # print(f"Len of sliced_trace[] = {len(i)}")
        total_messages_number += len(i)
    # print(sliced_traces[0][-3])
    # print(f"Total number of messages in the trace file: {total_messages_number}")

    print("--"*100)
    # print(after_pruning[0])

    selected_binary_patterns_graph     = nx.DiGraph()
    not_selected_binary_patterns_graph = nx.DiGraph()
    selected_binary_patterns_list = []
    binary_patterns_filename = output_folder.split("/")[-1] + "_minimal_binary_patterns.txt"
    with open(binary_patterns_filename, "w") as f:
        if pruning:
            description = "Computing minimal binary patterns (Pruned based on Mean Confidences)"
            slices = after_pruning
        else:
            description = "Computing minimal binary patterns (Not Pruned)"
            slices = interface_slices_info
        # for i, instance in enumerate(tqdm(after_pruning, desc="Computing minimal binary patterns (Pruned based on Mean Confidences)")):
        # for i, instance in enumerate(tqdm(interface_slices_info, desc="Computing minimal binary patterns (Not Pruned)")):
        for i, instance in enumerate(tqdm(slices, desc=description)):
            patterns = minimal_edges_cover_all_nodes(instance)
            # print(f"Minimal binary patterns: {patterns}")
            for pattern in patterns:
                selected_binary_patterns_list.append(pattern)
                src_node  = int(pattern.split("_")[0])
                dest_node = int(pattern.split("_")[1])
                selected_binary_patterns_graph.add_edge(src_node, dest_node)
                f.write(f"{pattern}\n")
        f.write("\n")
    f.close()

    # print(f"All: {all_binary_list}")
    # print(f"Selected: {selected_binary_patterns_list}")
    
    not_selected_binary_patterns_filename = output_folder.split("/")[-1] + "_not_selected_binary_patterns.txt"
    not_selected_binary_patterns_list = []
    with open(not_selected_binary_patterns_filename, "w") as f:
        for pattern in all_binary_list:
            if pattern not in selected_binary_patterns_list:
                not_selected_binary_patterns_list.append(pattern)
                src_node  = int(pattern.split("_")[0])
                dest_node = int(pattern.split("_")[1])
                not_selected_binary_patterns_graph.add_edge(src_node, dest_node)
                f.write(f"{pattern}\n")
        f.write("\n")
    f.close()
    # print(f"Not selected: {not_selected_binary_patterns_list}")

    return selected_binary_patterns_graph, not_selected_binary_patterns_graph

