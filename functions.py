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

def pruningGraph(inputGraph, graph, traceType):

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