# In this evaluation method, we will try to find the first initial message and the closest corresponding terminal message. Then the sub-trace with the 
# found initial and terminal message will be examined for the longest path that could be matched. If found, the instance will be removed from the trace 
# and it will be added to the accepted flows. It will continue to find the other available flows in the trace file until there are no flows left in the trace file.
# This is the new optimized version of the evaluation method that is proposed in the AutoModel paper.

import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
import sys
import joblib
from datetime import datetime
import scipy as sp
import time
from datetime import timedelta
from src.evaluation.linkedListDS import Node
from src.evaluation.linkedListDS import SLinkedList

class newEvaluationMethodOptimized:
	def __init__ (self, traceFilePath, availablePaths, initialNodes, terminalNodes, resultFileName, preFound=0):
		self.availablePaths = availablePaths
		self.availablePathsUsage = [[0 for x in range(len(y))] for y in self.availablePaths] 
		self.traceFilePath = traceFilePath
		self.fileName = resultFileName
		self.sizeOfAdMatrix = 155 #8
		self.adMatrix    = [[0 for x in range(self.sizeOfAdMatrix)] for y in range(self.sizeOfAdMatrix)] 
		# self.arrayOfActivePaths = [0] * self.sizeOfAdMatrix
		self.toBePrinted = ""
		self.maxIndexNode = 0
		self.tracesArray = []
		self.starting_nodes = initialNodes
		self.ending_nodes = terminalNodes
		self.all_paths_sorted = [[] for x in range(max(self.starting_nodes)+1)]  # = []
		self.startAndEnds = [[] for x in range(max(self.starting_nodes)+1)]
		self.counterForTest = 0
		self.traceList = SLinkedList()
		self.preFound = preFound


	def read_synthetic_traces(self, path):
		tempTraceArray = []
		flag = False
		try:
			with open(path, 'r') as File:
				infoFile = File.readlines() 
				eachFile = []
				for line in infoFile: 
					words = line.split(" ")
					for i in words:
						if (i != '' and i != '\n' and i != '-1' and i != '-2'):
							flag = True
							tempTraceArray.append(int(i))
						elif (i == '-2'):
							flag = False
							self.tracesArray.append(tempTraceArray)
							tempTraceArray = []
					if flag == True:
						self.tracesArray.append(tempTraceArray)
						tempTraceArray = []
		except FileNotFoundError as e:
			print ("\tError Reading Traces File!")
			exit()

	def read_jbl_traces (self, path):
		file = joblib.load(path)

		# print("Length of tracesArray = ", len(file))
		# exit()

		for i, j in enumerate(file):
			tempTrace = []
			for each in file[j]:
				tempTrace.append(int(each))
			self.tracesArray.append(tempTrace) 

	def read_inputs(self, traceFilePath):
		trace_file_type = traceFilePath.split('.')
		if trace_file_type[-1] == "txt":
			self.read_synthetic_traces(traceFilePath)
		else:
			self.read_jbl_traces(traceFilePath)

	def write_to_file (self, fileName, writeString):
		with open(fileName, 'a') as f:
    			f.write(writeString)
			    
	def checkIfPathIsAvailable(self, sub_trace, sub_traceUID):
		# print ("Length of sub_trace = ", len(sub_trace))
		# print (sub_trace)
		shouldBeRemoved = []
		shouldBeRemovedUID = []
		# print("Sub trace = ", sub_trace)
		for path_under_test in self.availablePaths[sub_trace[0]]:
			if set(path_under_test).issubset(set(sub_trace)):
				lastIndex = 0
				shouldBeRemoved = []
				shouldBeRemovedUID = []
				for anEvent in path_under_test:
					if anEvent in sub_trace[lastIndex:]:
						self.counterForTest += 1
						lastIndex += sub_trace[lastIndex:].index(anEvent)
						shouldBeRemoved.append(lastIndex)
						shouldBeRemovedUID.append(sub_traceUID[lastIndex])
					else:
						break
			if len(shouldBeRemoved) == len(path_under_test):
				self.availablePathsUsage[sub_trace[0]][self.availablePaths[sub_trace[0]].index(path_under_test)] += 1
				return shouldBeRemoved, shouldBeRemovedUID
			else:
				shouldBeRemoved = []
				shouldBeRemovedUID = []
		return 0, 0


	def find_path (self, inputTraceList):
		# toBeremoved = 0
		# totalNumberOfEvents = 0 
		totalNumberOfAccepted = 0
		# arrayOfActivePaths = [0] * self.sizeOfAdMatrix
		# checkFlag = False
		# ratio = 0
		# numberofRatios = 0

		indexCounter = 0
		i = inputTraceList.headval
		foundPathFlag = False

		printCounter = 0
		start_time = time.time()
		while i is not None:

			foundPathFlag = False
			breakFlag = False

			if i.dataval in self.starting_nodes:
				for terminal in self.startAndEnds[i.dataval]:
					j = i.nextval
					while j is not None:
						if terminal is not j.dataval:
							j = j.nextval
						# elif j.dataUID - i.dataUID > 2:
						# 	j = None
						# 	break
						else:
							# j = None
							break
					if j is not None:
						if terminal == j.dataval:
							k = i
							sub_trace = []
							sub_traceUID = []
							windowForEvaluation = 0
							while k != j.nextval:
								if k.dataUID - i.dataUID > 10000:
									indexCounter = i.dataUID
									i = i.nextval
									breakFlag = True
									break

								sub_trace.append(k.dataval)
								sub_traceUID.append(k.dataUID)
								k = k.nextval
								
							if breakFlag == False:
								remove, removeUID = self.checkIfPathIsAvailable(sub_trace, sub_traceUID)

							# start_time = time.time()

							if remove != 0 and breakFlag == False:
								foundPathFlag = True
								totalNumberOfAccepted += len(remove)
								k = i
								UIDindex = 0
								deleteFlag = False
								while k != j.nextval:
									if k.dataUID == removeUID[UIDindex]:
										if deleteFlag == False:
											i = k.nextval
										inputTraceList.remove(k)
										UIDindex += 1
									else:
										deleteFlag = True
									k = k.nextval
									if UIDindex == len(removeUID):
										break
								if UIDindex != len(removeUID):
									print("Something went wrong!")
									exit()
								elapsed_time = time.time() - start_time
								# if printCounter%30000 == 0:
								# 	print("Total removed till now =", totalNumberOfAccepted, "and UID =", indexCounter, "- it took:", timedelta(milliseconds=round(elapsed_time*1000)), "- length of sub_trace in this iteration =", len(sub_trace), "removed:", len(remove))
								printCounter += 1
								break
							else:
								foundPathFlag = False

						
			if i is not None and foundPathFlag == False:
				indexCounter = i.dataUID
				i = i.nextval


		old_method_accepted_events_number = 0


		############################################################################ For test only

		notAccepted = [0 for x in range(self.sizeOfAdMatrix)]
		totalNotAccepted = 0
		i = inputTraceList.headval
		while i is not None:
			notAccepted[i.dataval] += 1
			i = i.nextval

		#################################### Printing the not accepted ones
		# print("Not Accepted")
		# for i in range(len(notAccepted)):
		# 	if notAccepted[i]:
		# 		print (i, " : ", notAccepted[i])
		# 		totalNotAccepted += notAccepted[i]
		# print ("Not accepted in total = ", totalNotAccepted, "\n")

		notUsedPaths = []
		for i in range(len(self.availablePathsUsage)):
			if self.availablePathsUsage[i]:
				for j in range(len(self.availablePathsUsage[i])):
					if self.availablePathsUsage[i][j] == 0:
						notUsedPaths.append(self.availablePaths[i][j])
				# 	else:
				# 		print(self.availablePaths[i][j], ": ", self.availablePathsUsage[i][j])
				# print ("")
		# for apath in notUsedPaths:
		# 	print(apath)
		# print("test =", test)
		# exit()
		############################################################################ For test only

		self.toBePrinted += "\n\tAccepted Events Number = " + str(totalNumberOfAccepted + old_method_accepted_events_number)
		# print ("\tAccepted Events Number = ", totalNumberOfAccepted + old_method_accepted_events_number)
		return totalNumberOfAccepted + old_method_accepted_events_number, notAccepted, notUsedPaths

	def Evaluate(self):
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		self.toBePrinted += "\nStart Time =" + current_time
		# print("\nStart Time =", current_time)

		self.read_inputs(self.traceFilePath)
		# print("Trace Length =", len(self.tracesArray[0]))
		# exit()

		# everyInstance = [0 for x in range(self.sizeOfAdMatrix)]
		# for traces in self.tracesArray:
		# 	for event in traces:
		# 		everyInstance[event] += 1
		# 	for index in range(self.sizeOfAdMatrix):
		# 		if everyInstance[index]:
		# 			print("Index =", index, "and repeat =", everyInstance[index])
		# exit()

		self.toBePrinted += "\n\t------------------------------------------------------------------Creating the Finite State Machine------------------------------------------------------------------"
		# print ("\t------------------------------------------------------------------Creating the Finite State Machine------------------------------------------------------------------")


		################################################################################################################### For Efficiency
		# print("Starting nodes = ", self.starting_nodes)
		# print("Ending nodes = ", self.ending_nodes)

		for pathCases in self.availablePaths:
			for paths in pathCases:
				if paths[-1] not in self.startAndEnds[paths[0]]:
					self.startAndEnds[paths[0]].append(paths[-1])
		# print ("test startAndEnds: ", self.startAndEnds)		
		################################################################################################################### For Efficiency

		self.toBePrinted += "\n\tDone creating the Finite State Machine!"
		self.toBePrinted += "\n\t----------------------------------------------------------------From now on we are testing the parser----------------------------------------------------------------"
		# print ("\tDone creating the Finite State Machine!")
		# print ("\t----------------------------------------------------------------From now on we are testing the parser----------------------------------------------------------------")


		# finalRatio = 0
		finalNumberofRatios = 0
		totalCountOfEvents = 0
		totalAcceptedEventsNumber = 0

		# print ("\n")
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		# print("\tCurrent Time =", current_time)
		self.toBePrinted += "\n\n\tCurrent Time =" + current_time


		for traceIndex in range(len(self.tracesArray)):

			count = 0
			self.traceList = SLinkedList()
			# self.traceList.headval = Node(self.tracesArray[0])
			for index in range(len(self.tracesArray[traceIndex])):
				self.traceList.AddEnd(self.tracesArray[traceIndex][index], count)
				# if count%1000000 == 0:
				# 	print ("Count = ", count)
				count += 1
			# print("Length of trace = ", self.traceList.length())
			# exit()
			# self.traceList.listprint()
			# exit()

			totalCountOfEvents += len(self.tracesArray[traceIndex])
			self.toBePrinted += "\n\tTrace number " + str(traceIndex) + " : "
			# print ("\tTrace number ", traceIndex, " : ")
			# acceptedEventsNumber = self.find_path(self.tracesArray[traceIndex])

			acceptedEventsNumber, notAccepted, notUsedPaths = self.find_path(self.traceList)
			# acceptedEventsNumber = find_path_old_method(tracesArray[traceIndex])

			totalAcceptedEventsNumber += acceptedEventsNumber
			# finalRatio += R
			finalNumberofRatios += 1



		self.toBePrinted += "\n\tTotal number of events = " + str(totalCountOfEvents)
		self.toBePrinted += "\n\tNumber of traces = " + str(finalNumberofRatios)
		self.toBePrinted += "\n\tThe final answer is : " + str(totalAcceptedEventsNumber/totalCountOfEvents)
		# toBePrinted += "\n\tThe final answer is : " + str(finalRatio/finalNumberofRatios)
		# print ("\tTotal number of events = ", totalCountOfEvents)
		# print ("\tNumber of traces = ", finalNumberofRatios)
		# print ("\tThe final answer is : ", (totalAcceptedEventsNumber/totalCountOfEvents) )
		# print ("\tThe final answer is : ", (finalRatio/finalNumberofRatios) )


		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		# print("\nEnd Time =", current_time)
		# print ("")
		self.toBePrinted += "\n\nEnd Time =" + current_time + "\n\n\n\n\n\n\n\n"


		# FSMSize = 0
		# for i in range(sizeOfAdMatrix):
		# 	for j in range(sizeOfAdMatrix):
		# 			if FSMadMatrix[i][j] == 1:
		# 				FSMSize += 1
		# print ("FSM Size = ", FSMSize)
		# toBePrinted += "\n FSM Size =" + str(FSMSize) + "\n\n\n\n\n\n\n\n"

		self.write_to_file(self.fileName, self.toBePrinted)
		# print("Counter for test = ", self.counterForTest)
		return ((totalAcceptedEventsNumber+self.preFound)/(totalCountOfEvents+self.preFound)), ((totalAcceptedEventsNumber+self.preFound)/(totalCountOfEvents+self.preFound)), notAccepted, notUsedPaths 
