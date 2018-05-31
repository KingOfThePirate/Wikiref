import os,sys
import numpy as np

in_degree = open(sys.argv[1],"r").read().splitlines()

min_y = list()
min_n = list()
max_y = list()
max_n = list()
mean_y = list()
mean_n = list()
median_y = list()
median_n = list()

for x in range(0,len(in_degree)):
	if in_degree[x] == "$$$$Result_Y$$$$":
		min_y.append(float(in_degree[x+1].split(":")[1]))
		max_y.append(float(in_degree[x+2].split(":")[1]))
		median_y.append(float(in_degree[x+3].split(":")[1]))
		mean_y.append(float(in_degree[x+4].split(":")[1]))
		if in_degree[x+5] == "$$$$Result_N$$$$":
			min_n.append(float(in_degree[x+6].split(":")[1]))
			max_n.append(float(in_degree[x+7].split(":")[1]))
			median_n.append(float(in_degree[x+8].split(":")[1]))
			mean_n.append(float(in_degree[x+9].split(":")[1]))
		else:
			min_n.append(0.0)
			max_n.append(0.0)
			median_n.append(0.0)
			mean_n.append(0.0)

if len(mean_y) != len(mean_n):
	print "INcorrect"

print "Min Yes", round(sum(min_y)/len(min_y),3)
print "Max Yes", round(sum(max_y)/len(max_y),3)
print "Mean Yes", round(sum(mean_y)/len(mean_y),3)
print "Median Yes", round(sum(median_y)/len(median_y),3)

print "Min No", round(sum(min_n)/len(min_n),3)
print "Max No", round(sum(max_n)/len(max_n),3)
print "Mean No", round(sum(mean_n)/len(mean_n),3)
print "Median No", round(sum(median_n)/len(median_n),3)