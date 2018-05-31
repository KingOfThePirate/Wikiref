import os

f = open("./4_hops/Computer_engineering_4_hops_category","r").read().splitlines()

for line in f:
	print line.split("/")[-1]
