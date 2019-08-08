import csv
import os

######################################################################
# Author: 	Ian Ludden
# Date: 	08 August 2019
# 
# samplingFunctions.py
# 
# Utility functions for sampling NCAA tournament seeds 
# from distributions. 
# 
######################################################################

def sampleNCG(year):
	"""Returns..."""
	# TODO: implement
	if not os.exists(filepath):
		exit('Cannot find file with name \'{0}\'.'.format(filepath))

	with open(filepath, 'r') as f:
		reader = csv.reader(f)
		rawData = list(reader)

	return rawData
