#!/usr/bin/env python
# encoding: utf-8
"""
Plotting functions 
Created by Stijn Nuiten on 2019-03-28.
Copyright (c) 2019. All rights reserved.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as stats 
from math import *
from IPython import embed as shell
from itertools import combinations

def label_diff(ax,means,stds,txt,mark = True):
	# This method draws significans boxes in bar-plots. Up until now this works for barplots with 3 bars (i.e. 3 significance boxes).
	# Inputs are the handle of a (sub-)figure, the mean values and errors of the barplot, and lastly the actual text labels in a list.
	# TO ADD:
	# - automatically add ** when p<0.05/0.005
	# - exclude specific comparissons
	# - more than 3 bars
	means=np.array(means)
	stds = np.array(stds) 
	lims = means + stds
	num_bars = len(means)
	num_diff = factorial(num_bars)/2
	h = np.max(lims)*2*0.1

	if num_diff > 1:
		lines = [abs(a - b) for a, b in combinations(range(num_diff), 2)]
	else:
		lines = [1]
	bar = 0

	for d in range(num_diff):
		if mark:
			if float(txt[0].split(' ')[-1]) < 0.05:
				txt[d] = '*'
			elif float(txt[0].split(' ')[-1]) <0.005:
				txt[d] = '**'
			elif float(txt[0].split(' ')[-1]) > 0.05:
				txt[d] = 'n.s.'

		if lines[d]==1 and d>0:
			bar =+ 1
		x = [bar, bar + lines[d]]

		high=np.array(means[x[0]:x[1]+1])+np.array(stds[x[0]:x[1]+1])
		low=np.array(means[x[0]:x[1]+1])-np.array(stds[x[0]:x[1]+1])		
		maxdat = np.max((high,low))
		mindat = np.min((high,low))
		if abs(mindat) > abs(maxdat):
			y = mindat -  h - ((lines[d]-1) * 0.2 )
		else:
			y = maxdat +  h + ((lines[d]-1) * 0.2 )

		ax.plot([x[0], x[0], x[1], x[1]], [y, y+h, y+h, y], lw=1.5, color='k')
		ax.text((x[0]+x[1])*.5, y+h, txt[d], ha='center', va='bottom')

	return ax


	
