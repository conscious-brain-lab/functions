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
import seaborn as sns
from math import *
from IPython import embed as shell
from itertools import combinations

def label_diff(ax,means,stds,txt,width=None,pos=0, mark = True):
	# This method draws significans boxes in bar-plots. Up until now this works for barplots with 3 bars (i.e. 3 significance boxes).
	# Inputs are the handle of a (sub-)figure, the mean values and errors of the barplot, and lastly the actual text labels in a list.
	# TO ADD:
	# - exclude specific comparissons
	# - more than 3 bars
	# shell()
	means=np.array(means)
	stds = np.array(stds) 
	# lims = means + stds
	num_bars = len(means)
	num_diff = factorial(num_bars)/2
	h =  np.diff(ax.get_ylim())[0] *0.05

	# abs(np.array(means)).max()*0.25 


	if num_diff > 1:
		lines = [abs(a - b) for a, b in combinations(range(num_diff), 2)]
	else:
		lines = [1]
	bar = 0

	for d in range(int(num_diff)):
		if mark:
			if float(txt[0].split(' ')[-1]) < 0.05:
				txt[d] = '*'
			elif float(txt[0].split(' ')[-1]) <0.005:
				txt[d] = '**'
			elif float(txt[0].split(' ')[-1]) > 0.05:
				txt[d] = 'n.s.'

		if lines[d]==1 and d>0:
			bar =+ 1
		# shell()
		if width:
			x =[bar-width, bar+width]
		else:
			x = [bar, bar + lines[d]]

		x = [x[0] + pos , x[1] +pos]
		high=means + stds
		low=means - stds		
		# shell()
		maxdat = np.max((high,low))
		mindat = np.min((high,low))
		maxval = np.max([abs(mindat),abs(maxdat)])
		y_box = maxval + h
		y_text = y_box + 3* h
		# h = [lim*1.5 for lim in lims]
		# shell()
		if abs(mindat) > abs(maxdat):
			if mindat < 0 :
				y_box = -y_box 
				y_text = -y_text
				h = -h

		else:
			if maxdat < 0:
				y_box = -y_box 
				y_text = -y_text
				h=-h
		# shell()
		ax.plot([x[0], x[0], x[1], x[1]], [y_box, y_box+h, y_box+h, y_box], lw=1.5, color='k')
		ax.text((x[0]+x[1])*.5, y_text, txt[d], ha='center', va='center')
	return ax


def rainbowPlot(dx,dy,data,ort='h'):
	#adding color
	pal = sns.colorpalette('RdBu_r')
	f, ax = plt.subplots(figsize=(7, 5))
	ax=pt.half_violinplot( x = dx, y = dy, data = df, palette = pal,
	     bw = .2, cut = 0.,scale = "area", width = .6, 
	     inner = None, orient = ort)

	ax=sns.stripplot( x = dx, y = dy, data = df, palette = pal,
	      edgecolor = "white",size = 3, jitter = 1, zorder = 0,
	      orient = ort)

	ax=sns.boxplot( x = dx, y = dy, data = df, color = "black",
	      width = .15, zorder = 10, showcaps = True,
	      boxprops = {'facecolor':'none', "zorder":10}, showfliers=True,
	      whiskerprops = {'linewidth':2, "zorder":10}, 
	      saturation = 1, orient = ort)	

	return fig,ax
