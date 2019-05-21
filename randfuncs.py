#!/usr/bin/env python
# encoding: utf-8
"""
Random functions

Created by Stijn Nuiten on 2019-03-29.
Copyright (c) 2019. All rights reserved.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as stats 
from math import *
from IPython import embed as shell
from itertools import combinations


def SDT(target, hit, fa):
	"""
	Calculate d' and criterion     """

	target = np.array(target, dtype=bool)
	hit = np.array(hit, dtype=bool)
	fa = np.array(fa, dtype=bool)
	hit_rate = (np.sum(hit)) / (float(np.sum(target)))
	fa_rate = (np.sum(fa)) / (float(np.sum(~target)))

	if hit_rate == 1:
		hit_rate = hit_rate-0.00001
	elif hit_rate == 0:
		hit_rate = hit_rate+0.00001

	if fa_rate == 1:
		fa_rate = fa_rate-0.00001
	elif fa_rate == 0:
		fa_rate = fa_rate+0.00001

	hit_rate_z = stats.norm.isf(1-hit_rate)
	fa_rate_z = stats.norm.isf(1-fa_rate)

	d = hit_rate_z - fa_rate_z
	c = -(hit_rate_z + fa_rate_z) / 2.0

	return(d, c, hit_rate, fa_rate)
