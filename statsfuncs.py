#!/usr/bin/env python
# encoding: utf-8
"""
Statistical functions

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
from scipy.ndimage import measurements

	
def prevInference( data, perm1=100, perm2=10000, alpha=0.05,gamma0=0.5):
	# Prevalence inference analysis for decoding accuracies.
	# Derived from iPython Notebook by Lukas Snoek:
	# https://github.com/lukassnoek/random_notebooks/blob/master/prevalence_inference.ipynb
	N = len(data)
	K = data[0].shape[0]

	perms = np.random.normal(loc=0.5, scale=0.05, size=(N,K , perm1))

	m = np.min(data,axis=0)
	u_rank = np.zeros(K)  # uncorrected
	c_rank = np.zeros(K)  # corrected
	for it in range(perm2):
	    these_perms = np.vstack([perms[k, :, np.random.choice(np.arange(perm1), size=1)]
	                             for k in range(N)])
	    min_vals = these_perms.min(axis=0)
	    u_rank += m <= min_vals
	    c_rank += m <= min_vals.max()
	
	pu_GN = (1 + u_rank) / (perm2 + 1)
	pc_GN = (1 + c_rank) / (perm2 + 1)
	pu_MN = ((1 - gamma0) * pu_GN ** (1 / N) + gamma0) ** N
	pc_MN = pc_GN + (1 - pc_GN) * pu_MN

	return pc_MN, pc_GN


def cluster_ttest(cond1, cond2, n_permutes, pval):
	# cluster_ttest runs cluster-base corrected pairwise t-tests for time-frequency data. 
	# This function has been adapted from scripts by Michael X Cohen.
	# Inputs are:
	#     - data for two conditions you want to test (subject x frequency x time)
	#     - number of permutations
	#     - threshold p-value
	# It returns the t-values for significant clusters (non-cluster t-values are set to 0).
			
	zval=sp.stats.norm.ppf(1-pval);
	# Now, let's start statistics stuff
	tnum = np.mean(cond2-cond1,axis=0);
	tdenom = np.std(np.array([cond1,cond2]).mean(axis=0),axis=0,ddof=1)**2/np.sqrt(cond1.shape[0]);  
	real_t = tnum/tdenom;

	# % initialize null hypothesis matrices
	permuted_tvals  = np.zeros((n_permutes,real_t.shape[0], real_t.shape[1]))
	max_cluster_sizes = np.zeros((1,n_permutes))
	# % generate pixel-specific null hypothesis parameter distributions
	for permi in range(n_permutes):
	    # % permuted condition mapping
		fake_condition_mapping = np.sign(np.random.normal(size=cond1.shape[0]))
		# % compute t-map of null hypothesis
		tnumfake = np.array([[(cond2[:,i,j]-cond1[:,i,j]) * fake_condition_mapping for i in range(cond1.shape[1])] for j in range(cond1.shape[2])]).transpose(2,1,0).mean(axis=0)
		# tnumfake = squeeze(mean(bsxfun(@times,cond2-cond1,fake_condition_mapping),1))
		permuted_tvals[permi,:,:] = tnumfake/tdenom;

	# % compute mean and standard deviation (used for thresholding)
	mean_h0 = np.mean(permuted_tvals,axis=0)
	var_h0  = np.std(permuted_tvals,axis=0, ddof=1);

	# % loop through permutations to cluster sizes under the null hypothesis
	for permi in range(n_permutes):
		threshimg = permuted_tvals[permi,:,:];
		threshimg = (threshimg-mean_h0)/var_h0; # Normalize (transform t- to Z-value)
		threshimg[abs(threshimg)<zval] = 0 # % threshold image at p-value

		# threshimg[abs(threshimg)<zval] = 0;
		# % find clusters (need image processing toolbox for this!)
		labeled, islands = measurements.label(threshimg)
		if islands>0:
			area = measurements.sum((threshimg!=0), labeled, index=np.arange(labeled.max() + 1))
			max_cluster_sizes[0,permi] = np.max(area)

	# % find clusters (need image processing toolbox for this!)
	cluster_thresh = np.percentile(max_cluster_sizes,100-(100*pval))
	print(cluster_thresh)
	# % now threshold real data...
	real_t_thresh = (real_t-mean_h0)/var_h0; # % first Z-score

	real_t_thresh[abs(real_t_thresh)<zval] = 0 	# % next threshold image at p-value

	# % now cluster-based testing
	real_island, realnumclust = measurements.label(real_t_thresh)
	for i in range(realnumclust):
	    # %if real clusters are too small, remove them by setting to zero!  
	    if np.sum(real_island==i+1)<cluster_thresh:
	        real_t_thresh[real_island==i+1]=0
	return np.array(real_t_thresh,dtype=bool)

def topo_cluster_ttest(cond1, cond2, pval, n_permutes, interdist):

	dif = cond2-cond1
	permm=np.zeros((n_permutes,cond1.shape[1]))
	for i in range(n_permutes):
		permm[i,:] = np.mean(dif*np.sign(np.random.normal(size=cond1.shape)),axis=0)

	# cluster size
	maxclustsize = np.zeros((n_permutes,1))
	for i in range(n_permutes):
	    tempz = (permm[i,:]-np.mean(permm,axis=0))/np.std(permm,axis=0, ddof=1)
	    ph = 1-sp.stats.norm.cdf(tempz)
	    tempz[ph>.1]=0
	    whereSig = np.nonzero(tempz)[0]


	    if not whereSig.size:
	    	maxclustsize[i]=0
	    else:
	        clusts = np.zeros(whereSig.shape);
	        for ci in range(len(whereSig)):
	            clusts[ci] = sum(tempz[interdist[whereSig[ci],:]<5]);
	        
	        maxclustsize[i] = max(clusts);
	    
	# now real data
	z=(np.mean(dif,axis=0)-np.mean(permm,axis=0))/np.std(permm,axis=0,ddof=1)

	ph=1-sp.stats.norm.cdf(z)
	z[ph>pval]=0;
	whereSig = np.nonzero(z)[0]
	cluster_thresh = np.percentile(maxclustsize,100-(100*pval))

	for ci in range(len(whereSig)):
	    if sum(z[interdist[whereSig[ci],:]<10]!=0) < cluster_thresh:
	        z[whereSig[ci]] = 0

	return z
