#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:15:53 2019

Analyzing degree distirbutions and other graph characteristics for
ConsensusDB

@author: Yusuf
"""

import snap
import numpy as np
from matplotlib import pyplot as plt

def get_hist_ranges(hist):
    x = []
    for idx in range(len(hist[1])-1):
        x.append( np.mean( [hist[1][idx],hist[1][idx+1]]))
    y = hist[0]
    return (x,y)

def plot_hist(all_outdeg):
    hist = np.histogram(all_outdeg, range = (min(all_outdeg), max(all_outdeg)),\
                    bins =500)
    
    fig = plt.figure()   
    ax = fig.add_subplot(1,1,1)  
    ax.set_xscale('log')
    ax.set_yscale('log')
    x,y = get_hist_ranges(hist)
    y[y==0] = 1
    
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([max(min(y),1), max(y)])
    ax.scatter(x,y)
    ##coeffs = np.polyfit(x,y,deg=1)
    coeffs = np.polyfit(np.log10(x),np.log10(y),deg=1)
    polynomial = np.poly1d(coeffs)
    log10_y_fit = polynomial(np.log10(x)) 
    
    ax.plot([0,10**(coeffs[1]/coeffs[0])],[10**coeffs[1],0])
    ax.plot(x, 10**log10_y_fit, '*-') 
    return(x,y,ax)
    
# Plotting degree distributions
## Read in snap.py version of graph
s = snap.LoadEdgeList(snap.PUNGraph,'ConsensusPathDB_human_PPI_HiConfidence_snap.csv',
                      0,1,',')    
    
all_outdeg = []
for n in s.Nodes():
        all_outdeg.append(n.GetOutDeg())
   
all_outdeg = np.array(all_outdeg)
all_outdeg = all_outdeg[all_outdeg!= 0]

(x,y,ax) = plot_hist(all_outdeg)