#/!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pickle as pickle
import numpy as np
#from sklearn.decomposition import PCA
import uproot
import sys
sys.path.append('/var/www/cgi-bin/')
from modules.histCollection import HistCollection
from modules.histCollection import HistCleaner
from modules.dqmpca import DQMPCA
from autodqm.plugin_results import PluginResults
import plotly.graph_objects as go

def comparators():
    return {
        'pca': pca        
    }

def pca(histpair,
        sse_percentile=5, exp_var=0.95, norm_type='all', min_entries=100000,
        **kwargs):

    data_name = histpair.data_name
    data_hist = histpair.data_hist.values(flow=False)
    jar_dir = histpair.config["jar_dir"]

    data_year = int(histpair.data_series[-4:])
    possible_pickles = glob.glob("/var/www/cgi-bin/models/pca/{0}/{1}/*_{2}.pkl".format(jar_dir,data_year, data_name))
    

    if len(possible_pickles) != 1:
        return None
    pca_pickle = pickle.load(open(possible_pickles[0], "rb"))

    #Decalre data_hist_Entries
    data_hist_Entries = np.sum(data_hist)


    # Check that the hist is a histogram
    if "1" not in str(type(histpair.data_hist)):
        return None

    # Normalize data_hist
    if data_hist_Entries > 0:
        data_hist = data_hist * 1.0 / data_hist_Entries

    # Reject empty histograms
    is_good = data_hist_Entries != 0 and data_hist_Entries >= min_entries

    np_data = np.copy(data_hist)

    # Get 'good' (non-zero) bins
    np_data = np_data[pca_pickle.__dict__['_hist_cleaner'].good_bins]

    n_components = len(pca_pickle.__dict__['_DQMPCA__sse_ncomps'])

    sse, reco_data_raw = PCATest(np_data, pca_pickle, n_components)

    #Rebin reco_data to account for empty bins being removed in reco_data_raw
    reco_data = np.zeros(len(data_hist))
    for i in range(0, len(pca_pickle.__dict__['_hist_cleaner'].good_bins)):
        reco_data[pca_pickle.__dict__['_hist_cleaner'].good_bins[i]] = reco_data_raw[i]
    
    # Get SSE cut
    sse_cut = np.average(pca_pickle.__dict__['sse_cuts'][n_components])

    is_outlier = is_good and bool(sse > sse_cut)

    #Get bin centers from edges()
    bins = [(histpair.data_hist.axes[0].edges()[x] + histpair.data_hist.axes[0].edges()[x+1])/2 for x in range(0,len(histpair.data_hist.axes[0]))]
    if bins[0] < -999:
        bins[0]=2*bins[1]-bins[2]

    #Get Titles for histogram, x-axis, y-axis
    xAxisTitle = histpair.data_hist.axes[0]._bases[0]._members["fTitle"]
    if(len(histpair.data_hist.axes) > 1):
        yAxisTitle = histpair.data_hist.axes[1]._bases[0]._members["fTitle"]
    else:
        yAxisTitle = ""
    plotTitle = histpair.data_name + " PCA Test  |  data:" + str(histpair.data_run) + " & ref:" + str(histpair.ref_run)

    #Plotly doesn't support #circ, #theta, #phi but does support unicode
    xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")

    #Plot histograms with settings similar to PyRoot plots
    c = go.Figure()
    c.add_trace(go.Bar(name="data:"+str(histpair.data_run), x=bins, y=data_hist, marker_color='rgb(125,153,207)', opacity=.7))
    c.add_trace(go.Bar(name="PCA Reco", x=bins, y=reco_data, marker_color='rgb(192, 255, 195)', opacity=.7))
    c['layout'].update(bargap=0)
    c['layout'].update(barmode='overlay')
    c['layout'].update(plot_bgcolor='white')
    c.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False)
    c.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False)
    c.update_layout(
        title=plotTitle , title_x=0.5,
        xaxis_title= xAxisTitle,
        yaxis_title= yAxisTitle,
        font=dict(
            family="Times New Roman",
            size=9,
            color="black"
        )
    )
    artifacts = {
        'data' :data_hist,
        'reco' :reco_data
    }

    info = {
        'Data_Entries': str(data_hist_Entries),
        'Sum of Squared Errors': round(sse, 3),
        'PCA Components': n_components
    }

    return PluginResults(
            c,
            show=bool(is_outlier),
            info=info,
            artifacts=artifacts)

def PCATest(np_data, pca_obj, n_components):
    # Transform data in terms of principle component vectors
    transf = pca_obj.transform(np_data.reshape(1,-1))
    # Zero out components beyond n_components cap
    transf[0,n_components:] *= 0
    # Reconstruct data using N components
    reco_data = pca_obj.inverse_transform(transf)
    reco_data = reco_data.flatten()
    # Get sum of squared errors
    sse = np.sqrt(np.sum((reco_data - np_data)**2))

    return sse, reco_data
