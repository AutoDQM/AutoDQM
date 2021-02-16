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

    data_hist_Entries = np.sum(data_hist)


    # Check that the hist is a histogram
    if "1" not in str(type(histpair.data_hist)):
        return None

    # Normalize data_hist
    if data_hist_Entries > 0:
        data_hist = data_hist * 1.0 / data_hist_Entries
#### TEMP FIX NEED TO USE INTEGREAL AND ACCOUNT FOR BIN SIZE^^^^^^####

    # Reject empty histograms
    is_good = data_hist_Entries != 0 and data_hist_Entries >= min_entries

    np_data = np.copy(data_hist)

    # Get 'good' (non-zero) bins
    np_data = np_data[pca_pickle.__dict__['_hist_cleaner'].good_bins]

    n_components = len(pca_pickle.__dict__['_DQMPCA__sse_ncomps'])

    sse, reco_data_raw = PCATest(np_data, pca_pickle, n_components)

    reco_data = np.zeros(len(data_hist))
    for i in range(0, len(pca_pickle.__dict__['_hist_cleaner'].good_bins)):
        reco_data[pca_pickle.__dict__['_hist_cleaner'].good_bins[i]] = reco_data_raw[i]
    
    # Get SSE cut
    sse_cut = np.average(pca_pickle.__dict__['sse_cuts'][n_components])

    is_outlier = is_good and bool(sse > sse_cut)

    bins = [(histpair.data_hist.axes[0].edges(flow=False)[x] + histpair.data_hist.axes[0].edges(flow=False)[x+1])/2 for x in range(0,len(histpair.data_hist.axes[0]))]
    if bins[0] < -999:
        bins[0]=2*bins[1]-bins[2]

    xAxisTitle = histpair.data_hist.axes[0]._bases[0]._members["fTitle"]
    if(len(histpair.data_hist.axes) > 1):
        yAxisTitle = histpair.data_hist.axes[1]._bases[0]._members["fTitle"]
    else:
        yAxisTitle = ""
    plotTitle = histpair.data_name + " PCA Test  |  data:" + str(histpair.data_run) + " & ref:" + str(histpair.ref_run)

    xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")

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

def draw_same(data_hist, reco_data, reco_bins, data_run, is_outlier):
    # Set up canvas
    c = ROOT.TCanvas('c', 'c')
    reco_hist = data_hist.Clone("reco_hist")
    reco_hist.Reset()
    
    # Fill Reco hist
    for i in range(0, len(reco_bins)):
        reco_hist.SetBinContent(int(reco_bins[i])+1, float(reco_data[i]))

    ROOT.gStyle.SetOptStat(0)
    data_hist.SetStats(False)
    reco_hist.SetStats(False)

    # Set hist style
    data_hist.SetLineColor(ROOT.kBlue)
    data_hist.SetFillColor(38)
    data_hist.SetLineWidth(2)
    if is_outlier:
        reco_hist.SetLineColor(ROOT.kRed)
        reco_hist.SetFillColorAlpha(ROOT.kRed, 0.25) 
    else:
        reco_hist.SetLineColor(ROOT.kGreen)
        reco_hist.SetFillColorAlpha(ROOT.kGreen, 0.25) 
    reco_hist.SetLineStyle(7)
    reco_hist.SetLineWidth(2)

    # Name histograms
    data_hist.SetName("Data")
    reco_hist.SetName("Reconstructed")

    # Plot hist
    data_hist.Draw("hist")
    reco_hist.Draw("hist same")
    c.Update()

    # Text box
    data_text = ROOT.TLatex(.72, .91, "#scale[0.6]{Data: " + data_run + "}")
    data_text.SetNDC(ROOT.kTRUE)
    data_text.Draw("same")
    c.Update()

    # Draw legend
    legend = ROOT.TLegend(0.9,0.9,0.75,0.75)
    legend.AddEntry(data_hist, "Data")
    legend.AddEntry(reco_hist, "PCA Reco")
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.Draw("same")
    c.Update()

    artifacts = [data_hist, reco_hist, data_text, legend]
    return c, artifacts

