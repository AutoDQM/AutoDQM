#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import glob
import cPickle as pickle
import numpy as np
from sklearn.decomposition import PCA
import ROOT

from autodqm.plugin_results import PluginResults

def comparators():
    return {
        'pca': pca        
    }

def pca(histpair,
        sse_percentile=5, exp_var=0.95, norm_type='all', min_entries=100000,
        **kwargs):

    data_name = histpair.data_name
    data_hist = histpair.data_hist.Clone()
    jar_dir = histpair.config["jar_dir"]

    # Check for unique Pickle file
    possible_pickles = glob.glob("/var/www/cgi-bin/pickle_jar/{0}/*_{1}.pkl".format(jar_dir, data_name))
    if len(possible_pickles) != 1:
        return None

    # Load Pickle file
    pca_dict = pickle.load(open(possible_pickles[0], "rb"))

    # Check that the hist is a histogram
    if not data_hist.InheritsFrom('TH1'):
        return None

    # Normalize data_hist
    if data_hist.GetEntries() > 0:
        data_hist.Scale(1.0 / data_hist.Integral())

    # Reject empty histograms
    is_good = data_hist.GetEntries() != 0 and data_hist.GetEntries() >= min_entries

    np_data = get_np_data(data_hist)
    # Get 'good' (non-zero) bins
    np_data = np_data[pca_dict["good_bins"]]

    n_components = get_components(exp_var, pca_dict["pca"].explained_variance_ratio_)
    sse, reco_data = PCATest(np_data, pca_dict["pca"], n_components)

    # Get SSE cut
    sse_cut = pca_dict["sses_{}comp".format(n_components)]["{}pct".format(sse_percentile)]

    is_outlier = is_good and bool(sse > sse_cut)

    c, artifacts = draw_same(data_hist, reco_data, pca_dict["good_bins"], histpair.data_run, is_outlier)

    info = {
        'Data_Entries': data_hist.GetEntries(),
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

def get_components(exp_var, exp_var_ratios_, n_cap=3):
    """Get PCA components that explain variance to some percentage (exp_var)"""

    sum_var = 0
    n_components = 0
    for var_ratio in exp_var_ratios_:
        n_components += 1
        sum_var += var_ratio
        if sum_var >= exp_var or n_components >= n_cap:
            return n_components 

    return n_components

def get_np_data(data_hist):
    """Turn TH1F bin content into a numpy array"""
    np_data = []
    for x in range(1, data_hist.GetNbinsX() +1):
        np_data.append(data_hist.GetBinContent(x))

    return np.array(np_data)

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
    legend.AddEntry(reco_hist, "Reco")
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.Draw("same")
    c.Update()

    artifacts = [data_hist, reco_hist, data_text, legend]
    return c, artifacts
