#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import uproot
import numpy
from scipy.stats import chi2
from autodqm.plugin_results import PluginResults
import scipy
import plotly.graph_objects as go

def comparators():
    return {
        'pull_values': pullvals
    }


def pullvals(histpair,
             pull_cap=25, chi2_cut=500, pull_cut=20, min_entries=10000, norm_type='all',
             **kwargs):

    """Can handle poisson driven TH2s or generic TProfile2Ds"""
    data_hist = histpair.data_hist
    ref_hist = histpair.ref_hist

    # Check that the hists are histograms
    # Check that the hists are 2 dimensional
    if not ( (       "TH2" in str(type(data_hist)) and       "TH2" in str(type(ref_hist)) ) or
             ("TProfile2D" in str(type(data_hist)) and "TProfile2" in str(type(ref_hist)) ) ):
        return None
    # Extract values from TH2F or TProfile2D Format
    data_hist_norm = None
    ref_hist_norm = None
    #raise Exception(projectionXY(data_hist))
    #raise Exception(data_hist.xnumbins, data_hist._fBinEntries)
    data_hist_norm = numpy.copy(data_hist.values())
    ref_hist_norm = numpy.copy(ref_hist.values())

    # Clone data_hist array to create pull_hist array to be filled later
    pull_hist = numpy.copy(data_hist_norm)

    # Declare data_hist_Entries and ref_hist_Entries
    data_hist_Entries = numpy.sum(data_hist_norm);
    ref_hist_Entries = numpy.sum(ref_hist_norm);
    # Reject empty histograms
    is_good = data_hist_Entries != 0 and data_hist_Entries >= min_entries

    # Normalize data_hist (Note if col is selected numpy just transposes normalizes by rows then transposes again)
    if norm_type == "row":
        data_hist_norm = normalize_rows(data_hist_norm, ref_hist_norm)
    elif norm_type == "col":
        data_hist_norm = normalize_rows(numpy.transpose(data_hist_norm), numpy.transpose(ref_hist_norm))
        data_hist_norm = numpy.transpose(data_hist_norm)
    else:
        if data_hist_Entries > 0:
            data_hist_norm = data_hist_norm * ref_hist_Entries / data_hist_Entries

    #Calculate asymmetric error bars
    data_hist_errs = numpy.nan_to_num(abs(numpy.array(scipy.stats.chi2.interval(0.6827, 2 * data_hist_norm)) / 2 - 1 - data_hist_norm))
    ref_hist_errs = numpy.nan_to_num(abs(numpy.array(scipy.stats.chi2.interval(0.6827, 2 * ref_hist_norm)) / 2 - 1 - ref_hist_norm))

    # pull values
    if (data_hist_Entries + ref_hist_Entries).sum() == 0:
        max_pull = 0
        chi2 = 0
        new_pull = numpy.zeros_like(data_hist_norm)
    else:
        nBins = data_hist_norm.size
        data_hist_err, ref_hist_err = data_hist_errs[0, :, :], ref_hist_errs[1, :, :]
        mask = data_hist_norm < ref_hist_norm
        data_hist_err[mask] = data_hist_errs[1, :, :][mask]
        ref_hist_err[mask] = ref_hist_errs[0, :, :][mask]
        new_pull = pull(data_hist_norm, data_hist_err, ref_hist_norm, ref_hist_err)
        max_pull = numpy.abs(new_pull).max()

        ## compute chi2
        chi2 = (new_pull*new_pull).sum()/nBins


    # Clamp the displayed value
    fill_val = numpy.clip(new_pull, -pull_cap, pull_cap)

    # If the input bins were explicitly empty, make this bin white by
    # setting it out of range
    mask = data_hist_norm + ref_hist_norm == 0
    fill_val[mask] = -999

    # Fill Pull Histogram
    pull_hist = fill_val

    is_outlier = is_good and (chi2 > chi2_cut or abs(max_pull) > pull_cut)

    # Setting empty bins to be blank
    pull_hist = numpy.where(pull_hist < -2*pull_cap, None, pull_hist)
    colors = ['rgb(26, 42, 198)', 'rgb(118, 167, 231)', 'rgb(215, 226, 194)', 'rgb(212, 190, 109)', 'rgb(188, 76, 38)']

    #Getting Plot labels for x-axis and y-axis as well as type (linear or categorical)
    xLabels = None
    yLabels = None
    c = None
    x_axis_type = 'linear'
    y_axis_type = 'linear';
    if data_hist.axes[0].labels():
       xLabels = [str(x) for x in data_hist.axes[0].labels()]
       x_axis_type = 'category'
    else:
       xLabels = [str(data_hist.axes[0]._members["fXmin"] + x * (data_hist.axes[0]._members["fXmax"]-data_hist.axes[0]._members["fXmin"])/data_hist.axes[0]._members["fNbins"]) for x in range(0,data_hist.axes[0]._members["fNbins"]+1)]

    if data_hist.axes[1].labels():
       yLabels = [str(x) for x in data_hist.axes[1].labels()]
       y_axis_type = 'category'
    else:
       yLabels = [str(data_hist.axes[1]._members["fXmin"] + x * (data_hist.axes[1]._members["fXmax"]-data_hist.axes[1]._members["fXmin"])/data_hist.axes[1]._members["fNbins"]) for x in range(0,data_hist.axes[1]._members["fNbins"]+1)]

    if("xlabels" in histpair.config.keys()):
        xLabels=histpair.config["xlabels"]
        x_axis_type = 'category'
    if("ylabels" in histpair.config.keys()):
        yLabels=histpair.config["ylabels"]
        y_axis_type = 'category'

    pull_hist = numpy.transpose(pull_hist)

    #Getting Plot Titles for histogram, x-axis and y-axis
    xAxisTitle = data_hist.axes[0]._bases[0]._members["fTitle"]
    yAxisTitle = data_hist.axes[1]._bases[0]._members["fTitle"]
    plotTitle = histpair.data_name + " Pull Values  |  data:" + str(histpair.data_run) + " & ref:" + str(histpair.ref_run)

    #Plotly doesn't support #circ, #theta, #phi but does support unicode
    xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")

    #Plot pull-values using 2d heatmap will settings to look similar to old Pyroot version
    c  = go.Figure(data=go.Heatmap(z=pull_hist, zmin=-pull_cap, zmax=pull_cap, colorscale=colors, x=xLabels, y=yLabels))
    c['layout'].update(plot_bgcolor='white')
    c.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False, type=x_axis_type)
    c.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False, type=y_axis_type)
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


    ## write csv files for analysis
    with open("csv/pullvals.csv", "a") as myfile:
        myfile.write(f'{histpair.data_name},{max_pull},{chi2},{histpair.ref_run},{histpair.data_run}\n')


    info = {
        'Chi_Squared': f"{chi2:.2f}", # 2 decimal points
        'Max_Pull_Val': f"{max_pull:.2f}",
        'Data_Entries': str(data_hist_Entries),
        'Ref_Entries': str(ref_hist_Entries),
    }


    artifacts = [pull_hist, str(data_hist_Entries), str(ref_hist_Entries)]

    return PluginResults(
        c,
        show=bool(is_outlier),
        info=info,
        artifacts=artifacts)


def pull(bin1, binerr1, bin2, binerr2):
    ''' Calculate the pull value between two bins.
        pull = (data - expected)/sqrt(sum of errors in quadrature))
        data = |bin1 - bin2|, expected = 0

        only divide where bin1+bin2 != 0, output zero where that happens
    '''
    return numpy.divide( (bin1 - bin2) , ((binerr1**2 + binerr2**2)**0.5), out=numpy.zeros_like(bin1), where=(binerr1+binerr2)!=0)

def normalize_rows(data_hist_norm, ref_hist_norm):
    ref_sum = ref_hist_norm.sum(axis=0)
    data_sum = data_hist_norm.sum(axis=0)
    sf = numpy.divide(ref_sum, data_sum, where=data_sum!=0, out=numpy.ones_like(data_sum))

    return data_hist_norm*sf
