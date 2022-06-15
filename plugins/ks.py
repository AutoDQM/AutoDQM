#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uproot
import numpy
import scipy.stats
from autodqm.plugin_results import PluginResults
import plotly.graph_objects as go

def comparators():
    return {
        "ks_test": ks
    }


def ks(histpair, ks_cut=0.09, min_entries=10000, **kwargs):

    data_name = histpair.data_name
    ref_name = histpair.ref_name

    data_hist = histpair.data_hist
    ref_hist = histpair.ref_hist

    # Check that the hists are 1 dimensional
    if "1" not in str(type(data_hist)) or "1" not in str(type(ref_hist)):
        return None

    # Normalize data_hist by copying histogram and then normalizing (Note declaration of data_hist_Entries & ref_hist_Entries)
    data_hist_norm = numpy.copy(data_hist.values())
    ref_hist_norm = numpy.copy(ref_hist.values())
    data_hist_Entries = numpy.sum(data_hist_norm)
    ref_hist_Entries = numpy.sum(ref_hist_norm)
    if data_hist_Entries > 0:
        data_hist_norm = data_hist_norm * (ref_hist_Entries / data_hist_Entries)

    # Reject empty histograms
    is_good = data_hist_Entries != 0 and data_hist_Entries >= min_entries

    ks = scipy.stats.kstest(ref_hist_norm, data_hist_norm)[0]

    is_outlier = is_good and ks > ks_cut


    #Get bin centers from edges() stored by uproot
    bins = data_hist.axes[0].edges();
    if bins[0] < -999:
        bins[0]=2*bins[1]-bins[2]

    #Get Titles for histogram, X-axis, Y-axis (Note data_hist.axes will have length > 1 if y-axis title is declared even with 1d plot)
    xAxisTitle = data_hist.axes[0]._bases[0]._members["fTitle"]
    if(len(data_hist.axes) > 1):
        yAxisTitle = data_hist.axes[1]._bases[0]._members["fTitle"]
    else:
        yAxisTitle = ""
    plotTitle = histpair.data_name + " KS Test  |  data:" + str(histpair.data_run) + " & ref:" + str(histpair.ref_run)

    #Plotly doesn't support #circ, #theta, #phi but it does support unicode
    xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")


    #Plot histogram with previously declared axes and settings to look similar to PyRoot
    c = go.Figure()
    c.add_trace(go.Bar(name="data:"+str(histpair.data_run), x=bins, y=data_hist_norm, marker_color='white', marker=dict(line=dict(width=1,color='red'))))
    c.add_trace(go.Bar(name="ref:"+str(histpair.ref_run), x=bins, y=ref_hist_norm, marker_color='rgb(204, 188, 172)', opacity=.9))
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
    ref_text = "ref:"+str(histpair.ref_run)
    data_text = "data:"+str(histpair.data_run)
    artifacts = [data_hist_norm, ref_hist_norm, data_text, ref_text]

    info = {
        'Data_Entries': str(data_hist_Entries),
        'Ref_Entries': str(ref_hist_Entries),
        'KS_Val': ks
    }

    return PluginResults(
        c,
        show=is_outlier,
        info=info,
        artifacts=artifacts)
