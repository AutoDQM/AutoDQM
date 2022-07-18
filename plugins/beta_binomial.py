#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import uproot
import numpy
from autodqm.plugin_results import PluginResults
import scipy.stats as stats
from scipy.special import gammaln
import plotly.graph_objects as go
from plugins.pullvals import normalize_rows

def comparators():
    return { 
        'beta_binomial' : beta_binomial
    }

def beta_binomial(histpair, pull_cap=15, chi2_cut=100, pull_cut=10, min_entries=1, tol=0.01, norm_type='all', **kwargs):
    """beta_binomial works on both 1D and 2D"""
    data_hist = histpair.data_hist
    ref_hist = histpair.ref_hist

    data_hist_raw = numpy.round(numpy.copy(data_hist.values()))
    ref_hist_raw = numpy.round(numpy.copy(ref_hist.values()))

    ## num entries
    data_hist_Entries = numpy.sum(data_hist_raw)
    ref_hist_Entries = numpy.sum(ref_hist_raw)

    ## does not run beta_binomial if data or ref is 0
    if data_hist_Entries <= 0 or ref_hist_Entries <= 0:
        return None

    # Normalize data_hist (Note if col is selected numpy just transposes normalizes by rows then transposes again)
    data_hist_norm = numpy.copy(data_hist.values())
    data_hist_norm = data_hist_norm * ref_hist_Entries / data_hist_Entries

    ## only filled bins used for chi2
    nBinsUsed = numpy.count_nonzero(numpy.add(ref_hist_raw, data_hist_raw))
    nBins = data_hist.values().size

    ## calculte pull and chi2
    pull_hist = pull(data_hist_raw, ref_hist_raw, tol)
    pull_hist = pull_hist*numpy.sign(data_hist_norm-ref_hist_raw)
    chi2 = numpy.square(pull_hist).sum()/nBinsUsed
    max_pull = maxPullNorm(numpy.amax(pull_hist), nBinsUsed)
    min_pull = maxPullNorm(numpy.amin(pull_hist), nBinsUsed)
    if abs(min_pull) > max_pull:
        max_pull = min_pull

    ## define if plot anomalous
    is_outlier = data_hist_Entries >= min_entries and (chi2 > chi2_cut or abs(max_pull) > pull_cut)

    ## plotting
    # Setting empty bins to be blank
    pull_hist = numpy.where(numpy.add(ref_hist_raw, data_hist_raw) == 0, None, pull_hist)

    ##--------- 1D Plotting --------------
    # Check that the hists are 1 dimensional
    if ("TH1" in str(type(data_hist))) and ("TH1" in str(type(ref_hist))):
        #Get bin centers from edges() stored by uproot
        bins = data_hist.axes[0].edges()
        if bins[0] < -999:
            bins[0]=2*bins[1]-bins[2]

        #Truncate empty space on high end of histograms with large axes
        if data_hist_Entries > 0 and ref_hist_Entries > 0 and len(bins) > 15:
            last_bin = max( [15, max(numpy.nonzero(data_hist_raw)[0]), max(numpy.nonzero(ref_hist_raw)[0])] )
            if last_bin+2 < len(bins):
                bins = bins[:(last_bin+2)]

        #Get Titles for histogram, X-axis, Y-axis (Note data_hist.axes will have length > 1 if y-axis title is declared even with 1d plot)
        xAxisTitle = data_hist.axes[0]._bases[0]._members["fTitle"]
        if(len(data_hist.axes) > 1):
            yAxisTitle = data_hist.axes[1]._bases[0]._members["fTitle"]
        else:
            yAxisTitle = ""
        plotTitle = histpair.data_name + " beta-binomial  |  data:" + str(histpair.data_run) + " & ref:" + str(histpair.ref_run)
    
        #Plotly doesn't support #circ, #theta, #phi but it does support unicode
        xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
        yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
        plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    
    
        #Plot histogram with previously declared axes and settings to look similar to PyRoot
        c = go.Figure()
        c.add_trace(go.Bar(name="data:"+str(histpair.data_run), x=bins, y=data_hist_norm, marker_color='red'))
        c.add_trace(go.Bar(name="ref:"+str(histpair.ref_run), x=bins, y=ref_hist_raw, marker_color='blue', opacity=.5))
        c.update_traces(marker_line_width=0)
        c.update_layout(bargap=0, bargroupgap=0, barmode='overlay', plot_bgcolor='white')
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

    ## --------- end 1D plotting ---------


    ##---------- 2d Plotting --------------
    # Check that the hists are 2 dimensional
    if ( (       "TH2" in str(type(data_hist)) and       "TH2" in str(type(ref_hist)) ) or
         ("TProfile2D" in str(type(data_hist)) and "TProfile2" in str(type(ref_hist)) ) ):
        
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
        plotTitle = histpair.data_name + " beta-binomial  |  data:" + str(histpair.data_run) + " & ref:" + str(histpair.ref_run)
    
        #Plotly doesn't support #circ, #theta, #phi but does support unicode
        xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
        yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
        plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    
        #Plot pull-values using 2d heatmap will settings to look similar to old Pyroot version
        c = go.Figure(data=go.Heatmap(z=pull_hist, zmin=-pull_cap, zmax=pull_cap, colorscale=colors, x=xLabels, y=yLabels))
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
    ##----- end 2D plotting --------


    info = {
        'Chi_Squared': float(round(chi2, 2)),
        'Max_Pull_Val': float(round(max_pull,2)),
        'Data_Entries': str(int(data_hist_Entries)),
        'Ref_Entries': str(int(ref_hist_Entries)),
    }

    artifacts = [pull_hist, str(data_hist_Entries), str(ref_hist_Entries)]

    return PluginResults(
        c,
        show=bool(is_outlier),
        info=info,
        artifacts=artifacts)


def pull(D_raw, R_raw, tol=0.01):
    prob = numpy.zeros_like(D_raw)
    prob = ProbRel(D_raw, R_raw, 'BetaB', tol)
    pull = Sigmas(prob)

    return pull

def maxPullNorm(maxPull, nBinsUsed, cutoff=pow(10,-15)):
    sign = numpy.sign(maxPull)
    ## sf (survival function) better than 1-cdf for large pulls (no precision error)
    probGood = stats.chi2.sf(numpy.power(min(abs(maxPull), 37), 2), 1)

    ## Use binomial approximation for low probs (accurate within 1%)
    if nBinsUsed * probGood < 0.01:
        probGoodNorm = nBinsUsed * probGood
    else:
        probGoodNorm = 1 - numpy.power(1 - probGood, nBinsUsed)

    pullNorm = Sigmas(probGoodNorm) * sign

    return pullNorm


## Mean expectation for number of expected data events
def Mean(Data, Ref, func):
    nRef = Ref.sum()
    nData = Data.sum()
    if func == 'Gaus1' or func == 'Gaus2':
        return 1.0*nData*Ref/nRef
    ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#Moments_and_properties
    ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
    if func == 'BetaB' or func == 'Gamma':
        return 1.0*nData*(Ref+1)/(nRef+2)

    print('\nInside Mean, no valid func = %s. Quitting.\n' % func)
    sys.exit()
## Standard deviation of gaussian and beta-binomial functions
def StdDev(Data, Ref, func):
    nData = Data.sum()
    nRef = Ref.sum()
    mask = Ref > 0.5*nRef
    if func == 'Gaus1':
        ## whole array is calculated using the (Ref <= 0.5*nRef) formula, then the ones where the
        ## conditions are actually failed is replaced using mask with the (Ref > 0.5*nRef) formula
        output = 1.0*nData*numpy.sqrt(numpy.clip(Ref, a_min=1, a_max=None))/nRef
        output[mask] = (1.0*nData*numpy.sqrt(numpy.clip(nRef-Ref, a_min=1, a_max=None)))[mask]/nRef
    elif func == 'Gaus2':
        ## instead of calculating max(Ref, 1), set the whole array to have a lower limit of 1
        clipped = numpy.clip(Ref, a_min=1, a_max=None)
        output = 1.0*nData*numpy.sqrt( clipped/numpy.square(nRef) + Mean(nData, Ref, nRef, func)/numpy.square(nData) )
        clipped = numpy.clip(nRef-Ref, a_min=1, a_max=None)
        output[mask] = (1.0*nData*numpy.sqrt( clipped/numpy.square(nRef) + (nData - Mean(nData, Ref, nRef, func))/numpy.square(nData) ))
    elif (func == 'BetaB') or (func == 'Gamma'):
        output = 1.0*numpy.sqrt( nData*(Ref+1)*(nRef-Ref+1)*(nRef+2+nData) / (numpy.power(nRef+2, 2)*(nRef+3)) )
        
    else:
        print('\nInside StdDev, no valid func = %s. Quitting.\n' % func)
        sys.exit()

    return output


## Number of standard devations from the mean in any function
def numStdDev(Data, Ref, func):
    nData = Data.sum()
    nRef = Ref.sum()
    return (Data - Mean(Data, Ref, func)) / StdDev(Data, Ref, func)


## Predicted probability of observing Data / nData given a reference of Ref / nRef
def Prob(Data, nData, Ref, nRef, func, tol=0.01):
    scaleTol = numpy.power(1 + numpy.power(Ref * tol**2, 2), -0.5)
    nRef_tol = numpy.round(scaleTol * nRef)
    Ref_tol = numpy.round(Ref * scaleTol)

    if func == 'Gaus1' or func == 'Gaus2':
        return stats.norm.pdf( numStdDev(Data, Ref_tol, func) )
    if func == 'BetaB':
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        #return stats.betabinom.pmf(Data, nData, Ref+1, nRef-Ref+1)
        return stats.betabinom.pmf(Data, nData, Ref_tol + 1, nRef_tol - Ref_tol + 1)
    ## Expression for beta-binomial using definition in terms of gamma functions
    ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#As_a_compound_distribution
    if func == 'Gamma':
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        n_  = numpy.zeros_like(Data) + nData
        k_  = Data
        a_  = Ref_tol + 1
        b_  = nRef_tol - Ref_tol + 1
        ab_ = nRef_tol + 2
        logProb  = gammaln(n_+1) + gammaln(k_+a_) + gammaln(n_-k_+b_) + gammaln(ab_)
        logProb -= ( gammaln(k_+1) + gammaln(n_-k_+1) + gammaln(n_+ab_) + gammaln(a_) + gammaln(b_) )
        return numpy.exp(logProb)

    print('\nInside Prob, no valid func = %s. Quitting.\n' % func)
    sys.exit()


## Predicted probability relative to the maximum probability (i.e. at the mean)
def ProbRel(Data, Ref, func, tol=0.01):
    nData = Data.sum()
    nRef = Ref.sum()
    ## Find the most likely expected data value
    exp_up = numpy.clip(numpy.ceil(Mean(Data, Ref, 'Gaus1')), a_min=None, a_max=nData) # make sure nothing goes above nData
    exp_down = numpy.clip(numpy.floor(Mean(Data, Ref, 'Gaus1')), a_min=0, a_max=None) # make sure nothing goes below zero

    ## Find the maximum likelihood
    maxProb_up  = Prob(exp_up, nData, Ref, nRef,func, tol)
    maxProb_down = Prob(exp_down, nData, Ref, nRef,func, tol)
    maxProb = numpy.maximum(maxProb_up, maxProb_down)
    thisProb = Prob(Data, nData, Ref, nRef, func, tol)

    ## Sanity check to not have relative likelihood > 1
    ratio = numpy.divide(thisProb, maxProb, out=numpy.zeros_like(thisProb), where=maxProb!=0)
    cond = thisProb > maxProb
    ratio[cond] = 1

    return ratio


## Convert relative probability to number of standard deviations in normal distribution
def Sigmas(probRel):
    ## chi2.isf function fails for probRel < 10^-323, so cap at 10^-300 (37 sigma)
    probRel = numpy.maximum(probRel, pow(10, -300))
    return numpy.sqrt(stats.chi2.isf(probRel, 1))
    ## For very low prob, can use logarithmic approximation:
    ## chi2.isf(prob, 1) = 2 * (numpy.log(2) - numpy.log(prob) - 3)
