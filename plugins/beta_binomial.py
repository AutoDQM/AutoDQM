#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import uproot
import numpy
from autodqm.plugin_results import PluginResults
import scipy.stats as stats
from scipy.special import gammaln
import plotly.graph_objects as go

def comparators():
    return { 
        'beta_binomial' : beta_binomial
    }

def beta_binomial(histpair, pull_cap=40, chi2_cut=100, pull_cut=25, min_entries=10000, norm_type='all', **kwargs): 

    """beta_binomial works on both 1D and 2D"""

    data_hist = histpair.data_hist
    ref_hist = histpair.ref_hist

    data_hist_raw = numpy.copy(data_hist.values())
    ref_hist_raw = numpy.copy(ref_hist.values())

    ## num entries
    data_hist_Entries = numpy.sum(data_hist_raw)
    ref_hist_Entries = numpy.sum(ref_hist_raw)

    nRef = 1

    # Reject empty and low stat hist
    is_good = data_hist_Entries > min_entries

    ## only filled bins used for chi2
    nBinsUsed = numpy.count_nonzero(numpy.add(ref_hist_raw.sum(axis=0), data_hist_raw))
    nBins = data_hist.values().size

    ## calculte pull and chi2
    if nBinsUsed > 0: 
        pull_hist = pull(data_hist_raw, ref_hist_raw)
        chi2 = numpy.square(pull_hist).sum()/nBinsUsed if nBinsUsed > 0 else 0
        max_pull = maxPullNorm(numpy.amax(pull_hist), nBinsUsed)
    else:
        pull_hist = numpy.zeros_like(data_hist_raw)
        chi2 = 0
        max_pull = 0

    ## define if plot anomalous
    is_outlier = is_good and (chi2 > chi2_cut or abs(max_pull) > pull_cut)


    ## plotting
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


    info = {
        'Chi_Squared': float(chi2),
        'Max_Pull_Val': float(max_pull),
        'Data_Entries': str(data_hist_Entries),
        'Ref_Entries': str(ref_hist_Entries),
    }

    artifacts = [pull_hist, str(data_hist_Entries), str(ref_hist_Entries)]


    return PluginResults(
        c,
        show=bool(is_outlier),
        info=info,
        artifacts=artifacts)


def pull(D_raw, R_raw):
    nRef = 1
    tol = 0.01
    prob = numpy.zeros_like(D_raw)
    prob = ProbRel(D_raw, R_raw, 'BetaB')
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

    ## Use logarithmic approximation for very low probs
    if probGoodNorm < cutoff:
        pullNorm = numpy.sqrt(2 * (numpy.log(2) - numpy.log(probGoodNorm) - 3)) * sign
    else:
        pullNorm = numpy.sqrt(stats.chi2.ppf(1-probGoodNorm, 1)) * sign

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
def Pull(Data, Ref, func):
    nData = Data.sum()
    nRef = Ref.sum()
    return (Data - Mean(Data, Ref, func)) / StdDev(Data, Ref, func)


## Exact and approximate values for natural log of the Gamma function
def LogGam(z):
    return gammaln(z)

## Predicted probability of observing Data / nData given a reference of Ref / nRef
def Prob(Data, nData, Ref, nRef, func, kurt=0):
    if func == 'Gaus1' or func == 'Gaus2':
        return stats.norm.pdf( Pull(Data, Ref, func) )
    if func == 'BetaB':
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        return stats.betabinom.pmf(Data, nData, Ref+1, nRef-Ref+1)
    ## Expression for beta-binomial using definition in terms of gamma functions
    ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#As_a_compound_distribution
    if func == 'Gamma':
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        n_  = nData
        k_  = Data
        a_  = Ref+1
        b_  = nRef-Ref+1
        ab_ = nRef+2
        logProb  = LogGam(n_+1) + LogGam(k_+a_) + LogGam(n_-k_+b_) + LogGam(ab_)
        logProb -= ( LogGam(k_+1) + LogGam(n_-k_+1) + LogGam(n_+ab_) + LogGam(a_) + LogGam(b_) )
        return numpy.exp(logProb)

    print('\nInside Prob, no valid func = %s. Quitting.\n' % func)
    sys.exit()


## Predicted probability relative to the maximum probability (i.e. at the mean)
def ProbRel(Data, Ref, func, kurt=0):
    nData = Data.sum()
    nRef = Ref.sum()
    ## Find the most likely expected data value
    exp_up = numpy.ceil(Mean(nData, Ref, 'Gaus1'))
    exp_down = numpy.clip(numpy.floor(Mean(nData, Ref, 'Gaus1')), a_min=0, a_max=None) # make sure nothing goes below zero

    ## Find the maximum likelihood
    maxProb_up  = Prob(exp_up, nData, Ref, nRef,func, kurt)
    maxProb_down = Prob(exp_down, nData, Ref, nRef,func, kurt)
    maxProb = numpy.maximum(maxProb_up, maxProb_down)
    thisProb = Prob(Data, nData, Ref, nRef, func, kurt)
    ## Sanity check to not have relative likelihood > 1
    cond = maxProb < thisProb

    if any(cond.flatten()):
        print(f'for ProbRel')
        print(f'Data: {Data[cond]}\nnData: {nData}\nRef: {Ref[cond]}\nnRef: {nRef}')

    ## make sure check for thisProb < maxProb*0.001 (account for floating point inaccuracies) and just set the ratio to 1 if that is the case
    ratio = thisProb/maxProb
    cond = thisProb > maxProb*0.001
    ratio[cond] = 1

    return ratio #thisProb / maxProb


## Negative log likelihood
def NLL(prob):
    nllprob = -1.0*numpy.log(prob, where=(prob>0))
    nllprob[prob==0] = 999
    nllprob[prob < 0] == -999

    return nllprob


## Convert relative probability to number of standard deviations in normal distribution
def Sigmas(probRel):
    return numpy.sqrt(2.0*NLL(probRel))
