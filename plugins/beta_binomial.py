#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import uproot
import numpy as np
from autodqm.plugin_results import PluginResults
import scipy.stats as stats
from scipy.special import gammaln
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plugins.pullvals import normalize_rows

def comparators():
    return { 
        'beta_binomial' : beta_binomial
    }

def beta_binomial(histpair, pull_cap=15, chi2_cut=10, pull_cut=10, min_entries=1, tol=0.01, norm_type='all', **kwargs):
    """beta_binomial works on both 1D and 2D"""
    data_hist_orig = histpair.data_hist
    ref_hists_orig = []
    for rh in histpair.ref_hists:
        if np.round(rh.values()).sum() > 0 and rh.values().size == data_hist_orig.values().size:
            ref_hists_orig.append(rh)

    ## Observed that float64 is necessary for betabinom to preserve precision with arrays as input.
    ## (Not needed for single values.)  Deep magic that we do not undertand - AWB 2022.08.02
    data_hist_raw = np.round(np.copy(np.float64(data_hist_orig.values())))
    ref_hists_raw = np.round(np.array([np.copy(np.float64(rh.values())) for rh in ref_hists_orig]))
    nRef = len(ref_hists_raw)

    ## Get bin centers from edges() stored by uproot
    x_bins = data_hist_orig.axes[0].edges()
    ## Adjust x-axis range if option set in config file
    if data_hist_raw.ndim == 1 and len(x_bins) > 4:
        binLo, binHi = 0, len(x_bins) - 1
        if 'xmin' in histpair.config.keys() and histpair.config['xmin'] < x_bins[-2]:
            binLo = max( np.nonzero(x_bins >= histpair.config['xmin'])[0][0] - 1, 0 )
        if 'xmax' in histpair.config.keys() and histpair.config['xmax'] > x_bins[1]:
            binHi = min( np.nonzero(x_bins <= histpair.config['xmax'])[0][-1] + 1, len(x_bins) - 1 )

        x_bins = x_bins[binLo:binHi+1]
        data_hist_raw = data_hist_raw[binLo:binHi+1]
        ref_hists_raw = np.array([r[binLo:binHi+1] for r in ref_hists_raw])

    ## does not run beta_binomial if data or ref is 0
    if np.sum(data_hist_raw) <= 0 or nRef == 0:
        return None

    ## summed ref_hist
    ref_hist_sum = ref_hists_raw.sum(axis=0)

    ## Delete leading and trailing bins of 1D plots which are all zeros
    if data_hist_raw.ndim == 1 and len(x_bins) > 20:
        binHi = max( min( np.nonzero(data_hist_raw + ref_hist_sum > 0)[0][-1] + 1, len(x_bins) - 1 ), 20 )
        binLo = min( max( np.nonzero(data_hist_raw + ref_hist_sum > 0)[0][0] - 1, 0 ), binHi - 20 )

        x_bins = x_bins[binLo:binHi+1]
        data_hist_raw = data_hist_raw[binLo:binHi+1]
        ref_hists_raw = np.array([r[binLo:binHi+1] for r in ref_hists_raw])
        ref_hist_sum  = ref_hist_sum[binLo:binHi+1]

    ## num entries
    data_hist_Entries = np.sum(data_hist_raw)
    ref_hist_Entries = [np.sum(rh) for rh in ref_hists_raw]
    ref_hist_Entries_avg = np.round(np.sum(ref_hist_Entries) / nRef)

    # ## normalized ref_hist
    # ref_hist_norm = np.zeros_like(ref_hist_sum)
    # for ref_hist_raw in ref_hists_raw:
    #     ref_hist_norm = np.add(ref_hist_norm, (ref_hist_raw / np.sum(ref_hist_raw)))
    # ref_hist_norm = ref_hist_norm * data_hist_Entries / nRef

    ## only filled bins used for chi2
    nBinsUsed = np.count_nonzero(np.add(ref_hist_sum, data_hist_raw))
    nBins = data_hist_raw.size

    ## calculte pull and chi2, and get probability-weighted reference histogram
    [pull_hist, ref_hist_prob_wgt] = pull(data_hist_raw, ref_hists_raw, tol)
    pull_hist = pull_hist*np.sign(data_hist_raw-ref_hist_prob_wgt)
    chi2 = np.square(pull_hist).sum()/nBinsUsed
    max_pull = maxPullNorm(np.amax(pull_hist), nBinsUsed)
    min_pull = maxPullNorm(np.amin(pull_hist), nBinsUsed)
    if abs(min_pull) > max_pull:
        max_pull = min_pull

    ## define if plot anomalous
    is_outlier = data_hist_Entries >= min_entries and (chi2 > chi2_cut or abs(max_pull) > pull_cut)

    ## plotting
    # For 1D histograms, set pulls larger than pull_cap to pull_cap
    if data_hist_raw.ndim == 1:
        pull_hist = np.where(pull_hist >  pull_cap,  pull_cap, pull_hist)
        pull_hist = np.where(pull_hist < -pull_cap, -pull_cap, pull_hist)
    # For 2D histograms, set empty bins to be blank
    if data_hist_raw.ndim == 2:
        pull_hist = np.where(np.add(ref_hist_sum, data_hist_raw) == 0, None, pull_hist)

    if nRef == 1:
        ref_runs_str = histpair.ref_runs[0]
    else:
        ref_runs_str = str(min([int(x) for x in histpair.ref_runs])) + ' - '
        ref_runs_str += str(max([int(x) for x in histpair.ref_runs]))
        ref_runs_str += ' (' + str(nRef) + ')'

    ##--------- 1D Plotting --------------
    #Check that the hists are 1 dimensional
    if ("TH1" in str(type(data_hist_orig))) and ("TH1" in str(type(ref_hists_orig[0]))):
        if x_bins[0] < -999:
            x_bins[0]=2*x_bins[1]-x_bins[2]

        #Get Titles for histogram, X-axis, Y-axis (Note data_hist_orig.axes will have length > 1 if y-axis title is declared even with 1d plot)
        xAxisTitle = data_hist_orig.axes[0]._bases[0]._members["fTitle"]
        if(len(data_hist_orig.axes) > 1):
            yAxisTitle = data_hist_orig.axes[1]._bases[0]._members["fTitle"]
        else:
            yAxisTitle = ""
        plotTitle = histpair.data_name + " beta-binomial  |  data:" + str(histpair.data_run) + " & ref:" + ref_runs_str
    
        #Plotly doesn't support #circ, #theta, #phi but it does support unicode
        xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
        yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
        plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    
        #For some 1D plots, use a log scale x- or y-axis
        set_logx = (x_bins[1] > 0 and 'opts' in histpair.config.keys() and 'logx' in histpair.config['opts'] and len(x_bins) > 30)
        set_logy = ('opts' in histpair.config.keys() and 'logy' in histpair.config['opts'])
        if set_logx:
            #If first or last bin is an outlier, adjust to be 10% away from next bin (relative to furthest bin)
            if len(x_bins) > 4 and x_bins[1] > 0:
                x_bins[0]  = max(x_bins[0],  pow(10, 1.1*np.log10(x_bins[1]) - 0.1*np.log10(x_bins[-2])))
                x_bins[-1] = min(x_bins[-1], pow(10, 1.1*np.log10(x_bins[-2]) - 0.1*np.log10(x_bins[1])))                
            xAxisTitle = 'log10' + xAxisTitle
        if set_logy:
            yAxisTitle = 'log10' + yAxisTitle
    
        #Plot histogram with previously declared axes and settings to look similar to PyRoot
        c = make_subplots(specs=[[{"secondary_y": True}]])

        c.add_trace( go.Bar(name="data:"+str(histpair.data_run), x=x_bins, y=data_hist_raw, marker_color='red',
                            marker_line_color='red', marker_line_width=1) )
        c.add_trace( go.Scatter(name="ref:"+ref_runs_str, x=x_bins, y=ref_hist_prob_wgt, marker_color='blue',
                                mode='markers', marker_size=(10 / np.log10(len(x_bins)))), secondary_y=False)
        c.add_trace( go.Scatter(name="Pull", x=x_bins, y=pull_hist, marker_color='green',
                                mode='markers', marker_size=[abs(p) for p in pull_hist],
                                marker_symbol='x', marker_line_width=0), secondary_y=True)

        c.update_layout(bargap=0, bargroupgap=0, barmode='overlay', plot_bgcolor='white', legend_itemsizing='constant')
        c.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False, title_text=xAxisTitle)
        c.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False, title_text=yAxisTitle, secondary_y=False)
        c.update_yaxes(showline=False, showgrid=False, title_text="Pull", range=[-pull_cap*1.05, pull_cap*1.05], secondary_y=True)
        if set_logx:
            c.update_xaxes(type="log")
        if set_logy:
            c.update_yaxes(type="log", secondary_y=False)

        c.update_layout(
            title=plotTitle , title_x=0.5,
            # xaxis_title= xAxisTitle,
            # yaxis_title= yAxisTitle,
            font=dict(
                family="Times New Roman",
                size=9,
                color="black"
            )
        )
        ref_text = "ref:"+ref_runs_str
        data_text = "data:"+str(histpair.data_run)

    ## --------- end 1D plotting ---------


    ##---------- 2d Plotting --------------
    # Check that the hists are 2 dimensional
    if ( (       "TH2" in str(type(data_hist_orig)) and       "TH2" in str(type(ref_hists_orig[0])) ) or
         ("TProfile2D" in str(type(data_hist_orig)) and "TProfile2" in str(type(ref_hists_orig[0])) ) ):
        
        colors = ['rgb(26, 42, 198)', 'rgb(118, 167, 231)', 'rgb(215, 226, 194)', 'rgb(212, 190, 109)', 'rgb(188, 76, 38)']
        #Getting Plot labels for x-axis and y-axis as well as type (linear or categorical)
        xLabels = None
        yLabels = None
        c = None
        x_axis_type = 'linear'
        y_axis_type = 'linear';
        if data_hist_orig.axes[0].labels():
           xLabels = [str(x) for x in data_hist_orig.axes[0].labels()]
           x_axis_type = 'category'
        else:
           xLabels = [str(data_hist_orig.axes[0]._members["fXmin"] + x * (data_hist_orig.axes[0]._members["fXmax"]-data_hist_orig.axes[0]._members["fXmin"])/data_hist_orig.axes[0]._members["fNbins"]) for x in range(0,data_hist_orig.axes[0]._members["fNbins"]+1)]
    
        if data_hist_orig.axes[1].labels():
           yLabels = [str(x) for x in data_hist_orig.axes[1].labels()]
           y_axis_type = 'category'
        else:
           yLabels = [str(data_hist_orig.axes[1]._members["fXmin"] + x * (data_hist_orig.axes[1]._members["fXmax"]-data_hist_orig.axes[1]._members["fXmin"])/data_hist_orig.axes[1]._members["fNbins"]) for x in range(0,data_hist_orig.axes[1]._members["fNbins"]+1)]
    
        if("xlabels" in histpair.config.keys()):
            xLabels=histpair.config["xlabels"]
            x_axis_type = 'category'
        if("ylabels" in histpair.config.keys()):
            yLabels=histpair.config["ylabels"]
            y_axis_type = 'category'
    
        pull_hist = np.transpose(pull_hist)
    
        #Getting Plot Titles for histogram, x-axis and y-axis
        xAxisTitle = data_hist_orig.axes[0]._bases[0]._members["fTitle"]
        yAxisTitle = data_hist_orig.axes[1]._bases[0]._members["fTitle"]
        plotTitle = histpair.data_name + " beta-binomial  |  data:" + str(histpair.data_run) + " & ref:" + ref_runs_str
    
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

    if nRef == 1:
        Ref_Entries_str = str(int(ref_hist_Entries[0]))
    else:
        Ref_Entries_str = " - ".join([str(int(min(ref_hist_Entries))), str(int(max(ref_hist_Entries)))])

    info = {
        'Chi_Squared': float(round(chi2, 2)),
        'Max_Pull_Val': float(round(max_pull,2)),
        'Data_Entries': str(int(data_hist_Entries)),
        'Ref_Entries': Ref_Entries_str
    }

    artifacts = [pull_hist, str(int(data_hist_Entries)), Ref_Entries_str]

    return PluginResults(
        c,
        show=bool(is_outlier),
        info=info,
        artifacts=artifacts)


def pull(D_raw, R_list_raw, tol=0.01):
    nRef = len(R_list_raw)
    probs = []

    for R_raw in R_list_raw:
        ## Compute per-bin probabilities with beta-binomial function
        ## Protect against zero values with a floor at 10^-300 (37 sigma)
        probs.append( np.maximum(ProbRel(D_raw, R_raw, 'BetaB', tol), pow(10, -300)) )

    ## Per-bin probability is the per-bin average over all ref hists
    prob = np.array(probs).sum(axis=0) / nRef
    pull = Sigmas(prob)

    ## Reference histogram weighted by per-bin probabilities
    R_prob_wgt_avg = np.zeros_like(D_raw)

    for iR in range(len(R_list_raw)):
        R_raw = R_list_raw[iR]
        ## Get reference hist normalized to 1
        R_prob_wgt = R_raw / np.sum(R_raw)
        ## Compute per-bin probabilities relative to sum of probabilites
        prob_rel = np.divide(probs[iR], np.array(probs).sum(axis=0))
        ## Scale normalized reference by per-bin relative probabilities
        R_prob_wgt = np.multiply(R_prob_wgt, prob_rel)
        ## Add into average probability-weighted distribution
        R_prob_wgt_avg = np.add(R_prob_wgt_avg, R_prob_wgt)

    ## Normalize to data
    R_prob_wgt_avg = R_prob_wgt_avg * np.sum(D_raw)

    return [pull, R_prob_wgt_avg]

def maxPullNorm(maxPull, nBinsUsed, cutoff=pow(10,-15)):
    sign = np.sign(maxPull)
    ## sf (survival function) better than 1-cdf for large pulls (no precision error)
    probGood = stats.chi2.sf(np.power(min(abs(maxPull), 37), 2), 1)

    ## Use binomial approximation for low probs (accurate within 1%)
    if nBinsUsed * probGood < 0.01:
        probGoodNorm = nBinsUsed * probGood
    else:
        probGoodNorm = 1 - np.power(1 - probGood, nBinsUsed)

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
        output = 1.0*nData*np.sqrt(np.clip(Ref, a_min=1, a_max=None))/nRef
        output[mask] = (1.0*nData*np.sqrt(np.clip(nRef-Ref, a_min=1, a_max=None)))[mask]/nRef
    elif func == 'Gaus2':
        ## instead of calculating max(Ref, 1), set the whole array to have a lower limit of 1
        clipped = np.clip(Ref, a_min=1, a_max=None)
        output = 1.0*nData*np.sqrt( clipped/np.square(nRef) + Mean(nData, Ref, nRef, func)/np.square(nData) )
        clipped = np.clip(nRef-Ref, a_min=1, a_max=None)
        output[mask] = (1.0*nData*np.sqrt( clipped/np.square(nRef) + (nData - Mean(nData, Ref, nRef, func))/np.square(nData) ))
    elif (func == 'BetaB') or (func == 'Gamma'):
        output = 1.0*np.sqrt( nData*(Ref+1)*(nRef-Ref+1)*(nRef+2+nData) / (np.power(nRef+2, 2)*(nRef+3)) )
        
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
    scaleTol = np.power(1 + np.power(Ref * tol**2, 2), -0.5)
    nRef_tol = np.round(scaleTol * nRef)
    Ref_tol = np.round(Ref * scaleTol)
    nData_arr = np.zeros_like(Data) + np.float64(nData)

    if func == 'Gaus1' or func == 'Gaus2':
        return stats.norm.pdf( numStdDev(Data, Ref_tol, func) )
    if func == 'BetaB':
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        return stats.betabinom.pmf(Data, nData_arr, Ref_tol + 1, nRef_tol - Ref_tol + 1)
    ## Expression for beta-binomial using definition in terms of gamma functions
    ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#As_a_compound_distribution
    if func == 'Gamma':
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        n_  = nData_arr
        k_  = Data
        a_  = Ref_tol + 1
        b_  = nRef_tol - Ref_tol + 1
        ab_ = nRef_tol + 2
        logProb  = gammaln(n_+1) + gammaln(k_+a_) + gammaln(n_-k_+b_) + gammaln(ab_)
        logProb -= ( gammaln(k_+1) + gammaln(n_-k_+1) + gammaln(n_+ab_) + gammaln(a_) + gammaln(b_) )
        return np.exp(logProb)

    print('\nInside Prob, no valid func = %s. Quitting.\n' % func)
    sys.exit()


## Predicted probability relative to the maximum probability (i.e. at the mean)
def ProbRel(Data, Ref, func, tol=0.01):
    nData = Data.sum()
    nRef = Ref.sum()
    ## Find the most likely expected data value
    exp_up = np.clip(np.ceil(Mean(Data, Ref, 'Gaus1')), a_min=None, a_max=nData) # make sure nothing goes above nData
    exp_down = np.clip(np.floor(Mean(Data, Ref, 'Gaus1')), a_min=0, a_max=None) # make sure nothing goes below zero

    ## Find the maximum likelihood
    maxProb_up  = Prob(exp_up, nData, Ref, nRef, func, tol)
    maxProb_down = Prob(exp_down, nData, Ref, nRef, func, tol)
    maxProb = np.maximum(maxProb_up, maxProb_down)
    thisProb = Prob(Data, nData, Ref, nRef, func, tol)

    ## Sanity check to not have relative likelihood > 1
    ratio = np.divide(thisProb, maxProb, out=np.zeros_like(thisProb), where=maxProb!=0)
    cond = thisProb > maxProb
    ratio[cond] = 1
        
    return ratio


## Convert relative probability to number of standard deviations in normal distribution
def Sigmas(probRel):
    ## chi2.isf function fails for probRel < 10^-323, so cap at 10^-300 (37 sigma)
    probRel = np.maximum(probRel, pow(10, -300))
    return np.sqrt(stats.chi2.isf(probRel, 1))
    ## For very low prob, can use logarithmic approximation:
    ## chi2.isf(prob, 1) = 2 * (np.log(2) - np.log(prob) - 3)
