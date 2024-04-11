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

def beta_binomial(histpair, pull_cap=10, chi2_cut=10, pull_cut=10, min_entries=1, tol=0.01, norm_type='all', **kwargs):
    """beta_binomial works on both 1D and 2D"""
    data_hist_orig = histpair.data_hist
    ref_hists_orig = [rh for rh in histpair.ref_hists if rh.values().size == data_hist_orig.values().size]

    ## Observed that float64 is necessary for betabinom to preserve precision with arrays as input.
    ## (Not needed for single values.)  Deep magic that we do not undertand - AWB 2022.08.02
    data_hist_raw = np.round(np.copy(np.float64(data_hist_orig.values())))
    ref_hists_raw = np.round(np.array([np.copy(np.float64(rh.values())) for rh in ref_hists_orig]))

    ## Get bin centers from edges() stored by uproot
    x_bins = data_hist_orig.axes[0].edges()

    ## Concatenate multiple histograms together
    do_concat = histpair.data_concat and histpair.ref_concat
    if do_concat:
        for dhc in histpair.data_concat:
            data_hist_raw = np.concatenate((data_hist_raw, np.round(np.copy(np.float64(dhc.values())))))
            x_offset = x_bins[-1] - dhc.axes[0].edges()[0]
            x_bins = np.concatenate((x_bins[:-1], dhc.axes[0].edges() + x_offset))

        ref_hists_concat = []
        for ii in range(len(ref_hists_raw)):
            iRef_concat = np.copy(ref_hists_raw[ii])
            for rhc in histpair.ref_concat[ii]:
                iRef_concat = np.concatenate((iRef_concat, np.copy(np.float64(rhc.values()))))
            ref_hists_concat.append(iRef_concat)
        ref_hists_raw = np.array(ref_hists_concat)

    ## Delete empty reference histograms
    ref_hists_raw = np.array([rhr for rhr in ref_hists_raw if np.sum(rhr) > 0])
    nRef = len(ref_hists_raw)

    ## Does not run beta_binomial if data or ref is 0
    if np.sum(data_hist_raw) <= 0 or nRef == 0:
        return None

    ## Adjust x-axis range for 1D plots if option set in config file
    if data_hist_raw.ndim == 1 and len(x_bins) > 4 and not do_concat:
        binLo, binHi = 0, len(x_bins) - 1
        if 'xmin' in histpair.config.keys() and histpair.config['xmin'] < x_bins[-2]:
            binLo = max( np.nonzero(x_bins >= histpair.config['xmin'])[0][0], 0 )
        if 'xmax' in histpair.config.keys() and histpair.config['xmax'] > x_bins[1]:
            binHi = min( np.nonzero(x_bins <= histpair.config['xmax'])[0][-1], len(x_bins) - 1 )

        ## Check if new binning makes data or sum of references have all empty bins
        if np.sum(data_hist_raw[binLo:binHi+1]) <= 0 or sum(np.sum(r[binLo:binHi+1]) > 0 for r in ref_hists_raw) == 0:
            binLo, binHi = 0, len(x_bins) - 1

        x_bins = x_bins[binLo:binHi+1]
        data_hist_raw = data_hist_raw[binLo:binHi+1]
        ref_hists_raw = np.array([r[binLo:binHi+1] for r in ref_hists_raw if np.sum(r[binLo:binHi+1]) > 0])

    ## Update nRef and again don't run on empty histograms
    nRef = len(ref_hists_raw)
    if nRef == 0:
        return None

    ## Summed ref_hist
    ref_hist_sum = ref_hists_raw.sum(axis=0)

    ## Delete leading and trailing bins of 1D plots which are all zeros
    if data_hist_raw.ndim == 1 and len(x_bins) > 20 and not do_concat:
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

    ## access per-histogram settings for max_pull, chi2, axis titles
    plot_title  = None
    xaxis_title = None
    yaxis_title = None
    if 'opts' in histpair.config.keys():
        for opt in histpair.config['opts']:
            if 'pull_cap' in opt: pull_cap = float(opt.split('=')[1])
            if 'chi2_cut' in opt: chi2_cut = float(opt.split('=')[1])
            if 'pull_cut' in opt: pull_cut = float(opt.split('=')[1])
            if 'plot_title' in opt: plot_title = str(opt.split('=')[1])
            if 'xaxis_title' in opt: xaxis_title = str(opt.split('=')[1])
            if 'yaxis_title' in opt: yaxis_title = str(opt.split('=')[1])

    ## define if plot anomalous
    is_outlier = data_hist_Entries >= min_entries and (chi2 > chi2_cut or abs(max_pull) > pull_cut)

    ## For subsystems with many many plots (e.g. DT_DOC1), only generate somewhat anomalous plots
    sel_display_chi2    = -1.0
    sel_display_maxPull = -1.0
    if 'sel_display' in histpair.config.keys() and not histpair.config['sel_display'] is None:
        for sel in histpair.config['sel_display']:
            sels = sel.split('_')  ## Comparator, test, threshold
            if not sels[0] == 'BB': continue
            if sels[1] == 'Chi2':    sel_display_chi2    = float(sels[2])
            if sels[1] == 'MaxPull': sel_display_maxPull = float(sels[2])
    
        # Defining the Selective display thresholds based on the Chi2 and MaxPull choosen thresholds 
        if chi2 < chi2_cut*0.5 and max_pull < pull_cut*0.5 and not is_outlier:
            return None

    ##--------- Plotting --------------
    # For 1D histograms, set pulls larger than pull_cap to pull_cap
    if data_hist_raw.ndim == 1:
        pull_hist = np.where(pull_hist >  pull_cap,  pull_cap, pull_hist)
        pull_hist = np.where(pull_hist < -pull_cap, -pull_cap, pull_hist)
    # For 2D histograms, set empty bins to be blank
    if data_hist_raw.ndim == 2:
        pull_hist = np.where(np.add(ref_hist_sum, data_hist_raw) == 0, None, pull_hist)

    if nRef == 1:
        ref_runs_str = "Reference:<br>"+histpair.ref_runs[0]
    else:
        ref_runs_str = "References:"
        for rr in histpair.ref_runs:
            ref_runs_str += "<br>"+str(rr)

    # Set titles. Note data_hist_orig.axes will have length > 1 if y-axis title is declared, even with 1D plot.
    plotTitle = (plot_title if plot_title else histpair.data_name)
    if (nRef == 1):
        plotTitle += ('<br><span style="font-size: 14px;">Run '+
                      str(histpair.data_run)+' vs. '+histpair.ref_runs[0]+'</span>')
    else:
        plotTitle += ('<br><span style="font-size: 14px;">Run '+
                      str(histpair.data_run)+' vs. '+str(nRef)+' references</span>')
    xAxisTitle = (xaxis_title if xaxis_title else \
                  data_hist_orig.axes[0]._bases[0]._members["fTitle"])
    yAxisTitle = (yaxis_title if yaxis_title else \
                  ("" if (len(data_hist_orig.axes) <= 1) else \
                   data_hist_orig.axes[1]._bases[0]._members["fTitle"]))

    # Plotly doesn't support #circ, #theta, #phi but it does support unicode
    plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7").replace("#mu","\u03BC")
    yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7").replace("#mu","\u03BC")


    ##--------- 1D Plotting --------------
    #Check that the hists are 1 dimensional
    if ("TH1" in str(type(data_hist_orig))) and ("TH1" in str(type(ref_hists_orig[0]))):
        if x_bins[0] < -999:
            x_bins[0]=2*x_bins[1]-x_bins[2]
    
        #For some 1D plots, use a log scale x- or y-axis
        set_logx = (x_bins[1] > 0 and 'opts' in histpair.config.keys() and 'logx' in histpair.config['opts'] and len(x_bins) > 30) and not do_concat
        set_logy = ('opts' in histpair.config.keys() and 'logy' in histpair.config['opts'])
        if set_logx:
            #If first or last bin is an outlier, adjust to be 10% away from next bin (relative to furthest bin)
            if len(x_bins) > 4 and x_bins[1] > 0:
                x_bins[0]  = max(x_bins[0],  pow(10, 1.1*np.log10(x_bins[1]) - 0.1*np.log10(x_bins[-2])))
                x_bins[-1] = min(x_bins[-1], pow(10, 1.1*np.log10(x_bins[-2]) - 0.1*np.log10(x_bins[1])))                
            xAxisTitle = 'log10' + xAxisTitle
        if set_logy:
            yAxisTitle = 'log10' + yAxisTitle
            data_hist_raw = np.where(data_hist_raw < 0.1, 0.1, data_hist_raw)
            ref_hist_prob_wgt = np.where(ref_hist_prob_wgt < 0.1, 0.1, ref_hist_prob_wgt)
    
        #Plot histogram with previously declared axes and settings to look similar to PyRoot
        #See https://plotly.com/python/subplots/#custom-sized-subplot
        can = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing = 0.05)

        #Colors from Google colour picker: https://g.co/kgs/a8Uc8Yw
        can.add_trace( go.Bar(name="Beta-binomial<br>pull value", x=x_bins, y=pull_hist, marker_color='green',
                              marker_line_color='green', marker_line_width=1), row=2, col=1 )
        can.add_trace( go.Bar(name=ref_runs_str, x=x_bins, y=ref_hist_prob_wgt, marker_color='#f58282',
                              marker_line_color='#f58282', marker_line_width=1), row=1, col=1)
        can.add_trace( go.Scatter(name="Data run:<br>"+str(histpair.data_run), x=x_bins, y=data_hist_raw,
                                  marker_color='blue', marker_line_color='blue', mode='markers',
                                  marker_size=[np.power(1+abs(p),0.2)*(7 / np.log10(len(x_bins))) for p in pull_hist]),
                       row=1, col=1 )

        can.update_layout(bargap=0, bargroupgap=0, barmode='overlay', plot_bgcolor='white',
                          legend_itemsizing='constant', legend_traceorder='reversed')
        can.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False, title_text=xAxisTitle, row=2, col=1)
        can.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False, title_text=yAxisTitle)
        can.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=True,  title_text="Pull", range=[-pull_cap*1.05, pull_cap*1.05], row=2, col=1)
        if set_logx:
            can.update_xaxes(type="log")
        if set_logy:
            can.update_yaxes(type="log", row=1, col=1)

        can.update_layout(
            title=plotTitle , title_x=0.5,
            font=dict(
                family="Times New Roman",
                size=12,
                color="black"
            )
        )
        ref_text = ref_runs_str
        data_text = "Data run:<br>"+str(histpair.data_run)

    ## --------- end 1D plotting ---------


    ##---------- 2d Plotting --------------
    # Check that the hists are 2 dimensional
    if ( (       "TH2" in str(type(data_hist_orig)) and       "TH2" in str(type(ref_hists_orig[0])) ) or
         ("TProfile2D" in str(type(data_hist_orig)) and "TProfile2" in str(type(ref_hists_orig[0])) ) ):
        
        colors = ['rgb(26, 42, 198)', 'rgb(118, 167, 231)', 'rgb(215, 226, 194)', 'rgb(212, 190, 109)', 'rgb(188, 76, 38)']
        #Getting Plot labels for x-axis and y-axis as well as type (linear or categorical)
        xLabels = None
        yLabels = None
        can = None
        x_axis_type = 'linear'
        y_axis_type = 'linear'
        if data_hist_orig.axes[0].labels():
           xLabels = [str(x) for x in data_hist_orig.axes[0].labels()]
           x_axis_type = 'category'
        else:
           xLabels = [ str( data_hist_orig.axes[0]._members["fXmin"] +
                      x * ( data_hist_orig.axes[0]._members["fXmax"] -
                            data_hist_orig.axes[0]._members["fXmin"] ) /
                            data_hist_orig.axes[0]._members["fNbins"] )
                       for x in range(0, data_hist_orig.axes[0]._members["fNbins"] + 1) ]

        if data_hist_orig.axes[1].labels():
           yLabels = [str(x) for x in data_hist_orig.axes[1].labels()]
           y_axis_type = 'category'
        else:
           yLabels = [ str(data_hist_orig.axes[1]._members["fXmin"] +
                      x * (data_hist_orig.axes[1]._members["fXmax"] -
                           data_hist_orig.axes[1]._members["fXmin"] ) /
                           data_hist_orig.axes[1]._members["fNbins"] )
                       for x in range(0, data_hist_orig.axes[1]._members["fNbins"] + 1) ]
    
        if("xlabels" in histpair.config.keys()):
            xLabels=histpair.config["xlabels"]
            x_axis_type = 'category'
        if("ylabels" in histpair.config.keys()):
            yLabels=histpair.config["ylabels"]
            y_axis_type = 'category'

        pull_hist = np.transpose(pull_hist)
    
        #Repeat labels for concatenated histograms
        if do_concat:
            xLabels = None
            xAxisTitle += ' (bin indices from concatenated histograms)'
            ## For some reason the below doesn't work, even though it produces
            ## xLabels with the correct dimension. Strange and frustrating. - AWB 2022.08.09
            # xLabels_orig = xLabels.copy()
            # iCat = 1
            # while len(xLabels) < len(x_bins):
            #     xLabels = xLabels + ['%s (C%d)' % (xx, iCat) for xx in xLabels_orig[1:]]
            #     iCat += 1
    
        #Plot pull-values using 2d heatmap will settings to look similar to old Pyroot version
        can = go.Figure(data=go.Heatmap(z=pull_hist, zmin=-pull_cap, zmax=pull_cap, colorscale=colors,
                                        x=xLabels, y=yLabels, colorbar={"title": "Beta-binomial pull value", "titleside": "right"}))
        can['layout'].update(plot_bgcolor='white')
        can.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False, type=x_axis_type)
        can.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False, type=y_axis_type)
        can.update_layout(
            title=plotTitle , title_x=0.5,
            xaxis_title=xAxisTitle,
            yaxis_title=yAxisTitle,
            font=dict(
                family="Times New Roman",
                size=12,
                color="black"
            )
        )
    ##----- end 2D plotting --------

    if nRef == 1:
        Ref_Entries_str = str(int(ref_hist_Entries[0]))
    else:
        Ref_Entries_str = " - ".join([str(int(min(ref_hist_Entries))), str(int(max(ref_hist_Entries)))])

    info = {
        'Chi_Squared': np.nan_to_num(float(round(chi2, 2))),
        'Max_Pull_Val': np.nan_to_num(float(round(max_pull,2))),
        'Data_Entries': str(int(data_hist_Entries)),
        'Ref_Entries': Ref_Entries_str
    }

    artifacts = [pull_hist, str(int(data_hist_Entries)), Ref_Entries_str]

    return PluginResults(
        can,
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
