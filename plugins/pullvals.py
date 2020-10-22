#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ROOT
from autodqm.plugin_results import PluginResults


def comparators():
    return {
        'pull_values': pullvals
    }


def pullvals(histpair,
             pull_cap=25, chi2_cut=500, pull_cut=20, min_entries=100000, norm_type='all',
             **kwargs):
    """Can handle poisson driven TH2s or generic TProfile2Ds"""
    data_hist = histpair.data_hist
    ref_hist = histpair.ref_hist

    # Check that the hists are histograms
    if not data_hist.InheritsFrom('TH1') or not ref_hist.InheritsFrom('TH1'):
        return None

    # Check that the hists are 2 dimensional
    if data_hist.GetDimension() != 2 or ref_hist.GetDimension() != 2:
        return None

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(ROOT.kLightTemperature)
    ROOT.gStyle.SetNumberContours(255)

    # Get empty clone of reference histogram for pull hist
    if ref_hist.InheritsFrom('TProfile2D'):
        pull_hist = ref_hist.ProjectionXY("pull_hist")
    else:
        pull_hist = ref_hist.Clone("pull_hist")
    pull_hist.Reset()

    # Reject empty histograms
    is_good = data_hist.GetEntries() != 0 and data_hist.GetEntries() >= min_entries

    # Normalize data_hist
    if norm_type == "row":
        normalize_rows(data_hist, ref_hist)
    else:
        if data_hist.GetEntries() > 0:
            data_hist.Scale(ref_hist.GetSumOfWeights() / data_hist.GetSumOfWeights())

    max_pull = 0
    nBins = 0
    chi2 = 0
    for x in range(1, ref_hist.GetNbinsX() + 1):
        for y in range(1, ref_hist.GetNbinsY() + 1):

            # Bin 1 data
            bin1 = data_hist.GetBinContent(x, y)

            # Bin 2 data
            bin2 = ref_hist.GetBinContent(x, y)

            # TEMPERARY - Getting Symmetric Error - Need to update with >Proper Poisson error 
            if ref_hist.InheritsFrom('TProfile2D'):
                bin1err = data_hist.GetBinError(x, y)
                bin2err = ref_hist.GetBinError(x, y)
            else:
                bin1err, bin2err = bin1**(.5), bin2**(.5)

            # Count bins for chi2 calculation
            nBins += 1

            # Ensure that divide-by-zero error is not thrown when calculating pull
            if bin1err == 0 and bin2err == 0:
                new_pull = 0
            else:
                new_pull = pull(bin1, bin1err, bin2, bin2err)

            # Sum pulls
            chi2 += new_pull**2

            # Check if max_pull
            max_pull = max(max_pull, abs(new_pull))

            # Clamp the displayed value
            fill_val = max(min(new_pull, pull_cap), -pull_cap)

            # If the input bins were explicitly empty, make this bin white by
            # setting it out of range
            if bin1 == bin2 == 0:
                fill_val = -999

            # Fill Pull Histogram
            pull_hist.SetBinContent(x, y, fill_val)

    # Compute chi2
    chi2 = (chi2 / nBins)

    is_outlier = is_good and (chi2 > chi2_cut or abs(max_pull) > pull_cut)

    # Set up canvas
    c = ROOT.TCanvas('c', 'Pull')

    # Plot pull hist
    pull_hist.GetZaxis().SetRangeUser(-(pull_cap), pull_cap)
    pull_hist.SetTitle(pull_hist.GetTitle() + " Pull Values")
    pull_hist.Draw("colz")

    # Text box
    data_text = ROOT.TLatex(.52, .91,
                            "#scale[0.6]{Data: " + histpair.data_run + "}")
    ref_text = ROOT.TLatex(.72, .91,
                           "#scale[0.6]{Ref: " + histpair.ref_run + "}")
    data_text.SetNDC(ROOT.kTRUE)
    ref_text.SetNDC(ROOT.kTRUE)
    data_text.Draw()
    ref_text.Draw()

    info = {
        'Chi_Squared': chi2,
        'Max_Pull_Val': max_pull,
        'Data_Entries': data_hist.GetEntries(),
        'Ref_Entries': ref_hist.GetEntries(),
    }

    artifacts = [pull_hist, data_text, ref_text]

    return PluginResults(
        c,
        show=is_outlier,
        info=info,
        artifacts=artifacts)


def pull(bin1, binerr1, bin2, binerr2):
    ''' Calculate the pull value between two bins.
        pull = (data - expected)/sqrt(sum of errors in quadrature))
        data = |bin1 - bin2|, expected = 0
    '''
    return (bin1 - bin2) / ((binerr1**2 + binerr2**2)**0.5)


def normalize_rows(data_hist, ref_hist):

    for y in range(1, ref_hist.GetNbinsY() + 1):

        # Stores sum of row elements
        rrow = 0
        frow = 0

        # Sum over row elements
        for x in range(1, ref_hist.GetNbinsX() + 1):

            # Bin data
            rbin = ref_hist.GetBinContent(x, y)
            fbin = data_hist.GetBinContent(x, y)

            rrow += rbin
            frow += fbin

        # Scaling factors
        # Prevent divide-by-zero error
        if frow == 0:
            frow = 1
        if frow > 0:
            sf = float(rrow) / frow
        else:
            sf = 1
        # Prevent scaling everything to zero
        if sf == 0:
            sf = 1

        # Normalization
        for x in range(1, data_hist.GetNbinsX() + 1):
            # Bin data
            fbin = data_hist.GetBinContent(x, y)
            fbin_err = data_hist.GetBinError(x, y)

            # Normalize bin
            data_hist.SetBinContent(x, y, (fbin * sf))
            data_hist.SetBinError(x, y, (fbin_err * sf))

    return
