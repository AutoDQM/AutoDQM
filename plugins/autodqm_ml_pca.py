import os
import json
import numpy

from sklearn import decomposition
import plotly.graph_objects as go

from autodqm.plugin_results import PluginResults

def comparators():
    return {
            "autodqm_ml_pca" : autodqm_ml_pca
    }


def load_model(model_file):
    """
    Load PCA model from pickle file

    :param model_file: folder containing PCA pickles
    :type model_file: str
    
    """
    with open(model_file, "r") as f_in:
        pcaParams = json.load(f_in)

    pca = decomposition.PCA(random_state = 0)

    pca.components_ = numpy.array(pcaParams['components_'])
    pca.explained_variance_ = numpy.array(pcaParams['explained_variance_'])
    pca.explained_variance_ratio_ = numpy.array(pcaParams['explained_variance_ratio_'])
    pca.singular_values_ = numpy.array(pcaParams['singular_values_'])
    pca.mean_ = numpy.array(pcaParams['mean_'])
    pca.n_components_ = numpy.array(pcaParams['n_components_'])
    pca.n_features_ = numpy.array(pcaParams['n_features_'])
    pca.n_samples_ = numpy.array(pcaParams['n_samples_'])
    pca.noise_variance_ = numpy.array(pcaParams['noise_variance_'])

    # Need to implement in AutoDQM_ML
    if "normalize" in pcaParams.keys():
        pca.normalize = pcaParams["normalize"]
    else:
        pca.normalize = True

    # We should also add sse quantiles to the saved PCA results
    # this way, we can show each reconstructed PCA with a message like:
    #   "SSE = 0.XXX (greater than SSE of Y% of good histograms)"
    # rather than making a binary decision about whether its anomalous

    return pca


def normalize(hist):
    sum_entries = numpy.sum(hist)
    if sum_entries > 1:
        return hist * ( 1. / sum_entries)


def predict(hist, pca):
    n_dim = len(hist.shape)
    if n_dim != 1: # only 1d hists for now 
        raise ValueError()

    hist_reshape = hist.reshape(1,-1)
    hist_transformed = pca.transform(hist_reshape)
    hist_reconstructed = pca.inverse_transform(hist_transformed)

    sse = numpy.sum(
        (hist - hist_reconstructed) ** 2
    ).flatten()

    return float(sse), hist_reconstructed.flatten()


def plot(histpair, hist, hist_reconstructed):
    """

    """
    bins = [(histpair.data_hist.axes[0].edges()[x] + histpair.data_hist.axes[0].edges()[x+1])/2 for x in range(0,len(histpair.data_hist.axes[0]))]
    if bins[0] < -999:
        bins[0]=2*bins[1]-bins[2]

    xAxisTitle = histpair.data_hist.axes[0]._bases[0]._members["fTitle"]
    if(len(histpair.data_hist.axes) > 1):
        yAxisTitle = histpair.data_hist.axes[1]._bases[0]._members["fTitle"]
    else:
        yAxisTitle = ""
    plotTitle = histpair.data_name + " PCA Test  |  data:" + str(histpair.data_run) + " & ref:" + str(histpair.ref_runs[0])

    #Plotly doesn't support #circ, #theta, #phi but does support unicode
    xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")

    c = go.Figure()
    c.add_trace(go.Bar(name="Data:"+str(histpair.data_run), x=bins, y=hist, marker_color='rgb(125,153,207)', opacity=.7))
    c.add_trace(go.Scatter(name="PCA Reco", x=bins, y=hist_reconstructed, marker_color='rgb(192, 255, 195)', opacity=.7))

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
    return c


def autodqm_ml_pca(histpair, **kwargs):
    """
    Takes in an autodqm.histpair.HistPair object and 
    returns an autodqm.plugin_results.PluginResults object
    """

    # Check to see if a trained PCA matches this histpair
    year = histpair.data_series[-4:]

    if not "jar_dir" in histpair.config:
        show = False
        return None

    f_pca = "/var/www/cgi-bin/models/autodqm_ml_pca/{0}/{1}/{2}.json".format(histpair.config["jar_dir"], year, histpair.data_name)

    if not os.path.exists(f_pca):
        show = False
        return None
    else:
        show = True

    # Found matched PCA
    pca = load_model(f_pca)

    # Grab histogram
    hist = histpair.data_hist.values()

    # Normalize entries if this was done during training (usually yes)
    n_entries = int(numpy.sum(hist))
    if pca.normalize:
        hist = normalize(hist)

    # Check for cases where PCA could fail/is not applicable 
    if hist is None:
        show = False
        return None

    if len(hist.shape) != 1: # only for 1d histograms
        show = False
        return None

    if hist.shape[0] != pca.n_features_: # number of bins don't match (can happen if you try to apply a PCA trained on histograms from one year on another year, where that histogram has the same naming but its binning has been changed)
        show = False
        return None

    # Apply PCA
    sse, hist_reconstructed = predict(hist, pca) 

    # Create plot
    canvas = plot(histpair, hist, hist_reconstructed)

    # Metadata
    artifacts = {
        "data" : hist,
        "reco" : hist_reconstructed
    } 
    info = {
        "Data_Entries" : str(n_entries),
        "Sum of Squared Errors" : round(sse, 6),
        "PCA Components" : int(pca.n_components_)
    }

    return PluginResults(
        canvas,
        show=show,
        info=info,
        artifacts=artifacts
    )
