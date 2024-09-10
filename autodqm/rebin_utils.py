"""
This script stores useful functions to perform the rebinning of histograms and the "hot-bin" algorithm.

In occupancy histograms, noisy channels are common and usually have large values in the pull histogram.
This raises the chi-squared (chi2) and max-pull thresholds, sometimes "burying" the true regional anomalies.

These algorithms were developed to mitigate this effect and increase the relative chi-squared (Chi2) with respect to "local" anomalies.
"""

import numpy as np

# Instead of rebinning the data and reference histograms, we will rebin the pull histogram
def rebin_pull_hist(pull_hist_orig, nBinsUsed):

        import plugins.beta_binomial as bb

        nBinsUsed_orig = nBinsUsed #np.count_nonzero(pull_hist_orig)	
        pull_hist = pull_hist_orig.copy()

	# Applying the hot-bin algorithm
        pull_hist = substitute_max_bin_with_average(pull_hist)

	# Store chi2 values
        chi2_keep = []
        chi2_keep.append(np.square(pull_hist).sum() / nBinsUsed_orig)
        ## maxpull_keep = []
	
        # Padding the pull histogram
        pull_hist = pad_histogram(pull_hist)

        nBinsX, nBinsY = np.shape(pull_hist)[0], np.shape(pull_hist)[1]

        for rbX in range(1, int(np.sqrt(nBinsX))): 
            if (nBinsX % rbX != 0): continue
            for rbY in range(1, int(np.sqrt(nBinsY))):  
                if (nBinsY % rbY != 0): continue     

                pull_hist_rb = Rebin(pull_hist, rbX, rbY, pull = True)     

                # Number of bins in the new histogram
                nBinsUsed_rb = np.count_nonzero(pull_hist_rb)

                # Calculate the scaled chi2 and MaxPull value for the rebinned histogram
                if nBinsUsed_rb == 0: continue
                scaler   = np.sqrt(nBinsUsed_orig / nBinsUsed_rb)
                chi2     = scaler * np.square(pull_hist_rb).sum() / nBinsUsed_rb
                ## max_pull = bb.maxPullNorm(np.amax(pull_hist_rb), nBinsUsed_rb)
                ## min_pull = bb.maxPullNorm(np.amin(pull_hist_rb), nBinsUsed_rb)
                ## if abs(min_pull) > max_pull:
                    ## max_pull = min_pull  

                chi2_keep.append(chi2)
                ## maxpull_keep.append(max_pull)

        return np.max(chi2_keep)

def Rebin(hist, rbX, rbY, pull = True):

        # Get the dimensions of the input histogram
        x_bins, y_bins = hist.shape

        # Calculate the dimensions of the rebinned histogram
        new_x_bins = x_bins // rbX
        new_y_bins = y_bins // rbY

        # Initialize the rebinned histogram
        rebinned_hist = np.zeros((new_x_bins, new_y_bins))

        # Loop over the new bins and sum the appropriate values
        for i in range(new_x_bins):
            for j in range(new_y_bins):
                # Sum the values in the current rebinning block
                if (pull):
                    rebinned_hist[i, j] = hist[i*rbX:(i+1)*rbX, j*rbY:(j+1)*rbY].sum()/(rbX*rbY)
                else:
                    rebinned_hist[i, j] = int(hist[i*rbX:(i+1)*rbX, j*rbY:(j+1)*rbY].sum())

        return rebinned_hist     

def num_factors(nBins):
    nFact = 0
    for div in [2, 3, 4, 5]:
        if (nBins % div) == 0:
            nFact += 1
    return nFact

def pad_histogram(hist):
    hist_new = hist.copy()
    
    # Determine if the histogram is 1D or 2D
    if hist.ndim == 1:
        nBinsX = len(hist_new)
        nBinsY = 1
        hist_new = hist_new.reshape(-1, 1)  # Convert to 2D for uniform handling
    else:
        nBinsX, nBinsY = hist_new.shape

    nFactX = num_factors(nBinsX)
    nFactY = num_factors(nBinsY)
    nPadX = 0
    nPadY = 0

    # Compute padding for the X axis
    iX = 1
    while nFactX < 3 and 10 * iX < nBinsX:
        if num_factors(nBinsX + iX) > nFactX:
            nFactX = num_factors(nBinsX + iX)
            nPadX = iX
        iX += 1
        
    for jX in range(nPadX):
        if (jX % 2) == 0:
            value_edge = hist_new[nBinsX-1, :].reshape(1, -1)
            hist_new = np.concatenate((hist_new, value_edge), axis=0)        
        else:
            value_edge = hist_new[jX-1, :].reshape(1, -1)
            hist_new = np.concatenate((value_edge, hist_new), axis=0)

    # Compute padding for the Y axis
    iY = 1
    while nFactY < 3 and 10 * iY < nBinsY:
        if num_factors(nBinsY + iY) > nFactY:
            nFactY = num_factors(nBinsY + iY)
            nPadY = iY
        iY += 1
        
    for jY in range(nPadY):
        if (jY % 2) == 0:
            value_edge = hist_new[:, nBinsY-1].reshape(-1, 1)
            hist_new = np.concatenate((hist_new, value_edge), axis=1)        
        else:
            value_edge = hist_new[:, jY-1].reshape(-1, 1)
            hist_new = np.concatenate((value_edge, hist_new), axis=1)

    if hist.ndim == 1:
        hist_new = hist_new.flatten()  # Convert back to 1D if the original was 1D

    return hist_new

def substitute_max_bin_with_average(hist):
    """
    Finds the bin with the maximum value in a 2D histogram and substitutes its value 
    with the average value of the surrounding bins.
    
    Parameters:
    hist (numpy.ndarray): 2D histogram (2D numpy array)
    
    Returns:
    numpy.ndarray: Modified 2D histogram with the maximum bin substituted by the average of surrounding bins
    """

    chi_before = np.square(hist).sum()
    chi_after  = chi_before / 2.0  # initializing as half due to the while condition

    # if the difference in chi2 is bigger than 10% of the original chi2, we will repeat the algorithm!
    ii = 0
    max_iterations = int(np.sqrt(np.count_nonzero(hist))/5)

    while 2 * (chi_before - chi_after) / (chi_before + chi_after) > 0.1 and ii < max_iterations:
        if ii != 0:
            chi_before = chi_after

        rows, cols = hist.shape
        max_bin_value = np.max(np.abs(hist))  # max of abs since we have positive and negative pulls
        max_bin_indices = np.unravel_index(np.argmax(np.abs(hist), axis=None), hist.shape)

        max_row, max_col = max_bin_indices

        surrounding_values = []

        # Collect values from the surrounding bins
        for m in range(max_row - 1, max_row + 2):  
            for n in range(max_col - 1, max_col + 2):
                if (0 <= m < rows) and (0 <= n < cols) and not (m == max_row and n == max_col):
                    surrounding_values.append(hist[m, n])

        # Calculate the average of the surrounding bins
        if surrounding_values:
            average_value = np.mean(surrounding_values)
        else:
            average_value = max_bin_value  

        # Substitute the maximum bin's value with the average value
        hist[max_row, max_col] = average_value

        chi_after = np.square(hist).sum() + 1e-3  # adding a small value to avoid division by zero

        ii += 1

    return hist        

