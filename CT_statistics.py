import numpy as np
import scipy.stats as sp_stats


def histogram(x, allow_emptybins=False):
    """
    Determine a suitable number of bins and create a histogram.

    Input:
    x -> 1D numpy array
    allow_emptybins -> Boolean, set to true to allow the result contain empty bins

    Output:
    hist -> Number of values per bin
    bins -> Bin edges
    """

    # Initial bin number setup
    number_bins = int(np.round(np.sqrt(len(x))))

    if allow_emptybins:
        # Create the histogram without checking for empty bins
        hist, bins = np.histogram(x, number_bins)
    else:
        # Check for empty bins, reduce bin count until only filled bins remain.
        check_nonzero_bins = False
        while not check_nonzero_bins:
            hist, bins = np.histogram(x, number_bins)
            if np.isin(0, np.min(hist)):
                number_bins -= 1
            else:
                check_nonzero_bins = True

    return hist, bins


def summary_univar(x, create_plots=True):
    """
    Return univariate characterization.
    Input:
    x   -> 1D Numpy array
    Output:
    A dictionary of summary statistics
    """

    summary = []

    ''' Basic characterization '''
    summary.append(['Value count', len(x)])
    summary.append(['Minimum', np.nanmin(x)])
    summary.append(['Maximum', np.nanmax(x)])
    value_range = np.nanmax(x) - np.nanmin(x)
    summary.append(['Range', value_range])

    ''' Measures of Central tendency: 
        - arithmetic mean, 
        - median, 
        - mode, -
        - quartiles, and quintiles. '''

    summary.append(['Mean', np.average(x)])
    summary.append(['Median', np.median(x)])

    histogram_x = histogram(x)
    summary.append(['Historgram', histogram_x])
    # print("ISIN result: ", np.isin(histogram_x[0], np.max(histogram_x[0])))

    def mode_from_hist(input_histogram):
        mode_array = np.isin(input_histogram[0], np.max(input_histogram[0]))
        for i in range(len(input_histogram[1])):
            # print("Round ", i, "mode_array value", mode_array[i], " - edge: ", histogram[1][i])
            if mode_array[i]:
                return input_histogram[1][i] + 0.5 * (input_histogram[1][i + 1] - input_histogram[1][i])

    mode_x = mode_from_hist(histogram_x)
    summary.append(['Mode', mode_x])

    # bin_center_min = np.nanmin(x) + 0.5 * value_range / number_bins
    # bin_center_max = np.nanmax(x) - 0.5 * value_range / number_bins
    # bin_width = value_range / number_bins
    # bin_centers = np.arange(bin_center_min, bin_center_max, bin_width)

    # summary.append(['Mode', np.nanmax(np.percentile(x, np.arange(0, 100, 25))])
    # summary.append(['Mode', np.mode(x)])

    summary.append(['Quartiles', np.percentile(x, np.arange(0, 100, 25))])
    summary.append(['Quintiles', np.percentile(x, np.arange(0, 100, 20))])

    # harmonic_mean_x = statistics.harmonic_mean(x)
    # quartile_x = statistics.median_grouped()

    ''' Measures of Dispersion: 
        - Standard deviation,
        - Variance,
        - Skewness,
        - Kurtosis. '''

    summary.append(['Variance', np.var(x)])
    summary.append(['Standard deviation', np.std(x)])
    summary.append(['Skewness', sp_stats.skew(x)])
    summary.append(['Kurtosis', sp_stats.kurtosis(x)])

    ''' Theoretical distributions:
        - Probability density function,
        - Cumulative density function. '''

    return summary


def multivar_pca(x, create_plots=True):
    """ Conduct Principal Component Analysis """
    ''' Validate Gaussian distribution of input vars using chi² tests '''
    from scipy.stats import chisquare
    print(x)
    print(chisquare(x))

    ''' Optional: display histograms of all input vars '''

    ''' Conduct PCA '''

    ''' Optional: make plot(s): coefficient matrix, tree '''

    pass



