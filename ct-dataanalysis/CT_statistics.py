import numpy as np
import scipy.stats as sp_stats
import matplotlib.pyplot as plt


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

    from scipy.stats import chisquare, chi2
    from sklearn.decomposition import pca
    from math import pi

    for x_line in range(x.shape[1]):
        hist_x, bins_x = histogram(x[:, x_line], False)
        bin_width = bins_x[1] - bins_x[0]

        v = np.zeros((len(bins_x) - 1, 1))

        for i in range(len(hist_x)):
            v[i, 0] = bin_width * 0.5 + bins_x[i]
        y = (v - np.average(x[:, x_line])) / np.std(x[:, x_line])
        pdf_x = np.exp(-y ** 2 / 2) / np.sqrt(2 * pi)

        # Scale pdf to value range of observation dataset
        # Step 1: normalize
        exp_freq_dist = pdf_x / np.sum(pdf_x)
        # Step 2: scale to sum of
        norm_pdf_x = np.sum(hist_x) * exp_freq_dist

        x_plus_pdf = np.zeros((len(hist_x), 2))

        x_plus_pdf[:, 0] = hist_x
        x_plus_pdf[:, 1] = norm_pdf_x[:, 0]

        # chi2_stat, chi2_p = chisquare(hist_x, x_plus_pdf[:, 1])

        chi2calc = np.sum(((x_plus_pdf[:, 0] - x_plus_pdf[:, 1]) ** 2) / x_plus_pdf[:, 1])

        print("Chi2calc", chi2calc)

        # Calculate degrees of freedom:
        # sigma = number of classes - (params for exp freq dist + number variables
        deg_freedom = len(hist_x) - (2 + 1)

        critical_chi2 = chi2.ppf([0.25, 0.5, 0.8, 0.95, 0.999], deg_freedom)
        print("Critical chi2", critical_chi2)
        if critical_chi2[4] < chi2calc:
            print("Unbelievable: Chi2 is above critical value for 99.9% confidence.")
        elif critical_chi2[3] < chi2calc:
            print("Chi2 is above critical value for 95% confidence.")
        elif critical_chi2[2] < chi2calc:
            print("Chi2 is above critical value for 80% confidence.")
        elif critical_chi2[1] < chi2calc:
            print("Chi2 is above critical value for 50% confidence.")
        elif critical_chi2[0] < chi2calc:
            print("WARNING: less than 50 % confidence for Gaussian distribution.")
        else:
            print("WARNING: Gaussian distribution not significant.")

        ''' Optional: display histograms of all input vars '''

        if create_plots:
            plt.bar(v[:, 0], x_plus_pdf[:, 0], width=10)
            plt.plot(v[:, 0], x_plus_pdf[:, 1])
            plt.show()

    ''' Conduct PCA '''

    pca_model = pca.PCA(n_components=x.shape[1])
    pca_model.fit(x)
    X_r = pca_model.transform(x)

    print("pca model", pca_model)
    print("Explained variance ratio", pca_model.explained_variance_ratio_)

    ''' Optional: make plot(s): coefficient matrix, tree '''

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    print("shape X_r", X_r.shape[1])
    print("X_r", X_r)
    for color, i in zip(colors, range(X_r.shape[1] - 1)):
        print("X_r i", X_r[i, 0])
        print("X_r i", X_r[i, 1])
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')

    plt.figure()



