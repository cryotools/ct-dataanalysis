import numpy as np
import scipy.stats as sp_stats


class DAStatistics:
    """
    Tools for statistical analyzis of input data.
    """
    def __init__(self, x):
        self.x = x

    def summary_univar(self, x, create_plots=True):
        """
        Return univariate characterization.
        Input:
        x   -> 1D Numpy array
        Output:
        A dictionary of summary stat
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

        number_bins = int(np.round(np.sqrt(len(x))))
        histogram_x = np.histogram(x, number_bins)
        summary.append(['Historgram', histogram_x])
        print("ISIN result: ", np.isin(histogram_x[0], np.max(histogram_x[0])))

        def mode_from_hist(histogram):
            mode_array = np.isin(histogram[0], np.max(histogram[0]))
            for i in range(len(histogram[1])):
                print("Round ", i, "mode_array value", mode_array[i], " - edge: ", histogram[1][i])
                if mode_array[i]:
                    return histogram[1][i] + 0.5 * (histogram[1][i + 1] - histogram[1][i])

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



