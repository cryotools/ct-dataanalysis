import CT_statistics as stats
import CT_plots as plots
import numpy as np
import pandas as pd


# Create Gaussian distributed sample sets
s1 = np.random.random(300)   # (0, 10, 30)
s2 = np.random.random(300)   # np.random.normal(0, 7, 30)
s3 = np.random.normal(0, 30, 300)

# Make an array and add weighted samples to it
x = np.zeros((300, 4))
x[:, 0] = 15.4 + 7.2 * s1 + 10.5 * s2 + 2.5 * s3
x[:, 1] = 124.0 - 8.7 * s1 + 0.1 * s2 + 2.6 * s3
x[:, 2] = 100.0 + 5.25 * s1 - 6.5 * s2 + 3.5 * s3
x[:, 3] = 78.0 - 1.2 * s1 - 4.3 * s2 - 5.9 * s3
'''x[:, 4] = 30.0 + 1.25 * s1 + 2.5 * s2 + 10.5 * s3
x[:, 5] = 65.0 - 1.25 * s1 + 2.5 * s2 + 10.5 * s3
x[:, 6] = 21.0 + 1.25 * s1 - 2.5 * s2 + 10.5 * s3
x[:, 7] = 15.4 + 7.2 * s1 + 10.5 * s2 + 2.5 * s3
x[:, 8] = 124.0 - 8.7 * s1 + 0.1 * s2 + 2.6 * s3
x[:, 9] = 100.0 + 5.25 * s1 - 6.5 * s2 + 3.5 * s3
x[:, 10] = 78.0 - 1.2 * s1 - 4.3 * s2 - 5.9 * s3
x[:, 11] = 30.0 + 1.25 * s1 + 2.5 * s2 + 10.5 * s3
x[:, 12] = 65.0 - 1.25 * s1 + 2.5 * s2 + 10.5 * s3
x[:, 13] = 21.0 + 1.25 * s1 - 2.5 * s2 + 10.5 * s3
x[:, 14] = 30.0 + 1.25 * s1 + 2.5 * s2 + 10.5 * s3
x[:, 15] = 124.0 - 8.7 * s1 + 0.1 * s2 + 2.6 * s3
x[:, 16] = 45.4 + 7.2 * s1 - 10.5 * s2 - 2.5 * s3
'''

# Add noise
x += 3.8 * np.random.normal(0, 1, np.shape(x))


from scipy.stats import chisquare, chi2
from math import pi

# plots.plot_histograms(x)
hist_x, bins_x = stats.histogram(x[:, 1], False)
print("hist", hist_x, "bins", bins_x)

bin_width = bins_x[1] - bins_x[0]
print("bin width", bin_width)

v = np.zeros((len(bins_x) - 1, 1))
print("v zeros", v, "shape v zeros", v.shape)

for i in range(len(hist_x)):
    print("new v is", bin_width * 0.5 + bins_x[i])
    v[i, 0] = bin_width * 0.5 + bins_x[i]
print("Mean:", np.average(x[:, 1]), "STD:", np.std(x[:, 1]))
y = (v - np.average(x[:, 1])) / np.std(x[:, 1])
pdf_x = np.exp(-y**2/2) / np.sqrt(2 * pi)
# norm_pdf_x = spstats.norm.pdf(v, np.average(x[:, 1]), np.std(x[:, 1]))
# stats.multivar_pca(hist_x, norm_pdf_x)

# Scale pdf to value range of observation dataset
# Step 1: normalize
exp_freq_dist = pdf_x / np.sum(pdf_x)
print("Expected freq dist", exp_freq_dist)
# Step 2: scale to sum of
norm_pdf_x = np.sum(hist_x) * exp_freq_dist
print("Scaled exp freq dist", norm_pdf_x)

print("v", v, "norm_pdf", norm_pdf_x)
x_plus_pdf = np.zeros((len(hist_x), 2))

x_plus_pdf[:, 0] = hist_x
x_plus_pdf[:, 1] = norm_pdf_x[:, 0]

print("x + pdf", x_plus_pdf)

chi2_stat, chi2_p = chisquare(hist_x, x_plus_pdf[:, 1])

print("chi2 * 100", chi2_p * 100)
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

import matplotlib.pyplot as plt
plt.bar(v[:, 0], x_plus_pdf[:, 0], width=10)
plt.plot(v[:, 0], x_plus_pdf[:, 1])
plt.show()



def univariate_demo():
    # df = pd.read_csv('example_data/organic_matter.txt', header=None)
    df = pd.read_csv('example_data/sodium_content.txt', header=None)
    input_data = np.array(df).reshape(1, len(df))[0]

    print("Input data:", input_data)

    summary = stats.summary_univar(input_data)

    print("Summary statistics:")
    for measure in summary:
        print("{title}: {value}".format(title=measure[0], value=measure[1]))
    # print(summary.summary_univar(df))

    plots.data_histogram(input_data)


