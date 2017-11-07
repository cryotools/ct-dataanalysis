import CT_plots as plots
import CT_statistics as stats
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

#print("X shape:", x.shape[1])
#for i in range(x.shape[1]):
stats.multivar_pca(x, True)


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


