import DA_statistics as stats
import DA_plots as plots
import numpy as np
import pandas as pd

# df = pd.read_csv('example_data/organic_matter.txt', header=None)
df = pd.read_csv('example_data/sodium_content.txt', header=None)
input_data = np.array(df).reshape(1, len(df))[0]

print("Input data:", input_data)

stats_object = stats.DAStatistics(input_data)
summary = stats_object.summary_univar(input_data)

print("Summary statistics:")
for measure in summary:
    print("{title}: {value}".format(title=measure[0], value=measure[1]))
# print(summary.summary_univar(df))

plot = plots.DAPlots(input_data)
plot.data_histogram(input_data)


