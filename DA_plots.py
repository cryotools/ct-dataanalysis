"""
Plots to support data analysis.
Plot styles optimized for scientific publication.
"""
import matplotlib.pyplot as plt


class DAPlots:
    """ Plots facilitating data analysis """

    def __init__(self, x):
        self.x = x

    def data_histogram(self, x):
        # Calculate number of bins

        import matplotlib.cm as cm
        import matplotlib.colors as colors

        fig, ax = plt.subplots()
        N, bins, patches = ax.hist(x, 20)

        # I'll color code by height, but you could use any scalar


        # we need to normalize the data to 0..1 for the full
        # range of the colormap
        fracs = N.astype(float) / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())

        for thisfrac, thispatch in zip(fracs, patches):
            color = cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        plt.show()


'''

spread = np.random.rand(50) * 100
        center = np.ones(25) * 50
        flier_high = np.random.rand(10) * 100 + 100
        flier_low = np.random.rand(10) * -100
        data = np.concatenate((x, np.median(x), flier_high, flier_low), 0)

        import matplotlib.pyplot as plt
        #print("boxplot arguments", x, np.median(x), np.percentile(x, 25), np.percentile(x, 75), 0)
        plt.boxplot(data)
        plt.show()

'''