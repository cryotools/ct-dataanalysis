"""
Plots to support data analysis.
Plot styles optimized for scientific publication.
"""
import matplotlib.pyplot as plt
plt.style.use('./CT_style.mplstyle')


def plot_histograms(x, share_axis=True):
    """ Plot histogram(s) """
    import numpy as np
    from CT_statistics import histogram

    def prepare_histogram_plot(x):
        p_hist, p_bins = histogram(x)
        p_width = 0.9 * (p_bins[1] - p_bins[0])
        p_center = (p_bins[:-1] + p_bins[1:]) / 2
        return p_hist, p_bins, p_width, p_center

    if x.shape[1] == 1:
        hist, bins, width, center = prepare_histogram_plot(x)
        plt.bar(center, hist, align='center', width=width)
    elif x.shape[1] == 2:
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=share_axis)
        hist1, bins1, width1, center1 = prepare_histogram_plot(x[:, 0])
        hist2, bins2, width2, center2 = prepare_histogram_plot(x[:, 1])
        ax1.bar(center1, hist1, align='center', width=width1)
        ax2.bar(center2, hist2, align='center', width=width2)
    elif x.shape[1] == 3:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=share_axis)
        hist1, bins1, width1, center1 = prepare_histogram_plot(x[:, 0])
        hist2, bins2, width2, center2 = prepare_histogram_plot(x[:, 1])
        hist3, bins3, width3, center3 = prepare_histogram_plot(x[:, 2])
        ax1.bar(center1, hist1, align='center', width=width1)
        ax2.bar(center2, hist2, align='center', width=width2)
        ax3.bar(center3, hist3, align='center', width=width3)
    else:
        # num_cols = int(np.round(x.shape[1] / 2))
        num_cols = int(np.sqrt(x.shape[1]))

        if np.sqrt(x.shape[1]) % 2 == 0:
            num_rows = num_cols
        else:
            if x.shape[1] % 2 == 0:
                j = int(np.round(x.shape[1] / 2))
                for i in range(j):
                    if (x.shape[1] / num_cols) % 2 != 0:
                        num_cols += 1
                    else:
                        break
            num_rows = int(np.round(x.shape[1] / num_cols))
            if num_cols * num_rows < x.shape[1]:
                num_cols += 1


        # Configure subplot matrix
        f, axarr = plt.subplots(num_rows, num_cols, sharex=share_axis, sharey=share_axis)

        i = 0
        for row in range(num_rows):
            for col in range(num_cols):
                if i <= x.shape[1] - 1:
                    hist, bins, width, center = prepare_histogram_plot(x[:, i])
                    axarr[row, col].bar(center, hist, align='center', width=width)
                i += 1

    plt.tight_layout(pad=0.8, w_pad=0.2, h_pad=0.2)
    plt.show()




    '''
    axarr[0, 0].set_title('Axis [0,0]')
    axarr[0, 1].scatter(x, y)
    axarr[0, 1].set_title('Axis [0,1]')
    axarr[1, 0].plot(x, y ** 2)
    axarr[1, 0].set_title('Axis [1,0]')
    axarr[1, 1].scatter(x, y ** 2)
    axarr[1, 1].set_title('Axis [1,1]')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    '''

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