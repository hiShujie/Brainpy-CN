import numpy as np
import matplotlib.pyplot as plt

def barweb(barvalues, errors, width=1, groupnames=None, bw_title=None, bw_xlabel=None, bw_ylabel=None, bw_colormap='jet', gridstatus='none', bw_legend=None):
    if barvalues.shape != errors.shape:
        raise ValueError("barvalues and errors matrix must be of the same dimension")

    numgroups, numbars = barvalues.shape

    fig, ax = plt.subplots()
    bar_width = width * min(0.8, numbars / (numbars + 1.5))
    x = np.arange(numgroups)
    
    for i in range(numbars):
        bar_x = x - (numbars - 1) * bar_width / 2 + i * bar_width
        ax.bar(bar_x, barvalues[:, i], width=bar_width, color=plt.cm.get_cmap(bw_colormap)(i / numbars))
        ax.errorbar(bar_x, barvalues[:, i], yerr=errors[:, i], fmt='k', linestyle='none')

    ax.set_facecolor('white')
    if bw_title:
        ax.set_title(bw_title, fontsize=14)
    if bw_xlabel:
        ax.set_xlabel(bw_xlabel, fontsize=12)
    if bw_ylabel:
        ax.set_ylabel(bw_ylabel, fontsize=12)
    if groupnames is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(groupnames)
    ax.tick_params(axis='x', labelsize=12)
    ax.box(False)

    if gridstatus in ['x', 'xy']:
        ax.xaxis.grid(True)
    if gridstatus in ['y', 'xy']:
        ax.yaxis.grid(True)

    if bw_legend:
        ax.legend(bw_legend, fontsize=12, loc='best')
        ax.get_legend().set_frame_on(False)

    plt.show()

    return fig, ax
