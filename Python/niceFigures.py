import matplotlib.pyplot as plt
import matplotlib

font = {"family": "serif",
        "weight": "normal",
        "size": 15}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (8, 6)
matplotlib.rcParams['figure.dpi'] = 75

# ===
# Nice figures
def makeNiceAxes(ax, xlab=None, ylab=None):
    ax.tick_params(axis='both', reset=False, which='both', length=5, width=2)
    ax.tick_params(axis='y', direction='out', which="both")
    ax.tick_params(axis='x', direction='out', which="both")
    for spine in ['top', 'right','bottom','left']:
        ax.spines[spine].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

def niceFig(xlab=None, ylab=None, figsize=None, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if (nrows == 1) and (ncols == 1):
        makeNiceAxes(ax, xlab, ylab)
    else:
        for axx in ax:
            makeNiceAxes(axx, xlab, ylab)
    return fig, ax

def addColourBar(fig, cs, cbarLabel, pos=[0.85, .25, 0.03, 0.5], orientation="vertical"):
    position = fig.add_axes(pos)
    cbar = fig.colorbar(cs, cax=position, orientation=orientation)
    cbar.set_label(cbarLabel)
    return cbar