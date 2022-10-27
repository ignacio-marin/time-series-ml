import matplotlib.pyplot as plt


## Walmart plot
def plot_censored(id, dl, column):
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    fig.suptitle(id)
    dl.df[dl.df[column] == id][['date', 'y_0', 'y_0_mean_roll_7']].set_index('date').plot(ax=ax1)
    dl.df[dl.df[column] == id][['date','is_censored']].set_index('date').plot(ax=ax2)
    plt.show()