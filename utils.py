import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output

def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',10)) # outward by 10 points
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def visualize(text, z_space_image, z_space_range, z1, z2, labels):
    z1 = np.array(z1)
    z2 = np.array(z2)
    labels = np.array(labels)

    # Set up the axes with gridspec and add subplots
    fig = plt.figure(figsize=(32, 16))
    scale = 5
    n_rows = 3 * scale
    n_cols = 6 * scale
    grid = plt.GridSpec(n_rows, n_cols, hspace=0.125, wspace=0.125)
    text_area = fig.add_subplot(grid[0, :n_cols // 2])
    space_img = fig.add_subplot(grid[1:, :n_cols // 2])
    scatter_ax = fig.add_subplot(grid[1:, n_cols // 2:-1])
    x_hist = fig.add_subplot(grid[0, n_cols // 2:-1], yticklabels=[], sharex=scatter_ax)
    y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=scatter_ax)

    # Removing boxes
    text_area.axis('off')
    x_hist.axis('off')
    y_hist.axis('off')
    
    # Text area
    text_area.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', transform=text_area.transAxes, fontsize=24)
    
    # Z Space image
    min_, max_ = z_space_range
    space_img.imshow(z_space_image, cmap='gray', extent=[min_, max_, min_, max_])
    adjust_spines(space_img, ['left', 'bottom'])
    
    # Scatter
    cmap = plt.cm.get_cmap('Spectral')
    for c in set(labels):
        scatter_ax.scatter(z1[labels == c], z2[labels == c], label=c, c=[cmap(c / 10.)])
        
    box_width = max_ - min_
    p = patches.Rectangle((min_, min_), box_width, box_width, fill=False, clip_on=False, linewidth=5, linestyle='--', color='k')
    scatter_ax.add_patch(p)
        
    adjust_spines(scatter_ax, ['left', 'bottom'])
    scatter_ax.legend(loc=1, ncol=10, mode="expand", borderaxespad=0., prop={'size': 14})

    # Histograms
    x_hist.hist(z1, 40, histtype='stepfilled', orientation='vertical', color='gray')
    x_hist.axhline(y=0, xmin=0, xmax=1, linewidth=5, color='gray')
    
    y_hist.hist(z2, 40, histtype='stepfilled', orientation='horizontal', color='gray')
    y_hist.axvline(x=0, ymin=0, ymax=1, linewidth=5, color='gray')
    
class VisualizeTraining(keras.callbacks.Callback):
    
    def __init__(self, plot_params=({'text':'', 'loss_signature':'', 'constants':{}}, {'text':'', 'loss_signature':'', 'constants':{}})):
        assert 1 <= len(plot_params) < 3, 'Only 1 or 2 plots is allowed.'
        self.plot_params = plot_params
        self.history = {}

    def plot(self, epoch):
        clear_output(wait=True)
        
        fig = plt.figure(figsize=(32, 8))
        scale = 5
        n_rows = 3 * scale
        n_cols = 5 * scale
        grid = plt.GridSpec(n_rows, n_cols, hspace=0.125, wspace=3)
              
        n_plots = len(self.plot_params)
        n_cols_per_plot = n_cols // n_plots
        for i, p in enumerate(self.plot_params):
            text_area = fig.add_subplot(grid[0, n_cols_per_plot * i:n_cols_per_plot * (i+1)])
            text_area.axis('off')
            text_area.text(0.5, 0.5, p['text'], horizontalalignment='center', verticalalignment='center', transform=text_area.transAxes, fontsize=24)
                 
            plot = fig.add_subplot(grid[1:, n_cols_per_plot * i:n_cols_per_plot * (i+1)])
            plot.grid(True)
            for key in self.history.keys():
                if p['loss_signature'] in key:
                    plot.plot(np.arange(epoch+1), self.history[key], label=key)

            for key, value in p['constants'].items():
                plot.axhline(y=value, xmin=0, xmax=1, label=key, linewidth=3, linestyle='--', color='k', alpha=0.5)
            plot.legend(prop={'size': 16})

    def on_epoch_end(self, epoch, logs={}):
        for key, value in logs.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]
                
        self.plot(epoch)
        plt.show()