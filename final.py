print('Use python3 final.py -h to display arguments help.')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Parser is useful to change plot and other parameters.
parser = argparse.ArgumentParser()

# Extension
parser.add_argument('-e', '--extension',
    help = 'Sets the export format of the plot. Defaults to png.',
    choices = ['png', 'jpg', 'pgf', 'pdf', 'svg'],
    default = 'png'
    )
# Resolution (dpi)
parser.add_argument('-r', '--resolution',
    help = 'Sets the resolution of the plot in dpi. Defaults to 300.',
    type = int,
    default = 300
    )
# Output file name
parser.add_argument('-o', '--output',
    help = 'Sets the output file name. Defaults to plot.',
    type = str,
    default = 'plot'
    )
# Title
parser.add_argument('-t', '--title',
    help = 'Sets the title of the plot. Defaults to "Model Output."',
    type = str,
    default = 'Model Output'
    )
# X label
parser.add_argument('-x', '--xlabel',
    help = 'Sets the x label of the plot. Defaults to "x."',
    type = str,
    default = r'x'
    )
# Y label
parser.add_argument('-y', '--ylabel',
    help = 'Sets the y label of the plot. Defaults to "y."',
    type = str,
    default = 'y'
    )
# Data file name
parser.add_argument('-d', '--data',
    help = 'Sets the input data file name. Defaults to "data.csv."',
    type = str,
    default = 'data.csv')
# Scatter plot
parser.add_argument('-s', '--scatter',
    help = 'Sets the plot type to scatter plot.',
    action = 'store_true')
# Caption
parser.add_argument('-c', '--caption',
    help = 'Enables captions with extra information for individual plots.',
    action = 'store_true')
# Analysis (Question 2)
parser.add_argument('-a', '--analysis',
    help = 'Analyze data for Question 2. \
        The outputs directory must be specified with the --data argument. \
        The file name format is assumed to be data_<layer size>_<trial number>.csv. \
        Note that this argument does not work with -t, -x, -y or -c.',
    action='store_true')

args = parser.parse_args()

# ======================
# DEFAULT CONFIGURATIONS
# ======================

file_name = args.output
title = args.title
xlabel = args.xlabel
ylabel = args.ylabel
resolution = args.resolution
scatter = args.scatter

bbox_inches = 'tight'
figsize_inches = (6.4, 4.8)

# Relevant for Question 2 only
n_trials = 10 # Number of trials for each N
N_min = 2 # Minimum layer size.
N_max = 10 # Maximum layer size.

# ===================
# BEAUTIFUL FUNCTIONS
# ===================
def file_exists(name, ext): # file name.ext exists ?
    return os.path.exists(f'{name}.{ext}')
def savefigure(fig, name, ext): # save figure object
    i=1
    while(os.path.exists(f'{name}{i}.{ext}')): i+=1
    fig.savefig(f'{name}{i}.{ext}',
        dpi=resolution,
        bbox_inches=bbox_inches
        )
    print(f'Figure saved: {name}{i}.{ext}')

# If the output format is pgf, we use pdflatex
if args.extension == 'pgf':
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    extension = 'pgf'
else:
    extension = args.extension

# Extract data and further modify plot
if not args.analysis:
    # Data is stored as a 2d dimensional array. The
    # columns corresponds to "x", "y" and "yfit", in
    # this order.
    data = np.genfromtxt(args.data, delimiter=',')
    def get_data(data):
        x = data[1:, 0]
        y = data[1:, 1]
        yfit = data[1:, 2]

        k = data[1, 5]
        mse = data[1, 6]
        t = data[1, 7]
        caption = rf'Number of Iterations: ${int(k)}$. $MSE = {mse:.5e}$. Runtime: ${1000*t:.3f} \ ms$.'

        arrays = {'x': x, 'y':y, 'yfit': yfit, 'caption': caption}
        return arrays
    # Common kwargs to matplotlib.pyplot.scatter and matplotlib.pyplot.plot
    def kwargs(arrays, fig, ax): return {'marker': 'o', 'zorder': 3}
else:
    # Data is stored as a 4 dimensional array. A 2d slice of the first two dimensions
    # represents a single trial. Note there will be many empty (nan) values because they
    # are all different sizes, depending on the layer size.
    datas = np.empty((2*N_max+1, 8, N_max-N_min+1, n_trials))
    for N in range(N_min, N_max+1):
        for i in range(n_trials):
            data = np.genfromtxt(f'{args.data}/data_{N}_{i}.csv', delimiter=',')

            if data.shape[0] < 2*N_max+1:
                data = np.vstack((data, np.full((2*N_max+1-data.shape[0], 8), np.nan)))

            datas[:,:, N-N_min, i] = data
    # Extract the means of the desired quantities.
    def get_data(datas):
        N = np.arange(N_min, N_max+1)
        k = np.mean(datas[1, 5, :, :], axis=1)
        mse = np.mean(datas[1, 6, :, :], axis=1)
        t = 1000*np.mean(datas[1, 7, :, :], axis=1)
        arrays = {'N': N, 'k':k, 'mse':mse, 't':t}
        return arrays
    # Common kwargs to matplotlib.pyplot.scatter and matplotlib.pyplot.plot
    def kwargs(arrays, fig, ax): return {'marker': 'o', 'zorder': 3}

# ==============
# BEAUTIFUL PLOT
# ==============

if __name__ == '__main__' and not args.analysis:
    # fig, ax
    fig, ax = plt.subplots(figsize=figsize_inches)
    fig.set_dpi = resolution
    # Labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, rotation=90, ha='right')
    ax.set_title(title)
    # Ticks
    ax.minorticks_on()
    ax.tick_params(which='both', axis='both', direction='in')

    # Get data
    arrays = get_data(data)

    # Plot
    if scatter:
        ax.scatter(arrays['x'], arrays['y'], c='r', label=r'$y$ (given)', **kwargs(arrays, fig, ax))
        ax.scatter(arrays['x'], arrays['yfit'], c='b', label=r'$\hat{y}$ (model)', **kwargs(arrays, fig, ax))
    else:
        ax.plot(arrays['x'], arrays['y'], c='r', label=r'$y$ (given)', **kwargs(arrays, fig, ax))
        ax.plot(arrays['x'], arrays['yfit'], c='b', label=r'$\hat{y}$ (model)', **kwargs(arrays, fig, ax))
    
    # Legend
    ax.legend()
    
    if args.caption:
        fig.text(0.5, -0.025, arrays['caption'], ha='center')

    # Save
    savefigure(fig, file_name, extension)
elif __name__ == '__main__' and args.analysis:
    # fig, ax
    fig, axs = plt.subplots(3, figsize=figsize_inches)
    fig.set_dpi = resolution
    fig.suptitle('Gradient Descent Characteristics')

    fig.tight_layout()
    
    # Labels
    labels = ['Number of Iterations', r'$MSE$ value', r'Runtime ($ms$)']
    for i in range(3):
        axs[i].set_xlabel(r'$N$ value', fontsize=12)
        axs[i].set_ylabel(labels[i], fontsize=8, rotation=90)
        axs[i].get_yaxis().set_label_coords(-0.12, 0.5)
        # Ticks
        axs[i].minorticks_on()
        axs[i].tick_params(which='both', axis='both', direction='in')
    
    # Get data
    arrays = get_data(datas)

    # Plot
    if scatter:
        axs[0].scatter(arrays['N'], arrays['k'], c='r', label=r'$k$', **kwargs(arrays, fig, axs[0]))
        axs[1].scatter(arrays['N'], arrays['mse'], c='g', label=r'$MSE$', **kwargs(arrays, fig, axs[1]))
        axs[2].scatter(arrays['N'], arrays['t'], c='b', label=r'$t$', **kwargs(arrays, fig, axs[2]))
    else:
        axs[0].plot(arrays['N'], arrays['k'], c='r', label=r'$k$', **kwargs(arrays, fig, axs[0]))
        axs[1].plot(arrays['N'], arrays['mse'], c='g', label=r'$MSE$', **kwargs(arrays, fig, axs[1]))
        axs[2].plot(arrays['N'], arrays['t'], c='b', label=r'$t$', **kwargs(arrays, fig, axs[2]))
    
    # Save
    savefigure(fig, file_name, extension)
