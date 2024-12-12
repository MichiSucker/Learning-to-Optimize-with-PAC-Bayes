import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from experiments.helpers import set_size
import numpy as np


def plot_results(loading_path):

    width = 469.75499  # Arxiv
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts quantile_distance little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    plt.rcParams.update(tex_fonts)

    names = {'with': 'with', 'without': 'without', 'other': 'other'}
    colors = {'with': '#FFBE0B', 'without': '#FB5607', 'other': '#613f75'}

    loading_path_ratio = loading_path + '/train_with_ratio_of_losses/data/'
    number_of_iterations_ratio = np.load(loading_path_ratio + 'number_of_iterations_testing.npy')
    number_of_iterations_training_ratio = np.load(loading_path_ratio + 'number_of_iterations_training.npy')
    losses_with_ratio = np.load(loading_path_ratio + 'losses_ratio_train.npy')
    losses_without_ratio = np.load(loading_path_ratio + 'losses_sum_train.npy')
    iterates_ratio = np.arange(number_of_iterations_ratio + 1)

    loading_path_rand = loading_path + '/train_without_randomization/data/'
    number_of_iterations_rand = np.load(loading_path_rand + 'number_of_iterations_testing.npy')
    number_of_iterations_training_rand = np.load(loading_path_rand + 'number_of_iterations_training.npy')
    losses_with_rand = np.load(loading_path_rand + 'losses_with_randomization_train.npy')
    losses_without_rand = np.load(loading_path_rand + 'losses_without_randomization_train.npy')
    iterates_rand = np.arange(number_of_iterations_rand + 1)

    subplots = (2, 4)
    size = set_size(width=width, subplots=subplots)
    fig = plt.figure(figsize=size)
    G = gridspec.GridSpec(subplots[0], subplots[1])

    axes_1 = fig.add_subplot(G[:, 0:2])
    axes_2 = fig.add_subplot(G[:, 2:])

    axes_1.plot(iterates_ratio, np.mean(losses_with_ratio, axis=0),
                linestyle='dashed', label=names['with'], color=colors['with'])
    axes_1.plot(iterates_ratio, np.median(losses_with_ratio, axis=0),
                linestyle='dotted', color=colors['with'])
    axes_1.fill_between(iterates_ratio, np.quantile(losses_with_ratio, q=0, axis=0),
                        np.quantile(losses_with_ratio, q=0.95, axis=0),
                        color=colors['with'], alpha=0.5)

    axes_1.plot(iterates_ratio, np.mean(losses_without_ratio, axis=0),
                linestyle='dashed', label=names['without'], color=colors['without'])
    axes_1.plot(iterates_ratio, np.median(losses_without_ratio, axis=0),
                linestyle='dotted', color=colors['without'])
    axes_1.fill_between(iterates_ratio,
                        np.quantile(losses_without_ratio, q=0, axis=0),
                        np.quantile(losses_without_ratio, q=0.95, axis=0),
                        color=colors['without'], alpha=0.5)

    axes_1.axvline(x=number_of_iterations_training_ratio, ymin=0, ymax=1, linestyle='dashed', alpha=0.5,
                   label='$n_{\\rm{train}}$', color='black')

    axes_1.set_ylabel('$\\ell(x^{(k)})$')
    axes_1.set_xlabel('$n_{\\rm{it}}$')
    axes_1.set_yscale('log')
    axes_1.set_title('Ratio')
    axes_1.legend()
    axes_1.grid('on')

    axes_2.plot(iterates_rand, np.mean(losses_with_rand, axis=0),
                linestyle='dashed', label=names['with'], color=colors['with'])
    axes_2.plot(iterates_rand, np.median(losses_with_rand, axis=0),
                linestyle='dotted', color=colors['with'])
    axes_2.fill_between(iterates_rand, np.quantile(losses_with_rand, q=0, axis=0),
                        np.quantile(losses_with_rand, q=0.95, axis=0),
                        color=colors['with'], alpha=0.5)

    axes_2.plot(iterates_rand, np.mean(losses_without_rand, axis=0),
                linestyle='dashed', label=names['without'], color=colors['without'])
    axes_2.plot(iterates_rand, np.median(losses_without_rand, axis=0),
                linestyle='dotted', color=colors['without'])
    axes_2.fill_between(iterates_rand, np.quantile(losses_without_rand, q=0, axis=0),
                        np.quantile(losses_without_rand, q=0.95, axis=0),
                        color=colors['without'], alpha=0.5)

    axes_2.axvline(x=number_of_iterations_training_rand, ymin=0, ymax=1, linestyle='dashed', alpha=0.5,
                   label='$n_{\\rm{train}}$', color='black')

    axes_2.set_ylabel('$\\ell(x^{(k)})$')
    axes_2.set_xlabel('$n_{\\rm{it}}$')
    axes_2.set_yscale('log')
    axes_2.set_title('Randomization')
    axes_2.legend()
    axes_2.grid('on')

    plt.tight_layout()
    savings_path = loading_path + '/comparison_design_choices/'
    fig.savefig(savings_path + 'comparison_design_choices.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig(savings_path + 'comparison_design_choices.png', dpi=300, bbox_inches='tight', pad_inches=0)
