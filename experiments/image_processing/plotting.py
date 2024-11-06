import pickle
import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import set_size
from scipy.stats import beta


def create_evaluation_plots(loading_path, path_of_experiment):

    width = 469.75499  # Arxiv
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts quantile_distance little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7
    }
    plt.rcParams.update(tex_fonts)

    names = {'std': 'NAG', 'pac': 'Learned', 'other': 'other'}
    colors = {'std': '#C0002B', 'pac': '#00D3FF', 'other': '#006a78'}

    losses_of_baseline_algorithm = np.load(loading_path + 'losses_of_baseline_algorithm.npy')
    losses_of_learned_algorithm = np.load(loading_path + 'losses_of_learned_algorithm.npy')
    n_train = np.load(loading_path + 'number_of_iterations.npy')
    pac_bound = np.load(loading_path + 'pac_bound.npy')
    empirical_probability = np.load(loading_path + 'empirical_probability.npy')
    number_of_iterations_for_approximation = np.load(loading_path + 'number_of_iterations_for_approximation.npy')
    assert number_of_iterations_for_approximation == 1000
    approximate_optimal_losses = np.load(loading_path + 'approximate_optimal_losses.npy')

    with open(loading_path + 'times_of_learned_algorithm', 'rb') as file:
        times_of_learned_algorithm = pickle.load(file)

    with open(loading_path + 'times_of_baseline_algorithm', 'rb') as file:
        times_of_baseline_algorithm = pickle.load(file)

    with open(loading_path + 'parameters_of_estimation', 'rb') as file:
        parameters_of_estimation = pickle.load(file)

    levels_of_accuracy = list(times_of_baseline_algorithm.keys())
    linestyles = ['solid', 'dashed', 'dotted']
    assert len(levels_of_accuracy) == len(linestyles)

    subplots = (2, 2)
    fig, ax = plt.subplots(subplots[0], subplots[1], figsize=set_size(width=width, subplots=subplots))

    ####################################################################################################################
    # Plot loss over Iterations
    ####################################################################################################################

    # Compute mean and median for learned and standard losses
    mean_std, mean_pac = np.mean(losses_of_baseline_algorithm, axis=0), np.mean(losses_of_learned_algorithm, axis=0)
    median_std, median_pac = np.median(losses_of_baseline_algorithm, axis=0), np.median(losses_of_learned_algorithm,
                                                                                        axis=0)
    iterations = np.arange(0, losses_of_baseline_algorithm.shape[1])

    # Plot standard losses
    ax[0, 0].plot(iterations, mean_std, color=colors['std'], linestyle='dashed', label=names['std'])
    ax[0, 0].plot(iterations, median_std, color=colors['std'], linestyle='dotted')
    ax[0, 0].fill_between(iterations, np.quantile(losses_of_baseline_algorithm, q=0.1, axis=0),
                          np.quantile(losses_of_baseline_algorithm, q=0.9, axis=0),
                          color=colors['std'], alpha=0.5)

    # Plot pac losses
    ax[0, 0].plot(iterations, mean_pac, color=colors['pac'], linestyle='dashed', label=names['pac'])
    ax[0, 0].plot(iterations, median_pac, color=colors['pac'], linestyle='dotted')
    ax[0, 0].fill_between(iterations, np.quantile(losses_of_learned_algorithm, q=0.1, axis=0),
                          np.quantile(losses_of_learned_algorithm, q=0.9, axis=0),
                          color=colors['pac'], alpha=0.5)

    # Compute also the same statistics for the losses at the optimum
    mean_opt = np.mean(approximate_optimal_losses)
    median_opt = np.median(approximate_optimal_losses)

    # Plot optimal loss
    ax[0, 0].axhline(mean_opt, 0, 1, color=colors['other'], linestyle='dashed',
                     label='$\ell(x_{\\rm{NAG}}^{(1000)}, \\theta)$')
    ax[0, 0].axhline(median_opt, 0, 1, color=colors['other'], linestyle='dotted')

    # Highlight the number of iterations the algorithm was trained for
    ax[0, 0].axvline(n_train, 0, 1, color=colors['pac'], linestyle='dashdot', alpha=0.5, label='$n_{\\rm{train}}$')

    # Finalize plot for loss over iterations
    ax[0, 0].set(title=f'Loss over Iterations', xlabel='$n_{\\rm{it}}$', ylabel='$\ell(x^{(i)}, \\theta)$')
    ax[0, 0].legend()
    ax[0, 0].grid('on')
    ax[0, 0].set_yscale('log')

    ####################################################################################################################
    # Plot loss histogram
    ####################################################################################################################

    # Note that the loss, for which the PAC-bound holds, is stored in the (n+1)th-column.
    bins_pac = np.logspace(np.log10(np.min(losses_of_learned_algorithm[:, n_train + 1])),
                           np.log10(np.max(losses_of_learned_algorithm[:, n_train + 1])), 25)
    ax[1, 0].hist(losses_of_learned_algorithm[:, n_train + 1], color=colors['pac'], label=names['pac'],
                  alpha=0.5, bins=bins_pac, edgecolor='black')
    bins_std = np.logspace(np.log10(np.min(losses_of_baseline_algorithm[:, n_train + 1])),
                           np.log10(np.max(losses_of_baseline_algorithm[:, n_train + 1])), 25)
    ax[1, 0].hist(losses_of_baseline_algorithm[:, n_train + 1], color=colors['std'], label=names['std'],
                  alpha=0.5, bins=bins_std, edgecolor='black')

    ax[1, 0].axvline(np.mean(losses_of_learned_algorithm[:, n_train + 1]), color=colors['pac'], linestyle='dashed',
                     alpha=0.5)
    ax[1, 0].axvline(np.median(losses_of_learned_algorithm[:, n_train + 1]), color=colors['pac'], linestyle='dotted',
                     alpha=0.5)
    ax[1, 0].axvline(np.mean(losses_of_baseline_algorithm[:, n_train + 1]), color=colors['std'], linestyle='dashed',
                     alpha=0.5)
    ax[1, 0].axvline(np.median(losses_of_baseline_algorithm[:, n_train + 1]), color=colors['std'], linestyle='dotted',
                     alpha=0.5)

    ax[1, 0].axvline(pac_bound, color='#ffbf4e', linestyle='dashed', label='PAC-Bound')

    ax[1, 0].grid('on')
    ax[1, 0].legend()
    ax[1, 0].set(title='Loss Histogram', xlabel='$\ell(x^{(n_{\\rm{max}})})$', xscale='log')

    ####################################################################################################################
    # Plot bilevel performance
    ####################################################################################################################
    assert len(linestyles) == len(levels_of_accuracy)

    for epsilon, linestyle in zip(levels_of_accuracy, linestyles):

        # Compute the cumulative time
        total_time_pac = np.cumsum(times_of_learned_algorithm[epsilon])
        total_time_std = np.cumsum(times_of_baseline_algorithm[epsilon])

        # Plot
        ax[0, 1].plot(total_time_pac, color=colors['pac'], linestyle=linestyle, marker='o', ms=3., markevery=10)
        ax[0, 1].plot(total_time_std, color=colors['std'], linestyle=linestyle, marker='x', ms=3., markevery=10,
                      label=f"{epsilon:.2E}")

    ax[0, 1].legend()
    ax[0, 1].grid('on')
    title = 'Cumulative Time'
    ax[0, 1].set(xlabel='$n_{\\rm{problem}}$', ylabel='t [s]', title=title)

    ####################################################################################################################
    # Plot sublevel probability
    ####################################################################################################################

    p_l, p_u = parameters_of_estimation['probabilities']
    N = losses_of_learned_algorithm.shape[0]

    # Count number of successes (this can be done like this, as the empirical conv. prob. is calculated as success/N)
    success = empirical_probability * N
    failure = N - success

    # a and b for posterior regarding an uninformative prior
    a = 1 + success
    b = 1 + failure

    # Corresponding beta distribution
    rv = beta(a, b)
    x = np.linspace(0, 1, 1000, endpoint=True)
    pdf = rv.pdf(x)

    # 'Bayesian confidence'
    mass_inside = rv.cdf(p_u) - rv.cdf(p_l)
    height = 0.5 * np.max(pdf)

    if p_l > 0.0:
        ax[1, 1].plot([0, p_l], [height, height], color='#ff4545')

    if p_u < 1.0:
        ax[1, 1].plot([p_u, 1], [height, height], color='#ff4545')

    ax[1, 1].plot([p_l, p_u], [height, height], color='#09bc8a')
    ax[1, 1].plot(x, pdf, color='orange', alpha=0.5)
    ax[1, 1].axvline(p_l, 0, 1, color='black', alpha=0.5, linestyle='dotted', label='$p_l, p_u$')
    ax[1, 1].axvline(p_u, 0, 1, color='black', alpha=0.5, linestyle='dotted')
    fill_x = np.linspace(p_l, p_u, 100, endpoint=True)
    ax[1, 1].fill_between(fill_x, rv.pdf(fill_x), color='orange', alpha=0.3,
                          label=f'$F(p_u) - F(p_l)$ = {mass_inside:.4f}')
    ax[1, 1].axvline(empirical_probability, 0, 1, color='orange', linestyle='dashed',
                     label=f'$p(\\alpha)$ = {100 * empirical_probability:.1f} \%')
    ax[1, 1].set(title='Sublevel Prob.', xlabel='p')
    ax[1, 1].set_xticks(np.linspace(0.0, 1.0, num=11, endpoint=True))
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xticks(np.linspace(0.0, 1.0, num=21, endpoint=True), minor=True)
    ax[1, 1].grid('on')
    ax[1, 1].legend()

    plt.tight_layout()
    fig.savefig(path_of_experiment + 'evaluation_plot.pdf', dpi=300, bbox_inches='tight')
