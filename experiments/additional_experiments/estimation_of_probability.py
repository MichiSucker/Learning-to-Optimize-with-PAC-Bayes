import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import beta
import numpy as np
from experiments.helpers import set_size
from pathlib import Path


def get_bernoulli_sample(p):
    return int(np.random.uniform(low=0.0, high=1.0) <= p)


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/estimation_probability/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def bayesian_estimation_of_probability(path_of_experiment):

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

    savings_path = create_folder_for_storing_data(path_of_experiment)

    # Specify colorscheme
    color_density = 'black'
    color_true_probability = '#EA738D'
    color_lines = '#89ABE3'

    # Specify accuracy parameters
    quantile_distance = 0.075
    lower_quantile, upper_quantile = 0.05, 0.95

    # Specify underlying probability
    true_p = 0.8

    # Setup prior based on parameters quantile_distance and b
    a, b = 1, 1
    prior = beta(a=a, b=b)
    cur_upper_quantile, cur_lower_quantile = prior.ppf(upper_quantile), prior.ppf(lower_quantile)

    # Count number of iterates needed
    distributions = [(a, b)]
    quantiles = [(cur_lower_quantile, cur_upper_quantile)]
    n_iterates = 0
    while cur_upper_quantile - cur_lower_quantile > quantile_distance:

        # Increase counter
        n_iterates += 1

        # Get new sample (corresponds to evaluating whether the loss did decrease sufficiently)
        s = get_bernoulli_sample(p=true_p)
        a = a + s
        b = b + (1 - s)

        # Update the bounds
        posterior = beta(a=a, b=b)
        cur_upper_quantile = posterior.ppf(upper_quantile)
        cur_lower_quantile = posterior.ppf(lower_quantile)

        # Store the new estimate
        distributions.append((a, b))
        quantiles.append((cur_lower_quantile, cur_upper_quantile))

    # Create the plot. which shows the evolution of the posterior
    xes = np.linspace(start=0, stop=1, num=1000)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=set_size(width, subplots=(1, 3)), sharex='all', sharey='all')

    indices = [0, 30, 60]
    for i, idx in enumerate(indices):

        f_xes = beta.pdf(a=distributions[idx][0], b=distributions[idx][1], x=xes)
        ax[i].plot(xes, f_xes, color=color_density, label='$f_{a,b}(p)$')
        ax[i].axvline(true_p, 0, 1, color=color_true_probability, linestyle='dotted', label='p')
        ax[i].annotate(text='', xy=(quantiles[idx][0], 2), xytext=(quantiles[idx][1], 2),
                       arrowprops=dict(arrowstyle='<->', color=color_lines))
        ax[i].axvline(quantiles[idx][0], 0, 1, color=color_lines, linestyle='dashed', label='$q_l, q_u$')
        ax[i].axvline(quantiles[idx][1], 0, 1, color=color_lines, linestyle='dashed')

        ax[i].grid('on')
        ax[i].set(title=f'n = {idx}', xlabel='$p$')
        ax[i].legend()
    ax[0].set_ylabel('$f_{a,b}(p)$')
    plt.tight_layout()
    fig.savefig(fname=savings_path + 'bayesian_estimation.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)
