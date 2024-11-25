import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.gridspec as gridspec
import torch
from sympy import false

from experiments.helpers import set_size
from pathlib import Path
from classes.LossFunction.class_LossFunction import LossFunction
from scipy.stats import beta
from tqdm import tqdm


def estimate_probability(x, y):
    # Setup non-informative prior
    a, b = 1, 1
    prior = beta(a=a, b=b)
    upper_quantile = 0.99
    lower_quantile = 0.01
    quantile_distance = 0.05
    current_upper_quantile, current_lower_quantile = prior.ppf(upper_quantile), prior.ppf(lower_quantile)

    # Get true probability for the given point.
    # Run estimation
    true_p = true_probability_scalar(x, y)
    while current_upper_quantile - current_lower_quantile > quantile_distance:

        n = 50
        sample = np.random.uniform(low=0, high=1, size=n)  # To speed-up for the plot, do batches of n.
        result = np.sum(sample <= true_p)

        a += result
        b += n - result
        posterior = beta(a=a, b=b)
        current_upper_quantile = posterior.ppf(upper_quantile)
        current_lower_quantile = posterior.ppf(lower_quantile)

    # Compute posterior mean (closed form, since still Beta distribution)
    posterior_mean = a / (a + b)
    return posterior_mean


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/probabilistically_constrained_sampling_2d/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def true_probability_array(x, y):
    probabilities = np.maximum(0., np.minimum(1., 1 - (np.cos(2 * x) + np.sin(y+0.5))))
    probabilities[np.abs(y) >= 1.5] = 0.
    probabilities[np.abs(x) >= 1.5] = 0.
    probabilities[y >= 2 * x] = 0.
    return probabilities


def true_probability_scalar(x, y):
    probabilities = np.maximum(0., np.minimum(1., 1 - (np.cos(2 * x) + np.sin(y+0.5))))
    if (np.abs(y) >= 1.5) or (np.abs(x) >= 1.5) or (y >= 2*x):
        return 0
    return probabilities


def phi(x):
    return 0.5 * (x[0] + 0.5*x[1] + 0.5*x[0]*x[1]) ** 2


def run_gld(x, number_of_steps):
    samples = [x]
    rejected = []
    x = torch.tensor(x)
    step_size = 5e-3
    dist = torch.distributions.MultivariateNormal(torch.zeros((2,)), torch.eye(2))
    potential = LossFunction(function=phi)
    for k in tqdm(range(number_of_steps)):
        x_tilde = x - 0.5 * step_size * potential.compute_gradient(x) + step_size**0.5 * dist.sample()
        if estimate_probability(np.array(x_tilde[0]), np.array(x_tilde[1])) > 0.6:
            samples.append(x_tilde.numpy())
            x = x_tilde.clone()
        else:
            rejected.append(x_tilde.numpy())
    return np.array(samples), np.array(rejected)


def constrained_potential(x, y):
    prob = true_probability_array(x, y)
    potential = np.exp(-0.5 * (x + 0.5*y + 0.5*x*y) ** 2)
    potential[prob < 0.6] = 0.
    return potential


def create_samples(path: str):
    savings_path = create_folder_for_storing_data(path_of_experiment=path)
    x = np.array([1., -1.])
    samples, rejected = run_gld(x=x, number_of_steps=100000)
    np.save(savings_path + 'samples', samples)
    np.save(savings_path + 'rejected', rejected)


def load_samples(path: str):
    samples = np.load(path + 'samples.npy')
    rejected = np.load(path + 'rejected.npy')
    return samples, rejected


def create_plot(path):

    width = 469.75499  # Arxiv
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts quantile_distance little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    }
    plt.rcParams.update(tex_fonts)

    subplots = (4, 4)
    size = set_size(width=width, subplots=subplots)
    fig = plt.figure(figsize=size)
    G = gridspec.GridSpec(subplots[0], subplots[1])

    axes_1 = fig.add_subplot(G[0:2, 0:2])
    axes_2 = fig.add_subplot(G[0:2, 2:])
    axes_3 = fig.add_subplot(G[2:, 0:2])
    axes_4 = fig.add_subplot(G[2:, 2:])

    n_points = 500
    x, y = np.linspace(-2, 2, n_points), np.linspace(-2, 2, n_points)
    xx, yy = np.meshgrid(x, y)
    zz = true_probability_array(xx.copy(), yy.copy())

    my_cmap = matplotlib.cm.get_cmap('YlGn')
    my_cmap.set_bad('white')
    zz = np.ma.masked_equal(zz, 0)
    axes_1.contourf(xx, yy, zz, alpha=0.75, cmap=my_cmap)
    axes_1.set_title('Underlying Probability $p$')
    axes_1.grid('on')

    zz_2 = constrained_potential(xx, yy)
    my_cmap = matplotlib.cm.get_cmap('OrRd')
    my_cmap.set_bad('white')
    zz_2 = np.ma.masked_equal(zz_2, 0)
    axes_2.contourf(xx, yy, zz_2, alpha=0.75, cmap=my_cmap)
    axes_2.set_title('Constrained Potential ($p \ge 0.6$)')
    axes_2.grid('on')

    savings_path = create_folder_for_storing_data(path_of_experiment=path)
    try:
        samples, rejected = load_samples(savings_path)
    except FileNotFoundError:
        create_samples(path)
        samples, rejected = load_samples(savings_path)

    false_positive = samples[true_probability_array(samples[:, 0], samples[:, 1]) < 0.6]
    false_negative = rejected[true_probability_array(rejected[:, 0], rejected[:, 1]) >= 0.6]

    axes_3.scatter(samples[:, 0], samples[:, 1], color='#2B2D42', s=1, alpha=0.25, label='accepted')
    axes_3.scatter(rejected[:, 0], rejected[:, 1], color='#8D99AE', s=1, alpha=0.25, label='rejected')
    axes_3.scatter(false_positive[:, 0], false_positive[:, 1], color='#EF233C', s=1, alpha=0.75, label='false positive')
    axes_3.scatter(false_negative[:, 0], false_negative[:, 1], color='orange', s=1, alpha=0.75, label='false negative')

    axes_3.set_title(f'Accepted/Rejected = {len(samples)/len(rejected):.1f}')
    axes_3.contourf(xx, yy, zz_2, alpha=0., cmap=my_cmap)
    axes_3.grid('on')
    axes_3.legend()

    my_cmap = matplotlib.cm.get_cmap('Blues')
    axes_4.contourf(xx, yy, zz_2, alpha=0., cmap='OrRd')
    axes_4.hist2d(samples[:, 0], samples[:, 1], bins=[100, 100],
                  range=[[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)]], cmap=my_cmap, cmin=1)
    axes_4.set_title('Estimated Potential')
    axes_4.grid('on')

    plt.tight_layout()
    fig.savefig(savings_path + '2d_example.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig(savings_path + '2d_example.png', dpi=300, bbox_inches='tight', pad_inches=0)


