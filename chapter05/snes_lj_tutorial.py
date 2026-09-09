#!/usr/bin/env python3
"""Teaching example: SNES fitting of a noisy Lennard-Jones curve.

The program uses only NumPy, Matplotlib, and Python's standard library. It is
intended to accompany Chapter 5 of the GPUMD book. It demonstrates

1. SNES optimization;
2. training/test errors and overfitting;
3. NEP-style L1/L2 regularization and sparsity;
4. data quality (label noise);
5. data efficiency (training-set size);
6. the importance of configuration-space coverage.

The main model is a deliberately over-parameterized 1-H-H-1 tanh network.
Default H=10 gives 141 trainable parameters for only 30 training points.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
N_TRAIN = 30
N_TEST = 1000
NOISE_STD = 0.10
N_NEURONS = 10
N_GENERATIONS = 3000
RECORD_EVERY = 20

LAMBDA_1 = 0.02
LAMBDA_2 = 0.02
SPARSITY_THRESHOLD = 1.0e-2
SIGMA_0 = 0.10
POPULATION_SIZE = 0  # 0: 4 + floor(3*log(number_of_parameters))
SEED = 1234
SMALL_NUMBER = 1.0e-30

SWEEP_GENERATIONS = 1000
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10, 0.20)
TRAIN_SIZES = (5, 10, 20, 30, 50, 100)

REGULARIZATION_FIGURE = "fig-c05-lj-regularization.png"
SPARSITY_FIGURE = "fig-c05-lj-sparsity.png"
SNES_FIGURE = "fig-c05-lj-snes.png"
DATA_FIGURE = "fig-c05-lj-data-quality-efficiency.png"
COVERAGE_FIGURE = "fig-c05-lj-coverage.png"


# -----------------------------------------------------------------------------
# Model and data
# -----------------------------------------------------------------------------
def lj_potential(r):
    """Lennard-Jones-type reference function used in this teaching example."""
    return 10.0 / r**12 - 10.0 / r**6


def scale_input(x):
    """Map r in [1, 3] to [-1, 1]."""
    return x - 2.0


def number_of_parameters(n_neurons):
    """Number of trainable parameters in a 1-H-H-1 network."""
    return n_neurons * (n_neurons + 4) + 1


def initial_parameters(n_neurons, rng):
    """Initial SNES distribution mean."""
    u = rng.normal(0.0, np.sqrt(2.0 / (1 + n_neurons)), n_neurons)
    v = rng.normal(0.0, np.sqrt(1.0 / n_neurons), (n_neurons, n_neurons))
    w = rng.normal(0.0, np.sqrt(2.0 / (n_neurons + 1)), n_neurons)
    a = np.zeros(n_neurons)
    b = np.zeros(n_neurons)
    c = np.zeros(1)
    return np.concatenate((u, v.ravel(), w, a, b, c))


def unpack_parameters(theta, n_neurons):
    offset = 0
    u = theta[offset : offset + n_neurons]
    offset += n_neurons
    v = theta[offset : offset + n_neurons**2].reshape(n_neurons, n_neurons)
    offset += n_neurons**2
    w = theta[offset : offset + n_neurons]
    offset += n_neurons
    a = theta[offset : offset + n_neurons]
    offset += n_neurons
    b = theta[offset : offset + n_neurons]
    c = theta[-1]
    return u, v, w, a, b, c


def predict(theta, x, n_neurons):
    """Prediction from a 1-H-H-1 tanh network."""
    u, v, w, a, b, c = unpack_parameters(theta, n_neurons)
    xs = scale_input(x)
    h1 = np.tanh(u[:, None] * xs[None, :] - a[:, None])
    h2 = np.tanh(v @ h1 - b[:, None])
    return w @ h2 - c


def make_uniform_dataset(n_train, n_test, noise_std, data_seed):
    rng = np.random.default_rng(data_seed)
    x_train = np.linspace(1.0, 3.0, n_train)
    y_train = lj_potential(x_train) + rng.normal(0.0, noise_std, n_train)
    x_test = np.linspace(1.0, 3.0, n_test)
    y_test = lj_potential(x_test)
    return x_train, y_train, x_test, y_test


def make_dataset_from_x(x_train, n_test, noise_std, data_seed):
    rng = np.random.default_rng(data_seed)
    x_train = np.asarray(x_train, dtype=float)
    y_train = lj_potential(x_train) + rng.normal(0.0, noise_std, x_train.size)
    x_test = np.linspace(1.0, 3.0, n_test)
    y_test = lj_potential(x_test)
    return x_train, y_train, x_test, y_test


# -----------------------------------------------------------------------------
# SNES
# -----------------------------------------------------------------------------
def population_loss(population, x, target, n_neurons, lambda_1, lambda_2):
    """NEP-style data RMSE + L1 + L2 loss for one SNES population."""
    population_size = population.shape[0]
    offset = 0
    u = population[:, offset : offset + n_neurons]
    offset += n_neurons
    v = population[:, offset : offset + n_neurons**2].reshape(
        population_size, n_neurons, n_neurons
    )
    offset += n_neurons**2
    w = population[:, offset : offset + n_neurons]
    offset += n_neurons
    a = population[:, offset : offset + n_neurons]
    offset += n_neurons
    b = population[:, offset : offset + n_neurons]
    c = population[:, -1]

    xs = scale_input(x)
    h1 = np.tanh(u[:, :, None] * xs[None, None, :] - a[:, :, None])
    h2 = np.tanh(np.einsum("pij,pjn->pin", v, h1) - b[:, :, None])
    pred = np.einsum("pi,pin->pn", w, h2) - c[:, None]

    data_rmse = np.sqrt(np.mean((pred - target[None, :]) ** 2, axis=1) + SMALL_NUMBER)
    r1 = np.mean(np.abs(population), axis=1)
    r2 = np.sqrt(np.mean(population**2, axis=1) + SMALL_NUMBER)
    return data_rmse + lambda_1 * r1 + lambda_2 * r2


def rmse(theta, x, target, n_neurons):
    return np.sqrt(np.mean((predict(theta, x, n_neurons) - target) ** 2))


def parameter_rms(theta):
    return np.sqrt(np.mean(theta**2))


def snes_utilities(population_size):
    ranks = np.arange(1, population_size + 1)
    u = np.maximum(0.0, np.log(population_size / 2.0 + 1.0) - np.log(ranks))
    return u / np.sum(u) - 1.0 / population_size


def new_history():
    return {
        "generation": [],
        "train_rmse": [],
        "test_rmse": [],
        "parameter_rms": [],
        "sigma_rms": [],
    }


def record(history, generation, mean, sigma, data, n_neurons):
    x_train, y_train, x_test, y_test = data
    history["generation"].append(generation)
    history["train_rmse"].append(rmse(mean, x_train, y_train, n_neurons))
    history["test_rmse"].append(rmse(mean, x_test, y_test, n_neurons))
    history["parameter_rms"].append(parameter_rms(mean))
    history["sigma_rms"].append(np.sqrt(np.mean(sigma**2)))


def finish_history(history):
    return {key: np.asarray(value) for key, value in history.items()}


def run_snes(
    theta0,
    data,
    n_neurons,
    generations,
    record_every,
    sigma0,
    population_size,
    lambda_1,
    lambda_2,
    random_seed,
    snapshot_generations=(),
):
    """Run separable natural evolution strategies."""
    x_train, y_train, _, _ = data
    dimension = theta0.size
    if population_size == 0:
        population_size = 4 + int(np.floor(3.0 * np.log(dimension)))

    rng = np.random.default_rng(random_seed)
    mean = theta0.copy()
    sigma = np.full(dimension, sigma0)
    utilities = snes_utilities(population_size)

    # Equivalent to eta_sigma/2 in the common SNES notation.
    lr_sigma_half = (3.0 + np.log(dimension)) / (5.0 * np.sqrt(dimension)) / 2.0

    history = new_history()
    snapshots = {}
    snapshot_generations = set(int(x) for x in snapshot_generations)

    record(history, 0, mean, sigma, data, n_neurons)
    if 0 in snapshot_generations:
        snapshots[0] = mean.copy()

    for generation in range(1, generations + 1):
        samples = rng.standard_normal((population_size, dimension))
        population = mean[None, :] + sigma[None, :] * samples
        losses = population_loss(
            population, x_train, y_train, n_neurons, lambda_1, lambda_2
        )
        ranked_samples = samples[np.argsort(losses)]

        mean += sigma * (utilities @ ranked_samples)
        sigma_gradient = utilities @ (ranked_samples**2 - 1.0)
        sigma *= np.exp(lr_sigma_half * sigma_gradient)
        sigma = np.clip(sigma, 1.0e-12, 1.0e2)

        if generation in snapshot_generations:
            snapshots[generation] = mean.copy()
        if generation % record_every == 0 or generation == generations:
            record(history, generation, mean, sigma, data, n_neurons)

    return {
        "mean": mean,
        "sigma": sigma,
        "history": finish_history(history),
        "snapshots": snapshots,
        "population_size": population_size,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def set_plot_style():
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 21,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
        }
    )


def plot_regularization(data, results, args):
    x_train, y_train, x_test, y_test = data
    colors = ("#D55E00", "#0072B2", "#009E73")
    parameter_color = "#CC79A7"
    titles = (
        "No regularization",
        rf"L2: $\lambda_2={args.lambda2:g}$",
        rf"L1 + L2: $\lambda_1={args.lambda1:g}$, $\lambda_2={args.lambda2:g}$",
    )

    figure, axes = plt.subplots(2, 3, figsize=(22.0, 10.5), sharex="row", sharey="row")
    parameter_axes = []

    for column, (result, color, title) in enumerate(zip(results, colors, titles)):
        history = result["history"]

        ax = axes[0, column]
        ax.plot(x_test, y_test, color="black", linewidth=3.0, label="Exact LJ")
        ax.scatter(x_train, y_train, s=48, color="#777777", alpha=0.8,
                   label="Noisy training data", zorder=3)
        ax.plot(x_test, predict(result["mean"], x_test, args.neurons),
                color=color, linewidth=3.0, label="Final SNES fit")
        ax.set_title(title)
        ax.set_xlabel("Distance r")
        ax.legend(frameon=False)

        ax = axes[1, column]
        xg = history["generation"] + 1
        ax.plot(xg, history["train_rmse"], color=color, linewidth=2.8,
                label="Training RMSE")
        ax.plot(xg, history["test_rmse"], color=color, linewidth=2.8,
                linestyle="--", label="Test RMSE")
        best_index = np.argmin(history["test_rmse"])
        ax.scatter(xg[best_index], history["test_rmse"][best_index], color=color,
                   marker="*", s=180, zorder=3)
        ax.text(0.67, 0.95,
                f"Best test: {history['test_rmse'][best_index]:.4f}\n"
                f"Final test: {history['test_rmse'][-1]:.4f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=15,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Generation + 1")

        ax2 = ax.twinx()
        ax2.plot(xg, history["parameter_rms"], color=parameter_color,
                 linewidth=2.8, linestyle="-.", label=r"Parameter RMS $R_2$")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelsize=16, colors=parameter_color)
        parameter_axes.append(ax2)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left")

    axes[0, 0].set_ylabel("Potential U(r)")
    axes[1, 0].set_ylabel("RMSE")
    for ax2 in parameter_axes[:-1]:
        ax2.tick_params(axis="y", right=False, labelright=False)
    parameter_axes[-1].set_ylabel(r"Parameter RMS $R_2$", color=parameter_color)

    all_rmse = np.concatenate([
        h[key] for h in [r["history"] for r in results]
        for key in ("train_rmse", "test_rmse")
    ])
    for ax in axes[1, :]:
        ax.set_ylim(np.min(all_rmse) / 1.1, np.max(all_rmse) * 1.1)

    pvals = np.concatenate([r["history"]["parameter_rms"] for r in results])
    for ax2 in parameter_axes:
        ax2.set_ylim(np.min(pvals) / 1.25, np.max(pvals) * 1.25)

    pop = results[0]["population_size"]
    figure.suptitle(
        "SNES fitting of noisy Lennard-Jones data\n"
        f"1-{args.neurons}-{args.neurons}-1 network, {args.train_size} training points, "
        f"{args.generations:,} generations, population {pop}",
        fontsize=25,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return figure


def plot_sparsity(results, args):
    colors = ("#D55E00", "#0072B2", "#009E73")
    labels = ("No regularization", "L2 regularization", "L1 + L2 regularization")
    figure, ax = plt.subplots(figsize=(12.5, 7.5))

    mags = [np.maximum(np.abs(r["mean"]), SMALL_NUMBER) for r in results]
    combined = np.concatenate(mags)
    lo = np.floor(np.log10(np.min(combined)))
    hi = np.ceil(np.log10(np.max(combined)))
    bins = np.logspace(lo, hi, 36)

    ax.axvspan(bins[0], args.sparsity_threshold, color="#999999", alpha=0.10)
    for result, values, color, label in zip(results, mags, colors, labels):
        near_zero = np.count_nonzero(values < args.sparsity_threshold)
        fraction = 100.0 * near_zero / values.size
        final_test = result["history"]["test_rmse"][-1]
        ax.hist(values, bins=bins, histtype="step", color=color, linewidth=3.5,
                label=(f"{label}: {near_zero}/{values.size} near-zero "
                       f"({fraction:.1f}%), test RMSE={final_test:.4f}"))

    ax.axvline(args.sparsity_threshold, color="black", linewidth=2.2,
               linestyle="--",
               label=rf"Near-zero threshold: $|\theta|<{args.sparsity_threshold:g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Final parameter magnitude $|\theta|$")
    ax.set_ylabel("Number of parameters")
    ax.set_title("Parameter distributions after SNES training")
    ax.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    return figure


def plot_snes_evolution(data, result, args):
    x_train, y_train, x_test, y_test = data
    history = result["history"]
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.5))

    ax = axes[0, 0]
    ax.plot(x_test, y_test, color="black", linewidth=3.0, label="Exact LJ")
    ax.scatter(x_train, y_train, s=40, color="#777777", alpha=0.65,
               label="Training data", zorder=3)
    snapshot_keys = sorted(result["snapshots"].keys())
    for generation in snapshot_keys:
        theta = result["snapshots"][generation]
        ax.plot(x_test, predict(theta, x_test, args.neurons), linewidth=2.0,
                label=f"Generation {generation}")
    ax.set_xlabel("Distance r")
    ax.set_ylabel("Potential U(r)")
    ax.set_title("Evolution of the fitted potential")
    ax.legend(frameon=False, fontsize=12)

    xg = history["generation"] + 1
    ax = axes[0, 1]
    ax.plot(xg, history["train_rmse"], linewidth=2.8, label="Training RMSE")
    ax.plot(xg, history["test_rmse"], linewidth=2.8, linestyle="--", label="Test RMSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Generation + 1")
    ax.set_ylabel("RMSE")
    ax.set_title("Training and generalization")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(xg, history["parameter_rms"], linewidth=2.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Generation + 1")
    ax.set_ylabel(r"Parameter RMS $R_2$")
    ax.set_title("Model parameter scale")

    ax = axes[1, 1]
    ax.plot(xg, history["sigma_rms"], linewidth=2.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Generation + 1")
    ax.set_ylabel(r"SNES search-width RMS")
    ax.set_title(r"Contraction of the search distribution $\sigma$")

    figure.suptitle("How SNES learns the Lennard-Jones curve", fontsize=24)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return figure


def plot_data_quality_efficiency(noise_stats, size_stats):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    noise_levels, noise_mean, noise_std = noise_stats
    ax = axes[0]
    ax.errorbar(noise_levels, noise_mean, yerr=noise_std, marker="o", linewidth=2.5,
                capsize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Training-label noise standard deviation")
    ax.set_ylabel("Clean test RMSE")
    ax.set_title("Data quality")
    ax.grid(alpha=0.2)

    train_sizes, size_mean, size_std = size_stats
    ax = axes[1]
    ax.errorbar(train_sizes, size_mean, yerr=size_std, marker="o", linewidth=2.5,
                capsize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of training points")
    ax.set_ylabel("Clean test RMSE")
    ax.set_title("Data efficiency")
    ax.grid(alpha=0.2)

    figure.suptitle("Reference-data quality and quantity", fontsize=23)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


def plot_coverage(coverage_results, args):
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), sharex=True, sharey=True)
    for ax, item in zip(axes, coverage_results):
        name, data, result = item
        x_train, y_train, x_test, y_test = data
        ax.plot(x_test, y_test, color="black", linewidth=2.8, label="Exact LJ")
        ax.scatter(x_train, y_train, s=38, color="#777777", alpha=0.75,
                   label="Training data", zorder=3)
        ax.plot(x_test, predict(result["mean"], x_test, args.neurons), linewidth=2.8,
                label="SNES fit")
        test_rmse = result["history"]["test_rmse"][-1]
        ax.set_title(f"{name}\nTest RMSE = {test_rmse:.3f}")
        ax.set_xlabel("Distance r")
    axes[0].set_ylabel("Potential U(r)")
    axes[0].legend(frameon=False, fontsize=12)
    figure.suptitle("The number of data points is not enough: coverage matters", fontsize=23)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return figure


# -----------------------------------------------------------------------------
# Sweep helpers
# -----------------------------------------------------------------------------
def run_data_sweeps(args, theta0):
    """Controlled single-realization sweeps.

    All conditions use the same initial parameter vector and the same SNES
    random-sample sequence. This isolates the effect of changing the data.
    """
    noise_rmse = []
    for noise in args.noise_levels:
        data = make_uniform_dataset(
            args.train_size, args.test_size, noise, args.seed
        )
        result = run_snes(
            theta0, data, args.neurons, args.sweep_generations,
            max(args.sweep_generations // 10, 1), args.sigma0,
            args.population, args.lambda1, args.lambda2, args.seed + 2
        )
        noise_rmse.append(result["history"]["test_rmse"][-1])

    size_rmse = []
    for n_train in args.train_sizes:
        data = make_uniform_dataset(
            n_train, args.test_size, args.noise, args.seed
        )
        result = run_snes(
            theta0, data, args.neurons, args.sweep_generations,
            max(args.sweep_generations // 10, 1), args.sigma0,
            args.population, args.lambda1, args.lambda2, args.seed + 2
        )
        size_rmse.append(result["history"]["test_rmse"][-1])

    return (
        (np.asarray(args.noise_levels), np.asarray(noise_rmse), np.zeros(len(noise_rmse))),
        (np.asarray(args.train_sizes), np.asarray(size_rmse), np.zeros(len(size_rmse))),
    )


def run_coverage(args, theta0):
    n = args.train_size
    x_uniform = np.linspace(1.0, 3.0, n)
    x_no_short = np.linspace(1.2, 3.0, n)
    n1 = n // 2
    x_gap = np.concatenate((
        np.linspace(1.0, 1.25, n1),
        np.linspace(1.8, 3.0, n - n1),
    ))
    cases = (
        ("Uniform coverage", x_uniform),
        ("Missing short-range data", x_no_short),
        ("A gap in configuration space", x_gap),
    )
    out = []
    for name, x_train in cases:
        data = make_dataset_from_x(
            x_train, args.test_size, args.noise, args.seed
        )
        result = run_snes(
            theta0, data, args.neurons, args.coverage_generations,
            max(args.coverage_generations // 20, 1), args.sigma0,
            args.population, args.lambda1, args.lambda2, args.seed + 2
        )
        out.append((name, data, result))
    return out


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="SNES/LJ teaching program for the GPUMD book")
    parser.add_argument("--train-size", type=int, default=N_TRAIN)
    parser.add_argument("--test-size", type=int, default=N_TEST)
    parser.add_argument("--noise", type=float, default=NOISE_STD)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--record-every", type=int, default=RECORD_EVERY)
    parser.add_argument("--lambda1", type=float, default=LAMBDA_1)
    parser.add_argument("--lambda2", type=float, default=LAMBDA_2)
    parser.add_argument("--sparsity-threshold", type=float, default=SPARSITY_THRESHOLD)
    parser.add_argument("--sigma0", type=float, default=SIGMA_0)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sweep-generations", type=int, default=SWEEP_GENERATIONS)
    parser.add_argument("--coverage-generations", type=int, default=1500)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(NOISE_LEVELS))
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(TRAIN_SIZES))
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    if args.train_size < 2 or args.test_size < 2:
        parser.error("train-size and test-size must be at least 2")
    if args.neurons < 1 or args.generations < 1 or args.sweep_generations < 1:
        parser.error("neurons and generation counts must be positive")
    if args.noise < 0 or args.lambda1 < 0 or args.lambda2 < 0 or args.sigma0 <= 0:
        parser.error("noise/lambdas must be nonnegative and sigma0 positive")
    return args


def save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    set_plot_style()

    data = make_uniform_dataset(args.train_size, args.test_size, args.noise, args.seed)
    theta0 = initial_parameters(args.neurons, np.random.default_rng(args.seed + 1))
    snes_seed = args.seed + 2
    snapshot_generations = sorted(set(
        [0, min(100, args.generations), min(1000, args.generations), args.generations]
    ))

    # Run the data sweeps first. For this small NumPy example, doing the larger
    # collection of short SNES runs before the long regularization runs avoids
    # a large performance penalty on some BLAS/threading configurations.
    noise_stats, size_stats = run_data_sweeps(args, theta0)
    coverage_results = run_coverage(args, theta0)

    main_specs = ((0.0, 0.0), (0.0, args.lambda2), (args.lambda1, args.lambda2))
    main_results = []
    for lambda1, lambda2 in main_specs:
        main_results.append(
            run_snes(theta0, data, args.neurons, args.generations, args.record_every,
                     args.sigma0, args.population, lambda1, lambda2, snes_seed,
                     snapshot_generations=snapshot_generations)
        )

    save_figure(plot_regularization(data, main_results, args),
                os.path.join(args.output_dir, REGULARIZATION_FIGURE))
    save_figure(plot_sparsity(main_results, args),
                os.path.join(args.output_dir, SPARSITY_FIGURE))
    save_figure(plot_snes_evolution(data, main_results[2], args),
                os.path.join(args.output_dir, SNES_FIGURE))
    save_figure(plot_data_quality_efficiency(noise_stats, size_stats),
                os.path.join(args.output_dir, DATA_FIGURE))
    save_figure(plot_coverage(coverage_results, args),
                os.path.join(args.output_dir, COVERAGE_FIGURE))

    print("Done.")
    print(f"Network parameters: {number_of_parameters(args.neurons)}")
    print(f"Population size: {main_results[0]['population_size']}")
    print("Main regularization runs (final train/test RMSE):")
    for label, result in zip(("none", "L2", "L1+L2"), main_results):
        h = result["history"]
        print(f"  {label:6s}: {h['train_rmse'][-1]:.6f} / {h['test_rmse'][-1]:.6f}")
    print("Data-quality sweep (noise -> mean test RMSE):")
    for x, y in zip(noise_stats[0], noise_stats[1]):
        print(f"  {x:g} -> {y:.6f}")
    print("Data-efficiency sweep (N_train -> mean test RMSE):")
    for x, y in zip(size_stats[0], size_stats[1]):
        print(f"  {int(x)} -> {y:.6f}")
    print("Coverage study (final test RMSE):")
    for name, _, result in coverage_results:
        print(f"  {name}: {result['history']['test_rmse'][-1]:.6f}")


if __name__ == "__main__":
    main()
