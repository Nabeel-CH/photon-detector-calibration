"""Functions for s1 plots"""

import matplotlib.pyplot as plt
import numpy as np

from s1_sol import s1funcs


def plot1_1(sample_df):
    """
    Make Figure 1.1: histogram of the energy difference E_rec - E_true and save the figure in figs folder.

    Parameters:
        sample_df (pandas.DataFrame): DataFrame containing the sample data.
                                      Must contain columns "E_rec" and "E_true".
    Returns:
        fig, ax (matplotlib Figure and Axes): The figure and axis used for the plot.
    """
    # Dimensions of the plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Calculate the energy difference and make a plot
    sample_df["E_difference"] = sample_df["E_rec"] - sample_df["E_true"]
    ax.hist(sample_df["E_difference"], bins=50, histtype="step")

    # Add labels and title 
    ax.set_xlabel("Measured Energy - True Energy (E - E_0)/ GeV")
    ax.set_ylabel("Number of values")
    ax.set_title("Distribution of Energy Difference (E - E_0)")

    # Save the figure
    fig.savefig("../figs/Figure1.1.pdf")

    return fig, ax


def plot1_2(sample_df):
    """
    Make Figure 1.2: overlaid probability-density histograms of the
    energy difference (E_rec - E_true) for each E_true value.
    Save the figure in the figs folder.

    Parameters:
        sample_df (pandas.DataFrame): DataFrame containing the sample data.
                                      Must contain columns "E_rec" and "E_true".

    Returns:
        fig, ax (matplotlib Figure and Axes): The figure and axis used for the plot.
    """
    # Dimensions of the plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Get all unique E_0 values
    unique_E0 = sample_df["E_true"].unique()

    for E0 in unique_E0:
        # Select only rows that correspond to this E0
        sub_df = sample_df[sample_df["E_true"] == E0]

        # Get the (E - E0) values
        diff_values = sub_df["E_rec"] - sub_df["E_true"]

        # Make the histogram (probability density)
        ax.hist(
            diff_values,
            bins=50,           # number of bins
            histtype="step",
            density=True,      # probability density
            label=f"E_true = {E0}",
        )

    # Add labels and title
    ax.set_xlabel("Measured Energy - True Energy (E - E_0)")
    ax.set_ylabel("Probability density")
    ax.set_title("Distribution of (E - E_0) for different E_0 values")

    # Add legend
    ax.legend()

    # Adjust layout
    fig.tight_layout()

    # Save the figure
    fig.savefig("../figs/Figure1.2.pdf")

    return fig, ax


def plot1_3(summary):
    """
    Make Figure 1.3: scatter plots of the estimated sample mean and
    sample standard deviation of E_rec plotted against E_true.
    Save the figure in the figs folder.

    Parameters:
        summary (pandas.DataFrame): DataFrame containing a summary of the means and standard deviations calculated.
                                    Must contain following columns:
                                    - "E_true"
                                    - "mu_est"
                                    - "sigma_est"

    Returns:
        fig, ax (matplotlib Figure and Axes array): The figure and axes used for the plots.
    """
    # Create a figure with 2 subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # Plot 1: mean vs E_true
    ax[0].plot(
        summary["E_true"],
        summary["mu_est"],
        "o",
    )
    ax[0].set_xlabel("E_0/GeV")
    ax[0].set_ylabel("estimate of sample mean for E_rec")
    ax[0].set_title("Sample Mean vs E₀")

    # Plot 2: std dev vs E_true
    ax[1].plot(
        summary["E_true"],
        summary["sigma_est"],
        "o",
    )
    ax[1].set_xlabel("E_0/GeV")
    ax[1].set_ylabel("estimate of sample std dev for E_rec")
    ax[1].set_title("Sample Std Dev vs E₀")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    fig.savefig("../figs/Figure1.3.pdf")

    return fig, ax


def plot1_4(summary,
            lb_samp, dE_samp,
            a_samp, b_samp, c_samp,
            cov_mu_samp, cov_sigma_samp,
            N_BOOT=1000, random_state=30):
    """
    Make Figure 1.4:
        Left plot : (estimate of sample mean - E0) vs E0
        Right plot: (estimate of sample std dev / E0) vs E0
        Both plots with fitted curves and bootstrap ±1sigma bands.

    Parameters:
        summary (pandas.DataFrame): 
            Must contain columns "E_true", "mu_est", "sigma_est".
        lb_samp, dE_samp (float):
            parameters for the mean model.
        a_samp, b_samp, c_samp (float):
            parameters for the width model.
        cov_mu_samp (array):
            Covariance matrix for (lb_samp, dE_samp).
        cov_sigma_samp (array):
            Covariance matrix for (a_samp, b_samp, c_samp).

    Returns:
        fig, ax (matplotlib Figure and Axes array):
            The figure and axes used for the plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # PART 1: plot estimate of sample mean - E0 against E0 
    #       and estimate of sample std dev / E0 against E0
    # get x-values
    E0 = summary["E_true"].to_numpy()

    # get values needed to calculate y-values
    mu_values   = summary["mu_est"].to_numpy()   # measured sample means
    sigma_values = summary["sigma_est"].to_numpy()  # measured sample widths
    # calculate y-values
    mu_shift    = mu_values - E0
    sigma_ratio = sigma_values / E0

    # plot data points
    ax[0].plot(E0, mu_shift, "o", label="data")
    ax[1].plot(E0, sigma_ratio, "o", label="data")

    # set labels and titles
    ax[0].set_xlabel("E0")
    ax[0].set_ylabel("estimate of sample mean - E0")
    ax[0].set_title("mean offset vs E0")

    ax[1].set_xlabel("E0")
    ax[1].set_ylabel("estimate of sample std dev / E0")
    ax[1].set_title("relative width vs E0")


    # Part 2: overlay the fitted curves
    # E0 values for the fits
    E_plot = np.linspace(E0.min(), E0.max(), 200)

    # fitted curves using best–fit parameters
    mu_curve    = s1funcs.mu_model(E_plot, lb_samp, dE_samp) - E_plot
    sigma_curve = s1funcs.sigma_model(E_plot, a_samp, b_samp, c_samp) / E_plot

    # overlay the fitted curves
    ax[0].plot(E_plot, mu_curve, label="fit")
    ax[1].plot(E_plot, sigma_curve, label="fit")


    # Part 3: add bootstrap error bands (±1sigma bands) to the curves

    # parameters of mean and width models
    mean_params  = np.array([lb_samp, dE_samp])
    width_params = np.array([a_samp, b_samp, c_samp])

    # covariance matrices of parameters for mean and width models
    mean_cov  = np.array(cov_mu_samp)     
    width_cov = np.array(cov_sigma_samp)   

    # get the bootstrap bands
    mu_band, sigma_band = s1funcs.bootstrap_bands_q1(
        E_plot,
        mean_params,
        mean_cov,
        width_params,
        width_cov,
        N_BOOT=N_BOOT,
        random_state=random_state #set random state for reproducibility
    )

    # add shaded bootstrap bands
    ax[0].fill_between(
        E_plot,
        mu_curve - mu_band,
        mu_curve + mu_band,
        alpha=0.3,
        label="bootstrap ±1sigma",
    )

    ax[1].fill_between(
        E_plot,
        sigma_curve - sigma_band,
        sigma_curve + sigma_band,
        alpha=0.3,
        label="bootstrap ±1sigma",
    )

    # add legends now that all elements are on the axes
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    fig.savefig("../figs/Figure1.4.pdf")

    return fig, ax
