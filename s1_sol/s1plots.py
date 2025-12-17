"""Functions for all plots in solution.ipynb"""

import matplotlib.pyplot as plt
import numpy as np

from s1_sol import s1funcs


# Formatting constants used for plots 3.2, 4.1, 4.2
method_order  = ["sample", "individual", "simultaneous"]
method_colors = {
    "sample":       "#3B82F6",
    "individual":   "#F59E0B",  
    "simultaneous": "#EC4899",  
}
# parameters labels
param_order = ["lb", "dE", "a", "b", "c"]
param_labels = ["$\\lambda$", "$\\Delta$", "$a$", "$b$", "$c$"]


####Functions for making the plots####

def plot1_1(sample_df, bins=50):
    """
    Make Figure 1.1: histogram of the energy difference E_rec - E_true and save the figure in figs folder.

    Parameters:
        sample_df (pandas.DataFrame): DataFrame containing the sample data.
                                      Must contain columns "E_rec" and "E_true".
        bins (int): Number of bins for the histogram.

    Returns:
        fig, ax (matplotlib Figure and Axes): The figure and axis used for the plot.
    """
    # Dimensions of the plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Plot the histogram
    ax.hist(sample_df["E_difference"], bins=bins, histtype="bar")

    # Add labels and title 
    ax.set_xlabel("$(E - E_0)$ [GeV]")
    ax.set_ylabel("Number of events")
    ax.set_title("Distribution of Energy Difference $(E - E_0)$")

    # Save the figure
    fig.savefig("../figs/Figure1.1.pdf")

    return fig, ax


def plot1_2(sample_df, bins=50):
    """
    Make Figure 1.2: Distribution of `E - E_0`, with the histograms for each different value of `E_0` overlaid.
    Save the figure in the figs folder.

    Parameters:
        sample_df (pandas.DataFrame): DataFrame containing the sample data.
                                      Must contain columns "E_rec" and "E_true".
        bins (int): Number of bins for the histograms.

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
            bins=bins,           # number of bins
            histtype="step",
            density=True,      # probability density
            label=f"$E_0$ = {E0}",
        )

    # Add labels and title
    ax.set_xlabel("$(E - E_0)$ [GeV]")
    ax.set_ylabel("Probability density")
    ax.set_title(r"Distribution of $(E - E_0)$ for different $E_0$ values")

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
    sample standard deviation of E_rec plotted against E_true, along with their error bars.
    Save the figure in the figs folder.

    Parameters:
        summary (pandas.DataFrame): DataFrame containing a summary of the means and standard deviations calculated.
                                    Must contain the following columns:
                                    - "E_true"    (true energy)
                                    - "mu_est"    (estimate of sample mean)
                                    - "mu_err"    (error on sample mean)
                                    - "sigma_est" (estimate of sample std dev)
                                    - "sigma_err" (error on sample std dev)

    Returns:
        fig, ax (matplotlib Figure and Axes array): The figure and axes used for the plots.
    """
    # Create a figure with 2 subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # Plot 1: mean vs E_true (with error bars)
    ax[0].errorbar(
        summary["E_true"],
        summary["mu_est"],
        yerr=summary["mu_err"],
        fmt="x",
        capsize=3,
    )
    ax[0].set_xlabel("$E_0$ [GeV]")
    ax[0].set_ylabel("$\\hat{\\mu}_{\\rm samp}$ [GeV]")
    ax[0].set_title("Estimate of Sample Mean vs $E_0$")

    # Plot 2: std dev vs E_true (with error bars)
    ax[1].errorbar(
        summary["E_true"],
        summary["sigma_est"],
        yerr=summary["sigma_err"],
        fmt="x",
        capsize=3,
    )
    ax[1].set_xlabel("$E_0$ [GeV]")
    ax[1].set_ylabel("$\\hat{\\sigma}_{\\rm samp}$ [GeV]")
    ax[1].set_title("Estimate of Sample Std Dev vs $E_0$")

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
        The figure is saved as in the figs folder.

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
    ax[0].set_xlabel("$E_0$ [GeV]")
    ax[0].set_ylabel("($\\hat{\\mu}_{\\rm samp}$ - $E_0$) [GeV]")
    ax[0].set_title("($\\hat{\\mu}_{\\rm samp}$ - $E_0$) vs $E_0$")

    ax[1].set_xlabel("$E_0$ [GeV]")
    ax[1].set_ylabel("($\\hat{\\sigma}_{\\rm samp}$ / $E_0$)")
    ax[1].set_title("($\\hat{\\sigma}_{\\rm samp}$ / $E_0$) vs $E_0$")
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
        label="bootstrap ±1${\\sigma}$",
    )

    ax[1].fill_between(
        E_plot,
        sigma_curve - sigma_band,
        sigma_curve + sigma_band,
        alpha=0.3,
        label="bootstrap ±1${\\sigma}$",
    )

    # add legends now that all elements are on the axes
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    fig.savefig("../figs/Figure1.4.pdf")

    return fig, ax

def plot2_1(sample_df, indiv_summary):
    """
    Make Figure 2.1:
      Left plot: for each E_0, histogram of (E_rec - E_0) with the
                   corresponding individual ML Gaussian fit overlaid.
      Right plot: histogram of (E_rec - E_0) for all events together,
                   with the weighted sum of all individual ML fits overlaid.
    The figure is saved in the figs folder.

    Parameters:
        sample_df (pandas.DataFrame):
            Must contain columns:
               - "E_true"
                - "E_rec"
                - "E_difference" 
        indiv_summary (pandas.DataFrame):
            Summary of individual ML fits per E_true.
            Must contain columns:
                - "E_true"
                - "mu_est"
                - "sigma_est"

    Returns:
        fig, ax (matplotlib Figure and Axes array):
            The figure and axes used for the plot.
    """
    
    #Set up the figure with 2 subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # differences E - E0 
    E_diff_all = sample_df["E_difference"] 

    # x-range for plotting fitted curves
    x_min, x_max = E_diff_all.min(), E_diff_all.max()
    x_grid = np.linspace(x_min, x_max, 300)


    # Part 1: histogram + fit for each E0
    # List of distinct E0 values (sorted, just to be tidy)
    E0_values = sorted(sample_df["E_true"].unique())

    # Total number of events (used for the right panel later)
    N_total = len(sample_df)

    # Colour cycle from matplotlib (so each E0 gets its own colour)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


    for i, E0 in enumerate(E0_values):
        # Pick a colour for this E0
        color = colors[i % len(colors)]

        # All rows with this E0
        this_group = sample_df[sample_df["E_true"] == E0]

        # Differences E - E0 for this E0
        diff = this_group["E_rec"] - this_group["E_true"]

        # Histogram (probability density)
        ax[0].hist(
            diff,
            bins=50,
            histtype="step",
            density=True,
            color=color,
            label=fr"$E_0 = {E0}$",
        )

        # Get the fitted mean and sigma for this E0 from indiv_summary
        this_row   = indiv_summary[indiv_summary["E_true"] == E0].iloc[0]
        mu_hat     = this_row["mu_est"]
        sigma_hat  = this_row["sigma_est"]

        # For E - E0, the Gaussian mean is mean_indiv − E0
        mu_shift = mu_hat - E0

        # Fitted Gaussian curve on the common x_grid
        pdf = s1funcs.normal_pdf(x_grid, mu_shift, sigma_hat)


        ax[0].plot(
            x_grid,
            pdf,
            linestyle="--",
            color=color  # same colour as the histogram
        )
    ax[0].set_xlabel("($E - E_0$) [GeV]")
    ax[0].set_ylabel("Probability density")
    ax[0].set_title("Distributions of ($E - E_0$) at each $E_0$ along with ML fits")
    ax[0].legend()


    # Part 2: total histogram + appropriately normalised sum of sub-distributions
    # Histogram of E - E0 for all events
    ax[1].hist(
        E_diff_all,
        bins=50,
        #histtype="step",
        density=True,
        label="all $E_0$ data",
    )

    # Build the weighted sum of all fitted Gaussians
    total_pdf = np.zeros_like(x_grid)

    for E0 in E0_values:
        # Get the fitted μ and σ for this E0
        this_row   = indiv_summary[indiv_summary["E_true"] == E0].iloc[0]
        mu_hat     = this_row["mu_est"]
        sigma_hat  = this_row["sigma_est"]

        # In E - E0 space, the mean is μ_indiv − E0
        mu_shift = mu_hat - E0

        # Gaussian for this E0 in E - E0 space
        pdf = s1funcs.normal_pdf(x_grid, mu_shift, sigma_hat)

        # Number of events at this E0
        n_E0 = len(sample_df[sample_df["E_true"] == E0])

        # Weight by fraction of events at this E0 so total area stays 1
        weight = n_E0 / N_total

        total_pdf += weight * pdf
    # Overlay the summed model
    ax[1].plot(x_grid, total_pdf, label="weighted sum of ML fits")

    ax[1].set_xlabel("($E - E_0$) [GeV]")
    ax[1].set_ylabel("Probability density")
    ax[1].set_title("All ($E - E_0$) overlaid with sum of sub-distributions")
    ax[1].legend()


    plt.tight_layout()

    fig.savefig("../figs/Figure2.1.pdf")

    return fig, ax


def plot2_2(indiv_summary,
            lb_indiv, dE_indiv,
            a_indiv, b_indiv, c_indiv,
            cov_mu_indiv, cov_sigma_indiv,
            N_BOOT=1000, random_state=30):
    """
    Make Figure 2.2:
        - Left plot : (estimate of sample mean - E0) vs E0 using individual ML estimates
        - Right plot: (estimate of sample std dev / E0) vs E0 using individual ML estimates
        Both plots with fitted curves and bootstrap ±1sigma bands.

    Parameters:
        indiv_summary (pandas.DataFrame):
            Must contain columns:
                - "E_true"
                - "mu_est"
                - "sigma_est"
        lb_indiv, dE_indiv (float):
            Parameters for the mean model from the individual-fits method.
        a_indiv, b_indiv, c_indiv (float):
            Parameters for the width model from the individual-fits method.
        cov_mu_indiv (array):
            Covariance matrix for (lb_indiv, dE_indiv).
        cov_sigma_indiv (array):
            Covariance matrix for (a_indiv, b_indiv, c_indiv).

    Returns:
        fig, ax (matplotlib Figure and Axes array):
            The figure and axes used for the plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # PART 1: plot estimate of sample mean - E0 against E0 
    #       and estimate of sample std dev / E0 against E0
    # x-values
    E0 = indiv_summary["E_true"].to_numpy()
    # y-values 
    mu_values = indiv_summary["mu_est"].to_numpy()
    sigma_values = indiv_summary["sigma_est"].to_numpy()
    mu_shift    = mu_values - E0
    sigma_ratio = sigma_values / E0

    # plot data points
    ax[0].plot(E0, mu_shift, "o", label="data")
    ax[1].plot(E0, sigma_ratio, "o", label="data")

    # set labels and titles
    ax[0].set_xlabel("$E_0$ [GeV]")
    ax[0].set_ylabel("($\\hat{\\mu}_{\\rm indiv} - E_0$) [GeV]")
    ax[0].set_title("($\\hat{\\mu}_{\\rm indiv} - E_0$) vs $E_0$")

    ax[1].set_xlabel("$E_0$ [GeV]")
    ax[1].set_ylabel("($\\hat{\\sigma}_{\\rm indiv}$ / $E_0$)")
    ax[1].set_title("($\\hat{\\sigma}_{\\rm indiv}$ / $E_0$) vs $E_0$")

    # Part 2: overlay the fitted curves
    # E0 values for the fits
    E_plot = np.linspace(E0.min(), E0.max(), 200)

    # fitted curves using best–fit parameters
    mu_curve    = s1funcs.mu_model(E_plot, lb_indiv, dE_indiv) - E_plot
    sigma_curve = s1funcs.sigma_model(E_plot, a_indiv, b_indiv, c_indiv) / E_plot

    # overlay the fitted curves
    ax[0].plot(E_plot, mu_curve, label="fit")
    ax[1].plot(E_plot, sigma_curve, label="fit")


    # Part 3: add bootstrap error bands (±1sigma bands) to the curves

    # parameters of mean and width models
    mean_params  = np.array([lb_indiv, dE_indiv])
    width_params = np.array([a_indiv, b_indiv, c_indiv])

    # covariance matrices of parameters for mean and width models
    mean_cov  = np.array(cov_mu_indiv)     
    width_cov = np.array(cov_sigma_indiv)   


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
        label="bootstrap ±1${\\sigma}$",
    )

    ax[1].fill_between(
        E_plot,
        sigma_curve - sigma_band,
        sigma_curve + sigma_band,
        alpha=0.3,
        label="bootstrap ±1${\\sigma}$",
    )

    # add legends now that all elements are on the axes
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()

    fig.savefig("../figs/Figure2.2.pdf")

    return fig, ax


def plot3_1(sample_df,
            lam_sim, Delta_sim,
            a_sim, b_sim, c_sim,
            sim_covariance,
            N_BOOT=500, random_state=30):
    """
    Make Figure 3.1:
        Left panel : (mean_E - E_0) vs E_0 for the simultaneous fit,
                   with parametric bootstrap ±1sigma band.
        Right panel: (std_dev_E / E_0) vs E_0 for the simultaneous fit,
                   with parametric bootstrap ±1sigma band.

    Parameters
    ----------
    sample_df (pandas.DataFrame)
        DataFrame containing the sample data.
        Must contain column "E_true".
    lam_sim, Delta_sim : float
        Parameters for the mean model from the simultaneous fit.
    a_sim, b_sim, c_sim : float
        Parameters for the width model from the simultaneous fit.
    sim_covariance : array
        Covariance matrix for (lam_sim, Delta_sim, a_sim, b_sim, c_sim).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes array
        The figure and axes used for the plot.
    """
    
    # set up the figure with 2 subplots 
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))


    # Use the full range of true energies for the x-axis
    E0_min = sample_df["E_true"].min()
    E0_max = sample_df["E_true"].max()
    E_plot = np.linspace(E0_min, E0_max, 200)

    # Central curves from the simultaneous fit
    mu_curve    = s1funcs.mu_model(E_plot, lam_sim, Delta_sim)
    sigma_curve = s1funcs.sigma_model(E_plot, a_sim, b_sim, c_sim)

    # For plotting convert to shift and ratio
    mu_shift_curve    = mu_curve - E_plot
    sigma_ratio_curve = sigma_curve / E_plot


    # Parametric bootstrap to get ±1sigma bands
    # Mean parameter vector and covariance matrix from Minuit
    param_mean = np.array([lam_sim, Delta_sim, a_sim, b_sim, c_sim])
    param_cov  = np.array(sim_covariance)  


    mu_band, sigma_band = s1funcs.bootstrap_bands_q3(
        E_plot, 
        param_mean, 
        param_cov, 
        N_BOOT,
        random_state=random_state  # set random state for reproducibility
    )

    # Make the two plots

    # Left plot
    ax[0].plot(E_plot, mu_shift_curve, label="simultaneous fit")
    ax[0].fill_between(
        E_plot,
        mu_shift_curve - mu_band,
        mu_shift_curve + mu_band,
        alpha=0.3,
        label="bootstrap ±1${\\sigma}$",
    )
    ax[0].set_xlabel("$E_0$ [GeV]")
    ax[0].set_ylabel("(${\\mu}_{\\rm E}$ - $E_0$) [GeV]")
    ax[0].set_title("(${\\mu}_{\\rm E}$ - $E_0$) vs $E_0$")
    ax[0].legend()

    # Right plot
    ax[1].plot(E_plot, sigma_ratio_curve, label="simultaneous fit")
    ax[1].fill_between(
        E_plot,
        sigma_ratio_curve - sigma_band,
        sigma_ratio_curve + sigma_band,
        alpha=0.3,
        label="bootstrap ±1${\\sigma}$",
    )
    ax[1].set_xlabel("$E_0$ [GeV]")
    ax[1].set_ylabel("(${\\sigma}_{\\rm E}$ / $E_0$)")
    ax[1].set_title("(${\\sigma}_{\\rm E}$ / $E_0$) vs $E_0$")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig("../figs/Figure3.1.pdf")

    return fig, ax


def plot3_2(df_results):
    """
    Make Figure 3.2: comparison of parameter estimates and errors
    for the three methods (sample, individual, simultaneous).

    Parameters
    ----------
    df_results (pandas.DataFrame):
        Summary table created from results.json file with the columns:
          - "method"    : one of {"sample", "individual", "simultaneous"}
          - "parameter" : one of {"lb", "dE", "a", "b", "c"}
          - "value"     : fitted parameter value
          - "error"     : fitted parameter uncertainty

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axis used for the plot.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    # x positions for parameter groups
    x_base = np.arange(len(param_order))
    width  = 0.25  # horizontal offset between methods

    for i, method in enumerate(method_order):
        # x positions for this method (shifted left/right)
        x = x_base + (i - 1) * width

        # subset df_results for this method
        sub = df_results[df_results["method"] == method]

        # values and errors in parameter order
        vals = [sub[sub["parameter"] == p]["value"].iloc[0] for p in param_order]
        errs = [sub[sub["parameter"] == p]["error"].iloc[0] for p in param_order]

        ax.errorbar(
            x,
            vals,
            yerr=errs,
            fmt="x",
            capsize=3,
            label=method,
            color=method_colors[method],
        )

    # x-axis: parameter labels
    ax.set_xticks(x_base)
    ax.set_xticklabels(param_labels)

    ax.set_ylabel("Parameter value")
    ax.set_xlabel("Parameter")
    ax.set_title("Comparison of parameter estimates and errors")
    ax.grid(alpha=0.3)
    ax.legend(title="estimation type")

    plt.tight_layout()

    fig.savefig("../figs/Figure3.2.pdf")

    return fig, ax

def plot4_1(boot_df):
    """
    Make Figure 4.1: bootstrap distributions of parameters
    for the three methods (sample, individual, simultaneous).

    Parameters
    ----------
    boot_df : pandas.DataFrame
        Tidy bootstrap table with columns:
          - "method"   : "sample", "individual", or "simultaneous"
          - "parameter": "lb", "dE", "a", "b", or "c"
          - "value"    : bootstrapped parameter value

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and array of axes used for the plot.
    """
    fig, ax = plt.subplots(2, 3, figsize=(19.2, 9.6))

    
    # Set up the subplots for each parameter
    param_axes = {
        "lb": (0, 0),   
        "dE": (0, 1),   
        "a":  (1, 0),
        "b":  (1, 1),
        "c":  (1, 2),
    }

    # Plot bootstrap histograms for each parameter and method
    for param, (i, j) in param_axes.items():
        ax_ij = ax[i, j]
        sub = boot_df[boot_df["parameter"] == param]

        for method in method_order:
            vals = sub[sub["method"] == method]["value"].to_numpy()

            # only label once (on λ panel) so legend isn't duplicated
            label = method if param == "lb" else None

            ax_ij.hist(
                vals,
                bins=40,
                histtype="step",
                density=True,
                color=method_colors[method],
                label=label,
            )

        ax_ij.set_ylabel("bootstrap density")
        ax_ij.grid(alpha=0.3)

    # add legend on one of the subplots
    ax[0, 0].legend(title="estimation type")

    # put your code here
    ax[0,0].set_xlabel("$\\lambda$")
    ax[0,1].set_xlabel("$\\Delta$")
    ax[0,2].set_visible(False)
    ax[1,0].set_xlabel("$a$")
    ax[1,1].set_xlabel("$b$")
    ax[1,2].set_xlabel("$c$")

    fig.savefig("../figs/Figure4.1.pdf")

    return fig, ax

def plot4_2(boot_df, df_results):
    """
    Make Figure 4.2: comparison of original vs bootstrap parameter estimates
    for the three methods (sample, individual, simultaneous).
    Save the figure in the figs folder.

    Parameters:
        boot_df (pandas.DataFrame):
            Table with columns:
            - "method"   : "sample", "individual", or "simultaneous"
            - "parameter": "lb", "dE", "a", "b", or "c"
            - "value"    : bootstrapped parameter value
        df_results (pandas.DataFrame):
            Table of parameter estimates from the three methods with columns:
            - "method"   : "sample", "individual", or "simultaneous"
            - "parameter": "lb", "dE", "a", "b", or "c"
            - "value"    : estimate of parameter
            - "error"    : uncertainty on estimate

    Returns:
        fig, ax (matplotlib Figure and Axes):
            The figure and axis used for the plot.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    # For each method/parameter: mean and std over bootstrap samples
    boot_summary = (
        boot_df
        .groupby(["method", "parameter"])["value"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "boot_value", "std": "boot_error"})
    )

    # base x-positions for parameters
    x_base = np.arange(len(param_order))
    width  = 0.20  # horizontal offset between methods

    # Plot original and bootstrap estimates with error bars
    for i, method in enumerate(method_order):
        # x positions for this method
        x_orig = x_base + (i - 1) * width          # Original estimates 
        x_boot = x_base + (i - 1) * width + 0.06   # Bootstrap estimates

        # Original estimates 
        sub_orig = df_results[df_results["method"] == method]
        vals_orig = [sub_orig[sub_orig["parameter"] == p]["value"].iloc[0]
                     for p in param_order]
        err_orig  = [sub_orig[sub_orig["parameter"] == p]["error"].iloc[0]
                     for p in param_order]

        ax.errorbar(
            x_orig,
            vals_orig,
            yerr=err_orig,
            fmt="x",                
            capsize=3,
            color=method_colors[method],
            label=f"{method}",
        )

        # ----- bootstrap estimates from boot_summary -----
        sub_boot = boot_summary[boot_summary["method"] == method]
        vals_boot = [sub_boot[sub_boot["parameter"] == p]["boot_value"].iloc[0]
                     for p in param_order]
        err_boot  = [sub_boot[sub_boot["parameter"] == p]["boot_error"].iloc[0]
                     for p in param_order]

        ax.errorbar(
            x_boot,
            vals_boot,
            yerr=err_boot,
            fmt="o",                 
            mfc="white",              
            capsize=3,
            color=method_colors[method],
            label=f"{method} (boot)",
        )

    # Set axis labels and title
    ax.set_xticks(x_base)
    ax.set_xticklabels(param_labels)
    ax.set_xlabel("Parameter")

    ax.set_ylabel("Parameter value")
    ax.set_title("Original vs bootstrap parameter estimates")

    ax.grid(alpha=0.3)
    ax.legend(title="estimation type")

    plt.tight_layout()

    fig.savefig("../figs/Figure4.2.pdf")

    return fig, ax
