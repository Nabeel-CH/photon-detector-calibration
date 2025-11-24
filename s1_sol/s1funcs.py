"""Dummy functionality of s1_sol"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from iminuit.cost import UnbinnedNLL


def make_me_a_plot(size, bins=50):
    """
    A a function to demonstrate making a plot

    Parameters
    ----------
    size : int
        Size of random sample to produce
    bins : int, optional
        Number of bins for the plotted histogram

    Returns
    -------
    fig, ax
        The matplotlib Figure and Axes
    """

    x = np.random.normal(size=size)

    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)

    return fig, ax


def make_results_json(filename):
    """
    Make an example of the results.json output file

    Parameters
    ----------
    filename : str
        Path to save the file to

    """

    ex_dict = {
        "sample_ests": {
            "values": {
                "lb": np.nan,  # estimated value of lambda
                "dE": np.nan,  # estimated value of DeltaE
                "a": np.nan,  # estimated value of a
                "b": np.nan,  # estimated value of b
                "c": np.nan,  # estimated value of c
            },
            "errors": {
                "lb": np.nan,  # estimated error of lambda
                "dE": np.nan,  # estimated error of DeltaE
                "a": np.nan,  # estimated error of a
                "b": np.nan,  # estimated error of b
                "c": np.nan,  # estimated error of c
            },
        },
        "individual_fits": {
            "values": {
                "lb": np.nan,  # estimated value of lambda
                "dE": np.nan,  # estimated value of DeltaE
                "a": np.nan,  # estimated value of a
                "b": np.nan,  # estimated value of b
                "c": np.nan,  # estimated value of c
            },
            "errors": {
                "lb": np.nan,  # estimated error of lambda
                "dE": np.nan,  # estimated error of DeltaE
                "a": np.nan,  # estimated error of a
                "b": np.nan,  # estimated error of b
                "c": np.nan,  # estimated error of c
            },
        },
        "simultaneous_fit": {
            "values": {
                "lb": np.nan,  # estimated value of lambda
                "dE": np.nan,  # estimated value of DeltaE
                "a": np.nan,  # estimated value of a
                "b": np.nan,  # estimated value of b
                "c": np.nan,  # estimated value of c
            },
            "errors": {
                "lb": np.nan,  # estimated error of lambda
                "dE": np.nan,  # estimated error of DeltaE
                "a": np.nan,  # estimated error of a
                "b": np.nan,  # estimated error of b
                "c": np.nan,  # estimated error of c
            },
        },
    }
    with open(filename, "w") as f:
        json.dump(ex_dict, f, indent=4)

def update_results_json(section_key,
                   lb, lb_err,
                   dE, dE_err,
                   a, a_err,
                   b, b_err,
                   c, c_err):
    """
    Update the chosen section of results.json with the given
    parameter values and errors.

    section_key must be one of:
        'sample_ests'
        'individual_fits'
        'simultaneous_fit'
    """

    filename = "../results.json"

    # Load the JSON file
    with open(filename, "r") as f:
        results = json.load(f)

    # Select the correct section
    section = results[section_key]

    # Update values
    section["values"]["lb"] = float(lb)
    section["values"]["dE"] = float(dE)
    section["values"]["a"]  = float(a)
    section["values"]["b"]  = float(b)
    section["values"]["c"]  = float(c)

    # Update errors
    section["errors"]["lb"] = float(lb_err)
    section["errors"]["dE"] = float(dE_err)
    section["errors"]["a"]  = float(a_err)
    section["errors"]["b"]  = float(b_err)
    section["errors"]["c"]  = float(c_err)

    # Save back to results.json
    with open(filename, "w") as f:
        json.dump(results, f, indent=4, allow_nan=True)

    print(f"Updated '{section_key}' in results.json")


##### Models #####
# define the model for the mean
def mu_model(E0, lb, dE):
    return lb * E0 + dE

# define the model for the width
def sigma_model(E0, a, b, c):
    return E0 * np.sqrt((a / np.sqrt(E0))**2 + (b / E0)**2 + c**2)

#define normal distribution PDF
def normal_pdf(x, mu, sigma):
    """
    Normal distribution PDF 
    """
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

#define the model
def sim_pdf(data, lam, Delta, a, b, c):
    """
    Gaussian PDF for E given E0.
    """

    # unpack data
    E  = data[0]
    E0 = data[1]

    # mean 
    mu = lam * E0 + Delta

    # width 
    term = (a / np.sqrt(E0))**2 + (b / E0)**2 + c**2
    sigma = E0 * np.sqrt(term)

    # Gaussian PDF 
    pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((E - mu) / sigma) ** 2
    )

    return pdf


###Question 1####
def sample_est(sample_df):
    """
    For each E_true value in sample_df, compute:
      - N          : number of events
      - mu_est     : sample mean of E_rec
      - SE_mu      : standard error on the mean
      - sigma_est : sample standard deviation of E_rec
      - SE_sigma   : standard error on the standard deviation

    Returns a DataFrame called summary with one row per E_true
    """
    # Empty list to store results
    results = []

    # Go over each E0 value
    for E0 in sample_df["E_true"].unique():

        # Select all E_rec values only with this E0
        data = sample_df[sample_df["E_true"] == E0]["E_rec"]

        # Number of samples
        N = len(data)

        # Sample mean
        mu_est = np.mean(data)

        # Sample standard deviation (ddof=1 for sample std)
        sigma_est = np.std(data, ddof=1)

        # Standard error on the mean
        SE_mu = sigma_est / np.sqrt(N)

        # Standard error on the standard deviation
        SE_sigma = sigma_est / np.sqrt(2 * (N - 1))

        # Store the values 
        results.append([E0, N, mu_est, SE_mu, sigma_est, SE_sigma])
    # Store data in a summary table
    summary = pd.DataFrame(results, columns=[
        "E_true", "N", "mu_est", "mu_err", "sigma_est", "sigma_err"
    ])

    return summary

def fit_sample_method(summary):
    """
    Fit parameters using sample estimates with Least Squares. Returning the results and 
    the covariance matrices (used later for bootstrap bands).

    Input:
        summary : DataFrame with columns
                  E_true, mu_est, mu_err, sigma_est, sigma_err
    Returns:
        lb, lb_err,
        dE, dE_err,
        a, a_err,
        b, b_err,
        c, c_err,
        mean_minuit.covariance,   
        width_minuit.covariance
    """

    # x-values: E0
    E0 = summary["E_true"].to_numpy()

    # Get means values for mean fit
    mu_values   = summary["mu_est"].to_numpy()   # measured sample means
    mu_errors   = summary["mu_err"].to_numpy()     # errors on the means

    # Get standard deviation values for width fit
    sigma_values = summary["sigma_est"].to_numpy()  # measured sample widths
    sigma_errors = summary["sigma_err"].to_numpy()    # errors on the widths
    # LeastSquares cost function for mean fit
    mean_cost = LeastSquares(E0, mu_values, mu_errors, mu_model)

    # create a Minuit object with starting guesses
    mean_minuit = Minuit(mean_cost, lb=1.0, dE=0.0)

    # run the minimisation
    mean_minuit.migrad()
    mean_minuit.hesse()

    # extract best-fit parameters and their errors 
    lb       = mean_minuit.values["lb"]
    dE       = mean_minuit.values["dE"]
    lb_err   = mean_minuit.errors["lb"]
    dE_err   = mean_minuit.errors["dE"]

    # LeastSquares cost function for width fit
    width_cost = LeastSquares(E0, sigma_values, sigma_errors, sigma_model)

    # Minuit object with starting guesses
    width_minuit = Minuit(width_cost, a=1.0, b=1.0, c=0.1)

    # run the minimisation
    width_minuit.migrad()
    width_minuit.hesse()

    # extract best-fit parameters and their errors
    a      = width_minuit.values["a"]
    b       = width_minuit.values["b"]
    c       = width_minuit.values["c"]
    a_err   = width_minuit.errors["a"]
    b_err   = width_minuit.errors["b"]
    c_err   = width_minuit.errors["c"]
    # Return all fitted parameters and their errors
    # and covariance matrices
    return (
        lb, lb_err,
        dE, dE_err,
        a, a_err,
        b, b_err,
        c, c_err,
        mean_minuit.covariance,   
        width_minuit.covariance,
    )


# Bootstrap bands for Q1 part 4
def bootstrap_bands_q1(E_plot,
                       mean_params,
                       mean_cov,
                       width_params,
                       width_cov,
                       N_BOOT=1000):
    """
    Parametric bootstrap for Q1 part 4.

    Inputs:
        E_plot      : 1D array of E0 values where you want the curves
        mean_params : array-like [lb, dE]
        mean_cov    : covariance matrix for (lb, dE)
        width_params: array-like [a, b, c]
        width_cov   : covariance matrix for (a, b, c)
        N_BOOT      : number of bootstrap replicas (default 1000)

    Returns:
        mu_band    : 1D array, std dev of μ(E0) - E0 over bootstraps
        sigma_band : 1D array, std dev of σ(E0)/E0 over bootstraps
    """

    # Make sure inputs are numpy arrays
    mean_params  = np.array(mean_params)
    width_params = np.array(width_params)
    mean_cov     = np.array(mean_cov)
    width_cov    = np.array(width_cov)

    # Lists to store all bootstrap curves
    mu_boot_curves    = []
    sigma_boot_curves = []

    # Draw new parameter sets from the covariance matrices
    for _ in range(N_BOOT):
        # draw one random parameter set for each model
        lb_bs, dE_bs = np.random.multivariate_normal(mean_params,  mean_cov)
        a_bs,  b_bs, c_bs = np.random.multivariate_normal(width_params, width_cov)

        # compute and save the corresponding curves
        mu_boot_curves.append(
            mu_model(E_plot, lb_bs, dE_bs) - E_plot
        )
        sigma_boot_curves.append(
            sigma_model(E_plot, a_bs, b_bs, c_bs) / E_plot
        )

    # turn into numpy arrays to calculate std devs
    mu_boot_curves    = np.array(mu_boot_curves)
    sigma_boot_curves = np.array(sigma_boot_curves)

    # ±1sigma band height at each E_plot point
    mu_band    = mu_boot_curves.std(axis=0)
    sigma_band = sigma_boot_curves.std(axis=0)

    return mu_band, sigma_band



####### Question 2 #####
def indiv_MLsummary(sample_df):
    """
    For each E_true in sample_df, do an unbinned ML Gaussian fit
    to the E_rec values and return a summary table.

    The returned DataFrame indiv_summary has columns:
      - E_true
      - mu_est
      - mu_err
      - sigma_est
      - sigma_err
    """

    # Empty list to store results
    results_list = []  # store one row per E0 here

    # Go over each E0 value
    for E0 in sample_df["E_true"].unique():

        # Select all E_rec values only with this E0
        data = sample_df[sample_df["E_true"] == E0]["E_rec"]

        # quick initial guesses using sample mean/std
        mu_initial    = np.mean(data)
        sigma_initial = np.std(data, ddof=1)

        # build the negative log-likelihood
        cost = UnbinnedNLL(data, normal_pdf)

        # set up Minuit with starting values
        m = Minuit(cost, mu=mu_initial, sigma=sigma_initial)

        # run the minimiser
        m.migrad()
        m.hesse()

        # best-fit values and errors
        mu_fit    = m.values["mu"]
        sigma_fit = m.values["sigma"]
        mu_err    = m.errors["mu"]
        sigma_err = m.errors["sigma"]

        # store results in a list
        results_list.append(
            {
                "E_true": E0,
                "mu_est": mu_fit,
                "mu_err": mu_err,
                "sigma_est": sigma_fit,
                "sigma_err": sigma_err,
            }
        )

    # Convert list of dicts into a DataFrame
    indiv_summary = pd.DataFrame(results_list)

    return indiv_summary


##### Question 3 #####
def simultaneous_ML_fit(sample_df):
    """
    Simultaneous unbinned ML fit for all events.

    Uses sim_pdf as the model PDF and returns:
        lam_sim, lam_err,
        Delta_sim, Delta_err,
        a_sim, a_err,
        b_sim, b_err,
        c_sim, c_err
    """

    # Get data for fit
    # Measured energy E and true energy E0 for every event
    E_data  = sample_df["E_rec"].to_numpy()
    E0_data = sample_df["E_true"].to_numpy()

    # Stack into shape (D, N) = (2, N), which is necessary for UnbinnedNLL function 
    data_all = np.vstack([E_data, E0_data])

    # Build the unbinned negative log-likelihood over all events.
    sim_cost = UnbinnedNLL(data_all, sim_pdf)

    # Starting guesses for the parameters
    sim_minuit = Minuit(
        sim_cost,
        lam=1.0,
        Delta=0.0,
        a=0.3,
        b=1.0,
        c=0.05,
    )

    # Minimise the negative log-likelihood and get errors
    sim_minuit.migrad()
    sim_minuit.hesse()

    # get fitted values
    lam_sim   = sim_minuit.values["lam"]
    Delta_sim = sim_minuit.values["Delta"]
    a_sim     = sim_minuit.values["a"]
    b_sim     = sim_minuit.values["b"]
    c_sim     = sim_minuit.values["c"]

    lam_err   = sim_minuit.errors["lam"]
    Delta_err = sim_minuit.errors["Delta"]
    a_err     = sim_minuit.errors["a"]
    b_err     = sim_minuit.errors["b"]
    c_err     = sim_minuit.errors["c"]

    return (
        lam_sim,   lam_err,
        Delta_sim, Delta_err,
        a_sim,     a_err,
        b_sim,     b_err,
        c_sim,     c_err,
        sim_minuit.covariance
    )

def bootstrap_bands_q3(E_plot, param_mean, param_cov, N_BOOT=500):
    """
    Parametric bootstrap for the simultaneous fit (Q3).

    param_mean : [lam, Delta, a, b, c]
    param_cov  : covariance matrix
    """

    param_mean = np.array(param_mean)
    param_cov  = np.array(param_cov)

    mu_boot_curves    = []
    sigma_boot_curves = []

    for _ in range(N_BOOT):
        lam_b, Delta_b, a_b, b_b, c_b = np.random.multivariate_normal(
            param_mean, param_cov
        )

        mu_boot_curves.append(
            mu_model(E_plot, lam_b, Delta_b) - E_plot
        )
        sigma_boot_curves.append(
            sigma_model(E_plot, a_b, b_b, c_b) / E_plot
        )

    mu_boot_curves    = np.array(mu_boot_curves)
    sigma_boot_curves = np.array(sigma_boot_curves)

    mu_band    = mu_boot_curves.std(axis=0)
    sigma_band = sigma_boot_curves.std(axis=0)

    return mu_band, sigma_band