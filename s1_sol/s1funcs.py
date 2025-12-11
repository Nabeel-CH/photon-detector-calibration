"""Function definitions for solution.ipynb"""

import numpy as np
import json
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares, UnbinnedNLL

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
    Update a selected section of results.json with parameter values and errors.

    Parameters:
        section_key (str): The section to update.
                           Must be one of "sample_ests", "individual_fits",
                           or "simultaneous_fit".
        lb, dE, a, b, c (float): Parameter values.
        lb_err, dE_err, a_err, b_err, c_err (float): Parameter uncertainties.
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


########################## Models ##########################

def mu_model(E0, lb, dE):
    """
    Mean model μ_E(E0) = λ E0 + Δ.

    Parameters:
        E0: True energy 
        lb: λ.
        dE: Δ.

    Returns:
        Mean μ_E(E0).
    """
    return lb * E0 + dE

# define the model for the width
def sigma_model(E0, a, b, c):
    """
    Width model: σ_E(E0) = E0 * sqrt((a / sqrt(E0))^2 + (b / E0)^2 + c^2).

    Parameters:
        E0, a, b, c: model parameters.

    Returns:
        Width σ_E(E0).
    """
    return E0 * np.sqrt((a / np.sqrt(E0))**2 + (b / E0)**2 + c**2)

#define normal distribution PDF
def normal_pdf(x, mu, sigma):
    """
    Normal probability density function (PDF).

    Parameters:
        x: Points at which to evaluate the PDF.
        mu: Mean of the normal distribution.
        sigma: Standard deviation.

    Returns:
        PDF values at x.
    """
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

#define the model
def sim_pdf(data, lb, dE, a, b, c):
    """
    Gaussian PDF for simultaneous likelihood.

    Parameters:
        data (array): array with
                           data[0] = E,
                           data[1] = E0.
        lb (float)
        dE (float)
        a (float)
        b (float)
        c (float)

    Returns:
        PDF values for provided parameters.
    """

    # unpack data
    E  = data[0]
    E0 = data[1]

    # mean 
    mu = lb * E0 + dE

    # width 
    term = (a / np.sqrt(E0))**2 + (b / E0)**2 + c**2
    sigma = E0 * np.sqrt(term)

    # Gaussian PDF 
    pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((E - mu) / sigma) ** 2
    )

    return pdf


########################## Q1 functions ##########################

def sample_est(sample_df):
    """
    For each unique E_true value in sample_df, compute sample estimates of the mean
    and standard deviation of E_rec, along with their errors.

    Parameters:
        sample_df (pandas.DataFrame): Input data with columns
                                      "E_true" and "E_rec".

    Returns:
        pandas.DataFrame: Summary table with columns
                          "E_true", "N",
                          "mu_est", "mu_err",
                          "sigma_est", "sigma_err".
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
    Used for Q1(iv) and Q2(ii)

    Parameters:
        summary (Pandas.DataFrame): Columns 
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

    # set limit: lb > 0 
    mean_minuit.limits["lb"] = (0.0, None)

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
    width_minuit = Minuit(width_cost, a=0.3, b=1.0, c=0.05)

    # set limits: a, b, c > 0 
    width_minuit.limits["a"] = (0.0, None)
    width_minuit.limits["b"] = (0.0, None)
    width_minuit.limits["c"] = (0.0, None)

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
                       N_BOOT=1000,
                       random_state=None):
    """
    Parametric bootstrap for Q1 part 4.

    Parameters:
        E_plot      : E0 values
        mean_params : array [lb, dE]
        mean_cov    : covariance matrix for (lb, dE)
        width_params: array [a, b, c]
        width_cov   : covariance matrix for (a, b, c)
        N_BOOT      : number of bootstraps (default 1000)
        random_state: int or None, optional
                      Seed for the random number generator to make the
                      bootstrap reproducible. If None, use default RNG.

    Returns:
        mu_band    : 1D array, std dev of μ(E0) - E0 over bootstraps
        sigma_band : 1D array, std dev of σ(E0)/E0 over bootstraps
    """

    # Make sure inputs are numpy arrays
    mean_params  = np.array(mean_params)
    width_params = np.array(width_params)
    mean_cov     = np.array(mean_cov)
    width_cov    = np.array(width_cov)

    # Local random number generator (for reproducibility if random_state is set)
    rng = np.random.default_rng(random_state)

    # Lists to store all bootstrap curves
    mu_boot_curves    = []
    sigma_boot_curves = []

    # Draw new parameter sets from the covariance matrices
    for _ in range(N_BOOT):
        # draw one random parameter set for each model
        lb_bs, dE_bs = rng.multivariate_normal(mean_params,  mean_cov)      
        a_bs,  b_bs, c_bs = rng.multivariate_normal(width_params, width_cov)

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


########################## Q2 functions ##########################

def indiv_MLsummary(sample_df):
    """
    For each E_true in sample_df, do an unbinned ML Gaussian fit
    to the E_rec values and return a summary table.
    Perform individual unbinned ML Gaussian fits for each true energy E_true.

    For each distinct E_true, fit a normal PDF to the the
    E_rec values using UnbinnedNLL + Minuit.

    Parameters:
        sample_df (pandas.DataFrame): Input data with columns
                                      "E_true" and "E_rec".

    Returns:
        indiv_summary (pandas.DataFrame): Summary table with one row per E_true with columns
                          "E_true", "mu_est", "mu_err",
                          "sigma_est", "sigma_err".
    """

    # Empty list to store results
    results_list = []  # store one row per E0 here

    # Go over each E0 value
    for E0 in sample_df["E_true"].unique():

        # Select all E_rec values only with this E0
        data = sample_df[sample_df["E_true"] == E0]["E_rec"]

        # initial guesses using sample mean/std
        mu_initial    = np.mean(data)
        sigma_initial = np.std(data, ddof=1)

        # build the negative log-likelihood
        cost = UnbinnedNLL(data, normal_pdf)

        # set up Minuit with starting values
        m = Minuit(cost, mu=mu_initial, sigma=sigma_initial)

        # set limit: 
        # sigma > 0 as sigma must be positive
        # mu > 0 as energies are positive
        m.limits["sigma"] = (0.0, None)
        m.limits["mu"] = (0.0, None)


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


########################## Q3 functions ##########################

def simultaneous_ML_fit(sample_df):
    """
    Simultaneous unbinned ML fit for all events.
    Fits all (E_rec, E_true) pairs at once using the likihood function defined in `sim_pdf` 
    and UnbinnedNLL + Minuit, to estimate all parameters.

    Parameters:
        sample_df (pandas.DataFrame): Input data with columns
                                      "E_true" and "E_rec".

    Returns:
             lb_sim, lb_err,
             dE_sim, dE_err,
             a_sim, a_err,
             b_sim, b_err,
             c_sim, c_err,
             covariance matrix
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
        lb=1.0,
        dE=0.0,
        a=0.3,
        b=1.0,
        c=0.05,
    )

    # set limits: all parameters > 0
    sim_minuit.limits["lb"] = (0.0, None)
    sim_minuit.limits["a"]  = (0.0, None)
    sim_minuit.limits["b"]  = (0.0, None)
    sim_minuit.limits["c"]  = (0.0, None)

    # Minimise the negative log-likelihood and get errors
    sim_minuit.migrad()
    sim_minuit.hesse()

    # get fitted values
    lb_sim   = sim_minuit.values["lb"]
    dE_sim = sim_minuit.values["dE"]
    a_sim     = sim_minuit.values["a"]
    b_sim     = sim_minuit.values["b"]
    c_sim     = sim_minuit.values["c"]

    lb_err   = sim_minuit.errors["lb"]
    dE_err = sim_minuit.errors["dE"]
    a_err     = sim_minuit.errors["a"]
    b_err     = sim_minuit.errors["b"]
    c_err     = sim_minuit.errors["c"]

    return (
        lb_sim,   lb_err,
        dE_sim, dE_err,
        a_sim,     a_err,
        b_sim,     b_err,
        c_sim,     c_err,
        sim_minuit.covariance
    )

def bootstrap_bands_q3(E_plot, param_mean, param_cov, N_BOOT=1000, random_state=None):
    """
    Parametric bootstrap for the simultaneous fit (Q3).

    Parameters:
        E_plot (array):
            E0 where the curves are evaluated.
        param_mean (array):
            Parameter values [lb, dE, a, b, c].
        param_cov (array):
            Covariance matrix for [lb, dE, a, b, c].
        N_BOOT (int, optional):
            Number of bootstraps (default 1000).
        random_state (int or None, optional):
            Seed for the random number generator to make the bootstrap
            reproducible. If None, use the default RNG.
        
    Returns:
        mu_band (numpy.ndarray):
            Standard deviation of (μ(E0) − E0) across bootstrap replicas, at each E0 in E_plot.
        sigma_band (numpy.ndarray):
            Standard deviation of (σ(E0)/E0) across bootstrap replicas, at each E0 in E_plot.
    """

    param_mean = np.array(param_mean)
    param_cov  = np.array(param_cov)

    # Local RNG for reproducibility
    rng = np.random.default_rng(random_state)

    mu_boot_curves    = []
    sigma_boot_curves = []

    for _ in range(N_BOOT):
        lb_b, dE_b, a_b, b_b, c_b = rng.multivariate_normal( 
            param_mean, param_cov
        )

        mu_boot_curves.append(
            mu_model(E_plot, lb_b, dE_b) - E_plot
        )
        sigma_boot_curves.append(
            sigma_model(E_plot, a_b, b_b, c_b) / E_plot
        )

    mu_boot_curves    = np.array(mu_boot_curves)
    sigma_boot_curves = np.array(sigma_boot_curves)

    mu_band    = mu_boot_curves.std(axis=0)
    sigma_band = sigma_boot_curves.std(axis=0)

    return mu_band, sigma_band

def load_results_table(filename="../results.json"):
    """
    Load the parameter estimates and errors from results.json
    and return them as a tidy DataFrame.

    Parameters:
        filename (str, optional):
            Path to the JSON results file. Defaults to "../results.json".

    Returns:
        df_results (pandas.DataFrame):
            DataFrame with columns:
              - 'method'   : 'sample', 'individual', or 'simultaneous'
              - 'parameter': 'lb', 'dE', 'a', 'b', or 'c'
              - 'value'    : values of the parameters
              - 'error'    : uncertainty of the parameters
    """
    # Load the JSON file
    with open(filename, "r") as f:
        results = json.load(f)

    rows = []

    method_names = {
        "sample_ests":      "sample",
        "individual_fits":  "individual",
        "simultaneous_fit": "simultaneous",
    }

    params = ["lb", "dE", "a", "b", "c"]

    for key_json, method_label in method_names.items():
        vals = results[key_json]["values"]
        errs = results[key_json]["errors"]

        for p in params:
            rows.append({
                "method":   method_label,
                "parameter": p,
                "value":    vals[p],
                "error":    errs[p],
            })

    # make table
    df_results = pd.DataFrame(rows)

    return df_results

########################## Q4 functions ##########################

def run_bootstrap_all_methods(sample_df, N_BOOT=2500, random_state=30):
    """
    Run the non-parametric bootstrap for all three methods
    (sample, individual, simultaneous) and return a DataFrame of bootstrap parameter values.

    Parameters:
        sample_df (pandas.DataFrame):
            Data with columns "E_true" and "E_rec".
        N_BOOT (int, optional):
            Number of bootstrap resamples (default 2500).
        random_state (int, optional):
            Seed for the random number generator to make the
            bootstrap reproducible (default 30).

    Returns:
        boot_df (pandas.DataFrame):
            Bootstrap results with columns:
              - "method"   : "sample", "individual", or "simultaneous"
              - "parameter": "lb", "dE", "a", "b", or "c"
              - "value"    : bootstrapped parameter value
    """
    # set up bootstrap
    N = len(sample_df)  # original sample size

    # fixed seed for reproducibility
    rng = np.random.default_rng(random_state)

    methods = ["sample", "individual", "simultaneous"]
    params  = ["lb", "dE", "a", "b", "c"]

    # store bootstrapped parameter values
    boot_results = {m: {p: [] for p in params} for m in methods}

    # bootstrap loop
    for i in range(N_BOOT):
        # sample rows with replacement
        idx   = rng.integers(0, N, size=N)
        df_bs = sample_df.iloc[idx].reset_index(drop=True)

        ##Method 1: sample estimates##
        summary_bs = sample_est(df_bs)

        (
            lb_s,  lb_err_s,
            dE_s,  dE_err_s,
            a_s,   a_err_s,
            b_s,   b_err_s,
            c_s,   c_err_s,
            cov_mu_s, cov_sigma_s,
        ) = fit_sample_method(summary_bs)

        boot_results["sample"]["lb"].append(lb_s)
        boot_results["sample"]["dE"].append(dE_s)
        boot_results["sample"]["a"].append(a_s)
        boot_results["sample"]["b"].append(b_s)
        boot_results["sample"]["c"].append(c_s)

        ##Method 2: individual fits##
        indiv_bs = indiv_MLsummary(df_bs)

        (
            lb_i,  lb_err_i,
            dE_i,  dE_err_i,
            a_i,   a_err_i,
            b_i,   b_err_i,
            c_i,   c_err_i,
            cov_mu_i, cov_sigma_i,
        ) = fit_sample_method(indiv_bs)

        boot_results["individual"]["lb"].append(lb_i)
        boot_results["individual"]["dE"].append(dE_i)
        boot_results["individual"]["a"].append(a_i)
        boot_results["individual"]["b"].append(b_i)
        boot_results["individual"]["c"].append(c_i)

        ##Method 3: simultaneous fit##
        (
            lb_sim,   lb_err_sim,
            dE_sim,   dE_err_sim,
            a_sim,    a_err_sim,
            b_sim,    b_err_sim,
            c_sim,    c_err_sim,
            sim_cov,
        ) = simultaneous_ML_fit(df_bs)

        boot_results["simultaneous"]["lb"].append(lb_sim)
        boot_results["simultaneous"]["dE"].append(dE_sim)
        boot_results["simultaneous"]["a"].append(a_sim)
        boot_results["simultaneous"]["b"].append(b_sim)
        boot_results["simultaneous"]["c"].append(c_sim)

    # Put results into a DataFrame
    rows = []
    for method in methods:
        for p in params:
            for val in boot_results[method][p]:
                rows.append({"method": method, "parameter": p, "value": val})

    boot_df = pd.DataFrame(rows)

    return boot_df

