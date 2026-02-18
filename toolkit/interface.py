from .Engine_v07 import (
    generate_z_score_problem,
    generate_t_test_problem,
    generate_independent_t_test_problem,
    generate_repeated_t_test_problem,
    generate_Independent_ANOVA,
    generate_pearson_correlation,
    generate_1_predictor_regression
)


def run_z_test(dataset=None,
                population_mean=0,population_std=15, n=10, seed=None,
                tx_effect=5, noise_sd=3,
                alpha=0.05, two_tailed=True):
    """
    Run a one-sample Z-test and return the dataset and results table.

    Wraps generate_z_score_problem from the statistical engine. Used when
    the population standard deviation is known.

    Parameters
    ----------
    dataset : array-like or None, optional
        Pre-existing dataset to analyze. If None, data is generated.
    population_mean : float, optional
        Null hypothesis population mean. Default is 0.
    population_std : float, optional
        Known population standard deviation. Default is 15.
    n : int, optional
        Sample size for data generation. Default is 10.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    tx_effect : float, optional
        Treatment effect applied to generated data. Default is 5.
    noise_sd : float, optional
        Noise standard deviation for data generation. Default is 3.
    alpha : float, optional
        Significance level. Default is 0.05.
    two_tailed : bool, optional
        If True, two-tailed test. Default is True.

    Returns
    -------
    numpy.ndarray
        The dataset analyzed.
    pandas.DataFrame
        Results table with Z-test statistics and decision.
    """
    dataset, results_table = generate_z_score_problem(dataset=dataset,
                                                      population_mean=population_mean,
                                                      population_std=population_std,
                                                      n=n,
                                                      seed=seed,
                                                      tx_effect=tx_effect,
                                                      noise_sd=noise_sd,
                                                      alpha=alpha,
                                                      two_tailed=two_tailed)
    return dataset, results_table

def run_t_test(dataset=None,
                population_mean=0, population_std=15, n=10, seed=None,
                tx_effect=5, noise_sd=3,
                alpha=0.05, two_tailed=True):
    """
    Run a one-sample t-test and return the dataset and results table.

    Wraps generate_t_test_problem from the statistical engine. Used when
    the population standard deviation is unknown and estimated from the sample.

    Parameters
    ----------
    dataset : array-like or None, optional
        Pre-existing dataset to analyze. If None, data is generated.
    population_mean : float, optional
        Null hypothesis population mean. Default is 0.
    population_std : float, optional
        Population SD used only for data generation (not the test). Default is 15.
    n : int, optional
        Sample size for data generation. Default is 10.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    tx_effect : float, optional
        Treatment effect applied to generated data. Default is 5.
    noise_sd : float, optional
        Noise standard deviation for data generation. Default is 3.
    alpha : float, optional
        Significance level. Default is 0.05.
    two_tailed : bool, optional
        If True, two-tailed test. Default is True.

    Returns
    -------
    numpy.ndarray
        The dataset analyzed.
    pandas.DataFrame
        Results table with t-test statistics and decision.
    """
    dataset, results_table = generate_t_test_problem(dataset=dataset,
                                                    population_mean=population_mean,
                                                    population_std=population_std,
                                                    n=n,
                                                    seed=seed,
                                                    tx_effect=tx_effect,
                                                    noise_sd=noise_sd,
                                                    alpha=alpha,
                                                    two_tailed=two_tailed)
    return dataset, results_table

def run_independent_t_test(dataset1=None, dataset2=None,
        population_mean1=10, population_sd1=15, n1=10, seed1=None,
        population_mean2=20, population_sd2=15, n2=10, seed2=None,
        alpha=0.05, two_tailed=True):
    """
    Run an independent-samples t-test and return both datasets and results.

    Wraps generate_independent_t_test_problem from the statistical engine.
    Compares the means of two independent groups using pooled variance.

    Parameters
    ----------
    dataset1 : array-like or None, optional
        Pre-existing data for group 1. If None, data is generated.
    dataset2 : array-like or None, optional
        Pre-existing data for group 2. If None, data is generated.
    population_mean1 : float, optional
        Population mean for group 1 data generation. Default is 10.
    population_sd1 : float, optional
        Population SD for group 1. Default is 15.
    n1 : int, optional
        Sample size for group 1. Default is 10.
    seed1 : int or None, optional
        Random seed for group 1. Default is None.
    population_mean2 : float, optional
        Population mean for group 2 data generation. Default is 20.
    population_sd2 : float, optional
        Population SD for group 2. Default is 15.
    n2 : int, optional
        Sample size for group 2. Default is 10.
    seed2 : int or None, optional
        Random seed for group 2. Default is None.
    alpha : float, optional
        Significance level. Default is 0.05.
    two_tailed : bool, optional
        If True, two-tailed test. Default is True.

    Returns
    -------
    numpy.ndarray
        Dataset for group 1.
    numpy.ndarray
        Dataset for group 2.
    pandas.DataFrame
        Results table with independent t-test statistics and decision.
    """
    dataset1, dataset2, results_table = generate_independent_t_test_problem(
        dataset1=dataset1,
        dataset2=dataset2,
        population_mean1=population_mean1,
        population_sd1=population_sd1,
        n1=n1,
        seed1=seed1,
        population_mean2=population_mean2,
        population_sd2=population_sd2,
        n2=n2,
        seed2=seed2,
        alpha=alpha,
        two_tailed=two_tailed
    )
    return dataset1, dataset2, results_table

def run_repeated_t_test(pre_dataset=None, post_dataset=None,
        population_mean=0, population_std=15, n=10, seed=None,
        tx_effect=5, noise_sd=3,
        alpha=0.05, two_tailed=True):
    """
    Run a repeated-measures (paired) t-test and return both datasets and results.

    Wraps generate_repeated_t_test_problem from the statistical engine.
    Compares pre- and post-treatment scores for the same participants using
    difference scores.

    Parameters
    ----------
    pre_dataset : array-like or None, optional
        Baseline (pre-treatment) scores. If None, data is generated.
    post_dataset : array-like or None, optional
        Post-treatment scores. If None, derived from pre_dataset with a treatment effect.
    population_mean : float, optional
        Population mean for baseline data generation. Default is 0.
    population_std : float, optional
        Population SD for baseline data generation. Default is 15.
    n : int, optional
        Number of participants. Default is 10.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    tx_effect : float, optional
        Mean treatment effect. Default is 5.
    noise_sd : float, optional
        Noise SD applied to the treatment. Default is 3.
    alpha : float, optional
        Significance level. Default is 0.05.
    two_tailed : bool, optional
        If True, two-tailed test. Default is True.

    Returns
    -------
    numpy.ndarray
        Pre-treatment dataset.
    numpy.ndarray
        Post-treatment dataset.
    pandas.DataFrame
        Results table with paired t-test statistics and decision.
    """
    predataset, postdataset, results_table = generate_repeated_t_test_problem(
        pre_dataset=pre_dataset,
        post_dataset=post_dataset,
        population_mean=population_mean,
        population_std=population_std,
        n=n,
        seed=seed,
        tx_effect=tx_effect,
        noise_sd=noise_sd,
        alpha=alpha,
        two_tailed=two_tailed
    )
    return predataset, postdataset, results_table

def run_independent_anova(factors_dictionary={"A":3, "B":2},
                               n = 10, mean=10, std=2, alpha=0.05,
                               effect_size=2.0, seed=None):
    """
    Run an independent-groups ANOVA and return the dataset and ANOVA table.

    Wraps generate_Independent_ANOVA from the statistical engine. Supports
    both one-way and factorial designs specified through factors_dictionary.

    Parameters
    ----------
    factors_dictionary : dict, optional
        Maps factor names to their number of levels.
        Example: {"A": 3} for one-way, {"A": 3, "B": 2} for 3x2 factorial.
        Default is {"A": 3, "B": 2}.
    n : int, optional
        Number of observations per cell. Default is 10.
    mean : float, optional
        Grand population mean. Default is 10.
    std : float, optional
        Within-cell population standard deviation. Default is 2.
    alpha : float, optional
        Significance level. Default is 0.05.
    effect_size : float, optional
        SD of random mean shifts per cell (controls between-group variability).
        Default is 2.0.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    pandas.DataFrame
        Long-format dataset with scores and factor membership columns.
    pandas.DataFrame
        ANOVA summary table with SS, df, MS, F, and p-value per source.
    """
    df, anova_table = generate_Independent_ANOVA(factors_dictionary=factors_dictionary,
                                                 n=n, mean=mean, std=std, alpha=alpha,
                                                 effect_size=effect_size, seed=seed)
    return df, anova_table

def run_pearson_correlation(x_dataset=None, y_dataset=None,
                                 x_mean=10, x_std=1, y_mean=20, y_std=3,
                                 ro=0, n=10, alpha=0.05, seed=None, two_tailed=True):
    """
    Run a Pearson correlation analysis and return both datasets and results.

    Wraps generate_pearson_correlation from the statistical engine.
    Tests the linear relationship between two variables.

    Parameters
    ----------
    x_dataset : array-like or None, optional
        Pre-existing X variable data. If None, data is generated.
    y_dataset : array-like or None, optional
        Pre-existing Y variable data. If None, data is generated.
    x_mean : float, optional
        Population mean for X generation. Default is 10.
    x_std : float, optional
        Population SD for X generation. Default is 1.
    y_mean : float, optional
        Population mean for Y generation. Default is 20.
    y_std : float, optional
        Population SD for Y generation. Default is 3.
    ro : float, optional
        Null hypothesis population correlation (rho_0). Default is 0.
    n : int, optional
        Sample size. Default is 10.
    alpha : float, optional
        Significance level. Default is 0.05.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    two_tailed : bool, optional
        If True, two-tailed test. Default is True.

    Returns
    -------
    numpy.ndarray
        X dataset analyzed.
    numpy.ndarray
        Y dataset analyzed.
    pandas.DataFrame
        Results table with correlation statistics and decision.
    """
    x_dataset, y_dataset, results_table = generate_pearson_correlation(
        x_dataset=x_dataset,
        y_dataset=y_dataset,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        ro=ro,
        n=n,
        alpha=alpha,
        seed=seed,
        two_tailed=two_tailed
    )
    return x_dataset, y_dataset, results_table

def run_1_predictor_regression(x_dataset=None, y_dataset=None,
                                    x_mean=10, x_std=1, y_mean=20, y_std=3,
                                    n=10, alpha=0.05, seed=None):
    """
    Run a simple (one-predictor) linear regression and return datasets, table, and equation.

    Wraps generate_1_predictor_regression from the statistical engine.
    Fits a least-squares regression line predicting Y from X and tests
    model significance with an F test.

    Parameters
    ----------
    x_dataset : array-like or None, optional
        Pre-existing predictor (X) data. If None, data is generated.
    y_dataset : array-like or None, optional
        Pre-existing outcome (Y) data. If None, data is generated.
    x_mean : float, optional
        Population mean for X generation. Default is 10.
    x_std : float, optional
        Population SD for X generation. Default is 1.
    y_mean : float, optional
        Population mean for Y generation. Default is 20.
    y_std : float, optional
        Population SD for Y generation. Default is 3.
    n : int, optional
        Sample size. Default is 10.
    alpha : float, optional
        Significance level. Default is 0.05.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    numpy.ndarray
        Y (outcome) dataset analyzed.
    numpy.ndarray
        X (predictor) dataset analyzed.
    pandas.DataFrame
        Regression ANOVA table (Regression, Residual, Total rows).
    str
        Regression equation string in the format "Y=bX+a".
    """
    y_dataset, x_dataset, results_table, regression_equation = generate_1_predictor_regression(
        x_dataset=x_dataset,
        y_dataset=y_dataset,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        n=n,
        alpha=alpha,
        seed=seed
    )
    return y_dataset, x_dataset, results_table, regression_equation

