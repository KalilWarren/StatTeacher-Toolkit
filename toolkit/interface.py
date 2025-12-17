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
    df, anova_table = generate_Independent_ANOVA(factors_dictionary=factors_dictionary,
                                                 n=n, mean=mean, std=std, alpha=alpha,
                                                 effect_size=effect_size, seed=seed)
    return df, anova_table

def run_pearson_correlation(x_dataset=None, y_dataset=None, 
                                 x_mean=10, x_std=1, y_mean=20, y_std=3,
                                 ro=0, n=10, alpha=0.05, seed=None, two_tailed=True):
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

def run_1_predictor_regression(x_dataset = None, y_dataset = None,
                                    x_mean=10, x_std=1, y_mean=20, y_std=3,
                                    n=10, alpha=0.05, seed=None):
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

