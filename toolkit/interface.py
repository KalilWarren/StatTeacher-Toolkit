from .Engine_v07 import (
    generate_z_score_problem,
    generate_t_test_problem,
    generate_independent_t_test_problem,
    generate_repeated_t_test_problem,
    generate_Independent_ANOVA,
    generate_pearson_correlation,
    generate_1_predictor_regression
)
def run_z_test():
    dataset, results_table = generate_z_score_problem()
    return dataset, results_table

def run_t_test():
    dataset, results_table = generate_t_test_problem()
    return dataset, results_table

def run_independent_t_test():
    dataset1, dataset2, results_table = generate_independent_t_test_problem()
    return dataset1, dataset2, results_table

def run_repeated_t_test():
    predataset, postdataset, results_table = generate_repeated_t_test_problem()
    return predataset, postdataset, results_table

def run_independent_anova():
    df, anova_table = generate_Independent_ANOVA()
    return df, anova_table

def run_pearson_correlation():
    x_dataset, y_dataset, results_table = generate_pearson_correlation()
    return x_dataset, y_dataset, results_table

def run_1_predictor_regression():
    y_dataset, x_dataset, results_table, regression_equation = generate_1_predictor_regression()
    return y_dataset, x_dataset, results_table, regression_equation

