#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 19:08:56 2025

@author: kalilwarren
"""

import itertools
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import f
import pandas as pd

def generate_normal_data(mean=0, std=1, n=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data = np.random.normal(loc=mean, scale=std, size=n)
    return np.round(data)   # round to 1 decimal place

def _apply_treatment(pre_scores, effect=5, noise_sd=3):
    noise = np.random.normal(loc=0, scale=noise_sd, size=len(pre_scores))
    post_scores = pre_scores + effect + noise
    return np.round(post_scores, 1)

def _z_critical(alpha=0.05, two_tailed=True):
    if two_tailed:
        return norm.ppf(1 - alpha/2)
    else:
        return norm.ppf(1 - alpha)

def _t_critical(df, alpha=0.05, two_tailed=True):
    if two_tailed:
        return t.ppf(1 - alpha/2, df)
    else:
        return t.ppf(1 - alpha, df)

def _f_critical(df_Between, df_Within, alpha=0.05):
    return f.ppf(1 - alpha, df_Between, df_Within)

def _sum_of_squares(x):
    return np.sum((x - np.mean(x))**2)

def _sum_product(x, y):
    return np.sum((x-np.mean(x))*(y-np.mean(y)))

def _compute_factor_level_SS(df, C, dv="Scores"):
    factor_cols = [col for col in df.columns if col !=dv]
    
    ss_main={}
    
    for factor in factor_cols:
        T = df.groupby(factor)[dv].sum()
        n = df.groupby(factor)[dv].count()
        ss_factor = (T**2 / n).sum() - C
        ss_main[factor] = ss_factor

    return ss_main


def _build_independent_anova_table(
    factor_SS,
    factor_df,
    SS_Within,
    df_Within,
    SS_Total,
    SS_Between=None,
    df_Between=None,
    SS_Interaction=None,
    df_Interaction=None,
    alpha=0.05
):

    rows = []

    MS_Within = SS_Within / df_Within

    for factor in factor_SS:
        SS = factor_SS[factor]
        df_num = factor_df[factor]
        MS = SS / df_num
        F_value = MS / MS_Within
        p_value = 1 - f.cdf(F_value, df_num, df_Within)

        rows.append({
            "Source": factor,
            "SS": SS,
            "df": df_num,
            "MS": MS,
            "F": F_value,
            "p-value": p_value
        })
        
    if SS_Interaction is not None and df_Interaction is not None:
        MS_Int = SS_Interaction / df_Interaction
        F_Int = MS_Int / MS_Within
        p_Int = 1 - f.cdf(F_Int, df_Interaction, df_Within)

        rows.append({
            "Source": "Interaction",
            "SS": SS_Interaction,
            "df": df_Interaction,
            "MS": MS_Int,
            "F": F_Int,
            "p-value": p_Int
        })
        rows = [{
            "Source": "Between",
            "SS": SS_Between,
            "df": df_Between,
            "MS": None,
            "F": None,
            "p-value": None
            }] + rows

    rows.append({
        "Source": "Within",
        "SS": SS_Within,
        "df": df_Within,
        "MS": MS_Within,
        "F": None,
        "p-value": None
    })

    df_total = df_Within + sum(factor_df.values())
    if df_Interaction is not None:
        df_total += df_Interaction

    rows.append({
        "Source": "Total",
        "SS": SS_Total,
        "df": df_total,
        "MS": None,
        "F": None,
        "p-value": None
    })

    return pd.DataFrame(rows)


def generate_z_score_problem(dataset=None,
                            population_mean=0,population_std=15, n=10, seed=None,
                            tx_effect=5, noise_sd=3,
                            alpha=0.05, two_tailed=True):
    if dataset == None:
        dataset = generate_normal_data(population_mean, population_std, n, seed)
        dataset = _apply_treatment(dataset, tx_effect, noise_sd)
    n = len(dataset)
    sample_mean = np.mean(dataset)
    standard_error = population_std / np.sqrt(n)
    z = (sample_mean - population_mean) / standard_error
    cohen_d = (sample_mean - population_mean) / population_std
    
    z_critical =_z_critical(alpha, two_tailed)
    
    CI_Upper = sample_mean + z_critical * standard_error
    CI_Lower = sample_mean - z_critical * standard_error
    
    if abs(z) > z_critical:
        results = "Reject Null"
    else:
        results = "Fail to Reject Null"
    
    df_out = pd.DataFrame({
        "Statistic": ["N", "Population Mean", "Population SD", "Sample Mean", "Standard Error",
                      "Z Score", "Z_Critical", "Decision","Cohen's d", "95% CI Upper",
                      "95% CI Lower"],
        "Value": [n,
                  round(population_mean, 2),
                  round(population_std, 2),
                  round(sample_mean, 2),
                  round(standard_error, 2),
                  round(z, 2),
                  z_critical,
                  results,
                  round(cohen_d, 2),
                  round(CI_Upper, 2),
                  round(CI_Lower, 2)]
        
        })
    return dataset, df_out  

def generate_t_test_problem(dataset=None,
                            population_mean=0, population_std=15, n=10, seed=None,
                            tx_effect=5, noise_sd=3,
                            alpha=0.05, two_tailed=True):
    if dataset == None:
        dataset = generate_normal_data(population_mean, population_std, n, seed)
        dataset = _apply_treatment(dataset, tx_effect, noise_sd)
    sample_std = np.std(dataset)
    sample_mean = np.mean(dataset)
    standard_error = sample_std / np.sqrt(n)
    t = (sample_mean - population_mean) / standard_error
    cohen_d = (sample_mean - population_mean) / sample_std
    df = n - 1
    r_squared = (t**2) / (t**2 + df)

    t_critical =_t_critical(df, alpha, two_tailed)

    CI_Upper = sample_mean + t_critical * standard_error
    CI_Lower = sample_mean - t_critical * standard_error
    
    if abs(t) > t_critical:
        results = "Reject Null"
    else:
        results = "Fail to Reject Null"
    
    df_out = pd.DataFrame({
        "Statistic": ["N", "Sample SD", "Sample Mean", "Standard Error",
                      "t Score", "t_Critical", "Decision","Cohen's d", "R-Squared",
                      "95% CI Upper", "95% CI Lower"],
        "Value": [n,
                  round(sample_std, 2),
                  round(sample_mean, 2),
                  round(standard_error, 2),
                  round(t, 2),
                  t_critical,
                  results,
                  round(cohen_d, 2),
                  r_squared,
                  round(CI_Upper, 2),
                  round(CI_Lower, 2)]
        
        })
    return dataset, df_out  

def generate_independent_t_test_problem(
        dataset1=None, dataset2=None,
        population_mean1=10, population_sd1=15, n1=10, seed1=None,
        population_mean2=20, population_sd2=15, n2=10, seed2=None,
        alpha=0.05, two_tailed=True):
    if dataset1 == None:
        dataset1 = generate_normal_data(population_mean1, population_sd1, n1, seed1)
    if dataset2 == None:
        dataset2 = generate_normal_data(population_mean2, population_sd2, n2, seed2)
    sample_std1 = np.std(dataset1)
    sample_mean1 = np.mean(dataset1)
    sample_std2 = np.std(dataset2)
    sample_mean2 = np.mean(dataset2)
    df1 = n1 - 1
    df2 = n2 - 1
    df = df1 + df2
    sum_of_squares1 = _sum_of_squares(dataset1)
    sum_of_squares2 = _sum_of_squares(dataset2)
    sum_product = (sum_of_squares1 + sum_of_squares2) / (df1 + df2)
    standard_error = np.sqrt((sum_product / n1) + (sum_product / n2))
    t = (sample_mean1 - sample_mean2) / standard_error
    cohen_d = (sample_mean1 - sample_mean2) / np.sqrt(sum_product)
    r_squared = (t**2) / (t**2 + df)
    
    t_critical =_t_critical(df, alpha, two_tailed)
       
    CI_Upper = (sample_mean1 - sample_mean2) + t_critical * standard_error
    CI_Lower = (sample_mean1 - sample_mean2) - t_critical * standard_error
    
    if abs(t) > t_critical:
        results = "Reject Null"
    else:
        results = "Fail to Reject Null"
    
    df_out = pd.DataFrame({
        "Statistic": ["N1", "Sample SD1", "Sample Mean1", "df1", "Sum of Squares1",
                      "N2", "Sample SD2", "Sample Mean2", "df2", "Sum of Squares2", 
                      "Sum Of Product", "Standard Error", "t Score", "t_Critical", 
                      "Decision","Cohen's d", "R-Squared", "95% CI Upper", "95% CI Lower"],
        "Value": [n1,
                  round(sample_std1, 2),
                  round(sample_mean1, 2),
                  df1,
                  round(sum_of_squares1, 2), 
                  n2,
                  round(sample_std2, 2),
                  round(sample_mean2, 2),
                  df2,
                  round(sum_of_squares2, 2),
                  round(sum_product, 2),
                  round(standard_error, 2),
                  round(t, 2),
                  t_critical,
                  results,
                  round(cohen_d, 2),
                  r_squared,
                  round(CI_Upper, 2),
                  round(CI_Lower, 2)]
        
        })
    return dataset1, dataset2, df_out

def generate_repeated_t_test_problem(
        pre_dataset=None, post_dataset=None,
        population_mean=0, population_std=15, n=10, seed=None,
        tx_effect=5, noise_sd=3,
        alpha=0.05, two_tailed=True):
    
    if pre_dataset == None and post_dataset == None:
        pre_dataset = generate_normal_data(population_mean, population_std, n, seed)
        post_dataset = _apply_treatment(pre_dataset, tx_effect, noise_sd)
    
    differences = []
    for x1, x2 in zip(pre_dataset, post_dataset):
        d = x1 - x2
        differences.append(d)
    
    mean_differences = np.mean(differences)
    std_differences = np.std(differences)
    standard_error = std_differences / np.sqrt(n)
    t = mean_differences / standard_error
    df = n - 1
    cohen_d = mean_differences / std_differences
    r_squared = (t**2) / (t**2 + df)
    t_critical = _t_critical(df, alpha, two_tailed)
    
    CI_Upper = mean_differences + t_critical * standard_error
    CI_Lower = mean_differences - t_critical * standard_error
    
    if abs(t) > t_critical:
        results = "Reject Null"
    else:
        results = "Fail to Reject Null"
    
    df_out = pd.DataFrame({
        "Statistic": ["N", "Differences SD", "Differences Mean", "Standard Error",
                      "t Score", "t_Critical", "Decision","Cohen's d", "R-Squared",
                      "95% CI Upper", "95% CI Lower"],
        "Value": [n,
                  round(std_differences, 2),
                  round(mean_differences, 2),
                  round(standard_error, 2),
                  round(t, 2),
                  t_critical,
                  results,
                  round(cohen_d, 2),
                  r_squared,
                  round(CI_Upper, 2),
                  round(CI_Lower, 2)]
        
        })
    
    return pre_dataset, post_dataset, df_out

def generate_Independent_ANOVA(factors_dictionary={"A":3, "B":2},
                               n = 10, mean=10, std=2, alpha=0.05, 
                               effect_size=2.0, seed=None):

    factor_levels = {factor: [f"{factor}{i+1}" for i in range(levels)]
        for factor, levels in factors_dictionary.items()}

    all_cells = list(itertools.product(*factor_levels.values()))

    rows = []
    mean_shifts = {cell: np.random.normal(loc=0, scale=effect_size)
                   for cell in all_cells}
    
    for cell in all_cells:
        shift = mean_shifts[cell]
        cell_mean = mean + shift
        data = generate_normal_data(mean=cell_mean, std=std, n=n, seed=seed)

        for value in data:
            row = {"Scores": value}

            for factor_name, level in zip(factor_levels.keys(), cell):
                row[factor_name] = level

            rows.append(row)

    df = pd.DataFrame(rows)
    cells = len(all_cells)
    N = cells * n
    df_Total = N - 1
    SS_Total = _sum_of_squares(df["Scores"])
    df_Within = cells * (n - 1)
    
    G = 0
    for values in df["Scores"]:
        G += values
        
    C = (G**2) / N
    
    factor_cols = [col for col in df.columns if col != "Scores"]
    cell_SS = {}

    for cell_levels, group in df.groupby(factor_cols):
        ss = _sum_of_squares(group["Scores"])
        cell_SS[cell_levels] = ss
    
    SS_Within = 0
    for values in cell_SS.values():
        SS_Within += values
    
    SS_Between = SS_Total - SS_Within 
    df_Between = df_Total - df_Within
    
    if len(factors_dictionary) > 1:
        level_SS = _compute_factor_level_SS(df, C)   # SS for each level of each factor
        level_df = {factor: (levels - 1) for factor, levels in factors_dictionary.items()}
        
        SS_Interaction = SS_Between - sum(level_SS.values())
        df_Interaction = df_Between - sum(level_df.values())
        MS_Interaction = SS_Interaction / df_Interaction
        
        # Compute MS_Factor for each factor
        factor_MS = {
        factor: level_SS[factor] / level_df[factor]
        for factor in level_SS
        }
        
        factor_MS["Interaction"] = MS_Interaction
        anova_table =  _build_independent_anova_table(factor_SS=level_SS,
                                              factor_df=level_df,
                                              SS_Within=SS_Within,
                                              df_Within=df_Within,
                                              SS_Total=SS_Total,
                                              SS_Between=SS_Between,
                                              df_Between=df_Between,
                                              SS_Interaction=SS_Interaction,
                                              df_Interaction=df_Interaction,
                                              alpha=alpha
                                              )
        return df, anova_table

    elif len(factors_dictionary) == 1:
        factor = list(factors_dictionary.keys())[0]
        levels = factors_dictionary[factor]
        
        factor_df = {factor: levels - 1}
        factor_SS = {}
        factor_SS[factor] = SS_Between
        SS_Interaction = None
        df_Interaction = None
        
        anova_table = _build_independent_anova_table(factor_SS=factor_SS,
                                              factor_df=factor_df,
                                              SS_Within=SS_Within,
                                              df_Within=df_Within,
                                              SS_Total=SS_Total,
                                              SS_Interaction=SS_Interaction,
                                              df_Interaction=df_Interaction,
                                              alpha=alpha
                                              )
        return df, anova_table

    return "ANOVA Aborted: There's a problem with your factors dictionary."

def generate_pearson_correlation(x_dataset=None, y_dataset=None, 
                                 x_mean=10, x_std=1, y_mean=20, y_std=3,
                                 ro=0, n=10, alpha=0.05, seed=None, two_tailed=True):
    
    if x_dataset == None:
        x_dataset = generate_normal_data(x_mean, x_std, n, seed)
    if y_dataset == None:
        y_dataset = generate_normal_data(y_mean, y_std, n, seed)
        y_dataset = _apply_treatment(y_dataset, effect=10)
    
    mean_x = np.mean(x_dataset)
    mean_y = np.mean(y_dataset)
    
    SS_X = _sum_of_squares(x_dataset)
    SS_Y = _sum_of_squares(y_dataset)
    
    SP_XY = _sum_product(x_dataset, y_dataset)
    
    r = (SP_XY) / np.sqrt(SS_X*SS_Y)
    r_squared = r**2
    standard_error = np.sqrt((1 - r_squared)/(n - 2))
    t = (r - ro) / standard_error
    df = n - 2 
    t_critical = _t_critical(df, alpha, two_tailed)
    
    if abs(t) > t_critical:
        results = "Reject Null"
    else:
        results = "Fail to Reject Null"
        
    df_out = pd.DataFrame({
        "Statistic": ["n", "df", "Mean_X", "SS_X", "Mean_Y", "SS_Y",
                      "SP_XY", "r", "r_squared", "Standard Error", "t", 
                      "t_critical", "Decision"],
        "Value": [n,
                  df,
                  round(mean_x, 2),
                  round(SS_X, 2),
                  round(mean_y, 2), 
                  round(SS_Y, 2),
                  round(SP_XY, 2),
                  round(r, 2),
                  round(r_squared, 2),
                  round(standard_error, 2),
                  round(t, 2),
                  round(t_critical,2),
                  results]
        
        })
    return x_dataset, y_dataset, df_out

def generate_1_predictor_regression(x_mean=10, x_std=1, y_mean=20, y_std=3,
                                 ro=0, n=10, alpha=0.05, seed=None):
    x_dataset = generate_normal_data(x_mean, x_std, n, seed)
    y_dataset = generate_normal_data(y_mean, y_std, n, seed)
    y_dataset = _apply_treatment(y_dataset, effect=10)
    
    mean_x = np.mean(x_dataset)
    mean_y = np.mean(y_dataset)
    
    SS_X = _sum_of_squares(x_dataset)
    SS_Y = _sum_of_squares(y_dataset)
    df_Y = n - 1
    
    SP_XY = _sum_product(x_dataset, y_dataset)
    
    r = (SP_XY) / np.sqrt(SS_X*SS_Y)
    r_squared = r**2
    b = SP_XY / SS_X
    a = mean_y - (b*mean_x)
    
    SS_regression = r_squared*SS_Y
    SS_residual = (1-r_squared)*SS_Y
    df_regression = 1
    df_residual = n - 2
    MS_regression = SS_regression / df_regression
    MS_residual = SS_residual / df_residual
    F = MS_regression / MS_residual
    
    f_critical = _f_critical(df_regression, df_residual, alpha)
    
    if F > f_critical:
        results = "Reject Null"
    else:
        results = "Fail to Reject Null"
   
    rows = []
    rows.append({
        "Source": "Regression",
        "SS": SS_regression,
        "df": df_regression,
        "MS": MS_regression,
        "F": F,
        "Decision": results
    })
    
    rows.append({
        "Source": "Residual",
        "SS": SS_residual,
        "df": df_residual,
        "MS": MS_residual,
        "F": None,
        "Decision": None
    })
    
    rows.append({
        "Source": "Total",
        "SS": SS_Y,
        "df": df_Y,
        "MS": None,
        "F": None,
        "Decision": None
    })
    
    regression_equation = f"Y={b}X+{a}"
    
    return y_dataset, x_dataset, pd.DataFrame(rows), regression_equation


