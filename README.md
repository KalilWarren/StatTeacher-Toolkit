# StatTeacher Toolkit

> **This repository is no longer actively maintained.**
> Development has moved to [kalilwarren.github.io/toolkit.html](https://kalilwarren.github.io/toolkit.html).
> This codebase is preserved here as a legacy reference.

---

A Flask-based web application for generating statistical hypothesis-testing practice problems. Instructors and students can configure parameters, generate synthetic datasets, and instantly see step-by-step results — including test statistics, critical values, effect sizes, confidence intervals, and a pass/fail decision — all exportable to Excel.

---

## Features

- **Seven statistical tests** available through a point-and-click interface
- **Synthetic data generation** with configurable population parameters and optional random seeds for reproducibility
- **Complete results tables** — every intermediate value shown, not just the final answer
- **Excel export** — download the dataset and results as a multi-sheet `.xlsx` file
- **Factorial ANOVA support** — specify any fully-crossed design (e.g. `A:3,B:2`)

---

## Supported Tests

| Test | When to Use |
|---|---|
| **Z-Test** | Population SD is known |
| **One-Sample t-Test** | Population SD is unknown, estimated from sample |
| **Independent-Samples t-Test** | Compare two independent groups |
| **Repeated-Measures t-Test** | Compare pre/post scores for the same participants |
| **One-Way / Factorial ANOVA** | Compare three or more groups; supports multi-factor designs |
| **Pearson Correlation** | Test the linear relationship between two variables |
| **Simple Linear Regression** | Predict Y from a single predictor X |

---

## Project Structure

```
StatTeacher_Toolkit/
├── app.py                   # Flask application and route handlers
├── templates/
│   ├── index.html           # Parameter-entry form (test selection + inputs)
│   └── results.html         # Results display (dataset tables + statistics)
├── toolkit/
│   ├── __init__.py          # Package init — re-exports interface functions
│   ├── interface.py         # Thin wrappers that connect Flask to the engine
│   └── Engine_v07.py        # Core statistical computation engine
└── static/                  # Static assets (currently unused)
```

### Layer Overview

```
Browser  →  app.py (Flask routes)
              ↓
         toolkit/interface.py  (parameter forwarding)
              ↓
         toolkit/Engine_v07.py (data generation + statistics)
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/StatTeacher_Toolkit.git
cd StatTeacher_Toolkit

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install flask numpy scipy pandas openpyxl
```

### Running the App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Usage

1. **Select a test** from the dropdown on the home page.
2. **Enter parameters** — population mean, standard deviation, sample size, alpha, tail type, and any test-specific options (e.g. treatment effect, ANOVA factors).
3. **Set a seed** (optional) if you want reproducible data across runs.
4. Click **Generate Problem**.
5. Review the **dataset table** and **results table** on the results page.
6. Click **Download as Excel** to export both sheets as a `.xlsx` file.

### ANOVA Factor Syntax

For the ANOVA test, enter factors in the format `FactorName:Levels` separated by commas:

```
A:3          # One-way ANOVA with 3 levels
A:3,B:2      # 3×2 factorial ANOVA
A:2,B:2,C:3  # Three-factor ANOVA
```

---

## Output Metrics by Test

### Z-Test
N, Population Mean, Population SD, Sample Mean, Standard Error, Z Score, Critical Z, Decision, Cohen's d, 95% CI

### One-Sample t-Test
N, Sample SD, Sample Mean, Standard Error, t Score, Critical t, Decision, Cohen's d, R², 95% CI

### Independent-Samples t-Test
Per-group: N, SD, Mean, df, SS
Combined: Pooled variance, Standard Error, t Score, Critical t, Decision, Cohen's d, R², 95% CI for mean difference

### Repeated-Measures t-Test
N, SD of Differences, Mean Difference, Standard Error, t Score, Critical t, Decision, Cohen's d, R², 95% CI for mean difference

### ANOVA
Per-source ANOVA table: SS, df, MS, F, p-value
Sources: main effects, interaction (factorial), Within (error), Total

### Pearson Correlation
n, df, Mean X/Y, SS\_X, SS\_Y, SP\_XY, r, r², Standard Error, t, Critical t, Decision

### Simple Linear Regression
Regression equation (Y = bX + a)
ANOVA table: Regression, Residual, Total rows with SS, df, MS, F, Decision

---

## Module Reference

### `toolkit/Engine_v07.py`

The core computational engine. All functions return NumPy arrays and/or Pandas DataFrames.

#### Data Generation Utilities

| Function | Description |
|---|---|
| `generate_normal_data(mean, std, n, seed, return_df)` | Generate a normally distributed sample |
| `z_score_tranformation(data, rescale, new_mean, new_std)` | Convert raw scores to z-scores, with optional rescaling |
| `_apply_treatment(pre_scores, effect, noise_sd)` | Add a treatment effect + random noise to simulate pre→post data |

#### Critical Value Helpers

| Function | Description |
|---|---|
| `_z_critical(alpha, two_tailed)` | Critical Z value |
| `_t_critical(df, alpha, two_tailed)` | Critical t value |
| `_f_critical(df_Between, df_Within, alpha)` | Critical F value |

#### Statistical Computation Helpers

| Function | Description |
|---|---|
| `_sum_of_squares(x)` | SS: Σ(x − x̄)² |
| `_sum_product(x, y)` | SP\_XY: Σ(x − x̄)(y − ȳ) |
| `_compute_factor_level_SS(df, C, dv)` | Per-factor SS for factorial ANOVA |
| `_build_independent_anova_table(...)` | Assemble ANOVA summary DataFrame |

#### Main Problem Generators

| Function | Returns |
|---|---|
| `generate_z_score_problem(...)` | `(dataset, results_df)` |
| `generate_t_test_problem(...)` | `(dataset, results_df)` |
| `generate_independent_t_test_problem(...)` | `(dataset1, dataset2, results_df)` |
| `generate_repeated_t_test_problem(...)` | `(pre_dataset, post_dataset, results_df)` |
| `generate_Independent_ANOVA(...)` | `(long_df, anova_table_df)` |
| `generate_pearson_correlation(...)` | `(x_dataset, y_dataset, results_df)` |
| `generate_1_predictor_regression(...)` | `(y_dataset, x_dataset, anova_table_df, equation_str)` |

### `toolkit/interface.py`

Thin wrapper functions that forward arguments from the Flask layer to the engine. Each `run_*` function mirrors its corresponding `generate_*` function and returns identical values. See `Engine_v07.py` for parameter details.

| Interface Function | Engine Function |
|---|---|
| `run_z_test(...)` | `generate_z_score_problem` |
| `run_t_test(...)` | `generate_t_test_problem` |
| `run_independent_t_test(...)` | `generate_independent_t_test_problem` |
| `run_repeated_t_test(...)` | `generate_repeated_t_test_problem` |
| `run_independent_anova(...)` | `generate_Independent_ANOVA` |
| `run_pearson_correlation(...)` | `generate_pearson_correlation` |
| `run_1_predictor_regression(...)` | `generate_1_predictor_regression` |

### `app.py`

| Route / Function | Description |
|---|---|
| `GET /` → `index()` | Render the parameter-entry form |
| `POST /run_test` → `run_test()` | Execute the selected test; render results |
| `GET /download_dataset` → `download_dataset()` | Stream results as an Excel file |
| `get_float(name, default)` | Helper: parse a float from the request form |
| `parse_factors_string(factors_str)` | Helper: parse `"A:3,B:2"` into `{"A": 3, "B": 2}` |

---

## Dependencies

| Package | Purpose |
|---|---|
| Flask | Web framework and routing |
| NumPy | Array operations and random number generation |
| SciPy | Normal, t, and F distribution functions |
| Pandas | DataFrame construction and HTML/Excel rendering |
| OpenPyXL | Writing `.xlsx` files |

---

## License

This project is for educational use. No license has been specified — please contact the author before redistributing or using in commercial settings.
