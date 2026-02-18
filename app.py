import pandas as pd
import io
from flask import Flask, render_template, request, session, send_file
from datetime import datetime
from toolkit.interface import (
    run_z_test, 
    run_t_test, 
    run_independent_t_test,
    run_repeated_t_test,
    run_independent_anova,
    run_pearson_correlation,
    run_1_predictor_regression
)

app = Flask(__name__)
app.secret_key = "statteacher-dev-key"

@app.route("/download_dataset")
def download_dataset():
    """
    Flask route: export the most recent test results as a timestamped Excel file.

    Reads the serialized DataFrames stored in the Flask session under
    "excel_payload" and writes each entry as a separate sheet in an in-memory
    .xlsx workbook. Streams the file to the browser as a download attachment.

    Returns
    -------
    flask.Response
        An Excel file (.xlsx) download response, or a plain-text error message
        if no dataset is available in the current session.
    """

    if "excel_payload" not in session:
        return "No dataset available for download."

    payload = session["excel_payload"]

    test_type = session.get("last_test_type", "results")
    safe_test_type = test_type.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_test_type = f"{safe_test_type}_{timestamp}"

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, data_dict in payload.items():
            df = pd.DataFrame.from_dict(data_dict)

            safe_sheet = sheet_name.replace("/", "_")[:31]
            df.to_excel(writer, index=False, sheet_name=safe_sheet)

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=f"{safe_test_type}_results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



def get_float(name, default=None):
    """
    Safely parse a float value from the current request form.

    Parameters
    ----------
    name : str
        The form field name to read.
    default : float or None, optional
        Value to return if the field is absent or empty. Default is None.

    Returns
    -------
    float or None
        The parsed float value, or `default` if the field was missing/empty.
    """
    val = request.form.get(name)
    return float(val) if val not in (None, "") else default

def parse_factors_string(factors_str):
    """
    Parses 'A:3,B:2' -> {'A': 3, 'B': 2}
    """
    factors = {}

    if not factors_str:
        raise ValueError("No factors provided.")

    pairs = factors_str.split(",")

    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid format: '{pair}'. Expected Factor:Levels.")

        name, level = pair.split(":")

        name = name.strip()
        level = level.strip()

        if not name:
            raise ValueError("Factor name cannot be empty.")

        if not level.isdigit():
            raise ValueError(f"Levels for factor '{name}' must be an integer.")

        level = int(level)

        if level < 2:
            raise ValueError(f"Factor '{name}' must have at least 2 levels.")

        factors[name] = level

    return factors

@app.route("/", methods=["GET"])
def index():
    """
    Flask route: render the main parameter-entry form.

    Returns
    -------
    flask.Response
        Rendered index.html template with the test-selection form.
    """
    return render_template("index.html")


@app.route("/run_test", methods=["POST"])
def run_test():
    """
    Flask route: execute the selected statistical test and render the results page.

    Reads parameters from the submitted form, delegates to the appropriate
    interface function (run_z_test, run_t_test, etc.), stores the resulting
    DataFrames in the session for later Excel export, and renders results.html
    with the dataset and statistics tables.

    Supported test_type values
    --------------------------
    - "z_test"             : One-sample Z-test
    - "t_test"             : One-sample t-test
    - "independent_t_test" : Independent-samples t-test
    - "paired_t_test"      : Repeated-measures (paired) t-test
    - "Independent_anova"  : Independent-groups ANOVA (one-way or factorial)
    - "pearson_correlation": Pearson correlation
    - "simple_regression"  : One-predictor linear regression

    Returns
    -------
    flask.Response
        Rendered results.html template with HTML tables for the dataset and
        statistical results, or a plain-text error message for an unrecognized
        test type or invalid ANOVA factors string.
    """

    test_type = request.form.get("test_type")

    # Z-TEST -------------------------------------------------
    if test_type == "z_test":
        population_mean = get_float("population_mean", 0)
        population_std = get_float("population_std", 15)
        n = int(get_float("n", 10))
        tx_effect = get_float("tx_effect", 5)
        noise_sd = get_float("noise_sd", 3)
        alpha = get_float("alpha", 0.05)
        two_tailed = request.form.get("tail_type") == "two_tailed"
        seed = int(request.form.get("seed")) if request.form.get("seed") else None
        dataset, table = run_z_test(
            population_mean=population_mean,
            population_std=population_std,
            n=n,
            tx_effect=tx_effect,
            noise_sd=noise_sd,
            alpha=alpha,
            two_tailed=two_tailed,
            seed=seed
        )

        session["excel_payload"] = {
        "Data": pd.DataFrame({"Scores": dataset}).to_dict(),
        "Results": table.to_dict()
        }
        session["last_test_type"] = "z_test"
        df1 = pd.DataFrame({"Dataset": dataset})
        return render_template(
            "results.html",
            dataset=df1.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered"),
            show_download=True
        )

    # T-TEST -------------------------------------------------
    if test_type == "t_test":
        population_mean = get_float("population_mean", 0)
        population_std = get_float("population_std", 15)
        n = int(get_float("n", 10))
        tx_effect = get_float("tx_effect", 5)
        noise_sd = get_float("noise_sd", 3)
        alpha = get_float("alpha", 0.05)
        two_tailed = request.form.get("tail_type") == "two_tailed"
        seed = int(request.form.get("seed")) if request.form.get("seed") else None
        dataset, table = run_t_test(
            population_mean=population_mean,
            population_std=population_std,
            n=n,
            tx_effect=tx_effect,
            noise_sd=noise_sd,
            alpha=alpha,
            two_tailed=two_tailed,
            seed=seed
        )
        session["excel_payload"] = {
        "Data": pd.DataFrame({"Scores": dataset}).to_dict(),
        "Results": table.to_dict()
        }
        session["last_test_type"] = "t_test"
        df1 = pd.DataFrame({"Dataset": dataset})
        return render_template(
            "results.html",
            dataset=df1.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered"),
            show_download=True
        )
    
    # Independent T-Test --------------------------------------
    if test_type == "independent_t_test":
        population_mean1 = get_float("population_mean1", 10)
        population_sd1 = get_float("population_sd1", 15)
        n1 = int(get_float("n1", 10))
        seed1 = int(request.form.get("seed1")) if request.form.get("seed1") else None
        population_mean2 = get_float("population_mean2", 20)
        population_sd2 = get_float("population_sd2", 15)
        n2 = int(get_float("n2", 10))
        seed2 = int(request.form.get("seed2")) if request.form.get("seed2") else None
        alpha = get_float("alpha", 0.05)
        two_tailed = request.form.get("tail_type") == "two_tailed"
        dataset1, dataset2, table = run_independent_t_test(
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

        session["excel_payload"] = {
        "Data": pd.DataFrame({
        "Dataset1": dataset1,
        "Dataset2": dataset2
        }).to_dict(),
        "Results": table.to_dict()
        }
        session["last_test_type"] = "independent_t_test"

        df1 = pd.DataFrame({"Dataset 1": dataset1})
        df2 = pd.DataFrame({"Dataset 2": dataset2})

        return render_template(
            "results.html",
            dataset1=df1.to_html(index=False),
            dataset2=df2.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered"),
            show_download=True
        )

    # Repeated T-TEST ----------------------------------------
    if test_type == "paired_t_test":
        population_mean = get_float("population_mean", 0)
        population_std = get_float("population_std", 15)
        n = int(get_float("n", 10))
        tx_effect = get_float("tx_effect", 5)
        noise_sd = get_float("noise_sd", 3)
        alpha = get_float("alpha", 0.05)
        two_tailed = request.form.get("tail_type") == "two_tailed"
        seed = int(request.form.get("seed")) if request.form.get("seed") else None
        
        predataset, postdataset, table = run_repeated_t_test(
            population_mean=population_mean,
            population_std=population_std,
            n=n,
            tx_effect=tx_effect,
            noise_sd=noise_sd,
            alpha=alpha,
            two_tailed=two_tailed,
            seed=seed
        )
        session["excel_payload"] = {
        "Data": pd.DataFrame({
        "Pre": predataset,
        "Post": postdataset
        }).to_dict(),
        "Results": table.to_dict()
        }
        session["last_test_type"] = "paired_t_test"
        
        df1 = pd.DataFrame({"Pre-Treatment": predataset})
        df2 = pd.DataFrame({"Post-Treatment": postdataset})

        return render_template(
            "results.html",
            predataset=df1.to_html(index=False),
            postdataset=df2.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered"),
            show_download=True
        )
    
    # Independent ANOVA -------------------------------------
    if test_type == "Independent_anova": 
        n = int(get_float("n", 10))
        mean = get_float("mean", 10)
        std = get_float("std", 2)
        alpha = get_float("alpha", 0.05)
        effect_size = get_float("effect_size", 2.0)
        seed = int(request.form.get("seed")) if request.form.get("seed") else None
        factors_str = request.form.get("factors_dictionary")

        if not factors_str or not factors_str.strip():
            factors_str = "A:3,B:2"
        try:
            factors_dictionary = parse_factors_string(factors_str)
        except ValueError as e:
            return f"Error parsing factors: {e}"
        
        df, table = run_independent_anova(
            factors_dictionary=factors_dictionary,
            n=n,
            mean=mean,
            std=std,
            alpha=alpha,
            effect_size=effect_size,
            seed=seed
        ) 
        session["excel_payload"] = {
        "Data": df.to_dict(orient="list"),
        "ANOVA Table": table.to_dict(orient="list"),
        "Design": pd.DataFrame(
        list(factors_dictionary.items()),
        columns=["Factor", "Levels"]
        ).to_dict(orient="list")
        }

        session["last_test_type"] = "independent_anova"


        df1 = pd.DataFrame(df)
        return render_template(
            "results.html", 
            dataset=df1.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered"),
            show_download=True
            )
    
    # Pearson Correlation -----------------------------------
    if test_type == "pearson_correlation":
        seed = int(request.form.get("seed")) if request.form.get("seed") else None
        x_mean = get_float("x_mean", 10)
        x_std = get_float("x_std", 1)
        y_mean = get_float("y_mean", 20)
        y_std = get_float("y_std", 3)
        n = int(get_float("n", 10))
        ro = get_float("ro", 0)
        alpha = get_float("alpha", 0.05)
        two_tailed = request.form.get("tail_type") == "two_tailed"
        x_dataset, y_dataset, table = run_pearson_correlation(
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
        session["excel_payload"] = {
        "Data": pd.DataFrame({
        "X_Dataset": x_dataset,
        "Y_Dataset": y_dataset
        }).to_dict(),
        "Results": table.to_dict()
        }

        session["last_test_type"] = "pearson_correlation"

        return render_template(
            "results.html",
            test_type=test_type,
            x_dataset=x_dataset,
            y_dataset=y_dataset,
            table_html=table.to_html(classes="table table-bordered"),
            show_download=True
        )
    
    # 1-Predictor Regression --------------------------------
    if test_type == "simple_regression":
        seed = int(request.form.get("seed")) if request.form.get("seed") else None
        x_mean = get_float("x_mean", 10)
        x_std = get_float("x_std", 1)
        y_mean = get_float("y_mean", 20)
        y_std = get_float("y_std", 3)
        n = int(get_float("n", 10))
        alpha = get_float("alpha", 0.05)
        y_dataset, x_dataset, table, equation = run_1_predictor_regression(
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            n=n,
            alpha=alpha,
            seed=seed
        )
        session["excel_payload"] = {
        "Data": pd.DataFrame({
        "X": x_dataset,
        "Y": y_dataset
        }).to_dict(orient="list"),

        "Results": table.to_dict(orient="list"),

        "Model": pd.DataFrame({
        "Item": ["Regression Equation"],
        "Value": [equation]
        }).to_dict(orient="list")
        }


        session["last_test_type"] = "simple_regression"

        return render_template(
            "results.html",
            test_type=test_type,
            y_dataset=y_dataset,
            x_dataset=x_dataset,
            table_html=table.to_html(classes="table table-bordered"),
            equation=equation, 
            show_download=True
        )

    return "Invalid test type selected."


if __name__ == "__main__":
    app.run(debug=True)


