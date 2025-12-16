import pandas as pd
from flask import Flask, render_template, request
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

def get_float(name, default=None):
    val = request.form.get(name)
    return float(val) if val not in (None, "") else default

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/run_test", methods=["POST"])
def run_test():

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
        df1 = pd.DataFrame({"Dataset": dataset})
        return render_template(
            "results.html",
            dataset=df1.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered")
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
        df1 = pd.DataFrame({"Dataset": dataset})
        return render_template(
            "results.html",
            dataset=df1.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered")
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

        df1 = pd.DataFrame({"Dataset 1": dataset1})
        df2 = pd.DataFrame({"Dataset 2": dataset2})

        return render_template(
            "results.html",
            dataset1=df1.to_html(index=False),
            dataset2=df2.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered")
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
        df1 = pd.DataFrame({"Pre-Treatment": predataset})
        df2 = pd.DataFrame({"Post-Treatment": postdataset})

        return render_template(
            "results.html",
            predataset=df1.to_html(index=False),
            postdataset=df2.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered")
        )
    
    # Independent ANOVA -------------------------------------
    if test_type == "Independent_anova": 
        df, table = run_independent_anova() 
        df1 = pd.DataFrame(df)
        return render_template(
            "results.html", 
            dataset=df1.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered") 
            )
    
    # Pearson Correlation -----------------------------------
    if test_type == "pearson_correlation":
        x_dataset, y_dataset, table = run_pearson_correlation()
        return render_template(
            "results.html",
            test_type=test_type,
            x_dataset=x_dataset,
            y_dataset=y_dataset,
            table_html=table.to_html(classes="table table-bordered")
        )
    
    # 1-Predictor Regression --------------------------------
    if test_type == "1_predictor_regression":
        y_dataset, x_dataset, table, equation = run_1_predictor_regression()
        return render_template(
            "results.html",
            test_type=test_type,
            y_dataset=y_dataset,
            x_dataset=x_dataset,
            table_html=table.to_html(classes="table table-bordered"),
            equation=equation
        )

    return "Invalid test type selected."


if __name__ == "__main__":
    app.run(debug=True)


