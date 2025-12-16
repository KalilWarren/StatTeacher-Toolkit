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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/run_test", methods=["POST"])
def run_test():

    test_type = request.form.get("test_type")

    # Z-TEST -------------------------------------------------
    if test_type == "z_test":
        dataset, table = run_z_test()
        df1 = pd.DataFrame({"Dataset": dataset})
        return render_template(
            "results.html",
            dataset=df1.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered")
        )

    # T-TEST -------------------------------------------------
    if test_type == "t_test":
        dataset, table = run_t_test()
        df1 = pd.DataFrame({"Dataset": dataset})
        return render_template(
            "results.html",
            dataset=df1.to_html(index=False),
            table_html=table.to_html(classes="table table-bordered")
        )
    
    # Independent T-Test --------------------------------------
    if test_type == "independent_t_test":
        dataset1, dataset2, table = run_independent_t_test()

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
        predataset, postdataset, table = run_repeated_t_test()

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


