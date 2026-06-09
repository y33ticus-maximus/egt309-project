from logistic_regression import main as run_logistic_regression
from dt_model import main as run_decision_tree
from random_forest import RandomForestPipeline


def main():
    print("\n==============================")
    print("Running Logistic Regression")
    print("==============================")
    run_logistic_regression()

    print("\n==============================")
    print("Running Decision Tree")
    print("==============================")
    run_decision_tree()

    print("\n==============================")
    print("Running Random Forest")
    print("==============================")
    rf_pipeline = RandomForestPipeline()
    rf_pipeline.run()

    print("\n==============================")
    print("All models completed.")
    print("==============================")


if __name__ == "__main__":
    main()