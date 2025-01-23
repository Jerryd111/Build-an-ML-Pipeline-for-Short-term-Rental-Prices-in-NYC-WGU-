import json
import mlflow
import tempfile
import os
import argparse

# Define the available steps
_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: "test_regression_model" is omitted unless explicitly promoted to "prod"
]

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=str, required=True, help="Comma-separated steps to execute")
    parser.add_argument("--csv", type=str, help="Input CSV file to be tested")
    parser.add_argument("--ref", type=str, help="Reference CSV file to compare the new CSV to")
    parser.add_argument("--kl_threshold", type=float, help="Threshold for the KL divergence test")
    parser.add_argument("--min_price", type=float, help="Minimum accepted price")
    parser.add_argument("--max_price", type=float, help="Maximum accepted price")

    args = parser.parse_args()

    # Determine active steps
    active_steps = args.steps.split(",") if args.steps != "all" else _steps

    # Process each step
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in active_steps:
            _ = mlflow.run(
                f"{os.getcwd()}/src/download",
                "main",
                parameters={
                    "sample": "sample.csv",
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(os.getcwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": args.csv,
                    "output_artifact": "clean_sample.csv",
                    "output_type": "cleaned_data",
                    "output_description": "Cleaned dataset",
                    "min_price": args.min_price,
                    "max_price": args.max_price,
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(os.getcwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": args.csv,
                    "ref": args.ref,
                    "kl_threshold": args.kl_threshold,
                    "min_price": args.min_price,
                    "max_price": args.max_price,
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(os.getcwd(), "src", "data_split"),
                "main",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "artifact_root": "data",
                    "artifact_type": "split_data",
                },
            )

        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w") as fp:
                # Serialize random forest parameters into JSON
                json.dump({
                    "n_estimators": 100,
                    "max_depth": None,
                    "random_state": 42,
                }, fp)

            _ = mlflow.run(
                os.path.join(os.getcwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "train_data": "data/train.csv:latest",
                    "val_data": "data/val.csv:latest",
                    "rf_config": rf_config,
                    "max_tfidf_features": 100,
                    "output_artifact": "random_forest_model",
                },
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                os.path.join(os.getcwd(), "src", "test_regression_model"),
                "main",
                parameters={
                    "model": "random_forest_model:prod",
                    "test_data": "data/test.csv:latest",
                },
            )


if __name__ == "__main__":
    main()

