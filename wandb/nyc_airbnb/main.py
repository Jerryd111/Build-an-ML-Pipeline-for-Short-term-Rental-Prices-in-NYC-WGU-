import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

@hydra.main(config_name='config')
def go(config: DictConfig):

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
    
        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(os.getcwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": config["etl"]["sample"],
                    "output_artifact": "clean_sample.csv",
                    "output_type": "cleaned_data",
                    "output_description": "Cleaned dataset after basic preprocessing",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )
    
        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(os.getcwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
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
                    "val_size": config["modeling"]["val_size"],
                    "test_size": config["modeling"]["test_size"],
                },
            )
                    
        if "train_random_forest" in active_steps: 
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                os.path.join(os.getcwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "output_artifact": "random_forest_export",
                    "rf_config": rf_config,
                    "val_size": config["modeling"]["val_size"],
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
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
    go()
