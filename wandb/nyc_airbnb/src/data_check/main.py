import os
import hydra
import mlflow
from omegaconf import DictConfig

@hydra.main(config_name="config")
def go(config: DictConfig):
    root_path = hydra.utils.get_original_cwd()
    _steps = config["main"]["steps"].split(",")  # Steps defined in config.yaml

    for step in _steps:
        if step == "basic_cleaning":
            _ = mlflow.run(
                "src/basic_cleaning",
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "cleaned_data",
                    "output_description": "Data cleaned of outliers and bad values",
                    "min_price": config["basic_cleaning"]["min_price"],
                    "max_price": config["basic_cleaning"]["max_price"],
                },
            )

        elif step == "data_check":
            _ = mlflow.run(
                "src/data_check",
                "main",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "ref_artifact": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"],
                },
            )

if __name__ == "__main__":
    main()

