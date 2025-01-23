_ = mlflow.run(
    f"{config['main']['components_repository']}/train_val_test_split",
    'main',
    parameters={
        "input": "clean_sample.csv:latest",
        "test_size": config["modeling"]["test_size"],
        "random_seed": config["modeling"]["random_seed"],
        "stratify_by": config["modeling"].get("stratify_by", None),
    },
)

