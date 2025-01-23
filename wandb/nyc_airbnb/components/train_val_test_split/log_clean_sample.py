import wandb

# Initialize W&B
run = wandb.init(
    project="Project-Build-an-ML-Pipeline-Starter-components_train_val_test_split",
    entity="gerald-donofrio1-western-governors-university",  # Updated team name
    job_type="log_artifact"
)

# Create an artifact
artifact = wandb.Artifact(
    name="clean_sample",
    type="dataset",
    description="Cleaned dataset for training and testing"
)

# Add the file
artifact.add_file("clean_sample.csv")

# Log the artifact
run.log_artifact(artifact)
run.finish()
