import wandb

# Initialize a W&B run
run = wandb.init(
    project="Project-Build-an-ML-Pipeline-Starter-src_data_check", 
    job_type="upload"
)

# Create a new artifact
artifact = wandb.Artifact(
    name="clean_sample",
    type="dataset",
    description="Cleaned dataset after basic cleaning"
)

# Add the file to the artifact
artifact.add_file("clean_sample.csv")

# Log the artifact with the alias "latest"
run.log_artifact(artifact, aliases=["latest"])

# Finish the W&B run
run.finish()
