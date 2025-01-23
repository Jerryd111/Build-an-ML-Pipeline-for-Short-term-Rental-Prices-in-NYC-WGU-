#!/usr/bin/env python
"""
Cleaning of data and handling outliers
"""
import os
import wandb
import logging
import pandas as pd
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig


# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Define the configuration schema
class Config:
    input_artifact: str
    output_artifact_name: str
    output_artifact_type: str
    output_artifact_description: str
    min_price: float
    max_price: float

# Register the configuration schema
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(config_name="config")
def go(cfg: Config):
    # Initialize the wandb run
    run = wandb.init(job_type="data_clean")
    run.config.update(cfg)

    # Download artifact
    logger.info("Downloading artifact")
    artifact = run.use_artifact(cfg.input_artifact)
    artifact_path = artifact.file()

    # Load artifact to dataframe
    logger.info("Loading artifact to dataframe")
    df = pd.read_csv(artifact_path)

    # Cleaning the data
    logger.info("Cleaning the data")
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['price'].between(cfg.min_price, cfg.max_price)
    df = df[idx].copy()

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    filename = "clean_data"
    df.to_csv(filename, index=False)

    # Creating the artifact
    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=cfg.output_artifact_name,
        type=cfg.output_artifact_type,
        description=cfg.output_artifact_description,
    )
    artifact.add_file(filename)

    # Logging the artifact
    logger.info("Logging artifact")
    run.log_artifact(artifact)
    
    os.remove(filename)

if __name__ == "__main__":
    go()
