#!/usr/bin/env python
"""
Data validation and testing for cleaned dataset.
"""
import argparse
import logging
import pandas as pd
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

def test_row_count(dataframe):
    """
    Ensure the dataset has a reasonable number of rows.
    """
    if not 15000 <= len(dataframe) <= 20000:
        raise ValueError(f"Dataset has {len(dataframe)} rows. Expected between 15,000 and 20,000.")

def test_price_range(dataframe, min_price, max_price):
    """
    Ensure prices are within the specified range.
    """
    if not ((dataframe["price"] >= min_price).all() and (dataframe["price"] <= max_price).all()):
        raise ValueError("Prices are out of the specified range!")

def main(args):
    """
    Main execution function for the step.
    """
    run = wandb.init(job_type="data_check")
    logger.info("Downloading artifact.")
    try:
        artifact_path = wandb.use_artifact(args.input_artifact).file()
    except wandb.errors.CommError as e:
        raise ValueError(f"Invalid artifact format: {args.input_artifact}. Use 'collection:alias'.")
    
    logger.info("Loading data.")
    dataframe = pd.read_csv(artifact_path)

    logger.info("Running tests.")
    test_row_count(dataframe)
    test_price_range(dataframe, args.min_price, args.max_price)

    logger.info("All tests passed. Logging results.")
    wandb.log({"row_count_test_passed": True, "price_range_test_passed": True})

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data checks on input artifact")
    
    # Define the expected arguments
    parser.add_argument("--input_artifact", type=str, required=True, help="Input CSV file to validate")
    parser.add_argument("--ref_artifact", type=str, required=True, help="Reference CSV file for comparison")
    parser.add_argument("--kl_threshold", type=float, required=True, help="Threshold for KL divergence")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum acceptable price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum acceptable price")
    
    args = parser.parse_args()
    main(args)

