import os
import sys

sys.path.append('/container/mount/point')

import json
import pandas as pd
import numpy as np
import argparse
from utils.pair_matching import (
    discrepancyMatrix,
    construct_network,
    process_matched_pairs,
    generate_simulated_outcomes,
)
from contextlib import redirect_stdout

def main():
    """
    Main function to perform pair matching and simulated outcome generation.

    This script reads preprocessed covariates, applies pair matching between treated and control groups
    based on the provided hyperparameters, and generates simulated outcomes.

    Command-line arguments:
        -covariates (str): Path to the covariates file containing the dataset.
        -var (str): The target variable name in the dataset.
        -params (str): Path to the file containing hyperparameters for matching.
        -output (str): Directory to save output files.

    Outputs:
        - A CSV file of matched pairs (`matched_pairs.csv`) saved to the output directory.
        - A CSV file of simulated outcomes (`W_paired_<target_variable>.csv`) saved to the output directory.
        - A text file (`log.txt`) containing print outputs saved to the output directory.
    """
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Pair matching and simulated outcome generation script.")
    
    # Define the flags and expected inputs
    parser.add_argument("-covariates", required=True, help="Path to the covariates file (CSV).")
    parser.add_argument("-params", required=True, help="Path to the JSON file with hyperparameters.")
    parser.add_argument("-output", required=True, help="Directory to save output files.")

    # Parse the arguments
    args = parser.parse_args()

    # Extract arguments
    covariates_file = args.covariates
    hyperparameters_file = args.params
    output_directory = args.output

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Check if the hyperparameters file exists
    if not os.path.exists(hyperparameters_file):
        raise FileNotFoundError(f"Hyperparameters file '{hyperparameters_file}' not found.")

    # Define log file path
    log_file_path = os.path.join(output_directory, ".log_pairMatching.txt")

    # Load hyperparameters from JSON file
    with open(hyperparameters_file, "r") as f:
        hyperparameters = json.load(f)

    target_variable = hyperparameters["target_variable"]
    target_encoding = {int(k): v for k, v in hyperparameters["target_encoding"].items()}
    matching_criteria = hyperparameters["matching_criteria"]
    n_permutations = hyperparameters["n_permutations"]

    # Redirect stdout to the log file
    with open(log_file_path, "w") as log_file, redirect_stdout(log_file):
        print("Starting pair matching and simulated outcome generation...")
        print(f"Loaded hyperparameters: {hyperparameters}")

        # Load dataset
        df = pd.read_csv(covariates_file, index_col="u3_16s_id")
        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' is not in the dataset.")

        # Create treatment indicator and map encoding
        df["W"] = df[target_variable]
        df["W_str"] = df["W"].map(target_encoding)
        df["is_treated"] = df["W"].astype(bool)  # Boolean flag for treatment
        df["pair_nb"] = np.nan  # Placeholder for pair IDs

        test, control = df[df["W"] == 0], df[df["W"] == 1]

        print("Number of test - {0}".format(len(test)))
        print("Number of control - {0}".format(len(control)))

        # Set the thresholds for each covariate, default is Inf (i.e. no matching)
        thresholds =  np.empty((df.shape[1], ))
        thresholds[:] = np.nan

        # Set thresholds using the dictionary
        for column_name, threshold_value in matching_criteria.items():
            
            if column_name not in df.columns:
                continue

            column_index = df.columns.get_loc(column_name)
            thresholds[column_index] = threshold_value

        # Split data into treated and control groups
        treated_units = df[df["is_treated"]]
        control_units = df[~df["is_treated"]]

        N_treated, N_control = treated_units.shape[0], control_units.shape[0]
        print("Number of treated units: {0}".format(N_treated))
        print("Number of control units: {0}".format(N_control))

        # Optional weights for each covariate when computing the distances
        # WARNING: the order of the items in scaling needs to be the same as the order of the covariates (i.e. columns)
        scaling =  np.ones((df.shape[1], ), dtype=int)

        print("Calculating discrepancies and performing matching...")

        # Calculate discrepancies and perform matching
        discrepancies = discrepancyMatrix(treated_units, control_units, thresholds, scaling)
        g, pairs_dict = construct_network(discrepancies, N_treated, N_control)
        matched_df = process_matched_pairs(pairs_dict, treated_units, control_units)

        print(f"Number of pairs: {len(matched_df)}")
        print(f"Number of test individuals: {len(matched_df[matched_df.W == 0])}")
        print(f"Number of control individuals: {len(matched_df[matched_df.W == 1])}")

        # Generate simulated outcomes
        simulated_outcomes = generate_simulated_outcomes(matched_df, n_permutations)

        # Generate file paths
        simulated_outcomes_file = os.path.join(output_directory, f"W_paired_{target_variable}.csv")
        matched_df_file = os.path.join(output_directory, f"matched_pairs_{target_variable}.csv")

        # Save results
        simulated_outcomes.to_csv(simulated_outcomes_file, index=True)
        matched_df.to_csv(matched_df_file, index=True)

        print(f"Simulated outcomes saved to: {simulated_outcomes_file}")
        print(f"Matched pairs saved to: {matched_df_file}")
        print("Process completed successfully.")


if __name__ == "__main__":
    main()
