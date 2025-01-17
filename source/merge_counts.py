import os
import sys
sys.path.append('/container/mount/point')

import pandas as pd
import argparse
import json
import pickle

from rpy2.robjects import pandas2ri
from contextlib import redirect_stdout

from utils.helper import check_samples_overlap, generate_taxa_dict
from utils.preprocessing import filter_and_process_asv_table

# Convert pandas.DataFrames to R dataframes automatically.
pandas2ri.activate()

def main():
    """
    Main function to merge count data with taxonomic annotations.

    This script reads preprocessed data, applies pair matching between treated and control groups
    based on the provided hyperparameters, and generates simulated outcomes.

    Command-line arguments:
        -counts (str): Path to the data file (CSV).
        -taxonomy (str): Path to the JSON file with taxonomic annotations.
        -params (str): Path to the JSON file with hyperparameters.
        -output (str): Directory to save output files.

    Outputs:
        - A CSV file of matched pairs (`matched_pairs_<target_variable>.csv`) saved to the output directory.
        - A CSV file of simulated outcomes (`W_paired_<target_variable>.csv`) saved to the output directory.
        - A text file (`log.txt`) containing print outputs saved to the output directory.
    """
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Pair matching and simulated outcome generation script.")
    
    # Define the flags and expected inputs
    parser.add_argument("-counts", required=True, help="Path to the data file (CSV).")
    parser.add_argument("-taxonomy", required=True, help="Path to the JSON file with taxonomic annotations.")
    parser.add_argument("-params", required=True, help="Path to the JSON file with hyperparameters.")
    parser.add_argument("-output", required=True, help="Directory to save output files.")

    # Parse the arguments
    args = parser.parse_args()

    # Extract arguments
    count_data_file = args.counts
    taxonomy_file = args.taxonomy
    hyperparameters_file = args.params
    output_directory = args.output

    # Load hyperparameters from JSON file
    with open(hyperparameters_file, "r") as f:
        hyperparameters = json.load(f)

    target_variable = hyperparameters["target_variable"]
    
    # Check if matched pairs exist in the output directory
    pairs_file = os.path.join(output_directory, f"matched_pairs_{target_variable}.csv")
    
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"The matched pairs file does not exist in the output directory: {pairs_file}")
    
    # Define log file path
    log_file_path = os.path.join(output_directory, "log_Mergecounts.txt")

    # Redirect stdout to the log file
    with open(log_file_path, "w") as log_file, redirect_stdout(log_file):
        print("Loading data...")
        taxa = pd.read_csv(taxonomy_file, sep=',', index_col=0)
        sample_df = pd.read_csv(pairs_file, sep=',', index_col=0)
        asv = pd.read_csv(count_data_file, sep='\t', index_col=0)

        print(f"Taxonomy shape: {taxa.shape}")
        print(f"Matched samples shape: {sample_df.shape}")
        print(f"ASV shape: {asv.shape}")

        # Sort ASVs by pair-matching order
        print("Sorting ASVs by pair-matching order...")
        arr = list(sample_df.index)
        ASV_table = asv.reindex(
            sorted(asv.columns, key=lambda x: arr.index(int(x)) if x.isdigit() and int(x) in arr else float('inf')),
            axis=1
        )
        print(f"ASV table shape after sorting: {ASV_table.shape}")

        # Add taxonomic annotations
        print("Adding taxonomic annotations...")
        taxa_dict = {}
        for level in taxa.columns:
            df_level = ASV_table.join(taxa[level])
            df_level = df_level.groupby(level).sum()
            taxa_dict[level] = df_level
        taxa_dict["ASVs"] = ASV_table

        for level, table in taxa_dict.items():
            print(f"{level} count table shape: {table.shape}")

        # Overlap samples
        print("Checking sample overlap...")
        df = check_samples_overlap(sample_df, ASV_table)
        str_sample_ids = set(df.index.astype(str))

        # Filter ASV table by overlapping samples
        print("Filtering ASV table by overlapping samples...")
        ASV_table = ASV_table.loc[:, ASV_table.columns.isin(str_sample_ids)]
        asv_top99_samples, asv_sample_ids = filter_and_process_asv_table(
            ASV_table, freq_threshold=hyperparameters.get("freq_threshold", 0.01)
        )

        # Regenerate `taxa_dict` using filtered ASV table
        print("Generating filtered taxa dictionary...")
        taxa_dict = generate_taxa_dict(asv=asv_top99_samples, taxa=taxa)
        taxa_dict.pop("name", None)

        for level, table in taxa_dict.items():
            print(f"{level} count table shape after filtering: {table.shape}")

        # Align ASV and IgE samples
        print("Aligning ASV and IgE samples...")
        sample_ids = asv_top99_samples.columns.astype(int)
        w = pd.DataFrame(df[df.index.isin(sample_ids)]["W"].values, index=sample_ids, columns=["w"])

        print("Merge process completed successfully.")

        # Save results to output directory
        w.to_csv(os.path.join(output_directory, f"W_paired_{target_variable}.csv"))
        with open(os.path.join(output_directory, "taxa_dict.pkl"), "wb") as f:
            pickle.dump(taxa_dict, f)

if __name__ == "__main__":
    main()
