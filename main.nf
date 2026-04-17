process pairMatching {
    input:
    val target_variable
    path covariates_file
    path hyperparameters_file
    path output_directory

    // output:
    // path "${output_directory}/W_paired_${target_variable}.csv"
    // path "${output_directory}/matched_pairs_${target_variable}.csv"

    script:
    """
    echo "Covariates file: ${covariates_file}"
    echo "Hyperparameters file: ${hyperparameters_file}"
    echo "Output directory: ${output_directory}"

    echo "Running Pair Matching"

    python source/pair_matching.py \
        -covariates "/container/mount/point/data/${covariates_file}" \
        -params "/container/mount/point/data/${hyperparameters_file}" \
        -output "/container/mount/point/data/${output_directory}"
    """
}

process mergeCounts {
    input:
    path count_data
    path taxonomy_data
    path hyperparameters_file
    path output_directory

    script:
    """
    python source/merge_counts.py \
        -counts "/container/mount/point/data/${count_data}" \
        -taxonomy "/container/mount/point/data/${taxonomy_data}" \
        -params "/container/mount/point/data/${hyperparameters_file}" \
        -output "/container/mount/point/data/${output_directory}"
    """
}

workflow {
    // Define the input files of covariates, count data, taxonomy, and hyperparameters
    target_variable = "smoking_bin"
    covariates_file = Channel.fromPath("smoking_data_preprocessed.csv")
    count_data_file = Channel.fromPath("feature_table.tsv")
    taxonomy_file = Channel.fromPath("taxonomy_clean.csv")
    hyperparameters_file = Channel.fromPath("hyperparameters.json")

    output_directory = Channel.fromPath("results")

    // Pass the channels as inputs to the pairMatching process
    pairMatching(
        target_variable,
        covariates_file,
        hyperparameters_file,
        output_directory
    )
    
    mergeCounts(
        count_data_file,
        taxonomy_file,
        hyperparameters_file,
        output_directory
    )
}
