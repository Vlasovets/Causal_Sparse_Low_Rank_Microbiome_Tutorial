process pairMatching {
    input:
    path data_file
    path hyperparameters_file
    path output_directory

    script:
    """
    echo "Data file: ${data_file}"
    echo "Hyperparameters file: ${hyperparameters_file}"
    echo "Output directory: ${output_directory}"

    echo "Running Pair Matching"

    python source/pair_matching.py \
        -data "/container/mount/point/data/${data_file}" \
        -params "/container/mount/point/data/${hyperparameters_file}" \
        -output "/container/mount/point/data/${output_directory}"
    """
}

workflow {
    // Define the paths for data, hyperparameters, and output directories
    data_file = Channel.fromPath("~/Causal_Sparse_Low_Rank_Microbiome_Tutorial/data/smoking_data_preprocessed.csv")
    hyperparameters_file = Channel.fromPath("~/Causal_Sparse_Low_Rank_Microbiome_Tutorial/data/hyperparameters.json")
    output_directory = Channel.fromPath("~/Causal_Sparse_Low_Rank_Microbiome_Tutorial/data/results")

    // Pass the channels as inputs to the pairMatching process
    pairMatching(
        data_file,
        hyperparameters_file,
        output_directory
    )
}
