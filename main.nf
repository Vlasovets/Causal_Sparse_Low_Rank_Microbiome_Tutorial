process pairMatching {
    container 'rpy2:latest'

    input:
    path data_file
    path hyperparameters_file
    path output_directory

    script:
    """
    # activate Python environment and set up directories
    source /opt/python3_env/bin/activate

    # Ensure the output directory exists
    if [ ! -d "/container/mount/point/data/results" ]; then
        mkdir -p "/container/mount/point/data/results"
    fi

    # Change to the mount point directory
    cd /container/mount/point || exit 1

    # Run the pair-matching script with provided parameters
    python source/pair_matching.py -data "/container/mount/point/data/smoking_data_preprocessed.csv" -var smoking_bin -params "/container/mount/point/data/hyperparameters.json" -output "/container/mount/point/data/results"
    """
}

workflow {
    data_channel = Channel.fromPath("~/Causal_Sparse_Low_Rank_Microbiome_Tutorial/data/smoking_data_preprocessed.csv")
    params_channel = Channel.fromPath("~/Causal_Sparse_Low_Rank_Microbiome_Tutorial/data/hyperparameters.json")
    output_channel = Channel.fromPath("~/Causal_Sparse_Low_Rank_Microbiome_Tutorial/data/results")

    pairMatching(
        data_channel,
        params_channel,
        output_channel
    )
}
