process pairMatching {
    input:
    path data_file
    path hyperparameters_file
    path output_directory

    script:
    """
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
