import os
import sys
sys.path.append('/container/mount/point')

import pandas as pd
import numpy as np

from utils.helper import check_samples_overlap, generate_taxa_dict, transform_features, scale_array_by_diagonal
from utils.preprocessing import filter_and_process_asv_table
from utils.stability_selection import subsample, estimate_instability
from utils.solver import ADMM_single
from gglasso.helper.utils import normalize, log_transform

from gglasso.helper.model_selection import ebic


target_variable = "smoking_bin"
count_data_path = "data/feature_table.tsv"
freq_threshold = 0.05

sample_df = pd.read_csv(f'data/matched_df_{target_variable}.csv', sep=',', index_col=0)
print("Pair-matched data shape: {0}".format(sample_df.shape))

### Match covariates with 16S data
asv = pd.read_csv(str(count_data_path), index_col=0, sep='\t')

p, N = asv.shape
print("Raw count data shape: p={0}, N={1}".format(p, N))

### Add taxonomic annotation
taxa = pd.read_csv('data/taxonomy_clean.csv', sep=',', index_col=0)
print("Taxonomic data shape: {0}".format(taxa.shape))

### Check and select samples that are present in both datasets
df, diff_samples = check_samples_overlap(sample_df, asv, show_diff=False)
str_sample_ids = set(df.index.astype(str))
ASV_table = asv.loc[:, asv.columns.isin(str_sample_ids)] 
   
### Filter count data by frequency threshold  
asv_top_samples, asv_samples_ids = filter_and_process_asv_table(ASV_table, freq_threshold=freq_threshold)

sample_ids = asv_top_samples.columns.astype(int)
w = pd.DataFrame(df[df.index.isin(sample_ids)]["W"].values, index=sample_ids, columns=["w"])

### Aggregated count data
taxa_dict = generate_taxa_dict(asv=asv_top_samples, taxa=taxa)

print("\n After processing with filtering threshold={0}".format(freq_threshold))
for level in taxa_dict.keys():
    if level != "name":
        print("{0} count table shape: {1}".format(level, taxa_dict[level].shape))




### Sparse model

level = "genus"
model = "sparse"
N = 10  # Number of subsamples
tol = 1e-7
rtol = 1e-5
verbose = False
beta = 0.05 ### stability threshold
lambda1_range = np.logspace(0, -2, 10)

# select count table according to selected taxonomic level
counts = taxa_dict[level]
p, n = counts.shape

print("Level {0}: \n Shape(p,N): {1}".format(level, counts.shape))

X = transform_features(counts, transformation="clr")

subsamples = subsample(X.values, N)

for i in range(subsamples.shape[2]):  # Loop through each subsample
    subsample_slice = subsamples[:, :, i]        # Access the i-th subsample
    print(f"Subsample {i+1}:")
    print(subsample_slice.shape)

print("Print subsamples object: {0}".format(subsamples.shape))


lambda1_range = np.logspace(0, -2, 2)
sol_dict = dict()

for lambda1 in lambda1_range:

    print(f"-------------- Lambda: {lambda1} --------------")

    sub_sol_dict = dict()

    ### Loop through each subsample
    for i in range(N):
        subsample_slice = subsamples[:, :, i]
        print(f"Subsample {i+1}:")

        S0 = np.cov(subsample_slice, bias = True)
        S = scale_array_by_diagonal(S0)

        if p != S.shape[0]:
            raise Exception("Check covariance shape!")

        ### Solve the problem
        sol_sub, _ = ADMM_single(S=S, lambda1=lambda1, Omega_0=np.eye(p),
                          verbose=verbose, latent=False, tol=tol, rtol=rtol)

        G = sol_sub['Theta'].astype(bool).astype(int)

        sub_sol_dict[f"subsample_{i+1}"] = {"Theta": sol_sub['Theta'],
                                            "Omega": sol_sub['Omega'],
                                            "X": sol_sub['X'], "S": S,
                                            "adjacency_matrix": G}

    sol_dict[lambda1] = sub_sol_dict

    estimates = list()

    for i in range(N):

        ### adjacency matrix (like in the paper) or estimate (like in R)?
        estimate = sol_dict[lambda1][f'subsample_{i+1}']["adjacency_matrix"]

        ### psi_Lambda_st in the paper notation
        estimates.append(estimate)

    ### theta_b_st_hat in the paper notation
    edge_average = np.mean(estimates, axis=0)

    ### the variance of a Bernoulli distribution (ksi_b_st in the paper notation)
    edge_instability = 2 * edge_average * (1 - edge_average)

    ### D_b in the paper notation, note we use just lambda not 1/lambda as in the original paper
    total_instability = 2 * np.sum(edge_instability) / (p * (p - 1))

    sol_dict[lambda1]["D_b"] = total_instability

instabilities = [sol_dict[lambda1]["D_b"] for lambda1 in sol_dict]

# find the index of the largest lambda where total_instability <= stars_thresh
indices = np.where(np.array(instabilities) <= beta)[0]

if len(indices) > 0:
    opt_index = np.max(indices)
else:
    opt_index = indices[0] # pick the only one

lambda_star = lambda1_range[opt_index]

print(f"Optimal lambda: {lambda_star}")

### do the refit of the model
S0 = np.cov(X, bias = True)
S = scale_array_by_diagonal(S0)
if p != S.shape[0]:
    raise Exception("Check covariance shape!")


sol, _ = ADMM_single(S=S, lambda1=lambda_star, Omega_0=np.eye(p),
                  verbose=verbose, latent=False, tol=tol, rtol=rtol)

sol_dict["refit"] = {"Theta": sol['Theta'],
                     "Omega": sol['Omega'],
                     "X": sol['X'], "S": S, lambda_star: lambda_star,
                     "adjacency_matrix": sol['Theta'].astype(bool).astype(int)}



### SLR Model (84 minutes)

mu1 = 1
lambda1_range = np.logspace(0, -2, 10)
rank_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

n_samples = X.shape[1]
low_rank_dict = dict()

for r in rank_range:
    print(f"--------------     Rank: {r} --------------")

    sol_dict = dict()

    for lambda1 in lambda1_range:
        print(f"-------------- Lambda: {lambda1} --------------")

        sub_sol_dict = dict()

        ### Loop through each subsample
        for i in range(N):
            subsample_slice = subsamples[:, :, i]
            print(f"Subsample {i+1}:")

            S0 = np.cov(subsample_slice, bias=True)
            S = scale_array_by_diagonal(S0)

            if p != S.shape[0]:
                raise Exception("Check covariance shape!")

            ### Solve the sparse problem
            sub_sol, _ = ADMM_single(S=S, lambda1=lambda1, mu1=mu1, r=r, Omega_0=np.eye(p), 
                            verbose=verbose, latent=True, tol=tol, rtol=rtol)

            G = sub_sol['Theta'].astype(bool).astype(int)

            sub_sol_dict[f"subsample_{i+1}"] = {"Theta": sub_sol['Theta'],
                                                "L": sub_sol['L'],
                                                "Omega": sub_sol['Omega'],
                                                "X": sub_sol["X"], "S": S,
                                                "adjacency_matrix": G}

        # Store subsample solutions in sol_dict[lambda1]
        sol_dict[lambda1] = {"sub_samples": sub_sol_dict}

        estimates = list()

        for i in range(N):
            ### adjacency matrix (like in the paper) or estimate (like in R)?
            estimate = sol_dict[lambda1]["sub_samples"][f'subsample_{i+1}']["adjacency_matrix"]

            ### psi_Lambda_st in the paper notation
            estimates.append(estimate)

        ### theta_b_st_hat in the paper notation
        edge_average = np.mean(estimates, axis=0)

        ### the variance of a Bernoulli distribution (ksi_b_st in the paper notation)
        edge_instability = 2 * edge_average * (1 - edge_average)

        ### D_b in the paper notation
        total_instability = 2 * np.sum(edge_instability) / (p * (p - 1))

        # Add instability measure to sol_dict[lambda1]
        sol_dict[lambda1]["D_b"] = total_instability

    # Collect all instabilities for current mu1
    instabilities = [sol_dict[lambda1]["D_b"] for lambda1 in sol_dict if "D_b" in sol_dict[lambda1]]

    # Find the index of the largest lambda where total_instability <= stars_thresh
    indices = np.where(np.array(instabilities) <= beta)[0]

    if len(indices) > 0:
        opt_index = np.max(indices)
    else:
        opt_index = indices[0]  # pick the only one

    lambda_star = lambda1_range[opt_index]

    print(f"Optimal lambda: {lambda_star}")

    ### Do the refit of the model
    S0 = np.cov(X, bias=True)
    S = scale_array_by_diagonal(S0)
    if p != S.shape[0]:
        raise Exception("Check covariance shape!")

    sol, _ = ADMM_single(S=S, lambda1=lambda_star, mu1=mu1, r=r, Omega_0=np.eye(p),
                    verbose=verbose, latent=True, tol=tol, rtol=rtol)
    
    print(f"Rank of the low-rank matrix: {np.linalg.matrix_rank(sol['L'])}")


    ebic_result = ebic(S, sol['Theta'], N=n_samples, gamma=0.5)

    sol_dict["refit"] = {"Theta": sol['Theta'], "Omega": sol['Omega'], "X": sol['X'],
                        "L": sol["L"], "S": S, "lambda_star": lambda_star, "eBIC": ebic_result,
                        "adjacency_matrix": sol['Theta'].astype(bool).astype(int)}

    low_rank_dict[r] = sol_dict