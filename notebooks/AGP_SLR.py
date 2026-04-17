#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('/container/mount/point')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.cluster.hierarchy import linkage, leaves_list

from gglasso.helper.utils import sparsity
from gglasso.helper.model_selection import ebic

from utils.helper import scale_array_by_diagonal
from utils.stability_selection import subsample
from utils.solver import ADMM_single


# FIXED: SE-style CLR replaces transform_features multiplicative zero-replacement
# — see analysis/SLR_diagnostic_summary.md §3
def _clr_se(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    CLR transformation matching SPIEC-EASI's .spiec.easi.norm: t(clr(data + 1, 1)).
    Adds a uniform pseudocount of +1 to ALL entries (not just zeros), then applies
    centered log-ratio per sample.  Input: p×n DataFrame of raw counts.
    """
    X = counts_df.copy().astype(float) + 1.0
    log_X = np.log(X)
    # subtract per-sample log-geometric-mean (mean over taxa axis for each sample)
    return log_X.sub(log_X.mean(axis=0), axis=1)


# Docker:
# 
# docker run -v ~/Causal_Sparse_Low_Rank_Microbiome_Tutorial:/container/mount/point -it -p 8888:8888 --rm --name rpy rpy2:latest
# 
# source opt/python3_env/bin/activate
# 
# jupyter lab --port=8888 --ip 0.0.0.0 --no-browser --allow-root &
# 
# Check the order of clustering and use covarince instead of correlation because there is a bug in SPIEC-EASI
# Run SE with these covariances and reproduce the eigen plot of Theta -> proove that SE and my implementation gives the same solution
# Only after that do networks and plotting
# 

# ### Import American Gut Project data

# In[2]:


matched_df = pd.read_csv('data/AGP/matched_df_W.csv', sep=',', index_col=0)

f_taxa_smoker = pd.read_csv('data/AGP/tax_table_smoker.csv', sep=',', index_col=0)
f_count_smoker = pd.read_csv('data/AGP/otu_table_smoker.csv', sep=',', index_col=0)  # N x p

f_taxa_non_smoker = pd.read_csv('data/AGP/tax_table_non_smoker.csv', sep=',', index_col=0)
f_count_non_smoker = pd.read_csv('data/AGP/otu_table_non_smoker.csv', sep=',', index_col=0)  # N x p

print("Bacterial families for both cases are the same:", (f_count_smoker.columns == f_count_non_smoker.columns).all())
family_names = f_count_smoker.columns
names_dict = {index: (f_taxa_smoker.loc[index, "Family"] if pd.notna(f_taxa_smoker.loc[index, "Family"]) else f"f_unknown_{index}") for index in family_names.map(int)}


# ### Perform Subsampling

# In[3]:


level = "family"
model = "sparse"
N = 20  # FIXED: N=10 → N=20 to match SE rep.num=20 — see analysis/SLR_diagnostic_summary.md §4
lambda1_range = np.logspace(0, -2, 10)

# Process both smoker and non-smoker cases
data_dict = dict()

for case, f_count in zip(["smoker", "non_smoker"], [f_count_non_smoker, f_count_smoker]):
    print(f"Processing {case} case")
    
    # Select count table according to selected taxonomic level
    counts = f_count.copy().T
    p, n = counts.shape
    print(f"Level {level}: \n Shape(p,N): {counts.shape}")

    sample_ids = list(map(str, matched_df[matched_df["W"] == 1].index))

    # FIXED: uniform +1 CLR matches SPIEC-EASI .spiec.easi.norm — see analysis/SLR_diagnostic_summary.md §3
    clr_counts = _clr_se(counts)
    np.random.seed(42)  # FIXED: reproducibility — see analysis/SLR_diagnostic_summary.md §4
    subsamples = subsample(clr_counts.values, N)

    for i in range(subsamples.shape[2]):  # Loop through each subsample
        subsample_slice = subsamples[:, :, i]  # Access the i-th subsample
        print(f"Subsample {i+1}:")
        print(subsample_slice.shape)

    print(f"Print subsamples object: {subsamples.shape}")

    data_dict[case] = {
        "counts": counts,
        "clr_counts": clr_counts,
        "subsamples": subsamples
    }


# In[4]:


test = pd.DataFrame(np.cov(data_dict["smoker"]["clr_counts"], bias=True))

# fig = px.imshow(test,
#                 color_continuous_scale='RdBu_r',
#                 labels={'color': 'Value'},
#                 title=f'Test: p={p}, n={n}',
#                 color_continuous_midpoint=0)

# # Customize layout for better visuals (optional)
# fig.update_layout(
#     width=850,
#     height=850,
#     xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in S_hier.index.map(int)]),
#     yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in S_hier.columns.map(int)])
# )

# fig.show()

off_diagonal_elements = test.values[np.triu_indices_from(test.values, k=1)]
max_off_diagonal_value = np.max(abs(off_diagonal_elements))
max_off_diagonal_value


# ### Empirical covariance

# In[5]:


cov_dict = {"smoker": dict(), "non_smoker": dict()}
corr_dict = {"smoker": dict(), "non_smoker": dict()}

for case, f_count in zip(["smoker", "non_smoker"], [f_count_non_smoker, f_count_smoker]):
    print(f"Processing {case} case")

    clr_counts = data_dict[case]["clr_counts"]
    
    # Empriical covariance matrix
    S = np.cov(clr_counts, bias=True)
    corr = scale_array_by_diagonal(S)
    if p != S.shape[0]:
        raise Exception("Check covariance shape!")
    
    cov_dict[case] = S
    corr_dict[case] = corr


# In[6]:


### Hierarchical clustering of the correlation matrix
linkage_counts = linkage(cov_dict["smoker"], method='average')
ordered_indices = leaves_list(linkage_counts)

for case, f_count in zip(["smoker", "non_smoker"], [f_count_non_smoker, f_count_smoker]):
    print(f"Processing {case} case")

    S = cov_dict[case]
    # S = corr_dict[case]

    S_hier = pd.DataFrame(S, index=family_names, columns=family_names)
    S_hier = S_hier.iloc[ordered_indices, ordered_indices]

    fig = px.imshow(S_hier,
                color_continuous_scale='RdBu_r',
                labels={'color': 'Value'},
                title=f'Empirical covariance: p={p}, n={n} ({case})', 
                zmin=-1, 
                zmax=1)
                

    # Customize layout for better visuals (optional)
    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in S_hier.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in S_hier.columns.map(int)])
    )

    fig.show()

    fig.write_image(f"plots/png/covaraince_{case}.png")
    fig.write_image(f"plots/pdf/covaraince_{case}.pdf")
    fig.write_image(f"plots/svg/covaraince_{case}.svg")
    fig.write_html(f"plots/html/covaraince_{case}.html")

    # fig.write_image(f"plots/png/correlation_{case}.png")
    # fig.write_image(f"plots/pdf/correlation_{case}.pdf")
    # fig.write_image(f"plots/svg/correlation_{case}.svg")
    # fig.write_html(f"plots/html/correlation_{case}.html")


# ## SPIEC-EASI Solution

# (corresponding code in R)

# ### Sparse part

# In[7]:


se_Theta_smoker = pd.read_csv('data/AGP/theta_smoker.csv', sep=',', index_col=0)
se_Theta_nonsmoker = pd.read_csv('data/AGP/theta_non_smoker.csv', sep=',', index_col=0)

for case, se_Theta in zip(["smoker", "non_smoker"], [se_Theta_smoker, se_Theta_nonsmoker]):
    counts = data_dict[case]["counts"]
    se_Theta.index, se_Theta.columns = family_names, family_names
    se_Theta = se_Theta.iloc[ordered_indices, ordered_indices]

    SP = np.round(sparsity(se_Theta), 4)

    fig = px.imshow(se_Theta,
                    color_continuous_scale='RdBu_r',
                    labels={'color': 'Value'},
                    title=f'SPIEC-EASI: Theta, SP={SP} ({case})',
                    color_continuous_midpoint=0)

    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in se_Theta.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in se_Theta.columns.map(int)])
    )

    fig.show()

    fig.write_image(f"plots/png/se_sparse_part_{case}.png")
    fig.write_image(f"plots/pdf/se_sparse_part_{case}.pdf")
    fig.write_image(f"plots/svg/se_sparse_part_{case}.svg")
    fig.write_html(f"plots/html/se_sparse_part_{case}.html")


# ### Low-rank part

# In[8]:


se_L_smoker = pd.read_csv('data/AGP/low_rank_smoker.csv', sep=',', index_col=0)
se_L_nonsmoker = pd.read_csv('data/AGP/low_rank_non_smoker.csv', sep=',', index_col=0)

for case, se_L in zip(["smoker", "non_smoker"], [se_L_smoker, se_L_nonsmoker]):
    rank_se_L = np.linalg.matrix_rank(se_L)
    print(f"Rank of the low-rank matrix from SPIEC-EASI ({case}): {rank_se_L}")

    se_L.index, se_L.columns = family_names, family_names
    se_L = se_L.iloc[ordered_indices, ordered_indices]

    fig = px.imshow(se_L,
                    color_continuous_scale='RdBu_r',
                    labels={'color': 'Value'},
                    title=f'SPIEC-EASI: L (rank={rank_se_L}) ({case})',
                    color_continuous_midpoint=0)

    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in se_L.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in se_L.columns.map(int)])
    )

    fig.show()

    fig.write_image(f"plots/png/se_lowrank_part_{case}.png")
    fig.write_image(f"plots/pdf/se_lowrank_part_{case}.pdf")
    fig.write_image(f"plots/svg/se_lowrank_part_{case}.svg")
    fig.write_html(f"plots/html/se_lowrank_part_{case}.html")


# ### Likelihood (Omega=S-L)

# In[9]:


for case, se_Theta in zip(["smoker", "non_smoker"], [se_Theta_smoker, se_Theta_nonsmoker]):
    se_Theta.index, se_Theta.columns = family_names, family_names
    se_Theta = se_Theta.iloc[ordered_indices, ordered_indices]

    se_L = pd.read_csv(f'data/AGP/low_rank_{case}.csv', sep=',', index_col=0)
    se_L.index, se_L.columns = family_names, family_names
    se_L = se_L.iloc[ordered_indices, ordered_indices]

    se_Omega = se_Theta - se_L

    fig = px.imshow(se_Omega,
                    color_continuous_scale='RdBu_r',
                    labels={'color': 'Value'},
                    title=f'SPIEC-EASI: Omega ({case})',
                    color_continuous_midpoint=0)

    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in se_Omega.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in se_Omega.columns.map(int)])
    )

    fig.show()

    fig.write_image(f"plots/png/se_omega_part_{case}.png")
    fig.write_image(f"plots/pdf/se_omega_part_{case}.pdf")
    fig.write_image(f"plots/svg/se_omega_part_{case}.svg")
    fig.write_html(f"plots/html/se_omega_part_{case}.html")


# ## GGLasso solution

# In[10]:


### lambda path from SPIEC-EASI
lambda1_range_non_smoker = [3.02721826, 2.37563971, 1.86430695, 1.46303347, 1.14813010, 0.90100654, 0.70707387, 0.55488327, 0.43545018, 0.34172387, 
                            0.26817121, 0.21045003, 0.16515276, 0.12960528, 0.10170903, 0.07981718, 0.06263733, 0.04915527, 0.03857509, 0.03027218]

lambda1_range_smoker = [4.66364059, 3.65983845, 2.87209471, 2.25390496, 1.76877439, 1.38806334, 1.08929654, 0.85483632, 0.67084133, 0.52644942, 
                        0.41313643, 0.32421293, 0.25442933, 0.19966596, 0.15668985, 0.12296392, 0.09649716, 0.07572711, 0.05942761, 0.04663641]

rank_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
verbose = False
tol, rtol = 1e-5, 1e-5
beta = 0  # FIXED: beta=0.05 → beta=0 to match SE default — see analysis/SLR_diagnostic_summary.md §1
mu1 = 1

low_rank_dict = {"smoker": dict(), "non_smoker": dict()}

for case in ["smoker", "non_smoker"]:
    print(f"Processing {case} case")

    control_counts = data_dict[case]["counts"]
    control_clr_counts = data_dict[case]["clr_counts"]
    subsamples = data_dict[case]["subsamples"]

    n_samples = control_clr_counts.shape[1]

    for r in rank_range:
        print(f"--------------     Rank: {r} --------------")

        sol_dict = dict()

        lambda1_range = lambda1_range_smoker if case == "smoker" else lambda1_range_non_smoker

        for lambda1 in lambda1_range:
            print(f"-------------- Lambda: {lambda1} --------------")

            sub_sol_dict = dict()

            ### Loop through each subsample
            for i in range(N):
                subsample_slice = subsamples[:, :, i]
                print(f"Subsample {i+1}:")

                S = np.cov(subsample_slice, bias=True)
                # S = scale_array_by_diagonal(S0)

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

        ### Do the refit of the model on the exactly the same empirical covariance matrix
        S = cov_dict[case]
        # S0 = np.cov(control_clr_counts, bias=True)
        # S = scale_array_by_diagonal(S0)
        # if p != S.shape[0]:
        #     raise Exception("Check covariance shape!")
        

        sol, _ = ADMM_single(S=S, lambda1=lambda_star, mu1=mu1, r=r, Omega_0=np.eye(p),
                        verbose=verbose, latent=True, tol=tol, rtol=rtol)
        
        print(f"Rank of the low-rank matrix: {np.linalg.matrix_rank(sol['L'])}")


        ebic_result = ebic(S, sol['Theta'], N=n_samples, gamma=0.5)

        sol_dict["refit"] = {"Theta": sol['Theta'], "Omega": sol['Omega'], "X": sol['X'],
                            "L": sol["L"], "S": S, "lambda_star": lambda_star, "eBIC": ebic_result,
                            "adjacency_matrix": sol['Theta'].astype(bool).astype(int)}

        low_rank_dict[case][r] = sol_dict


# ### Empirical covariance

# In[11]:


selected_rank = 10

for case in ["smoker", "non_smoker"]:
    gg_S = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["S"], index=family_names, columns=family_names)
    gg_S = gg_S.iloc[ordered_indices, ordered_indices]

    fig = px.imshow(gg_S,
                    color_continuous_scale='RdBu_r',
                    labels={'color': 'Value'},
                    title=f'GGLasso: S_hat ({case})')

    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in gg_S.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in gg_S.columns.map(int)])
    )

    fig.show()


# ### Sparse part

# In[12]:


selected_rank = 10

for case in ["smoker", "non_smoker"]:
    gg_Theta = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["Theta"], index=family_names, columns=family_names)
    gg_Theta = gg_Theta.iloc[ordered_indices, ordered_indices]
    SP = np.round(sparsity(gg_Theta), 4)

    fig = px.imshow(gg_Theta,
                    color_continuous_scale='RdBu_r',
                    labels={'color': 'Value'},
                    title=f'GGLasso: Theta, SP={SP} ({case})',
                    color_continuous_midpoint=0)

    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in gg_Theta.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in gg_Theta.columns.map(int)])
    )

    fig.show()

    fig.write_image(f"plots/png/gg_sparse_part_{case}.png")
    fig.write_image(f"plots/pdf/gg_sparse_part_{case}.pdf")
    fig.write_image(f"plots/svg/gg_sparse_part_{case}.svg")
    fig.write_html(f"plots/html/gg_sparse_part_{case}.html")


# ### Low-rank part

# In[13]:


selected_rank = 10

for case in ["smoker", "non_smoker"]:
    gg_L = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["L"], index=family_names, columns=family_names)
    gg_L = gg_L.iloc[ordered_indices, ordered_indices]
    rank_gg_L = np.linalg.matrix_rank(gg_L)
    print(f"Rank of the low-rank matrix from GGLasso ({case}): {rank_gg_L}")

    fig = px.imshow(gg_L,
                    color_continuous_scale='RdBu_r',  # Adjust the color scale
                    labels={'color': 'Value'},          # Label for color bar
                    title=f'GGLasso: L (rank={rank_gg_L}) ({case})',
                    color_continuous_midpoint=0)

    # Customize layout for better visuals (optional)
    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in gg_L.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in gg_L.columns.map(int)])
    )

    fig.show()

    fig.write_image(f"plots/png/gg_lowrank_part_{case}.png")
    fig.write_image(f"plots/pdf/gg_lowrank_part_{case}.pdf")
    fig.write_image(f"plots/svg/gg_lowrank_part_{case}.svg")
    fig.write_html(f"plots/html/gg_lowrank_part_{case}.html")


# ### Likelihood

# In[14]:


for case in ["smoker", "non_smoker"]:
    gg_Omega = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["Omega"], index=family_names, columns=family_names)
    gg_Omega = gg_Omega.iloc[ordered_indices, ordered_indices]

    fig = px.imshow(gg_Omega,
                    color_continuous_scale='RdBu_r',  
                    labels={'color': 'Value'},
                    title=f'GGLasso: Omega ({case})',
                    color_continuous_midpoint=0)

    # Customize layout for better visuals (optional)
    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in gg_Omega.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in gg_Omega.columns.map(int)])
    )

    fig.show()

    fig.write_image(f"plots/png/gg_omega_part_{case}.png")
    fig.write_image(f"plots/pdf/gg_omega_part_{case}.pdf")
    fig.write_image(f"plots/svg/gg_omega_part_{case}.svg")
    fig.write_html(f"plots/html/gg_omega_part_{case}.html")


# In[15]:


for case in ["smoker", "non_smoker"]:
    gg_S = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["S"], index=family_names, columns=family_names)
    gg_Omega = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["Omega"], index=family_names, columns=family_names)

    I = gg_Omega.dot(gg_S)
    I_reoder = I.iloc[ordered_indices, ordered_indices]

    fig = px.imshow(I_reoder,
                    color_continuous_scale='RdBu_r',  
                    labels={'color': 'Value'},
                    title=f'GGLasso: Omega x S_hat ({case})',
                    color_continuous_midpoint=0)

    # Customize layout for better visuals (optional)
    fig.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in I_reoder.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in I_reoder.columns.map(int)])
    )

    fig.show()

    fig.write_image(f"plots/png/gg_identity_{case}.png")
    fig.write_image(f"plots/pdf/gg_identity_{case}.pdf")
    fig.write_image(f"plots/svg/gg_identity_{case}.svg")
    fig.write_html(f"plots/html/gg_identity_{case}.html")


# In[16]:


for case in ["smoker", "non_smoker"]:
    ranks = list(low_rank_dict[case].keys())
    sparsities = [sparsity(low_rank_dict[case][rank]["refit"]["Theta"]) for rank in ranks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ranks, y=sparsities, mode='lines+markers', name=f'Sparsity ({case})'))

    fig.update_layout(
        title=f'Sparsity vs Rank ({case})',
        xaxis_title='Rank',
        yaxis_title='Sparsity',
        template='plotly_white'
    )

    fig.show()

    for key in low_rank_dict[case].keys():
        print(f"Case: {case}, Rank: {key}")
        print(low_rank_dict[case][key]["refit"]["eBIC"])


# ### SVD of both solutions

# In[306]:


for case in ["smoker", "non_smoker"]:
    se_L = pd.read_csv(f'data/AGP/low_rank_{case}.csv', sep=',', index_col=0)
    gg_L = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["L"], index=family_names, columns=family_names)

    # Perform Singular Value Decomposition (SVD) on both matrices
    U_se, s_se, Vt_se = np.linalg.svd(se_L)
    U_gg, s_gg, Vt_gg = np.linalg.svd(gg_L)

    # Compare eigenvalues
    print(f"Eigenvalues of se_L ({case}):")
    print(s_se)
    print(f"\nEigenvalues of gg_L ({case}):")
    print(s_gg)
    print(f"\nDifference between SE and GG ({case}):")
    print(np.linalg.norm(se_L.values - gg_L.values))
    
    # Plot the spectrum of eigenvalues
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(s_se))), y=s_se, mode='lines+markers', name=f'SE Eigenvalues ({case})'))
    fig.add_trace(go.Scatter(x=list(range(len(s_gg))), y=s_gg, mode='lines+markers', name=f'GG Eigenvalues ({case})'))

    fig.update_layout(
        title=f'Eigenvalue Spectrum Comparison ({case})',
        xaxis_title='Eigenvalue Index',
        yaxis_title='Eigenvalue',
        template='plotly_white'
    )

    fig.show()

    fig.write_image(f"plots/png/spectrum_plot_{case}.png")
    fig.write_image(f"plots/pdf/spectrum_plot_{case}.pdf")
    fig.write_image(f"plots/svg/spectrum_plot_{case}.svg")
    fig.write_html(f"plots/html/spectrum_plot_{case}.html")


# In[328]:


for case, se_Theta in zip(["smoker", "non_smoker"], [se_Theta_smoker, se_Theta_nonsmoker]):
    print(f"{case} case")
    ### SPIEC-EASI
    se_Theta.index, se_Theta.columns = family_names, family_names
    se_Theta = se_Theta.iloc[ordered_indices, ordered_indices]
    se_SP = np.round(sparsity(se_Theta), 4)
    print(f"SE Theta Sparsity {se_SP}")

    se_L = pd.read_csv(f'data/AGP/low_rank_{case}.csv', sep=',', index_col=0)
    se_L.index, se_L.columns = family_names, family_names
    se_L = se_L.iloc[ordered_indices, ordered_indices]

    se_Omega = se_Theta - se_L

    ### GGLasso
    gg_Theta = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["Theta"], index=family_names, columns=family_names)
    gg_Theta = gg_Theta.iloc[ordered_indices, ordered_indices]
    gg_Omega = pd.DataFrame(low_rank_dict[case][selected_rank]["refit"]["Omega"], index=family_names, columns=family_names)
    gg_Omega = gg_Omega.iloc[ordered_indices, ordered_indices]
    
    gg_SP = np.round(sparsity(gg_Theta), 4)
    print(f"GG Theta Sparsity {gg_SP} \n")

    diff_Omega = se_Omega - gg_Omega
    diff_Theta = se_Theta - gg_Theta

    fig_omega = px.imshow(diff_Omega,
                          color_continuous_scale='RdBu_r',  
                          labels={'color': 'Value'},
                          title=f'Difference (SE-GG): Omega ({case})',
                          color_continuous_midpoint=0)

    fig_omega.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in diff_Omega.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in diff_Omega.columns.map(int)])
    )

    fig_omega.show()

    fig_omega.write_image(f"plots/png/difference_omega_{case}.png")
    fig_omega.write_image(f"plots/pdf/difference_omega_{case}.pdf")
    fig_omega.write_image(f"plots/svg/difference_omega_{case}.svg")
    fig_omega.write_html(f"plots/html/difference_omega_{case}.html")

    fig_theta = px.imshow(diff_Theta,
                          color_continuous_scale='RdBu_r',  
                          labels={'color': 'Value'},
                          title=f'Difference (SE-GG): Theta ({case})',
                          color_continuous_midpoint=0)

    fig_theta.update_layout(
        width=850,
        height=850,
        xaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in diff_Theta.index.map(int)]),
        yaxis=dict(tickmode='array', tickvals=list(range(len(names_dict))), ticktext=[names_dict[idx] for idx in diff_Theta.columns.map(int)])
    )

    fig_theta.show()

    fig_theta.write_image(f"plots/png/difference_theta_{case}.png")
    fig_theta.write_image(f"plots/pdf/difference_theta_{case}.pdf")
    fig_theta.write_image(f"plots/svg/difference_theta_{case}.svg")
    fig_theta.write_html(f"plots/html/difference_theta_{case}.html")


# In[ ]:




