import numpy as np
import pandas as pd
import igraph as ig


# pairwise difference between real-univariate covariate of treated VS control group
def pairDist(treated=np.array, control=np.array):
        
    D = treated[:, None] - control
    
    return D


# pairwise absolute difference between real-univariate covariates of treated VS control group
def abs_pairDist(treated=np.array, control=np.array):
        
    D = np.abs(treated[:, None] - control)
    
    return D


# pairwise difference between factor-valued (i.e. bounded integer-valued) covariates 
# (e.g. day of the week, month, ...) of treated VS control group, assuming the facotr levels are cyclic
# and only the shortest difference modulo nb_levels matters.
def pairModuloDist(treated=np.array, control=np.array, nb_levels=int):
    # test here
    
    categorical_treated = False
    t_str_value = []
    
    for i in treated:
        if isinstance(i, str):
            t_str_value.append(True)
            
    if np.any(t_str_value):
        categorical_treated = True
    
    if categorical_treated:
        treated = pd.get_dummies(treated, dummy_na=True)
        treated = treated.values.argmax(1)
        
    categorical_control = False
    c_str_value = []
    
    for i in control:
        if isinstance(i, str):
            c_str_value.append(True)
            
    if np.any(c_str_value):
        categorical_control = True
            
    if categorical_control:
        control = pd.get_dummies(control, dummy_na=True) # Add a column to indicate NaNs, if False NaNs are ignored.
        control = control.values.argmax(1) #Returns the indices of the maximum values along the y-axis.
        
    
    treated_control = pairDist(treated.astype(int), control.astype(int)) % nb_levels
    control_treated = pairDist(control.astype(int), treated.astype(int)) % nb_levels
    
    pmin = np.minimum(treated_control, np.transpose(control_treated))
    
    return pmin



# pairwise difference between covariates of treated VS control group
# Inputs: treated/control are of covariate vectors (one entry per unit, for a given covariate)
# Outputs: pairwise difference matrix
def pairdifference(treated=np.array, control=np.array):
    
    categorical = False
    str_value = []
    
    for i in treated:
        if isinstance(i, str):
            str_value.append(True)
            
    if np.any(str_value):
        categorical = True
            
    
    if categorical:
        
        nb_levels = len(set(treated))
        D_mod = pairModuloDist(treated, control, nb_levels)
        
        return D_mod
    
    else:
        
        D_abs = abs_pairDist(treated,control)
        
        return D_abs
    
    
def discrepancyMatrix(treated, control, thresholds, scaling=None):
    """
    Compute the discrepancy matrix between treated and control units based on given thresholds and scaling factors.

    Parameters:
    - treated (pd.DataFrame): DataFrame containing the treated units with covariates.
    - control (pd.DataFrame): DataFrame containing the control units with covariates.
    - thresholds (list or np.ndarray): List or array of thresholds for each covariate. If a threshold is NaN, it is ignored.
    - scaling (list or np.ndarray, optional): List or array of scaling factors for each covariate. If None, no scaling is applied.

    Returns:
    - np.ndarray: A matrix (2D array) of discrepancies where each entry (i, j) corresponds to the discrepancy between treated unit i and control unit j. 
    
    """
    
    nb_covariates = treated.shape[1]
    
    nrow = treated.shape[0]
    ncol = control.shape[0]
    D = np.zeros(shape=(nrow, ncol))
    
    non_admissible = np.full((nrow, ncol), False)
    
    for i in range(0, nb_covariates):
        
        if not np.isnan(thresholds[i]):
            
            t = np.array(treated.iloc[:, i])
            c = np.array(control.iloc[:, i])
            
            differences = pairdifference(t, c)
            D = D + differences*scaling[i]
            
            differences[np.isnan(differences)] = 0
             
            if thresholds[i] >= 0:
            
                non_admissible = non_admissible + (differences > thresholds[i])
            
            elif thresholds[i] < 0:
            
                non_admissible = non_admissible + (differences <= np.abs(thresholds[i]))
    
    D = D / nb_covariates
    
    D[non_admissible] = np.nan
    
    return D


def construct_network(discrepancies, N_treated, N_control):
    """
    Construct a bipartite network from discrepancies matrix.

    Parameters:
    - discrepancies (numpy.ndarray): Matrix of discrepancies.
    - N_treated (int): Number of treated units.
    - N_control (int): Number of control units.

    Returns:
    - igraph.Graph: Bipartite network.
    - dict: Dictionary of matched pairs.
    - pd.DataFrame: DataFrame of matched pairs.
    """
    adj = np.isnan(discrepancies)
    edges_mat = np.argwhere(adj == False) # !!! indices are sorted
    weights = []

    for i in range(0, edges_mat.shape[0]):
        row = edges_mat[i][0]
        col = edges_mat[i][1]
        w = 1 / (1 + discrepancies[row][col])
        weights.append(w)

    weights = np.array(weights)

    edges_vector = edges_mat

    for i in range(0, edges_vector.shape[0]):
        edges_mat[i][1] = edges_vector[i][1] + N_treated

    edges_vector = edges_mat.flatten()
    
    t_nodes = np.repeat(True, N_treated)
    c_nodes = np.repeat(False, N_control)
    nodes = np.concatenate((t_nodes, c_nodes), axis=None)
    
    g = ig.Graph.Bipartite(nodes, edges_mat)
    assert g.is_bipartite()
    
    matching = g.maximum_bipartite_matching(weights=weights)
    
    pairs_dict = dict()
    N_matched = 0

    for i in range(0, N_treated+1):
        if matching.is_matched(i):
            N_matched += 1
            pairs_dict[N_matched] = [i, matching.match_of(i) - N_treated]
            
    return g, pairs_dict


def process_matched_pairs(pairs_dict, treated_units, control_units):
    """
    Process matched pairs and create a DataFrame of matched units.

    Parameters:
    - pairs_dict (dict): Dictionary of matched pairs.
    - treated_units (pd.DataFrame): DataFrame of treated units.
    - control_units (pd.DataFrame): DataFrame of control units.

    Returns:
    - pd.DataFrame: DataFrame of matched pairs.
    """
    treated_units = treated_units.copy()
    control_units = control_units.copy()
    
    
    matched = []
    total_nb_match = 0

    for i in range(1, len(pairs_dict)):
        total_nb_match = total_nb_match + 1

        # Save pair number
        treated_units.loc[pairs_dict[i][0], "pair_nb"] = total_nb_match
        control_units.loc[pairs_dict[i][1], "pair_nb"] = total_nb_match

        matched.append(treated_units.iloc[pairs_dict[i][0], :])
        matched.append(control_units.iloc[pairs_dict[i][1], :])

    matched_df = pd.DataFrame(matched, columns=treated_units.columns)
    
    return matched_df


def generate_simulated_outcomes(matched_df, n_col):
    """
    Generate simulated outcomes based on matched pairs data.

    Parameters:
    - matched_df (pd.DataFrame): DataFrame of matched pairs data.
    - n_col (int): Number of columns in the simulated outcomes matrix.

    Returns:
    - pd.DataFrame: DataFrame of simulated outcomes.
    """
    n_total = len(matched_df.W)
    n_treated = len(matched_df[matched_df.W == 0])

    # Initialize an empty matrix to store simulated outcomes
    W_sim = np.empty([n_total, n_col])
    np.random.seed(123)

    for t in range(n_col):
        W_sim_to_fill = np.empty(n_total)
        flip_coin = np.random.binomial(n=1, p=0.5, size=n_treated)
        W_sim_to_fill[np.arange(start=0, stop=(n_total - 1), step=2)] = flip_coin
        W_sim_to_fill[np.arange(start=1, stop=n_total, step=2)] = 1 - flip_coin
        W_sim[:, t] = W_sim_to_fill

    # Create a DataFrame with unique simulated outcomes
    W_unique = np.unique(W_sim, axis=1)
    W_unique = pd.DataFrame(W_unique, index=matched_df.index)

    return W_unique

