import pandas as pd
import numpy as np
from collections import Counter


def merge_ige(id_df, ige_df, taxa_df):
    """
    Merge the input DataFrames based on specific columns, drop columns and rows with NaN values,
    calculate column count differences, and merge with a taxa DataFrame.

    Args:
        id_df (pandas.DataFrame): DataFrame containing identifier data.
        ige_df (pandas.DataFrame): DataFrame containing ige data.
        taxa_df (pandas.DataFrame): DataFrame containing taxa data.

    Returns:
        pandas.DataFrame: The merged and processed DataFrame.

    """
    # Merge by zz_nr
    ige = id_df.merge(ige_df, on='zz_nr', how='inner')

    # Delete zz_nr column
    del ige['zz_nr']

    # Set u3_16s_id as the index
    ige = ige.set_index("u3_16s_id")

    # Drop NaNs by rows
    ige = ige.dropna()

    # Calculate the difference in column counts
    c1 = Counter(ige.columns)
    ige = ige.loc[:, (ige != 0).any(axis=0)]
    c2 = Counter(ige.columns)
    column_diff = c1 - c2

    # Print the difference in column counts
    print("Drop", column_diff)

    print("Any null values?", ige.isnull().values.any())
    
    ige = taxa_df.merge(ige, left_index=True, right_index=True)

    return ige


def overlap_with_16s_samples(dataframe, ids_to_select):
    """
    Filter a DataFrame based on a set of indices.

    Parameters:
    - dataframe: pandas DataFrame
    - indices_to_select: set of indices to select

    Returns:
    - filtered DataFrame
    """
    return dataframe[dataframe.index.isin(ids_to_select)]


def replace_imputed_and_empty_with_nan(dataframe, selected_cols: list, imputed_value=-999.0):
    """
    Replace imputed missing values and empty strings with NaN in the DataFrame.

    This function takes a pandas DataFrame as input and replaces the imputed missing
    values (specified by `imputed_value`) and empty strings with NaN.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame to process.
    imputed_value (int or float, optional): The value to be replaced with NaN for imputed missing values.
        Default is -999.0.

    Returns:
    pandas.DataFrame: A new DataFrame with imputed missing values and empty strings replaced by NaN.
    """
    df = dataframe.copy()
    ### replace imputed missing values with NaN
    df = df.replace(imputed_value, np.nan)
    ### replace empty strings with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    for var in selected_cols:
        df[var] = df[var].replace({9: np.nan})
    return df


def filter_and_process_asv_table(ASV_table, freq_threshold):
    """
    Filter and process the ASV table to retain only relevant columns and samples.

    This function performs the following steps:
    1. Counts the presence of bacteria in the ASV table.
    2. Calculates the frequency of each bacterium across all samples.
    3. Filters the ASV table to retain only those taxa with non-zero frequencies.
    4. Further filters the ASV table to retain taxa with a frequency of at least 0.01 across samples.
    5. Removes columns (samples) from the ASV table where the mean abundance is zero.

    Parameters:
    - ASV_table (pd.DataFrame): Input ASV table DataFrame with taxa as rows and samples as columns.

    Returns:
    - asv_top_samples (pd.DataFrame): Processed ASV table containing only the relevant taxa and samples.
    - asv_samples_ids (set): Set of column IDs (sample IDs) in the resulting ASV table.
    """
    # Count the presence of bacteria
    taxa_freq = ASV_table > 0
    
    # Frequency of bacterium across all samples
    taxa_freq = taxa_freq.sum(axis=1)
    
    # Create DataFrame with taxa frequencies
    freq_frame = pd.DataFrame(taxa_freq, columns=['taxa frequency'])
    
    # Filter taxa frequencies for non-zero values
    nonzero_freq = freq_frame[freq_frame['taxa frequency'] > 0]
    
    # Filter ASV table based on non-zero frequencies
    ASV_table = ASV_table[ASV_table.index.isin(nonzero_freq.index)]
    
    # Calculate sample frequencies and filter for "freq_threshold"
    freq_sample = nonzero_freq['taxa frequency'] / ASV_table.shape[1]
    freq_sample = pd.DataFrame(freq_sample)
    freq_sample = freq_sample[freq_sample['taxa frequency'] > freq_threshold]
    
    # Filter ASV table based on >= 0.01 frequencies
    asv_top_samples = ASV_table[ASV_table.index.isin(freq_sample.index)]
    
    # Filter columns with zero mean
    asv_samples_ids = set(asv_top_samples.columns)
    means = asv_top_samples.mean()
    zero_mean_cols = means[means == 0].index
    
    # Drop columns with zero mean if any
    asv_top_samples = asv_top_samples.drop(zero_mean_cols, axis=1)
    asv_samples_ids = set(asv_top_samples.columns)
    
    return asv_top_samples, asv_samples_ids


def process_taxonomic_data(taxa, taxonomy_levels):
    """
    Process taxonomic data in the provided DataFrame.

    Parameters:
    - taxa (pd.DataFrame): DataFrame containing a column named 'Taxon' with taxonomic information.
    - taxonomy_levels (dict): Dictionary mapping taxonomic levels to their corresponding regular expressions.

    Returns:
    - pd.DataFrame: Processed DataFrame with taxonomic ranks split into different columns, 
      blank spaces removed, and a new column 'ASVs' added.
    """
    # Split taxonomic ranks into different columns
    taxa_sep = taxa['Taxon'].str.split(';', expand=True)

    # Rename taxonomic ranks with full names
    taxa_sep.columns = taxonomy_levels.keys()

    # Drop rows with missing species
    taxa_sep = taxa_sep[taxa_sep.species.notnull()]

    # Remove blank spaces from taxonomic ranks
    taxa_sep[taxa_sep.columns] = taxa_sep.apply(lambda x: x.str.strip())

    # Subtract "s" from the string names in the 'species' column
    taxa_sep['species'] = taxa_sep['species'].map(lambda x: x.lstrip('s'))
    taxa_sep['species'] = taxa_sep['genus'] + taxa_sep['species']

    # Add 'ASVs' from index values
    taxa_sep["ASVs"] = taxa_sep.index

    return taxa_sep


def filter_samples_by_status(w, status):
    """
    Filter sample IDs based on the given status.

    Parameters:
    - w (pandas.Series): A pandas Series containing status values (0 or 1).
    - status (str): The status to filter samples by. Should be either "allergy" or "control".

    Returns:
    - list of str: List of sample IDs corresponding to the specified status.

    Raises:
    - ValueError: If the provided status is not valid.
    """
    if "allergy" in status:
        sample_ids = w[w == 0].index
    elif "control" in status:
        sample_ids = w[w == 1].index
    else:
        raise ValueError("Invalid status value")

    sample_ids = [str(x) for x in sample_ids]
    
    return sample_ids


def create_taxa_dict(ASV_table, taxa_sep):
    """
    Create a dictionary of grouped DataFrames based on taxonomic levels.

    Parameters:
    - ASV_table (pd.DataFrame): DataFrame containing ASV information.
    - taxa_sep (pd.DataFrame): DataFrame with taxonomic information, where each column represents a taxonomic level.

    Returns:
    - dict: Dictionary where keys are taxonomic levels, and values are DataFrames with grouped and summed ASV data.
    """
    taxa_dict = dict()

    for level in taxa_sep.columns:
        df_level = ASV_table.join(taxa_sep[level])
        df_level = df_level.groupby(level).sum()

        taxa_dict[level] = df_level

    return taxa_dict


def rename_columns(ige_df, mask: dict):
    """
    Rename columns in the input DataFrame based on a predefined mapping.

    Args:
        ige_df (pandas.DataFrame): DataFrame containing the columns to be renamed.
        mask (dict): Dictionary containing new names for proteins.

    Returns:
        pandas.DataFrame: The DataFrame with renamed columns.

    """
    ige_names = {}
    
    for col in ige_df.columns:
        for prefix, new_prefix in mask.items():
            if prefix in col:
                ige_names[col] = new_prefix
                break

    new_names = ige_names | mask
    
    # Rename columns using the dictionary
    ige = ige_df.rename(columns=new_names, inplace=False)
    
    return ige


def count_taxa_presence(table):
    """
    Count the presence of bacteria in the table and filter out rows with all zeros.

    Parameters:
    - table (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Filtered DataFrame with rows containing at least one non-zero value.
    """
    taxa_freq = table > 0
    taxa_freq = taxa_freq.sum(axis=1)
    freq_frame = pd.DataFrame(taxa_freq, columns=['taxa frequency'])
    nonzero_freq = freq_frame[freq_frame['taxa frequency'] > 0]
    
    return table[table.index.isin(nonzero_freq.index)]


def filter_low_frequency_samples(table, threshold=0.01):
    """
    Filter samples with low taxa frequencies.

    Parameters:
    - table (pd.DataFrame): Input DataFrame.
    - threshold (float): Threshold for taxa frequency.

    Returns:
    pd.DataFrame: Filtered DataFrame with samples meeting the frequency threshold.
    """
    freq_sample = table.sum(axis=1) / table.shape[1]
    freq_sample = pd.DataFrame(freq_sample, columns=['taxa frequency'])
    freq_sample = freq_sample[freq_sample['taxa frequency'] >= threshold]
    
    return table[table.index.isin(freq_sample.index)]


def filter_zero_mean_columns(table):
    """
    Filter out columns with zero mean.

    Parameters:
    - table (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with columns having non-zero mean.
    """
    means = table.mean()
    zero_mean_cols = means[means == 0].index
    
    return table.drop(zero_mean_cols, axis=1)


def exclude_values_difference(set_a, set_b):
    """
    Compute the difference between two sets and convert the result to a list of integers.

    Parameters:
    - set_a (set): First set.
    - set_b (set): Second set.

    Returns:
    list: List of integers representing the difference between set_a and set_b.
    """
    return [int(x) for x in set_a.difference(set_b)]


def threshold_and_get_dropped_columns(df, threshold):
    """
    Drop columns from the DataFrame that have more than a certain threshold
    percentage of missing values and return the set of dropped columns.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        threshold (float): The threshold value (between 0 and 1) for removing columns
                           based on the percentage of missing values. Columns with
                           missing values exceeding this threshold will be dropped.

    Returns:
        pandas.DataFrame: The modified DataFrame after dropping columns with more
                          than the specified threshold of missing values.
        set: A set containing the names of the dropped columns.
    """
    all_columns = set(df.columns)
    # Drop columns with more than 90% of missing values
    df = df.dropna(thresh=df.shape[0] * threshold, axis=1)

    # Compute dropped columns and print the result
    filtered_columns = set(df.columns)
    print("Number of features left after {0}% thresholding - {1}".format(int((1 - threshold) * 100), len(filtered_columns)))
    dropped_columns = all_columns - filtered_columns
    print(dropped_columns)

    return df, dropped_columns


def drop_selected_features(df, mask, col_exclude):
    """
    Drop selected features from dataframe.

    Args:
        df (pandas.DataFrame): DataFrame containing the original data.
        mask (dict): Dictionary mapping original column names to desired column names.
        col_exclude (list): List of columns to drop.

    Returns:
        pandas.DataFrame: The subsetted DataFrame containing only IgE values.

    """
    cols = [col for col in mask.values() if col in df.columns]
    
    sub_df = df.loc[:, cols]
    
    for col in col_exclude:
        # Select column
        cols = sub_df.filter(regex=col).columns
        sub_df = sub_df.drop(columns=cols, inplace=False)

    return sub_df


def drop_zero_features(df):
    """
    Drops columns in the given DataFrame where all values are zero.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame from which columns with all zero values will be removed.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with columns containing only zeros removed.
    """
    zero_columns = df.columns[(df == 0).all()]
    print(f"Columns dropped (all zeros): {list(zero_columns)}")
    
    return df.drop(columns=zero_columns)
    


def select_features(ige_df, columns):
    """
    Selects IgE columns related to the specified features from the input DataFrame.

    Args:
        ige_df (pandas.DataFrame): Input DataFrame containing IgE columns.
        columns (list): List of features to select.

    Returns:
        pandas.DataFrame: DataFrame with selected IgE columns related to the specified features.
    """
    # Select IgE columns related to food
    food_columns = set(columns)
    food = [column for column in ige_df.columns if any(food_col in column for food_col in food_columns)]
    food_ige = ige_df[food]

    return food_ige


def aggregate_features(df, columns):
    """
    Calculates the column sums for the specified columns in a DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (list): List of column prefixes to match and calculate sums.

    Returns:
        pandas.DataFrame: DataFrame with aggregated column sums.
    """
    ige_dict = {}

    for col in columns:
        # Filter columns that start with the given prefix
        matched_columns = df.filter(regex=f"{col}")
        
        bool_df = matched_columns.astype(bool).astype(int)
        
        col_sum = bool_df.sum(axis=1)
        
        ige_dict[col] = col_sum.to_dict()
        
    print("Now, data represents presence/absence information!")
        
    ige = pd.DataFrame(ige_dict)

    return ige


def drop_zeros(df):
    """
    Drops rows and columns containing only zeros from a Pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with rows and columns containing only zeros removed.
    """
    # Drop rows with all zeros
    df = df[~(df == 0).all(axis=1)]
    
    # Drop columns with all zeros
    df = df.loc[:, (df != 0).any(axis=0)]

    return df


def filter_counts(df, threshold):
    """
    Filters columns from a DataFrame based on the sum of their values.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        threshold (int): Minimum sum threshold for column filtering.

    Returns:
        pandas.DataFrame: DataFrame with filtered columns.
    """
    column_sums = df.sum(axis=0)
    selected_columns = column_sums[column_sums > threshold].index
    filtered_df = df[selected_columns]
    
    filtered_df = filtered_df.astype(bool).astype(int)

    return filtered_df


def intersect_features(ige_df: pd.DataFrame, count_df: pd.DataFrame):
    """
    Selects rows from ige_df based on the intersection of indices in count_df columns.

    Args:
        ige_df (pd.DataFrame): DataFrame containing IgE measurements.
        count_df (pd.DataFrame): DataFrame containing counts.

    Returns:
        pd.DataFrame: Selected rows from ige_df based on the intersection of indices.

    """
    # Extract features index
    features_index = list(map(int, count_df.columns))

    # Print number of samples in count table
    num_samples_count_table = len(features_index)
    print("Number of samples in count table:", num_samples_count_table)

    # Extract index list
    index_list = list(ige_df.index)

    # Print number of samples with IgE measurements
    num_samples_ige_measurements = len(index_list)
    print("Number of samples with IgE measurements:", num_samples_ige_measurements)

    # Get the intersection of valid indices
    valid_indices = list(set(index_list) & set(features_index))

    # Print number of samples in intersection
    num_samples_intersection = len(valid_indices)
    print("Number of samples in intersection:", num_samples_intersection)

    # Select rows by index from the list
    selected_rows = ige_df.loc[valid_indices]

    return selected_rows


def add_noise_to_data(df, delta_correction=1000, seed=None):
    """
    Adds a small amount of random noise to the values in a DataFrame to prevent tied ranks.

    This function takes a DataFrame and generates a new DataFrame with the same shape,
    where random noise is added to each element. The noise for each row is generated
    based on the smallest difference between unique values in that row, ensuring that
    ties are broken while maintaining the original data's structure.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing numeric values to which noise will be added.
    delta_correction : float, optional (default=1000)
        A factor used to adjust the interval of added noise. The smaller this value,
        the larger the noise added. Increasing `delta_correction` decreases the noise 
        magnitude, making it closer to zero.
    seed : int, optional
        Seed for the random number generator for reproducibility.

    Returns:
    -------
    pandas.DataFrame
        A new DataFrame with the same shape as the input, but with small random noise
        added to each element.

    Notes:
    -----
    - The amount of noise added is determined by the smallest difference between consecutive
      unique values in each row, divided by `delta_correction`. This ensures that the noise 
      is very small relative to the original values.
    - If a row contains only a single unique value, a default noise interval is used.
    - Adjust `delta_correction` to control the precision of noise addition.
    
    """
    A = df.copy()
    
    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Create an empty DataFrame E with the same shape and index/column structure as A
    E = pd.DataFrame(index=A.index, columns=A.columns)
    
    # Iterate over each row in the DataFrame A
    for idx, row in A.iterrows():
        unique_vals = np.unique(row.values)
        
        # Calculate delta as the smallest difference between consecutive unique values
        delta = np.min(np.diff(np.sort(unique_vals))) if len(unique_vals) > 1 else 1
        a = -delta / delta_correction  # Define a small interval around zero
        b = -a
        
        # Generate random noise for the row
        epsilon = a + (b - a) * np.random.rand(len(row))
        
        # Assign this noise to the corresponding row in E
        E.loc[idx] = epsilon
    
    # Add noise to the original DataFrame and convert to float64
    A_eps = A + E
    
    return A_eps.astype(np.float64)