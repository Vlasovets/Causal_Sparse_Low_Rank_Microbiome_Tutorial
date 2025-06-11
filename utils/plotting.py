import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import networkx as nx
import networkx as nx
import bisect
import plotly.express as px
import scipy.spatial as scs

from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, HoverTool, LabelSet, PointDrawTool
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import RdBu, Blues8
from bokeh.models import Tabs, ColorBar, LinearColorMapper
from bokeh.layouts import row
from bokeh.io import export_svgs, export_png


from pyvis.network import Network
from matplotlib import pyplot as plt
from .heatmap import Clustergram

def presence_absence_heatmap(data, row_labels, column_labels, height=1200, width=1400, colorscale='Viridis'):
    """
    Create a presence-absence heatmap using Plotly.

    Args:
        data (list of lists): A 2D matrix of data for the heatmap, where rows represent samples and columns represent IgE types.
        row_labels (list): Labels for the rows (samples).
        column_labels (list): Labels for the columns (IgE types).
        height (int, optional): Height of the heatmap figure in pixels. Default is 1200.
        width (int, optional): Width of the heatmap figure in pixels. Default is 1400.

    Returns:
        go.Figure: A Plotly figure representing the presence-absence heatmap.
    """
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=column_labels,
        y=row_labels,
        colorscale=colorscale
    ))

    fig.update_layout(
        title='Presence-Absence Heatmap IgEs',
        xaxis_title='IgE type',
        yaxis_title='Samples'
    )
    # Update the layout for the subplots
    fig.update_layout(height=height, width=width, xaxis_tickangle=45)

    return fig



def map_color(row: pd.DataFrame):
    """
    Maps values based on column names to corresponding color values.
    
    Args:
        row (DataFrame): A row from a DataFrame.
        
    Returns:
        str: The color value corresponding to the first column with a value of 1 in the row.
    """
    color_mapping = {
    'nut': "#8E0202",
    'shrimp': "#02688E",
    'kiwi': "#8E8602",
    'meat': "#FFA500"}
    
    for column in row.index:
        if row[column] == 1:
            return color_mapping.get(column)
    # If no allergy
        if row.sum() == 0:
            return "#026E68"


def plot_missing_percentages(df, mask_names, threshold: int=5):
    """
    Plot the percentage of missing values in columns of a DataFrame greater than 10%.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        mask_names (dict): A dictionary with column names mapping to their corresponding labels.

    Returns:
        fig (plotly.graph_objs.Figure): The Plotly figure object containing the bar chart.

    Raises:
        None.

    Notes:
        This function uses Plotly to create a bar chart displaying the percentage of missing values
        for columns with missing percentages greater than 10%. The column names are replaced with
        their corresponding labels using the 'mask_names' dictionary.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    plot_df = df.copy()

    # Rename columns using the dictionary
    plot_df = plot_df.rename(columns=mask_names)

    # Calculate the percentage of missing values for each column
    missing_percentages = (plot_df.isnull().mean() * 100).round(2)

    # Filter columns to keep only those with missing percentages greater than 5%
    missing_percentages = missing_percentages[missing_percentages > threshold]

    # Sort the missing_percentages DataFrame in descending order
    missing_percentages = missing_percentages.sort_values(ascending=False)

    # Create a bar chart showing the percentage of missing values
    fig = go.Figure(data=go.Bar(x=missing_percentages.index, y=missing_percentages.values, marker=dict(color='#960b81')))
    fig.update_layout(title='Percentage of Missing Covariates (>{0}%)'.format(threshold), 
                      xaxis_title='Covariate', yaxis_title='Percentage (%)',
                      width=1000, height=500, xaxis_tickangle=45)
    
    return fig


def count_histogram_plot(df: pd.DataFrame, height: int, width: int, xaxis_title_text: str, yaxis_title_text: str):
    """
    Creates a histogram plot using Plotly.

    Args:
        df (pandas.DataFrame): DataFrame with column counts.
        height (int): Height of the plot in pixels.
        width (int): Width of the plot in pixels.
        xaxis_title_text (str): Title text for the X-axis.
        yaxis_title_text (str): Title text for the Y-axis.

    Returns:
        None
    """
    fig = go.Figure()

    hist_ige = go.Histogram(
        x=df.index,
        y=df['counts'],
        name="total food IgEs",
        histfunc='sum',
        marker_color='#5C1360'
    )

    fig.add_trace(hist_ige)

    fig.update_layout(
        title_x=0.5,
        xaxis={'type': 'category', 'categoryorder': 'total descending'},
        yaxis={'title': 'Number of allergics'},
        xaxis_title_text=xaxis_title_text,
        yaxis_title_text=yaxis_title_text,
        bargap=0.5,
        height=height,
        width=width
    )

    # Add annotations to the histogram bars
    for i in range(len(df.index)):
        fig.add_annotation(
            x=df.index[i],
            y=df['counts'][i] - 1,
            text=str(df['counts'][i]),
            showarrow=False,
            font=dict(size=12, color='white')
        )

    fig.update_layout(
        xaxis_tickangle=45
    )
    
    return fig


def plot_latent_correlation(rLT, width: int=1400, height: int=1400, title: str = None):
    """
    Plots a heatmap of the latent correlation matrix using Plotly.

    Parameters:
        rLT (pandas.DataFrame): The latent correlation matrix to be visualized.

        width (int, optional): The width of the resulting plot in pixels.
            Default is 1400.

        height (int, optional): The height of the resulting plot in pixels.
            Default is 1400.
            
        title (str, optional): The title of the plot.
            Default is None

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the heatmap.

    This function generates a heatmap to visualize the correlations between latent variables
    in a given correlation matrix. It uses Plotly for interactive plotting.

    The provided `rLT` DataFrame should contain correlation values, where rows and columns
    represent latent variables, and the values indicate their pairwise correlations.

    """
    mask = np.triu(np.ones_like(rLT, dtype=bool))
    rLT_masked = rLT.mask(mask)

    heat = go.Heatmap(
        z=rLT_masked,
        x=rLT.columns.values,
        y=rLT.columns.values,
        zmin=-1,  # Sets the lower bound of the color domain
        zmax=1,
        xgap=1,  # Sets the horizontal gap (in pixels) between bricks
        ygap=1,
        colorscale='RdBu_r'
    )

    layout = go.Layout(
        title_text=title,
        title_x=0.5,
        width=width,
        height=height,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
        xaxis_tickangle=45
    )

    fig = go.Figure(data=[heat], layout=layout)
    return fig


def correlation_heatmap(df:pd.DataFrame):
    """
    Creates a correlation heatmap using Plotly.

    Args:
        df(pandas.DataFrame): DataFrame containing the correlation values.

    Returns:
        None
    """
    mask = np.triu(np.ones_like(df, dtype=bool))
    rLT = df.mask(mask)

    heat = go.Heatmap(
        z=rLT,
        x=rLT.columns.values,
        y=rLT.columns.values,
        zmin=0,  # Sets the lower bound of the color domain
        zmax=1,
        xgap=1,  # Sets the horizontal gap (in pixels) between bricks
        ygap=1,
        colorscale='RdBu_r'
    )

    layout = go.Layout(
        title_x=0.5,
        width=850,
        height=850,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig = go.Figure(data=[heat], layout=layout)
    
    return fig


def map_color(row: pd.DataFrame):
    """
    Maps values based on column names to corresponding color values.
    
    Args:
        row (DataFrame): A row from a DataFrame.
        
    Returns:
        str: The color value corresponding to the first column with a value of 1 in the row.
    """
    color_mapping = {
    'nut': "#824432",
    'wheat': "#F5DEB3",
    'shrimp': "#e29a86",
    'kiwi': "#8ee53f",
    'meat': "#f9906f",
    'egg': "#F0EAD6"}
    
    for column in row.index:
        if row[column] == 1:
            return color_mapping.get(column)
        

def abundance_heatmap(df, labels_dict=dict, height=int, width=int, test_ids=list, control_ids=list,  cluster='raw', color_map="PuBu"):
    """
    Create an abundance heatmap using the given DataFrame and labels dictionary.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing abundance data.

    labels_dict : dict
        A dictionary mapping row labels to their corresponding label values.

    height : int
        The height of the heatmap.

    width : int
        The width of the heatmap.

    cluster : str, optional
        The clustering method to use, can be 'raw' for no clustering,
        or 'average', 'single', 'complete', 'ward', etc.

    Returns:
    --------
    heatmap.Clustergram
        An instance of the Clustergram class representing the abundance heatmap.
    """

    columns = list(df.columns.values)
    rows = list(df.index)
    #row_labels = list(labels_dict.values())
    level = df.index.name

    clustergram = Clustergram(
        data=df.loc[rows].values,
        row_labels=rows,
        column_labels=columns,
        color_threshold={
            'row': 250,
            'col': 700 
        },
        column_colors=len(test_ids)*['#F66A71'] + len(control_ids)*['#641E62'],
        color_list={'col': ['#641E62', '#F66A71']},
        color_map=color_map,
        height=height,
        width=width,
        hidden_labels='column',
        col_dist='euclidean',
        row_dist='euclidean',
        dist_fun=scs.distance.pdist,
        cluster=cluster,
        line_width=2
    )

    return clustergram


def simple_heatmap(df, title="Heatmap with Plotly", width=2000, height=1500, colorscale="Greys"):
    """
    Generates a heatmap from a given DataFrame using Plotly.
    
    Parameters:
    - df: DataFrame to be visualized in the heatmap.
    - title: Title of the heatmap.
    - width: Width of the plot in pixels.
    - height: Height of the plot in pixels.
    - colorscale: Colorscale of the heatmap.
    
    Returns:
    - fig: Plotly Figure object with the heatmap.
    """
    # Create the heatmap
    data = [
        go.Heatmap(
            z=df.iloc[:, :-1].values.tolist(),  # Exclude the last column for correlation if it exists
            colorscale=colorscale,
            x=df.iloc[:, :-1].columns,
            y=df.index,  # Use the index of df for y-axis
        )
    ]

    # Define the layout
    layout = go.Layout(
        width=width,
        height=height,
        title=title,
        xaxis=dict(title="Columns"),
        yaxis=dict(title="Index"),
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)
    
    return fig


def plot_sex_ratio(df, show: bool = False, save: bool=True, name: str=None):
    """
    Plot a histogram to show the ratio of allergics between males and females.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing data to be plotted.
    """
    # Replace numeric values in 'sex' column with string labels
    plot_df = df.copy()
    plot_df['sex_str'] = plot_df['sex'].replace({1: 'male', 2: 'female'})

    # Create the histogram plot
    fig = px.histogram(plot_df, x="W_str", color="sex_str", color_discrete_sequence=px.colors.qualitative.G10,
                       barnorm='percent', text_auto=".2f",
                       width=800, height=400)

    # Customize the layout of the plot
    fig.update_layout(
        title_text='KORA: Allergics ratio between males and females',  # title of plot
        xaxis_title_text='Allergic?',  # x-axis label
        yaxis_title_text='Percentage',  # y-axis label
        bargap=0.1,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
    )
    
    if show:
        fig.show()
    if save:
        fig.write_image("plots/png/{0}.png".format(name))
        fig.write_image("plots/svg/{0}.svg".format(name))
        fig.write_html("plots/html/{0}.html".format(name))
    
    return fig


def plot_age_ratio(df, show: bool = False, save: bool=True, name: str=None):
    """
    Plot a histogram to show the age ratio between (non)allergics.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing data to be plotted.
    """
    plot_df = df.copy()
    
    # Define age bins and labels
    bins = [30, 40, 50, 60, 70]
    labels = ['30s', '40s', '50s', '60s']

    # Create an 'age_cat' column based on age bins
    plot_df['age_cat'] = pd.cut(plot_df['age'], bins=bins, labels=labels, right=False)

    # Define a color map for age categories
    color_discrete_map = {'30s': '#3C15DB', '40s': '#17C428', '50s': '#F626E1', '60s': "#F6F626"}

    # Create the histogram plot
    fig = px.histogram(plot_df, x="W_str", color="age_cat", barnorm='percent', text_auto=".2f",
                       color_discrete_map=color_discrete_map, width=800, height=400)

    # Customize the layout of the plot
    fig.update_layout(
        title_text='KORA: Age ratio between (non)allergics',  # title of the plot
        xaxis_title_text='Allergic?',  # x-axis label
        yaxis_title_text='Percentage',  # y-axis label
        bargap=0.1,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
    )
    
    if show:
        fig.show()
    if save:
        fig.write_image("plots/png/{0}.png".format(name))
        fig.write_image("plots/svg/{0}.svg".format(name))
        fig.write_html("plots/html/{0}.html".format(name))
    
    return fig


def plot_activity_ratio(df, show: bool = False, save: bool=True, name: str=None):
    """
    Plot a histogram to show the ratio between active and inactive people among allergics.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing data to be plotted.
    """
    plot_df = df.copy()
    # Replace numeric values in 'phys activ' column with string labels
    plot_df['phys_act_str'] = plot_df['phys_activ'].replace({0: 'active', 1: 'inactive'})

    # Create the histogram plot
    fig = px.histogram(plot_df, x="W_str", color="phys_act_str", color_discrete_sequence=px.colors.qualitative.G10,
                       barnorm='percent', text_auto=".2f",
                       width=800, height=400)

    # Customize the layout of the plot
    fig.update_layout(
        title_text='KORA: Allergics ratio between active and inactive people',  # title of the plot
        xaxis_title_text='Allergic?',  # x-axis label
        yaxis_title_text='Percentage',  # y-axis label
        bargap=0.1,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
    )
    
    if show:
        fig.show()
    if save:
        fig.write_image("plots/png/{0}.png".format(name))
        fig.write_image("plots/svg/{0}.svg".format(name))
        fig.write_html("plots/html/{0}.html".format(name))
    
    return fig


def plot_smoking_ratio(df, show: bool = False, save: bool=True, name: str=None):
    """
    Plot a histogram to show the ratio between (non)smoking among allergics.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing data to be plotted.
    """
    plot_df = df.copy()
    # Replace numeric values in 'smoking behaviour' column with string labels
    smoking_labels = {1: "regular_smokers", 2: "irregular_smokers", 3: "ex_smoker", 4: "never_smoker"}
    plot_df['smoking_str'] = plot_df['smoking_behaviour'].replace(smoking_labels)

    # Create the histogram plot
    fig = px.histogram(plot_df, x="W_str", color="smoking_str", color_discrete_sequence=px.colors.qualitative.G10,
                       barnorm='percent', text_auto=".2f",
                       width=800, height=400)

    # Customize the layout of the plot
    fig.update_layout(
        title_text='KORA: Allergics ratio between (non)smoking',  # title of the plot
        xaxis_title_text='Allergic?',  # x-axis label
        yaxis_title_text='Percentage',  # y-axis label
        bargap=0.1,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
    )
    
    if show:
        fig.show()
    if save:
        fig.write_image("plots/png/{0}.png".format(name))
        fig.write_image("plots/svg/{0}.svg".format(name))
        fig.write_html("plots/html/{0}.html".format(name))
    
    return fig 


def plot_bmi_probability_density(allergic, non_allergic, show: bool = False, save: bool=True, name: str=None):
    """
    Plot the probability density of BMI for (non)allergics.

    Parameters:
    - allergic (pandas.DataFrame): DataFrame containing data for allergics.
    - non_allergic (pandas.DataFrame): DataFrame containing data for non-allergics.
    """
    # Define the data for the plot
    hist_data = [allergic['bmi'], non_allergic['bmi']]
    group_labels = ['Allergics', 'Non-allergics']
    colors = ['slategray', 'magenta']

    # Create the probability density plot
    fig = ff.create_distplot(hist_data, group_labels, bin_size=2, show_rug=False,
                             histnorm="probability density", colors=colors)

    # Customize the layout of the plot
    fig.update_layout(
        title_text='KORA: BMI probability density of (non)allergics',  # title of the plot
        xaxis_title_text="BMI (kg/m2)",  # x-axis label
        yaxis_title_text='Probability density',  # y-axis label
        bargap=0.1,  # gap between bars of adjacent location coordinates
        bargroupgap=0.01  # gap between bars of the same location coordinates
    )
    
    if show:
        fig.show()
    if save:
        fig.write_image("plots/png/{0}.png".format(name))
        fig.write_image("plots/svg/{0}.svg".format(name))
        fig.write_html("plots/html/{0}.html".format(name))

    return fig


def plot_waist_hip_ratio_probability_density(allergic, non_allergic, show: bool = False, save: bool=True, name: str=None):
    """
    Plot the probability density of waist-hip ratio for (non)allergics.

    Parameters:
    - allergic (pandas.DataFrame): DataFrame containing data for allergics.
    - non_allergic (pandas.DataFrame): DataFrame containing data for non-allergics.
    """
    # Define the data for the plot
    hist_data = [allergic['waist-hip-ratio'], non_allergic['waist-hip-ratio']]
    group_labels = ['Allergics', 'Non-allergics']
    colors = ['aqua', 'pink']

    # Create the probability density plot
    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.001, show_rug=False,
                             histnorm="probability density", colors=colors)

    # Customize the layout of the plot
    fig.update_layout(
        title_text='KORA: Waist-hip ratio probability density of (non)allergics',  # title of the plot
        xaxis_title_text="Waist-hip ratio",  # x-axis label
        yaxis_title_text='Probability density',  # y-axis label
        bargap=0.1,  # gap between bars of adjacent location coordinates
        bargroupgap=0.01  # gap between bars of the same location coordinates
    )
    
    if show:
        fig.show()
    if save:
        fig.write_image("plots/png/{0}.png".format(name))
        fig.write_image("plots/svg/{0}.svg".format(name))
        fig.write_html("plots/html/{0}.html".format(name))

    return fig


def create_graph(corr_matrix: pd.DataFrame, threshold: float) -> nx.Graph:
    """
    Create a graph from a correlation matrix based on a specified threshold.

    Parameters:
        corr_matrix (pd.DataFrame): The input correlation matrix as a DataFrame.
        threshold (float): The threshold value for edge inclusion in the graph.

    Returns:
        nx.Graph: A NetworkX graph representing correlations between elements in the matrix.
    """
    # Take the upper part only
    upper = np.triu(np.ones(corr_matrix.shape)).astype(bool)
    df = corr_matrix.where(upper)
    df = pd.DataFrame(corr_matrix.stack(), columns=['covariance']).reset_index()
    df.columns = ["source", "target", "covariance"]

    # Remove diagonal entries
    df = df[abs(df['covariance']) >= threshold]
    # Remove self-loops
    df = df[df['source'] != df['target']]
    # Remove zero entries
    df = df[df['covariance'] != 0]

    # Build graph
    G = nx.from_pandas_edgelist(df, edge_attr="covariance")

    return G


def create_network_visualization(G_adapt, height: int = 1500, width: int = 1800, show_labels: bool = False, 
                                 size_degree: bool = False, scale_edge: int = 2, scale_node: int = 1):
    """
    Creates a network visualization using Pyvis library and returns the Pyvis Network object.

    Parameters:
        G_adapt (networkx.Graph): The graph data loaded from NetworkX.
        height (int): The height of the network visualization in pixels. Default is 1500.
        width (int): The width of the network visualization in pixels. Default is 1800.
        show_labels (bool): Whether to show edge labels. Default is False.
        size_degree (bool): Whether to adjust node sizes based on degree. Default is False.
        scale_edge (int): Scaling factor for edge widths. Default is 2.
        scale_node (int): Scaling factor for node sizes when using size_degree. Default is 1.

    Returns:
        pyvis.network.Network: The Pyvis Network object representing the network visualization.
    """
    # Create a Pyvis network
    net = Network(height=f"{height}px", width=f"{width}px", directed=False, cdn_resources='in_line', notebook=True)

    # Load the graph data from NetworkX
    net.from_nx(G_adapt)

    # Disable physics simulation
    net.toggle_physics(False)

    if size_degree:
        # Calculate the degree of each node
        degrees = dict(G_adapt.degree())

        # Set node sizes based on degree
        for node in net.nodes:
            node_id = node['id']
            if node_id in degrees:
                node['size'] = degrees[node_id] * scale_node  # Adjust the scaling factor as needed

    # Set edge and node styles
    for edge in net.edges:
        edge['width'] = abs(edge['covariance'])
        edge['length'] = 2000

        if show_labels:
            edge['label'] = str(round(edge['covariance'], 2))
        if edge['covariance'] >= 0:
            edge['color'] = '#f1ac8b' #red
            edge['width'] = abs(edge['covariance']) * scale_edge
            if show_labels:
                edge['font'] = {'multi': 'true', 'size': 15, 'color': 'blue', 'face': 'arial', 'align': 'top'}
        else:
            edge['color'] = "#abe4ff" #blue
            edge['width'] = abs(edge['covariance']) * scale_edge
            if show_labels:
                edge['font'] = {'multi': 'true', 'size': 15, 'color': 'red', 'face': 'arial', 'align': 'top'}

    for node in net.nodes:
        if "ASV" in node['label'] or len(node['label']) == 1 or "g_" in node['label']:
            node['color'] = '#610053'
        else:
            node['color'] = '#FA4665'

        node['font'] = {'multi': 'true', 'size': 20, 'color': 'black', 'face': 'arial', 'align': 'left'}

    return net





