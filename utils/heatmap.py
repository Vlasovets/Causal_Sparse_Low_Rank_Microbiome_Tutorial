import numpy as np
import pandas as pd
import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns
import scipy.spatial as scs
import networkx as nx
import bisect

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly import subplots

from sklearn.impute import SimpleImputer


from math import pi
from itertools import chain
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, HoverTool, LabelSet, PointDrawTool
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import RdBu, Blues8
from bokeh.models import Tabs, ColorBar, LinearColorMapper
from bokeh.layouts import row
from bokeh.io import export_svgs, export_png




# pylint: disable=assignment-from-no-return, no-self-use
def Clustergram(
    data,
    generate_curves_dict=False,
    return_computed_traces=False,
    computed_traces=None,
    row_labels=None,
    row_colors=None,
    row_colors_label=None,
    column_labels=None,
    column_colors=None,
    column_colors_label=None,
    hidden_labels=None,
    standardize="none",
    cluster="all",
    row_dist="euclidean",
    col_dist="euclidean",
    dist_fun=scs.distance.pdist,
    link_fun=None,
    link_method=None,
    color_threshold=None,
    optimal_leaf_order=False,
    color_map=None,
    color_list=None,
    display_range=3,
    center_values=True,
    log_transform=False,
    display_ratio=0.2,
    imputer_parameters=None,
    row_group_marker=None,  # group number, annotation, color
    col_group_marker=None,  # same as above
    tick_font=None,
    annotation_font=None,
    line_width=0.5,
    paper_bg_color="rgba(0,0,0,0)",
    plot_bg_color="rgba(0,0,0,0)",
    height=500,
    width=500,
):
    """Return a Dash Bio Clustergram object.
Keyword arguments:
- data (2D array-like; required): Matrix or table of observations (dropping
    columns of non-numeric dtype).
- generate_curves_dict (bool; default False): Whether or not to return a
    dictionary containing information about the cluster number
    associated with each curve number in the graph. (May be useful
    for capturing the cluster number that is clicked.)
- return_computed_traces (bool; default False): Whether or not to return
    the precomputed dendrogram traces. (May be useful if one wishes
    to add, e.g., group markers to the figure without recalculating
    the clustering in the entire figure.)
- computed_traces (dict; optional): The dendrogram traces from another
   (precomputed) Clustergram component.
- row_labels (list; optional): List of row category labels
   (observation labels).
- row_colors (list; optional): List of row colors
   (observation colors).
- row_colors_label (string; optional): String which describes the annotation
    label for row_colors.
- column_labels (list; optional): List of column category labels
   (observation labels).
- column_colors (list; optional): List of column colors
   (observation colors).
- column_colors_label (string; optional): String which describes the annotation
    label for column_colors.
- hidden_labels (list; optional): List containing strings 'row' and/or 'col'
    if row and/or column labels should be hidden on the final plot.
- standardize (string; default 'none'): The dimension for standardizing
    values, so that the mean is 0 and the standard deviation is 1,
    along the specified dimension: 'row', 'column', or 'none'.
- cluster (string; default 'all'): The dimension along which the data will
    be clustered: 'row', 'column', or 'all'; 'all' means data will be
    clustered along columns, then clustered along rows of
    column-clustered data.
- row_dist (string; default 'euclidean'): Distance metric for rows.
    Passed as argument `metric` to the function specified in `dist_fun`
    when called for clustering along rows.
- col_dist (string; default 'euclidean'): Distance metric for columns.
    Passed as argument `metric` to the function specified in `dist_fun`
    when called for clustering along columns.
- dist_fun (function; default scipy.spatial.distance.pdist): Function
    to compute the pairwise distance from the observations (see docs for
    scipy.spatial.distance.pdist).
- link_fun (function; default scipy.cluster.hierarchy.linkage): Function to
    compute the linkage matrix from the pairwise distances (see docs for
    scipy.cluster.hierarchy.linkage).
- link_method (string; default 'complete'): The linkage algorithm to use
    if link_fun not set. For method 'single', an optimized algorithm based
    on minimum spanning, for methods 'complete', 'average', 'weighted' and
    'ward', an algorithm called nearest-neighbors chain is implemented
    (see docs for scipy.cluster.hierarchy.linkage).
- color_threshold (dict; default {'row': 0, 'col': 0}): Maximum
    linkage value for which unique colors are assigned to clusters;
    'row' for rows, and 'col' for columns.
- optimal_leaf_order (bool; default False): Whether to enable (True) or
    disable (False) the option to determine leaf order that maximizes
    similarity between neighboring leaves.
- color_map (list; default [[0.0, 'rgb(255,0,0)'], [0.5,
    'rgb(0,0,0)'], [1.0, 'rgb(0,255,0)']]): Colorscale for the heatmap.
    Top-level elements contain two elements, the first of which refers to
    the percentile rank, and the second to the applied color. For instance,
    [[0.0, 'white'], [0.5, 'gray'], [1.0, 'black']] means that cells in the
    49th percentile would be white; cells at the 50th or higher percentiles,
    excluding the 100th percentile, would be gray; and the cell(s) at the
    100th percentile would be black.
- color_list (dict; optional): The list of colors to use for different
   clusters in the dendrogram that have a root under the threshold for
   each dimension. If there are fewer colors than there are clusters
   along a specific dimension, the colors of the clusters will cycle
   through the colors specified in the list. The keys are: 'row' (for
   row clusters), 'col' (for column clusters), and 'bg' (for all
   traces above the clustering threshold for both row and column).
- display_range (double; default 3.0): In the heatmap, standardized
    values from the dataset that are below the negative of this value
    will be colored with one shade, and the values that are above this
    value will be colored with another.
- center_values (bool; default True): Whether or not to center the
    values of the heatmap about zero.
- log_transform (bool; default False): Whether or not to transform
    the data by taking the base-two logarithm of all values in the
    dataset.
- display_ratio (list | number; default 0.2): The dendrograms' heights with
    respect to the size of the heatmap; with one element, both the row
    and column dendrograms have the same ratio; with two, the row
    dendrogram ratio corresponds to the first element of the list and
    the column dendrogram ratio corresponds to the second element of
    the list.
- imputer_parameters (dict; optional): Specifies the parameters
    'missing_values' and 'strategy' of the SimpleImputer class from
    scikit-learn 0.20.1 (both of these parameters must be keys in the
    dictionary). An additional parameter, 'axis', is used to specify
    the direction along which to impute (a parameter of Imputer, which
    was deprecated in scikit-learn 0.20.0): 'axis=0' indicates that
    imputing should happen along columns, while 'axis=1' indicates
    that it should happen along rows (see: https://scikit
    -learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html).
- row_group_marker (list; optional): A list containing the annotations
    for row clusters in the dendrogram. Each annotation is a
    dictionary with the keys 'group_number' (the cluster number to
    highlight), 'annotation' (a string containing the text of the
    annotation), and 'color' (a string representation of the color of
    the annotation).
- col_group_marker (list; optional): A list containing the annotations for
    column clusters in the dendrogram. Each annotation is a dictionary
    with the keys 'group_number' (the cluster number to highlight),
    'annotation' (a string containing the text of the annotation), and
    'color' (a string representation of the color of the
    annotation).
- tick_font (dict; optional): The font options for ticks, as specified
    in the Plotly graph_objects documentation (see:
    https://plotly.com/python/reference/#bar-marker-colorbar-tickfont).
- annotation_font (dict; optional): The font options for annotations,
    as specified in the Plotly graph_objects documentation (see:
    https://plotly.cp,/python/reference/#layout-scene-annotations-items-annotation-font).
- line_width (list | number; default 0.5): The line width for the
    dendrograms. If in list format, the first element corresponds to
    the width of the row dendrogram traces, and the second corresponds
    to the width of the column dendrogram traces.
- paper_bg_color (string; default 'rgba(0,0,0,0)'): The background
    color of the paper on the graph.
- plot_bg_color (string; default 'rgba(0,0,0,0)'): The background
    color of the subplots on the graph.
- height (number; default 500): The height of the graph, in px.
- width (number; default 500): The width of the graph, in px.
    """

    if color_threshold is None:
        color_threshold = dict(row=0, col=0)

    # get rid of arguments that are not used by _Clustergram
    kwargs = locals()
    kwargs.pop("return_computed_traces")
    kwargs.pop("computed_traces")
    kwargs.pop("generate_curves_dict")

    (fig, ct, curves_dict) = _Clustergram(**kwargs).figure(
        computed_traces=computed_traces
    )

    return_values = [go.Figure(fig)]

    if generate_curves_dict:
        return_values.append(curves_dict)
    if return_computed_traces:
        return_values.append(ct)

    # return only the figure by default
    if len(return_values) == 1:
        return return_values[0]

    # otherwise, return all requested values
    return tuple(return_values)


class _Clustergram:
    """A Dash Bio Clustergram class.
Methods:
- figure(computed_traces=None): Return a figure object compatible with plotly.graph_objects.
    """

    def __init__(
        self,
        data,
        row_labels=None,
        row_colors=None,
        row_colors_label=None,
        column_labels=None,
        column_colors=None,
        column_colors_label=None,
        hidden_labels=None,
        standardize="none",
        cluster="all",
        row_dist="euclidean",
        col_dist="euclidean",
        dist_fun=scs.distance.pdist,
        link_fun=None,
        link_method=None,
        color_threshold=None,
        optimal_leaf_order=False,
        color_map=None,
        color_list=None,
        display_range=3,
        center_values=True,
        log_transform=False,
        display_ratio=0.2,
        imputer_parameters=None,
        row_group_marker=None,  # group number, annotation, color
        col_group_marker=None,  # same as above
        tick_font=None,
        annotation_font=None,
        line_width=0.5,
        paper_bg_color="rgba(0,0,0,0)",
        plot_bg_color="rgba(0,0,0,0)",
        height=500,
        width=500,
    ):
        """Construct a Dash Bio Clustergram object.
    See docstring of the `Clustergram` function, where the same keyword arguments (and a couple
    of other ones) are documented.
        """
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes("number")
            data = data.values
        if hidden_labels is None:
            hidden_labels = []
        if color_threshold is None:
            color_threshold = dict(row=0, col=0)
        # Always keep unique identifiers for rows
        row_ids = list(range(data.shape[0]))
        # Always keep unique identifiers for columns
        column_ids = list(range(data.shape[1]))

        if link_method is None:
            link_method = "complete"

        if link_fun is None:
            def linkage(x, **kwargs):
                return sch.linkage(x, link_method, **kwargs)

            link_fun = linkage

        self._data = data
        self._row_labels = row_labels
        self._row_colors = row_colors
        self._row_colors_label = row_colors_label
        self._row_ids = row_ids
        self._column_labels = column_labels
        self._column_colors = column_colors
        self._column_colors_label = column_colors_label
        self._column_ids = column_ids
        self._cluster = cluster
        self._row_dist = row_dist
        self._col_dist = col_dist
        self._dist_fun = dist_fun
        self._link_fun = link_fun
        self._color_threshold = color_threshold
        self._optimal_leaf_order = optimal_leaf_order
        if color_map is None:
            self._color_map = [
                [0.0, "rgb(84,48,5)"],
                [0.5, "rgb(246,232,195)"],
                [1.0, "rgb(0,60,48)"],
            ]
        else:
            self._color_map = color_map
        self._color_list = color_list
        self._display_range = display_range
        self._center_values = center_values
        self._display_ratio = display_ratio
        self._imputer_parameters = imputer_parameters
        if row_group_marker is None:
            self._row_group_marker = []
        else:
            self._row_group_marker = row_group_marker
        if col_group_marker is None:
            self._col_group_marker = []
        else:
            self._col_group_marker = col_group_marker
        if tick_font is None:
            self._tick_font = {}
        else:
            self._tick_font = tick_font
        if annotation_font is None:
            self._annotation_font = {}
        else:
            self._annotation_font = annotation_font
        self._paper_bg_color = paper_bg_color
        self._plot_bg_color = plot_bg_color
        self._height = height
        self._width = width

        # convert line width to list if necessary
        if isinstance(line_width, list):
            if len(line_width) == 2:
                self._line_width = line_width
            elif len(line_width) == 1:
                self._line_width = [line_width[0], line_width[0]]
            else:
                raise ValueError("line_width cannot have more than 2 elements")
        else:
            self._line_width = [line_width, line_width]

        # convert display ratio to list if necessary
        if not isinstance(display_ratio, list):
            self._display_ratio = [display_ratio, display_ratio]
        if self._cluster == "row":
            self._display_ratio = [self._display_ratio[0], 0]
        elif self._cluster == "col":
            self._display_ratio = [0, self._display_ratio[1]]
        elif self._cluster is None:
            self._display_ratio = [0, 0]

        self._hidden_labels = []

        if "row" in hidden_labels:
            self._hidden_labels.append("yaxis11")
        if "col" in hidden_labels:
            self._hidden_labels.append("xaxis11")

        # preprocessing data
        if self._imputer_parameters is not None:

            # numpy NaN values are not serializable and turn into
            # 'None' by the time they get here; passing a string
            # means that it can be converted in the clustergram
            # component itself
            if self._imputer_parameters["missing_values"].lower() == "nan":
                self._imputer_parameters.update(missing_values=np.nan)

            imp = SimpleImputer(
                missing_values=self._imputer_parameters["missing_values"],
                strategy=self._imputer_parameters["strategy"],
            )

            if self._imputer_parameters["axis"] == 0:
                self._data = imp.fit_transform(self._data.T).T
            else:
                self._data = imp.fit_transform(self._data)

        if log_transform:
            self._data = np.log2(self._data)
        if standardize in ["row", "column"]:
            self._data = self._scale(standardize)

    def figure(self, computed_traces=None):
        """Return a figure object compatible with plotly.graph_objects.
    Parameters:
    - computed_traces (dict; optional): The dendrogram traces from another
        (precomputed) Clustergram component.
        """
        dt, heatmap = None, None

        if computed_traces is None:
            (
                dt,
                self._data,
                self._row_ids,
                self._column_ids,
            ) = self._compute_clustered_data()
        else:
            # use, if available, the precomputed dendrogram and heatmap
            # traces (as well as the row and column labels)
            dt = computed_traces["dendro_traces"]
            heatmap = computed_traces["heatmap"]
            self._row_ids = computed_traces["row_ids"]
            self._column_ids = computed_traces["column_ids"]

        # Match reordered rows and columns with their respective labels
        if self._row_labels:
            self._row_labels = [self._row_labels[r] for r in self._row_ids]
        if self._row_colors:
            self._row_colors = [self._row_colors[r] if r < len(self._row_colors) else 'gray' for r
                                in self._row_ids]
        if self._column_labels:
            self._column_labels = [self._column_labels[r] for r in self._column_ids]
        if self._column_colors:
            self._column_colors = [
                self._column_colors[r] if r < len(self._column_colors) else 'gray' for r in
                self._column_ids]

        # this dictionary relates curve numbers (accessible from the
        # hoverData/clickData props) to cluster numbers
        cluster_curve_numbers = {}

        # initialize plot; GM is for group markers
        # [empty]      [col. dendro] [col. dendro] [empty]
        # [row dendro] [heatmap]     [heatmap]     [row GM]
        # [row dendro] [heatmap]     [heatmap]     [row GM]
        # [empty]      [col. GM]     [col. GM]     [empty]
        fig = subplots.make_subplots(
            rows=5,
            cols=5,
            specs=[
                [{}, {}, {"colspan": 2}, None, {}],
                [{}, {}, {"colspan": 2}, None, {}],
                [{"rowspan": 2}, {"rowspan": 2}, {"colspan": 2, "rowspan": 2}, None,
                 {"rowspan": 2}],
                [None, None, None, None, None],
                [{}, {}, {"colspan": 2}, None, {}],
            ],
            vertical_spacing=0,
            horizontal_spacing=0,
            print_grid=False,
        )

        fig["layout"].update(hovermode="closest")

        # get the tick values; these will be at the leaves of the
        # dendrogram

        tickvals_col = []
        tickvals_row = []

        # for column dendrogram, leaves are at bottom (y=0)
        for i in range(len(dt["col"])):
            xs = dt["col"][i]["x"]
            ys = dt["col"][i]["y"]

            # during serialization (e.g., in a dcc.Store, the NaN
            # values become None and the arrays get turned into lists;
            # they must be converted back
            if isinstance(xs, list):
                xs = np.array(xs, dtype=np.float)
                dt["col"][i].update(x=xs)
            if isinstance(ys, list):
                ys = np.array(ys, dtype=np.float)
                dt["col"][i].update(y=ys)
            tickvals_col += [
                xs.flatten()[j]
                for j in range(len(xs.flatten()))
                if ys.flatten()[j] == 0.0 and xs.flatten()[j] % 10 == 5
            ]
        tickvals_col = list(set(tickvals_col))

        # for row dendrogram, leaves are at right(x=0, since we
        # horizontally flipped it)
        for i in range(len(dt["row"])):
            xs = dt["row"][i]["x"]
            ys = dt["row"][i]["y"]

            if isinstance(xs, list):
                xs = np.array(xs, dtype=np.float)
                dt["row"][i].update(x=xs)
            if isinstance(ys, list):
                ys = np.array(ys, dtype=np.float)
                dt["row"][i].update(y=ys)

            tickvals_row += [
                ys.flatten()[j]
                for j in range(len(ys.flatten()))
                if xs.flatten()[j] == 0.0 and ys.flatten()[j] % 10 == 5
            ]

        tickvals_row = list(set(tickvals_row))

        # sort so they are in the right order (lowest to highest)
        tickvals_col.sort()
        tickvals_row.sort()

        # update axis settings for dendrograms and heatmap
        axes = [
            "xaxis1",
            "xaxis3",
            "xaxis5",
            "xaxis6",
            "xaxis7",
            "xaxis9",
            "xaxis10",
            "xaxis11",
            "yaxis1",
            "yaxis3",
            "yaxis5",
            "yaxis6",
            "yaxis7",
            "yaxis9",
            "yaxis10",
            "yaxis11",
        ]

        for a in axes:
            fig["layout"][a].update(
                type="linear",
                showline=False,
                showgrid=False,
                zeroline=False,
                mirror=False,
                fixedrange=False,
                showticklabels=False,
            )

        (row_dendro_traces, col_dendro_traces) = self._sort_traces(dt["row"], dt["col"])

        for i in range(len(col_dendro_traces)):
            cdt = col_dendro_traces[i]
            cdt["name"] = "Col Cluster %d" % i
            cdt["line"] = dict(width=self._line_width[1])
            cdt["hoverinfo"] = "y+name"
            cluster_curve_numbers[len(fig.data)] = ["col", i]
            fig.append_trace(cdt, 1, 3)

        # row dendrogram (displays on left side)
        for i in range(len(row_dendro_traces)):
            rdt = row_dendro_traces[i]
            rdt["name"] = "Row Cluster %d" % i
            rdt["line"] = dict(width=self._line_width[0])
            rdt["hoverinfo"] = "x+name"
            cluster_curve_numbers[len(fig.data)] = ["row", i]
            fig.append_trace(rdt, 3, 1)

        col_dendro_traces_y = [r["y"] for r in col_dendro_traces]
        # arbitrary extrema if col_dendro_traces_y is empty
        col_dendro_traces_min_y = 0
        col_dendro_traces_max_y = 1
        if len(col_dendro_traces_y):
            col_dendro_traces_min_y = np.concatenate(col_dendro_traces_y).min()
            col_dendro_traces_max_y = np.concatenate(col_dendro_traces_y).max()

        # ensure that everything is aligned properly
        # with the heatmap
        yaxis9 = fig["layout"]["yaxis9"]  # pylint: disable=invalid-sequence-index
        yaxis9.update(scaleanchor="y11")
        xaxis3 = fig["layout"]["xaxis3"]  # pylint: disable=invalid-sequence-index
        xaxis3.update(scaleanchor="x11")

        if len(tickvals_col) == 0:
            tickvals_col = [10 * i + 5 for i in range(len(self._column_ids))]

        # add in all of the labels
        fig["layout"]["xaxis11"].update(  # pylint: disable=invalid-sequence-index
            tickmode="array",
            tickvals=tickvals_col,
            ticktext=self._column_labels,
            tickfont=self._tick_font,
            showticklabels=True,
            side="bottom",
            showline=False,
            range=[min(tickvals_col) - 5, max(tickvals_col) + 5]
            # workaround for autoscale issues above; otherwise
            # the graph cuts off and must be scaled manually
        )

        if len(tickvals_row) == 0:
            tickvals_row = [10 * i + 5 for i in range(len(self._row_ids))]

        fig["layout"]["yaxis11"].update(  # pylint: disable=invalid-sequence-index
            tickmode="array",
            tickvals=tickvals_row,
            ticktext=self._row_labels,
            tickfont=self._tick_font,
            showticklabels=True,
            side="right",
            showline=False,
        )

        # hide labels, if necessary
        for label in self._hidden_labels:
            fig["layout"][label].update(ticks="", showticklabels=False)

        row_colors_heatmap = self._get_row_colors_heatmap()
        if row_colors_heatmap is not None:
            fig.append_trace(self._get_row_colors_heatmap(), 3, 2)

        col_colors_heatmap = self._get_column_colors_heatmap()
        if col_colors_heatmap is not None:
            fig.append_trace(col_colors_heatmap, 2, 3)

        # recalculate the heatmap, if necessary
        if heatmap is None:

            # heatmap
            heat_data = self._data

            # symmetrize the heatmap about zero, if necessary
            if self._center_values:
                heat_data = np.subtract(heat_data, np.mean(heat_data))

            heatmap = go.Heatmap(
                x=tickvals_col,
                y=tickvals_row,
                z=heat_data,
                colorscale=self._color_map,
                # TODO: This should be based on the text width of the labels, or
                # at least passable by the user, so they can adjust it
                colorbar={"xpad": 300},
            )

        fig.append_trace(heatmap, 3, 3)

        # it seems the range must be set after heatmap is appended to the
        # traces, otherwise the range gets overwritten
        fig["layout"]["yaxis9"].update(  # pylint: disable=invalid-sequence-index
            range=[min(tickvals_row), max(tickvals_row)],
        )

        # hide all legends
        fig["layout"].update(showlegend=False,)

        # apply the display ratio
        row_ratio = 0
        col_ratio = 0

        # the argument can be either in list form or float form
        # first is ratio for row; second is ratio for column
        if self._display_ratio[0] != 0:
            row_ratio = 0 if len(row_dendro_traces) == 0 else 0.95 / float(
                1 + int(1 / self._display_ratio[0]))
        if self._display_ratio[1] != 0:
            col_ratio = 0 if len(col_dendro_traces) == 0 else 0.95 / float(
                1 + int(1 / self._display_ratio[1]))

        # the row/column labels take up 0.05 of the graph, and the rest
        # is taken up by the heatmap and dendrogram for each dimension

        # row: dendrogram, heatmap, row labels (left-to-right)
        # column: dendrogram, column labels, heatmap (top-to-bottom)

        row_colors_ratio = 0.02 if row_colors_heatmap is not None else 0
        col_colors_ratio = 0.02 if col_colors_heatmap is not None else 0

        # width adjustment for row dendrogram
        fig["layout"]["xaxis1"].update(  # pylint: disable=invalid-sequence-index
            domain=[0, 0.95]
        )
        fig["layout"]["xaxis3"].update(  # pylint: disable=invalid-sequence-index
            domain=[row_ratio + row_colors_ratio, 0.95], anchor="y9"
        )
        fig["layout"]["xaxis5"].update(  # pylint: disable=invalid-sequence-index
            domain=[0, 0.95]
        )
        fig["layout"]["xaxis7"].update(  # pylint: disable=invalid-sequence-index
            domain=[row_ratio + row_colors_ratio, 0.95], anchor="y9"
        )
        fig["layout"]["xaxis9"].update(  # pylint: disable=invalid-sequence-index
            domain=[0, row_ratio]
        )
        fig["layout"]["xaxis10"].update(  # pylint: disable=invalid-sequence-index
            domain=[row_ratio, row_ratio + row_colors_ratio]
        )
        fig["layout"]["xaxis11"].update(  # pylint: disable=invalid-sequence-index
            domain=[row_ratio + row_colors_ratio, 0.95]
        )

        # height adjustment for column dendrogram
        fig["layout"]["yaxis1"].update(  # pylint: disable=invalid-sequence-index
            domain=[1 - col_ratio, 1]
        )
        fig["layout"]["yaxis3"].update(  # pylint: disable=invalid-sequence-index
            domain=[1 - col_ratio, 1],
            range=[col_dendro_traces_min_y, col_dendro_traces_max_y],
        )
        fig["layout"]["yaxis5"].update(  # pylint: disable=invalid-sequence-index
            domain=[1 - col_ratio - col_colors_ratio, 1 - col_ratio]
        )

        fig["layout"]["yaxis6"].update(  # pylint: disable=invalid-sequence-index
            domain=[1 - col_ratio - col_colors_ratio, 1 - col_ratio]
        )
        fig["layout"]["yaxis7"].update(  # pylint: disable=invalid-sequence-index
            domain=[1 - col_ratio - col_colors_ratio, 1 - col_ratio]
        )

        fig["layout"]["yaxis9"].update(  # pylint: disable=invalid-sequence-index
            domain=[0, 1 - col_ratio - col_colors_ratio]
        )

        fig["layout"]["yaxis10"].update(  # pylint: disable=invalid-sequence-index
            domain=[0, 1 - col_ratio - col_colors_ratio]
        )
        fig["layout"]["yaxis11"].update(  # pylint: disable=invalid-sequence-index
            domain=[0, 1 - col_ratio - col_colors_ratio]
        )

        fig["layout"][
            "legend"
        ] = dict(  # pylint: disable=unsupported-assignment-operation
            x=0.7, y=0.7
        )

        # annotations
        color_labels = self._get_color_labels()

        # axis settings for subplots that will display group labels
        axes = ["xaxis12", "yaxis12", "xaxis15", "yaxis15"]

        for a in axes:
            fig["layout"][a].update(
                type="linear",
                showline=False,
                showgrid=False,
                zeroline=False,
                mirror=False,
                fixedrange=False,
                showticklabels=False,
            )

        # group labels for row dendrogram
        fig["layout"]["yaxis12"].update(  # pylint: disable=invalid-sequence-index
            domain=[0, 0.95 - col_ratio], scaleanchor="y11", scaleratio=1
        )
        if len(tickvals_row) > 0:
            fig["layout"]["yaxis12"].update(  # pylint: disable=invalid-sequence-index
                range=[min(tickvals_row), max(tickvals_row)]
            )
        # padding between group label line and dendrogram
        fig["layout"]["xaxis12"].update(  # pylint: disable=invalid-sequence-index
            domain=[0.95, 1], range=[-5, 1]
        )

        # group labels for column dendrogram
        fig["layout"]["xaxis15"].update(  # pylint: disable=invalid-sequence-index
            domain=[row_ratio, 0.95], scaleanchor="x11", scaleratio=1
        )
        if len(tickvals_col) > 0:
            fig["layout"]["xaxis15"].update(  # pylint: disable=invalid-sequence-index
                range=[min(tickvals_col), max(tickvals_col)]
            )
        fig["layout"]["yaxis15"].update(  # pylint: disable=invalid-sequence-index
            domain=[0.95 - col_ratio, 1 - col_ratio], range=[-0.5, 0.5]
        )

        # get group label annotations and label traces
        (
            row_group_labels,
            col_group_labels,
            row_annotations,
            col_annotations,
        ) = self._group_label_traces(row_dendro_traces, col_dendro_traces)
        # add annotations to graph
        fig["layout"].update(annotations=row_annotations + col_annotations + color_labels)
        # add label traces to graph
        for rgl in row_group_labels:
            fig.append_trace(rgl, 3, 5)
        for cgl in col_group_labels:
            fig.append_trace(cgl, 5, 3)

        # set background colors
        fig["layout"].update(
            paper_bgcolor=self._paper_bg_color, plot_bgcolor=self._plot_bg_color
        )

        # finally add height and width
        fig["layout"].update(height=self._height, width=self._width)

        computed_traces = {
            "dendro_traces": dt,
            "heatmap": heatmap,
            "row_ids": self._row_ids,
            "column_ids": self._column_ids,
        }

        return (fig, computed_traces, cluster_curve_numbers)

    def _scale(self, dim):
        """Return standardized data based on user parameters.
        Parameters:
        - dim (string): The dimension, row or column, to standardize across.
        Returns:
        - ndarray: An array containing the standardized data.
        """

        std = np.zeros(self._data.shape)

        if dim == "row":
            std = scipy.stats.zscore(self._data, axis=1)
        elif dim == "column":
            std = scipy.stats.zscore(self._data, axis=0)

        return std

    def _get_clusters(self):
        """Cluster the data according to the specified dimensions.
        Returns:
        - tuple: The linkage matrices for the columns and/or rows.
        """

        Zcol = None
        Zrow = None

        # cluster along columns
        if self._cluster in ["col", "all"]:
            tmp = np.transpose(self._data)
            dcol = self._dist_fun(tmp, metric=self._col_dist)
            Zcol = self._link_fun(dcol, optimal_ordering=self._optimal_leaf_order)
        # cluster along rows only if 'all' is selected
        if self._cluster in ["row", "all"]:
            drow = self._dist_fun(self._data, metric=self._row_dist)
            Zrow = self._link_fun(drow, optimal_ordering=self._optimal_leaf_order)

        return (Zcol, Zrow)

    def _get_row_colors_heatmap(self):
        colors = self._row_colors

        if colors is None:
            return None

        colorscale = []

        i = 0

        step = round(1 / len(colors), 10)

        for color in colors:
            colorscale.append([i, color])
            i = round(i + step, 10)
            colorscale.append([i, color])

        colorscale[-1][0] = 1

        z = [[i] for i in range(len(colors))]

        return go.Heatmap(
            z=z,
            colorscale=colorscale,
            colorbar={"xpad": 300},
            showscale=False
        )

    def _get_column_colors_heatmap(self):
        colors = self._column_colors

        if colors is None:
            return None

        colorscale = []

        i = 0

        step = round(1 / len(colors), 10)

        for color in colors:
            colorscale.append([i, color])
            i = round(i + step, 10)
            colorscale.append([i, color])

        colorscale[-1][0] = 1

        z = [[i * 5 for i in range(len(colors))]]

        return go.Heatmap(
            z=z,
            colorscale=colorscale,
            colorbar={"xpad": 300},
            showscale=False
        )
    

    def _compute_clustered_data(self):
        """Get the traces that need to be plotted for the row and column
        dendrograms, and update the ordering of the 2D data array,
        row labels, and column labels to match the reordered
        dendrogram leaves.
        Returns:
        - dict: A dictionary containing entries for the row and column
        dendrogram traces.
        - ndarray: The original 2D data array that has been reordered to
        match the ordering of the row and column dendrogram leaves.
        - list: A list of the row labels that have been reordered to match
        the ordering of the row dendrogram leaves.
        - list: a list of the column labels that have been reordered to match
        the ordering of the column dendrogram leaves.
        """

        # initialize return dict
        trace_list = {"col": [], "row": []}

        clustered_column_ids = self._column_ids
        clustered_row_ids = self._row_ids

        # cluster the data and calculate dendrogram

        # allow referring to protected member
        # pylint: disable=W0212
        # columns
        if self._cluster in ["col", "all"]:
            cols_dendro = ff._dendrogram._Dendrogram(
                np.transpose(self._data),
                orientation="bottom",
                labels=self._column_ids,
                # TODO: How does colormap work?
                # colorscale=self._color_map["cols"],
                distfun=lambda X: self._dist_fun(X, metric=self._col_dist),
                linkagefun=lambda d: self._link_fun(
                    d, optimal_ordering=self._optimal_leaf_order
                ),
                color_threshold=self._color_threshold["col"],
            )
            clustered_column_ids = cols_dendro.labels
            trace_list["col"] = cols_dendro.data

        # rows
        if self._cluster in ["row", "all"]:
            rows_dendro = ff._dendrogram._Dendrogram(
                self._data,
                orientation="right",
                labels=self._row_ids,
                # TODO: How does colormap work?
                # colorscale=self._color_map,
                distfun=lambda X: self._dist_fun(X, metric=self._row_dist),
                linkagefun=lambda d: self._link_fun(
                    d, optimal_ordering=self._optimal_leaf_order
                ),
                color_threshold=self._color_threshold["row"],
            )
            clustered_row_ids = rows_dendro.labels
            trace_list["row"] = rows_dendro.data
        # pylint: enable=W0212

        # now, we need to rearrange the data array to fit the labels

        # first get reordered indices
        rl_indices = [self._row_ids.index(r) for r in clustered_row_ids]
        cl_indices = [self._column_ids.index(c) for c in clustered_column_ids]

        # modify the data here; first shuffle rows,
        # then transpose and shuffle columns,
        # then transpose again
        clustered_data = self._data[rl_indices].T[cl_indices].T

        return trace_list, clustered_data, clustered_row_ids, clustered_column_ids

    def _sort_traces(self, rdt, cdt):
        """Sort row dendrogram clusters and column dendrogram clusters
        so that the background trace (above threshold) is trace 0
        and all other traces are ordered top-to-bottom (row dendrogram)
        or left-to-right (column dendrogram).
        Parameters:
        - rdt (list[dict]): The row dendrogram cluster traces.
        - cdt (list[dict]): The column dendrogram cluster traces.
        Returns:
        - tuple: The sorted row dendrogram clusters and column
            dendrogram clusters.
        """

        tmp_rdt = []
        tmp_cdt = []

        if len(rdt) > 0:
            # first, find background trace: (max 'x')
            rdt.sort(key=lambda t: -1 * max(list(t["x"])))
            tmp_rdt.append(rdt[0])
            # then, sort top-to-bottom
            r = rdt[1:]
            r.sort(key=lambda t: -1 * min(list(t["y"])))
            tmp_rdt += r
        if len(cdt) > 0:
            # background trace has max 'y'
            cdt.sort(key=lambda t: -1 * max(list(t["y"])))
            tmp_cdt.append(cdt[0])
            # sort left to right
            c = cdt[1:]
            c.sort(key=lambda t: min(list(t["x"])))
            tmp_cdt += c

        return (tmp_rdt, tmp_cdt)

    def _get_color_labels(self):
        """Return annotations positioned on the figure to describe the
        features represented by the row and/or column colors.
        Parameters:
        - row_colors_label (string; optional): String which describes the annotation
            label for row_colors.
        - column_colors_label (string; optional): String which describes the annotation
            label for column_colors.
        Returns:
        - list: A list of dicts describing the row and column color labels.
        """
        labels = []

        if self._row_colors_label is not None:
            row_label = {
                "x": "1",
                "y": "0.85",
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "top",
                "text": self._row_colors_label,
                "font": self._annotation_font,
                "showarrow": False
            }

            labels.append(row_label)

        if self._column_colors_label is not None:
            column_label = {
                "x": "0.1",
                "y": "0",
                "xref": "paper",
                "yref": "paper",
                "xanchor": "right",
                "yanchor": "bottom",
                "text": self._column_colors_label,
                "font": self._annotation_font,
                "showarrow": False
            }

            labels.append(column_label)

        return labels

    def _group_label_traces(self, row_clusters, col_clusters):
        """Calculate the traces and annotations that correspond to group
        labels.
        Parameters:
        - row_clusters (list[dict]): List of all row traces (each
            trace corresponds to a cluster)
        - col_clusters (list[dict]): List of all column traces (each
            trace corresponds to a cluster)
        Returns:
        - tuple: The row label traces, column label traces, row group
            annotations, and column group annotations.
        """

        row_group_labels = []
        col_group_labels = []

        row_annotations = []
        col_annotations = []

        for rgm in self._row_group_marker:
            if len(row_clusters) == 0:
                break
            if rgm["group"] >= len(row_clusters):
                continue
            # get upper and lower bounds of group
            ymin = min(row_clusters[rgm["group"]]["y"])
            ymax = max(row_clusters[rgm["group"]]["y"])
            trace = go.Scatter(
                x=[0, 0],
                y=[ymin, ymax],
                mode="lines",
                line=dict(width=6, color=rgm["color"]),
                marker=dict(size=0),
                hoverinfo="none",
            )
            row_group_labels.append(trace)
            row_annotations.append(
                dict(
                    x=0.5,
                    y=1 / 2 * (ymin + ymax),
                    xref="x12",
                    yref="y12",
                    text=rgm["annotation"],
                    font=self._annotation_font,
                    showarrow=False,
                    xanchor="left",
                )
            )

        for cgm in self._col_group_marker:
            if len(col_clusters) == 0:
                break
            if cgm["group"] >= len(col_clusters):
                continue
            # get leftmost and rightmost bounds of group
            xmin = min(col_clusters[cgm["group"]]["x"])
            xmax = max(col_clusters[cgm["group"]]["x"])
            trace = go.Scatter(
                x=[xmin, xmax],
                y=[0, 0],
                mode="lines",
                line=dict(width=6, color=cgm["color"]),
                marker=dict(size=0),
                hoverinfo="none",
            )
            col_group_labels.append(trace)
            col_annotations.append(
                dict(
                    x=1 / 2 * (xmin + xmax),
                    y=-0.5,
                    xref="x15",
                    yref="y15",
                    text=cgm["annotation"],
                    font=self._annotation_font,
                    showarrow=False,
                )
            )

        return (row_group_labels, col_group_labels, row_annotations, col_annotations)
    

    
    
def _make_heatmap(data: pd.DataFrame(), title: str = None, labels_dict: dict = None,
                  labels_dict_reversed: dict = None,
                  width: int = 1500, height: int = 1500, label_size: str = "5pt",
                  title_size: str = "24pt", not_low_rank: bool = True):
    nlabels = len(labels_dict)
    shifted_labels_dict = {k + 0.5: v for k, v in labels_dict.items()}
    shifted_labels_dict_reversed = {k + 0.5: v for k, v in labels_dict_reversed.items()}

    df = data.iloc[::-1]  # rotate matrix 90 degrees
    df = pd.DataFrame(df.stack(), columns=['covariance']).reset_index()
    df.columns = ["taxa_y", "taxa_x", "covariance"]
    df = df.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})

    color_list, colors = _get_colors(df=df)
    # min_value = df['covariance'].min()
    # max_value = df['covariance'].max()
    # mapper = LinearColorMapper(palette=colors, low=min_value, high=max_value)
    mapper = LinearColorMapper(palette=colors, low=-1, high=1)
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))

    bottom, top, left, right = _get_bounds(nlabels=nlabels)

    source = ColumnDataSource(
        dict(top=top, bottom=bottom, left=left, right=right, color_list=color_list,
             taxa_x=df['taxa_x'], taxa_y=df['taxa_y'], covariance=df['covariance']))

    bokeh_tools = ["save, zoom_in, zoom_out, wheel_zoom, box_zoom, crosshair, reset, hover"]

    p = figure(width=width, height=height, x_range=(0, nlabels), y_range=(0, nlabels),
               title=title, title_location='above', x_axis_location="below",
               tools=bokeh_tools, toolbar_location='left')

    p.quad(top="top", bottom="bottom", left="left", right="right", line_color='white',
           color="color_list", source=source)
    p.xaxis.major_label_orientation = pi / 4
    p.yaxis.major_label_orientation = "horizontal"
    p.xaxis.major_label_text_font_size = label_size
    p.yaxis.major_label_text_font_size = label_size
    p.title.text_font_size = title_size
    p.add_layout(color_bar, 'right')
    p.toolbar.autohide = True
    p.xaxis.ticker = [x + 0.5 for x in
                      list(range(0, nlabels))]  ### shift label position to the center
    p.yaxis.ticker = [x + 0.5 for x in list(range(0, nlabels))]
    p.xaxis.major_label_overrides = shifted_labels_dict
    p.yaxis.major_label_overrides = shifted_labels_dict_reversed

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [
        ("taxa_x", "@taxa_x"),
        ("taxa_y", "@taxa_y"),
        ("covariance", "@covariance"),
    ]

    return p


def hierarchical_clustering(data: pd.DataFrame, order: list, n_covariates: int = None):
    """
    Performs hierarchical clustering on the input DataFrame using 
    the provided cluster order and returns the reorganized DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        order (list): The cluster order as a list of indices.
        n_covariates (int, optional): The number of covariates. Defaults to None.

    Returns:
        pd.DataFrame: The reorganized DataFrame.
    """
    if n_covariates is None:
        re_data = data.iloc[order, order]

    else:
        asv_part = data.iloc[:-n_covariates, :-n_covariates]
        re_asv_part = asv_part.iloc[order, order]
        cov_asv_part = data.iloc[:-n_covariates, -n_covariates:].iloc[order, :]
        cov_part = data.iloc[-n_covariates:, -n_covariates:]

        res = np.block([[re_asv_part.values, cov_asv_part.values],
                        [cov_asv_part.T.values, cov_part.values]])

        labels = list(re_asv_part.columns) + list(cov_part.columns)
        re_data = pd.DataFrame(res, index=labels, columns=labels)

    return re_data


def create_label_dict(df):
    """
    Creates a dictionary to map column indices to column labels and another 
    dictionary to map column indices to reversed column labels.
    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary maps column indices to column labels, 
        and the second dictionary maps column indices to reversed column labels.
    """
    n_labels = len(df.columns)
    labels_dict = dict(zip(range(n_labels), df.columns))
    labels_dict_reversed = dict(zip(range(n_labels),list(labels_dict.values())[::-1]))
    
    return labels_dict, labels_dict_reversed



def _get_bounds(nlabels: int):
    """
    Calculate bounds for creating a heatmap grid.

    Parameters:
        nlabels (int): The number of labels.

    Returns:
        tuple: A tuple of four lists containing the bottom, top, left, and right bounds for creating a heatmap grid.
    """
    bottom = list(chain.from_iterable([[ii] * nlabels for ii in range(nlabels)]))
    top = list(chain.from_iterable([[ii + 1] * nlabels for ii in range(nlabels)]))
    left = list(chain.from_iterable([list(range(nlabels)) for ii in range(nlabels)]))
    right = list(chain.from_iterable([list(range(1, nlabels + 1)) for ii in range(nlabels)]))

    return bottom, top, left, right


def _get_colors(df: pd.DataFrame):
    """
    Get colors for visualizing covariance values in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing covariance values.

    Returns:
        tuple: A tuple containing two lists - `color_list` and `colors`.
            - `color_list`: List of color codes based on covariance values.
            - `colors`: List of all available color codes.
    """
    rdbu = plt.get_cmap('RdBu')
    cmap = ListedColormap(rdbu(np.arange(256)))
    
    # Create a list of hex color codes from the colormap
    colors = [cmap(i)[:3] for i in range(256)]
    colors = ['#' + ''.join([format(int(c * 255), '02x') for c in color]) for color in colors]
    colors = colors[::-1]  # red - positive, blue - negative

    ccorr = np.arange(-1, 1, 1 / (len(colors) / 2))
    color_list = []
    for value in df.covariance.values:
        ind = bisect.bisect_left(ccorr, value)  # smart array insertion
        if ind == 0:  # avoid ind == -1 on the next step
            ind = ind + 1
        color_list.append(colors[ind - 1])
    return color_list, colors


def _get_order(data: pd.DataFrame, method: str = 'average', metric: str = 'euclidean'):
    """
    Performs hierarchical clustering on the input DataFrame and returns the cluster order.

    Args:
        data (pd.DataFrame): The input DataFrame.
        method (str, optional): The clustering method. Defaults to 'average'.
        metric (str, optional): The distance metric. Defaults to 'euclidean'.

    Returns:
        list: The cluster order as a list of indices.
    """
    grid = sns.clustermap(data, method=method, metric=metric, robust=True)
    plt.close()
    clust_order = grid.dendrogram_row.reordered_ind

    return clust_order



def plot_ordered_heatmap(data: pd.DataFrame, order: list(), n_covariates: int=None,
                 title: str=" ", width: int=1500, height: int=1500, label_size: str="16pt"):
    """
    Plots a heatmap using the input DataFrame and additional parameters.

    Args:
        data (pd.DataFrame): The input DataFrame.
        order: The order as a list of indices.
        n_covariates (int, optional): The number of covariates. Defaults to None.
        title (str, optional): The title of the heatmap. Defaults to " ".
        width (int, optional): The width of the heatmap in pixels. Defaults to 1500.
        height (int, optional): The height of the heatmap in pixels. Defaults to 1500.
        label_size (str, optional): The size of the labels. Defaults to "16pt".

    Returns:
        [type]: The plotted heatmap.
    """
    clust_data = hierarchical_clustering(data, order=order, n_covariates=n_covariates)
    lables_clust, re_labels_clust = create_label_dict(clust_data)

    p = _make_heatmap(data=clust_data, title=title, width=width, height=height, label_size=label_size,
                      labels_dict=lables_clust, labels_dict_reversed=re_labels_clust)
    return p
# column_order = list(map(str, status_dict.keys()))
# colors = [item['color'] for item in status_dict.values()]

# # sort columns by allergy status
# count_table = count_table.loc[:, column_order]

# columns = list(count_table.columns.values)
# rows = list(count_table.index)

# fig = dashbio.Clustergram(
#     data=count_table,
#     row_labels=rows,
#     column_labels=columns,
#     column_colors=colors,
#     cluster='row',
#     link_method="average",
#     height=400,
#     width=6400,
#     color_map='RdYlBu_r',
#     hidden_labels='column'
# )

# fig.show()

# fig.write_image("plots/png/ige_clustegram.png")
# fig.write_image("plots/pdf/ige_clustegram.pdf")
# fig.write_html("plots/html/ige_clustegram.html")