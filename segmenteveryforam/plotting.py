import matplotlib.pyplot as plt
import pandas as pd

def _format_metric_label(metric):
    """
    Convert morphometric column names into readable plot labels.
    """

    labels = {
        "area_mm2": "Area (mm²)",
        "perimeter_mm": "Perimeter (mm)",
        "major_diameter_mm": "Major diameter (mm)",
        "minor_diameter_mm": "Minor diameter (mm)",
        "orientation_rad": "Orientation (rad)",
        "orientation_deg": "Orientation (°)",
        "aspect_ratio": "Aspect ratio",
        "elongation": "Elongation",
        "roundness": "Roundness",
        "equivalent_diameter_mm": "Equivalent diameter (mm)",
        "circularity": "Circularity",
        "perimeter_area_ratio": "Perimeter-area ratio",
    }

    return labels.get(
        metric,
        metric.replace("_", " ").capitalize(),
    )


def plot_morphometric_histogram(
    data,
    metric="major_diameter_mm",
    bins=20,
    title=None,
    xlabel=None,
    ylabel="Number of specimens",
    show=True,
):
    """
    Plot a histogram for a specimen-level morphometric variable.

    Parameters
    ----------
    data : pandas.DataFrame
        Specimen-level morphometric data.

    metric : str, default="major_diameter_mm"
        Numeric column to plot.

    bins : int, default=20
        Number of histogram bins.

    title : str or None
        Optional custom figure title.

    xlabel : str or None
        Optional custom x-axis label. If None, the metric name
        is converted into a readable label.

    ylabel : str, default="Number of specimens"
        Y-axis label.

    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if metric not in data.columns:
        raise ValueError(
            f"Metric '{metric}' was not found in the DataFrame."
        )

    if not pd.api.types.is_numeric_dtype(data[metric]):
        raise TypeError(
            f"Metric '{metric}' must contain numeric data."
        )

    values = data[metric].dropna()

    if values.empty:
        raise ValueError(
            f"Metric '{metric}' contains no valid numeric values."
        )

    # Create readable label from column name
    metric_label = _format_metric_label(metric)

    if xlabel is None:
        xlabel = metric_label

    if title is None:
        sample_id = (
            data["sample_id"].iloc[0]
            if "sample_id" in data.columns
            else None
        )

        species = (
            data["species"].iloc[0]
            if "species" in data.columns
            else None
        )

        if sample_id is not None and species is not None:
            title = (
                f"{sample_id} | {species}\n"
                f"{metric_label} distribution"
            )
        else:
            title = f"{metric_label} distribution"

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        values,
        bins=bins,
        edgecolor="black",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_morphometric_boxplot(
    data,
    metric="major_diameter_mm",
    vertical="depth_ccsf",
    show_n=False,
    show_mean=True,
    connect_mean=True,
    title=None,
    xlabel=None,
    ylabel=None,
    invert_vertical=True,
    show=True,
):
    """
    Plot morphometric distributions by depth or age.

    Parameters
    ----------
    data : pandas.DataFrame
        Specimen-level morphometric data.

    metric : str, default="major_diameter_mm"
        Numeric morphometric column to plot.

    vertical : str, default="depth_ccsf"
        Vertical grouping variable, typically "depth_ccsf" or "Age_Ma".

    show_n : bool, default=False
        Whether to include specimen counts in vertical-axis labels.

    show_mean : bool, default=True
        Whether to show mean markers on each boxplot.

    connect_mean : bool, default=True
        Whether to connect mean values across depth/age groups.

    title : str or None
        Optional custom title.

    xlabel : str or None
        Optional custom x-axis label.

    ylabel : str or None
        Optional custom vertical-axis label.

    invert_vertical : bool, default=True
        Whether to invert the vertical axis.

    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if metric not in data.columns:
        raise ValueError(
            f"Metric '{metric}' was not found in the DataFrame."
        )

    if vertical not in data.columns:
        raise ValueError(
            f"Vertical variable '{vertical}' was not found in the DataFrame."
        )

    if not pd.api.types.is_numeric_dtype(data[metric]):
        raise TypeError(
            f"Metric '{metric}' must contain numeric data."
        )

    if not pd.api.types.is_numeric_dtype(data[vertical]):
        raise TypeError(
            f"Vertical variable '{vertical}' must contain numeric data."
        )

    plot_data = data[
        [metric, vertical]
    ].dropna()

    if plot_data.empty:
        raise ValueError(
            "No valid data remain after removing missing values."
        )

    positions = sorted(
        plot_data[vertical].unique()
    )

    boxplot_data = [
        plot_data.loc[
            plot_data[vertical] == value,
            metric
        ]
        for value in positions
    ]

    counts = (
        plot_data
        .groupby(vertical)
        .size()
        .reindex(positions)
    )

    mean_values = (
        plot_data
        .groupby(vertical)[metric]
        .mean()
        .reindex(positions)
    )

    metric_label = _format_metric_label(metric)

    if xlabel is None:
        xlabel = metric_label

    if ylabel is None:
        if vertical == "depth_ccsf":
            ylabel = "Depth (ccsf)"
        elif vertical == "Age_Ma":
            ylabel = "Age (Ma)"
        else:
            ylabel = vertical.replace("_", " ").capitalize()

    if title is None:
        species = (
            data["species"].iloc[0]
            if "species" in data.columns
            else None
        )

        if species is not None:
            title = (
                f"{species}: {metric_label} distribution by {ylabel}"
            )
        else:
            title = (
                f"{metric_label} distribution by {ylabel}"
            )

    meanprops = dict(
        marker="o",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=6,
    )

    boxprops = dict(
        facecolor="lightblue",
        edgecolor="black",
    )

    medianprops = dict(
        color="orange",
        linewidth=2,
    )

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.boxplot(
        boxplot_data,
        positions=positions,
        vert=False,
        widths=0.6,
        notch=True,
        showmeans=show_mean,
        patch_artist=True,
        meanprops=meanprops,
        boxprops=boxprops,
        medianprops=medianprops,
    )

    if connect_mean:
        ax.plot(
            mean_values.values,
            positions,
            linewidth=1.5,
            marker="o",
            markersize=6,
            zorder=4,
            label="Mean",
        )

        ax.legend(loc="best")

    if show_n:
        labels = [
            f"{value:.2f} (n={int(count)})"
            for value, count in zip(
                positions,
                counts.values,
            )
        ]

        ax.set_yticks(positions)
        ax.set_yticklabels(labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if invert_vertical:
        ax.invert_yaxis()

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def plot_morphometric_profile(
    summary,
    metric="major_diameter_mm",
    vertical="depth_ccsf",
    statistic="mean",
    uncertainty=None,
    show_n=False,
    title=None,
    xlabel=None,
    ylabel=None,
    invert_vertical=True,
    show=True,
):
    """
    Plot a morphometric statistic through depth or age.

    Parameters
    ----------
    summary : pandas.DataFrame
        Summary-level morphometric data produced by
        summarize_foram_morphometrics().

    metric : str, default="major_diameter_mm"
        Morphometric variable to plot.

    vertical : str, default="depth_ccsf"
        Vertical-axis variable, typically "depth_ccsf" or "Age_Ma".

    statistic : str, default="mean"
        Summary statistic to plot, such as:
        "mean", "median", "p25", "p75", or "p95".

    uncertainty : str or None, default=None
        Optional uncertainty to display.

        Supported values:
        - None
        - "sem"
        - "std"

        SEM is calculated from:
        std / sqrt(n_specimens)

    show_n : bool, default=False
        Whether to annotate specimen counts.

    title : str or None
        Optional custom title.

    xlabel : str or None
        Optional custom horizontal-axis label.

    ylabel : str or None
        Optional custom vertical-axis label.

    invert_vertical : bool, default=True
        Whether to invert the vertical axis.

    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """

    import numpy as np

    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame.")

    if vertical not in summary.columns:
        raise ValueError(
            f"Vertical variable '{vertical}' was not found in the DataFrame."
        )

    statistic_column = f"{statistic}_{metric}"

    if statistic_column not in summary.columns:
        raise ValueError(
            f"Column '{statistic_column}' was not found in the summary DataFrame."
        )

    if not pd.api.types.is_numeric_dtype(summary[statistic_column]):
        raise TypeError(
            f"Column '{statistic_column}' must contain numeric data."
        )

    if not pd.api.types.is_numeric_dtype(summary[vertical]):
        raise TypeError(
            f"Vertical variable '{vertical}' must contain numeric data."
        )

    plot_data = summary.copy()

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    uncertainty_values = None

    if uncertainty is not None:

        if uncertainty == "sem":

            if statistic != "mean":
                raise ValueError(
                    "SEM is currently supported only when statistic='mean'."
                )

            std_column = f"std_{metric}"

            if std_column not in plot_data.columns:
                raise ValueError(
                    f"Column '{std_column}' is required to calculate SEM."
                )

            if "n_specimens" not in plot_data.columns:
                raise ValueError(
                    "Column 'n_specimens' is required to calculate SEM."
                )

            uncertainty_values = (
                plot_data[std_column]
                / np.sqrt(plot_data["n_specimens"])
            )

        elif uncertainty == "std":

            std_column = f"std_{metric}"

            if std_column not in plot_data.columns:
                raise ValueError(
                    f"Column '{std_column}' was not found."
                )

            uncertainty_values = plot_data[std_column]

        else:
            raise ValueError(
                "uncertainty must be None, 'sem', or 'std'."
            )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS AND SORT
    # --------------------------------------------------------

    valid = (
        plot_data[statistic_column].notna()
        & plot_data[vertical].notna()
    )

    plot_data = plot_data.loc[valid].copy()

    if uncertainty_values is not None:
        uncertainty_values = uncertainty_values.loc[valid]

    if plot_data.empty:
        raise ValueError(
            "No valid data remain after removing missing values."
        )

    sort_order = plot_data[vertical].sort_values().index

    plot_data = plot_data.loc[sort_order]

    if uncertainty_values is not None:
        uncertainty_values = uncertainty_values.loc[sort_order]

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    metric_label = _format_metric_label(metric)

    statistic_labels = {
        "mean": "Mean",
        "median": "Median",
        "p25": "25th percentile",
        "p75": "75th percentile",
        "p95": "95th percentile",
    }

    statistic_label = statistic_labels.get(
        statistic,
        statistic.replace("_", " ").capitalize(),
    )

    if xlabel is None:
        xlabel = f"{statistic_label} {metric_label.lower()}"

    if ylabel is None:
        if vertical == "depth_ccsf":
            ylabel = "Depth (ccsf)"
        elif vertical == "Age_Ma":
            ylabel = "Age (Ma)"
        else:
            ylabel = vertical.replace("_", " ").capitalize()

    if title is None:
        species = (
            summary["species"].iloc[0]
            if "species" in summary.columns
            else None
        )

        if species is not None:
            title = (
                f"{species}: {statistic_label} "
                f"{metric_label.lower()}"
            )
        else:
            title = (
                f"{statistic_label} {metric_label.lower()}"
            )

        if uncertainty is not None:
            title += f" ± {uncertainty.upper()}"

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 8))

    x = plot_data[statistic_column]
    y = plot_data[vertical]

    # Optional uncertainty
    if uncertainty_values is not None:
        ax.errorbar(
            x,
            y,
            xerr=uncertainty_values,
            fmt="none",
            elinewidth=1.5,
            capsize=4,
            capthick=1,
            zorder=1,
        )

    # Main profile
    ax.plot(
        x,
        y,
        "o-",
        linewidth=2,
        markersize=6,
        zorder=2,
    )

    # --------------------------------------------------------
    # OPTIONAL SPECIMEN COUNTS
    # --------------------------------------------------------

    if show_n:

        if "n_specimens" not in plot_data.columns:
            raise ValueError(
                "Column 'n_specimens' is required when show_n=True."
            )

        x_range = x.max() - x.min()

        offset = (
            0.02 * x_range
            if x_range > 0
            else 0.01
        )

        for idx, row in plot_data.iterrows():

            x_position = row[statistic_column]

            if uncertainty_values is not None:
                uncertainty_value = uncertainty_values.loc[idx]

                if pd.notna(uncertainty_value):
                    x_position += uncertainty_value

            ax.text(
                x_position + offset,
                row[vertical],
                f"n={int(row['n_specimens'])}",
                fontsize=8,
                va="center",
            )

    # --------------------------------------------------------
    # FINAL FORMATTING
    # --------------------------------------------------------

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if invert_vertical:
        ax.invert_yaxis()

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax