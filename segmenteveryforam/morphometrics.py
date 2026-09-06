import numpy as np
import pandas as pd


def get_foram_morphometrics(summary, metadata=None):
    """
    Build a specimen-level foraminiferal morphometric dataset from
    SegmentEveryForam summary measurements.

    Parameters
    ----------
    summary : pandas.DataFrame
        Base specimen measurements produced by SegmentEveryForam.

    metadata : pandas.Series, dict, or None, optional
        Metadata associated with the image or sample. Each metadata
        field is repeated for every specimen in the output.

    Returns
    -------
    pandas.DataFrame
        One row per foraminifer containing core and derived
        morphometric measurements.
    """

    required_columns = [
        "area",
        "perimeter",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "centroid-0",
        "centroid-1",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in summary.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in summary: "
            + ", ".join(missing_columns)
        )

    foram_data = pd.DataFrame(index=summary.index.copy())

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if metadata is not None:
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()

        # Metadata used internally by the workflow but not needed
        # in specimen-level morphometric output
        excluded_metadata = {
            "image_path",
        }

        for key, value in metadata.items():
            if key not in excluded_metadata:
                foram_data[key] = value
    
    # --------------------------------------------------------
    # Specimen identifier
    # --------------------------------------------------------

    foram_data["foram_id"] = np.arange(
        1,
        len(summary) + 1
    )

    # --------------------------------------------------------
    # Core morphometrics
    # --------------------------------------------------------

    # SegmentEveryForam measurements are assumed to be in
    # meters and square meters after scale calibration.
    foram_data["area_mm2"] = (
        summary["area"] * 1_000_000
    )

    foram_data["perimeter_mm"] = (
        summary["perimeter"] * 1000
    )

    foram_data["major_diameter_mm"] = (
        summary["major_axis_length"] * 1000
    )

    foram_data["minor_diameter_mm"] = (
        summary["minor_axis_length"] * 1000
    )

    foram_data["orientation_rad"] = (
        summary["orientation"]
    )

    # --------------------------------------------------------
    # Derived morphometrics
    # --------------------------------------------------------

    major = foram_data["major_diameter_mm"]
    minor = foram_data["minor_diameter_mm"]
    area = foram_data["area_mm2"]
    perimeter = foram_data["perimeter_mm"]

    foram_data["aspect_ratio"] = (
        major / minor
    )

    foram_data["elongation"] = (
        1 - (minor / major)
    )

    foram_data["roundness"] = (
        minor / major
    )

    foram_data["equivalent_diameter_mm"] = np.sqrt(
        4 * area / np.pi
    )

    foram_data["circularity"] = (
        4 * np.pi * area / (perimeter ** 2)
    )

    foram_data["perimeter_area_ratio"] = (
        perimeter / area
    )

    foram_data["orientation_deg"] = np.degrees(
        foram_data["orientation_rad"]
    )

    # --------------------------------------------------------
    # Specimen position in image
    # --------------------------------------------------------

    foram_data["centroid_y_px"] = (
        summary["centroid-0"]
    )

    foram_data["centroid_x_px"] = (
        summary["centroid-1"]
    )

    # Replace invalid divisions with NaN
    foram_data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return foram_data

def summarize_foram_morphometrics(
    foram_data,
    group_by=None,
):
    """
    Summarize specimen-level foraminiferal morphometrics.

    Parameters
    ----------
    foram_data : pandas.DataFrame
        Specimen-level output from get_foram_morphometrics().

    group_by : list of str or None, optional
        Columns used to define sample groups. If None, all rows are
        summarized together.

    Returns
    -------
    pandas.DataFrame
        One row per group containing specimen counts and descriptive
        statistics for morphometric variables.
    """

    metrics = [
        "area_mm2",
        "perimeter_mm",
        "major_diameter_mm",
        "minor_diameter_mm",
        "equivalent_diameter_mm",
        "aspect_ratio",
        "elongation",
        "roundness",
        "circularity",
        "perimeter_area_ratio",
    ]

    missing_metrics = [
        metric
        for metric in metrics
        if metric not in foram_data.columns
    ]

    if missing_metrics:
        raise ValueError(
            "Missing required morphometric columns: "
            + ", ".join(missing_metrics)
        )

    if group_by is None:
        grouped = [(None, foram_data)]

    else:
        missing_group_columns = [
            column
            for column in group_by
            if column not in foram_data.columns
        ]

        if missing_group_columns:
            raise ValueError(
                "Missing group_by columns: "
                + ", ".join(missing_group_columns)
            )

        grouped = foram_data.groupby(
            group_by,
            dropna=False,
            sort=False,
        )

    rows = []

    for group_key, group in grouped:

        row = {}

        # --------------------------------------------
        # Group metadata
        # --------------------------------------------

        if group_by is not None:

            if len(group_by) == 1:
                group_key = (group_key,)

            for column, value in zip(
                group_by,
                group_key,
            ):
                row[column] = value

        # --------------------------------------------
        # Basic specimen information
        # --------------------------------------------

        row["n_specimens"] = len(group)

        if "image_id" in group.columns:
            image_ids = (
                group["image_id"]
                .dropna()
                .astype(str)
                .unique()
            )

            row["image_ids"] = ",".join(
                sorted(image_ids)
            )

        # --------------------------------------------
        # Morphometric statistics
        # --------------------------------------------

        for metric in metrics:

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()
            
            if len(values) == 0:

                row[f"mean_{metric}"] = np.nan
                row[f"median_{metric}"] = np.nan
                row[f"std_{metric}"] = np.nan
                row[f"p25_{metric}"] = np.nan
                row[f"p75_{metric}"] = np.nan
                row[f"p95_{metric}"] = np.nan
                row[f"iqr_{metric}"] = np.nan

                continue

            row[f"mean_{metric}"] = values.mean()
            row[f"median_{metric}"] = values.median()
            row[f"std_{metric}"] = values.std()

            p25 = values.quantile(0.25)
            p75 = values.quantile(0.75)

            row[f"p25_{metric}"] = p25
            row[f"p75_{metric}"] = p75
            row[f"p95_{metric}"] = values.quantile(0.95)
            row[f"iqr_{metric}"] = p75 - p25

        rows.append(row)

    return pd.DataFrame(rows)