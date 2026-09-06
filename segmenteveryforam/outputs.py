from pathlib import Path

import pandas as pd


def save_foram_morphometrics(
    foram_data,
    output_dir="foram_morphometrics_outputs",
):
    """
    Save or append specimen-level foraminiferal morphometric data.

    One CSV file is created per sample and species.

    Existing data are preserved, while duplicate specimens from
    reprocessed images are removed using image_id and foram_id.

    Parameters
    ----------
    foram_data : pandas.DataFrame
        Specimen-level morphometric output from
        get_foram_morphometrics().

    output_dir : str or pathlib.Path, optional
        Directory where specimen-level CSV files are stored.

    Returns
    -------
    pathlib.Path
        Path to the saved CSV file.

    pandas.DataFrame
        Combined specimen-level dataset saved to disk.
    """

    required_columns = [
        "sample_id",
        "species",
        "image_id",
        "foram_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in foram_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in foram_data: "
            + ", ".join(missing_columns)
        )

    if foram_data.empty:
        raise ValueError(
            "foram_data is empty and cannot be saved."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_ids = foram_data["sample_id"].dropna().unique()
    species_values = foram_data["species"].dropna().unique()

    if len(sample_ids) != 1:
        raise ValueError(
            "foram_data must contain exactly one sample_id "
            "when using save_foram_morphometrics()."
        )

    if len(species_values) != 1:
        raise ValueError(
            "foram_data must contain exactly one species "
            "when using save_foram_morphometrics()."
        )

    sample_id = str(sample_ids[0])
    species = str(species_values[0])

    foram_data_path = (
        output_dir
        / f"{sample_id}_{species}_foram_data.csv"
    )

    if foram_data_path.exists():
        existing_data = pd.read_csv(
            foram_data_path
        )

        combined_data = pd.concat(
            [existing_data, foram_data],
            ignore_index=True,
        )

    else:
        combined_data = foram_data.copy()

    combined_data = combined_data.drop_duplicates(
        subset=[
            "image_id",
            "foram_id",
        ],
        keep="last",
    )

    combined_data = combined_data.reset_index(
        drop=True
    )

    combined_data.to_csv(
        foram_data_path,
        index=False,
    )

    return foram_data_path, combined_data

def save_foram_summary(
    sample_summary,
    output_dir="foram_average_measurement_outputs",
):
    """
    Save sample-level foraminiferal morphometric summary data.

    One CSV file is created per sample and species.

    Parameters
    ----------
    sample_summary : pandas.DataFrame
        Sample-level output from summarize_foram_morphometrics().

    output_dir : str or pathlib.Path, optional
        Directory where sample-summary CSV files are stored.

    Returns
    -------
    pathlib.Path
        Path to the saved CSV file.
    """

    required_columns = [
        "sample_id",
        "species",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in sample_summary.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in sample_summary: "
            + ", ".join(missing_columns)
        )

    if sample_summary.empty:
        raise ValueError(
            "sample_summary is empty and cannot be saved."
        )

    sample_ids = sample_summary["sample_id"].dropna().unique()
    species_values = sample_summary["species"].dropna().unique()

    if len(sample_ids) != 1:
        raise ValueError(
            "sample_summary must contain exactly one sample_id "
            "when using save_foram_summary()."
        )

    if len(species_values) != 1:
        raise ValueError(
            "sample_summary must contain exactly one species "
            "when using save_foram_summary()."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_id = str(sample_ids[0])
    species = str(species_values[0])

    summary_path = (
        output_dir
        / f"{species}_summary.csv"
    )

    # If a species summary already exists, combine it with
    # the newly calculated sample summary.
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)

        combined_summary = pd.concat(
            [existing_summary, sample_summary],
            ignore_index=True,
        )

    else:
        combined_summary = sample_summary.copy()


    # If a sample is reprocessed, keep the newest summary
    # rather than creating a duplicate row.
    combined_summary = combined_summary.drop_duplicates(
        subset=["sample_id", "species"],
        keep="last",
    )


    # Keep rows organized by depth when available.
    if "depth_ccsf" in combined_summary.columns:
        combined_summary = combined_summary.sort_values(
            "depth_ccsf"
        )


    combined_summary = combined_summary.reset_index(
        drop=True
    )


    combined_summary.to_csv(
        summary_path,
        index=False,
    )

    return summary_path, combined_summary     