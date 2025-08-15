#  python ~/work/activitysim/activitysim/examples/example_estimation/build_example_data/build_stop_coeffs.py

from __future__ import annotations

import numpy as np
import polars as pl

FIRST_RUN = True

# work, school, univ, social, shopping, eatout, escort,atwork,othmaint,othdiscr
for what in [
    "work",
    "school",
    "univ",
    "social",
    "shopping",
    "eatout",
    "escort",
    "atwork",
    "othmaint",
    "othdiscr",
]:
    if FIRST_RUN:
        df = pl.read_csv(f"stop_frequency_{what}.csv", comment_prefix="#")
        df.write_csv(f"stop_frequency_backup_{what}.csv")
    else:
        df = pl.read_csv(f"stop_frequency_backup_{what}.csv", comment_prefix="#")

    # Drop Expression column
    df = df.drop("Expression")

    # Convert to long format (equivalent to pandas unstack)
    df_melted = df.melt(id_vars="Description", variable_name="alt", value_name="value")

    # Drop null values
    df_melted = df_melted.drop_nulls()

    # Drop duplicates based on Description and value (keep first)
    df_melted = df_melted.unique(subset=["Description", "value"], keep="first")

    # Check for duplicates in Description column to create coefficient names
    description_counts = df_melted.group_by("Description").agg(pl.count().alias("count"))
    duplicate_descriptions = (
        description_counts
        .filter(pl.col("count") > 1)
        .get_column("Description")
        .to_list()
    )

    # Create coefficient names based on whether there are duplicates
    df_melted = df_melted.with_columns(
        pl.when(pl.col("Description").is_in(duplicate_descriptions))
        .then(pl.lit("coef_") + pl.col("Description") + "_" + pl.col("alt"))
        .otherwise(pl.lit("coef_") + pl.col("Description"))
        .alias("coefficient_name")
    )

    # Clean up coefficient names
    df_melted = df_melted.with_columns(
        pl.col("coefficient_name")
        .str.to_lowercase()
        .str.replace_all(r"[^a-zA-Z0-9]+", "_")
        .alias("coefficient_name")
    )

    # Remove alt column and save coefficients
    df_final = df_melted.drop("alt")
    df_final.write_csv(f"stop_frequency_coefficients_{what}.csv")

    # Load original spec for updating - convert to pandas for easier manipulation
    # then convert back to polars for output
    spec = pl.read_csv(f"stop_frequency_backup_{what}.csv", comment_prefix="#")
    spec_pd = spec.to_pandas()
    
    alt_cols = spec.columns[2:]

    # Update spec with coefficient names - use the original pandas-style logic
    # for easier row/column manipulation
    for row in df_final.iter_rows(named=True):
        description = row["Description"]
        value = row["value"]
        coeff_name = row["coefficient_name"]
        
        # Create mapping for this coefficient
        value_to_coeff = {value: coeff_name}
        
        # Get the row(s) matching this description
        mask = spec_pd["Description"] == description
        if mask.any():
            # For each alternative column, replace matching values
            for alt_col in alt_cols:
                original_values = spec_pd.loc[mask, alt_col].values
                updated_values = [value_to_coeff.get(val, val) for val in original_values]
                spec_pd.loc[mask, alt_col] = updated_values

    # Convert back to polars and add Label column
    spec = pl.from_pandas(spec_pd)
    spec = spec.with_columns(
        (pl.lit("util_") + pl.col("Description")).alias("Label")
    )

    # Clean up Label column
    spec = spec.with_columns(
        pl.col("Label")
        .str.to_lowercase()
        .str.replace_all(r"[^a-zA-Z0-9]+", "_")
        .alias("Label")
    )

    # Reorder columns to put Label first
    columns = ["Label"] + [col for col in spec.columns if col != "Label"]
    spec = spec.select(columns)

    # Save final files
    df_final.write_csv(f"stop_frequency_coefficients_{what}.csv")
    spec.write_csv(f"stop_frequency_{what}.csv")
