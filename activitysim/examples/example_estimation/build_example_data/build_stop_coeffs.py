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

    # Drop duplicates based on Description and value
    df_melted = df_melted.unique(subset=["Description", "value"], keep="first")

    # Check for duplicates in Description column to create coefficient names
    description_counts = df_melted.group_by("Description").agg(pl.count().alias("count"))
    dupes_df = description_counts.filter(pl.col("count") > 1)
    duplicate_descriptions = dupes_df.get_column("Description").to_list()

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

    # Load original spec for updating
    spec = pl.read_csv(f"stop_frequency_backup_{what}.csv", comment_prefix="#")
    alt_cols = spec.columns[2:]

    # Update spec with coefficient names
    for row in df_final.iter_rows(named=True):
        description = row["Description"]
        value = row["value"]
        coeff_name = row["coefficient_name"]
        
        # Get the row matching this description
        mask = spec.get_column("Description") == description
        if mask.any():
            row_idx = mask.arg_max()  # Get first matching index
            
            # Update alt columns for this row
            for alt_col in alt_cols:
                current_val = spec[row_idx, alt_col]
                if current_val == value:
                    # Create a new dataframe with the updated value
                    spec = spec.with_columns(
                        pl.when(
                            (pl.col("Description") == description) & 
                            (pl.col(alt_col) == value)
                        )
                        .then(pl.lit(coeff_name))
                        .otherwise(pl.col(alt_col))
                        .alias(alt_col)
                    )

    # Add Label column
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
