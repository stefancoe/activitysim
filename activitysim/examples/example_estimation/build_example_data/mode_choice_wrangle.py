import numpy as np
import polars as pl

df = pl.read_csv("trip_mode_coefficients_p.csv", comment_prefix="#")

alts = [col for col in df.columns if col != "Expression"]
alts_str = "_".join(alts)

# Convert to long format (equivalent to pandas unstack)
df_melted = df.melt(id_vars="Expression", variable_name="alts", value_name="value")

# Group by Expression and value, then aggregate alts with string join
df_grouped = (
    df_melted
    .group_by(["Expression", "value"])
    .agg(pl.col("alts").str.concat("_").alias("alts"))
)

# Create coefficient names
df_grouped = df_grouped.with_columns(
    pl.when(pl.col("alts") == alts_str)
    .then(pl.lit("coef_") + pl.col("Expression"))
    .otherwise(pl.lit("coef_") + pl.col("Expression") + "_" + pl.col("alts"))
    .alias("coefficient_name")
)

coefficients_df = df_grouped

# Re-read the original data
df = pl.read_csv("trip_mode_coefficients_p.csv", comment_prefix="#")

# For each alternative, merge with coefficients to replace values with coefficient names
for alt in alts:
    alt_df = (
        df.select(["Expression", alt])
        .rename({alt: "value"})
        .join(
            coefficients_df.select(["Expression", "value", "coefficient_name"]),
            on=["Expression", "value"],
            how="left"
        )
    )
    # Update the original dataframe with coefficient names
    df = df.with_columns(
        alt_df.get_column("coefficient_name").alias(alt)
    )

# Write outputs
coefficients_df.select(["coefficient_name", "value"]).write_csv("trip_mode_choice_coefficients.csv")
df.write_csv("trip_mode_choice_coefficients_template.csv")
