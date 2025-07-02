import greeks
import polars as pl

from datetime import datetime

def get_underlying():
    """Fetches the SPX index data from a CSV file.

    Returns:
        DataFrame: A Polars DataFrame containing the SPX index data.
    """
    spx_df = pl.read_csv("../spx-index.csv", try_parse_dates=True)
    spx_df = spx_df.with_columns(pl.col("Date").dt.date())
    spx_df = spx_df.drop(["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"])
    spx_df = spx_df.rename({"Date": "date", "Close": "spx_close"})
    return spx_df

def add_underlying(options_df: pl.DataFrame) -> pl.DataFrame:
    """Adds the SPX index data to the options DataFrame.

    Args:
        options_df (pl.DataFrame): The options DataFrame.
        
    Returns:
        pl.DataFrame: The options DataFrame with SPX index data added.  
    """
    options_df = options_df.with_columns(date = pl.col("datetime").dt.date())
    options_df = pl.concat([options_df, spx_df], how="align_left")
    return options_df

if __name__ == "__main__":

    # Import SOFR interest rates data
    sofr = greeks.get_sofr()

    # Import options data
    #options_df = pl.read_parquet("../data/spxw_quotes_20240705.parquet")
    asset = "SPXW"
    date = "20240624"
    
    options_df = pl.read_parquet("../data/" + asset +"-" + date + "-all-strikes.parquet")  

    # options_df = pl.read_csv(
    #     "../data/" + asset +"-" + date + "-all-strikes.csv",
    #     try_parse_dates=True,
    #     infer_schema_length=1000000,
    # )

    # We get the daily prices of the underlying
    spx_df = get_underlying()

    # We add a column with the underlying prices to the options dataframe
    options_df = add_underlying(options_df)

    # Join the two dataframes on the date_time column to have the interest_rate column
    options_df = options_df.join(sofr, on="date", how="inner")

    # Add a column with the number of days to expiration
    options_df = options_df.with_columns(
        ((pl.col("expiration").cast(pl.Date) - pl.col("date").cast(pl.Date)).cast(pl.Int64)
            / (24.0 * 60 * 60 * 1_000)
        ).alias("days_to_exp")
    )

    # Add a column with the moneyness
    options_df = options_df.with_columns(moneyness = pl.col("strike") / pl.col("spx_close"))

    # Filter only the EOD options
    options_df = options_df.filter(pl.col("datetime").dt.time() == datetime.strptime("16:00:00", "%H:%M:%S").time())

    # Add a column with the Implied Volatility, Delta, Theta and Rho
    options_df = greeks.iv_calculator(options_df)

    # Add a column with the Gamma (calculation is vectorized using Polars)
    options_df = greeks.gamma_calculator(options_df)

    # Add columns with the Vega (calculation is vectorized using Polars)
    options_df = greeks.vega_calculator(options_df)

    # Save the final DataFrame to a Parquet file
    options_df.write_parquet("../data/greeks-" + asset + "-" + date + ".parquet")