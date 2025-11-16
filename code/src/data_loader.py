import pandas as pd


def _read_csv(path, dt_col, freq):
    """
    Read a CSV file, parse the datetime column, resample to a fixed
    frequency, and return a regularized DataFrame.
    """
    df = pd.read_csv(path)
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.dropna(subset=[dt_col])
    df = df.set_index(dt_col).resample(freq).mean().reset_index()
    return df


def load_and_merge(cfg):
    """
    Load forecast, weather, and emissions CSVs according to the config,
    merge them on the datetime column, and return (df, meta).

    cfg should contain:
      - cfg["dataset"]: dataset-level options (column names, frequency, etc.)
      - cfg["paths"]: paths to the individual CSV files:
          * "forecast_csv"
          * "weather_csv"
          * "emission_csv"
    """
    ds = cfg["dataset"]
    paths = cfg["paths"]

    dt_col = ds["datetime_col"]
    freq = ds.get("freq", "h")

    # Load each source and regularize to the same time grid
    f = _read_csv(paths["forecast_csv"], dt_col, freq)
    w = _read_csv(paths["weather_csv"], dt_col, freq)
    e = _read_csv(paths["emission_csv"], dt_col, freq)

    # Outer-join forecast + weather, then inner-join emissions
    fw = pd.merge(f, w, on=dt_col, how="outer", suffixes=("_f", "_w"))
    df = pd.merge(fw, e, on=dt_col, how="inner")

    # Ensure a region column exists
    region_col = ds.get("region_col", None)
    if region_col is None or region_col not in df.columns:
        df["region"] = ds.get("region_default", "REGION")
        region_col = "region"

    # Drop rows without target values
    df = df.dropna(subset=[ds["target_col"]])

    meta = {
        "dt_col": dt_col,
        "region_col": region_col,
        "target_col": ds["target_col"],
        "freq": freq,
    }
    return df, meta