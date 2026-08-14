def to_jsonable_df(df):
    """
    Convert a pandas DataFrame (or dataframe-like) into JSON-safe python types.

    We keep this generic so preprocessing tests can run in minimal environments
    when `preprocessing_option=none`.
    """

    if df is None:
        return None

    if hasattr(df, "to_dict"):
        data = df.to_dict(orient="list")
        # Convert numpy arrays/scalars to plain python lists/numbers.
        for k, v in list(data.items()):
            if hasattr(v, "tolist"):
                data[k] = v.tolist()
        return data

    return df
