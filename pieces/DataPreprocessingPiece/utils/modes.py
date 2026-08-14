def _resolve_features(payload, data, *, target_columns=None):
    configured_features = payload.get("preprocessor_features")
    if configured_features:
        return list(configured_features)

    excluded = set(target_columns or [])

    # Auto-select numeric columns only: trainers coerce features via pd.to_numeric, so
    # date/datetime/string columns would otherwise drop every row.
    if hasattr(data, "select_dtypes"):
        numeric_columns = list(data.select_dtypes(include="number").columns)
    else:
        numeric_columns = list(getattr(data, "columns", []))

    auto_features = [column for column in numeric_columns if column not in excluded]
    if not auto_features:
        raise ValueError(
            "Could not infer feature columns. Provide `preprocessor_features` in payload."
        )
    return auto_features


def _read_input_dataframe(payload):
    for key in ("dataframe", "X", "data"):
        if payload.get(key) is not None:
            return payload.get(key)
    return None


def _ensure_datetime_column(data):
    import pandas as pd  # type: ignore

    if "datetime" in data.columns:
        return data
    if "Date" in data.columns and "Time" in data.columns:
        data = data.copy()
        data["datetime"] = pd.to_datetime(
            data["Date"].astype(str) + " " + data["Time"].astype(str),
            dayfirst=True,
            errors="coerce",
        )
        return data
    if "timestamp_utc" in data.columns:
        data = data.copy()
        data["datetime"] = pd.to_datetime(data["timestamp_utc"], errors="coerce")
        return data
    return data


def _load_prediction_data(path):
    import pandas as pd  # type: ignore

    if str(path).lower().endswith(".json"):
        data = pd.read_json(path)
        return _ensure_datetime_column(data)

    try:
        data = pd.read_csv(
            path,
            sep=";",
            skiprows=58,
            parse_dates={"datetime": ["Date", "Time"]},
            dayfirst=True,
        )
        return data
    except (ValueError, KeyError):
        data = pd.read_csv(path)
        return _ensure_datetime_column(data)


def preprocess_prediction(payload):
    import os
    import pandas as pd  # type: ignore

    from .preprocessor_utils import (
        ensure_datetime_column,
        flag_each_day,
        preprocess_solargis_data,
    )
    from .serialization import to_jsonable_df

    df = _read_input_dataframe(payload)
    data_path = payload.get("data_path")
    data_path_solargis = payload.get("data_path_solargis")
    data_path_okte = payload.get("data_path_okte")
    save_data_path = payload.get("save_data_path")
    flag_each_day_enabled = bool(payload.get("flag_each_day", False))
    keep_datetime = bool(payload.get("keep_datetime", False))

    def _read_supported_csv(path: str):
        from pathlib import Path as _Path

        if not path:
            raise ValueError("Empty CSV path.")
        if not _Path(path).is_file():
            raise FileNotFoundError(
                f"Input CSV not found at `{path}`. "
                "Domino must use Shared Storage = Local so Open-Meteo output "
                "is visible to Data Preprocessing."
            )
        read_attempts = [
            {"sep": ";", "skiprows": 0, "dayfirst": True},
            {"sep": ",", "skiprows": 0, "dayfirst": True},
            {"sep": None, "engine": "python", "skiprows": 0, "dayfirst": True},
            # Legacy commercial SolarGIS exports with a long header block.
            {"sep": ";", "skiprows": 58, "dayfirst": True},
            {"sep": None, "engine": "python", "skiprows": 58, "dayfirst": True},
        ]
        last_err: Exception | None = None
        seen_cols: list[str] = []
        for kwargs in read_attempts:
            try:
                candidate = pd.read_csv(path, **kwargs)
            except Exception as exc:  # noqa: BLE001 - try next dialect
                last_err = exc
                continue
            seen_cols = [str(c) for c in candidate.columns]
            if (
                "datetime" in candidate.columns
                or ("Date" in candidate.columns and "Time" in candidate.columns)
                or "timestamp_utc" in candidate.columns
            ):
                return candidate
        detail = f" Last read error: {last_err}." if last_err else ""
        raise ValueError(
            "Unable to read input CSV with supported schemas. "
            "Expected a `datetime`, `timestamp_utc`, or `Date`+`Time` column. "
            f"Path=`{path}` columns={seen_cols}.{detail}"
        )

    target_col_input = payload.get("target_column")
    target_col = str(target_col_input) if target_col_input else "PVOUT"

    # Prefer generic data_path (Open-Meteo). data_path_solargis remains as alias only.
    weather_path = data_path or data_path_solargis

    if df is None:
        if not weather_path and not data_path_okte:
            raise ValueError(
                "preprocessing_option='prediction' requires `payload['dataframe']`, "
                "or `payload['data_path']` (Open-Meteo CSV), or "
                "`payload['data_path_okte']`."
            )

        if weather_path and data_path_okte:
            weather_df = _read_supported_csv(weather_path)
            weather_df = ensure_datetime_column(weather_df)

            okte_df = _read_supported_csv(data_path_okte)
            okte_df = ensure_datetime_column(okte_df)

            okte_drop = [
                c for c in ("Date", "Time")
                if c in okte_df.columns and c not in weather_df.columns
            ]
            if okte_drop:
                okte_df = okte_df.drop(columns=okte_drop)

            df = pd.merge(
                weather_df,
                okte_df,
                on="datetime",
                how="inner",
                suffixes=("", "_okte"),
            )
            if df.empty:
                raise ValueError(
                    "Inner-join on `datetime` produced 0 rows. Check that the Open-Meteo "
                    "and OKTE generators emit overlapping timestamps."
                )
        elif weather_path:
            df = _read_supported_csv(weather_path)
        else:
            df = _read_supported_csv(data_path_okte)

    data = df
    data = ensure_datetime_column(data)
    if flag_each_day_enabled:
        data = flag_each_day(data)

    # SolarGIS-specific preprocessing only when the dataset has the expected columns.
    if all(col in data.columns for col in ("GHI", "DIF", "SE")):
        data = preprocess_solargis_data(data)
    else:
        data = data.dropna()

    if save_data_path:
        os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
        data.to_csv(save_data_path, index=False)

    if target_col not in data.columns:
        if target_col_input is not None:
            raise ValueError(
                f"Target column `{target_col}` not found in data. "
                f"Available columns: {list(data.columns)}. "
                "Pass `target_column` in piece input to match your dataset."
            )
        # Default PVOUT not found — dataset has no standard target (e.g. OKTE).
        # Return only X so downstream normalization/inference pieces still work.
        features = _resolve_features(payload, data, target_columns=[])
        if keep_datetime and "datetime" in data.columns and "datetime" not in features:
            features = ["datetime"] + features
        X = data[features]
        return {
            "message": "DataPreprocessingPiece executed (prediction, no target column).",
            "artifacts": {
                "X": to_jsonable_df(X),
                "y": {},
                "features": features,
                "target_column": None,
            },
        }

    features = _resolve_features(payload, data, target_columns=[target_col])
    if keep_datetime and "datetime" in data.columns and "datetime" not in features:
        features = ["datetime"] + features

    X = data[features]
    y = data[target_col]

    return {
        "message": "DataPreprocessingPiece executed (prediction).",
        "artifacts": {
            "X": to_jsonable_df(X),
            "y": to_jsonable_df(y.to_frame()),
            "features": features,
            "target_column": target_col,
        },
    }


def preprocess_correction(payload):
    import os
    import pandas as pd  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore

    from .preprocessor_utils import (
        ensure_datetime_column,
        flag_each_day,
        preprocess_solargis_data,
    )
    from .serialization import to_jsonable_df

    df = _read_input_dataframe(payload)
    data_path = payload.get("data_path")
    save_data_path = payload.get("save_data_path")
    flag_each_day_enabled = bool(payload.get("flag_each_day", False))
    test_size = payload.get("test_size")  # optional
    load_all_data = bool(payload.get("load_all_data", False))

    def _load_single(path: str):
        # Special handling for PVOD-derived Solargis-like CSV.
        if os.path.basename(path).startswith("error_correction_pvod"):
            data = pd.read_csv(path)
            data = ensure_datetime_column(data)
            if flag_each_day_enabled:
                data = flag_each_day(data)
            return data

        try:
            data = pd.read_csv(
                path,
                sep=";",
                skiprows=58,
                dayfirst=True,
            )
            data = ensure_datetime_column(data)
        except ValueError:
            data = pd.read_csv(path, sep=None, engine="python", dayfirst=True)
            data = ensure_datetime_column(data)
        if flag_each_day_enabled:
            data = flag_each_day(data)
        return data

    if df is None:
        if not data_path:
            raise ValueError(
                "preprocessing_option='correction' requires either `payload['dataframe']` "
                "or `payload['data_path']`."
            )

        if load_all_data:
            data = pd.DataFrame()
            for file in os.listdir(data_path):
                part = _load_single(os.path.join(data_path, file))
                data = pd.concat([data, part])
        else:
            data = _load_single(data_path)
    else:
        data = df

    data = ensure_datetime_column(data)

    # Separate true sequence one predictions from the rest
    true_sequence_one = data[data["pred_sequence_id"] == 1]
    pred_sequence_one = data[data["pred_sequence_id"] != 1]

    # Insert the true pvout into the pred sequence one dataframe.
    true_pvout_map = true_sequence_one.set_index("datetime")["PVOUT"].to_dict()
    pred_sequence_one = pred_sequence_one.copy()
    pred_sequence_one["true_pvout"] = pred_sequence_one["datetime"].map(true_pvout_map)
    pred_sequence_one = pred_sequence_one.dropna(subset=["true_pvout"])

    # Preprocess data
    pred_sequence_one = preprocess_solargis_data(pred_sequence_one)
    true_sequence_one = preprocess_solargis_data(true_sequence_one)

    if save_data_path:
        # Derive two output paths: *_pred.csv and *_true.csv
        root, ext = os.path.splitext(save_data_path)
        os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
        pred_sequence_one.to_csv(f"{root}_pred{ext}", index=False)
        true_sequence_one.to_csv(f"{root}_true{ext}", index=False)

    features = _resolve_features(
        payload, pred_sequence_one, target_columns=["true_pvout"]
    )
    if "datetime" not in features:
        features.append("datetime")
    if (
        "pred_sequence_id" in pred_sequence_one.columns
        and "pred_sequence_id" not in features
    ):
        features.append("pred_sequence_id")

    X = pred_sequence_one[features]
    y_pred = pred_sequence_one["PVOUT"]
    y_true = pred_sequence_one["true_pvout"]

    if test_size is not None:
        x_train, x_test, y_pred_train, y_pred_test, y_true_train, y_true_test = (
            train_test_split(X, y_pred, y_true, test_size=test_size)
        )
        artifacts = {
            "true_sequence_one": to_jsonable_df(true_sequence_one),
            "x_train": to_jsonable_df(x_train),
            "x_test": to_jsonable_df(x_test),
            "y_pred_train": to_jsonable_df(y_pred_train.to_frame()),
            "y_pred_test": to_jsonable_df(y_pred_test.to_frame()),
            "y_true_train": to_jsonable_df(y_true_train.to_frame()),
            "y_true_test": to_jsonable_df(y_true_test.to_frame()),
        }
    else:
        artifacts = {
            "true_sequence_one": to_jsonable_df(true_sequence_one),
            "X": to_jsonable_df(X),
            "y_pred": to_jsonable_df(y_pred.to_frame()),
            "y_true": to_jsonable_df(y_true.to_frame()),
        }

    return {
        "message": "DataPreprocessingPiece executed (correction).",
        "artifacts": artifacts,
    }
