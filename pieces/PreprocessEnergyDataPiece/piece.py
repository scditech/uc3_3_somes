
try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
import json
import sys
import pandas as pd
import traceback


class PreprocessEnergyDataPiece(BasePiece):
    """
    Prepare training data only (train_dataset.parquet).
    The prediction input for PredictPiece is a separate CSV and is not generated here.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        log_path = Path(self.results_path) / "preprocess_energy_data.log"
        err_path = Path(self.results_path) / "preprocess_energy_data_error.txt"
        try:
            print("[INFO] PreprocessEnergyDataPiece started")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[INFO] PreprocessEnergyDataPiece started\n")

            input_path = Path(input_data.input_path)
            generate_predict = getattr(input_data, "generate_predict_dataset", False)

            print(f"[INFO] Using input file: {input_path}")
            print(f"[INFO] generate_predict_dataset: {generate_predict}")

            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            df = pd.read_parquet(input_path)

            if "datetime" not in df.columns:
                raise ValueError(f"Input must contain datetime column. Found: {df.columns}")

            repo_root = Path(__file__).resolve().parents[2]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from pieces.common_somes.quality import (
                build_quality_bundle,
                missing_data_frame,
                quality_indicator_frame,
            )

            profiled = {}
            if "department_id" in df.columns:
                for dept, group in df.groupby("department_id"):
                    profiled[f"load_{dept}"] = group.drop(columns=["department_id"])
            else:
                profiled["merged_energy"] = df
            quality = build_quality_bundle(profiled)

            df["datetime"] = pd.to_datetime(df["datetime"])
            rows_before = len(df)
            if "department_id" in df.columns:
                df = df.drop_duplicates(subset=["department_id", "datetime"])
            else:
                df = df.drop_duplicates(subset=["datetime"])
            dropped_duplicates = rows_before - len(df)
            df = df.sort_values("datetime")
            df = df.set_index("datetime")

            if "department_id" in df.columns:
                parts = []
                for dept, g in df.groupby("department_id"):
                    g2 = g.drop(columns=["department_id"]).resample("15min").mean().ffill()
                    g2["department_id"] = dept
                    parts.append(g2.reset_index())
                train_df = pd.concat(parts, ignore_index=True).sort_values(["department_id", "datetime"]).reset_index(drop=True)
            else:
                df_15min = df.resample("15min").mean().ffill()
                train_df = df_15min.reset_index()
            train_df.rename(columns={"index": "datetime"}, inplace=True)

            train_path = Path(self.results_path) / "train_dataset.parquet"
            train_df.to_parquet(train_path, index=False)

            quality["corrections_applied"] = {
                "dropped_duplicates": int(dropped_duplicates),
                "resampled_to": "15min",
                "rows_in": int(rows_before),
                "rows_out": int(len(train_df)),
                "inserted_steps": int(max(0, len(train_df) - len(df))),
                "imputation": "resample.mean + ffill",
            }
            out_dir = Path(self.results_path)
            quality_path = out_dir / "data_quality_report.json"
            quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
            indicators_path = out_dir / "data_quality_indicators.csv"
            quality_indicator_frame(quality).to_csv(indicators_path, index=False)
            missing_path = out_dir / "missing_data_report.csv"
            missing_data_frame(quality).to_csv(missing_path, index=False)

            print("[SUCCESS] Preprocessing finished (train_dataset only)")
            print(f"[INFO] Train rows: {len(train_df)}")
            print(f"[INFO] Data quality severity: {quality['overall_severity']}")

            predict_path_str = ""
            if generate_predict:
                print("[WARN] generate_predict_dataset=True is deprecated here; use separate CSV for PredictPiece.")

            self.display_result = {"file_type": "parquet", "file_path": str(train_path)}

            return OutputModel(
                message=f"Preprocessing finished (quality: {quality['overall_severity']})",
                train_file_path=str(train_path),
                predict_file_path=predict_path_str,
                data_quality_report_json=str(quality_path),
                data_quality_indicators_csv=str(indicators_path),
                missing_data_report_csv=str(missing_path),
                data_quality_severity=str(quality["overall_severity"]),
            )
        except Exception:
            err = traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[ERROR] PreprocessEnergyDataPiece failed\n")
                f.write(err + "\n")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err)
            raise
