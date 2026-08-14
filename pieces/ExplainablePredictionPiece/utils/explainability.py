from __future__ import annotations

from typing import Any, Optional, Tuple, Union


def _get_predict_fn(model: Any, feature_names: list, is_classification: bool = False):
    """Return a predict function that takes a 2D array and returns 1D predictions."""
    import numpy as np
    import pandas as pd

    if hasattr(model, "model"):
        # Wrapper around PredictionModel (has .model and .predict(DataFrame))
        raw = model.model

        def predict_fn(x: np.ndarray) -> np.ndarray:
            x2 = x.reshape(1, -1) if x.ndim == 1 else x
            df = pd.DataFrame(x2, columns=feature_names)
            out = model.predict(df)
            return np.asarray(out).ravel()

        return predict_fn

    def predict_fn(x: np.ndarray) -> np.ndarray:
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        out = model.predict(x2)
        return np.asarray(out).ravel()

    return predict_fn


def _parse_data(
    data: Union["Any", Tuple, dict],
) -> Tuple["Any", Optional["Any"], list]:
    """
    Return (X, y, feature_names) where X is a 2D numpy array.
    """
    import numpy as np
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        X = data.values
        feature_names = list(data.columns)
        return X, None, feature_names

    if isinstance(data, (list, tuple)) and len(data) >= 1:
        X = np.asarray(data[0])
        y = np.asarray(data[1]) if len(data) > 1 else None
        feature_names = (
            list(data[2]) if len(data) > 2 else [str(i) for i in range(X.shape[1])]
        )
        return X, y, feature_names

    if isinstance(data, dict):
        X = np.asarray(data["X"])
        y = np.asarray(data["y"]) if "y" in data else None
        feature_names = data.get("feature_names", [str(i) for i in range(X.shape[1])])
        return X, y, feature_names

    raise TypeError(
        "data must be DataFrame, (X, y) tuple, or dict with 'X' and optional 'y'"
    )


def _is_tree_model(model: Any) -> bool:
    """Heuristic: True if model is tree-based (XGBoost, sklearn tree, etc.)."""
    if model is None:
        return False
    name = type(model).__name__
    if "XGB" in name or "XGBRegressor" in name or "XGBClassifier" in name:
        return True
    module = getattr(type(model), "__module__", "") or ""
    if "xgboost" in module or "sklearn.ensemble" in module or "sklearn.tree" in module:
        return True
    return "Tree" in name or "Forest" in name or "GradientBoosting" in name


class ExplainableModule:
    def __init__(self, mode: str = "regression"):
        """
        mode: 'regression' or 'classification'
        """

        self.mode = mode

    def explain(self, model: Any, data: Any, method: str = "shap", **kwargs) -> Any:
        """Dispatch to LIME or SHAP. method in ('lime', 'shap')."""
        m = str(method).lower()
        if m == "lime":
            return self.lime(model, data, **kwargs)
        if m == "shap":
            return self.shap(model, data, **kwargs)
        raise ValueError("method must be 'lime' or 'shap'")

    def lime(
        self,
        model: Any,
        data: Any,
        num_samples: int = 5000,
        num_features: int = 10,
        num_explanations: int = 5,
        instance_idx: Optional[int] = None,
        **kwargs,
    ) -> dict:
        import numpy as np

        try:
            from lime import lime_tabular  # type: ignore
        except ImportError as e:
            raise ImportError("Install lime: pip install lime") from e

        X, _, feature_names = _parse_data(data)
        X = X.reshape(1, -1) if X.ndim == 1 else X

        predict_fn = _get_predict_fn(
            model, feature_names, is_classification=(self.mode == "classification")
        )

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X,
            feature_names=feature_names,
            mode=self.mode,
            **kwargs,
        )

        if instance_idx is not None:
            exp = explainer.explain_instance(
                X[int(instance_idx)],
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )
            return {
                "method": "lime",
                "instance_idx": int(instance_idx),
                "explanation": exp.as_list(),
            }

        n = min(int(num_explanations), len(X))
        results = []
        for i in range(n):
            exp = explainer.explain_instance(
                X[i],
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )
            results.append({"instance_idx": i, "explanation": exp.as_list()})

        return {
            "method": "lime",
            "num_explanations": n,
            "explanations": results,
            "feature_names": feature_names,
        }

    def shap(
        self,
        model: Any,
        data: Any,
        background_size: Optional[int] = 100,
        max_evals: Optional[int] = 500,
        tree_fallback: bool = True,
        **kwargs,
    ) -> dict:
        import numpy as np

        try:
            import shap  # type: ignore
        except ImportError as e:
            raise ImportError("Install shap: pip install shap") from e

        X, _, feature_names = _parse_data(data)
        X = X.reshape(1, -1) if X.ndim == 1 else X

        # Resolve underlying model for tree explainer
        raw_model = model.model if hasattr(model, "model") else model

        use_tree = bool(tree_fallback) and _is_tree_model(raw_model)
        if use_tree:
            try:
                explainer = shap.TreeExplainer(
                    raw_model,
                    X if background_size is None else X[: int(background_size)],
                    **kwargs,
                )
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                base_value = getattr(explainer, "expected_value", None)
                return {
                    "method": "shap",
                    "explainer_type": "TreeExplainer",
                    "shap_values": np.asarray(shap_values).tolist(),
                    "feature_names": feature_names,
                    "base_value": (
                        base_value.tolist()
                        if hasattr(base_value, "tolist")
                        else base_value
                    ),
                }
            except Exception:
                # Fall back to KernelExplainer
                tree_error = "TreeExplainer failed; falling back to KernelExplainer"
        else:
            tree_error = None

        predict_fn = _get_predict_fn(model, feature_names)
        if background_size is None or int(background_size) >= len(X):
            background = X
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(
                len(X), size=min(int(background_size), len(X)), replace=False
            )
            background = X[idx]

        explainer = shap.KernelExplainer(predict_fn, background, **kwargs)
        shap_values = explainer.shap_values(X, nsamples=max_evals or 500)
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        return {
            "method": "shap",
            "explainer_type": "KernelExplainer",
            "shap_values": np.asarray(shap_values).tolist(),
            "feature_names": feature_names,
            "base_value": getattr(explainer, "expected_value", None),
            "tree_error": tree_error,
        }


def run_explainability(
    model: Any, data: Any, method: str, mode: str, cfg: dict
) -> dict:
    """
    Convenience wrapper so the piece can call a single function.
    """
    m = str(method).lower()
    explain_cfg = cfg or {}
    module = ExplainableModule(mode=mode)

    if m == "lime":
        return module.lime(
            model,
            data,
            num_samples=int(explain_cfg.get("num_samples", 5000)),
            num_features=int(explain_cfg.get("num_features", 10)),
            num_explanations=int(explain_cfg.get("num_explanations", 5)),
            instance_idx=explain_cfg.get("instance_idx"),
            **(explain_cfg.get("lime_kwargs") or {}),
        )

    if m == "shap":
        return module.shap(
            model,
            data,
            background_size=explain_cfg.get("background_size", 100),
            max_evals=explain_cfg.get("max_evals", 500),
            tree_fallback=bool(explain_cfg.get("tree_fallback", True)),
            **(explain_cfg.get("shap_kwargs") or {}),
        )

    raise ValueError("method must be 'lime' or 'shap'")
