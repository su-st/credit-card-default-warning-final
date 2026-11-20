"""T-陳4: SHAP analysis of an explicitly selected registered baseline.

Run from the repository root, for example:
  python experiments/04_shap.py --mode global --source-run-id RUN --sensitive false
  python experiments/04_shap.py --mode cases --source-run-id DEV_RUN --sensitive false

Global importance uses L1. Case illustrations use the separate L1-H model.
This script does not train models, select policies, or evaluate test data.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import protocol as p
from src.explain_shap import global_importance, shap_values
from src.registry import IntegrityError, Registry, content_hash, file_hash, identifier, read_json


def load_source(registry, source_run_id, *, mode, sensitive):
    """Reject the wrong layer before the registry loads any model/predictions."""
    expected_level = "L1" if mode == "global" else "L1-H"
    meta = read_json(registry.runs / identifier(source_run_id) / "run.json")
    if meta.get("level") != expected_level or meta.get("status") != "complete":
        raise IntegrityError(f"{mode} requires a completed {expected_level} source run")
    cfg = meta.get("configuration", {})
    if (cfg.get("name") != "XGBoost" or cfg.get("lag") != 0
            or cfg.get("engineered") is not False
            or cfg.get("sensitive") is not sensitive):
        raise IntegrityError("Source must match the explicitly selected baseline features")
    if not meta.get("rules_freeze_id"):
        raise IntegrityError("Source is missing its approved rules identity")
    source = registry.load_run(source_run_id)
    if source["run"] != meta:
        raise IntegrityError("Source metadata changed while loading")
    return source


def prepare_source(source, *, mode):
    """Use protocol data and verify it against the registered training evidence."""
    meta, record = source["run"], source["model_record"]
    cfg = meta["configuration"]
    if file_hash(p.DATA_DIR / "default_of_credit_card_clients.csv") != record["data_sha256"]:
        raise IntegrityError("Local data differs from the source model's training data")
    if mode == "global":
        training = p.dev_data()
        evaluation = training
        expected_count = 24000
        if training.attrs.get("level") != "L1":
            raise IntegrityError("Global data must be L1")
    else:
        training, evaluation = p.dev_holdout()
        expected_count = 19200
        if (training.attrs.get("level") != "L1-H"
                or evaluation.attrs.get("level") != "L1-H"
                or training.attrs.get("role") != "train"
                or evaluation.attrs.get("role") != "holdout"
                or len(evaluation) != 4800
                or len(training.index.intersection(evaluation.index))):
            raise IntegrityError("Cases require the separate L1-H training/holdout split")
        if (content_hash(evaluation.index.tolist()) != record["prediction_ids_hash"]
                or content_hash(sorted(evaluation.index.tolist())) != record["prediction_ids_set_hash"]):
            raise IntegrityError("Holdout IDs differ from the source model's declared holdout")
    if (len(training) != expected_count or record["training_count"] != expected_count
            or content_hash(training.index.tolist()) != record["training_ids_hash"]):
        raise IntegrityError("Training IDs differ from the registered source model")
    X, columns = p.prepare(evaluation, lag=cfg["lag"],
                           sensitive=cfg["sensitive"], engineered=cfg["engineered"])
    if columns != record["columns"] or X.columns.tolist() != columns:
        raise IntegrityError("Feature columns/order differ from the registered source model")
    return X


def explain_frames(model, X):
    """Shared signed merge, followed by mean absolute values for global ranking."""
    merged = shap_values(model, X, merge_onehot=True)
    unmerged = shap_values(model, X, merge_onehot=False)
    if (not merged.index.equals(X.index) or not unmerged.index.equals(X.index)
            or set(merged.columns) != set(X.columns)
            or not merged.columns.is_unique
            or not np.isfinite(merged.to_numpy()).all()
            or not np.isfinite(unmerged.to_numpy()).all()
            or not np.allclose(merged.sum(axis=1), unmerged.sum(axis=1), rtol=1e-6, atol=1e-6)):
        raise IntegrityError("SHAP merge failed its alignment/additivity checks")
    return merged, unmerged


def importance_figure(importance):
    import matplotlib.pyplot as plt

    shown = importance.head(20).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, .32 * len(shown) + 1.8)))
    ax.barh(shown.index, shown.to_numpy(), color="#2563a6")
    ax.set_xlabel("Mean absolute SHAP value (raw margin / log-odds)")
    ax.set_title("L1 global model importance | signed One-Hot merge")
    fig.text(.02, .015, "Model contributions describe predictions; they are not causes of default.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    return fig


def case_figure(values, *, base, margin, risk_score, number):
    import matplotlib.pyplot as plt

    ordered = values.reindex(values.abs().sort_values(ascending=False).index)
    shown = ordered.iloc[:10].copy()
    if len(ordered) > 10:
        shown.loc["Other features (signed sum)"] = ordered.iloc[10:].sum()
    starts = base + np.r_[0., np.cumsum(shown.to_numpy()[:-1])]
    ends = starts + shown.to_numpy()
    fig, ax = plt.subplots(figsize=(10, max(5, .4 * len(shown) + 2)))
    ypos = np.arange(len(shown))
    ax.barh(ypos, np.abs(shown.to_numpy()), left=np.minimum(starts, ends),
            color=["#c43b42" if value >= 0 else "#2877af" for value in shown])
    ax.set_yticks(ypos, [f"{name} ({value:+.4f})" for name, value in shown.items()])
    ax.invert_yaxis()
    ax.axvline(base, color="#777777", linestyle="--", label=f"Base margin {base:.4f}")
    ax.axvline(margin, color="#222222", linestyle=":", label=f"Predicted margin {margin:.4f}")
    ax.set_xlabel("Cumulative signed SHAP contribution (raw margin / log-odds)")
    ax.set_title(f"Development example {number} | L1-H | risk score {risk_score:.4f}")
    ax.legend(loc="best", fontsize=8)
    fig.text(.02, .015, "L1-H is for illustration only. Risk scores are not calibrated probabilities; factors are not causes.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    return fig


def select_cases(X, case_ids=None):
    ids = sorted(X.index.tolist())[:3] if case_ids is None else list(case_ids)
    if len(ids) not in (2, 3) or len(set(ids)) != len(ids) or not set(ids) <= set(X.index):
        raise ValueError("Select two or three distinct IDs from the L1-H holdout only")
    return X.loc[ids]


def analyse(source_run_id, *, mode, sensitive, case_ids=None):
    if mode not in {"global", "cases"}:
        raise ValueError("mode must be global or cases")
    if not isinstance(sensitive, bool):
        raise ValueError("sensitive must explicitly be True or False")
    if mode == "global" and case_ids is not None:
        raise ValueError("Case IDs may only be provided in cases mode")
    registry = Registry()
    source = load_source(registry, source_run_id, mode=mode, sensitive=sensitive)
    X = prepare_source(source, mode=mode)
    if mode == "cases":
        X = select_cases(X, case_ids)
    # Explain before starting a run so input/helper errors do not leave empty runs.
    merged, unmerged = explain_frames(source["model"], X)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta, record = source["run"], source["model_record"]
    run = p.start_run(meta["level"], f"T-陳4 SHAP {mode} from registered baseline",
                      model_id=meta["model_id"], rules_freeze_id=meta["rules_freeze_id"])
    try:
        run.bind_model(record)
        run.update(source_run_id=source_run_id, source_model_hash=record["model_hash"],
                   source_git_commit=meta.get("git_commit"),
                   configuration=meta["configuration"], analysis="T-陳4 SHAP", shap_mode=mode,
                   shap_scale="XGBoost raw margin / log-odds for default=1",
                   aggregation="signed sum per original feature, then mean absolute value",
                   n_explained=len(X), descriptive_only=True)
        if mode == "global":
            importance = global_importance(merged)
            for name, series in (("shap_global", importance),
                                 ("shap_global_unmerged", global_importance(unmerged))):
                table = series.rename_axis("feature").reset_index()
                table["source_run_id"] = source_run_id
                table["source_level"] = "L1"
                table["n_explained"] = len(X)
                table["shap_scale"] = "raw margin / log-odds"
                run.save_csv(name, table)
            fig = importance_figure(importance)
            try:
                run.save_fig("shap_global", fig)
            finally:
                plt.close(fig)
        else:
            import xgboost as xgb

            model = source["model"]
            transformed = model.named_steps["pre"].transform(X)
            booster = model.named_steps["clf"].get_booster()
            matrix = xgb.DMatrix(transformed)
            margins = booster.predict(matrix, output_margin=True)
            base = booster.predict(matrix, pred_contribs=True)[:, -1]
            scores = model.predict_proba(X)[:, 1]
            if not np.allclose(base + merged.sum(axis=1).to_numpy(), margins, rtol=1e-5, atol=1e-5):
                raise IntegrityError("Case SHAP values do not reconstruct the model raw margin")
            run.update(case_customer_ids=X.index.tolist(),
                       case_selection="user supplied IDs" if case_ids is not None else "three smallest holdout IDs; no score/label selection",
                       not_for_performance_or_model_selection=True)
            for number, (_, values) in enumerate(merged.iterrows(), start=1):
                i = number - 1
                fig = case_figure(values, base=float(base[i]), margin=float(margins[i]),
                                  risk_score=float(scores[i]), number=number)
                try:
                    run.save_fig(f"shap_case_{number}", fig)
                finally:
                    plt.close(fig)
        run.finish()
    except Exception as exc:
        if run.meta["status"] == "running":
            run.fail(f"{type(exc).__name__}: {exc}")
        raise
    print(f"Completed SHAP {mode}: run_id={run.run_id}", flush=True)
    print(f"Source run: {source_run_id}; sensitive={sensitive}; level={meta['level']}", flush=True)
    print(f"Results: {run.result_path}", flush=True)
    return run.run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("global", "cases"), required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--sensitive", choices=("true", "false"), required=True,
                        help="Must match the leader-confirmed model and the source run")
    parser.add_argument("--case-ids", nargs="+", type=int,
                        help="Optional two or three holdout IDs; default is the three smallest IDs")
    args = parser.parse_args(argv)
    return analyse(args.source_run_id, mode=args.mode,
                   sensitive=args.sensitive == "true", case_ids=args.case_ids)


if __name__ == "__main__":
    main()
