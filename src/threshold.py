"""門檻自動選擇。門檻只用訓練集的交叉驗證預測（驗證資料）決定，測試集只用來報告最後結果。"""
import numpy as np
import pandas as pd

from .config import LIST_THRESHOLD
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def confusion(y_true, prob, t):
    y = np.asarray(y_true); pred = np.asarray(prob) >= t
    return dict(TP=int((pred & (y == 1)).sum()), FP=int((pred & (y == 0)).sum()),
                FN=int((~pred & (y == 1)).sum()), TN=int((~pred & (y == 0)).sum()))


def total_cost(c, loss, contact, rescue):
    """總成本 = 漏判損失 + 聯絡了但沒救回的損失 + 聯絡成本。"""
    return c["FN"] * loss + c["TP"] * (1 - rescue) * loss + (c["TP"] + c["FP"]) * contact


def threshold_table(y_true, prob, loss, contact, rescue) -> pd.DataFrame:
    """每個門檻的名單人數、Precision、Recall、F1、誤報人數與總成本。"""
    rows = []
    for t in GRID:
        c = confusion(y_true, prob, t); n = c["TP"] + c["FP"]
        prec = c["TP"] / n if n else 0.0
        rec = c["TP"] / (c["TP"] + c["FN"])
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"門檻": t, "名單人數": n, "Precision": prec, "Recall": rec, "F1-score": f1,
                     "誤報人數": c["FP"], "漏判人數": c["FN"], "總成本": total_cost(c, loss, contact, rescue)})
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame, rule: str, top_n_rate: float = 0.2, n_total: int | None = None,
                     max_list_rate: float | None = None) -> float:
    """依規則從表中選出門檻。cost 規則可加上人力上限 max_list_rate。"""
    if rule == "cost":
        ok = table if max_list_rate is None else table[table["名單人數"] <= max_list_rate * n_total]
        return float(ok.loc[ok["總成本"].idxmin(), "門檻"])
    if rule == "f1":
        return float(table.loc[table["F1-score"].idxmax(), "門檻"])
    if rule == "top_n":
        target = top_n_rate * n_total
        return float(table.loc[(table["名單人數"] - target).abs().idxmin(), "門檻"])
    raise ValueError(f"未知的規則：{rule}")


def policy_list(score: pd.Series, record: dict) -> pd.Series:
    """C1 single-threshold export policy; C2 owns the general Policy interface."""
    from .registry import validate_ids

    if "protocol_policy" in record:
        from .protocol import Policy, apply_policy
        return apply_policy(score, Policy(**record["protocol_policy"]))[0]

    validate_ids(score.index)
    values = score.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid risk scores")
    policy = record["policy"]
    if policy["kind"] != "single":
        raise ValueError("C1 export supports only a single-threshold policy")
    threshold, cap = policy["threshold"], policy["max_list_rate"]
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    if cap is not None and (not np.isfinite(cap) or not 0 <= cap <= 1):
        raise ValueError("Invalid list capacity")
    selected = score >= threshold
    if cap is not None and selected.sum() > int(np.floor(cap * len(score))):
        ranked = pd.DataFrame({"score": score, "customer_id": score.index})
        ranked.index.name = None
        keep = ranked.loc[selected].sort_values(["score", "customer_id"], ascending=[False, True]).head(int(np.floor(cap * len(score))))
        selected = pd.Series(score.index.isin(keep.index), index=score.index)
    return selected.astype(int)


def validate_threshold(info, model_record, *, level, registry):
    """Check source-level separation and recompute every training-evidence hash."""
    from .registry import IntegrityError, content_hash

    registry.load_model(model_record)  # checks actual OOF/fold files, not just JSON
    for key in ("model_id", "model_hash", "params_hash", "features_hash", "training_ids_hash",
                "seed", "n_splits", "folds_sha256", "oof_sha256"):
        if info[key] != model_record[key]:
            raise IntegrityError(f"Threshold {key} mismatch")
    if info["source_level"] != model_record["level"] or (
            (level == "L1-H") != (info["source_level"] == "L1-H")):
        raise IntegrityError("Development holdout policy cannot be mixed with a formal model")
    expected = content_hash({"policy": info["policy"], "source": {k: info[k] for k in (
        "model_id", "model_hash", "params_hash", "features_hash", "training_ids_hash",
        "seed", "n_splits", "folds_sha256", "oof_sha256", "rules_freeze_id", "source_level")}})
    if info["policy_hash"] != expected or info["policy_id"] != f"single-{expected[:16]}":
        raise IntegrityError("Policy hash/identity mismatch")
    if info["rules_freeze_id"] is not None:
        registry.load_rules(info["rules_freeze_id"])
    if info["threshold"] != info["policy"]["threshold"] or info["scenario"] != info["policy"]["scenario"]:
        raise IntegrityError("Threshold policy content mismatch")
    # Also validate numerical ranges without making any policy choice.
    policy_list(pd.Series([0.0], index=[1]), info)


def save_threshold(info: dict, *, run, model_record):
    """Bind the supplied selection to its own run; never writes a global file.

    Selection rules/calculations above remain unchanged for L0 and the existing
    02 analysis. Official cost selection belongs to C3 and frozen rules.
    """
    from .registry import IntegrityError, content_hash, write_json

    run._running()
    if run.meta["model_id"] != model_record["model_id"]:
        raise IntegrityError("Threshold model differs from run")
    source = {k: model_record[k] for k in (
        "model_id", "model_hash", "params_hash", "features_hash", "training_ids_hash",
        "seed", "n_splits", "folds_sha256", "oof_sha256")}
    source.update(rules_freeze_id=run.meta["rules_freeze_id"], source_level=model_record["level"])
    policy = dict(kind="single", threshold=float(info["threshold"]),
                  max_list_rate=info.get("max_list_rate"), scenario=info.get("assumptions"),
                  rule=info["rule"], selected_on=info["selected_on"])
    digest = content_hash({"policy": policy, "source": source})
    record = dict(info, **source, run_id=run.run_id, policy=policy,
                  policy_id=f"single-{digest[:16]}", policy_hash=digest, scenario=policy["scenario"])
    validate_threshold(record, model_record, level=run.meta["level"], registry=run.registry)
    write_json(run.path / "threshold.json", record)
    run.update(policy_id=record["policy_id"], policy_hash=digest, scenario=record["scenario"])
    return record


def load_threshold(run_id=None, *, registry=None, allow_fallback=False) -> tuple[float, str]:
    """Require a completed coherent run. Fallback needs explicit opt-in.

    A corrupt existing run always raises, even with allow_fallback=True.
    """
    from .registry import Registry

    if run_id is None:
        if allow_fallback:
            return LIST_THRESHOLD, "未使用正式門檻（明確啟用 config 備用值）"
        raise ValueError("run_id is required; global threshold.json is no longer accepted")
    bundle = (registry or Registry()).load_run(run_id)
    info = bundle["threshold"]
    return float(info["threshold"]), f"{info['source_level']} OOF（run_id={run_id}，規則：{info['rule']}）"
