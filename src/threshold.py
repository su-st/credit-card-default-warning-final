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


def analyze_cost_scenarios(y, oof_score, bill, scenarios):
    """Summarize supplied scenarios using C2 selection, policy and evaluation.

    Loss_i = max(September bill_i, 0) * loss_ratio. Total cost is the sum
    of FN losses + TP losses * (1-rescue_rate) + FINAL list size * contact_cost.
    Pass bill_at(df, 0); choose_threshold evaluates the truncated final list at
    all 91 thresholds (0.05..0.95), choosing the higher threshold on cost ties.
    This is a numerical helper, not authorization to select on held-out data.
    Formal callers use run_cost_analysis, which verifies registered L1 OOF.

    Returns (summary DataFrame, policies in row order). Each policy retains its
    scenario and OOF evidence; persistence binds it to a model/run separately.
    """
    from dataclasses import asdict
    from .protocol import choose_threshold, evaluate

    rows, policies = [], []
    for scenario in scenarios:
        chosen = choose_threshold(y, oof_score, bill, scenario)
        policy = chosen["政策物件"]
        metrics = evaluate(y, score=oof_score, policy=policy, bill=bill,
                           scenario=scenario)["overall"].iloc[0].to_dict()
        rows.append({**asdict(scenario), "門檻": chosen["門檻"], **metrics})
        policies.append(policy)
    return pd.DataFrame(rows), policies


def run_cost_analysis(model_record, *, rules_freeze_id):
    """Run D2 on a registered L1 model's own OOF; never fit or load test data.

    Uses the per-customer cost formula documented in analyze_cost_scenarios.
    Main settings come from load_rules(id).scenario. The frozen JSON must also
    include cost_sensitivity with four nonempty axes: max_list_rate, loss_ratio,
    contact_cost, rescue_rate. Their Cartesian product is evaluated; there is no
    config fallback. This function does not create or change a freeze.

    Reports the bounded cost policy, unbounded cost policy (Scenario cap=1),
    unbounded F1 comparison, and sensitivity policies. F1 uses the preserved
    legacy select_threshold rule (first/lower threshold on F1 ties). Its cost
    and list evaluation still use C2. All rows also show the preserved original
    average-default-loss cost as a comparison, using the same final list.

    Every policy gets a new run/threshold.json, sharing only the source model.
    Results are in outputs/results/<run_id>/; the main run additionally holds
    cost_comparison.csv with all three IDs for every row. Returns that table.
    The report describes scenarios under assumptions, not bank-optimal policy.
    """
    from dataclasses import asdict, replace
    from itertools import product
    from . import protocol as p
    from .registry import IntegrityError, content_hash, file_hash

    rules = p.load_rules(rules_freeze_id)
    registry = p._registry()
    registry.load_model(model_record)
    if model_record["level"] != "L1":
        raise IntegrityError("D2 research selection requires registered L1 OOF, never L1-H/test")
    axes = rules.cost_sensitivity
    keys = ("max_list_rate", "loss_ratio", "contact_cost", "rescue_rate")
    if set(axes) != set(keys) or any(not axes[key] for key in keys):
        raise ValueError("cost_sensitivity requires exactly four nonempty scenario axes")
    scenarios = [p.Scenario(**dict(zip(keys, values)))
                 for values in product(*(axes[key] for key in keys))]
    # Validate every frozen assumption before reading development data or writing runs.
    main = rules.scenario
    if file_hash(p.DATA_DIR / "default_of_credit_card_clients.csv") != model_record["data_sha256"]:
        raise IntegrityError("Current data differs from the registered model")
    train = p.dev_data()
    if content_hash(train.index.tolist()) != model_record["training_ids_hash"]:
        raise IntegrityError("L1 IDs/order differ from registered OOF")
    saved = pd.read_csv(registry.path(model_record["model_path"]).parent / "oof.csv",
                        converters={"oof_score": float})
    if (saved["customer_id"].tolist() != train.index.tolist()
            or not np.array_equal(saved["y"].to_numpy(), train["default"].to_numpy())):
        raise IntegrityError("OOF customer IDs/labels differ from L1 data")
    y = train["default"]
    score = pd.Series(saved["oof_score"].to_numpy(), index=train.index)
    bill = p.bill_at(train, 0)
    all_scenarios = [main, replace(main, max_list_rate=1.0), *scenarios]
    summary, policies = analyze_cost_scenarios(y, score, bill, all_scenarios)
    labels = ["主要成本政策", "無上限成本對照", *(["敏感度情境"] * len(scenarios))]

    # F1 is a reference on the same OOF, without changing its historical selector.
    f1_scenario = replace(main, max_list_rate=1.0)
    candidates = []
    for t in GRID:
        policy = p.Policy.single(float(t), None)
        metrics = p.evaluate(y, score=score, policy=policy)["overall"].iloc[0]
        candidates.append({"門檻": t, "F1-score": metrics["F1"]})
    t = select_threshold(pd.DataFrame(candidates), "f1")
    policy = replace(p.Policy.single(t, None), selection="f1",
                     scenario=asdict(f1_scenario), evidence=policies[0].evidence)
    metrics = p.evaluate(y, score=score, policy=policy, bill=bill,
                         scenario=f1_scenario)["overall"].iloc[0].to_dict()
    summary = pd.concat([summary, pd.DataFrame([{**asdict(f1_scenario), "門檻": t, **metrics}])],
                        ignore_index=True)
    policies.append(policy)
    all_scenarios.append(f1_scenario)
    labels.append("無上限F1對照")
    # Original average-loss formula is unchanged and deliberately not used for selection.
    average_loss = float(bill.loc[y == 1].mean())
    main_run = None
    rows = []
    try:
        for i, (policy, scenario) in enumerate(zip(policies, all_scenarios)):
            run = p.start_run("L1", f"D2 {labels[i]}: scenario assumptions, not final performance",
                              model_id=model_record["model_id"], rules_freeze_id=rules_freeze_id)
            if i == 0:
                main_run = run
            try:
                run.bind_model(model_record)
                policy = run.record_policy(policy)
                pred, _ = p.apply_policy(score, policy)
                row = summary.loc[i].to_dict()
                row.update(比較=labels[i], model_id=model_record["model_id"],
                           policy_id=policy.policy_id, run_id=run.run_id,
                           平均損失對照總成本=total_cost(confusion(y, pred, .5), average_loss,
                                                      scenario.contact_cost, scenario.rescue_rate))
                rows.append(row)
                run.save_csv("cost_summary", pd.DataFrame([row]))
                if i != 0:
                    run.finish()
            except Exception as exc:
                run.fail(f"{type(exc).__name__}: {exc}")
                raise
        summary = pd.DataFrame(rows)
        main_run.save_csv("cost_comparison", summary)
        main_run.finish()
    except Exception as exc:
        if main_run is not None and main_run.meta["status"] == "running":
            main_run.fail(f"{type(exc).__name__}: {exc}")
        raise
    return summary
