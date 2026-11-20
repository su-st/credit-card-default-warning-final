"""T-陳2: compare numeric PAY with the frozen categorical method on L1 only.

UCI does not define codes 0 or -2; their meanings are not assumed here.
The mapping and fold-fitted One-Hot are provided by src.frozen_methods.
Run from the repository root. Model artifacts/OOF remain local under outputs/.
"""
import argparse
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import frozen_methods as fm, protocol as p
from src.data import age_group
from src.model_freeze import configurations, features, model_factory, engineering_check, rank_candidates
from src.registry import IntegrityError, content_hash, file_hash
from src.training_time import execution_environment, timed_call


def selection_data(rules_id):
    rules = p.load_rules(rules_id)
    df = p.dev_data()
    proof = rules.pay_categorical["l1_verification"]
    if (df.attrs.get("level") != "L1"
            or len(df) != rules.data_protocol["selection"]["train_rows"]
            or content_hash(df.index.tolist()) != proof["training_ids_hash"]
            or file_hash(p.DATA_DIR / "default_of_credit_card_clients.csv") != proof["data_sha256"]):
        raise IntegrityError("Expected the frozen seed-42 L1 customers and data file")
    return df, rules


def groups_of(df, rules_id):
    groups = {"性別": df["SEX"].map({1: "男性", 2: "女性"}),
              "年齡層": age_group(df["AGE"]).astype(str),
              "有無延遲": fm.delay_partition(df, rules_freeze_id=rules_id)}
    if any(g.isna().any() for g in groups.values()):
        raise IntegrityError("Incomplete group labels")
    return groups


def fail(run, exc):
    if run.meta["status"] == "running":
        run.fail(f"{type(exc).__name__}: {exc}")


def build_method(df, rules, rules_id, key, token, reference=None):
    config = configurations(rules)[key]
    X, _ = features(df, config, rules_id)
    factory = lambda: model_factory(config, rules_id)
    run = p.start_run("L1", TASK + " model source: " + key,
                      model_id=f"xgb-{key.lower().replace(' ', '-')}-{token}",
                      rules_freeze_id=rules_id)
    try:
        cv_args = dict(folds=rules.cv["n_splits"], seed=rules.cv["random_state"])
        print(f"{TASK}: {key} CV AUC / OOF / full L1 fit", flush=True)
        cv, cv_seconds = timed_call(p.cv_evaluate, factory, X, df["default"], **cv_args)
        score, oof_seconds = timed_call(p.oof_predict, factory, X, df["default"], **cv_args)
        score.attrs["level"] = "L1"
        if reference is not None:
            if (not X.index.equals(reference["X"].index)
                    or not score.attrs["fold_assignment"].equals(reference["oof"].attrs["fold_assignment"])):
                raise IntegrityError("Models do not use identical ordered L1 customers and folds")
        model, fit_seconds = timed_call(factory().fit, X, df["default"])
        record = run.attach_model(model, X, df["default"], score, seed=cv_args["seed"])
        if reference is not None and record["folds_sha256"] != reference["record"]["folds_sha256"]:
            raise IntegrityError("Persisted fold evidence differs")
        run.update(configuration=config, method=key, analysis=TASK, evaluation_kind="score_only",
                   cv_auc_mean=float(cv["平均"].iloc[0]), cv_auc_std=float(cv["標準差"].iloc[0]),
                   cv_std_ddof=0, selection_evidence_only=True,
                   training_timing=dict(fit_seconds=fit_seconds, oof_seconds=oof_seconds,
                                        cv_evaluate_seconds=cv_seconds, environment=execution_environment()),
                   limitation="L1 OOF selection analysis; policies selected on these same labels are not independent validation")
        run.save_csv("cv_auc", cv)
        run.finish()
        return dict(key=key, run=run, record=record, configuration=config, model=model,
                    X=X, oof=score, cv=cv, policies={})
    except Exception as exc:
        fail(run, exc)
        raise


def tag_table(frame, run, bundle, policy, comparison, stage="final"):
    out = frame.copy()
    values = dict(run_id=run.run_id, model_id=bundle["record"]["model_id"],
                  model_hash=bundle["record"]["model_hash"], policy_id=policy.policy_id,
                  source_run_id=bundle["run"].run_id, method=bundle["key"],
                  comparison=comparison, list_stage=stage, level="L1",
                  rules_freeze_id=run.meta["rules_freeze_id"],
                  threshold_source=f"outputs/runs/{run.run_id}/threshold.json")
    for key, value in values.items():
        out[key] = value
    return out


def policy_result(df, rules, rules_id, bundle, comparison, policy, partition=None, method=None):
    # A policy owns its own run and threshold.json, bound to the actual model/OOF.
    run = p.start_run("L1", TASK + " policy comparison: " + (method or bundle["key"]) + " / " + comparison,
                      model_id=bundle["record"]["model_id"], rules_freeze_id=rules_id)
    try:
        run.bind_model(bundle["record"])
        policy = run.record_policy(policy)
        run.update(configuration=bundle["configuration"], analysis=TASK,
                   method=method or bundle["key"], comparison=comparison,
                   model_source_run_id=bundle["run"].run_id, evaluation_kind="policy",
                   selection_evidence_only=True)
        groups = groups_of(df, rules_id)
        out = p.evaluate(df["default"], score=bundle["oof"], policy=policy, partition=partition,
                         groups=groups, bill=p.bill_at(df, 0), scenario=rules.scenario)
        row = out["overall"].copy()
        gaps = out["gaps"].set_index("分組方式")
        row["gender_eo_gap"] = gaps.loc["性別", "EO 差距"]
        row["age_eo_gap"] = gaps.loc["年齡層", "EO 差距"]
        no_delay = out["groups"].loc[(out["groups"]["分組方式"] == "有無延遲")
                                      & (out["groups"]["族群"] == "無延遲")]
        row["no_delay_default_recall"] = no_delay["Recall"].iloc[0] if len(no_delay) else np.nan
        row["cv_auc_mean"] = float(bundle["cv"]["平均"].iloc[0])
        row["cv_auc_std"] = float(bundle["cv"]["標準差"].iloc[0])
        row["threshold"] = policy.threshold if policy.kind == "single" else np.nan
        row["input_feature_count"] = len(bundle["X"].columns)
        row["folds_sha256"] = bundle["record"]["folds_sha256"]
        row = tag_table(row, run, bundle, policy, comparison)
        row["method"] = method or bundle["key"]
        for name, table in out.items():
            tagged = tag_table(table, run, bundle, policy, comparison)
            tagged["method"] = method or bundle["key"]
            run.save_csv(name, tagged)
        run.save_csv("comparison_row", row)
        pred, info = p.apply_policy(bundle["oof"], policy, partition=partition)
        bundle["policies"][comparison] = dict(run_id=run.run_id, policy=policy, pred=pred, row=row)
        run.finish()
        return row, run
    except Exception as exc:
        fail(run, exc)
        raise


def standard_comparisons(df, rules, rules_id, bundle):
    formal = p.choose_threshold(df["default"], bundle["oof"], p.bill_at(df, 0), rules.scenario)["政策物件"]
    rows = []
    for key, policy in [("single_0.5", p.Policy.single(0.5, None)),
                        ("top_20pct", p.Policy.top_n(rules.g2b["policy"]["rate"])),
                        ("formal_cost", formal)]:
        row, _ = policy_result(df, rules, rules_id, bundle, key, policy)
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def replacement_evidence(bundles, rules, rules_id, df):
    base = bundles["M1-NS"]["policies"]["top_20pct"]["row"].iloc[0]
    rows = []
    for key, b in bundles.items():
        if key == "M1-NS":
            continue
        row = b["policies"]["top_20pct"]["row"].iloc[0]
        check = engineering_check(b, df, rules_id)
        rows.append(dict(candidate=key, model_id=b["record"]["model_id"],
                         source_run_id=row["run_id"], policy_id=row["policy_id"],
                         baseline_model_id=base["model_id"], baseline_run_id=base["run_id"],
                         recall_gain=row["Recall"] - base["Recall"], auc_gain=row["AUC"] - base["AUC"],
                         gender_eo_increase=row["gender_eo_gap"] - base["gender_eo_gap"],
                         age_eo_increase=row["age_eo_gap"] - base["age_eo_gap"],
                         input_feature_count_before_onehot=len(b["X"].columns), **check))
    result = rank_candidates(pd.DataFrame(rows), rules)
    result["decision_scope"] = "L1 evidence only; no model freeze or platform replacement"
    return result


def comparison_figure(table, run_id):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    cv = table.drop_duplicates("method").set_index("method")
    axes[0].bar(cv.index, cv["cv_auc_mean"], yerr=cv["cv_auc_std"], capsize=4)
    axes[0].set(ylabel="CV AUC (mean +/- population SD)", ylim=(0, 1), title="L1, shared five folds")
    table.pivot(index="method", columns="comparison", values="Recall").plot.bar(ax=axes[1], rot=0)
    axes[1].set(ylabel="Recall", ylim=(0, 1), title="Three distinct list policies")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{TASK.replace('陳', 'Chen')}: selection evidence, not independent validation", fontsize=11)
    fig.text(.01, .01, "summary run_id=" + run_id, fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    return fig


def summary_run(baseline, rules_id):
    run = p.start_run("L1", TASK + " combined analysis",
                      model_id=baseline["record"]["model_id"], rules_freeze_id=rules_id)
    run.bind_model(baseline["record"])
    run.update(analysis=TASK, comparison_summary=True, policy_id_scope="per-row policy sources",
               limitation="No independent performance estimate; all design evidence uses L1 OOF only")
    return run


def save_figure(run, name, fig):
    try:
        run.save_fig(name, fig)
    finally:
        plt.close(fig)


TASK = "T-陳2"


def run_analysis(rules_id=fm.RULES_FREEZE_ID):
    df, rules = selection_data(rules_id)
    token = uuid4().hex[:12]
    baseline = build_method(df, rules, rules_id, "M1-NS", token)
    categorical = build_method(df, rules, rules_id, "05", token, reference=baseline)
    bundles = {"M1-NS": baseline, "05": categorical}
    compare = pd.concat([standard_comparisons(df, rules, rules_id, b)
                         for b in bundles.values()], ignore_index=True)
    run = summary_run(baseline, rules_id)
    try:
        eligibility = replacement_evidence(bundles, rules, rules_id, df)
        run.update(source_run_ids=compare["run_id"].tolist(), configuration=baseline["configuration"],
                   cv_std_ddof=0, candidate_key="05")
        run.save_csv("pay_categorical_compare", compare)
        run.save_csv("pay_categorical_replacement", eligibility)
        save_figure(run, "pay_categorical_compare", comparison_figure(compare, run.run_id))
        run.finish()
        print(f"Completed {TASK}: summary_run_id={run.run_id}", flush=True)
        print(f"Results: {run.result_path}", flush=True)
        return run, compare, eligibility
    except Exception as exc:
        fail(run, exc)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules-freeze-id", default=fm.RULES_FREEZE_ID)
    args = parser.parse_args(argv)
    return run_analysis(args.rules_freeze_id)


if __name__ == "__main__":
    main()
