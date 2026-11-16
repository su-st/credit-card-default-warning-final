"""C1 acceptance tests use synthetic customers only; no UCI/test-set access."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.export import export_run, main as export_main, top_factors, verify_predictions
from src.registry import (GroupedModel, IntegrityError, Registry, content_hash,
                          file_hash, read_json, write_json)
from src.threshold import load_threshold, policy_list, save_threshold, validate_threshold


@pytest.fixture(scope="session")
def synthetic():
    # 19,200 / 4,800 synthetic rows; fixture parameters are not research choices.
    rng = np.random.default_rng(12)
    n = 24000
    frame = pd.DataFrame({
        "LIMIT_BAL": rng.uniform(10000, 100000, n),
        "PAY_1": rng.integers(-2, 4, n),
        "BILL_AMT1": rng.uniform(0, 50000, n),
        "SEX": rng.integers(1, 3, n), "AGE": rng.integers(22, 71, n),
        "EDUCATION": rng.integers(1, 5, n), "MARRIAGE": rng.integers(1, 4, n),
        "default": rng.integers(0, 2, n),
    }, index=pd.Index(range(100001, 124001), name="ID"))
    train, hold = frame.iloc[:19200], frame.iloc[19200:]
    cols = ["LIMIT_BAL", "PAY_1", "BILL_AMT1"]
    model = Pipeline([("pre", ColumnTransformer([("num", StandardScaler(), cols)])),
                      ("clf", XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, random_state=42))])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    score = cross_val_predict(model, train[cols], train["default"], cv=cv, method="predict_proba")[:, 1]
    folds = np.empty(len(train), dtype=int)
    for fold, (_, valid) in enumerate(cv.split(train, train["default"])):
        folds[valid] = fold
    oof = pd.DataFrame({"customer_id": train.index, "oof_score": score, "y": train["default"].to_numpy()})
    model.fit(train[cols], train["default"])
    return SimpleNamespace(train=train, hold=hold, cols=cols, model=model, oof=oof,
                           folds=pd.Series(folds, index=train.index))


def register(registry, fixture, *, model=None, model_id="xgb-synthetic-ns-dev19200", level="L1-H"):
    return registry.register_model(model_id, model or fixture.model, columns=fixture.cols,
        training_ids=fixture.train.index, oof=fixture.oof, folds=fixture.folds,
        prediction_ids=fixture.hold.index if level == "L1-H" else None,
        level=level, data_sha256="a" * 64,
        class_weight_algorithm="Synthetic fixture only; unit class weight, not a research setting")


@pytest.fixture
def registered(tmp_path, synthetic):
    registry = Registry(tmp_path)
    record = register(registry, synthetic)
    return registry, record, synthetic


def pending(registry, record, *, run_id=None, threshold=.5000123456789, cap=None, rules_freeze_id=None):
    run = registry.start_run(record["level"], "C1 synthetic functional verification",
                             model_id=record["model_id"], run_id=run_id, rules_freeze_id=rules_freeze_id)
    run.bind_model(record)
    policy = save_threshold(dict(threshold=threshold, rule="synthetic-fixture",
        selected_on="Synthetic five-fold OOF, not an official policy", max_list_rate=cap,
        assumptions=None), run=run, model_record=record)
    return run, policy


@pytest.fixture
def completed(registered):
    registry, record, data = registered
    run, _ = pending(registry, record)
    bundle = export_run(run, data.hold)
    return registry, run, bundle, data


def rewrite(path, value):
    # Deliberate tampering tests; production never edits completed artifacts.
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def test_repeat_exports_all_4800_full_precision_and_factors(registered):
    registry, record, data = registered
    first, _ = pending(registry, record)
    second, _ = pending(registry, record)
    a = export_run(first, data.hold)
    b = export_run(second, data.hold)
    assert first.run_id != second.run_id
    assert_frame_equal(a["predictions"].drop(columns="run_id"), b["predictions"].drop(columns="run_id"))
    assert a["run"]["verified_customers"] == 4800
    assert a["run"]["predictions_verified"]
    ordered = data.hold.loc[a["predictions"]["customer_id"], data.cols]
    actual = data.model.predict_proba(ordered)[:, 1]
    np.testing.assert_array_equal(a["predictions"]["risk_score"], actual.astype(float))
    assert (actual != np.round(actual, 4)).any()
    assert (a["predictions"]["in_list"] == (actual >= a["threshold"]["threshold"])).all()
    assert read_json(first.result_path / "run.json") == a["run"]


@pytest.mark.parametrize("file,key", [
    ("run.json", "model_id"), ("run.json", "run_id"),
    ("threshold.json", "model_id"), ("threshold.json", "run_id"),
    ("predictions.csv", "model_id"), ("predictions.csv", "run_id"),
    ("predictions.csv", "policy_id"),
])
def test_reject_cross_file_identity_mismatch(completed, file, key):
    registry, run, _, _ = completed
    path = run.path / file
    if file.endswith("csv"):
        rows = pd.read_csv(path)
        rows.loc[0, key] = "wrong-identity"
        rows.to_csv(path, index=False)
        # Keep outer checksum valid so this tests identity checking separately.
        meta = read_json(run.path / "run.json")
        meta["predictions_sha256"] = file_hash(path)
        rewrite(run.path / "run.json", meta)
    else:
        meta = read_json(path)
        meta[key] = "wrong-identity"
        rewrite(path, meta)
    with pytest.raises(IntegrityError):
        registry.load_run(run.run_id)


@pytest.mark.parametrize("which", ["model", "predictions", "oof", "folds"])
def test_reject_actual_file_hash_mismatch(completed, which):
    registry, run, bundle, _ = completed
    model_path = registry.path(bundle["model_record"]["model_path"])
    path = {"model": model_path, "predictions": run.path / "predictions.csv",
            "oof": model_path.parent / "oof.csv", "folds": model_path.parent / "folds.csv"}[which]
    with path.open("ab") as f:
        f.write(b"\ncorrupted")
    with pytest.raises(IntegrityError):
        registry.load_run(run.run_id)


@pytest.mark.parametrize("which", ["threshold.json", "predictions.csv", "run.json"])
def test_missing_file_never_falls_back(completed, which):
    registry, run, _, _ = completed
    (run.path / which).unlink()
    with pytest.raises(FileNotFoundError):
        load_threshold(run.run_id, registry=registry, allow_fallback=True)


def test_no_implicit_global_threshold_or_default(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        load_threshold(registry=Registry(tmp_path))
    threshold, label = load_threshold(registry=Registry(tmp_path), allow_fallback=True)
    assert threshold == .5 and "未使用正式門檻" in label


def test_completed_runs_are_immutable_and_existing_id_is_rejected(completed):
    registry, run, bundle, _ = completed
    original = file_hash(run.path / "run.json")
    for action in [lambda: run.finish(), lambda: run.save_csv("overwrite", pd.DataFrame()),
                   lambda: run.update(extra="changed"), lambda: run.fail("changed")]:
        with pytest.raises(IntegrityError):
            action()
    with pytest.raises(FileExistsError):
        registry.start_run("L1-H", "retry", model_id=bundle["run"]["model_id"], run_id=run.run_id)
    assert file_hash(run.path / "run.json") == original


def test_same_model_id_different_fitted_models_have_separate_storage(registered):
    registry, record, data = registered
    other = deepcopy(data.model)
    other.set_params(clf__n_estimators=4)
    other.fit(data.train[data.cols], data.train["default"])
    second = register(registry, data, model=other)
    assert second["model_id"] == record["model_id"]
    assert second["model_hash"] != record["model_hash"]
    assert second["model_path"] != record["model_path"]
    # Re-registering the identical model+evidence reuses it, without overwriting.
    assert register(registry, data) == record
    assert not list(registry.models.glob(".register-*"))


def test_reject_test_export_before_any_prediction(registered, monkeypatch):
    registry, record, _ = registered
    run, _ = pending(registry, record)
    run.meta["level"] = "L2"
    monkeypatch.setattr(registry, "load_run", lambda *a, **kw: pytest.fail("loaded before rejecting test export"))
    with pytest.raises(PermissionError):
        export_run(run, None)
    with pytest.raises(PermissionError):
        export_main(["--run-id", "does-not-exist", "--level", "L2"])


def test_all_customers_verified_even_when_only_last_score_is_forged(completed):
    registry, run, bundle, data = completed
    rows = bundle["predictions"].copy()
    rows.loc[len(rows) - 1, "risk_score"] -= 1e-5
    rows.to_csv(run.path / "predictions.csv", index=False, float_format="%.17g")
    meta = read_json(run.path / "run.json")
    meta["predictions_sha256"] = file_hash(run.path / "predictions.csv")
    rewrite(run.path / "run.json", meta)
    with pytest.raises(IntegrityError, match="all exported scores"):
        verify_predictions(registry, run.run_id, data.hold[data.cols])


def test_development_policy_cannot_bind_to_formal_model(registered):
    registry, record, data = registered
    run, threshold = pending(registry, record)
    assert threshold["source_level"] == "L1-H"
    formal = register(registry, data, model_id="xgb-synthetic-ns-s42train", level="L1")
    with pytest.raises(IntegrityError):
        validate_threshold(threshold, formal, level="L1", registry=registry)
    with pytest.raises(IntegrityError, match="formal"):
        validate_threshold(threshold, record, level="L2", registry=registry)


def test_threshold_has_all_section_5_3_2_source_fields(registered):
    registry, record, _ = registered
    _, threshold = pending(registry, record)
    required = {"model_id", "model_hash", "params_hash", "features_hash", "training_ids_hash",
        "seed", "n_splits", "folds_sha256", "oof_sha256", "rules_freeze_id", "scenario", "policy"}
    assert required <= threshold.keys()
    for key in required - {"rules_freeze_id", "scenario", "policy"}:
        assert threshold[key] == record[key]


def test_grouped_model_routes_every_id_after_reload_and_export(tmp_path, synthetic):
    columns = synthetic.cols
    X = synthetic.train[columns]
    no_delay = DummyClassifier(strategy="constant", constant=0).fit(X, synthetic.train["default"])
    delay = DummyClassifier(strategy="constant", constant=1).fit(X, synthetic.train["default"])
    model = GroupedModel(no_delay, delay, ["PAY_1"], columns)
    registry = Registry(tmp_path)
    record = register(registry, synthetic, model=model, model_id="grouped-synthetic-dev19200")
    reloaded, _ = registry.load_model(record)
    expected = (synthetic.hold["PAY_1"] > 0).astype(float).to_numpy()
    np.testing.assert_array_equal(reloaded.predict_proba(synthetic.hold)[:, 1], expected)
    # Factors here are synthetic; this verifies packaging+full handoff verification.
    run, _ = pending(registry, record, threshold=.5)
    bundle = export_run(run, synthetic.hold, factors_fn=lambda m, X: [["synthetic", "", ""] for _ in range(len(X))])
    got = bundle["predictions"].set_index("customer_id").loc[synthetic.hold.index, "risk_score"]
    np.testing.assert_array_equal(got, expected)
    assert bundle["run"]["verified_customers"] == 4800


def test_capacity_ties_use_customer_id_and_full_score():
    score = pd.Series([.50000003, .50000003, .50000001, .49999999], index=[8, 2, 5, 3])
    policy = {"policy": {"kind": "single", "threshold": .5, "max_list_rate": .25}}
    assert policy_list(score, policy).to_dict() == {8: 0, 2: 1, 5: 0, 3: 0}


def test_wrong_holdout_ids_fail_run_without_overwriting(registered):
    registry, record, data = registered
    run, _ = pending(registry, record)
    with pytest.raises(IntegrityError, match="holdout IDs"):
        export_run(run, data.hold.iloc[::-1])
    assert read_json(run.path / "run.json")["status"] == "failed"
    assert not (run.path / "predictions.csv").exists()
    with pytest.raises(FileExistsError):
        registry.start_run("L1-H", "retry", model_id=record["model_id"], run_id=run.run_id)


def test_frozen_rules_content_is_read_from_record_not_markdown(tmp_path):
    registry = Registry(tmp_path)
    frozen = registry.outputs / "frozen"
    frozen.mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "frozen_rules.md").write_text("Human readable frozen rules")
    write_json(frozen / "rules-fixture.json", {"rules_freeze_id": "fixture", "setting": "from JSON",
        "markdown_sha256": file_hash(docs / "frozen_rules.md")})
    assert registry.load_rules("fixture")["setting"] == "from JSON"
    (docs / "frozen_rules.md").write_text("tampered")
    with pytest.raises(IntegrityError, match="Markdown"):
        registry.load_rules("fixture")


def test_run_summaries_include_identities_and_refuse_overwrite(registered):
    registry, record, _ = registered
    run, threshold = pending(registry, record)
    run.save_csv("summary", pd.DataFrame({"value": [3]}))
    rows = pd.read_csv(run.result_path / "summary.csv")
    assert rows.loc[0, "run_id"] == run.run_id
    assert rows.loc[0, "model_id"] == record["model_id"]
    assert rows.loc[0, "policy_id"] == threshold["policy_id"]
    with pytest.raises(FileExistsError):
        run.save_csv("summary", pd.DataFrame())
    run.finish()
    assert registry.load_run(run.run_id)["run"]["status"] == "complete"


def test_sensitive_and_non_sensitive_dev_models_export_separately(registered):
    registry, ns_record, data = registered
    cols = data.cols + ["SEX", "AGE", "EDUCATION", "MARRIAGE"]
    model = Pipeline([("pre", ColumnTransformer([("num", StandardScaler(), cols)])),
                      ("clf", XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, random_state=42))])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_predict(model, data.train[cols], data.train["default"], cv=cv, method="predict_proba")[:, 1]
    model.fit(data.train[cols], data.train["default"])
    oof = data.oof.copy()
    oof["oof_score"] = scores
    sensitive = registry.register_model("xgb-synthetic-sensitive-dev19200", model, columns=cols,
        training_ids=data.train.index, oof=oof, folds=data.folds, prediction_ids=data.hold.index,
        level="L1-H", data_sha256="a" * 64, class_weight_algorithm="Synthetic fixture; unit weights")
    a, pa = pending(registry, ns_record)
    b, pb = pending(registry, sensitive)
    ba, bb = export_run(a, data.hold), export_run(b, data.hold)
    assert ba["run"]["model_hash"] != bb["run"]["model_hash"]
    assert pa["policy_id"] != pb["policy_id"]
    assert set(ba["predictions"]["customer_id"]) == set(bb["predictions"]["customer_id"])
    assert bb["model_record"]["columns"] == cols
    with pytest.raises(IntegrityError):
        validate_threshold(pa, sensitive, level="L1-H", registry=registry)


def test_same_model_can_have_distinct_policies_without_duplicating_artifact(registered):
    registry, record, _ = registered
    a, pa = pending(registry, record, threshold=.4)
    b, pb = pending(registry, record, threshold=.6)
    a.finish()
    b.finish()
    assert a.meta["model_hash"] == b.meta["model_hash"]
    assert pa["policy_id"] != pb["policy_id"]
    assert len(list(registry.models.glob("*/model.joblib"))) == 1


@pytest.mark.parametrize("key", ["params_hash", "features_hash", "training_ids_hash", "folds_sha256", "oof_sha256"])
def test_threshold_rejects_changed_source_evidence(registered, key):
    registry, record, _ = registered
    _, policy = pending(registry, record)
    policy[key] = "0" * 64
    with pytest.raises(IntegrityError, match=key):
        validate_threshold(policy, record, level="L1-H", registry=registry)


def test_export_requires_4800_rows_and_same_registered_order(registered):
    registry, record, data = registered
    run, _ = pending(registry, record)
    with pytest.raises(IntegrityError, match="4,800"):
        export_run(run, data.hold.head(100))


def test_failed_run_retry_uses_new_id(registered):
    registry, record, data = registered
    failed, _ = pending(registry, record)
    with pytest.raises(RuntimeError, match="fixture failure"):
        export_run(failed, data.hold, factors_fn=lambda m, X: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    retry, _ = pending(registry, record)
    result = export_run(retry, data.hold)
    assert failed.run_id != retry.run_id
    assert read_json(failed.path / "run.json")["status"] == "failed"
    assert result["run"]["status"] == "complete"


def test_invalid_scores_or_policy_ranges_are_rejected():
    for scores, threshold, cap in [([float("nan")], .5, None), ([1.1], .5, None),
                                    ([.5], 1.1, None), ([.5], .5, -.1)]:
        with pytest.raises(ValueError):
            policy_list(pd.Series(scores, index=[1]), {"policy": {"kind": "single", "threshold": threshold, "max_list_rate": cap}})


def test_02_io_migration_keeps_original_analysis_results(tmp_path, synthetic, capsys):
    # Execute 02's unchanged numerical body on synthetic L1 OOF only. No protocol
    # data access or real rules are exercised; fixture records are in tmp_path.
    spec = importlib.util.spec_from_file_location("threshold_experiment", Path(__file__).parents[1] / "experiments/02_threshold.py")
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    registry = Registry(tmp_path)
    record = register(registry, synthetic, level="L1", model_id="synthetic-l1")
    run = registry.start_run("L1", "synthetic legacy regression", model_id=record["model_id"])
    run.bind_model(record)
    from src.threshold import select_threshold, threshold_table
    y = synthetic.train["default"]
    loss = float(synthetic.train.loc[y == 1, "BILL_AMT1"].clip(lower=0).mean())
    expected = threshold_table(y, synthetic.oof["oof_score"], loss, 50, .2)
    experiment.analyze(registry, run, record, SimpleNamespace(rules_freeze_id=None),
        synthetic.train, y, synthetic.oof["oof_score"].to_numpy(), None, 50, .2, "f1", .2, None,
        [50, 500, 2000, 5000], [.1, .2, .3])
    saved = pd.read_csv(run.result_path / "threshold_table.csv")
    assert_frame_equal(saved[list(expected.columns)], expected.round(4), check_dtype=False)
    assert registry.load_run(run.run_id)["threshold"]["threshold"] == select_threshold(expected, "f1")
    sensitivity = pd.read_csv(run.result_path / "threshold_sensitivity.csv")
    assert len(sensitivity) == 12
    for _, row in sensitivity.iterrows():
        table = threshold_table(y, synthetic.oof["oof_score"], loss, row["聯絡成本"], row["挽回比例"])
        assert row["最佳門檻"] == select_threshold(table, "cost")
        assert registry.load_run(row["run_id"])["threshold"]["policy_id"] == row["policy_id"]


def test_02_import_does_not_load_test_data(monkeypatch):
    import src.data
    monkeypatch.setattr(src.data, "load_data", lambda: pytest.fail("02 imported raw data"))
    spec = importlib.util.spec_from_file_location("threshold_import_only", Path(__file__).parents[1] / "experiments/02_threshold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_no_formal_run_without_model_freeze(tmp_path):
    registry = Registry(tmp_path)
    with pytest.raises(ValueError, match="requires"):
        registry.start_run("L2", "not frozen")
    with pytest.raises(FileNotFoundError):
        registry.start_run("L2", "missing freeze", model_freeze_id="absent", entry="platform")
    assert not registry.runs.exists()


def test_frozen_6000_customer_reload_verification_without_real_test_data(tmp_path, synthetic):
    registry = Registry(tmp_path)
    frozen_dir = registry.outputs / "frozen"
    frozen_dir.mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    md = tmp_path / "docs/frozen_rules.md"
    md.write_text("Synthetic freeze fixture, not a project freeze")
    write_json(frozen_dir / "rules-fixture.json", {"rules_freeze_id": "fixture", "markdown_sha256": file_hash(md)})
    train = pd.concat([synthetic.train, synthetic.hold])
    cols = synthetic.cols
    model = deepcopy(synthetic.model)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_predict(model, train[cols], train["default"], cv=cv, method="predict_proba")[:, 1]
    folds = pd.Series(-1, index=train.index, dtype=int)
    for k, (_, valid) in enumerate(cv.split(train, train["default"])):
        folds.iloc[valid] = k
    model.fit(train[cols], train["default"])
    record = registry.register_model("synthetic-formal-s42train", model, columns=cols,
        training_ids=train.index, oof=pd.DataFrame({"customer_id": train.index, "oof_score": scores, "y": train["default"].to_numpy()}),
        folds=folds, level="L1", data_sha256="a" * 64, class_weight_algorithm="Synthetic fixture, unit weights")
    source, _ = pending(registry, record, rules_freeze_id="fixture")
    source.finish()
    selected = {k: source.meta[k] for k in ("model_id", "model_hash", "params_hash", "policy_id", "policy_hash")}
    selected.update(entry_id="platform", source_run_id=source.run_id, columns=cols,
                    role="platform", comparison_id="synthetic-comparison")
    freeze_path = frozen_dir / "model-fixture.json"
    write_json(freeze_path, {"model_freeze_id": "fixture", "rules_freeze_id": "fixture", "approved_at": "synthetic-test-only", "entries": [selected]})
    run = registry.start_run("L2", "synthetic 6000-row verification", model_freeze_id="fixture", entry="platform")
    policy = read_json(run.path / "threshold.json")
    X = pd.concat([synthetic.hold, synthetic.hold.iloc[:1200]], ignore_index=True)
    X.index = pd.Index(range(200001, 206001), name="ID")
    score = model.predict_proba(X[cols])[:, 1]
    rows = pd.DataFrame({"customer_id": X.index, "risk_score": score,
        "in_list": policy_list(pd.Series(score, index=X.index), policy).to_numpy(),
        "run_id": run.run_id, "model_id": run.meta["model_id"], "policy_id": policy["policy_id"], "data_split": "test"})
    rows = rows.sort_values(["risk_score", "customer_id"], ascending=[False, True])
    rows["risk_rank"] = range(1, 6001)
    rows.to_csv(run.path / "predictions.csv", index=False, float_format="%.17g")
    run.update(predictions_sha256=file_hash(run.path / "predictions.csv"))
    assert verify_predictions(registry, run.run_id, X[cols]) == 6000
    run.update(predictions_verified=True, verified_customers=6000)
    run.finish()
    assert registry.load_run(run.run_id)["run"]["model_hash"] == source.meta["model_hash"]
    # Editing the freeze after completion is detected; no production freeze is touched.
    changed = read_json(freeze_path)
    changed["approved_at"] = "changed"
    rewrite(freeze_path, changed)
    with pytest.raises(IntegrityError, match="changed"):
        registry.load_run(run.run_id)
