"""Private two-stage L4 implementation. Never exposes outer scores to analysis."""
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import getpass
import time
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .registry import (_SEALED_ACCESS, IntegrityError, content_hash, file_hash,
                       read_json, write_json)


@contextmanager
def _access():
    token = _SEALED_ACCESS.set(True)
    try:
        yield
    finally:
        _SEALED_ACCESS.reset(token)


def _configuration(rules):
    cfg = rules.l4
    if content_hash(cfg) != rules.l4_configuration_hash:
        raise IntegrityError("L4 settings differ from the frozen configuration hash")
    if (cfg["model"] != "M1-NS" or cfg["outer_folds"] != 5
            or cfg["n_iter"] != rules.n_iter or cfg["search_space"] != rules.search_space
            or cfg["scenario"] != asdict(rules.scenario)):
        raise IntegrityError("L4 model/search/count/scenario differ from frozen rules")
    if cfg["n_iter"] < 30 or cfg["inner_folds"] != 5:
        raise IntegrityError("L4 requires at least 30 searches and five inner folds")
    return cfg


def _sealed_folders(registry, rules_id):
    root = registry.outputs / "sealed"
    completed = []
    for path in sorted(root.glob(f"l4-{rules_id}-*")):
        manifest = path / "seal.json"
        if manifest.exists():
            record = read_json(manifest)
            if record.get("status") == "sealed" and record.get("rules_freeze_id") == rules_id:
                completed.append(path)
    return completed


def seal_nested(rules_freeze_id):
    from . import protocol as p
    registry = p._registry()
    rules = p.load_rules(rules_freeze_id)
    cfg = _configuration(rules)
    rules_path = registry.outputs / "frozen" / f"rules-{rules_freeze_id}.json"
    rules_hash = file_hash(rules_path)
    with _access():
        if _sealed_folders(registry, rules_freeze_id):
            raise FileExistsError("This rules freeze already has completed sealed results")
        seal_id = f"l4-{rules_freeze_id}-{uuid4().hex[:12]}"
        folder = registry.outputs / "sealed" / seal_id
        folder.mkdir(parents=True, exist_ok=False)
        try:
            data = p._full_data()
            p.validate_ids(data.index)
            X, columns = p.prepare(data, lag=0, sensitive=False, engineered=False)
            outer = StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=cfg["seed"])
            assignment = pd.Series(-1, index=data.index, dtype=int)
            files, fold_metadata = {}, []
            for fold, (tr, va) in enumerate(outer.split(X, data["default"])):
                start = time.perf_counter()
                Xtr, ytr = X.iloc[tr], data["default"].iloc[tr]
                result = p.tune_model(Xtr, ytr, cfg["search_space"], n_iter=cfg["n_iter"],
                    folds=cfg["inner_folds"], seed=cfg["seed"])
                if result["實際搜尋組數"] != cfg["n_iter"] or result["搜尋範圍雜湊"] != content_hash(cfg["search_space"]):
                    raise IntegrityError("Actual L4 search count/space differs from frozen rules")
                params = result["最佳參數"]
                factory = lambda: p.make_model("XGBoost", params=params)
                oof = p.oof_predict(factory, Xtr, ytr, folds=cfg["inner_folds"], seed=cfg["seed"])
                chosen = p.choose_threshold(ytr, oof, p.bill_at(data.iloc[tr], 0), rules.scenario)
                model = factory().fit(Xtr, ytr)
                score = model.predict_proba(X.iloc[va])[:, 1]
                assignment.iloc[va] = fold
                fold_path = folder / f"fold-{fold}"
                fold_path.mkdir()
                pd.DataFrame({"customer_id": X.index[va], "fold": fold, "score": score}).to_csv(
                    fold_path / "outer.csv", index=False, float_format="%.17g")
                pd.DataFrame({"customer_id": Xtr.index, "oof_score": oof.to_numpy(), "y": ytr.to_numpy(),
                              "fold": oof.attrs["fold_assignment"].to_numpy()}).to_csv(fold_path / "inner.csv", index=False, float_format="%.17g")
                model_id = f"xgb-l4-ns-fold{fold}-{seal_id}"
                evidence = dict(model_id=model_id, params_hash=content_hash({"params": params, "columns": columns}),
                    features_hash=content_hash(columns), training_ids_hash=content_hash(Xtr.index.tolist()),
                    oof_sha256=file_hash(fold_path / "inner.csv"), folds_sha256=content_hash(oof.attrs["fold_assignment"].tolist()),
                    seed=cfg["seed"], n_splits=cfg["inner_folds"], rules_freeze_id=rules_freeze_id)
                policy = replace(chosen["政策物件"], evidence=evidence)
                meta = dict(fold=fold, model_id=model_id, params=params, params_hash=evidence["params_hash"],
                    columns=columns, policy=asdict(policy), policy_id=policy.policy_id, policy_hash=policy.policy_hash,
                    actual_n_iter=result["實際搜尋組數"], search_space_hash=result["搜尋範圍雜湊"],
                    oof_sha256=evidence["oof_sha256"], folds_sha256=evidence["folds_sha256"])
                write_json(fold_path / "model-policy.json", meta)
                fold_metadata.append(meta)
                for child in fold_path.iterdir():
                    files[str(child.relative_to(folder))] = file_hash(child)
                # Never print outer metrics or scores, including in verbose search output.
                print(f"outer fold {fold}: {time.perf_counter()-start:.2f}s", flush=True)
            assignments = pd.DataFrame({"customer_id": data.index, "fold": assignment.to_numpy()})
            assignments.to_csv(folder / "assignments.csv", index=False)
            files["assignments.csv"] = file_hash(folder / "assignments.csv")
            registry.load_rules(rules_freeze_id)
            if file_hash(rules_path) != rules_hash:
                raise IntegrityError("Frozen rules changed during sealing")
            seal = dict(seal_id=seal_id, rules_freeze_id=rules_freeze_id, status="sealed",
                rules_freeze_sha256=rules_hash,
                completed_at=datetime.now(timezone.utc).isoformat(), files=files,
                configuration_hash=rules.l4_configuration_hash, customer_count=len(data),
                data_sha256=file_hash(p.DATA_DIR / "default_of_credit_card_clients.csv"),
                assignment_hash=content_hash(assignments.to_dict("records")))
            write_json(folder / "seal.json", seal)
            return seal_id
        except Exception as exc:
            write_json(folder / "seal.json", {"seal_id": seal_id, "rules_freeze_id": rules_freeze_id,
                "status": "failed", "failure": type(exc).__name__})
            raise


def open_nested(rules_freeze_id, model_freeze_id):
    from . import protocol as p
    from ._protocol_runs import _OPENING
    # Freeze checks precede any sealed/raw-data read.
    freeze = p._freeze(model_freeze_id)
    if freeze["rules_freeze_id"] != rules_freeze_id:
        raise IntegrityError("Model and sealed rules freezes differ")
    rules = p.load_rules(rules_freeze_id)
    cfg = _configuration(rules)
    registry = p._registry()
    with _access():
        folders = _sealed_folders(registry, rules_freeze_id)
        if len(folders) != 1:
            raise IntegrityError("Exactly one completed seal is required")
        folder = folders[0]
        seal_path = folder / "seal.json"
        seal = read_json(seal_path)
        seal_hash = file_hash(seal_path)
        for path in registry.runs.glob("*/run.json"):
            existing = read_json(path)
            if existing.get("status") == "complete" and (existing.get("seal_sha256") == seal_hash or existing.get("seal_id") == seal["seal_id"]):
                if existing.get("seal_sha256") != seal_hash:
                    raise IntegrityError("A previously opened seal was modified")
                raise FileExistsError("This seal already has a completed opening run")
        if seal["rules_freeze_id"] != rules_freeze_id or seal["configuration_hash"] != rules.l4_configuration_hash:
            raise IntegrityError("Sealed configuration mismatch")
        if file_hash(registry.outputs / "frozen" / f"rules-{rules_freeze_id}.json") != seal["rules_freeze_sha256"]:
            raise IntegrityError("Frozen rules changed since sealing")
        expected_files = {"assignments.csv"} | {f"fold-{k}/{name}" for k in range(cfg["outer_folds"])
            for name in ("outer.csv", "inner.csv", "model-policy.json")}
        if set(seal["files"]) != expected_files:
            raise IntegrityError("Incomplete or unexpected sealed files")
        for relative, expected_hash in seal["files"].items():
            if file_hash(folder / relative) != expected_hash:
                raise IntegrityError("Sealed file hash mismatch")
        if file_hash(p.DATA_DIR / "default_of_credit_card_clients.csv") != seal["data_sha256"]:
            raise IntegrityError("Sealed data hash mismatch")
        data = p._full_data()
        p.validate_ids(data.index)
        if len(data) != seal["customer_count"]:
            raise IntegrityError("Sealed customer count mismatch")
        X, columns = p.prepare(data, lag=0, sensitive=False, engineered=False)
        cv = StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=cfg["seed"])
        expected_assignment = pd.Series(-1, index=data.index, dtype=int)
        splits = list(cv.split(X, data["default"]))
        for fold, (_, va) in enumerate(splits):
            expected_assignment.iloc[va] = fold
        assigned = pd.read_csv(folder / "assignments.csv")
        p.validate_ids(assigned["customer_id"])
        if (set(assigned["customer_id"]) != set(data.index)
                or content_hash(assigned.to_dict("records")) != seal["assignment_hash"]
                or not np.array_equal(assigned.set_index("customer_id").loc[data.index, "fold"], expected_assignment)):
            raise IntegrityError("Outer fold assignment mismatch")
        batches, metadata = [], []
        for fold, (tr, va) in enumerate(splits):
            fold_path = folder / f"fold-{fold}"
            batch = pd.read_csv(fold_path / "outer.csv", converters={"score": float})
            p.validate_ids(batch["customer_id"])
            if set(batch["customer_id"]) != set(data.index[va]) or set(batch["fold"]) != {fold}:
                raise IntegrityError("Missing, duplicate, extra or wrong-fold outer IDs")
            if not np.isfinite(batch["score"]).all() or not batch["score"].between(0, 1).all():
                raise IntegrityError("Invalid sealed scores")
            meta = read_json(fold_path / "model-policy.json")
            inner = pd.read_csv(fold_path / "inner.csv", converters={"oof_score": float})
            if (meta["fold"] != fold or meta["columns"] != columns or meta["actual_n_iter"] != cfg["n_iter"]
                    or meta["search_space_hash"] != content_hash(cfg["search_space"])
                    or meta["oof_sha256"] != file_hash(fold_path / "inner.csv")
                    or not np.array_equal(inner["customer_id"], data.index[tr])
                    or not np.array_equal(inner["y"], data["default"].iloc[tr])
                    or meta["folds_sha256"] != content_hash(inner["fold"].tolist())
                    or meta["params_hash"] != content_hash({"params": meta["params"], "columns": columns})):
                raise IntegrityError("Sealed fold training/search evidence mismatch")
            policy = p.Policy(**meta["policy"])
            if (policy.policy_hash != meta["policy_hash"] or policy.policy_id != meta["policy_id"]
                    or policy.evidence["training_ids_hash"] != content_hash(data.index[tr].tolist())
                    or policy.evidence["rules_freeze_id"] != rules_freeze_id
                    or policy.evidence["model_id"] != meta["model_id"]
                    or policy.evidence["params_hash"] != meta["params_hash"]
                    or policy.evidence["oof_sha256"] != meta["oof_sha256"]
                    or policy.evidence["folds_sha256"] != meta["folds_sha256"]
                    or policy.evidence["features_hash"] != content_hash(columns)
                    or policy.scenario != asdict(rules.scenario)
                    or policy.kind != "single" or policy.selection != "cost"):
                raise IntegrityError("Sealed policy evidence mismatch")
            expected_inner = np.empty(len(tr), dtype=int)
            inner_cv = StratifiedKFold(cfg["inner_folds"], shuffle=True, random_state=cfg["seed"])
            for inner_fold, (_, inner_va) in enumerate(inner_cv.split(X.iloc[tr], data["default"].iloc[tr])):
                expected_inner[inner_va] = inner_fold
            if not np.array_equal(inner["fold"], expected_inner):
                raise IntegrityError("Sealed inner folds differ from the frozen seed")
            for key, value in meta["params"].items():
                if key == "random_state":
                    valid = value == cfg["seed"]
                else:
                    valid = key in cfg["search_space"] and value in cfg["search_space"][key]
                if not valid:
                    raise IntegrityError("Sealed parameters lie outside the frozen search")
            batches.append(batch)
            metadata.append(meta)
        combined = pd.concat(batches, ignore_index=True)
        p.validate_ids(combined["customer_id"])
        if set(combined["customer_id"]) != set(data.index):
            raise IntegrityError("Every customer must appear exactly once")
        # Only now may any outer metric be computed. Labels align by UCI ID.
        token = _OPENING.set(dict(seal_id=seal["seal_id"], seal_sha256=seal_hash,
            opened_at=datetime.now(timezone.utc).isoformat(), opened_by=getpass.getuser(),
            data_sha256=seal["data_sha256"], fold_policies_hash=content_hash([m["policy_hash"] for m in metadata]),
            folds=metadata))
        try:
            run = p.start_run("L4", "Opened nested tuning and threshold evaluation",
                rules_freeze_id=rules_freeze_id, model_freeze_id=model_freeze_id)
        finally:
            _OPENING.reset(token)
        try:
            rows = []
            for fold, meta in enumerate(metadata):
                batch = batches[fold].set_index("customer_id").loc[data.index[splits[fold][1]]]
                score = batch["score"].rename("score")
                result = p.evaluate(data["default"].loc[score.index], score=score,
                    policy=p.Policy(**meta["policy"]), bill=p.bill_at(data.loc[score.index], 0), scenario=rules.scenario)
                row = result["overall"].copy()
                row["fold"] = fold
                row["model_id"] = meta["model_id"]
                row["run_id"] = run.run_id
                fold_run = run.path / "folds" / str(fold)
                fold_run.mkdir(parents=True, exist_ok=False)
                write_json(fold_run / "model-policy.json", meta)
                row.to_csv(fold_run / "overall.csv", index=False)
                run.save_csv(f"fold-{fold}", row)
                rows.append(row)
            run.save_csv("nested_summary", pd.concat(rows, ignore_index=True))
            run.finish()
            if file_hash(seal_path) != seal_hash:
                raise IntegrityError("Seal changed during opening")
            return run
        except Exception as exc:
            if run.meta["status"] == "running":
                run.fail(type(exc).__name__)
            raise
