"""Private C2 run lifecycle; public construction is protocol.start_run."""
from contextvars import ContextVar
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd

from .registry import (Run, IntegrityError, content_hash, file_hash, identifier,
                       read_json, write_json)

from .git_state import git_dirty

_OPENING = ContextVar("opening_nested_seal", default=None)
_L3_SOURCE = ContextVar("verified_l3_training_source", default=None)


def _metadata(level, description, run_id, **kwargs):
    from . import protocol as p
    from .config import ROOT as root
    return dict(run_id=run_id, level=level, description=description, status="running",
        protocol_run=True, created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_dirty=git_dirty(root),
        python=platform.python_version(), packages={name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "pytest")}, **kwargs)


def start_protocol_run(level, description, *, model_id, rules_freeze_id, model_freeze_id, entry):
    from . import protocol as p
    registry = p._registry()
    if level in {"L0", "L1", "L1-H"}:
        run = registry.start_run(level, description, model_id=model_id, rules_freeze_id=rules_freeze_id,
                                  model_freeze_id=model_freeze_id, entry=entry)
        run.update(protocol_run=True)
        if rules_freeze_id is not None:
            run.update(rules_freeze_sha256=file_hash(registry.outputs / "frozen" / f"rules-{rules_freeze_id}.json"))
        return ProtocolRun(registry, run.meta)
    if level == "L2":
        run = registry.start_run(level, description, model_id=model_id, rules_freeze_id=rules_freeze_id,
                                  model_freeze_id=model_freeze_id, entry=entry)
        run.update(protocol_run=True)
        run.update(rules_freeze_sha256=file_hash(registry.outputs / "frozen" / f"rules-{run.meta['rules_freeze_id']}.json"))
        return ProtocolRun(registry, run.meta)
    if level not in {"L3", "L4"} or model_id is not None or model_freeze_id is None:
        raise ValueError("L3/L4 require a model freeze and cannot supply a model identity/hash")
    if level == "L3":
        if entry is None:
            raise ValueError("L3 requires a frozen entry")
        freeze, chosen, source, digest = registry.frozen_entry(model_freeze_id, entry)
        if rules_freeze_id not in (None, freeze["rules_freeze_id"]):
            raise IntegrityError("Rules/model freeze mismatch")
        rules_freeze_id = freeze["rules_freeze_id"]
        extra = dict(model_id=f"{chosen['model_id']}-l3-{uuid4().hex[:8]}", entry=entry,
                     model_freeze_id=model_freeze_id, model_freeze_sha256=digest)
        shared = _L3_SOURCE.get()
        if shared is not None:
            source = registry.load_run(shared)['run']
            if (source['level'] != 'L3' or source['model_freeze_id'] != model_freeze_id
                    or source['rules_freeze_id'] != rules_freeze_id):
                raise IntegrityError('Invalid shared L3 training source')
            extra.update(model_id=source['model_id'], retraining=source['retraining'],
                         model_source_run_id=shared, training_timing=source['training_timing'], seed=source['seed'])
    else:
        if entry is not None or _OPENING.get() is None:
            raise PermissionError("L4 runs are created only by open_sealed after seal verification")
        freeze = p._freeze(model_freeze_id)
        if rules_freeze_id != freeze["rules_freeze_id"]:
            raise IntegrityError("Rules/model freeze mismatch")
        extra = dict(model_freeze_id=model_freeze_id,
                     model_freeze_sha256=file_hash(registry.outputs / "frozen" / f"model-{model_freeze_id}.json"),
                     **_OPENING.get())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:12]
    path = registry.runs / run_id
    path.mkdir(parents=True, exist_ok=False)
    meta = _metadata(level, description, run_id, rules_freeze_id=rules_freeze_id, **extra)
    meta["rules_freeze_sha256"] = file_hash(registry.outputs / "frozen" / f"rules-{rules_freeze_id}.json")
    write_json(path / "run.json", meta)
    return ProtocolRun(registry, meta)


def _record_policy(run, policy):
    from .protocol import Policy
    run._running()
    if not isinstance(policy, Policy) or run.meta["level"] == "L4":
        raise ValueError("Per-model policy required; L4 stores policies per fold")
    record = read_json(run.registry.path(run.meta["model_record"]))
    source = {k: record[k] for k in ("model_id", "model_hash", "params_hash", "features_hash",
        "training_ids_hash", "seed", "n_splits", "folds_sha256", "oof_sha256")}
    source.update(rules_freeze_id=run.meta.get("rules_freeze_id"), source_level=record["level"])
    # Fixed comparison policies keep their identity across splits. The threshold
    # document still binds that identity to this particular model/OOF evidence.
    if policy.selection != "fixed":
        policy = replace(policy, evidence={**policy.evidence, **source})
    threshold = dict(source, run_id=run.run_id, policy_id=policy.policy_id, policy_hash=policy.policy_hash,
        protocol_policy=asdict(policy), scenario=policy.scenario, threshold=policy.threshold)
    write_json(run.path / "threshold.json", threshold)
    run.update(policy_id=policy.policy_id, policy_hash=policy.policy_hash)
    if policy.scenario is not None:
        run.update(scenario=policy.scenario)
    return policy


def validate_policy_record(registry, meta, record, threshold):
    from .protocol import Policy
    registry.load_model(record)
    if threshold["source_level"] != record["level"]:
        raise IntegrityError("Policy/model data level mismatch")
    for key in ("model_id", "model_hash", "params_hash", "features_hash", "training_ids_hash", "seed", "n_splits", "folds_sha256", "oof_sha256"):
        if threshold[key] != record[key]:
            raise IntegrityError(f"Policy {key} evidence mismatch")
    for key in ("run_id", "model_id", "policy_id", "policy_hash", "rules_freeze_id"):
        if threshold[key] != meta.get(key):
            raise IntegrityError(f"Run/policy {key} mismatch")
    policy = Policy(**threshold["protocol_policy"])
    if policy.policy_hash != threshold["policy_hash"] or policy.policy_id != threshold["policy_id"]:
        raise IntegrityError("Policy content hash mismatch")
    if meta["level"] == "L1-H" and threshold["source_level"] != "L1-H":
        raise IntegrityError("Holdout policy cannot mix with formal policy")
    if meta["level"] in {"L2", "L3"} and threshold["source_level"] == "L1-H":
        raise IntegrityError("Formal model cannot use a development policy")
    if "calibrator" in meta:
        evidence = policy.evidence
        if (evidence.get("calibrator_hash") != meta["calibrator"]["calibrator_hash"]
                or evidence.get("crossfit_oof_sha256") != meta["calibrator"]["crossfit_oof_sha256"]):
            raise IntegrityError("Policy lacks matching cross-fitted calibration evidence")
    return policy


def training_evidence(run, X, oof, config, seed, fit_seconds):
    """Recorded only after an actual fit, before registering its L3 artifact."""
    return dict(independent_fit=True, fit_run_id=run.run_id, seed=int(seed),
                cv_seed=oof.attrs['seed'], n_splits=oof.attrs['folds'],
                training_ids_hash=content_hash(X.index.tolist()),
                configuration_hash=content_hash(config), fit_seconds=fit_seconds,
                oof_scores_hash=content_hash(oof.tolist()),
                fold_assignment_hash=content_hash(oof.attrs['fold_assignment'].tolist()))


def validate_l3_training(registry, meta, record, chosen):
    evidence=meta.get('retraining', {})
    if (record['level']!='L3' or record['model_id']==chosen['model_id']
            or record['model_id']!=meta['model_id'] or evidence.get('independent_fit') is not True
            or evidence.get('seed')!=record['seed'] or evidence.get('seed')!=meta.get('seed')
            or evidence.get('training_ids_hash')!=record['training_ids_hash']
            or evidence.get('configuration_hash')!=content_hash(chosen['configuration'])
            or evidence.get('n_splits')!=record['n_splits']
            or isinstance(evidence.get('fit_seconds'),bool)
            or not isinstance(evidence.get('fit_seconds'),(int,float))
            or not np.isfinite(evidence['fit_seconds']) or evidence['fit_seconds']<=0):
        raise IntegrityError('L3 requires a new identity, independent fit and matching training evidence')
    if ('training_timing' in meta and
            meta['training_timing']['fit_seconds']!=evidence['fit_seconds']):
        raise IntegrityError('L3 fit timing differs from training evidence')
    if evidence.get('fit_run_id')!=meta['run_id']:
        source=registry.load_run(evidence.get('fit_run_id'))['run']
        if (source['level']!='L3' or source['model_id']!=record['model_id']
                or source['model_hash']!=record['model_hash'] or source['retraining']!=evidence
                or source['model_freeze_id']!=meta['model_freeze_id']):
            raise IntegrityError('L3 shared model lacks a completed independent fit source')
    path=registry.path(record['model_path']).parent
    oof=pd.read_csv(path/'oof.csv',converters={'oof_score':float})
    folds=pd.read_csv(path/'folds.csv')
    if (content_hash(oof.oof_score.tolist())!=evidence.get('oof_scores_hash')
            or content_hash(folds['fold'].tolist())!=evidence.get('fold_assignment_hash')):
        raise IntegrityError('L3 OOF/fold evidence mismatch')


def load_protocol_run(registry, run_id, *, allow_running=False):
    from . import protocol as p
    path = registry.runs / identifier(run_id)
    meta = read_json(path / "run.json")
    if meta["run_id"] != run_id or (meta["status"] != "complete" and not (allow_running and meta["status"] == "running")):
        raise IntegrityError("Invalid run identity/status")
    if meta.get("rules_freeze_id"):
        registry.load_rules(meta["rules_freeze_id"])
        if file_hash(registry.outputs / "frozen" / f"rules-{meta['rules_freeze_id']}.json") != meta["rules_freeze_sha256"]:
            raise IntegrityError("Frozen rules changed after run creation")
    if meta["level"] == "L4":
        forbidden = {"model_id", "model_hash", "params_hash", "policy_id", "policy_hash"}
        if forbidden & meta.keys():
            raise IntegrityError("L4 cannot claim a single model/policy hash")
        freeze = p._freeze(meta["model_freeze_id"])
        if (freeze["rules_freeze_id"] != meta["rules_freeze_id"] or not meta.get("seal_sha256")
                or file_hash(registry.outputs / "frozen" / f"model-{meta['model_freeze_id']}.json") != meta["model_freeze_sha256"]):
            raise IntegrityError("L4 freeze/seal binding mismatch")
        if (len(meta["folds"]) != 5 or meta["fold_policies_hash"] != content_hash([m["policy_hash"] for m in meta["folds"]])):
            raise IntegrityError("L4 per-fold policy summary mismatch")
        return {"run": meta}
    record = read_json(registry.path(meta["model_record"]))
    expected_level = "L1" if meta["level"] == "L2" else meta["level"]
    if record["level"] != expected_level:
        raise IntegrityError("Model data level mismatch")
    for key in ("model_id", "model_hash", "params_hash", "data_sha256"):
        if record[key] != meta[key]:
            raise IntegrityError(f"Model/run {key} mismatch")
    if meta["level"] in {"L2", "L3"}:
        freeze, chosen, _, digest = registry.frozen_entry(meta["model_freeze_id"], meta["entry"])
        if digest != meta["model_freeze_sha256"] or freeze["rules_freeze_id"] != meta["rules_freeze_id"]:
            raise IntegrityError("Frozen record mismatch")
        if meta["level"] == "L3":
            validate_l3_training(registry, meta, record, chosen)
        if meta["level"] == "L2":
            keys = ("model_id", "model_hash", "params_hash") if chosen.get("evaluation_kind") == "score_only" else ("model_id", "model_hash", "params_hash", "policy_id", "policy_hash")
            if chosen.get("evaluation_kind") == "score_only" and ("policy_id" in meta or (path / "threshold.json").exists()):
                raise IntegrityError("Score-only run cannot have policy artifacts")
            if meta.get("calibrator") != chosen.get("calibrator"):
                raise IntegrityError("Frozen calibrator mismatch")
            for key in keys:
                if meta[key] != chosen[key]:
                    raise IntegrityError(f"L2 frozen {key} mismatch")
    model, _ = registry.load_model(record)
    if "calibrator" in meta:
        from .model_freeze import load_calibrator
        if (meta.get("calibrator_hash") != meta["calibrator"]["calibrator_hash"]
                or meta.get("calibrator_record") != meta["calibrator"]["calibrator_record"]):
            raise IntegrityError("Calibrator/run identity mismatch")
        load_calibrator(registry, meta["calibrator"], record)
    threshold = None
    policy = None
    if "policy_id" in meta:
        threshold = read_json(path / "threshold.json")
        if "protocol_policy" in threshold:
            policy = validate_policy_record(registry, meta, record, threshold)
        else:
            from .threshold import validate_threshold
            validate_threshold(threshold, record, level=meta["level"], registry=registry)
            for key in ("run_id", "model_id", "policy_id", "policy_hash"):
                if meta[key] != threshold[key]:
                    raise IntegrityError(f"Legacy policy {key} mismatch")
            policy = p.Policy.single(threshold["threshold"], threshold["policy"]["max_list_rate"])
    predictions = None
    if "predictions_sha256" in meta:
        if threshold is None or file_hash(path / "predictions.csv") != meta["predictions_sha256"]:
            raise IntegrityError("Prediction checksum/policy mismatch")
        predictions = pd.read_csv(path / "predictions.csv", converters={"risk_score": float}, keep_default_na=False)
        for key in ("run_id", "model_id", "policy_id"):
            if set(predictions[key]) != {meta[key]}:
                raise IntegrityError(f"Predictions {key} mismatch")
        p.validate_ids(predictions["customer_id"])
        score = pd.Series(predictions["risk_score"].to_numpy(), index=predictions["customer_id"])
        partition = pd.Series(predictions["partition"].to_numpy(), index=score.index) if policy.kind == "group" else None
        expected, _ = p.apply_policy(score, policy, partition=partition)
        if not np.array_equal(expected, predictions["in_list"]):
            raise IntegrityError("Predictions do not match policy")
        if meta["level"] == "L1-H" and (len(score) != 4800 or content_hash(sorted(score.index.tolist())) != record["prediction_ids_set_hash"]):
            raise IntegrityError("Wrong development holdout")
        if meta["level"] == "L1-H" and set(predictions["data_split"]) != {"dev_holdout"}:
            raise IntegrityError("Wrong development data_split")
        if meta["level"] == "L2":
            training = pd.read_csv(registry.path(record["model_path"]).parent / "oof.csv", usecols=["customer_id"])
            if (len(score) != 6000 or set(predictions["data_split"]) != {"test"}
                    or set(training["customer_id"]) & set(score.index)):
                raise IntegrityError("L2 requires 6,000 distinct non-training test customers")
        ranked = predictions.sort_values(["risk_score", "customer_id"], ascending=[False, True])
        if not np.array_equal(predictions["customer_id"], ranked["customer_id"]) or not np.array_equal(predictions["risk_rank"], np.arange(1, len(score)+1)):
            raise IntegrityError("Rank mismatch")
    return {"run": meta, "model": model, "model_record": record, "threshold": threshold, "predictions": predictions}


class ProtocolRun(Run):
    def save_csv(self, name, df):
        return super().save_csv(name, df)

    def update(self, **fields):
        if self.meta["level"] == "L4" and {"model_id", "model_hash", "params_hash", "policy_id", "policy_hash"} & fields.keys():
            raise IntegrityError("L4 cannot impersonate a single frozen model/policy")
        if self.meta["level"] == "L3" and "model_hash" in fields:
            _, chosen, _, _ = self.registry.frozen_entry(self.meta["model_freeze_id"], self.meta["entry"])
            if fields["model_hash"] == chosen["model_hash"] and not self.meta.get('retraining', {}).get('independent_fit'):
                raise IntegrityError("L3 requires independent fit evidence even when model bytes match")
        super().update(**fields)

    def bind_model(self, record):
        if self.meta["level"] == "L4":
            raise IntegrityError("L4 records models per fold only")
        if self.meta["level"] == "L3":
            _, chosen, _, _ = self.registry.frozen_entry(self.meta["model_freeze_id"], self.meta["entry"])
            if record["columns"] != chosen["columns"]:
                raise IntegrityError("L3 features differ from frozen configuration")
            model, _ = self.registry.load_model(record)
            models = model.models_.values() if hasattr(model, 'models_') else [model]
            for submodel in models:
                for key, value in chosen["configuration"]["params"].items():
                    if content_hash(submodel.named_steps["clf"].get_params()[key]) != content_hash(value):
                        raise IntegrityError("L3 parameters differ from frozen configuration")
            validate_l3_training(self.registry, self.meta, record, chosen)
        super().bind_model(record)

    def attach_model(self, model, X, y, score, *, seed=42, prediction_ids=None):
        """Persist the model and actual OOF evidence, without choosing any setting."""
        from . import protocol as p
        if self.meta["level"] not in {"L0", "L1", "L1-H", "L3"}:
            raise ValueError("Cannot replace a frozen model or an L4 fold model")
        p._aligned(y, score)
        if not X.index.equals(y.index):
            raise ValueError("X/y IDs differ")
        assignment = score.attrs.get("fold_assignment")
        if assignment is None:
            raise ValueError("OOF fold provenance is required")
        record = self.registry.register_model(self.meta["model_id"], model,
            columns=X.columns, training_ids=X.index,
            oof=pd.DataFrame({"customer_id": X.index, "oof_score": score.to_numpy(), "y": y.to_numpy()}),
            folds=assignment, level=self.meta["level"],
            data_sha256=file_hash(p.DATA_DIR / "default_of_credit_card_clients.csv"),
            class_weight_algorithm=("Historical full-training-label weight before CV" if self.meta["level"] == "L0"
                                    else "Per fit: only that fit's training labels"),
            prediction_ids=prediction_ids, seed=seed, n_splits=score.attrs["folds"])
        self.bind_model(record)
        return record

    def record_policy(self, policy):
        return _record_policy(self, policy)

    def evaluate_split(self, seed, train, test):
        """L3: use only the frozen configuration, one new model/run per seed."""
        from . import protocol as p
        if self.meta["level"] != "L3":
            raise ValueError("This method is for an L3 run")
        _, chosen, _, _ = self.registry.frozen_entry(self.meta["model_freeze_id"], self.meta["entry"])
        expected_train, expected_test = p._split(p._full_data(), seed)
        if not train.index.equals(expected_train.index) or not test.index.equals(expected_test.index):
            raise IntegrityError("L3 input split does not match the specified seed")
        cfg = chosen["configuration"]
        X, _ = p.prepare(train, lag=cfg["lag"], sensitive=cfg["sensitive"], engineered=cfg["engineered"])
        Z, _ = p.prepare(test, lag=cfg["lag"], sensitive=cfg["sensitive"], engineered=cfg["engineered"])
        factory = lambda: p.make_model(cfg["name"], params=cfg["params"])
        score = p.oof_predict(factory, X, train["default"], seed=seed)
        from .training_time import timed_call
        model, fit_seconds = timed_call(factory().fit, X, train["default"])
        self.update(seed=int(seed), retraining=training_evidence(self, X, score, cfg, seed, fit_seconds))
        self.attach_model(model, X, train["default"], score, seed=seed)
        frozen_policy = p.Policy(**chosen["protocol_policy"])
        rules = p.load_rules(self.meta["rules_freeze_id"])
        partition_train = _partition(train, frozen_policy.partition_name) if frozen_policy.kind == "group" else None
        policy = _policy_for_split(train["default"], score, p.bill_at(train, cfg["lag"]), frozen_policy, rules, partition_train)
        policy = self.record_policy(policy)
        test_score = pd.Series(model.predict_proba(Z)[:, 1], index=test.index)
        partition_test = _partition(test, policy.partition_name) if policy.kind == "group" else None
        report = p.evaluate(test["default"], score=test_score, policy=policy, partition=partition_test,
                            bill=p.bill_at(test, cfg["lag"]), scenario=rules.scenario)
        self.update(seed=int(seed))
        for name, frame in report.items():
            self.save_csv(name, frame)
        self.finish()
        return report


def _partition(frame, name):
    from .data import age_group
    if name == "性別":
        return frame["SEX"].map({1: "男性", 2: "女性"})
    if name == "年齡層":
        return age_group(frame["AGE"]).astype(str)
    if name == "性別×年齡層":
        return frame["SEX"].map({1: "男性", 2: "女性"}) + "・" + age_group(frame["AGE"]).astype(str)
    raise ValueError("Unknown frozen mutually exclusive partition")


def _policy_for_split(y, score, bill, frozen_policy, rules, partition=None):
    from . import protocol as p
    if frozen_policy.kind == "top_n" or frozen_policy.selection == "fixed":
        return frozen_policy
    if frozen_policy.selection == "cost":
        return p.choose_threshold(y, score, bill, rules.scenario)["政策物件"]
    if frozen_policy.kind != "group" or rules.group_threshold["target_definition"] != "single_policy_oof_recall":
        raise ValueError("Unsupported frozen group target definition")
    base = p.choose_threshold(y, score, bill, rules.scenario)["政策物件"]
    # The existing 07 method designs targets using the raw single-threshold list.
    target = float((score[y == 1] >= base.threshold).mean())
    grid = np.round(np.arange(.05, .951, .01), 2)
    thresholds = {}
    for group in sorted(partition.unique()):
        positive = (partition == group) & (y == 1)
        if not positive.any():
            raise ValueError("Cannot fit a group threshold without positives")
        gaps = [abs(float((score[positive] >= t).mean())-target) for t in grid]
        best = min(range(len(grid)), key=lambda i: (round(gaps[i], 4), -grid[i]))
        thresholds[group] = float(grid[best])
    return replace(p.Policy.group(frozen_policy.partition_name, thresholds, rules.scenario.max_list_rate),
                   scenario=asdict(rules.scenario))


def summarize_l3(runs):
    """A separate aggregate; never put multiple model hashes in one run field."""
    rows = []
    for run in runs:
        meta = run.registry.load_run(run.run_id)["run"]
        if meta["level"] != "L3":
            raise ValueError("Only completed L3 runs may be summarized")
        frame = pd.read_csv(run.result_path / "overall.csv")
        rows.append({"seed": meta["seed"], **frame.iloc[0].to_dict()})
    by_seed = pd.DataFrame(rows)
    numeric = [c for c in by_seed.select_dtypes(include="number") if c != "seed"]
    summary = by_seed[numeric].agg(["mean", "std", "min", "max"]).reset_index(names="統計")
    return {"by_seed": by_seed, "summary": summary, "source_run_ids": [r.run_id for r in runs]}
