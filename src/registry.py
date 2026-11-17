"""Content-addressed model artifacts and write-once C1 runs.

No data is loaded and no model is fitted on import. C2 owns the public protocol,
data access, training and freeze orchestration; this module owns artifact I/O.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from uuid import uuid4
from contextvars import ContextVar

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import ROOT
from .git_state import git_dirty

_SEALED_ACCESS = ContextVar("sealed_artifact_access", default=False)


def _guard_sealed(path):
    parts = Path(path).resolve().parts
    if any(parts[i:i+2] == ("outputs", "sealed") for i in range(len(parts)-1)) and not _SEALED_ACCESS.get():
        raise PermissionError("Sealed artifacts may only be read by nested_eval/open_sealed")


class IntegrityError(ValueError):
    """An artifact disagrees with its recorded identity or evidence."""


def now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    """Represent full estimator parameters without repr/memory addresses."""
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, float) and not np.isfinite(value):
        return {"float": str(value)}
    if hasattr(value, "get_params"):
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}",
                "params": json_value(value.get_params(deep=False))}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported parameter type: {type(value).__name__}")


def content_hash(value):
    raw = json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    _guard_sealed(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Invalid artifact identifier")
    return value


def read_json(path):
    _guard_sealed(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, exclusive=True):
    text = json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False) + "\n"
    path = Path(path)
    if exclusive:
        with path.open("x", encoding="utf-8") as f:
            f.write(text)
    else:
        # Only the owning, still-running Run calls this for state transitions.
        temp = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
        try:
            temp.write_text(text, encoding="utf-8")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def validate_ids(ids):
    ids = pd.Index(ids)
    if not len(ids) or ids.hasnans or not ids.is_unique:
        raise ValueError("Customer IDs must be nonempty, unique and non-null")
    if not pd.api.types.is_integer_dtype(ids.dtype) or (ids <= 0).any():
        raise ValueError("Customer IDs must be positive integers")
    return ids


class GroupedModel(ClassifierMixin, BaseEstimator):
    """Persist fitted F6-D submodels together with the existing delay rule.

    Routing uses only the supplied PAY columns: any positive code selects the
    delayed model. Feature engineering stays with the experiment's owner.
    """
    def __init__(self, no_delay_model, delay_model, pay_columns, columns):
        self.no_delay_model = no_delay_model
        self.delay_model = delay_model
        self.pay_columns = pay_columns
        self.columns = columns
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        delayed = (X[list(self.pay_columns)] > 0).any(axis=1).to_numpy()
        result = np.empty((len(X), 2), dtype=float)
        for mask, model in ((~delayed, self.no_delay_model), (delayed, self.delay_model)):
            if mask.any():
                if list(model.classes_) != [0, 1]:
                    raise IntegrityError("Grouped submodel classes must be [0, 1]")
                result[mask] = model.predict_proba(X.loc[mask, list(self.columns)])
        return result


class Registry:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.outputs = self.root / "outputs"
        self.models = self.outputs / "models"
        self.runs = self.outputs / "runs"
        self.results = self.root / "results"

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.outputs.resolve()):
            raise IntegrityError("Artifact path must stay inside outputs/")
        if path.is_relative_to((self.outputs / "sealed").resolve()):
            raise IntegrityError("C1 does not open sealed artifacts")
        return path

    def relative(self, path):
        return str(Path(path).relative_to(self.root))

    def register_model(self, model_id, model, *, columns, training_ids, oof, folds,
                       level, data_sha256, class_weight_algorithm, prediction_ids=None,
                       seed=42, n_splits=5):
        """Store an already-fitted model, actual OOF scores and fold evidence.

        Never trains or chooses settings. OOF rows must follow training ID order.
        L1-H artifacts additionally declare their separate development holdout.
        """
        identifier(model_id)
        if level not in {"L0", "L1", "L1-H", "L3"}:
            raise ValueError("C1 model registration requires L0, L1 or L1-H")
        ids = validate_ids(training_ids)
        columns = list(columns)
        if not columns or len(set(columns)) != len(columns) or {"default", "ID", "customer_id"} & set(columns):
            raise ValueError("Invalid model feature columns")
        if list(oof.columns) != ["customer_id", "oof_score", "y"]:
            raise ValueError("OOF columns must be customer_id, oof_score, y")
        if not np.array_equal(ids, oof["customer_id"].to_numpy()):
            raise IntegrityError("OOF IDs do not match ordered training IDs")
        scores = oof["oof_score"].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Invalid OOF risk scores")
        if not oof["y"].isin([0, 1]).all():
            raise ValueError("OOF labels must be binary")
        if not isinstance(folds, pd.Series) or not folds.index.equals(ids):
            raise IntegrityError("Fold assignment must align with training IDs")
        if not pd.api.types.is_integer_dtype(folds.dtype) or set(folds) != set(range(n_splits)):
            raise ValueError("Fold assignment must cover all zero-based folds")
        if not class_weight_algorithm or not re.fullmatch(r"[a-f0-9]{64}", data_sha256):
            raise ValueError("Actual data hash and class-weight description required")
        prediction_hash = None
        prediction_set_hash = None
        if level == "L1-H":
            hold_ids = validate_ids(prediction_ids if prediction_ids is not None else [])
            if len(ids.intersection(hold_ids)):
                raise IntegrityError("Training and development holdout overlap")
            prediction_hash = content_hash(hold_ids.tolist())
            prediction_set_hash = content_hash(sorted(hold_ids.tolist()))
        self.models.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".register-", dir=self.models))
        try:
            joblib.dump(model, temp / "model.joblib", compress=3)
            model_hash = file_hash(temp / "model.joblib")
            oof.to_csv(temp / "oof.csv", index=False)
            pd.DataFrame({"customer_id": ids, "fold": folds.to_numpy()}).to_csv(temp / "folds.csv", index=False)
            params = json_value(model)
            record = dict(model_id=model_id, model_hash=model_hash,
                          params=params, columns=columns,
                          params_hash=content_hash({"params": params, "columns": columns}),
                          features_hash=content_hash(columns), level=level,
                          training_ids_hash=content_hash(ids.tolist()), training_count=len(ids),
                          prediction_ids_hash=prediction_hash,
                          prediction_ids_set_hash=prediction_set_hash,
                          class_weight_algorithm=class_weight_algorithm, data_sha256=data_sha256,
                          seed=seed, n_splits=n_splits,
                          oof_sha256=file_hash(temp / "oof.csv"),
                          folds_sha256=file_hash(temp / "folds.csv"))
            dest = self.models / f"{model_id}-{model_hash[:12]}"
            record["model_path"] = self.relative(dest / "model.joblib")
            record["model_record"] = self.relative(dest / "model.json")
            write_json(temp / "model.json", record)
            if dest.exists():
                if self.load_model(record)[1] != record:
                    raise IntegrityError("Same model bytes have conflicting training evidence")
            else:
                temp.rename(dest)
            return deepcopy(record)
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    def load_model(self, record):
        path = self.path(record["model_path"])
        stored = read_json(path.parent / "model.json")
        if stored != record:
            raise IntegrityError("Model record mismatch")
        if file_hash(path) != record["model_hash"]:
            raise IntegrityError("Model file hash mismatch")
        for name, key in (("oof.csv", "oof_sha256"), ("folds.csv", "folds_sha256")):
            if file_hash(path.parent / name) != record[key]:
                raise IntegrityError(f"{name} hash mismatch")
        oof = pd.read_csv(path.parent / "oof.csv", converters={"oof_score": float})
        folds = pd.read_csv(path.parent / "folds.csv")
        if (content_hash(oof["customer_id"].tolist()) != record["training_ids_hash"]
                or not np.array_equal(oof["customer_id"], folds["customer_id"])
                or set(folds["fold"]) != set(range(record["n_splits"]))):
            raise IntegrityError("Training IDs or folds disagree with model record")
        if content_hash(record["columns"]) != record["features_hash"] or content_hash(
                {"params": record["params"], "columns": record["columns"]}) != record["params_hash"]:
            raise IntegrityError("Model parameter/feature hash mismatch")
        model = joblib.load(path)
        if json_value(model) != record["params"]:
            raise IntegrityError("Loaded model parameters disagree with record")
        return model, stored

    def load_rules(self, rules_freeze_id):
        identifier(rules_freeze_id)
        rules = read_json(self.outputs / "frozen" / f"rules-{rules_freeze_id}.json")
        if rules["rules_freeze_id"] != rules_freeze_id:
            raise IntegrityError("Rules freeze identity mismatch")
        if file_hash(self.root / "docs" / "frozen_rules.md") != rules["markdown_sha256"]:
            raise IntegrityError(
                "Frozen Markdown hash mismatch. Possible Windows line-ending conversion; "
                "see README.md「Windows 使用者」for recovery steps.")
        return rules

    def frozen_entry(self, model_freeze_id, entry):
        """Read, never create or alter, a leader-approved model freeze entry."""
        identifier(model_freeze_id)
        path = self.outputs / "frozen" / f"model-{model_freeze_id}.json"
        freeze = read_json(path)
        if freeze["model_freeze_id"] != model_freeze_id or not freeze["approved_at"]:
            raise IntegrityError("Model freeze identity/leader approval is missing")
        self.load_rules(freeze["rules_freeze_id"])
        matches = [row for row in freeze["entries"] if row["entry_id"] == entry]
        if len(matches) != 1:
            raise IntegrityError("Frozen entry must identify exactly one model/policy")
        selected = matches[0]
        if not selected.get("role") or not selected.get("comparison_id"):
            raise IntegrityError("Frozen entry requires a role and comparison_id")
        source_meta = read_json(self.runs / identifier(selected["source_run_id"]) / "run.json")
        if source_meta["level"] != "L1":
            raise IntegrityError("Formal freeze cannot reference a development or formal-test run")
        source = self.load_run(selected["source_run_id"])
        if source["run"]["level"] != "L1" or source["run"]["rules_freeze_id"] != freeze["rules_freeze_id"]:
            raise IntegrityError("Formal freeze must reference its L1 rules/model/policy source")
        for key in ("model_id", "model_hash", "params_hash", "policy_id", "policy_hash"):
            if selected[key] != source["run"][key]:
                raise IntegrityError(f"Frozen {key} disagrees with source evidence")
        if selected["columns"] != source["model_record"]["columns"]:
            raise IntegrityError("Frozen feature list disagrees with model")
        if source["model_record"]["training_count"] != 24000:
            raise IntegrityError("Formal model must have 24,000 training customers")
        return freeze, selected, source, file_hash(path)

    def start_run(self, level, description, *, model_id=None, rules_freeze_id=None,
                  model_freeze_id=None, entry=None, run_id=None):
        frozen = None
        if level == "L2":
            if model_id is not None or model_freeze_id is None or entry is None:
                raise ValueError("L2 requires model_freeze_id and entry; model_id comes from the freeze")
            freeze, selected, source, freeze_hash = self.frozen_entry(model_freeze_id, entry)
            if rules_freeze_id not in (None, freeze["rules_freeze_id"]):
                raise IntegrityError("Rules freeze does not match the model freeze")
            rules_freeze_id = freeze["rules_freeze_id"]
            model_id = selected["model_id"]
            frozen = (freeze, selected, source, freeze_hash)
        elif level not in {"L0", "L1", "L1-H"}:
            # C2/final_eval owns formal frozen evaluation and L3/L4 orchestration.
            raise ValueError("C2 owns L3/L4 per-seed/sealed-run orchestration")
        if level != "L2" and (model_freeze_id is not None or entry is not None):
            raise ValueError("Development runs cannot impersonate frozen evaluation")
        identifier(model_id)
        if rules_freeze_id is not None:
            self.load_rules(rules_freeze_id)
        run_id = identifier(run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:12])
        path = self.runs / run_id
        result = self.results / run_id
        if result.exists():
            raise FileExistsError(result)
        path.mkdir(parents=True, exist_ok=False)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            dirty = git_dirty(ROOT)
            packages = {p: importlib.metadata.version(p) for p in (
                "pandas", "numpy", "scikit-learn", "xgboost", "joblib", "pytest")}
            meta = dict(run_id=run_id, level=level, model_id=model_id, description=description,
                        rules_freeze_id=rules_freeze_id, status="running", created_at=now(),
                        git_commit=commit, git_dirty=dirty, python=platform.python_version(), packages=packages)
            if frozen is not None:
                _, selected, source, freeze_hash = frozen
                meta.update(model_freeze_id=model_freeze_id, entry=entry,
                            model_freeze_sha256=freeze_hash)
            write_json(path / "run.json", meta)
            run = Run(self, meta)
            if frozen is not None:
                run.bind_model(source["model_record"])
                policy = dict(source["threshold"], run_id=run_id)
                write_json(path / "threshold.json", policy)
                run.update(policy_id=policy["policy_id"], policy_hash=policy["policy_hash"], scenario=policy["scenario"])
            return run
        except Exception:
            # Leave a failed marker; retries always need another ID.
            write_json(path / "run.json", {"run_id": run_id, "status": "failed"}, exclusive=not (path / "run.json").exists())
            raise

    def load_run(self, run_id, *, allow_running=False):
        from .threshold import validate_threshold

        path = self.runs / identifier(run_id)
        meta = read_json(path / "run.json")
        if meta.get("protocol_run"):
            from ._protocol_runs import load_protocol_run
            return load_protocol_run(self, run_id, allow_running=allow_running)
        if meta["run_id"] != run_id:
            raise IntegrityError("Run identity mismatch")
        if meta["status"] != "complete" and not (allow_running and meta["status"] == "running"):
            raise IntegrityError("Run is not complete")
        record = read_json(self.path(meta["model_record"]))
        if record["level"] != ("L1" if meta["level"] == "L2" else meta["level"]):
            raise IntegrityError("Model/run data level mismatch")
        if meta["level"] == "L2":
            freeze, selected, _, digest = self.frozen_entry(meta["model_freeze_id"], meta["entry"])
            if digest != meta["model_freeze_sha256"] or freeze["rules_freeze_id"] != meta["rules_freeze_id"]:
                raise IntegrityError("Frozen record changed after run creation")
            for key in ("model_id", "model_hash", "params_hash", "policy_id", "policy_hash"):
                if meta[key] != selected[key]:
                    raise IntegrityError(f"L2 {key} differs from frozen entry")
        for key in ("model_id", "model_hash", "params_hash", "data_sha256"):
            if record[key] != meta[key]:
                raise IntegrityError(f"Run/model {key} mismatch")
        model, _ = self.load_model(record)
        threshold = read_json(path / "threshold.json")
        validate_threshold(threshold, record, level=meta["level"], registry=self)
        for key in ("run_id", "model_id", "model_hash", "policy_id", "policy_hash", "rules_freeze_id"):
            if threshold[key] != meta[key]:
                raise IntegrityError(f"Run/threshold {key} mismatch")
        predictions = None
        if "predictions_sha256" in meta:
            if file_hash(path / "predictions.csv") != meta["predictions_sha256"]:
                raise IntegrityError("Predictions hash mismatch")
            predictions = pd.read_csv(path / "predictions.csv", converters={"risk_score": float}, keep_default_na=False)
            for key in ("run_id", "model_id", "policy_id"):
                if set(predictions[key]) != {meta[key]}:
                    raise IntegrityError(f"Predictions {key} mismatch")
            validate_ids(predictions["customer_id"])
            if meta["level"] == "L1-H":
                if (len(predictions) != 4800
                        or content_hash(sorted(predictions["customer_id"].tolist())) != record["prediction_ids_set_hash"]
                        or set(predictions["data_split"]) != {"dev_holdout"}):
                    raise IntegrityError("Prediction IDs or data level do not match L1-H holdout")
            if meta["level"] == "L2":
                training = pd.read_csv(self.path(record["model_path"]).parent / "oof.csv", usecols=["customer_id"])
                if (len(predictions) != 6000 or set(predictions["data_split"]) != {"test"}
                        or set(training["customer_id"]) & set(predictions["customer_id"])):
                    raise IntegrityError("L2 must contain 6,000 distinct non-training test customers")
            expected_order = predictions.sort_values(["risk_score", "customer_id"], ascending=[False, True])
            if (not np.array_equal(predictions["customer_id"], expected_order["customer_id"])
                    or not np.array_equal(predictions["risk_rank"], np.arange(1, len(predictions) + 1))):
                raise IntegrityError("Risk ranking is inconsistent with full-precision scores")
            from .threshold import policy_list
            score = pd.Series(predictions["risk_score"].to_numpy(), index=predictions["customer_id"])
            if not np.array_equal(predictions["in_list"], policy_list(score, threshold).to_numpy()):
                raise IntegrityError("Prediction list does not match full-precision scores")
        return {"run": meta, "threshold": threshold, "model": model,
                "model_record": record, "predictions": predictions}


class Run:
    def __init__(self, registry, meta):
        self.registry = registry
        self.meta = deepcopy(meta)
        self.run_id = meta["run_id"]
        self.path = registry.runs / self.run_id
        self.result_path = registry.results / self.run_id

    def _running(self):
        if self.meta["status"] != "running" or read_json(self.path / "run.json")["status"] != "running":
            raise IntegrityError("Run is immutable after completion or failure")

    def update(self, **fields):
        self._running()
        if {"run_id", "level", "model_id", "status"} & fields.keys():
            raise ValueError("Run identity/status cannot be changed")
        self.meta.update(json_value(fields))
        write_json(self.path / "run.json", self.meta, exclusive=False)

    def bind_model(self, record):
        self._running()
        self.registry.load_model(record)
        expected_level = "L1" if self.meta["level"] == "L2" else self.meta["level"]
        if self.meta["model_id"] != record["model_id"] or expected_level != record["level"]:
            raise IntegrityError("Model belongs to another run identity or data level")
        if self.meta["level"] == "L2":
            _, selected, _, _ = self.registry.frozen_entry(self.meta["model_freeze_id"], self.meta["entry"])
            if record["model_hash"] != selected["model_hash"]:
                raise IntegrityError("L2 cannot replace or retrain the frozen model")
        self.update(**{k: record[k] for k in ("model_hash", "params_hash", "model_path", "model_record", "data_sha256")})

    def save_csv(self, name, frame):
        self._running()
        identifier(name)
        path = self.result_path / (name if name.endswith(".csv") else name + ".csv")
        self.result_path.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as f:
            # Every summary row keeps the relevant run/model/policy identities.
            result = frame.copy()
            for key in ("run_id", "model_id", "policy_id"):
                if key in self.meta and key not in result:
                    result[key] = self.meta[key]
            result.to_csv(f, index=False)

    def save_fig(self, name, fig):
        self._running()
        identifier(name)
        path = self.result_path / (name if Path(name).suffix else name + ".png")
        self.result_path.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            fig.savefig(f, format=path.suffix[1:])

    def finish(self):
        self._running()
        self.registry.load_run(self.run_id, allow_running=True)
        if "predictions_sha256" in self.meta and not self.meta.get("predictions_verified"):
            raise IntegrityError("Full-customer model reload verification is required")
        meta = dict(self.meta, status="complete", finished_at=now())
        self.result_path.mkdir(parents=True, exist_ok=True)
        write_json(self.result_path / "run.json", meta)
        write_json(self.path / "run.json", meta, exclusive=False)
        self.meta = meta

    def fail(self, reason):
        self._running()
        self.meta.update(status="failed", failure=str(reason), finished_at=now())
        write_json(self.path / "run.json", self.meta, exclusive=False)
