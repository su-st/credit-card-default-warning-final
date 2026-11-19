"""C2 public experiment contract (final_spec sections 4.4–5).

L1 is the only selection data. Test access checks the leader's model freeze
before loading data. No data, model or frozen configuration is loaded on import.

cv_evaluate repeats overall mean/population standard deviation in each fold row.
tune_model's five Chinese keys and evaluate's columns are fixed below and tested.
load_rules returns attribute access; scenario is a Scenario. Formal L4 requires
search_space, n_iter, scenario, l4 and l4_configuration_hash in the frozen JSON.
l4 specifies model=M1-NS, outer_folds, inner_folds, seed, n_iter, search_space,
scenario; the duplicated settings must match. No Markdown values are parsed.

Runs persist fitted models using run.attach_model(model, X, y, oof_score), then
run.record_policy(policy) when a policy is used, save_csv/save_fig and finish.
This records actual artifacts rather than inventing a model hash from a name.
L3 entries additionally carry configuration (name, params, lag, sensitive,
engineered) and protocol_policy; run.evaluate_split retrains each seed using
those frozen values. summarize_l3 in the private backend returns a separate
aggregate with source_run_ids and no single model/policy hash.

G2b gate: python -m src.protocol --check-g2b. It must return zero before freezing;
the currently unmerged PR #9/07 migration is reported, never silently exempted.
"""
from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from scipy.stats import norm

from . import data as data_module
from .config import DATA_DIR, ROOT
from .features import add_available_features, engineered_cols
from .models import make_fold_model
from .registry import IntegrityError, Registry, content_hash, validate_ids


def _registry():
    return Registry(ROOT)


def _full_data():
    return data_module.load_data()


def _split(data, seed):
    return train_test_split(data, test_size=.2, random_state=seed, stratify=data["default"])


def _freeze(model_freeze_id):
    from .registry import read_json, identifier
    record = read_json(_registry().outputs / "frozen" / f"model-{identifier(model_freeze_id)}.json")
    if record["model_freeze_id"] != model_freeze_id or not record["approved_at"]:
        raise IntegrityError("Approved model freeze is required")
    _registry().load_rules(record["rules_freeze_id"])
    if record.get("schema_version") == 2:
        for row in record["entries"]:
            _registry().frozen_entry(model_freeze_id, row["entry_id"])
    return record


def dev_data():
    train, _ = data_module.split(_full_data())
    train.attrs.update(level="L1", seed=42)
    return train


def dev_holdout():
    train, hold = data_module.split(dev_data())
    train.attrs.update(level="L1-H", role="train", seed=42)
    hold.attrs.update(level="L1-H", role="holdout", seed=42)
    return train, hold


def final_test(model_freeze_id):
    _freeze(model_freeze_id)  # Must happen before any raw/test data is loaded.
    _, test = data_module.split(_full_data())
    test.attrs.update(level="L2", model_freeze_id=model_freeze_id, seed=42)
    return test


def outer_splits(model_freeze_id, seeds=range(42, 52)):
    _freeze(model_freeze_id)
    data = _full_data()
    for seed in seeds:
        train, test = _split(data, seed)
        for frame in (train, test):
            frame.attrs.update(level="L3", seed=int(seed), model_freeze_id=model_freeze_id)
        yield seed, train, test


def _lag(lag):
    if isinstance(lag, bool) or not isinstance(lag, int) or not 0 <= lag <= 5:
        raise ValueError("lag must be an integer from 0 through 5")


def bill_at(df, lag):
    _lag(lag)
    return df[f"BILL_AMT{lag+1}"].clip(lower=0)


def prepare(df, *, lag, sensitive, engineered):
    _lag(lag)
    if not isinstance(sensitive, bool) or not isinstance(engineered, bool):
        raise TypeError("sensitive and engineered must be explicit booleans")
    columns = data_module.feature_cols(lag=lag, sensitive=sensitive)
    X = df.copy()
    if engineered:
        # Do not pick up stale engineered columns belonging to unavailable months.
        X = X.drop(columns=engineered_cols(X))
        X = add_available_features(X, lag)
        columns += engineered_cols(X)
    X = X.loc[:, columns].copy()
    X.attrs.update(df.attrs)
    X.attrs.update(lag=lag, sensitive=sensitive, engineered=engineered)
    return X, columns


def make_model(name, *, params=None):
    return make_fold_model(name, params=params)


def _aligned(reference, *others):
    if not isinstance(reference, pd.Series):
        raise TypeError("Inputs must be indexed Series")
    validate_ids(reference.index)
    for other in others:
        if not isinstance(other, pd.Series) or not other.index.equals(reference.index):
            raise ValueError("Series must have identical ordered customer IDs")


def _labels(y):
    if not y.isin([0, 1]).all():
        raise ValueError("Labels must be binary and non-null")


def _cv(X, y, folds, seed):
    if not X.index.equals(y.index):
        raise ValueError("X/y customer IDs differ")
    validate_ids(X.index)
    _labels(y)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2 or y.nunique() != 2 or y.value_counts().min() < folds:
        raise ValueError("Each class needs at least folds observations")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def oof_predict(model_fn, X, y, *, folds=5, seed=42):
    scores = pd.Series(np.nan, index=X.index, name="oof_score")
    assignment = pd.Series(-1, index=X.index, dtype=int)
    sources = []
    for fold, (tr, va) in enumerate(_cv(X, y, folds, seed).split(X, y)):
        model = model_fn()
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.iloc[va] = model.predict_proba(X.iloc[va])[:, 1]
        assignment.iloc[va] = fold
        sources.append({"fold": fold, "training_ids_hash": content_hash(X.index[tr].tolist()),
                        "validation_ids_hash": content_hash(X.index[va].tolist())})
    scores.attrs.update(seed=seed, folds=folds, fold_assignment=assignment,
                        fold_sources=sources, folds_hash=content_hash(assignment.tolist()))
    return scores


def cv_evaluate(model_fn, X, y, *, folds=5, seed=42):
    rows = []
    for fold, (tr, va) in enumerate(_cv(X, y, folds, seed).split(X, y)):
        model = model_fn()
        model.fit(X.iloc[tr], y.iloc[tr])
        rows.append({"折編號": fold, "AUC": roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1])})
    result = pd.DataFrame(rows)
    result["平均"] = result["AUC"].mean()
    result["標準差"] = result["AUC"].std(ddof=0)
    return result


def tune_model(X, y, search_space, *, n_iter, folds=5, seed=42):
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be positive")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a nonempty mapping of finite choices")
    if any(not isinstance(v, (list, tuple)) or not v for v in search_space.values()):
        raise ValueError("Search choices must be nonempty finite lists")
    if "scale_pos_weight" in search_space:
        raise ValueError("Class weights are computed inside each fit")
    size = math.prod(len(v) for v in search_space.values())
    if n_iter > size:
        raise ValueError("Requested search count exceeds the finite search space")
    rows = []
    for iteration, params in enumerate(ParameterSampler(search_space, n_iter=n_iter, random_state=seed)):
        params = dict(params)
        params.setdefault("random_state", seed)
        cv = cv_evaluate(lambda: make_model("XGBoost", params=params), X, y, folds=folds, seed=seed)
        rows.append(dict(params=params, iteration=iteration, cv_auc_mean=float(cv["AUC"].mean()),
                         cv_auc_std=float(cv["AUC"].std(ddof=0)), **params))
    history = pd.DataFrame(rows)
    best = history.loc[history["cv_auc_mean"].idxmax()]
    return {"最佳參數": dict(best["params"]), "最佳 CV AUC": float(best["cv_auc_mean"]),
            "完整搜尋紀錄": history, "實際搜尋組數": len(history), "搜尋範圍雜湊": content_hash(search_space)}


@dataclass(frozen=True)
class Scenario:
    max_list_rate: float
    loss_ratio: float
    contact_cost: float
    rescue_rate: float

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.max_list_rate > 1 or self.rescue_rate > 1 or self.loss_ratio > 1:
            raise ValueError("Rates must not exceed one")


@dataclass(frozen=True)
class Policy:
    kind: str
    threshold: float | None = None
    max_list_rate: float | None = None
    partition_name: str | None = None
    thresholds: dict | None = None
    rate: float | None = None
    selection: str = "fixed"
    scenario: dict | None = None
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"single", "group", "top_n"}:
            raise ValueError("Unknown policy kind")
        for value in (self.threshold, self.max_list_rate, self.rate):
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Threshold/capacity/rate must be in [0, 1]")
        if self.kind == "single" and self.threshold is None:
            raise ValueError("Single policy requires a threshold")
        if self.kind == "group":
            if not self.partition_name or not isinstance(self.thresholds, dict) or not self.thresholds:
                raise ValueError("Group policy requires a complete partition and thresholds")
            if any(not np.isfinite(t) or not 0 <= t <= 1 for t in self.thresholds.values()):
                raise ValueError("Invalid group threshold")
            normalized = {str(k): float(v) for k, v in self.thresholds.items()}
            if len(normalized) != len(self.thresholds):
                raise ValueError("Ambiguous group labels")
            object.__setattr__(self, "thresholds", normalized)
        if self.kind == "top_n" and self.rate is None:
            raise ValueError("Top-N policy requires a rate")
        object.__setattr__(self, "thresholds", deepcopy(self.thresholds))
        object.__setattr__(self, "evidence", deepcopy(self.evidence))
        object.__setattr__(self, "scenario", deepcopy(self.scenario))

    @staticmethod
    def single(threshold, max_list_rate):
        return Policy("single", threshold=threshold, max_list_rate=max_list_rate)

    @staticmethod
    def group(partition_name, thresholds, max_list_rate):
        return Policy("group", partition_name=partition_name, thresholds=thresholds, max_list_rate=max_list_rate,
                      selection="group-recall")

    @staticmethod
    def top_n(rate):
        return Policy("top_n", rate=rate)

    @property
    def policy_hash(self):
        return content_hash(asdict(self))

    @property
    def policy_id(self):
        return f"{self.kind}-{self.policy_hash[:16]}"


def _rank(score):
    frame = pd.DataFrame({"score": score.to_numpy(), "customer_id": score.index}, index=score.index)
    frame.index.name = None
    return frame.sort_values(["score", "customer_id"], ascending=[False, True]).index


def apply_policy(score, policy, *, partition=None):
    _aligned(score)
    if not isinstance(policy, Policy):
        raise TypeError("policy must be a Policy")
    if not np.isfinite(score.to_numpy()).all() or not score.between(0, 1).all():
        raise ValueError("Invalid risk scores")
    if policy.kind == "group":
        if partition is None:
            raise ValueError("Group policy requires partition")
        _aligned(score, partition)
        labels = partition.astype(str)
        if partition.isna().any() or labels.nunique() != partition.nunique() or set(labels.unique()) != set(policy.thresholds):
            raise ValueError("Partition must be mutually exclusive, complete and match all threshold groups")
        raw = score >= labels.map(policy.thresholds)
    elif policy.kind == "single":
        raw = score >= policy.threshold
    else:
        raw = pd.Series(score.index.isin(_rank(score)[:math.floor(policy.rate * len(score))]), index=score.index)
    final = raw.copy()
    triggered = False
    if policy.max_list_rate is not None and int(raw.sum()) > math.floor(policy.max_list_rate * len(score)):
        triggered = True
        keep = _rank(score.loc[raw])[:math.floor(policy.max_list_rate * len(score))]
        final = pd.Series(score.index.isin(keep), index=score.index)
    return final.astype(int), {"pred_raw": raw.astype(int), "原始名單人數": int(raw.sum()), "名單上限是否觸發": triggered}


def _total_cost(y, pred, bill, scenario):
    loss = bill.to_numpy() * scenario.loss_ratio
    flag, actual = pred.to_numpy().astype(bool), y.to_numpy().astype(bool)
    return float(loss[actual & ~flag].sum() + (loss[actual & flag] * (1 - scenario.rescue_rate)).sum()
                 + flag.sum() * scenario.contact_cost)


def _bills(y, bill):
    _aligned(y, bill)
    if not np.isfinite(bill.to_numpy()).all() or (bill < 0).any():
        raise ValueError("bill must be finite and nonnegative; use bill_at")


def choose_threshold(y, score, bill, scenario):
    _aligned(y, score)
    _labels(y)
    _bills(y, bill)
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be Scenario")
    best = None
    for t in np.round(np.arange(.05, .951, .01), 2):
        policy = Policy.single(float(t), scenario.max_list_rate)
        pred, info = apply_policy(score, policy)
        cost = _total_cost(y, pred, bill, scenario)
        if best is None or cost < best[0] or (cost == best[0] and t > best[1]):
            best = (cost, float(t), pred, info)
    cost, threshold, pred, info = best
    evidence = {"training_ids_hash": content_hash(y.index.tolist()),
                "oof_scores_hash": content_hash(score.tolist())}
    policy = replace(Policy.single(threshold, scenario.max_list_rate), selection="cost",
                     scenario=asdict(scenario), evidence=evidence)
    return {"政策物件": policy, "門檻": threshold, "最終名單人數": int(pred.sum()),
            "最終名單比例": float(pred.mean()), "原始名單人數": info["原始名單人數"],
            "上限是否觸發": info["名單上限是否觸發"], "總成本": cost, "情境": scenario}


def coverage_curve(y, score, rates, bill=None):
    _aligned(y, score)
    _labels(y)
    if bill is not None:
        _bills(y, bill)
    rows = []
    for rate in rates:
        pred, _ = apply_policy(score, Policy.top_n(rate))
        actual = y == 1
        rows.append({"名單比例": rate, "名單人數": int(pred.sum()),
                     "違約者涵蓋率": float(pred[actual].sum() / actual.sum()) if actual.sum() else np.nan,
                     "帳單金額涵蓋率": float(bill[actual & pred.astype(bool)].sum() / bill[actual].sum())
                     if bill is not None and bill[actual].sum() > 0 else np.nan})
    return pd.DataFrame(rows, columns=["名單比例", "名單人數", "違約者涵蓋率", "帳單金額涵蓋率"])


OVERALL_COLUMNS = ["policy_id", "AUC", "Precision", "Recall", "F1", "名單人數", "名單比例", "原始名單人數",
    "名單上限是否觸發", "誤報人數", "漏判人數", "違約者涵蓋率", "帳單金額涵蓋率", "總成本"]
GROUP_COLUMNS = ["分組方式", "族群", "總人數", "違約人數", "未違約人數", "Recall", "Recall 下限", "Recall 上限", "FNR",
    "FPR", "FPR 下限", "FPR 上限", "小樣本"]
GAP_COLUMNS = ["分組方式", "EO 差距", "FPR 差距"]


def _wilson(numerator, denominator):
    if not denominator:
        return np.nan, np.nan
    z = norm.ppf(.975)
    p = numerator / denominator
    center = (p + z*z / (2*denominator)) / (1 + z*z / denominator)
    half = z * np.sqrt(p*(1-p)/denominator + z*z/(4*denominator**2)) / (1 + z*z/denominator)
    return max(0., float(center-half)), min(1., float(center+half))


def _groups(y, pred, groups):
    rows, gaps = [], []
    for name, partition in (groups or {}).items():
        _aligned(y, partition)
        if partition.isna().any():
            raise ValueError("Group labels must be complete")
        start = len(rows)
        for group in pd.unique(partition):
            idx = partition == group
            actual, flag = y[idx], pred[idx].astype(bool)
            positives, negatives = int((actual == 1).sum()), int((actual == 0).sum())
            tp = int((flag & (actual == 1)).sum())
            fp = int((flag & (actual == 0)).sum())
            recall = tp / positives if positives else np.nan
            fpr = fp / negatives if negatives else np.nan
            rl, ru = _wilson(tp, positives)
            fl, fu = _wilson(fp, negatives)
            rows.append(dict(zip(GROUP_COLUMNS, [name, group, len(actual), positives, negatives,
                recall, rl, ru, 1-recall, fpr, fl, fu, positives < 93 or negatives < 93])))
        subset = pd.DataFrame(rows[start:])
        valid_recall, valid_fpr = subset["Recall"].dropna(), subset["FPR"].dropna()
        gaps.append({"分組方式": name, "EO 差距": float(valid_recall.max()-valid_recall.min()) if len(valid_recall) else np.nan,
                     "FPR 差距": float(valid_fpr.max()-valid_fpr.min()) if len(valid_fpr) else np.nan})
    return pd.DataFrame(rows, columns=GROUP_COLUMNS), pd.DataFrame(gaps, columns=GAP_COLUMNS)


def evaluate(y, *, score=None, policy=None, pred=None, partition=None, groups=None, bill=None, scenario=None):
    _aligned(y)
    _labels(y)
    if not isinstance(policy, Policy) or (score is None and pred is None):
        raise ValueError("Use score+policy, score+pred+policy, or pred+policy")
    if policy.kind == "group":
        if partition is None:
            raise ValueError("Group policy requires partition")
        _aligned(y, partition)
        if partition.isna().any() or set(partition.astype(str).unique()) != set(policy.thresholds):
            raise ValueError("Invalid group partition")
    info = {"原始名單人數": np.nan, "名單上限是否觸發": np.nan}
    if score is not None:
        _aligned(y, score)
        expected, info = apply_policy(score, policy, partition=partition)
        if pred is not None:
            _aligned(y, pred)
            if not pred.isin([0, 1]).all() or not np.array_equal(pred.to_numpy(), expected.to_numpy()):
                raise ValueError("External pred disagrees with apply_policy")
        pred = expected
    else:
        _aligned(y, pred)
        if not pred.isin([0, 1]).all():
            raise ValueError("Predictions must be binary")
    actual, flag = y == 1, pred.astype(bool)
    tp, fp, fn = int((actual & flag).sum()), int((~actual & flag).sum()), int((actual & ~flag).sum())
    precision = tp/(tp+fp) if tp+fp else np.nan
    recall = tp/(tp+fn) if tp+fn else np.nan
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
    amount_coverage = np.nan
    if bill is not None:
        _bills(y, bill)
        if scenario is not None and bill[actual].sum() > 0:
            amount_coverage = float(bill[actual & flag].sum()/bill[actual].sum())
    if scenario is not None and bill is None:
        raise ValueError("Cost evaluation requires bill")
    overall = dict(zip(OVERALL_COLUMNS, [policy.policy_id,
        float(roc_auc_score(y, score)) if score is not None and y.nunique() == 2 else np.nan,
        precision, recall, f1, int(flag.sum()), float(flag.mean()), info["原始名單人數"],
        info["名單上限是否觸發"], fp, fn, recall, amount_coverage,
        _total_cost(y, pred, bill, scenario) if bill is not None and scenario is not None else np.nan]))
    per_group, gaps = _groups(y, pred, groups)
    return {"overall": pd.DataFrame([overall], columns=OVERALL_COLUMNS), "groups": per_group, "gaps": gaps}


def gap_ci(y, pred, partition, *, n_boot=2000, seed=42, pred_b=None):
    _aligned(y, pred, partition)
    _labels(y)
    if not pred.isin([0, 1]).all() or partition.isna().any():
        raise ValueError("Complete partition and binary predictions required")
    if pred_b is not None:
        _aligned(y, pred_b)
        if not pred_b.isin([0, 1]).all():
            raise ValueError("pred_b must be binary")
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be positive")
    strata = []
    for group in pd.unique(partition):
        for label in (0, 1):
            pos = np.flatnonzero(((partition == group) & (y == label)).to_numpy())
            if len(pos):
                strata.append(pos)
    def gap(positions, flags):
        yy = pd.Series(y.to_numpy()[positions], index=range(1, len(positions)+1))
        pp = pd.Series(flags.to_numpy()[positions], index=yy.index)
        gg = pd.Series(partition.to_numpy()[positions], index=yy.index)
        _, result = _groups(yy, pp, {"partition": gg})
        return result.iloc[0][["EO 差距", "FPR 差距"]].to_numpy(dtype=float)
    positions = np.arange(len(y))
    estimate = gap(positions, pred)
    if pred_b is not None:
        estimate = gap(positions, pred_b) - estimate
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        value = gap(idx, pred)
        if pred_b is not None:
            value = gap(idx, pred_b) - value
        samples.append(value)
    values = np.asarray(samples)
    rows = []
    for col, name in enumerate(["EO 差距", "FPR 差距"]):
        valid = values[np.isfinite(values[:, col]), col]
        reliable = len(valid) >= math.ceil(.95*n_boot)
        lo, hi = np.percentile(valid, [2.5, 97.5]) if reliable else (np.nan, np.nan)
        rows.append({"指標": name, "估計值": estimate[col], "下限": lo, "上限": hi,
                     "有效重抽": len(valid), "無效重抽": n_boot-len(valid), "可靠": reliable,
                     "比較": "B-A" if pred_b is not None else "單一名單"})
    return pd.DataFrame(rows)


def load_rules(rules_freeze_id):
    values = deepcopy(_registry().load_rules(rules_freeze_id))
    values["scenario"] = Scenario(**values["scenario"])
    return SimpleNamespace(**values)


def start_run(level, description, *, model_id=None, rules_freeze_id=None, model_freeze_id=None, entry=None):
    from ._protocol_runs import start_protocol_run
    return start_protocol_run(level, description, model_id=model_id, rules_freeze_id=rules_freeze_id,
                              model_freeze_id=model_freeze_id, entry=entry)


def nested_eval(rules_freeze_id):
    from ._sealed import seal_nested
    return seal_nested(rules_freeze_id)


def open_sealed(rules_freeze_id, model_freeze_id):
    from ._sealed import open_nested
    return open_nested(rules_freeze_id, model_freeze_id)


def audit_sources(root=ROOT):
    """Static G2b gate: find direct test/raw/sealed access in experiment code.

    Data primitives, the historical L0 entry and guarded protocol backends are
    the only low-level readers. This is an auditable process check, not a Python
    sandbox. The separately callable gate must be empty before G2b.
    """
    import ast
    root = Path(root)
    allowed = {'src/data.py', 'src/protocol.py', 'src/_sealed.py', 'src/_protocol_runs.py',
               'src/registry.py', 'experiments/00_reproduce_midterm.py'}
    findings = []
    for directory in ('src', 'experiments'):
        for file in (root/directory).glob('*.py'):
            relative = file.relative_to(root).as_posix()
            if relative in allowed:
                continue
            tree = ast.parse(file.read_text(encoding='utf-8'))
            raw_aliases = {'load_data', 'load_raw', 'split', 'train_test_split'}
            raw_modules = {'src.data'}
            builder_forbidden = {'final_test', 'outer_splits', 'nested_eval', 'open_sealed', 'dev_holdout', '_full_data'}
            formal_readers = {'final_test': 'src/final_eval.py', 'outer_splits': 'src/stability_eval.py'}
            readers = {'read_csv', 'read_json', 'read_table', 'read_parquet', 'loadtxt', 'fromfile'}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in {'src.protocol', 'protocol'}:
                    builder_forbidden.update(a.asname or a.name for a in node.names if a.name in builder_forbidden)
                    formal_readers.update({a.asname or a.name: formal_readers[a.name]
                                           for a in node.names if a.name in formal_readers})
                if isinstance(node, ast.ImportFrom) and node.module in {'src.data', 'data', 'sklearn.model_selection'}:
                    raw_aliases.update(a.asname or a.name for a in node.names if a.name in raw_aliases)
                if isinstance(node, ast.ImportFrom) and node.module in {'pandas', 'numpy'}:
                    readers.update(a.asname or a.name for a in node.names if a.name in readers)
                if isinstance(node, ast.ImportFrom) and node.module == 'src':
                    raw_modules.update(a.asname or a.name for a in node.names if a.name == 'data')
                if isinstance(node, ast.Import):
                    raw_modules.update(a.asname or a.name for a in node.names if a.name == 'src.data')
            unsafe_paths = set()
            def unsafe(node):
                return any(isinstance(s,ast.Name) and s.id in unsafe_paths or
                    isinstance(s,ast.Constant) and isinstance(s.value,str) and ('sealed' in s.value or 'predictions.csv' in s.value)
                    for s in ast.walk(node))
            for _ in range(3):
                for node in ast.walk(tree):
                    if isinstance(node,ast.Assign) and unsafe(node.value):
                        unsafe_paths.update(t.id for target in node.targets for t in ast.walk(target) if isinstance(t,ast.Name))
            for node in ast.walk(tree):
                reason = None
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
                    if name in raw_aliases and (isinstance(node.func,ast.Name) or
                            isinstance(node.func,ast.Attribute) and ast.unparse(node.func.value) in raw_modules):
                        reason = 'direct data/test split bypasses protocol'
                    if name in readers and any(unsafe(arg) for arg in node.args):
                        reason = 'direct sealed/legacy handoff read'
                    if relative in {'src/model_freeze.py', 'experiments/00_build_model_freeze.py'} and name in builder_forbidden:
                        reason = 'model-freeze builder may only obtain L1 dev_data'
                    if name in formal_readers and relative != formal_readers[name]:
                        reason = 'formal data loader is restricted to its guarded evaluation entry'
                    if name == 'load_dotenv':
                        reason = 'experiment reads .env instead of process configuration'
                if relative.startswith('experiments/') and isinstance(node, ast.Name) and isinstance(node.ctx,ast.Load) and node.id in {'test','y_test'}:
                    reason = 'unmigrated test-set global'
                if reason:
                    findings.append({'file':relative,'line':node.lineno,'reason':reason})
    return sorted(findings,key=lambda r:(r['file'],r['line'],r['reason']))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='C2 data isolation gate')
    parser.add_argument('--check-g2b',action='store_true',required=True)
    parser.parse_args()
    violations = audit_sources()
    for item in violations:
        print(f"{item['file']}:{item['line']}: {item['reason']}")
    raise SystemExit(1 if violations else 0)
