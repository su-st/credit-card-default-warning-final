"""F4 shared signed One-Hot aggregation; no data is read on import."""
import pandas as pd
import xgboost as xgb


def shap_values(model, X, *, merge_onehot=True):
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    transformed = pre.transform(X)
    values = clf.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[:, :-1]
    names = pre.get_feature_names_out()
    result = pd.DataFrame(values, index=X.index, columns=names)
    if not merge_onehot:
        return result
    original = []
    for name in names:
        prefix, value = name.split("__", 1)
        if prefix == "cat":
            candidates = [c for c in X.columns if value.startswith(c + "_")]
            if not candidates:
                raise ValueError("Cannot map encoded feature to its original column")
            original.append(max(candidates, key=len))
        else:
            original.append(value)
    # Sum with sign first. Taking absolute values before grouping is incorrect.
    return result.T.groupby(original, sort=False).sum().T


def global_importance(shap_df):
    return shap_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
