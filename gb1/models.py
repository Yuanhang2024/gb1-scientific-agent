# 适应度预测器：学习四个位点的氨基酸组合与实验fitness之间的关系。
"""Fixed-feature model comparison and repeated fitting of one selected family."""
from pathlib import Path
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from .data import one_hot
from .storage import read_json, write_json
from .schema import compatible_config


# Spearman评价排序，Pearson评价线性关联，MSE评价数值误差。
# 样本太少或分数恒定时，相关系数记为None。
def regression_metrics(actual, predicted, k=10):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(actual) == 0 or actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require equal, finite, nonempty arrays")
    correlations_defined = len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0
    k = min(k, len(actual))
    # Top-k命中率是预测前k与实测前k的交集数量除以k；并列时保持输入顺序。
    top_actual = set(np.argsort(-actual, kind="stable")[:k].tolist())
    top_predicted = set(np.argsort(-predicted, kind="stable")[:k].tolist())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"n": len(actual), "spearman": float(spearmanr(actual, predicted).statistic) if correlations_defined else None,
                "pearson": float(pearsonr(actual, predicted).statistic) if correlations_defined else None,
                "mse": float(mean_squared_error(actual, predicted)), "top_k": k,
                "top_k_hit_rate": len(top_actual & top_predicted) / k}


# 按已声明类别与参数创建回归器；树模型使用实验配置中的随机种子。
def estimator(spec, config):
    params = dict(spec["parameters"])
    if spec["family"] == "ridge":
        return Ridge(**params)
    params.update(random_state=config.model_seed if config.model_seed is not None else config.seed, n_jobs=config.n_jobs)
    if spec["family"] == "random_forest":
        return RandomForestRegressor(**params)
    if spec["family"] == "extra_trees":
        return ExtraTreesRegressor(**params)
    raise ValueError("Unknown model family")


# 展开待比较的参数组合；所有组合使用相同训练集和验证集。
def model_specs(config):
    result = []
    for family in config.model_families:
        if family == "ridge":
            result.extend({"family": family, "parameters": {"alpha": alpha}} for alpha in config.ridge_alphas)
        else:
            for leaf in config.tree_min_samples_leaf:
                result.append({"family": family, "parameters": {"n_estimators": config.tree_estimators,
                              "min_samples_leaf": leaf, "max_depth": config.tree_max_depth}})
    return result


# X为四位点one-hot，y为已知fitness；每次调用都会重新训练一个模型。
def fit_model(spec, config, known):
    model = estimator(spec, config)
    model.fit(one_hot(known.variant), known.fitness.to_numpy(dtype=float))
    return model


# 只用验证集选择模型；候选池标签和最终测试标签不参与选优。
def select_model(dataset, config, output):
    output = Path(output)
    if (output / "selection.json").exists():
        raise ValueError("Selection already frozen; use the existing selection or a new output directory")
    validation = pd.read_csv(dataset.directory / "validation.csv")
    if set(validation.variant) & (set(dataset.train.variant) | set(dataset.candidates.variant)):
        raise ValueError("Validation overlaps training or candidate rows")
    results, trained = [], []
    for index, spec in enumerate(model_specs(config)):
        start = time.perf_counter()
        model = fit_model(spec, config, dataset.train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        metrics = regression_metrics(validation.fitness, model.predict(one_hot(validation.variant)), config.top_k_metric)
        results.append({"index": index, **spec, **metrics, "fit_seconds": fit_seconds,
                        "predict_seconds": time.perf_counter() - start})
        trained.append(model if config.save_predictors else None)
    # 先比验证集排序相关，再比Top-k命中率和MSE；完全相同时按配置顺序确定结果。
    winner = max(results, key=lambda row: (row["spearman"] if row["spearman"] is not None else -2,
                                           row["top_k_hit_rate"], -row["mse"], -row["index"]))
    spec = {"family": winner["family"], "parameters": winner["parameters"]}
    output.mkdir(parents=True, exist_ok=True)
    if config.save_predictors:
        joblib.dump(trained[winner["index"]], output / "initial_model.joblib")
    selection = {"format_version": 1, "dataset_manifest": dataset.manifest,
                 "config": config.model_dump(), "spec": spec, "comparison": results,
                 "selection_rule": "validation Spearman, top-k hit rate, lower MSE, declared candidate order",
                 "top_k_definition": "overlap of predicted top k and true top k divided by k; stable input-order ties",
                 "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}}
    write_json(output / "selection.json", selection)
    pd.json_normalize(results).to_csv(output / "model_comparison.csv", index=False)
    return selection


# 后续轮次沿用已选类别和超参数，配置或数据不同就要求新建实验。
def load_selection(path, dataset, config):
    selection = read_json(path)
    if not compatible_config(selection["config"], config) or selection["dataset_manifest"] != dataset.manifest:
        raise ValueError("Model selection config or dataset differs from this experiment")
    return selection
