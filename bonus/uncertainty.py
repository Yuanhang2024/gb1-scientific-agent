# 不确定性评价：先计算森林成员的预测分歧，再用独立样本校准预测区间。
"""Forest disagreement and independent absolute/scaled split-conformal evaluation."""
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from gb1.data import one_hot
from gb1.models import regression_metrics


# 只用当前known训练森林，校准和测试标签不进入fit。
def fit_forest(known, config):
    model = RandomForestRegressor(n_estimators=config.rf_estimators, min_samples_leaf=config.rf_min_samples_leaf,
                                  max_depth=config.rf_max_depth, random_state=config.seed, n_jobs=config.n_jobs)
    model.fit(one_hot(known.variant), known.fitness.to_numpy(dtype=float))
    return model


# 分批预测控制内存占用；每棵树给出一个估计，成员差异作为分歧分数。
def predict_distribution(model, variants, batch_size=8192):
    variants = list(variants)
    if len(model.estimators_) < 2:
        raise ValueError("Sample tree disagreement requires at least two fitted trees")
    means, deviations = [], []
    for start in range(0, len(variants), batch_size):
        encoded = one_hot(variants[start:start + batch_size])
        # 矩阵每行是一棵树、每列是一个候选；沿axis=0计算均值和样本标准差。
        members = np.stack([tree.predict(encoded) for tree in model.estimators_], axis=0)
        means.append(members.mean(axis=0))
        deviations.append(members.std(axis=0, ddof=1))
    if not means:
        return np.array([], dtype=float), np.array([], dtype=float)
    mean, std = np.concatenate(means), np.concatenate(deviations)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Forest predictions and disagreement must be finite")
    return mean, std


# 有限样本校准取排序后的第ceil((n+1)*(1-alpha))项；它对应数组下标index-1。
def finite_quantile(scores, alpha):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Calibration scores must be a nonempty finite nonnegative vector")
    index = math.ceil((len(values) + 1) * (1 - alpha))
    if index > len(values) or index < 1:
        raise ValueError("Calibration sample too small for a finite split-conformal quantile")
    return float(np.partition(values, index - 1)[index - 1]), index


# 分别校准绝对残差和按分歧缩放的残差；两个分位数都来自独立校准集。
def calibrate(actual, mean, std, config):
    actual, mean, std = (np.asarray(values, dtype=float) for values in (actual, mean, std))
    if actual.shape != mean.shape or mean.shape != std.shape or (std < 0).any():
        raise ValueError("Calibration arrays must have matching shapes and nonnegative disagreement")
    residual = np.abs(actual - mean)
    q_abs, index = finite_quantile(residual, config.alpha)
    q_scaled, _ = finite_quantile(residual / np.maximum(std, config.std_floor), config.alpha)
    return {"n_calibration": len(actual), "alpha": config.alpha, "nominal_coverage": 1 - config.alpha,
            "order_index": index, "q_abs": q_abs, "q_scaled": q_scaled,
            "std_floor": config.std_floor, "std_ddof": config.std_ddof,
            "uncertainty_meaning": "Sample standard deviation across fitted trees; a disagreement score, not a posterior confidence interval",
            "coverage_scope": "Random held-out marginal evaluation; adaptive selected-point coverage is empirical only"}


# 用校准阈值扩展预测均值；S1分数非负，因此将区间下界截到0。
def interval_bounds(mean, std, calibration):
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    half_scaled = calibration["q_scaled"] * np.maximum(std, calibration["std_floor"])
    return {"abs_lower": np.maximum(0, mean - calibration["q_abs"]), "abs_upper": mean + calibration["q_abs"],
            "scaled_lower": np.maximum(0, mean - half_scaled), "scaled_upper": mean + half_scaled}


# 覆盖率说明实际值落入区间的比例，宽度说明区间有多宽，两者需要一起看。
def evaluate_predictions(actual, mean, std, bounds):
    actual, mean, std = (np.asarray(values, dtype=float) for values in (actual, mean, std))
    fields = ("spearman", "pearson", "mse", "mae", "abs_error_std_spearman", "mean_tree_disagreement_std",
              "abs_coverage", "abs_mean_width", "scaled_coverage", "scaled_mean_width")
    if len(actual) == 0:
        return {"n": 0, **{field: None for field in fields}}
    core = regression_metrics(actual, mean)
    error = np.abs(actual - mean)
    correlation = float(spearmanr(error, std).statistic) if len(actual) > 1 and np.ptp(error) > 0 and np.ptp(std) > 0 else None
    result = {"n": len(actual), **{key: core[key] for key in ("spearman", "pearson", "mse")},
              "mae": float(error.mean()), "abs_error_std_spearman": correlation,
              "mean_tree_disagreement_std": float(std.mean())}
    for kind in ("abs", "scaled"):
        lower, upper = np.asarray(bounds[f"{kind}_lower"]), np.asarray(bounds[f"{kind}_upper"])
        result[f"{kind}_coverage"] = float(((actual >= lower) & (actual <= upper)).mean())
        result[f"{kind}_mean_width"] = float((upper - lower).mean())
    return result


# 对已训练模型做校准和留出评价；这些标签不传给候选选择函数。
# 随机留出的覆盖结果与主动选中样本的经验覆盖应分别解释。
def evaluate_frozen(model, dataset, config):
    """Invoked after acquisition is frozen; these labels never enter the ranking interface."""
    calibration_rows = pd.read_csv(dataset.directory / "hidden/calibration.csv")
    test_rows = pd.read_csv(dataset.directory / "hidden/test.csv")
    role_ids = [set(rows.variant) for rows in (dataset.train, dataset.candidates, calibration_rows, test_rows)]
    if any(role_ids[i] & role_ids[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Calibration/test data overlap another role")
    for rows in (calibration_rows, test_rows):
        if rows.variant.duplicated().any() or not np.isfinite(rows.fitness.to_numpy(dtype=float)).all():
            raise ValueError("Held-out rows must have unique variants and finite labels")
    if len(calibration_rows) != config.calibration_size or len(test_rows) != config.test_size:
        raise ValueError("Held-out sample counts differ from the fixed configuration")
    cal_mean, cal_std = predict_distribution(model, calibration_rows.variant, config.prediction_batch_size)
    calibration = calibrate(calibration_rows.fitness, cal_mean, cal_std, config)
    test_mean, test_std = predict_distribution(model, test_rows.variant, config.prediction_batch_size)
    bounds = interval_bounds(test_mean, test_std, calibration)
    actual = test_rows.fitness.to_numpy(dtype=float)
    hd = np.array([sum(a != b for a, b in zip(v, "VDGV")) for v in test_rows.variant])
    report = {"overall": evaluate_predictions(actual, test_mean, test_std, bounds), "by_hd": {}}
    for group in range(5):
        mask = hd == group
        report["by_hd"][str(group)] = evaluate_predictions(actual[mask], test_mean[mask], test_std[mask],
                                                          {name: values[mask] for name, values in bounds.items()})
    predictions = pd.DataFrame({"variant": test_rows.variant, "mutation_count": hd, "actual_fitness": actual,
                                "predicted_fitness": test_mean, "tree_disagreement_std": test_std, **bounds})
    return calibration, report, predictions
