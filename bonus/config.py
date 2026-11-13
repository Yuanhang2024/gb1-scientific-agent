# 扩展实验配置：明确数据数量、选样策略和查询预算，运行前统一检查。
"""Independent, explicit settings for CPU forest and acquisition experiments."""
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

GLOBAL_STRATEGIES = ("random", "greedy", "std_only", "mean_plus_std")
HD_STRATEGIES = ("random", "mean_plus_std")


# 这组配置只控制扩展实验；k为每轮查询数，hd_pool_size为每个突变数量组的候选库大小。
class BonusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 44
    seeds: list[int] = [44, 55, 66]
    train_size: int = Field(default=2000, ge=3)
    initial_single_count: int = Field(default=16, ge=1)
    calibration_size: int = Field(default=1000, ge=1)
    test_size: int = Field(default=5000, ge=1)
    reserve_single_count: int = Field(default=40, ge=1)
    holdout_max_attempts: int = Field(default=100, ge=1)
    hd_pool_size: int = Field(default=40, ge=1)
    rounds: int = Field(default=3, ge=1)
    k: int = Field(default=10, ge=1)
    rf_estimators: int = Field(default=100, ge=2)
    rf_min_samples_leaf: int = Field(default=1, ge=1)
    rf_max_depth: int | None = Field(default=None, ge=1)
    n_jobs: int = Field(default=4, ge=1)
    # 按样本标准差计算成员分歧；std_floor在尺度校准中防止除以0。
    std_ddof: Literal[1] = 1
    std_floor: float = Field(default=1e-6, gt=0)
    # alpha是目标未覆盖比例，1-alpha为标称覆盖率；kappa控制选样分数中分歧的权重。
    alpha: float = Field(default=0.1, gt=0, lt=1)
    kappa: float = Field(default=1.0, ge=0)
    prediction_batch_size: int = Field(default=8192, ge=1)
    save_pool_scores: bool = True
    global_strategies: list[Literal["random", "greedy", "std_only", "mean_plus_std"]] = list(GLOBAL_STRATEGIES)
    hd_strategies: list[Literal["random", "mean_plus_std"]] = list(HD_STRATEGIES)
    hd_groups: list[Literal[1, 2, 3, 4]] = [1, 2, 3, 4]

    # 候选库要足以支持各轮不重复查询，校准样本也要足以给出有限分位数。
    @model_validator(mode="after")
    def validate_budgets(self):
        if self.train_size <= 1 + self.initial_single_count:
            raise ValueError("Training requires WT, the specified singles and HD>=2 rows")
        if self.hd_pool_size < self.rounds * self.k or self.reserve_single_count < self.hd_pool_size:
            raise ValueError("HD pools and single reserve must satisfy the query budget")
        if math.ceil((self.calibration_size + 1) * (1 - self.alpha)) > self.calibration_size:
            raise ValueError("Calibration sample too small for a finite split-conformal quantile")
        for values in (self.seeds, self.global_strategies, self.hd_strategies, self.hd_groups):
            if not values or len(values) != len(set(values)):
                raise ValueError("Seed, strategy and HD lists must be nonempty and unique")
        return self
