"""
Simple-14days 增强版（Level Shift + Weekend Shift）
目标：在保持模型极简的前提下，增加 1~2 个有效特征，看看对每日7天预测是否有提升。
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


class Simple14EnhancedForecaster:
    """基于 Simple-14days 的轻量增强版本：

    特征：
      1. dayofweek：星期几（原始特征）
      2. level_shift：最近7天 vs 前7天 的整体均值差
      3. weekend_shift：最近14天内 周末均值 - 工作日均值
    """

    def __init__(self, window_days: int = 14, alpha: float = 0.4, beta: float = 0.3):
        self.window_days = window_days
        self.alpha = alpha  # level_shift 权重
        self.beta = beta    # weekend_shift 权重

    def _compute_features(self, df: pd.DataFrame):
        """基于最近 window_days 天的数据构建特征。"""
        recent = df.tail(self.window_days).copy()
        recent["dayofweek"] = recent["ds"].dt.dayofweek

        # 1. 按星期几的均值和标准差（原始 Simple-14days 特征）
        weekday_means = recent.groupby("dayofweek")["y"].mean()
        weekday_stds = recent.groupby("dayofweek")["y"].std()

        # 2. level_shift：最近7天 vs 前7天 的整体均值差
        if len(recent) >= 14:
            prev7 = recent.iloc[:7]["y"]
            last7 = recent.iloc[7:]["y"]
        else:
            # 数据不足14天时，退化为前半段、后半段
            half = len(recent) // 2
            prev7 = recent.iloc[:half]["y"]
            last7 = recent.iloc[half:]["y"]

        prev7_mean = prev7.mean()
        last7_mean = last7.mean()
        level_shift = last7_mean - prev7_mean

        # 3. weekend_shift：周末均值 - 工作日均值
        weekdays = recent[recent["dayofweek"] <= 4]["y"]
        weekends = recent[recent["dayofweek"] >= 5]["y"]
        if len(weekdays) > 0 and len(weekends) > 0:
            weekend_shift = weekends.mean() - weekdays.mean()
        else:
            weekend_shift = 0.0

        # 全局均值与标准差备用
        global_mean = recent["y"].mean()
        global_std = recent["y"].std()

        features = {
            "weekday_means": weekday_means,
            "weekday_stds": weekday_stds,
            "level_shift": level_shift,
            "weekend_shift": weekend_shift,
            "global_mean": global_mean,
            "global_std": global_std,
        }
        return features, recent

    def forecast_next_7days(self, df: pd.DataFrame):
        """以最近 window_days 天为训练数据，预测未来7天每日值。"""
        features, recent = self._compute_features(df)
        last_date = df["ds"].max()

        weekday_means = features["weekday_means"]
        weekday_stds = features["weekday_stds"]
        level_shift = features["level_shift"]
        weekend_shift = features["weekend_shift"]
        global_mean = features["global_mean"]
        global_std = features["global_std"]

        dates = []
        preds = []
        lowers = []
        uppers = []

        for k in range(1, 8):
            date = last_date + pd.Timedelta(days=k)
            dow = date.dayofweek

            # 基础预测：该星期几在最近14天的均值
            if dow in weekday_means.index:
                base = weekday_means[dow]
                std = weekday_stds[dow]
            else:
                base = global_mean
                std = global_std

            # Level shift：对所有天统一做一个偏移（同一个值）
            base += self.alpha * level_shift

            # Weekend shift：只对周六/周日做额外修正
            if dow >= 5:  # 5=周六, 6=周日
                base += self.beta * weekend_shift

            # 非负约束
            pred = max(0.0, base)

            # 置信区间：基于该dow历史 std，简单 ±1.5σ
            std_use = std if not np.isnan(std) and std > 0 else global_std
            if np.isnan(std_use) or std_use <= 0:
                std_use = 0.0
            lower = max(0.0, pred - 1.5 * std_use)
            upper = pred + 1.5 * std_use

            dates.append(date)
            preds.append(pred)
            lowers.append(lower)
            uppers.append(upper)

        daily_forecast = pd.DataFrame(
            {"date": dates, "prediction": preds, "lower": lowers, "upper": uppers}
        )
        return daily_forecast


def evaluate_on_range(df: pd.DataFrame, cutoff: str, days: int = 7):
    """在指定截止日前训练，之后连续 days 天做每日预测，比对真实值。

    按用户要求：使用 2025-10-18 ~ 2025-10-25 的真实数据进行逐日对比。
    """
    cutoff_date = pd.Timestamp(cutoff)
    df_train = df[df["ds"] <= cutoff_date].copy()
    df_test = df[(df["ds"] > cutoff_date) & (df["ds"] <= cutoff_date + pd.Timedelta(days=days))].copy()

    forecaster = Simple14EnhancedForecaster(window_days=14)
    daily_forecast = forecaster.forecast_next_7days(df_train)

    # 对齐前 days 天
    daily_forecast = daily_forecast.iloc[: len(df_test)].reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    print("=" * 80)
    print(f"增强版 Simple-14days 在区间 {df_test['ds'].min().date()} ~ {df_test['ds'].max().date()} 的每日表现")
    print("=" * 80)
    print(f"{'日期':<12} {'星期':<6} {'预测':<10} {'实际':<10} {'误差':<10} {'误差%':<10}")
    print("-" * 70)

    errors = []
    preds = []
    actuals = []

    for i in range(len(df_test)):
        date = daily_forecast.loc[i, "date"]
        pred = daily_forecast.loc[i, "prediction"]
        real = df_test.loc[i, "y"]
        err = pred - real
        err_pct = abs(err) / real * 100 if real > 0 else 0.0

        dow_name = weekday_names[date.dayofweek]
        print(
            f"{date.strftime('%Y-%m-%d'):<12} {dow_name:<6} "
            f"{pred:<10.1f} {real:<10.0f} {err:<10.1f} {err_pct:<10.1f}"
        )

        errors.append(abs(err))
        preds.append(pred)
        actuals.append(real)

    mae = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))) * 100

    print("-" * 70)
    print(f"日均 MAE:  {mae:.2f}")
    print(f"日均 MAPE: {mape:.2f}%")
    print(f"最大绝对误差: {max(errors):.1f}")
    print(f"最小绝对误差: {min(errors):.1f}")
    print()


def main():
    df = pd.read_csv("./rawdata/raw_data.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # 严格按你的验证要求：以 2025-10-17 为截止，预测 10-18 ~ 10-24 每日
    evaluate_on_range(df, cutoff="2025-10-17", days=7)


if __name__ == "__main__":
    main()
