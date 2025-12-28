# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # ✅ 新增
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # ✅ 新增

# =========================
# 配置参数
# =========================
TRAIN_PATH = "delhi_climate_train.csv"
DATE_COL = "date"
TARGET_COL = "meantemp"
DIFF_ORDER = 1
LOG_EPS = 1e-8


# =========================

def load_timeseries(path: str, date_col: str, target_col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col)
    if df[date_col].duplicated().any():
        df = df[~df[date_col].duplicated(keep="first")]
    df = df.set_index(date_col)
    return pd.to_numeric(df[target_col], errors="coerce").dropna()


def safe_log(y: pd.Series, eps: float = 1e-8) -> pd.Series:
    min_val = float(y.min())
    if min_val <= 0:
        y = y + abs(min_val) + eps
    return np.log(y)


def difference(y: pd.Series, d: int = 1) -> pd.Series:
    z = y.copy()
    for _ in range(d):
        z = z.diff()
    return z.dropna()


def plot_acf_pacf(y: pd.Series, lags: int = 40):
    """绘制 ACF 和 PACF 图"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ACF
    plot_acf(y, lags=lags, ax=axes[0], title=f"ACF")
    # PACF
    plot_pacf(y, lags=lags, ax=axes[1], title=f"PACF")

    plt.tight_layout()
    plt.show()


# -------------------------
# 统计检验函数
# -------------------------
def adf_test(x: pd.Series) -> dict:
    res = adfuller(x, autolag="AIC")
    return {
        "test": "ADF (单位根检验)",
        "pvalue": res[1],
        "decision": "stationary" if res[1] <= 0.05 else "non-stationary"
    }


def kpss_test(x: pd.Series, regression: str = "c") -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pval, _, _ = kpss(x, regression=regression, nlags="auto")
    return {
        "test": f"KPSS ({'趋势' if regression == 'ct' else '水平'}平稳检验)",
        "pvalue": pval,
        "decision": "stationary" if pval > 0.05 else "non-stationary"
    }


def arch_lm_test(x: pd.Series, lags: int = 12) -> dict:
    lm_stat, lm_pval, _, _ = het_arch(x, nlags=lags)
    return {
        "test": f"ARCH-LM (异方差检验, lags={lags})",
        "pvalue": lm_pval,
        "decision": "heteroskedastic" if lm_pval <= 0.05 else "homoskedastic"
    }


def main():
    if not os.path.exists(TRAIN_PATH):
        print(f"错误: 找不到文件 {TRAIN_PATH}")
        return

    # 1. 加载与变换
    print(f"正在处理训练集: {TRAIN_PATH}...")
    y = load_timeseries(TRAIN_PATH, DATE_COL, TARGET_COL)
    y_log = safe_log(y, eps=LOG_EPS)
    y_log_diff = difference(y_log, d=DIFF_ORDER)

    # 2. 绘制 ACF / PACF
    # 这有助于确定 ARIMA 模型的 p 和 q 参数
    plot_acf_pacf(y_log_diff, lags=40)

    # 3. 统计检验
    results = [
        adf_test(y_log_diff),
        kpss_test(y_log_diff, regression="c"),
        kpss_test(y_log_diff, regression="ct"),
        arch_lm_test(y_log_diff, lags=12),
        arch_lm_test(y_log_diff, lags=24)
    ]

    # 4. 打印结果
    print("\n" + "=" * 50)
    print(f" 训练集检验结果: d{DIFF_ORDER}(log({TARGET_COL}))")
    print("=" * 50)
    for r in results:
        print(f"[{r['test']}]")
        print(f"  P-Value: {r['pvalue']:.4f}")
        print(f"  Decision: {r['decision']}")
        print("-" * 20)


if __name__ == "__main__":
    main()