# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

# 忽略模型不收敛等警告
warnings.filterwarnings("ignore")

# =========================
# 1. 配置与路径
# =========================
TRAIN_PATH = "delhi_climate_train.csv"
DATE_COL = "date"
TARGET_COL = "meantemp"
OUTPUT_DIR = Path("Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 数据转换函数
# =========================
def load_and_transform(path: str):
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
    df.set_index(DATE_COL, inplace=True)
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

    # 手动执行 Log + Diff
    # 注意：这里我们只做变换，不修改原始 y
    y_log = np.log(y + 1e-8)
    y_log_diff = y_log.diff().dropna()

    return y_log_diff


# =========================
# 3. 执行主流程
# =========================
def main():
    # A. 准备处理后的平稳序列
    # 这里的 processed_data 是 log(y) 的一阶差分
    processed_data = load_and_transform(TRAIN_PATH)
    print(f"数据预处理完成 (Log + Diff)，有效观测数: {len(processed_data)}")

    # B. 网格搜索 (p: 0-30, q: 0-30)
    # ⚠️ 警告：范围较大，计算时间会较长
    p_values = range(0, 10)
    q_values = range(0, 10)
    results_list = []

    print(f"\n开始网格搜索 ARIMA(p, 0, q)... 预计总任务数: {len(p_values) * len(q_values)}")

    for p in p_values:
        for q in q_values:
            try:
                # 关键修改：对已差分数据做拟合，d=0
                model = ARIMA(processed_data, order=(p, 0, q))
                res = model.fit()
                results_list.append({
                    'p': p, 'q': q,
                    'AIC': res.aic, 'BIC': res.bic,
                    'order': (p, 0, q)
                })
            except Exception as e:
                # 捕获高阶模型可能不收敛的情况
                continue

    # C. 筛选最优模型 (按 BIC)
    results_df = pd.DataFrame(results_list)
    if results_df.empty:
        print("网格搜索未找到有效模型。")
        return

    top_models = results_df.sort_values('BIC').head(4)
    print("\n" + "=" * 50)
    print("--- 最佳模型候选 (针对 Log-Diff 序列) ---")
    print(top_models[['order', 'AIC', 'BIC']])
    print("=" * 50)

    # D. 自动化诊断
    for idx, row in top_models.iterrows():
        order = row['order']
        order_str = f"p{order[0]}_d{order[1]}_q{order[2]}"

        print(f"\n正在诊断最佳模型: ARIMA{order}")
        model = ARIMA(processed_data, order=order)
        res = model.fit()

        # 统计检验
        lb_test = acorr_ljungbox(res.resid, lags=[10], return_df=True)
        p_lb = lb_test['lb_pvalue'].values[0]

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 残差自相关 (ACF)
        plot_acf(res.resid, lags=40, ax=axes[0])
        axes[0].set_title(f"Residual ACF: {order}\nLjung-Box p: {p_lb:.4f}")

        # 正态性检查 (Q-Q Plot)
        stats.probplot(res.resid, dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q Plot: {order}")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"Diagnostic_{order_str}.png")
        plt.show()

    print(f"\n任务完成！请在 {OUTPUT_DIR} 文件夹查看诊断图。")


if __name__ == "__main__":
    main()