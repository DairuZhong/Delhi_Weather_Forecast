import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
import os  # 用于文件路径操作
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 忽略收敛警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 准备输出目录
# ==========================================
output_dir = 'Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建文件夹: {output_dir}")

# ==========================================
# 1. 数据加载
# ==========================================
df = pd.read_csv('delhi_climate_train.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
series = df['meantemp']

# ==========================================
# 2. 扩展网格搜索 (p: 0-10, q: 0-10)
# ==========================================
p_values = range(0, 11)
d_values = [1]
q_values = range(0, 11)

results_list = []

print("正在执行网格搜索，请稍候...")
for p in p_values:
    for q in q_values:
        order = (p, 1, q)
        try:
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            results_list.append({
                'p': p,
                'q': q,
                'AIC': model_fit.aic,
                'BIC': model_fit.bic,
                'order': order
            })
        except:
            continue

# 按 BIC 排序（BIC 对模型复杂度惩罚更严，通常选出的模型更简洁）
results_df = pd.DataFrame(results_list)
top_4_models = results_df.sort_values('BIC').head(4)

print("\n--- 排名前 4 的模型 (按 BIC 排序) ---")
print(top_4_models[['order', 'AIC', 'BIC']])

# ==========================================
# 3. 自动化诊断并保存绘图
# ==========================================
for i, row in top_4_models.iterrows():
    current_order = row['order']
    order_str = f"{current_order[0]}_{current_order[1]}_{current_order[2]}"

    print(f"\n" + "=" * 50)
    print(f"正在诊断模型: ARIMA{current_order}")
    print("=" * 50)

    # 拟合模型
    model = ARIMA(series, order=current_order)
    res = model.fit()

    # --- A. 提取残差 ---
    residuals = res.resid

    # --- B. 绘图与保存 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 残差 ACF 图
    plot_acf(residuals, lags=30, ax=axes[0])
    axes[0].set_title(f"Residual ACF: ARIMA{current_order}")

    # Q-Q 图
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot: ARIMA{current_order}")

    plt.tight_layout()

    # 保存图片到 Outputs 文件夹
    file_name = f"Diagnostic_ARIMA_{order_str}.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path)
    print(f"已保存诊断图至: {save_path}")

    # 显示并关闭，防止内存溢出
    plt.show()
    plt.close(fig)

    # --- C. 统计检验输出 (打印在控制台) ---
    print("\n[Ljung-Box Test]")
    lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print(lb_test)

    # 将参数表也保存为 CSV (可选)
    param_table_path = os.path.join(output_dir, f"Params_ARIMA_{order_str}.csv")
    res.params.to_csv(param_table_path)
    print(f"已保存参数表至: {param_table_path}")

print("\n所有任务已完成！请查看 Outputs 文件夹。")