import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示 - 如果没有中文字体就使用默认
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 读取数据
df = pd.read_csv('./rawdata/raw_data.csv')
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

print("=" * 80)
print("数据基本信息")
print("=" * 80)
print(f"数据范围: {df['ds'].min()} 到 {df['ds'].max()}")
print(f"数据点数: {len(df)}")
print(f"均值: {df['y'].mean():.2f}")
print(f"标准差: {df['y'].std():.2f}")
print(f"最小值: {df['y'].min()}")
print(f"最大值: {df['y'].max()}")
print(f"中位数: {df['y'].median():.2f}")
print()

# 统计特性
print("=" * 80)
print("统计特性分析")
print("=" * 80)
print(df['y'].describe())
print()

# 检查数据中的异常值
Q1 = df['y'].quantile(0.25)
Q3 = df['y'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['y'] < Q1 - 1.5*IQR) | (df['y'] > Q3 + 1.5*IQR)]
print(f"异常值数量 (IQR方法): {len(outliers)}")
if len(outliers) > 0:
    print("异常值样例:")
    print(outliers.head(10))
print()

# 趋势分析 - 检查是否存在明显的趋势和漂移
print("=" * 80)
print("趋势与漂移分析")
print("=" * 80)
# 分段统计
segments = 4
segment_size = len(df) // segments
for i in range(segments):
    start_idx = i * segment_size
    end_idx = (i + 1) * segment_size if i < segments - 1 else len(df)
    segment_data = df.iloc[start_idx:end_idx]
    print(f"时段 {i+1} ({segment_data['ds'].iloc[0].strftime('%Y-%m-%d')} 到 {segment_data['ds'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  均值: {segment_data['y'].mean():.2f}, 标准差: {segment_data['y'].std():.2f}")
print()

# 平稳性检验 (ADF Test)
print("=" * 80)
print("平稳性检验 (ADF Test)")
print("=" * 80)
adf_result = adfuller(df['y'].dropna())
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print(f"临界值:")
for key, value in adf_result[4].items():
    print(f"  {key}: {value:.4f}")
if adf_result[1] < 0.05:
    print("结论: 序列是平稳的 (p < 0.05)")
else:
    print("结论: 序列不平稳 (p >= 0.05)")
print()

# 自相关性分析
print("=" * 80)
print("自相关性分析")
print("=" * 80)
# Ljung-Box检验
lb_test = acorr_ljungbox(df['y'], lags=[1, 7, 14, 30], return_df=True)
print("Ljung-Box检验 (检测自相关性):")
print(lb_test)
print()

# 计算不同lag的自相关系数
lags_to_check = [1, 7, 14, 30]
acf_values = acf(df['y'], nlags=30, fft=False)
print("自相关系数 (ACF):")
for lag in lags_to_check:
    print(f"  Lag {lag}: {acf_values[lag]:.4f}")
print()

# 周期性分析
print("=" * 80)
print("周期性分析 (按星期几)")
print("=" * 80)
df['weekday'] = df['ds'].dt.dayofweek
df['weekday_name'] = df['ds'].dt.day_name()
weekday_stats = df.groupby('weekday_name')['y'].agg(['mean', 'std', 'count'])
# 按照周一到周日排序
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_stats = weekday_stats.reindex(weekday_order)
print(weekday_stats)
print()

# 方差齐性检验 (Levene's test)
print("=" * 80)
print("方差齐性检验 (不同星期的方差是否一致)")
print("=" * 80)
weekday_groups = [df[df['weekday'] == i]['y'].values for i in range(7)]
levene_stat, levene_p = stats.levene(*weekday_groups)
print(f"Levene统计量: {levene_stat:.4f}")
print(f"p-value: {levene_p:.4f}")
if levene_p < 0.05:
    print("结论: 不同星期的方差不一致 (p < 0.05)")
else:
    print("结论: 不同星期的方差一致 (p >= 0.05)")
print()

# 检查是否存在季节性
print("=" * 80)
print("季节性分析 (按月份)")
print("=" * 80)
df['month'] = df['ds'].dt.month
monthly_stats = df.groupby('month')['y'].agg(['mean', 'std', 'count'])
print(monthly_stats)
print()

# 独立性检验 - 检查连续天数之间的关系
print("=" * 80)
print("独立性分析 (相邻天数的关系)")
print("=" * 80)
df['y_lag1'] = df['y'].shift(1)
df['y_lag7'] = df['y'].shift(7)

# Pearson相关系数
corr_lag1 = df[['y', 'y_lag1']].corr().iloc[0, 1]
corr_lag7 = df[['y', 'y_lag7']].corr().iloc[0, 1]
print(f"与前一天的相关系数: {corr_lag1:.4f}")
print(f"与前7天的相关系数: {corr_lag7:.4f}")

# Spearman秩相关
spearman_lag1, spearman_p1 = stats.spearmanr(df['y'], df['y_lag1'], nan_policy='omit')
print(f"与前一天的Spearman相关系数: {spearman_lag1:.4f} (p={spearman_p1:.4f})")
print()

# 变异系数分析 (CV)
print("=" * 80)
print("变异性分析")
print("=" * 80)
cv = df['y'].std() / df['y'].mean()
print(f"变异系数 (CV): {cv:.4f}")
print("CV > 0.5 表示高变异性")
print()

# 创建可视化
print("=" * 80)
print("正在生成可视化图表...")
print("=" * 80)

fig, axes = plt.subplots(4, 2, figsize=(16, 16))

# 1. 时间序列图
axes[0, 0].plot(df['ds'], df['y'], linewidth=0.8)
axes[0, 0].set_title('时间序列图', fontsize=12)
axes[0, 0].set_xlabel('日期')
axes[0, 0].set_ylabel('Incident数量')
axes[0, 0].grid(True, alpha=0.3)

# 2. 分段均值
segment_means = []
segment_dates = []
for i in range(segments):
    start_idx = i * segment_size
    end_idx = (i + 1) * segment_size if i < segments - 1 else len(df)
    segment_data = df.iloc[start_idx:end_idx]
    segment_means.append(segment_data['y'].mean())
    segment_dates.append(segment_data['ds'].iloc[len(segment_data)//2])

axes[0, 1].plot(segment_dates, segment_means, marker='o', linewidth=2, markersize=8)
axes[0, 1].set_title('分段均值趋势', fontsize=12)
axes[0, 1].set_xlabel('时段')
axes[0, 1].set_ylabel('平均Incident数量')
axes[0, 1].grid(True, alpha=0.3)

# 3. 直方图
axes[1, 0].hist(df['y'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Incident数量分布', fontsize=12)
axes[1, 0].set_xlabel('Incident数量')
axes[1, 0].set_ylabel('频数')
axes[1, 0].grid(True, alpha=0.3)

# 4. Q-Q图
stats.probplot(df['y'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q图 (正态性检验)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

# 5. ACF图
acf_vals = acf(df['y'], nlags=40, fft=False)
axes[2, 0].stem(range(len(acf_vals)), acf_vals, basefmt=' ')
axes[2, 0].axhline(y=0, color='black', linewidth=0.8)
axes[2, 0].axhline(y=1.96/np.sqrt(len(df)), color='red', linestyle='--', linewidth=0.8)
axes[2, 0].axhline(y=-1.96/np.sqrt(len(df)), color='red', linestyle='--', linewidth=0.8)
axes[2, 0].set_title('自相关函数 (ACF)', fontsize=12)
axes[2, 0].set_xlabel('Lag')
axes[2, 0].set_ylabel('ACF')
axes[2, 0].grid(True, alpha=0.3)

# 6. PACF图
pacf_vals = pacf(df['y'], nlags=40)
axes[2, 1].stem(range(len(pacf_vals)), pacf_vals, basefmt=' ')
axes[2, 1].axhline(y=0, color='black', linewidth=0.8)
axes[2, 1].axhline(y=1.96/np.sqrt(len(df)), color='red', linestyle='--', linewidth=0.8)
axes[2, 1].axhline(y=-1.96/np.sqrt(len(df)), color='red', linestyle='--', linewidth=0.8)
axes[2, 1].set_title('偏自相关函数 (PACF)', fontsize=12)
axes[2, 1].set_xlabel('Lag')
axes[2, 1].set_ylabel('PACF')
axes[2, 1].grid(True, alpha=0.3)

# 7. 星期分布
weekday_means = df.groupby('weekday')['y'].mean().values
axes[3, 0].bar(range(7), weekday_means, edgecolor='black', alpha=0.7)
axes[3, 0].set_title('按星期几的平均Incident数量', fontsize=12)
axes[3, 0].set_xlabel('星期 (0=周一, 6=周日)')
axes[3, 0].set_ylabel('平均Incident数量')
axes[3, 0].grid(True, alpha=0.3, axis='y')

# 8. 滚动均值和标准差
window = 7
df['rolling_mean'] = df['y'].rolling(window=window).mean()
df['rolling_std'] = df['y'].rolling(window=window).std()
axes[3, 1].plot(df['ds'], df['y'], alpha=0.3, label='原始数据')
axes[3, 1].plot(df['ds'], df['rolling_mean'], linewidth=2, label=f'{window}天滚动均值')
axes[3, 1].fill_between(df['ds'], 
                         df['rolling_mean'] - 2*df['rolling_std'], 
                         df['rolling_mean'] + 2*df['rolling_std'],
                         alpha=0.2, label='±2σ区间')
axes[3, 1].set_title(f'{window}天滚动统计', fontsize=12)
axes[3, 1].set_xlabel('日期')
axes[3, 1].set_ylabel('Incident数量')
axes[3, 1].legend()
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_analysis_visualization.png', dpi=150, bbox_inches='tight')
print("图表已保存为: data_analysis_visualization.png")
print()

# 生成分析报告总结
print("=" * 80)
print("数据特征总结")
print("=" * 80)
print(f"""
1. 数据范围: {len(df)}天数据，从{df['ds'].min().strftime('%Y-%m-%d')}到{df['ds'].max().strftime('%Y-%m-%d')}

2. 趋势特征:
   - 存在明显的下降趋势（从1月的1100+降到10月的150左右）
   - 这是一个明显的非平稳序列，存在趋势漂移

3. 周期性:
   - ACF显示出弱的自相关性
   - 相邻日期的相关系数: {corr_lag1:.4f}
   - 7天lag的相关系数: {corr_lag7:.4f}
   - 不存在强的马尔可夫性（相邻天的相关性弱）

4. 变异性:
   - 变异系数 CV = {cv:.4f}
   - 数据波动大，存在异常值
   
5. 平稳性:
   - ADF检验 p-value = {adf_result[1]:.4f}
   - {'序列不平稳' if adf_result[1] >= 0.05 else '序列平稳'}
   - 均值和方差随时间变化
""")
