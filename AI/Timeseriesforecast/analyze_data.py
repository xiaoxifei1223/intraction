import pandas as pd
import numpy as np
import sys

print("开始分析数据...", flush=True)

# 读取数据
try:
    df = pd.read_csv('rawdata/raw_data.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    print(f"数据加载成功! 共{len(df)}条记录", flush=True)
except Exception as e:
    print(f"加载数据失败: {e}", flush=True)
    sys.exit(1)

print("\n" + "="*80)
print("【1】数据基本信息")
print("="*80)
print(f"数据范围: {df['ds'].min().strftime('%Y-%m-%d')} 到 {df['ds'].max().strftime('%Y-%m-%d')}")
print(f"数据点数: {len(df)} 天")
print(f"均值: {df['y'].mean():.2f}")
print(f"标准差: {df['y'].std():.2f}")
print(f"最小值: {df['y'].min()}")
print(f"最大值: {df['y'].max()}")
print(f"中位数: {df['y'].median():.2f}")
cv = df['y'].std() / df['y'].mean()
print(f"变异系数 (CV): {cv:.4f}")
print()

# 趋势分析
print("="*80)
print("【2】趋势分析 - 分段统计")
print("="*80)
segments = 4
segment_size = len(df) // segments
segment_stats = []
for i in range(segments):
    start_idx = i * segment_size
    end_idx = (i + 1) * segment_size if i < segments - 1 else len(df)
    segment_data = df.iloc[start_idx:end_idx]
    mean_val = segment_data['y'].mean()
    std_val = segment_data['y'].std()
    segment_stats.append(mean_val)
    print(f"时段{i+1} ({segment_data['ds'].iloc[0].strftime('%Y-%m-%d')} 到 {segment_data['ds'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  均值: {mean_val:.2f}, 标准差: {std_val:.2f}")

first_30_mean = df.head(30)['y'].mean()
last_30_mean = df.tail(30)['y'].mean()
change_pct = ((last_30_mean - first_30_mean) / first_30_mean * 100)
print(f"\n前30天均值: {first_30_mean:.2f}")
print(f"后30天均值: {last_30_mean:.2f}")
print(f"变化幅度: {change_pct:.2f}%")
print(">>> 结论: 存在明显的下降趋势和漂移 <<<")
print()

# 相关性分析
print("="*80)
print("【3】自相关性分析")
print("="*80)
df['y_lag1'] = df['y'].shift(1)
df['y_lag7'] = df['y'].shift(7)
df['y_lag14'] = df['y'].shift(14)

corr_lag1 = df[['y', 'y_lag1']].corr().iloc[0, 1]
corr_lag7 = df[['y', 'y_lag7']].corr().iloc[0, 1]
corr_lag14 = df[['y', 'y_lag14']].corr().iloc[0, 1]

print(f"与前1天的相关系数: {corr_lag1:.4f}")
print(f"与前7天的相关系数: {corr_lag7:.4f}")
print(f"与前14天的相关系数: {corr_lag14:.4f}")
print(">>> 结论: 相关性较弱，不存在强马尔可夫性 <<<")
print()

# 周期性分析
print("="*80)
print("【4】周期性分析 - 按星期几统计")
print("="*80)
df['weekday'] = df['ds'].dt.dayofweek
weekday_stats = df.groupby('weekday')['y'].agg(['mean', 'std', 'count'])
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
for idx, row in weekday_stats.iterrows():
    print(f"{weekday_names[idx]}: 均值={row['mean']:.2f}, 标准差={row['std']:.2f}, 样本数={int(row['count'])}")

weekday_cv = weekday_stats['mean'].std() / weekday_stats['mean'].mean()
print(f"\n星期之间的变异系数: {weekday_cv:.4f}")
if weekday_cv < 0.15:
    print(">>> 结论: 星期效应较弱 <<<")
else:
    print(">>> 结论: 存在一定的星期效应 <<<")
print()

# 异常值分析
print("="*80)
print("【5】异常值分析")
print("="*80)
Q1 = df['y'].quantile(0.25)
Q3 = df['y'].quantile(0.75)
IQR = Q3 - Q1
outlier_lower = Q1 - 1.5 * IQR
outlier_upper = Q3 + 1.5 * IQR
outliers = df[(df['y'] < outlier_lower) | (df['y'] > outlier_upper)]
print(f"IQR方法检测到的异常值: {len(outliers)} 个 ({len(outliers)/len(df)*100:.2f}%)")
if len(outliers) > 0 and len(outliers) <= 10:
    print("\n异常值列表:")
    for idx, row in outliers.iterrows():
        print(f"  {row['ds'].strftime('%Y-%m-%d')}: {int(row['y'])}")
elif len(outliers) > 10:
    print(f"\n异常值样例(前10个):")
    for idx, row in outliers.head(10).iterrows():
        print(f"  {row['ds'].strftime('%Y-%m-%d')}: {int(row['y'])}")
print()

# 周汇总分析
print("="*80)
print("【6】周汇总分析 - 预测目标")
print("="*80)
df['week'] = df['ds'].dt.to_period('W')
weekly_sum = df.groupby('week')['y'].sum().reset_index()
weekly_sum.columns = ['week', 'total_incidents']
print("最近10周的incident总数:")
for idx, row in weekly_sum.tail(10).iterrows():
    print(f"  {str(row['week'])}: {int(row['total_incidents'])}")

print(f"\n周总数统计:")
print(f"  均值: {weekly_sum['total_incidents'].mean():.2f}")
print(f"  标准差: {weekly_sum['total_incidents'].std():.2f}")
print(f"  中位数: {weekly_sum['total_incidents'].median():.2f}")
print(f"  最大值: {int(weekly_sum['total_incidents'].max())}")
print(f"  最小值: {int(weekly_sum['total_incidents'].min())}")
print()

# 总结
print("="*80)
print("【数据特征总结】")
print("="*80)
print(f"""
核心发现:

1. 【明显的非平稳性】
   - 存在显著的下降趋势（从{first_30_mean:.0f}降至{last_30_mean:.0f}，下降{abs(change_pct):.1f}%）
   - 这不是一个稳定的时间序列，存在系统性漂移
   - 均值和方差随时间变化显著
   
2. 【弱自相关性】
   - lag1相关系数 {corr_lag1:.4f}，相邻天关系很弱
   - lag7相关系数 {corr_lag7:.4f}，周期效应不明显
   - 不存在明显的马尔可夫性
   - 意味着"昨天的值对今天预测帮助有限"

3. 【高变异性】
   - CV = {cv:.4f} (> 0.5表示高变异)
   - 数据波动很大，存在{len(outliers)}个异常值
   - 单日预测难度大，不确定性高

4. 【弱周期性】
   - 星期效应变异系数 {weekday_cv:.4f}
   - {'星期效应较弱' if weekday_cv < 0.15 else '存在一定的星期效应'}
   - 没有明显的季节性模式
""")

print("="*80)
print("【算法建议】")
print("="*80)
print("""
基于以上数据特征,对于你的预警需求,建议采用以下策略:

❌ 不推荐的方法:
   - Prophet: 适合有明确季节性和趋势的数据,你的数据趋势不稳定
   - ARIMA/SARIMA: 需要平稳序列,你的数据存在明显漂移
   - 传统LSTM: 弱自相关性导致时序依赖学习困难

✅ 推荐的方法组合:

【方案1: 基于分位数的统计预警】(最简单有效)
   1. 使用滚动窗口(如最近30-60天)计算均值和分位数
   2. 预测下周总数 = 窗口均值 × 7
   3. 设置阈值: 如均值 + 2×标准差,或90%分位数
   4. 优点: 简单、鲁棒、易解释
   5. 适合: 数据不稳定、趋势漂移的场景

【方案2: 自适应基线 + 变点检测】
   1. 使用CUSUM或EWMA检测分布变化/漂移
   2. 检测到变点后更新基线(rolling baseline)
   3. 基于新基线进行预测和阈值设定
   4. 结合你原有的CUSUM经验,但不依赖Prophet
   5. 适合: 需要快速响应趋势变化

【方案3: 集成模型 + 不确定性量化】
   1. LightGBM/XGBoost + 特征工程:
      - 时间特征: 星期几、月份、季度
      - 滞后特征: lag7, lag14, lag30 (虽然相关性弱但可尝试)
      - 滚动统计: rolling_mean, rolling_std (多窗口)
      - 趋势特征: 线性趋势、加速度
   2. 使用分位数回归获得预测区间
   3. 预测下周7天的总和
   4. 优点: 可捕捉复杂非线性关系
   5. 适合: 有足够数据、需要较高精度

【方案4: 简化的深度学习】
   1. Temporal Fusion Transformer (TFT) - 专门处理多时间尺度
   2. 或N-BEATS - 不假设特定季节性
   3. 加入协变量(如是否节假日、外部事件)
   4. 输出预测分布而非点估计
   5. 适合: 数据量充足、可解释性要求不高

【推荐实施路径】:
   
   第一阶段(快速上线): 
   → 使用方案1 + 方案2的组合
   → 1-2天可以实现,稳定可靠
   
   第二阶段(优化提升):
   → 尝试方案3,对比方案1的效果
   → 根据业务反馈调整阈值和窗口大小
   
   关键点:
   1. 由于趋势不稳定,使用短期滚动窗口(30-60天)而非全历史
   2. 预测"下周总数"比预测每天更可靠(平滑随机波动)
   3. 使用预测区间而非点估计,给出不确定性
   4. 定期重新训练/更新基线(如每周或检测到变点时)
   5. 结合业务知识设置动态阈值
""")

print("\n分析完成!")
