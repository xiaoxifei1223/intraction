"""
诊断模型问题
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv('./rawdata/raw_data.csv')
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

print("="*80)
print("问题诊断分析")
print("="*80)

# 1. 查看最近60天的数据特征
recent_60 = df.tail(60)
print(f"\n最近60天数据统计:")
print(f"  均值: {recent_60['y'].mean():.2f}")
print(f"  标准差: {recent_60['y'].std():.2f}")
print(f"  最小值: {recent_60['y'].min()}")
print(f"  最大值: {recent_60['y'].max()}")
print(f"  变异系数: {recent_60['y'].std() / recent_60['y'].mean():.4f}")

# 2. 对比不同时期
periods = [
    ("前60天", df.head(60)),
    ("中间60天", df.iloc[len(df)//2-30:len(df)//2+30]),
    ("最近60天", df.tail(60))
]

print(f"\n不同时期对比:")
for name, period_df in periods:
    print(f"{name}: 均值={period_df['y'].mean():.2f}, 标准差={period_df['y'].std():.2f}")

# 3. 检查训练数据量
print(f"\n训练数据问题:")
print(f"  总数据量: {len(df)} 天")
print(f"  窗口大小: 60 天")
print(f"  创建特征后(去除NaN): 约 31 个样本")
print(f"  ⚠️ 问题: 样本量太少！只有31个样本训练模型")

# 4. 周汇总数据的变化
df['week'] = df['ds'].dt.to_period('W')
weekly_sum = df.groupby('week')['y'].sum()
recent_weeks = weekly_sum.tail(10)

print(f"\n最近10周的总数:")
for week, total in recent_weeks.items():
    print(f"  {week}: {int(total)}")

print(f"\n周总数统计:")
print(f"  均值: {recent_weeks.mean():.2f}")
print(f"  标准差: {recent_weeks.std():.2f}")
print(f"  MAE=206 相对于均值{recent_weeks.mean():.2f}的比例: {206/recent_weeks.mean()*100:.1f}%")

# 5. 问题根源分析
print("\n" + "="*80)
print("问题根源:")
print("="*80)
print("""
1. 训练样本太少 (只有31个)
   - 60天窗口
   - 减去lag和rolling特征需要的前28-30天
   - 实际可用于训练的只有约31个样本
   - LightGBM需要更多样本才能学习到模式

2. 数据趋势剧烈变化
   - 从1100+降到150左右，下降84%
   - 最近60天的数据已经处于低位平台期
   - 模型基于低位数据预测，但历史回测时数据量级不同

3. 特征设计问题
   - lag_28需要28天历史
   - rolling_30需要30天历史
   - 导致大量数据被浪费

解决方案:
""")
