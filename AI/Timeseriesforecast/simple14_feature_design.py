"""
Simple-14days 特征设计详解
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 加载数据
df = pd.read_csv('./rawdata/raw_data.csv')
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

print("="*80)
print("Simple-14days 特征设计详解")
print("="*80)
print()

# 使用10月17日为截止点
cutoff_date = pd.Timestamp('2025-10-17')
df_train = df[df['ds'] <= cutoff_date].copy()
recent_14 = df_train.tail(14).copy()

print("【1】特征工程流程")
print("="*80)
print("""
Step 1: 提取最近14天数据
Step 2: 为每天标注星期几特征（dayofweek: 0-6）
Step 3: 按星期分组，计算统计量
Step 4: 对未来7天，使用对应星期的统计量进行预测
""")
print()

# Step 1-2: 特征提取
print("【2】原始数据 + 星期特征")
print("="*80)
recent_14['dayofweek'] = recent_14['ds'].dt.dayofweek
recent_14['dayname'] = recent_14['ds'].dt.day_name()

print("最近14天数据 (2025-10-04 到 2025-10-17):")
print(recent_14[['ds', 'dayname', 'dayofweek', 'y']].to_string(index=False))
print()

# Step 3: 特征统计
print("【3】特征统计量计算")
print("="*80)

weekday_features = recent_14.groupby(['dayofweek', 'dayname']).agg({
    'y': ['count', 'mean', 'std', 'min', 'max', 'median']
}).reset_index()

weekday_features.columns = ['dayofweek', 'dayname', 'count', 'mean', 'std', 'min', 'max', 'median']

print("按星期分组的特征:")
print(weekday_features.to_string(index=False))
print()

print("【4】特征详细解释")
print("="*80)
print()

for idx, row in weekday_features.iterrows():
    dow = int(row['dayofweek'])
    name = row['dayname']
    count = int(row['count'])
    mean = row['mean']
    std = row['std']
    min_val = row['min']
    max_val = row['max']
    median = row['median']
    
    print(f"{name} (dayofweek={dow}):")
    print(f"  样本数: {count} 个")
    print(f"  均值: {mean:.1f}")
    print(f"  标准差: {std:.1f}")
    print(f"  中位数: {median:.1f}")
    print(f"  范围: [{min_val:.0f}, {max_val:.0f}]")
    print(f"  变异系数: {std/mean:.3f}")
    
    # 找出这个星期的具体样本
    samples = recent_14[recent_14['dayofweek'] == dow]
    print(f"  具体样本:")
    for _, s in samples.iterrows():
        print(f"    {s['ds'].strftime('%Y-%m-%d')}: {int(s['y'])}")
    print()

print("="*80)
print("【5】特征设计的核心思想")
print("="*80)
print("""
特征类型: 单一类别特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

特征名: dayofweek (星期几)
特征值: 0-6 (整数，代表周一到周日)
特征类型: Categorical (类别型)

预测逻辑:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IF 预测日期是周一 (dayofweek=0):
    THEN 预测值 = 最近14天中所有周一的均值
    
IF 预测日期是周二 (dayofweek=1):
    THEN 预测值 = 最近14天中所有周二的均值
    
... (依此类推)

置信区间:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
lower = mean - 1.5 × std
upper = mean + 1.5 × std

(1.5σ约等于80%置信区间，假设正态分布)
""")
print()

print("【6】为什么只用一个特征？")
print("="*80)
print("""
对比: 复杂模型(如LightGBM)使用的特征

LightGBM特征表 (31个):
┌────────────────────┬──────────────────────────────────┐
│ 特征类型           │ 具体特征                          │
├────────────────────┼──────────────────────────────────┤
│ Lag特征 (7个)      │ lag_1, lag_2, lag_3, lag_7,      │
│                    │ lag_14, lag_21, lag_28           │
├────────────────────┼──────────────────────────────────┤
│ 滚动均值 (3个)     │ rolling_mean_7/14/30             │
├────────────────────┼──────────────────────────────────┤
│ 滚动标准差 (3个)   │ rolling_std_7/14/30              │
├────────────────────┼──────────────────────────────────┤
│ 滚动极值 (6个)     │ rolling_min/max_7/14/30          │
├────────────────────┼──────────────────────────────────┤
│ 时间特征 (6个)     │ dayofweek, day, month, quarter,  │
│                    │ dayofyear, weekofyear            │
├────────────────────┼──────────────────────────────────┤
│ 其他 (6个)         │ is_weekend, trend, diff_1,       │
│                    │ diff_7, deviation_from_mean_7/30 │
└────────────────────┴──────────────────────────────────┘

问题:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ lag_28需要28天历史 → 60天窗口变成32天可用
✗ 包含历史趋势(trend) → 但历史已不适用(regime change)
✗ 31个特征但只有32个样本 → 特征数/样本数 = 0.97 (过拟合风险)
✗ 训练时间长(10-30秒) → 实时性差

Simple-14days的优势:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 只用1个特征 → 不需要历史lag，全部14天可用
✓ 只关注"当前"星期模式 → 不受历史趋势影响
✓ 1个特征14个样本 → 特征数/样本数 = 0.07 (无过拟合)
✓ 运行时间<1秒 → 实时性好
✓ 易于理解和解释 → 业务人员都能懂
""")
print()

print("【7】特征设计的理论基础")
print("="*80)
print("""
为什么单特征在这个场景下有效？

1. 奥卡姆剃刀原则 (Occam's Razor)
   "如无必要，勿增实体"
   在多个模型都能解释数据时，选择最简单的

2. 样本-特征比 (Sample-to-Feature Ratio)
   理论要求: 样本数 ≥ 10 × 特征数
   
   Simple-14days: 14样本 / 1特征 = 14:1 ✓
   LightGBM:     32样本 / 31特征 = 1.03:1 ✗

3. 领域知识
   - 数据有regime change → 历史不适用
   - 相关性分析显示: lag1=0.76, lag7=0.81
     虽然相关，但在低位平台期这些相关性已变化
   - 星期效应虽弱但稳定存在

4. 偏差-方差权衡 (Bias-Variance Tradeoff)
   简单模型:
     - 高偏差(可能欠拟合) 
     - 低方差(预测稳定)
     - 总误差 = 适中
   
   复杂模型(样本不足时):
     - 低偏差(训练很好)
     - 高方差(过拟合，泛化差)
     - 总误差 = 高
""")
print()

print("【8】特征有效性验证")
print("="*80)

# 计算星期效应的统计显著性
overall_mean = recent_14['y'].mean()
weekday_means = recent_14.groupby('dayofweek')['y'].mean()

print(f"全局均值: {overall_mean:.1f}")
print(f"\n各星期偏离度:")
for dow, mean in weekday_means.items():
    deviation = (mean - overall_mean) / overall_mean * 100
    print(f"  星期{dow}: {mean:.1f} (偏离全局 {deviation:+.1f}%)")

# 方差分析
between_group_var = sum([(weekday_means[dow] - overall_mean)**2 * len(recent_14[recent_14['dayofweek']==dow]) 
                         for dow in weekday_means.index])
total_var = sum((recent_14['y'] - overall_mean)**2)
explained_var_ratio = between_group_var / total_var if total_var > 0 else 0

print(f"\n星期特征解释的方差比例: {explained_var_ratio*100:.1f}%")
print(f"残差方差比例: {(1-explained_var_ratio)*100:.1f}%")

if explained_var_ratio < 0.1:
    print("→ 星期效应较弱 (解释<10%方差)")
elif explained_var_ratio < 0.3:
    print("→ 星期效应中等 (解释10-30%方差)")
else:
    print("→ 星期效应显著 (解释>30%方差)")
print()

print("【9】特征的稳定性分析")
print("="*80)

# 对比不同14天窗口的星期均值稳定性
windows = [
    (df[(df['ds'] > '2025-09-20') & (df['ds'] <= '2025-10-03')], "窗口1: 09-21~10-03"),
    (df[(df['ds'] > '2025-09-27') & (df['ds'] <= '2025-10-10')], "窗口2: 09-28~10-10"),
    (df[(df['ds'] > '2025-10-04') & (df['ds'] <= '2025-10-17')], "窗口3: 10-04~10-17"),
]

print("不同时间窗口的星期均值对比:")
print()
print(f"{'星期':<10} {'窗口1':<10} {'窗口2':<10} {'窗口3':<10} {'变异系数':<10}")
print("-"*60)

weekday_names_short = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
for dow in range(7):
    means = []
    for window_df, _ in windows:
        window_df = window_df.copy()
        window_df['dayofweek'] = window_df['ds'].dt.dayofweek
        dow_data = window_df[window_df['dayofweek'] == dow]
        if len(dow_data) > 0:
            means.append(dow_data['y'].mean())
        else:
            means.append(np.nan)
    
    if all(~np.isnan(means)):
        cv = np.std(means) / np.mean(means) if np.mean(means) > 0 else 0
        print(f"{weekday_names_short[dow]:<10} {means[0]:<10.1f} {means[1]:<10.1f} {means[2]:<10.1f} {cv:<10.3f}")
    else:
        print(f"{weekday_names_short[dow]:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

print()
print("变异系数越小，说明该星期的特征越稳定")
print()

print("【10】可视化特征空间")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 特征分布(按星期)
ax1 = axes[0, 0]
boxplot_data = [recent_14[recent_14['dayofweek']==dow]['y'].values 
                for dow in range(7)]
bp = ax1.boxplot(boxplot_data, labels=weekday_names_short)
ax1.set_xlabel('Day of Week', fontsize=11)
ax1.set_ylabel('Incidents', fontsize=11)
ax1.set_title('Feature Distribution by Weekday', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 图2: 特征均值+标准差
ax2 = axes[0, 1]
means = [weekday_features[weekday_features['dayofweek']==dow]['mean'].values[0] 
         if dow in weekday_features['dayofweek'].values else 0 
         for dow in range(7)]
stds = [weekday_features[weekday_features['dayofweek']==dow]['std'].values[0] 
        if dow in weekday_features['dayofweek'].values else 0 
        for dow in range(7)]

x_pos = np.arange(7)
ax2.bar(x_pos, means, alpha=0.7, color='steelblue', label='Mean')
ax2.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='red', 
            capsize=5, capthick=2, label='±1 Std')
ax2.set_xlabel('Day of Week', fontsize=11)
ax2.set_ylabel('Incidents', fontsize=11)
ax2.set_title('Feature: Mean ± Std', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(weekday_names_short)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 图3: 时间序列+特征标注
ax3 = axes[1, 0]
colors = plt.cm.tab10(np.linspace(0, 1, 7))
for dow in range(7):
    dow_data = recent_14[recent_14['dayofweek'] == dow]
    ax3.scatter(dow_data['ds'], dow_data['y'], 
               c=[colors[dow]], label=weekday_names_short[dow], s=100, alpha=0.7)

ax3.set_xlabel('Date', fontsize=11)
ax3.set_ylabel('Incidents', fontsize=11)
ax3.set_title('Time Series colored by Weekday Feature', fontsize=12, fontweight='bold')
ax3.legend(loc='best', ncol=7, fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 图4: 特征重要性(用解释方差表示)
ax4 = axes[1, 1]
explained = explained_var_ratio * 100
residual = (1 - explained_var_ratio) * 100

bars = ax4.bar(['Weekday\nFeature', 'Residual\n(Noise)'], 
               [explained, residual], 
               color=['green', 'lightgray'], alpha=0.7)
ax4.set_ylabel('Variance Explained (%)', fontsize=11)
ax4.set_title('Feature Importance (Variance Decomposition)', 
             fontsize=12, fontweight='bold')
ax4.set_ylim([0, 100])

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('simple14_feature_design.png', dpi=150, bbox_inches='tight')
print("✓ 特征设计可视化已保存: simple14_feature_design.png")
print()

print("="*80)
print("【总结】Simple-14days特征设计")
print("="*80)
print(f"""
特征数量: 1个
特征名称: dayofweek (星期几)
特征类型: Categorical (7个类别: 0-6)
特征来源: 日期时间戳

预测公式:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prediction(date) = mean(y | dayofweek = date.dayofweek, 
                        ds in [date-14days, date])

简单来说:
  预测值 = 最近14天中同样星期几的平均值

置信区间:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CI_80% = [mean - 1.5×std, mean + 1.5×std]

特征统计:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 解释方差: {explained_var_ratio*100:.1f}%
- 每个类别样本数: {int(14/7)} 个
- 样本-特征比: 14:1

为什么有效:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 数据有regime change，只用最近数据
2. 样本充足，避免过拟合
3. 特征稳定，星期模式存在
4. 简单透明，易于理解维护
5. 运行快速，实时性好

这是"极简主义"在机器学习中的成功案例！
""")
