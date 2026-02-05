"""
验证Simple-14days在10月18-25这一周的实际表现
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# 加载数据
df = pd.read_csv('./rawdata/raw_data.csv')
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

print("="*80)
print("Simple-14days 模型验证: 2025-10-18 到 2025-10-25")
print("="*80)
print()

# 截取到10月17日的数据(模拟只有到这一天的数据)
cutoff_date = pd.Timestamp('2025-10-17')
df_train = df[df['ds'] <= cutoff_date].copy()

print(f"【训练数据】截止到: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"  共 {len(df_train)} 天数据")
print()

# 使用最近14天数据进行预测
recent_14 = df_train.tail(14).copy()
recent_14['dayofweek'] = recent_14['ds'].dt.dayofweek

print("【使用的最近14天数据】(2025-10-04 到 2025-10-17):")
print("-"*80)
print(recent_14[['ds', 'y']].to_string(index=False))
print()

# 按星期分组统计
weekday_stats = recent_14.groupby('dayofweek')['y'].agg(['mean', 'std', 'count']).reset_index()
print("【星期统计特征】:")
print("-"*80)
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
for idx, row in weekday_stats.iterrows():
    print(f"{weekday_names[int(row['dayofweek'])]}: "
          f"样本数={int(row['count'])}, "
          f"均值={row['mean']:.1f}, "
          f"标准差={row['std']:.1f}")
print()

# 预测未来7天(10月18-24，注意原数据只到10-25)
predictions = []
dates = []
lowers = []
uppers = []

for i in range(1, 8):
    next_date = cutoff_date + pd.Timedelta(days=i)
    dow = next_date.dayofweek
    
    # 根据星期几获取预测
    if dow in weekday_stats['dayofweek'].values:
        row = weekday_stats[weekday_stats['dayofweek'] == dow].iloc[0]
        pred = row['mean']
        std = row['std']
    else:
        # 如果该星期没有数据，使用全局均值
        pred = recent_14['y'].mean()
        std = recent_14['y'].std()
    
    predictions.append(max(0, pred))
    dates.append(next_date)
    lowers.append(max(0, pred - 1.5 * std))
    uppers.append(pred + 1.5 * std)

# 获取实际值
actual_data = df[(df['ds'] > cutoff_date) & (df['ds'] <= cutoff_date + pd.Timedelta(days=7))]

print("【预测 vs 实际对比】:")
print("="*80)
print(f"{'日期':<12} {'星期':<8} {'预测值':<10} {'实际值':<10} {'误差':<10} {'误差%':<10} {'在CI内?':<10}")
print("-"*80)

daily_errors = []
in_ci_count = 0

for i, (pred, lower, upper, date) in enumerate(zip(predictions, lowers, uppers, dates)):
    if i < len(actual_data):
        actual = actual_data.iloc[i]['y']
        error = pred - actual
        error_pct = abs(error) / actual * 100 if actual > 0 else 0
        in_ci = lower <= actual <= upper
        
        daily_errors.append(abs(error))
        if in_ci:
            in_ci_count += 1
        
        dow_name = date.strftime('%A')
        dow_cn = weekday_names[date.dayofweek]
        
        print(f"{date.strftime('%Y-%m-%d'):<12} {dow_cn:<8} "
              f"{pred:<10.1f} {actual:<10.0f} "
              f"{error:<10.1f} {error_pct:<10.1f} "
              f"{'✓' if in_ci else '✗':<10}")

print()
print("【汇总统计】:")
print("-"*80)

if len(actual_data) > 0:
    week_pred = sum(predictions[:len(actual_data)])
    week_actual = actual_data['y'].sum()
    week_error = week_pred - week_actual
    week_error_pct = abs(week_error) / week_actual * 100
    
    print(f"周总数:")
    print(f"  预测: {week_pred:.0f}")
    print(f"  实际: {week_actual:.0f}")
    print(f"  误差: {week_error:.1f} ({week_error_pct:.1f}%)")
    print()
    
    print(f"每日误差:")
    print(f"  MAE: {np.mean(daily_errors):.2f}")
    print(f"  MAPE: {np.mean([abs(predictions[i] - actual_data.iloc[i]['y']) / actual_data.iloc[i]['y'] * 100 for i in range(len(actual_data))]):.2f}%")
    print(f"  最大误差: {max(daily_errors):.0f}")
    print(f"  最小误差: {min(daily_errors):.0f}")
    print()
    
    print(f"置信区间覆盖率:")
    print(f"  {in_ci_count}/{len(actual_data)} 天在置信区间内 ({in_ci_count/len(actual_data)*100:.1f}%)")
    print()

# 详细分析每一天
print("【逐日详细分析】:")
print("="*80)

for i in range(len(actual_data)):
    date = dates[i]
    dow_name = weekday_names[date.dayofweek]
    pred = predictions[i]
    actual = actual_data.iloc[i]['y']
    error = pred - actual
    error_pct = abs(error) / actual * 100 if actual > 0 else 0
    
    print(f"\n{date.strftime('%Y-%m-%d')} ({dow_name}):")
    print(f"  预测: {pred:.1f}")
    print(f"  实际: {actual}")
    print(f"  误差: {error:+.1f} ({error_pct:.1f}%)")
    
    # 找出该星期在训练数据中的样本
    train_samples = recent_14[recent_14['dayofweek'] == date.dayofweek]
    if len(train_samples) > 0:
        print(f"  训练样本 ({len(train_samples)}个):")
        for idx, row in train_samples.iterrows():
            print(f"    {row['ds'].strftime('%Y-%m-%d')}: {int(row['y'])}")
        print(f"  训练均值: {train_samples['y'].mean():.1f}")
    
    # 判断预测质量
    if error_pct < 10:
        quality = "✓ 非常准确"
    elif error_pct < 20:
        quality = "✓ 较准确"
    elif error_pct < 30:
        quality = "△ 一般"
    else:
        quality = "✗ 偏差较大"
    
    print(f"  评价: {quality}")

print()
print("="*80)
print("【结论】")
print("="*80)

if len(actual_data) > 0:
    avg_mape = np.mean([abs(predictions[i] - actual_data.iloc[i]['y']) / actual_data.iloc[i]['y'] * 100 for i in range(len(actual_data))])
    
    if avg_mape < 20:
        conclusion = "表现优秀"
    elif avg_mape < 30:
        conclusion = "表现良好"
    elif avg_mape < 50:
        conclusion = "表现一般"
    else:
        conclusion = "表现较差"
    
    print(f"Simple-14days模型在10月18-25这周的表现: {conclusion}")
    print(f"  周总数MAPE: {week_error_pct:.1f}%")
    print(f"  日均MAPE: {avg_mape:.1f}%")
    print()
    
    print("优点:")
    print("  ✓ 使用最近14天数据，能反映当前趋势")
    print("  ✓ 按星期分组，利用了弱周期性")
    print("  ✓ 简单透明，易于理解和调试")
    print()
    
    print("不足:")
    if in_ci_count < len(actual_data) * 0.7:
        print("  - 置信区间覆盖率偏低，可能需要调整")
    if any([e > 50 for e in daily_errors]):
        print("  - 个别日期误差较大，可能受异常波动影响")
    if avg_mape > 30:
        print("  - 整体MAPE偏高，可能需要考虑其他方法")
    
    if avg_mape <= 30 and in_ci_count >= len(actual_data) * 0.5:
        print("  (整体表现良好，无明显不足)")

print()
