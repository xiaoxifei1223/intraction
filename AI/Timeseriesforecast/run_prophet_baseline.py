"""
运行Prophet + CUSUM基准方案
"""

import pandas as pd
import sys
from prophet_cusum_baseline import ProphetCUSUMForecaster, evaluate_prophet_cusum

def main():
    print("="*80)
    print("基准方案: Prophet + CUSUM 漂移检测")
    print("="*80)
    print()
    
    # 1. 加载数据
    print("【步骤1】加载数据...")
    try:
        df = pd.read_csv('./rawdata/raw_data.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        print(f"✓ 数据加载成功! 共{len(df)}条记录")
        print(f"  数据范围: {df['ds'].min().strftime('%Y-%m-%d')} 到 {df['ds'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        sys.exit(1)
    
    print()
    
    # 2. 回测评估
    print("【步骤2】回测评估...")
    print("-"*60)
    evaluate_prophet_cusum(df, n_weeks=8)
    
    print()
    
    # 3. 训练并预测
    print("【步骤3】训练模型并预测下周...")
    print("-"*60)
    
    forecaster = ProphetCUSUMForecaster(cusum_threshold=3.0, cusum_drift=0.5)
    result = forecaster.forecast_next_week(df)
    
    print(f"\n下周预测结果:")
    print(f"  预测总数: {result['prediction']:.2f}")
    print(f"  置信区间: [{result['lower']:.2f}, {result['upper']:.2f}]")
    print(f"  是否检测到漂移: {'是' if result['drift_detected'] else '否'}")
    
    if len(forecaster.changepoints) > 0:
        print(f"\n检测到的Changepoints:")
        for i, cp in enumerate(forecaster.changepoints):
            print(f"  {i+1}. {cp.strftime('%Y-%m-%d')}")
    
    print()
    
    # 4. 预警检查
    print("【步骤4】预警检查...")
    print("-"*60)
    
    # 计算阈值
    df['week'] = df['ds'].dt.to_period('W')
    weekly_sum = df.groupby('week')['y'].sum()
    recent_weeks = weekly_sum.tail(8)
    
    threshold_mean = recent_weeks.mean()
    threshold_std = recent_weeks.std()
    THRESHOLD = threshold_mean + 1.5 * threshold_std
    
    print(f"最近8周统计:")
    print(f"  均值: {threshold_mean:.2f}")
    print(f"  标准差: {threshold_std:.2f}")
    print(f"  阈值: {THRESHOLD:.2f}")
    print()
    
    if result['prediction'] > THRESHOLD:
        print(f"⚠️  警告！预测值 {result['prediction']:.2f} 超过阈值 {THRESHOLD:.2f}")
        is_alert = True
    else:
        print(f"✓ 正常。预测值 {result['prediction']:.2f} 低于阈值 {THRESHOLD:.2f}")
        is_alert = False
    
    print()
    
    # 5. 生成可视化
    print("【步骤5】生成可视化...")
    print("-"*60)
    forecaster.plot_forecast(df, result)
    
    # 6. 保存结果
    print("\n【步骤6】保存结果...")
    print("-"*60)
    
    # 保存每日预测
    result['daily_predictions'].to_csv('prophet_daily_forecast.csv', index=False)
    print("✓ 每日预测已保存: prophet_daily_forecast.csv")
    
    # 保存周预测摘要
    week_summary = pd.DataFrame({
        'metric': ['prediction', 'lower', 'upper', 'threshold', 'alert', 'drift_detected'],
        'value': [
            result['prediction'],
            result['lower'],
            result['upper'],
            THRESHOLD,
            int(is_alert),
            int(result['drift_detected'])
        ]
    })
    week_summary.to_csv('prophet_week_summary.csv', index=False)
    print("✓ 周预测摘要已保存: prophet_week_summary.csv")
    
    print()
    print("="*80)
    print("Prophet + CUSUM 基准方案完成!")
    print("="*80)
    print()
    
    print("输出文件:")
    print("  1. prophet_daily_forecast.csv - 下周每日预测")
    print("  2. prophet_week_summary.csv - 周预测摘要")
    print("  3. prophet_cusum_forecast.png - 预测可视化")
    print("  4. cusum_detection.png - CUSUM漂移检测图")
    print()


if __name__ == '__main__':
    main()
