"""
改进版预测系统 - 直接预测周总数
主要改进:
1. 窗口增加到120天,样本从31增加到约15周
2. 直接预测周总数,而非每日累加
3. 简化特征,提高训练效率
"""

import pandas as pd
import sys
from lgbm_forecast_v2 import ImprovedIncidentForecaster, evaluate_weekly_model

def main():
    print("="*80)
    print("改进版 Incident预测和预警系统 V2")
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
    
    # 2. 创建并训练模型
    print("【步骤2】创建并训练改进模型...")
    print("-"*60)
    
    forecaster = ImprovedIncidentForecaster(window_size=120)
    forecaster.train(df)
    
    print()
    
    # 3. 回测评估
    print("【步骤3】模型回测评估...")
    print("-"*60)
    evaluate_weekly_model(df, n_weeks=8)
    
    print()
    
    # 4. 预测下周
    print("【步骤4】预测下周总数...")
    print("-"*60)
    
    week_forecast = forecaster.predict_next_week(df)
    
    print()
    
    # 5. 预警检查
    print("【步骤5】预警检查...")
    print("-"*60)
    
    # 计算阈值
    df['week'] = df['ds'].dt.to_period('W')
    weekly_sum = df.groupby('week')['y'].sum()
    recent_weeks = weekly_sum.tail(8)
    
    threshold_mean = recent_weeks.mean()
    threshold_std = recent_weeks.std()
    
    print(f"最近8周统计:")
    print(f"  均值: {threshold_mean:.2f}")
    print(f"  标准差: {threshold_std:.2f}")
    print()
    
    THRESHOLD = threshold_mean + 1.5 * threshold_std
    print(f"设定阈值: {THRESHOLD:.2f}")
    
    is_alert = forecaster.check_alert(week_forecast, THRESHOLD)
    
    print()
    
    # 6. 保存结果
    print("【步骤6】保存预测结果...")
    print("-"*60)
    
    week_summary = pd.DataFrame({
        'metric': ['prediction', 'q10', 'q25', 'q50', 'q75', 'q90', 'threshold', 'alert'],
        'value': [
            week_forecast['prediction'],
            week_forecast['q10'],
            week_forecast['q25'],
            week_forecast['q50'],
            week_forecast['q75'],
            week_forecast['q90'],
            THRESHOLD,
            int(is_alert)
        ]
    })
    week_summary.to_csv('week_forecast_v2.csv', index=False)
    print("✓ 周预测摘要已保存: week_forecast_v2.csv")
    
    print()
    print("="*80)
    print("预测完成!")
    print("="*80)
    print()
    
    if is_alert:
        print("⚠️  预警建议:")
        print("  - 下周预测incident数量较高，建议提前准备")
    else:
        print("✓ 下周预测在正常范围内")
    
    print()


if __name__ == '__main__':
    main()
