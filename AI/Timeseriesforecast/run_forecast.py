"""
主运行脚本 - Incident预测和预警系统
使用方法: python run_forecast.py
"""

import pandas as pd
import sys
from lgbm_forecast_model import IncidentForecaster, evaluate_model

def main():
    print("="*80)
    print("Incident预测和预警系统")
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
    print("【步骤2】创建并训练模型...")
    print("-"*60)
    
    # 初始化预测器
    forecaster = IncidentForecaster(
        window_size=60,  # 使用最近60天数据（应对非平稳性）
        retrain_freq=7   # 每周重新训练一次
    )
    
    # 训练模型
    forecaster.train(df)
    
    print()
    
    # 3. 回测评估（可选）
    print("【步骤3】模型回测评估...")
    print("-"*60)
    evaluate_model(df, forecaster, n_weeks=4)
    
    print()
    
    # 4. 预测下周
    print("【步骤4】预测下周incident总数...")
    print("-"*60)
    
    week_forecast, daily_forecast = forecaster.predict_next_week(df, return_daily=True)
    
    print()
    
    # 5. 预警检查
    print("【步骤5】预警检查...")
    print("-"*60)
    
    # 设置阈值 - 这里使用历史数据的统计信息
    df['week'] = df['ds'].dt.to_period('W')
    weekly_sum = df.groupby('week')['y'].sum()
    
    # 方法1: 使用最近几周的均值 + 2倍标准差
    recent_weeks = weekly_sum.tail(8)  # 最近8周
    threshold_mean = recent_weeks.mean()
    threshold_std = recent_weeks.std()
    threshold_statistical = threshold_mean + 2 * threshold_std
    
    # 方法2: 使用90%分位数
    threshold_percentile = recent_weeks.quantile(0.9)
    
    print(f"最近8周统计:")
    print(f"  均值: {threshold_mean:.2f}")
    print(f"  标准差: {threshold_std:.2f}")
    print(f"  90%分位数: {threshold_percentile:.2f}")
    print()
    
    # 选择阈值（可以根据业务需求调整）
    # 这里使用均值 + 1.5倍标准差（更灵敏）
    THRESHOLD = threshold_mean + 1.5 * threshold_std
    
    print(f"设定阈值: {THRESHOLD:.2f}")
    print()
    
    # 检查预警
    is_alert = forecaster.check_alert(week_forecast, THRESHOLD, threshold_type='fixed')
    
    print()
    
    # 6. 生成可视化报告
    print("【步骤6】生成可视化报告...")
    print("-"*60)
    forecaster.plot_forecast_visualization(df, week_forecast, daily_forecast)
    
    print()
    
    # 7. 保存预测结果
    print("【步骤7】保存预测结果...")
    print("-"*60)
    
    # 保存每日预测
    daily_forecast.to_csv('daily_forecast.csv', index=False)
    print("✓ 每日预测已保存: daily_forecast.csv")
    
    # 保存周预测
    week_summary = pd.DataFrame({
        'metric': ['prediction', 'q10', 'q25', 'q75', 'q90', 'threshold', 'alert'],
        'value': [
            week_forecast['prediction'],
            week_forecast['q10'],
            week_forecast['q25'],
            week_forecast['q75'],
            week_forecast['q90'],
            THRESHOLD,
            int(is_alert)
        ]
    })
    week_summary.to_csv('week_forecast_summary.csv', index=False)
    print("✓ 周预测摘要已保存: week_forecast_summary.csv")
    
    print()
    print("="*80)
    print("预测完成!")
    print("="*80)
    print()
    print("输出文件:")
    print("  1. daily_forecast.csv - 下周每日预测")
    print("  2. week_forecast_summary.csv - 周预测摘要")
    print("  3. forecast_visualization.png - 预测可视化图")
    print("  4. feature_importance.png - 特征重要性图")
    print()
    
    # 8. 生成预警建议
    if is_alert:
        print("⚠️  预警建议:")
        print("  - 下周预测incident数量较高，建议:")
        print("    1. 提前准备应急资源")
        print("    2. 增加人员值班")
        print("    3. 关联其他系统进行报警")
        print("    4. 监控实时数据，及时调整")
    else:
        print("✓ 下周预测在正常范围内，继续监控")
    
    print()
    
    # 9. 下次重训练建议
    last_date = df['ds'].max()
    next_retrain_date = last_date + pd.Timedelta(days=forecaster.retrain_freq)
    print(f"建议下次重训练日期: {next_retrain_date.strftime('%Y-%m-%d')}")
    print()


if __name__ == '__main__':
    main()
