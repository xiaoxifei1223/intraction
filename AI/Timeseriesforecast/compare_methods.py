"""
对比不同预测方法
1. Prophet (不用CUSUM)
2. Prophet + CUSUM (原方案)
3. 简单统计方法 (最近N天均值)
"""

import pandas as pd
import numpy as np
from prophet_cusum_v2 import ProphetCUSUMDailyForecaster, evaluate_prophet_daily
from sklearn.metrics import mean_absolute_error
from simple14_enhanced_level_weekend import Simple14EnhancedForecaster
from lgbm_forecast_v2 import evaluate_weekly_model
def simple_forecast(df, recent_days=30):
    """简单方法: 使用最近N天的数据预测未来7天"""
    recent_data = df.tail(recent_days)
    
    # 按星期几分组
    recent_data['dayofweek'] = recent_data['ds'].dt.dayofweek
    weekday_means = recent_data.groupby('dayofweek')['y'].mean()
    
    # 预测未来7天
    last_date = df['ds'].max()
    predictions = []
    
    for i in range(1, 8):
        next_date = last_date + pd.Timedelta(days=i)
        dow = next_date.dayofweek
        # 如果该星期没有数据,用全局均值
        if dow in weekday_means.index:
            pred = weekday_means[dow]
        else:
            pred = recent_data['y'].mean()
        predictions.append(pred)
    
    return np.array(predictions)

def evaluate_simple_method(df, recent_days=30, n_weeks=8):
    """回测简单方法"""
    print("\n" + "="*60)
    print(f"简单统计方法回测 (使用最近{recent_days}天均值)")
    print("="*60)
    
    predictions = []
    actuals = []
    
    for i in range(n_weeks, 0, -1):
        cutoff_date = df['ds'].max() - pd.Timedelta(days=i*7)
        df_train = df[df['ds'] <= cutoff_date].copy()
        
        if len(df_train) < recent_days:
            continue
        
        # 简单预测
        pred_daily = simple_forecast(df_train, recent_days=recent_days)
        pred_total = pred_daily.sum()
        
        # 实际值
        actual_start = cutoff_date + pd.Timedelta(days=1)
        actual_end = cutoff_date + pd.Timedelta(days=7)
        actual_week = df[(df['ds'] > cutoff_date) & (df['ds'] <= actual_end)]
        actual_total = actual_week['y'].sum()
        
        predictions.append(pred_total)
        actuals.append(actual_total)
        
        error = abs(pred_total - actual_total)
        pct_error = error / actual_total * 100 if actual_total > 0 else 0
        
        print(f"Week ending {actual_end.strftime('%Y-%m-%d')}: "
              f"Pred={pred_total:.0f}, Actual={actual_total:.0f}, "
              f"Error={error:.0f} ({pct_error:.1f}%)")
    
    if len(predictions) > 0:
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        print(f"\n{'='*60}")
        print(f"回测性能指标:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"{'='*60}")
    
    return predictions, actuals

def evaluate_simple14_enhanced(df, n_weeks=8):
    """回测 Simple-14days 增强版 (level_shift + weekend_shift)"""
    print("\n" + "="*60)
    print("Simple-14days 增强版回测 (level_shift + weekend_shift)")
    print("="*60)
    
    predictions = []
    actuals = []
    
    for i in range(n_weeks, 0, -1):
        cutoff_date = df['ds'].max() - pd.Timedelta(days=i*7)
        df_train = df[df['ds'] <= cutoff_date].copy()
        
        if len(df_train) < 14:
            continue
        
        forecaster = Simple14EnhancedForecaster(window_days=14)
        daily_forecast = forecaster.forecast_next_7days(df_train)
        pred_total = daily_forecast['prediction'].sum()
        
        # 实际值
        actual_start = cutoff_date + pd.Timedelta(days=1)
        actual_end = cutoff_date + pd.Timedelta(days=7)
        actual_week = df[(df['ds'] > cutoff_date) & (df['ds'] <= actual_end)]
        actual_total = actual_week['y'].sum()
        
        predictions.append(pred_total)
        actuals.append(actual_total)
        
        error = abs(pred_total - actual_total)
        pct_error = error / actual_total * 100 if actual_total > 0 else 0
        
        print(f"Week ending {actual_end.strftime('%Y-%m-%d')}: "
              f"Pred={pred_total:.0f}, Actual={actual_total:.0f}, "
              f"Error={error:.0f} ({pct_error:.1f}%)")
    
    return predictions, actuals


def main():
    print("="*80)
    print("预测方法全面对比")
    print("="*80)
    print()
    
    # 加载数据
    df = pd.read_csv('./rawdata/raw_data.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    print(f"数据: {len(df)}天, {df['ds'].min().strftime('%Y-%m-%d')} 到 {df['ds'].max().strftime('%Y-%m-%d')}")
    print()
    
    # 存储结果
    results = {}
    
    # 方法1: Prophet (不用CUSUM)
    print("\n" + "#"*80)
    print("方法1: Prophet (不使用CUSUM)")
    print("#"*80)
    pred1, act1 = evaluate_prophet_daily(df, n_weeks=8, use_cusum=False)
    if len(pred1) > 0:
        results['Prophet'] = {
            'MAE': mean_absolute_error(act1, pred1),
            'MAPE': np.mean(np.abs((np.array(act1) - np.array(pred1)) / np.array(act1))) * 100
        }
    
    # 方法2: Prophet + CUSUM
    print("\n" + "#"*80)
    print("方法2: Prophet + CUSUM (你的原方案)")
    print("#"*80)
    pred2, act2 = evaluate_prophet_daily(df, n_weeks=8, cusum_threshold=5.0, use_cusum=True)
    if len(pred2) > 0:
        results['Prophet+CUSUM'] = {
            'MAE': mean_absolute_error(act2, pred2),
            'MAPE': np.mean(np.abs((np.array(act2) - np.array(pred2)) / np.array(act2))) * 100
        }
    
    # 方法3: 简单统计方法 (不同窗口)
    for days in [14, 30, 60]:
        print("\n" + "#"*80)
        print(f"方法3: 简单统计方法 (最近{days}天)")
        print("#"*80)
        pred3, act3 = evaluate_simple_method(df, recent_days=days, n_weeks=8)
        if len(pred3) > 0:
            results[f'Simple-{days}days'] = {
                'MAE': mean_absolute_error(act3, pred3),
                'MAPE': np.mean(np.abs((np.array(act3) - np.array(pred3)) / np.array(act3))) * 100
            }
    
    # 方法4: Simple-14days增强版 (level_shift + weekend_shift)
    print("\n" + "#"*80)
    print("方法4: Simple-14days增强版 (level_shift + weekend_shift)")
    print("#"*80)
    pred4, act4 = evaluate_simple14_enhanced(df, n_weeks=8)
    if len(pred4) > 0:
        results['Simple-14d-Enhanced'] = {
            'MAE': mean_absolute_error(act4, pred4),
            'MAPE': np.mean(np.abs((np.array(act4) - np.array(pred4)) / np.array(act4))) * 100
        }
    
    # 方法5: LightGBM 周模型 (lgbm_forecast_v2)
    print("\n" + "#"*80)
    print("方法5: LightGBM 周模型 (lgbm_forecast_v2)")
    print("#"*80)
    pred5, act5 = evaluate_weekly_model(df, n_weeks=8)
    if len(pred5) > 0:
        results['LightGBM-weekly-v2'] = {
            'MAE': mean_absolute_error(act5, pred5),
            'MAPE': np.mean(np.abs((np.array(act5) - np.array(pred5)) / np.array(act5))) * 100
        }
    
    # 输出对比总结
    print("\n" + "="*80)
    print("最终对比结果")
    print("="*80)
    print()
    
    # 按MAPE排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MAPE'])
    
    print(f"{'方法':<25} {'MAE':>10} {'MAPE':>10}")
    print("-"*50)
    for method, metrics in sorted_results:
        print(f"{method:<25} {metrics['MAE']:>10.2f} {metrics['MAPE']:>9.1f}%")
    
    print()
    print("="*80)
    print("结论:")
    best_method = sorted_results[0][0]
    best_mape = sorted_results[0][1]['MAPE']
    print(f"✓ 最佳方法: {best_method} (MAPE={best_mape:.1f}%)")
    print("="*80)
    print()

if __name__ == '__main__':
    main()
