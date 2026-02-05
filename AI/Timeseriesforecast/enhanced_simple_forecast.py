"""
增强版Simple预测 - 在14天基础特征上添加衍生特征
目标: 提升预测精度同时保持简单性
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class EnhancedSimpleForecaster:
    """增强版简单预测器"""
    
    def __init__(self, window_days=14):
        self.window_days = window_days
        
    def create_enhanced_features(self, df):
        """
        创建增强特征
        
        特征1: dayofweek (基础特征)
        特征2: recent_trend (最近趋势)
        特征3: volatility_level (波动水平)
        """
        recent_data = df.tail(self.window_days).copy()
        recent_data['dayofweek'] = recent_data['ds'].dt.dayofweek
        
        # 特征1: 基础星期均值 (已有)
        weekday_means = recent_data.groupby('dayofweek')['y'].mean()
        weekday_stds = recent_data.groupby('dayofweek')['y'].std()
        
        # 特征2: 最近趋势 (线性拟合斜率)
        # 用最近7天的数据计算趋势
        recent_7 = recent_data.tail(7)
        if len(recent_7) >= 3:
            x = np.arange(len(recent_7))
            y = recent_7['y'].values
            # 简单线性回归
            trend_slope = np.polyfit(x, y, 1)[0]  # 斜率
        else:
            trend_slope = 0
        
        # 特征3: 波动水平 (最近7天的变异系数)
        recent_7_std = recent_7['y'].std()
        recent_7_mean = recent_7['y'].mean()
        volatility = recent_7_std / recent_7_mean if recent_7_mean > 0 else 0
        
        # 特征4: 最近一天的值 (作为参考)
        last_value = recent_data['y'].iloc[-1]
        
        features = {
            'weekday_means': weekday_means,
            'weekday_stds': weekday_stds,
            'trend_slope': trend_slope,
            'volatility': volatility,
            'last_value': last_value,
            'recent_mean': recent_data['y'].mean()
        }
        
        return features, recent_data
    
    def predict_next_7days(self, df):
        """
        预测未来7天 - 使用增强特征
        
        预测策略:
        1. 基础预测 = 该星期的历史均值 (权重70%)
        2. 趋势调整 = trend_slope × 天数 (权重15%)
        3. 最近值参考 = last_value的影响 (权重15%)
        """
        features, recent_data = self.create_enhanced_features(df)
        
        last_date = df['ds'].max()
        predictions = []
        dates = []
        lowers = []
        uppers = []
        
        # 提取特征
        weekday_means = features['weekday_means']
        weekday_stds = features['weekday_stds']
        trend_slope = features['trend_slope']
        last_value = features['last_value']
        recent_mean = features['recent_mean']
        
        for i in range(1, 8):
            next_date = last_date + pd.Timedelta(days=i)
            dow = next_date.dayofweek
            
            # 1. 基础预测 (星期均值)
            if dow in weekday_means.index:
                base_pred = weekday_means[dow]
                std = weekday_stds[dow]
            else:
                base_pred = recent_mean
                std = recent_data['y'].std()
            
            # 2. 趋势调整
            # 假设趋势会延续，但影响力随时间衰减
            trend_adjustment = trend_slope * i * 0.5  # 乘以天数和衰减系数
            
            # 3. 最近值影响
            # 如果最近值偏离均值较多，给予一定权重
            last_value_deviation = last_value - recent_mean
            last_value_adjustment = last_value_deviation * (0.3 / (i + 1))  # 影响随时间递减
            
            # 组合预测 (加权平均)
            weight_base = 0.70
            weight_trend = 0.15
            weight_last = 0.15
            
            final_pred = (
                weight_base * base_pred + 
                weight_trend * (base_pred + trend_adjustment) +
                weight_last * (base_pred + last_value_adjustment)
            )
            
            # 确保非负
            final_pred = max(0, final_pred)
            
            # 置信区间 (考虑波动性)
            volatility_multiplier = 1.0 + features['volatility']
            adjusted_std = std * volatility_multiplier
            
            predictions.append(final_pred)
            dates.append(next_date)
            lowers.append(max(0, final_pred - 1.5 * adjusted_std))
            uppers.append(final_pred + 1.5 * adjusted_std)
        
        daily_forecast = pd.DataFrame({
            'date': dates,
            'prediction': predictions,
            'lower': lowers,
            'upper': uppers
        })
        
        week_total = {
            'prediction': sum(predictions),
            'lower': sum(lowers),
            'upper': sum(uppers)
        }
        
        return daily_forecast, week_total, features


def evaluate_enhanced_model(df, window_days=14, n_weeks=8):
    """回测增强模型"""
    print("\n" + "="*60)
    print(f"增强版Simple模型回测 (窗口={window_days}天)")
    print("="*60)
    
    forecaster = EnhancedSimpleForecaster(window_days=window_days)
    predictions = []
    actuals = []
    daily_preds_all = []
    daily_actuals_all = []
    
    for i in range(n_weeks, 0, -1):
        cutoff_date = df['ds'].max() - pd.Timedelta(days=i*7)
        df_train = df[df['ds'] <= cutoff_date].copy()
        
        if len(df_train) < window_days:
            continue
        
        daily_forecast, week_total, features = forecaster.predict_next_7days(df_train)
        
        # 实际值
        actual_start = cutoff_date + pd.Timedelta(days=1)
        actual_end = cutoff_date + pd.Timedelta(days=7)
        actual_week = df[(df['ds'] > cutoff_date) & (df['ds'] <= actual_end)]
        actual_total = actual_week['y'].sum()
        
        predictions.append(week_total['prediction'])
        actuals.append(actual_total)
        
        # 记录每日预测
        for j in range(min(len(daily_forecast), len(actual_week))):
            daily_preds_all.append(daily_forecast.iloc[j]['prediction'])
            daily_actuals_all.append(actual_week.iloc[j]['y'])
        
        error = abs(week_total['prediction'] - actual_total)
        pct_error = error / actual_total * 100 if actual_total > 0 else 0
        
        print(f"Week ending {actual_end.strftime('%Y-%m-%d')}: "
              f"Pred={week_total['prediction']:.0f}, Actual={actual_total:.0f}, "
              f"Error={error:.0f} ({pct_error:.1f}%)")
    
    if len(predictions) > 0:
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        daily_mae = mean_absolute_error(daily_actuals_all, daily_preds_all)
        daily_mape = np.mean(np.abs((np.array(daily_actuals_all) - np.array(daily_preds_all)) / np.array(daily_actuals_all))) * 100
        
        print(f"\n{'='*60}")
        print(f"周总数性能:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"\n每日性能:")
        print(f"  MAE:  {daily_mae:.2f}")
        print(f"  MAPE: {daily_mape:.2f}%")
        print(f"{'='*60}")
    
    return predictions, actuals


def test_on_oct18_25(df):
    """在10月18-25验证集上测试"""
    print("\n" + "="*80)
    print("增强模型在10月18-25的表现")
    print("="*80)
    
    cutoff_date = pd.Timestamp('2025-10-17')
    df_train = df[df['ds'] <= cutoff_date].copy()
    
    forecaster = EnhancedSimpleForecaster(window_days=14)
    daily_forecast, week_total, features = forecaster.predict_next_7days(df_train)
    
    print("\n【提取的特征】:")
    print(f"  趋势斜率: {features['trend_slope']:.2f} (每天变化)")
    print(f"  波动水平: {features['volatility']:.3f}")
    print(f"  最近一天值: {features['last_value']:.0f}")
    print(f"  最近均值: {features['recent_mean']:.1f}")
    
    print("\n【逐日预测 vs 实际】:")
    print("-"*80)
    
    actual_data = df[(df['ds'] > cutoff_date) & (df['ds'] <= cutoff_date + pd.Timedelta(days=7))]
    
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    print(f"{'日期':<12} {'星期':<8} {'预测':<10} {'实际':<10} {'误差':<10} {'误差%':<10}")
    print("-"*70)
    
    daily_errors = []
    for i in range(len(actual_data)):
        pred = daily_forecast.iloc[i]['prediction']
        actual = actual_data.iloc[i]['y']
        date = daily_forecast.iloc[i]['date']
        dow_name = weekday_names[date.dayofweek]
        
        error = pred - actual
        error_pct = abs(error) / actual * 100 if actual > 0 else 0
        daily_errors.append(abs(error))
        
        print(f"{date.strftime('%Y-%m-%d'):<12} {dow_name:<8} "
              f"{pred:<10.1f} {actual:<10.0f} "
              f"{error:<10.1f} {error_pct:<10.1f}")
    
    week_pred = sum([daily_forecast.iloc[i]['prediction'] for i in range(len(actual_data))])
    week_actual = actual_data['y'].sum()
    week_error = abs(week_pred - week_actual)
    week_error_pct = week_error / week_actual * 100
    
    print()
    print(f"周总数: 预测={week_pred:.0f}, 实际={week_actual:.0f}, "
          f"误差={week_error:.0f} ({week_error_pct:.1f}%)")
    
    daily_mae = np.mean(daily_errors)
    daily_mape = np.mean([abs(daily_forecast.iloc[i]['prediction'] - actual_data.iloc[i]['y']) / actual_data.iloc[i]['y'] * 100 
                          for i in range(len(actual_data))])
    
    print(f"\n每日MAE: {daily_mae:.2f}")
    print(f"每日MAPE: {daily_mape:.2f}%")


def compare_models(df):
    """对比原始Simple-14days和增强版"""
    print("\n" + "="*80)
    print("模型对比: Simple-14days vs Enhanced")
    print("="*80)
    
    # 原始Simple-14days
    from final_recomm_forecast import SimpleAdaptiveForecaster
    
    print("\n【模型1: 原始Simple-14days】")
    simple_forecaster = SimpleAdaptiveForecaster(window_days=14)
    
    predictions_simple = []
    actuals_simple = []
    
    for i in range(8, 0, -1):
        cutoff_date = df['ds'].max() - pd.Timedelta(days=i*7)
        df_train = df[df['ds'] <= cutoff_date].copy()
        
        if len(df_train) < 14:
            continue
        
        _, week_total = simple_forecaster.forecast_next_7days(df_train)
        
        actual_start = cutoff_date + pd.Timedelta(days=1)
        actual_end = cutoff_date + pd.Timedelta(days=7)
        actual_week = df[(df['ds'] > cutoff_date) & (df['ds'] <= actual_end)]
        actual_total = actual_week['y'].sum()
        
        predictions_simple.append(week_total['prediction'])
        actuals_simple.append(actual_total)
    
    mae_simple = mean_absolute_error(actuals_simple, predictions_simple)
    mape_simple = np.mean(np.abs((np.array(actuals_simple) - np.array(predictions_simple)) / np.array(actuals_simple))) * 100
    
    print(f"  周总数 MAE:  {mae_simple:.2f}")
    print(f"  周总数 MAPE: {mape_simple:.2f}%")
    
    # 增强版
    print("\n【模型2: Enhanced (添加趋势+波动特征)】")
    predictions_enhanced, actuals_enhanced = evaluate_enhanced_model(df, window_days=14, n_weeks=8)
    
    # 对比总结
    print("\n" + "="*80)
    print("【对比总结】")
    print("="*80)
    
    if len(predictions_simple) > 0 and len(predictions_enhanced) > 0:
        mae_enhanced = mean_absolute_error(actuals_enhanced, predictions_enhanced)
        mape_enhanced = np.mean(np.abs((np.array(actuals_enhanced) - np.array(predictions_enhanced)) / np.array(actuals_enhanced))) * 100
        
        improvement_mae = (mae_simple - mae_enhanced) / mae_simple * 100
        improvement_mape = (mape_simple - mape_enhanced) / mape_simple * 100
        
        print(f"{'指标':<20} {'Simple-14d':<15} {'Enhanced':<15} {'改进':<15}")
        print("-"*70)
        print(f"{'MAE':<20} {mae_simple:<15.2f} {mae_enhanced:<15.2f} {improvement_mae:+.1f}%")
        print(f"{'MAPE':<20} {mape_simple:<15.2f}% {mape_enhanced:<15.2f}% {improvement_mape:+.1f}%")
        
        print()
        if improvement_mape > 5:
            print(f"✓ 增强版显著优于原始版 (MAPE改进{improvement_mape:.1f}%)")
        elif improvement_mape > 0:
            print(f"△ 增强版略优于原始版 (MAPE改进{improvement_mape:.1f}%)")
        else:
            print(f"✗ 增强版未能改进 (MAPE恶化{abs(improvement_mape):.1f}%)")
            print("  → 建议保持原始Simple-14days")


def main():
    # 加载数据
    df = pd.read_csv('./rawdata/raw_data.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    print("="*80)
    print("增强版Simple预测 - 特征工程测试")
    print("="*80)
    print()
    
    print("【设计的增强特征】")
    print("-"*80)
    print("""
    基础特征:
      1. dayofweek - 星期几 (0-6)
    
    新增特征:
      2. recent_trend - 最近7天的趋势斜率
         · 计算方法: 线性拟合最近7天数据
         · 作用: 捕捉短期上升/下降趋势
         · 权重: 15%
      
      3. last_value_influence - 最近一天的影响
         · 计算方法: 最近一天与均值的偏离
         · 作用: 考虑最近状态的惯性
         · 权重: 15%
      
      4. volatility - 波动水平
         · 计算方法: 最近7天的变异系数
         · 作用: 动态调整置信区间
         · 用途: 高波动时扩大CI
    
    预测公式:
      final_pred = 0.70 × weekday_mean 
                 + 0.15 × (weekday_mean + trend_adjustment)
                 + 0.15 × (weekday_mean + last_value_influence)
    """)
    
    # 回测对比
    compare_models(df)
    
    # 在10月18-25验证
    test_on_oct18_25(df)
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == '__main__':
    main()
