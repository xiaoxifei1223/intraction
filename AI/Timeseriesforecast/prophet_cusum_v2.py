"""
Prophet + CUSUM 方案 (修正版)
目标: 预测未来7天的每日incident数量，然后求和判断是否报警
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ProphetCUSUMDailyForecaster:
    """Prophet + CUSUM - 预测每日数据"""
    
    def __init__(self, cusum_threshold=5.0, cusum_drift=1.0, use_cusum=True):
        """
        参数:
            cusum_threshold: CUSUM检测阈值
            cusum_drift: 漂移参数
            use_cusum: 是否使用CUSUM检测
        """
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.use_cusum = use_cusum
        self.model = None
        self.changepoints = []
        self.last_checkpoint = None
        
    def detect_drift_cusum(self, residuals, dates=None):
        """CUSUM漂移检测"""
        n = len(residuals)
        
        # 计算残差均值
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        
        # 标准化残差
        if residual_std > 0:
            residuals_std = (residuals - residual_mean) / residual_std
        else:
            residuals_std = residuals - residual_mean
        
        # CUSUM统计量
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i-1] + residuals_std[i] - self.cusum_drift)
            cusum_neg[i] = min(0, cusum_neg[i-1] + residuals_std[i] + self.cusum_drift)
        
        # 检测漂移
        drift_pos_indices = np.where(cusum_pos > self.cusum_threshold)[0]
        drift_neg_indices = np.where(np.abs(cusum_neg) > self.cusum_threshold)[0]
        
        drift_detected = len(drift_pos_indices) > 0 or len(drift_neg_indices) > 0
        
        if drift_detected:
            if len(drift_pos_indices) > 0 and len(drift_neg_indices) > 0:
                drift_point = min(drift_pos_indices[0], drift_neg_indices[0])
            elif len(drift_pos_indices) > 0:
                drift_point = drift_pos_indices[0]
            else:
                drift_point = drift_neg_indices[0]
        else:
            drift_point = None
        
        return drift_detected, drift_point, cusum_pos, cusum_neg, residuals_std
    
    def train_prophet(self, df, use_checkpoint=False):
        """训练Prophet模型"""
        
        if use_checkpoint and self.last_checkpoint is not None:
            df_train = df[df['ds'] >= self.last_checkpoint].copy()
            print(f"  使用checkpoint {self.last_checkpoint.strftime('%Y-%m-%d')} 之后的 {len(df_train)} 天数据")
        else:
            df_train = df.copy()
            print(f"  使用全部 {len(df_train)} 天数据")
        
        # Prophet模型配置
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='additive',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.8
        )
        
        self.model.fit(df_train)
        
        return self.model
    
    def forecast_next_7days(self, df, detect_drift=True):
        """
        预测未来7天的每日incident数量
        
        返回:
            daily_forecast: 每日预测
            week_total: 7天总和
            drift_detected: 是否检测到漂移
        """
        
        # 第一次训练 - 用于检测漂移
        if detect_drift and self.use_cusum:
            print("检测漂移...")
            self.train_prophet(df, use_checkpoint=False)
            
            # 预测历史数据计算残差
            forecast_hist = self.model.predict(df[['ds']])
            residuals = df['y'].values - forecast_hist['yhat'].values
            
            # CUSUM检测
            drift_detected, drift_point, _, _, _ = self.detect_drift_cusum(residuals, df['ds'])
            
            if drift_detected and drift_point is not None:
                self.last_checkpoint = df.iloc[drift_point]['ds']
                self.changepoints.append(self.last_checkpoint)
                print(f"  ⚠️ 检测到漂移! Checkpoint: {self.last_checkpoint.strftime('%Y-%m-%d')}")
                
                # 使用checkpoint后的数据重新训练
                print("  使用新数据重新训练...")
                self.train_prophet(df, use_checkpoint=True)
            else:
                print("  ✓ 未检测到显著漂移")
        else:
            print("训练模型...")
            self.train_prophet(df, use_checkpoint=False)
            drift_detected = False
        
        # 预测未来7天
        future = self.model.make_future_dataframe(periods=7)
        forecast = self.model.predict(future)
        
        # 提取未来7天的预测
        daily_forecast = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        daily_forecast.columns = ['date', 'prediction', 'lower', 'upper']
        
        # 确保预测值非负
        daily_forecast['prediction'] = daily_forecast['prediction'].clip(lower=0)
        daily_forecast['lower'] = daily_forecast['lower'].clip(lower=0)
        daily_forecast['upper'] = daily_forecast['upper'].clip(lower=0)
        
        # 计算7天总和
        week_total = {
            'prediction': daily_forecast['prediction'].sum(),
            'lower': daily_forecast['lower'].sum(),
            'upper': daily_forecast['upper'].sum(),
            'drift_detected': drift_detected
        }
        
        return daily_forecast, week_total
    
    def plot_forecast(self, df, daily_forecast, week_total):
        """绘制预测结果"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 图1: 每日预测
        ax1 = axes[0]
        
        recent_days = 60
        df_recent = df.tail(recent_days)
        
        ax1.plot(df_recent['ds'], df_recent['y'], 'o-', 
                label='Historical Data', alpha=0.7, linewidth=1.5, color='blue')
        
        ax1.plot(daily_forecast['date'], daily_forecast['prediction'], 
                'ro-', label='Forecast (Next 7 Days)', linewidth=2, markersize=8)
        
        ax1.fill_between(daily_forecast['date'], 
                         daily_forecast['lower'], 
                         daily_forecast['upper'],
                         alpha=0.3, color='red', label='80% CI')
        
        # 标记changepoints
        for cp in self.changepoints:
            if cp >= df_recent['ds'].min():
                ax1.axvline(cp, color='orange', linestyle='--', 
                           alpha=0.7, linewidth=2, label='Drift Point')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Daily Incidents', fontsize=12)
        ax1.set_title('Prophet + CUSUM: Daily Forecast', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 图2: 周总和对比
        ax2 = axes[1]
        
        df['week'] = df['ds'].dt.to_period('W')
        weekly_sum = df.groupby('week')['y'].sum().reset_index()
        weekly_sum = weekly_sum.tail(10)
        
        x_pos = range(len(weekly_sum))
        ax2.bar(x_pos, weekly_sum['y'], alpha=0.6, label='Historical Weekly Total', color='steelblue')
        
        next_week_pos = len(weekly_sum)
        ax2.bar(next_week_pos, week_total['prediction'], 
               color='red', alpha=0.7, label='Forecasted Week Total')
        
        error_lower = week_total['prediction'] - week_total['lower']
        error_upper = week_total['upper'] - week_total['prediction']
        ax2.errorbar(next_week_pos, week_total['prediction'],
                    yerr=[[error_lower], [error_upper]],
                    fmt='none', ecolor='red', capsize=10, capthick=2,
                    label='80% CI')
        
        week_labels = [str(w) for w in weekly_sum['week']] + ['Next Week']
        ax2.set_xticks(range(len(week_labels)))
        ax2.set_xticklabels(week_labels, rotation=45, ha='right')
        
        ax2.set_xlabel('Week', fontsize=12)
        ax2.set_ylabel('Total Incidents', fontsize=12)
        ax2.set_title('Weekly Total: Forecast vs Historical', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('prophet_daily_forecast_v2.png', dpi=150, bbox_inches='tight')
        print("预测图已保存: prophet_daily_forecast_v2.png")


def evaluate_prophet_daily(df, n_weeks=8, cusum_threshold=5.0, use_cusum=True):
    """回测Prophet每日预测"""
    print("\n" + "="*60)
    print(f"Prophet + CUSUM 每日预测回测 (CUSUM={'启用' if use_cusum else '禁用'})")
    print("="*60)
    
    predictions = []
    actuals = []
    
    for i in range(n_weeks, 0, -1):
        print(f"\n--- 回测第 {n_weeks-i+1}/{n_weeks} 周 ---")
        
        cutoff_date = df['ds'].max() - pd.Timedelta(days=i*7)
        df_train = df[df['ds'] <= cutoff_date].copy()
        
        if len(df_train) < 60:
            continue
        
        # 创建模型
        forecaster = ProphetCUSUMDailyForecaster(
            cusum_threshold=cusum_threshold, 
            cusum_drift=0.5,
            use_cusum=use_cusum
        )
        
        # 预测未来7天
        daily_forecast, week_total = forecaster.forecast_next_7days(df_train, detect_drift=use_cusum)
        
        # 实际值
        actual_start = cutoff_date + pd.Timedelta(days=1)
        actual_end = cutoff_date + pd.Timedelta(days=7)
        actual_week = df[(df['ds'] > cutoff_date) & (df['ds'] <= actual_end)]
        actual_total = actual_week['y'].sum()
        
        # 逐日误差
        actual_daily = actual_week['y'].values
        pred_daily = daily_forecast['prediction'].values[:len(actual_daily)]
        daily_mae = mean_absolute_error(actual_daily, pred_daily)
        
        predictions.append(week_total['prediction'])
        actuals.append(actual_total)
        
        error = abs(week_total['prediction'] - actual_total)
        pct_error = error / actual_total * 100 if actual_total > 0 else 0
        
        print(f"  周总数: Pred={week_total['prediction']:.0f}, Actual={actual_total:.0f}, Error={error:.0f} ({pct_error:.1f}%)")
        print(f"  日均MAE: {daily_mae:.2f}")
    
    if len(predictions) > 0:
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        print(f"\n{'='*60}")
        print(f"回测性能指标:")
        print(f"  周总数 MAE:  {mae:.2f}")
        print(f"  周总数 MAPE: {mape:.2f}%")
        print(f"{'='*60}")
    
    return predictions, actuals
