"""
基准方案: Prophet + CUSUM 漂移检测
复现你原来的方法作为baseline
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

class ProphetCUSUMForecaster:
    """Prophet + CUSUM 漂移检测预测器"""
    
    def __init__(self, cusum_threshold=5.0, cusum_drift=1.0):
        """
        参数:
            cusum_threshold: CUSUM检测阈值
            cusum_drift: 漂移参数
        """
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.model = None
        self.changepoints = []
        self.last_checkpoint = None
        
    def detect_drift_cusum(self, residuals, plot=False):
        """
        CUSUM漂移检测
        
        参数:
            residuals: 残差序列
            plot: 是否绘制CUSUM图
            
        返回:
            drift_detected: 是否检测到漂移
            drift_point: 漂移点索引
        """
        n = len(residuals)
        
        # 标准化残差
        residuals_std = (residuals - residuals.mean()) / (residuals.std() + 1e-8)
        
        # CUSUM统计量 (双边检测)
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i-1] + residuals_std[i] - self.cusum_drift)
            cusum_neg[i] = min(0, cusum_neg[i-1] + residuals_std[i] + self.cusum_drift)
        
        # 检测是否超过阈值
        drift_pos = np.where(cusum_pos > self.cusum_threshold)[0]
        drift_neg = np.where(np.abs(cusum_neg) > self.cusum_threshold)[0]
        
        drift_detected = len(drift_pos) > 0 or len(drift_neg) > 0
        
        if drift_detected:
            # 找到第一个漂移点
            if len(drift_pos) > 0 and len(drift_neg) > 0:
                drift_point = min(drift_pos[0], drift_neg[0])
            elif len(drift_pos) > 0:
                drift_point = drift_pos[0]
            else:
                drift_point = drift_neg[0]
        else:
            drift_point = None
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(residuals_std, 'b-', alpha=0.5, label='Standardized Residuals')
            plt.axhline(0, color='k', linestyle='--', alpha=0.3)
            plt.ylabel('Standardized Residuals')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(cusum_pos, 'r-', label='CUSUM+')
            plt.plot(np.abs(cusum_neg), 'b-', label='|CUSUM-|')
            plt.axhline(self.cusum_threshold, color='k', linestyle='--', 
                       label=f'Threshold={self.cusum_threshold}')
            if drift_detected:
                plt.axvline(drift_point, color='orange', linestyle='--', 
                           linewidth=2, label=f'Drift at {drift_point}')
            plt.xlabel('Time')
            plt.ylabel('CUSUM')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('cusum_detection.png', dpi=150, bbox_inches='tight')
            print("CUSUM检测图已保存: cusum_detection.png")
        
        return drift_detected, drift_point
    
    def train_prophet(self, df, use_all_data=False):
        """
        训练Prophet模型
        
        参数:
            df: 数据框 (需要ds, y列)
            use_all_data: 是否使用全部数据(False时使用checkpoint之后的数据)
        """
        if not use_all_data and self.last_checkpoint is not None:
            # 只使用checkpoint之后的数据
            df_train = df[df['ds'] >= self.last_checkpoint].copy()
            print(f"使用checkpoint {self.last_checkpoint.strftime('%Y-%m-%d')} 之后的数据")
            print(f"训练样本数: {len(df_train)}")
        else:
            df_train = df.copy()
            print(f"使用全部数据训练")
            print(f"训练样本数: {len(df_train)}")
        
        # 创建Prophet模型
        self.model = Prophet(
            changepoint_prior_scale=0.05,  # 趋势灵活性
            seasonality_prior_scale=10,
            seasonality_mode='additive',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.8
        )
        
        # 训练
        self.model.fit(df_train)
        
        return self.model
    
    def predict_and_check_drift(self, df, plot=False):
        """
        预测并检查漂移
        
        返回:
            forecast: 预测结果
            drift_detected: 是否检测到漂移
        """
        # 训练模型
        self.train_prophet(df)
        
        # 预测历史数据(用于计算残差)
        forecast = self.model.predict(df[['ds']])
        
        # 计算残差
        residuals = df['y'].values - forecast['yhat'].values
        
        # CUSUM检测
        drift_detected, drift_point = self.detect_drift_cusum(residuals, plot=plot)
        
        if drift_detected and drift_point is not None:
            # 设置checkpoint
            self.last_checkpoint = df.iloc[drift_point]['ds']
            self.changepoints.append(self.last_checkpoint)
            print(f"\n⚠️ 检测到漂移! Checkpoint: {self.last_checkpoint.strftime('%Y-%m-%d')}")
            print(f"   将使用checkpoint之后的数据重新训练")
            
            # 使用checkpoint之后的数据重新训练
            self.train_prophet(df, use_all_data=False)
        else:
            print(f"\n✓ 未检测到显著漂移")
        
        return forecast, drift_detected
    
    def forecast_next_week(self, df):
        """预测下周"""
        # 先做漂移检测
        forecast_hist, drift_detected = self.predict_and_check_drift(df, plot=True)
        
        # 预测未来7天
        future = self.model.make_future_dataframe(periods=7)
        forecast = self.model.predict(future)
        
        # 提取下周的预测
        next_week_forecast = forecast.tail(7)
        
        # 计算周总数
        week_total = next_week_forecast['yhat'].sum()
        week_lower = next_week_forecast['yhat_lower'].sum()
        week_upper = next_week_forecast['yhat_upper'].sum()
        
        result = {
            'prediction': week_total,
            'lower': week_lower,
            'upper': week_upper,
            'daily_predictions': next_week_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            'drift_detected': drift_detected
        }
        
        return result
    
    def plot_forecast(self, df, forecast_result):
        """绘制预测结果"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 图1: 时间序列和预测
        ax1 = axes[0]
        
        # 历史数据
        recent_days = 60
        df_recent = df.tail(recent_days)
        ax1.plot(df_recent['ds'], df_recent['y'], 'o-', 
                label='Historical Data', alpha=0.7, linewidth=1.5)
        
        # 下周预测
        daily_pred = forecast_result['daily_predictions']
        ax1.plot(daily_pred['ds'], daily_pred['yhat'], 
                'ro-', label='Forecast', linewidth=2, markersize=8)
        ax1.fill_between(daily_pred['ds'], 
                         daily_pred['yhat_lower'], 
                         daily_pred['yhat_upper'],
                         alpha=0.3, color='red', label='80% CI')
        
        # 标记changepoints
        for cp in self.changepoints:
            if cp >= df_recent['ds'].min():
                ax1.axvline(cp, color='orange', linestyle='--', 
                           alpha=0.7, linewidth=2, label='Drift Point')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Incidents', fontsize=12)
        ax1.set_title('Prophet + CUSUM Forecast', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 图2: 周总数
        ax2 = axes[1]
        
        df['week'] = df['ds'].dt.to_period('W')
        weekly_sum = df.groupby('week')['y'].sum().reset_index()
        weekly_sum = weekly_sum.tail(10)
        
        x_pos = range(len(weekly_sum))
        ax2.bar(x_pos, weekly_sum['y'], alpha=0.6, label='Historical Weekly Total')
        
        # 预测周
        next_week_pos = len(weekly_sum)
        ax2.bar(next_week_pos, forecast_result['prediction'], 
               color='red', alpha=0.7, label='Forecasted Week')
        
        # 误差线
        error_lower = forecast_result['prediction'] - forecast_result['lower']
        error_upper = forecast_result['upper'] - forecast_result['prediction']
        ax2.errorbar(next_week_pos, forecast_result['prediction'],
                    yerr=[[error_lower], [error_upper]],
                    fmt='none', ecolor='red', capsize=10, capthick=2,
                    label='80% CI')
        
        week_labels = [str(w) for w in weekly_sum['week']] + ['Next Week']
        ax2.set_xticks(range(len(week_labels)))
        ax2.set_xticklabels(week_labels, rotation=45, ha='right')
        
        ax2.set_xlabel('Week', fontsize=12)
        ax2.set_ylabel('Total Incidents', fontsize=12)
        ax2.set_title('Weekly Forecast - Prophet + CUSUM', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('prophet_cusum_forecast.png', dpi=150, bbox_inches='tight')
        print("\n预测可视化图已保存: prophet_cusum_forecast.png")


def evaluate_prophet_cusum(df, n_weeks=8):
    """回测Prophet+CUSUM方法"""
    print("\n" + "="*60)
    print("Prophet + CUSUM 回测评估")
    print("="*60)
    
    predictions = []
    actuals = []
    
    for i in range(n_weeks, 0, -1):
        print(f"\n--- 回测第 {n_weeks-i+1}/{n_weeks} 周 ---")
        
        # 截取数据
        cutoff_date = df['ds'].max() - pd.Timedelta(days=i*7)
        df_train = df[df['ds'] <= cutoff_date].copy()
        
        if len(df_train) < 60:
            continue
        
        # 创建模型
        forecaster = ProphetCUSUMForecaster(cusum_threshold=3.0, cusum_drift=0.5)
        
        # 预测
        result = forecaster.forecast_next_week(df_train)
        
        # 实际值
        actual_start = cutoff_date + pd.Timedelta(days=1)
        actual_end = cutoff_date + pd.Timedelta(days=7)
        actual_week = df[(df['ds'] > cutoff_date) & (df['ds'] <= actual_end)]
        actual_total = actual_week['y'].sum()
        
        predictions.append(result['prediction'])
        actuals.append(actual_total)
        
        error = abs(result['prediction'] - actual_total)
        pct_error = error / actual_total * 100
        
        print(f"Week ending {actual_end.strftime('%Y-%m-%d')}: "
              f"Pred={result['prediction']:.0f}, "
              f"Actual={actual_total:.0f}, "
              f"Error={error:.0f} ({pct_error:.1f}%)")
    
    if len(predictions) > 0:
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        print(f"\n回测性能指标:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return predictions, actuals
