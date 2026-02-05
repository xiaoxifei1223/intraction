"""
LightGBM时间序列预测模型 - 用于Incident预警
特点：
1. 利用滚动窗口处理非平稳序列
2. 自动特征工程（lag特征、滚动统计、时间特征）
3. 预测下周总incident数并提供不确定性区间
4. 支持阈值预警
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class IncidentForecaster:
    """Incident预测和预警系统"""
    
    def __init__(self, window_size=60, retrain_freq=7):
        """
        参数:
            window_size: 滚动窗口大小（天），用于应对非平稳性
            retrain_freq: 重新训练频率（天）
        """
        self.window_size = window_size
        self.retrain_freq = retrain_freq
        self.model = None
        self.quantile_models = {}  # 分位数回归模型
        self.feature_names = None
        
    def create_features(self, df, is_train=True):
        """
        创建特征
        
        特征类型:
        1. Lag特征: 利用强自相关性
        2. 滚动统计: 捕捉局部趋势
        3. 时间特征: 星期、月份等
        4. 趋势特征: 线性趋势
        """
        df = df.copy()
        
        # 1. Lag特征 (基于分析，lag1和lag7相关性强)
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # 2. 滚动统计特征
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['y'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['y'].rolling(window=window).max()
            
        # 3. 时间特征
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['day'] = df['ds'].dt.day
        df['month'] = df['ds'].dt.month
        df['quarter'] = df['ds'].dt.quarter
        df['dayofyear'] = df['ds'].dt.dayofyear
        df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
        
        # 周末标志
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # 4. 趋势特征（局部线性趋势）
        df['trend'] = np.arange(len(df))
        
        # 5. 滞后差分（捕捉变化率）
        df['diff_1'] = df['y'].diff(1)
        df['diff_7'] = df['y'].diff(7)
        
        # 6. 与滚动均值的偏离
        df['deviation_from_mean_7'] = df['y'] - df['rolling_mean_7']
        df['deviation_from_mean_30'] = df['y'] - df['rolling_mean_30']
        
        if is_train:
            # 删除含有NaN的行（前面的lag和rolling导致的）
            df = df.dropna()
        
        return df
    
    def prepare_data(self, df, window_size=None):
        """
        准备训练数据，使用滚动窗口
        """
        if window_size is None:
            window_size = self.window_size
            
        # 只使用最近的window_size天数据
        if len(df) > window_size:
            df = df.iloc[-window_size:].copy()
        
        # 创建特征
        df_features = self.create_features(df, is_train=True)
        
        # 分离特征和目标
        feature_cols = [col for col in df_features.columns 
                       if col not in ['ds', 'y']]
        
        X = df_features[feature_cols]
        y = df_features['y']
        
        self.feature_names = feature_cols
        
        return X, y, df_features
    
    def train(self, df, params=None):
        """
        训练模型（包括主模型和分位数模型）
        """
        print("开始训练模型...")
        print(f"使用最近 {self.window_size} 天数据")
        
        # 准备数据
        X, y, df_features = self.prepare_data(df)
        
        print(f"训练样本数: {len(X)}")
        print(f"特征数: {len(self.feature_names)}")
        
        # 默认参数
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(mae)
            print(f"  Fold {fold+1} MAE: {mae:.2f}")
        
        print(f"平均 MAE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
        
        # 在全部数据上训练最终模型
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, train_data, num_boost_round=200)
        
        # 训练分位数模型 (用于预测区间)
        print("\n训练分位数模型...")
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            params_quantile = params.copy()
            params_quantile['objective'] = 'quantile'
            params_quantile['alpha'] = q
            
            train_data = lgb.Dataset(X, label=y)
            self.quantile_models[q] = lgb.train(
                params_quantile, 
                train_data, 
                num_boost_round=200
            )
        
        print("模型训练完成!")
        
        # 特征重要性
        self.plot_feature_importance()
        
        return self
    
    def predict_next_week(self, df, return_daily=False):
        """
        预测下周的incident总数
        
        返回:
            week_forecast: dict包含预测值和置信区间
            daily_forecast: (可选) 每天的预测
        """
        print("\n开始预测下周...")
        
        # 获取最后一天
        last_date = df['ds'].max()
        print(f"最后一天数据: {last_date.strftime('%Y-%m-%d')}")
        
        # 预测未来7天
        daily_predictions = []
        daily_quantiles = {q: [] for q in self.quantile_models.keys()}
        
        df_extended = df.copy()
        
        for day in range(1, 8):
            pred_date = last_date + pd.Timedelta(days=day)
            
            # 创建新行
            new_row = pd.DataFrame({
                'ds': [pred_date],
                'y': [np.nan]  # 未知的目标值
            })
            
            df_extended = pd.concat([df_extended, new_row], ignore_index=True)
            
            # 创建特征（使用历史数据和之前的预测）
            df_features = self.create_features(df_extended, is_train=False)
            
            # 获取最后一行的特征
            X_pred = df_features[self.feature_names].iloc[-1:].ffill()
            
            # 预测
            pred = self.model.predict(X_pred)[0]
            daily_predictions.append(pred)
            
            # 预测分位数
            for q, model in self.quantile_models.items():
                q_pred = model.predict(X_pred)[0]
                daily_quantiles[q].append(q_pred)
            
            # 更新df_extended的y值（用预测值替代，供下一次迭代使用）
            df_extended.loc[df_extended['ds'] == pred_date, 'y'] = pred
            
            print(f"  {pred_date.strftime('%Y-%m-%d')} ({pred_date.strftime('%A')}): {pred:.2f}")
        
        # 计算周总数
        week_total = sum(daily_predictions)
        week_q10 = sum(daily_quantiles[0.1])
        week_q25 = sum(daily_quantiles[0.25])
        week_q75 = sum(daily_quantiles[0.75])
        week_q90 = sum(daily_quantiles[0.9])
        
        week_forecast = {
            'prediction': week_total,
            'q10': week_q10,
            'q25': week_q25,
            'q75': week_q75,
            'q90': week_q90,
            'confidence_interval_50': (week_q25, week_q75),
            'confidence_interval_80': (week_q10, week_q90),
        }
        
        print(f"\n下周预测总数: {week_total:.2f}")
        print(f"  50%置信区间: [{week_q25:.2f}, {week_q75:.2f}]")
        print(f"  80%置信区间: [{week_q10:.2f}, {week_q90:.2f}]")
        
        if return_daily:
            daily_forecast = pd.DataFrame({
                'date': pd.date_range(last_date + pd.Timedelta(days=1), periods=7),
                'prediction': daily_predictions,
                'q10': daily_quantiles[0.1],
                'q25': daily_quantiles[0.25],
                'q75': daily_quantiles[0.75],
                'q90': daily_quantiles[0.9],
            })
            return week_forecast, daily_forecast
        
        return week_forecast
    
    def check_alert(self, week_forecast, threshold, threshold_type='fixed'):
        """
        检查是否需要报警
        
        参数:
            week_forecast: 预测结果
            threshold: 阈值
            threshold_type: 'fixed' (固定值) 或 'percentile' (基于历史百分位)
        """
        prediction = week_forecast['prediction']
        q90 = week_forecast['q90']
        
        print("\n" + "="*60)
        print("预警检查")
        print("="*60)
        
        if threshold_type == 'fixed':
            if prediction > threshold:
                print(f"⚠️  警告！预测值 {prediction:.2f} 超过阈值 {threshold}")
                print(f"   90%分位数: {q90:.2f}")
                return True
            else:
                print(f"✓ 正常。预测值 {prediction:.2f} 低于阈值 {threshold}")
                return False
        
        return False
    
    def plot_feature_importance(self, top_n=15):
        """绘制特征重要性"""
        if self.model is None:
            return
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.feature_names
        
        # 排序
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance (Gain)')
        plt.title('Top Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        print("\n特征重要性图已保存: feature_importance.png")
    
    def plot_forecast_visualization(self, df, week_forecast, daily_forecast=None):
        """可视化预测结果"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 图1: 历史数据 + 预测
        ax1 = axes[0]
        
        # 显示最近60天的历史数据
        recent_days = 60
        df_recent = df.tail(recent_days).copy()
        
        ax1.plot(df_recent['ds'], df_recent['y'], 'o-', 
                label='Historical Data', alpha=0.7, linewidth=1.5)
        
        if daily_forecast is not None:
            # 预测值
            ax1.plot(daily_forecast['date'], daily_forecast['prediction'], 
                    'ro-', label='Forecast', linewidth=2, markersize=8)
            
            # 置信区间
            ax1.fill_between(daily_forecast['date'], 
                           daily_forecast['q10'], 
                           daily_forecast['q90'],
                           alpha=0.2, color='red', label='80% CI')
            ax1.fill_between(daily_forecast['date'], 
                           daily_forecast['q25'], 
                           daily_forecast['q75'],
                           alpha=0.3, color='red', label='50% CI')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Incidents', fontsize=12)
        ax1.set_title('Incident Forecast - Next Week', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 图2: 周总数预测
        ax2 = axes[1]
        
        # 计算最近几周的总数
        df['week'] = df['ds'].dt.to_period('W')
        weekly_sum = df.groupby('week')['y'].sum().reset_index()
        weekly_sum = weekly_sum.tail(10)  # 最近10周
        
        x_pos = range(len(weekly_sum))
        ax2.bar(x_pos, weekly_sum['y'], alpha=0.6, label='Historical Weekly Total')
        
        # 添加预测周
        next_week_pos = len(weekly_sum)
        ax2.bar(next_week_pos, week_forecast['prediction'], 
               color='red', alpha=0.7, label='Forecasted Week')
        
        # 误差线（置信区间）
        error_lower = week_forecast['prediction'] - week_forecast['q10']
        error_upper = week_forecast['q90'] - week_forecast['prediction']
        ax2.errorbar(next_week_pos, week_forecast['prediction'],
                    yerr=[[error_lower], [error_upper]],
                    fmt='none', ecolor='red', capsize=10, capthick=2,
                    label='80% CI')
        
        # X轴标签
        week_labels = [str(w) for w in weekly_sum['week']] + ['Next Week']
        ax2.set_xticks(range(len(week_labels)))
        ax2.set_xticklabels(week_labels, rotation=45, ha='right')
        
        ax2.set_xlabel('Week', fontsize=12)
        ax2.set_ylabel('Total Incidents', fontsize=12)
        ax2.set_title('Weekly Total Incidents - Forecast vs Historical', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('forecast_visualization.png', dpi=150, bbox_inches='tight')
        print("预测可视化图已保存: forecast_visualization.png")


def evaluate_model(df, forecaster, n_weeks=4):
    """在历史数据上回测模型性能"""
    print("\n" + "="*60)
    print("模型回测评估")
    print("="*60)
    
    predictions = []
    actuals = []
    
    # 回测最近n周
    for i in range(n_weeks, 0, -1):
        # 截取数据（模拟只有到某一天的数据）
        cutoff_date = df['ds'].max() - pd.Timedelta(days=i*7)
        df_train = df[df['ds'] <= cutoff_date].copy()
        
        if len(df_train) < 60:  # 数据不足
            continue
        
        # 训练并预测
        temp_forecaster = IncidentForecaster(window_size=60)
        temp_forecaster.train(df_train, params={
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'seed': 42
        })
        
        week_forecast = temp_forecaster.predict_next_week(df_train)
        
        # 获取实际值
        actual_start = cutoff_date + pd.Timedelta(days=1)
        actual_end = cutoff_date + pd.Timedelta(days=7)
        actual_week = df[(df['ds'] > cutoff_date) & (df['ds'] <= actual_end)]
        actual_total = actual_week['y'].sum()
        
        predictions.append(week_forecast['prediction'])
        actuals.append(actual_total)
        
        print(f"Week ending {actual_end.strftime('%Y-%m-%d')}: "
              f"Predicted={week_forecast['prediction']:.2f}, "
              f"Actual={actual_total:.2f}, "
              f"Error={abs(week_forecast['prediction']-actual_total):.2f}")
    
    if len(predictions) > 0:
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        print(f"\n回测性能指标:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return predictions, actuals