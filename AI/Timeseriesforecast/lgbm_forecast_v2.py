"""
改进版LightGBM时间序列预测模型
优化点:
1. 增加窗口大小到120天，获得更多训练样本
2. 简化特征，减少NaN数量
3. 直接预测周总数而非每日预测累加
4. 添加更多针对性特征
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

class ImprovedIncidentForecaster:
    """改进的Incident预测系统 - 直接预测周总数"""
    
    def __init__(self, window_size=120):
        """
        参数:
            window_size: 滚动窗口大小（天）
        """
        self.window_size = window_size
        self.model = None
        self.quantile_models = {}
        self.feature_names = None
        
    def create_weekly_features(self, df):
        """
        创建周级别特征 - 直接预测周总数
        """
        df = df.copy()
        df['week'] = df['ds'].dt.to_period('W')
        
        # 按周汇总
        weekly_df = df.groupby('week').agg({
            'y': 'sum',
            'ds': 'last'
        }).reset_index()
        weekly_df.columns = ['week', 'weekly_total', 'week_end_date']
        weekly_df = weekly_df.sort_values('week_end_date').reset_index(drop=True)
        
        # 1. Lag特征 (周级别)
        for lag in [1, 2, 3, 4]:
            weekly_df[f'lag_week_{lag}'] = weekly_df['weekly_total'].shift(lag)
        
        # 2. 滚动统计 (周级别)
        for window in [2, 4, 8]:
            weekly_df[f'rolling_mean_{window}w'] = weekly_df['weekly_total'].rolling(window=window).mean()
            weekly_df[f'rolling_std_{window}w'] = weekly_df['weekly_total'].rolling(window=window).std()
            weekly_df[f'rolling_min_{window}w'] = weekly_df['weekly_total'].rolling(window=window).min()
            weekly_df[f'rolling_max_{window}w'] = weekly_df['weekly_total'].rolling(window=window).max()
        
        # 3. 趋势特征
        weekly_df['trend'] = np.arange(len(weekly_df))
        weekly_df['diff_1w'] = weekly_df['weekly_total'].diff(1)
        weekly_df['diff_2w'] = weekly_df['weekly_total'].diff(2)
        
        # 4. 与均值的偏离
        weekly_df['deviation_from_mean_4w'] = weekly_df['weekly_total'] - weekly_df['rolling_mean_4w']
        weekly_df['deviation_from_mean_8w'] = weekly_df['weekly_total'] - weekly_df['rolling_mean_8w']
        
        # 5. 时间特征
        weekly_df['week_of_year'] = weekly_df['week_end_date'].dt.isocalendar().week.astype(int)
        weekly_df['month'] = weekly_df['week_end_date'].dt.month
        weekly_df['quarter'] = weekly_df['week_end_date'].dt.quarter
        
        return weekly_df
    
    def prepare_data(self, df):
        """准备训练数据"""
        # 创建周特征
        weekly_df = self.create_weekly_features(df)
        
        # 使用最近的周数据
        weeks_needed = self.window_size // 7
        if len(weekly_df) > weeks_needed:
            weekly_df = weekly_df.iloc[-weeks_needed:].copy()
        
        # 去除NaN
        weekly_df = weekly_df.dropna()
        
        # 分离特征和目标
        feature_cols = [col for col in weekly_df.columns 
                       if col not in ['week', 'weekly_total', 'week_end_date']]
        
        X = weekly_df[feature_cols]
        y = weekly_df['weekly_total']
        
        self.feature_names = feature_cols
        
        return X, y, weekly_df
    
    def train(self, df, params=None):
        """训练模型"""
        print("开始训练改进模型...")
        print(f"使用最近 {self.window_size} 天数据 (约{self.window_size//7}周)")
        
        X, y, weekly_df = self.prepare_data(df)
        
        print(f"训练样本数: {len(X)} 周")
        print(f"特征数: {len(self.feature_names)}")
        print(f"目标范围: [{y.min():.0f}, {y.max():.0f}], 均值: {y.mean():.0f}")
        
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 15,  # 减少复杂度避免过拟合
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 3,
                'verbose': -1,
                'seed': 42
            }
        
        # 时间序列交叉验证
        if len(X) >= 15:  # 至少15周数据才做交叉验证
            tscv = TimeSeriesSplit(n_splits=min(3, len(X)//5))
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                )
                
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
                cv_scores.append((mae, mape))
                print(f"  Fold {fold+1} MAE: {mae:.2f}, MAPE: {mape:.2f}%")
            
            avg_mae = np.mean([s[0] for s in cv_scores])
            avg_mape = np.mean([s[1] for s in cv_scores])
            print(f"平均 MAE: {avg_mae:.2f}, MAPE: {avg_mape:.2f}%")
        
        # 在全部数据上训练最终模型
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, train_data, num_boost_round=100)
        
        # 训练分位数模型
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
                num_boost_round=100
            )
        
        print("模型训练完成!")
        
        # 特征重要性
        self.plot_feature_importance()
        
        return self
    
    def predict_next_week(self, df):
        """预测下周总数"""
        print("\n开始预测下周...")
        
        # 创建周特征
        weekly_df = self.create_weekly_features(df)
        
        # 获取最后一周的特征
        last_week_features = weekly_df[self.feature_names].iloc[-1:].copy()
        
        # 更新trend (下一周)
        if 'trend' in last_week_features.columns:
            last_week_features['trend'] = last_week_features['trend'] + 1
        
        # 填充缺失值
        last_week_features = last_week_features.ffill().bfill()
        
        # 预测
        pred = self.model.predict(last_week_features)[0]
        
        # 预测分位数
        quantile_preds = {}
        for q, model in self.quantile_models.items():
            quantile_preds[q] = model.predict(last_week_features)[0]
        
        week_forecast = {
            'prediction': pred,
            'q10': quantile_preds[0.1],
            'q25': quantile_preds[0.25],
            'q50': quantile_preds[0.5],
            'q75': quantile_preds[0.75],
            'q90': quantile_preds[0.9],
            'confidence_interval_50': (quantile_preds[0.25], quantile_preds[0.75]),
            'confidence_interval_80': (quantile_preds[0.1], quantile_preds[0.9]),
        }
        
        last_date = df['ds'].max()
        print(f"最后数据日期: {last_date.strftime('%Y-%m-%d')}")
        print(f"\n下周预测总数: {pred:.2f}")
        print(f"  50%置信区间: [{quantile_preds[0.25]:.2f}, {quantile_preds[0.75]:.2f}]")
        print(f"  80%置信区间: [{quantile_preds[0.1]:.2f}, {quantile_preds[0.9]:.2f}]")
        
        return week_forecast
    
    def check_alert(self, week_forecast, threshold):
        """检查预警"""
        prediction = week_forecast['prediction']
        q90 = week_forecast['q90']
        
        print("\n" + "="*60)
        print("预警检查")
        print("="*60)
        
        if prediction > threshold:
            print(f"⚠️  警告！预测值 {prediction:.2f} 超过阈值 {threshold:.2f}")
            print(f"   90%分位数: {q90:.2f}")
            return True
        else:
            print(f"✓ 正常。预测值 {prediction:.2f} 低于阈值 {threshold:.2f}")
            return False
    
    def plot_feature_importance(self, top_n=15):
        """绘制特征重要性"""
        if self.model is None:
            return
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.feature_names
        
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance (Gain)')
        plt.title('Top Feature Importance - Weekly Model')
        plt.tight_layout()
        plt.savefig('feature_importance_v2.png', dpi=150, bbox_inches='tight')
        print("\n特征重要性图已保存: feature_importance_v2.png")


def evaluate_weekly_model(df, n_weeks=8):
    """回测周模型"""
    print("\n" + "="*60)
    print("模型回测评估")
    print("="*60)
    
    predictions = []
    actuals = []
    
    for i in range(n_weeks, 0, -1):
        # 截取数据
        cutoff_date = df['ds'].max() - pd.Timedelta(days=i*7)
        df_train = df[df['ds'] <= cutoff_date].copy()
        
        if len(df_train) < 120:
            continue
        
        # 训练
        forecaster = ImprovedIncidentForecaster(window_size=120)
        forecaster.train(df_train, params={
            'objective': 'regression',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'verbose': -1
        })
        
        week_forecast = forecaster.predict_next_week(df_train)
        
        # 获取实际值
        actual_start = cutoff_date + pd.Timedelta(days=1)
        actual_end = cutoff_date + pd.Timedelta(days=7)
        actual_week = df[(df['ds'] > cutoff_date) & (df['ds'] <= actual_end)]
        actual_total = actual_week['y'].sum()
        
        predictions.append(week_forecast['prediction'])
        actuals.append(actual_total)
        
        error = abs(week_forecast['prediction'] - actual_total)
        pct_error = error / actual_total * 100
        print(f"Week ending {actual_end.strftime('%Y-%m-%d')}: "
              f"Pred={week_forecast['prediction']:.0f}, "
              f"Actual={actual_total:.0f}, "
              f"Error={error:.0f} ({pct_error:.1f}%)")
    
    if len(predictions) > 0:
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        print(f"\n回测性能指标:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return predictions, actuals
