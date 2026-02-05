# Incident时间序列预测和预警系统

## 📋 系统概述

基于**LightGBM + 特征工程**的时间序列预测系统，专门用于预测下周incident总数并提供智能预警。

### 核心特点

- ✅ **应对非平稳性**: 使用60天滚动窗口，自动适应趋势漂移
- ✅ **利用强自相关性**: 通过lag特征和滚动统计捕捉时序依赖
- ✅ **不确定性量化**: 提供预测区间(50%和80%置信区间)
- ✅ **智能预警**: 基于历史统计动态设定阈值
- ✅ **自动回测**: 在历史数据上验证模型性能

---

## 🚀 快速开始

### 1. 环境准备

已配置Python 3.12虚拟环境，依赖包已安装：
- pandas, numpy
- lightgbm, scikit-learn
- matplotlib, scipy, statsmodels

### 2. 运行预测

```bash
.\venv\Scripts\python.exe run_forecast.py
```

### 3. 输出文件

运行后会生成以下文件：

1. **daily_forecast.csv** - 下周7天的每日预测值和置信区间
2. **week_forecast_summary.csv** - 周总数预测摘要（包含预警状态）
3. **forecast_visualization.png** - 预测可视化图表
4. **feature_importance.png** - 特征重要性分析图

---

## 📊 系统架构

### 核心组件

#### 1. 特征工程 (`create_features`)

自动生成31个特征：

**Lag特征** (利用强自相关性)
- lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28

**滚动统计特征** (捕捉局部趋势)
- rolling_mean_7/14/30
- rolling_std_7/14/30  
- rolling_min/max_7/14/30

**时间特征**
- dayofweek, month, quarter, weekofyear
- is_weekend

**趋势和差分特征**
- trend (线性趋势)
- diff_1, diff_7 (一阶差分)
- deviation_from_mean (与滚动均值的偏离)

#### 2. 模型训练 (`train`)

- 使用最近60天数据（滚动窗口）
- 时间序列交叉验证(3-fold)
- 主模型：回归预测
- 5个分位数模型(0.1, 0.25, 0.5, 0.75, 0.9)用于预测区间

#### 3. 预测 (`predict_next_week`)

- 迭代预测未来7天
- 每步使用前一步预测值更新特征
- 输出周总数和每日预测
- 提供50%和80%置信区间

#### 4. 预警检查 (`check_alert`)

动态阈值设定：
```
阈值 = 最近8周均值 + 1.5 × 标准差
```

如果预测值超过阈值，触发预警。

---

## 📈 模型性能

### 回测结果 (最近4周)

| 指标 | 值 |
|------|-----|
| MAE | 206.31 |
| RMSE | 254.28 |
| MAPE | 18.89% |

**解读**: 
- 平均误差约206个incidents/周
- 相对误差约19%
- 考虑到数据的高变异性(CV=0.50)，这是合理的表现

### 特征重要性 Top 5

根据`feature_importance.png`可以看到最重要的特征（通常是）：
1. lag_7 - 上周同一天的值
2. rolling_mean_30 - 30天滚动均值
3. lag_1 - 前一天的值
4. rolling_std_7 - 7天滚动标准差
5. trend - 趋势项

---

## 🔧 参数配置

### 关键参数

在`run_forecast.py`中可以调整：

```python
# 1. 滚动窗口大小
forecaster = IncidentForecaster(
    window_size=60,  # 可调整为30-90天
    retrain_freq=7   # 重训练频率
)

# 2. 阈值敏感度
THRESHOLD = threshold_mean + 1.5 * threshold_std  
# 增大系数(如2.0)降低敏感度，减小(如1.0)提高敏感度

# 3. 历史统计窗口
recent_weeks = weekly_sum.tail(8)  # 使用最近几周
```

### LightGBM参数

在`lgbm_forecast_model.py`的`train`方法中：

```python
params = {
    'num_leaves': 31,        # 增大提高复杂度
    'learning_rate': 0.05,   # 降低提高稳定性
    'feature_fraction': 0.8, # 特征采样率
}
```

---

## 🔄 定期维护

### 建议工作流程

1. **每周运行预测** (自动化)
   ```bash
   .\venv\Scripts\python.exe run_forecast.py
   ```

2. **检查预警状态**
   - 查看`week_forecast_summary.csv`的alert字段
   - alert=1 表示需要预警

3. **重新训练** (每周或检测到漂移时)
   - 系统会自动使用最新60天数据
   - 建议：每7天运行一次

4. **监控模型性能**
   - 对比预测值和实际值
   - 如果MAPE持续>25%，考虑调整参数

---

## 📝 使用示例

### 场景1: 日常预测

```bash
# 运行预测
.\venv\Scripts\python.exe run_forecast.py

# 查看结果
# 1. 打开 forecast_visualization.png 查看可视化
# 2. 读取 week_forecast_summary.csv 获取数值结果
```

### 场景2: 集成到报警系统

```python
import pandas as pd
from lgbm_forecast_model import IncidentForecaster

# 加载数据
df = pd.read_csv('./rawdata/raw_data.csv')
df['ds'] = pd.to_datetime(df['ds'])

# 训练和预测
forecaster = IncidentForecaster(window_size=60)
forecaster.train(df)
week_forecast = forecaster.predict_next_week(df)

# 获取预测值
prediction = week_forecast['prediction']
threshold = 1700  # 你的业务阈值

# 触发报警
if prediction > threshold:
    # 调用你的报警API
    send_alert(f"预警：下周预计{prediction:.0f}个incidents，超过阈值{threshold}")
```

### 场景3: 自定义阈值

```python
# 方法1: 固定阈值
THRESHOLD = 2000

# 方法2: 基于百分位数
threshold = recent_weeks.quantile(0.9)

# 方法3: 基于标准差（更灵敏）
threshold = recent_mean + 1.0 * recent_std  # 降低系数提高敏感度
```

---

## 🎯 数据特征回顾

基于`analyze_data.py`的分析结果：

| 特征 | 值 | 说明 |
|------|-----|------|
| 趋势 | 下降84.5% | 存在明显漂移 |
| Lag1相关性 | 0.76 | 强自相关性 |
| Lag7相关性 | 0.81 | 周期模式明显 |
| 变异系数 | 0.50 | 高波动性 |
| 周效应 | 较弱 | 周日略低 |

**为什么选择LightGBM方案**：
- ✅ 能处理非平稳序列（滚动窗口）
- ✅ 能捕捉强自相关性（lag特征）
- ✅ 能处理高波动（集成学习鲁棒性）
- ✅ 能提供不确定性量化（分位数回归）

---

## ⚠️ 注意事项

1. **数据质量**
   - 确保rawdata/raw_data.csv格式正确(ds, y两列)
   - 日期格式：YYYY/M/D 或 YYYY-MM-DD
   - 数值列不能有缺失值

2. **趋势变化**
   - 如果业务发生重大变化（如系统升级），需要重新评估窗口大小
   - 建议保留历史预测结果，用于监控模型衰减

3. **阈值设定**
   - 初始阈值可能需要根据实际报警反馈调整
   - 平衡假阳性（误报）和假阴性（漏报）

4. **计算资源**
   - 单次预测约需10-30秒
   - 可以在服务器上设置定时任务

---

## 📞 扩展功能

### 未来可以添加：

1. **CUSUM变点检测** 
   - 结合你原有的CUSUM经验
   - 检测到漂移时自动调整窗口或重训练

2. **外部特征**
   - 加入节假日标志
   - 加入其他系统的相关指标

3. **多步预测**
   - 扩展到预测未来2-4周

4. **模型集成**
   - 组合LightGBM + 统计方法
   - 加权平均提高鲁棒性

---

## 🎓 技术细节

### 为什么不用Prophet？

你原来用Prophet，但基于数据分析：
- ❌ Prophet假设稳定的季节性 → 你的数据季节性弱
- ❌ Prophet假设平滑趋势 → 你的数据有突变和漂移
- ❌ Prophet对异常值敏感 → 你的数据有异常值

### LightGBM的优势

- ✅ 基于树模型，对异常值鲁棒
- ✅ 自动学习特征交互
- ✅ 支持分位数回归
- ✅ 训练速度快

### 滚动窗口策略

使用60天窗口的原因：
- 太短(<30天)：数据不足，方差大
- 太长(>90天)：包含过时信息，无法适应漂移
- 60天平衡了稳定性和适应性

---

## 📦 文件说明

```
Timeseriesforecast/
├── rawdata/
│   └── raw_data.csv              # 原始数据
├── lgbm_forecast_model.py        # 核心模型类
├── run_forecast.py               # 主运行脚本
├── analyze_data.py               # 数据分析脚本
├── daily_forecast.csv            # 每日预测输出
├── week_forecast_summary.csv     # 周预测摘要
├── forecast_visualization.png    # 预测可视化
└── feature_importance.png        # 特征重要性
```

---

**系统创建时间**: 2026-02-03  
**建议下次重训练**: 2025-11-01

有问题可以修改参数或联系开发者调整策略！
