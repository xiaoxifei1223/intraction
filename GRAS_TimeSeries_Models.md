# GRAS 时序预测算法选型与冷启动策略

> 针对 GRAS 指标预测场景，基于 Prophet、N-BEATS、ARIMA 三种算法的技术选型、优势对比与冷启动实施方案

---

## 一、Python 包支持情况

三种算法均有成熟的 Python 生态支持：

### 1.1 Prophet

- **包名**：`prophet` (Meta/Facebook 开源)
- **安装**：
  ```bash
  pip install prophet
  ```
- **依赖**：需要 `pystan`（较重，编译时间较长）
- **版本建议**：`prophet>=1.1`
- **文档**：https://facebook.github.io/prophet/

### 1.2 N-BEATS

- **包名**：`darts` (Darts 库包含 N-BEATS 实现，推荐)
- **安装**：
  ```bash
  pip install darts
  ```
- **或者**：`neuralforecast` 包也有实现
  ```bash
  pip install neuralforecast
  ```
- **依赖**：PyTorch (需要 GPU 支持以获得最佳性能)
- **文档**：https://unit8co.github.io/darts/

### 1.3 ARIMA

- **包名**：
  - `statsmodels` (最常用，功能全面)
  - `pmdarima` (提供 AutoARIMA，自动超参数搜索)
- **安装**：
  ```bash
  pip install statsmodels
  # 或
  pip install pmdarima
  ```
- **依赖**：轻量级，无深度学习依赖
- **文档**：https://www.statsmodels.org/

---

## 二、三种算法优势对比

### 2.1 Prophet 优势

#### 适用场景
- **有明显周期性、节假日影响的业务指标**
- 例如：流量指标、错误率、用户活跃度等

#### 核心优势
✅ **业务友好**
- 可解释性强，能明确看到趋势、季节性、节假日效应
- 输出结果易于向业务人员说明

✅ **缺失值容忍**
- 对缺失数据和异常值具有鲁棒性
- 无需完美的数据预处理

✅ **节假日建模**
- 内置节假日效果建模（非常适合 GRAS 场景）
- 可自定义发布窗口、大促等特殊事件

✅ **快速上手**
- API 简单直观
- 超参数少，调优成本低

#### 主要局限
❌ 对短期/高频数据效果一般  
❌ 不适合复杂非线性模式  
❌ 长期预测（>30 天）准确度衰减

#### 典型应用
```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False
)

# 添加节假日
holidays = pd.DataFrame({
    'holiday': 'release_window',
    'ds': pd.to_datetime(['2024-01-15', '2024-02-01']),
    'lower_window': -1,
    'upper_window': 1
})
model.add_country_holidays(country_name='CN')
model = model.add_holidays(holidays)

model.fit(df_train)
forecast = model.predict(future)
```

---

### 2.2 N-BEATS 优势

#### 适用场景
- **数据量大、模式复杂、需要长期预测**
- 例如：容量规划、资源预测等

#### 核心优势
✅ **深度学习 SOTA**
- 纯神经网络架构，无需手动特征工程
- 在 M4 竞赛等基准测试中表现优异

✅ **多步预测强**
- 一次可预测未来多个时间点（如未来 7 天）
- 预测精度衰减较慢

✅ **非线性能力**
- 能捕捉复杂、非线性的时序模式
- 适合高度动态的业务场景

✅ **可解释分解**
- 可输出趋势 + 季节性分量（类似 Prophet）
- 支持 interpretable 模式

#### 主要局限
❌ 训练慢，需要 GPU 加速  
❌ 数据量要求高（至少几百个样本）  
❌ 超参数调优复杂  
❌ 模型文件较大，部署成本高

#### 典型应用
```python
from darts.models import NBEATSModel
from darts import TimeSeries

# 转换为 Darts 时间序列格式
series = TimeSeries.from_dataframe(df, 'timestamp', 'value')

# 训练 N-BEATS
model = NBEATSModel(
    input_chunk_length=30,
    output_chunk_length=7,
    n_epochs=100,
    batch_size=32
)

model.fit(series)
forecast = model.predict(n=7)
```

---

### 2.3 ARIMA 优势

#### 适用场景
- **短期预测、数据量少、线性趋势为主**
- 例如：短期容量告警、近期趋势判断

#### 核心优势
✅ **经典可靠**
- 统计学基础扎实，结果可解释
- 学术界和工业界长期验证

✅ **小样本友好**
- 数据量少时依然有效（最低 30-50 个样本）
- 适合冷启动场景

✅ **轻量快速**
- 无需 GPU，推理极快（毫秒级）
- 适合实时在线预测

✅ **置信区间**
- 自带预测区间估计（上下界）
- 符合统计规范

#### 主要局限
❌ 不适合复杂季节性（需要 SARIMA 扩展）  
❌ 需要数据平稳（可能需要差分处理）  
❌ 长期预测衰减快（>7 天效果差）  
❌ 超参数（p, d, q）选择需要经验

#### 典型应用
```python
from statsmodels.tsa.arima.model import ARIMA

# 训练 ARIMA
model = ARIMA(df['value'], order=(1, 1, 1))
fitted = model.fit()

# 预测未来 7 天
forecast = fitted.forecast(steps=7)
conf_int = fitted.get_forecast(steps=7).conf_int()
```

---

## 三、算法对比总结表

| 维度 | Prophet | N-BEATS | ARIMA |
|------|---------|---------|-------|
| **最低数据量** | 14-30 天 | 100+ 样本 | 30-50 样本 |
| **训练速度** | 快 (秒级) | 慢 (分钟-小时) | 快 (秒级) |
| **推理速度** | 快 | 中等 | 极快 |
| **可解释性** | 强 | 中等 | 强 |
| **季节性支持** | 强 (多层) | 强 | 弱 (需 SARIMA) |
| **节假日建模** | 原生支持 | 需手动特征 | 不支持 |
| **非线性能力** | 弱 | 强 | 弱 |
| **长期预测** | 中等 (7-30 天) | 强 (30+ 天) | 弱 (3-7 天) |
| **超参数调优** | 简单 | 复杂 | 中等 |
| **GPU 需求** | 否 | 是 | 否 |
| **适用场景** | 周期业务指标 | 复杂长期预测 | 短期快速预测 |

---

## 四、无前期数据时的冷启动策略

这是 GRAS 新应用上线时的核心挑战。以下是四种务实方案：

### 4.1 策略 1：迁移学习（Transfer Learning）— **推荐**

#### 核心思路
从相似应用/指标借用历史数据，训练"通用模型"，为新应用提供初始预测能力。

#### 实施步骤

**Step 1：识别相似应用**
- 同技术栈（如都是 Java 微服务）
- 同业务类型（如都是 API 网关）
- 同指标类型（如都是 `G.error_rate`）

**Step 2：聚合历史数据**
```python
# 收集多个相似应用的历史数据
similar_apps = ['app_b', 'app_c', 'app_d']
df_train = pd.concat([
    fetch_metric_history('app_b', 'G.error_rate', days=60),
    fetch_metric_history('app_c', 'G.error_rate', days=60),
    fetch_metric_history('app_d', 'G.error_rate', days=60)
])

df_train['ds'] = df_train['timestamp']
df_train['y'] = df_train['value']
```

**Step 3：训练通用模型**
```python
from prophet import Prophet

model_generic = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # 控制趋势灵活度
)

# 添加 GRAS 场景的关键事件
holidays = pd.DataFrame({
    'holiday': ['release_window', 'major_promotion'],
    'ds': pd.to_datetime(['2024-01-15', '2024-02-01']),
    'lower_window': [-1, -2],
    'upper_window': [1, 1]
})
model_generic = model_generic.add_holidays(holidays)

model_generic.fit(df_train)
```

**Step 4：新应用冷启动使用**
```python
# 新应用 app_new 上线，立即用通用模型预测
future = model_generic.make_future_dataframe(periods=7)
forecast_generic = model_generic.predict(future)

# 写入预测结果（标记为通用模型）
write_to_db(
    app_id='app_new',
    metric_id='G.error_rate',
    forecast=forecast_generic,
    source='AI_MODEL_GENERIC',
    model_version='generic_v1.0'
)
```

**Step 5：数据积累后切换专属模型**
```python
# 2 周后，app_new 积累足够数据
if len(app_new_history) >= 14:
    model_specific = Prophet()
    model_specific.fit(app_new_history)
    
    # 在 decision 层逐步提升专属模型权重
    # 从 generic 0.8 / specific 0.2 → generic 0.2 / specific 0.8
```

#### 适用模型
- ✅ **Prophet**：直接合并多应用数据训练
- ✅ **N-BEATS**：可在多应用数据上预训练，再 fine-tune
- ❌ **ARIMA**：不太适合（过于依赖单序列特性）

---

### 4.2 策略 2：合成数据 + 规则生成

#### 核心思路
利用现有 Java 规则引擎，"模拟"生成一段伪历史数据，用于初始化 AI 模型。

#### 实施步骤

**Step 1：规则反推历史**
```python
# 假设 Java 规则：error_rate = f(request_count, deploy_frequency, ...)
# 用过去 30 天的上下文数据 + 规则反推

synthetic_history = []

for day in past_30_days:
    # 获取该天的上下文（请求量、发布次数等）
    context = fetch_context(day)
    
    # 用 Java 规则计算该天的理论指标值
    predicted_value = java_rule_engine.calculate(
        request_count=context['requests'],
        deploy_count=context['deploys'],
        is_holiday=context['is_holiday']
    )
    
    synthetic_history.append({
        'ds': day,
        'y': predicted_value
    })

df_synthetic = pd.DataFrame(synthetic_history)
```

**Step 2：用合成数据训练初始模型**
```python
model = Prophet()
model.fit(df_synthetic)

# 立即可用于预测
forecast = model.predict(model.make_future_dataframe(periods=7))
```

**Step 3：真实数据替代**
```python
# 一旦有真实数据（如 3 天后）
if len(real_data) > 0:
    # 混合训练：80% 合成 + 20% 真实
    df_mixed = pd.concat([
        df_synthetic.sample(frac=0.8),
        real_data
    ])
    model.fit(df_mixed)

# 逐步降低合成数据权重，直到完全使用真实数据
```

#### 优缺点
- ✅ **优点**：有总比没有强，能立即"跑起来"
- ⚠️ **缺点**：合成数据可能不真实，需快速用真实数据替代
- 💡 **建议**：仅作为 1-2 周的过渡方案

---

### 4.3 策略 3：分阶段启动（Bootstrap）— **稳妥方案**

#### 核心思路
第一阶段只用 Java 规则，积累真实数据后再逐步引入 AI。

#### 实施时间线

**Phase 1：冷启动期（0-2 周）**
- **策略**：
  - 只用 Java 规则预测
  - 写入 `metric_prediction_history (source='JAVA_RULE')`
  - 同时收集真实指标值 → `metric_value_history`
  
- **代码示例**：
  ```python
  # Decision 层配置
  prediction_weights = {
      'JAVA_RULE': 1.0,
      'AI_MODEL': 0.0  # AI 暂不启用
  }
  
  final_prediction = (
      java_rule_prediction * prediction_weights['JAVA_RULE']
  )
  ```

**Phase 2：混合期（2-4 周）**
- **条件**：数据积累 ≥14 天（Prophet 最低要求）
- **策略**：
  - 开始训练 Prophet/ARIMA
  - 与规则并行，标记为 `source='AI_MODEL'`
  - Decision 层对比两者，逐步提升 AI 权重
  
- **代码示例**：
  ```python
  # 训练第一个 AI 模型
  if len(real_data) >= 14:
      model_prophet = Prophet()
      model_prophet.fit(real_data)
      
      # 权重逐步调整
      prediction_weights = {
          'JAVA_RULE': 0.7,
          'AI_MODEL': 0.3  # 初始给予 30% 权重
      }
  
  # Decision 层融合
  final_prediction = (
      java_rule_prediction * 0.7 +
      ai_prediction * 0.3
  )
  ```

**Phase 3：成熟期（1-3 月后）**
- **条件**：数据 ≥100 样本
- **策略**：
  - 尝试 N-BEATS（如需高精度）
  - 根据历史对比（Actual vs Forecast MAPE），动态调整权重
  - 可能完全切换到 AI（规则仅作兜底）
  
- **代码示例**：
  ```python
  # 根据历史准确率动态调权
  mape_rule = calculate_mape(java_rule_history, actual_history)
  mape_ai = calculate_mape(ai_history, actual_history)
  
  # 权重与准确率反向关联
  total_error = mape_rule + mape_ai
  weight_rule = (1 - mape_rule / total_error) if total_error > 0 else 0.5
  weight_ai = (1 - mape_ai / total_error) if total_error > 0 else 0.5
  
  # 归一化
  total_weight = weight_rule + weight_ai
  final_weights = {
      'JAVA_RULE': weight_rule / total_weight,
      'AI_MODEL': weight_ai / total_weight
  }
  ```

---

### 4.4 策略 4：外部数据增强

#### 核心思路
即使没有历史指标值，也可引入"已知事件"特征辅助建模。

#### 可用外部数据

**1. 日历特征**
- 工作日 / 周末
- 节假日（国家法定假日）
- 月初 / 月末（可能影响业务）

**2. 业务事件**
- 发布窗口（从 CI/CD 系统获取）
- 大促活动（从运营日历获取）
- 系统维护窗口

**3. 全局指标**
- 整体流量趋势（如果新应用是从老系统迁移）
- 行业基准数据

#### 实施方式
```python
from prophet import Prophet
import pandas as pd

# 构建"事件骨架"（无需历史指标值）
model = Prophet()

# 添加已知事件
holidays = pd.DataFrame({
    'holiday': 'release',
    'ds': pd.to_datetime(['2024-01-15', '2024-02-01', '2024-02-15']),
    'lower_window': -1,
    'upper_window': 1
})
model = model.add_holidays(holidays)

# 添加外部回归变量（如全局流量）
model.add_regressor('global_traffic')

# 即使没有目标变量历史，也可以先"预热"模型结构
# 等真实数据来了，快速 fit
```

---

## 五、针对 GRAS 的推荐组合策略

根据 GRAS 双源（Java 规则 + AI）架构，推荐分阶段组合：

### 5.1 启动期（数据 < 2 周）

**主力模型**
- Java 规则（确定性强，可立即使用）

**备选模型**
- ARIMA（如果有任何相似指标的少量历史）
- 通用 Prophet 模型（基于迁移学习）

**Decision 层配置**
```python
prediction_weights = {
    'JAVA_RULE': 0.8,
    'AI_GENERIC': 0.2  # 通用模型辅助
}
```

---

### 5.2 成长期（数据 2 周 - 2 月）

**主力模型**
- Prophet（易用、可解释、适合周期性）
- 专属模型开始训练

**辅助模型**
- Java 规则（兜底 + 对比验证）
- ARIMA（用于短期预测）

**Decision 层配置**
```python
# 动态调整权重
if data_quality_score > 0.8:
    prediction_weights = {
        'JAVA_RULE': 0.4,
        'PROPHET': 0.6
    }
else:
    prediction_weights = {
        'JAVA_RULE': 0.6,
        'PROPHET': 0.4
    }
```

---

### 5.3 成熟期（数据 > 2 月）

**主力模型**
- N-BEATS（追求精度，用于关键指标）
- 或 Prophet（追求稳定，用于一般指标）

**辅助模型**
- ARIMA（短期预测，< 3 天）
- Java 规则（兜底，防止 AI 失效）

**Decision 层配置**
```python
# 基于历史准确率自适应
mape_scores = {
    'JAVA_RULE': calculate_mape_last_30d('JAVA_RULE'),
    'PROPHET': calculate_mape_last_30d('PROPHET'),
    'NBEATS': calculate_mape_last_30d('NBEATS')
}

# 逆 MAPE 归一化权重
weights = inverse_normalize(mape_scores)

# 最终预测
final_prediction = sum(
    model_predict(model) * weights[model]
    for model in ['JAVA_RULE', 'PROPHET', 'NBEATS']
)
```

---

## 六、完整冷启动实施示例

### 6.1 场景描述
- 新应用 `app_new` 刚上线
- 指标：`G.error_rate`
- 无任何历史数据

### 6.2 实施代码

```python
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

# ============================================
# Step 1: 准备"种子数据"（迁移学习）
# ============================================

# 从相似应用收集历史数据
similar_apps = ['app_a', 'app_b', 'app_c']
seed_data = []

for app_id in similar_apps:
    df = fetch_metric_history(
        app_id=app_id,
        metric='G.error_rate',
        days=60
    )
    df['ds'] = pd.to_datetime(df['timestamp'])
    df['y'] = df['value']
    seed_data.append(df[['ds', 'y']])

df_seed = pd.concat(seed_data, ignore_index=True)

# ============================================
# Step 2: 训练通用模型
# ============================================

model_generic = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

# 添加 GRAS 关键事件
holidays = pd.DataFrame({
    'holiday': 'release_window',
    'ds': pd.to_datetime([
        '2024-01-15', '2024-02-01', '2024-02-15',
        '2024-03-01', '2024-03-15', '2024-04-01'
    ]),
    'lower_window': -1,
    'upper_window': 1
})

model_generic.add_country_holidays(country_name='CN')
model_generic = model_generic.add_holidays(holidays)

model_generic.fit(df_seed)

# ============================================
# Step 3: 新应用立即使用通用模型
# ============================================

# 预测未来 7 天
future = model_generic.make_future_dataframe(periods=7)
forecast_generic = model_generic.predict(future)

# 写入数据库（Hot Storage）
write_prediction_to_dynamodb(
    app_id='app_new',
    metric_id='G.error_rate',
    forecast_value=forecast_generic['yhat'].iloc[-1],
    lower_bound=forecast_generic['yhat_lower'].iloc[-1],
    upper_bound=forecast_generic['yhat_upper'].iloc[-1],
    source='AI_MODEL_GENERIC',
    model_version='generic_v1.0',
    horizon='7d'
)

# 写入历史表（Warm Storage）
for idx, row in forecast_generic.tail(7).iterrows():
    write_prediction_history_to_postgres(
        app_id='app_new',
        metric_id='G.error_rate',
        target_time=row['ds'],
        forecast_value=row['yhat'],
        lower_bound=row['yhat_lower'],
        upper_bound=row['yhat_upper'],
        source='AI_MODEL_GENERIC',
        model_version='generic_v1.0'
    )

# ============================================
# Step 4: 并行运行 Java 规则（兜底）
# ============================================

rule_prediction = java_rule_engine.predict_error_rate(
    app_id='app_new',
    context=get_current_context()
)

write_prediction_to_dynamodb(
    app_id='app_new',
    metric_id='G.error_rate',
    forecast_value=rule_prediction,
    source='JAVA_RULE',
    model_version='rule_v1.0'
)

# ============================================
# Step 5: Decision 层融合（冷启动期权重）
# ============================================

weights = {
    'JAVA_RULE': 0.7,        # 规则更可靠
    'AI_MODEL_GENERIC': 0.3  # 通用模型辅助
}

final_prediction = (
    rule_prediction * weights['JAVA_RULE'] +
    forecast_generic['yhat'].iloc[-1] * weights['AI_MODEL_GENERIC']
)

# 写入最终预测（标记为 FUSION）
write_prediction_to_dynamodb(
    app_id='app_new',
    metric_id='G.error_rate',
    forecast_value=final_prediction,
    source='FUSION',
    model_version='bootstrap_v1.0'
)

# ============================================
# Step 6: 定期检查，切换专属模型
# ============================================

def check_and_upgrade_model(app_id, metric_id):
    """
    定期任务：检查数据积累情况，决定是否切换模型
    """
    real_data = fetch_metric_history(app_id, metric_id, days=30)
    
    if len(real_data) >= 14:
        print(f"[{app_id}] 数据充足，训练专属模型...")
        
        # 训练专属 Prophet
        model_specific = Prophet()
        model_specific.fit(real_data)
        
        # 更新权重配置
        update_decision_weights(
            app_id=app_id,
            weights={
                'JAVA_RULE': 0.5,
                'AI_MODEL_SPECIFIC': 0.5  # 专属模型权重提升
            }
        )
        
        print(f"[{app_id}] 已切换至专属模型")
    
    if len(real_data) >= 100:
        print(f"[{app_id}] 数据成熟，尝试 N-BEATS...")
        # 训练 N-BEATS（如需要）
        # ...

# 每天运行一次检查
schedule.every().day.at("02:00").do(
    check_and_upgrade_model,
    app_id='app_new',
    metric_id='G.error_rate'
)
```

---

## 七、关键决策点总结

### 7.1 模型选择决策树

```
START
  |
  ├─ 数据量 < 2 周?
  │    ├─ YES → 使用 Java 规则 + 通用 Prophet (迁移学习)
  │    └─ NO → 继续
  |
  ├─ 数据量 2 周 - 2 月?
  │    ├─ 有明显周期? → Prophet (推荐)
  │    ├─ 短期预测 < 7 天? → ARIMA
  │    └─ 数据质量差? → ARIMA (鲁棒)
  |
  └─ 数据量 > 2 月?
       ├─ 追求极致精度? → N-BEATS
       ├─ 追求稳定可靠? → Prophet
       └─ 多模型融合 → Prophet + ARIMA + N-BEATS (ensemble)
```

### 7.2 冷启动策略优先级

1. **迁移学习**（最推荐）
   - 适用：有相似应用历史
   - 效果：★★★★★
   - 实施难度：★★★☆☆

2. **分阶段启动**（最稳妥）
   - 适用：任何场景
   - 效果：★★★★☆
   - 实施难度：★★☆☆☆

3. **合成数据**（快速但不准）
   - 适用：规则引擎完善
   - 效果：★★★☆☆
   - 实施难度：★★★★☆

4. **外部数据增强**（辅助手段）
   - 适用：配合其他策略
   - 效果：★★☆☆☆
   - 实施难度：★★★☆☆

---

## 八、监控与评估指标

### 8.1 模型性能指标

在 Decision 层需持续监控以下指标：

```python
# 1. MAPE (平均绝对百分比误差)
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# 2. RMSE (均方根误差)
rmse = np.sqrt(np.mean((actual - predicted) ** 2))

# 3. Coverage (预测区间覆盖率)
coverage = np.mean(
    (actual >= lower_bound) & (actual <= upper_bound)
)

# 4. 提前量准确率
# 对于 GRAS，关键是能否提前 X 小时预警
early_warning_accuracy = calculate_early_warning_hit_rate(
    predictions=predictions,
    actuals=actuals,
    lead_time_hours=2
)
```

### 8.2 写入监控表

```sql
-- 在 PostgreSQL 中记录模型评估
CREATE TABLE model_evaluation (
    id              BIGSERIAL PRIMARY KEY,
    app_id          VARCHAR(100) NOT NULL,
    metric_id       VARCHAR(100) NOT NULL,
    model_source    VARCHAR(50) NOT NULL,  -- 'PROPHET'/'NBEATS'/'ARIMA'/'JAVA_RULE'
    eval_date       DATE NOT NULL,
    mape            DOUBLE PRECISION,
    rmse            DOUBLE PRECISION,
    coverage        DOUBLE PRECISION,
    sample_count    INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_model_eval_app_metric_date
    ON model_evaluation (app_id, metric_id, eval_date);
```

---

## 九、参考资料

### 官方文档
- Prophet: https://facebook.github.io/prophet/
- Darts (N-BEATS): https://unit8co.github.io/darts/
- Statsmodels (ARIMA): https://www.statsmodels.org/

### 论文
- N-BEATS: Neural basis expansion analysis for interpretable time series forecasting (ICLR 2020)
- Prophet: Forecasting at Scale (PeerJ 2018)

### 最佳实践
- AWS Forecast Best Practices: https://docs.aws.amazon.com/forecast/
- Google Cloud AI Platform Time Series: https://cloud.google.com/solutions/machine-learning/
