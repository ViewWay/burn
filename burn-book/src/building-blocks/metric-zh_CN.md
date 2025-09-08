# 指标

在使用学习器时，您可以选择记录在整个训练过程中监控的指标。我们目前提供有限范围的指标。

| 指标           | 描述                                             |
| ---------------- | ------------------------------------------------------- |
| 准确率         | 计算准确率百分比                    |
| TopK准确率     | 计算top-k准确率百分比              |
| 精确率        | 计算精确率百分比                       |
| 召回率           | 计算召回率百分比                          |
| Fβ分数       | 计算F<sub>β </sub>分数百分比             |
| AUROC            | 计算ROC曲线下面积百分比     |
| 损失             | 输出用于反向传播的损失              |
| CPU温度  | 获取CPU温度                           |
| CPU使用率        | 获取CPU利用率                               |
| CPU内存使用率 | 获取CPU RAM使用率                                 |
| GPU温度  | 获取GPU温度                               |
| 学习率    | 获取每个优化器步骤的当前学习率 |
| CUDA             | 获取一般CUDA指标，如利用率          |

为了使用指标，您的训练步骤输出必须实现来自`burn-train::metric`的`Adaptor` trait。以下是分类输出的示例，该示例已随crate提供。

```rust , ignore
/// 适用于多种指标的简单分类输出。
#[derive(new)]
pub struct ClassificationOutput<B: Backend> {
    /// 损失。
    pub loss: Tensor<B, 1>,

    /// 输出。
    pub output: Tensor<B, 2>,

    /// 目标。
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
```

# 自定义指标

通过实现`Metric` trait来生成您自己的自定义指标。

```rust , ignore

/// 指标trait。
///
/// # 注意
///
/// 实现应该定义自己的输入类型，仅由指标使用。
/// 这很重要，因为当模型输出适应每个指标的输入类型时可能会发生冲突。
pub trait Metric: Send + Sync {
    /// 指标的输入类型。
    type Input;

    /// 指标的参数化名称。
    ///
    /// 这应该是唯一的，所以避免使用短的通用名称，更倾向于使用长名称。
    ///
    /// 对于可以在不同参数下存在的指标（例如，不同k值的top-k准确率），
    /// 名称对于每个实例都应该是唯一的。
    fn name(&self) -> String;

    /// 更新指标状态并返回当前指标条目。
    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry;
    /// 清除指标状态。
    fn clear(&mut self);
}
```

作为示例，让我们看看损失指标是如何实现的。

```rust, ignore
/// 损失指标。
#[derive(Default)]
pub struct LossMetric<B: Backend> {
    state: NumericMetricState,
    _b: B,
}

/// 损失指标输入类型。
#[derive(new)]
pub struct LossInput<B: Backend> {
    tensor: Tensor<B, 1>,
}


impl<B: Backend> Metric for LossMetric<B> {
    type Input = LossInput<B>;

    fn update(&mut self, loss: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size] = loss.tensor.dims();
        let loss = loss
            .tensor
            .clone()
            .mean()
            .into_data()
            .iter::<f64>()
            .next()
            .unwrap();

        self.state.update(
            loss,
            batch_size,
            FormatOptions::new(self.name()).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> String {
        "Loss".to_string()
    }
}
```

当您实现的指标本质上是数值型时，您可能还想实现`Numeric` trait。这将允许您的指标被绘制。

```rust, ignore
impl<B: Backend> Numeric for LossMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
```