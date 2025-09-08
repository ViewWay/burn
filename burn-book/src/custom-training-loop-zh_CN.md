# 自定义训练循环

尽管 Burn 提供了一个专门用于简化训练的项目，但这并不意味着您必须使用它。有时您可能对训练有特殊需求，重新实现训练循环可能更快。此外，您可能只是更喜欢实现自己的训练循环，而不是使用预构建的训练循环。

Burn 为您提供了支持！

我们将从 [基本工作流程](./basic-workflow) 部分中展示的相同示例开始，但不使用 [Learner](file:///Users/yimiliya/RustroverProjects/burn/crates/burn-train/src/learner/base.rs#L54-L131) 结构体。

```rust, ignore
#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: &B::Device) {
    // 创建配置。
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    B::seed(config.seed);

    // 创建模型和优化器。
    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    // 创建批处理器。
    let batcher = MnistBatcher::default();

    // 创建数据加载器。
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    ...
}
```

正如在前面的示例中看到的，配置和数据加载器的设置没有改变。现在，让我们继续编写自己的训练循环：

```rust, ignore
pub fn run<B: AutodiffBackend>(device: B::Device) {
    ...

    // 迭代训练和验证循环 X 个周期。
    for epoch in 1..config.num_epochs + 1 {
        // 实现我们的训练循环。
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.images);
            let loss = CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone(), batch.targets.clone());
            let accuracy = accuracy(output, batch.targets);

            println!(
                "[训练 - 周期 {} - 迭代 {}] 损失 {:.3} | 准确率 {:.3} %",
                epoch,
                iteration,
                loss.clone().into_scalar(),
                accuracy,
            );

            // 当前反向传播的梯度
            let grads = loss.backward();
            // 与模型每个参数关联的梯度。
            let grads = GradientsParams::from_grads(grads, &model);
            // 使用优化器更新模型。
            model = optim.step(config.lr, model, grads);
        }

        // 获取没有自动微分的模型。
        let model_valid = model.valid();

        // 实现我们的验证循环。
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.images);
            let loss = CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone(), batch.targets.clone());
            let accuracy = accuracy(output, batch.targets);

            println!(
                "[验证 - 周期 {} - 迭代 {}] 损失 {} | 准确率 {}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
                accuracy,
            );
        }
    }
}
```

在前面的代码片段中，我们可以观察到循环从周期 `1` 开始到 `num_epochs` 结束。在每个周期内，我们迭代训练数据加载器。在此过程中，我们执行前向传播，这对于计算损失和准确率是必要的。为了保持简单，我们将结果打印到 stdout。

获得损失后，我们可以调用 `backward()` 函数，它返回每个变量的特定梯度。需要注意的是，我们需要使用 `GradientsParams` 类型将这些梯度映射到相应的参数。这一步很重要，因为您可能会运行多个不同的自动微分图，并为每个参数 ID 累积梯度。

最后，我们可以使用学习率、模型和计算出的梯度执行优化步骤。值得一提的是，与 PyTorch 不同，您无需向优化器注册梯度，也无需调用 `zero_grad`。梯度会在优化步骤中自动消耗。如果您对梯度累积感兴趣，可以通过使用 `GradientsAccumulator` 轻松实现。

```rust, ignore
let mut accumulator = GradientsAccumulator::new();
let grads = model.backward();
let grads = GradientsParams::from_grads(grads, &model);
accumulator.accumulate(&model, grads); ...
let grads = accumulator.grads(); // 弹出累积的梯度。
```

注意，在每个周期后，我们包含一个验证循环来评估模型在以前未见过的数据上的性能。为了在验证步骤中禁用梯度跟踪，我们可以调用 `model.valid()`，它提供了一个没有自动微分功能的内部后端模型。需要强调的是，我们已经声明验证批处理器在内部后端上，即 `MnistBatcher<B::InnerBackend>`；不使用 `model.valid()` 将导致编译错误。

您可以在 [示例](https://github.com/tracel-ai/burn/tree/main/examples/custom-training-loop) 中找到上述代码供您测试。

## 多个优化器

为模型的不同部分设置不同的学习率、优化器参数或使用完全不同的优化器是常见做法。在 Burn 中，每个 `GradientParams` 只能包含实际应用优化器的梯度子集。这允许您灵活地混合和匹配优化器！

```rust,ignore
// 首先计算所有梯度
let grads = loss.backward();

// 现在将梯度分割成各个部分。
let grads_conv1 = GradientParams::from_module(&mut grads, &model.conv1);
let grads_conv2 = GradientParams::from_module(&mut grads, &model.conv2);

// 您可以使用这些梯度更新模型，为每个参数使用不同的学习率。您也可以在这里使用完全不同的优化器！
model = optim.step(config.lr * 2.0, model, grads_conv1);
model = optim.step(config.lr * 4.0, model, grads_conv2);

// 为了更精细的控制，您可以分离单个参数
// 例如，线性偏置通常需要较小的学习率。
if let Some(bias) == model.linear1.bias {
    let grads_bias = GradientParams::from_params(&mut grads, &model.linear1, &[bias.id]);
    model = optim.step(config.lr * 0.1, model, grads_bias);
}

// 注意，上述调用会移除梯度，因此我们可以获取所有"剩余"的梯度。
let grads = GradientsParams::from_grads(grads, &model);
model = optim.step(config.lr, model, grads);
```

## 自定义类型

上面的解释演示了如何创建基本的训练循环。但是，您可能会发现使用中间类型组织程序是有益的。有多种方法可以做到这一点，但这需要熟悉泛型。

如果您希望将优化器和模型组合到同一结构中，您有几个选项。需要注意的是，优化器 trait 依赖于 `AutodiffModule` trait 和 `AutodiffBackend` trait，而模块只依赖于 `AutodiffBackend` trait。

让我们更仔细地看看如何创建您的类型：

**创建一个在后端和优化器上泛型的结构体，使用预定义模型。**

```rust, ignore
struct Learner<B, O>
where
    B: AutodiffBackend,
{
    model: Model<B>,
    optim: O,
}
```

这相当简单。您可以对后端进行泛型化，因为它在此情况下与具体类型 `Model` 一起使用。

**创建一个在模型和优化器上泛型的结构体。**

```rust, ignore
struct Learner<M, O> {
    model: M,
    optim: O,
}
```

这个选项是声明结构体的直观方式。在定义结构体时，您不需要使用 `where` 语句编写类型约束；您可以等到实现实际功能时再写。但是，使用这个结构体时，在尝试为结构体实现代码块时可能会遇到一些问题。

```rust, ignore
impl<B, M, O> Learner<M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    pub fn step(&mut self, _batch: MnistBatch<B>) {
        //
    }
}
```

这将导致以下编译错误：

```console
1. 类型参数 `B` 不受 impl trait、self 类型或谓词的约束
   未约束的类型参数 [E0207]
```

为了解决这个问题，您有两个选项。第一个是使您的函数在后端上泛型化，并在其定义中添加 trait 约束：

```rust, ignore
#[allow(dead_code)]
impl<M, O> Learner2<M, O> {
    pub fn step<B: AutodiffBackend>(&mut self, _batch: MnistBatch<B>)
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
        O: Optimizer<M, B>,
    {
        //
    }
}
```

然而，有些人可能更喜欢在实现块本身上有约束。在这种情况下，您可以使用 `PhantomData<B>` 使您的结构体在后端上泛型化。

**创建一个在后端、模型和优化器上泛型的结构体。**

```rust, ignore
struct Learner3<B, M, O> {
    model: M,
    optim: O,
    _b: PhantomData<B>,
}
```

您可能想知道为什么需要 `PhantomData`。在声明结构体时，每个泛型参数都必须用作字段。当您不需要泛型参数时，可以使用 `PhantomData` 将其标记为零大小类型。

这些只是关于如何定义自己类型的一些建议，但您可以自由使用任何您喜欢的模式。