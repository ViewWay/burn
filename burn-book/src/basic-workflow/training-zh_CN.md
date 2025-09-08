# 训练

我们现在准备编写在 MNIST 数据集上训练模型所需的代码。我们将在文件 `src/training.rs` 中定义此训练部分的代码。

模型应该输出一个可以被学习器理解的项目，而不是简单张量，学习器的职责是将优化器应用于模型。输出结构体用于训练期间计算的所有指标。因此，它应包含计算任务所需指标的所有必要信息。

Burn 提供了两种基本输出类型：`ClassificationOutput` 和 `RegressionOutput`。它们实现了与指标一起使用的必要 trait。可以创建自己的项目，但这超出了本指南的范围。

由于 MNIST 任务是一个分类问题，我们将使用 `ClassificationOutput` 类型。

```rust , ignore
# use crate::{
#     data::{MnistBatch, MnistBatcher},
#     model::{Model, ModelConfig},
# };
# use burn::{
#     data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
#     nn::loss::CrossEntropyLossConfig,
#     optim::AdamConfig,
#     prelude::*,
#     record::CompactRecorder,
#     tensor::backend::AutodiffBackend,
#     train::{
#         metric::{AccuracyMetric, LossMetric},
#         ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
#     },
# };
# 
impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}
```

从前面的代码块中可以明显看出，我们使用交叉熵损失模块进行损失计算，不包括任何填充标记。然后我们返回包含损失、带有所有逻辑值的输出张量和目标的分类输出。

请注意，张量操作接收拥有的张量作为输入。要多次使用张量，您需要使用 `clone()` 函数。不用担心；这个过程不会涉及实际复制张量数据。相反，它只会表明张量在多个实例中使用，这意味着某些操作不会就地执行。总之，我们的 API 设计为拥有张量以优化性能。

接下来，我们将继续实现模型的训练和验证步骤。

```rust , ignore
# use crate::{
#     data::{MnistBatch, MnistBatcher},
#     model::{Model, ModelConfig},
# };
# use burn::{
#     data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
#     nn::loss::CrossEntropyLossConfig,
#     optim::AdamConfig,
#     prelude::*,
#     record::CompactRecorder,
#     tensor::backend::AutodiffBackend,
#     train::{
#         metric::{AccuracyMetric, LossMetric},
#         ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
#     },
# };
# 
# impl<B: Backend> Model<B> {
#     pub fn forward_classification(
#         &self,
#         images: Tensor<B, 3>,
#         targets: Tensor<B, 1, Int>,
#     ) -> ClassificationOutput<B> {
#         let output = self.forward(images);
#         let loss = CrossEntropyLossConfig::new()
#             .init(&output.device())
#             .forward(output.clone(), targets.clone());
# 
#         ClassificationOutput::new(loss, output, targets)
#     }
# }
# 
impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
```

在这里，我们在 `TrainStep` 和 `ValidStep` 中将输入和输出类型定义为泛型参数。我们将它们称为 `MnistBatch` 和 `ClassificationOutput`。在训练步骤中，梯度的计算很简单，只需在损失上调用 `backward()`。请注意，与 PyTorch 不同，梯度不存储在每个张量参数旁边，而是由反向传递返回，如下所示：`let gradients = loss.backward();`。可以使用 grad 函数获得参数的梯度：`let grad = tensor.grad(&gradients);`。虽然在使用学习器结构体和优化器时不需要这样做，但在调试或编写自定义训练循环时，这可能非常有用。训练步骤和验证步骤之间的一个区别是，前者要求后端实现 `AutodiffBackend` 而不仅仅是 `Backend`。否则，`backward` 函数不可用，因为后端不支持自动微分。稍后我们将看到如何创建支持自动微分的后端。

<details>
<summary><strong>🦀 方法定义中的泛型类型约束</strong></summary>

尽管在本指南的前面部分已经介绍了泛型数据类型、trait 和 trait 边界，但前面的代码片段可能一开始看起来很多。

在上面的例子中，我们为 `Model` 结构体实现了 `TrainStep` 和 `ValidStep` trait，它在 `Backend` trait 上是泛型的，如前所述。这些 trait 由 `burn::train` 提供，定义了应该为所有结构体实现的通用 `step` 方法。由于 trait 在输入和输出类型上是泛型的，trait 实现必须指定使用的具体类型。这就是额外类型约束出现的地方 `<MnistBatch<B>, ClassificationOutput<B>>`。正如我们之前看到的，批处理的具体输入类型是 `MnistBatch`，前向传递的输出是 `ClassificationOutput`。`step` 方法签名与具体输入和输出类型匹配。

有关定义方法时泛型类型约束的更多详细信息，请查看 Rust Book 的[这一部分](https://doc.rust-lang.org/book/ch10-01-syntax.html#in-method-definitions)。

</details><br>

让我们继续建立实际的训练配置。

```rust , ignore
# use crate::{
#     data::{MnistBatch, MnistBatcher},
#     model::{Model, ModelConfig},
# };
# use burn::{
#     data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
#     nn::loss::CrossEntropyLossConfig,
#     optim::AdamConfig,
#     prelude::*,
#     record::CompactRecorder,
#     tensor::backend::AutodiffBackend,
#     train::{
#         metric::{AccuracyMetric, LossMetric},
#         ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
#     },
# };
# 
# impl<B: Backend> Model<B> {
#     pub fn forward_classification(
#         &self,
#         images: Tensor<B, 3>,
#         targets: Tensor<B, 1, Int>,
#     ) -> ClassificationOutput<B> {
#         let output = self.forward(images);
#         let loss = CrossEntropyLossConfig::new()
#             .init(&output.device())
#             .forward(output.clone(), targets.clone());
# 
#         ClassificationOutput::new(loss, output, targets)
#     }
# }
# 
# impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
#     fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
#         let item = self.forward_classification(batch.images, batch.targets);
# 
#         TrainOutput::new(self, item.loss.backward(), item)
#     }
# }
# 
# impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
#     fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
#         self.forward_classification(batch.images, batch.targets)
#     }
# }
# 
#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // 在获取准确的学习器摘要之前删除现有工件
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("配置应该成功保存");

    B::seed(config.seed);

    let batcher = MnistBatcher::default();

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

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("训练模型应该成功保存");
}
```

使用 `Config` derive 创建实验配置是一个好习惯。在 `train` 函数中，我们首先要确保 `artifact_dir` 存在，使用标准 rust 库进行文件操作。所有检查点、日志和指标都将存储在此目录下。我们使用之前创建的批处理器初始化数据加载器。由于在验证阶段不需要自动微分，`learner.fit(...)` 方法为数据加载器定义了 `B::InnerBackend` 的必要后端边界（参见[后端](./backend.md)）。自动微分功能通过类型系统提供，使得几乎不可能忘记停用梯度计算。

接下来，我们创建学习器，在训练和验证步骤中都包含准确率和损失指标，以及设备和周期。我们还使用 `CompactRecorder` 配置检查点，以指示权重应如何存储。这个结构体实现了 `Recorder` trait，使其能够保存记录以实现持久性。

然后，我们使用模型、优化器和学习率构建学习器。值得注意的是，构建函数的第三个参数实际上应该是一个学习率 _调度器_。在我们的示例中提供浮点数时，它会自动转换为 _常数_ 学习率调度器。学习率不是优化器配置的一部分，这与其他框架中的做法不同，而是在执行优化器步骤时作为参数传递。这避免了必须改变优化器的状态，因此更加函数式。在使用学习器结构体时没有区别，但如果您实现自己的训练循环，这将是一个必须掌握的重要细微差别。

一旦创建了学习器，我们就可以简单地调用 `fit` 并提供训练和验证数据加载器。为了简化此示例，我们使用测试集作为验证集；但是，我们不建议在实际使用中采用这种做法。

最后，训练好的模型由 `fit` 方法返回。然后使用 `CompactRecorder` 保存训练好的权重。这个记录器使用 `MessagePack` 格式，浮点数为半精度 `f16`，整数为 `i16`。还有其他记录器可用，支持各种格式，如 `BinCode` 和 `JSON`，有或没有压缩。任何后端，无论精度如何，都可以加载任何类型的记录数据。