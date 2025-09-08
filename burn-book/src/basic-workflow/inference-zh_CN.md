# 推理

现在我们已经训练了模型，下一步自然是要将其用于推理。

您需要两样东西来为模型加载权重：模型的记录和模型的配置。由于 Burn 中的参数是惰性初始化的，`ModelConfig::init` 函数不会执行分配和 GPU/CPU 内核。权重在首次使用时初始化，因此您可以安全地使用 `config.init(device).load_record(record)` 而不会产生任何有意义的性能成本。让我们在新文件 `src/inference.rs` 中创建一个简单的 `infer` 方法，用于加载我们训练好的模型。

```rust , ignore
# use crate::{data::MnistBatcher, training::TrainingConfig};
# use burn::{
#     data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
#     prelude::*,
#     record::{CompactRecorder, Recorder},
# };
# 
pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("模型的配置应该存在；请先运行训练");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("训练模型应该存在；请先运行训练");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("预测值 {predicted} 期望值 {label}");
}
```

第一步是加载训练配置以获取正确的模型配置。然后我们可以使用与训练期间相同的记录器获取记录。最后我们可以使用配置和记录初始化模型。为了简单起见，我们可以使用训练期间使用的相同批处理器将 MnistItem 转换为张量。

通过运行 infer 函数，您应该能看到模型的预测！

在 `main.rs` 文件中 `train` 函数调用后添加对 `infer` 的调用：

```rust , ignore
# #![recursion_limit = "256"]
# mod data;
# mod inference;
# mod model;
# mod training;
# 
# use crate::{model::ModelConfig, training::TrainingConfig};
# use burn::{
#     backend::{Autodiff, Wgpu},
#     data::dataset::Dataset,
#     optim::AdamConfig,
# };
# 
# fn main() {
#     type MyBackend = Wgpu<f32, i32>;
#     type MyAutodiffBackend = Autodiff<MyBackend>;
# 
#     let device = burn::backend::wgpu::WgpuDevice::default();
#     let artifact_dir = "/tmp/guide";
#     crate::training::train::<MyAutodiffBackend>(
#         artifact_dir,
#         TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
#         device.clone(),
#     );
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
# }
```

数字 `42` 是 MNIST 数据集中图像的索引。您可以使用这个 [MNIST 查看器](https://observablehq.com/@davidalber/mnist-viewer) 来探索和验证它们。