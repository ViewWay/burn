# 后端

我们已经有效地编写了训练模型所需的大部分必要代码。但是，我们没有在任何地方明确指定要使用的后端。这将在我们程序的主入口点中定义，即 `src/main.rs` 中定义的 `main` 函数。

```rust , ignore
# #![recursion_limit = "256"]
# mod data;
# mod model;
# mod training;
#
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu},
#     data::dataset::Dataset,
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
```

在此代码片段中，我们使用 `Wgpu` 后端，它与任何操作系统兼容并将使用 GPU。有关其他选项，请参阅 Burn README。此后端类型将图形 API、浮点类型和整数类型作为将在训练期间使用的泛型参数。自动微分后端只是相同的后端，包装在 `Autodiff` 结构体中，该结构体为任何后端赋予可微分性。

我们使用工件目录、模型配置（数字类别的数量是 10，隐藏维度是 512）、优化器配置（在我们的例子中将是默认的 Adam 配置）以及可以从后端获得的设备调用前面定义的 `train` 函数。

您现在可以使用以下命令训练您新创建的模型：

```console
cargo run --release
```

当使用上述命令运行项目时，您应该通过基本 CLI 仪表板看到训练进度：

<img title="标题" alt="替代文本" src="./training-output.png">