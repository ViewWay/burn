# 数据

通常，人们会在某些数据集上训练模型。Burn 提供了一个非常有用的数据集来源和转换库，例如 Hugging Face 数据集工具，允许将数据下载并存储到 SQLite 数据库中以实现极高效的数据流和存储。但在本指南中，我们将使用来自 `burn::data::dataset::vision` 的 MNIST 数据集，它不需要外部依赖。

为了高效地迭代数据集，我们将定义一个实现 `Batcher` trait 的结构体。批处理器的目标是将单个数据集项目映射到可以作为输入提供给我们之前定义的模型的批处理张量。

让我们首先在文件 `src/data.rs` 中定义我们的数据集功能。为了简洁起见，我们将省略一些导入，但遵循本指南的完整代码可以在 `examples/guide/` [目录](https://github.com/tracel-ai/burn/tree/main/examples/guide) 中找到。

```rust , ignore
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};


#[derive(Clone, Default)]
pub struct MnistBatcher {}
```

这个批处理器非常简单，因为它只定义了一个将实现 `Batcher` trait 的结构体。该 trait 在 `Backend` trait 上是泛型的，其中包括设备的关联类型，因为并非所有后端都暴露相同的设备。例如，基于 Libtorch 的后端暴露了 `Cuda(gpu_index)`、`Cpu`、`Vulkan` 和 `Metal` 设备，而 ndarray 后端只暴露了 `Cpu` 设备。

接下来，我们需要实际实现批处理逻辑。

```rust , ignore
# use burn::{
#     data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
#     prelude::*,
# };
#
# #[derive(Clone, Default)]
# pub struct MnistBatcher {}
#
#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // 归一化：缩放到 [0,1] 并使均值=0 和标准差=1
            // 值 mean=0.1307,std=0.3081 来自 PyTorch MNIST 示例
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
```

<details>
<summary><strong>🦀 迭代器和闭包</strong></summary>

迭代器模式允许您依次对一系列项目执行某些任务。

在这个例子中，通过调用 `iter` 方法在向量 `items` 中的 `MnistItem` 上创建了一个迭代器。

_迭代器适配器_是在 `Iterator` trait 上定义的方法，通过改变原始迭代器的某些方面来产生不同的迭代器。在这里，`map` 方法被链式调用，以在使用 `collect` 消费最终迭代器以获得 `images` 和 `targets` 向量之前转换原始数据。然后将两个向量连接成一个张量用于当前批次。

您可能注意到每次调用 `map` 都不同，因为它定义了在每一步对迭代器项目执行的函数。这些匿名函数在 Rust 中被称为 [_闭包_](https://doc.rust-lang.org/book/ch13-01-closures.html)。由于它们的语法使用竖线 `||`，它们很容易识别。竖线捕获输入变量（如果适用），而表达式的其余部分定义要执行的函数。

如果我们回到这个例子，我们可以分解并注释用于处理图像的表达式。

```rust, ignore
let images = items                                                       // 获取 items Vec<MnistItem>
    .iter()                                                              // 在其上创建一个迭代器
    .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())  // 对于每个项目，将图像转换为浮点数据结构
    .map(|data| Tensor::<B, 2>::from_data(data, device))                 // 对于每个数据结构，在设备上创建一个张量
    .map(|tensor| tensor.reshape([1, 28, 28]))                           // 对于每个张量，重塑为图像维度 [C, H, W]
    .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)                    // 对于每个图像张量，应用归一化
    .collect();                                                          // 消费结果迭代器并将值收集到新向量中
```

有关迭代器和闭包的更多信息，请务必查看 Rust Book 中的[相关章节](https://doc.rust-lang.org/book/ch13-00-functional-features.html)。

</details><br>

在前面的例子中，我们实现了带有 `MnistItem` 列表作为输入和单个 `MnistBatch` 作为输出的 `Batcher` trait。批次以 3D 张量的形式包含图像，以及包含正确数字类别索引的目标张量。第一步是将图像数组解析为 `TensorData` 结构体。Burn 提供了 `TensorData` 结构体来封装张量存储信息，而不需要特定于后端。在从数据创建张量时，我们通常需要将数据精度转换为正在使用的当前后端。这可以通过 `.convert()` 方法完成（在这个例子中，数据被转换为后端的浮点元素类型 `B::FloatElem`）。在导入 `burn::tensor::ElementConversion` trait 时，您可以对特定数字调用 `.elem()` 将其转换为正在使用的当前后端元素类型。