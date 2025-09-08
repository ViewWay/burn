# 示例

在 [下一章](./basic-workflow) 中，您将有机会以逐步的方式自己实现整个 Burn `guide` 示例。

在 [examples](https://github.com/tracel-ai/burn/tree/main/examples) 目录中提供了许多额外的 Burn 示例。Burn 示例被组织为库 crate，其中一个或多个示例是可执行的二进制文件。然后可以使用以下 cargo 命令行在 Burn 仓库的根目录中执行示例：

```bash
cargo run --example <示例名称>
```

要了解更多关于 crate 和示例的信息，请阅读下面的 Rust 部分。

<details>
<summary><strong>🦀 关于 Rust crate</strong></summary>

每个 Burn 示例都是一个 **package**，它们是 `examples` 目录的子目录。一个 package 由一个或多个 **crates** 组成。

package 是提供一组功能的一个或多个 crate 的捆绑包。package 包含一个 `Cargo.toml` 文件，该文件描述了如何构建这些 crate。

crate 是 Rust 中的编译单元。它可以是单个文件，但通常将 crate 分割成多个 **modules** 更容易。

module 让我们能够在 crate 内组织代码以提高可读性和易于重用。module 还允许我们控制项目的 _隐私性_。例如，`pub(crate)` 关键字用于使模块在 crate 内部公开可用。在下面的代码片段中声明了四个模块，其中两个是公开的，对 crate 的用户可见，其中一个仅在 crate 内部公开，crate 用户无法看到，最后一个是没有关键字的私有模块。这些模块可以是单个文件，也可以是包含 `mod.rs` 文件的目录。

```rust, ignore
pub mod data;
pub mod inference;
pub(crate) mod model;
mod training;
```

crate 可以有两种形式之一：**二进制 crate** 或 **库 crate**。编译 crate 时，编译器首先在 crate 根文件中查找（库 crate 为 `src/lib.rs`，二进制 crate 为 `src/main.rs`）。在 crate 根文件中声明的任何模块都将被插入到 crate 中进行编译。

所有 Burn 示例都是库 crate，它们可以包含一个或多个使用该库的可执行示例。我们甚至有一些 Burn 示例使用其他示例的库 crate。

示例是 `examples` 目录下的唯一文件。每个文件生成一个同名的可执行文件，然后可以使用 `cargo run --example <可执行文件名>` 执行每个示例。

以下是典型 Burn 示例 package 的文件树：

```
examples/burn-example
├── Cargo.toml
├── examples
│   ├── example1.rs      ---> 编译为 example1 二进制文件
│   ├── example2.rs      ---> 编译为 example2 二进制文件
│   └── ...
└── src
    ├── lib.rs           ---> 这是库的根文件
    ├── module1.rs
    ├── module2.rs
    └── ...
```

</details><br>

如果您想查看，以下是一些当前可用的额外示例：

| 示例                                                                                                   | 描述                                                                                                                                                                                  |
| :-------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [自定义 CSV 数据集](https://github.com/tracel-ai/burn/tree/main/examples/custom-csv-dataset)             | 实现一个数据集来解析用于回归任务的 CSV 数据。                                                                                                                                |
| [回归](https://github.com/tracel-ai/burn/tree/main/examples/simple-regression)                      | 在加州住房数据集上训练一个简单的 MLP 来预测一个地区的房屋中位数价格。                                                                                      |
| [自定义图像数据集](https://github.com/tracel-ai/burn/tree/main/examples/custom-image-dataset)         | 在遵循简单文件夹结构的自定义图像数据集上训练一个简单的 CNN。                                                                                                             |
| [自定义渲染器](https://github.com/tracel-ai/burn/tree/main/examples/custom-renderer)                   | 实现一个自定义渲染器来显示 [`Learner`](./building-blocks/learner.md) 进度。                                                                                              |
| [Web 图像分类](https://github.com/tracel-ai/burn/tree/main/examples/image-classification-web) | 使用 Burn、WGPU 和 WebAssembly 在浏览器中进行图像分类的演示。                                                                                                                      |
| [Web 上的 MNIST 推理](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web)        | 浏览器中的交互式 MNIST 推理演示。该演示在 [online](https://burn.dev/demo/) 上可用。                                                                                  |
| [MNIST 训练](https://github.com/tracel-ai/burn/tree/main/examples/mnist)                              | 演示如何使用配置的 [`Learner`](./building-blocks/learner.md) 训练自定义 [`Module`](./building-blocks/module.md) (MLP) 来记录指标并保持训练检查点。 |
| [命名张量](https://github.com/tracel-ai/burn/tree/main/examples/named-tensor)                         | 使用实验性的 `NamedTensor` 功能执行操作。                                                                                                                             |
| [ONNX 导入推理](https://github.com/tracel-ai/burn/tree/main/examples/onnx-inference)              | 导入在 MNIST 上预训练的 ONNX 模型，使用 Burn 对样本图像进行推理。                                                                                                 |
| [PyTorch 导入推理](https://github.com/tracel-ai/burn/tree/main/examples/import-model-weights)          | 导入在 MNIST 上预训练的 PyTorch 模型，使用 Burn 对样本图像进行推理。                                                                                               |
| [文本分类](https://github.com/tracel-ai/burn/tree/main/examples/text-classification)           | 在 AG News 或 DbPedia 数据集上训练文本分类 transformer 模型。训练后的模型可用于分类文本样本。                                             |
| [文本生成](https://github.com/tracel-ai/burn/tree/main/examples/text-generation)                   | 在 DbPedia 数据集上训练文本生成 transformer 模型。                                                                                                                           |
| [Wasserstein GAN MNIST](https://github.com/tracel-ai/burn/tree/main/examples/wgan)                        | 训练 WGAN 模型以基于 MNIST 生成新的手写数字。                                                                                                                       |

有关每个示例的更多信息，请参阅其各自的 `README.md` 文件。请务必查看 [examples](https://github.com/tracel-ai/burn/tree/main/examples) 目录以获取最新列表。

<div class="warning">

请注意，一些示例使用 [HuggingFace 的 `datasets` 库](https://huggingface.co/docs/datasets/index) 来下载示例中所需的数据库。这是一个 Python 库，这意味着您需要先安装 Python 才能运行这些示例。在适用时，此要求将在示例的 README 中明确说明。

</div>