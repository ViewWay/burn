# 模型

第一步是创建一个项目并添加不同的 Burn 依赖项。首先使用 Cargo 创建一个新项目：

```console
cargo new guide
```

正如[之前提到的](../getting-started.md#creating-a-burn-application)，这将使用 `Cargo.toml` 和 `src/main.rs` 文件初始化您的 `guide` 项目目录。

在 `Cargo.toml` 文件中，添加带有 `train`、`vision` 和 `wgpu` 特性的 `burn` 依赖项。由于我们禁用了默认特性，我们还希望启用 `std`、`tui`（用于仪表板）和 `fusion` 用于 wgpu。然后运行 `cargo build` 构建项目并导入所有依赖项。

```toml
[package]
name = "guide"
version = "0.1.0"
edition = "2024"

[dependencies]
# 禁用卷积的自动调优默认设置
burn = { version = "~0.19", features = ["std", "tui", "train", "vision", "wgpu", "fusion"], default-features = false }
# burn = { version = "~0.19", features = ["train", "vision", "wgpu"] }
```

我们的目标是创建一个用于图像分类的基本卷积神经网络。我们将通过使用两个卷积层后跟两个线性层、一些池化和 ReLU 激活函数来保持模型的简单性。我们还将使用 dropout 来提高训练性能。

让我们首先在新文件 `src/model.rs` 中定义我们的模型结构体。

```rust , ignore
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}
```

这段代码示例中有两个主要的事情。

1. 您可以通过在结构体上使用 `#[derive(Module)]` 属性来创建深度学习模块。这将生成必要的代码，使结构体实现 `Module` trait。这个 trait 将使您的模块既可训练又可（反）序列化，同时添加相关功能。就像 Rust 中经常使用的其他属性一样，例如 `Clone`、`PartialEq` 或 `Debug`，结构体中的每个字段也必须实现 `Module` trait。

   <details>
   <summary><strong>🦀 Trait</strong></summary>

   Trait 是 Rust 语言的一个强大而灵活的特性。它们提供了一种为特定类型定义共享行为的方法，可以与其他类型共享。

   类型的行为由在该类型上调用的方法组成。由于所有 `Module` 都应该实现相同的功能，因此它被定义为一个 trait。在特定类型上实现 trait 通常需要用户为他们的类型实现 trait 定义的行为，但如上所述使用 `derive` 属性时并非如此。请查看下面的[解释](#derive-attribute)来了解原因。

   有关 trait 的更多详细信息，请查看 Rust Book 中的[相关章节](https://doc.rust-lang.org/book/ch10-02-traits.html)。
   </details><br>

   <details id="derive-attribute">
   <summary><strong>🦀 Derive 宏</strong></summary>

   `derive` 属性允许通过生成代码轻松实现 trait，这些代码将在使用 `derive` 语法注释的类型上实现具有其自己默认实现的 trait。

   这是通过 Rust 的一个称为[过程宏](https://doc.rust-lang.org/reference/procedural-macros.html)的功能实现的，它允许我们在编译时运行操作 Rust 语法的代码，既消费又生成 Rust 语法。使用 `#[my_macro]` 属性，您可以有效地扩展提供的代码。您将看到 derive 宏经常被用于递归实现 trait，其中实现由所有字段的组合组成。

   在这个例子中，我们想要派生 [`Module`](../building-blocks/module.md) 和 `Debug` trait。

   ```rust, ignore
   #[derive(Module, Debug)]
   pub struct MyCustomModule<B: Backend> {
       linear1: Linear<B>,
       linear2: Linear<B>,
       activation: Relu,
   }
   ```

   基本的 `Debug` 实现由编译器提供，用于使用 `{:?}` 格式化器格式化值。为了便于使用，`Module` trait 实现由 Burn 自动处理，因此您无需做任何特殊事情。它本质上充当参数容器。

   有关可派生 trait 的更多详细信息，请查看 Rust [附录](https://doc.rust-lang.org/book/appendix-03-derivable-traits.html)、[参考](https://doc.rust-lang.org/reference/attributes/derive.html)或[示例](https://doc.rust-lang.org/rust-by-example/trait/derive.html)。
   </details><br>

2. 注意结构体在 [`Backend`](../building-blocks/backend.md) trait 上是泛型的。后端 trait 抽象了张量操作的底层低级实现，允许您的新模型在任何后端上运行。与其他框架不同，后端抽象不是由编译标志或设备类型确定的。这很重要，因为您可以扩展特定后端的功能（参见[后端扩展部分](../advanced/backend-extension)），并且它允许创新的[自动微分系统](../building-blocks/autodiff.md)。您还可以在运行时更改后端，例如在使用 GPU 后端训练模型时使用 CPU 后端计算训练指标。在我们的例子中，使用的后端将在稍后确定。

   <details>
   <summary><strong>🦀 Trait 边界</strong></summary>

   Trait 边界提供了一种方法，使泛型项能够限制用作其参数的类型。trait 边界规定了类型实现的功能。因此，边界限制泛型为符合边界的类型。它还允许泛型实例访问边界中指定的 trait 的方法。

   对于一个简单但具体的例子，请查看 [Rust By Example on bounds](https://doc.rust-lang.org/rust-by-example/generics/bounds.html)。

   在 Burn 中，`Backend` trait 使您能够使用不同的实现运行张量操作，因为它抽象了张量、设备和元素类型。[入门示例](../getting-started.md#writing-a-code-snippet)说明了拥有适用于不同后端实现的简单 API 的优势。虽然它使用了 WGPU 后端，但您可以轻松地将其替换为任何其他受支持的后端。

   ```rust, ignore
   // 从任何受支持的后端中选择。
   // type Backend = Candle<f32, i64>;
   // type Backend = LibTorch<f32>;
   // type Backend = NdArray<f32>;
   type Backend = Wgpu;

   // 创建两个张量。
   let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
   let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

   // 打印两个张量的元素级加法（使用选定的后端完成）。
   println!("{}", tensor_1 + tensor_2);
   ```

   有关 trait 边界的更多详细信息，请查看 Rust [trait 边界部分](https://doc.rust-lang.org/book/ch10-02-traits.html#trait-bound-syntax)或[参考](https://doc.rust-lang.org/reference/items/traits.html#trait-bounds)。

   </details><br>

请注意，每次在 `src` 目录中创建新文件时，您还需要将此模块显式添加到 `main.rs` 文件中。例如，在创建 `model.rs` 后，您需要在主文件顶部添加以下内容：

```rust , ignore
mod model;
#
# fn main() {
# }
```

接下来，我们需要实例化模型进行训练。

```rust , ignore
# use burn::{
#     nn::{
#         conv::{Conv2d, Conv2dConfig},
#         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
#         Dropout, DropoutConfig, Linear, LinearConfig, Relu,
#     },
#     prelude::*,
# };
#
# #[derive(Module, Debug)]
# pub struct Model<B: Backend> {
#     conv1: Conv2d<B>,
#     conv2: Conv2d<B>,
#     pool: AdaptiveAvgPool2d,
#     dropout: Dropout,
#     linear1: Linear<B>,
#     linear2: Linear<B>,
#     activation: Relu,
# }
#
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// 返回初始化的模型。
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
```

一眼望去，您可以通过打印模型实例来查看模型配置：

```rust , ignore
#![recursion_limit = "256"]
mod model;

use crate::model::ModelConfig;
use burn::backend::Wgpu;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    println!("{model}");
}
```

输出：

```rust , ignore
Model {
  conv1: Conv2d {ch_in: 1, ch_out: 8, stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 80}
  conv2: Conv2d {ch_in: 8, ch_out: 16, stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 1168}
  pool: AdaptiveAvgPool2d {output_size: [8, 8]}
  dropout: Dropout {prob: 0.5}
  linear1: Linear {d_input: 1024, d_output: 512, bias: true, params: 524800}
  linear2: Linear {d_input: 512, d_output: 10, bias: true, params: 5130}
  activation: Relu
  params: 531178
}
```

<details>
<summary><strong>🦀 引用</strong></summary>

在前面的示例中，`init()` 方法签名使用 `&` 来表示参数类型是引用：`&self`，对当前接收者（`ModelConfig`）的引用，和 `device: &B::Device`，对后端设备的引用。

```rust, ignore
pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
    Model {
        // ...
    }
}
```

Rust 中的引用允许我们指向资源以访问其数据而无需拥有它。所有权的概念是 Rust 的核心，值得[深入了解](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)。

在像 C 这样的语言中，内存管理是显式的，由程序员负责，这意味着很容易犯错误。在像 Java 或 Python 这样的语言中，内存管理在垃圾收集器的帮助下是自动的。这非常安全和直接，但也会产生运行时成本。

在 Rust 中，内存管理相当独特。除了实现 [`Copy`](https://doc.rust-lang.org/std/marker/trait.Copy.html) 的简单类型（例如，[原语](https://doc.rust-lang.org/rust-by-example/primitives.html)如整数、浮点数、布尔值和 `char`），每个值都由称为 _owner_ 的某个变量 _owned_。所有权可以从一个变量转移到另一个变量，有时值可以被 _borrowed_。一旦 _owner_ 变量超出作用域，值就会被 _dropped_，这意味着它分配的任何内存都可以安全地释放。

由于方法不拥有 `self` 和 `device` 变量，引用指向的值在引用停止使用时（即方法的作用域）不会被丢弃。

有关引用和借用的更多信息，请务必阅读 Rust Book 中的[相关章节](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html)。

</details><br>

在创建自定义神经网络模块时，通常最好在模型结构体旁边创建一个配置。这允许您为网络定义默认值，这要归功于 `Config` 属性。此属性的好处是它使配置可序列化，使您能够轻松保存模型超参数，增强您的实验过程。请注意，将为您的配置自动生成一个构造函数，它将把没有默认值的参数作为输入：`let config = ModelConfig::new(num_classes, hidden_size);`。可以使用类似构建器的方法轻松覆盖默认值：（例如 `config.with_dropout(0.2);`）

第一个实现块与初始化方法相关。正如我们所见，所有字段都使用相应神经网络底层模块的配置进行设置。在这个特定情况下，我们选择在第一层将张量通道从 1 扩展到 8，然后在第二层从 8 扩展到 16，在所有维度上使用 3 的内核大小。我们还使用自适应平均池化模块将图像的维度降低到 8×8 矩阵，我们将在前向传递中将其展平为 1024（16 * 8 * 8）的结果张量。

现在让我们看看前向传递是如何定义的。

```rust , ignore
# use burn::{
#     nn::{
#         conv::{Conv2d, Conv2dConfig},
#         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
#         Dropout, DropoutConfig, Linear, LinearConfig, Relu,
#     },
#     prelude::*,
# };
#
# #[derive(Module, Debug)]
# pub struct Model<B: Backend> {
#     conv1: Conv2d<B>,
#     conv2: Conv2d<B>,
#     pool: AdaptiveAvgPool2d,
#     dropout: Dropout,
#     linear1: Linear<B>,
#     linear2: Linear<B>,
#     activation: Relu,
# }
#
# #[derive(Config, Debug)]
# pub struct ModelConfig {
#     num_classes: usize,
#     hidden_size: usize,
#     #[config(default = "0.5")]
#     dropout: f64,
# }
#
# impl ModelConfig {
#     /// 返回初始化的模型。
#     pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
#         Model {
#             conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
#             conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
#             pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
#             activation: Relu::new(),
#             linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
#             linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
#             dropout: DropoutConfig::new(self.dropout).init(),
#         }
#     }
# }
#
impl<B: Backend> Model<B> {
    /// # 形状
    ///   - 图像 [batch_size, height, width]
    ///   - 输出 [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // 在第二维创建一个通道。
        let x = images.reshape([batch_size, 1, height, width]);


        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
```

对于前 PyTorch 用户来说，这可能感觉非常直观，因为每个模块都直接使用急切 API 合并到代码中。请注意，对于前向方法没有强制抽象。您可以自由定义多个具有您喜欢的名称的前向函数。大多数使用 Burn 构建的神经网络模块都使用 `forward` 命名法，仅仅因为这是该领域的标准。

与神经网络模块类似，作为参数给出的 [`Tensor`](../building-blocks/tensor.md) 结构体也将后端 trait 作为泛型参数，以及其维度。即使在此特定示例中未使用，也可以将张量的类型作为第三个泛型参数添加。例如，不同类型（浮点、整数、布尔）的 3 维张量将定义如下：

```rust , ignore
Tensor<B, 3> // 浮点张量（默认）
Tensor<B, 3, Float> // 浮点张量（显式）
Tensor<B, 3, Int> // 整数张量
Tensor<B, 3, Bool> // 布尔张量
```

请注意，具体的元素类型，如 `f16`、`f32` 等，将在后端中定义。