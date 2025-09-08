# 入门指南

Burn 是一个使用 Rust 编程语言编写的深度学习框架。因此，不言而喻，您必须了解 Rust 的基本概念。建议阅读 [Rust Book](https://doc.rust-lang.org/book/) 的前几章，但如果您刚刚开始学习，不用担心。在需要时，我们会尽量提供足够的上下文和外部资源参考。只需留意 **🦀 Rust Note** 指示器。

## 安装 Rust

有关安装说明，请参阅 [安装页面](https://doc.rust-lang.org/book/ch01-01-installation.html)。它详细解释了在您的计算机上安装 Rust 的最便捷方法，这是开始使用 Burn 要做的第一件事。

## 创建 Burn 应用程序

正确安装 Rust 后，使用 Rust 的构建系统和包管理器 Cargo 创建一个新的 Rust 应用程序。它会随 Rust 自动安装。

<details>
<summary><strong>🦀 Cargo 速查表</strong></summary>

[Cargo](https://doc.rust-lang.org/cargo/) 是一个非常有用的工具来管理 Rust 项目，因为它处理许多任务。更准确地说，它用于编译您的代码，下载您的代码所依赖的库/包，并构建这些库。

以下是您在本指南中可能使用的主要 `cargo` 命令的速查表。

| 命令                | 描述                                                                                      |
| ------------------- | ----------------------------------------------------------------------------------------- |
| `cargo new` _path_  | 在给定目录中创建一个新的 Cargo 包。                                                        |
| `cargo add` _crate_ | 向 Cargo.toml 配置文件添加依赖项。                                                         |
| `cargo build`       | 编译本地包及其所有依赖项（在调试模式下，使用 `-r` 表示发布模式）。                          |
| `cargo check`       | 检查本地包的编译错误（速度更快）。                                                         |
| `cargo run`         | 运行本地包的二进制文件。                                                                   |

更多信息，请查看 Rust Book 中的 [Hello, Cargo!](https://doc.rust-lang.org/book/ch01-03-hello-cargo.html)。

</details><br>

在您选择的目录中，运行以下命令：

```console
cargo new my_burn_app
```

这将在 `my_burn_app` 项目目录中初始化一个 `Cargo.toml` 文件和一个包含自动生成的 `main.rs` 文件的 `src` 目录。进入该目录查看：

```console
cd my_burn_app
```

然后，添加 Burn 作为依赖项：

```console
cargo add burn --features wgpu
```

最后，通过执行以下命令编译本地包：

```console
cargo build
```

就是这样，您已经准备好了！您已经配置了一个使用 Burn 和 WGPU 后端的项目，该后端允许在任何平台上使用 GPU 执行低级操作。

<div class="warning">

当使用 `wgpu` 后端之一时，您可能会遇到与递归类型评估相关的编译错误。这是由于 `wgpu` 依赖链中的复杂类型嵌套造成的。

要解决此问题，请在您的 `main.rs` 或 `lib.rs` 文件顶部添加以下行：

```rust
#![recursion_limit = "256"]
```

默认的递归限制（128）通常略低于所需深度（通常为 130-150），这是由于深度嵌套的关联类型和特征边界造成的。

</div>

## 编写代码片段

`src/main.rs` 是由 Cargo 自动生成的，让我们将其内容替换为以下内容：

```rust, ignore
use burn::tensor::Tensor;
use burn::backend::Wgpu;

// 后端类型的别名。
type Backend = Wgpu;

fn main() {
    let device = Default::default();
    // 创建两个张量，第一个具有显式值，第二个为与第一个相同形状的全1张量
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // 打印两个张量的元素级加法（使用 WGPU 后端完成）结果。
    println!("{}", tensor_1 + tensor_2);
}
```

<details>
<summary><strong>🦀 Use 声明</strong></summary>

要将 Burn 模块或项目引入作用域，需要添加 `use` 声明。

在上面的示例中，我们想将 `Tensor` 结构体和 `Wgpu` 后端引入作用域：

```rust, ignore
use burn::tensor::Tensor;
use burn::backend::Wgpu;
```

在这个例子中这很容易理解。但是，同样的声明可以写成一种快捷方式，同时绑定具有共同前缀的多个路径：

```rust, ignore
use burn::{tensor::Tensor, backend::Wgpu};
```

在这个例子中，共同前缀很短，而且只有两个项目要绑定到本地。因此，使用两个 `use` 声明的第一种用法可能更受欢迎。但要知道这两个例子都是有效的。有关 `use` 关键字的更多详细信息，请查看 Rust Book 中的 [这一部分](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html) 或 [Rust 参考手册](https://doc.rust-lang.org/reference/items/use-declarations.html)。

</details><br>

<details>
<summary><strong>🦀 泛型数据类型</strong></summary>

如果您是 Rust 新手，您可能想知道为什么我们必须使用 `Tensor::<Backend, 2>::...`。这是因为 `Tensor` 结构体在多个具体数据类型上是 [泛型](https://doc.rust-lang.org/book/ch10-01-syntax.html) 的。更具体地说，`Tensor` 可以使用三个泛型参数定义：后端、维度数（秩）和数据类型（默认为 `Float`）。在这里，我们只指定后端和维度数，因为默认使用 `Float` 张量。有关 `Tensor` 结构体的更多详细信息，请查看 [这一部分](./building-blocks/tensor.md)。

大多数情况下涉及泛型时，编译器可以自动推断泛型参数。在这种情况下，编译器需要一点帮助。这通常可以通过两种方式之一完成：提供类型注解或通过 _turbofish_ `::<>` 语法绑定泛型参数。在上面的示例中，我们使用了所谓的 _turbofish_ 语法，但我们也可以使用类型注解代替，如下所示：

```rust, ignore
let tensor_1: Tensor<Backend, 2> = Tensor::from_data([[2., 3.], [4., 5.]]);
let tensor_2 = Tensor::ones_like(&tensor_1);
```

您可能注意到我们只为第一个张量提供了类型注解，但这个示例仍然有效。那是因为编译器（正确地）推断出 `tensor_2` 具有相同的泛型参数。在原始示例中也可以这样做，但为两者指定参数更加明确。

</details><br>

通过运行 `cargo run`，您现在应该看到加法的结果：

```console
Tensor {
  data:
[[3.0, 4.0],
 [5.0, 6.0]],
  shape:  [2, 2],
  device:  DefaultDevice,
  backend:  "wgpu",
  kind:  "Float",
  dtype:  "f32",
}
```

虽然前面的示例有些简单，但接下来的基本工作流程部分将引导您完成一个对深度学习应用更相关的示例。

## 使用 `prelude`

Burn 的核心库中有各种各样的内容。当创建新模型或使用现有模型进行推理时，您可能需要导入使用的每个组件，这可能会有些冗长。

为了解决这个问题，提供了 `prelude` 模块，允许您轻松地将常用的结构体和宏作为一组导入：

```rust, ignore
use burn::prelude::*;
```

这等同于：

```rust, ignore
use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{
        backend::Backend, Bool, Device, ElementConversion, Float, Int, Shape, Tensor,
        TensorData,
    },
};
```

<div class="warning">

为简单起见，本书后续章节都将使用这种导入形式，除了 [构建块](./building-blocks) 章节，因为显式导入有助于用户掌握特定结构和宏的用法。

</div>