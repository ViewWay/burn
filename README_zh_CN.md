<div align="center">
<img src="https://raw.githubusercontent.com/tracel-ai/burn/main/assets/logo-burn-neutral.webp" width="350px"/>

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)
[![Current Crates.io Version](https://img.shields.io/crates/v/burn.svg)](https://crates.io/crates/burn)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/burn)](https://crates.io/crates/burn)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://burn.dev/docs/burn)
[![Test Status](https://github.com/tracel-ai/burn/actions/workflows/test.yml/badge.svg)](https://github.com/tracel-ai/burn/actions/workflows/test.yml)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tracel-ai/burn)

[<img src="https://www.runblaze.dev/ci-blaze-powered.png" width="125px"/>](https://www.runblaze.dev)

---

**Burn 是下一代深度学习框架，毫不妥协地兼顾了<br />灵活性、效率与可移植性。**

<br/>
</div>

<div align="left">

## 性能表现

<div align="left">
<img align="right" src="https://raw.githubusercontent.com/tracel-ai/burn/main/assets/ember-blazingly-fast.png" height="96px"/>

我们深信深度学习框架的目标是将计算转化为有用的智能，因此性能是 Burn 的核心支柱之一。我们致力于通过多种优化技术实现卓越效率，具体如下所述。

**点击各部分了解详情** 👇

</div>

<br />

<details>
<summary>
自动内核融合 💥
</summary>
<br />

使用 Burn 意味着您的模型将在任何后端上自动优化。我们提供一种自动且动态创建自定义内核的方法，最大限度减少不同内存空间之间的数据迁移，这在内存移动成为瓶颈时极为有效。

例如，您可以使用高级张量 API 编写自己的 GELU 激活函数（如下 Rust 代码片段所示）。

```rust
fn gelu_custom<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let x = x.clone() * ((x / SQRT_2).erf() + 1);
    x / 2
}
```

在运行时，针对您的特定实现，将自动生成一个自定义低级内核，其性能堪比手写的 GPU 实现。该内核约60行 WGSL [WebGPU 着色语言](https://www.w3.org/TR/WGSL/)代码，而这种冗长的低级着色语言通常不适合直接编写深度学习模型！

</details>

<details>
<summary>
异步执行 ❤️‍🔥
</summary>
<br />

对于[第一方后端](#backends)，采用异步执行方式，这使得多种优化成为可能，例如前述的自动内核融合。

异步执行还能确保框架的正常运行不会阻塞模型计算，从而使框架开销对执行速度影响极小。反过来，模型中的密集计算也不会影响框架的响应能力。更多关于异步后端的信息，请参见[这篇博客文章](https://burn.dev/blog/creating-high-performance-asynchronous-backends-with-burn-compute)。

</details>

<details>
<summary>
线程安全构建模块 🦞
</summary>
<br />

Burn 利用 Rust 的[所有权系统](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)强调线程安全。每个模块拥有自己的权重，因此可以将模块发送到另一个线程计算梯度，再将梯度发送回主线程进行聚合，从而实现多设备训练。

这与 PyTorch 不同，PyTorch 的反向传播实际会修改每个张量参数的 _grad_ 属性，这不是线程安全操作，需要使用底层同步机制，详见[分布式训练](https://pytorch.org/docs/stable/distributed.html)。虽然 PyTorch 方式速度快，但不支持跨多后端，且实现较为复杂。

</details>

<details>
<summary>
智能内存管理 🦀
</summary>
<br />

深度学习框架的重要职责之一是减少模型运行所需内存。最初级的内存管理是每个张量拥有独立内存，创建时分配，超出作用域时释放。但频繁分配和释放内存开销巨大，因此通常需要内存池来提高吞吐量。Burn 提供基础设施，方便创建和选择后端的内存管理策略。详情参见[这篇博客](https://burn.dev/blog/creating-high-performance-asynchronous-backends-with-burn-compute)。

另一个关键优化是利用所有权系统跟踪张量何时可以就地变异，虽然单次优化不大，但在训练或推理大型模型时累计显著节省内存。详情见[这篇关于张量处理的博客](https://burn.dev/blog/burn-rusty-approach-to-tensor-handling)。

</details>

<details>
<summary>
自动内核选择 🎯
</summary>
<br />

优秀的深度学习框架应保证模型在所有硬件上流畅运行。不同硬件的执行速度表现差异甚大。例如，矩阵乘法内核有多种参数配置，受矩阵尺寸和硬件影响巨大。错误配置可能导致执行速度降低十倍甚至更多。因此，选择合适内核至关重要。

借助自研后端，我们自动运行基准测试，针对当前硬件和矩阵大小挑选最佳配置，并采用合理的缓存策略。

虽然这会略微增加预热执行时间，但在经过几轮前向和反向传播后能迅速稳定，长期节约大量时间。此功能可关闭，适用于冷启动优先的场景。

</details>

<details>
<summary>
硬件特定特性 🔥
</summary>
<br />

众所周知，深度学习核心操作主要是矩阵乘法，全连接神经网络即基于此。

越来越多硬件厂商针对矩阵乘法工作负载优化芯片。例如，Nvidia 的 _Tensor Cores_，以及多数手机内置 AI 专用芯片。目前，我们的 LibTorch、Candle、CUDA、Metal 和 WGPU/SPIR-V 后端支持 Tensor Cores，但其他加速器尚未支持。期待[此问题](https://github.com/gpuweb/gpuweb/issues/4195)解决后能将支持带给 WGPU 后端。

</details>

<details>
<summary>
自定义后端扩展 🎒
</summary>
<br />

Burn 致力于成为最灵活的深度学习框架。除了兼容多种后端，Burn 还允许您扩展后端功能，以满足个性化建模需求。

此灵活性带来多重优势，例如支持自定义操作（如闪电注意力），或为特定后端手写内核以提升性能。详情见 Burn 书籍[该章节](https://burn.dev/books/burn/advanced/backend-extension/index.html)🔥。

</details>

<br />

## 后端支持

<div align="left">
<img align="right" src="https://raw.githubusercontent.com/tracel-ai/burn/main/assets/backend-chip.png" height="96px"/>

Burn 力求在尽可能多的硬件上实现卓越性能和稳健实现。我们认为，这种灵活性对现代需求至关重要，因为您可能在云端训练模型，但部署环境却因用户而异。

</div>

<br />

**支持的后端**

| 后端     | 设备                        | 类型        |
| -------- | --------------------------- | ----------- |
| CUDA     | NVIDIA GPU                  | 第一方      |
| ROCm     | AMD GPU                    | 第一方      |
| Metal    | Apple GPU                  | 第一方      |
| Vulkan   | Linux & Windows 大多数 GPU | 第一方      |
| Wgpu     | 多数 GPU                   | 第一方      |
| NdArray  | 多数 CPU                   | 第三方      |
| LibTorch | 多数 GPU & CPU             | 第三方      |
| Candle   | Nvidia、Apple GPU 和 CPU   | 第三方      |

<br />

与其他框架相比，Burn 支持多后端的方式非常不同。大部分代码基于 Backend trait，实现后端可插拔。这使得后端组合成为可能，并能通过自动微分和自动内核融合等功能增强它们。

<details>
<summary>
自动微分：为任意后端添加反向传播支持的装饰器 🔄
</summary>
<br />

自动微分实际上是一个后端 _装饰器_，不能单独存在，必须包裹另一个后端。

简单地将基础后端封装为 Autodiff，便可透明地赋予其自动微分能力，从而支持模型调用 backward。

```rust
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Distribution, Tensor};

fn main() {
    type Backend = Autodiff<Wgpu>;

    let device = Default::default();

    let x: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device);
    let y: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device).require_grad();

    let tmp = x.clone() + y.clone();
    let tmp = tmp.matmul(x);
    let tmp = tmp.exp();

    let grads = tmp.backward();
    let y_grad = y.grad(&grads).unwrap();
    println!("{y_grad}");
}
```

值得注意的是，只有 Autodiff 后端才提供 backward 方法，避免调用不支持自动微分后端的错误。

详见 [Autodiff Backend README](./crates/burn-autodiff/README.md)。

</details>

<details>
<summary>
Fusion：为所有第一方后端带来内核融合的装饰器
</summary>
<br />

该后端装饰器为支持融合的内部后端增强内核融合能力。您可将此装饰器与 Autodiff 等其他后端装饰器组合。所有第一方加速后端（如 WGPU 和 CUDA）默认开启 Fusion（`burn/fusion` 特性标志），通常无需手动应用。

```rust
#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;
```

未来计划基于计算和内存瓶颈实现自动梯度检查点，与融合后端配合提升训练速度，详见[此问题](https://github.com/tracel-ai/burn/issues/936)。

详见 [Fusion Backend README](./crates/burn-fusion/README.md)。

</details>

<details>
<summary>
Router（Beta）：将多个后端合成为一个的装饰器
</summary>
<br />

此后端简化硬件操作，例如您希望部分操作在 CPU 上执行，部分操作在 GPU 上执行。

```rust
use burn::tensor::{Distribution, Tensor};
use burn::backend::{
    NdArray, Router, Wgpu, ndarray::NdArrayDevice, router::duo::MultiDevice, wgpu::WgpuDevice,
};

fn main() {
    type Backend = Router<(Wgpu, NdArray)>;

    let device_0 = MultiDevice::B1(WgpuDevice::DiscreteGpu(0));
    let device_1 = MultiDevice::B2(NdArrayDevice::Cpu);

    let tensor_gpu =
        Tensor::<Backend, 2>::random([3, 3], burn::tensor::Distribution::Default, &device_0);
    let tensor_cpu =
        Tensor::<Backend, 2>::random([3, 3], burn::tensor::Distribution::Default, &device_1);
}

```

</details>

<details>
<summary>
Remote（Beta）：远程后端执行装饰器，适用于分布式计算
</summary>
<br />

该后端包括客户端和服务器两部分。客户端通过网络发送张量操作到远程计算后端。您可用任一第一方后端作为服务器，一行代码即可启动：

```rust
fn main_server() {
    // 在3000端口启动服务器。
    burn::server::start::<burn::backend::Cuda>(Default::default(), 3000);
}

fn main_client() {
    // 创建与3000端口服务器通信的客户端。
    use burn::backend::{Autodiff, RemoteBackend};

    type Backend = Autodiff<RemoteDevice>;

    let device = RemoteDevice::new("ws://localhost:3000");
    let tensor_gpu =
        Tensor::<Backend, 2>::random([3, 3], Distribution::Default, &device);
}

```

</details>

<br />

## 训练与推理

<div align="left">
<img align="right" src="https://raw.githubusercontent.com/tracel-ai/burn/main/assets/ember-wall.png" height="96px"/>

Burn 致力于简化完整深度学习工作流程，您可通过友好的终端仪表盘实时监控训练进展，且能在从嵌入式设备到大型 GPU 集群的多样环境中运行推理。

Burn 自设计之初即兼顾训练和推理。相比于 PyTorch 等框架，Burn 简化了训练到部署的过渡过程，无需修改代码。

</div>

<div align="center">

<br />

<a href="https://www.youtube.com/watch?v=N9RM5CQbNQc" target="_blank">
    <img src="https://raw.githubusercontent.com/tracel-ai/burn/main/assets/burn-train-tui.png" alt="Burn Train TUI" width="75%">
  </a>
</div>

<br />

**点击下方分类以展开查看更多👇**

<details>
<summary>
训练仪表盘 📈
</summary>
<br />

如前视频所示（点击图片即可观看），基于[Ratatui](https://github.com/ratatui-org/ratatui)库的新终端用户界面仪表盘使用户无需连接外部软件即可轻松跟踪训练。

您可以实时查看训练和验证指标，使用方向键分析指标的长期趋势及近期历史。训练循环可安全中断，确保检查点完整写入或关键代码执行不被打断 🛡。

</details>

<details>
<summary>
ONNX 支持 🐫
</summary>
<br />

ONNX（Open Neural Network Exchange）是一种开放标准格式，支持导出深度学习模型的结构和权重。

Burn 支持导入符合 ONNX 标准的模型，使您能轻松将 TensorFlow、PyTorch 等框架开发的模型迁移到 Burn，享受框架优势。

关于 ONNX 支持的更多内容，请见 Burn 书籍[相关章节🔥](https://burn.dev/books/burn/import/onnx-model.html)。

> **注意**：该组件仍在积极开发中，目前仅支持[有限的 ONNX 操作符集](./crates/burn-import/SUPPORTED-ONNX-OPS.md)。

</details>

<details>
<summary>
导入 PyTorch 或 Safetensors 模型 🚚
</summary>
<br />

您可以直接从 PyTorch 或 Safetensors 格式加载权重至 Burn 定义的模型，轻松复用已有模型，同时享受 Burn 的性能和部署优势。

了解更多：

- [将预训练 PyTorch 模型导入 Burn](https://burn.dev/books/burn/import/pytorch-model.html)
- [加载 Safetensors 格式模型](https://burn.dev/books/burn/import/safetensors-model.html)

</details>

<details>
<summary>
浏览器内推理 🌐
</summary>
<br />

若干后端支持运行于 WebAssembly 环境：Candle 和 NdArray 用于 CPU 运行，WGPU 则通过 WebGPU 实现 GPU 加速。这意味着您可以直接在浏览器内运行推理。提供若干示例：

- [MNIST](./examples/mnist-inference-web)：允许绘制数字，小型卷积网络尝试识别！2️⃣ 7️⃣ 😰
- [图像分类](./examples/image-classification-web)：上传图像并进行分类！🌄

</details>

<details>
<summary>
嵌入式设备：<i>no_std</i> 支持 ⚙️
</summary>
<br />

Burn 核心组件支持[no_std](https://docs.rust-embedded.org/book/intro/no-std.html)，可运行在无操作系统的裸机环境（如嵌入式设备）中。

> 目前仅 NdArray 后端支持 _no_std_ 环境。

</details>

<br />

### 性能基准

为了评估各后端性能及追踪改进，我们提供专门的基准测试套件。

使用 [burn-bench](https://github.com/tracel-ai/burn-bench) 运行并比较基准。

> ⚠️ **注意**：使用 `wgpu` 后端时，可能遇到递归类型求值的编译错误，因 `wgpu` 依赖链的类型嵌套复杂。解决方案是在 `main.rs` 或 `lib.rs` 顶部添加：

```rust
#![recursion_limit = "256"]
```

默认递归限制（128）通常略低于所需（约130-150），因深层嵌套的关联类型和 trait 约束。

## 快速入门

<div align="left">
<img align="right" src="https://raw.githubusercontent.com/tracel-ai/burn/main/assets/ember-walking.png" height="96px"/>

刚刚接触 Burn？您来对地方了！继续阅读，下文将助您快速上手。

</div>

<details>
<summary>
Burn 书籍 🔥
</summary>
<br />

想要高效使用 Burn，了解其核心组件和设计理念至关重要。我们强烈建议新用户阅读[Burn 书籍 🔥](https://burn.dev/books/burn/)的前几章，书中详细展示了框架各方面，包括张量、模块、优化器等基础构件，以及进阶内容，如自定义 GPU 内核编写。

> 项目持续发展，我们努力保持书籍内容更新，但难免有遗漏。如遇异常，请告知我们！同时欢迎提交代码贡献 😄

</details>

<details>
<summary>
示例代码 🙏
</summary>
<br />

下面的代码片段展示了框架的易用性！定义了一个带参数的神经网络模块及其前向传播方法。

```rust
use burn::nn;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: nn::Linear<B>,
    linear_outer: nn::Linear<B>,
    dropout: nn::Dropout,
    gelu: nn::Gelu,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
```

仓库中拥有大量[示例](./examples)，涵盖多种使用场景。

按照[书籍](https://burn.dev/books/burn/)：

- [基础工作流](./examples/guide)：创建自定义卷积神经网络 `Module`，用于 MNIST 训练和推理。
- [自定义训练循环](./examples/custom-training-loop)：实现基础训练循环，不使用 `Learner`。
- [自定义 WGPU 内核](./examples/custom-wgpu-kernel)：学习如何为 WGPU 后端编写自定义操作。

补充示例：

- [自定义 CSV 数据集](./examples/custom-csv-dataset)：实现解析 CSV 数据的回归任务数据集。
- [回归示例](./examples/simple-regression)：训练简单多层感知机预测加州房价中位数。
- [自定义图像数据集](./examples/custom-image-dataset)：根据简易文件夹结构训练卷积神经网络。
- [自定义渲染器](./examples/custom-renderer)：实现自定义渲染器展示 [`Learner`](./building-blocks/learner.md) 训练进度。
- [浏览器图像分类](./examples/image-classification-web)：Burn、WGPU 与 WebAssembly 实现的浏览器图像分类演示。
- [浏览器 MNIST 推理](./examples/mnist-inference-web)：交互式 MNIST 推理演示，线上体验见[此处](https://burn.dev/demo/)。
- [MNIST 训练](./examples/mnist)：使用 `Learner` 配置日志和检查点训练自定义 MLP。
- [具名张量](./examples/named-tensor)：实验性 `NamedTensor` 功能示例。
- [ONNX 导入推理](./examples/onnx-inference)：导入预训练 MNIST ONNX 模型并推理。
- [PyTorch 导入推理](./examples/import-model-weights)：导入预训练 MNIST PyTorch 模型并推理。
- [文本分类](./examples/text-classification)：训练文本分类 Transformer，使用 AG News 或 DbPedia 数据集。
- [文本生成](./examples/text-generation)：训练生成文本的 Transformer 模型。
- [Wasserstein GAN MNIST](./examples/wgan)：训练 WGAN 模型生成手写数字。

建议克隆仓库，亲自运行示例，获得实践体验！

</details>

<details>
<summary>
预训练模型 🤖
</summary>
<br />

我们维护了使用 Burn 构建的模型与示例的最新汇总，详见[tracel-ai/models 仓库](https://github.com/tracel-ai/models)。

找不到您需要的模型？欢迎开 issue，我们或将优先开发。构建了 Burn 模型想分享？欢迎提交 Pull Request，将您的模型纳入社区列表！

</details>

<details>
<summary>
为何选择 Rust 进行深度学习？🦀
</summary>
<br />

深度学习要求极高层次的抽象与极致的执行速度。Rust 是理想选择，因其提供零成本抽象，便于构建神经网络模块，且细粒度内存控制助力性能优化。

框架需在高层次易用性和底层性能间平衡，令用户专注创新。当前主流方案多为 Python API，底层依赖 C/C++，这降低了可移植性，提高复杂度，阻碍研究与工程协作。Rust 的抽象方式灵活，能消弭两者隔阂。

Rust 配备 Cargo 包管理器，极大简化构建、测试和部署流程，远胜 Python。

尽管 Rust 学习曲线陡峭，但我们坚信它最终可构建更稳定、少 Bug 的高效解决方案（适当练习后😅）！

</details>

<br />

> **废弃提示**<br />自 `0.14.0` 起，张量数据内部结构发生变更。旧版的 `Data` 结构已弃用，且自 `0.17.0` 起被新结构 `TensorData` 取代，后者通过字节存储底层数据并保留数据类型字段，提供更大灵活性。若代码中仍使用 `Data`，请务必切换为 `TensorData`。

<details id="deprecation">
<summary>
加载旧版本模型记录 ⚠️
</summary>
<br />

若尝试加载版本低于 `0.14.0` 保存的模型记录，请使用兼容版本（`0.14`、`0.15` 或 `0.16`）并启用 `record-backward-compat` 特性：

```
features = [..., "record-backward-compat"]
```

否则记录无法正确反序列化，会报错并提示开启兼容特性。

兼容性仅保证反序列化（加载）过程生效。重新保存后，模型将采用新结构，后续可升级至当前版本。

请注意二进制格式不兼容旧版，需先用旧版加载，再以自描述格式（如 `NamedMpkFileRecorder`）保存，方能正常加载新版。

</details>

## 社区

<div align="left">
<img align="right" src="https://raw.githubusercontent.com/tracel-ai/burn/main/assets/ember-community.png" height="96px"/>

若您对项目感兴趣，欢迎加入我们的[Discord](https://discord.gg/uPEBbYYDB6)！我们致力于为所有背景的成员营造友好环境，您可在此提问并分享成果。

</div>

<br/>

**贡献指南**

贡献前，请先阅读我们的[行为准则](https://github.com/tracel-ai/burn/tree/main/CODE-OF-CONDUCT.md)。强烈建议了解[架构概览](https://github.com/tracel-ai/burn/tree/main/contributor-book/src/project-architecture)，了解部分设计决策。更多细节见[贡献指南](/CONTRIBUTING.md)。

## 项目状态

Burn 正处于积极开发阶段，可能存在破坏性改动。虽大多问题易于修复，目前仍无绝对保证。

## 许可协议

Burn 采用 MIT 和 Apache License 2.0 双许可协议发布。详见 [LICENSE-APACHE](./LICENSE-APACHE) 与 [LICENSE-MIT](./LICENSE-MIT) 文件。提交 Pull Request 即视为同意授权条款。

</div>