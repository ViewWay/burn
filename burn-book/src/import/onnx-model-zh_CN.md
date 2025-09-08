# 在Burn中导入ONNX模型

## 简介

随着深度学习的发展，框架间的互操作性变得至关重要。Burn是一个现代的Rust深度学习框架，提供了从其他流行框架导入模型的强大支持。本节重点介绍将[ONNX（开放神经网络交换）](https://onnx.ai/onnx/intro/index.html)模型导入Burn，使您能够在基于Rust的深度学习项目中利用预训练模型。

## 为什么要导入模型？

导入预训练模型有以下几个优势：

1. **节省时间**：跳过从头开始训练模型的资源密集型过程。
2. **访问最先进的架构**：利用研究人员和行业领导者开发的前沿模型。
3. **迁移学习**：为您的特定任务微调导入的模型，受益于知识迁移。
4. **框架间一致性**：在框架间移动时保持一致的性能。

## 了解ONNX

ONNX（开放神经网络交换）是一种开放格式，旨在表示机器学习模型，具有以下关键特性：

- **框架无关**：提供适用于各种深度学习框架的通用格式。
- **全面表示**：捕捉模型架构和训练权重。
- **广泛支持**：与PyTorch、TensorFlow和scikit-learn等流行框架兼容。

这种标准化允许在不同框架和部署环境之间无缝移动模型。

## Burn的ONNX支持

Burn的ONNX导入方法提供了独特的优势：

1. **原生Rust代码生成**：将ONNX模型转换为Rust源代码，以便与Burn生态系统深度集成。
2. **编译时优化**：利用Rust编译器优化生成的代码，可能提高性能。
3. **无运行时依赖**：消除了对ONNX运行时的需求，这与许多其他解决方案不同。
4. **可训练性**：允许使用Burn进一步训练或微调导入的模型。
5. **可移植性**：能够为各种目标编译，包括WebAssembly和嵌入式设备。
6. **后端灵活性**：适用于Burn支持的任何后端。

## ONNX兼容性

Burn要求ONNX模型使用**opset版本16或更高版本**。如果您的模型使用较旧版本，您需要使用ONNX版本转换器升级它。

### 升级ONNX模型

有两种简单的方法将ONNX模型升级到所需的opset版本：

选项1：使用提供的实用脚本：

```
uv run --script https://raw.githubusercontent.com/tracel-ai/burn/refs/heads/main/crates/burn-import/onnx_opset_upgrade.py
```

选项2：使用自定义Python脚本：

```python
import onnx
from onnx import version_converter, shape_inference

# 加载您的ONNX模型
model = onnx.load('path/to/your/model.onnx')

# 将模型转换为opset版本16
upgraded_model = version_converter.convert_version(model, 16)

# 对升级后的模型应用形状推断
inferred_model = shape_inference.infer_shapes(upgraded_model)

# 保存转换后的模型
onnx.save(inferred_model, 'upgraded_model.onnx')
```

## 分步指南

按照以下步骤将ONNX模型导入您的Burn项目：

### 步骤1：更新`build.rs`

首先，将`burn-import` crate添加到您的`Cargo.toml`：

```toml
[build-dependencies]
burn-import = "~0.19"
```

然后，在您的`build.rs`文件中：

```rust
use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/my_model.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

这会在构建过程中从您的ONNX模型生成Rust代码。

### 步骤2：修改`mod.rs`

在您的`src/model/mod.rs`文件中，包含生成的代码：

```rust
pub mod my_model {
    include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
}
```

### 步骤3：使用导入的模型

现在您可以在代码中使用导入的模型：

```rust
use burn::tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use model::my_model::Model;

fn main() {
    let device = NdArrayDevice::default();

    // 创建模型实例并从目标目录默认设备加载权重
    let model: Model<NdArray<f32>> = Model::default();

    // 创建输入张量（替换为您的实际输入）
    let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 3, 224, 224], &device);

    // 执行推理
    let output = model.forward(input);

    println!("模型输出: {:?}", output);
}
```

## 高级配置

`ModelGen`结构体提供了几个配置选项：

```rust
ModelGen::new()
    .input("path/to/model.onnx")
    .out_dir("model/")
    .record_type(RecordType::NamedMpk)
    .half_precision(false)
    .embed_states(false)
    .run_from_script();
```

- `record_type`：定义存储权重的格式（Bincode、NamedMpk、NamedMpkGz或PrettyJson）。
- `half_precision`：通过使用半精度（f16）权重来减少模型大小。
- `embed_states`：将模型权重直接嵌入到生成的Rust代码中（需要记录类型为`Bincode`）。

## 加载和使用模型

根据您的配置，您可以通过几种方式加载模型：

```rust
// 使用设备创建新模型实例
//（随机初始化权重并惰性加载；之后通过`load_record`加载权重）
let model = Model::<Backend>::new(&device);

// 从文件加载
//（文件类型应与`ModelGen`中指定的记录类型匹配）
let model = Model::<Backend>::from_file("path/to/weights", &device);

// 从嵌入权重加载（如果embed_states为true）
let model = Model::<Backend>::from_embedded(&device);

// 从输出目录使用默认设备加载（适用于测试）
let model = Model::<Backend>::default();
```

## 故障排除

常见问题和解决方案：

1. **不支持的ONNX操作符**：检查[支持的ONNX操作符列表](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md)。您可能需要简化模型或等待支持。

2. **构建错误**：确保您的`burn-import`版本与Burn版本匹配，并验证`build.rs`中的ONNX文件路径。

3. **运行时错误**：确认您的输入张量与模型的预期形状和数据类型匹配。

4. **性能问题**：尝试使用`half_precision`选项减少内存使用，或尝试不同的`record_type`选项。

5. **查看生成的文件**：在`OUT_DIR`目录中找到生成的Rust代码和权重（通常是`target/debug/build/<project>/out`）。

## 示例和资源

有关实际示例，请查看：

1. [MNIST推理示例](https://github.com/tracel-ai/burn/tree/main/examples/onnx-inference)
2. [SqueezeNet图像分类](https://github.com/tracel-ai/models/tree/main/squeezenet-burn)

这些示例演示了Burn项目中ONNX导入的实际使用。

## 结论

将ONNX模型导入Burn结合了预训练模型的庞大生态系统与Burn的性能和Rust的安全特性。按照本指南，您可以无缝地将ONNX模型集成到您的Burn项目中，用于推理、微调或进一步开发。

`burn-import` crate正在积极开发中，正在进行工作以支持更多ONNX操作符并提高性能。请关注Burn仓库以获取更新！

---

> 🚨**注意**：`burn-import` crate正在积极开发中。有关支持的ONNX操作符的最新信息，请参考[官方文档](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md)。