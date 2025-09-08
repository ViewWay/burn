# Safetensors模型

## 简介

Burn支持从Safetensors格式导入模型权重，这是一种安全高效的pickle格式替代品。无论您是在PyTorch中训练了模型，还是想要使用提供Safetensors格式权重的预训练模型，都可以轻松地将它们导入到Burn中。

本指南演示了将模型导出为Safetensors格式并导入到Burn中的完整工作流程。

## 将模型导出为Safetensors格式

要将PyTorch模型导出为Safetensors格式，您需要`safetensors` Python库。该库提供了一个简单的API，用于将模型权重保存为Safetensors格式。

### 示例：导出PyTorch模型

```python
import torch
import torch.nn as nn
from safetensors.torch import save_file

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, (2,2))
        self.conv2 = nn.Conv2d(2, 2, (2,2), bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    # 设置种子以确保可重现性
    torch.manual_seed(42)

    # 初始化模型并确保它在CPU上
    model = Net().to(torch.device("cpu"))

    # 提取模型权重字典
    model_weights = model.state_dict()

    # 保存为Safetensors格式
    save_file(model_weights, "conv2d.safetensors")
```

### 验证导出

您可以使用神经网络可视化工具[Netron](https://github.com/lutzroeder/netron)查看`.safetensors`文件来验证导出的模型。正确导出的文件将显示扁平的张量结构，类似于PyTorch `.pt`权重文件。

## 将Safetensors模型导入Burn

将Safetensors模型导入Burn涉及两个主要步骤：

1. 在Burn中定义模型架构
2. 从Safetensors文件加载权重

### 步骤1：在Burn中定义模型

首先，您需要创建一个与导出模型架构匹配的Burn模型：

```rust
use burn::{
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> Net<B> {
    /// 创建新模型。
    pub fn init(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([2, 2], [2, 2])
            .init(device);
        let conv2 = Conv2dConfig::new([2, 2], [2, 2])
            .with_bias(false)
            .init(device);
        Self { conv1, conv2 }
    }

    /// 模型的前向传递。
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        self.conv2.forward(x)
    }
}
```

### 步骤2：加载权重

您有两种加载权重的选项：

#### 选项A：在运行时动态加载

这种方法在运行时直接加载Safetensors文件，需要`burn-import`依赖：

```rust
use crate::model;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;

type Backend = burn_ndarray::NdArray<f32>;

fn main() {
    let device = Default::default();

    // 从Safetensors文件加载权重
    let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load("./conv2d.safetensors".into(), &device)
        .expect("应该成功解码状态");

    // 初始化模型并加载权重
    let model = model::Net::<Backend>::init(&device).load_record(record);
}
```

#### 选项B：预转换为Burn的二进制格式

这种方法在构建时将Safetensors文件转换为Burn的优化二进制格式，消除了对`burn-import`的运行时依赖：

```rust
// 此代码将放在build.rs或单独的工具中

use crate::model;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;

type Backend = burn_ndarray::NdArray<f32>;

fn convert_model() {
    let device = Default::default();

    // 从Safetensors加载
    let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load("./conv2d.safetensors".into(), &device)
        .expect("应该成功解码状态");

    // 保存为Burn的二进制格式
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, "model.mpk".into())
        .expect("保存模型记录失败");
}

// 在您的应用程序代码中
fn load_model() -> Net<Backend> {
    let device = Default::default();

    // 从Burn的二进制格式加载
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load("./model.mpk".into(), &device)
        .expect("应该成功解码状态");

    Net::<Backend>::init(&device).load_record(record)
}
```

> **注意**：有关预转换模型的示例，请参见Burn仓库中的`examples/import-model-weights`目录。

## 高级配置选项

### 框架特定适配器

导入Safetensors模型时，您可以指定适配器类型来处理框架特定的张量转换。这在从不同ML框架导入模型时至关重要，因为张量布局和命名约定可能有所不同：

```rust
let device = Default::default();

// 使用框架特定适配器创建加载参数
let load_args = LoadArgs::new("model.safetensors".into())
    .with_adapter_type(AdapterType::PyTorch); // 默认适配器

// 使用指定的适配器加载
let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("应该成功解码状态");
```

#### 可用的适配器类型

| 适配器类型          | 描述                                                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PyTorch** (默认) | 自动应用PyTorch特定的转换：<br>- 转置线性层的权重<br>- 重命名归一化参数 (weight→gamma, bias→beta) |
| **NoAdapter**         | 直接加载张量而不进行任何转换<br>- 在从已经匹配Burn张量布局的框架导入时很有用                             |

## 故障排除和高级功能

### 不同模型架构的关键重映射

如果您的Burn模型结构与Safetensors文件中的参数名称不匹配，您可以使用正则表达式重新映射键：

```rust
let device = Default::default();

// 使用键重映射创建加载参数
let load_args = LoadArgs::new("model.safetensors".into())
    // 移除"conv"前缀，例如"conv.conv1" -> "conv1"
    .with_key_remap("conv\\.(.*)", "$1");

let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("应该成功解码状态");

let model = Net::<Backend>::init(&device).load_record(record);
```

### 使用键检查进行调试

为了帮助解决导入问题，您可以启用调试以打印原始键和重新映射的键：

```rust
let device = Default::default();

// 启用键的调试打印
let load_args = LoadArgs::new("model.safetensors".into())
    .with_key_remap("conv\\.(.*)", "$1")
    .with_debug_print();  // 打印原始键和重新映射的键

let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("应该成功解码状态");
```

### 自动处理非连续索引

SafetensorsFileRecorder自动处理模型层名称中的非连续索引。例如，如果源模型包含有间隙的索引：

```
"model.layers.0.weight"
"model.layers.0.bias"
"model.layers.2.weight"  // 注意间隙（没有索引1）
"model.layers.2.bias"
"model.layers.4.weight"
"model.layers.4.bias"
```

记录器将自动重新索引这些以使其连续，同时保持它们的顺序：

```
"model.layers.0.weight"
"model.layers.0.bias"
"model.layers.1.weight"  // 从2重新索引
"model.layers.1.bias"
"model.layers.2.weight"  // 从4重新索引
"model.layers.2.bias"
```

### 部分模型加载

您可以选择性地将权重加载到部分模型中，这对于以下情况很有用：

- 仅从编码器-解码器架构中加载编码器
- 微调特定层，同时随机初始化其他层
- 创建结合不同来源部分的混合模型

### 支持枚举模块

SafetensorsFileRecorder支持包含新类型变体的枚举模块。枚举变体会根据枚举变体类型自动选择，允许灵活的模型架构。