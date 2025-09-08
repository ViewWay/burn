go# PyTorch模型

## 简介

Burn支持从PyTorch导入模型权重，无论您是在PyTorch中训练了模型还是想要使用预训练模型。Burn支持导入带有`.pt`和`.safetensors`文件扩展名的PyTorch模型权重。与ONNX模型相比，这些文件只包含模型的权重，因此您需要在Burn中重建模型架构。

本指南演示了从PyTorch导出模型并导入到Burn的完整工作流程。您也可以参考这篇[从PyTorch过渡到Burn](https://dev.to/laggui/transitioning-from-pytorch-to-burn-45m)教程来导入更复杂的模型。

## 将模型导出为PyTorch格式

要正确导出PyTorch模型，您需要使用`torch.save`函数仅保存模型权重(state_dict)，而不是整个模型。

### 示例：导出PyTorch模型

```python
import torch
import torch.nn as nn

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

    # 仅保存权重，而不是整个模型
    torch.save(model_weights, "conv2d.pt")
```

如果您意外保存了整个模型而不是仅保存权重，在导入时可能会遇到如下错误：

```
Failed to decode foobar: DeserializeError("Serde error: other error:
Missing source values for the 'foo1' field of type 'BarRecordItem'.
Please verify the source data and ensure the field name is correct")
```

### 验证导出

您可以使用神经网络可视化工具[Netron](https://github.com/lutzroeder/netron)查看`.pt`文件来验证导出的模型。正确导出的权重文件将显示扁平的张量结构，而错误导出的文件将显示代表整个模型架构的嵌套块。

在Netron中查看导出的模型时，您应该看到类似这样的内容：

![image alt>](./conv2d.svg)

## 将PyTorch模型导入Burn

将PyTorch模型导入Burn涉及两个主要步骤：

1. 在Burn中定义模型架构
2. 从导出的PyTorch模型加载权重

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

这种方法在运行时直接加载PyTorch文件，需要`burn-import`依赖：

```rust
use crate::model;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;

type Backend = burn_ndarray::NdArray<f32>;

fn main() {
    let device = Default::default();

    // 从PyTorch文件加载权重
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load("./conv2d.pt".into(), &device)
        .expect("应该成功解码状态");

    // 初始化模型并加载权重
    let model = model::Net::<Backend>::init(&device).load_record(record);
}
```

#### 选项B：预转换为Burn的二进制格式

这种方法在构建时将PyTorch文件转换为Burn的优化二进制格式，消除了对`burn-import`的运行时依赖：

```rust
// 此代码将放在build.rs或单独的工具中

use crate::model;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;

type Backend = burn_ndarray::NdArray<f32>;

fn convert_model() {
    let device = Default::default();

    // 从PyTorch加载
    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load("./conv2d.pt".into(), &device)
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

## 提取配置

在某些情况下，模型可能需要额外的配置设置，这些设置通常在导出期间包含在`.pt`文件中。`burn-import` cargo包中的`config_from_file`函数允许直接从`.pt`文件中提取这些配置。

```rust
use std::collections::HashMap;

use burn::config::Config;
use burn_import::pytorch::config_from_file;

#[derive(Debug, Config)]
struct NetConfig {
    n_head: usize,
    n_layer: usize,
    d_model: usize,
    some_float: f64,
    some_int: i32,
    some_bool: bool,
    some_str: String,
    some_list_int: Vec<i32>,
    some_list_str: Vec<String>,
    some_list_float: Vec<f64>,
    some_dict: HashMap<String, String>,
}

fn main() {
    let path = "weights_with_config.pt";
    let top_level_key = Some("my_config");
    let config: NetConfig = config_from_file(path, top_level_key).unwrap();
    println!("{:#?}", config);

    // 提取后，建议将其保存为json文件。
    config.save("my_config.json").unwrap();
}
```

## 故障排除和高级功能

### 不同模型架构的关键重映射

如果您的Burn模型结构与PyTorch文件中的参数名称不匹配，您可以使用正则表达式重新映射键：

```rust
let device = Default::default();
let load_args = LoadArgs::new("tests/key_remap/key_remap.pt".into())
    // 移除"conv"前缀，例如"conv.conv1" -> "conv1"
    .with_key_remap("conv\\.(.*)", "$1");

let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("应该成功解码状态");

let model = Net::<Backend>::init(&device).load_record(record);
```

### 使用键检查进行调试

为了帮助解决导入问题，您可以启用调试以打印原始键和重新映射的键：

```rust
let device = Default::default();
let load_args = LoadArgs::new("tests/key_remap/key_remap.pt".into())
    // 移除"conv"前缀，例如"conv.conv1" -> "conv1"
    .with_key_remap("conv\\.(.*)", "$1")
    .with_debug_print(); // 打印键和重新映射的键

let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("应该成功解码状态");

let model = Net::<Backend>::init(&device).load_record(record);
```

以下是输出示例：

```text
键和张量形状的调试信息：
---
原始键: conv.conv1.bias
重新映射键: conv1.bias
形状: [2]
数据类型: F32
---
原始键: conv.conv1.weight
重新映射键: conv1.weight
形状: [2, 2, 2, 2]
数据类型: F32
---
原始键: conv.conv2.weight
重新映射键: conv2.weight
形状: [2, 2, 2, 2]
数据类型: F32
---
```

### 自动处理非连续索引

PyTorchFileRecorder自动处理模型层名称中的非连续索引。例如，如果源模型包含有间隙的索引：

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

### 为state_dict指定顶级键

有时[`state_dict`](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)与其他元数据一起嵌套在顶级键下。在这种情况下，您可以在`LoadArgs`中指定顶级键：

```rust
let device = Default::default();
let load_args = LoadArgs::new("tiny.en.pt".into())
    .with_top_level_key("my_state_dict");

let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("应该成功解码状态")
```

### 支持枚举模块

PyTorchFileRecorder支持包含新类型变体的枚举模块。枚举变体会根据枚举变体类型自动选择，允许灵活的模型架构。

## 当前已知问题

1. [Candle的pickle目前不支持解包布尔张量](https://github.com/tracel-ai/burn/issues/1179)。