# 模块

`Module` derive允许您创建自己的神经网络模块，类似于PyTorch。derive函数只生成必要的方法，本质上作为类型的参数容器，它不对前向传递的声明做任何假设。

```rust, ignore
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: Gelu,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    /// 添加到结构体的普通方法。
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
```

请注意，结构体中声明的所有字段也必须实现`Module` trait。

## 张量

如果您想创建包含张量的自己的模块，而不仅仅是使用`Module` derive定义的其他模块，您需要小心以实现您想要的行为。

- `Param<Tensor<B, D>>`：如果您希望张量作为模块的参数包含在内，您需要将张量包装在`Param`结构体中。这将创建一个ID，用于标识此参数。这在执行模块优化和保存优化器和模块检查点等状态时至关重要。请注意，模块的记录只包含参数。

- `Param<Tensor<B, D>>.set_require_grad(false)`：如果您希望张量作为模块的参数包含在内，因此与模块权重一起保存，但不希望它被优化器更新。

- `Tensor<B, D>`：如果您希望张量作为可以重新创建的常量在实例化模块时使用。这在生成正弦嵌入时很有用，例如。

## 方法

这些方法适用于所有模块。

| Burn API                                | PyTorch等效项                       |
| --------------------------------------- | ---------------------------------------- |
| `module.devices()`                      | N/A                                      |
| `module.fork(device)`                   | 类似于 `module.to(device).detach()`  |
| `module.to_device(device)`              | `module.to(device)`                      |
| `module.no_grad()`                      | `module.require_grad_(False)`            |
| `module.num_params()`                   | N/A                                      |
| `module.visit(visitor)`                 | N/A                                      |
| `module.map(mapper)`                    | N/A                                      |
| `module.into_record()`                  | 类似于 `state_dict`                  |
| `module.load_record(record)`            | 类似于 `load_state_dict(state_dict)` |
| `module.save_file(file_path, recorder)` | N/A                                      |
| `module.load_file(file_path, recorder)` | N/A                                      |

类似于后端trait，还有`AutodiffModule` trait来表示具有自动微分支持的模块。

| Burn API         | PyTorch等效项 |
| ---------------- | ------------------ |
| `module.valid()` | `module.eval()`    |

## 访问器和映射器

如前所述，模块主要作为参数容器。因此，我们自然提供了几种对每个参数执行函数的方法。这与PyTorch不同，在PyTorch中扩展模块功能并不那么简单。

`map`和`visitor`方法非常相似，但有不同的用途。映射用于可能的可变操作，其中模块的每个参数都可以更新为新值。在Burn中，优化器本质上只是复杂的模块映射器。另一方面，访问器用于您不打算修改模块但需要从中检索特定信息的情况，例如参数数量或正在使用的设备列表。

您可以通过实现这些简单trait来实现自己的映射器或访问器：

```rust, ignore
/// 模块访问器trait。
pub trait ModuleVisitor<B: Backend> {
    /// 访问模块中的浮点张量。
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>);
    /// 访问模块中的整数张量。
    fn visit_int<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Int>);
    /// 访问模块中的布尔张量。
    fn visit_bool<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Bool>);
}

/// 模块映射器trait。
pub trait ModuleMapper<B: Backend> {
    /// 映射模块中的浮点张量。
    fn map_float<const D: usize>(&mut self, id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D>;
    /// 映射模块中的整数张量。
    fn map_int<const D: usize>(&mut self, id: ParamId, tensor: Tensor<B, D, Int>) -> Tensor<B, D, Int>;
    /// 映射模块中的布尔张量。
    fn map_bool<const D: usize>(&mut self, id: ParamId, tensor: Tensor<B, D, Bool>) -> Tensor<B, D, Bool>;
}
```

请注意，trait不要求实现所有方法，因为它们已经被定义为不执行任何操作。如果您只对浮点张量感兴趣（如大多数用例），那么您可以简单地实现`map_float`或`visit_float`。

例如，`ModuleMapper` trait可以实现将所有参数限制在范围`[min, max]`内。

```rust, ignore
/// 将参数限制在范围`[min, max]`内。
pub struct Clamp {
    /// 范围的下界。
    pub min: f32,
    /// 范围的上界。
    pub max: f32,
}

// 将所有浮点参数张量限制在`[min, max]`范围内。
impl<B: Backend> ModuleMapper<B> for Clamp {
    fn map_float<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        tensor: burn::prelude::Tensor<B, D>,
    ) -> burn::prelude::Tensor<B, D> {
        tensor.clamp(self.min, self.max)
    }
}

// 将模块映射器限制在范围`[-0.5, 0.5]`内
let mut clamp = Clamp {
    min: -0.5,
    max: 0.5,
};
let model = model.map(&mut clamp);
```

如果您想在训练期间使用此功能来约束模型参数，请确保参数张量仍然被跟踪用于自动微分。这可以通过对实现进行简单调整来完成。

```rust, ignore
impl<B: AutodiffBackend> ModuleMapper<B> for Clamp {
    fn map_float<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        tensor: burn::prelude::Tensor<B, D>,
    ) -> burn::prelude::Tensor<B, D> {
        let is_require_grad = tensor.is_require_grad();

        let mut tensor = Tensor::from_inner(tensor.inner().clamp(self.min, self.max));

        if is_require_grad {
            tensor = tensor.require_grad();
        }

        tensor
    }
}
```

## 模块显示

Burn提供了一种简单的方法来一目了然地显示模块及其配置的结构。您可以打印模块以查看其结构，这在调试和跟踪模块不同版本之间的变化时很有用。（请参见[基本工作流模型](../basic-workflow/model.md)示例的打印输出。）

要自定义模块的显示，您可以为模块实现`ModuleDisplay` trait。这将更改模块及其子模块的默认显示设置。请注意，`ModuleDisplay`会自动为所有模块实现，但您可以通过使用`#[module(custom_display)]`注释模块来覆盖它以自定义显示。

```rust
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: Gelu,
}

impl<B: Backend> ModuleDisplay for PositionWiseFeedForward<B> {
    /// 模块显示的自定义设置。
    /// 如果返回`None`，将使用默认设置。
    fn custom_settings(&self) -> Option<burn::module::DisplaySettings> {
        DisplaySettings::new()
            // 将显示所有属性（默认为false）
            .with_show_all_attributes(false)
            // 将在每个属性后换行（默认为true）
            .with_new_line_after_attribute(true)
            // 将显示参数数量（默认为true）
            .with_show_num_parameters(true)
            // 将缩进2个空格（默认为2）
            .with_indentation_size(2)
            // 将显示参数ID（默认为false）
            .with_show_param_id(false)
            // 便利方法将设置包装在Some()中
            .optional()
    }

    /// 要显示的自定义内容。
    /// 如果返回`None`，将使用默认内容
    ///（模块的所有属性）
    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("linear_inner", &self.linear_inner)
            .add("linear_outer", &self.linear_outer)
            .add("anything", "anything_else")
            .optional()
    }
}
```

## 内置模块

Burn带有内置模块，您可以使用它们来构建自己的模块。

### 通用

| Burn API        | PyTorch等效项                            |
| --------------- | --------------------------------------------- |
| `BatchNorm`     | `nn.BatchNorm1d`, `nn.BatchNorm2d` 等       |
| `Dropout`       | `nn.Dropout`                                  |
| `Embedding`     | `nn.Embedding`                                |
| `Gelu`          | `nn.Gelu`                                     |
| `GroupNorm`     | `nn.GroupNorm`                                |
| `HardSigmoid`   | `nn.Hardsigmoid`                              |
| `InstanceNorm`  | `nn.InstanceNorm1d`, `nn.InstanceNorm2d` 等 |
| `LayerNorm`     | `nn.LayerNorm`                                |
| `LeakyRelu`     | `nn.LeakyReLU`                                |
| `Linear`        | `nn.Linear`                                   |
| `Prelu`         | `nn.PReLu`                                    |
| `Relu`          | `nn.ReLU`                                     |
| `RmsNorm`       | _无直接等效项_                        |
| `SwiGlu`        | _无直接等效项_                        |
| `Interpolate1d` | _无直接等效项_                        |
| `Interpolate2d` | _无直接等效项_                        |

### 卷积

| Burn API          | PyTorch等效项             |
| ----------------- | ------------------------------ |
| `Conv1d`          | `nn.Conv1d`                    |
| `Conv2d`          | `nn.Conv2d`                    |
| `Conv3d`          | `nn.Conv3d`                    |
| `ConvTranspose1d` | `nn.ConvTranspose1d`           |
| `ConvTranspose2d` | `nn.ConvTranspose2d`           |
| `ConvTranspose3d` | `nn.ConvTranspose3d`           |
| `DeformConv2d`    | `torchvision.ops.DeformConv2d` |

### 池化

| Burn API            | PyTorch等效项     |
| ------------------- | ---------------------- |
| `AdaptiveAvgPool1d` | `nn.AdaptiveAvgPool1d` |
| `AdaptiveAvgPool2d` | `nn.AdaptiveAvgPool2d` |
| `AvgPool1d`         | `nn.AvgPool1d`         |
| `AvgPool2d`         | `nn.AvgPool2d`         |
| `MaxPool1d`         | `nn.MaxPool1d`         |
| `MaxPool2d`         | `nn.MaxPool2d`         |

### RNN

| Burn API         | PyTorch等效项     |
| ---------------- | ---------------------- |
| `Gru`            | `nn.GRU`               |
| `Lstm`/`BiLstm`  | `nn.LSTM`              |
| `GateController` | _无直接等效项_ |

### Transformer

| Burn API             | PyTorch等效项      |
| -------------------- | ----------------------- |
| `MultiHeadAttention` | `nn.MultiheadAttention` |
| `TransformerDecoder` | `nn.TransformerDecoder` |
| `TransformerEncoder` | `nn.TransformerEncoder` |
| `PositionalEncoding` | _无直接等效项_  |
| `RotaryEncoding`     | _无直接等效项_  |

### 损失

| Burn API                 | PyTorch等效项       |
| ------------------------ | ------------------------ |
| `BinaryCrossEntropyLoss` | `nn.BCELoss`             |
| `CosineEmbeddingLoss`    | `nn.CosineEmbeddingLoss` |
| `CrossEntropyLoss`       | `nn.CrossEntropyLoss`    |
| `HuberLoss`              | `nn.HuberLoss`           |
| `MseLoss`                | `nn.MSELoss`             |
| `PoissonNllLoss`         | `nn.PoissonNLLLoss`      |