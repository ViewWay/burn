# 量化（测试版）

量化技术使用较低精度的数据类型（如8位整数）而不是浮点精度来执行计算和存储张量。有多种方法对深度学习模型进行量化，分类如下：

- 训练后量化（PTQ）
- 量化感知训练（QAT）

在训练后量化中，模型以浮点精度进行训练，然后转换为较低精度的数据类型。

训练后量化有两种类型：

1. 静态量化：量化模型的权重和激活值。静态量化激活值需要数据校准（即记录激活值以使用代表性数据计算最佳量化参数）。
1. 动态量化：提前量化权重（类似于静态量化），但激活值在运行时动态量化。

有时训练后量化无法达到可接受的任务准确性。这就是量化感知训练发挥作用的地方，因为它在训练期间建模量化效果。量化误差在前向和后向传递中使用伪量化模块进行建模，这有助于模型学习对精度降低更鲁棒的表示。

<div class="warning">

Burn 中的量化支持目前正处于积极开发中。

它在某些后端支持以下模式：

- 静态每张量量化为有符号8位整数（`i8`）

目前不支持整数运算，这意味着张量被反量化以浮点精度执行运算。

</div>

## 模块量化

训练后量化模型的权重相当简单。我们可以访问权重张量并收集其统计数据，例如使用 `MinMaxCalibration` 时的最小值和最大值，以计算量化参数。

```rust , ignore
# use burn::module::Quantizer;
# use burn::tensor::quantization::{Calibration, QuantizationScheme, QuantizationType};
#
// 量化配置
let mut quantizer = Quantizer {
    calibration: Calibration::MinMax,
    scheme: QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8),
};

// 量化权重
let model = model.quantize_weights(&mut quantizer);
```

> 鉴于目前所有运算都以浮点精度执行，在推理前对模块参数进行反量化可能是明智的。这允许我们以降低的精度存储模型以节省磁盘空间，同时保持推理速度。
>
> 这可以通过 `ModuleMapper` 轻松实现。
>
> ```rust, ignore
> # use burn::module::{ModuleMapper, ParamId};
> # use burn::tensor::{backend::Backend, Tensor};
> #
> /// 用于反量化加载的模型参数的模块映射器。
> pub struct Dequantize {}
>
> impl<B: Backend> ModuleMapper<B> for Dequantize {
>     fn map_float<const D: usize>(
>         &mut self,
>         _id: ParamId,
>         tensor: Tensor<B, D>,
>     ) -> Tensor<B, D> {
>         tensor.dequantize()
>     }
> }
>
> // 以浮点精度加载保存的量化模型
> model = model
>     .load_file(file_path, recorder, &device)
>     .expect("应该能够加载量化模型权重")
>     .map(&mut Dequantize {});
> ```

### 校准

校准是量化过程中的一个步骤，在此步骤中计算所有浮点张量的范围。对于权重来说这相当简单，因为在 _量化时_ 已知实际范围（权重是静态的），但激活值需要更多关注。

为了计算量化参数，Burn 支持以下 `Calibration` 方法。

| 方法     | 描述                                                                      |
| :------- | :------------------------------------------------------------------------ |
| `MinMax` | 基于运行时最小值和最大值计算量化范围映射。                                  |

### 量化方案

量化方案定义了量化类型、量化粒度和范围映射技术。

Burn 目前支持以下 `QuantizationType` 变体。

| 类型    | 描述                        |
| :------ | :-------------------------- |
| `QInt8` | 8位有符号整数量化。          |

量化参数基于要表示的值范围定义，通常可以通过每张量量化为整个权重张量计算，或者通过每通道量化为每个通道分别计算（常用于CNN）。

Burn 目前支持以下 `QuantizationScheme` 变体。

| 变体                         | 描述                                                                                                                                                              |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PerTensor(mode, type)`      | 将单组量化参数应用于整个张量。`mode` 定义值的转换方式，`type` 表示目标量化类型。                                                                                    |

#### 量化模式

| 模式        | 描述                                                          |
| ----------- | ------------------------------------------------------------- |
| `Symmetric` | 使用以零为中心的范围的比例因子映射值。                          |