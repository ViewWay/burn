# 张量

如在[模型部分](../basic-workflow/model.md)中所解释的，张量结构体有3个泛型参数：后端B、维度D和数据类型。

```rust , ignore
Tensor<B, D>           // 浮点张量（默认）
Tensor<B, D, Float>    // 显式浮点张量
Tensor<B, D, Int>      // 整数张量
Tensor<B, D, Bool>     // 布尔张量
```

请注意，用于`Float`、`Int`和`Bool`张量的具体元素类型由后端实现定义。

Burn张量在其声明中通过维度数D定义，而不是通过其形状定义。张量的实际形状从其初始化中推断出来。例如，大小为(5,)的张量初始化如下：

```rust, ignore
let floats = [1.0, 2.0, 3.0, 4.0, 5.0];

// 获取默认设备
let device = Default::default();

// 正确：张量是1维的，有5个元素
let tensor_1 = Tensor::<Backend, 1>::from_floats(floats, &device);

// 错误：let tensor_1 = Tensor::<Backend, 5>::from_floats(floats, &device);
// 这将导致错误，用于创建5维张量
```

### 初始化

Burn张量主要使用`from_data()`方法初始化，该方法以`TensorData`结构体作为输入。`TensorData`结构体有两个公共字段：`shape`和`dtype`。`value`现在以字节形式存储，是私有的，但可以通过以下任何方法访问：`as_slice`、`as_mut_slice`、`to_vec`和`iter`。要从张量中检索数据，在打算之后重用张量时应使用`.to_data()`方法。或者，对于一次性使用，推荐使用`.into_data()`。让我们看几个从不同输入初始化张量的示例。

```rust, ignore

// 从给定后端（Wgpu）初始化
let tensor_1 = Tensor::<Wgpu, 1>::from_data([1.0, 2.0, 3.0], &device);

// 从通用后端初始化
let tensor_2 = Tensor::<Backend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);

// 使用from_floats初始化（推荐用于f32元素类型）
// 将在内部转换为TensorData。
let tensor_3 = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);

// 从数组切片初始化整数张量
let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];
let tensor_4 = Tensor::<Backend, 1, Int>::from_data(TensorData::from(&arr[0..3]), &device);

// 从自定义类型初始化

struct BodyMetrics {
    age: i8,
    height: i16,
    weight: f32
}

let bmi = BodyMetrics{
        age: 25,
        height: 180,
        weight: 80.0
    };
let data  = TensorData::from([bmi.age as f32, bmi.height as f32, bmi.weight]);
let tensor_5 = Tensor::<Backend, 1>::from_data(data, &device);

```

## 所有权和克隆

几乎所有Burn操作都获取输入张量的所有权。因此，多次重用张量将需要克隆它。让我们看一个例子来更好地理解所有权规则和克隆。假设我们想要对输入张量进行简单的最小-最大归一化。

```rust, ignore
let input = Tensor::<Wgpu, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
let min = input.min();
let max = input.max();
let input = (input - min).div(max - min);
```

使用PyTorch张量，上述代码将按预期工作。然而，Rust严格的所有权规则将给出错误并阻止在第一次`.min()`操作后使用输入张量。输入张量的所有权转移到变量`min`，输入张量不再可用于进一步操作。像大多数复杂原语一样，Burn张量不实现`Copy` trait，因此必须显式克隆。现在让我们重写一个使用克隆进行最小-最大归一化的工作示例。

```rust, ignore
let input = Tensor::<Wgpu, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
let min = input.clone().min();
let max = input.clone().max();
let input = (input.clone() - min.clone()).div(max - min);
println!("{}", input.to_data());// 成功：[0.0, 0.33333334, 0.6666667, 1.0]

// 注意max、min在最后一次操作中被移动了
// 所以下面的打印将给出错误。
// 如果我们想将它们用于进一步操作，
// 需要以类似的方式克隆它们。
// println!("{:?}", min.to_data());
```

我们不需要担心内存开销，因为通过克隆，张量的缓冲区不会被复制，只是对其的引用增加了。这使得能够准确确定张量被使用的次数，这对于重用张量缓冲区甚至将操作融合到单个内核中非常方便（[burn-fusion](https://burn.dev/docs/burn_fusion/index.htmls)）。出于这个原因，我们不提供显式的就地操作。如果张量只使用一次，当可用时将始终使用就地操作。

## 张量操作

通常使用PyTorch时，在反向传递期间不支持显式的就地操作，这使得它们仅对数据预处理或仅推理模型实现有用。使用Burn，您可以更多地关注模型应该做什么，而不是如何做。我们负责确保您的代码在训练和推理期间尽可能快地运行。同样的原则也适用于广播；除非另有说明，所有操作都支持广播。

在这里，我们提供了所有支持的操作及其PyTorch等效项的列表。请注意，为了简单起见，我们忽略了类型签名。有关更多详细信息，请参阅[完整文档](https://docs.rs/burn/latest/burn/tensor/struct.Tensor.html)。

### 基本操作

这些操作适用于所有张量类型：`Int`、`Float`和`Bool`。

| Burn                                        | PyTorch等效项                                                        |
|---------------------------------------------|---------------------------------------------------------------------------|
| `Tensor::cat(tensors, dim)`                 | `torch.cat(tensors, dim)`                                                 |
| `Tensor::empty(shape, device)`              | `torch.empty(shape, device=device)`                                       |
| `Tensor::from_primitive(primitive)`         | N/A                                                                       |
| `Tensor::stack(tensors, dim)`               | `torch.stack(tensors, dim)`                                               |
| `tensor.all()`                              | `tensor.all()`                                                            |
| `tensor.all_dim(dim)`                       | `tensor.all(dim)`                                                         |
| `tensor.any()`                              | `tensor.any()`                                                            |
| `tensor.any_dim(dim)`                       | `tensor.any(dim)`                                                         |
| `tensor.chunk(num_chunks, dim)`             | `tensor.chunk(num_chunks, dim)`                                           |
| `tensor.split(split_size, dim)`             | `tensor.split(split_size, dim)`                                           |
| `tensor.split_with_sizes(split_sizes, dim)` | `tensor.split([split_sizes], dim)`                                        |
| `tensor.device()`                           | `tensor.device`                                                           |
| `tensor.dtype()`                            | `tensor.dtype`                                                            |
| `tensor.dims()`                             | `tensor.size()`                                                           |
| `tensor.equal(other)`                       | `x == y`                                                                  |
| `tensor.expand(shape)`                      | `tensor.expand(shape)`                                                    |
| `tensor.expand(shape)`                      | `tensor.expand(shape)`                                                    |
| `tensor.flatten(start_dim, end_dim)`        | `tensor.flatten(start_dim, end_dim)`                                      |
| `tensor.flip(axes)`                         | `tensor.flip(axes)`                                                       |
| `tensor.into_data()`                        | N/A                                                                       |
| `tensor.into_primitive()`                   | N/A                                                                       |
| `tensor.into_scalar()`                      | `tensor.item()`                                                           |
| `tensor.narrow(dim, start, length)`         | `tensor.narrow(dim, start, length)`                                       |
| `tensor.not_equal(other)`                   | `x != y`                                                                  |
| `tensor.permute(axes)`                      | `tensor.permute(axes)`                                                    |
| `tensor.movedim(src, dst)`                  | `tensor.movedim(src, dst)`                                                |
| `tensor.repeat_dim(dim, times)`             | `tensor.repeat(*[times if i == dim else 1 for i in range(tensor.dim())])` |
| `tensor.repeat(sizes)`                      | `tensor.repeat(sizes)`                                                    |
| `tensor.reshape(shape)`                     | `tensor.view(shape)`                                                      |
| `tensor.roll(shfts, dims)`                  | `tensor.roll(shifts, dims)`                                               |
| `tensor.roll_dim(shift, dim)`               | `tensor.roll([shift], [dim])`                                             |
| `tensor.shape()`                            | `tensor.shape`                                                            |
| `tensor.slice(ranges)`                      | `tensor[(*ranges,)]`                                                      |
| `tensor.slice_assign(ranges, values)`       | `tensor[(*ranges,)] = values`                                             |
| `tensor.slice_fill(ranges, value)`          | `tensor[(*ranges,)] = value`                                              |
| `tensor.slice_dim(dim, range)`              | N/A                                                                       |
| `tensor.squeeze(dim)`                       | `tensor.squeeze(dim)`                                                     |
| `tensor.swap_dims(dim1, dim2)`              | `tensor.transpose(dim1, dim2)`                                            |
| `tensor.to_data()`                          | N/A                                                                       |
| `tensor.to_device(device)`                  | `tensor.to(device)`                                                       |
| `tensor.transpose()`                        | `tensor.T`                                                                |
| `tensor.t()`                                | `tensor.T`                                                                |
| `tensor.unsqueeze()`                        | `tensor.unsqueeze(0)`                                                     |
| `tensor.unsqueeze_dim(dim)`                 | `tensor.unsqueeze(dim)`                                                   |
| `tensor.unsqueeze_dims(dims)`               | N/A                                                                       |

### 数值操作

这些操作适用于数值张量类型：`Float`和`Int`。

| Burn                                                            | PyTorch等效项                             |
| --------------------------------------------------------------- | ---------------------------------------------- |
| `Tensor::eye(size, device)`                                     | `torch.eye(size, device=device)`               |
| `Tensor::full(shape, fill_value, device)`                       | `torch.full(shape, fill_value, device=device)` |
| `Tensor::ones(shape, device)`                                   | `torch.ones(shape, device=device)`             |
| `Tensor::zeros(shape, device)`                                  | `torch.zeros(shape, device=device)`            |
| `tensor.abs()`                                                  | `torch.abs(tensor)`                            |
| `tensor.add(other)` or `tensor + other`                         | `tensor + other`                               |
| `tensor.add_scalar(scalar)` or `tensor + scalar`                | `tensor + scalar`                              |
| `tensor.all_close(other, atol, rtol)`                           | `torch.allclose(tensor, other, atol, rtol)`    |
| `tensor.argmax(dim)`                                            | `tensor.argmax(dim)`                           |
| `tensor.argmin(dim)`                                            | `tensor.argmin(dim)`                           |
| `tensor.argsort(dim)`                                           | `tensor.argsort(dim)`                          |
| `tensor.argsort_descending(dim)`                                | `tensor.argsort(dim, descending=True)`         |
| `tensor.bool()`                                                 | `tensor.bool()`                                |
| `tensor.clamp(min, max)`                                        | `torch.clamp(tensor, min=min, max=max)`        |
| `tensor.clamp_max(max)`                                         | `torch.clamp(tensor, max=max)`                 |
| `tensor.clamp_max(max)`                                         | `torch.clamp(tensor, max=max)`                 |
| `tensor.clamp_min(min)`                                         | `torch.clamp(tensor, min=min)`                 |
| `tensor.contains_nan()`                                         | N/A                                            |
| `tensor.div(other)` or `tensor / other`                         | `tensor / other`                               |
| `tensor.div_scalar(scalar)` or `tensor / scalar`                | `tensor / scalar`                              |
| `tensor.dot()`                                                  | `torch.dot()`                                  |
| `tensor.equal_elem(other)`                                      | `tensor.eq(other)`                             |
| `tensor.full_like(fill_value)`                                  | `torch.full_like(tensor, fill_value)`          |
| `tensor.gather(dim, indices)`                                   | `torch.gather(tensor, dim, indices)`           |
| `tensor.greater(other)`                                         | `tensor.gt(other)`                             |
| `tensor.greater_elem(scalar)`                                   | `tensor.gt(scalar)`                            |
| `tensor.greater_equal(other)`                                   | `tensor.ge(other)`                             |
| `tensor.greater_equal_elem(scalar)`                             | `tensor.ge(scalar)`                            |
| `tensor.lower(other)`                                           | `tensor.lt(other)`                             |
| `tensor.lower_elem(scalar)`                                     | `tensor.lt(scalar)`                            |
| `tensor.lower_equal(other)`                                     | `tensor.le(other)`                             |
| `tensor.lower_equal_elem(scalar)`                               | `tensor.le(scalar)`                            |
| `tensor.mask_fill(mask, value)`                                 | `tensor.masked_fill(mask, value)`              |
| `tensor.mask_where(mask, value_tensor)`                         | `torch.where(mask, value_tensor, tensor)`      |
| `tensor.max()`                                                  | `tensor.max()`                                 |
| `tensor.max_abs()`                                              | `tensor.abs().max()`                           |
| `tensor.max_abs_dim(dim)`                                       | `tensor.abs().max(dim, keepdim=True)`          |
| `tensor.max_dim(dim)`                                           | `tensor.max(dim, keepdim=True)`                |
| `tensor.max_dim_with_indices(dim)`                              | N/A                                            |
| `tensor.max_pair(other)`                                        | `torch.Tensor.max(a,b)`                        |
| `tensor.mean()`                                                 | `tensor.mean()`                                |
| `tensor.mean_dim(dim)`                                          | `tensor.mean(dim, keepdim=True)`               |
| `tensor.min()`                                                  | `tensor.min()`                                 |
| `tensor.min_dim(dim)`                                           | `tensor.min(dim, keepdim=True)`                |
| `tensor.min_dim_with_indices(dim)`                              | N/A                                            |
| `tensor.min_pair(other)`                                        | `torch.Tensor.min(a,b)`                        |
| `tensor.mul(other)` or `tensor * other`                         | `tensor * other`                               |
| `tensor.mul_scalar(scalar)` or `tensor * scalar`                | `tensor * scalar`                              |
| `tensor.neg()` or `-tensor`                                     | `-tensor`                                      |
| `tensor.not_equal_elem(scalar)`                                 | `tensor.ne(scalar)`                            |
| `tensor.ones_like()`                                            | `torch.ones_like(tensor)`                      |
| `tensor.one_hot(num_classes)`                                   | `torch.nn.functional.one_hot`                  |
| `tensor.one_hot_fill(num_classes, on_value, off_value, axis)`   | N/A                                            |
| `tensor.pad(pads, value)`                                       | `torch.nn.functional.pad(input, pad, value)`   |
| `tensor.powf(other)` or `tensor.powi(intother)`                 | `tensor.pow(other)`                            |
| `tensor.powf_scalar(scalar)` or `tensor.powi_scalar(intscalar)` | `tensor.pow(scalar)`                           |
| `tensor.prod()`                                                 | `tensor.prod()`                                |
| `tensor.prod_dim(dim)`                                          | `tensor.prod(dim, keepdim=True)`               |
| `tensor.rem(other)` or `tensor % other`                         | `tensor % other`                               |
| `tensor.scatter(dim, indices, values)`                          | `tensor.scatter_add(dim, indices, values)`     |
| `tensor.select(dim, indices)`                                   | `tensor.index_select(dim, indices)`            |
| `tensor.select_assign(dim, indices, values)`                    | N/A                                            |
| `tensor.sign()`                                                 | `tensor.sign()`                                |
| `tensor.sort(dim)`                                              | `tensor.sort(dim).values`                      |
| `tensor.sort_descending(dim)`                                   | `tensor.sort(dim, descending=True).values`     |
| `tensor.sort_descending_with_indices(dim)`                      | `tensor.sort(dim, descending=True)`            |
| `tensor.sort_descending_with_indices(dim)`                      | `tensor.sort(dim, descending=True)`            |
| `tensor.sort_with_indices(dim)`                                 | `tensor.sort(dim)`                             |
| `tensor.sub(other)` or `tensor - other`                         | `tensor - other`                               |
| `tensor.sub_scalar(scalar)` or `tensor - scalar`                | `tensor - scalar`                              |
| `scalar - tensor`                                               | `scalar - tensor`                              |
| `tensor.sum()`                                                  | `tensor.sum()`                                 |
| `tensor.sum_dim(dim)`                                           | `tensor.sum(dim, keepdim=True)`                |
| `tensor.topk(k, dim)`                                           | `tensor.topk(k, dim).values`                   |
| `tensor.topk_with_indices(k, dim)`                              | `tensor.topk(k, dim)`                          |
| `tensor.tril(diagonal)`                                         | `torch.tril(tensor, diagonal)`                 |
| `tensor.triu(diagonal)`                                         | `torch.triu(tensor, diagonal)`                 |
| `tensor.zeros_like()`                                           | `torch.zeros_like(tensor)`                     |

### 浮点操作

这些操作仅适用于`Float`张量。

| Burn API                                     | PyTorch等效项                      |
| -------------------------------------------- | ---------------------------------------    |
| `tensor.cast(dtype)`                         | `tensor.to(dtype)`                         |
| `tensor.ceil()`                              | `tensor.ceil()`                            |
| `tensor.cos()`                               | `tensor.cos()`                             |
| `tensor.cosh()`                              | `tensor.cosh()`                            |
| `tensor.erf()`                               | `tensor.erf()`                             |
| `tensor.exp()`                               | `tensor.exp()`                             |
| `tensor.floor()`                             | `tensor.floor()`                           |
| `tensor.from_floats(floats, device)`         | N/A                                        |
| `tensor.from_full_precision(tensor)`         | N/A                                        |
| `tensor.int()`                               | Similar to `tensor.to(torch.long)`         |
| `tensor.is_close(other, atol, rtol)`         | `torch.isclose(tensor, other, atol, rtol)` |
| `tensor.is_finite()`                         | `torch.isfinite(tensor)`                   |
| `tensor.is_inf()`                            | `torch.isinf(tensor)`                      |
| `tensor.is_nan()`                            | `torch.isnan(tensor)`                      |
| `tensor.log()`                               | `tensor.log()`                             |
| `tensor.log1p()`                             | `tensor.log1p()`                           |
| `tensor.matmul(other)`                       | `tensor.matmul(other)`                     |
| `tensor.random(shape, distribution, device)` | N/A                                        |
| `tensor.random_like(distribution)`           | `torch.rand_like()` only uniform           |
| `tensor.recip()` or `1.0 / tensor`           | `tensor.reciprocal()` or `1.0 / tensor`    |
| `tensor.round()`                             | `tensor.round()`                           |
| `tensor.sin()`                               | `tensor.sin()`                             |
| `tensor.sinh()`                              | `tensor.sinh()`                            |
| `tensor.sqrt()`                              | `tensor.sqrt()`                            |
| `tensor.tan()`                               | `tensor.tan()`                             |
| `tensor.tanh()`                              | `tensor.tanh()`                            |
| `tensor.to_full_precision()`                 | `tensor.to(torch.float)`                   |
| `tensor.var(dim)`                            | `tensor.var(dim)`                          |
| `tensor.var_bias(dim)`                       | N/A                                        |
| `tensor.var_mean(dim)`                       | N/A                                        |
| `tensor.var_mean_bias(dim)`                  | N/A                                        |

### 整数操作

这些操作仅适用于`Int`张量。

| Burn API                                         | PyTorch等效项                                      |
| ------------------------------------------------ | ------------------------------------------------------- |
| `Tensor::arange(5..10, device)`                  | `tensor.arange(start=5, end=10, device=device)`         |
| `Tensor::arange_step(5..10, 2, device)`          | `tensor.arange(start=5, end=10, step=2, device=device)` |
| `tensor.bitwise_and(other)`                      | `torch.bitwise_and(tensor, other)`                      |
| `tensor.bitwise_and_scalar(scalar)`              | `torch.bitwise_and(tensor, scalar)`                     |
| `tensor.bitwise_not()`                           | `torch.bitwise_not(tensor)`                             |
| `tensor.bitwise_left_shift(other)`               | `torch.bitwise_left_shift(tensor, other)`               |
| `tensor.bitwise_left_shift_scalar(scalar)`       | `torch.bitwise_left_shift(tensor, scalar)`              |
| `tensor.bitwise_right_shift(other)`              | `torch.bitwise_right_shift(tensor, other)`              |
| `tensor.bitwise_right_shift_scalar(scalar)`      | `torch.bitwise_right_shift(tensor, scalar)`             |
| `tensor.bitwise_or(other)`                       | `torch.bitwise_or(tensor, other)`                       |
| `tensor.bitwise_or_scalar(scalar)`               | `torch.bitwise_or(tensor, scalar)`                      |
| `tensor.bitwise_xor(other)`                      | `torch.bitwise_xor(tensor, other)`                      |
| `tensor.bitwise_xor_scalar(scalar)`              | `torch.bitwise_xor(tensor, scalar)`                     |
| `tensor.float()`                                 | `tensor.to(torch.float)`                                |
| `tensor.from_ints(ints)`                         | N/A                                                     |
| `tensor.int_random(shape, distribution, device)` | N/A                                                     |
| `tensor.cartesian_grid(shape, device)`           | N/A                                                     |

### 布尔操作

这些操作仅适用于`Bool`张量。

| Burn API                             | PyTorch等效项              |
| ------------------------------------ | ------------------------------- |
| `Tensor::diag_mask(shape, diagonal)` | N/A                             |
| `Tensor::tril_mask(shape, diagonal)` | N/A                             |
| `Tensor::triu_mask(shape, diagonal)` | N/A                             |
| `tensor.argwhere()`                  | `tensor.argwhere()`             |
| `tensor.bool_and()`                  | `tensor.logical_and()`          |
| `tensor.bool_not()`                  | `tensor.logical_not()`          |
| `tensor.bool_or()`                   | `tensor.logical_or()`           |
| `tensor.float()`                     | `tensor.to(torch.float)`        |
| `tensor.int()`                       | `tensor.to(torch.long)`         |
| `tensor.nonzero()`                   | `tensor.nonzero(as_tuple=True)` |

### 量化操作

这些操作仅适用于实现量化策略的后端上的`Float`张量。

| Burn API                           | PyTorch等效项 |
| ---------------------------------- | ------------------ |
| `tensor.quantize(scheme, qparams)` | N/A                |
| `tensor.dequantize()`              | N/A                |

## 激活函数

| Burn API                                         | PyTorch等效项                                 |
| ------------------------------------------------ | -------------------------------------------------- |
| `activation::gelu(tensor)`                       | `nn.functional.gelu(tensor)`                       |
| `activation::hard_sigmoid(tensor, alpha, beta)`  | `nn.functional.hardsigmoid(tensor)`                |
| `activation::leaky_relu(tensor, negative_slope)` | `nn.functional.leaky_relu(tensor, negative_slope)` |
| `activation::log_sigmoid(tensor)`                | `nn.functional.log_sigmoid(tensor)`                |
| `activation::log_softmax(tensor, dim)`           | `nn.functional.log_softmax(tensor, dim)`           |
| `activation::mish(tensor)`                       | `nn.functional.mish(tensor)`                       |
| `activation::prelu(tensor,alpha)`                | `nn.functional.prelu(tensor,weight)`               |
| `activation::quiet_softmax(tensor, dim)`         | `nn.functional.quiet_softmax(tensor, dim)`         |
| `activation::relu(tensor)`                       | `nn.functional.relu(tensor)`                       |
| `activation::sigmoid(tensor)`                    | `nn.functional.sigmoid(tensor)`                    |
| `activation::silu(tensor)`                       | `nn.functional.silu(tensor)`                       |
| `activation::softmax(tensor, dim)`               | `nn.functional.softmax(tensor, dim)`               |
| `activation::softmin(tensor, dim)`               | `nn.functional.softmin(tensor, dim)`               |
| `activation::softplus(tensor, beta)`             | `nn.functional.softplus(tensor, beta)`             |
| `activation::tanh(tensor)`                       | `nn.functional.tanh(tensor)`                       |

## 网格函数

| Burn API                                           | PyTorch等效项                      |
|----------------------------------------------------|-----------------------------------------|
| `grid::meshgrid(tensors, GridIndexing::Matrix)`    | `torch.meshgrid(tensors, indexing="ij") |
| `grid::meshgrid(tensors, GridIndexing::Cartesian)` | `torch.meshgrid(tensors, indexing="xy") |

## 线性代数函数

| Burn API                               | PyTorch等效项                        |
|----------------------------------------|-------------------------------------------|
| `linalg::vector_norm(tensors, p, dim)` | `torch.linalg.vector_norm(tensor, p, dim) |

## 显示张量详细信息

Burn提供了灵活的选项来显示张量信息，允许您控制详细程度和格式以满足您的需求。

### 基本显示

要显示张量的详细视图，您可以简单地使用Rust的`println!`或`format!`宏：

```rust
let tensor = Tensor::<Backend, 2>::full([2, 3], 0.123456789, &Default::default());
println!("{}", tensor);
```

这将输出：

```
Tensor {
  data:
[[0.12345679, 0.12345679, 0.12345679],
 [0.12345679, 0.12345679, 0.12345679]],
  shape:  [2, 3],
  device:  Cpu,
  backend:  "ndarray",
  kind:  "Float",
  dtype:  "f32",
}
```

### 控制精度

您可以使用Rust的格式化语法控制显示的小数位数：

```rust
println!("{:.2}", tensor);
```

输出：

```
Tensor {
  data:
[[0.12, 0.12, 0.12],
 [0.12, 0.12, 0.12]],
  shape:  [2, 3],
  device:  Cpu,
  backend:  "ndarray",
  kind:  "Float",
  dtype:  "f32",
}
```

### 全局打印选项

为了更精细地控制张量打印，Burn提供了一个`PrintOptions`结构体和一个`set_print_options`函数：

```rust
use burn::tensor::{set_print_options, PrintOptions};

let print_options = PrintOptions {
    precision: Some(2),
    ..Default::default()
};

set_print_options(print_options);
```

选项：

- `precision`：浮点数的小数位数（默认：None）
- `threshold`：显示前 summarizing 的最大元素数（默认：1000）
- `edge_items`：summarizing时在每个维度的开头和结尾显示的项目数（默认：3）

### 检查张量接近度

Burn提供了一个实用函数`check_closeness`来比较两个张量并评估它们的相似性。这个函数在调试和验证张量操作时特别有用，特别是在处理浮点运算时，小的数值差异可能会累积。在从其他框架导入模型的过程中，它也很有价值，有助于确保导入的模型产生与原始模型一致的结果。

以下是使用`check_closeness`的示例：

```rust
use burn::tensor::{check_closeness, Tensor};
type B = burn::backend::NdArray;

let device = Default::default();
let tensor1 = Tensor::<B, 1>::from_floats(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.001, 7.002, 8.003, 9.004, 10.1],
    &device,
);
let tensor2 = Tensor::<B, 1>::from_floats(
    [1.0, 2.0, 3.0, 4.000, 5.0, 6.0, 7.001, 8.002, 9.003, 10.004],
    &device,
);

check_closeness(&tensor1, &tensor2);
```

`check_closeness`函数逐元素比较两个输入张量，检查它们与一系列epsilon值的绝对差异。然后它打印一个详细报告，显示在每个容差级别内的元素百分比。

输出为不同的epsilon值提供了一个细分，允许您在各种精度级别上评估张量的接近度。这在处理可能引入小数值差异的操作时特别有帮助。

该函数使用彩色编码输出来突出显示结果：

- 绿色 [PASS]：所有元素都在指定的容差范围内。
- 黄色 [WARN]：大多数元素（90%或更多）在容差范围内。
- 红色 [FAIL]：检测到显著差异。

这个实用程序在实现或调试张量操作时非常有价值，特别是那些涉及复杂数学计算的操作，或者在移植算法时。在验证导入模型的准确性时，它也是一个必不可少的工具，确保Burn实现产生的结果与原始模型紧密匹配。