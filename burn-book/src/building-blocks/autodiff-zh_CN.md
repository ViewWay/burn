# 自动微分

Burn的张量也支持自动微分，这是任何深度学习框架的重要组成部分。我们在[上一节](./backend.md)中介绍了`Backend` trait，但Burn还有另一个用于自动微分的trait：`AutodiffBackend`。

然而，并非所有张量都支持自动微分；您需要一个实现了`Backend`和`AutodiffBackend` trait的后端。幸运的是，您可以使用后端装饰器为任何后端添加自动微分功能：`type MyAutodiffBackend = Autodiff<MyBackend>`。这个装饰器通过维护动态计算图并利用内部后端执行张量操作来实现`AutodiffBackend`和`Backend` trait。

`AutodiffBackend` trait添加了在浮点张量上不能被调用的新操作。它还提供了一个新的关联类型`B::Gradients`，其中存储了每个计算出的梯度。

```rust, ignore
fn calculate_gradients<B: AutodiffBackend>(tensor: Tensor<B, 2>) -> B::Gradients {
    let mut gradients = tensor.clone().backward();

    let tensor_grad = tensor.grad(&gradients);        // 获取
    let tensor_grad = tensor.grad_remove(&mut gradients); // 弹出

    gradients
}
```

请注意，即使后端没有实现`AutodiffBackend` trait，某些函数也始终可用。在这种情况下，这些函数将不执行任何操作。

| Burn API                                | PyTorch等效项            |
| --------------------------------------- | ----------------------------- |
| `tensor.detach()`                       | `tensor.detach()`             |
| `tensor.require_grad()`                 | `tensor.requires_grad()`      |
| `tensor.is_require_grad()`              | `tensor.requires_grad`        |
| `tensor.set_require_grad(require_grad)` | `tensor.requires_grad(False)` |

然而，您不太可能犯任何错误，因为您无法在未实现`AutodiffBackend` trait的后端上调用张量的`backward`。此外，没有自动微分后端，您无法检索张量的梯度。

## 与PyTorch的区别

Burn处理梯度的方式与PyTorch不同。首先，调用`backward`时，每个参数的`grad`字段不会更新。相反，反向传递会在容器中返回所有计算出的梯度。这种方法提供了许多好处，例如能够轻松地将梯度发送到其他线程。

您还可以使用张量上的`grad`方法检索特定参数的梯度。由于此方法将梯度作为输入，因此很难忘记事先调用`backward`。请注意，有时使用`grad_remove`可以通过允许就地操作来提高性能。

在PyTorch中，当您在推理或验证中不需要梯度时，通常需要用代码块来限定您的代码范围。

```python
# 推理模式
torch.inference():
   # 您的代码
   ...

# 或者无梯度
torch.no_grad():
   # 您的代码
   ...
```

使用Burn时，您不需要用`Autodiff`包装后端进行推理，并且可以调用`inner()`来获取内部张量，这在验证时很有用。

```rust, ignore
/// 使用 `B: AutodiffBackend`
fn example_validation<B: AutodiffBackend>(tensor: Tensor<B, 2>) {
    let inner_tensor: Tensor<B::InnerBackend, 2> = tensor.inner();
    let _ = inner_tensor + 5;
}

/// 使用 `B: Backend`
fn example_inference<B: Backend>(tensor: Tensor<B, 2>) {
    let _ = tensor + 5;
    ...
}
```

**优化器中的梯度**

我们已经看到了如何将梯度与张量一起使用，但在使用`burn-core`中的优化器时，过程略有不同。为了与`Module` trait配合使用，需要一个转换步骤来将张量参数与其梯度链接起来。这个步骤对于轻松支持梯度累积和在多个设备上训练是必要的，在多个设备上，每个模块可以被分叉并在不同设备上并行运行。我们将在[Module](./module.md)部分深入探讨这个主题。