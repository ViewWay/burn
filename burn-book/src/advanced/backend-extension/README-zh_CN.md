# 后端扩展

Burn旨在成为最灵活的深度学习框架。虽然保持与各种后端的兼容性至关重要，但Burn提供了扩展后端实现功能的能力，以满足您的建模需求。这种多功能性在许多方面都有优势，例如支持自定义操作如flash attention，或手动融合操作以提高性能。

在本节中，我们将深入探讨扩展后端的过程，并提供多个示例。但在我们继续之前，让我们先建立基本原理，这些原理将使您能够创建自己的后端扩展。

正如您所观察到的，Burn中的大多数类型都是针对后端trait的泛型。这可能会给人一种印象，即Burn在后端层之上运行在较高层次。然而，将trait显式化而不是通过编译标志选择，这是一个经过深思熟虑的设计决策。这种显式性并不意味着所有后端必须相同；相反，它在组合后端时提供了极大的灵活性。自动微分后端trait（参见[自动微分部分](../../building-blocks/autodiff.md)）是后端trait如何被扩展以启用反向传播梯度计算的一个例子。此外，这种设计允许您创建自己的后端扩展。为此，您需要设计自己的后端trait，指定应支持哪些函数。

```rust, ignore
pub trait Backend: burn::tensor::backend::Backend {
    fn my_new_function(tensor: B::TensorPrimitive<2>) -> B::TensorPrimitive<2> {
        // 您可以定义一个基本实现，重用Burn后端API。
        // 这很有用，因为所有后端现在都将自动支持您的模型。
        // 但通过在特定后端中实现此块，可以提高此新操作的性能。
    }
}
```

然后，您可以为任何想要支持的后端实现您的新自定义后端trait：

```rust, ignore
impl<E: TchElement> Backend for burn_tch::LibTorch<E> {
   fn my_new_function(tensor: TchTensor<E, 2>) -> TchTensor<E, 2> {
      // 我的Tch实现
   }
}

impl<E: NdArrayElement> Backend for burn_ndarray::NdArray<E> {
    // 没有特定实现，但后端仍可使用。
}
```

您可以使用相同的模式支持反向传递。

```rust, ignore
impl<B: Backend> Backend for burn_autodiff::Autodiff<B> {
    // 没有特定实现；autodiff将与默认实现一起工作。
    // 如果您仍想训练模型，但在推理期间观察到性能提升，这很有用。
}

impl<B: Backend> Backend for burn_autodiff::Autodiff<B> {
   fn my_new_function(tensor: AutodiffTensor<E, 2>) -> AutodiffTensor<E, 2> {
      // 我自己的反向实现，针对我的自定义后端trait泛型。
      //
      // 如果您想在反向传递期间调用自定义内核，可以为您的自定义后端trait添加一个新的
      // `my_new_function_backward`方法。
   }
}

impl<E: TchElement> Backend for burn_autodiff::Autodiff<burn_tch::LibTorch<E>> {
   fn my_new_function(tensor: AutodiffTensor<E, 2>) -> AutodiffTensor<E, 2> {
      // 我自己的反向实现，针对后端实现泛型。
      //
      // 这是调用反向传递自定义内核的另一种方法，
      // 不需要在自定义后端中添加新的`backward`函数。
      // 如果您不希望所有后端都支持训练，这很有用，
      // 当您知道模型只会在一个特定后端上训练时，
      // 可以减少额外代码的需要。
   }
}
```

每个实现的具体细节将由本节提供的示例涵盖。`cubecl`编译器前端是实现自定义内核的推荐方法，因为它支持多个后端，包括`wgpu`和`CUDA`，并且是第一方`burn`内核的编写方式。