# 自定义`cubecl`内核

在本节中，您将学习如何通过使用cubecl编译器前端编写自己的内核来创建自己的自定义操作。我们将以深度学习领域中的常见工作流程为例，在其中我们创建一个内核来融合多个操作。请注意，`burn`会自动执行此操作，但在某些情况下手动实现可能更高效。我们将融合一个矩阵乘法内核，后跟加法和ReLU激活函数，这在各种模型中都很常见。所有代码都可以在[示例目录](https://github.com/tracel-ai/burn/tree/main/examples/custom-cubecl-kernel)中找到。

## 自定义后端Trait

首先，我们需要通过定义自定义后端trait来确定新创建操作的类型签名。由于我们将使用`Backend` trait的关联类型`TensorPrimitive`，它封装了后端的底层张量实现，我们将使用类型别名来避免与关联类型的丑陋消歧。

```rust, ignore
/// 我们创建自己的后端trait，扩展Burn后端trait。
pub trait Backend: burn::tensor::backend::Backend {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// 我们创建自己的自动微分后端trait，扩展Burn自动微分后端trait。
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}
```

在我们的项目中，我们可以使用这些trait而不是Burn提供的`burn::tensor::backend::{Backend, AutodiffBackend}` trait。Burn的用户API通常使用`Tensor`结构体而不是直接处理原始张量类型。因此，我们可以用暴露新操作的函数来封装我们新定义的后端trait，同时保持一致的API。

```rust, ignore
/// 我们使用自定义后端上的添加函数定义自定义实现。
pub fn matmul_add_relu_custom<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let output = B::fused_matmul_add_relu(
        lhs.into_primitive().tensor(),
        rhs.into_primitive().tensor(),
        bias.into_primitive().tensor(),
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// 我们使用基本张量操作定义参考实现。
pub fn matmul_add_relu_reference<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let x = lhs.matmul(rhs) + bias;

    activation::relu(x)
}

```

请注意，我们还提供了一个参考实现用于测试目的，这使我们能够轻松验证我们的新实现。虽然不是必需的，但拥有参考实现是有价值的，特别是在创建参考实现仅使用基本张量操作就可行的项目中。

## 前向内核

现在，让我们使用`cubecl`编译器前端编写融合内核。为了简单起见，我们将创建一个简单的矩阵乘法内核，而不使用任何复杂的技术。我们不会深入探讨`cube`宏的细节，但如果您有兴趣了解更多，请参见[`cubecl` Book](https://github.com/tracel-ai/cubecl/tree/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/cubecl-book)。实际的矩阵乘法、加法和relu计算在 extensive prelude 之后找到，该 prelude 用于正确地将每个计算单元映射到它负责的数据，并支持批次。

```rust, ignore
use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn fused_matmul_add_relu_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    let batch = ABSOLUTE_POS_Z;

    let n_rows = output.shape(output.rank() - 2);
    let n_cols = output.shape(output.rank() - 1);
    let dim_k = rhs.shape(rhs.rank() - 1);

    if row >= n_rows || col >= n_cols {
        return;
    }

    let offset_output = batch * n_rows * n_cols;
    let mut offset_lhs = 0;
    let mut offset_rhs = 0;

    let batch_dims = output.rank() - 2;
    for dim in 0..batch_dims {
        offset_lhs += offset_output / output.stride(dim) % lhs.shape(dim) * lhs.stride(dim);
        offset_rhs += offset_output / output.stride(dim) % rhs.shape(dim) * rhs.stride(dim);
    }

    let mut sum = F::new(0.0);
    for k in 0..dim_k {
        let lhs_index = row * dim_k + k;
        let rhs_index = k * n_cols + col;

        sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
    }

    let out_index = row * n_cols + col;
    let index = offset_output + out_index;

    output[index] = F::max(sum + bias[index], F::new(0.0));
}
```

现在，让我们进入下一步，实现剩余代码来启动内核。我们将深入实现自定义后端trait的通用JIT后端。这会自动为`burn-cuda`、`burn-wgpu`以及融合实现trait。

```rust, ignore
/// 为通用`CubeBackend`实现我们的自定义后端trait。
impl<R: CubeRuntime, F: FloatElement, I: IntElement> Backend for CubeBackend<R, F, I> {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // 定义立方体维度，为简单起见硬编码。
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        lhs.assert_is_on_same_device(&rhs);
        lhs.assert_is_on_same_device(&bias);

        // 为简单起见，确保每个张量都是连续的。
        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);
        let bias = into_contiguous(bias);

        // 获取矩阵乘法相关的形状。
        let ndims = lhs.shape.num_dims();
        let num_rows = lhs.shape.dims[ndims - 2];
        let num_cols = rhs.shape.dims[ndims - 1];

        // 计算输出形状，同时跟踪批次数量。
        let mut num_batches = 1;
        let mut shape_out = vec![0; ndims];
        for i in shape_out.clone().into_iter().take(ndims - 2) {
            shape_out[i] = usize::max(lhs.shape.dims[i], rhs.shape.dims[i]);
            num_batches *= shape_out[i];
        }
        shape_out[ndims - 2] = num_rows;
        shape_out[ndims - 1] = num_cols;
        let shape_out = Shape::from(shape_out);

        // 为输出张量创建缓冲区。
        let buffer = lhs
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        // 创建输出张量原语。
        // 创建输出张量原语。
        let output = CubeTensor::new_contiguous(
            lhs.client.clone(),
            lhs.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // 使用x、y和z中的立方体数量声明wgsl工作组。
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // 使用启动信息和给定缓冲区懒加载执行内核。为简单起见，不执行向量化
        fused_matmul_add_relu_kernel::launch::<F, R>(
            &lhs.client,
            cube_count,
            cube_dim,
            lhs.as_tensor_arg::<F>(1),
            rhs.as_tensor_arg::<F>(1),
            bias.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
        );

        // 返回输出张量。
        output
    }
}
```

在前面的代码块中，我们演示了如何启动修改正确缓冲区的内核。需要注意的是，Rust的可变性安全性在这里不适用；上下文有能力对任何缓冲区执行任何可变操作。虽然在之前我们只修改新创建的输出缓冲区的场景中这不是问题，但明智的做法是记住这一点。

## 反向

现在自定义后端trait已为JIT后端实现，您可以使用它来调用`matmul_add_relu_custom`函数。然而，在这个阶段还不能计算梯度。如果您的用例不超过推理，那么不需要实现以下任何代码。

对于反向传递，我们将利用来自`burn-autodiff`的后端实现，它实际上对后端是泛型的。我们不会为反向传递制作自己的`cubecl`内核，而只会将我们的融合内核用于前向传递，并使用基本操作计算梯度。

```rust, ignore
// 为任何也实现我们自定义后端trait的后端实现我们的自定义后端trait。
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // 创建将实现Backward trait的零大小类型。
        #[derive(Debug)]
        struct FusedMatmulAddReluBackward;

        // 为给定后端B实现反向trait，节点梯度
        // 有三个其他梯度需要计算（lhs、rhs和bias）。
        impl<B: Backend> Backward<B, 3> for FusedMatmulAddReluBackward {
            // 我们必须在前向传递中构建的状态，用于计算反向传递。
            //
            // 请注意，我们可以通过仅保留被跟踪张量的状态来进一步提高性能，
            // 改进内存管理，但为简单起见，我们避免了这部分。
            type State = (NodeID, NodeID, FloatTensor<B>, Shape);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                // 获取每个变量的节点。
                let [node_lhs, node_rhs, node_bias] = ops.parents;
                // 获取当前节点的梯度。
                let grad = grads.consume::<B>(&ops.node);

                // 设置我们的状态。
                let (lhs_state, rhs_state, output, shape_bias) = ops.state;
                let lhs: FloatTensor<B> = checkpointer.retrieve_node_output(lhs_state);
                let rhs: FloatTensor<B> = checkpointer.retrieve_node_output(rhs_state);

                // 获取张量的形状以支持广播。
                let shape_lhs = lhs.shape();
                let shape_rhs = rhs.shape();

                // 使用基本Burn后端trait中已存在的`relu_backward`
                // 函数计算输出的梯度。
                let grad_output = B::relu_backward(output, grad);

                // 计算lhs梯度，即支持广播的矩阵乘法导数。
                let grad_lhs = broadcast_shape::<B>(
                    B::float_matmul(grad_output.clone(), B::float_transpose(rhs)),
                    &shape_lhs,
                );
                // 计算rhs梯度，即支持广播的矩阵乘法导数。
                let grad_rhs = broadcast_shape::<B>(
                    B::float_matmul(B::float_transpose(lhs), grad_output.clone()),
                    &shape_rhs,
                );
                // 加法的导数仅为1，因此我们只需要支持广播来
                // 计算偏置梯度。
                let grad_bias = broadcast_shape::<B>(grad_output, &shape_bias);

                // 根据变量是否标记为`tracked`为每个变量注册梯度。
                if let Some(node) = node_bias {
                    grads.register::<B>(node.id, grad_bias);
                }
                if let Some(node) = node_lhs {
                    grads.register::<B>(node.id, grad_lhs);
                }
                if let Some(node) = node_rhs {
                    grads.register::<B>(node.id, grad_rhs);
                }
            }
        }

        // 使用每个变量节点和相应图形准备有状态操作。
```

```rust, ignore
        // 使用每个变量节点和相应图准备有状态操作。
        //
        // 每个节点可以以与这里定义的相同顺序使用`ops.parents`获取。
        match FusedMatmulAddReluBackward
            .prepare::<C>([lhs.node.clone(), rhs.node.clone(), bias.node.clone()])
            // 将操作标记为计算绑定，意味着它将保存其
            // 状态而不是在检查点期间重新计算
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // 当至少有一个节点被跟踪时，我们应该注册我们的反向步骤。

                // 状态由该操作反向传递所需的内容组成。
                // 由于我们需要父级的输出，我们必须检查点它们的ID以在反向传递开始时检索
                // 它们的节点输出。我们还可以保存辅助数据，如偏置形状。如果我们还需要此操作的输出，
                // 我们可以将其保存在状态中或在反向传递期间重新计算。
                // 这里我们选择将其保存在状态中，因为它是一个计算绑定操作。
                let lhs_state = prep.checkpoint(&lhs);
                let rhs_state = prep.checkpoint(&rhs);
                let bias_shape = bias.primitive.shape();

                let output = B::fused_matmul_add_relu(
                    lhs.primitive.clone(),
                    rhs.primitive.clone(),
                    bias.primitive,
                );

                let state = (lhs_state, rhs_state, output.clone(), bias_shape);

                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                // 当没有节点被跟踪时，我们可以只计算原始操作而
                // 不保留任何状态。
                let output = B::fused_matmul_add_relu(lhs.primitive, rhs.primitive, bias.primitive);
                prep.finish(output)
            }
        }
    }
}
```

前面的代码是自文档化的，以使其更清晰，但以下是它的摘要：

我们在`Autodiff<B>`中定义`fused_matmul_add_relu`，允许任何自动微分装饰的后端从我们的实现中受益。在自动微分装饰的后端中，前向传递仍然需要实现。这是通过一个全面的匹配语句块实现的，其中计算被委托给内部后端，同时跟踪状态。状态包括对反向传递相关的任何信息，如输入和输出张量以及偏置形状。当操作未被跟踪时（意味着在此特定操作的图中不会有反向传递），存储状态变得不必要，我们只需执行前向计算。

反向传递使用从计算图中前一个节点获得的梯度。它计算`relu`（`relu_backward`）、加法（这里不需要操作，因为导数为一）和`matmul`（另一个带有转置输入的`matmul`）的导数。这导致输入张量和偏置的梯度，这些梯度被注册供后续操作节点使用。

唯一剩下的部分是为我们的JIT后端实现自动微分装饰的后端trait。

```rust, ignore
impl<R: CubeRuntime, F: FloatElement, I: IntElement> AutodiffBackend
    for Autodiff<CubeBackend<R, F, I>>
{
}
```

## 结论

在本指南中，我们使用`cubecl`编译器前端实现了一个融合内核，使其能够在任何GPU和任何`cubecl`后端上执行。通过深入了解JIT后端和自动微分后端的内部工作原理，我们对这些系统有了更深入的理解。

虽然扩展后端可能比使用简单张量更困难，但好处可能是值得的。这种方法使我们能够创建具有更大执行控制权的自定义模型，这可能会大大提高模型的性能。

在结束本指南时，我们希望您对Burn的后端扩展世界有了深入了解，并且它将帮助您释放项目的全部潜力。