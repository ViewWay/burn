# 自定义WGPU内核

在本节中，您将学习如何通过使用WGPU后端编写自己的内核来创建自己的自定义操作。我们将以深度学习领域中的常见工作流程为例，在其中我们创建一个内核来融合多个操作。请注意，`burn`会自动执行此操作，但在某些情况下手动实现可能更高效。我们将融合一个矩阵乘法内核，后跟加法和ReLU激活函数，这在各种模型中都很常见。所有代码都可以在[示例目录](https://github.com/tracel-ai/burn/tree/main/examples/custom-wgpu-kernel)中找到。

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

现在，让我们使用WGSL着色语言编写融合内核。为了简单起见，我们将创建一个简单的矩阵乘法内核，而不使用任何复杂的技术。虽然我们不会深入探讨WGSL语法的细节，因为它超出了本指南的范围，但我们仍然为好奇的读者提供以下实现。实际的矩阵乘法、加法和relu计算在 extensive overhead 之后找到，该 overhead 的作用是正确地将每个计算单元映射到它负责的数据，并支持批次。

```wgsl, ignore
@group(0)
@binding(0)
var<storage, read_write> lhs: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> rhs: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read_write> bias: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(4)
var<storage, read_write> info: array<u32>;

```

const BLOCK_SIZE = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // 索引
    let row = workgroup_id.x * BLOCK_SIZE + (local_idx / BLOCK_SIZE);
    let col = workgroup_id.y * BLOCK_SIZE + (local_idx % BLOCK_SIZE);
    let batch = global_id.z;

    // 基本信息
    let dim = info[0];
    let n_rows = info[6u * dim - 1u];
    let n_cols = info[6u * dim];
    let K = info[5u * dim - 1u];

    // 如果超出输出维度则返回
    if row >= n_rows || col >= n_cols {
        return;
    }

    // 计算相应的偏移量，支持广播。
    let offset_output = batch * n_rows * n_cols;
    var offset_lhs: u32 = 0u;
    var offset_rhs: u32 = 0u;

    let batch_dims = dim - 2u;
    for (var b: u32 = 1u; b <= batch_dims; b++) {
        let stride_lhs = info[b];
        let stride_rhs = info[b + dim];
        let stride_output = info[b + 2u * dim];
        let shape_lhs = info[b + 3u * dim];
        let shape_rhs = info[b + 4u * dim];

        offset_lhs += offset_output / stride_output % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_output % shape_rhs * stride_rhs;
    }

    // 基本矩阵乘法实现
    var sum = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        let lhs_index = row * K + k;
        let rhs_index = k * n_cols + col;

        sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
    }

    let output_index = row * n_cols + col;
    let index = offset_output + output_index;

    // 加法和ReLU
    output[index] = max(sum + bias[index], 0.0);
}

现在，让我们进入下一步，实现剩余代码来启动内核。初始部分涉及加载模板并用适当的变量填充它。`register(name, value)`方法只是在上述WGSL代码中将`{{ name }}`的出现替换为其他字符串，然后进行编译。为了使用模板工具，您需要在`cargo.toml`中激活Burn的`template`特性。

```rust, ignore
// 源自用WGSL编写的内核。
kernel_wgsl!(FusedMatmulAddReluRaw, "./kernel.wgsl");

// 使用立方体信息定义我们的内核类型。
#[derive(new, Debug)]
struct FusedMatmulAddRelu<E: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<E>,
}

// 为我们的内核类型实现动态内核trait。
impl<E: FloatElement> KernelSource for FusedMatmulAddRelu<E> {
    fn source(&self) -> SourceTemplate {
        // 使用`SourceTemplate` trait将我们的原始内核与立方体大小信息扩展。
        FusedMatmulAddReluRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>().info(self.cube_dim)
    }
}
```

随后，我们将深入实现WGPU后端的自定义后端trait。请注意，在本教程中我们不会涉及支持`fusion`特性标志，因此我们为原始`WgpuBackend`类型实现trait。

```rust, ignore
/// 为现有的后端`WgpuBackend`实现我们的自定义后端trait。
impl<F: FloatElement, I: IntElement> Backend for CubeBackend<WgpuRuntime, F, I> {
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
        let output = CubeTensor::new_contiguous(
            lhs.client.clone(),
            lhs.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // 创建内核。
        let kernel = FusedMatmulAddRelu::<F>::new(cube_dim);

        // 构建内核所需的信息缓冲区，如形状和步幅。
        let info = build_info::<_, F>(&[&lhs, &rhs, &output]);
        let info_handle = lhs.client.create(bytemuck::cast_slice(&info));

        // 使用x、y和z中的立方体数量声明wgsl工作组。
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // 使用启动信息和给定缓冲区懒加载执行内核。
        lhs.client.execute(
            Box::new(SourceKernel::new(kernel, cube_dim)),
            cube_count,
            vec![
                lhs.handle.binding(),
                rhs.handle.binding(),
                bias.handle.binding(),
                output.handle.clone().binding(),
                info_handle.binding(),
            ],
        );

        // 返回输出张量。
        output
    }
}
```

在前面的代码块中，我们演示了如何启动修改正确缓冲区的内核。需要注意的是，Rust的可变性安全性在这里不适用；上下文有能力对任何缓冲区执行任何可变操作。虽然在之前我们只修改新创建的输出缓冲区的场景中这不是问题，但明智的做法是记住这一点。

## 反向

现在自定义后端trait已为WGPU后端实现，您可以使用它来调用`matmul_add_relu_custom`函数。然而，在这个阶段还不能计算梯度。如果您的用例不超过推理，那么不需要实现以下任何代码。

对于反向传递，我们将利用来自`burn-autodiff`的后端实现，它实际上对后端是泛型的。我们不会为反向传递制作自己的WGSL内核，而只会将我们的融合内核用于前向传递，并使用基本操作计算梯度。

```rust, ignore
// 为任何也实现我们自定义后端trait的后端实现我们的自定义后端trait。
//
// 请注意，我们可以只为Wgpu后端实现后端trait，而不是为任何也实现我们自己API的后端实现。
// 这将允许我们只调用仅为Wgpu实现的函数，并可能调用仅为该任务制作的自定义内核。
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
                // 由于我们需要父级的输出，我们必须检查点它们的ID以在反向开始时检索它们的节点
                // 输出。我们还可以保存辅助数据，如偏置形状
                // 如果我们还需要此操作的输出，我们可以将其保存在状态中或在反向传递期间重新计算
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

前面的代码是自文档化的，以使其更清晰，但以下是它的摘要。

我们在`Autodiff<B>`中定义`fused_matmul_add_relu`，允许任何自动微分装饰的后端从我们的实现中受益。在自动微分装饰的后端中，前向传递仍然需要实现。这是通过一个全面的匹配语句块实现的，其中计算被委托给内部后端，同时跟踪状态。状态包括对反向传递相关的任何信息，如输入和输出张量，以及偏置形状。当操作未被跟踪时（意味着在此特定操作的图中不会有反向传递），存储状态变得不必要，我们只需执行前向计算。

反向传递使用从计算图中前一个节点获得的梯度。它计算`relu`（`relu_backward`）、加法（这里不需要操作，因为导数为一）和`matmul`（另一个带有转置输入的`matmul`）的导数。这导致输入张量和偏置的梯度，这些梯度被注册供后续操作节点使用。

唯一剩下的部分是为我们的WGPU后端实现自动微分装饰的后端trait。

```rust, ignore
impl<G: GraphicsApi, F: FloatElement, I: IntElement> AutodiffBackend for Autodiff<WgpuBackend<G, F, I>>
{
}
```

## 结论

在本指南中，我们使用WGPU后端实现了一个融合内核，使其能够在任何GPU上执行。通过深入了解WGPU后端和自动微分后端的内部工作原理，我们对这些系统有了更深入的理解。

虽然扩展后端可能比使用简单张量更困难，但好处可能是值得的。这种方法使我们能够创建具有更大执行控制权的自定义模型，这可能会大大提高模型的性能。

在结束本指南时，我们希望您对Burn的后端扩展世界有了深入了解，并且它将帮助您释放项目的全部潜力。