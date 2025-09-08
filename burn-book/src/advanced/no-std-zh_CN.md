# 无标准库

在本节中，您将学习如何在嵌入式系统上运行ONNX推理模型，在没有标准库支持的Raspberry Pi Pico上运行。这应该普遍适用于其他平台。所有代码都可以在[示例目录](https://github.com/tracel-ai/burn/tree/main/examples/raspberry-pi-pico)中找到。

## 分步指南

让我们逐步了解运行嵌入式ONNX模型的过程：

### 设置
按照[embassy指南](https://embassy.dev/book/#_getting_started)为您的特定环境进行设置。设置完成后，您应该有类似以下的内容。
```
./inference
├── Cargo.lock
├── Cargo.toml
├── build.rs
├── memory.x
└── src
    └── main.rs
```

需要添加一些其他依赖项
```toml
[dependencies]
embedded-alloc = "0.6.0" # 仅当您的芯片没有默认分配器时
burn = { version = "0.19", default-features = false, features = ["ndarray"] } # 后端必须是ndarray

[build-dependencies]
burn-import = { version = "0.19" } # 用于自动生成导入模型的rust代码
```

### 导入模型
按照[导入模型](../import/README.md)的说明操作。

使用以下ModelGen配置
```rs
ModelGen::new()
    .input(my_model)
    .out_dir("model/")
    .record_type(RecordType::Bincode)
    .embed_states(true)
    .run_from_script();
```

### 全局分配器
首先定义一个全局分配器（如果您在没有alloc的no_std系统上）。

```rs
use embedded_alloc::LlffHeap as Heap;

#[global_allocator]
```

```rs
#[global_allocator]
static HEAP: Heap = Heap::empty();

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
	{
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 100 * 1024; // 这取决于模型在内存中的大小。
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(&raw mut HEAP_MEM as usize, HEAP_SIZE) }
    }
}
```

### 定义后端
我们使用ndarray，所以只需像往常一样定义NdArray后端
```rs
use burn::{backend::NdArray, tensor::Tensor};

type Backend = NdArray<f32>;
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;
```

然后在`main`函数内添加
```rs
use your_model::Model;

// 获取后端的默认设备
let device = BackendDevice::default();

// 创建新模型并加载状态
let model: Model<Backend> = Model::default();
```

### 运行模型
要运行模型，只需像平常一样调用它
```rs
// 定义张量
let input = Tensor::<Backend, 2>::from_floats([[input]], &device);

// 在输入上运行模型
let output = model.forward(input);
```

## 结论
在no_std环境中运行模型与在正常环境中运行几乎相同。所需要的只是一个全局分配器。