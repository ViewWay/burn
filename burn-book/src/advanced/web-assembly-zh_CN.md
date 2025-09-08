# WebAssembly

Burn支持使用`NdArray`和`WebGpu`后端执行WebAssembly(WASM)，允许模型直接在浏览器中运行。

查看以下示例：

- [图像分类Web](https://github.com/tracel-ai/burn/tree/main/examples/image-classification-web)
- [Web上的MNIST推理](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web)

当以WebAssembly为目标时，某些依赖项需要额外配置。特别是，使用`WebGpu`时，`getrandom` crate需要显式设置。

按照[推荐用法](https://github.com/rust-random/getrandom/#webassembly-support)，确保为您的项目显式添加带有`wasm_js`特性标志的依赖项。

```toml
[dependencies]
getrandom = { version = "0.3.2", default-features = false, features = [
    "wasm_js",
] }
```

您还需要通过rust-flags相应地设置`getrandom_backend`。该标志可以通过在`.cargo/config.toml`中指定`rustflags`字段来设置

```toml
[target.wasm32-unknown-unknown]
rustflags = ['--cfg', 'getrandom_backend="wasm_js"']
```

或者使用`RUSTFLAGS`环境变量：

```
RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
```

按照`getrandom`的建议，这个更改现在在最新版本的Burn中是显式必需的。这避免了不以Web为目标的WASM开发人员可能出现的问题。