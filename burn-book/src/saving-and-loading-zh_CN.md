# 模型的保存与加载

保存您训练好的机器学习模型非常简单，无论您选择哪种输出格式。正如在 [Record](./building-blocks/record.md) 部分中提到的，支持不同的格式来序列化/反序列化模型。默认情况下，我们使用 `NamedMpkFileRecorder`，它在 [rmp_serde](https://docs.rs/rmp-serde/) 的帮助下使用 [MessagePack](https://msgpack.org/) 二进制序列化格式。

```rust, ignore
// 以完整精度保存模型为 MessagePack 格式
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model
    .save_file(model_path, &recorder)
    .expect("应该能够保存模型");
```

请注意，文件扩展名由记录器根据您选择的格式自动处理。因此，只需提供文件路径和基本名称。

现在您已经将训练好的模型保存到磁盘上，可以以类似的方式轻松加载它。

```rust, ignore
// 从 MessagePack 文件以完整精度加载模型
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model = model
    .load_file(model_path, &recorder, device)
    .expect("应该能够从提供的文件加载模型权重");
```

**注意：** 模型可以保存为不同的输出格式，只需确保在加载保存的模型时使用正确的记录器类型。不同精度设置之间的类型转换会自动处理，但格式不可互换。可以从一种格式加载模型并保存为另一种格式，只要之后使用新的记录器类型加载即可。

## 从记录权重初始化

加载模块权重的最直接方法是使用生成的方法 [load_record](https://burn.dev/docs/burn/module/trait.Module.html#tymethod.load_record)。请注意，参数初始化是惰性的，因此在使用模块之前不会实际执行张量分配和 GPU/CPU 内核。这意味着您可以使用 `init(device)` 后跟 `load_record(record)` 而不会产生任何有意义的性能成本。

```rust, ignore
// 创建一个虚拟初始化的模型以保存
let device = Default::default();
let model = Model::<MyBackend>::init(&device);

// 以完整精度保存模型为 MessagePack 格式
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model
    .save_file(model_path, &recorder)
    .expect("应该能够保存模型");
```

之后，可以同样轻松地从保存在磁盘上的记录加载模型。

```rust, ignore
// 在后端的默认设备上加载模型记录
let record: ModelRecord<MyBackend> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
    .load(model_path.into(), &device)
    .expect("应该能够从提供的文件加载模型权重");

// 使用加载的记录/权重初始化新模型
let model = Model::init(&device).load_record(record);
```

## 没有存储，没有问题！

对于可能在运行时无法使用（或不需要）文件存储的应用程序，您可以使用 `BinBytesRecorder`。

在前面的示例中，我们使用了基于 MessagePack 格式的 `FileRecorder`，可以替换为您选择的 [另一种文件记录器](./building-blocks/record.md#recorder)。要将模型嵌入为运行时应用程序的一部分，首先使用 `BinFileRecorder` 将模型保存为二进制文件。

```rust, ignore
// 以完整精度保存模型为二进制格式
let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
model
    .save_file(model_path, &recorder)
    .expect("应该能够保存模型");
```

然后，在您的最终应用程序中，包含模型并使用 `BinBytesRecorder` 加载它。

将模型作为应用程序的一部分嵌入特别适用于较小的模型，但不建议用于非常大的模型，因为这会显著增加二进制文件大小，并在运行时消耗更多内存。

```rust, ignore
// 将模型文件包含为字节数组的引用
static MODEL_BYTES: &[u8] = include_bytes!("path/to/model.bin");

// 以完整精度加载模型二进制记录
let record = BinBytesRecorder::<FullPrecisionSettings>::default()
    .load(MODEL_BYTES.to_vec(), device)
    .expect("应该能够从字节加载模型权重");

// 使用记录加载模型
model.load_record(record);
```

此示例假设在加载模型记录之前已经创建了模型。如果您想跳过随机初始化并直接使用提供的记录初始化权重，可以像 [前面的示例](#initialization-from-recorded-weights) 那样进行调整。