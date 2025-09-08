# 导入模型

Burn支持从其他框架和文件格式导入模型，使您能够在Burn应用程序中使用预训练权重。

## 支持的格式

Burn目前支持三种主要的模型导入格式：

| 格式 | 描述 | 用例 |
|--------|-------------|----------|
| [**ONNX**](./onnx-model.md) | 开放神经网络交换格式 | 直接从任何支持ONNX导出的框架导入完整的模型架构和权重 |
| [**PyTorch**](./pytorch-model.md) | PyTorch权重(.pt, .pth) | 将PyTorch模型的权重加载到匹配的Burn架构中 |
| [**Safetensors**](./safetensors-model.md) | Hugging Face的模型序列化格式 | 将模型的张量权重加载到匹配的Burn架构中 |