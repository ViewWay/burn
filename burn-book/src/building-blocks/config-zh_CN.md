# 配置

在编写科学代码时，通常会有很多设置的值，深度学习也不例外。Python有可能为函数定义默认参数，这有助于改善开发者体验。然而，这有一个缺点，即在升级到新版本时可能会破坏您的代码，因为默认值可能会在您不知情的情况下改变，使调试变得非常困难。

考虑到这一点，我们提出了配置系统。这是一个简单的Rust derive，您可以应用到您的类型上，让您能够轻松定义默认值。此外，所有配置都可以序列化，减少在升级版本时出现潜在错误的可能性，并提高可重现性。

```rust , ignore
use burn::config::Config;

#[derive(Config)]
pub struct MyModuleConfig {
    d_model: usize,
    d_ff: usize,
    #[config(default = 0.1)]
    dropout: f64,
}
```

该derive还为配置的每个属性添加了有用的`with_`方法，类似于构建器模式，以及一个`save`方法。

```rust, ignore
fn main() {
    let config = MyModuleConfig::new(512, 2048);
    println!("{}", config.d_model); // 512
    println!("{}", config.d_ff); // 2048
    println!("{}", config.dropout); // 0.1
    let config =  config.with_dropout(0.2);
    println!("{}", config.dropout); // 0.2

    config.save("config.json").unwrap();
}
```

## 良好实践

通过使用配置类型，可以轻松创建新的模块实例。初始化方法应该在配置类型上实现，并以设备作为参数。

```rust, ignore
impl MyModuleConfig {
    /// 在给定设备上创建模块。
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyModule {
        MyModule {
            linear: LinearConfig::new(self.d_model, self.d_ff).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
```

然后我们可以在这段代码中添加以下行：

```rust, ignore
use burn::backend::Wgpu;
let device = Default::default();
let my_module = config.init::<Wgpu>(&device);
```