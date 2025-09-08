# 数据集

从根本上讲，数据集是与特定分析或处理任务相关的数据集合。数据模态根据任务的不同而变化，但大多数数据集主要由图像、文本、音频或视频组成。

这个数据源是机器学习成功训练模型的重要组成部分。因此，提供一个方便且高性能的API来处理您的数据至关重要。由于这个过程在不同问题之间差异很大，它被定义为应在您的类型上实现的trait。数据集trait与PyTorch中的数据集抽象类非常相似：

```rust, ignore
pub trait Dataset<I>: Send + Sync {
    fn get(&self, index: usize) -> Option<I>;
    fn len(&self) -> usize;
}
```

数据集trait假设一个固定长度的项目集合，可以以常数时间随机访问。这与使用Apache Arrow底层来提高流性能的数据集有重大区别。Burn中的数据集不假设它们将如何被访问；它只是一个项目集合。

然而，您可以组合多个数据集转换来懒加载获得您想要的内容，而无需预处理，这样您的训练可以立即开始！

## 转换

Burn中的转换都是懒加载的，修改一个或多个输入数据集。这些转换的目标是为您提供必要的工具，以便您可以建模复杂的数据分布。

| 转换     | 描述                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `SamplerDataset`   | 从数据集中采样项目。这是一种方便的方法，可以将数据集建模为固定大小的概率分布。 |
| `SelectionDataset` | 通过索引从数据集中选择项目子集。可以随机打乱；可以重新打乱。                         |
| `ShuffledDataset`  | 打乱包装的数据集；这是`SelectionDataset`的薄包装。                                            |
| `PartialDataset`   | 返回具有指定范围的输入数据集视图。                                                              |
| `MapperDataset`    | 在输入数据集上懒加载计算转换。                                                                   |
| `ComposedDataset`  | 将多个数据集组合在一起创建一个更大的数据集，而无需复制任何数据。                                     |
| `WindowsDataset`   | 设计用于处理从输入数据集中提取的重叠数据窗口的数据集。                               |

让我们看看每种数据集转换的基本用法以及它们如何组合在一起。除特别说明外，这些转换默认都是懒加载的，减少了不必要的中间分配的需要，提高了性能。每种转换的完整文档可以在[API参考](https://burn.dev/docs/burn/data/dataset/transform/index.html)中找到。

- **SamplerDataset**：此转换可用于有（默认）或无替换地从数据集中采样项目。转换使用采样大小初始化，该大小可以大于或小于输入数据集大小。当我们要在训练期间更频繁地检查点较大数据集，较少检查点较小数据集时，这特别有用，因为现在epoch的大小由采样大小控制。使用示例：

```rust, ignore
type DbPedia = SqliteDataset<DbPediaItem>;
let dataset: DbPedia = HuggingfaceDatasetLoader::new("dbpedia_14")
        .dataset("train").
        .unwrap();

let dataset = SamplerDataset<DbPedia, DbPediaItem>::new(dataset, 10000);
```

- **SelectionDataset**：此转换可用于通过索引从数据集中选择项目子集。它可以使用要从输入数据集中选择的索引列表进行初始化。当您想要从较大数据集中创建较小数据集时，这特别有用，例如从训练集中创建验证集。

  `SelectionDataset`也可以使用随机种子初始化，以在选择前打乱索引。当您想要从数据集中随机选择项目子集时，这很有用。

  基础数据集项目可能在选择中包含多次。

```rust, ignore
let explicit = SelectionDataset::from_indices_checked(dataset.clone(), vec![0, 1, 2, 0]);

let shuffled = SelectionDataset::new_shuffled(dataset.clone(), &mut rng);
let shuffled = SelectionDataset::new_shuffled(dataset.clone(), 42);

let mut mutable = SelectionDataset::new_select_all(dataset.clone(), vec![0, 1, 2, 0]);
mutable.shuffle(42);
mutable.shuffle(&mut rng);
```

- **ShuffledDataset**：此转换可用于打乱数据集的项目。在将原始数据集拆分为训练/测试拆分之前特别有用。可以使用种子初始化以确保可重现性。

  `ShuffledDataset`是`SelectionDataset`的薄包装。

```rust, ignore
let dataset = ShuffledDataset<DbPedia, DbPediaItem>::new(dataset, &mut rng);
let dataset = ShuffledDataset<DbPedia, DbPediaItem>::new(dataset, 42);
```

- **PartialDataset**：此转换对于返回具有指定开始和结束索引的数据集视图很有用。用于创建训练/验证/测试拆分。在下面的示例中，我们展示了如何链式使用ShuffledDataset和PartialDataset来创建拆分。

```rust, ignore
// 为简洁起见在此处定义链式数据集类型
type PartialData = PartialDataset<ShuffledDataset<DbPedia, DbPediaItem>>;
let len = dataset.len();
let split = "train"; // 或 "val"/"test"

let data_split = match split {
    "train" => PartialData::new(dataset, 0, len * 8 / 10),  // 获取前80%数据集
    "test" => PartialData::new(dataset, len * 8 / 10, len), // 获取剩余20%
    _ => panic!("无效的拆分类型"),                      // 处理意外的拆分类型
};
```

- **MapperDataset**：此转换对于对数据集的每个项目应用转换很有用。在已知通道均值时对图像数据进行归一化时特别有用。

- **ComposedDataset**：此转换对于将从多个源（比如不同的HuggingfaceDatasetLoader源）下载的多个数据集组合成一个更大的数据集很有用，可以从一个源进行采样。

- **WindowsDataset**：此转换对于创建数据集的重叠窗口很有用。对于序列时间序列数据特别有用，例如在使用LSTM时。

## 存储

您可以选择多种数据集存储选项。数据集的选择应基于数据集的大小及其预期用途。

| 存储            | 描述                                                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InMemDataset`     | 内存中的数据集，使用向量存储项目。非常适合较小的数据集。                                                               |
| `SqliteDataset`    | 使用[SQLite](https://www.sqlite.org/)索引项目的数据集，可以保存在简单的SQL数据库文件中。非常适合较大的数据集。 |
| `DataframeDataset` | 使用[Polars](https://www.pola.rs/)数据框存储和管理数据的数据集。非常适合高效的数据操作和分析。       |

## 源

目前，Burn只有少数几个可用的数据集源，但会增加更多！

### Hugging Face

您可以轻松地使用Burn导入任何Hugging Face数据集。我们使用SQLite作为存储，以避免每次下载模型或启动Python进程。您需要事先知道数据集中每个项目的格式。以下是使用[dbpedia数据集](https://huggingface.co/datasets/dbpedia_14)的示例。

```rust, ignore
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    pub title: String,
    pub content: String,
    pub label: usize,
}
```

```rust, ignore
}

fn main() {
    let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
        .dataset("train") // 训练拆分。
        .unwrap();
}
```

我们看到项目必须派生`serde::Serialize`、`serde::Deserialize`、`Clone`和`Debug`，但这些是唯一的要求。

<div class="warning">

`HuggingfaceDatasetLoader`依赖于[HuggingFace的`datasets`库](https://huggingface.co/docs/datasets/index)来下载数据集。这是一个Python库，因此您必须有现有的Python安装才能使用此加载器。

</div>

### 图像

`ImageFolderDataset`是一个通用的视觉数据集，用于从磁盘加载图像。目前可用于多类和多标签分类任务以及语义分割和目标检测任务。

```rust, ignore
// 从根文件夹创建图像分类数据集，
// 其中每个类的图像存储在各自的文件夹中。
//
// 例如：
// root/dog/dog1.png
// root/dog/dog2.png
// ...
// root/cat/cat1.png
let dataset = ImageFolderDataset::new_classification("path/to/dataset/root").unwrap();
```

```rust, ignore
// 从项目列表创建多标签图像分类数据集，
// 其中每个项目是元组`(图像路径, 标签)`，以及数据集中的类列表。
//
// 例如：
let items = vec![
    ("root/dog/dog1.png", vec!["animal".to_string(), "dog".to_string()]),
    ("root/cat/cat1.png", vec!["animal".to_string(), "cat".to_string()]),
];
let dataset = ImageFolderDataset::new_multilabel_classification_with_items(
    items,
    &["animal", "cat", "dog"],
)
.unwrap();
```

```rust, ignore
// 从项目列表创建分割掩码数据集，其中每个
// 项目是元组`(图像路径, 掩码路径)`和对应于掩码中整数值的类列表。
let items = vec![
    (
        "path/to/images/image0.png",
        "path/to/annotations/mask0.png",
    ),
    (
        "path/to/images/image1.png",
        "path/to/annotations/mask1.png",
    ),
    (
        "path/to/images/image2.png",
        "path/to/annotations/mask2.png",
    ),
];
let dataset = ImageFolderDataset::new_segmentation_with_items(
    items,
    &[
        "cat", // 0
        "dog", // 1
        "background", // 2
    ],
)
.unwrap();
```

```rust, ignore
// 从COCO数据集创建目标检测数据集。目前仅支持
// 目标检测数据（边界框）的导入。
//
// COCO为训练和验证提供了单独的注释和图像档案，
// 需要将解压文件的路径作为参数传递：

let dataset = ImageFolderDataset::new_coco_detection(
    "/path/to/coco/instances_train2017.json",
    "/path/to/coco/images/train2017"
)
.unwrap();

```

### 逗号分隔值(CSV)

从内存中的简单CSV文件加载记录很简单，使用`InMemDataset`：

```rust, ignore
// 从制表符('\t')分隔符的csv构建数据集。
// 读取器可以为您的特定文件进行配置。
let mut rdr = csv::ReaderBuilder::new();
let rdr = rdr.delimiter(b'\t');

let dataset = InMemDataset::from_csv("path/to/csv", rdr).unwrap();
```

请注意，这需要`csv` crate。

**流式数据集怎么样？**

Burn没有流式数据集API，这是设计使然！学习器结构体会多次迭代数据集，只在完成时进行检查点。您可以将数据集的长度视为在执行检查点和运行验证之前的迭代次数。没有什么能阻止您在多次使用相同`index`调用时返回不同的项目。

## 数据集如何使用？

在训练期间，数据集用于访问数据样本，对于监督学习中的大多数用例，还包括其对应的地面实况标签。请记住，`Dataset` trait实现负责从其源检索数据，通常是某种数据存储。此时，可以天真地迭代数据集为模型提供单个样本进行处理，但这效率不高。

相反，我们收集多个样本，模型可以将它们作为_批次_处理，以充分利用现代硬件（例如，具有令人印象深刻的并行处理能力的GPU）。由于数据集中的每个数据样本都可以独立收集，数据加载通常并行进行以进一步加快速度。在这种情况下，我们使用多线程`BatchDataLoader`并行化数据加载，以从`Dataset`实现中获取项目序列。最后，项目序列通过`Batcher` trait实现组合成批处理张量，可以用作模型的输入。在此步骤中可以执行其他张量操作来准备批处理数据，就像在[基本工作流指南](../basic-workflow/data.md)中所做的那样。下图说明了MNIST数据集的过程。

<img title="Burn数据加载管道" alt="Burn数据加载管道" src="./dataset.png">

虽然我们已经方便地实现了指南中使用的[`MnistDataset`](https://github.com/tracel-ai/burn/blob/main/crates/burn-dataset/src/vision/mnist.rs)，但我们将回顾其实现以演示如何使用`Dataset`和`Batcher` trait。

手写数字的[MNIST数据集](http://yann.lecun.com/exdb/mnist/)有60,000个训练样本和10,000个测试样本。数据集中的单个项目由一个\\(28 \times 28\\)像素的黑白图像（存储为原始字节）及其对应的标签（\\(0\\)到\\(9\\)之间的数字）表示。这由`MnistItemRaw`结构体定义。

```rust, ignore
# #[derive(Deserialize, Debug, Clone)]
struct MnistItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}
```

对于这种低分辨率的单通道图像，整个训练集和测试集可以一次性加载到内存中。因此，我们利用现有的`InMemDataset`来检索原始图像和标签数据。此时，图像数据仍然只是一堆字节，但我们想要以预期形式检索_结构化_图像数据。为此，我们可以定义一个`MapperDataset`，将原始图像字节转换为二维数组图像（在此过程中我们将其转换为浮点数）。

```rust, ignore
const WIDTH: usize = 28;
const HEIGHT: usize = 28;

# /// MNIST项目。
# #[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MnistItem {
    /// 作为二维浮点数组的图像。
    pub image: [[f32; WIDTH]; HEIGHT],

    /// 图像的标签。
    pub label: u8,
}

struct BytesToImage;

impl Mapper<MnistItemRaw, MnistItem> for BytesToImage {
    /// 将原始MNIST项目（图像字节）转换为MNIST项目（二维数组图像）。
    fn map(&self, item: &MnistItemRaw) -> MnistItem {
        // 确保图像尺寸正确。
        debug_assert_eq!(item.image_bytes.len(), WIDTH * HEIGHT);

        // 将图像转换为二维浮点数组。
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in item.image_bytes.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            image_array[y][x] = *pixel as f32;
        }

        MnistItem {
            image: image_array,
            label: item.label,
        }
    }
}

type MappedDataset = MapperDataset<InMemDataset<MnistItemRaw>, BytesToImage, MnistItemRaw>;

# /// MNIST数据集包含70,000个28x28的黑白图像，分为10个类别（每个数字一个类别），每类有7,000个图像。有60,000个训练图像和10,000个测试图像。
# ///
# /// 数据从[CVDF镜像](https://github.com/cvdfoundation/mnist)从网络下载。
pub struct MnistDataset {
    dataset: MappedDataset,
}
```

要构造`MnistDataset`，必须将数据源解析为预期的`MappedDataset`类型。由于训练集和测试集使用相同的文件格式，我们可以将功能分离为加载`train()`和`test()`数据集。

```rust, ignore

impl MnistDataset {
    /// 创建新的训练数据集。
    pub fn train() -> Self {
        Self::new("train")
    }

    /// 创建新的测试数据集。
    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        // 下载数据集
        let root = MnistDataset::download(split);

        // 将数据解析为图像字节向量和标签向量
        let images: Vec<Vec<u8>> = MnistDataset::read_images(&root, split);
        let labels: Vec<u8> = MnistDataset::read_labels(&root, split);

        // 收集为MnistItemRaw向量
        let items: Vec<_> = images
            .into_iter()
            .zip(labels)
            .map(|(image_bytes, label)| MnistItemRaw { image_bytes, label })
            .collect();

        // 为InMemDataset<MnistItemRaw>创建MapperDataset以转换
        // 项目（MnistItemRaw -> MnistItem）
        let dataset = InMemDataset::new(items);
        let dataset = MapperDataset::new(dataset, BytesToImage);

        Self { dataset }
    }

#    /// 从网络下载MNIST数据集文件。
#    /// 如果无法完成下载或无法将文件内容写入磁盘，则会panic。
#    fn download(split: &str) -> PathBuf {
#        // 数据集文件存储在burn-dataset缓存目录中
#        let cache_dir = dirs::home_dir()
#            .expect("无法获取主目录")
#            .join(".cache")
#            .join("burn-dataset");
#        let split_dir = cache_dir.join("mnist").join(split);
#
#        if !split_dir.exists() {
#            create_dir_all(&split_dir).expect("无法创建基础目录");
#        }
#
#        // 下载拆分文件
#        match split {
#            "train" => {
#                MnistDataset::download_file(TRAIN_IMAGES, &split_dir);
#                MnistDataset::download_file(TRAIN_LABELS, &split_dir);
#            }
#            "test" => {
#                MnistDataset::download_file(TEST_IMAGES, &split_dir);
#                MnistDataset::download_file(TEST_LABELS, &split_dir);
#            }
#            _ => panic!("指定了无效的拆分 {}", split),
#        };
#
#        split_dir
#    }
#
#    /// 从MNIST数据集URL下载文件到目标目录。
#    /// 使用[进度条](indicatif)报告文件下载进度。
#    fn download_file<P: AsRef<Path>>(name: &str, dest_dir: &P) -> PathBuf {
#        // 输出文件名
#        let file_name = dest_dir.as_ref().join(name);
#
#        if !file_name.exists() {
#            // 下载gzip文件
#            let bytes = download_file_as_bytes(&format!("{URL}{name}.gz"), name);
#
#            // 创建文件以写入下载的内容
#            let mut output_file = File::create(&file_name).unwrap();
#
#            // 解码gzip文件内容并写入磁盘
#            let mut gz_buffer = GzDecoder::new(&bytes[..]);
#            std::io::copy(&mut gz_buffer, &mut output_file).unwrap();
#        }
#
#        file_name
#    }
#
#    /// 从提供的路径读取指定拆分的图像。
#    /// 每个图像都是一个字节向量。
#    fn read_images<P: AsRef<Path>>(root: &P, split: &str) -> Vec<Vec<u8>> {
#        let file_name = if split == "train" {
#            TRAIN_IMAGES
#        } else {
#            TEST_IMAGES
#        };
#        let file_name = root.as_ref().join(file_name);
#
#        // 从16字节头部元数据读取图像数量
#        let mut f = File::open(file_name).unwrap();
#        let mut buf = [0u8; 4];
#        let _ = f.seek(SeekFrom::Start(4)).unwrap();
#        f.read_exact(&mut buf)
#            .expect("应该能够读取图像文件头部");
#        let size = u32::from_be_bytes(buf);
#
#        let mut buf_images: Vec<u8> = vec![0u8; WIDTH * HEIGHT * (size as usize)];
#        let _ = f.seek(SeekFrom::Start(16)).unwrap();
#        f.read_exact(&mut buf_images)
#            .expect("应该能够读取图像文件头部");
#
#        buf_images
#            .chunks(WIDTH * HEIGHT)
#            .map(|chunk| chunk.to_vec())
#            .collect()
#    }
#
#    /// 从提供的路径读取指定拆分的标签。
#    fn read_labels<P: AsRef<Path>>(root: &P, split: &str) -> Vec<u8> {
#        let file_name = if split == "train" {
#            TRAIN_LABELS
#        } else {
#            TEST_LABELS
#        };
#        let file_name = root.as_ref().join(file_name);
#
#        // 从8字节头部元数据读取标签数量
#        let mut f = File::open(file_name).unwrap();
#        let mut buf = [0u8; 4];
#        let _ = f.seek(SeekFrom::Start(4)).unwrap();
#        f.read_exact(&mut buf)
#            .expect("应该能够读取标签文件头部");
#        let size = u32::from_be_bytes(buf);
#
#        let mut buf_labels: Vec<u8> = vec![0u8; size as usize];
#        let _ = f.seek(SeekFrom::Start(8)).unwrap();
#        f.read_exact(&mut buf_labels)
#            .expect("应该能够从文件读取标签");
#
#        buf_labels
#    }
}
```

由于`MnistDataset`只是包装了带有`InMemDataset`的`MapperDataset`实例，我们可以轻松实现`Dataset` trait。

```rust, ignore
impl Dataset<MnistItem> for MnistDataset {
    fn get(&self, index: usize) -> Option<MnistItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
```

现在唯一缺少的是`Batcher`，我们已经在[基本工作流指南](../basic-workflow/data.md)中讨论过了。`Batcher`以数据加载器检索的`MnistItem`列表作为输入，并返回一批图像作为3D张量及其目标。