# 2 使用预训练 LLM 生成文本

本章涵盖以下内容
- 搭建用于处理 LLM 的代码环境
- 使用 tokenizer 准备输入文本
- 使用预训练 LLM 生成文本的逐步过程
- 通过缓存和编译技术加速 LLM 文本生成

在上一章中，我们讨论了传统大语言模型（LLM）与推理模型之间的区别。此外，我们还介绍了几种提升 LLM 推理能力的技术。这些推理技术通常应用于传统（基础）LLM 之上。

在本章中，我们将通过加载一个预训练的基础模型来为后续章节打下基础，如图 2.1 所示。此前我们讨论了推理方法通常是在常规的后训练阶段之后添加的。然而，从基础模型开始更容易看出哪些能力来自推理方法本身。换句话说，本章中的传统 LLM 是一个非推理 LLM，更具体地说，它是一个预训练的基础模型，而不是经过指令微调或偏好微调的助手模型。

---

**图 2.1** 一个心智模型，描绘了开发推理模型的四个主要阶段。本章聚焦于阶段 1：加载一个传统 LLM 并实现文本生成功能。

---

除了搭建编码环境和加载预训练 LLM 之外，你还将学习如何使用 tokenizer 为模型准备文本输入。如图 2.1 所示，你还将实现一个文本生成函数，使 LLM 能够实际用于生成文本。该功能将在后续章节中被使用并进一步改进。

## 2.1 LLM 文本生成简介

在本章中，我们将实现所有必要的 LLM 基础知识，从搭建编码环境、加载预训练 LLM 到生成文本，这些知识将在本书中复用和扩展。从这个意义上说，本章可以被理解为一个**搭建章节**。

尽管该 LLM 是一个仅经过预训练的基础模型，但它已经能够生成连贯的文本，并在某些情况下遵循基本指令，如图 2.2 所示。

---

**图 2.2** 一个概览图，展示了 LLM 在给定用户查询（输入文本）时生成回复（输出文本）的过程。

---

图 2.2 总结了 LLM 文本生成流水线的组成部分，我们将在本章稍后更详细地讨论和实现这些步骤。

> **注意** 按照惯例，涉及神经网络（如 LLM）的示意图通常按垂直方向从下到上（从输入到输出）绘制和阅读。箭头表示信息向上流经模型。

如果你之前没有编写过 LLM 或没有以编程方式使用过 LLM，本章将教会你文本生成过程是如何工作的。在本章中，我们不会深入探讨 LLM 的内部结构，例如注意力机制和其他架构组件；这是我的另一本书《Build a Large Language Model (from Scratch)》的主题。请注意，理解这些内部结构并不是阅读本书所必需的，而且如果你感兴趣，可以在读完本书之后再学习它们。

即使你之前已经读过《Build a Large Language Model (From Scratch)》这本书，本章也会有所帮助，因为它新增了关于加速推理和其他实际优化方面的内容。

在我们开始实现图 2.2 中所示的组件（包括输入准备、加载 LLM 和生成文本）之前，我们首先需要搭建编码环境。这是下一节的重点。

## 2.2 搭建编码环境

本节描述了两种主要的搭建 Python 编码环境以跟随本书示例的方法：一种简单的基于 pip 的搭建方式，以及一种基于 uv 的工作流。我建议在决定哪种方式最适合你之前完整阅读本节。

如果你正在阅读本书，你可能之前已经用 Python 编写过代码。在这种情况下，如果你已经设置好了 Python 环境（Python 3.10 或更新版本），安装依赖项最简单的方法是使用 Python 的包安装器（`pip`）在终端中操作。

如果你已经从出版商网站下载了代码，可以使用 `requirements.txt` 文件来安装本书所需的 Python 库：

```bash
pip install -r requirements.txt
```

或者，如果你想直接安装所需的包而不下载 `requirements.txt` 文件，可以使用：

```bash
pip install -r https://raw.githubusercontent.com/\
rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt
```

> **本章使用的 Python 包**
> 如果你只想安装本章中使用的包，可以从以下命令开始：
> ```bash
> pip install torch>=2.10.0 tokenizers>=0.22.2 reasoning-from-scratch
> ```
> 然而，PyTorch 的安装可能因平台和硬件而异，特别是如果你想要 GPU 加速的话。如果此命令没有为你的系统安装正确的版本，我建议首先使用官方 PyTorch 网站（https://pytorch.org/get-started/locally/）上的安装选择器，然后再单独安装剩余的包。
> - `torch` 指的是 PyTorch，一个广泛使用的深度学习库，提供构建和训练神经网络的工具。
> - `tokenizers` 是一个提供高效分词算法的库，用于为 LLM 准备输入数据。
> - `reasoning-from-scratch` 是我为本书开发的一个自定义库。它包含各章节中实现的所有代码示例，以及我们将要使用的额外工具函数。

虽然 `pip` 是安装 Python 包的标准方式，但我个人更推荐通过广泛推荐的 uv Python 包和项目管理器来使用 Python。像许多人一样，我现在推荐 uv，因为它明显更快，并且比 pip 更可靠地处理依赖解析。它还会自动创建隔离环境，并自带自己的 Python 可执行文件（但如果已安装兼容版本，则会使用系统 Python），因此如果你尚未在系统上安装 Python，它也是一个很好的选择。

图 2.3 概述了从安装 uv 到准备好执行本章代码的 4 步流程，我们将在本节的剩余部分介绍这些内容。

---

**图 2.3** 通过 macOS 终端安装和使用 uv Python 包和项目管理器。

---

请注意，图 2.3 展示了在 macOS 终端上的 uv 安装和使用步骤，但 uv 也支持 Linux 和 Windows。

1) 要安装 uv，请从官方网站运行适用于你操作系统的安装程序：https://docs.astral.sh/uv/getting-started/installation/

2) 接下来，克隆 GitHub 仓库：

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
```

这里，`--depth 1` 选项告诉 git 执行浅克隆，即只下载代码的最新版本而不下载完整的版本历史。这使得下载更快，占用的空间更少。

如果你没有安装 git，也可以手动从出版商网站下载源代码仓库，或者在浏览器中打开此链接：https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip（下载后解压）。

3) 接下来，在终端中导航到 `reasoning-from-scratch` 文件夹。

4) 在 `reasoning-from-scratch` 文件夹内，执行：

```bash
uv run jupyter lab
```

上面的命令将启动 JupyterLab，你可以在其中打开一个空白的 Jupyter 笔记本来输入和执行代码，或者打开包含本章所有代码的第 2 章笔记本。（你不需要先创建或激活虚拟环境。）

> **提示** Python 脚本文件可以通过 `uv run script-name.py` 执行。

上面的 `uv run...` 命令还会设置一个本地虚拟环境（通常在隐藏的 `.venv/` 文件夹内），并自动安装 `reasoning-from-scratch` 文件夹内 `pyproject.toml` 文件中的所有依赖项。因此，不需要通过 requirements 文件手动安装代码依赖项。但是，如果你计划安装额外的包，可以使用以下命令：

```bash
uv add packagename
```

补充代码仓库在 `ch02` 子文件夹中包含了额外的安装说明和详细信息，如有需要可以查阅。

## 2.3 了解硬件需求和建议

你可能听说过训练 LLM 非常昂贵。对于领先的 LLM 公司来说，训练一个新的基础模型 LLM 的计算成本，在低端花费 100 万到 1000 万美元并不罕见，而在高端则超过 5000 万美元——这甚至还没有添加任何推理技术。

例如，DeepSeek V3 模型作为 DeepSeek R1 推理系统的基础检查点，在 2,048 块 Nvidia H800 GPU 上训练了大约 11 周，估计成本为 550 万美元。（DeepSeek V3 是少数几个计算披露完全透明的较新模型之一。）

此外，根据技术报告，最终训练运行使用了 1480 万 GPU 小时的计算量。能耗大约为 620 兆瓦时，这大约相当于一个普通美国家庭约 55 年的用电量。

这种资源需求和高昂的价格标签使得我和大多数读者都无法开发 LLM。因此，我们将使用一个相对较小（但功能强大）的预训练 LLM，在其上实现推理技术。

请注意，这个较小的 LLM 是一个按比例缩小的版本，但其他方面遵循与当代最先进模型相同的架构。我们将应用的推理方法与较大 LLM 使用的方法相同。区别在于，较小的 LLM 使我们能够以经济实惠的方式探索这些方法。

打个比方，假设你对汽车的工作原理很好奇。如果你是汽车新手，作为学习练习，你可能不会一开始就立即建造一辆昂贵的法拉利。相反，你会先制造一辆像大众甲壳虫这样的小型汽车，这仍然能教会你很多关于发动机和变速箱如何工作的知识。相反，我甚至会认为，在小型汽车上工作能帮助你更好地理解发动机和变速箱的工作原理，因为它排除了复杂的改进和其他细节。

虽然我们在本书中出于教育目的使用一个相对较小的模型，但推理技术的使用、开发和应用仍然是计算密集型的，后续章节（如第 5-8 章）将受益于使用 GPU。

如果你遵循了上一节的内容，你应该已经安装了 PyTorch，你可以通过执行以下 PyTorch 代码来查看你的计算机是否有 PyTorch 支持的 GPU：

```python
import torch
print(f"PyTorch version {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA/ROCm GPU: {torch.cuda.get_device_name(0)}")
elif torch.xpu.is_available():
    print(f"Intel GPU: {torch.xpu.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("Apple Silicon GPU")
else:
    print("Only CPU")
```

根据你的机器不同，代码可能返回：

```
PyTorch version 2.10.0
Only CPU
```

如果你的机器没有 GPU 来运行代码，也不用担心。第 2-5 章可以在 CPU 上合理的时间内执行。

> **注意** 如果你想要 GPU 加速但不确定要安装哪个 PyTorch 版本，请使用官方 PyTorch 安装选择器（https://pytorch.org/get-started）为你的系统选择正确的 CPU、CUDA 或 ROCm 版本。对于本书，任何最近支持的 PyTorch CUDA 版本都可以。

根据章节不同，代码将自动使用 NVIDIA GPU（如果可用），否则在 CPU 上运行（或在特定章节或部分推荐使用 Apple Silicon GPU 时在其上运行）。我将在相应的章节和部分中提供更多信息。

像许多其他每天研究和使用 LLM 的 AI 研究人员一样，我在家里没有具备必要 GPU 硬件的机器来训练 LLM，而是使用云资源。如果你在寻找云提供商选项，我个人偏好是 Lightning AI Studio（https://lightning.ai/），因为它易于使用且功能支持完善，如图 2.4 所示。或者，Google Colab（https://colab.research.google.com/）是另一个不错的选择。

---

**图 2.4** Lightning AI GPU 云平台在 Web 浏览器中的概览。该界面支持 Python 脚本、Jupyter 笔记本、终端访问，并允许用户根据计算需求在 CPU 和各种 GPU 类型之间切换。

---

截至本文撰写时，Lightning AI 还在注册和验证流程后为用户提供免费的计算积分，可用于图 2.4 中显示的不同 GPU 选择。（如前所述，本章不需要 GPU；但是，如果你想使用 GPU，L4 GPU 对于本章来说绰绰有余。）

> **注意** 需要披露的是，我曾在 2023 年帮助构建和发布 Lightning AI 平台，但目前不再持有该公司的任何经济利益。我没有被赞助推荐它，而是自己付费使用它。我使用它是因为我发现它是最方便的选择。它支持多种类型的 GPU，允许在它们之间以及回到 CPU 轻松切换以节省成本，并允许我暂停或恢复环境而无需重新设置。

补充代码仓库在 `ch02` 子文件夹中包含了额外的 GPU 平台推荐，如有需要可以查阅。

> **使用 PyTorch**
> 在本节中，我们导入并使用了 PyTorch 库，它目前是最广泛使用的通用库。我们将在本书中使用它来运行和训练 LLM，包括我们将开发的推理方法。如果你是 PyTorch 新手，为了充分利用本书，我推荐阅读我的《PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs》教程，该教程在我的网站上免费提供：https://sebastianraschka.com/teaching/pytorch-1h。

## 2.4 为 LLM 准备输入文本

在本节中，我们将探讨如何使用 tokenizer 为 LLM 处理输入和输出文本，如图 2.5 所示，该图扩展了图 2.2 中先前显示的输入和输出准备步骤，提供了对分词流水线的更详细视图。

这在本书的其余部分中很重要，因为每个提示、推理轨迹和生成的答案最终都必须表示为 token ID，然后模型才能处理它。

---

**图 2.5** 一个简化的示意图，展示了 LLM 如何接收输入数据并生成输出。用户提供的文本通过 tokenizer 的 `encode` 方法被分词为 ID，然后由 LLM 处理以生成输出 token ID。这些 ID 再通过 tokenizer 的 `decode` 方法解码回人类可读的文本。

---

为了看看这在实践中是如何工作的，我们将从本书的 `reasoning-from-scratch` Python 包中加载一个 tokenizer，该包应该已经按照第 2.2 节中的说明安装好了。

要下载 tokenizer 文件（对应于我们将在下一节介绍的 Qwen3 0.6B 基础 LLM），请运行：

```python
from reasoning_from_scratch.qwen3 import download_qwen3_small
download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")
```

这将显示一个类似于以下的进度条：

```
tokenizer-base.json: 100% (6 MiB / 6 MiB)
```

该命令下载 `tokenizer-base.json` 文件（大小约为 6 兆字节）并将其保存在 `qwen3` 子目录中。

现在，我们可以从 tokenizer 文件中将 tokenizer 设置加载到 `Qwen3Tokenizer` 中：

```python
from pathlib import Path
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer

tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
```

由于我们尚未加载 LLM（图 2.5 中所示的核心组件），我们将首先仅使用 tokenizer 进行一个简单的试运行。具体来说，我们将进行一个分词往返测试，即我们将一段文本编码为 token ID，然后将这些 ID 解码回文本，如图 2.6 所示。

---

**图 2.6** 使用 tokenizer 进行往返分词过程的演示。用户提供的输入文本首先通过 `encode` 方法转换为 token ID，然后通过 `decode` 方法准确地重建回原始文本。

---

以下代码片段实现了图 2.6 底部所示的编码过程：

```python
prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
```

以下代码实现了图 2.6 顶部所示的解码过程，将 token ID 转换回文本：

```python
text = tokenizer.decode(input_token_ids_list)
print(text)
```

根据打印结果，我们可以看到 tokenizer 从 token ID 重建了原始输入提示：

```
'Explain large language models.'
```

在继续讨论 LLM 之前，让我们看看 `encode` 方法生成的 token ID。以下代码打印每个 token ID 及其对应的解码字符串，以帮助说明 tokenizer 的工作原理：

```python
for i in input_token_ids_list:
    print(f"{i} --> {tokenizer.decode([i])}")
```

输出如下：

```
840 --> Ex
20772 --> plain
3460 -->  large
4128 -->  language
4119 -->  models
13 --> .
```

如输出所示，原始文本被分成六个 token ID。每个 token 代表一个单词或子词，具体取决于 tokenizer 如何分割输入。

例如，"Explain" 被分成两个独立的 token，"Ex" 和 "plain"。这是因为 tokenizer 算法使用基于 Byte Pair Encoding（BPE）的子词方法。BPE 可以使用完整单词和子词单元的混合来表示常见和罕见单词。空格也经常包含在 token 中（例如，" large"），这有助于 LLM 检测单词边界。

`Qwen3Tokenizer` 的词汇表大约有 151,000 个 token，按本文撰写时的标准来看这被认为是相对较大的（作为对比，早期的 GPT-2 词汇表大小约为 50,000 个 token，Llama 3 的词汇表大小约为 128,000 个 token）。

语言模型中较大的词汇表会增加模型大小，因为嵌入层和输出层必须存储更多的 token 表示，并且它还会增加生成下一个 token 概率的每个 token 计算成本。较大的词汇表还允许更多单词被表示为单个 token，而不是被拆分为子词组件。这可以减少序列长度。例如，将像 "Explain" 这样的单词拆分为 "Ex" 和 "plain" 会产生更多的输入 token。更多的 token 会导致更长的输入序列，从而增加处理时间和资源使用。因此，权衡在于较大的词汇表具有略高的每个 token 成本，而较小的词汇表通常会产生更长的 token 序列。例如，将序列中的 token 数量加倍大致可以使运行模型的计算成本加倍，因为模型必须处理和生成更多的 token。

不幸的是，tokenizer 的详细覆盖和从零实现超出了本书的范围。感兴趣的读者可以在附录 A 的进一步资源和阅读部分找到额外的资源，包括我的从零实现。

> **练习 2.1：编码未知单词**
> 尝试使用 tokenizer 看看它如何处理未知单词。为此，发挥创意，编造一些不存在的单词。此外，如果你会说多种语言，尝试编码英语以外的单词。

## 2.5 加载预训练模型

在上一节中，我们加载并熟悉了为 LLM 准备输入数据并将 LLM 输出转换回人类可读文本表示的 tokenizer。在本节中，我们将加载 LLM 本身，如图 2.7 的概览所示。

一旦这个基础模型就位，后续章节将直接在其基础上进行评估、以不同方式提示它，并最终将其训练成更强的推理模型。

---

**图 2.7** 本书中开发推理模型的四个关键阶段概览。本节聚焦于阶段 1 中加载预训练 LLM。

---

如前所述，本书使用 Qwen3 0.6B 作为预训练基础模型。在本节中，我们加载其预训练权重，如图 2.7 所示。模型名称中的 "0.6B" 表示该模型包含大约 0.6 亿个权重参数。

为什么选择 Qwen3？在仔细评估了几个开源权重基础模型后，我选择 Qwen3 0.6B 的原因如下：
- 对于本书，我们想要一个小而强大的开源权重模型（开源权重模型是指其训练权重可公开下载的模型），能够在消费级硬件上运行。
- Qwen3 既提供基础模型（我们推理模型开发的重点），也提供官方推理变体，我们可以将其用作评估参考。

> **注意** "Qwen3" 的规范拼写不包含空格，而例如 "Llama 3" 则包含空格。

本着"从零开始"构建的精神，本书使用我用纯 PyTorch 编写的 Qwen3 自定义重新实现，它完全独立于外部 LLM 库。这种重新实现的重点在于代码的可读性和可调整性，以防你以后想修改它用于自己的实验。尽管是从头开始构建的，但该实现与原始预训练 Qwen3 模型权重完全兼容。

本书不会深入介绍 Qwen3 代码实现。仅这个主题就可以填满一整本单独的书，类似于我的另一本书《Build A Large Language Model (From Scratch)》。相反，这本《Build A Reasoning Model (From Scratch)》专门专注于在基础模型之上实现推理方法，在本例中是 Qwen3。

> **注意** 这个重新实现的 Qwen3 LLM 完全在本地运行，就像任何其他用 PyTorch 实现的神经网络一样。不涉及任何服务器端组件或外部 API 调用。所有模型使用都在你自己的机器上进行，你的数据保留在你的设备上。如果你关心隐私，我们使用的设置确保对 LLM 输入和输出的完全控制。

对于那些对 Qwen3 的额外细节以及模型代码感兴趣的人，请参阅附录 C。

在加载模型之前，我们可以指定要使用的设备，即 CPU 或 GPU。清单 2.1 中的代码将自动选择最佳可用设备：

**清单 2.1 自动获取设备**

```python
def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
```

请注意，如果你拥有现代的 NVIDIA GPU（基于 Ampere 架构或更新版本），当 `enable_tensor_cores=True` 时，`get_device()` 函数会自动启用 Tensor Core 以加快矩阵乘法运算。这可能会略微改变浮点舍入，但不会明显影响本书中的结果，而且在较新的显卡上速度优势是值得的。在非 NVIDIA 设备上，这些设置将被忽略。

使用清单 2.1 中的代码，我们可以然后获取设备如下：

```python
device = get_device()
```

虽然 GPU 通常提供实质性的速度和性能提升，但为了兼容性和调试目的，最初使用 CPU 运行本章剩余的代码可能会有所帮助。你可以通过显式设置来临时覆盖自动选择：

```python
device = torch.device("cpu")
```

在完成本章并验证代码在 CPU 上正常工作后，移除或注释掉手动覆盖并重新运行代码。如果你的系统有 GPU，你应该能观察到性能的提升。

> **注意** 本章剩余部分的代码是在配备 Apple M4 CPU 的 Mac Mini 上执行的。与 Apple Silicon M4 GPU 和 NVIDIA H100 GPU 的性能比较包含在本章末尾。

在加载模型并将其放到选定的 `device` 上之前，我们首先需要下载 Qwen3 0.6B 的权重。这些文件是正确初始化预训练模型所必需的。在上一节中，我们调用相同的函数时设置了 `tokenizer_only=True` 以仅下载 tokenizer 文件。这里，我们设置 `tokenizer_only=False`，以便同时下载模型权重：

```python
download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
```

输出如下：

```
qwen3-0.6B-base.pth: 100% (1433 MiB / 1433 MiB)
✓ qwen3/tokenizer-base.json already up-to-date
```

（tokenizer 前面有一个对勾，因为我们在上一节中已经下载过了。）

通过上一步下载模型权重后，我们现在可以将 `Qwen3Model` 类实例化，并通过 PyTorch 的 `load_state_dict` 方法将预训练权重加载到其中：

```python
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)  #A 实例化一个带有随机权重作为占位符的 Qwen3 模型
model.load_state_dict(torch.load(model_path))  #B 将预训练权重加载到模型中
model.to(device)  #C 将模型转移到指定设备（例如，"cuda"）
```

请注意，如果设备设置为 `"cpu"`，`model.to(device)` 操作将被跳过，因为模型默认已经位于 CPU 内存中。

执行上述代码后，你应该会看到以下输出（如果你不是在交互式环境（如 Jupyter 笔记本）中运行代码，则必须运行 `print(model)` 才能看到输出）：

```
Qwen3Model(
  (tok_emb): Embedding(151936, 1024)
  (trf_blocks): ModuleList(
    (0-27): 28 x TransformerBlock(
      (att): GroupedQueryAttention(
        (W_query): Linear(in_features=1024, out_features=2048, bias=False)
        (W_key): Linear(in_features=1024, out_features=1024, bias=False)
        (W_value): Linear(in_features=1024, out_features=1024, bias=False)
        (out_proj): Linear(in_features=2048, out_features=1024, bias=False)
        (q_norm): RMSNorm()
        (k_norm): RMSNorm()
      )
      (ff): FeedForward(
        (fc1): Linear(in_features=1024, out_features=3072, bias=False)
        (fc2): Linear(in_features=1024, out_features=3072, bias=False)
        (fc3): Linear(in_features=3072, out_features=1024, bias=False)
      )
      (norm1): RMSNorm()
      (norm2): RMSNorm()
    )
  )
  (final_norm): RMSNorm()
  (out_head): Linear(in_features=1024, out_features=151936, bias=False)
)
```

此输出是 PyTorch 打印的 Qwen3 0.6B 基础模型架构摘要。它突出了模型的核心组件：一个嵌入层、28 个 transformer 块的堆栈，以及一个最终的线性投影头。每个 transformer 块包括一个分组查询注意力机制和一个多层前馈网络，以及贯穿始终的归一化层。

这些组件也在图 2.8 中以可视化方式展示，供熟悉 LLM 架构的读者参考。详细理解此架构对于本书来说不是必需的。由于我们不会修改基础模型本身，而是在其之上构建推理方法，你现在可以安全地将架构视为黑盒。但是，感兴趣的读者可以在附录 C 中找到有关这些组件（如 `RMSNorm`）的更多信息。

---

**图 2.8** 前面实例化的 Qwen3 0.6B 模型架构概览。输入文本被分词并通过嵌入层，然后是 28 个重复的 transformer 块。每个块包含分组查询注意力、前馈层和 RMS 归一化。模型以最终归一化和线性输出层结束。箭头显示数据流经模型的路径。

---

本节的关键要点是，我们现在已经加载了一个预训练模型，其架构如图 2.8 所示，它应该能够生成连贯的文本。在下一节中，我们将编写一个文本生成函数，将分词后的数据输入模型并以人类可读的格式返回响应。

## 2.6 理解顺序 LLM 文本生成过程

加载预训练 LLM 后，我们的目标是编写一个利用 LLM 生成文本的函数。该函数构成了我们将在本书后面实现的推理改进方法的基础，如图 2.9 所示。

---

**图 2.9** 本书中开发推理模型的四个关键阶段概览。本节解释了 LLM 中文本生成背后的主要概念，这使我们能够在本章剩余部分实现一个用于使用预训练 LLM 的文本生成函数。

---

在我们实现这个将在本章和后续章节中使用的文本生成函数（如图 2.9 所示）之前，让我们先回顾一下 LLM 中文本生成的基本概念。

你可能已经知道，LLM 中的文本生成是一个顺序过程，LLM 一次生成一个单词（或 token）。这通常也被称为**自回归文本生成**，如图 2.10 所示。

---

**图 2.10** LLM 中顺序（自回归）文本生成的示意图。在每次迭代中，模型基于输入和先前生成的 token 生成下一个 token，这些 token 被累积地反馈回模型以产生连贯的输出。

---

请注意，图 2.10 中展示的顺序文本生成过程是一个广泛的概述。该图显示了在输入提示时每一步生成一个输出 token（顶行）。这样做是为了简化解释基于 LLM 的文本生成背后的主要概念。

> **聊天机器人界面**
> 虽然图 2.10 中的图表展示了底层的下一个 token 预测过程，但聊天界面只是将这种机制包装在一个对话循环中。模型仍然一次预测一个 token，但系统将整个对话历史反馈回模型，以便每个新的回复在轮次之间感觉具有上下文感知和连贯性。本书专注于单轮对话，其中当前答案独立于先前的答案。但是，感兴趣的读者可以在附录 G 中找到带有回答历史的聊天界面的实现。

现在，如果我们更仔细地查看其中一次迭代，LLM 会在一次前向传播中处理完整的输入序列，并为每个输入位置生成一个预测。这意味着如果我们有六个输入 token，模型会返回六个相应的预测，如图 2.11 所示。该图的排列方式使得按位置的结构易于查看，但模型输出通常不仅仅是向右移动的输入提示。相反，每个位置都会产生一个可能下一个 token 的分布。对于文本生成，我们只使用最后一个位置的预测，因为那是告诉我们接下来要附加哪个新 token 的位置。

---

**图 2.11** LLM 处理完整的输入序列，并为每个输入位置生成一个下一个 token 的预测。这些预测在概念上提前一步，因为每个位置都预测下一个 token。该图示为直观理解而按位置对齐，但输出并非字面上就是向右移动的输入提示。在生成过程中，我们只保留最终预测，并用它一次一个 token 地继续输入提示。

---

在实现一个文本生成函数，该函数使用图 2.11 中所示的概念进行每次迭代以实现图 2.10 中所示的自回归文本生成过程之前，让我们先看一个代码示例，通过重用第 2.4 节中的 "Explain large language models." 示例提示来进一步说明图 2.11：

```python
prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
print(f"Number of input tokens: {len(input_token_ids_list)}")
input_tensor = torch.tensor(input_token_ids_list)  #A 将 Python 列表转换为 PyTorch 张量
input_tensor_fmt = input_tensor.unsqueeze(0)  #B 添加一个额外的维度
input_tensor_fmt = input_tensor_fmt.to(device)
with torch.inference_mode():
    output_tensor = model(input_tensor_fmt)  #C 生成输出
output_tensor_fmt = output_tensor.squeeze(0)  #D 移除额外的维度
print(f"Formatted Output tensor shape: {output_tensor_fmt.shape}")
```

> **张量的 Squeeze 和 Unsqueeze**
> 张量是 n 维的广义矩阵。许多 PyTorch 函数和模型组件期望具有特定维度的张量，因此能够添加或移除维度对于使数据与这些操作兼容非常重要。
> PyTorch 中的 `.squeeze()` 和 `.unsqueeze()` 操作用于通过移除或添加大小为 1 的维度来改变张量的形状。这通常对于重塑张量以匹配模型的期望很有用。例如，模型可能期望具有两个维度（例如，行和列）的输入张量，以便它可以处理批量输入（参见附录 E）。但如果输入只是一个行向量，我们可以使用 `.unsqueeze(0)` 来添加一个额外的维度并使其兼容：
> ```python
> example = torch.tensor([1, 2, 3])
> print(example)
> print(example.unsqueeze(0))
> ```
> 这返回：
> ```
> tensor([1, 2, 3])
> tensor([[1, 2, 3]])
> ```
> 这里，`.unsqueeze(0)` 在位置 0 添加了一个新维度，将一个 1D 张量转换为形状为 `(1, 3)` 的 2D 张量。相反，`.squeeze(0)` 从位置 0 移除一个大小为 1 的维度：
> ```python
> example = torch.tensor([[1, 2, 3]])
> print(example)
> print(example.squeeze(0))
> ```
> 这返回：
> ```
> tensor([[1, 2, 3]])
> tensor([1, 2, 3])
> ```
> 当你想要移除不需要的额外维度时，这很有用。

前面代码示例的输出如下：

```
Number of input tokens: 6
Formatted Output tensor shape: torch.Size([6, 151936])
```

正如我们所见，我们将六个输入 token 输入模型，模型返回一个 6×151,936 维的矩阵。这个矩阵中的 6 对应于六个输入 token。第二个维度 151,936 对应于模型支持的词汇表大小。例如，六个 token 中的每一个都由一个包含 151,936 个值的向量表示。我们可以将这些向量中的值视为词汇表中每个可能单词的分数，其中最高分对应于被选为生成 token 的最可能的单词或子词（在 151,936 条目的词汇表中）。

因此，为了获得下一个生成的单词，我们提取这个 6×151,936 维矩阵的最后一行，找到该行中最大分数值对应的 token ID，并通过 tokenizer 将此 token ID 转换回文本，如图 2.12 所示。

---

**图 2.12** 更仔细地观察 LLM 在单次文本生成迭代中输出的原始分数如何转换为 token ID 及其对应的文本表示。

---

让我们看看如何将 LLM 输出矩阵转换为生成的文本 token（如图 2.12 所示）。

请注意，对于文本生成，我们在推理模式而非训练模式下运行模型，因此 PyTorch 不需要跟踪梯度。如图 2.11 所示，较早的位置对应于提示中已有 token 的预测。只有最后一个位置预测我们想要附加的下一个未见过的 token，我们可以通过 `[-1]` 索引获取它：

```python
last_token = output_tensor_fmt[-1]
print(last_token)
```

在上一步中使用 `torch.inference_mode()` 避免了存储我们在生成过程中不需要的梯度信息。这减少了内存使用，通常还能提高速度。

这会打印对应于最后一个 token 的 151,936 个值：

```
tensor([ 7.3750,  2.0312,  8.0000,  ..., -2.5469, -2.5469, -2.5469],
       dtype=torch.bfloat16)
```

`dtype=torch.bfloat16` 后缀表示这些分数以 bfloat16 格式存储，这是一种降低内存使用和提高 LLM 推理效率的常用降精度格式。根据你的硬件和设置，你可能会看到不同的 `dtype`。

然后，我们可以使用 `argmax` 函数获取此张量中最大分数值（value）的位置：

```python
print(torch.argmax(last_token, dim=-1, keepdim=True))
```

结果是：

```
tensor([20286])
```

这个返回的整数值是该向量中最大值的位置，它也对应于生成 token（`last_token`）的 token ID，我们可以通过 tokenizer 将其转换回文本：

```python
print(tokenizer.decode([20286]))
```

这会打印生成的 token：

```
Large
```

> **Max 与 Argmax**
> 简要回顾一下 max 和 argmax 的工作原理以及它们之间的区别是很有帮助的，因为我们稍后在实现文本生成函数选择下一个 token 时会使用 `torch.argmax()`。在本章中，使用 argmax 对应于**贪心解码（greedy decoding）**，即我们总是选择单个分数最高的下一个 token。我们将在本书后面讨论替代方案。
> `torch.max()` 函数返回张量中的最大值，而 `torch.argmax()` 返回该值的索引。例如：
> ```python
> example = torch.tensor([-2, 1, 3, 1])
> print(torch.max(example))
> print(torch.argmax(example))
> ```
> 这返回：
> ```
> tensor(3)
> tensor(2)
> ```
> 最大值是 3，它首次出现在索引 2 处。
> 我们还可以对 `torch.argmax()` 使用 `keepdim=True` 来通过保留被减少的维度来保持输出形状一致：
> ```python
> print(torch.argmax(example, keepdim=True))
> ```
> 这返回：
> ```
> tensor([2])
> ```
> 这里，`keepdim=True` 将结果保持为与输入具有相同维度数的 1D 张量，这对于保持 tokenizer 所需的形状以及稍后在我们文本生成函数中进行连接很有帮助。

总结一下，图 2.10 展示了 LLM 中的迭代（自回归）文本生成过程。然后，图 2.11 放大了这个过程中的一次迭代。图 2.12 则进一步放大了这一次迭代，展示了分数矩阵（由 LLM 输出）如何转换为 token ID（及其对应的文本表示）。

虽然我们已经看到了如何使用 LLM 生成单个 token，但在下一节中，我们将把这些概念付诸实践，并实现一个函数，该函数顺序应用此概念以生成连贯的输出文本。

## 2.7 编写一个最小化的文本生成函数

上一节解释了 LLM 中基本顺序文本生成过程中的单次迭代。在本节中，基于该概念，我们将实现一个文本生成函数，该函数使用预训练 LLM 根据用户提示生成连贯的文本，如图 2.13 的章节概览所示。

---

**图 2.13** 本书中开发推理模型的四个关键阶段概览。在本节中，我们为预训练 LLM 实现一个文本生成函数。

---

图 2.13 中提到的这个文本生成函数的工作原理是：首先将输入提示转换为模型可以处理的 token ID。然后模型预测下一个最可能的 token，将其附加到序列中，并重新处理扩展后的序列以生成下一个 token。这个迭代过程持续进行，直到满足停止条件，然后将生成的 token ID 解码回文本。

图 2.14 逐步展示了这一过程，显示了每个阶段生成的 token ID 及其对应的文本。（此图与本节开头展示的图 2.10 类似，只是它同时显示了生成的 token ID 及其文本表示。）

---

**图 2.14** 大语言模型（LLM）中顺序（自回归）文本生成的示意图，显式展示了 token ID。在每次迭代中，模型基于原始输入和所有先前生成的 token 生成下一个 token。预测的 token 以其文本和 token ID 形式被添加到序列中。

---

下面清单 2.2 中的 `generate_text_basic_stream` 函数使用上一节介绍的 `argmax` 函数实现了顺序文本生成过程（图 2.14）：

**清单 2.2 一个基本的文本生成函数**

```python
@torch.inference_mode()  #A 禁用梯度跟踪以提高速度和内存效率
 def generate_text_basic_stream(
     model,
     token_ids,
     max_new_tokens,
     eos_token_id=None
 ):
     model.eval()  #B 将模型切换到评估模式以启用确定性行为（最佳实践）
     for _ in range(max_new_tokens):
         out = model(token_ids)[:, -1]  #C 获取最后一个 token 的分数
         next_token = torch.argmax(out, dim=-1, keepdim=True)
         if (eos_token_id is not None  #D 如果批次中所有序列都生成了 EOS，则停止
             and torch.all(next_token == eos_token_id)):
             break
         yield next_token  #E 每个 token 一生成就立即产出
         token_ids = torch.cat([token_ids, next_token], dim=1)  #F 将新预测的 token 附加到序列中
```

本质上，清单 2.2 中的 `generate_text_basic_stream` 函数通过 for 循环应用基于 `argmax` 的 token ID 提取，循环次数由用户指定的迭代次数（`max_new_tokens`）决定。它返回生成的 token ID，类似于图 2.14 中所示，然后我们可以将其转换回文本。

让我们使用该函数为一个简单的 "Explain large language models in a single sentence." 提示生成一个 100 token 的回复，以确保 `Qwen3Model` 和 `generate_text_basic_stream` 函数正常工作（我们将在后续章节中讨论推理任务示例）。

请注意，以下代码会很慢，可能需要 1-3 分钟才能完成，具体取决于你的计算机（我们将在后面的章节中加速它）：

```python
prompt = "Explain large language models in a single sentence."
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device  #A 将输入 token ID 转移到模型所在的同一设备（CPU、GPU）上
).unsqueeze(0)
max_new_tokens = 100  #B 让模型最多生成 100 个新 token

for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
):
    token_id = token.squeeze(0).tolist()  #C 将输出 token ID 从 PyTorch 张量转换为 Python 列表
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True  #D 停用缓冲，以便 token 实时打印
    )
```

生成的输出文本如下：

```
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.<|endoftext|>Human language
is a complex and dynamic system that has evolved over millions of
years to enable effective communication and social interaction. It is
composed of a vast array of symbols, including letters, numbers, and
words, which are used to convey meaning and express thoughts and
ideas. The evolution of language has
```

请注意，输出是在 CPU 上生成的。根据设备（例如，CPU 与 GPU）的不同，由于不同硬件上的浮点行为差异，确切的措辞可能会略有不同。

根据上面的输出，我们可以看到模型很好地遵循了指令，对提示产生了一个单一、清晰的句子作为回复。它在特殊 token `<|endoftext|>` 之后继续生成额外的、离题的文本。这个 token 在训练期间用于标记文档的结束并分隔不同的样本。

> **提示** " Large"（第一个输出词）中的前导空格是预期的，因为许多 tokenizer 在单词跟随前文时会对前面的空格进行编码。在清单 2.2 中，token 是逐个流式传输的，跟随输入文本，因此这个前导空格自然地出现在第一个发出的 token 中。如果我们想要更干净的输出，可以在第一个 token 或最终组装的字符串上调用 `.lstrip()`。

在使用模型进行推理（训练后生成文本）时，我们通常希望它在产生特殊 token `<|endoftext|>` 时立即停止。这个 token 由 ID 151643 表示，我们可以通过以下方式确认：

```python
print(tokenizer.encode("<|endoftext|>"))
```

为了方便，这个 token ID 也通过 `tokenizer.eos_token_id` 属性保存。我们可以将此 ID 传递给 `generate_text_basic_stream` 函数以指示生成应在何时停止：

```python
#A 传递序列结束（eos）token ID
for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id  #A
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

如果我们将此响应与之前的响应进行比较，我们可以看到一旦遇到序列结束 token，文本生成就会停止。

输出如下：

```
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.
```

你可能已经注意到，生成响应相对较慢，可能需要几秒钟到几分钟不等，具体取决于硬件。

在结束并学习如何大幅加速此函数之前，让我们在清单 2.3 中实现一个简单的工具函数，用于测量文本生成过程的运行时间：

**清单 2.3 Token 生成速度和内存使用量**

```python
import warnings

def generate_stats(output_token_ids, tokenizer, start_time, end_time):
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")
    for name, backend in (("CUDA", getattr(torch, "cuda", None)),
                          ("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():
            #A 检查我们是否实际在使用此后端
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )
            #B 如果支持则同步（对异步后端很重要）
            if hasattr(backend, "synchronize"):
                backend.synchronize()
            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
```

清单 2.3 中的 `generate_stats` 函数将计算总运行时间，给定开始和结束时间戳，以每秒 token 数（tokens/sec）为单位的生成速度，以及使用的 GPU 内存。请注意，GPU 内存使用量目前仅针对支持 CUDA 的 GPU（以及通过 XPU 支持的一些较新的 Intel GPU）计算，因为 PyTorch 缺乏用于 CPU 和 Apple Silicon GPU 的类似工具函数。

要应用 `generate_stats` 函数，我们通过 Python 的 `time` 模块在运行 `generate_text_basic_stream` 函数之前和之后立即获取 `start_time` 和 `end_time` 时间戳：

```python
import time

start_time = time.time()
generated_ids = []
for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
    next_token_id = token.squeeze(0)
    generated_ids.append(next_token_id)  #A 收集生成的 token
end_time = time.time()

output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)
```

在 Mac Mini M4 CPU 上，输出如下：

```
Time: 7.94 sec
5 tokens/sec

Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.
```

以每秒 5 个 token 的速度，生成速度相对较慢。在下一节中，我们将实现一种缓存技术，将生成速度提高 5-6 倍。

> **文本生成和推理术语**
> 在阅读 LLM 文献或软件文档时，你经常会看到**推理（inference）**这个词出现在你可能期望看到文本生成的地方。在神经网络上下文中，推理有非常具体的含义，即：采用一个参数已经学习并固定的模型，运行前向传播，并产生预测（例如，生成下一个 token）。在这个阶段没有进行任何估计或学习。在前向传播中，我们只是应用一个函数。
> 这与统计学中的推理不同，统计学中的目标是从数据中学习未知信息。统计推理涉及估计参数、量化不确定性或对总体或数据生成过程进行假设检验。
> 请注意，虽然在大型语言模型上下文中，模型正在计算下一个 token 的分布，人们有时会说模型正在"估计"或"推断"下一个 token，但这不是统计学意义上的估计。在这个阶段，模型是一个固定函数，其参数已经学习过了。
> 因此，当我们调用 `generate_text_basic_stream` 函数时，我们不是在执行统计推理。我们是在执行神经网络推理，这只是训练好的模型的前向应用，以产生下一个 token 的分布，并从此分布中选择下一个生成的 token。

## 2.8 通过 KV 缓存实现更快的推理

既然我们已经有了一个基本的文本生成函数，我们可以将注意力转向实际运行它时会发生什么。正如你可能已经注意到的，上一节中的文本生成可能有点慢。这种减速指向了一个关键问题：推理期间的性能。

在使用 LLM 运行推理时，在这个上下文中意味着从提示生成文本，运行时性能（效率）很快变得重要，特别是对于长序列。虽然本书中的代码强调清晰而非速度，但现实世界的系统经常使用工程技巧来提高推理效率。

这对于后面的章节也很有用，因为评估和推理工作流通常需要生成许多 token 或许多候选答案，因此这里的小幅加速会迅速累积。

在剩下的两节中，我们将介绍两种基本技术，KV 缓存和模型编译，如图 2.15 的概览所示，以加速文本生成。

---

**图 2.15** 本书中开发推理模型的四个关键阶段概览。本节建立在预训练 LLM 和我们之前编码的基本文本生成函数之上，并应用 KV 缓存来加速执行。

---

如图 2.15 所述，一种提高文本生成速度的工程技巧是 KV 缓存，其中 KV 指的是模型注意力机制中使用的 keys 和 values。如果你不熟悉这些术语，没关系。关键思想是，我们可以在文本生成的每一步缓存某些中间值并重用它们，如图 2.16 所示，这有助于加速推理。

---

**图 2.16** KV 缓存在自回归文本生成过程中如何提高效率的示意图。KV 缓存存储中间表示，而不是在每一步重新处理整个输入序列，这样 LLM 可以重用它们来生成下一个 token。这消除了在每次后续迭代中将生成的 token 与先前输入连接的需要。

---

如图 2.16 所示，KV 缓存的关键思想是在缓存中存储每次迭代中计算的中间值。以前，网络生成的每个新 token 都会与整个输入序列连接，并反复反馈回模型（图中用带叉的框表示）。这种方法效率低下，因为在后续迭代中，除了新生成的 token 之外，所有 token 都保持不变。通过使用 KV 缓存，我们避免了冗余计算，而是直接检索存储的中间表示。

粗略地说，在没有 KV 缓存的情况下生成 n 个新 token 需要反复重新计算越来越长序列上的工作，这总共增加了大约 O(n²) 的解码工作量。这里，O(n²) 意味着总工作量大致随输出长度的平方增长。使用 KV 缓存后，在初始传递之后，每个新步骤只处理新添加的 token，将其减少到大约 O(n)，这意味着总工作量大致与生成 token 的数量成正比。

如前所述，像 KV 缓存这样用于提高 token 生成速度的非推理聚焦的 LLM 细节超出了本书的范围，对于本书后面涵盖的主题也不是必需的。但是，感兴趣的读者可以在我的免费文章中找到有关 KV 缓存机制的更多信息：《Understanding and Coding the KV Cache in LLMs from Scratch》（https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms）。

下面是 `generate_text_basic_stream` 函数的修改版本，它整合了 KV 缓存，与清单 2.2 中的基本文本生成函数几乎完全相同，除了通过注释突出显示的 KV 缓存相关更改：

**清单 2.4 一个带有 KV 缓存的基本文本生成函数**

```python
from reasoning_from_scratch.qwen3 import KVCache

@torch.inference_mode()
def generate_text_basic_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])  #A 初始化 KV 缓存
    model.reset_kv_cache()                           #A
    out = model(token_ids, cache=cache)[:, -1]       #B 在第一轮中，像之前一样将整个输入提供给模型
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        if (eos_token_id is not None
            and torch.all(next_token == eos_token_id)):
            break
        yield next_token
        out = model(next_token, cache=cache)[:, -1]  #C 后续迭代只将 next_token 输入模型
```

清单 2.4 中的 `generate_text_basic_stream_cache` 函数与清单 2.2 中的 `generate_text_basic_stream` 函数只有细微差别。主要区别是引入了 `KVCache` 对象。

在第一次迭代期间，模型像之前一样被给予完整的输入 token 序列，使用 `model(token_ids, cache=cache)`。在幕后，KV 缓存为所有这些输入 token 计算并存储注意力 keys 和 values，因此它们在后续迭代中不需要重新计算。

在后续迭代中，我们不再需要传递整个序列。相反，我们只使用 `model(next_token, cache=cache)` 将 `next_token` 提供给模型。然后模型从先前存储的 KV 缓存中检索必要的上下文。

让我们为这个函数计时，看看它是否提供了任何性能优势：

```python
start_time = time.time()
generated_ids = []
for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
    next_token_id = token.squeeze(0)
    generated_ids.append(next_token_id)
end_time = time.time()

output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)
```

输出如下：

```
Time: 1.40 sec
29 tokens/sec

Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.
```

正如我们所见，这种方法明显更快，在 Mac Mini M4 CPU 上每秒生成 29 个 token，而之前仅为每秒 5 个 token。

重要的是，我们还看到生成的文本与之前相同，这是确保 KV 缓存被正确实现和使用的重要完整性检查。

在下一节中，我们将学习另一种可以进一步提高生成速度的技术，这在我们在 upcoming chapters 中评估模型时会派上用场。更快的生成使我们能够在更短的时间内运行更多的评估，并更容易高效地比较不同的模型或设置。

## 2.9 通过 PyTorch 模型编译实现更快的推理

在上一节中，我们介绍了 KV 缓存作为一种提高运行时效率的技术，如图 2.17 的概览所示。

---

**图 2.17** 本书中开发推理模型的四个关键阶段概览。本节建立在预训练 LLM 和我们之前编码的基本文本生成函数（包括 KV 缓存）之上，并添加模型编译以进一步提高执行速度。

---

如图 2.17 所示，在本章的剩余部分，我们将应用另一种可以显著提高模型推理速度的技术：使用 `torch.compile` 进行模型编译。简单来说，`torch.compile` 分析模型的内部计算图，并尝试将操作组转换为更优化的内核，这减少了 Python 开销和其他执行效率低下的问题。这可以提高文本生成期间的运行时性能，特别是当我们在循环中反复调用相同的模型代码时。

这在以后运行更大的评估和更复杂的推理工作流时很重要，因为即使是适度的每步加速也能节省大量的总运行时间。

```python
major, minor = map(int, torch.__version__.split(".")[:2])
if (major, minor) >= (2, 8):
    # 这避免了在 PyTorch 2.8 及更新版本中
    # 如果模型包含像 self.pos = self.pos + 1 这样的代码时
    # 重新触发模型重新编译
    torch._dynamo.config.allow_unspec_int_on_nn_module = True

model_compiled = torch.compile(model)
```

如果你使用的是配备 Apple Silicon 的 Mac 并遇到 `InductorError`，请确保使用 PyTorch 2.9 或更新版本。

值得注意的是，使用编译模型的第一次执行可能比平常慢，因为存在初始编译和优化步骤。为了更好地衡量性能改进，我们将重复文本生成过程多次。

首先，我们将使用非缓存版本的生成函数来测试这一点。清单 2.5 中的代码与我们之前使用的类似，只是我们连续运行了三次。代码执行可能需要几分钟才能完成，具体取决于系统：

**清单 2.5 使用编译模型生成文本**

```python
for i in range(3):  #A 我们运行 token 生成三次
    start_time = time.time()
    generated_ids = []
    for token in generate_text_basic_stream(
        model=model_compiled,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id)
    end_time = time.time()
    if i == 0:  #B 第一次运行标记为"Warm-up run"
        print("\n\nWarm-up run")
    else:
        print(f"\n\nTimed run {i}:")
    output_token_ids_tensor = torch.cat(generated_ids, dim=0)
    generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)
    print(f"\n{30*'-'}\n")
```

输出如下：

```
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Warm-up run
Time: 11.68 sec
3 tokens/sec
------------------------------
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Timed run 1:
Time: 6.78 sec
6 tokens/sec
------------------------------
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Timed run 2:
Time: 6.80 sec
6 tokens/sec
------------------------------
```

从上面的结果可以看出，编译后的模型在速度上略有提升，达到每秒约 6 个 token，而之前为每秒 5 个 token。

接下来，让我们看看 KV 缓存版本的表现如何，使用与之前相同的代码，只是将 `generate_text_basic_stream` 替换为 `generate_text_basic_stream_cache`：

**清单 2.6 使用带有 KV 缓存的编译模型生成文本**

```python
for i in range(3):
    start_time = time.time()
    generated_ids = []
    for token in generate_text_basic_stream_cache(
        model=model_compiled,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id)
    end_time = time.time()
    if i == 0:
        print("\n\nWarm-up run")
    else:
        print(f"\n\nTimed run {i}:")
    output_token_ids_tensor = torch.cat(generated_ids, dim=0)
    generate_stats(
        output_token_ids_tensor, tokenizer, start_time, end_time
    )
    print(f"\n{30*'-'}\n")
```

输出如下：

```
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Warm-up run
Time: 8.07 sec
5 tokens/sec
------------------------------
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Timed run 1:
Time: 0.60 sec
68 tokens/sec
------------------------------
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Timed run 2:
Time: 0.60 sec
68 tokens/sec
------------------------------
```

根据上面的输出，模型生成速度从带有 KV 缓存的未编译模型的每秒 29 个 token 提高到编译后的相同模型的每秒 68 个 token（在 Mac Mini M4 CPU 上），这是超过 2 倍的速度提升。

如果你没有看到任何改进，请尝试使用 `"max-autotune"` 模式而不是默认设置运行 `torch.compile`。例如，将

```python
model = torch.compile(model)
```

替换为

```python
model = torch.compile(model, mode="max-autotune")
```

> **练习 2.2：在非 CPU 设备上重新运行代码**
> 如果你有 GPU 访问权限，请在 GPU 设备上重新运行本章中的代码，并将运行时间与 CPU 运行时间进行比较。

如果你好奇不同的模型配置在 Apple Silicon GPU 和高端 NVIDIA GPU 上的表现如何，请参见表 2.1。

**表 2.1** 不同硬件上不同模型配置的 token 生成速度和 GPU 内存使用量

| Mode | Hardware | Tokens/sec | GPU memory |
|------|----------|-----------|------------|
| Regular | Mac Mini M4 CPU | 5 | - |
| Regular compiled | Mac Mini M4 CPU | 6 | - |
| KV cache | Mac Mini M4 CPU | 28 | - |
| KV cache compiled | Mac Mini M4 CPU | 68 | - |
| Regular | Mac Mini M4 GPU | 27 | - |
| Regular compiled | Mac Mini M4 GPU | 43 | - |
| KV cache | Mac Mini M4 GPU | 41 | - |
| KV cache compiled | Mac Mini M4 GPU | 71 | - |
| Regular | NVIDIA H100 GPU | 51 | 1.55 GB |
| Regular compiled | NVIDIA H100 GPU | 164 | 1.81 GB |
| KV cache | NVIDIA H100 GPU | 48 | 1.52 GB |
| KV cache compiled | NVIDIA H100 GPU | 141 | 1.81 GB |

如表所示，NVIDIA GPU 提供了最佳性能，这是预期的。一旦启用 KV 缓存和编译模型，CPU 的表现仍然出人意料地好。然而，有一些重要的细节可以解释为什么 GPU 在这种特定设置下的优势没有更大。

首先，本章中的模型没有针对 GPU 进行优化。该实现旨在内存使用和计算速度之间取得良好的平衡，因为内存是大多数读者的主要瓶颈。针对 GPU 优化的变体会预先分配最大上下文长度（在本例中为 4 万个 token）的完整 K 和 V 张量。这种预分配避免了通过 `torch.cat` 进行重复连接，但也会增加内存消耗。

对于大的上下文大小，预分配例如 K 和 V 各 4 万个条目会增加明显的占用。另一种方法是让用户在构造时指定固定的上下文大小，但这会引入额外的配置开销。这里的 KV 缓存通过 `torch.cat` 按需增长，这更简单且更节省内存，尽管在 GPU 上连接非预分配的张量会稍慢一些。

我在 bonus materials 中包含了一个 GPU 优化版本，其中 KV 缓存版本比普通版本稍快，但它使用更多内存（参见 https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM）。

其次，用于基准测试的模型很小。较大的模型从 KV 缓存和 GPU 优化的内存布局中获益更多。批量推理也是如此，GPU 可以更好地饱和其计算单元。使用更大的模型或针对 GPU 的实现，NVIDIA GPU 的性能优势会更加明显。

所有示例都使用单个提示运行（即，batch size 为 1）。对于对多输入下性能如何扩展感兴趣的读者，附录 E 中讨论了批量推理。

## 2.10 小结

使用 LLM 生成文本涉及多个关键步骤：
- 搭建运行 LLM 代码的编码环境并安装必要的依赖项。
- 加载预训练的基础 LLM（例如 Qwen3 0.6B），它将在后续章节中扩展推理能力。
- 初始化并使用 tokenizer，它将文本输入转换为 token ID，并将输出解码回人类可读的形式。
- LLM 中的文本生成遵循顺序（自回归）过程，模型通过预测下一个最可能的 token 一次生成一个 token。
- 文本生成的速度和效率可以通过以下方式提高：
  - **KV 缓存**，它存储中间状态以避免在每一步重新计算先前遇到的输入 token。
  - **模型编译**，使用 `torch.compile`，它可以优化运行时性能。

本章通过使用预训练的基础 LLM 实现一个功能完善且高效的文本生成流水线，为后续章节中的推理能力奠定了技术基础。
