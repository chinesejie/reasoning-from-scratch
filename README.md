# 从零构建推理模型(中文版)

本仓库包含开发大语言模型（LLM）推理模型的代码，是[*从零构建推理模型*](https://mng.bz/lZ5B)一书的官方代码仓库。


<br>
<br>

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="250px"></a>

（彩色印刷。）

<br>

在[*从零构建推理模型*](https://mng.bz/lZ5B)中，你将学习并理解推理大语言模型（LLM）的工作原理。

推理是近年来改进LLM最令人兴奋也最重要的进展之一，但如果你只听说过"推理"这个词并从理论上了解它，它也是最容易被误解的概念之一。这就是为什么本书采用实践方法的原因。我们将从一个预训练的基础LLM开始，然后一步步亲自用代码为其添加推理能力，这样你就能清楚地看到它是如何工作的。

本书描述的方法将指导你开发一个用于教育目的小型但功能完备的推理模型。它模拟了创建大规模推理模型（如DeepSeek R1、GPT-5 Thinking等）所使用的方法。此外，本书还包括加载现有预训练模型权重的代码。

- 官方[源代码仓库](https://github.com/rasbt/reasoning-from-scratch)链接
- [Manning出版社网站](https://mng.bz/lZ5B)上的书籍链接
- Amazon.com上的书籍页面链接（待定）
- ISBN 9781633434677



<br>
<br>

要下载本仓库的副本，请点击[下载ZIP](https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip)按钮，或在终端中执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
```

<br>


> **提示：**
> 第2章提供了关于安装Python、管理Python包以及设置编码环境的额外提示。

<br>
<br>

## 目录（进行中）

[![代码测试 Linux](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml)
[![代码测试 macOS](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml)
[![代码测试 Windows](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml)

- [故障排除指南](./troubleshooting.md)

| 章节标题                                               | 主要代码                                                    |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| 第1章：理解推理模型                        | 无代码                                                      |
| 第2章：使用预训练LLM生成文本                | - [ch02_main.ipynb](ch02/01_main-chapter-code/ch02_main.ipynb)<br/>- [ch02_exercise-solutions.ipynb](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb) |
| 第3章：评估推理模型                           | - [ch03_main.ipynb](ch03/01_main-chapter-code/ch03_main.ipynb)<br/>- [ch03_exercise-solutions.ipynb](ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb) |
| 第4章：通过推理时缩放改进推理       | - [ch04_main.ipynb](ch04/01_main-chapter-code/ch04_main.ipynb)<br/>- [ch04_exercise-solutions.ipynb](ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb) |
| 第5章：通过自我完善实现推理时缩放            | - [ch05_main.ipynb](ch05/01_main-chapter-code/ch05_main.ipynb)<br/>- [ch05_exercise-solutions.ipynb](ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb) |
| 第6章：使用强化学习训练推理模型 | - [ch06_main.ipynb](ch06/01_main-chapter-code/ch06_main.ipynb)<br/>- [ch06_exercise-solutions.ipynb](ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb) |
| 第7章：改进GRPO以进行强化学习             | - [ch07_main.ipynb](ch07/01_main-chapter-code/ch07_main.ipynb)<br/>- [ch07_exercise-solutions.ipynb](ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb) |
| 第8章：蒸馏推理模型以实现高效推理   | - [ch08_main.ipynb](ch08/01_main-chapter-code/ch08_main.ipynb)<br/>- [ch08_exercise-solutions.ipynb](ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb) |
| 附录A：参考文献和延伸阅读                  | 无代码                                                      |
| 附录B：习题解答                              | 代码和解答位于各章节的子文件夹中           |
| 附录C：Qwen3 LLM源代码                           | - [chC_main.ipynb](chC/01_main-chapter-code/chC_main.ipynb)  |
| 附录D：使用更大的LLM                               | - [chD_main.ipynb](chD/chD_main.ipynb)                       |
| 附录E：面向批处理和吞吐量的执行      | - [chE_main.ipynb](chE/chE_main.ipynb)                       |
| 附录F：LLM评估的常见方法             | - [chF_main.ipynb](chF/01_main-chapter-code/chF_main.ipynb)  |
| 附录G：构建聊天界面                       | - [chG](chG)                                                 |

<br>
&nbsp;

下面的心智模型总结了本书涵盖的主要技术。

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/mental-model.webp" width="650px">



<br>



&nbsp;
## 配套书籍

请注意，*从零构建推理模型*是一本专注于改进LLM推理方法的独立书籍。

在本书中，我们使用一个预训练的开源基础LLM（Qwen3），并在其之上从头开始应用推理方法。这包括推理时缩放、强化学习和蒸馏。

然而，如果你有兴趣了解常规基础LLM是如何实现的，你可能会喜欢我的前一本书[*从零构建大语言模型*](https://amzn.to/4fqvn0D)。

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="120px"></a>

- [Amazon链接](https://amzn.to/4fqvn0D)
- [Manning链接](http://mng.bz/orYv)
- [GitHub仓库](https://github.com/rasbt/LLMs-from-scratch)


<br>
&nbsp;

## 硬件要求

本书主要章节的代码设计为在消费级硬件上在合理的时间内运行，不需要专门的服务器硬件。这种方法确保了广泛的读者可以参与学习。此外，如果有GPU可用，代码会自动利用它们。也就是说，第2-4章在CPU和GPU上都能很好地运行。对于第5章和第6章，如果你想复现章节中的结果，建议使用GPU。


（请参阅[setup_tips](ch02/02_setup-tips/python-instructions.md)文档以获取额外建议。）

&nbsp;
## 练习

本书的每一章都包含若干练习。解答汇总在附录B中，相应的代码笔记本可在本仓库的主要章节文件夹中找到（例如，[`ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb`](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb)）。


&nbsp;
## 附加资料

有几个文件夹包含为感兴趣的读者提供的可选附加材料：

- **第2章：使用预训练LLM生成文本**
  - [可选Python设置和云GPU推荐](ch02/02_setup-tips)
  - [使用GPU优化的LLM版本](ch02/03_optimized-LLM)
  - [在Windows上使用`torch.compile()`](ch02/04_torch-compile-windows)
  - [运行推理并与模型聊天](ch02/05_use_model)
- **第3章：评估LLM**
  - [MATH-500验证器脚本](ch03/02_math500-verifier-scripts)
  - [高级解析器](ch03/03_advanced-parser)（混合LaTeX解析器）
- **第4章：通过推理时缩放改进推理**
  - [MATH-500上的推理缩放](ch04/02_math500-inference-scaling-scripts)（思维链提示、自洽性）
- **第5章：通过自我完善实现推理时缩放**
  - [MATH-500上的更多推理缩放](ch05/02_math500-more-inference-scaling-scripts)（Best-of-N、自我完善）
- **第6章：使用强化学习训练推理模型**
  - [GRPO脚本](ch06/02_rlvr_grpo_scripts_intro)（批处理模式）
- **第7章：改进GRPO以进行强化学习**
  - [高级GRPO脚本](ch07/03_rlvr_grpo_scripts_advanced)（包括DeepSeek-V3.2-、Olmo3-和GDPO风格训练）
  - [下载训练检查点](ch07/04_download_trainining_checkpoints)（如何下载和使用第6章和第7章的GRPO检查点）
- **第8章：蒸馏推理模型以实现高效推理**
  - [生成蒸馏数据](ch08/02_generate_distillation_data)（通过Ollama或OpenRouter生成教师输出）
  - [使用蒸馏进行训练](ch08/04_train_with_distillation)（包括单示例和批处理蒸馏脚本）
  - [下载训练检查点](ch08/05_download_training_checkpoints)（如何下载和使用第8章蒸馏检查点）
  - [通过Hugging Face使用Qwen3](ch08/06_use_via_huggingface)（如何使用基础模型和第6-8章的检查点与`transformers`）
- **附录F：LLM评估的常见方法**
  - [MMLU评估方法](chF/02_mmlu)
  - [LLM排行榜](chF/03_leaderboards)
  - [LLM作为评判者](chF/04_llm-judge)
- **附录G：构建聊天界面**
  - [聊天界面代码](chG/01_main-chapter-code)


&nbsp;
## 问题、反馈和为本仓库做出贡献

对于常见问题，请参阅[故障排除指南](./troubleshooting.md)。

我欢迎各种反馈，最好通过[Manning讨论论坛](https://livebook.manning.com/forum?product=raschka2&page=1)或[GitHub Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions)分享。同样，如果你有任何问题或只是想与其他人交流想法，也请随时在论坛中发帖。

请注意，由于本仓库包含与实体书对应的代码，我目前无法接受会扩展主要章节内容的贡献，因为这会引入与实体书的偏差。保持一致有助于确保每个人都能获得顺畅的体验。

&nbsp;
## 引用

如果你发现本书或代码对你的研究有用，请考虑引用它。

芝加哥格式引用：

> Raschka, Sebastian. *从零构建推理模型*. Manning, 2025. ISBN: 9781633434677.

BibTeX条目：

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Reasoning Model (From Scratch)},
  publisher    = {Manning},
  year         = {2025},
  isbn         = {9781633434677},
  url          = {https://mng.bz/lZ5B},
  github       = {https://github.com/rasbt/reasoning-from-scratch}
}
```
