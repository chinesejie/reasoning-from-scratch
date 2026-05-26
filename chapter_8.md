# 8 蒸馏推理模型以实现高效推理

本章涵盖
- 推理模型的硬蒸馏与软蒸馏
- 创建并准备教师生成的推理数据集
- 使用交叉熵损失训练并评估蒸馏后的学生模型

推理性能不仅可以通过推理时扩展和强化学习来提升，还可以通过蒸馏来实现。在蒸馏中，一个较小的学生模型会在一个较大的教师模型生成的推理轨迹和答案上进行训练。如图 8.1 的概述所示，本章重点介绍这种训练时技术。

图 8.1 本书涵盖主题的思维模型。本章重点介绍蒸馏，即使用一个较小的学生模型在较大的教师模型生成的推理轨迹上进行训练。

首先，我们将对模型蒸馏进行一般性介绍，然后再更详细地讨论图 8.1 中展示的各个步骤。

## 8.1 推理任务中的模型蒸馏简介

模型蒸馏是指在一个较小的 LLM（学生模型）上，使用一个较大的 LLM（教师模型）产生的输出来进行训练。对于推理模型，这些输出通常不仅包括最终答案，还包括得出最终答案的中间推理轨迹。

蒸馏尤其重要，因为最强的推理模型往往过于庞大且昂贵，无法直接部署使用。例如，DeepSeek-R1 拥有 6710 亿参数。如此规模的系统开发成本高昂、部署成本高昂，而且远超大多数从业者能在本地硬件上运行的范围。

本章旨在展示一个更小规模的通用生产模式。DeepSeek-R1 团队通过蒸馏这个 6710 亿参数的教师模型，创建了较小的模型变体。我们在这里遵循相同的基本思路，但采用适合本书的实用规模。

在本书中，我们有意使用小模型，而不是从头训练一个非常大的 LLM。但本章的工作流程仍然相同。我们使用一个更强的教师模型来生成推理轨迹，然后训练一个较小的学生模型来复现它们。在我们的设置中，这个蒸馏阶段也只需要很少的时间。这里，在 DGX Spark 上的蒸馏训练运行大约需要 3 小时，使用约 15 GB 的 RAM，而在相同硬件上，第 6 章中即使进行几轮 GRPO 也需要大约 12 小时和约 70 GB 的 RAM。

蒸馏也可能比从头开始使用可验证奖励的强化学习（RLVR）训练小模型更有效。例如，DeepSeek-R1 论文报告称，其较小的蒸馏变体优于仅使用强化学习训练的同类模型。在该设置中，最大的 DeepSeek-R1 模型（6710 亿参数）充当教师模型，生成用于训练较小学生模型的监督信号。

蒸馏主要有两种类型：硬蒸馏（hard distillation）和软蒸馏（soft distillation）。在硬蒸馏中，学生模型在教师模型生成的文本上进行训练，因此教师模型的 token 被视为目标。在软蒸馏中，学生模型通过最小化 KL 散度（衡量两个概率分布差异的指标）来训练，以匹配教师模型在整个词表上的概率分布。这两种方法如图 8.2 所示。

图 8.2 硬蒸馏使用教师模型生成的 token 来训练学生模型；软蒸馏使用教师模型的完整输出分布来训练学生模型。

如图 8.2 所示，第一种选择是纯硬蒸馏。在这里，我们只使用教师模型生成的文本作为训练目标。

对于熟悉典型 LLM 训练流程的读者（在我的另一本书《从零构建大型语言模型》中有更详细的介绍），硬蒸馏本质上就是在合成数据上进行监督微调。

这也是较小的 DeepSeek-R1 蒸馏模型所使用的设置。一个大型教师模型生成推理轨迹和答案，学生模型经过微调以复现它们。

根据 DeepSeek-R1 论文，对于小模型来说，这种蒸馏方法可以获得比使用强化学习训练更高的准确率。

硬蒸馏的主要实际优势在于，我们只需要访问教师模型生成的文本，而不需要它的 logits。

第二种选择是纯软蒸馏。学生模型不是仅匹配教师模型选定的 token，而是被训练来匹配教师模型在整个词表上的完整输出分布。这为学生模型提供了更丰富的信息，了解教师模型认为哪些替代 token 是合理的，但这需要在训练时访问教师模型的 logits 或 log-probabilities。

第三种选择是结合硬蒸馏和软蒸馏。这是经典的“知识蒸馏”设置，早期在计算机视觉领域流行，例如在论文《Distilling the Knowledge in a Neural Network》中。在这种情况下，我们在教师模型实际输出 token 上进行训练，同时鼓励学生模型匹配教师模型的完整分布。

在实践中，硬蒸馏对于 LLM 来说要常见得多。一个原因是完整的教师模型 logits 通常无法获取。像 ChatGPT 或 Claude 这样的专有系统可能会暴露生成的文本，但通常不会暴露经典软蒸馏所需的完整词表分布。

> **注意** 将生成的文本用于蒸馏可能会受到提供商特定使用政策的限制，包括 OpenAI 和 Anthropic 的服务条款，因此在实践中使用此类输出进行训练之前，应仔细审查这些条款。

即使 logits 可用，软蒸馏也更麻烦。学生模型和教师模型通常需要相同的分词器，以便它们的词表分布对齐，这使得在同一模型家族内使用这种方法更容易。此外，存储和使用长推理轨迹的完整 token 分布也要昂贵得多。相比之下，存储纯文本输出既便宜又简单。

在这里，我们专注于 DeepSeek-R1 风格的硬蒸馏，因为它对大多数读者来说是更实用的设置。本章步骤总结如图 8.3。

图 8.3 本章有四个主要步骤：(1) 蒸馏介绍与概述；(2) 数据集准备（步骤 2a-2c）；(3) 通过蒸馏进行训练（步骤 3a-3c）；(4) 评估。

有了这个概述，我们现在可以从蒸馏背后的高级思想转向实现它所需的实际步骤。在下一节中，我们首先开始准备用于蒸馏的数据集，这是训练学生模型的基础。

## 8.2 生成用于推理蒸馏的数据集

第一个实际步骤是创建用于训练学生模型的数据集。对于我们来说，学生模型仍然是我们在前面章节中使用的 Qwen3 0.6B 基础模型。

为了构建数据集，我们使用来自 MATH 数据分割的 12,000 道数学题，这些题目与 MATH-500 评估集不重叠。这些与我们在第 6 章和第 7 章中用于 RLVR 的 12,000 道题目相同。我们现在不再采样多个学生响应并计算奖励，而是将这些题目输入到一个现有的推理模型 DeepSeek-R1（教师模型）中，并将其响应收集为训练目标。这个设置如图 8.4 所示。

图 8.4 本章使用的蒸馏设置。我们使用 12,000 道不重叠的 MATH 训练题目从 DeepSeek-R1 获取合成解答，然后在单独的 MATH-500 测试集上评估蒸馏后的 Qwen3 学生模型。

在第 6 章和第 7 章的 RLVR 设置中，我们训练 Qwen3 基础模型产生正确的解答，然后使用验证器将模型的最终答案与参考答案进行比较。验证器产生奖励信号。

在蒸馏中，监督更为直接。我们不是使用奖励函数将学生模型的答案与参考答案进行比较，而是将学生模型生成的 token 与教师模型生成的 token 进行比较，如图 8.4 所示。换句话说，教师模型的响应成为目标序列。与 RLVR 的这种比较在图 8.5 中有更详细的说明。

图 8.5 在 RLVR 中，生成的答案与真实参考解答进行比较（上方子图）；而在蒸馏中，学生答案与教师生成的解答进行比较（下方子图）。

蒸馏的一个实际优势是，我们可以在训练学生模型之前提前生成教师数据集。

因为为所有 12,000 道数学题生成教师答案可能非常耗时且耗费资源，所以我提前使用通过 OpenRouter 托管的 6710 亿参数 DeepSeek-R1 模型创建了这个数据集。

完整的数据生成成本约为 50 美元的 API 使用费。接下来，我们只需加载这个预生成的数据集，因此你不需要自己生成它即可跟随学习。如果你好奇数据生成过程，代码和使用说明可在补充材料中找到：https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/02_generate_distillation_data。

## 8.3 加载用于蒸馏的 MATH 训练数据集

我们现在加载上一步中由 DeepSeek-R1 生成的蒸馏 MATH 数据集。

每个示例包含一道数学题，以及我们可以用作监督微调目标的推理轨迹和最终答案。这一步在图 8.6 中突出显示。

图 8.6 突出显示当前章节的概述。在这里，我们从 JSON 文件加载 DeepSeek-R1 生成的数据集，然后准备它用于训练。

我将数据集通过 Hugging Face Hub 提供，而不是 GitHub，因为 JSON 文件大约为 107 MB，超过了 GitHub 100 MB 的文件大小限制：https://huggingface.co/datasets/rasbt/math_distill。以下辅助函数会在所选分区尚未在本地缓存时下载它，并将其作为 Python 对象返回。

**代码清单 8.1 加载蒸馏后的 MATH 训练分割**

```python
import json
import requests
from pathlib import Path

def load_distill_data(
    local_path=None,
    partition="deepseek-r1-math-train",
    save_copy=True,
):
    if local_path is None:
        local_path = f"{partition}.json"
    local_path = Path(local_path)
    url = (
        "https://huggingface.co/datasets/rasbt/math_distill"
        "/resolve/main/data/"
        f"{partition}.json"
    )
    backup_url = (
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        f"MATH/{partition}.json"
    )
    if local_path.exists():  # A
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        size_kb = local_path.stat().st_size / 1e3
        print(f"{local_path}: {size_kb:.1f} KB (cached)")
        return data
    assert partition in (
        "deepseek-r1-math-train",
        "deepseek-r1-math500",
        "qwen3-235b-a22b-math-train",
        "qwen3-235b-a22b-math500",
    )
    try:  # B
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        print("Using backup URL.")
        r = requests.get(backup_url, timeout=30)
        r.raise_for_status()
    data = r.json()
    if save_copy:  # C
        with local_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        size_kb = local_path.stat().st_size / 1e3
        print(f"{local_path}: {size_kb:.1f} KB")
    return data
```

输出为：

```
deepseek-r1-math-train.json: 107538.0 KB
Dataset size: 12000
```

接下来，让我们检查其中一个训练示例，以了解数据集结构并确切查看教师生成的数据包含哪些字段。为此，我们选取其中一个训练示例（第五个）进行说明：

```python
from pprint import pprint
pprint(math_train[4])
```

打印的输出如下：

```python
{'gtruth_answer': '6',
 'message_content': 'Sam worked \\( x \\) days and did...'
 'message_thinking': "Okay, let's see. Sam was hired for 20 days..."
 'problem': 'Sam is hired for a 20-day period...'
}
```

每个数据集条目包含数学题本身（`problem`）、真实答案（`gtruth_answer`）、教师模型的推理轨迹（`message_thinking`）和最终答案（`message_content`）。

对于蒸馏来说，最重要的两个字段是推理轨迹和最终答案，因为它们共同构成了学生模型应该学习复现的目标文本。

`format_distilled_answer` 函数（代码清单 8.2）通过将推理轨迹放在 `<think>...</think>` 标签内，然后附加最终答案，将这两个字段组合成一个单一的训练目标。

打印的输出为：

```
<think>Okay, let's see. Sam was hired for 20 days. Each day he works, he earns
$60...So answer is 6 days not worked.</think>

Sam worked \( x \) days and did not work \( y \) days. We know:...
Sam did not work \(\boxed{6}\) days.
```

**代码清单 8.2 格式化教师响应以进行蒸馏**

```python
def format_distilled_answer(entry):
    content = str(entry["message_content"]).strip()
    if not content:
        raise ValueError("Missing non-empty 'message_content' field.")
    thinking = str(entry["message_thinking"]).strip()
    return f"<think>{thinking}</think>\n\n{content}"

print(format_distilled_answer(math_train[4]))
```

如前一章所讨论的，`<think></think>` token 是可选的。它们对于蒸馏本身不是必需的，尽管它们有助于清晰地区分推理轨迹和最终答案。

这种区分在实施向最终用户隐藏冗长推理轨迹的用户界面时非常有用。一些系统，包括像 ChatGPT 这样的产品，可能只显示最终答案，同时隐藏部分内部推理。教模型使用显式的 `<think>` 标签可以使这些轨迹更容易解析和处理。

由于数据集还包含真实标签，我们可以衡量教师模型在此数据集上的准确性。为了方便，我们使用第 3 章补充材料中的 `evaluate_json.py` 脚本，该脚本使用在该章中实现的验证器将生成的答案与参考答案进行比较。这让我们可以快速估计 DeepSeek-R1 在蒸馏数据集上的表现。

```python
from reasoning_from_scratch.ch07 import download_from_github
_ = download_from_github(
    "ch03/02_math500-verifier-scripts/evaluate_json.py"
)
```

通过前面的 Python 代码下载脚本后，我们可以在代码终端中如下运行它：

```bash
uv run evaluate_json.py \
  --json_path "deepseek-r1-math-train.json" \
  --gtruth_answer gtruth_answer \
  --generated_text message_content
```

输出为：

```
Accuracy: 90.6% (10871/12000)
```

（如果你不是 uv 用户，请将 `uv run` 替换为 `python`。）

虽然模型并不完美，但 90.6% 是一个相对较高的准确率。此外，在第 3 章的 MATH-500 测试集上，它达到了 91.2% 的准确率，这远高于我们将要训练的 Qwen3 0.6B 基础模型（MATH-500 上为 15.2% 的准确率）或官方 Qwen3 0.6B 推理参考模型（MATH-500 上为 50.8%）。

## 8.4 构建训练示例

接下来，我们将原始数据集条目转换为模型可以使用的训练示例。从高层次来看，这意味着一致地格式化提示和答案，对它们进行分词，并将生成的 token ID 与提示长度一起存储。整个预处理阶段在图 8.7 中突出显示。

图 8.7 通过理解分词器、对示例进行分词以及过滤和分割数据集，将加载的数据集转换为适合模型训练的格式。

这里的大部分工作都是简单直接的预处理。在 RLVR 章节中，我们即时执行了类似的准备，因为每个训练示例只采样一次然后被丢弃。蒸馏不同，我们通常对相同的示例进行多个训练 epoch 的循环。因此，一次性格式化和分词数据集、存储处理后的示例，并在训练期间重用它们会更高效。这也是蒸馏一旦收集到教师数据后通常比 RLVR 更容易迭代的原因之一。

> **什么是训练 epoch？**
> 训练 epoch，简称 epoch，是机器学习和深度学习中的经典术语。一个 epoch 是对完整训练数据集的一次完整遍历。例如，如果我们训练 3 个 epoch，模型会看到每个训练示例三次，通常每次顺序不同。多个 epoch 帮助模型通过多次 revisit 相同数据来逐步改进。

### 8.4.1 加载并理解分词器

我们从分词器开始。在这里，我们使用 Qwen3 推理分词器，因为它支持第 7 章中引入的 `<think>...</think>` token。

**代码清单 8.3 加载 Qwen3 推理分词器**

```python
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
)

def load_reasoning_tokenizer(local_dir="qwen3"):
    download_qwen3_small(
        kind="reasoning", tokenizer_only=True, out_dir=local_dir
    )
    tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )
    return tokenizer

tokenizer = load_reasoning_tokenizer()
prompt = "Sam is hired for a 20-day period..."
prompt_ids = tokenizer.encode(prompt)
decoded_prompt = tokenizer.decode(prompt_ids)
print(decoded_prompt)
```

我们设置 `apply_chat_template=True` 和 `add_generation_prompt=True`，以便分词器应用与 Qwen3 的聊天和推理模型相同的提示格式样式。下面的示例展示了自动插入的额外包装 token。

格式化后的文本如下：

```
<|im_start|>user
Sam is hired for a 20-day period...<|im_end|>
<|im_start|>assistant
```

特别是，`<|im_start|>user` 标记用户提示的开始，`<|im_end|>` 标记提示的结束，`<|im_start|>assistant` 标记模型响应的开始。

这种聊天风格的包装是可选的，但它是指令和聊天微调的常见约定。

对于目标答案，我们通过设置 `chat_wrapped=False` 来禁用这种包装。否则，提示和答案都会引入它们自己的 assistant-start token，这在将它们连接成单个训练序列时不是我们想要的：

```python
answer = (
    "<think>Okay, let me try to solve "
    "this problem...</think> \\boxed{4}"
)
answer_ids = tokenizer.encode(answer, chat_wrapped=False)
decoded_answer = tokenizer.decode(answer_ids)
print(decoded_answer)
```

如所示，`chat_wrapped=False` 设置抑制了答案 token 的聊天模板：

```
<think>Okay, let me try to solve this problem...</think> \boxed{4}
```

对于训练，我们需要完整的 token ID 序列：包装的提示、后跟教师答案、后跟一个序列结束 token。这是模型在下一个 token 预测期间将看到的序列：

```python
token_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
decoded_token_ids = tokenizer.decode(token_ids)
print(decoded_token_ids)
```

格式化后的字符串如下所示：

```
<|im_start|>user
Sam is hired for a 20-day period...<|im_end|>
<|im_start|>assistant
<think>Okay, let me try to solve this problem...</think>\boxed{4}<|im_end|>
```

值得强调的是，推理分词器主要方便之处在于它已经包含了 `<think></think>` token。但原则上，我们也可以使用基础分词器。同样，聊天模板也不是严格必需的。我们在这里保留它，因为它与 Qwen3 自己的推理模型使用的格式相匹配，并有助于保持训练和评估设置的一致性。

如果你稍后使用前面章节的脚本评估模型，请记住使用 `--which_model "reasoning"` 设置，以便评估使用相同的分词器变体。

### 8.4.2 格式化并分词数据集

在加载并理解分词器之后，我们现在可以将格式化和分词步骤应用于整个数据集。这个过渡在图 8.8 中突出显示。

图 8.8 分词器步骤完成后，我们现在继续将格式化和分词步骤应用于整个数据集。

有了分词器，我们现在可以通过 `build_examples` 函数一致地应用整个数据集的格式化和分词步骤。单个训练样本的完整格式化和分词流程如图 8.9 所示。

图 8.9 一个训练样本的分词流程示例。数学题被渲染为聊天提示格式，教师推理轨迹和最终答案通过 `format_distilled_answer` 组合，然后两部分连接成一个 token 序列。

如图 8.9 所示，`build_examples` 函数遵循三个步骤。首先，它渲染并分词提示。其次，它格式化并分词教师答案。第三，它将两部分连接起来，并记录提示长度，以便我们稍后仅在答案 token 上计算损失。代码清单 8.4 展示了如何在代码中实现这一点。

**代码清单 8.4 构建并检查分词后的蒸馏示例**

```python
from reasoning_from_scratch.ch03 import render_prompt

def build_examples(data, tokenizer):
    examples = []
    skipped = 0
    for entry in data:
        try:
            # A
            prompt = render_prompt(entry["problem"])
            prompt_ids = tokenizer.encode(prompt)
            # B
            target_answer = format_distilled_answer(entry)
            answer_ids = tokenizer.encode(
                target_answer, chat_wrapped=False
            )
            # C
            token_ids = (
                prompt_ids + answer_ids + [tokenizer.eos_token_id]
            )
            if len(token_ids) < 2:
                skipped += 1
                continue
            # D
            examples.append({
                "token_ids": token_ids,
                "prompt_len": len(prompt_ids),
            })
        except (KeyError, TypeError, ValueError):
            # E
            skipped += 1
    return examples, skipped

examples, skipped = build_examples(math_train, tokenizer)
```

运行代码清单 8.4 中的代码后，得到的结果为：

```python
print("Number of examples:", len(examples))
print("Number of skipped examples:", skipped)
```

输出：

```
Number of examples: 12000
Number of skipped examples: 0
```

接下来，让我们解码其中一个训练示例以进一步检查它：

```python
print(tokenizer.decode(examples[4]["token_ids"]))
```

输出为：

```
<|im_start|>user
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}
Question:
Sam is hired for a 20-day period...
Answer:<|im_end|>
<|im_start|>assistant
<think>Okay, let's see. Sam was hired for 20 days.... So answer is 6 days not
worked.</think>
...Sam did not work \(\boxed{6}\) days.<|im_end|>
```

查看上面的输出，我们可以确认它符合所有格式要求。例如，它正确地使用了聊天模板，答案的推理轨迹正确地包含在 `<think></think>` 标签内。最后，答案以序列结束 token `<|im_end|>` 结尾。

### 8.4.3 过滤并分割数据集

示例被分词后，我们仍然需要过滤掉长序列，并将数据集分割为训练集和验证集子集。这一步在图 8.10 中突出显示。

图 8.10 分词后，我们过滤掉长序列，并将剩余示例分割为训练集和验证集子集。

分词后，检查序列长度是很有用的。这告诉我们示例的平均长度、哪些样本是极端异常值，以及我们的过滤需要多激进。然后，我们移除超过所选最大长度的示例，用固定的随机种子打乱剩余数据，并分割出一小部分验证集。

让我们首先通过代码清单 8.5 分析序列长度。

**代码清单 8.5 计算长度并过滤长示例**

```python
def compute_length(examples, answer_only=False):
    lengths = []
    for ex in examples:
        total = len(ex["token_ids"])
        length = total - ex["prompt_len"] if answer_only else total
        lengths.append(length)
    avg_len = round(sum(lengths) / len(lengths))
    shortest_len = min(lengths)
    longest_len = max(lengths)
    shortest_idx = lengths.index(shortest_len)
    longest_idx = lengths.index(longest_len)
    print(f"Average: {avg_len} tokens")
    print(f"Shortest: {shortest_len} tokens (index {shortest_idx})")
    print(f"Longest: {longest_len} tokens (index {longest_idx})")

compute_length(examples)
```

得到的输出为：

```
Average: 2946 tokens
Shortest: 236 tokens (index 10846)
Longest: 42005 tokens (index 2529)
```

如我们所见，平均响应长度为 2,946 个 token，这对于推理模型来说是典型的。不过，存在异常值。例如，最长的答案有 42,005 个 token，非常过长。索引位置（index 10846 和 index 2529）分别表示数据集中最短和最长示例的位置，以防我们想要检查它们。

为了在这个蒸馏示例中保持合理的计算成本，我们使用代码清单 8.6 中的代码将数据集过滤为仅包含最多 2,048 个 token 的数据集条目。在实践中，控制序列长度是使蒸馏在较小硬件上可行的主要步骤之一。

**代码清单 8.6 过滤长示例**

```python
def filter_examples_by_max_len(examples, max_len=2048):
    filtered_examples = [
        s for s in examples
        if len(s["token_ids"]) <= max_len
    ]
    print("Original:", len(examples))
    print("Filtered:", len(filtered_examples))
    print("Removed:", len(examples) - len(filtered_examples))
    return filtered_examples

filtered_examples = filter_examples_by_max_len(examples, max_len=2048)
```

运行代码清单 8.6 中的过滤代码后，有 5,305 个训练示例被移除：

```
Original: 12000
Filtered: 6695
Removed: 5305
```

让我们在这个新子集上计算数据集长度：

```python
compute_length(filtered_examples)
```

如我们所见，现在平均 token 长度下降到 1,180 个 token，并且没有格式化的训练示例超过 2,048 个 token。

输出：

```
Average: 1180 tokens
Shortest: 236 tokens (index 5971)
Longest: 2048 tokens (index 5587)
```

最后，我们将数据集分割为训练示例和验证示例，后者用于在训练运行期间进行快速评估。

**代码清单 8.7 分割为训练集和验证集**

```python
import random
rng = random.Random(123)
rng.shuffle(filtered_examples)
train_examples = filtered_examples[25:]
val_examples = filtered_examples[:25]
print("Number of train examples:", len(train_examples))
print("Number of validation examples:", len(val_examples))
```

得到的训练示例和验证示例如下：

```
Number of train examples: 6670
Number of validation examples: 25
```

请注意，我们有意识地保持验证集很小，以免不必要地减慢训练循环。此外，在训练完成后，我们还将使用 MATH-500 的 500 个样本来评估模型的性能。

> **练习 8.1：训练集和验证集长度**
> 将 `compute_length` 函数应用于新的 `train_examples` 和 `val_examples` 分区，以检查它们是否基于样本长度保持平衡。

## 8.5 加载预训练模型

数据集准备完成后，我们可以转向实际的蒸馏训练。我们首先加载预训练的 Qwen3 基础模型，如图 8.11 所示。

图 8.11 数据集准备完成后，我们通过加载预训练的 Qwen3 基础模型开始蒸馏训练。

我们从预训练的 Qwen3 基础模型开始，而不是从 RL 训练过的模型开始，因为蒸馏本身是我们想要在这里研究的训练阶段。这保持了设置的简洁性，并使任何改进更容易归因于蒸馏的推理轨迹。它也反映了一种常见的实际设置，即一个通用的基础模型稍后使用教师生成的数据进行适配。

**代码清单 8.8 加载用于蒸馏的 Qwen3 基础模型**

```python
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer,
)

device = get_device()
model, _ = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False,
)
```

请注意，我们可以忽略代码清单 8.8 中加载的基础分词器，因为我们将使用之前在第 8.4.1 节中加载的带有 `<think></think>` token 支持的分词器。

## 8.6 计算训练和验证损失

接下来，我们实现交叉熵损失，它将在我们稍后实现训练循环时作为蒸馏期间的训练信号。我们还将重用相同的计算在验证集上监控训练期间的进度。

具有深度学习背景的读者可能已经熟悉分类任务中的交叉熵。这里的想法相同，只是我们在每个位置想要预测的目标类别是教师生成序列中的下一个 token。

我们可以直接将此损失与第 5 章和第 6 章中的 log-probability 计算联系起来。如那里所讨论的，log-probability 衡量模型为正确的下一个 token 分配了多少概率。更高的 log-probability 意味着模型对正确 token 更有信心，更低的 log-probability 则意味着相反。重用前面章节中的“The capital of Germany is Berlin”示例，这在图 8.12 中进行了回顾。

图 8.12 token 和序列 log-probabilities 的图示。正确下一个 token 的 log-probabilities 被求和以得到序列 log-probability，这是本章后面使用的交叉熵损失的基础。

交叉熵就是图 8.12 中所示的这些 token log-probabilities 的负平均值。例如，如果 -16.6250 是 log-probability，那么负 log-probability 是 16.6250，负平均 log-probability 是 16.6250/5 = 3.325。注意，这里的 5 是因为序列有 5 个目标预测，其 log-probabilities 正在被平均：“The”、“capital”、“of”、“Germany”、“is”和“Berlin”。

交叉熵也是 3.325。（与 log-probability 类似，值越接近 0 越好，因为模型对正确的目标 token 更有信心。）

第 6 章中的 `sequence_logprob` 函数执行了图 8.12 中所示的精确计算，这使其成为理解蒸馏损失如何工作的有用起点。在这种情况下，我们使用 train_examples 中索引位置为 5730 的训练示例，因为它是 train_examples 中最短的示例，我们可以通过 `compute_length(train_examples)` 确定，因此比其他示例计算得更快：

**代码清单 8.9 计算平均负 log-probability**

```python
from reasoning_from_scratch.ch06 import sequence_logprob

tok = torch.tensor(token_ids, dtype=torch.long, device=device)
with torch.no_grad():
    seq_logprob = sequence_logprob(model, tok, prompt_len)
num_answer_tokens = tok.numel() - prompt_len
avg_logprob = -seq_logprob / num_answer_tokens
print(f"Average logprob: {avg_logprob:.2f}")
```

得到的负平均 log-probability 是 1.68。

现在，我们可以使用 PyTorch 的 `cross_entropy` 函数计算相同的量。虽然该函数通常是为分类任务介绍的，但它是这里的自然选择。例如，模型 logits 提供了对词表的预测类别分布，而目标序列在每个位置提供了正确的类别标签。

与之前的平均 log-probability 计算一样，我们仅在答案 token 上计算损失，并忽略提示 token（提示是作为输入提供的上下文，我们不希望模型因复现它而受到惩罚）。这在图 8.13 中进行了说明。

图 8.13 答案 token 上的交叉熵损失输入。模型接收提示和答案偏移一个 token 作为输入，答案 token 的 logits 与参考答案 token 进行比较。

在蒸馏期间，我们希望学生模型学习教师模型的推理轨迹和最终答案（以提示为条件），因此，如图 8.13 所示，在代码清单 8.10 中计算交叉熵时，我们丢弃仅对应于序列提示部分的 logits 和目标：

**代码清单 8.10 直接计算交叉熵损失**

```python
# A 将序列偏移一个 token，使得每个输入预测下一个 token 目标
input_ids = tok[:-1].unsqueeze(0)
target_ids = tok[1:]
logits = model(input_ids).squeeze(0)

# B 丢弃提示位置，使损失仅覆盖教师答案
first_answer_logit_idx = max(prompt_len - 1, 0)
answer_logits = logits[first_answer_logit_idx:]
answer_targets = target_ids[first_answer_logit_idx:]

# C 计算交叉熵损失
with torch.no_grad():
    ce_mean_direct = torch.nn.functional.cross_entropy(
        answer_logits, answer_targets
    )
print(f"Cross-entropy: {ce_mean_direct:.2f}")
```

得到的交叉熵损失是 1.68，这与代码清单 8.9 中使用 `sequence_logprob` 函数时相似。换句话说，`cross_entropy` 以更优化的方式实现了相同的核心计算。因此，对于训练，我们使用内置的 `cross_entropy` 函数而不是我们自定义的 log-probability 函数。

下面的 `compute_example_loss` 辅助函数（代码清单 8.11）将此逻辑包装成一个方便的函数，用于计算单个示例的仅答案损失。“仅答案损失”意味着训练损失仅在教师模型的答案 token 上计算，而不是在提示 token 上。

我们关注仅答案损失，因为提示已经给出。模型在蒸馏中的工作不是学习重构输入指令。它的工作是在给定该指令的条件下产生目标答案。因此，我们使用提示作为上下文，但不对提示 token 预测惩罚模型。

现在，让我们将所有内容整合到一个函数中，该函数将目标准备到交叉熵计算的整个逻辑应用于给定训练示例，如代码清单 8.11 所示：

**代码清单 8.11 定义一个蒸馏示例的损失**

```python
def compute_example_loss(model, example, device):
    token_ids = example["token_ids"]
    prompt_len = example["prompt_len"]
    # A 创建偏移一个 token 的输入-目标对
    input_ids = torch.tensor(
        token_ids[:-1], dtype=torch.long, device=device
    ).unsqueeze(0)
    target_ids = torch.tensor(
        token_ids[1:], dtype=torch.long, device=device
    )
    logits = model(input_ids).squeeze(0)
    # B 忽略提示 token，使损失仅在蒸馏答案上计算
    answer_start = max(prompt_len - 1, 0)
    answer_logits = logits[answer_start:]
    answer_targets = target_ids[answer_start:]
    # C 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(
        answer_logits, answer_targets
    )
    return loss

# D 用于验证辅助函数是否返回与之前相同的损失
with torch.no_grad():
    loss = compute_example_loss(
        model, train_examples[5730], device
    )
print(f"Loss: {loss:.2f}")
```

得到的损失再次是 1.68，这表明代码清单 8.11 中的函数按预期工作。

> **批处理**
> 也可以通过将多个示例批处理在一起来并行处理。我们在这里省略批处理以保持实现紧凑并降低资源需求。附录 E 更详细地讨论了批处理和面向吞吐量的执行，包括损失计算和一般训练。

接下来，我们定义一个小包装器，它迭代地计算多个示例的平均损失。这对于快速健全性检查和跟踪训练期间的验证损失都很有用。

**代码清单 8.12 评估多个示例的损失**

```python
@torch.no_grad()
def evaluate_examples(model, examples, device):
    was_training = model.training
    # A 在评分示例时临时切换到评估模式
    model.eval()
    total_loss = 0.0
    num_examples = 0
    # B 对所有示例的损失求和
    for example in examples:
        loss = compute_example_loss(model, example, device)
        total_loss += loss.item()
        num_examples += 1
    # C 恢复训练模式，以便此辅助函数在训练期间调用是安全的
    if was_training:
        model.train()
    # D 对所有示例的平均损失
    return total_loss / num_examples

# E 在一小部分子集上估计当前训练损失
train_loss = evaluate_examples(model, train_examples[:3], device)
print(f"Train loss (3 examples): {train_loss:.2f}")
val_loss = evaluate_examples(model, val_examples[:3], device)
print(f"Validation loss (3 examples): {val_loss:.2f}")
```

输出为：

```
Train loss (3 examples): 0.98
Validation loss (3 examples): 1.02
```

我们将在训练期间重用此评估函数。理想情况下，这两个量都应该随时间下降，这表明学生模型在匹配教师生成的目标序列方面变得更好。

在实践中，训练损失（在用于优化的训练示例上计算的损失）通常更嘈杂，因为它是在当前正在优化的示例上测量的，可能会根据样本顺序和最近的参数更新而波动。

验证损失（在单独的留出验证集上测量，而不是在当前用于优化的示例上）是在不更新模型权重的情况下计算的。因此，它通常提供更清晰的信号，表明学生模型是否以超越训练集的方式改进。由于这个原因，验证损失通常是要关注的更可靠指标。

## 8.7 实现蒸馏的训练循环

有了数据集准备和损失计算，我们现在可以实现蒸馏的训练循环。这个阶段在图 8.14 中突出显示。

图 8.14 数据集准备和损失计算完成后，我们现在转向蒸馏的训练循环。

训练循环与第 6 章中的非常相似。主要区别是我们现在在多个 epoch 中多次 revisit 相同的训练集，并优化交叉熵蒸馏损失，而不是 RLVR 中使用的 GRPO 目标（GRPO 损失）。详细步骤如图 8.15 所示。

图 8.15 蒸馏训练循环。在每个 epoch 中，训练示例被打乱，学生模型为每个示例计算交叉熵损失，梯度被反向传播，模型权重被更新。验证损失在某些间隔报告以跟踪进度。

`train_distillation` 函数（代码清单 8.13）实现了图 8.15 中所示的循环。它在每个 epoch 开始时打乱训练示例，为每个示例计算损失，应用优化器步骤，可选地裁剪大梯度，并定期在验证集上评估。指标还会写入 CSV 文件，以便我们稍后检查学习曲线。

**代码清单 8.13 实现蒸馏训练循环**

```python
import time

def train_distillation(
    model,
    train_examples,
    val_examples,
    device,
    epochs=2,
    lr=5e-6,
    grad_clip_norm=None,
    seed=123,
    log_every=50,
    checkpoint_dir="checkpoints",
    csv_log_path=None,
):
    # A 初始化优化器（模型已加载）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    total_steps = epochs * len(train_examples)
    global_step = 0
    rng = random.Random(seed)
    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_distill_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)
    # B 迭代训练 epoch
    for epoch in range(1, epochs + 1):
        # C 在 epoch 开始时打乱训练示例
        epoch_examples = list(train_examples)
        rng.shuffle(epoch_examples)
        # D 迭代 epoch 中的训练示例
        for example in epoch_examples:
            global_step += 1
            # E 重置损失梯度
            optimizer.zero_grad()
            # F 为当前示例计算交叉熵损失
            loss = compute_example_loss(model, example, device)
            # G 反向传播梯度
            loss.backward()
            # H 可选地裁剪大梯度以提高训练稳定性
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm
                )
            # I 更新模型权重
            optimizer.step()
            # J 定期在验证集上评估当前模型
            if log_every and global_step % log_every == 0:
                val_loss = evaluate_examples(
                    model=model,
                    examples=val_examples,
                    device=device,
                )
                model.train()
                print(
                    f"[Epoch {epoch}/{epochs} "
                    f"Step {global_step}/{total_steps}] "
                    f"train_loss={loss.item():.4f} "
                    f"val_loss={val_loss:.4f}"
                )
                append_csv_metrics(
                    csv_log_path=csv_log_path,
                    epoch_idx=epoch,
                    total_steps=global_step,
                    train_loss=loss.item(),
                    val_loss=val_loss,
                )
        # K 记录指标并保存此 epoch 的检查点
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=global_step,
            suffix=f"epoch{epoch}",
        )
        print(f"Saved checkpoint to {ckpt_path}")
    return model

def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-distill-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path

def append_csv_metrics(
    csv_log_path,
    epoch_idx,
    total_steps,
    train_loss,
    val_loss,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "epoch,total_steps,train_loss,val_loss\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{epoch_idx},{total_steps},{train_loss:.6f},"
            f"{val_loss:.6f}\n"
        )
```

在最大序列长度为 2048 的情况下，完整的训练运行需要大约 15 GB 的 GPU 内存。如果这对你的硬件来说太高，你可以通过在笔记本中更早地过滤掉更长的序列来降低资源需求，例如在代码清单 8.6（第 8.4.3 节）的 `filter_examples_by_max_len` 步骤中将 `max_len` 从 2048 改为 1024 或 512。

让我们执行一个简短的训练运行：

**代码清单 8.14 训练模型**

```python
# A 设置 PyTorch 种子以使短演示可复现
torch.manual_seed(0)
# B 在一小部分子集上训练，以保持此笔记本运行轻量
train_distillation(
    model,
    train_examples=train_examples[:10],  # B
    val_examples=val_examples[:10],      # B
    device=device,
    epochs=2,
    lr=5e-6,
    grad_clip_norm=1.0,  # C
    seed=123,
    log_every=5,
    csv_log_path="train_distill_metrics.csv",
)
```

在检查结果之前，让我们简要谈谈这些设置。我们有意识地保持训练运行较短。例如，我们只使用 10 个训练示例和 10 个验证示例，并且只训练 2 个 epoch。这纯粹是为了保持笔记本运行轻量且足够快以进行实验。对于真正的蒸馏运行，我们当然会在更多示例上训练，理想情况下是整个训练集。

学习率（`lr=5e-6`）对于微调预训练模型来说是一个合理的范围，并且在此设置中以及在我的实验实践中效果良好，如接下来讨论的损失曲线所反映的。

梯度裁剪设置（`grad_clip_norm=1.0`）与第 6 章中的相同，有助于防止当特定示例产生异常大的梯度时出现不稳定的更新。

`log_every=5` 设置意味着每 5 个训练步骤测量一次验证损失。由于此演示仅使用少量示例，这会产生频繁的进度更新，以便我们可以快速验证训练循环是否按预期工作。在更大的运行中，我们通常会增加此间隔以减少评估开销。

最后，`csv_log_path="train_distill_metrics.csv"` 参数将训练和验证损失存储在 CSV 文件中，以便我们稍后检查和绘制它们。

此运行的主要目标不是实现最佳的推理性能，而是确认蒸馏管道端到端地工作。一旦确认，我们就可以转向更大的运行，并更详细地检查生成的学习曲线和检查点。

现在让我们简要看看运行的输出：

```
[Epoch 1/2 Step 5/20] train_loss=0.9648 val_loss=0.9082
[Epoch 1/2 Step 10/20] train_loss=0.9844 val_loss=0.8871
Saved checkpoint to checkpoints/qwen3-0.6B-distill-step00010-epoch1.pth
[Epoch 2/2 Step 15/20] train_loss=0.8008 val_loss=0.8707
[Epoch 2/2 Step 20/20] train_loss=0.7148 val_loss=0.8586
Saved checkpoint to checkpoints/qwen3-0.6B-distill-step00020-epoch2.pth
```

即使这只是一个微小的演示运行，输出已经显示了预期的整体行为。训练损失和验证损失都随时间下降，这表明学生模型在匹配教师生成的目标序列方面变得更好。验证损失在这里特别有用，因为它是在留出示例上计算的，因此提供了更清晰的信号，表明改进不仅限于训练样本。

我们还可以看到检查点在每个 epoch 结束时保存。这在实践中很有用，因为它允许我们稍后恢复训练或使用 MATH-500 测试集评估蒸馏模型的中间版本。

## 8.8 评估蒸馏后的模型

在实现训练循环之后，最后阶段是评估蒸馏后的模型，如图 8.16 所示。

图 8.16 实现训练循环后，我们在 MATH-500 测试集上评估蒸馏后的模型。

我们不必在此笔记本内运行完整的蒸馏过程，还可以从补充材料中下载一个便利脚本。如前一章关于 RLVR 所述，将笔记本专注于核心思想，并将长时间运行的训练作业移到独立脚本中通常很方便。

通过前面的代码下载脚本后，我们可以在代码终端中如下运行它（如果你不是 uv 用户，请将 `uv run` 替换为 `python`）：

```python
from reasoning_from_scratch.ch07 import download_from_github
download_from_github(
    "ch08/04_train_with_distillation/distill.py"
)
```

```bash
uv run distill.py \
  --data_path deepseek-r1-math-train.json \
  --validation_size 25 \
  --epochs 3 \
  --lr 1e-5 \
  --max_seq_len 2048 \
  --use_think_tokens \
  --grad_clip 1.0
```

使用这些设置，完整的训练运行在 DGX Spark 上大约需要 3 小时 5 分钟，使用大约 15.02 GB 的 GPU 内存。（与早期的 RLVR 运行相比，这相对适中，因为大部分昂贵的工作已经移到了一次性的教师数据生成步骤中。）如果你不想自己运行训练，我们可以下载生成的指标文件并直接检查它。

```python
download_from_github(
    "ch08/03_logs/deepseek-r1-2048_distill_metrics.csv"
)
```

代码清单 8.15 实现了一个实用函数，用于可视化存储在 CSV 文件中的训练指标：

**代码清单 8.15 从 CSV 日志绘制蒸馏损失**

```python
import csv
import matplotlib.pyplot as plt

def plot_distill_metrics(csv_path="train_distill_metrics.csv"):
    total_steps, train_losses, val_losses, epoch_bounds = [], [], [], {}
    # A 加载并绘制记录的损失
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["total_steps"])
            epoch = int(row["epoch"])
            total_steps.append(step)
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            epoch_bounds.setdefault(epoch, [step, step])[1] = step
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(total_steps, train_losses, label="train_loss", alpha=0.3)
    ax.plot(total_steps, val_losses, label="val_loss")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    # B 添加第二个 x 轴，以便 epoch 编号在 step 轴下方可见
    epoch_axis = ax.secondary_xaxis("bottom")
    epoch_axis.spines["bottom"].set_position(("outward", 45))
    epochs = sorted(epoch_bounds)
    epoch_axis.set_xticks(
        [
            (epoch_bounds[epoch][0] + epoch_bounds[epoch][1]) / 2
            for epoch in epochs
        ]
    )
    epoch_axis.set_xticklabels(epochs)
    epoch_axis.set_xlabel("Epoch")
    plt.tight_layout()
    plt.show()

plot_distill_metrics("deepseek-r1-2048_distill_metrics.csv")
```

得到的图如图 8.17 所示。

图 8.17 在 DeepSeek-R1 推理轨迹上进行 3 个 epoch 的蒸馏训练运行的训练和验证损失。

图 8.17 中所示的训练损失曲线和验证损失曲线应该略有不同的解释。训练损失是在训练期间当前步骤为每个训练示例计算的，而验证损失是在一小部分固定的留出集上计算的。因此，验证曲线噪音更小，是这里更有信息的信号。我们也可以在一部分示例上计算训练损失，类似于验证损失，但这会减慢训练速度；训练损失对于估计训练进度远不如验证损失重要。

正如我们所希望的，验证损失首先急剧下降，然后开始趋于平缓，表明模型正在从蒸馏数据中学习，但额外的训练带来的收益递减。我们可以尝试更大的学习率或不同的调度来更积极地训练，但总体而言曲线看起来是健康的。

正如在第 7 章中一样，我们可以使用第 3 章中基于验证器的实用程序评估保存的检查点。以下命令从补充材料下载评估脚本：

```python
download_from_github(
    "ch03/02_math500-verifier-scripts/evaluate_math500.py"
)
```

接下来，我们在代码终端中使用推理分词器在 MATH-500 上评估蒸馏后的检查点，如下所示：

```bash
uv run evaluate_math500.py \
  --dataset_size 500 \
  --which_model reasoning \
  --max_new_tokens 4096 \
  --checkpoint_path \
  "run_11/checkpoints/distill/qwen3-0.6B-distill-step06682-epoch1.pth"
```

对于同一 DeepSeek-R1 运行的后续检查点，将 `...step06682-epoch1.pth` 分别替换为 `...step13364-epoch2.pth` 和 `...step20046-epoch3.pth`。

本章使用的 DeepSeek-R1 运行的评估结果总结在表 8.1 中。作为参考，我还包括了在 Qwen3 235B-A22B 教师输出上训练的第二次运行（一个 2350 亿参数的 Qwen3 模型）。

**表 8.1 不同模型检查点在 MATH-500 任务上的准确率**

| 方法 | Epoch | 最终验证损失 | MATH-500 准确率 |
|------|-------|-------------|----------------|
| 1 Qwen3 0.6B 基础模型（第 3 章） | - | - | 15.2% |
| 2 Qwen3 0.6B 推理模型（第 3 章） | - | - | 48.2% |
| 3 DeepSeek-R1 | 1 | 0.5436 | 30.6% |
| 4 DeepSeek-R1 | 2 | 0.5349 | 32.4% |
| 5 DeepSeek-R1 | 3 | 0.5343 | 33.6% |
| 6 Qwen3 235B-A22B | 1 | 0.4043 | 45.0% |
| 7 Qwen3 235B-A22B | 2 | 0.3963 | 43.8% |
| 8 Qwen3 235B-A22B | 3 | 0.3948 | 44.2% |

根据表 8.1 中所示的结果，对于 DeepSeek-R1 运行，我们看到 MATH-500 准确率从第一个 epoch 后的 30.6% 提高到第三个 epoch 后的 33.6%，而验证损失从 0.5436 下降到 0.5343。这与之前基于图 8.17 的学习曲线讨论相符，学生模型明显从教师生成的推理轨迹中学习，但初始改进后的收益开始逐渐减少。

Qwen3 235B-A22B 运行在此设置中表现明显更好。一个可能的原因是教师和学生来自同一模型家族，这意味着分词器、提示约定和整体响应风格更加一致。这可以使教师目标对较小的 Qwen3 学生模型来说更容易模仿。

考虑到 MATH-500 准确率为 45%（第 6 行），我们的蒸馏配方达到了与推理参考模型（48.2%，第 2 行）几乎相同的性能，而推理参考模型本身也是一个蒸馏模型，但它在由 Qwen3 235B-A22B 生成的更大的数据集上训练。这就是蒸馏背后的主要权衡。

请注意，一般来说，我们不期望较小的学生模型完全匹配教师模型。例如，Qwen3 235B-A22B 的 MATH-500 准确率为 92.4%，DeepSeek-R1 的 MATH-500 准确率为 91.2%。但我们仍然可以在一个便宜得多的模型中恢复教师模型推理行为的有用部分。另外，请注意，如果我们使用更大的模型（例如，Qwen3 的 40 亿或 300 亿参数版本而不是 Qwen3 0.6B），我们的蒸馏模型准确率可能会更高，但这会增加训练期间的计算成本。

让我们退一步，将我们实现的各个部分连接成一个完整的工作流程。图 8.18 总结了完整的蒸馏管道，从介绍性设置开始，然后是数据集生成和预处理，接着是蒸馏训练循环本身，最后是在 MATH-500 上评估蒸馏后的模型。

图 8.18 蒸馏模型的评估完成了本章的技术内容。

图 8.18 中所示的工作流程完成了从更强的教师模型蒸馏较小推理模型的核心技术配方。在实践中，每个阶段都提供了变化的空间，例如改变教师数据的生成方式以及修改训练设置。

> **练习 8.2：不使用 `<THINK>` token 进行蒸馏**
> 重复蒸馏实验，不使用 `--use_think_tokens`，并将结果与使用包裹在 `<think>...</think>` 中的推理轨迹训练的版本进行比较。检查验证损失，如果可能，在 MATH-500 上评估保存的检查点。然后将结果与表 8.1 中列出的结果进行比较。在此设置中，显式推理标签有多重要？

## 8.9 推理模型的未来方向

在结束本章之前，让我们讨论推理模型下一步的发展方向。

在可预见的未来，整体广泛模式保持不变。例如，一般策略是开发一个更强的推理教师模型，收集高质量的推理轨迹，并将它们蒸馏到较小的学生模型中。

一个明显的方向是继续完善 DeepSeek-R1 论文推广的训练配方，这包括用于旗舰模型的 RLVR（第 6 章和第 7 章）以及面向计算效率的较小模型的蒸馏。

第二个方向是推理时的优化。在实践中，推理模型不应该总是产生同样长的答案。一些任务受益于简短、直接的响应，而另一些任务则受益于更详细的多步推理。这在应用层创造了更多自动和灵活的推理扩展空间，周围的系统决定何时要求简短答案、何时分配更多推理预算以及何时提前停止。例如，OpenAI 在 2025 年推出 GPT-5 时实现了这样一个系统，他们添加了一个“auto”模式来引导推理努力和推理轨迹生成长度。

第三个改进方向是 RLVR 的奖励生成。当前的大部分工作仍然严重依赖基于最终答案的奖励，尤其是在数学和代码领域。但检查中间推理步骤而不仅仅是最终结果的过程奖励可能会提供更丰富的训练信号，并帮助模型学习更可靠的推理策略。例如，DeepSeek-Math-V2 论文（https://arxiv.org/abs/2511.22570）最近证明，在训练期间判断整个答案可以有意义地提高推理性能。

第四个方向是推理模型在更大的智能体应用（如 OpenAI Codex、Claude Code 和 OpenClaw）中作为引擎的角色日益增强。在这些设置中，模型不仅要解决简单的数学或编码问题，还要规划、调用工具、从失败中恢复以及协调更长的工作流程。这自然将奖励设计推向了数学和代码正确性之外。我们可能希望为成功的工具使用、信息检索、策略合规性等提供奖励。反过来，这导致了多奖励训练，其中多个目标一起优化，而不是依赖单一的 correctness 分数。

蒸馏在这种情况下可能仍然很重要，因为它提供了一种实用的方式，将这些更丰富的行为从更大、更昂贵的教师系统转移到更小、更容易部署甚至可以在本地运行的模型，同时对用户来说更具成本效益。

## 8.10 结论

这完成了本书的主要技术材料。剩余部分是关于接下来尝试什么、如何跟上快速发展的领域以及在哪里找到额外材料的简要指引。

### 8.10.1 接下来做什么

一个实际的下一步是开始结合本书中的方法，而不是将它们视为孤立的技术。例如，你可以从强大的教师模型蒸馏一个较小的模型，继续使用 RLVR 训练它，然后在部署时应用推理时扩展方法，如自一致性或自精炼。运行这类比较通常是培养哪种方法在给定设置中最有帮助的直觉的最快方式。

附录也是继续学习的好地方。它们涵盖了 LLM 架构细节、用于更高吞吐量的批处理执行以及替代评估方法等额外主题。

最后，补充代码仓库（https://github.com/rasbt/reasoning-from-scratch）包括 bonus 材料和独立脚本，这些脚本比笔记本更适合长时间运行和更大的实验。

### 8.10.2 在快速发展的领域中保持最新

我希望这本书让你对现代推理模型在实践中如何工作有了更清晰的认识！

推理模型研究发展迅速，具体的算法、数据集和最佳实践将继续变化。本书中的核心思想往往保持相关和有用。这包括仔细的模型评估、区分推理时方法和训练时方法，以及对训练损失和奖励信号如何计算的扎实理解。

当出现新方法时，将它们映射回这些基础是有帮助的。许多新技术最好理解为我们在这里实现的构建块的变体或组合，即使周围的训练配方变得更加复杂。

在实践中，我推荐几个资源。首先，你可以浏览 arXiv（https://arxiv.org/list/cs.LG/recent）上最近的机器学习和 AI 论文，以尽早发现新的训练和评估想法。然而，请注意，新论文的数量现在已经如此之大，以至于全面跟上几乎是不可能的，因此通常最好将 arXiv 视为扫描主题和有前途的论文的方式，而不是试图阅读所有内容。

其次，阅读新发布模型的技术摘要和报告，因为它们通常包含关于提示约定、基准测试和限制的最有用细节。这些通常由开发者在 Hugging Face 模型中心、公告博客文章和社交媒体上直接分享。

第三，关注社交媒体平台 X 和 r/LocalLLaMA（https://www.reddit.com/r/LocalLLaMA/）等社区中的从业者讨论，实现细节、复现和失败案例通常在它们进入 polished 论文之前就会出现。

AI 助手或“深度研究”工具对于监控这些来源和总结新发展也很有用，但它们最好作为过滤器而不是阅读原始论文和模型卡的替代品。

我还定期在我的博客 https://magazine.sebastianraschka.com 上撰写关于 AI 和 LLM 主题的文章。

## 8.11 总结

- 蒸馏在一个较小的学生 LLM 上训练，使用一个较大的教师 LLM 产生的输出。
- 硬蒸馏通常比软蒸馏更实用，因为教师模型的 logits 通常不可用，而教师模型的文本输出存储和重用要便宜得多。
- 我们使用 DeepSeek-R1 作为教师模型，Qwen3 0.6B 作为学生模型。
- 蒸馏数据集由与 MATH-500 不重叠的 12,000 道 MATH 训练题构建。
- 每个训练样本将渲染的提示与教师推理轨迹和最终答案组合，可选地通过 `<think>...</think>` 标签分隔。
- 为了效率，我们对数据集进行一次分词，按序列长度过滤，并在多个 epoch 中重用处理后的示例。
- 训练目标是仅答案交叉熵，它等价于正确下一个 token 的负平均 log-probability。
- 蒸馏训练循环是一个标准的监督学习循环，包括每个 epoch 打乱训练示例、计算损失、反向传播、更新模型权重以及跟踪验证损失。
- 验证损失是训练期间要关注的主要信号，保存的检查点可以稍后使用第 3 章的验证器在 MATH-500 上进行评估。
- 本章中的蒸馏方法将 Qwen3 0.6B 基础模型在 MATH-500 上的准确率从 15.2% 提高到 45.0%。

---

# 附录 A 参考文献与延伸阅读

## A.1 第 1 章：理解推理模型

### A.1.1 参考文献

OpenAI o1 模型的公告文章，它被视为第一个基于 LLM 的推理模型：
- Introducing OpenAI o1-preview, https://openai.com/index/introducing-openai-o1-preview/

DeepSeek-R1 是第一个附带全面技术报告的开源推理模型，该报告首次表明推理能力可以通过可验证奖励的强化学习涌现（第 5 章更详细地介绍了这一主题）：
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, https://arxiv.org/abs/2501.12948

OpenAI CEO 对未来模型推理（“思维链”）能力的评论：
- "[...] We will next ship GPT-4.5, the model we called Orion internally, as our last non-chain-of-thought model. [...]", https://x.com/sama/status/1889755723078443244

Apple AI 研究人员的一篇研究论文，发现推理模型是复杂的（但非常有能力的）模式匹配器：
- The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity, https://machinelearning.apple.com/research/illusion-of-thinking

一本关于逐步实现和训练大型语言模型的深入书籍和指南：
- Build a Large Language Model (From Scratch), http://mng.bz/orYv

### A.1.2 延伸阅读

一篇介绍 DeepSeek-R1 如何工作的文章，提供了对 LLM 中推理基础的洞察：
- Understanding Reasoning LLMs, https://magazine.sebastianraschka.com/p/understanding-reasoning-llms

## A.2 第 2 章：使用预训练 LLM 生成文本

### A.2.1 参考文献

uv Python 包和项目管理器的官方安装页面：
- Installing uv, https://docs.astral.sh/uv/getting-started/installation/

用户友好且流行的支持 GPU 的云计算平台：
- Lightning AI, https://lightning.ai/
- Google Colab, https://colab.research.google.com/

Qwen3 资源，包含额外的基准性能和其他模型的比较：
- Blog post, https://qwenlm.github.io/blog/qwen3/
- Technical report, https://arxiv.org/abs/2505.09388

对不同序列长度的 KV cache 大小感到好奇的读者可以在这里找到一个方便的计算器应用：
- KV cache size calculator, https://lmcache.ai/kv_cache_calculator.html

### A.2.2 延伸阅读

一篇面向 PyTorch 新读者或希望复习的读者的 PyTorch 教程：
- PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs tutorial, https://sebastianraschka.com/teaching/pytorch-1h

关于分词的额外资源：
- Build a Large Language Model (from Scratch) chapter 2, https://mng.bz/M96o
- Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch, https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

对于对更深入的 PyTorch 覆盖感兴趣的读者（可选），我可以推荐以下两本书：
- Deep Learning with PyTorch, https://www.manning.com/books/deep-learning-with-pytorch-second-edition
- Machine Learning with PyTorch and Scikit-Learn, https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning/dp/1801819319/

## A.3 第 3 章：评估推理模型

### A.3.1 参考文献

MATH-500 数据集源自 MATH 数据集（包含 12,500 道涵盖代数、几何、概率、数论等的问题），该数据集在以下论文中介绍：
- Measuring Mathematical Problem Solving With the MATH Dataset, https://arxiv.org/abs/2103.03874

MATH-500 分割（从原始 MATH 数据集创建）在以下论文中提出：
- Let's Verify Step by Step, https://arxiv.org/abs/2305.20050

### A.3.2 延伸阅读

对 SymPy（用于数学和符号计算的 Python 库，本书不需要）感兴趣的读者可以在这个官方教程中了解它：
- SymPy introductory tutorial, https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html

一个系统（这里是一个微调的 LLM）也评估中间推理步骤的示例：
- Evaluating Mathematical Reasoning Beyond Accuracy, https://arxiv.org/pdf/2404.05692

一个包含 800,000 个步骤级正确性标签的大规模数据集，用于模型生成的 MATH 数据集问题解答：
- Let's Verify Step by Step, https://arxiv.org/abs/2305.20050

一篇描述 LLM 评估成本上升的文章，发现在七个流行基准上评估 o1 等推理模型大约花费 1500 美元：
- The rise of AI "reasoning" models is making benchmarking more expensive, https://techcrunch.com/2025/04/10/the-rise-of-ai-reasoning-models-is-making-benchmarking-more-expensive/

一篇关于 LLM 基准测试的综合 2025 年综述：
- A Survey on Large Language Model Benchmarks, https://arxiv.org/abs/2508.15361

一个最近的研究项目强调，小型推理模型本身可以成功地用作其他推理模型的验证器，而不是仅依赖确定性和符号验证器：
- xVerify: Efficient Answer Verifier for Reasoning Model Evaluations, https://arxiv.org/abs/2504.10481

## A.4 第 4 章：通过推理时扩展改进推理

### A.4.1 参考文献

以下论文正式描述了思维链提示。请注意，该论文建议“Let's think step by step”作为提示修改。然而，在我的实验中，我发现当使用 Qwen3 基础模型时，“Explain step by step”表现更好，这就是我们在第 4 章中使用后者的原因。
- Large Language Models are Zero-Shot Reasoners, https://arxiv.org/abs/2205.11916

对自一致性采样及其额外比较研究的描述：
- Self-Consistency Improves Chain-of-Thought Reasoning in Language Models, https://arxiv.org/abs/2203.11171

### A.4.2 延伸阅读

对额外推理扩展方法的概述和讨论：
- The State of LLM Reasoning Model Inference, https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling

## A.5 第 5 章：通过自精炼进行推理时扩展

### A.5.1 参考文献

Google 对其专有 Gemini 3 模型背后的方法保密，但基于最近的公告，我们可以推测它使用了类似于自一致性或 Best-of-N 的推理扩展技术：“We’re pushing the boundaries of intelligence even further with Gemini 3 Deep Think. This mode meaningfully improves reasoning capabilities by exploring many hypotheses simultaneously to solve problems.”
- Public announcement by Google DeepMind, which develops Gemini, https://x.com/GoogleDeepMind/status/1996658401233842624?s=20

DeepSeekMath-V2 论文表明，自一致性扩展可以显著提高答案准确率，并且结合自一致性与他们版本的自精炼（图 2 中的 Best@32），该模型在几个数学竞赛中达到了金牌水平：
- DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning, https://arxiv.org/abs/2511.22570v1

在自一致性中，我们可以使用评分函数对不同答案进行排名并选择最佳答案，而不是使用多数投票。这种方法也称为 Best-of-N。然而，如果适用，多数投票往往能产生更好的结果：
- Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-time-scaling Methods, https://arxiv.org/abs/2504.14047

### A.5.2 延伸阅读

一篇解释概率与似然之间差异的短文：
- What is the difference between likelihood and probability?, https://sebastianraschka.com/faq/docs/probability-vs-likelihood.html

## A.6 第 6 章：使用强化学习训练推理模型

### A.6.1 参考文献

InstructGPT 论文展示了 RLHF 的有效性，并在普及 RLHF 作为 LLM 的标准对齐和微调方法方面发挥了重要作用：
- Training language models to follow instructions with human feedback, https://arxiv.org/abs/2203.02155

介绍 GRPO 算法的 DeepSeekMath 论文：
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, https://arxiv.org/abs/2402.03300

DeepSeek-R1 论文表明，强大的推理行为可以仅通过强化学习（通过使用 GRPO 的 RLVR）在 LLM 中涌现。这在 R1-Zero 变体中最为清楚地展示。然而，将这种方法与多阶段训练管道结合会产生更好的推理模型：
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, https://arxiv.org/abs/2501.12948

### A.6.2 延伸阅读

对涉及 RLVR 的 DeepSeek-R1 训练管道的全面讲解：
- Understanding Reasoning LLMs: Methods and Strategies for Building and Refining Reasoning Models, https://magazine.sebastianraschka.com/p/understanding-reasoning-llms

在 LLM 背景下对 GRPO 和 PPO 进行强化学习的比较：
- The State of Reinforcement Learning for LLM Reasoning: Understanding GRPO and New Insights from Reasoning Model Papers, https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training

## A.7 第 7 章：改进 GRPO 以进行强化学习

### A.7.1 参考文献

介绍裁剪策略比率的原始 PPO 论文，我们在这里也使用它来稳定 GRPO：
- Proximal Policy Optimization Algorithms, https://arxiv.org/abs/1707.06347

推荐改进 GRPO 算法的额外论文：
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale, https://arxiv.org/abs/2503.14476
- Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO), https://arxiv.org/abs/2503.20783
- Your Efficient RL Framework Secretly Brings You Off-Policy RL Training (VERL), https://fengyao.notion.site/off-policy-rl
- DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models, https://arxiv.org/abs/2512.02556
- GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization, https://arxiv.org/abs/2601.05242
- Group Sequence Policy Optimization (GSPO), https://arxiv.org/abs/2507.18071
- MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (CISPO), https://arxiv.org/abs/2506.13585

### A.7.2 延伸阅读

PPO（用于 RLHF 的原始算法）和 GRPO 之间的比较：
- The State of Reinforcement Learning for LLM Reasoning, https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training

一篇讨论不同 GRPO 改进的技术深入文章：
- GRPO++: Tricks for Making RL Actually Work, https://cameronrwolfe.substack.com/p/grpo-tricks

## A.8 第 8 章：蒸馏推理模型以实现高效推理

### A.8.1 参考文献

普及硬蒸馏和软蒸馏目标结合的经典知识蒸馏论文：
- Distilling the Knowledge in a Neural Network, https://arxiv.org/abs/1503.02531

描述推理蒸馏配方的 DeepSeek-R1 论文，这启发了本章，其中一个大型教师模型生成推理轨迹，然后用于训练较小的学生模型：
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, https://arxiv.org/abs/2501.12948

一篇关于蒸馏大型语言模型的论文，报告了精心设计的软蒸馏目标的强结果：
- MiniLLM: Knowledge Distillation of Large Language Models, https://arxiv.org/abs/2306.08543

### A.8.2 延伸阅读

关于监督微调（硬蒸馏中的技术）以及处理批量训练示例时掩码的更多细节：
- Build A Large Language Model (From Scratch) chapter 7, https://www.manning.com/books/build-a-large-language-model-from-scratch

对推理模型训练管道（包括蒸馏和 RLVR）的实用讲解：
- Understanding Reasoning LLMs: Methods and Strategies for Building and Refining Reasoning Models, https://magazine.sebastianraschka.com/p/understanding-reasoning-llms

## A.9 附录 F：LLM 评估的常见方法

### A.9.1 参考文献

介绍流行的多项选择 MMLU 数据集的论文：
- Measuring Massive Multitask Language Understanding, https://arxiv.org/abs/2009.03300

对 Elo 评分系统的详细描述：
- Elo rating system, https://en.wikipedia.org/wiki/Elo_rating_system

描述流行 LLM 排行榜背后原始方法的 Chatbot Arena 论文：
- Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference, https://arxiv.org/abs/2403.04132

### A.9.2 延伸阅读

一篇讨论 LM Arena 等排行榜问题的论文：
- The Leaderboard Illusion, http://arxiv.org/abs/2504.20879

作者更详细描述 gpt-oss 的文章：
- From GPT-2 to gpt-oss: Analyzing the Architectural Advances, https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the

对不同 LLM 评判方法的综述：
- A Survey on LLM-as-a-Judge, https://arxiv.org/abs/2411.15594

一个微调为评判者的小型 LLM 示例：
- PHUDGE: Phi-3 as Scalable Judge, https://arxiv.org/abs/2405.08029

---

# 附录 B 练习解答

练习解答的完整代码示例可以在补充 GitHub 仓库中找到：https://github.com/rasbt/reasoning-from-scratch。

## B.1 第 2 章

**练习 2.1：编码未知单词**

我们可以使用一个类似于 "Hello, Ardwarklethyrx. Haus und Garten." 的提示，它包含一个虚构的单词（"Ardwarklethyrx"）和三个非英语语言的单词（德语）：

```python
prompt = "Hello, Ardwarklethyrx. Haus und Garten."
input_token_ids_list = tokenizer.encode(prompt)
for i in input_token_ids_list:
    print(f"{[i]} --> {tokenizer.decode([i])}")
```

输出为：

```
[9707] --> Hello
[11] --> ,
[1644] -->  Ar
[29406] --> dw
[838] --> ark
[273] --> le
[339] --> th
[10920] --> yr
[87] --> x
[13] --> .
[47375] -->  Haus
[2030] -->  und
[93912] -->  Garten
[13] --> .
```

如我们所见，未知单词被分解成更小的子词片段甚至单个 token；这允许分词器和 LLM 处理任何输入。

德语单词在这里没有被分解成字符甚至子词，这表明分词器在训练期间见过德语文本。这也表明 LLM 很可能也在德语文本上训练过，应该能够很好地处理至少某些非英语语言。

**练习 2.2：在非 CPU 设备上重新运行代码**

我们可以简单地删除第 2.5 节中的 `device = torch.device("cpu")` 行，然后按原样重新运行第 2 章的其余代码。我在表 2.1 末尾提供了我尝试过的硬件的参考编号。

## B.2 第 3 章

**练习 3.1：添加更多测试用例**

我们可以添加无数不同的测试用例。下面是一些有趣的案例：

```python
from reasoning_from_scratch.ch03 import (
    run_demos_table
)

more_tests = [
    ("check_17", "[1, 2]", "(1, 2)", True),  # A 不同的括号类型
    ("check_18", "1e-3", "0.001", True),  # B 科学记数法
    ("check_19", "(-3)^2", "9", True),  # C 带插入符号指数的代数简化
    ("check_20", "−1", "-1", True),  # D Unicode 减号 (U+2212) 与 ASCII 连字符减号
]
run_demos_table(more_tests)
```

输出为：

```
Test     | Expect | Got   | Status
check_17 | True   | True  | PASS
check_18 | True   | True  | PASS
check_19 | True   | True  | PASS
check_20 | True   | False | FAIL
```

如我们所见，测试在所有情况下都通过了，除了 check_20，它将常规符号与 Unicode 版本的减号互换，这在人眼看来是无法区分的（取决于我们使用的字体或编辑器）。我们可以通过向 `normalize_text` 函数添加以下任一行来修复此测试用例：

```python
text = text.replace("−", "-")
# 或
text = text.replace("−", "-")
```

另一个有趣的测试如下：

```python
extra_tests_1 = [
    ("check_21", "Text around answer 3.", "3", True)
]
run_demos_table(extra_tests_1)
```

我们可以通过以下代码运行它：

```python
from reasoning_from_scratch.ch03 import (
    extract_final_candidate
)
extra_tests_2 = [
    ("check_21",
    extract_final_candidate("Text around answer 3."),
    "3", True)
]
run_demos_table(extra_tests_2)
```

然而，它失败了测试：

```
Test     | Expect | Got   | Status
check_21 | True   | False | FAIL  
Passed 0/1
```

虽然看起来我们的代码无法处理这种包含文本的案例，但这实际上是一个设计不佳的测试。在实践中，`run_demos_table` 函数专门用于测试 `grade_answer` 函数；仅此而已。

`grade_answer` 函数永远不会以这种文本形式接收整个答案，因为答案会在传递给 `grade_answer` 之前从文本中提取出来。例如，如果我们想测试文本答案，我们需要按如下方式调用测试：

```python
from reasoning_from_scratch.ch03 import (
    extract_final_candidate
)
extra_tests_2 = [
    ("check_21",
    extract_final_candidate("Text around answer 3."),
    "3", True)
]
run_demos_table(extra_tests_2)
```

如基于输出的所见，它现在通过了测试：

```
Test     | Expect | Got  | Status
check_21 | True   | True | PASS  
Passed 1/1
```

**练习 3.2：计算平均响应长度**

有两种计算平均响应长度的选项。第一种选项是通过添加以下行来修改 `evaluate_math500_stream` 函数（第 3 章中的代码清单 3.13）：

```python
# ...
# 在 `num_correct = 0` 下方
total_len = 0
# ...
# 在 for i, row in enumerate(math_data, start=1): 内部
# 在 `gen_text = ...` 下方的任何位置
total_len += len(tokenizer.encode(gen_text))
# ...
# 在 return 语句之前的底部任何位置
avg_len = total_len / num_examples
print(f"Average length: {avg_len:.2f} tokens")
```

第二种选项是从我们在第 3 章主章节中运行 `evaluate_math500_stream` 函数时创建的 `.jsonl` 文件中计算响应长度。这样，我们避免了重新运行评估。

首先，我们加载 `.jsonl` 文件如下：

```python
import json
from pathlib import Path

WHICH_MODEL = "base"
dev_name = "mps"
local_path = Path(f"math500-{dev_name}.jsonl")  # A
if not local_path.exists():
    raise FileNotFoundError(
        f"{local_path} not found. Run ch03_main.ipynb to create it."
    )
results = []
with open(local_path, "r") as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))
print("Number of entries:", len(results))
```

这打印：

```
Number of entries: 500
```

请注意，每个条目都有多个键，然而，我们只对 `"generated_text"` 键感兴趣，它包含模型的完整答案。接下来，我们需要加载分词器，以便在计算 token 数量之前对答案文本进行分词。这类似于我们在第 3 章代码清单 3.1 中使用的代码：

```python
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer
)

if WHICH_MODEL == "base":
    download_qwen3_small(
        kind="base", tokenizer_only=True, out_dir="qwen3"
    )
    tokenizer_path = Path("qwen3") / "tokenizer-base.json"
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
elif WHICH_MODEL == "reasoning":
    download_qwen3_small(
        kind="reasoning", tokenizer_only=True, out_dir="qwen3"
    )
    tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )
```

然后，我们可以如下计算平均长度，这类似于我们可以如何修改 `evaluate_math500_stream` 函数：

```python
total_len = 0
for item in results:
    num_tokens = len(tokenizer.encode(item["generated_text"]))
    total_len += num_tokens
avg_len = total_len / len(results)
print(f"Average length: {avg_len:.2f} tokens")
```

得到的平均长度如下：

```
Average length: 98.00 tokens
```

表 B.1 列出了不同模型和子集的平均长度。

**表 B.1 MATH-500 上的平均 token 数量**

| 模型 | 设备 | 平均长度 | MATH-500 大小 |
|------|------|---------|--------------|
| Base | CPU | 97.30 | 10 |
| Base | CUDA | 96.74 | 500 |
| Reasoning | CPU | 891.80 | 10 |
| Reasoning | CUDA | 1361.21 | 500 |

如基于表 B.1 中结果的所见，并且如预期的那样，推理模型生成的响应要长得多（在这种情况下，大约长 10 倍）。

**练习 3.3：扩展或更改评估数据集**

要在更大的数据集上评估模型，我们可以简单地将 `math_data[:10]` 更改为不同的切片或更大的数字（最多 500），在以下函数调用中：

```python
num_correct, num_examples, acc = evaluate_math500_stream(
    model, tokenizer, device, 
    math_data=math_data[:10],
    max_new_tokens=2048,
    verbose=False
)
```

表 B.2 下面显示了不同数据集大小的准确率值。（由于 MATH-500 测试集已经打乱，没有应用额外的打乱。）

**表 B.2 不同 MATH-500 数据集大小的准确率**

| 模型 | 设备 | 准确率 | MATH-500 大小 |
|------|------|-------|--------------|
| Base | CUDA | 30.0% | 10 |
| Base | CUDA | 34.0% | 50 |
| Base | CUDA | 27.0% | 100 |
| Base | CUDA | 15.3% | 500 |
| Reasoning | CUDA | 90.0% | 10 |
| Reasoning | CUDA | 58.0% | 50 |
| Reasoning | CUDA | 56.0% | 100 |
| Reasoning | CUDA | 48.2% | 500 |

如基于表 B.2 中结果的所见，前 10 个示例对于在全部 500 个示例上评估的 MATH-500 性能来说并不非常具有代表性。

此外，我们可以创建一个与 MATH-500 风格相似的全新数据集。例如，此仓库中包含一个 MATH-500 风格的数据集；我们可以在主章节中通过将文件名从 `math500_test.json` 更改为 `math_new50_exercise.json` 来使用它（此数据集包含在本书的 GitHub 仓库中：https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03/01_main-chapter-code）。

模型的性能如下：
- base: 36.0% (18/50)
- reasoning: 80.0% (40/50)

准确率对于基础模型与 MATH-500 测试集的 50 示例子集（表 B.2）相似，对于推理模型更高。这表明，尽管可能与 Qwen3 的训练数据有重叠，但模型很好地泛化到新的数学问题，并且没有显示出对原始 MATH-500 数据的广泛过拟合迹象。

**练习 3.4：尝试不同的提示模板**

我们可以使用本章中建议的替代提示，将提示修改为使用单词 "problem" 而不是 "question"：

```python
def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Solve the problem and write the final "
        "result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Problem:\n{prompt}\n\nAnswer:"
    )
    return template
```

使用此提示将基础模型在 500 个示例上的性能从 15.3% 提高到 31.2%。此外，它将推理模型的性能从 48.2% 提高到 50.0%。

从这些观察中，我们可以得出结论，基础模型对提示格式更敏感（可能是由于从训练集中记忆了一些提示格式化的 MATH-500 示例），而推理模型似乎基本不受影响。

## B.3 第 4 章

**练习 4.1：在 MATH-500 上使用思维链提示**

修改只需要在应用提示模板后添加一个提示后缀，例如 `"\n\nExplain step by step."`。只需要更新第 3 章中 MATH-500 评估函数的很小一部分代码，如下所示：

```python
def evaluate_math500_stream(...):
    # ...
    for i, row in enumerate(math_data, start=1):
        prompt = render_prompt(row["problem"])
        prompt += "\n\nExplain step by step."  # NEW
        gen_text = generate_text_stream_concat(
            model, tokenizer, prompt, device,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )
    # ...
```

改进显示在第 4 章表 4.1 的第 3 行中，可以在第 4 章第 4.6 节中找到。

**练习 4.2：在 MATH-500 上使用温度缩放和 Top-P 过滤**

在这里，我们将 `generate_text_stream_concat` 函数替换为 `generate_text_stream_concat_flex`，并将 `generate_text_top_p_stream_cache` 作为其生成函数传入。第 3 章中更新的 MATH-500 评估函数如下所示，更改用标记为 # NEW 的注释标注。

```python
def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
    temperature=1.0,  # NEW
    top_p=1.0,        # NEW
):
    # ...
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])
            gen_text = generate_text_stream_concat_flex( # NEW
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
                generate_func=generate_text_top_p_stream_cache,  # NEW
                temperature=temperature,                         # NEW
                top_p=top_p                                      # NEW
            )
        # ...
```

此修改函数与第 3 章基线之间的差异可以在第 4 章表 4.1 的第 1 行和第 4 行中看到，可以在第 4 章第 4.6 节中找到。

**练习 4.3：在 MATH-500 上使用自一致性采样**

从第 3 章的 `evaluate_math500_stream` 函数开始，第一个修改是将行 `gen_text = generate_text_stream_concat(...)` 替换为对 `results = self_consistency_vote(...)` 的调用。第二个修改添加了一个简单的平局决胜规则，选择最常见答案的第一次出现。例如，如果采样结果是 1, 3, 5, 3, 5，函数将返回 3，因为它是最频繁组中最早出现的成员。

由于最常见的答案存储在 `results["majority_winners"]` 中，打破平局的一种直接方法是取此列表的第一个元素，即 `results["majority_winners"][0]`。

这些更改在下面的代码摘录中说明：

```python
def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=2048,
    verbose=False,
    prompt_suffix="",    # NEW
    temperature=1.0,     # NEW
    top_p=1.0,           # NEW
    seed=None,           # NEW
    num_samples=10,      # NEW
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")
    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])
            ##############################################################
            # NEW
            prompt += prompt_suffix
            results = self_consistency_vote(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                num_samples=num_samples,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                show_progress=False,
                show_long_answer=False,
                seed=seed,
            )
            # 解决平局
            if results["final_answer"] is None:
                extracted = results["majority_winners"][0]
            else:
                extracted = results["final_answer"]
            # extracted = extract_final_candidate(
            #     gen_text
            # )
            # 可选地，获取长答案
            if extracted is not None:
                for idx, s in enumerate(results["short_answers"]):
                    if s == extracted:
                        long_answer = results["full_answers"][idx]
                        break
                gen_text = long_answer
            ##############################################################
            is_correct = grade_answer(
                extracted, row["answer"]
            )
            num_correct += int(is_correct)
            # ...
```

使用自一致性采样时的性能改进在第 4 章第 4.6 节的表 4.1 中总结和讨论（第 5-7 行和第 9-12 行）。

**练习 4.4：自一致性采样中的提前停止**

提前停止检查可以通过添加几行代码来实现，这些代码检查给定答案是否已经被计算多次，或者更具体地说，如果给定答案计数大于 `num_samples / 2`：

```python
def self_consistency_vote(
    # ...
    early_stop=True,   # NEW
):
    # ...
    if show_progress:
        print(f"[Sample {i+1}/{num_samples}] → {short!r}")
    #########################################################
    # NEW
    # 如果一个答案已经满足 >= 50% 多数，则提前停止
    if early_stop and counts[short] > num_samples / 2:
        majority_winners = [short]
        final_answer = short
        break
    #########################################################
    if final_answer is None:
        mc = counts.most_common()
        if mc:
            top_freq = mc[0][1]
            majority_winners = [s for s, f in mc if f == top_freq]
            final_answer = mc[0][0] if len(majority_winners) == 1 else None
    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }
```

修改后的 `self_consistency_vote` 函数的摘录更具体地说明了在哪里插入此代码：

```python
# ...
if early_stop and counts[short] > num_samples / 2:
    majority_winners = [short]
    final_answer = short
    break
```

## B.4 第 5 章

**练习 5.1：使用启发式评分器作为自一致性中的平局决胜者**

有很多方法可以实现这一点。也许最简单的方法是在自一致性函数外部处理它，直接使用返回的字典，类似于我们在练习 4.4 中实现平局决胜逻辑时直接在 `evaluate_math500_stream` 函数内部所做的那样。相关行如下所示：

```python
# ...
from reasoning_from_scratch.ch05 import heuristic_score

def evaluate_math500_stream(
    #...
    # ...
    results = self_consistency_vote(...)
    # 多数投票获胜者可用
    if results["final_answer"] is not None:
        extracted = results["final_answer"]
    ### NEW: 用 heuristic_score 打破平局
    else:
        best = None
        best_score = float("-inf")
        for cand in results["majority_winners"]:
            scores = [
                heuristic_score(results["full_answers"][idx],   
                prompt=prompt)
                for idx in results["groups"][cand]
            ]
            score = max(scores)
            if score > best_score:
                best_score = score
                best = cand
        extracted = best
    # ...
    # ...
    return num_correct, num_examples, acc
```

结果如表 B.3 所示。

**表 B.3 MATH-500 自一致性分数与不同平局决胜方法**

| 方法 | 模型 | 准确率 | 时间 |
|------|------|-------|------|
| 1 带思维链提示的基线 | Base | 33.4% | 129.2 min |
| 2 自一致性 (n=3) | Base | 43.2% | 328.2 min |
| 3 自一致性 (n=3) + 启发式 | Base | 43.4% | 326.5 min |
| 4 自一致性 (n=3) + 平均 logprob | Base | 44.8% | 327.7 min |

表中显示的准确率值和运行时间是使用 "cuda" GPU (DGX Spark) 在 MATH-500 测试集的全部 500 个样本上计算的。

表 B.3 中的第 1 行是第 4 章不带自一致性的基线。第 2 行不使用评分器进行平局决胜，因此如果答案之间存在平局，它会选择首次出现的答案。使用启发式评分器作为平局决胜者（第 3 行）会带来轻微改进。而使用 logprob 评分器作为平局决胜者（第 4 行）则获得了最好（但也是最小的）改进。

**练习 5.2：在 Best-of-N 设置中使用启发式评分器**

Best-of-N 类似于自一致性，因为我们生成多个答案。然而，我们不是通过多数投票选择最终答案，而是使用评分函数（如 `heuristic_score`）对所有生成的答案进行评分，并返回得分最高的答案。

有几种方法可以实现这种行为，但最简单的方法是使用第 4 章中现有的自一致性函数作为模板，并替换为 `heuristic_score`，如下所示：

```python
# ...
from reasoning_from_scratch.ch05 import (
    heuristic_score
)

def self_consistency_vote( #...):
    full_answers, short_answers = [], []
    counts = Counter()
    groups = {}
    majority_winners, final_answer = [], None
    best_score, best_idx = float("-inf"), None
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)
        answer = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )
        short = extract_final_candidate(answer, fallback="number_then_full")
        full_answers.append(answer)
        short_answers.append(short)
        counts[short] += 1
        if short in groups:
            groups[short].append(i)
        else:
            groups[short] = [i]
        score = heuristic_score(answer, prompt=prompt)
        if score > best_score:
            best_score, best_idx = score, i
```

**表 B.4 MATH-500 Best-of-N 分数与启发式和平均 logprob 评分**

| 方法 | 模型 | 准确率 | 时间 |
|------|------|-------|------|
| 1 带思维链提示的基线 | Base | 33.4% | 129.2 min |
| 2 Best-of-N (n=3) + 启发式 | Base | 40.6% | 327.7 min |
| 3 Best-of-N (n=3) + 平均 logprob | Base | 43.2% | 330.2 min |

表中显示的准确率值和运行时间是使用 "cuda" GPU (DGX Spark) 在 MATH-500 测试集的全部 500 个样本上计算的。

**练习 5.3：使用 Logprob 评分器作为自一致性中的平局决胜者**

代码与练习 5.1 类似，只是我们将 `heuristic_score` 替换为 `avg_logprob_answer`，如下所示：

```python
# ...
# from reasoning_from_scratch.ch05 import heuristic_score
from reasoning_from_scratch.ch05 import avg_logprob_answer

def evaluate_math500_stream(# ...)
    # ...
    # score = heuristic_score(
    #    candidate_full, prompt=prompt
    # )
    score = avg_logprob_answer(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        answer=candidate_full,
        device=device,
    )
    # ...
```

结果已包含在之前的表 B.3（练习 5.1）第 4 行中。

**练习 5.4：在 Best-of-N 设置中使用 Logprob 评分器**

要实现带有 logprob 评分器的 Best-of-N，我们可以使用练习 5.2 中的代码，并将 `heuristic_score` 替换为 `avg_logprob_answer`，如下所示：

```python
from reasoning_from_scratch.ch05 import (
    avg_logprob_answer
)
# ...
score = avg_logprob_answer(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    answer=answer,
    device=device
)
if score > best_score:
    best_score, best_idx = score, i
# ...
```

得到的 MATH-500 分数如上面的表 B.4（练习 5.2）所示。

**练习 5.5：使用启发式分数进行自精炼**

使用 `heuristic_score` 实际上比使用 logprob 分数更简单；我们只需要更改以下代码：

```python
from functools import partial

avg_logprob_score = partial(
    avg_logprob_answer,
    model=model,
    tokenizer=tokenizer,
    device=device
)
torch.manual_seed(0)
results_logprob = self_refinement_loop(
    model=model,
    tokenizer=tokenizer,
    raw_prompt=raw_prompt,
    device=device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=avg_logprob_score,
    verbose=True,
    temperature=0.7,
    top_p=0.9,
)
```

更新的代码为：

```python
torch.manual_seed(0)
results_logprob = self_refinement_loop(
    model=model,
    tokenizer=tokenizer,
    raw_prompt=raw_prompt,
    device=device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=heuristic_score,  # NEW
    verbose=True,
    temperature=0.7,
    top_p=0.9,
)
```

相对于第 3 章的基线和第 4 章的自一致性的改进显示在主章节的表 5.1（第 4、5 和 10 行）中。

## B.5 第 6 章

**练习 6.1：添加格式感知奖励塑造**

如果没有找到 "\boxed{}" 答案，我们可以分配部分奖励（分数 0.5），如下所示，使用我们在第 3 章中编码的 `fallback="number_then_full"` 回退：

```python
from reasoning_from_scratch.ch03 import (
    extract_final_candidate, grade_answer
)

def reward_rlvr(answer_text, ground_truth):
    # 1) 尝试提取框选答案
    boxed = extract_final_candidate(
        answer_text, fallback=None
    )
    if boxed:
        correct = grade_answer(boxed, ground_truth)
        return 1.0 if correct else 0.0
    # 2) 如果没有找到框选答案，查找数字
    unboxed = extract_final_candidate(
        answer_text, fallback="number_then_full"
    )
    if unboxed:
        correct = grade_answer(unboxed, ground_truth)
        return 0.5 if correct else 0.0
    return 0.0
```

当插入到第 6 章的代码中并在相同设置下训练时，部分奖励变体获得了较低的准确率（37.8%），而标准 GRPO 设置为 47.4%，尽管平均使用的 token 数量相似，如表 B.5 所示。

**表 B.5 严格奖励和部分奖励的 MATH-500 准确率**

| 方法 | 步数 | 最大 token 数 | 采样数 | 准确率 | 平均 token 数 |
|------|------|-------------|--------|-------|-------------|
| 1 GRPO (第 6 章) | 50 | 512 | 8 | 47.4% | 586.11 |
| 2 GRPO 部分奖励 (练习 6.2) | 50 | 512 | 8 | 37.8% | 550.33 |

**练习 6.2：零优势情况**

如果奖励都相等（例如，它们都是 0 或都是 1），优势将全部为 0，因为减去均值会移除共享的奖励值，只留下零，我们可以在下面演示：

```python
import torch
rollout_rewards = [0., 0., 0., 0.]
rewards = torch.tensor(rollout_rewards)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
print(advantages)
```

这返回 `tensor([0., 0., 0., 0.])`。

类似地，如果我们将 rollout 奖励更改为 `rollout_rewards = [0., 0., 0., 0.]`，我们得到相同的全零张量，`tensor([0., 0., 0., 0.])`。

简而言之，如果一组中的所有奖励都相同，例如所有奖励为 0 或所有奖励为 1，那么对于所有 i 个 rollout，$r_i - \mu_i = 0$。因此，策略梯度为零，该提示的模型参数不会更新。

这种行为是故意的。如果所有 rollout 都同样糟糕或同样好，就没有相对信号来告诉模型应该强化或抑制哪种行为。直观地说，如果模型正确回答了所有问题，就不需要更新它。反之，如果模型错误地回答了所有问题，我们不想更新模型来强化这种行为。

## B.6 第 7 章

**练习 7.1：测试 \<THINK\> 格式奖励**

以下代码检查如果 think token 使用不正确，格式奖励是否为零：

```python
from pathlib import Path
import torch
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
from reasoning_from_scratch.qwen3 import download_qwen3_small
from reasoning_from_scratch.ch07 import reward_format

download_qwen3_small(
    kind="reasoning", tokenizer_only=True, out_dir="qwen3"
)
tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
prompt = "Calculate ..."

def check_case(name, rollout):
    token_ids = tokenizer.encode(prompt + rollout)
    prompt_len = len(tokenizer.encode(prompt))
    reward = reward_format(
        token_ids=torch.tensor(token_ids),
        prompt_len=prompt_len,
    )
    print(f"{name}: {reward}")

# 1) 正确情况
check_case(
    "Correct order",
    "Let's ... <think> ... </think> ..."
)
# 2) 标签拼写错误
check_case(
    "Typo in <think>",
    "Let's ... <thnik> ... </think> ..."
)
# 3) 顺序颠倒
check_case(
    "Reversed order",
    "Let's ... </think> ... <think> ..."
)
# 4) 缺少一个标签
check_case(
    "Missing </think>",
    "Let's ... <think> ..."
)
```

输出如下，表明函数需要正确使用 `<think>...</think>` 标签才能授予 1.0 的奖励：

```
Correct order: 1.0
Typo in <think>: 0.0
Reversed order: 0.0
Missing </think>: 0.0
```

**练习 7.2：使格式奖励有条件**

条件奖励的实现非常简单；在主章节中，我们讨论了将总体奖励实现如下：

```python
reward = rlvr_reward + format_reward_weight * format_reward
```

因此，如果正确性奖励 (`rlvr_reward`) 为 0.0，禁用奖励的一种方法是：

```python
if conditional_reward:
    format_reward *= rlvr_reward
reward = rlvr_reward + format_reward_weight * format_reward
```

要在实践中使用它，你可以运行 `7_6_plus_format_reward.py` 脚本，我们在第 7 章第 7.6 节中使用过，并启用 `--conditional_reward` 标志。

我们可以下载此运行的日志文件（使用与第 7.6 节类似的设置）并绘制它：

```python
from reasoning_from_scratch.ch07 import download_from_github
from reasoning_from_scratch.ch07 import plot_grpo_metrics

download_from_github(
    "ch07/02_logs/7_6_plus_format_reward_conditional_metrics.csv"
)
plot_grpo_metrics(
    "7_6_plus_format_reward_conditional_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"],
)
```

图 B.1 显示了带有条件格式奖励的 GRPO 训练运行的基本指标。

图 B.1 中的图显示，评估准确率和平均奖励受到了很大打击，但似乎恢复了。

总体而言，尽管有这种性能崩溃，但这看起来比以前更稳定，并且趋势表明如果我们训练更长时间，性能会进一步提高。

```python
plot_grpo_metrics(
    "7_6_plus_format_reward_conditional_metrics.csv",
    columns=["reward_avg", "format_reward_avg", "adv_std", "entropy_avg"],
)
```

图 B.2 显示了带有条件格式奖励的 GRPO 训练运行的额外指标。

在图 B.2 中，我们看到平均格式奖励几乎完美地模仿了平均奖励图，这是一个很好的健全性检查，表明条件逻辑正在工作。此外，平均格式奖励显示了在正确答案子集中，总奖励中有多少来自格式项。

如我们所见，由于平均格式奖励图呼应了平均奖励图，它主要是一种奖励（而且看起来如果模型正确，它总是被授予；这很有道理，因为训练后的推理模型已经知道如何正确使用 `<think>...</think>` 标签（我们可以看到它没有忘记这种能力）。

然而，熵的增加仍然有点令人担忧，可能暗示着训练不稳定性，这可以通过其他方式解决（例如使用更小的 `clip_eps` 进行更紧的裁剪）。

## B.7 第 8 章

**练习 8.1：训练集和验证集长度**

要计算训练集和验证集的答案长度统计信息，你可以在分割后，在第 8.4.3 节末尾添加以下命令：

```python
compute_length(train_examples)
# 打印
# Average: 1180 tokens
# Shortest: 236 tokens (index 5730)
# Longest: 2048 tokens (index 1319)
```

以及：

```python
compute_length(val_examples)
# 打印
# Average: 1106 tokens
# Shortest: 236 tokens (index 12)
# Longest: 2048 tokens (index 22)
```

如我们所见，平均 token 长度（1180 对 1106）相当相似，数据集应该相对平衡。

作为奖励，我们还可以绘制直方图来可视化分布：

```python
import matplotlib.pyplot as plt

train_lengths = [len(ex["token_ids"]) for ex in train_examples]
val_lengths = [len(ex["token_ids"]) for ex in val_examples]
# 归一化计数，因为验证分割要小得多
bins = range(0, max(train_lengths + val_lengths) + 64, 64)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(train_lengths, bins=bins, density=True, alpha=0.6, label="Train")
ax.hist(val_lengths, bins=bins, density=True, alpha=0.6, label="Validation")
ax.set_xlabel("Token length")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()
```

得到的图如图 B.3 所示。

图 B.3 训练集和验证集长度分布。

验证样本要少得多，这就是为什么验证直方图看起来有点锯齿状，但如我们所见，它具有良好的分布覆盖范围。

**练习 8.2：不使用 \<THINK\> token 进行蒸馏**

要在脚本执行命令中复现不带 `<think></think>` token 的运行：

```bash
uv run distill.py \
  --data_path deepseek-r1-math-train.json \
  --validation_size 25 \
  --epochs 3 \
  --lr 1e-5 \
  --max_seq_len 2048 \
  --grad_clip 1.0
```

然后，对于评估，我们使用 base 而不是 reasoning 模型：

```bash
uv run evaluate_math500.py \
  --dataset_size 500 \
  --which_model base \
  --max_new_tokens 4096 \
  --checkpoint_path \
  run_11/checkpoints/distill/qwen3-0.6B-distill-step05746-epoch1.pth
```

结果如表 B.6 所示。

**表 B.6 有无 think token 的 MATH-500 任务准确率**

| 方法 | Epoch | 最终验证损失 | MATH-500 准确率 |
|------|-------|-------------|----------------|
| 1 Qwen3 0.6B 基础模型（第 3 章） | - | - | 15.2% |
| 2 Qwen3 0.6B 推理模型（第 3 章） | - | - | 48.2% |
| 3 DeepSeek-R1 | 1 | 0.5436 (0.5404) | 31.8% (30.6%) |
| 4 DeepSeek-R1 | 2 | 0.5349 (0.5339) | 31.8% (32.4%) |
| 5 DeepSeek-R1 | 3 | 0.5343 (0.5306) | 30.2% (33.6%) |
| 6 Qwen3 235B-A22B | 1 | 0.4043 (0.3130) | 44.8% (45.0%) |
| 7 Qwen3 235B-A22B | 2 | 0.3963 (0.3087) | 39.4% (43.8%) |
| 8 Qwen3 235B-A22B | 3 | 0.3948 (0.3078) | 39.8% (44.2%) |

在表 B.6 中，新结果（不带 think token）首先显示，对应的 think token 结果（来自主章节）在括号中。

有趣的是，当省略 `<think></think>` token 时，Qwen3 模型的验证损失更低，但这并没有转化为更好的建模性能。

如我们所见，省略 `<think></think>` 在几乎所有情况下都会使结果稍微变差。

---

# 附录 C Qwen3 LLM 源代码

虽然这是一本从零开始的书，但如主章节所述，从零开始的部分指的是推理技术，而不是 LLM 本身。完全从零开始实现一个 LLM 需要另一本书，这是我的《从零构建大型语言模型》一书的主题（http://mng.bz/orYv）。

然而，对于对我们在本书《从零构建推理模型》中使用的 Qwen3 实现感兴趣的读者，本附录列出了我在本书的 `reasoning_from_scratch` Python 包中实现并导入的 `Qwen3Model` 模型的源代码：

```python
from reasoning_from_scratch.qwen3 import Qwen3Model, Qwen3Tokenizer
```

如图 C.1 所示，Qwen3 架构与 GPT-2 非常相似，后者在我的《从零构建大型语言模型》一书中有介绍。虽然熟悉 GPT-2 不是本书的要求，但本附录为熟悉 GPT-2 的读者提到了与 GPT-2 的比较。事实上，我通过将另一本书中的 GPT-2 模型逐段移植到 Qwen3 架构中来编写 Qwen3 实现，使其遵循类似的风格约定以提高可读性。

图 C.1 Qwen3 与 GPT-2 的架构比较。两个模型都通过嵌入层和堆叠的 transformer 块处理文本，但它们在某些设计选择上有所不同。

如图 C.1 所示，Qwen3（2025 年发布）和 GPT-2（2019 年发布）总体上非常相似，因为它们都基于原始 transformer 架构的解码器子模块。然而，自 2019 年以来，一些设计选择已经演变。

请注意，Qwen3 中发现的大多数这些设计选择并不是 Qwen3 独有的，而是在许多其他当代 LLM 中都可以找到，我在我的《大型 LLM 架构比较》（https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison）文章中讨论了这些。

对于想要了解这些架构如何实现的新 LLM 读者，我建议从 GPT-2 开始。它的设计更简单，更容易实现，这使其成为探索更现代变体之前的更容易的入门点。

由于本书不关注架构实现，本附录的其余部分将只简要概述 Qwen3 的代码。

## C.1 均方根层归一化 (RMSNorm)

与使用标准 LayerNorm 的 GPT-2 相比，较新的 Qwen3 架构将其替换为均方根层归一化（RMSNorm）。这已成为近期模型架构中越来越常见的趋势。

RMSNorm 与 LayerNorm 的核心功能相同：归一化层激活以稳定和改进训练。然而，它通过移除均值中心化步骤来简化计算。这意味着激活仍然会被归一化，但它们不以 0 为中心。

图 C.2 比较了 GPT-2 中使用的 LayerNorm 和 Qwen3 中使用的 RMSNorm。LayerNorm（左）将激活归一化，使其平均值（均值）恰好为零，扩散（方差）恰好为一。RMSNorm（右）则根据其均方根缩放激活，不强制零均值或单位方差，但仍将均值和方差保持在合理范围内以实现稳定训练。

如图 C.2 所示，LayerNorm 和 RMSNorm 都将层输出缩放到合理范围。

LayerNorm 减去均值并除以标准差，使得层输出具有零均值和单位方差（方差为一，标准差为一），这在梯度值方面产生了有利于稳定训练的性质。

RMSNorm 将输入除以均方根。这会将激活缩放到可比较的大小，而不强制零均值或单位方差。在图 C.2 所示的特定示例中，均值为 0.77，方差为 0.41。

LayerNorm 和 RMSNorm 都稳定激活尺度并改进优化；然而，RMSNorm 通常在大规模 LLM 中更受青睐，因为它计算成本更低。与 LayerNorm 不同，RMSNorm 默认不使用偏置（偏移）项，这减少了可训练参数的数量。此外，RMSNorm 将昂贵的均值和方差计算减少为单一的均方根操作。这将跨特征归约的数量从两个减少到一个，从而降低了 GPU 上的通信开销并略微提高了训练效率。

代码清单 C.1 展示了 RMSNorm 在代码中的样子。

**代码清单 C.1 RMSNorm**

```python
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        emb_dim,
        eps=1e-6,
        bias=False,
        qwen3_compatible=True,
    ):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)
```

请注意，为简洁起见，本附录不为每个 LLM 组件提供详细的代码讲解。相反，在 C.6 节中，我们将把所有组件集成到 `Qwen3Model` 类中，将预训练权重加载到其中，然后在 C.9 节中使用此模型生成文本。

## C.2 前馈模块

前馈模块（一个小型多层感知器）被替换为门控线性单元（GLU）变体，该变体在 2020 年的一篇论文中介绍（https://arxiv.org/abs/2002.05202）。在这种设计中，标准的两个全连接层被三个替换，如图 C.3 所示。

图 C.3 在 GPT-2（上）中，前馈模块由两个全连接（线性）层组成，中间由一个非线性激活函数分隔。在 Qwen3（下）中，此模块是门控线性单元（GLU）变体，添加了第三个线性层（线性层 3），并将此线性层 3 的输出与线性层 1 的激活输出进行逐元素相乘。

Qwen3 的前馈模块（图 C.3）可以如代码清单 C.2 所示实现。

**代码清单 C.2 Qwen3 前馈模块**

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"],
            bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"],
            bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"],
            bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2  # A
        return self.fc3(x)
```

乍一看，Qwen3 中使用的 GLU 前馈变体似乎应该优于 GPT-2 中的标准前馈变体，仅仅因为它添加了一个额外的线性层（三个而不是两个），因此似乎有更多的参数。

然而，这种直觉是误导性的。在实践中，GLU 变体中的 fc1 和 fc2 层每个的宽度都是标准前馈模块中 fc1 层宽度的一半。在实践中，GLU 变体有更少的参数。

为了用具体示例说明这一点，假设图 C.3 中“线性层 1”的输入维度为 1024。这对应于代码清单 C.2 中的 `cfg["emb_dim"]`。fc1 的输出维度为 3,072（`cfg["hidden_dim"]`）。请注意，这些是 Qwen3 0.6B 变体中使用的实际数字。在这种情况下，代码清单 C.2 中 GLU 变体的参数计数如下：

- fc1: 1024 × 3,072 = 3,145,728
- fc2: 1024 × 3,072 = 3,145,728
- fc3: 1024 × 3,072 = 3,145,728
- 总计: 3 × 3,145,728 = 9,437,184 参数

如果我们假设此 GLU 变体中的 fc1 宽度是标准前馈模块中通常选择的 fc1 宽度的一半，则标准前馈模块的参数计数如下：

- fc1: 1024 × 2×3,072 = 6,291,456
- fc2: 1024 × 2×3,072 = 6,291,456
- 总计: 2 × 6,291,456 = 12,582,912 参数

虽然 GLU 变体通常比常规前馈模块有更少的参数，但它们表现更好。改进来自于门控机制引入的额外乘法交互，`activation(x_fc1) * x_fc2`，这增加了模型的表达能力。这类似于在给定适当训练的情况下，更深、更窄的网络如何优于更浅、更宽的网络。

在我们继续下一节之前，还有一件事要说明。请注意，图 C.3 中所示的前馈模块包含一个标记为“激活函数”的元素，而我们在代码清单 C.2 中使用了 `nn.functional.silu` 激活作为具体示例。

从历史上看，激活函数是一个热门话题，直到深度学习社区在大约十年前基本上汇聚到修正线性单元（ReLU）。ReLU 简单且计算成本低，但它在零处有一个尖锐的拐点。这促使研究人员探索更平滑的函数，如高斯误差线性单元（GELU）和 sigmoid 线性单元（SiLU），如图 C.4 所示。

图 C.4 可用于前馈模块（神经网络）的不同激活函数。GELU 和 SiLU（Swish）为 ReLU 提供了平滑的替代方案，ReLU 在输入为零处有一个尖锐的拐点。

GELU 涉及高斯累积分布函数（CDF）。计算此 CDF 很慢，因为它使用分段逻辑和指数，这使得编写融合的、优化的 GPU 内核变得困难（尽管存在使用更便宜操作且运行更快、结果几乎相同的 tanh 近似）。

简而言之，虽然 GELU 产生平滑的激活曲线，但总体上比更简单的函数计算成本更高。

较新的模型已经在很大程度上用 SiLU（也称为 Swish）函数替代了 GELU，该函数平滑地将大的负输入抑制到 ~0，并且对大的正输入近似线性，如图 C.4 所示。

SiLU 具有类似的平滑性，但计算比 GELU 稍微便宜，并提供可比的建模性能。在实践中，SiLU 现在用于大多数架构，而 GELU 仅在某些模型中继续使用，如 Google 的 Gemma 开放权重 LLM。在代码清单 C.2 中的前馈模块实现中，此 SiLU 函数通过 `nn.functional.silu` 调用。代码清单 C.2 中的前馈模块也常被称为 SwiGLU，这是一个源自 Swish 和 GLU 两个术语的缩写。

## C.3 旋转位置嵌入 (RoPE)

在基于 transformer 的 LLM 中，位置编码是必要的，因为注意力机制的原因。默认情况下，注意力将输入 token 视为没有顺序。在原始 GPT 架构中，绝对位置嵌入通过为序列中的每个位置添加一个学习到的嵌入向量来解决这个问题，然后将其添加到 token 嵌入中。

RoPE（旋转位置嵌入的缩写）引入了一种不同的方法：它不是将位置信息添加为单独的嵌入，而是通过以依赖于每个 token 位置的方式旋转注意力机制（C.4 节）中的查询和键向量来编码位置信息。RoPE 是一个优雅的想法，但本身也是一个长话题。感兴趣的读者可以在原始 RoPE 论文中找到更多信息：https://arxiv.org/abs/2104.09864。（虽然于 2021 年首次引入，但 RoPE 随着 2023 年原始 Llama 模型的发布而广泛采用，此后已成为现代 LLM 的主流，因此它不是 Qwen3 独有的。）

RoPE 可以用两种数学上等价的方式实现：交错形式，将相邻维度配对进行旋转；或两半形式，将维度分成余弦和正弦两半以方便处理。代码清单 C.3 实现了两半变体，这可能更容易阅读。

**代码清单 C.3 RoPE 函数**

```python
import torch

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096,
                        dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (
        torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
        / head_dim
    ))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape    # A
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]  # First half    # B
    x2 = x[..., head_dim // 2:]  # Second half    # B
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    # Shape after:  (1, 1, seq_len, head_dim)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)    # C
```

代码清单 C.3 中的 RoPE 代码将在 C.4 节的组查询注意力机制中使用。

> **RoPE 实现变体**
> 熟悉原始 RoPE 论文（https://arxiv.org/abs/2104.09864）的读者可能想知道我选择的特定实现。
> 有两种常见的 RoPE 实现风格，它们在数学上是等价的。实现主要区别在于旋转矩阵如何配对维度。我选择了 split-halves 风格，因为它更容易阅读和实现。
> 1) Split-halves 风格（本书，Hugging Face Transformers）：
> 旋转矩阵：
> ```
> [ cosθ   -sinθ    0      0   ... ]
> [ sinθ    cosθ    0      0   ... ]
> [  0       0    cosθ   -sinθ ... ]
> [  0       0    sinθ    cosθ ... ]
> ```
> 这里，嵌入维度被分成两半，然后每一块进行旋转。
> 2) 交错（偶数/奇数）风格（原始论文，Llama 仓库）：
> 旋转矩阵：
> ```
> [ cosθ  -sinθ    0      0   ... ]
> [ sinθ   cosθ    0      0   ... ]
> [  0      0    cosθ   -sinθ ... ]
> [  0      0    sinθ    cosθ ... ]
> ```
> 这里，嵌入维度交错为偶数/奇数余弦/正弦对。
> 两种布局编码相同的相对位置。唯一的区别是维度如何配对。

## C.4 组查询注意力 (GQA)

组查询注意力（GQA）已成为原始多头注意力（MHA）机制的标准、更计算和参数高效的替代方案。

与 MHA 不同，在 MHA 中每个头也有自己的一组键和值，为了减少内存使用，GQA 将多个头分组以共享相同的键和值投影，如图 C.5 所示。

图 C.5 MHA 和 GQA 的比较。这里，组大小为 2，其中键和值对在 2 个查询之间共享。

因此，如图 C.5 所示，GQA 的核心思想是通过在多个查询头之间共享键和值头来减少键和值头的数量。这 (1) 降低了模型的参数计数，(2) 减少了推理期间键和值张量的内存带宽使用，因为需要存储和从 KV 缓存（C.7 节）中检索的键和值更少。

虽然 GQA 主要是 MHA 的计算效率解决方法，但消融研究（如原始 GQA 论文中提出的，https://arxiv.org/abs/2305.13245）表明，在 LLM 建模性能方面，它与标准 MHA 表现相当。

代码清单 C.4 实现了带有 KV 缓存支持的 GQA 机制。

**代码清单 C.4 组查询注意力**

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None,
                 qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.W_query = nn.Linear(
            d_in, self.d_out, bias=False, dtype=dtype
        )
        self.W_key = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)            # A
        keys = self.W_key(x)                 # B
        values = self.W_value(x)             # B
        queries = queries.view(b, num_tokens, self.num_heads,
                               self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups,
                             self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups,
                                 self.head_dim).transpose(1, 2)
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0                    # C
            keys, values = keys_new, values_new            
        next_cache = (keys, values)
        keys = keys.repeat_interleave(       # D
            self.group_size, dim=1           # D
        )                                    # D
        values = values.repeat_interleave(   # D
            self.group_size, dim=1           # D
        )                                    # D
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / self.head_dim**0.5, dim=-1
        )
        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache
```

你可能已经注意到，代码清单 C.4 中的 GQA 机制还包括一个 `qk_norm` 参数。这不是标准 GQA 设计的一部分。当 `qk_norm=True` 时，会对查询和键应用额外的基于 Query/Key-RMSNorm 的归一化，称为 QKNorm，这是 Qwen3 中使用的一种技术。如前面在 RMSNorm 部分（C.1 节）所讨论的，QKNorm 有助于提高训练稳定性。

## C.5 Transformer 块

Transformer 块是 LLM 的核心组件，它结合了本附录到目前为止涵盖的所有单个元素。如图 C.6 所示，它重复多次；在 Qwen3 的 0.6B 参数版本中，它重复 28 次。

图 C.6 Qwen3 中 transformer 块的结构。每个块包括 RMSNorm、RoPE、掩码组查询注意力和前馈模块，在 0.6B 参数模型中重复 28 次。

代码清单 C.5 实现了图 C.6 中所示的 transformer 块。

**代码清单 C.5 Transformer 块**

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(
            x, mask, cos, sin, start_pos=start_pos, cache=cache
        )  # A
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x, next_cache
```

如我们所见，在代码清单 C.5 中，transformer 块简单地连接了我们在前面章节中实现的各种元素。

## C.6 主模型代码

在本节中，我们将定义在第 2 章中导入并使用的 `Qwen3Model` 类。为了实现 `Qwen3Model` 类，代码清单 C.6 遵循了前面图 C.6 中所示的架构，其中 transformer 块位于 LLM 的核心。

**代码清单 C.6 主 Qwen3Model 代码**

```python
class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 主模型参数
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"],
                                    dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"],
            bias=False, dtype=cfg["dtype"]
        )
        # 可重用工具
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # 跟踪 KV 缓存中的当前位置

    def forward(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(
                    pos_end, pos_end, device=x.device, dtype=torch.bool
                ),
                diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  # 不严格必要但有助于 torch.compile
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device,
                           dtype=torch.bool),
                diagonal=1
            )
        mask = mask[None, None, :, :]                 # A
        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0
```

由于我们已经有了所有主要组件，代码清单 C.6 中的 `Qwen3Model` 类只添加了围绕 transformer 块的几个额外组件，即嵌入层和输出层（包括另一个 RMSNorm 层）。然而，代码可能看起来有些复杂，这是由于 KV 缓存选项。

如第 2 章所讨论的，KV 缓存可以加速文本生成过程，但这是本书范围之外的话题。感兴趣的读者可以在我的文章《从零理解和编码 LLM 中的 KV 缓存》中找到有关 KV 缓存的更多信息：https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms。

请注意，如代码清单 C.6 中实现的 `Qwen3Model` 类支持各种模型大小（有关更多信息，请参见附录 D）。在第 2 章中，我们使用 0.6B 参数模型，因为它是 Qwen3 模型家族中资源需求最低的模型。此模型的具体配置在图 C.7 中可视化。

图 C.7 Qwen3 0.6B 模型的架构。该模型由一个 token 嵌入层组成，后跟 28 个 transformer 块，每个块包含 RMSNorm、RoPE、QKNorm、带有 16 个头的掩码组查询注意力，以及一个中间大小为 3,072 的前馈模块。

要通过 `Qwen3Model` 类使用图 C.7 中所示的 0.6B 模型，我们可以在代码清单 C.7 中定义以下配置，在实例化新的 `Qwen3Model` 实例时作为输入（`cfg=QWEN_CONFIG_06_B`）提供。

我们将在 C.9 节中使用代码清单 C.7 中的 `QWEN_CONFIG_06_B` 配置来实例化 Qwen3 0.6B 模型。

> **更精确的上下文长度信息**
> 对于 Qwen3 0.6B，默认最大支持的上下文长度为 40,960 个 token。然而，模型本身仅使用 32,768 个 token 进行训练。40,960 个 token 的分配为模型输出（生成的文本）保留 32,768 个 token，为典型提示（用户的问题或指令）保留 8,192 个 token。
> 为了更直观地理解，40,000 个 token 大约对应于第一本《哈利·波特》书的一半。
> 由于此长度足以满足大多数推理任务，开发人员指出，除非平均上下文经常超过 32,000 个 token，否则不推荐上下文扩展方法（如 YaRN），因为在较短的上下文中启用它们可能会略微降低性能。

**代码清单 C.7 Qwen3 0.6B 配置**

```python
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # 词汇表大小
    "context_length": 40_960,  # 训练期间最初使用的长度
    "emb_dim": 1024,           # 嵌入维度
    "n_heads": 16,             # 注意力头数
    "n_layers": 28,            # 层数
    "hidden_dim": 3072,        # FeedForward 中中间维度的大小
    "head_dim": 128,           # GQA 中头的大小
    "qk_norm": True,           # 是否在 GQA 中归一化查询和键
    "n_kv_groups": 8,          # GQA 的键值组数
    "rope_base": 1_000_000.0,  # RoPE 中 "theta" 的基数
    "dtype": torch.bfloat16,   # 降低精度的 dtype 以减少内存
}
```

## C.7 KV 缓存

与 KV 缓存相关的主要工作在 `Qwen3Model`（代码清单 C.6）和 `GroupedQueryAttention`（代码清单 C.4）代码中完成。代码清单 C.8 中所示的 `KVCache` 在文本生成期间存储键值对本身，这带来了我们在第 2 章中启用 KV 缓存时体验到的加速。

代码清单 C.8 中的 `KVCache` 类在第 2 章中实现的 `generate_text_basic_stream_cache` 函数内部使用。

**代码清单 C.8 KV 缓存**

```python
class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
```

## C.8 分词器

分词器代码有些复杂，因为它支持各种特殊 token，除了基础模型之外，还支持 Qwen3 的所谓的“Thinking”模型变体，这是一个推理模型。分词器的完整重新实现如代码清单 C.9 所示。

**代码清单 C.9 分词器**

```python
import re
from tokenizers import Tokenizer

class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self,
                 tokenizer_file_path="tokenizer-base.json",
                 apply_chat_template=False,
                 add_generation_prompt=False,
                 add_thinking=False):
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. "
            )
        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) 
                               for t in self._SPECIALS}
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)
        f = tok_path.name.lower()                      # A
        if "base" in f and "reasoning" not in f:       # A
            self.eos_token = "<|endoftext|>"           # A
        else:                                          # A
            self.eos_token = "<|im_end|>"              # A
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template
        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]
        if chat_wrapped:
            prompt = self._wrap_chat(prompt)
        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
        if self.add_thinking:
            s += "\n"          # B
        else:
            s += "\n<think>\n\n</think>\n\n"
        return s
```

请注意，我在代码清单 C.9 中的 `Qwen3Tokenizer` 重新实现可能看起来有些复杂，因为它旨在复制 Qwen3 团队在 Hugging Face Transformers 库中发布的官方分词器的行为。

乍一看，它似乎有一些怪癖。例如，当 `add_thinking=True` 时，不插入 `"\n<think>\n\n</think>\n\n"` token（其中 `\n` 是换行符），而当 `add_thinking=False` 时，会添加这些 token。这是有意为之的，因为非基础 Qwen3 0.6B 模型是一个混合模型，支持推理（"thinking"）和标准模式。

## C.9 使用模型

让我们现在实例化并使用模型，通过重用第 2 章中的文本生成方法来确认代码是否有效。

首先，我们使用预训练模型权重实例化模型：

```python
from pathlib import Path
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.qwen3 import download_qwen3_small

# device = get_device()         # A
device = torch.device("cpu")
download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
tokenizer_file_path = Path("qwen3") / "tokenizer-base.json"
model_file = Path("qwen3") / "qwen3-0.6B-base.pth"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_file_path)
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file))
model.to(device)
```

输出显示了实例化模型的结构，应该与我们在代码清单 C.7 中的配置文件中使用的值相匹配：

```
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
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

接下来，我们重用第 2 章中的文本生成函数来生成文本：

```python
import time
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache,
    generate_stats
)

prompt = "Explain large language models in a single sentence."
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device
).unsqueeze(0)
max_new_tokens = 200
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
    generated_ids.append(next_token_id)  # 收集生成的 token
end_time = time.time()
output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)
```

由于我们使用了与第 2 章相同的提示，生成的文本与第 2 章生成的文本完全匹配：

```
Time: 1.46 sec
28 tokens/sec
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.
```

虽然主章节使用 Qwen3 的 0.6B 参数变体来降低本书的资源需求，但感兴趣的读者可以在附录 D 中找到有关如何使用更大模型的更多信息。

---

# 附录 D 使用更大的 LLM

主章节使用 0.6B 参数（0.6B）的 Qwen3 基础模型，因为它是 Qwen3 家族中最小的模型，因此最容易在消费级硬件上运行。

然而，附录 C 中的相同 `Qwen3Model` 实现不仅限于 0.6B 检查点。我们还可以用它来加载更大的 Qwen3 检查点，使用相同的从零开始的 PyTorch 代码。在实践中，这意味着一旦我们理解了如何使用 0.6B 模型，转向更大的模型主要涉及三个变化：

1. 选择匹配的配置字典；
2. 从 Hugging Face 下载更大的检查点；
3. 加载基础或推理变体的适当分词器。

本附录使用 Qwen3 4B 模型作为具体示例来说明此过程，因为它足够大，比 0.6B 模型有意义地更强，同时仍然比更大的 8B、14B 和 32B 变体更容易处理。

## D.1 更大的密集 Qwen3 配置

仓库在 `reasoning_from_scratch.appendix_c` Python 库中包含了几个更大 Qwen3 模型的配置字典，如表 D.1 所列。（你也可以在补充材料中查看源代码：https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/appendix_c.py）。

**表 D.1 Qwen3 配置（大于 0.6B）**

| 模型大小 | 配置 Python 字典 |
|---------|----------------|
| 1.7B | QWEN3_CONFIG_1_7B |
| 4B | QWEN3_CONFIG_4B |
| 8B | QWEN3_CONFIG_8B |
| 14B | QWEN3_CONFIG_14B |
| 32B | QWEN3_CONFIG_32B |

请注意，表 D.1 中列出的模型是密集的 Qwen3 变体，类似于主章节中使用的 Qwen3 0.6B 模型和附录 C 中讨论的模型。这里，“密集”意味着这些不是 Qwen3 的稀疏混合专家（MoE）变体。MoE 模型不受本书代码支持，但从零开始的实现可以在这里找到：https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3。

图 D.1 更直观地总结了表 D.1 中模型之间的差异。

图 D.1 不同 Qwen3 变体的比较，从 0.6B 到 32B 参数。

如图 D.1 所示，所有不同的 Qwen3 大小变体都使用与附录 C 中 0.6B 模型相同的整体架构模式。具体而言，它们仍然使用 token 嵌入、组查询注意力、旋转位置嵌入、前馈层和 RMS 归一化。变化的是模型维度，如嵌入大小、层数、注意力头数和前馈隐藏维度。

表 D.2 列出了将模型加载到 RAM（和 GPU RAM）时的内存需求估计。这里，作为一个粗略的下界，以 bfloat16 精度存储权重（这是 LLM 的常见默认值）大约需要每个参数 2 字节。

**表 D.2 Qwen3 模型大小**

| 模型大小 | 大致权重内存（bfloat16） |
|---------|------------------------|
| 1.7B | 约 3.4 GB |
| 4B | 约 8 GB |
| 8B | 约 16 GB |
| 14B | 约 28 GB |
| 32B | 约 64 GB |

表 D.2 中的数字只是权重本身的大致下界。在实践中，实际运行时内存使用更高，因为我们还需要内存来存储激活、临时缓冲区、KV 缓存等。此外，模型加载可以暂时增加内存使用，因为在将张量复制到 `Qwen3Model` 实例中时，张量可能存在于多个位置。有关此问题的更多详细信息，请参阅我的《从零构建大型语言模型》一书的补充材料：https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/08_memory_efficient_weight_loading

## D.2 下载更大检查点概述

与主章节中使用的已转换为 PyTorch 检查点以最小化外部代码依赖的 0.6B 检查点不同，较大的官方 Qwen3 模型通常以 safetensors 文件的形式分发，有时跨多个分片分割。

`reasoning-from-scratch` Python 库提供了一个辅助函数 `download_from_huggingface_from_snapshots` 来处理这个问题。这个函数，我们将在下一节中使用，下载 Hugging Face 快照到本地目录，然后加载单个 `model.safetensors` 文件或多个通过 `model.safetensors.index.json` 文件引用的分片。

因为这个辅助函数依赖于早期章节中不需要的额外包，你可能需要先安装它们：

```bash
uv add huggingface_hub safetensors
```

或

```bash
pip install huggingface_hub safetensors
```

一旦这些包安装完成，加载更大的检查点遵循与之前相同的大致模式。例如，我们下载文件，实例化模型，加载权重，准备分词器，然后运行文本生成，如我们将在下一节中看到的。

## D.3 加载更大的基础模型

我们现在将使用官方 Qwen3 4B 基础模型完成一个完整的示例。第一步，如代码清单 D.1 所示，是从 Hugging Face Model Hub 下载模型快照（https://huggingface.co/Qwen/Qwen3-4B-Base）：

**代码清单 D.1 下载模型权重**

```python
from pathlib import Path
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.appendix_c import (
    download_from_huggingface_from_snapshots,
)

device = get_device()
local_dir = Path("qwen3-4b-base")
weights = download_from_huggingface_from_snapshots(
    repo_id="Qwen/Qwen3-4B-Base",
    local_dir=local_dir,
)
```

`local_dir` 路径指定快照应存储在磁盘上的位置。在此示例中，我们将 4B 基础模型保存在专用的 `qwen3-4b-base` 文件夹中，以便它与主章节中使用的较小 `qwen3` 目录分开。

接下来，我们使用代码清单 D.2 中的代码实例化 4B 模型并将 Hugging Face 权重加载到从零开始的 Qwen3Model 实现中：

**代码清单 D.2 将权重加载到 Qwen3Model**

```python
from reasoning_from_scratch.qwen3 import (
    Qwen3Model,
    load_hf_weights_into_qwen,
)
from reasoning_from_scratch.appendix_c import QWEN3_CONFIG_4B

model = Qwen3Model(QWEN3_CONFIG_4B)
load_hf_weights_into_qwen(
    model,
    param_config={
        "n_layers": QWEN3_CONFIG_4B["n_layers"],
        "hidden_dim": QWEN3_CONFIG_4B["hidden_dim"],
    },
    params=weights,
)
model.to(device)
model.eval()
```

`QWEN3_CONFIG_4B` 字典定义了 4B 模型的架构维度。`load_hf_weights_into_qwen` 辅助函数然后将 Hugging Face 参数名称映射到我们自己的实现中使用的参数名称。之后，`model.to(device)` 将模型移动到所选设备，`model.eval()` 将模型切换到推理模式。

模型加载后，我们通过代码清单 D.3 中的代码准备分词器：

**代码清单 D.3 加载基础分词器**

```python
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
import shutil

tokenizer_src = local_dir / "tokenizer.json"
tokenizer_path = local_dir / "tokenizer-base.json"
if not tokenizer_path.exists():
    shutil.copyfile(tokenizer_src, tokenizer_path)
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
```

Hugging Face 快照将分词器存储在通用名称 `"tokenizer.json"` 下。在代码清单 D.3 中，我们将其复制到 `"tokenizer-base.json"`，以便很容易区分于我们将在下一节中使用的推理分词器。

我们现在可以使用模型进行生成，与我们在第 2 章中使用较小的 0.6B 模型时完全相同，使用代码清单 D.4 中的代码：

**代码清单 D.4 生成文本**

```python
import torch
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache,
)

prompt = "Explain large language models in two sentences."
input_ids = torch.tensor(
    tokenizer.encode(prompt),
    device=device,
).unsqueeze(0)
for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_ids,
    max_new_tokens=64,
    eos_token_id=tokenizer.eos_token_id,
):
    print(tokenizer.decode(token.squeeze(0).tolist()), end="", flush=True)
```

类似于第 2 章中的主书内容，我们对输入提示进行编码，通过 `unsqueeze(0)` 添加批次维度，然后逐个流式传输生成的 token。

输出如下所示：

```
Large language models are artificial intelligence systems that use deep learning
techniques to understand and generate human-like text. They are trained on vast 
amounts of data and can perform a wide range of natural language processing
tasks, 
such as translation, summarization, and question answering.
```

这是跨模型大小重用相同 `Qwen3Model` 抽象的一个很好的特性。一旦检查点和分词器设置正确，推理代码看起来基本相同。

## D.4 加载更大的推理变体

同样的想法也适用于更大的推理风格 Qwen3 模型。给定模型大小的架构保持不变。与上一节相比，只有检查点和分词器设置发生变化。

例如，要加载 4B 推理变体而不是 4B 基础变体，我们需要：

- 将仓库 ID 从 `Qwen/Qwen3-4B-Base` 切换为 `Qwen/Qwen3-4B`；
- 将 `tokenizer.json` 文件复制或重命名为 `tokenizer-reasoning.json`；
- 使用代码清单 D.5 中的代码初始化分词器：

**代码清单 D.5 加载推理分词器**

```python
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_path,
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
```

这三个分词器设置与推理模型使用的聊天风格提示格式相匹配，如第 2 章和第 8 章所讨论的。换句话说，我们仍然使用相同的 4B 架构配置（`QWEN3_CONFIG_4B`），但我们将其与推理检查点和推理风格分词器行为配对。

其余的模型加载和模型使用代码保持不变。我们仍然下载快照，实例化 `Qwen3Model(QWEN3_CONFIG_4B)`，通过 `load_hf_weights_into_qwen` 加载权重，将模型移动到目标设备，然后使用第 2 章中的相同流式函数生成文本。

## D.5 实用建议

本附录的主要观点是，附录 C 中相同的从零开始模型代码可以在更大的官方 Qwen3 检查点中重用，前提是我们选择匹配的配置字典和分词器设置。

这意味着转向更大的模型主要是加载不同的检查点和配置的问题，而不是学习一个全新的概念。

如果你想超越这里展示的 4B 示例进行实验，1.7B 和 8B 变体是自然的下一步。如果你的硬件无法处理 4B，1.7B 模型是从 0.6B 迈出的适度一步。8B 模型 substantially 更大，如果你的硬件可以处理它，可能会产生更强的输出。

---

# 附录 E 批处理与面向吞吐量的执行

我们在主章节中实现的代码通常一次处理一个提示或示例。这保持了代码紧凑且更易于理解，也有助于使资源需求在某种程度上更易于管理。

实际上，本书中的代码运行起来已经很昂贵，因此在许多情况下添加批处理支持会增加复杂性而没有实际好处，这取决于你可用的硬件。

也就是说，如果你有访问相对现代的具有足够内存的 GPU，批处理执行（如图 E.1 所示）在某些设置中非常有用。例如，如果我们想评估许多问题、每个提示采样多个响应，或一次在多个示例上训练，那么批处理可以帮助提高吞吐量并降低总体时间。

本附录解释了批处理背后的广泛思想，并展示了如何在不同章节的补充材料中使用批处理实现。

图 E.1 单示例与批处理生成的概述。在批处理生成中，多个提示被打包到一个批次中，并行处理，然后一起解码以提高吞吐量。

## E.1 为什么批处理有帮助

当我们谈论计算性能时，区分以下总体目标是有帮助的：

1. **延迟**：我们为一个提示获得答案的速度有多快；
2. **吞吐量**：我们在固定时间内可以处理多少个提示。

单示例生成通常最适合理解概念实现、最小化延迟和调试。

批处理可以被视为单示例生成的扩展，主要针对吞吐量。例如，如果我们想在 MATH-500 上评估数百个问题，生成多个自一致性样本，或在更大的蒸馏数据集上训练，批处理可以在合适的硬件上 substantially 减少总运行时间。

然而，批处理在每个设备上不会自动更快。小模型在 CPU 或优化较少的 GPU 上可能受益很少或根本没有，因为我们必须引入额外的代码复杂性，包括填充和其他开销，这可能抵消我们通过批处理获得的并行性带来的收益。

因此，批处理最好被理解为吞吐量优化，而不是通用的加速。

## E.2 运行批处理生成

批处理中的主要技术障碍是提示通常具有不同的长度。一个数学问题可能分词为 40 个 token，而另一个可能分词为 120 个 token，但 PyTorch 张量仍然必须是矩形的。因此，如果我们想一起运行多个提示，我们需要一种方法来填充较短的行，并跟踪哪些位置是真实的 token，哪些只是填充 token。

填充 token 是添加到提示中的占位符 token，以使其达到一定长度，例如，将一个三 token 的提示 "3 + 3" 扩展为五 token 的提示，我们可以在右侧或左侧添加两个额外的填充 token："`<pad><pad> 3 + 3`"。（在实践中，我们使用哪个 token 作为填充 token 并不重要，通常使用序列结束 token 如 `<eos>` 或 `<|endoftext|>`。）

从概念上讲，填充和跟踪填充位置使批处理生成比单提示生成更复杂。在主章节中，我们使用了来自 `reasoning_from_scratch.qwen3` 的 `Qwen3Model` 类，这是附录 C 中解释的普通单示例实现。

对于批处理生成，补充材料因此在 `reasoning_from_scratch.qwen3_batched` 中包含了一个单独的 `Qwen3Model` 类，它可以作为直接替代品，但添加了批处理所需的填充感知执行簿记。

为了具体说明，从一个看起来非常类似于我们在主章节中已经使用的代码的单序列示例开始是有帮助的。在这里，在代码清单 E.1 中，我们将其应用于两个非常小的提示，但我们仍然一个接一个地运行它们：

**代码清单 E.1 单示例生成**

```python
import torch
from reasoning_from_scratch.ch02 import (
    get_device,
    generate_text_basic_stream_cache,
)
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer,
    render_prompt,
)

device = get_device()
model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False,
)

for problem in ["2+2?", "3+3=6?"]:  # A
    prompt = render_prompt(problem)
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=32,
        eos_token_id=tokenizer.eos_token_id,
    ):
        next_token_id = token.squeeze(0)
        print(tokenizer.decode(next_token_id.tolist()), end="", flush=True)
    print()  # B
```

得到的输出为：

```
\boxed{4}
\boxed{6}
```

代码清单 E.1 中的代码仍然是普通的单示例生成。我们渲染提示，对其进行分词，添加大小为 1 的批次维度，然后在 token 到达时流式传输生成的 token。这在概念上很简单，这就是主章节偏爱这种风格的原因。

代码清单 E.2 现在展示了相应的批处理版本。整体工作流程类似，但现在我们切换到 `reasoning_from_scratch.qwen3_batched`，预先对两个提示进行分词，左填充它们到相同长度，并一起生成它们：

**代码清单 E.2 批处理生成**

```python
from reasoning_from_scratch.qwen3_batched import (
    generate_text_basic_batched_cache,
    load_model_and_tokenizer,
)

model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False,
)

problems = ["2+2?", "3+3=6?"]
prompts = [render_prompt(problem) for problem in problems]
tokenized = [tokenizer.encode(p) for p in prompts]
pad_id = tokenizer.pad_token_id
max_len = max(len(t) for t in tokenized)
left_padded = [  # A
    [pad_id] * (max_len - len(t)) + t
    for t in tokenized
]
input_ids = torch.tensor(left_padded, dtype=torch.long, device=device)
generated = generate_text_basic_batched_cache(
    model=model,
    token_ids=input_ids,
    max_new_tokens=32,
    eos_token_id=tokenizer.eos_token_id,
    pad_id=pad_id,  # B
)
for row in generated:
    eos_pos = (row == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        row = row[:eos_pos[0]]
    print(tokenizer.decode(row.tolist()))
```

结果与之前相同，但这次它们是并行产生的：

```
\boxed{4}
\boxed{6}
```

在这个示例中，我们使用非流式批处理辅助函数，因此我们等待整个批次完成后再解码和打印结果。这保持了使用示例简单，并使我们在深入研究内部掩码逻辑之前更容易看到填充的作用。

还有一个更优化的文本生成函数变体，`generate_text_basic_batched_cache_stop`，它在 finished 行发出 EOS 时从活动计算批次中移除它们，而不是一直携带它们直到最长的行完成。相比之下，我们在代码清单 E.2 中使用的 `generate_text_basic_batched_cache` 在每个解码步骤中将每一行保持在活动批次中。

在 `generate_text_basic_batched_cache` 中，一旦一行发出了 EOS，实现会强制该行继续产生 EOS，以便批次形状保持对齐，即使从最终解码输出的角度来看该行已经完成。这两个函数之间的差异如图 E.2 所示。

图 E.2 顺序生成 (1) 与常规批处理生成 (2) 和提前停止批处理生成 (3)。常规批处理辅助函数 (2) 将 finished 行作为占位符 EOS 行保留在批次中，而 `_stop` 变体 (3) 在它们发出 EOS 时立即从活动计算批次中移除 finished 行。

作为旁注，Qwen3 使用 `<|endoftext|>` 作为基础模型的 EOS token，但图 E.2 将其简单标记为 `<eos>` 以视觉紧凑。

下一节将更详细地介绍内部填充是如何处理的。

## E.3 填充和注意力掩码

在单示例模式下，如果我们对一个短提示（如 "2+2?"）进行分词，我们可以将其作为形状为 `(1, 4)` 的简单张量传递给模型：

```python
input_ids = torch.tensor([[17, 10, 17, 30]])
```

在内部，模型内部构建一个标准的因果注意力掩码，以便每个位置只能关注自身和更早的 token。如果你不熟悉因果注意力掩码和自注意力，我有一篇文章提供了更多背景信息：https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention。

"2+2?" 提示的因果注意力掩码如图 E.3 所示。

图 E.3 单示例生成中使用的因果掩码。token 只能关注自己和更早的位置，这是标准的自回归设置。

在图 E.3 所示的因果掩码中，1 表示"被掩码掉"，0 表示"允许"。

在此上下文中，如图 E.3 所示，第一个 token 不能向前查看后面的位置，第二个 token 只能查看前两个位置，依此类推，这是标准的自回归掩码模式。

批处理改变了情况，因为不同的提示通常具有不同的长度。假设我们将 "2+2?" 与稍长的提示 "3+3=6?" 一起处理。

由于 PyTorch 张量必须是矩形的，较短的行必须填充以匹配较长的行，如图 E.4 所示，使用左填充。

图 E.4 批处理生成中的左填充。较短的行被填充到批次中的最大序列长度，填充感知掩码确保这些填充位置不会影响注意力。

请注意，图 E.4 显示我们在内部保留了一个额外的 `attn_mask`。此掩码用于跟踪填充位置，其中 True 表示已填充，False 表示未填充。

我们使用这个额外的 `attn_mask` 来识别因果掩码中对应于填充 token ID 的 token。

掩码填充的键和将填充的查询归零是使批处理行为类似于单示例执行的重要步骤。

在本附录的其余部分，我们将看到如何使用来自不同章节的补充材料中实现的可选批处理变体脚本。

## E.4 第 3 章：批处理 MATH-500 评估

补充材料包括一个用于第 3 章中单示例评估方法的脚本，我们可以使用第 7 章中介绍的下载函数直接下载和运行：

**代码清单 E.3 下载单示例评估脚本**

```python
from reasoning_from_scratch.ch07 import download_from_github
download_from_github(
    "ch03/02_math500-verifier-scripts/evaluate_math500.py"
)
download_from_github(
    "ch03/01_main-chapter-code/math500_test.json",
    out="math500_test.json",
)
```

通过代码清单 E.3 下载脚本和数据集后，我们可以从终端运行单示例评估，如下所示（如果你不使用 uv，只需将 `uv run` 替换为 `python`）：

```bash
uv run evaluate_math500.py \
  --dataset_size 500 \
  --which_model "reasoning"
```

补充材料还包括一个应用我们上面讨论的批处理方法的批处理版本。下载几乎相同，只是我们将 `evaluate_math500.py` 替换为 `evaluate_math500_batched.py`，如代码清单 E.4 所示：

**代码清单 E.4 下载支持批处理的评估脚本**

```python
download_from_github(
    "ch03/02_math500-verifier-scripts/evaluate_math500_batched.py"
)
```

批处理脚本的使用也与非批处理版本非常相似，只是我们现在提供了一个额外的 `--batch_size` 参数来指定模型应该并行处理多少个提示和答案：

```bash
uv run evaluate_math500_batched.py \
  --dataset_size 500 \
  --which_model "reasoning" \
  --batch_size 64
```

理想的批次大小完全取决于你的硬件可以处理什么，因此在实践中最好从较小的值开始，然后向上扩展，直到达到对你的设置有意义的吞吐量和内存限制。

我们将在本章末尾返回单示例和批处理生成性能的比较。

## E.5 第 4 章：批处理自一致性采样

第 4 章的可选 `self_consistency_math500_batched.py` 脚本不会将不同的提示混合到一个填充张量中。相反，它将相同的提示重复 `num_samples` 次，并并行采样多个延续以进行自一致性投票，因为并行化每个提示的不同样本数量是一个我们可以利用的自然角度。

由于每一行都从相同的提示长度开始，此脚本可以使用来自 `reasoning_from_scratch.qwen3` 的常规 `Qwen3Model`，而不是 `reasoning_from_scratch.qwen3_batched`。换句话说，这里的批次维度用于同一提示的多个采样 rollout，而不是将不同长度的无关提示填充在一起。

我们可以如代码清单 E.5 所示下载脚本：

**代码清单 E.5 下载支持批处理的自一致性采样**

```python
download_from_github(
    "ch04/02_math500-inference-scaling-scripts/"
    "self_consistency_math500_batched.py"
)
```

要下载非批处理版本，只需从代码清单 E.5 中的文件名中删除 "_batched" 部分。

我们可以如下运行批处理脚本，非批处理脚本的语法 otherwise 相同：

```bash
uv run self_consistency_math500_batched.py \
  --which_model base \
  --temperature 0.9 \
  --top_p 0.9 \
  --num_samples 3 \
  --dataset_size 500 \
  --prompt_suffix "\n\nExplain step by step."
```

## E.6 第 6 章：批处理 GRPO rollout

第 5 章中的自精炼本质上是顺序的，因此不会从简单的批处理中受益太多。原则上，可以为多个输入并行运行自精炼循环，但这实现起来非平凡，因此不是补充材料的一部分。相反，下一个批处理示例出现在第 6 章 RLVR 代码中。

在第 6 章中，我们再次对多个 rollout 使用相同的提示，因此不需要填充。如 E.5 节所述，代码因此可以保留来自 `reasoning_from_scratch.qwen3` 的常规 `Qwen3Model` 类。

相关脚本可以如下获取：

**代码清单 E.6 下载支持批处理的 RLVR 脚本**

```python
download_from_github(
    "ch06/02_rlvr_grpo_scripts_intro/"
    "rlvr_grpo_original_no_kl_batched.py"
)
```

与之前类似，要下载单生成脚本，只需从代码清单 E.6 中的文件名中删除 "_batched"。

典型的批处理运行如下所示：

```bash
uv run rlvr_grpo_original_no_kl_batched.py \
  --num_rollouts 8 \
  --batch_size 4 \
  --max_new_tokens 512
```

这里，`--batch_size` 控制在一个训练步骤中并行生成多少个 rollout。这提高了吞吐量，但也增加了内存压力，因此在实践中你可能需要减少 `--num_rollouts` 或 `--max_new_tokens`。

## E.7 第 8 章：批处理蒸馏

第 8 章返回到第 3 章的填充感知风格，因为蒸馏示例具有不同的提示和答案长度。换句话说，我们再次将不同长度的无关示例批处理到单个矩形张量中，这意味着掩码和填充再次变得重要。

我们可以如下下载批处理训练脚本并获取样本训练数据集：

**代码清单 E.7 下载支持批处理的蒸馏脚本**

```python
from reasoning_from_scratch.ch08 import load_distill_data

download_from_github(
    "ch08/04_train_with_distillation/distill_batched.py"
)
_ = load_distill_data(
    partition="deepseek-r1-math-train",
    local_path="deepseek-r1-math-train.json",
)
```

对于非批处理版本，从代码清单 E.7 中的文件名中删除 "_batched" 部分。

典型的批处理运行如下所示：

```bash
uv run distill_batched.py \
  --data_path deepseek-r1-math-train.json \
  --dataset_size 500 \
  --validation_size 10 \
  --epochs 2 \
  --use_think_tokens \
  --batch_size 32
```

这里，`--batch_size` 让我们每个优化步骤处理多个蒸馏示例，但由于训练还必须存储激活，资源需求可能比第 3 章的评估器高得多。因此，第 8 章为我们提供了本章前面介绍的填充感知批处理策略的训练类比。

## E.8 单序列与批处理生成

第 3 章和第 8 章的补充脚本将不同长度的示例批处理在一起，因此它们需要填充感知注意力掩码和更仔细的簿记。第 4 章和第 6 章将相同提示或 rollout 设置的重复副本批处理在一起，因此它们可以保留常规模型实现并避免更复杂的填充逻辑。

无论哪种方式，批处理生成都可以带来更高的吞吐量和更短的运行时间，代价是增加 RAM 使用。表 E.1 比较了使用前面章节中显示的设置的单示例与批处理生成。

**表 E.1 RAM 使用和运行时间比较**

| 脚本 | 批次大小 | H100 RAM | H100 总时间 | DGX Spark 总时间 |
|------|---------|---------|------------|----------------|
| 1 evaluate_math500.py | - | 1.8 GB | 90 min | 174 min |
| 2 evaluate_math500_batched.py | 64 | 23.39 GB | 16 min | 108 min |
| 3 self_consistency_math500.py | - | 1.79 GB | 252 min | 340 min |
| 4 self_consistency_math500_batched.py | 3 | 2.45 GB | 129 min | 243 min |
| 5 rlvr_grpo_original_no_kl.py | - | 43.35 GB | 68 min | 63 min |
| 6 rlvr_grpo_original_no_kl_batched.py | 4 | 44.91 GB | 19 min | 23 min |
| 7 distill.py | - | 8.29 GB | 10 min | 32 min |
| 8 distill_batched.py | 4 | 8.34 GB | 9 min | 28 min |

如表 E.1 所示，批处理变体总是比单序列变体更快。在大多数情况下，差异非常明显。一个例外是蒸馏（第 7 和第 8 行），其中批处理只产生适度的优势。

这可能是因为 GPU 在单批次模式下已经得到了很好的利用，而批处理生成中由于额外的掩码生成带来的额外开销不值得，鉴于 Qwen3 0.6B 参数模型的小模型大小。

---

# 附录 F LLM 评估的常见方法

## F.1 理解 LLM 的主要评估方法

在实践中评估训练好的 LLM 有四种常见方式：多项选择、验证器、排行榜和 LLM 评判者，如图 F.1 所示。研究论文、营销材料、技术报告和模型卡（LLM 特定技术报告的术语）通常包含来自这些类别中两个或更多类别的结果。

图 F.1 本书涵盖主题的思维模型，重点关注本附录涵盖的两大评估类别：基于基准的评估和基于评判的评估。

此外，如图 F.1 所示，这里介绍的四个类别分为两组：基于基准的评估和基于评判的评估。

其他度量，如训练损失、困惑度和奖励，通常在模型开发期间内部使用。（它们在模型训练章节中介绍。）

以下小节提供每种方法的简要概述。

## F.2 评估答案选择准确率

我们从基于基准的方法开始：多项选择问答。

历史上，最广泛使用的评估方法之一是多项选择基准，如 MMLU（Massive Multitask Language Understanding 的缩写，https://huggingface.co/datasets/cais/mmlu）。MMLU 数据集中的一个示例任务如图 F.2 所示。

图 F.2 通过比较 LLM 的多项选择预测与数据集中的正确答案来评估 LLM 在 MMLU 上的表现。

图 F.2 只显示了 MMLU 数据集中的一个示例。完整的 MMLU 数据集包含 57 个科目（从高中数学到生物学），总共约有 16,000 道多项选择题，性能以准确率（正确回答问题的比例）来衡量，例如，如果 16,000 个问题中有 14,000 个回答正确，则为 87.5%。

多项选择基准，如 MMLU，以类似于标准化测试、许多学校考试或理论驾驶测试的直接、可量化的方式测试 LLM 的知识回忆。

请注意，图 F.2 显示了一个简化的多项选择评估版本，其中模型的预测答案字母直接与正确答案进行比较。还存在另外两种流行的方法，涉及 log-probability 评分（log-probabilities 在第 4 章中有更详细的讨论）。

以下小节说明了图 F.2 中所示的 MMLU 评分如何在代码中实现。包括不同评分变体的端到端 MMLU 脚本将作为 bonus 材料提供在本书的代码仓库中。

### F.2.1 加载模型

首先，在我们可以评估 MMLU 之前，我们必须加载预训练模型。以下代码与第 3 章中的代码清单 3.1 相同。

**代码清单 F.1 加载预训练模型**

```python
from pathlib import Path
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small, Qwen3Tokenizer,
    Qwen3Model, QWEN_CONFIG_06_B
)

device = get_device()
torch.set_float32_matmul_precision("high")  # A
# device = "cpu"  # B
WHICH_MODEL = "base"  # C
if WHICH_MODEL == "base":
    download_qwen3_small(
        kind="base", tokenizer_only=False, out_dir="qwen3"
    )
    tokenizer_path = Path("qwen3") / "tokenizer-base.json"
    model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
elif WHICH_MODEL == "reasoning":
    download_qwen3_small(
        kind="reasoning", tokenizer_only=False, out_dir="qwen3"
    )
    tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
    model_path = Path("qwen3") / "qwen3-0.6B-reasoning.pth"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )
else:
    raise ValueError(f"Invalid choice: WHICH_MODEL={WHICH_MODEL}")
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))
model.to(device)
```

### F.2.2 检查生成的答案字母

在本节中，我们实现最简单且也许最直观的 MMLU 评分方法，它依赖于检查生成的多项选择答案字母是否与正确答案匹配。这与前面图 F.2 中所示的类似。

为此，我们将使用 MMLU 数据集中的一个示例：

```python
example = {
    "question": (
        "How many ways are there to put 4 distinguishable"
        " balls into 2 indistinguishable boxes?"
    ),
    "choices": ["7", "11", "16", "8"],
    "answer": "D",
}
```

接下来，我们定义一个函数来格式化 LLM 提示：

**代码清单 F.2 格式化 LLM 提示**

```python
def format_prompt(example):
    return (
        f"{example['question']}\n"
        f"A. {example['choices'][0]}\n"
        f"B. {example['choices'][1]}\n"
        f"C. {example['choices'][2]}\n"
        f"D. {example['choices'][3]}\n"
        "Answer: "  # A
    )
```

让我们在 MMLU 示例上执行该函数，以了解格式化后的 LLM 输入的样子：

```python
prompt = format_prompt(example)
print(prompt)
```

输出为：

```
How many ways are there to put 4 distinguishable balls into 2
indistinguishable boxes?
A. 7
B. 11
C. 16
D. 8
Answer:
```

如上面的模型提示所示，它为模型提供了不同答案选项的列表，并以 "Answer: " 文本结尾，鼓励模型生成正确答案。

虽然这不是严格必需的，但有时提供额外的问题以及正确答案作为输入也是有帮助的，这样模型可以观察它被期望如何解决任务。（例如，提供 5 个示例的情况也被称为 5-shot MMLU。）然而，对于当前一代的 LLM，即使基础模型也相当有能力，这并非必需。

> **加载不同的 MMLU 样本**
> 你可以通过 `datasets` 库直接加载 MMLU 数据集的示例（可以通过 `pip install datasets` 或 `uv add datasets` 安装）：
> ```python
> from datasets import load_dataset
> configs = get_dataset_config_names("cais/mmlu")
> dataset = load_dataset("cais/mmlu", "high_school_mathematics")> example = dataset["test"][0]  # A
> print(example)
> ```
> 上面，我们使用了 "high_school_mathematics" 子集；要获取其他子集的列表，请使用以下代码：
> ```python
> from datasets import get_dataset_config_names
> subsets = get_dataset_config_names("cais/mmlu")
> print(subsets)
> ```

接下来，我们对提示进行分词并将其包装在 PyTorch 张量对象中作为 LLM 的输入（类似于我们在第 2 章中所做的）：

```python
prompt_ids = tokenizer.encode(prompt)
prompt_fmt = torch.tensor(prompt_ids, device=device).unsqueeze(0)
```

然后，我们在代码清单 F.3 中定义主评分函数，它生成几个 token（这里默认 8 个 token）并提取模型打印的 A/B/C/D 字母的第一个实例。

**代码清单 F.3 提取生成的字母**

```python
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache
)

def predict_choice(
    model, tokenizer, prompt_fmt, max_new_tokens=8
):
    pred = None
    for t in generate_text_basic_stream_cache(
        model=model,
        token_ids=prompt_fmt,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        answer = tokenizer.decode(t.squeeze(0).tolist())
        for letter in answer:
            letter = letter.upper()
            if letter in "ABCD":  # A
                pred = letter
                break
        if pred:
            break
    return pred

pred1 = predict_choice(model, tokenizer, prompt_fmt)
print(
    f"Generated letter: {pred1}\n"
    f"Correct? {pred1 == example['answer']}"
)
```

结果如下：

```
Generated letter: C
Correct? False
```

如我们所见，在这种情况下生成的答案是不正确的（False）。

> **多项选择答案格式**
> 请注意，本节为实现说明目的实现了一个简化的多项选择评估版本，其中模型的预测答案字母直接与正确答案进行比较。在实践中，存在更流行的变体，如 log-probability 评分，我们衡量模型认为每个候选答案的可能性有多大，而不仅仅是检查最终字母选择。（我们在第 4 章中讨论基于概率的评分。）对于推理模型，评估还可以涉及评估生成正确答案的可能性，当正确答案作为输入提供时。
> 无论变体如何，评估仍然归结为检查模型是否从预定义的答案选项中选择。这些变体的示例将作为可选 bonus 材料包含在代码仓库中。

像 MMLU 这样的多项选择基准的一个限制是，它们只衡量 LLM 从预定义选项中选择的能力，因此在检查模型与基础模型相比遗忘了多少知识之外，对评估推理能力不是很有用。它不能捕捉自由形式写作能力或实际效用。不过，它仍然是一个简单且有用的诊断：高 MMLU 分数不一定意味着模型在实际使用中很强，但低分数可以突出潜在的知识差距。

## F.3 使用验证器检查答案

与上一节讨论的多项选择问答相关，基于验证的方法通过准确率指标来量化 LLM 的能力。然而，与多项选择基准相比，验证方法允许 LLM 提供自由形式的答案。然后我们提取相关答案部分，并使用所谓的验证器将答案部分与数据集中提供的正确答案进行比较，如图 F.3 所示。

图 F.3 使用基于验证的方法在自由形式问答中评估 LLM。模型生成自由形式的答案（可能包括多个步骤）和最终框选答案，提取该答案并与数据集中的正确答案进行比较。

如图 F.3 所示，当我们将提取的答案与提供的答案进行比较时，我们可以使用外部工具，如代码解释器或计算器软件。

缺点是，这种方法只能应用于可以轻松（并且理想情况下是确定性地）验证的领域，如数学和代码。此外，这种方法可能引入额外的复杂性和依赖性，并且可能将部分评估负担从模型本身转移到外部工具。

然而，因为它允许我们以编程方式生成无限数量的数学问题变体，并且受益于逐步推理，它已成为推理模型评估和开发的基石。

第 3 章提供了这种方法的广泛示例，这就是为什么我们在这里跳过代码演示。

## F.4 使用偏好和排行榜比较模型

到目前为止，我们已经介绍了两种提供易于量化指标（如模型准确率）的方法。然而，上述方法都没有以更全面的方式评估 LLM，包括判断响应的风格。在本节中，如图 F.4 所示，我们讨论一种基于评判的方法，即 LLM 排行榜。

图 F.4 本书涵盖主题的思维模型，重点关注本附录涵盖的基于评判和基于基准的评估方法。在上一节已经涵盖了基于基准的方法（多项选择、验证器）之后，我们现在介绍基于评判的方法来测量 LLM 性能，本小节重点关注排行榜。

图 F.4 中提到的排行榜方法是一种基于评判的方法，其中模型不是根据准确率值或其他固定基准分数排名，而是根据用户（或其他 LLM）对其输出的偏好排名。

一个流行的排行榜是 LM Arena（前身为 Chatbot Arena，https://lmarena.ai/），用户比较来自两个用户选择或匿名模型的响应，并为他们偏好的模型投票，如图 F.5 所示。

图 F.5 基于评判的排行榜界面示例（LM Arena）。两个 LLM 被给予相同的提示，它们的响应并排显示，用户投票选择偏好的答案。

如图 F.5 所示收集的这些偏好投票，然后汇总到所有用户的排行榜中，按用户偏好对不同模型进行排名。在本节的其余部分，我们将实现一个简单的排行榜示例。

为了创建一个具体示例，考虑用户在不同 LLM 的类似图 F.5 的设置中提示。下面的列表表示成对投票，其中第一个模型是获胜者：

```python
votes = [
    ("GPT-5", "Claude-3"),
    ("GPT-5", "Llama-4"),
    ("Claude-3", "Llama-3"),
    ("Llama-4", "Llama-3"),
    ("Claude-3", "Llama-3"),
    ("GPT-5", "Llama-3"),
]
```

在上面的列表中，`votes` 列表中的每个元组表示两个模型之间的成对偏好，写成 `(winner, loser)`。因此，`("GPT-5", "Claude-3")` 意味着用户偏好 GPT-5 而不是 Claude-3 模型答案。

在本节的其余部分，我们将把 `votes` 列表转换为排行榜。为此，我们将使用流行的 Elo 评分系统，该系统最初是为排名国际象棋选手而开发的。在我们查看具体代码实现之前，简而言之，它的工作原理如下。

每个模型从一个基线分数开始。然后，在每次比较和偏好投票之后，模型的评分会更新。具体而言，如果用户偏好当前模型而不是排名较高的模型，当前模型将获得相对较大的排名更新并在排行榜中排名更高。反之，如果当前模型战胜排名较低的模型，它只会增加一点点评分。（如果当前模型输了，它也会以类似方式更新，但排名分数会被减去而不是增加。）

将这些成对排名转换为排行榜的代码如代码清单 F.4 所示。

**代码清单 F.4 构建排行榜**

```python
def elo_ratings(vote_pairs, k_factor=32, initial_rating=1000):
    ratings = {  # A
        model: initial_rating
        for pair in vote_pairs
        for model in pair
    }
    for winner, loser in vote_pairs:  # B
        rating_winner, rating_loser = ratings[winner], ratings[loser]
        expected_winner = 1.0 / (  # C
            1.0 + 10 ** ((ratings[loser] - ratings[winner]) / 400.0)
        )
        ratings[winner] = (  # D
            ratings[winner] + k_factor * (1 - expected_winner)
        )
        ratings[loser] = (
            ratings[loser] + k_factor * (0 - (1 - expected_winner))
        )
    return ratings
```

代码清单 F.4 中的 `elo_ratings` 函数接受 `votes` 作为输入并将其转换为排行榜，如下所示：

```python
ratings = elo_ratings(votes, k_factor=32, initial_rating=1000)
for model in sorted(ratings, key=ratings.get, reverse=True):
    print(f"{model:8s} : {ratings[model]:.1f}")
```

这导致以下排行榜排名，其中分数越高越好：

```
GPT-5    : 1043.7
Claude-3 : 1015.2
Llama-4  : 1000.7
Llama-3  : 940.4
```

那么，这是如何工作的？对于每一对，我们使用以下公式计算获胜者的预期分数：

```python
expected_winner = 1 / (1 + 10 ** ((rating_loser - rating_winner) / 400))
```

`expected_winner` 值是模型在基于当前评分的无平局设置中的预测获胜机会。它决定了排名更新的大小。

首先，每个模型从 `initial_rating = 1000` 开始。如果两个评分（获胜者和失败者）相等，我们有 `expected_winner = 0.5`，表示势均力敌的比赛。在这种情况下，更新为：

```
rating_winner + k_factor * (1 - 0.5) = rating_winner + 16
rating_loser  + k_factor * (0 - (1 - 0.5)) = rating_loser - 16
```

现在，如果一个重量级热门（评分较高的模型）获胜，我们有 `expected_winner ≈ 1`。热门只获得少量，失败者只失去一点点：

```
rating_winner + 32 * (1 - 0.99) = rating_winner + 0.32
rating_loser  + 32 * (0 - (1 - 0.99)) = rating_loser - 0.32
```

然而，如果一个冷门（评分较低的模型）获胜，我们有 `expected_winner ≈ 0`，获胜者几乎获得完整的 `k_factor` 分数，而失败者失去大约相同的幅度：

```
rating_winner + 32 * (1 - 0.01) = rating_winner + 31.68
rating_loser  + 32 * (0 - (1 - 0.01)) = rating_loser - 31.68
```

> **顺序很重要**
> Elo 方法在每次比赛（模型比较）后更新评分，因此后面的结果建立在已经更新的评分之上。这意味着同一组结果，当以不同顺序呈现时，可能会以略微不同的最终分数结束。这种影响通常很轻微，但当 upset 发生得早或晚时可能会发生。
> 为了减少这种顺序效应，我们可以打乱 `votes` 对并多次运行 `elo_ratings` 函数，然后平均评分。

上述描述的排行榜方法提供了比静态基准分数更动态的模型质量视图。然而，结果可能受用户人口统计、提示选择和投票偏见的影响。基准和排行榜也可能被操纵，用户可能根据风格而不是正确性选择响应。最后，与自动化基准测试工具相比，排行榜不为新开发的变体提供即时反馈，这使得它们在积极的模型开发期间更难使用。

> **其他排名方法**
> LM Arena 最初使用本节描述的 Elo 方法，但最近过渡到基于 Bradley-Terry 模型的统计方法。Bradley-Terry 模型的主要优势在于，作为统计基础的方法，它允许构建置信区间来表达排名中的不确定性。此外，与 Elo 评分不同，Bradley-Terry 模型使用对整个数据集的统计拟合联合估计所有评分，这使其不受顺序效应影响。
> 为了使报告分数保持在熟悉的范围内，Bradley-Terry 模型被拟合以产生与 Elo 可比的值。尽管排行榜不再正式使用 Elo 评分，但术语 "Elo" 仍然被 LLM 研究人员和从业者广泛用于比较模型。本书的 bonus 材料中包含了显示 Elo 评分的代码示例：https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/03_leaderboards。

## F.5 使用其他 LLM 评判响应

在早期，LLM 使用统计和基于启发式的方法进行评估，包括一种称为 BLEU 的度量，它是生成文本与参考文本匹配程度的粗略度量。这种度量的问题在于它们需要精确的单词匹配，不考虑同义词、单词变化等。

如果我们想将书面答案文本作为一个整体来评判，解决这个问题的一种方法是使用上一节中讨论的相对排名和基于排行榜的方法。然而，排行榜的一个缺点是偏好比较的主观性，因为它涉及人类反馈（以及与收集此反馈相关的挑战）。

一种相关的方法是使用另一个具有预定义评分标准（即评估指南）的 LLM 来比较 LLM 的响应与参考响应，并根据预定义的标准评判响应质量，如图 F.6 所示。

图 F.6 LLM 评判评估示例。要评估的模型生成一个答案，然后由单独的评判 LLM 根据评分标准和提供的参考答案对答案进行评分。

在实践中，如图 F.6 所示的基于评判的方法在评判 LLM 强大时效果很好。常见的设置使用通过 API 访问的领先专有 LLM，尽管也存在专门的评判模型（参见附录 A 中的参考文献）。评判者工作得很好的原因之一是，评估答案通常比生成答案更容易。

要以编程方式在 Python 中实现如图 F.6 所示的基于评判的模型评估，我们可以加载 Qwen3 模型之一（附录 D），并用评分标准和我们要评估的模型答案提示它。

或者，我们可以通过 API 使用其他 LLM，例如 ChatGPT 或 Ollama API。在本节的其余部分，我们将使用 Python 中的 Ollama API 实现如图 F.6 所示的基于评判的评估。

具体而言，我们将使用 OpenAI 的 200 亿参数 gpt-oss 开放权重模型，因为它在能力和效率之间提供了良好的平衡。有关 gpt-oss 的更多信息，请参阅我的文章《From GPT-2 to gpt-oss: Analyzing the Architectural Advances》，网址为 https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the。

### F.5.1 在 Ollama 中实现 LLM 作为评判者的方法

Ollama（https://ollama.com）是一个用于在笔记本电脑上运行 LLM 的高效开源应用程序。它作为开源 llama.cpp 库（https://github.com/ggerganov/llama.cpp）的包装器，该库用纯 C/C++ 实现 LLM 以最大化效率。

然而，请注意，Ollama 只是使用 LLM 生成文本（推理）的工具，不支持训练或微调 LLM。

要执行以下代码，请通过访问 https://ollama.com 并按照操作系统提供的说明安装 Ollama：

- 对于 macOS 和 Windows 用户：打开下载的 Ollama 应用程序。如果提示安装命令行使用，请选择 "yes"。
- 对于 Linux 用户：使用 Ollama 网站上提供的安装命令。

在实现模型评估代码之前，让我们首先下载 gpt-oss 模型，并通过从命令行终端使用它来验证 Ollama 是否正常工作。在命令行（不是在 Python 会话中）执行以下命令来尝试 200 亿参数的 gpt-oss 模型：

```bash
ollama run gpt-oss:20b
```

第一次执行此命令时，将自动下载 200 亿参数的 gpt-oss 模型，该模型占用 14 GB 的存储空间。输出如下所示：

```
$ ollama run gpt-oss:20b
pulling manifest 
pulling b112e727c6f1: 100% ▕██████████████████████▏  13 GB
pulling fa6710a93d78: 100% ▕██████████████████████▏ 7.2 KB
pulling f60356777647: 100% ▕██████████████████████▏  11 KB
pulling d8ba2f9a17b3: 100% ▕██████████████████████▏   18 B
pulling 55c108d8e936: 100% ▕██████████████████████▏  489 B
verifying sha256 digest 
writing manifest 
removing unused layers 
success
>>> What is 1+2?
Thinking...
User asks: "What is 1+2?" This is simple: answer 3. Provide explanation? Possibly
ask for simple 
arithmetic. Provide answer: 3.
...done thinking.
1 + 2 = **3**
```

> **替代 Ollama 模型**
> 请注意，`ollama run gpt-oss:20b` 命令中的 `gpt-oss:20b` 指的是 200 亿参数的 gpt-oss 模型。使用 Ollama 配合 `gpt-oss:20b` 模型大约需要 13 GB 的 RAM。如果你的机器没有足够的 RAM，你可以尝试使用较小的模型，例如 40 亿参数的 `qwen3:4b` 模型，通过 `ollama run qwen3:4b`，它只需要大约 4 GB 的 RAM。
> 对于更强大的计算机，你也可以使用更大的 1200 亿参数 gpt-oss 模型，将 `gpt-oss:20b` 替换为 `gpt-oss:120b`。然而，请记住，此模型需要 significantly 更多的计算资源。

模型下载完成后，我们得到一个命令行界面，允许我们与模型交互。例如，尝试问模型 "What is 1+2?"：

你可以使用输入 `/bye` 结束此 `ollama run gpt-oss:20b` 会话。

在本节的其余部分，我们将使用 ollama API。这种方法要求 Ollama 在后台运行。有三种不同的选项可以实现这一点：

1. 在终端中运行 `ollama serve` 命令（推荐）。这将 Ollama 后端作为服务器运行，通常在 http://localhost:11434 上。请注意，直到通过 API 调用它时才加载模型（稍后在本节中）。
2. 类似于前面运行 `ollama run gpt-oss:20b` 命令，但保持打开状态，不要通过 `/bye` 退出会话。如前面所讨论的，这会打开一个围绕本地 Ollama 服务器的最小便利包装器。在幕后，它使用与 `ollama serve` 相同的服务器 API。
3. Ollama 桌面应用程序。打开桌面应用程序会自动运行相同的后端，并在其上提供图形界面，如前面图 F.6 所示。

> **Ollama 服务器 IP 地址**
> Ollama 通过启动本地服务器类进程在我们的机器上本地运行。当在终端中运行 `ollama serve` 时，如上所述，你可能会遇到错误消息说 `Error: listen tcp 127.0.0.1:11434: bind: address already in use`。
> 如果是这种情况，尝试使用命令 `OLLAMA_HOST=127.0.0.1:11435 ollama serve`（如果这个地址也在使用中，尝试将数字递增一，直到找到未使用的地址。）

以下代码验证 Ollama 会话在我们在上一节中使用 Ollama 评估生成的测试集响应之前是否正常运行：

**代码清单 F.5 检查 Ollama 是否正在运行**

```python
import psutil

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")
if not ollama_running:
    raise RuntimeError(
        "Ollama not running. Launch ollama before proceeding."
    )
print("Ollama running:", check_if_running("ollama"))
```

确保执行前面的代码显示的输出为 `Ollama running: True`。如果显示 `False`，请验证 `ollama serve` 命令或 Ollama 应用程序是否正在 actively 运行。

在本附录的其余部分，我们将通过 Python 中的 Ollama REST API 与在我们机器上运行的本地 gpt-oss 模型交互。以下 `query_model` 函数演示了如何使用 API：

**代码清单 F.6 查询本地 Ollama 模型**

```python
import json
import requests

def query_model(
    prompt,
    model="gpt-oss:20b",
    url="http://localhost:11434/api/chat"  # A
):
    data = {  # B
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {  # C
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    # D
    with requests.post(url, json=data, stream=True, timeout=30) as r:  
        r.raise_for_status()  # E
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):  # F
            if not line:
                continue
            response_json = json.loads(line)  # G
            if "message" in response_json:  # H
                response_data += response_json["message"]["content"]
        return response_data
    return response_data
```

以下是我们刚刚实现的代码清单 F.6 中 `query_model` 函数的使用示例：

```python
ollama_model = "gpt-oss:20b"
result = query_model("What is 1+2?", ollama_model)
print(result)
```

得到的结果是 "3"。（这与我们运行 Ollama run 或 Ollama 应用程序时得到的结果不同，因为默认设置不同。）

使用 `query_model` 函数，我们可以用一个提示来评估我们的模型生成的响应，该提示包括一个评分标准，要求 gpt-oss 模型根据正确答案作为参考，以 1 到 5 的等级评分我们目标模型的响应。

我们为此使用的提示如代码清单 F.7 所示：

**代码清单 F.7 设置包含评分标准的提示模板**

```python
def rubric_prompt(instruction, reference_answer, model_answer):
    rubric = (
        "You are a fair judge assistant. You will be given an instruction, "
        "a reference answer, and a candidate answer to evaluate, according "
        "to the following rubric:\n\n"
        "1: The response fails to address the instruction, providing "
        "irrelevant, incorrect, or excessively verbose content.\n"
        "2: The response partially addresses the instruction but contains "
        "major errors, omissions, or irrelevant details.\n"
        "3: The response addresses the instruction to some degree but is "
        "incomplete, partially correct, or unclear in places.\n"
        "4: The response mostly adheres to the instruction, with only "
        "minor errors, omissions, or lack of clarity.\n"
        "5: The response fully adheres to the instruction, providing a "
        "clear, accurate, and relevant answer in a concise and efficient "
        "manner.\n\n"
        "Now here is the instruction, the reference answer, and the "
        "response.\n"
    )
    prompt = (
        f"{rubric}\n"
        f"Instruction:\n{instruction}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Answer:\n{model_answer}\n\n"
        f"Evaluation: "
    )
    return prompt
```

`rubric_prompt` 中的 `model_answer` 旨在代表我们自己在实践中产生的模型响应。为说明目的，我们硬编码一个合理的模型答案，而不是动态生成它。（然而，请随意使用我们在 F.2.1 节中加载的 Qwen3 模型来生成真实的 `model_answer`。）

接下来，让我们为 Ollama 模型生成渲染后的提示：

```python
rendered_prompt = rubric_prompt(
    instruction=(
        "If all birds can fly, and a penguin is a bird, "
        "can a penguin fly?"
    ),
    reference_answer=(
        "Yes, according to the premise that all birds can fly, "
        "a penguin can fly."
    ),
    model_answer=(
        "Yes – under those premises a penguin would be able to fly."
    )
)
print(rendered_prompt)
```

输出如下：

```
You are a fair judge assistant. You will be given an instruction, a
reference answer, and a candidate answer to evaluate, according to the
following rubric:
1: The response fails to address the instruction, providing irrelevant,
incorrect, or excessively verbose content.
2: The response partially addresses the instruction but contains major
errors, omissions, or irrelevant details.
3: The response addresses the instruction to some degree but is
incomplete, partially correct, or unclear in places.
4: The response mostly adheres to the instruction, with only minor
errors, omissions, or lack of clarity.
5: The response fully adheres to the instruction, providing a clear,
accurate, and relevant answer in a concise and efficient manner.
Now here is the instruction, the reference answer, and the response.
Instruction:
If all birds can fly, and a penguin is a bird, can a penguin fly?
Reference Answer:
Yes, according to the premise that all birds can fly, a penguin can
fly.
Answer:
Yes – under those premises a penguin would be able to fly.
Evaluation: 
```

以 "Evaluation: " 结尾的提示激励模型生成答案。让我们看看 gpt-oss:20b 模型如何评判响应：

```python
result = query_model(rendered_prompt, ollama_model)
print(result)
```

响应如下：

```
**Score: 5**
The candidate answer directly addresses the question, correctly applies the given
premises, and concisely states that a penguin would be able to fly. It is
accurate, relevant, and clear.
```

如我们所见，答案获得了最高分，这是合理的，因为它确实是正确的。虽然这是一个手动逐步完成过程的简单示例，但我们可以进一步实现一个 for 循环，迭代地查询模型（例如，我们在第 2 章加载的 Qwen3 模型），使用评估数据集中的问题，并通过 gpt-oss 评估它并计算平均分数。

然后，对两个模型执行此操作（例如，Qwen3 基础和推理模型），我们可以相对地比较模型。

> **使用过程奖励模型评分中间推理步骤**
> 与符号验证器和 LLM 评判者相关，有一类称为过程奖励模型（PRM）的学习模型。与评判者一样，PRM 可以评估超越最终答案的推理轨迹，但与通用评判者不同，它们专门关注推理的中间步骤。与验证器不同，验证器符号化地检查正确性，通常只在结果层面，PRM 在强化学习训练期间提供逐步奖励信号。我们可以将 PRM 归类为“步骤级评判者”，它们主要是为训练而开发的，而不是纯粹的评估。（在实践中，PRM 很难在大规模上可靠地训练。例如，DeepSeek R1 没有采用 PRM，而是将验证器与推理训练相结合。）

基于评判的评估相对于基于偏好的排行榜具有优势，包括可扩展性和一致性，因为它们不依赖大量的人类投票者。（从技术上讲，也可以将排行榜背后的基于偏好的评分外包给 LLM 评判者。）然而，LLM 评判者也与人类投票者共享类似的弱点：结果可能受模型偏好、提示设计和答案风格的影响。此外，对评判模型和评分标准的选择有很强的依赖性，并且它们缺乏固定基准的可复现性。

---

# 附录 G 构建聊天界面

本书的补充代码仓库包含一个基于浏览器的聊天界面，使用开源的 Chainlit Python 包（https://github.com/Chainlit/chainlit）构建，通过类似 ChatGPT 的界面与本书中的各种 LLM 和推理模型交互，如图 G.1 的截图所示。

图 G.1 Qwen3 模型的 Chainlit 界面，模仿 ChatGPT 界面。

图 G.1 中所示的界面在我们想要与模型交互时非常有用，比在笔记本单元格或终端中输入提示更方便。

基于 Chainlit 库，实现相对简单直接。

我们不是将 token 打印到终端，而是将前面章节中相同的模型加载和生成功能包装在一个小型 Web 应用程序中，该应用程序保持在我们的计算机上本地运行，并在我们的浏览器中运行。

在实践中，这意味着我们可以重用从零开始的 Qwen3Model 实现和第 2 章的流式辅助函数，同时让 Chainlit 处理浏览器 UI 和消息事件。

补充材料在 chG/01_main-chapter-code（https://github.com/rasbt/reasoning-from-scratch/tree/main/chG/01_main-chapter-code）中包含两个变体：

- `qwen3_chat_interface.py`，这是一个单轮界面；
- `qwen3_chat_interface_multiturn.py`，它存储对话历史并支持多轮交互。

在本附录中，我们将介绍这两个版本。我们还将讨论如何安装 Chainlit，以及如何通过第 7 章中的 `download_from_github` 辅助函数下载脚本。

## G.1 安装 Chainlit

在运行聊天界面之前，我们首先需要安装 Chainlit。最简单的选项是：

```bash
pip install chainlit
```

如果你使用 uv，等效命令是：

```bash
uv add chainlit
```

如果你直接在 reasoning-from-scratch 仓库中工作，Chainlit 也作为可选依赖列在项目的 `pyproject.toml` 中。在这种情况下，一个方便的替代方案是：

```bash
uv sync --extra extra
```

或：

```bash
pip install -e ".[extra]"
```

## G.2 将代码作为脚本运行

Chainlit 运行普通的 Python 脚本。这很重要，因为 Chainlit 将脚本作为模块导入，并查找用 `@chainlit.on_chat_start` 和 `@chainlit.on_message` 装饰的特殊函数，我们稍后会看到。

这意味着如果我们想重用本附录中显示的代码，我们应该将其放入一个 `.py` 文件中，例如：

```
qwen3_chat_interface.py
```

或：

```
app.py
```

然后通过终端启动它：

```bash
chainlit run app.py
```

或者，如果你使用 uv，此命令变为：

```bash
uv run chainlit run app.py
```

换句话说，虽然我们可以在笔记本中检查代码，但实际的 Chainlit 应用程序预期存在于一个独立的 Python 脚本中。

## G.3 下载脚本

如果你克隆了仓库，文件已经存在于 chG/01_main-chapter-code 中，你可以跳过此步骤。否则，如第 7 章所讨论的，我们可以使用 `download_from_github(...)` 来获取两个脚本：

**代码清单 G.1 下载 Chainlit 脚本**

```python
from reasoning_from_scratch.ch07 import download_from_github
download_from_github(
    "chG/01_main-chapter-code/qwen3_chat_interface.py"
)
download_from_github(
    "chG/01_main-chapter-code/qwen3_chat_interface_multiturn.py"
)
```

## G.4 常规单轮脚本

`qwen3_chat_interface.py` 中的常规单轮版本有意保持紧凑。它几乎直接重用前面的章节代码，并将其包装在最小数量的 Chainlit 特定逻辑中。

这里，“单轮”意味着我们仍然可以提交多个查询，但每个查询彼此独立，不知道前一个查询。

代码相对较短，如代码清单 G.2 所示，完整显示。

**代码清单 G.2 Chainlit 单轮查询脚本**

```python
import os
from pathlib import Path
import torch
import chainlit
from reasoning_from_scratch.ch02 import (
    get_device,
)
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer,
    load_tokenizer_only
)
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

# A 配置部分
WHICH_MODEL = "reasoning"  # B 更改为 "base" 以使用基础模型
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
COMPILE = False
DEVICE = get_device()

def load_app_model_and_tokenizer():
    if CHECKPOINT_PATH is None:
        return load_model_and_tokenizer(
            which_model=WHICH_MODEL,
            device=DEVICE,
            use_compile=COMPILE,
            local_dir=LOCAL_DIR,
        )
    checkpoint_path = Path(CHECKPOINT_PATH)  # C 可选地从第 7 或 8 章加载训练检查点
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )
    tokenizer = load_tokenizer_only(
        which_model=WHICH_MODEL, local_dir=LOCAL_DIR
    )
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )
    model.to(DEVICE)
    if COMPILE:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)
    return model, tokenizer

MODEL, TOKENIZER = load_app_model_and_tokenizer()
EOS_TOKEN_IDS = (
    TOKENIZER.encode("<|im_end|>")[0],
    TOKENIZER.encode("<|endoftext|>")[0]
)

# D 初始化每会话状态并用默认系统提示填充
@chainlit.on_chat_start
async def on_start():
    chainlit.user_session.set("history", [])
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}
    )

# E 处理一条用户消息并将模型响应流式传输到 Chainlit UI
@chainlit.on_message
async def main(message: chainlit.Message):
    # F 编码输入
    input_ids = TOKENIZER.encode(message.content)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)
    # G 开始一条我们可以流式传输的传出消息
    out_msg = chainlit.Message(content="")
    await out_msg.send()
    # H 流式生成
    for tok in generate_text_basic_stream_cache(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        token_id = tok.squeeze(0).item()
        if token_id in EOS_TOKEN_IDS:
            break
        piece = TOKENIZER.decode([token_id])
        await out_msg.stream_token(piece)
    # I 完成流式传输的消息
    await out_msg.update()
```

代码清单 G.2 中的 `CHECKPOINT_PATH` 环境变量是可选的。如果设置了，脚本将加载该自定义 `.pth` 检查点，而不是默认的官方权重。`load_app_model_and_tokenizer()` 辅助函数处理这两种情况。如果没有提供自定义检查点，它将委托给第 3 章的 `load_model_and_tokenizer(...)`。如果提供了自定义检查点路径，它只通过 `load_tokenizer_only(...)` 加载分词器，构造一个新的 `Qwen3Model(QWEN_CONFIG_06_B)`，然后用自定义状态字典加载权重。

这很有用，因为它意味着单轮 Chainlit 界面也可以用于交互式检查第 6、7 或 8 章的检查点，前提是分词器设置与检查点保持对齐。下一节将解释如何从命令行终端提供自定义检查点作为输入。

`@chainlit.on_chat_start` 函数初始化一个历史对象并插入一个默认系统消息，这是一个常见的约定：

```python
{"role": "system", "content": "You are a helpful assistant."}
```

然而，单轮脚本在生成响应时实际上并不使用存储的历史记录。在 `main(...)` 内部，提示仅由当前的 `message.content` 构建：

```python
input_ids = TOKENIZER.encode(message.content)
```

因此，即使浏览器 UI 看起来像一个聊天应用程序，常规版本的行为更像一个提示框，它只是视觉上显示较早的消息，而模型实际上并不记得较早的对话。换句话说，模型本身只看到当前用户消息，并将每一轮视为独立的。

其余的生成逻辑与我们已经在第 2 章中使用的非常相似。

## G.5 运行单轮脚本

如果我们在补充代码仓库内部，启动用户界面最方便的命令通常是：

```bash
uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface.py
```

如果我们将代码保存在另一个文件名下，例如 `app.py`，我们将简单地相应地替换路径：

```bash
uv run chainlit run app.py
```

启动服务器后，Chainlit 通常会自动打开一个浏览器标签页。如果没有，我们可以手动将终端中显示的本地地址复制到浏览器中。通常是 http://localhost:8000。

图 G.2 显示了 Chainlit 界面中的一个示例输出。

图 G.2 Chainlit 界面中的示例响应。

默认情况下，脚本初始化我们在第 3 章中简要介绍的 Qwen3 推理变体，它提供了图 G.2 中所示的响应。要使用自定义检查点运行相同的脚本，我们通过 `CHECKPOINT_PATH` 环境变量传递路径。例如：

```bash
CHECKPOINT_PATH=path/to/qwen3-0.6B-distill-step06682-epoch1.pth \
uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface.py
```

这样做时，`WHICH_MODEL` 必须与检查点期望的分词器保持对齐。第 8 章的蒸馏检查点使用推理分词器，因此在这种情况下我们应该保持 `WHICH_MODEL = "reasoning"`。对于第 6 章的检查点，例如，我们将其更改为 `WHICH_MODEL = "base"`。

直接在脚本中编辑配置可能看起来有点麻烦。这样做的原因，而不是使用命令行参数，是因为 Chainlit 本身通过 `chainlit run ...` 启动和管理应用程序入口点，因此脚本不像我们可以自由定义和解析自定义命令行参数（例如使用 Python argparse 库）的普通独立 Python 程序那样运行。因此，对于模型选择、生成长度或检查点路径等简单选项，将它们保留在脚本中或通过环境变量传递更方便。

> **下载模型检查点**
> 如果你想探索第 6、7 或 8 章中的一些模型检查点，但没有计算能力自己训练它们，你可以在补充材料中找到如何下载它们的信息：https://github.com/rasbt/reasoning-from-scratch/tree/main/ch07/04_download_trainining_checkpoints 和 https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/05_download_training_checkpoints。

## G.6 多轮界面

第二个脚本 `qwen3_chat_interface_multiturn.py` 扩展了基本的 Chainlit 界面，使模型可以在同一次对话中基于较早的轮次进行条件生成。以下小节解释了多轮在实际中的含义、脚本如何实现它，以及何时适合使用它。

### G.6.1 多轮的含义

在单轮设置中，模型只看到当前输入。为说明目的，假设交互是：

```
User: What is 1 + 1?
Assistant: 2
User: What did I just ask?
```

在常规单轮脚本中，当模型收到第二条消息 "What did I just ask?" 时，它可能会回答 "I don't know what you are asking"。原因是较早的问题和答案在浏览器中可见，但它们没有包含在模型输入中。因此模型没有真正的对话记忆。

在多轮设置（`qwen3_chat_interface_multiturn.py`）中，我们显式地在提示中包含（或者更具体地说，前置）较早的消息，类似于 ChatGPT 和其他聊天助手的做法。因此，在这种情况下，模型可能会用 "You asked me what 1+1 is" 来回答 "What did I just ask?" 这个问题。

### G.6.2 多轮变体

多轮脚本的开始方式与常规版本相同，但主要区别在于它添加了两个辅助函数，然后在生成期间实际使用存储的会话历史记录。

第一个辅助函数，如代码清单 G.3 所示，将存储的消息历史转换为我们这里需要的 Qwen 聊天格式。

**代码清单 G.3 从对话历史构建提示**

```python
def build_prompt_from_history(history, add_assistant_header=True):
    parts = []
    for m in history:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_assistant_header:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)
```

`build_prompt_from_history` 函数将消息列表序列化为 Qwen 推理模型期望的聊天模板样式。从概念上讲，生成的提示如下所示：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 1 + 1?<|im_end|>
<|im_start|>assistant
2<|im_end|>
<|im_start|>user
What did I just ask?<|im_end|>
<|im_start|>assistant
```

模型然后从最后的 assistant 头部继续生成。

第二个辅助函数修剪消息，如代码清单 G.4 所示，如果它们超过上下文长度，则需要修剪消息，因为多轮对话可能变得非常长。

**代码清单 G.4 将提示修剪以适应上下文窗口**

```python
def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    assert max_new_tokens < context_len
    keep_len = max(1, context_len - max_new_tokens)
    # A 如果提示太长，左截断到 keep_len
    if input_ids_tensor.shape[1] > keep_len: # A
        input_ids_tensor = input_ids_tensor[:, -keep_len:]
    return input_ids_tensor
```

`trim_input_tensor` 辅助函数确保上下文窗口中仍有足够的空间容纳新 token。如果对话变得太长，最旧的 token 首先通过左截断被丢弃。（在实践中，还存在其他更复杂的选项，例如总结前面的上下文；然而，为简单起见，这里未实现。）

### G.6.3 多轮脚本如何使用历史记录

实际的多轮行为出现在消息处理程序中。在每一轮开始时，脚本检索当前会话历史记录并追加新的用户消息：

```python
history = chainlit.user_session.get("history")
history.append({"role": "user", "content": message.content})
```

然后它从历史记录构建完整提示，对其进行分词，可选地修剪它，并以与之前相同的方式流式传输答案。

在生成结束时，它将模型回复存储回同一个历史列表：

```python
history.append({"role": "assistant", "content": out_msg.content})
chainlit.user_session.set("history", history)
```

这是相对于单轮脚本的核心区别。多轮版本不仅仅在浏览器中显示聊天记录，即它实际上将较早的轮次反馈回模型，以便未来的响应可以以它们为条件。

启动命令类似于单轮情况：

```bash
uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface_multiturn.py
```

在操作上，脚本的行为与常规 Chainlit 界面完全相同，如图 G.3 所示。唯一的区别在于内部提示的构建方式。

图 G.3 Chainlit 界面中多个提示和响应的示例。

图 G.3 中的第二个响应表明模型通过提供的上下文记住了第一个问题。

### G.6.4 建议

有一个重要的实际限制值得强调。多轮 Chainlit 脚本适用于官方 Qwen3 推理变体。它不适合第 8 章的蒸馏检查点，例如。

原因是蒸馏检查点是在提示-响应风格的监督上训练的，而不是在持久多轮聊天历史上训练的。换句话说，它们没有以与官方推理检查点相同的方式训练为在多个轮次中继续对话。

因此，多轮脚本不应被期望与蒸馏检查点可靠地工作。实际建议是：

- 使用单轮 Chainlit 脚本进行快速交互测试，包括自定义的第 6、7 和 8 章检查点；
- 当你想要对话记忆时，使用多轮 Chainlit 脚本与官方推理检查点配合。


