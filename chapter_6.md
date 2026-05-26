# 6 使用强化学习训练推理模型

本章涵盖以下内容：
- 基于人类反馈的强化学习（RLHF）与基于可验证奖励的强化学习（RLVR）之间的区别
- 将推理 LLM 的训练作为带有任务正确性奖励的强化学习问题
- 对每个提示采样多个响应以计算组相对学习信号
- 使用基于组的策略优化更新 LLM 权重以提升推理能力

推理性能和答案准确性可以通过增加推理计算预算以及特定的模型训练方法来提升。如图 6.1 所示，本章聚焦于强化学习——这是训练推理模型最常用的方法。

---

**图 6.1** 本书涵盖主题的思维模型。本章聚焦于通过额外训练（阶段 4）来提升推理的技术。具体来说，本章涵盖强化学习。

下一节将概述 LLM 背景下的强化学习，然后讨论两种用于 LLM 的常见强化学习方法。

## 6.1 LLM 强化学习简介

推理时间缩放（inference-time scaling）和训练时间缩放（training-time scaling）是提升大语言模型推理性能的两种不同方法，如图 6.2 所示。推理时间缩放通过为每个生成的答案投入更多计算来提高准确性，而训练时间缩放则通过在训练期间投入额外计算来提升准确性。本章聚焦于训练时间缩放。

---

**图 6.2** 推理时间缩放与训练时间缩放的概念对比。增加推理时的计算量通过为每个答案生成投入更多资源来提升准确性，而增加训练时的计算量则通过预先投入更多资源来提升准确性。

尽管图 6.2 将推理时间缩放和训练时间缩放作为独立概念呈现，但在实践中它们可以结合使用。例如，在通过强化学习（RL）——本章的核心焦点——提升模型的推理能力后，可以应用第 4 章和第 5 章介绍的推理时间缩放技术来进一步提升性能。

强化学习关注模型如何从一系列动作及其结果中学习，经典例子包括训练智能体玩国际象棋、围棋或电子游戏。虽然 LLM 的强化学习建立在这些文献基础之上，但它在重要方面有所不同，并且通常类似于熟悉的 LLM 训练技术。由于本书聚焦于语言模型，我们不涵盖通用的强化学习，而是专注于强化学习在 LLM 训练中的应用。

那么，强化学习在实践中对 LLM 意味着什么？从高层次来看，强化学习通常作为后训练阶段应用于预训练语言模型之上，有时在指令微调之后。这一设置如图 6.3 所示。

---

**图 6.3** LLM 的常见训练阶段。推理训练和偏好调优阶段的顺序可以变化，某些流水线会交错进行推理训练和偏好调优，而不是严格按顺序执行。

LLM 有两个常见的强化学习阶段：推理训练和偏好调优。如图 6.3 所示，强化学习通常在指令微调之后应用，而偏好调优通常跟随推理训练。然而，正如 DeepSeek 团队在其 DeepSeek-R1 工作中所展示的，专注于推理的强化学习也可以直接应用于预训练的基础模型，跳过指令微调和偏好调优。

由此产生的模型在推理准确性方面通常比经过完整训练流水线的模型要弱，但它仍然表现出清晰且稳健的推理行为。更重要的是，直接将推理训练应用于基础模型可以避免混合多个训练阶段的效果，这使得更容易将任何观察到的改进具体归因于推理训练本身。

在深入了解强化学习在 LLM 中的实现细节之前，简要地在概念层面将强化学习与预训练进行比较是有用的。在预训练期间，LLM 被训练来预测大量文本中的下一个 token，许多模型的核心能力和涌现行为（例如，回答基于知识的问题和遵循简单指令）在这个阶段已经发展起来。

请注意，预训练是我之前的书《从零构建大语言模型》（Build a Large Language Model (From Scratch)）的主要焦点，但熟悉该材料并不是理解本章或本书其余部分的必要条件。

在预训练模型之上应用强化学习很有吸引力，因为它让我们能够优化整个输出，例如答案的正确性或偏好，而不是单个 token 和下一个 token 预测。从这个意义上说，预训练主要构建知识，而强化学习塑造模型如何使用这些知识，包括其推理行为。

如图 6.4 所示，接下来的两个小节将概述强化学习在 LLM 训练中的使用方式，并为本章使用的方法设定背景。

---

**图 6.4** 本章路线图。在本节简要介绍 LLM 的强化学习（RL）之后，我们将在下一节讨论两种强化学习阶段 RLHF 和 RLVR 之间的区别。然后，我们将在本章剩余部分聚焦于使用 GRPO 算法实现 RLVR。

### 6.1.1 基于人类反馈的原始强化学习流水线（RLHF）

2022 年，InstructGPT 论文（https://arxiv.org/abs/2203.02155）引入了基于人类反馈的强化学习（RLHF），这是一种使用人类偏好标签来修改模型行为的训练方法。

事实上，RLHF 是将 GPT-3 转变为 ChatGPT 最初使用的模型的关键成分，而 ChatGPT 可以说是让 LLM 在 2022 年广受欢迎的原因。

从高层次来看，RLHF 是实现上一节讨论的偏好调优阶段的最流行方法。在 RLHF 之前，LLM 主要通过预训练进行训练，在某些情况下还会进行监督微调，这两者都基于下一个 token 预测目标。

RLHF 超越了这种 token 级别的优化，并在整个输出上优化模型（在这种情况下，RLHF 优化的是人类如何对 LLM 输出进行排名和评估）。

为了更具体地说明这一点，请考虑以下示例。假设提示是："我想买一台用于编程和日常使用的笔记本电脑。我应该考虑什么？"

模型可能会生成两个候选回答：

**回答 A：** "你应该考虑 CPU、内存、存储、GPU、屏幕分辨率、电池续航、键盘质量、接口选择、散热设计和价格。"

**回答 B：** "对于编程和日常使用，重点关注快速的 CPU、至少 16 GB 的内存以及至少 256 GB 存储的 SSD。舒适的键盘和良好的电池续航也很重要。只有当你计划在本地训练小型 LLM 时，GPU 才可能重要。"

两个回答都提到了相关要点，但人类数据标注员可能更喜欢回答 B，因为它更具体，并且针对提示中陈述的用例。

在 RLHF 中，这种成对偏好在许多提示上被收集，并用于训练模型以偏爱人类持续排名更高的回答。

如图 6.5 所示，RLHF 有两个主要步骤：(1) 训练一个奖励模型（其本身也是一个 LLM），以及 (2) 使用该奖励模型为目标 LLM 的输出打分并进行微调。

---

**图 6.5** 基于人类反馈的强化学习（RLHF）两阶段概述。首先，奖励模型在人类排名的回答上进行训练，为每个回答分配偏好分数。其次，LLM 在强化学习目标中使用这些奖励分数进行更新，以鼓励偏好的回答并抑制不太理想的回答。

RLHF 的第一步是训练奖励模型，如图 6.5 的顶部子面板所示。在这里，我们让 LLM 为每个提示生成多个回答（例如，使用第 4 章解释的温度缩放和 top-p 过滤方法），并要求人类标注员根据他们的人类偏好将答案从最好到最差进行排名。

这些人类偏好通过统计偏好模型转换为奖励模型的训练目标，该模型将成对比较映射为相对质量分数。奖励模型通常是一个从预训练 LLM 初始化的独立模型，它被训练为每个输出一个单数字（标量）奖励分数，以反映其感知质量。

其动机是奖励模型可以自动为新的模型输出打分，从而消除在每个训练步骤中进行人工标注的需要。

RLHF 的第二步是在奖励模型分配给 LLM 每个答案的奖励分数上训练 LLM（例如，一个不好的答案可能获得 -2，一个好的答案可能获得 +2）。

我们在这里讨论 RLHF，是因为它可以被视为专注于推理的强化学习方法（例如基于可验证奖励的强化学习（RLVR））的前身。

虽然 RLHF 和 RLVR 的训练信号来源不同，但流水线的底层结构（生成回答、为它们打分、通过强化学习更新模型）密切相关。

### 6.1.2 从人类反馈到可验证奖励（RLVR）

RLHF 可能相当复杂，因为它需要训练一个额外的奖励模型，而该模型通常也是一个大型 LLM，训练成本（也）很高。

基于可验证奖励的强化学习（RLVR）通过用可验证奖励替换 LLM 奖励模型来简化这一设置。可验证奖励是确定性地计算出来的，无需人工标注。例如，第 3 章介绍的数学验证器就是数学问题的可验证奖励生成器。

例如，给定一个数学问题（和正确答案），数学验证器自动检查生成的答案的最终结果是否与真实答案匹配，并分配相应的奖励（正确答案得 1 分，错误答案得 0 分）。

因此，RLHF 的两步流水线在 RLVR 中坍缩为单个训练循环，类似于 RLHF 中的第 2 步：模型生成回答，验证器分配奖励，这些奖励被直接用于更新模型。

这个简化的 RLVR 训练过程如图 6.6 所示。

---

**图 6.6** 基于可验证奖励的强化学习（RLVR）概述。LLM 生成一个由确定性验证器评估的回答，例如第 3 章的数学验证器，它分配一个正确性标签，用作强化学习目标中的奖励信号来更新模型。

RLVR 的流行很大程度上可以追溯到 2025 年 DeepSeek-R1 的成功（https://arxiv.org/abs/2501.12948），它证明了无需依赖人类偏好数据或学习得到的奖励模型，也能实现强大的推理性能。

DeepSeek-R1 使用自动可验证奖励训练推理行为，包括数学问题的正确性检查（类似于我们第 3 章的验证器）以及代码相关任务的代码编译和执行。

代码执行超出了本书的范围，因为它需要在一个安全的执行环境中训练 LLM。在实践中，这意味着在隔离的沙箱中编译和运行模型生成的代码，使其无法以不安全的方式访问主机系统、私有文件或外部服务，这使得设置比解决数学问题复杂得多。

无论如何，验证数学任务（检查答案是否正确，作为二元标签）和代码编译（检查代码是否编译，也作为二元标签）背后的理念是相似的，并且 RLVR 训练算法将是相同的。

总而言之，与 RLHF 相比，RLVR 提供了几个实际优势。它消除了训练和维护一个独立奖励模型的需要，而该模型通常在规模和成本上与基础 LLM 相当。RLVR 需要访问可靠的验证信号，这将其限制在某些领域，例如数学和代码。

可验证奖励也是确定性和可复现的，意味着它们每次对相同的输出都会产生相同的结果，这避免了人类偏好标注中出现的噪声和不一致性。

最后，RLVR 自然地扩展到大型训练集，因为，给定一个参考解决方案，奖励可以自动为任意数量的模型生成答案计算，而无需额外的人工标注工作。

## 6.2 使用 GRPO 进行基于可验证奖励的强化学习逐步讲解

既然我们已经介绍了整体情况，并了解了强化学习如何融入 LLM 的整体开发周期，我们现在进入具体实现。具体来说，我们实现 RLVR 来训练一个推理模型，如图 6.7 的章节路线图所示。这种方法类似于 DeepSeek-R1-Zero，但我们在小得多的规模上应用它，使用更小的模型和更少的数据，因为否则一个可比的完整训练运行将需要数十万美元的 GPU 计算资源。

---

**图 6.7** 在介绍了 LLM 的两种主要强化学习方法 RLHF 和 RLVR 之后，剩余章节聚焦于使用 GRPO 算法实现 RLVR，从数据集加载到实现完整的训练循环。

请注意，RLVR 定义了整体训练设置，即使用自动可验证奖励。此外，我们需要一个具体的策略优化算法，可以在该 RLVR 框架内使用来更新模型权重并训练模型。术语"策略"（policy）是强化学习领域的行话，在这个特定上下文中指的是我们想要训练的 LLM。

具体来说，我们将使用组相对策略优化（Group Relative Policy Optimization, GRPO）算法进行策略优化。

简而言之，RLVR 决定了可用的学习信号是什么，而 GRPO 决定了如何使用该信号来更新模型权重。

> **策略梯度算法**
>
> 一种广泛用于 LLM 的策略梯度算法是近端策略优化（Proximal Policy Optimization, PPO），它在 RLHF 中得到了普及。原则上，相同的算法也可以应用于 RLVR 设置。
>
> 在训练 DeepSeek-R1 推理模型时，DeepSeek 团队选择了一种更简单的替代方案 GRPO，该算法最初在他们的 DeepSeekMath 论文（https://arxiv.org/abs/2402.03300）中引入。GRPO 比 PPO 更节省资源，因为它不需要一个单独的价值模型来估计价值函数。相反，GRPO 从一组采样回答的相对比较中推导其学习信号，这大大减少了计算开销。
>
> 感兴趣的读者可以在我的文章《LLM 推理强化学习的现状》（https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training）中找到 PPO 和 GRPO 的更详细的并排比较。在本章中，我们聚焦于使用 GRPO 实现 RLVR，这与 DeepSeek-R1 和其他流行推理模型使用的类似。
>
> 在下一章中，我们将在此基础上介绍几个实用的 GRPO 扩展，以进一步提升训练稳定性和最终的推理性能。

### 6.2.1 通过厨师类比理解 GRPO 的高层次直觉

RLVR 的 GRPO 算法的技术细节一开始可能有点让人不知所措，因为它有许多组成部分和大量领域特定的强化学习术语。因此，在我们开始具体的技术概述和实现之前，用一个例子来介绍 GRPO 背后的理念和机制可能是有益的。

图 6.8 使用一个厨师准备餐食的类比来介绍 GRPO 的技术术语和方法。

---

**图 6.8** 使用厨师类比概述 RLVR 的 GRPO 算法。生成多个 rollout 并打分，计算相对优势，并使用带有基于 KL 的正则化（损失）项的策略梯度目标来更新模型参数。

因此，沿着图 6.8 中的流程图走一遍，想象我们是一位经营小型外卖服务的厨师：

1. 每天，我们收到一个客户请求（提示），要求我们准备某道菜（例如，千层面）。
2. 对于该请求，我们烹饪我们著名的千层面，但同时尝试几种不同的食谱变体（rollout）。
3. 准备好食谱后，我们制作多道菜品（completions）。请注意，在 LLM 设置中，rollout 指的是针对一个提示的完整生成过程，而 completion 是生成的输出文本。在实践中，这两个术语经常互换使用。
4. 客户只在品尝完整道菜后才给我们反馈（奖励），而不是在准备过程中。这意味着我们评判的是最终结果，而不是单个烹饪步骤。
5. 我们根据第 4 阶段的客户反馈比较菜品之间的相对表现。这有助于我们识别哪些食谱变体在该特定请求中获得了更高的奖励，哪些获得了更低的奖励。
6. 对于每道完成的菜品，我们还记录它对我们当前烹饪风格的典型程度，即它在多大程度上遵循了我们通常的技术和食材选择（logprobs）。
7. 使用第 5 阶段的相对反馈和第 6 阶段每道菜品的典型程度，我们确定如何调整烹饪风格（策略梯度损失）。在这里，我们想要强化那些带来更好菜品的选择，并减少那些带来更差菜品的选择。
8. 同时，我们参考我们过去的原始食谱。
9. 然后我们应用一个风格保持惩罚，鼓励尝试新的食谱变体，但防止对我们的烹饪风格进行过于剧烈的改变，以免吓跑现有客户。
10. 基于偏好的调整和烹饪风格保持惩罚被组合成一个单一的整体度量或决策（总体损失），关于烹饪风格应该如何改变。这里的目标是平衡客户满意度和一致性。
11. 最后，我们更新烹饪风格（基于梯度的模型权重更新），以便未来的菜品更有可能满足客户偏好，同时仍然接近原始食谱。

尽管这是一个相对直观的类比，我们可以看到 GRPO 中有许多组成部分和环节。

### 6.2.2 高层次 GRPO 流程

遵循上一节的厨师类比，图 6.9 中的流程图提供了 RLVR 的 GRPO 算法的技术概述，使用我们将在逐步实现 GRPO 时计算的具体值和数字。

---

**图 6.9** RLVR 的逐步 GRPO 更新。(1) 采样一个提示并生成多个 rollout。(2) 使用可验证奖励为每个 rollout 打分。(3) 从这些奖励中计算组相对优势。(4) 计算当前模型下每个 rollout 的对数概率。(5) 将优势和对数概率组合形成策略梯度损失。(6) 添加一个针对参考模型的 KL 正则化项，并使用得到的总损失更新模型参数。

图 6.9 中的 GRPO 大纲包含许多技术组件，乍一看可能让人不知所措。我们将从实现一个简化的 GRPO 版本开始，省略图右侧所示的 KL 损失项。

在第 7 章中，我们将为了完整性添加这个缺失的 KL 损失项。虽然 KL 损失是原始 GRPO 公式的一部分，但它并非严格必要。事实上，许多 LLM 开发者在实践中省略了它，因为这样做既简化了实现，有时还能提升建模性能。

> **注** 有经验的 GRPO 或其他策略梯度方法的读者可能想知道裁剪策略比率的使用；这些将在下一章中介绍。

如果图 6.9 所示的概述在第一次阅读时感觉复杂，不用担心。我们将逐步构建这个算法。目前，最好将图 6.9 视为一个高层次路线图，在我们处理各个部分时帮助我们定位。

## 6.3 加载预训练模型

与前几章类似，我们首先加载预训练模型（见图 6.10 中的阶段 5），然后我们将使用它来生成 GRPO 所需的 rollout（对给定提示的答案）。

---

**图 6.10** 在阶段 5 中，我们加载预训练模型（本节）和数据集（下一节），用于模型训练。

清单 6.1 是我们之前用于加载分词器和基础模型的相同代码。

```python
# 清单 6.1 加载分词器和基础模型
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer
)

device = get_device()
device = torch.device("cpu")  # 删除这一行以在 GPU 上运行代码（如果你的机器支持）
model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)
```

请注意，清单 6.1 中的代码默认在 CPU 上运行，以确保你获得的结果与本章所示的结果更加一致。一旦你完成了本章的完整阅读，我建议删除 `device = torch.device("cpu")` 这一行，并在 GPU 上运行本章代码。

接下来，为了确保模型加载正确，让我们在简单的数学提示上结合第 4 章的温度和 top-p 采样器代码一起使用它：

```python
# 清单 6.2 使用温度缩放和 top-p 采样生成文本
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache
)

raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)
torch.manual_seed(0)
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9
)
print(response)
```

模型打印出 " \boxed{58}"，这是一个错误的答案（正确答案是 83），但这没关系，因为这里的目标仅仅是确保代码运行没有问题。

## 6.4 加载 MATH 训练子集

接下来，我们加载将用于使用 GRPO 训练模型的数据集。

我们使用从原始 MATH 数据集派生的非重叠训练子集，这意味着所有 MATH-500 评估问题都被明确排除在训练数据之外。这防止了数据泄漏，并确保了训练和评估之间的清晰分离，如图 6.11 所示。

> **注** 有关该数据集构建方式的详细信息，请参见 https://github.com/rasbt/math_full_minus_math500。

---

**图 6.11** MATH 数据集的结构和划分。完整数据集包含约 12,500 个问题，分为一个 500 问题的测试集（MATH-500），我们在第 3 章中用于模型评估。一个非重叠的 12,000 个问题集在本章中用于训练。

为了加载图 6.11 中所示的 MATH 训练集中的 12,000 个数学问题，我们在清单 6.3 中定义了以下 `load_math_train` 函数，它类似于第 3 章中的 `load_math500_test` 函数，只是我们指定了不同的文件路径。

```python
# 清单 6.3 加载 MATH 训练集
import json
import requests
from pathlib import Path

def load_math_train(local_path="math_train.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "math_full_minus_math500/refs/heads/main/"
        "math_full_minus_math500.json"
    )
    backup_url = (  # 如果 GitHub 托管的文件不可用，使用备用 URL
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        "MATH/math_full_minus_math500.json"
    )
    if local_path.exists():   # 如果已下载，从本地文件加载
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            print("Using backup URL.")
            r = requests.get(backup_url, timeout=30)
            r.raise_for_status()
        data = r.json()
        if save_copy:  # 可选地为未来运行缓存本地副本
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    return data
```

上述代码应该打印 "Dataset size: 12000" 以表明数据集已正确加载。

接下来，我们还可以打印 12,000 个条目中的一个，以了解数据集结构。为此，我们使用标准库中的 `pprint` 函数，因为它相比普通的 `print` 函数为字典条目提供了更好的格式：

```python
from pprint import pprint
pprint(math_train[4])
```

结果条目（索引 4 对应数据集中的第五个条目）如下所示：

```python
{
    "answer": "6",
    "level": "Level 3",
    "problem": (
        "Sam is hired for a 20-day period. On days that he "
        "works, he earns $\\$60. For each day that he does "
        "not work, $\\$30 is subtracted from his earnings. "
        "At the end of the 20-day period, he received "
        "$\\$660. How many days did he not work?"
    ),
    "solution": (
        "Call $x$ the number of days Sam works and $y$ the "
        "number of days he does not... Thus, Sam did not "
        "work for $\\boxed{6}$ days."
    ),
    "type": "Algebra",
    "unique_id": 4,
}
```

正如我们所见，每个数据集示例都存储为一个包含多个字段的字典。对于训练，相关字段是 `"problem"`，它用作提示，以及 `"answer"`，这是我们使用数学验证器验证模型输出所针对的目标。

数据集还在 `"solution"` 字段中包含完整的分步解答。虽然这样的分步解答原则上可以用于评估中间推理步骤，但这样做会不必要地约束模型，并可能导致对特定解答和风格的过拟合。相反，我们希望允许模型更自由地探索解空间。

## 6.5 采样 Rollout

在设置好预训练基础模型并加载数据集后，我们现在准备实现 GRPO 阶段，如图 6.12 所示。

---

**图 6.12** 在概述了 RLVR 方法和 GRPO 算法之后，以下各节实现了我们需要通过 MATH 数据集上的可验证奖励训练 LLM 的各个 GRPO 阶段。

在我们开始之前，图 6.13 回顾了之前介绍的 GRPO 概述，但采用了简化形式，省略了 KL 损失项（Kullback-Leibler 散度损失的缩写），我们将在下一章中添加。我们可以将其视为一个惩罚项，阻止更新后的模型偏离参考模型太远。目前，图 6.13 作为一个简化的路线图，在我们逐步讲解算法的组件时供我们参考。

---

**图 6.13** RLVR 的逐步 GRPO 更新（不含 KL 损失项）。我们首先提示 LLM 生成不同的 rollout。

如图 6.13 所示，我们首先使用 LLM 生成多个 rollout。这里，"rollout" 是一个强化学习术语，简单地指模型为给定提示生成的完整答案。

我们可以重用清单 6.2 中的 `generate_text_stream_concat_flex` 函数来生成 rollout。然而，当我们之前定义该函数时，我们使用了 `@inference_mode` 装饰器，它禁用了几个 PyTorch 功能以提高效率。由于我们稍后将执行反向传播，这使得该函数与我们的训练设置不兼容。（这里，"反向传播" 指的是 PyTorch 从损失计算梯度以便优化器可以更新模型权重的步骤。）相反，我们需要使用 `@torch.no_grad` 装饰器，它禁用前向传播的梯度跟踪，而不会将 PyTorch 切换到仅推理模式。

让我们以更紧凑的形式重写该函数，并将其命名为 `sample_response`。重要的是，该函数生成的文本与 `generate_text_stream_concat_flex(..., generate_func=generate_text_top_p_stream_cache)` 完全相同。我们可以通过观察在相同的 `random_seed`、`temperature` 和 `top_p` 设置下，两个函数产生相同的生成响应来验证这一点。

```python
# 清单 6.4 定义 rollout 生成函数
from reasoning_from_scratch.qwen3 import KVCache
from reasoning_from_scratch.ch04 import top_p_filter

@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
    )
    cache = KVCache(n_layers=model.cfg["n_layers"])  # 缓存过去的键和值以实现高效生成，如第 2 章所述
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]
    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:  # 应用第 4 章的温度缩放
            logits = logits / temperature
        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)  # 应用第 4 章的 top-p 过滤
        next_token = torch.multinomial(
            probas.cpu(), num_samples=1
        ).to(device)
        token_id = next_token.item()
        generated.append(token_id)
        if (
            tokenizer.eos_token_id is not None
            and token_id == tokenizer.eos_token_id
        ):
            break
        logits = model(next_token, cache=cache)[:, -1]
    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)
```

请注意，该函数现在还返回提示加答案 token 的 token ID 以及 token 数量（`input_ids.numel()`），连同答案文本（`tokenizer.decode(generated)`）。返回这些允许我们稍后简化几个下游函数。

除此之外，这里没有什么新内容。代码只是我们先前开发的更精简版本；它直接将第 2 章的 `generate_text_basic_stream_cache` 函数与第 4 章的温度和 top-p 采样结合起来。

接下来，我们在类似于清单 6.2 的示例提示上调用该函数：

```python
# 清单 6.5 使用温度缩放和 top-p 采样生成 rollout

torch.manual_seed(0)
raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)
token_ids, prompt_len, answer_text = sample_response(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    device=device,
    max_new_tokens=512,
    temperature=0.9,
    top_p=0.9,
)
print(answer_text)
```

之前，`generate_text_stream_concat_flex` 采样函数返回 " \boxed{58}"。现在，`sample_response` 函数显式地在响应中包含序列结束 token：" \boxed{58}<|endoftext|>"。包含这个 <|endoftext|> token 在理论上不是必需的，但它可以防止模型在非常长的训练运行中忘记生成序列结束 token。

接下来，我们可以多次调用 `sample_response` 来生成不同的 rollout，稍后实现完整训练循环时我们确实会这样做。目前，为了保持 GRPO 讲解示例的简单性，并更容易遵循图 6.13 的逐步 GRPO 大纲，我们假设模型产生了以下四个简短的响应：

```python
rollouts = [
    r"\boxed{83}",
    r"The correct answer is \boxed{83}",
    r"The final answer is 83",
    r"We get \boxed{38}",
]
```

## 6.6 计算奖励

GRPO 流程的第二阶段涉及为每个 rollout 计算奖励，如图 6.14 所示。

---

**图 6.14** GRPO 流水线中的第二阶段计算上一节 LLM 生成的每个 rollout 的奖励。

奖励使用第 3 章的数学验证器计算，主要基于答案的正确性。通过在清单 6.6 的 `reward_rlvr` 函数中设置 `fallback=None`，我们还包含了一个隐式的格式约束。例如，一个答案只有在正确且以所需的 `\boxed{}` 格式表达时才会获得 1.0 的奖励。

```python
# 清单 6.6 实现奖励函数
from reasoning_from_scratch.ch03 import (
    extract_final_candidate, grade_answer
)

def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # fallback=None 要求使用 \boxed{} 格式才能返回提取的答案
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)
```

由于模型只有在正确回答并以 `\boxed{}` 格式写出最终答案时才能获得非零奖励，我们鼓励它学习产生正确且格式规范的答案。

让我们试用一下 `reward_rlvr` 函数并将其应用于这些 rollout。

```python
# 清单 6.7 将奖励函数应用于所有 rollout
rollout_rewards = []
for answer in rollouts:
    reward = reward_rlvr(answer_text=answer, ground_truth="83")
    print(f"Answer: {answer!r}")
    print(f"Reward: {reward}\n")
    rollout_rewards.append(reward)
```

结果输出如下：

```
Answer: '\\boxed{83}'
Reward: 1.0

Answer: 'The correct answer is \\boxed{83}'
Reward: 1.0

Answer: 'The final answer is 83'
Reward: 0.0

Answer: 'We get \\boxed{38}'
Reward: 0.0
```

如上所示，奖励函数按预期工作，只有在答案包含正确结果（83）并使用 `\boxed{}` 格式时才提供 1.0 的奖励。

请注意，DeepSeek-R1 团队也尝试使用过程奖励模型在训练期间为中间解答步骤打分。这些尝试没有成功，研究人员得出结论，最好只在最终答案正确性奖励上训练，而不使用中间奖励。

> **练习 6.1：添加格式感知的奖励塑形**
>
> 扩展本章的 `reward_rlvr` 函数，使其基于输出格式分配部分分数。具体来说，修改奖励函数，使其在模型以所需的 `\boxed{}` 格式产生正确答案时返回 1.0，答案正确但未加框时返回 0.5，其他情况返回 0.0。

## 6.7 通过优势值从 Rollout 中准备学习信号

我们现在从奖励转向所谓的优势值（advantages），如图 6.15 所示。虽然奖励告诉我们每个单独的 rollout 表现如何，但优势值捕捉了一个 rollout 相对于为同一提示生成的其他 rollout 的表现如何。

---

**图 6.15** GRPO 的第三阶段从答案（rollout）正确性奖励中计算优势值。

图 6.15 所示的优势值通过一个简单的公式计算：

$$
\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}
$$

这里：
- $r_i$ 表示第 $i$ 个 rollout 的奖励，
- $\mu_r$ 是该组 rollout 的平均奖励，
- $\sigma_r$ 是相应的标准差，
- $\epsilon$ 是一个添加的小常数，用于数值稳定性以避免除零错误。

在代码中，我们可以如下实现优势计算：

```python
# 清单 6.8 计算优势值
rewards = torch.tensor(rollout_rewards, device=device)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
print(rewards)
print(advantages)
```

我们在上一节计算的奖励是 `[1., 1., 0., 0.]`，相应的优势值是 `[0.8659, 0.8659, -0.8659, -0.8659]`。

这些优势值捕捉了哪些回答在组内表现优于平均水平，哪些表现差于平均水平。

> **注** GRPO 中的 "GR"（组相对）指的是 GRPO 为每个提示生成多个答案（rollout），并将它们相互比较以构建学习信号。

你可能仍然想知道将奖励转换为优势值的目的是什么。在实际层面，优势值在后续的策略更新中直接缩放梯度。

如果优势值为正，梯度会增加产生该 rollout 的动作的可能性。如果为负，梯度会降低它们的可能性。优势值接近零的 rollout 贡献很小。

> **练习 6.2：零优势值情况**
>
> 修改 rollout 奖励，使所有 rollout 获得相同的奖励（例如，强制所有奖励为 0.0 或全部为 1.0）。运行优势计算并验证得到的 GRPO 损失为零。为什么在组相对策略优化中这种行为是可取的？

## 6.8 用序列对数概率为 Rollout 打分

我们已从奖励中计算了奖励和优势值。现在，我们从 rollout 中走一条独立的分支，计算它们的对数概率（logprobs），如图 6.16 所示。Logprobs 衡量模型在其当前参数下认为每个生成的 token 的可能性有多大，如上一章所讨论的。

---

**图 6.16** GRPO 流水线的第四阶段计算每个 rollout 的对数概率，这与我们在上一章开发的对数概率打分器相关。

如图 6.16 所示，这些 logprobs 与优势值一起构成了我们将在稍后实现的 GRPO 策略梯度损失的核心要素。

回到 logprobs，在上一章中我们实现了一个 `avg_logprob_answer` 函数，计算模型答案 token 的逐 token logprobs。这些值在文献中通常被称为 token-level logprobs，通过评估模型在其当前参数下为每个生成的 token 分配的可能性来获得。

当在响应中取平均时，这些值通常被称为 token-level logprobs，并常用于为 LLM 输出打分。在这种上下文中更倾向于取平均，因为它提供了长度归一化，使得不同长度的响应之间的分数具有可比性。

> **Token-Level 对数概率的数学定义**
>
> 从数学上讲，我们在上一章计算的 token-level logprobs 可以写成：
>
> $$
> \frac{1}{T} \sum_{t=1}^{T} \log P(y_t | y_{<t}, x; W)
> $$
>
> 这里：
> - $y_1, y_2, ..., y_T$ 表示生成长度为 $T$ 的响应中的 token，
> - $y_{<t}$ 表示所有先前生成的 token，
> - $x$ 是输入提示，
> - $W$ 表示模型的权重参数。
>
> 这个表达式在数学上与上一章使用的表达式相同；我们只是从 $x$ 切换到 $y$ 以清楚地区分生成的输出 token 和输入提示。

作为参考，上一章的 `avg_logprob_answer` 函数复制如下，我们用它来为先前的提示和 `answer_text` 计算 logprobs 以作说明：

```python
# 清单 6.9 计算 token-level 对数概率（类似于第 5 章）
@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)
    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1
    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]
    return torch.mean(next_token_logps).item()

avg_logprob_val = avg_logprob_answer(
    model, tokenizer, 
    prompt=prompt,
    answer=answer_text,
    device=device) 
print(avg_logprob_val)
```

这返回 `-0.061279296875`。

在 GRPO 中，我们不使用平均的 token-level logprobs。相反，我们使用序列级别的 logprobs（sequence-level logprobs）。

原因是 GRPO 为每个 rollout 分配单个奖励和优势值，这适用于整个生成的响应。直观地说，logprob 应该反映模型生成整个序列的可能性，而不是每个 token 的平均值。

我们通过将所有生成 token 的 logprobs 相加来计算序列级别的 logprobs。（这个和对应于在当前模型参数下生成完整响应的对数似然。）

相比之下，平均 token-level logprobs 会按序列长度进行归一化。虽然这种归一化对于为不同长度的响应打分和比较很有用，但它会无意中在策略优化中重新缩放学习信号，并导致较长和较短的响应对更新的贡献不均匀。（虽然这不是原始 GRPO 公式的意图，但我们将在下一章改进算法时重新讨论 token-level logprobs。）

我们可以通过去掉平均步骤并将清单 6.9 代码中的 `torch.mean(next_token_logps)` 替换为 `torch.sum(next_token_logps)`，将 token-level、长度归一化的分数转换为序列级别的 logprobs。

只要 token 数量完全匹配，相同的结果也可以通过将平均 logprob 乘以答案 token 的数量来获得：

```python
sequence_logprob_val = avg_logprob_val * (
    len(tokenizer.encode(answer_text))
)
print(sequence_logprob_val)
```

这现在返回 `-16.239013671875`。

这些序列级别的 logprobs 与序列长度 $T$ 成线性比例，这意味着较长的响应通常会收到更负的 logprob 值。因此，对于两个同样好的答案，优化会隐式地偏爱较短的那个，因为在似然方面它更便宜。因此，求和的 logprobs 鼓励模型更早停止，除非产生更长的响应被更高的奖励所证明。

因此，如前所述，我们可以将函数中的 `torch.mean` 替换为 `torch.sum` 以获得序列级别的 logprobs。由于该函数在上一章中使用 `@torch.inference_mode()` 装饰器在推理模式下运行，我们无论如何都需要重新定义它（不带装饰器），以便 PyTorch 可以跟踪梯度。

此外，由于 6.5 节的 `sample_response` 已经返回了 `token_ids` 和 `prompt_len`，我们可以通过去掉显式的分词步骤和 `full_ids` 的构造来简化实现。

```python
# 清单 6.10 计算序列级别的对数概率
# 我们想要预测下一个 token 概率的位置
def sequence_logprob_draft(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    start = prompt_len - 1         # 我们想要预测下一个 token 概率的位置
    end = token_ids.shape[0] - 1   # 我们想要预测下一个 token 概率的位置
    t_idx = torch.arange(start, end, device=token_ids.device)
    next_tokens = token_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]
    return torch.sum(next_token_logps)  # 对答案 token 的 logprobs 求和

print(sequence_logprob_draft(model, token_ids, prompt_len))
```

输出：

```
tensor(-16.2998, grad_fn=<SumBackward0>)
```

首先，结果与我们之前得到的 `-16.2421875` 相似（差异是由于浮点行为和舍入造成的）。

PyTorch 在内部构建一个计算图，记录应用于张量的每个可微操作。`SumBackward0` 条目表明 token 对数概率的求和是该图的一部分，这意味着梯度可以通过序列级别的对数概率反向传播到模型参数。

这正是我们稍后实现训练循环以通过反向传播更新模型权重时进行策略优化所需要的。反向传播应用反向传播算法，这是深度学习中用于计算梯度和更新神经网络权重的标准方法。

> **PyTorch 计算图、梯度和反向传播**
>
> 当模型产生输出时，PyTorch 记录从模型参数到该输出的每个数学操作。这个记录被称为计算图。当我们稍后运行反向传播时，PyTorch 使用这个图来执行反向传播，即计算描述模型参数的变化将如何影响最终结果的梯度。如果一个操作是该图的一部分，PyTorch 可以在训练期间通过它计算梯度。
>
> 理解 PyTorch 的计算图不是本章必需的，但感兴趣的读者可以在我的教程中阅读更多相关内容：https://sebastianraschka.com/teaching/pytorch-1h/#3-seeing-models-as-computation-graphs。

虽然这个草稿实现是一个完全可用的代码实现，但我们可以稍微简化它并使其对 GPU 更高效。特别是，我们可以避免显式构造索引范围，而是使用 `torch.gather` 直接选择对应于生成 token 的 logprob。清单 6.11 展示了相同计算的更紧凑、优化版本，它产生相同的结果，同时更易于阅读和更快执行。

```python
# 清单 6.11 优化的序列级别对数概率代码
def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(
        1, token_ids[1:].unsqueeze(-1)
    ).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])

print(sequence_logprob(model, token_ids, prompt_len))
```

与之前的代码类似，这返回 `tensor(-16.2998, grad_fn=<SumBackward0>)`。

> **提示** 我们可以直接在 `sample_response` 函数中计算这些 logprobs 以提高效率，避免两次调用 `model(...)`（分别在 `sample_response` 和 `sequence_logprob` 函数中）。

最后，在有了一个稳健且高效的序列级别 logprob 函数之后，我们可以计算四个不同 rollout 各自的 logprobs。

```python
# 清单 6.12 计算所有 rollout 的序列级别对数概率
rollouts = [
    r"\boxed{83}",
    r"The correct answer is \boxed{83}",
    r"The final answer is 83",
    r"We get \boxed{38}",
]

rollout_logps = []
for text in rollouts:
    token_ids = tokenizer.encode(prompt + " " + text)
    logprob = sequence_logprob(
        model=model,
        token_ids=torch.tensor(token_ids, device=device),
        prompt_len=prompt_len,
    )
    print(f"Answer:  {text}")
    print(f"Logprob: {logprob.item():.4f}\n")
    rollout_logps.append(logprob)
```

打印如下：

```
Answer:  \boxed{83}
Logprob: -7.9243

Answer:  The correct answer is \boxed{83}
Logprob: -20.1546

Answer:  The final answer is 83
Logprob: -16.6130

Answer:  We get \boxed{38}
Logprob: -23.3677
```

正如我们所见，较短且更简洁的答案比较长或更冗长的响应获得更高（较不负）的序列级别 logprobs。例外是最后一个答案，它获得了最低分。请注意，这是唯一一个包含错误数值答案（38 而不是 83）的答案，所以这也很符合直觉。

总的来说，这说明了求和的 logprobs 如何自然地偏爱简洁的输出，这与它们在 GRPO 中在序列级别应用奖励和优势时的作用是一致的。

## 6.9 从优势值到策略更新：通过 GRPO 损失

我们现在实现 GRPO 流水线的第五阶段，将先前计算的优势值和 logprobs 组合成策略梯度损失。我们稍后将实现随后的权重更新（第六阶段），作为训练循环的一部分，如图 6.17 所示。

---

**图 6.17** GRPO 流水线的第五阶段计算我们用于更新模型的策略梯度损失。第六阶段，模型权重更新，将稍后作为训练循环的一部分实现。

首先，我们将上一节的 rollout logprobs 列表转换为 PyTorch 张量。然后，我们通过将每个 rollout 的序列级别 logprob 乘以其对应的优势值，在 rollout 上取平均，并应用一个负号来计算策略梯度损失。关于负号，因为 PyTorch 优化器被定义为最小化损失，所以自然写为最大化问题的目标必须翻转符号。

```python
# 清单 6.13 计算策略梯度损失
logps = torch.stack(rollout_logps)
pg_loss = -(advantages.detach() * logps).mean()
print(logps)
print(pg_loss)
```

对于 logprobs，打印：

```
tensor([
    -7.9243, -20.1546, -16.6130, -23.3677],
    grad_fn=<StackBackward0>)
```

结果策略梯度损失值为：

```
tensor(-2.5764, grad_fn=<NegBackward0>)
```

请注意，我们在优势值上使用 `.detach()`，因为它们在策略更新期间被视为固定的学习信号。这防止了梯度通过优势计算反向传播。这确保了只有影响 logprobs 的模型参数被更新。

例如，在策略梯度方法中，我们想要最大化 rollout 的优势加权 logprob，因为较高优势 rollout 的较高对数概率会改善策略。

因此，通过将该目标乘以 -1，我们将最大化问题转换为等效的最小化问题。请注意，最小化负目标产生的参数更新与最大化原始目标完全相同，但这样，它与 PyTorch 的优化器实现保持兼容。

> **最大化与最小化**
>
> 为了用一个具体例子进一步说明符号翻转，假设我们想要最大化的目标是一个简单的标量值 $f(x) = 3$。
>
> 在最大化视角下，较大的值更好，所以 3 比 2 更可取。
>
> PyTorch 优化器最小化，所以它们会尝试减小这个值，这与我们想要的相反。
>
> 现在，假设我们翻转符号并将损失定义为 $L(x) = -f(x) = -3$。
>
> 最小化 $L(x)$ 现在等价于最大化 $f(x)$。例如，如果 $L(x)$ 从 -2 减小到 -3，$f(x)$ 从 2 增加到 3。

> **策略梯度损失的数学定义**
>
> 对于觉得通过数学符号更容易理解计算的读者，GRPO 中使用的策略梯度损失可以写成：
>
> $$
> \mathcal{L}_{PG} = -\frac{1}{N} \sum_{i=1}^{N} \hat{A}_i \sum_{t=1}^{T_i} \log P(y_t^{(i)} | y_{<t}^{(i)}, x^{(i)}; W)
> $$
>
> 这里：
> - $N$ 表示 rollout 的数量，
> - $y_1^{(i)}, ..., y_{T_i}^{(i)}$ 是第 $i$ 个生成长度为 $T_i$ 的响应中的 token，
> - $y_{<t}^{(i)}$ 表示该响应中所有先前生成的 token，
> - $x^{(i)}$ 是对应的输入提示，
> - $\hat{A}_i$ 是完整 rollout 的优势值。
>
> 内部求和计算一个 rollout 的序列级别对数概率，而外部平均计算跨 rollout 的优势加权对数概率。

## 6.10 将所有内容整合到单个 GRPO 函数中

我们现在已经完成了本章最具挑战性的部分，并单独走过了每个 GRPO 阶段。

接下来，我们将图 6.18 所示的五个 GRPO 阶段组合成一个单一的 `compute_grpo_loss` 函数，稍后我们将将其用作完整训练循环的一部分。（请注意，这里的术语"阶段"指的是基于我们如何逐步讲解 GRPO 算法的 GRPO 损失计算的概念组件，而不是外层优化循环中的训练步骤。）

---

**图 6.18** 完整的 GRPO 工作流程，其中 (1) 为提示生成多个 rollout，(2) 分配正确性奖励，(3) 转换为组相对优势值，以及 (4) 与对数概率组合以 (5) 计算策略梯度损失。损失梯度 (6) 将在下一节计算并用于更新模型。

下面清单 6.14 中展示了 `compute_grpo_loss` 函数，它组合了图 6.18 中我们之前讨论的所有阶段。

```python
# 清单 6.14 组合所有 GRPO 阶段
def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])
    was_training = model.training
    model.eval()
    for _ in range(num_rollouts):
        # 阶段 1：生成 rollout
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # 阶段 2.1：计算奖励
        reward = reward_rlvr(text, example["answer"])
        # 阶段 4.1：计算 logprobs
        logp = sequence_logprob(model, token_ids, prompt_len)
        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )
    if was_training:
        model.train()
    # 阶段 2.2：收集所有奖励
    rewards = torch.tensor(roll_rewards, device=device)
    # 阶段 3：计算优势值
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    # 阶段 4.2：收集所有 logprobs
    logps = torch.stack(roll_logps)
    # 阶段 5：计算策略梯度损失
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss  # 在下一章中我们在这里添加一个 KL 项
    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }
```

`compute_grpo_loss` 函数相对不言自明，因为它遵循了我们在前几节中手动实现的精确阶段。不过请注意，在阶段 1 之后，代码直接进行到阶段 2 和 4，而不是阶段 3 和 4。这纯粹是一个实现选择，通过避免多个嵌套 for 循环来简化代码结构，同时保留相同的操作逻辑顺序。

还要注意，`pg_loss`（策略梯度损失）在我们的示例中与总体损失相同。这是因为我们在这里有意省略了 KL 损失项，并将在第 7 章中为完整性添加它。如前所述，据报道当省略 KL 损失项时，GRPO 在数学问题上表现更好，这就是为什么我们在这个阶段采用这个简化目标。

接下来，让我们在 MATH 训练数据集的一个示例上尝试新的 `compute_grpo_loss` 函数，以确保它工作正常。在这里，我们仅采样少量 rollout（`num_rollouts=2`）并生成相对较少的 token（`max_new_tokens=256`）用于测试目的。你可能仍然需要几秒钟才能看到函数返回结果。

```python
# 清单 6.15 在 MATH 训练示例上计算 GRPO 阶段

torch.manual_seed(123)
stats = compute_grpo_loss(
    model=model,
    tokenizer=tokenizer,
    example=math_train[4],
    device=device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9
)
pprint(stats)
```

输出如下：

```python
{'advantages': [0.0, 0.0],
 'loss': -0.0,
 'loss_tensor': tensor(-0., grad_fn=<NegBackward0>),
 'pg_loss': -0.0,
 'rewards': [0.0, 0.0],
 'samples': [{'gen_len': 4, 'reward': 0.0, 'text': ' 14<|endoftext|>'}, 
             {'gen_len': 256,
              'reward': 0.0,
              'text': ' 4\n'
                      'To solve the problem, let\'s break it down step by step'
                      '...'}
```

正如我们从 `samples` 字段中的 `rewards` 和 `text` 条目所看到的，模型回答不正确。如 6.4 节 earlier 所讨论的，正确答案必须包含 `\boxed{6}`。由于这个条件没有满足，奖励为零，这反过来产生了零优势值和零损失。在这种情况下，如果我们正在训练模型，梯度将为零，模型参数将不会更新。

## 6.11 实现 GRPO 训练循环

我们现在已经具备了实现完整 GRPO 训练循环的所有必要组件。在本章的最后一节中，我们将所有内容整合在一起，使用 GRPO 算法通过基于可验证奖励的强化学习来训练模型，如图 6.19 所示。

---

**图 6.19** 在实现了各个 GRPO 阶段之后，我们现在实现周围的训练循环来更新模型权重。

训练循环由八个主要阶段组成，如图 6.20 所示。这些阶段中的大多数是用于深度神经网络的典型 PyTorch 训练循环的标准组件，包括 LLM。推理模型特有的唯一阶段是阶段 4，在那里我们计算 GRPO 损失，而不是许多其他类型的深度神经网络和 LLM 预训练中使用的标准分类损失。

---

**图 6.20** 训练循环大纲。整体结构遵循标准的深度学习训练循环。关键区别在于损失的计算方式：不是标准的监督目标，而是通过 GRPO 阶段获得损失（阶段 4）。

在逐一讨论这八个阶段之前，首先看看它们在代码中是如何实现的是有帮助的。清单 6.16 展示了对应于图 6.20 所示八个阶段的完整训练循环。

```python
# 清单 6.16 实现 RLVR 训练循环
import time

def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,
):
    if steps is None:
        steps = len(math_data)
    # 阶段 1：初始化优化器（模型已在函数外部初始化）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0
    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)
    try:
        # 阶段 2：遍历训练步骤
        for step in range(steps):
            # 阶段 3：重置损失梯度（最佳实践是在每个步骤开始时执行此操作）
            optimizer.zero_grad()
            current_step = step + 1
            example = math_data[step % len(math_data)]
            # 阶段 4：计算 GRPO 损失
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            # 阶段 5：反向传播计算损失梯度
            stats["loss_tensor"].backward()
            # 阶段 6：裁剪大梯度以提高训练稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 阶段 7：使用损失梯度更新模型权重
            optimizer.step()
            # 阶段 8：收集奖励、平均响应长度和损失
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(
                sample["gen_len"] for sample in stats["samples"]
            )
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path, current_step, steps,
                stats["loss"], reward_avg, avg_response_len,
            )
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )
            # 阶段 8：保存模型检查点
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")
    # 如果我们提前中断训练，保存模型检查点
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model
    return model

def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path

# 将结果保存到 CSV 文件的实用函数
def append_csv_metrics(
    csv_log_path, step_idx, total_steps,
    loss, reward_avg, avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )
```

该函数使用 GRPO 实现了完整的 RLVR 训练循环。如前所述，除了阶段 4（GRPO 损失计算）之外，该结构镜像了一个传统的 PyTorch 训练循环，在其中我们重置梯度、反向传播损失、可选地裁剪梯度、通过优化器步骤更新参数、记录指标，并定期保存检查点。

> **提示** 有关在 PyTorch 中训练神经网络的一般介绍，请参见我的文章《一小时学会 PyTorch：从张量到在多个 GPU 上训练神经网络》的第 3-8 节（https://sebastianraschka.com/teaching/pytorch-1h/）。

与其他标准训练循环相比，这里的关键区别在于损失的构建方式。不是标准的监督目标（用于大多数神经网络，例如分类器，但也用于 LLM 预训练），训练信号是通过 RLVR 过程中的 GRPO 算法计算的。

请注意，除了定期打印结果以跟踪进度的日志记录步骤之外，我们还定期保存模型检查点（模型权重的快照），以便我们稍后加载、评估和使用它，如下一节所讨论的。

让我们现在运行代码并训练模型。之前，我们将 PyTorch 设备硬编码为 `"cpu"`，以便所有中间结果与书中显示的结果更紧密地匹配，因为 `"mps"` 和 `"cuda"` 设备可能引入小的浮点差异。然而，在 CPU 上训练非常慢。

为了方便，下面的清单已经包含了两行代码，在调用 `train_rlvr_grpo()` 函数之前自动选择和激活适当的设备。例如，只需运行以下清单中的代码，它就会自动切换到 `"cuda"` 或 `"mps"`。

```python
# 清单 6.17 训练模型
device = get_device()
model.to(device)
torch.manual_seed(0)
train_rlvr_grpo(
    model=model,
    tokenizer=tokenizer,
    math_data=math_train,
    device=device,
    steps=50,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=5,
    checkpoint_dir=".",
)
```

在检查结果之前，让我们简要谈谈设置。`temperature` 和 `top_p` 设置选择在一个常见范围内，鼓励答案有一定多样性，但仍能为该模型产生连贯的文本。可选地，你可以更改这些设置并查看它们如何影响结果（合理和常见的范围是 0.7-0.9）。

步骤数相对较小，为 50，尽管数据集包含 12,000 个条目。这纯粹是为了在这种教育背景下实现合理的运行时间，可以增加。不过，现在它已经足以产生良好的结果。

学习率（`lr`）可以调整，但它处于一个合理的范围内并且效果很好。

Rollout 的数量（`num_rollouts=4`）和允许的 token 数（`max_new_tokens=512`）相对较小，以减少资源需求。如果你遇到内存不足错误，你可以进一步降低这些值，例如，通过设置 `num_rollouts=2` 和 `max_new_tokens=64` 用于测试目的。

在当前 `checkpoint_every=5` 设置下，模型每 5 步保存一次，一个检查点需要大约 1.5 GB 的磁盘空间。在实践中，在较长的运行中，我建议将此数字设置为 50 或 100。在这种情况下，它故意设置得较小以用于测试目的。另外请注意，如果训练运行被手动中断，例如，通过 Jupyter notebook 的 "Interrupt the kernel" 按钮中断执行，应该创建一个额外的检查点。

让我们现在简要看看运行的输出（我在第 7 步中断了 50 步的运行）：

```
Using Apple Silicon GPU (MPS)
[Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=88.0 
[Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=7.5 
[Step 3/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.5 
[Step 4/50] loss=0.0909 reward_avg=0.250 avg_resp_len=6.5 
[Step 5/50] loss=1.1001 reward_avg=0.500 avg_resp_len=300.5 
Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00005.pth
KeyboardInterrupt. Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00006-interrupt.pth
```

我们可以看到损失和奖励值都在波动，这在强化学习训练中是正常的。

请注意，我们打印的是平均奖励，它表示采样响应中正确的比例。例如，在我们的 `num_rollouts=4` 设置下，`reward_avg` 为 0.5 意味着 4 个 rollout 中有 2 个获得了正奖励。类似地，0.25 和 0.0 的值分别表示一个或零个正确响应。

损失值本身不应被过度解读。`reward_avg=0.000` 的步骤产生接近零的损失，因为所有 rollout 获得相同的奖励，导致组相对优势消失，几乎没有梯度信号。较大的损失幅度，例如在第 3 步（见上面输出的第 4 行），仅仅反映了 rollout 之间较大的相对差异，对于 GRPO 风格的目标来说这是典型的，尤其是在训练开始时。

理想情况下，我们希望随着时间的推移看到两个主要趋势：

1. 当对许多步骤取平均时，平均奖励应该增加，因为模型学习产生更准确的响应。
2. 推理准确性应该提高（我们在下一节评估这一点）。

补充材料包含一个稍微更复杂的脚本版本，还包含一个选项，可以定期在 MATH-500 测试数据集的子集上评估模型，以查看模型是否在目标任务上改进：https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_intro

> **批量训练**
>
> 强化学习可能相对资源密集，因为我们必须为每个训练步骤通过 LLM 生成多个（长的）rollout。因此，该实现不支持批处理。
>
> 如果你可以访问多个 GPU，你可以使用补充材料中的可选代码版本，支持批处理和多 GPU，可以在 https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_intro 找到，它训练模型的速度更快。

## 6.12 加载和评估保存的模型检查点

在最后一节中，如图 6.21 所示，我们讨论如何从上一节的 RLVR 训练运行中加载保存的检查点，以及如何在 MATH-500 数据集上评估模型。

---

**图 6.21** 本章的最后一步讨论我们如何加载保存的模型检查点并评估它们。

可以使用 PyTorch state dict 方法加载保存的检查点，如第 2 章所述，其中 `model_path` 指向相应的 `.pth` 检查点文件：

```python
model.load_state_dict(torch.load("qwen3-0.6B-rlvr-grpo-step00050.pth"))
```

> **下载 RLVR 检查点**
>
> 如果你由于运行时间成本而不想在本地运行 GRPO 训练循环，你可以转而下载我上传的预训练检查点。这些检查点可以手动获取，或者使用下面提供的辅助函数直接从 Python 下载，该函数下载选定的检查点并使其可用于评估或进一步实验：
>
> 请注意，此检查点是使用与清单 6.17 类似的设置创建的，只是 `num_rollouts=4` 增加到了 `num_rollouts=8`。
>
> 这些检查点与第 3 章介绍的模型评估工具完全兼容，这允许我们使用相同的 MATH-500 验证流程来评估强化学习训练的模型。
>
> 为了方便，你可以重用第 3 章 bonus 材料中提供的评估脚本（https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500.py）来在 MATH-500 数据集上运行评估，只需指定所需的检查点路径。
>
> 例如，要评估 `qwen3-0.6B-rlvr-grpo-step00050.pth` 检查点文件，你可以运行上述脚本：
>
> ```bash
> uv run evaluate_math500.py \
>     --dataset_size 500 \
>     --which_model base \
>     --checkpoint_path "qwen3-0.6B-rlvr-grpo-step00050.pth"
> ```
> （如果你不是 uv 用户，将 `uv run` 替换为 `python`。）

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints
download_qwen3_grpo_checkpoints(grpo_type="no_kl", step="00050")
```

表 6.1 比较了原始的基础模型和推理模型（第 1 行和第 2 行）与本章我们通过 GRPO 训练的基础模型（第 3 行和第 4 行）。

**表 6.1** 不同基础和推理模型的 MATH-500 任务准确率

| 方法 | 步骤 | 最大 Token 数 | Rollout 数量 | 准确率 | 平均 Token 数 |
|------|------|--------------|-------------|--------|-------------|
| 1 基础模型（第 3 章） | - | - | - | 15.2% | 78.85 |
| 2 推理模型（第 3 章） | - | - | - | 48.2% | 1369.79 |
| 3 GRPO（本章） | 50 | 256 | 4 | 43.2% | 560.22 |
| 4 GRPO（本章） | 50 | 512 | 4 | 45.6% | 579.81 |
| 5 GRPO（本章） | 50 | 512 | 8 | 47.4% | 586.11 |

表 6.1 中的准确率列指的是我们在第 3 章中使用的完整 500 样本 MATH-500 数据集的准确率。请注意，"最大 Token 数"列对应于训练期间每个 rollout 允许的 token 数量。如果数字是 512，这鼓励 LLM 在这个 512 token 限制内提供最终的框选答案，否则在训练期间它将不会获得奖励。

评估代码以最大 2048 的 token 限制执行，允许 LLM 在评估期间生成更长的响应。"平均 Token 数"列平均了在 MATH-500 数据集上评估期间的响应长度。我们可以看到，与参考推理模型（第 2 行）相比，我们的 GRPO 模型平均生成更短的响应，正如预期的那样，这是由于训练期间的 token 长度限制。

请注意，DeepSeek-R1 团队观察到，随着模型开始编写中间（思维链）解释，响应在训练过程中会变长。因此，为了最大化准确率，在训练期间不要过于激进地限制 token 长度是有道理的。更长的 token 长度，尤其是与多个 rollout 结合时，需要更多的计算资源，这就是为什么我们在本章中将它们限制在较低的上限。

如表 6.1 所示，通过 GRPO 训练推理模型产生了 47.4% 的准确率（第 5 行），这接近官方 Qwen3 0.6B 推理模型在 MATH-500 上的准确率。

模型的准确率可以通过增加响应长度、每个提示采样更多的 rollout 以及训练额外的步骤来进一步提升。下一章介绍了监控和改进 GRPO 训练结果的实用技巧和技巧。

我们只训练了模型 50 步，这已经足以解锁基础模型中的推理行为。在我的实验中，训练更长时间不一定会提高性能，甚至可能降低准确率，因为原始 GRPO 公式在较长运行中可能不稳定。

还值得记住的是，本章实现了一个简化的 GRPO 变体，没有 KL 正则化项。在下一章中，我们将回到完整的 GRPO 目标，将 KL 项重新加入，并讨论使训练在较长运行中更稳定的常见改进。

感兴趣的读者可以在 https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main/grpo_original_no_kl 找到并下载从 50 到 9000 的额外检查点。

## 6.13 总结

- 强化学习（RL）可用于在人工偏好标签和可验证奖励上训练 LLM。
- RL 通常作为后训练应用于预训练基础模型之上，并且可以插入到 LLM 流水线的不同阶段，包括推理训练和偏好调优。
- 基于人类反馈的强化学习（RLHF）通过一个两阶段设置优化人类偏好：从排名的回答中训练奖励模型，然后使用奖励分数更新 LLM。
- 基于可验证奖励的强化学习（RLVR）通过用确定性的、自动计算的验证器（例如，数学答案检查）替换学习得到的奖励模型来简化 RLHF。
- 我们聚焦于用于数学推理的 RLVR。
- 我们使用 GRPO 作为策略优化算法，将验证器奖励转化为参数更新；因为 GRPO 直接使用序列级别奖励优化模型，而无需单独的价值模型，所以特别方便。
- GRPO 是 LLM 其他 RL 算法更节省资源的替代方案，因为它避免了训练单独的价值模型，而是从一组采样 rollout 的内部比较中推导学习信号。
- "rollout" 指的是针对提示的完整模型答案（completion）；奖励、优势值和对数概率在后续步骤中从 rollout 计算。
- 奖励从验证器计算，只有当最终答案正确且以所需的 `\boxed{}` 等格式可提取时才授予奖励。
- 原始奖励通过将每个 rollout 奖励相对于组均值和标准差进行归一化，转换为优势值。
- GRPO 还依赖于序列级别的对数概率，通过将生成的答案 token 的 token 对数概率求和来计算。
- 序列对数概率与优势值一起构成了 GRPO 中的核心策略梯度目标。
- 完整的 GRPO 损失计算被组合成一个单一函数，执行 rollout 采样、奖励计算、优势计算、logprob 计算和策略梯度损失计算。
- 周围的训练循环是一个标准的深度学习循环，关键区别在于损失来自 GRPO 而不是传统的分类损失。
- 训练是资源密集的，因为每个步骤都需要生成多个可能很长的 rollout，但即使是短的 GRPO 运行也能将 MATH-500 准确率从 15% 提高到 47%。
