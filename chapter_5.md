# 5 通过自精炼实现推理时扩展

## 本章内容
- 使用简单的基于规则的评分器对 LLM 答案进行评分
- 计算 LLM 对其答案的置信度
- 编写自精炼循环，让 LLM 迭代改进其答案

上一章介绍了推理时扩展（简称推理扩展）的概念，它可以在不进一步训练模型的情况下提高模型回答的准确性。特别是，上一章的重点是自一致性，即模型生成多个答案，最终答案通过多数投票选出。

如图 5.1 所示，本章超越了推理扩展的简单多数投票，涵盖了另一种流行且有用的推理扩展技术：**自精炼**。与生成多个答案以供选择不同，自精炼专注于迭代地改进单个答案以纠正潜在错误。

**图 5.1** 本书所涵盖主题的心智模型。本章继续阶段 3，重点介绍无需额外训练即可改进推理的推理时技术。本章介绍自精炼，即模型迭代地批判和改进自己的答案。

## 5.1 评分并迭代改进模型回答

如上一章所述，推理扩展提供了一种用额外计算换取更好准确性的方法。我们还介绍了两种推理扩展技术：思维链提示和自一致性。

思维链提示，如图 5.2 所示，通过修改提示来触发基础模型写出更长的解释，例如添加短语 "Explain step by step."，这反过来可以提高答案的准确性。这种方法对于不自然提供推理式解释的基础模型特别有用。作为推理模型训练的模型通常不会从这种类型的推理扩展中受益，因为它们已经解释了它们的答案。

**图 5.2** 本书涵盖的三种改进推理的推理时方法。前两种方法在上一章中已涵盖。本章涵盖第三种方法，即自精炼，其中模型迭代地改进自己的答案。

自一致性，如图 5.2 所示的第二种方法，让模型并行产生多个答案。然后我们通过在这些候选答案上进行简单多数投票来选出最终答案。

尽管自一致性相当简单，但它通常能在答案准确性方面带来巨大提升，这就是为什么它已成为 LLM 应用中准确性优先于延迟的常见选择。最近的例子包括 DeepSeekMath-V2 和 Google 的 Gemini 3 Deep Think 模式（参见附录 A 中的参考文献）。

自一致性的一个缺点是，多数投票需要可以比较的简短答案。

在本章中，我们实现了一种更通用的技术，**自精炼**，其中 LLM 学会迭代地改进自己的答案（图 5.2 中的方法 3）。

但在实现自精炼技术之前，我们将首先实现评分函数，用于比较和排名不同的答案，如图 5.3 所示。

**图 5.3** 我们构建一个简单的基于规则的评分器，计算 token 概率和 log-probabilities，然后将这些评分作为自精炼方法的一部分，其中模型迭代地改进自己的答案。

在加载预训练 LLM 之后（如图 5.3 中的步骤 1，与前几章相同），我们将从本章开始用一个简单的基于规则的评分函数来说明评分的概念（步骤 2）。然后，我们将介绍 token 概率（步骤 3）和 token log-probabilities（步骤 4）的概念，这些是我们实现 logprob 评分方法（步骤 5）以用于自精炼循环（步骤 6）所需要的。

我们将主要使用 logprob 评分函数（图 5.3 中的步骤 5）来跟踪自精炼循环中的进度。这些评分函数也可用于打破自一致性中的平局，或选择最佳回答，而不是依赖多数投票。

图 5.3 中的本章概述看起来相对简短且直接。token 概率和 log-probability 的主题有些复杂，在下一章中也很重要，届时我们将实现强化学习方法来训练 LLM。因此，本章将花费大量篇幅解释 log-probability 评分的概念。

## 5.2 加载预训练模型

与上一章一样，我们首先加载本章使用的模型。

与前几章一样，清单 5.1 中的代码加载了本章使用的模型和 tokenizer。

请注意，上述代码默认在 CPU 上运行，以确保与本章所示结果通常更一致。虽然根据操作系统和机器的不同，CPU 上仍然可能出现较小的数值差异，但这些差异通常比在不同加速器后端上观察到的差异更小且更可预测。

后面的部分（5.4 和 5.5 节）涉及使用小数点后有很多位数的非常小的数字进行计算，这些计算对设备选择和底层实现细节更敏感。这在实践中不是问题，但这种不匹配在第一次阅读时可能会令人困惑。因此，我建议从 CPU 设备开始，稍后再考虑 MPS 或 CUDA 设备。

接下来，为了确保模型正确加载，让我们将其与上一章的温度和 top-p 采样器代码一起在 MATH-500 提示上使用：

**清单 5.1 加载 tokenizer 和基础模型**
```python
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer
)

device = get_device()
device = torch.device("cpu")  #A

model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)
```
#A 删除此行以在 GPU 上运行代码（如果您的机器支持）

模型产生以下回答：

**清单 5.2 使用温度缩放和 top-p 采样生成文本**
```python
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
prompt_cot = prompt + "\n\nExplain step by step."

torch.manual_seed(0)
response_1 = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9
)
```
#A 为节省空间而截断的回答

问题指出 $3x-9$ 的一半等于 $x+37$。我们需要求出 $x$ 的值。
### 步骤 2：将问题转化为方程
... #A
### 最终答案
\[
\boxed{83}
\]

因为我们使用温度缩放和 top-p 采样，改变随机种子会产生不同的回答：

```python
torch.manual_seed(3)
response_2 = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9, 
)
```
#A 为节省空间而截断的回答

这次，模型回答如下：
我们从给定的方程开始：
\[
\frac{1}{2} \times (3x - 9) = x + 37 
\]
... #A
最终答案：
\[
\boxed{83}
\]

在这两种情况下，模型都产生了正确的最终答案（83）。第二个回答要短得多，我们可以通过打印每个回答的字符数或 token 数来确认：

```python
print("Response 1 characters:", len(response_1))
print("Response 1 tokens:", len(tokenizer.encode(response_1)))
print("\nResponse 2 characters:", len(response_2))
print("Response 2 tokens:", len(tokenizer.encode(response_2)))
```

Response 1 characters: 1419 
Response 1 tokens: 534 
Response 2 characters: 533 
Response 2 tokens: 231

较短的回答不一定更好。如果两个回答都达到了相同的正确答案，判断哪个更好并不简单。这通常取决于人类对清晰度、有用性以及中间步骤部分正确性的偏好。

对回答的中间步骤进行评分仍然是一个活跃的研究领域（参见附录 A），诸如过程奖励模型之类评估推理本身的方法在实践中并不总能产生更好的输出。

如果两个回答的定性价值相当，有一点是确定的：较短的回答更便宜，因为它们需要生成更少的 token，因此更受青睐。

## 5.3 使用基于规则的评分器对 LLM 回答进行评分

在上一节中，LLM 生成了两个正确的回答。在本节中，我们开发一个简单的基于规则的评分函数来比较它们（图 5.4）。

**图 5.4** 本节实现一个基于规则的评分器，对预训练 LLM 生成的不同答案进行排名。

基于规则的评分函数为两个 LLM 回答中的每一个分配一个分数（图 5.5），这使我们能够对它们进行排名并选择更好的一个。这里，"更好"指的是格式和简洁性，而不是正确性。

**图 5.5** 两个生成的回答达到了相同的正确答案，但它们的解释不同。评分器评估回答并为每个回答分配分数。

实现评分函数（评分器）有几种方法，如图 5.5 所示。评分器可以是启发式的，即一种简单的基于规则的方法。它也可以是另一个 LLM 来对答案进行评分，通常被称为 LLM-as-a-judge（参见附录 F.5）。或者它可以依赖内部概率分数或似然度，这将在本章后面探讨。

在本节中，我们从启发式开始，即一个基于规则的评分器，我们在清单 5.3 中实现它。我们从这种启发式版本开始，不是因为它是最复杂的选项，而是因为它给了我们一个简单的基线，使后面基于概率的评分器更容易比较和推理。

**清单 5.3 一个简单的基于规则的评分器**
```python
from reasoning_from_scratch.ch03 import extract_final_candidate
import math

def heuristic_score(
    answer,
    prompt=None,  #A
    brevity_bonus=500.0,
    boxed_bonus=2.0,
    extract_bonus=1.0,
    fulltext_bonus=0.0,
):
    score = 0.0
    #B
    cand = extract_final_candidate(answer, fallback="none")
    if cand:
        score += boxed_bonus
    #C
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            cand = extract_final_candidate(
                answer, fallback="number_then_full"
            )
            if cand:
                score += fulltext_bonus
    #D
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score
```
#A 我们在本节中忽略的占位符
#B 奖励具有最终框定值的回答
#C 如果回答没有框定值，则给予较弱的奖励
#D 添加一个随文本长度衰减的简洁性奖励

这个 `heuristic_score` 根据答案可以被提取的干净程度以及答案的长度为 LLM 答案分配一个数值分数。例如，如果答案有 `\boxed{}` 回答，我们会给予奖励（`boxed_bonus`）。否则，如果它至少包含一个数字，我们会给予较小的奖励（`extract_bonus`）。`brevity_bonus` 根据答案的简短程度分配分数。

绝对值并不是很重要。更重要的是它们的相对比例。这里，`boxed_bonus=2.0` 故意大于 `extract_bonus=1.0`，以便优先选择框定清晰的最终答案，而不是我们只能提取数字的答案。简洁性项最多贡献 1.5 个额外分数，因此它主要用于打破质量相似的答案之间的平局，而不是主导提取奖励。我们在这里还保持 `fulltext_bonus=0.0`，以便除非我们能从中提取更清晰的最终候选答案，否则不会奖励任意长形式的答案。

> **注意** 我们开发了一个评分器，在这里不使用第 3 章的验证器，因为我们假设我们不知道问题的真实答案。验证器仅用于在现有测试集上评估模型时进行评估。

`prompt` 参数是一个占位符，在函数本身中没有用途，但当我们开发本章后面带有可交换评分器插件的自精炼函数时，它会使我们的生活更轻松（代码也更简单）。

虽然 `brevity_bonus=500.0` 可能看起来很高，但请注意，它通过 `1.5 * math.exp(-len(answer) / brevity_bonus)` 用作指数衰减项。下面的图说明了它对分数的影响。

**清单 5.4 绘制简洁性惩罚曲线**
```python
import matplotlib.pyplot as plt

def plot_brevity_curve(brevity_bonus, max_len=2048):
    lengths = torch.arange(1, max_len)
    scores = 1.5 * torch.exp(-lengths / brevity_bonus)
    plt.figure(figsize=(4, 3))
    plt.plot(lengths, scores)
    plt.xlabel("Text length (number of characters)")
    plt.ylabel("Score contribution")
    plt.tight_layout()
    #plt.savefig("brevity_curve.pdf")
    plt.show()

plot_brevity_curve(500)
```

**图 5.6** 我们的评分器使用的简单基于规则的长度惩罚。较长的解释获得较小的分数贡献。

分数奖励对于较短的答案接近 1.5，而超过 1,000 个字符的答案获得 0.2 或更少的奖励。

作为粗略的直觉，几百个字符对应于一个短段落或一个简洁的解题过程，而 1,000 个字符或更多通常意味着一个相当长的多句解释。因此，这一项适度地偏爱紧凑的答案，但它不会强迫模型将所有内容压缩成一行回答。

为简单起见，简洁性奖励使用字符数而不是 token 数来计算，这避免了将 tokenizer 传递给评分函数。使用 token 计数也是合理的（甚至可能更可取）。

我们现在将启发式评分器应用于上一节的第一个（较长的）回答：

```python
print(round(heuristic_score(response_1), 3))
```

得到的分数是 2.088。接下来，让我们尝试第二个（较短的）回答：

```python
print(round(heuristic_score(response_2), 3))
```

计算的分数是 2.517，这意味着第二个（较短的）回答将是首选答案。

本节介绍了评分方法的基本思想。在本章后面，我们将开发另一个评分器，并在将其用作自精炼方法的一部分时回到启发式评分器。

> **练习 5.1：在自一致性中使用启发式评分器作为平局决胜器**
> 扩展上一章中的自一致性实现（`self_consistency_vote`），使其能够处理候选答案之间的平局。例如，当两个或更多答案获得相同数量的投票时，将本节中的启发式评分器（`heuristic_score`）应用于平局的候选者，并选择分数最高的一个。提示：您不需要修改 `self_consistency_vote` 函数本身来应用平局决胜，但您可以将其应用于 `self_consistency_vote` 函数返回的结果字典。

> **练习 5.2：在 Best-of-N 设置中使用启发式评分器**
> 修改自一致性实现，以便使用启发式评分器而不是多数投票来选择最终答案。为每个问题生成 N 个候选答案（其中 N=2 或更高），使用启发式评分器对每个候选者进行评分，并选择分数最高的作为最终预测。（如果我们使用评分器而不是多数投票，文献中将该方法称为 Best-of-N，而不是自一致性。）
> 将此方法应用于 MATH-500 的一个小子集，并将结果与纯 Best-of-N 和上一练习中的自一致性平局决胜器进行比较。

请注意，评分器包含几个凭直觉选择的参数，这就是为什么我们将其称为启发式分数。为了优化提取和简洁性奖励设置，我们可以将此评分器插入练习 5.1 和 5.2 中的自一致性或 Best-of-N 方法中，并评估哪些设置在 MATH-500 等基准数据集上产生更高的准确性。

作为实用的经验法则，如果评分器过于频繁地偏好最终结果难以解析的答案，则增加与提取相关的奖励。如果评分器不断偏好不会改善最终结果的长篇大论答案，则增加简洁性压力。另一方面，如果评分器开始偏好过于简短但不完整的答案，则减少简洁性惩罚或增加对清晰可提取最终答案的奖励。思考这些权衡通常比孤立地关注确切的数值更有用。

## 5.4 理解 token 概率分数

在本节中，我们朝着构建基于模型自身置信度的评分器迈出第一步（图 5.7 中的步骤 5），其中置信度意味着模型分配的概率。这个想法是，在每个位置，模型将概率质量分布在可能的下一个 token 上。如果提议答案中的 token 始终获得高概率，这表明该答案与模型自身的内部偏好更兼容。相反，如果模型为许多答案 token 分配非常低的概率，则表明模型本身并不强烈支持该答案。具体来说，在本节中，我们首先计算提议答案的 token 概率分数，并用它们来估计模型认为该答案的可能性有多大（图 5.7 中的步骤 3）。

这起初可能看起来像是一个弯路，但这些概率和 log-probability 概念在下一章中再次变得重要，届时它们将出现在训练目标中，而不仅仅是作为推理时评分信号。

**图 5.7** 在本节中，我们从简单的基于规则评分器转向 token 级概率。这些概率构成了我们稍后将在自精炼方法中使用的 logprob 评分方法的基础。

本节计算的 token 概率分数在文献中也被称为 next-token probabilities、per-token probabilities、sequence likelihoods，或宽泛地称为 token-level likelihoods。它们代表模型为每个可能的下一个 token 分配的概率，表示为词汇表上的归一化（softmax）分布。虽然这起初听起来很复杂，但它依赖于与文本生成相同的机制，我们在上一章中已经介绍过。

> **注意** 术语 logprob 是 AI 文献中 log-probability 的常见简写。

例如，在上一章中，我们看到在每一步模型为词汇表中的每个 token 分配一个所谓的 logit 分数，然后基于这些分数选择下一个 token。这个我们在第 4 章 4.4 节讨论过的过程，为了参考在图 5.8 中再次说明。

**图 5.8** LLM 如何选择下一个 token。模型将输入文本转换为 token ID，计算每个词汇表 token 的分数，并选择分数最高的 token。右侧的图显示了词汇表子集的 logit 值，其中 Berlin 的 token 具有最高的分数。

在上一章中，我们查找了对应于最高分数的词汇表条目（图 5.8 中的词汇表索引 19846，对应于 token "Berlin"），以获取下一个生成的 token。

在这里，我们出于不同的目的重新审视 logits。如图 5.8 所示，我们不想生成下一个 token，而是想使用这些分数来量化模型对特定答案的置信度。换句话说，我们在这里没有基于这些分数生成任何内容，而只是在检查这些分数。

例如，考虑我们想要评分的两个候选答案：
1. "The capital of Germany is Berlin"
2. "The capital of Germany is Bridge"

目标是量化模型对每个答案的置信度。重要的是要记住，模型置信度并不自动意味着正确性。模型可以为某个答案分配高概率，仅仅是因为它很好地符合它学到的模式，即使该答案在事实上是错误的。在实践中，当这种情况发生时，我们仍然需要其他工具，例如验证器、外部工具或检索，或基于比较的方法（如自一致性）来检测和纠正自信的错误答案。

在图 5.8 中，输入文本 "The capital of Germany is" 被输入到模型中，并选择分数最高的下一个 token（"Berlin"）。在这里，我们不选择 token，而是比较分配给两个候选下一个 token "Berlin" 和 "Bridge" 的分数，如图 5.9 所示。（"Bridge" 被用作替代，因为它出现在图 5.8 所示的词汇表范围内。）

**图 5.9** 我们如何查找特定 token 的 logit 分数。在将输入文本通过模型传递后，我们为每个词汇表 token 获得一个 logit 值。然后我们将想要评分的候选 token 转换为 token ID，并从分布中读取它们对应的 logit 值。

与其使用图 5.9 所示的原始 logits，我们将它们转换为概率。概率更容易解释，在不同输入之间可比较，并构成本章后面使用的 logprob 评分方法的基础。

如第 4 章所述，应用 `torch.softmax` 将 logits 转换为概率值。图 5.10 显示了得到的概率分布。

**图 5.10** 下一个 token 评分。输入文本被转换为 token ID 并输入到 LLM，LLM 输出下一个 token 的 logits。应用 softmax 后，这些 logits 变成概率，其中 "Berlin" 等 token 获得高概率，而 "Bridge" 等不太可能的选择获得接近零的值。

图 5.10 所示的 token 概率与上一章中的计算方式相同，使用 `torch.softmax`，作为第 4 章 4.4.3 节中描述的多项式采样过程的一部分。这里的区别是我们不从分布中采样。相反，我们只是查找分配给特定 token 的概率。

在图 5.10 中，只有 "Berlin" 的条形可见，概率为 0.1695。绘制的范围内其他 token 的概率接近零。显示的条形之和不等于 1，因为完整词汇表包含 151,000 个 token，而图只显示了分布的一小部分（token 索引 19,800–19,900）。

这表明，给定输入文本 "The capital of Germany is"，模型对 "Berlin" 的置信度高于 "Bridge" 作为下一个 token。

在实践中，我们通常想要比较完整的答案而不是单个 token。从单 token 到序列级评分的扩展如图 5.11 所示。

**图 5.11** 计算给定序列的 token 概率分数。对于每个位置，我们将前面的文本输入模型并读取下一个 token 的 softmax 概率。将这些条件概率相乘得到完整序列的联合概率。

如图 5.11 所示，我们计算每个 token 给定其前面 token 的概率。这与图 5.10 中的过程相同，只是我们对序列中的每个 token 重复它。在普通文本生成过程中，模型在生成时也是一步一步这样做的，然后选择下一个 token。

在这里，我们以不同的方式重用相同的机制：我们不采样新 token，而是取一个完成的候选答案，并在模型下逐个评分其每个 token。然后我们将这些概率相乘以获得完整序列的概率，也称为**联合概率**。

**TOKEN 序列的联合概率**

对于熟悉数学符号的人来说，token 序列的联合概率可以紧凑地写成条件概率的乘积。对于序列 $x_1, x_2, ..., x_T$ 和模型权重 W，这是：

展开后，这变成：

这与我们在图 5.11 中计算的内容完全匹配。在每个位置，我们将上下文输入模型，获得下一个 token 的概率，并将这些分数相乘。

理想情况下，分配给序列 "The capital of Germany is Berlin" 的概率应该远高于无意义的答案 "The capital of Germany is Bridge"。因为联合概率是通过乘以许多小值获得的，所以图 5.11 中两个序列的结果概率都接近零。我们将在下一节解决这个问题。

图 5.11 中的两个答案几乎相同，仅最后一个 token 不同（"Berlin" 与 "Bridge"），我们在这里为了简单和说明而使用。相同的方法也可以应用于差异更大的序列，例如本章前面为 MATH-500 提示（"Half the value of $3x-9$ is $x+37$. What is the value of $x$?"）生成的答案。

**此评分与贪婪和温度加 top-p 采样的区别**

当我们计算 token 概率分数时，我们不是在生成文本。完整序列已经存在，我们只是查询模型在固定上下文作为输入的情况下每个下一个 token 的概率。因此，这些概率不会影响后面的输入。这是一个回顾性评分过程，而不是生成步骤。

生成方法，如我们在第 4 章编码的 `generate_text_stream_cache`（贪婪采样）和 `generate_text_top_p_stream_cache` 函数，行为非常不同。贪婪采样总是选择最可能的下一个 token，而温度采样和 top-p 采样从重塑的概率分布中抽取。这些过程在每一步修改分布，然后提交给单个 token，该 token 成为下一个位置的输入。

一个重要的后果是，这些采样策略（包括贪婪采样）都不能保证产生模型下整体概率最高的序列。在一个步骤看起来次优的 token 可能导致未来步骤中后续 token 的可能性大大提高。另一方面，局部最优选择可能将模型引向后来概率较低的序列。

即使我们确实找到了模型下全局最高概率的序列，那也不能保证它是正确答案或对用户最有用的答案。它只意味着它是模型在其学习分布下自身最偏好的序列。

温度和 top-p 采样通过在采样前扁平化或截断分布来增加更多可变性，这可能进一步将生成的文本与全局最可能的序列断开连接。

通过对现有答案进行评分，我们避免了所有这些复杂性。我们不运行采样算法，模型也不选择任何 token。我们只评估如果给定序列已经被写出，它会有多大概率。

在这么多概念解释之后，让我们现在看看 token 概率计算的实际操作：

**清单 5.5 计算下一个 token 概率**
```python
@torch.inference_mode()
def calc_next_token_probas(model, tokenizer, prompt, device, show=True):
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    logits = model(token_ids.unsqueeze(0)).squeeze(0)     #A
    all_probas = torch.softmax(logits, dim=-1)            #A
    #B
    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:] #C
    next_token_probas = all_probas[t_idx, next_ids] #D
    prod_next_token_probas = torch.prod(next_token_probas)  #E
    if show:
        print("Next-token probabilities:", next_token_probas)
        print("Joint probability:", prod_next_token_probas)
    else:
        return next_token_probas, prod_next_token_probas
```
#A 获取 logits 和概率，类似于文本生成函数
#B 选择我们评分的位��（这里：全部）
#C 由于我们有文本，我们知道真正的下一个 token
#D 获取每个下一个 token 的概率
#E 序列的似然性是概率分数的乘积

如上所示，`calc_next_token_probas` 函数计算下一个 token 概率，如图 5.11 所示，相对较短且乍一看非常简单。这个函数中发生了很多事情来执行计算，如图 5.12 中用一个更简单的输入文本示例和一个微小的五词词汇表所说明的。

**图 5.12** 提取下一个 token 概率。在将输入文本转换为 token ID 后，模型计算通过 softmax 函数转换为概率的 logits。然后使用位置索引张量和真实下一个 token，我们获得模型为每个下一个 token 计算的概率。

`calc_next_token_probas` 函数，如图 5.12 中用一个简单示例说明的，分几步计算下一个 token 概率。首先，它以与前面介绍的文本生成函数相同的方式计算 logits（例如，第 2 章 `generate_text_basic` 中的 `out = model(token_ids)` 行）。

得到的 logits（图 5.12 中的步骤 2）形状为 [sequence_length, vocab_size]，然后使用 `torch.softmax`（步骤 3）转换为归一化概率分布，以便每个位置的概率之和为 1。

为了提取每个位置分配给实际下一个 token 的概率，我们构造一个索引张量 `t_idx` 用于我们想要评分的位��，以及另一个张量 `next_ids`，包含相应的目标 token。例如，如果输入文本是 "The capital of Germany is Berlin"，那么 `t_idx` 指的是对应于 "The capital of Germany is" 的位置，而 `next_ids` 包含偏移一个位置的 token："capital of Germany is Berlin"。

这些目标 token 只是输入 token 偏移一个位置，因为 LLM 被训练来预测序列中的下一个 token。例如，给定输入 "The capital of Germany is Berlin"，模型在位置 2（"capital" 的 token）被要求预测位置 3（"of" 的 token）。使用 `all_probas[t_idx, next_ids]`（步骤 4 和 5）正好检索序列中每个位置的这些概率值。

最后，函数使用 `torch.prod` 计算序列似然性，即每个 token 概率的乘积（步骤 6）。

让我们最终看看这个函数的实际操作：

```python
torch.set_printoptions(precision=4, sci_mode=True)
calc_next_token_probas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Berlin"
)
```

得到的输出是：
Next-token probabilities: tensor([6.1512e-05, 4.6484e-01, 
1.6724e-02, 7.3828e-01, 1.6895e-01], dtype=torch.bfloat16) 
Joint probability: tensor(5.9372e-08, dtype=torch.bfloat16)

接下来，让我们尝试另一个文本：

```python
calc_next_token_probas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Bridge"
)
```

这导致：
Next-token probabilities: tensor([6.1512e-05, 4.6484e-01, 1.6724e-02,
7.3828e-01, 2.9802e-07], dtype=torch.bfloat16) 
Joint probability: tensor(1.0481e-13, dtype=torch.bfloat16)

分析上面的结果，第一个回答（以 "Berlin" 结尾）给我们的联合概率（序列似然性）为 5.9372e-08，大于第二个回答的联合概率 1.0481e-13。（十进制形式，5.9372e-08 是 0.000000059372，1.0481e-13 是 0.00000000000010481。）

这些结果是有道理的，因为 "Berlin" 答案获得了比 "Bridge" 答案更高的序列似然性，正如预期的那样。两个值都非常小，因为将许多小于 1 的概率相乘会很快产生接近零的数字。这使得原始似然性在实践中难以处理，特别是对于较长序列，下溢成为一个问题。

> **注意** 这些分数反映模型的内部似然性，而不是正确性的校准概率。换句话说，它们告诉我们模型在其自身分布下对一个答案的偏好程度超过另一个答案，而不是答案是否实际上正确。高分意味着答案很好地符合模型学到的模式，而不是保证它是正确的。

在下一节中，我们将介绍对此计算的修改，即 log-probabilities，它避免了这些数值问题，并为我们提供了一种更稳定的方式来评分整个序列。

**概率与似然性**

您可能已经注意到，本节使用了两个术语，"probability" 和 "likelihood"。有区别吗？在统计学领域，probabilities 描述在我们观察任何数据之前事件发生的可能性，并且必须在所有可能结果上求和为 1。Likelihoods 则衡量特定模型解释观察到的数据的程度，并被视为模型权重或参数的函数，而不是数据。简而言之，probabilities 预测给定模型的数据，而 likelihoods 评估给定数据的模型。（有关更详细的示例，请参阅我关于 probabilities 与 likelihoods 的文章：https://sebastianraschka.com/faq/docs/probability-vs-likelihood.html）

在 LLM 中，next-token probability 技术上是一个 probability，而不是 likelihood，因为它来自所有可能的下一个 token 的归一化分布，其和为 1。换句话说，`next_token_probas` 中的值是 probabilities，而不是 likelihoods。它们直接来自模型在每个位置对词汇表的 softmax，这是一个归一化的概率分布。

乘积，计算为联合概率（`torch.prod(next_token_probas)`），是模型分配给序列的概率。这个量通常被称为序列 likelihood（特别是当被视为模型参数的函数时）。

## 5.5 从 token 概率分数到 log-probabilities

上一节计算的 token 概率可以用作评分函数来对不同回答进行排名。正如我们所见，概率分数可能非常小，特别是当相乘以获得联合概率或序列似然性时。

在本节中，为了避免处理如此小值时经常出现的数值稳定性问题，我们对这些概率应用对数缩放（也称为 log scaling），如图 5.13 的概述所示。

**图 5.13** 概述我们如何从高 token 概率转向 token log-probabilities，它们为稍后用于自精炼的 log-probability 评分提供了数值上更稳定的基础。

请注意，本节只是对上一节中的概率应用缩放变换，总体目标仍然是计算 token 概率，或者在本例中，log-probabilities，如图 5.14 所示。

例如，考虑这个计算一组 logit 值概率的简单示例：

```python
torch.set_printoptions(precision=4, sci_mode=False)
logits = torch.linspace(-2, 2, steps=7)
probas = torch.softmax(logits, dim=-1)
print(probas)
```

概率值为：
tensor([0.0090, 0.0175, 0.0341, 0.0665, 0.1295, 0.2522, 0.4912])

我们可以通过 `torch.log` 函数将它们转换为 log-probability 分数：

```python
print(torch.log(probas))
```

log-probability 值为：
tensor([-4.7109, -4.0442, -3.3776, -2.7109, -2.0442, -1.3776, -0.7109])

这里，`torch.log` 应用数学上的自然对数，其中 $log(0.0090) = -4.7109$，反之 $e^{-4.7109} = 0.0090$。

与其链式调用 `torch.log(torch.softmax(...))`，PyTorch 还有一个优化的 `torch.log_softmax` 函数，它结合了这两个操作：

```python
log_probas = torch.log_softmax(logits, dim=-1)
print(log_probas)
```

与之前类似，这返回：
tensor([-4.7109, -4.0442, -3.3776, -2.7109, -2.0442, -1.3776, -0.7109])

请注意，log 缩放只改变值的大小，不改变排序顺序，因此具有更高似然性的序列总是也具有更高的 log-likelihood。为了直观地看到这一点，让我们并排放置 logits、probabilities 和 log-probabilities：

**清单 5.6 绘制 logits、softmax probabilities 和 log-softmax 值**
```python
plt.figure(figsize=(9, 4))
#A
plt.subplot(1, 3, 1)
plt.bar(range(len(logits)), logits, color="C0", alpha=0.7)
plt.title("Logits")
plt.xlabel("Token index")
plt.ylabel("Value")
plt.grid(alpha=0.3)
#B
plt.subplot(1, 3, 2)
plt.bar(range(len(probas)), probas, color="C1", alpha=0.7)
plt.title("torch.softmax(logits)")
plt.xlabel("Token index")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
#C
plt.subplot(1, 3, 3)
plt.bar(range(len(log_probas)), log_probas, color="C2", alpha=0.7)
plt.title("torch.log_softmax(logits)")
plt.xlabel("Token index")
plt.ylabel("Log-probability")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("logits_softmax_log_softmax.pdf")
plt.show()
```
#A 绘制 logits
#B 绘制 softmax probabilities
#C 绘制 log-softmax 值

**图 5.14** 简单示例中 logits、softmax probabilities 和 log-probabilities 的比较。Log-probabilities 保留了概率的排序顺序。

正如我们在结果图（图 5.14）中看到的，log-probabilities 与 logits 和概率值具有相同的排序顺序。较大的 logit 对应于较大的概率和较不负面的 log-probability。另一方面，较小的 logits 产生较小的概率和更负面的 log-probabilities。

在实际操作中，更高的概率更好，因为它们表明模型对 token 分配了更多的置信度。对于 log-probabilities，接近零的值更好，因为零对应于概率 1，而非常负面的值对应于极小的概率。这使得比较 token 变得容易：最不负面的 log-probability 是最可能的，最负面的 log-probability 是最不可能的。

虽然图 5.14 说明了 logits、probabilities 和 log-probabilities 之间关系的简单示例，但我们可以将其应用于前面 "The capital of Germany is" 提示示例的概率值，如图 5.15 所示。

**图 5.15** 对于下一个 token 评分，logits 如何转换为概率和 log-probabilities。正确的下一个 token（"Berlin"）获得高 logit，这变成高概率和较不负面的 log-probability，而不太可能的候选者如 "Bridge" 映射到非常小的概率和大的负面 log-probabilities。

我们使用概率而不是 logits，因为概率是归一化的，更容易解释为置信度值。前面我们还注意到，log-probabilities 提供了额外的好处，因为它们允许比使用原始概率进行数值上更稳定的计算。

数值稳定性来自于即使非常小的概率分数也能映射到合理数值范围内的 log-probabilities。另一个原因是，将许多概率相乘会很快将结果推向零，而取对数将这种乘法变成加法，这避免了下溢，对于长序列来说更加稳定。

图 5.16 展示了如何为前面使用的两个示例文本计算联合 log-probability（或序列 log-likelihood）。

**图 5.16** Token 级 log-probabilities 如何累积形成序列 log-probabilities。每行显示给定前面文本的下一个 token 的 log-probability。将这些值相加得到完整序列的联合 log-probability。

前面，当我们计算常规联合概率分数时，我们看到两个序列的值都接近零。现在，使用联合 log-probability，我们可以在图 5.16 中看到，即使只改变最后一个 token（例如，"Berlin" 与 "Bridge"）也会导致明显不同的总 log-probability（-16.6250 与 -29.8750）。

**使用 LOG-PROBABILITIES**

当我们计算序列的联合概率时，我们将许多介于 0 和 1 之间的数字相乘：

在展开形式中，我们可以将其写为：

这些乘积很快变得极小，这既不方便又可能导致数值下溢。简而言之，当数字变得太小时，计算机可能会将它们四舍五入为零，然后我们就失去了有用的信息。为了避免这种情况，我们经常在 log 空间中工作。取联合概率的对数得到

利用对数的乘积等于对数之和这一事实，这展开为

换句话说，序列的 log-probability 只是其各个 token 的 log-probabilities 之和。这在数值上更稳定，也更容易处理。它也匹配 `torch.log_softmax` 返回的值，它直接提供 log-probabilities。

因此，求和 log-probabilities 是机器学习和 AI 中的标准方法。

重用上一节计算 token 概率的代码（清单 5.6），实现 token log-probability 函数很简单，因为它只需要两个小的更改：将 `torch.softmax` 改为 `torch.log_softmax`，将 `torch.prod` 改为 `torch.sum`。

> **警告** 使用不匹配的组合，例如 `torch.softmax` 与 `torch.sum` 或 `torch.log_softmax` 与 `torch.prod`，在技术上是可能的，但在数学上是错误的。

让我们现在尝试用前面的示例提示更新后的函数：

**清单 5.7 计算下一个 token log-probabilities**
```python
@torch.inference_mode()
def calc_next_token_logprobas(model, tokenizer, prompt, device, show=True):
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    #A
    all_logprobas = torch.log_softmax(logits, dim=-1)
    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:]
    next_token_logprobas = all_logprobas[t_idx, next_ids]
    #B
    sum_next_token_logprobas = torch.sum(next_token_logprobas)
    if show:
        print("Next-token log-probabilities:", next_token_logprobas)
        print("Joint log-probability:", sum_next_token_logprobas)
    else:
        return next_token_logprobas, sum_next_token_logprobas

calc_next_token_logprobas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Berlin"
)
```
#A 我们现在使用 log_softmax
#B 我们用求和替换乘积

结果是：
Next-token log-probabilities: tensor([-9.6875, -0.7695, -4.0938,
-0.3008, -1.7812], dtype=torch.bfloat16) 
Joint log-probability: tensor(-16.6250, dtype=torch.bfloat16)

现在第二个序列：

```python
calc_next_token_logprobas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Bridge"
)
```

这返回：
Next-token log-probabilities: tensor([ -9.6875,  -0.7695,  -4.0938,
-0.3008, -15.0000], dtype=torch.bfloat16) 
Joint log-probability: tensor(-29.8750, dtype=torch.bfloat16)

正如我们所见，"Berlin" 和 "Bridge" 序列之间的差异现在更加明显（-16.6250 和 -29.8750），前者得分高得多（更好）。

## 5.6 使用 log-probabilities 评分模型置信度

前两节详细解释了 token 概率和 log-probabilities 的概念。在本节中，我们对 log-probability 计算进行一些轻微修改，以开发基于 log-probability 的评分函数，类似于我们在本章开头开发的启发式评分器，如图 5.17 所示。

**图 5.17** 本节实现一个基于 token log-probabilities 的 logprob 评分器，我们将在本章后面的自精炼方法中使用它。

我们开发的 logprob 评分方法与上一节介绍的过程非常相似。我们进行两个主要修改。

首先，我们从分数计算中排除提示，只计算答案 token 的分数。其次，我们对 token log-probabilities 取平均（而不是求和），以便更公平地比较两个不同长度的序列。这两个修改后的更新计算如图 5.18 所示。

**图 5.18** 修改后的 logprob 评分过程。提示 token 从计算中排除，只收集答案 token 的 log-probabilities。然后对这些值取平均以获得长度归一化的分数，这使我们能够公平地比较不同长度的答案。

为了说明目的，我们可以使用上一节的 `calc_next_token_logprobas` 函数来实现图 5.18 所示的两个修改。例如，假设我们有以下示例提示和答案：

```python
example_prompt = "What is the capital of Germany?"
example_answer = " The capital of Germany is Berlin."
next_token_logprobas, sum_next_token_logprobas = calc_next_token_logprobas(
    model, tokenizer, device=device,
    prompt=example_prompt+example_answer,
    show=False
)
print("Next-token logprobas:", next_token_logprobas)
print("Joint log-probability:", sum_next_token_logprobas)
```

这打印以下输出：
Next-token logprobas: tensor([-0.4512, -0.3418, -8.3125, -0.3906,
-3.8125, -3.0469, -1.1719,  0.0000, -0.0155,  0.0000, 
-0.0078, -0.0752, -0.1582], dtype=torch.bfloat16) 
Joint log-probability: tensor(-17.7500, dtype=torch.bfloat16)

（请注意，张量包含提示 token 的分数，这就是为什么数字看起来与图 5.18 不同，但当你继续阅读时，这会变得清楚。）

然后我们可以通过以下代码计算答案 token 的数量，结果为 7：

```python
print(len(tokenizer.encode(example_answer)))
```

然后，为了计算答案 token 上的平均 log-probability，如图 5.18 所示，我们需要对这 7 个答案 token 取平均：

```python
last_7 = next_token_logprobas[-7:]
print(last_7)
print(torch.mean(last_7))
```

这打印：
tensor([-1.1719,  0.0000, -0.0155,  0.0000, -0.0078, -0.0752, -0.1582], 
dtype=torch.bfloat16)
tensor(-0.2041, dtype=torch.bfloat16)

我们可以看到结果平均值 -0.2041 与图 5.18 中显示的相似。

我们可以将 `calc_next_token_logprobas` 代码与这个计算更方便地组合到一个新函数 `avg_logprob_answer` 中，如下所示：

**清单 5.8 答案 token 的平均 log-probability 评分**
```python
@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt)  #A
    answer_ids = tokenizer.encode(answer)  #A
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)
    logits = model(full_ids.unsqueeze(0)).squeeze(0)  #B
    logprobs = torch.log_softmax(logits, dim=-1)      #B
    start = len(prompt_ids) - 1  #C
    end = full_ids.shape[0] - 1  #C
    #D
    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]
    #E
    return torch.mean(next_token_logps)
```
#A 分别编码提示和答案 token 以稍后获取提示长度
#B 与之前 calc_next_token_logprobas 中相同
#C 答案 token 对应位置的索引范围
#D 与之前相同，只是使用 start 和 end
#E 对答案 token 分数取平均

`avg_logprob_answer` 函数总体上与上一节的 `calc_next_token_logprobas` 函数相似，除了我们只计算答案 token 的 logprobs，而不是包括提示在内的整个序列，并对计算的 logprob 值取平均而不是求和。

让我们将这个新函数应用于本节前面的提示和答案：

```python
score_1 = avg_logprob_answer(
    model, tokenizer,
    prompt="What is the capital of Germany?",
    answer=" The capital of Germany is Berlin.",
    device=device
)
print(score_1)
```

这返回 -0.2041，与之前相似，这表明我们正确地实现了该函数。我们现在也可以将这个函数应用于无意义的 "Bridge" 答案：

```python
score_2 = avg_logprob_answer(
    model, tokenizer,
    prompt="What is the capital of Germany?",
    answer=" The capital of Germany is Bridge.",
    device=device
)
print(score_2)
```

这里的结果分数是 -3.8906，远低于 "Berlin" 分数，正如预期的那样。

有了平均 logprob 评分函数，我们现在也可以计算本章开头定义的 MATH-500 提示（`prompt_cot`）和我们存储为 `response_1` 和 `response_2` 的相应回答的分数，这留给读者作为练习。

我们在本章中花费了大量时间来讲解下一个 token 概率和 log-probability 评分的概念。原因之一是我们可以将其用作即将介绍的自精炼部分的评分器。第二个原因是 log-probabilities 的概念在我们实现下一章中具有可验证奖励的强化学习时也会相关。

同时，重要的是不要过度解读 logprob 评分作为万能解决方案。因为它衡量的是模型自身强烈偏好的内容，所以它仍然可能偏爱自信的错误答案，而不是正确但不太强烈偏好的答案。因此，在实践中，logprob 评分最好被视为几个有用评分信号之一，而不是有保证的平局决胜器。

> **练习 5.3：在自一致性中使用 logprob 评分器作为平局决胜器**
> 使用 logprob 评分器代替练习 5.1 中的启发式评分器，扩展上一章中的自一致性实现（`self_consistency_vote`），使其能够处理候选答案之间的平局。然后在（MATH-500 数据集的子集上）运行两个实现，看看哪种平局决胜方法表现更好。

> **练习 5.4：在 Best-of-N 设置中使用 logprob 评分器**
> 扩展自一致性实现，以便使用 logprob 评分器（`avg_logprob_answer`）而不是启发式评分器或多数投票来选择最终答案（类似于练习 5.2）。然后在（MATH-500 数据集的子集上）运行不同的实现，看看哪种平局决胜方法表现更好。
> 提示：您不需要修改 `self_consistency_vote` 函数本身来应用平局决胜，但您可以将其应用于 `self_consistency_vote` 函数返回的结果字典，该字典在第 3 章的 `evaluate_math500_stream` 函数内部使用。

## 5.7 通过迭代反馈进行自精炼

在介绍了多种对 LLM 答案进行评分的方法之后，我们现在来到本章的核心推理扩展技术：自精炼（图 5.19）。

**图 5.19** 我们工作流中的最后一步，其中前面开发的 logprob 评分器在自精炼方法中使用。

本节介绍自精炼并手动演示该过程。下一节，如图 5.19 所示，然后将自动化自精炼循环，并添加对章前面开发的评分方法（启发式评分器和平均 logprob 评分器）的支持。

自精炼是一种 LLM 分析和改进自己答案的技术。如图 5.20 所示，LLM 从对提示的初始答案开始，与常规 LLM 使用相同。然后，它批判答案并改进它。

**图 5.20** 自精炼过程。LLM 首先产生对提示的初始答案，然后收到一个批判提示，要求它分析自己的回答并产生一个简短的批判以及改进计划。在最后一步，模型收到一个精炼提示，其中包含原始问题、其草稿答案和批判，并生成一个包含建议改进的修订答案。

图 5.20 中的自精炼程序乍一看可能很复杂，但它本质上只是对不同提示和输入上的文本生成函数的连续应用。为了更清楚，让我们用一个具体的代码示例逐步讲解它。

> **注意** 对于这个代码示例，为了说明目的，我们不使用思维链提示。在实践中，如果我们使用基础模型，可以将自精炼与思维链提示结合使用。

我们从基础提示和答案开始（图 5.20 中的步骤 1 和 2），基于本章开头 5.2 节 "加载预训练模型" 中的代码：

**清单 5.9 基础提示和答案**
```python
raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)

torch.manual_seed(123)
initial_response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7,
    top_p=0.9, 
)
```

LLM 回答 " \boxed{18}"，这是一个错误的答案（正确答案是 83）。

接下来，我们让 LLM 批判答案。为此，我们编写一个批判提示，其中包括原始问题（`raw_prompt`）和答案（draft），如图 5.20 中的步骤 3 和 4 所示。在代码中，它看起来像这样：

**清单 5.10 批判提示和改进计划**
```python
def make_critique_prompt(raw_prompt, draft):
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\n"
        "Critique:"
    )

critique_prompt = make_critique_prompt(raw_prompt, initial_response)
torch.manual_seed(123)
critique = generate_text_stream_concat_flex(
    model, tokenizer, critique_prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7,
    top_p=0.9, 
)
```

上面的批判提示让 LLM 写出详尽的批判，甚至包含正确答案本身：

问题似乎在设置中存在逻辑错误。"Half the value of $3x-9$ is $x+37$" 的陈述是不正确的，因为 $3x-9$ 的一半应该是 $(3x-9)/2$，而不是 $3x-9$。方程应该是 $\frac{1}{2}(3x-9) = x + 37$。
Fix Plan: 
1. Correct the equation to $\frac{1}{2}(3x-9) = x + 37$.
2. Multiply both sides by 2 to eliminate the fraction: $3x - 9 = 2(x + 37)$.
3. Distribute the 2 on the right side: $3x - 9 = 2x + 74$.
4. Subtract $2x$ from both sides: $x - 9 = 74$.
5. Add 9 to both sides: $x = 83$.

请注意，批判本身可能包含事实错误。例如，它说 "The question itself is incomplete"，这是不正确的，但它仍然继续提出一个正确的改进计划。这很重要，因为在自精炼中，即使批判的部分内容有误，只要它总体上推动模型走向更好的修订，它仍然可能是有用的。

最后，我们使用这个精炼提示来修订原始答案（图 5.20 中的步骤 5 和 6）：

**清单 5.11 答案精炼**
```python
def make_refine_prompt(raw_prompt, draft, critique):
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revised answer:"
    )

refine_prompt = make_refine_prompt(raw_prompt, initial_response, critique)
torch.manual_seed(123)
revised_answer = generate_text_stream_concat_flex(
    model, tokenizer, refine_prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7,
    top_p=0.9, 
)
```

虽然基础模型不是最好的指令跟随者，但精炼提示让 LLM 生成正确答案：
...
\boxed{83}
Final result: The value of $x$ is \boxed{83}.

本节中的程序说明了单次精炼循环。在实践中，重复此循环多次迭代并不罕见。在下一节中，我们将编写一个函数来自动化此过程。

## 5.8 编写自精炼循环

这最后一节将上一节的手动自精炼步骤打包到一个方便的函数中，该函数简化了自精炼的使用，并允许该过程重复固定次数的迭代。

此外，该函数支持插入本章前面开发的评分方法（启发式评分和 logprob 评分），为每个答案计算分数（图 5.21），这可用于决定是否应接受精炼后的答案。

**图 5.21** 带有可选评分的自精炼循环。模型首先产生初始答案，然后批判它，并根据批判生成修订答案。两个答案都可以用本章的评分函数（例如，logprob 评分）进行评估，并且只有当修订答案的分数优于前一个答案时，才接受修订答案。

在图 5.21 中，修订答案获得比初始答案（-1.258）更高的 logprob 分数（-0.377），这使得接受修订是合理的。在实践中，自精炼也可能产生更差的答案，尤其是在运行多次迭代时。因此，评分器提供了一种方法来确定修订答案是否代表改进。

> **注意** 评分并不总能改善结果。在自精炼中是否以及使用哪个评分器取决于 LLM，需要通过在 MATH-500 等基准数据集上进行实验来确定。

完整的代码结合了上一节的步骤，并添加了 `iterations` 参数和评分函数（`score_fn`）选项，如下面的清单 5.12 所示：

**清单 5.12 支持多次迭代和评分的自精炼**
```python
def self_refinement_loop(
    model,
    tokenizer,
    raw_prompt,
    device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=None,
    prompt_renderer=render_prompt,
    prompt_suffix="",
    verbose=False,
    temperature=0.7,
    top_p=0.9,
):
    steps = []
    #A
    prompt = prompt_renderer(raw_prompt) + prompt_suffix
    current_full = generate_text_stream_concat_flex(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_response_tokens,
        verbose=False,
        generate_func=generate_text_top_p_stream_cache,
        temperature=temperature,
        top_p=top_p,
    )
    current_extracted = extract_final_candidate(
        current_full, fallback="number_then_full"
    )
    if score_fn:
        current_score = score_fn(answer=current_full, prompt=prompt)
    else:
        current_score = 0.0
    #B
    for it in range(iterations):
        draft_before_full = current_full
        draft_before_extracted = current_extracted
        score_before = current_score
        #C
        critique_prompt = make_critique_prompt(
            raw_prompt, draft_before_full
        )
        critique_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=critique_prompt,
            device=device,
            max_new_tokens=max_critique_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )
        #D
        refine_prompt = make_refine_prompt(
            raw_prompt, draft_before_full, critique_full
        )
        revised_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=refine_prompt,
            device=device,
            max_new_tokens=max_response_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )
        revised_extracted = extract_final_candidate(
            revised_full, fallback="number_then_full"
        )
        if score_fn:
            revised_score = score_fn(
                answer=revised_full, prompt=prompt
            )
        else:
            revised_score = 0.0
        #E
        step = {
            "iteration": it + 1,
            "draft_full": draft_before_full,
            "draft_extracted": draft_before_extracted,
            "critique": critique_full,
            "revised_full": revised_full,
            "revised_extracted": revised_extracted,
            "score_before": score_before,
            "score_after": revised_score,
        }
        steps.append(step)
        if verbose:
            print(
                f"[Refinement {it+1}/{iterations}]"
                f"\nCurrent: {draft_before_extracted}"
                f"\nRevised: {revised_extracted}"
                f"\nScore before: {score_before:.3f}"
                f"\nScore after: {revised_score:.3f}"
                f"\n{'=' * 25}\n"
            )
        #F
        if revised_score >= current_score:
            current_full = revised_full
            current_extracted = revised_extracted
            current_score = revised_score
    return {
        "final_full": current_full,
        "final_extracted": current_extracted,
        "steps": steps,
    }
```
#A 初始回答（草稿）
#B 运行一次或多次迭代
#C 批判回答
#D 精炼回答
#E 记录结果
#F 如果修订回答不差，则接受修订回答

`self_refinement_loop` 函数运行上一节的自精炼程序一次或多次迭代（`for it in range(iterations)`）。在每次迭代结束时，它比较修订后的分数和当前分数（`revised_score >= current_score`），并且仅在分数相等或更高时保留修订后的答案。

使用评分函数是可选的。当 `score_fn=None`（默认值）时，分数始终设置为 0.0。由于 `0.0 >= 0.0` 评估为 True，因此在运行多次精炼迭代时，最新的答案总是被接受。

接下来，我们使用平均 logprob 评分器 `avg_logprob_answer` 运行自精炼循环。清单 5.8（5.6 节）中的 `avg_logprob_answer` 函数需要几个参数（`model`、`tokenizer`、`prompt`、`answer` 和 `device`）。如上一个清单所示，评分函数仅使用两个参数调用：

```python
# ...
if score_fn:
    revised_score = score_fn(
        answer=revised_full, prompt=prompt
    )
# ...
```

为了使 `avg_logprob_answer` 与此 `score_fn` 调用兼容，我们使用 Python 内置 `functools` 模块中的 `partial` 函数来预先指定剩余的参数：

**清单 5.13 创建平均 log-probability 评分器**
```python
from functools import partial

avg_logprob_score = partial(
    avg_logprob_answer,
    model=model,
    tokenizer=tokenizer,
    device=device
)
```

现在，我们可以在自精炼循环中使用 `avg_logprob_score` 函数：

```python
torch.manual_seed(1)
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

输出如下所示：
```
[Refinement 1/2]
Current: 10
Revised: 83
Score before: -0.855
Score after: -0.226
=========================
[Refinement 2/2]
Current: 83
Revised: 83
Score before: -0.226
Score after: -1.320
=========================
```

查看上面的结果，我们看到模型在第一次迭代中纠正了最初错误的答案（从 10 到 83），分数从 -0.855 提高到 -0.226。在第二次迭代中，分数变得更差，但这没关系，因为我们已经有了正确答案。

我们还可以通过 `results_logprob["final_extracted"]` 访问最佳评分答案的提取数字，在本例中返回 83。如果您对完整答案感兴趣，请使用 `results_logprob["final_full"]`。要阅读批判和更长的答案，您可以打印 `results_logprob` 字典，它包含所有迭代的详细结果。

> **练习 5.5：在自精炼中使用启发式分数**
> 使用我们在 5.3 节中定义的 `heuristic_score` 函数运行 `self_refinement_loop`。

前面的示例表明，自精炼可以帮助模型生成正确答案。为了更好地了解这种自精炼方法的实际用途，我在第 3 章的 MATH-500 上运行了此方法，并将结果总结在表 5.1 中。

**表 5.1 不同自精炼方法的 MATH-500 任务准确率**

| 方法 | 评分 | 迭代次数 | 模型 | 准确率 | 时间 |
|------|------|----------|------|--------|------|
| 1 Baseline (chapter 3) | - | - | Base | 15.2% | 10.1 min |
| 2 Self-refinement | None | 1 | Base | 25.0% | 84.8 min |
| 3 Self-refinement | None | 2 | Base | 22.0% | 165.4 min |
| 4 Self-refinement | Heuristic | 1 | Base | 21.6% | 84.7 min |
| 5 Self-refinement | Heuristic | 2 | Base | 20.8% | 151.4 min |
| 6 Self-refinement | Avg. logprob | 1 | Base | 21.4% | 85.3 min |
| 7 Self-refinement | Avg. logprob | 2 | Base | 22.0% | 165.3 min |
| 8 Baseline (chapter 3) | - | - | Reasoning | 48.2% | 182.1 min |
| 9 Self-refinement | None | 1 | Reasoning | 56.6% | 498.8 min |
| 10 Self-refinement | Heuristic | 1 | Reasoning | 57.8% | 498.6 min |
| 11 Self-refinement | Avg. logprob | 1 | Reasoning | 48.4% | 499.7 min |

表 5.1 中显示的准确率值是在 MATH-500 测试集的所有 500 个样本上使用 "cuda" GPU（DGX Spark）计算的。

正如我们在表 5.1 中看到的，自精炼提高了基础模型的性能（第 1-7 行）。改进非常有限，最佳准确率是在自精炼中不使用评分时实现的。这意味着启发式分数和平均 logprob 分数有时都可能导致错误答案被接受，而不是最初的正确答案。

请注意，当为基础模型和自精炼都添加 "Explain step by step." 思维链时，模型未能超过基础模型（结果未显示）。

查看推理模型结果（第 8-11 行），我们可以看到自精炼和启发式评分的组合将答案准确率提高了近 10%。（如第 3 章所述，可以通过将 5.2 节中 `load_model_and_tokenizer` 的 `which_model="base"` 更改为 `which_model="reasoning"` 来使用 "reasoning" 模型。）

在这两种情况下，平均 logprob 评分似乎都会导致比不使用评分器或启发式评分器更差的性能。这可能是因为平均 logprob 分数与答案在模型下看起来多么自然或预期更密切相关，而不是答案实际上是否正确。因此，即使它们在语义上是错误的，它也可能偏爱流畅、熟悉或语法清晰的答案。或者，换句话说，logprob 标准可能会无意中选择自信的错误，而启发式分数更侧重于答案的格式和结构。

在这一点上，似乎我们花了很多时间讨论平均 logprob 评分的概念。Logprob 评分是使用 LLM 时的基本概念，在即将到来的章节中，当我们实现强化学习训练过程时，它会派上用场。

将表 5.1 中的这些结果与上一章（表 4.1）中的结果进行比较，我们还观察到，对于这个模型在这个数学任务上，自精炼不如自一致性有效。尽管自一致性很简单，但它在实践中效果很好，因此仍然被广泛使用。

本章中我们没有探讨的另一种方法是使用外部模型进行自精炼，类似于附录 F 中 F.5 节描述的 LLM-as-a-judge 设置。例如，我们使用第二个 LLM 来计算分数并撰写批判，而不是使用启发式分数或平均 logprob。

2025 年 11 月，DeepSeek 团队通过 DeepSeekMath-V2 证明，使用第二个 LLM 进行自精炼可以非常成功，在多个数学竞赛中达到金牌级别的表现。具体来说，DeepSeek 团队提出了一种新方法，他们训练第二个 LLM 成为一个好的批判模型，并进一步训练他们的基础模型，使其在自精炼的背景下成为更好的数学问题解决者（相比之下，在本章中，我们在没有任何额外训练的情况下应用了自精炼）。

说到推理和训练，本章结束了无需额外训练的推理扩展（图 5.22）。在下一章中，我们将开始介绍更新模型权重以更好地完成推理任务的训练技术。

**图 5.22** 本书从基本 LLM 使用到推理时推理方法，最后到基于训练的技术的进展概述。本章结束了无需额外训练的推理扩展方法，接下来的章节介绍了更新模型权重以进一步提高推理性能的方法。

## 5.9 总结

- 自精炼通过迭代地批判和改进单个答案，扩展了上一章的推理时扩展思想，而不是像自一致性那样依赖多个独立样本。
- 一个简单的基于规则的评分函数通过奖励可提取的最终答案和更短、更经济的完成来排名模型输出。
- 下一个 token 评分通过将 logits 转换为归一化概率并将这些概率组合成序列级似然性来量化模型置信度。
- Log-probabilities 取代原始概率以避免数值下溢，并将多个 token 的乘积转换为稳定的和与平均值。
- 这些评分函数在自精炼之外更普遍有用，例如，用于打破自一致性中的平局或实现 Best-of-N 选择策略。
- 自精炼过程包括三个阶段：生成初始草稿、产生简短的批判和修复计划，以及生成修订答案。
- 一个可重用的精炼函数自动化了具有多次迭代（精炼轮次）的自精炼工作流，并且可以使用基于分数的接受来仅保留不降低我们前面开发的评分函数之一计算的答案分数的修订。
