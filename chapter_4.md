# 4 通过推理时扩展提升推理能力

## 本章内容
- 提示 LLM 解释其推理过程以提高答案准确性
- 修改文本生成函数以产生多样化的回复
- 通过采样多个回复来提高推理的可靠性

推理性能和答案准确性可以在不重新训练或修改模型本身的情况下得到提升。这些方法在推理阶段运作，即模型生成文本时。如图 4.1 的概览所示，本章将介绍两种推理时扩展方法。正如我们将在本章后面看到的，这两种方法都能使我们之前章节中使用的基础模型的准确率提高一倍以上。

![图 4.1](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F01_raschka.webp)

**图 4.1** 本书所涵盖主题的思维模型。本章聚焦于无需额外训练即可提升推理能力的技术（阶段 3）。具体而言，它扩展了文本生成函数，并实现了一种基于投票的方法来提升答案准确性。下一章将介绍一种推理时扩展方法，即模型迭代地优化自己的答案。

下一节将提供推理时扩展的总体介绍，然后更详细地讨论图 4.1 中所示的推理方法。

## 4.1 推理时扩展简介

总的来说，提升推理能力主要有两种策略：

1. 增加训练计算量（training compute）
2. 增加推理计算量（也称为推理时扩展或测试时扩展）

（在机器学习和 AI 中，"compute" 指的是训练或运行模型所需的计算资源。）这两种方法如图 4.2 所示。

![图 4.2](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F02_raschka.webp)

**图 4.2** 推理时扩展（本章）与训练时扩展（第 5 章之后）的对比。两者都通过使用更多计算资源来提高准确性，但推理时扩展是即时进行的，无需改变模型的权重参数。这些图表的灵感来自 OpenAI 介绍其首个推理模型的文章（https://openai.com/index/learning-to-reason-with-llms/）。

图 4.2 中的图表看起来似乎我们要么通过增加训练时计算量，要么通过增加推理时计算量来提升推理能力。LLM 通常的设计思路是结合大量的训练时计算量（未来章节的主题）和增加的推理时计算量（本章的主题）。

推理时计算扩展（也称为推理时扩展和测试时计算扩展）意味着在答案生成阶段花费额外的计算量，即在模型已经训练完成之后，以提高其回复质量。我们不改变模型的权重，而是让同一个训练好的模型为每个问题做更多的工作，例如生成更多的 token 并"思考"更长时间、采样多个答案，或逐步优化其答案。

在本书中，我们聚焦于三种实用且基础的推理时技术（图 4.3）：

- **方法 1**：扩展思维链（chain-of-thought）回复，提示模型解释其推理过程。这是一种简单的技术，可以显著提升准确性。
- **方法 2**：通过自一致性（self-consistency）进行并行采样，即模型生成多个回复并选择最频繁出现的那个。
- **方法 3**：迭代式自我优化（iterative self-refinement），即模型在多个步骤中审查并改进自己的推理和答案。（该主题在下一章中实现并更详细地介绍。）

![图 4.3](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F03_raschka.webp)

**图 4.3** 本书涵盖的三种推理时方法概览。第一种修改提示以鼓励逐步推理，第二种采样多个答案并选择最频繁的一个。这两种方法都在本章讨论。第三种方法，即模型迭代优化自己的回复，将在下一章介绍。

图 4.3 所示的方法属于推理时扩展的范畴，因为它们导致模型生成更多的 token，从而增加了推理过程中的计算资源消耗。换句话说，这些方法以让推理过程更昂贵为代价来获得更好的准确性，这是推理时扩展技术的一个共同主题。

本章的第一个方法通过提示模型解释其推理过程来提高答案准确性，这是一种简单但非常有效的方法。

接下来，我们扩展前面介绍的文本生成函数，以便为同一输入采样多个回复（图 4.3 中的第 2 种方法）。使用这个修改后的函数，我们实现自一致性，一种基于投票的推理时扩展技术，通过生成多个答案并选择最频繁的一个来提高答案准确性。

我总共评估了十几种不同的推理时扩展技术，进行了数千次实验。图 4.3 所示的三种方法被选入本书，是因为它们结合了三种理想的特性：它们在本书使用的模型上带来了显著的准确性提升，代表了推理时扩展的主要实用范式，并且足够简单，可以从头实现而无需引入太多额外的机制。具体而言，它们涵盖了生成更长回复和生成多个回复这两种主要模式。

包含推理和解释的更长回复（方法 1 和 3）正是我们通常对推理模型的期望，如前几章所讨论的。而并行采样（方法 2）也广泛应用于生产系统中，例如 Claude 4（如 https://www.anthropic.com/news/claude-4 所述）。

## 4.2 加载预训练模型

在开始实现上一节所述的推理时扩展方法之前，我们先加载前几节中使用的预训练基础模型。

清单 4.1 中的代码加载了本章要使用的模型和分词器。它与我们在前几章中使用的代码类似。

由于本章的代码运行成本较低，我建议首次运行时在 "cpu" 设备上运行本章，以获得与本章所示相同的结果（在 GPU 上运行本章可能会微妙地改变结果。）

让我们用 MATH-500 数据集中的一个提示来测试模型，这是我们在上一章中使用过的：

**清单 4.1 加载分词器和基础模型**

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

格式化后的提示如下：

```python
from reasoning_from_scratch.ch03 import render_prompt

raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)
print(prompt)
```

输出：

```
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}
Question:
Half the value of $3x-9$ is $x+37$. What is the value of $x$?
Answer:
```

我们可以将上面的提示作为输入，使用上一章中定义的 `generate_text_stream_concat`。我们希望在保持其余代码不变的情况下比较几种推理时扩展策略。为此，我们对文本生成包装器做了一个小的修改，以便可以替换不同的生成函数和设置，而无需重写周围的提示处理和输出代码。这些更改通过清单 4.2 中的代码注释高亮显示：

**清单 4.2 修改后的 generate_text_stream_concat_flex 函数**

```python
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache

def generate_text_stream_concat_flex(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
    generate_func=None,  #A
    **generate_kwargs    #A
):
    if generate_func is None:  #B
        generate_func = generate_text_basic_stream_cache
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)
    generated_ids = []
    for token in generate_func(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs,  #C
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)
```

简而言之，上面的 `generate_text_stream_concat_flex` 函数与上一章的 `generate_text_stream_concat` 函数类似，只是我们现在可以将文本生成函数（如 `generate_text_basic_stream_cache`）作为函数参数传入，而不是硬编码它。实际的好处是，外部包装器可以统一处理提示编码、流式传输和解码，而内部生成步骤可以被不同的解码策略替换。这使得公平地比较方法变得更加容易，因为我们只改变生成逻辑而不是整个流程。在后续章节中，我们将把 `generate_text_basic_stream_cache` 函数替换为更高级的函数。

用法也与之前类似，只是我们现在可以显式地传递文本生成器函数（例如 `generate_text_basic_stream_cache`）：

```python
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_basic_stream_cache
)
```

生成的输出是：

```
\boxed{20}
```

请注意，这个答案是错误的，正确答案是 83。在本章的剩余部分以及下一章中，我们将实现推理时扩展方法，以使模型生成正确答案。

## 4.3 通过思维链提示生成更好的回复

在上一节加载预训练基础模型并设置文本生成函数之后，本节重点通过所谓的思维链提示（chain-of-thought prompting）来改进模型输出。

思维链提示是一种经典、简单且有效的技术，它修改输入提示以鼓励 LLM 生成解释或所谓的思维链（也称为推理链），如图 4.4 所示。

![图 4.4](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F04_raschka.webp)

**图 4.4** 第一种推理时方法，思维链提示，修改提示以鼓励模型在给出最终答案之前逐步解释其推理过程。

尝试思维链提示的最简单方法是追加一条额外的指令，要求模型逐步推理。有多种方式可以表达这一点。例如，原始的零样本思维链论文使用了 "Let's think step by step."（https://arxiv.org/abs/2205.11916）。这只是一个例子，不同的措辞在实践中可能对不同的模型和任务效果更好。这里，我们使用 "Explain step by step."，因为它在本章的实验中表现良好，并且保持示例简单，如下所示：

```python
prompt_cot = prompt + " \n\nExplain step by step."
response_cot = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=True,
)
```

现在的回复如下：

```
To solve the problem, we need to find the value of \( x \) such that 
half the value of \( 3x - 9 \) is equal to \( x + 37 \).
# ...  #A
### Step 3: Solve for \( x \)
Subtract \( 2x \) from both sides to isolate \( x \):
\[
3x - 2x - 9 = 74
\]
Simplify:
\[
x - 9 = 74
\]
Add 9 to both sides to solve for \( x \):
\[
x = 74 + 9
\]
\[
x = 83
\]
### Final Answer:
\[
\boxed{83}
\]
```

正如我们所见，模型现在写出了一个详细的逐步解释，并且在这种情况下得出了正确答案。

这种简单的思维链提示很好地展示了推理时扩展的权衡。虽然模型现在正确回答了，但它比之前消耗了更多的 token。正如第 2 章所讨论的，LLM 一次生成一个 token，每个额外的 token 都需要模型再进行一次前向传播。因此，这些中间推理步骤不仅使答案更长，还直接增加了延迟、计算成本，以及实践中通常的 API 成本。

请注意，虽然模型在这种情况下生成了正确答案，但并非所有问题都能从思维链提示中受益。在简单问题上，它有时甚至可能降低模型的性能，因为模型有时可能会生成错误的解释并误导自己。这种现象也被称为"过度思考"（overthinking）。

最后，并非每个模型都能从 "Explain step by step"（或类似）指令中受益。在这种情况下，我们使用一个简单的基座模型，它并不总是生成解释，因此思维链提示可以明显帮助。训练过的推理模型，例如我们在上一章练习中使用的 Qwen3 模型的"reasoning"变体，已经在其回复旁边写出了解释，不需要或不会从这种类型的思维链提示中受益。

### 为什么思维链可以提高准确性

思维链提示要求模型写出导致最终答案的中间步骤。这在两个实际方面有所帮助。

首先，逐步走过这些步骤给了模型更多自我纠正的机会。

其次，逐步推理与许多训练示例的书写方式相匹配。例如，大型数学和逻辑数据集通常包含详细的解答，因此要求思维链会使模型与它已经学会的模式对齐。

同时，思维链并不能保证正确性。它仍然可能产生错误的推理，对于非常简单的问题，它甚至可能引入导致更多错误的不必要步骤。换句话说，思维链可以提高许多推理任务的准确性，但它并非普遍有益。

总的来说，思维链回答并没有给模型提供新知识，但它改变了模型使用其现有知识的方式。通常，这种转变可以带来更可靠的答案。这对于数学、代码、逻辑问题以及其他类型的多步问题尤其如此。

> **练习 4.1：在 MATH-500 上使用思维链提示**
>
> 修改第 3 章第 3.9 节中的 `evaluate_math500_stream` 函数，看看思维链提示是否能提高基座模型在 MATH-500 上的准确性。

## 4.4 用温度缩放控制输出多样性

上一节通过扩展模型的答案，让我们初步体验了推理时扩展，即思维链提示。思维链提示可以被视为一种顺序技术，因为我们扩展了下一个 token 预测步骤的数量。

在本章的剩余部分，我们将实现一种生成多个答案的技术，如图 4.5 所示。由于这些答案是相互独立的，这可以实现为并行采样（如果我们有必要资源，例如使用多个 GPU），在这种情况下不会增加用户获得答案的等待时间。

（请注意，思维链提示也可以与这种技术结合，但这将在后面的章节中介绍。）

![图 4.5](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F05_raschka.webp)

**图 4.5** 第二种推理时方法，自一致性采样，生成多个答案并选择最频繁的一个。该方法依赖于本节涵盖的温度缩放，它影响模型如何采样其下一个 token。

图 4.5 所示的自一致性技术，以及本章涵盖的内容，也被称为自一致性采样，并在 Google Research 的论文《Self-Consistency Improves Chain of Thought Reasoning in Language Models》（https://arxiv.org/abs/2203.11171）中被正式描述。

在实现自一致性采样（图 4.5 中的步骤 5）之前，我们首先需要扩展文本生成函数，使其能够为同一提示产生多个不同的答案。为了实现这一点，我们将实现两种技术，温度缩放（步骤 3）和 top-p 过滤（步骤 4），它们允许模型采样不同的回复。

温度缩放是本节的主要焦点，因为它成为本章剩余部分输出多样性的关键控制手段，也构成了后续 top-p 过滤方法的基础。在介绍温度缩放之前，下一小节简要概述了 LLM 中下一个 token 是如何采样的。

这些底层采样控制单独看起来可能很技术性，但它们正是让我们能够生成本章后面自一致性所需的多样化候选答案的关键。

### 4.4.1 理解选择下一个 token 的过程

本小节更仔细地审视了我们到目前为止实现的文本生成过程，并解释了下一个 token 选择过程在底层是如何工作的。这些信息将帮助你理解温度缩放背后的动机。

例如，假设我们有以下简单的提示：

```python
ex_prompt = "The capital of Germany is"
response = generate_text_stream_concat_flex(
    model, tokenizer, ex_prompt, device,
    max_new_tokens=1, verbose=True
)
```

模型的回复是 " Berlin"。

虽然这看起来相对简单，但底层发生了多个步骤，如图 4.6 所示。

![图 4.6](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F06_raschka.webp)

**图 4.6** LLM 如何生成下一个 token。与本书中的其他流程图一样，流程从下到上运行。模型将输入转换为 token ID，计算所有可能的下一个 token 的分数，并选择分数最高的那个作为下一个输出。

生成文本时，正如第 2 章所讨论的，输入首先被转换为 token ID：

```python
input_token_ids = torch.tensor(
    tokenizer.encode(ex_prompt), device=device
).unsqueeze(0)
print(input_token_ids)
```

输出：

```
tensor([[ 785, 6722,  315, 9856,  374]])
```

在第二步（图 4.6 中的步骤 2），我们获取要生成的输出 token 的分数。这些模型输出分数也称为 logits。请注意，LLM 为每个输入 token 生成一个输出 token，但我们只关心最后一个 token，我们通过 `[:, -1]` 张量索引来选择它。这个最后一个 token 对应于我们要生成的 token：

```python
with torch.inference_mode():
    next_token_logits = model(input_token_ids)[:, -1]
print(next_token_logits.shape)
```

打印的输出形状是 `[1, 151936]`，其中 151936 是这个分词器和模型的词汇表大小。词汇表大小包含分词器可以处理和 LLM 可以生成的所有唯一 token。

为了实际获得下一个生成的 token（这里是 " Berlin"），我们必须找到与最大分数相关联的词汇表条目（图 4.6 中的步骤 3）：

```python
max_token_id = torch.argmax(next_token_logits)
print(f"Token ID: {max_token_id}")
print(f"Decoded token: '{tokenizer.decode([max_token_id])}'")
```

输出：

```
Token ID: 19846
Decoded token: ' Berlin'
```

以上，我们涵盖了生成下一个 token 的三个主要步骤。在继续下一节之前，让我们更仔细地看看我们传入 `torch.argmax` 函数以获得 token ID 的 `next_token_logits` 张量的分数分布，并用 matplotlib 绘制它们：

**清单 4.3 绘制下一个 token 的 logit 分数**

```python
import matplotlib.pyplot as plt

def plot_scores_bar(
    next_token_logits, start=19_800, end=19_900,
    arrow=True, ylabel="Logit value"
):
    x = torch.arange(start, end)  #A
    logits_section = next_token_logits[0, start:end].float().cpu()  #B
    plt.bar(x, logits_section)  #C
    plt.xlabel("Vocabulary index")
    plt.ylabel(ylabel)
    if arrow:  #D
        max_idx = torch.argmax(logits_section)
        plt.annotate(
            "Berlin",
            xy=(x[max_idx], logits_section[max_idx]),
            xytext=(x[max_idx] - 25, logits_section[max_idx] - 2),
            arrowprops={
                "facecolor": "black", "arrowstyle": "->", "lw": 1.5
            },
            fontsize=10,
        )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_scores_bar(next_token_logits)
```

请注意，我们将绘图限制在 19,800 到 19,900 之间的 100 个词汇索引 token，而不是绘制所有 151,936 个条目的分数，那样会使图表过于拥挤。（我选择这个特定范围是因为它包含分数最高的条目。）生成的图如图 4.7 所示。

![图 4.7](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F07_raschka.webp)

**图 4.7** 语言模型更大词汇表中 100 个 token 切片的下一个 token logits 示例。每个条形代表该切片内一个可能 token 的分数，其中 "Berlin" 具有最高的 logit 值并被选为下一个 token。

图 4.7 中的图表显示了词汇索引 19,800-19,899 的所有 100 个 logit 值（分数）。这些值大约从 -8 到 20，其中 20 对应于 token " Berlin" 的词汇索引。

### 4.4.2 通过温度参数重新缩放 token 分数（logits）

既然我们已经走过了模型如何选择其下一个 token 的过程，我们就可以介绍温度的概念了。温度，或者更确切地说，所选择的温度参数，改变 logits（token 分数）的尖锐程度或分散程度，这反过来影响如何选择下一个 token。

如图 4.8 所示，本节重点在于在使用它们进行采样之前，用温度参数重新缩放下一个 token 的 logits。重新缩放在这里意味着调整分数的大小，以便采样步骤对分数差异变得或多或少敏感。

![图 4.8](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F08_raschka.webp)

**图 4.8** 在本节中，我们实现温度缩放的核心部分（步骤 3.2），它调整下一个 token 的分数。这使我们能够在后续步骤中控制模型选择其下一个 token 的自信程度。

下面清单 4.4 中实现温度重新缩放步骤（图 4.8 中的步骤 3.2）的代码相对简短且简单。

**清单 4.4 通过温度缩放重新缩放下一个 token 的分数**

```python
def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature
```

本质上，清单 4.4 中的代码在下一节将它们转换为概率之前重新缩放 logit 值。

在实践中，温度值预期为正数。温度为 1.0 意味着没有变化，因为将一个数除以 1 仍然是这个数本身。

我们将在稍后向文本生成函数添加温度缩放时添加额外的安全保护措施。

对于所有其他值，logits 被除以温度。低于 1.0 的温度使分布更尖锐（这将使模型在即将到来的部分中选择下一个 token 时更加自信）。高于 1.0 的温度使 logits 变平，这可以使采样（图 4.8 中的步骤 3.4）更加多样化。换句话说，更高的温度缩小了最高 token 与排名较低 token 之间的概率差距，这增加了采样选择非最大 token 的机会。

让我们看看这个 `scale_logits_by_temperature` 函数的实际效果，并尝试相对极端的温度值 0.5 和 5.0 以获得更强的效果：

**清单 4.5 绘制温度重新缩放后的 logits**

```python
def plot_logits_with_temperature(
    next_token_logits, start=19_800, end=19_900,
    temps=(0.5, 5.0),
):
    x = torch.arange(start, end)
    logits_orig = next_token_logits[0, start:end].float().cpu()
    logits_scaled = [  #A
        scale_logits_by_temperature(logits_orig, T) for T in temps  #A
    ]  #A
    plt.plot(x, logits_orig, label="Original logits", lw=2)  #B
    plt.plot(  #B
        x, logits_scaled[0],  #B
        label=f"T={temps[0]} (sharper)", ls="--", lw=1  #B
    )  #B
    plt.plot(  #B
        x, logits_scaled[1],  #B
        label=f"T={temps[1]} (flatter)", ls=":", lw=3  #B
    )  #B
    # Highlight max logit
    max_idx = torch.argmax(logits_orig)  #C
    plt.annotate(  #C
        "Berlin",
        xy=(x[max_idx], logits_orig[max_idx]),
        xytext=(x[max_idx] - 25, logits_orig[max_idx] + 2),
        arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 1.5},
        fontsize=12,
    )
    plt.xlabel("Vocabulary index")
    plt.ylabel("Logit value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_logits_with_temperature(
    next_token_logits,
    temps=(0.5, 5.0)  #D
)
```

如图 4.9 中的图所示，较大的温度（5.0）产生了更平坦的分数分布，而较小的温度（0.5）产生了更尖锐的分布。

![图 4.9](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F09_raschka.webp)

**图 4.9** 温度缩放对 logits 的影响。较低的温度使分布更尖锐，而较高的温度使其更平坦。（请注意，此可视化显示为折线图以便阅读，尽管条形图会更准确地表示离散的词汇分数。）

请注意，上面清单 4.5 中的绘图代码看起来与我们在上一节中使用的绘图代码（清单 4.3）非常相似。除了应用温度缩放之外，我们现在使用折线图（`plt.plot`）而不是条形图（`plt.bar`）。虽然条形图在技术上是 x 轴上离散词汇索引的更好选择，但折线图在这种情况下更容易可视化和比较重新缩放后的 logits。

重要的不是一个 logit 本身的绝对高度，而是 logits 之间的差距如何变化。如果一个 token（如 "Berlin"）比其他替代方案更突出，它在后面被选中的可能性就会更大。如果差距缩小，与之前相比，排名较低的 token 被选中的可能性就会增加。下一节通过将 logits 转换为概率来说明这一点。

### 为什么是温度？

温度一词来自物理学，在物理学中，温度控制系统中有多少随机性或运动。在 LLM 中，我们使用相同的想法来控制模型选择其下一个 token 的自信程度或创造性程度，正如我们将在接下来的两节中看到的。

### 4.4.3 从概率分布中采样下一个 token

上一节使用不同的温度值重新缩放了 logit 值。这样做的目的是让我们（稍后）影响模型如何选择下一个 token。在下一节进入下一个 token 采样部分之前，本节添加了一个中间步骤：将重新缩放的 logits 转换为概率分数，如图 4.10 所示。

![图 4.10](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F10_raschka.webp)

**图 4.10** 生成 token 的采样过程概览。在本节中，我们重点关注步骤 3.3 和 3.4，其中下一个 token 的分数被转换为概率分布，并基于该分布采样下一个 token。

为了演示如何将重新缩放的 logits 转换为概率分数，如 图 4.10 中的步骤 3.3 所述，我们将使用温度为 5.0。这使得在图中可视化结果概率变得更容易。例如，" Berlin" token 的 logit 值如此之高，以至于否则会主导尺度，使我们难以看到周围 token 的概率。

从重新缩放的 logits 到概率分数的转换可以通过单个函数调用 `torch.softmax` 完成，如下面的清单 4.6 所示。

**清单 4.6 从概率分布中采样下一个 token**

```python
rescaled_logits = scale_logits_by_temperature(next_token_logits, 5.0)  #A
next_token_probas = torch.softmax(  #B
    rescaled_logits, dim=-1
)
print("Probability sum:", torch.sum(next_token_probas))
plot_scores_bar(
    next_token_probas, arrow=False, ylabel="Probability value"
)
```

清单 4.6 中的 `torch.softmax` 函数将 logit 值归一化为 0 到 1 之间的值，并且这些值的总和为 1，我们可以通过以下代码确认：

```python
print("Probability sum:", torch.sum(next_token_probas))
```

此外，让我们通过重用前面清单 4.3 中的 `plot_scores_bar` 函数来可视化转换后的分数：

生成的图，现在 y 轴上是概率值，如图 4.11 所示。

![图 4.11](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F11_raschka.webp)

**图 4.11** 通过对重新缩放的 logits 应用 softmax 函数获得的 token 概率。概率最高的 token（对应于 " Berlin"，但为了代码简单起见省略了标签）被选为下一个输出。

在 图 4.11 的图中，我们可以看到 token ID 19,846（" Berlin"）在这个选定的词汇范围内具有最高的值。概率是 0.0003，我们可以通过以下代码确认：

```python
print("Token ID 19,846 probability:", next_token_probas[:, 19846])
print("Highest probability:", max(next_token_probas.squeeze(0)))
```

请注意，虽然这个 0.0003 的值相对较小，但在该图中选定的词汇范围之外，没有比这个值更大的值。例如，使用以下代码，我们可以确认 0.0003 确实是最大的值：

```python
print("Highest probability:", max(next_token_probas.squeeze(0)))
```

我们可以将这个分数解释为模型的置信度。这意味着模型对 " Berlin" (19,846) 作为下一个 token 的置信度高于其他 token。

这个值如此之小的原因是我们使用了一个很大的温度，这使得在图中将这个值与其他 token 值并排绘制变得更容易。如果我们将温度从 5 改为 0.5，概率分数将从 0.0003 增加到 0.3398，而其他 token 的概率分数将更接近 0。

### Softmax 函数的底层原理

Softmax 函数将原始分数向量（logits）转换为概率分布，其中每个值介于 0 和 1 之间，并且所有值的总和为 1。这种转换使它们更容易解释，并便于稍后从中采样。

如果你熟悉数学符号，softmax 函数背后的公式是

$$
\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

这里，$z$ 是一个实值输入向量

$$
z = [z_1, z_2, \ldots, z_n],
$$

其中

- $n$ 是向量中元素的数量，
- $i$ 是当前元素的索引 ($1 \leq i \leq n$)，
- $j$ 是用于对所有元素求和的索引 ($1 \leq j \leq n$)，

这为每个元素 $z_i$ 产生一个归一化的概率，使得

$$
\sum_i \text{softmax}(z_i) = 1
$$

在代码中，softmax 函数可以用简单的 3 行代码实现：

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```

在实践中，我们更喜欢使用 `torch.softmax` 函数，因为它有一些额外的数值稳定性改进，可以更可靠地处理非常小和非常大的值。

将 logits 转换为这些概率分数的目的是，概率分数在某种程度上更容易解释，并且我们现在可以使用 `torch.multinomial` 函数从中采样。

例如，如果我们从这个概率分布中抽取一个样本，以我们 5.0 的温度设置，我们有 0.03% 的机会得到 " Berlin"（而以 0.5 的温度，我们有 33.98% 的机会得到 " Berlin"）。

采样过程，对应于本节开头图中的步骤 3.4，可以实现如下：

```python
torch.manual_seed(123)
print(
    "Sampled token:",
    torch.multinomial(next_token_probas.cpu(), num_samples=1)
)
```

这段代码返回 token ID 65,094，对应单词 " mistress"。请注意，这个词在我们的上下文 "The capital of Germany is" 中没有意义，它在这种情况下是随机选择的，并受到高温设置的影响，高温设置鼓励采样 " Berlin" 以外的 token。

`torch.multinomial` 函数按概率比例采样词汇索引。换句话说，概率较高的词汇索引更有可能被采样。如果我们重复采样非常多次，以温度 5 计算，我们将以 0.03% 的概率采样对应于 token " Berlin" 的词汇索引。

在查看一些额外的例子之前，请注意我们在上面指定了一个随机种子，以使本章中的代码可复现。`torch.multinomial()` 函数在 "cuda" 和 "mps" 设备上可能仍然产生不同的结果，甚至在抽取较大数量的样本时可能会崩溃（我在 PyTorch 2.9 的 "cuda" 和 "mps" 设备上都观察到了这个问题），这就是为什么我们通过 `.cpu()` 在 CPU 上运行采样。

现在让我们采样更多的下一个 token 候选，以获得更具代表性的样本。

**清单 4.7 采样多个下一个 token 候选**

```python
def count_samples(probas, num_samples=1000, threshold=1, tokenizer=None):
    samples = torch.multinomial(  #A
        probas.cpu(), num_samples=num_samples, replacement=True
    )
    counts = torch.bincount(samples.squeeze(0), minlength=1)  #B
    for i, c in enumerate(counts):
        if c > threshold:  #C
            if tokenizer is None:
                print(f"Vocab index {i}: {c.item()}x")
            else:
                print(f"'{tokenizer.decode([i])}': {c.item()}x")
```

清单 4.7 中的 `count_samples` 函数从概率分布中采样 token 索引，并统计每个 token 被抽取的频率。它使用 `torch.multinomial` 按概率随机选择 `num_samples` 个索引。`replacement=True` 设置允许我们多次抽取同一个 token。

然后，`torch.bincount` 统计每个索引出现的频率。最后，它只打印出现次数超过指定阈值的 token，以免用更频繁抽取的 token  clutter 输出。

### 多项式采样

多项式采样是根据词汇表上的概率分布选择下一个 token 的过程，就像 softmax 概率分数一样。在多项式采样中，我们不是总是选择最可能的 token（称为贪婪解码），而是随机抽取一个 token，其中概率较高的 token 更有可能被选中。这种随机性对于生成多样化的回复很重要，我们稍后会在自一致性中使用它。

为了进一步说明，我们可以将概率分数视为一组加权选择。例如，概率为 0.40 的 token 出现的可能性是概率为 0.10 的 token 的四倍，但两者仍然是可能的结果。

假设模型分配了以下概率：

- "Berlin": 0.70
- "Munich": 0.20
- "Hamburg": 0.10

贪婪解码将总是返回 "Berlin"。

多项式采样则根据这些权重抽取一个 token。在非常大量的抽取中，"Berlin" 将出现最频繁（70% 的时间），"Munich" 有时出现（20% 的时间），而 "Hamburg" 只偶尔出现（10% 的时间）。

这种可变性使我们能够在后续章节中为自一致性生成多个候选答案。

请注意，`count_samples` 函数仅用于说明。它从分布中抽取许多样本，以便你可以看到每个 token 被选中的频率。在真实的文本生成中，我们每次只抽取一个 token，但在这里抽取大量样本可以使底层概率更容易可视化和理解。

首先，让我们在我们通过应用温度 5 获得的概率分数上运行 `count_samples` 函数：

```python
torch.manual_seed(123)
count_samples(next_token_probas, tokenizer=tokenizer)
```

输出如下：

```
'}': 2x
' </': 2x
' represent': 2x
' Inf': 2x
'()*': 2x
' beside': 2x
' Kob': 2x
'�': 2x
```

正如我们所见，即使默认样本大小为 1000，没有一个采样的 token 出现超过 2 次。而且，这些都是在 "The capital of Germany is" 查询上下文中的无意义 token。这些无意义结果的原因是我们使用了一个过高的温度值。

让我们尝试一个较低的温度值 0.35，它使分数分布更尖锐，并使选择有意义的下一个 token 更有可能：

```python
torch.manual_seed(123)
probas_lowT = torch.softmax(
    scale_logits_by_temperature(next_token_logits, 0.35), dim=-1
)
count_samples(probas_lowT, tokenizer=tokenizer)
```

在这种情况下，我们看到以下输出：

```
/' __': 158x
' Berlin': 435x
' ____': 169x
' ______': 209x
' Munich': 3x
' Hamburg': 3x
' _____': 18x
```

这个输出作为我们 "The capital of Germany is" 查询的下一个 token 候选更有意义。在 1000 个样本中，token " Berlin" 被抽取了 435 次。

你可以通过 `print(probas_lowT[0, 19_846])` 检查，给定温度值 0.35，抽取这个 token 的概率大约是 42%。为了使 " Berlin" 被采样的可能性更大，我们可以进一步降低温度。

请注意，一些其他较稀有的候选也是有意义的，例如，" Munich" 和 " Hamburg" 都是德国的大城市，所以它们与查询并非完全无关。下划线回复（" ____"）可能是由于模型见过以测验占位符形式出现的文本，例如 "The capital of Germany is ____"。

你可能会想，如果 " Berlin" 是正确答案，那么通过添加这种温度重新缩放和多项式采样让模型偶尔给出错误答案有什么意义？

一般来说，对于不同类型的查询，在采样过程中引入随机性有助于模型探索替代回复，而不是总是选择单个最可能的 token。这种可变性对于创造性或开放式任务很有用，因为这些任务可能有多个有效的补全。

具体而言，在推理任务中，我们可以通过诸如自一致性（第 4.6 节）之类的技术来利用这种采样多样性，该技术生成多个候选答案并进行比较以提高答案准确性。

### 4.4.4 将温度缩放添加到文本生成函数中

在继续下一节并介绍另一个通过添加 token 概率选择过滤器来改进文本生成过程的改进之前，让我们将温度缩放修改添加到文本生成函数中，以便我们在通过模型生成新 token 时可以更方便地使用它。

**清单 4.8 带温度缩放的文本生成**

```python
from reasoning_from_scratch.qwen3 import KVCache

@torch.inference_mode()
def generate_text_temp_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out = model(token_ids, cache=cache)[:, -1]  #A
    for _ in range(max_new_tokens):
        ########################################
        # NEW:
        orig_device = token_ids.device
        if temperature is None or temperature == 1.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        else:
            logits = scale_logits_by_temperature(out, temperature)  #B
            probas = torch.softmax(logits, dim=-1)  #C
            next_token = torch.multinomial(probas.cpu(), num_samples=1)  #D
            next_token = next_token.to(orig_device)
        #########################################
        if (eos_token_id is not None
            and torch.all(next_token == eos_token_id)):
            break
        yield next_token
        out = model(next_token, cache=cache)[:, -1]
```

清单 4.8 中的 `generate_text_temp_stream_cache` 与第 2 章练习中的 `generate_text_stream_cache` 函数类似，我们在第 3 章中也使用了它。新增加的是我们现在插入了温度重新缩放和采样。新的部分位于代码中 `# New` 注释指示符下方。

输出是 `\boxed{$x = \frac{90}{7}$}`。正确答案是 83，所以模型仍然是错误的，但这更多是作为演示，表明我们可以通过使用温度缩放和采样来调整答案。

### 选择温度设置

在实践中，温度选择取决于目标。温度为 0.0 对应于贪婪解码，即我们总是选择概率最高的 token。较小的非零值，如 0.3-0.8，通常在我们想要一点更多多样性而不使输出过于不稳定时很有用。高得多的值使模型更广泛地探索，这对于创造性生成或广泛搜索可能很有用，但通常会损害我们希望得到最可能答案的任务的可靠性。

在下一节中，我们将学习如何改进采样过程。

## 4.5 用 top-p 采样平衡多样性和连贯性

在上一节中，我们看到了温度缩放和采样（通过 `torch.multinomial`）如何增加 LLM 回复的多样性，无论是好是坏。具体而言，我们看到了使用上一节描述的方法，我们最终可能会采样到与用户查询无关的"奇怪" token。

在本节中，我们通过添加 top-p 过滤器（图 4.12）来改进采样过程，使得非常低置信度的 token 不会被意外采样。本节描述的 top-p 采样过程也被称为核采样（nucleus sampling）。

![图 4.12](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F12_raschka.webp)

**图 4.12** top-p 过滤过程概览。该过滤器通过排序 token 概率、应用累积截止、选择 top-p 子集并重新归一化结果来保留概率最高的 token。

图 4.12 总结了构成 top-p 过滤器的四个步骤：排序 token 概率（4.1）、计算它们的累积和（4.2）、选择满足 top-p 截止的子集（4.3），以及重新归一化剩余概率，使它们再次形成有效的分布（4.4）。

重新归一化步骤是必要的，因为一旦我们移除了所有低概率 token，剩余的概率就不再总和为 1。为了正确采样，我们重新缩放这些剩余的值，使它们代表一个适当的概率分布。

在图 4.12 中概述了完整的 top-p 过滤流程之后，接下来的小节将详细介绍每个步骤。我们将从简要回顾温度缩放开始，然后实现图 4.12 中所示的步骤 4.1 到 4.4。

### 4.5.1 选择 top-p token 的子集

在本节中，我们实现了图 4.12 中所示的 top-p 过滤器，我们将用它来改进文本生成函数，我们计划将其用于自一致性采样。

> **注意** 本节中 top-p 过滤的目的是丢弃低概率 token，以便在采样期间只剩下最合理的选项。这降低了生成不符合上下文的 token 的几率。

在实现 top-p 过滤器之前，让我们简要回顾一下温度缩放和采样过程，使用一个更简单的玩具数据集，这使得说明过程更容易。

例如，假设模型和分词器的词汇表只有 10 个条目，而不是 151,936。

请注意，`toy_logits` 变量保存了我们在上一节中通过 `model(token_ids, cache=cache)[:, -1]` 调用获得的下一个 token logit 分数的示例值，假设词汇表大小为 10。生成的图如图 4.13 所示。

**清单 4.9 用玩具数据回顾温度缩放和采样**

```python
toy_logits = torch.tensor(  #A
    [-0.7, -3.0, 0.1, -1.2, 2.0, -1.0, -0.5, -2.0, 0.3, 1.5]
)
toy_logits_scaled = scale_logits_by_temperature(toy_logits, 1.0)  #B
toy_probas = torch.softmax(toy_logits_scaled, dim=-1)  #C
plt.bar(  #D
    torch.arange(len(toy_probas)), toy_probas,
    alpha=0.5
)
plt.ylim([0, 1])
plt.xlabel("Vocabulary index")
plt.ylabel("Probability")
plt.show()
```

![图 4.13](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F13_raschka.webp)

**图 4.13** top-p 过滤前的 token 概率示例。该分布包含许多低概率 token，稍后将通过应用累积概率阈值来截断它们。

图 4.13 中的条形图显示了重新缩放到概率分数后的下一个 token logit 分数。到目前为止，这是对上一节的回顾。接下来，我们将添加图 4.12 中所示的前两个 top-p 过滤步骤（步骤 4.1 和 4.2）。这涉及按降序排序概率分数并计算累积和。

**清单 4.10 计算累积概率和**

```python
sorted_probas, sorted_idx = torch.sort(toy_probas, descending=True)  #A
cumsum = torch.cumsum(sorted_probas, dim=-1)  #B
plt.bar(
    torch.arange(len(sorted_probas)), sorted_probas,
    alpha=0.5
)
plt.step(
    torch.arange(len(cumsum)), cumsum,
    where="mid", color="C1", label="Cumulative sum"
)
plt.ylim([0, 1])
plt.xlabel("Token rank (sorted by probability)")
plt.ylabel("Probability")
plt.show()
```

上面清单 4.10 中使用的 `torch.cumsum` 函数沿给定维度计算元素的累积和。在这个例子中，它获取排序后的 token 概率并逐步累加，因此 `torch.cumsum` 中的每个位置代表到该 token 为止累积的总概率。

例如，第一个元素等于最高概率，第二个等于前两个的总和，依此类推，直到最终值达到 1。这最好通过查看清单 4.10 生成的累积步骤图来理解，如图 4.14 所示。

![图 4.14](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F14_raschka.webp)

**图 4.14** 排序后的 token 概率及其累积和的可视化。这一步通过展示概率从高到低排序时如何累积，为 top-p 过滤做准备，有助于确定在哪里设置截止阈值。

现在我们有了累积概率和，我们可以实现核心的 top-p 过滤步骤。top-p 中的 "p" 代表概率，top-p 可以翻译为"保留累积概率保持低于或等于 p 的最小 token 集合"。一个简单的 top-p 过滤代码实现可能如下所示：

**清单 4.11 一个简单的 top-p 过滤规则**

```python
top_p = 0.8
keep_mask = cumsum <= top_p
n_kept = torch.sum(keep_mask).item()
print("Cumulative sum:", cumsum)
print("Tokens kept:", n_kept)
```

在代码中，这被实现为 `keep_mask = cumsum <= top_p`，它标记所有累积概率质量尚未超过阈值 p 的 token（这里，通过 `top_p=0.8` 定义）。然后计算并赋值保留的 token 数量给变量 `n_kept`。输出是：

```
Cumulative sum: tensor([0.4538, 0.7290, 0.8119, 0.8798,
        0.9170, 0.9475, 0.9701, 0.9886, 0.9969, 1.0000])
Tokens kept: 2
```

查看返回的累积概率，第二个条目（0.7290）刚好低于 0.8 的 top-p 阈值，第三个条目（0.8119）超过了它，这就是为什么只保留前 2 个 token。

现在，一个更常见的 top-p 过滤变体包含超过阈值的 token：

**清单 4.12 一种常见的 top-p 过滤变体**

```python
keep_mask = (cumsum - sorted_probas) < top_p
n_kept = keep_mask.sum().item()
print("Tokens kept:", n_kept)
```

这段代码现在返回 3 作为保留的 token 数量。这遵循了 top-p 过滤的定义，即保留最小的 token 集合，使得累积概率质量至少为 p。

让我们用下面的代码用一个图来说明这个 top 过滤。

**清单 4.13 可视化 top-p 过滤**

```python
plt.bar(
    torch.arange(len(sorted_probas)), sorted_probas,
    alpha=0.5, label="Sorted probabilities"
)
plt.step(
    torch.arange(len(cumsum)), cumsum, where="mid",
    color="darkorange", label="Cumulative sum"
)
#A
plt.axhline(
    top_p, color="red", linestyle="--",
    label=f"top_p = {top_p}"
)
plt.axvline(
    n_kept - 0.5, color="gray", linestyle=":",
    label=f"Top-p cutoff at {n_kept} tokens"
)
plt.xlabel("Token rank (sorted by probability)")
plt.ylabel("Probability")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0, 1.05)
plt.show()
```

执行清单 4.13 中的代码产生的图如图 4.15 所示。

![图 4.15](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F15_raschka.webp)

**图 4.15** Top-p（核）过滤。token 按概率排序，保留累积概率超过阈值（p = 0.8）的最小子集用于采样。

图 4.15 中的图显示了定义哪些 token 被保留的 `top_p = 0.8` 阈值（水平虚线）。在这种情况下，前两个 token 的累积和低于阈值，因此我们排除截止线右侧的所有其他 token（垂直虚线）。

为了实现这个阈值截止，我们可以使用以下代码，它首先将截止线右侧（图 4.15 中的垂直虚线）的所有值清零，并恢复原始排序顺序。

**清单 4.14 应用 top-p 过滤**

```python
kept_sorted = torch.where(
    keep_mask, sorted_probas,
    torch.zeros_like(sorted_probas)
)
filtered = torch.zeros_like(toy_probas).scatter(0, sorted_idx, kept_sorted)
print(filtered)
```

结果张量如下所示：

```
tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.4538, 0.0000, 0.0000, 0.0000, 0.0829,
        0.2752])
```

我们可以看到，除了索引位置 4、8 和 9 之外的值都被清零了。这意味着如果我们现在使用多项式函数，只有这三个 token 会被考虑。最后，我们重新归一化这些值，使它们再次总和为 1：

```python
denom = torch.sum(filtered).clamp_min(1e-12)
renormalized = filtered / denom
print(renormalized)
```

结果归一化后的张量是：

```
tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.5589, 0.0000, 0.0000, 0.0000,
        0.1021, 0.3390])
```

我们在本节中实现的 top-p 过滤的目标是移除低概率 token，以避免它们在后面被采样。这有助于减少给定上下文中的无意义 token 回复。

### 4.5.2 将 top-p 过滤器添加到文本生成函数中

在上一节中，我们使用一个简单的玩具示例（词汇表大小为 10）走过了 top-p 过滤步骤，以便能够在条形图中可视化该过程。在本节中，我们将四个主要的 top-p 过滤步骤（图 4.16 中的步骤 4.1-4.4）添加到现有的文本生成函数中。

![图 4.16](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F16_raschka.webp)

**图 4.16** 将 top-p 过滤与温度缩放集成。在重新缩放下一个 token 分数后，top-p 过滤在步骤 3.3 和 3.4 之间应用，以将采样限制在最可能的 token 上。

鉴于我们之前的文本生成函数，如图 4.16 所示，我们在温度缩放部分早期实现的概率转换和采样之间添加了 top-p 过滤。

为此，我们首先将上一节中的 top-p 过滤步骤组合成一个单一的、方便的函数，我们可以调用它。

**清单 4.15 Top-p 过滤函数**

```python
def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)  #A
    cumprobas = torch.cumsum(sorted_probas, dim=1)  #B
    prefix = cumprobas - sorted_probas  #C
    keep = prefix < top_p  #C
    keep[:, 0] = True  #D
    kept_sorted = torch.where(  #E
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )
    #F
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)
    #G
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom
```

为了简要回顾 top-p 过滤中发生的事情，清单 4.15 中的 `top_p_filter` 函数首先对 token 概率进行排序并计算它们的累积和。然后它保留前缀累积质量（每个 token 之前的累积概率）低于 `top_p` 阈值的 token，这包括第一个超过阈值的 token。它将其余的清零，将保留的值映射回它们的原始顺序，并重新归一化剩余的概率，使它们再次总和为 1。

在将这个 `top_p_filter` 函数添加到我们的文本生成函数之前，让我们先试一试，看看它如何与之前的温度缩放方法一起工作。首先，我们获取 logits：

```python
with torch.inference_mode():
    next_token_logits = model(input_token_ids)[:, -1]
print(next_token_logits.shape)
```

上面的代码打印了 `next_token_logits` 张量的维度 `[1, 151936]`，因为我们现在处理的是真实数据和完整的词汇表，它有 151,936 个条目。

接下来，我们将 logits 重新缩放为概率分数，并应用温度为 0.35 的温度缩放，与之前类似：

```python
torch.manual_seed(123)
probas_lowT = torch.softmax(
    scale_logits_by_temperature(next_token_logits, 0.35), dim=-1
)
count_samples(probas_lowT, threshold=1, tokenizer=tokenizer)
```

到目前为止，这段代码与我们之前在温度缩放示例中使用的类似，并打印以下采样输出：

```
' __': 158x
' Berlin': 435x
' ____': 169x
' ______': 209x
' Munich': 3x
' Hamburg': 3x
' _____': 18x
```

现在，让我们添加 top-p 过滤器，看看它如何改变结果：

```python
torch.manual_seed(123)
probas_lowT = torch.softmax(
    scale_logits_by_temperature(next_token_logits, 0.35), dim=-1
)
probas_lowT_filtered = top_p_filter(probas_lowT, top_p=0.8)
count_samples(probas_lowT_filtered, threshold=1, tokenizer=tokenizer)
```

以 0.8 的 `top_p` 阈值，这是一个典型值，我们排除了 20% 最低概率的 token，采样输出现在看起来如下：

```
' Berlin': 534x
' ____': 217x
' ______': 249x
```

正如我们所见，我们现在要么选择正确的城市（并移除 " Munich" 和 " Hamburg" 作为采样选项），要么打印下划线 token，模型可能用它来将文本格式化为带有 ' ______' 占位符的测验问题。

既然我们已经使用一个简单的玩具示例走过了 top-p 过滤过程，让我们回到我们的数学查询，并将 top-p 过滤器添加到我们之前编码的 `generate_text_temp_stream_cache` 函数中。更新后的函数，现在称为 `generate_text_top_p_stream_cache`，如下面的清单 4.16 所示。

**清单 4.16 带 top-p 过滤的文本生成**

```python
@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.,
    top_p=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out = model(token_ids, cache=cache)[:, -1]  #A
    for _ in range(max_new_tokens):
        orig_device = token_ids.device
        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        else:
            logits = scale_logits_by_temperature(out, temperature)  #B
            probas = torch.softmax(logits, dim=-1)  #C
            probas = top_p_filter(probas, top_p)  #D
            next_token = torch.multinomial(probas.cpu(), num_samples=1)  #E
            next_token = next_token.to(orig_device)
        if (eos_token_id is not None
            and torch.all(next_token == eos_token_id)):
            break
        yield next_token
        out = model(next_token, cache=cache)[:, -1]
```

为了结束本节，让我们将这个新的文本生成函数插入到 `generate_text_stream_concat_flex` 中，类似于我们之前对温度缩放版本的文本生成函数所做的：

```python
torch.manual_seed(123)
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.5,
    top_p=0.8,
)
```

输出是 " \boxed{18}"，这仍然不正确。请注意，到目前为止我们所做的一切主要是为了能够在采样不同的输出。在下一节中，我们将使用 `generate_text_stream_concat_flex` 与我们增强的文本生成函数（`generate_text_top_p_stream_cache`）在实现自一致性推理时扩展技术时采样不同的输出。

### Top-k 过滤

Top-k 过滤是另一种在采样期间限制候选下一个 token 集合的方法。

与保留累积概率低于阈值的所有 token（如 top-p）不同，top-k 仅保留基于模型 logits 的 k 个最可能的 token。

按概率对词汇表排序后，前 k 个条目之后的所有内容都被移除。

然后对剩余的 k 个 token 进行重新归一化并从中采样。

简而言之，top-k 保留固定数量的最可能 token，而 top-p 根据它们的累积质量保留可变数量的 token。

Top-k 实现起来更简单，我在我的《Build a Large Language Model (From Scratch)》一书中介绍过它。Top-p 采样最近变得更加流行。

> **练习 4.2：在 MATH-500 上使用温度缩放和 top-p 过滤**
>
> 修改第 3 章第 3.9 节中的 `evaluate_math500_stream` 函数，看看添加温度缩放和 top-p 采样是否会改变基座模型在 MATH-500 上的准确性。你可以将温度缩放和 top-p 都设置为 0.9。

## 4.6 用自一致性提高回复准确性

在之前所有关于编码温度缩放和 top-p 过滤的工作之后，我们现在准备实现本章的第二种推理时扩展技术：自一致性采样。

从概念上讲，本节将前面几章的几个部分联系在一起：我们重用第 2 章的生成代码、第 3 章的答案提取逻辑，以及本章的采样控制来构建一个完整的基于投票的推理时流程。

自一致性采样的想法在 Google Research 的论文《Self-Consistency Improves Chain of Thought Reasoning in Language Models》（https://arxiv.org/abs/2203.11171）中被正式介绍。尽管名字听起来很花哨，但它本质上是一种简单的多数投票形式，我们使用温度缩放和 top-p 过滤来生成多个答案，然后选择最频繁的那个，如图 4.17 所示。

![图 4.17](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F17_raschka.webp)

**图 4.17** 自一致性采样方法从 LLM 生成多个回复，并选择最频繁的答案，通过对这些采样回复进行多数投票来提高答案准确性。

图 4.17 所示的自一致性采样技术算作一种推理时扩展技术，因为我们不更新模型本身，并且我们花费更多的计算资源来提高回复准确性（关于准确性的更多内容稍后，在我们实现并尝试这种技术之后）。

自一致性代码实现，多亏了 `generate_text_stream_concat_flex` 函数，相对简单，主要过程可以总结为图 4.18 所示的三个主要步骤。

![图 4.18](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F18_raschka.webp)

**图 4.18** 实现自一致性采样的三个主要步骤。首先，我们使用大于零的温度和 top-p 过滤为同一提示生成多个答案。其次，我们从每个生成的解答中提取最终的框选答案。第三，我们选择最频繁提取的答案作为最终预测。

为了实现图 4.18 所示的三个步骤，在下面的清单 4.17 中，我们简单地重复调用 `generate_text_stream_concat_flex` 来生成多个答案，并重用第 3 章的 `extract_final_candidate` 函数。唯一的新代码是基于答案频率的多数投票。

**清单 4.17 带 top-p 过滤的文本生成**

```python
from reasoning_from_scratch.ch03 import extract_final_candidate
from collections import Counter

def self_consistency_vote(
    model, tokenizer, prompt, device,
    num_samples=10, temperature=0.8, top_p=0.9, max_new_tokens=2048,
    show_progress=True, show_long_answer=False, seed=None,
):
    full_answers, short_answers = [], []
    for i in range(num_samples):  #A
        if seed is not None:
            torch.manual_seed(seed + i + 1)
        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )
        short = extract_final_candidate(  #B
            answer, fallback="number_then_full"  #B
        )  #B
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)
    mc = counts.most_common()  #C
    if not mc:
        majority_winners, final_answer = [], None
    else:
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

简而言之，上面清单 4.17 中的自一致性方法：

1. 用大于 0 的温度和 top-p 采样多个答案
2. 从每个答案中提取最终的框选答案
3. 选择最频繁的最终答案

请注意，在我们的实现中，我们使用 for 循环顺序生成这些答案。在实践中，也常见在不同的设备上生成答案，以便采样可以并行化。

此外，在上面的代码中，我们为每一轮单独设置随机种子：

```python
if seed is not None:
    torch.manual_seed(seed + i + 1)
```

从技术上讲，这不是必需的，设置一次随机种子就足以生成多样化的样本。这种显式种子设置在我们想要单独重新运行某些轮次时很有用。

让我们在实践中尝试这个函数，看看我们得到什么：

```python
results = self_consistency_vote(
    model,
    tokenizer,
    prompt,
    device=device,
    num_samples=5,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    seed=123,
    show_progress=True,
)
```

打印的结果如下所示：

```
[Sample 1/5] → '83'
[Sample 2/5] → '22'
[Sample 3/5] → '54'
[Sample 4/5] → '83'
[Sample 5/5] → '61'
```

最终答案，由于 83 是最频繁的答案，是 83（我们可以通过 `results["final_answer"]` 以编程方式访问这个数字）。如果你想阅读解释和完整答案，你可以访问返回 83 的正确解答之一，例如 `print(results["full_answers"][0])`，它打印：

```
To find the value of x, let's solve the equation step by step.
1. **Given Equation:**
\[
\frac{1}{2} \times (3x - 9) = x + 37
\]
# ...
**Final Answer:**
\[
\boxed{83}
\]
```

最后，通过这种自一致性方法，我们能够生成正确答案。（请注意，在 "mps" 或 "cuda" 设备上执行代码时，结果可能会有所不同。）

`self_consistency_vote` 函数目前也不处理平局情况，所以如果多个样本具有相同的频率，它会返回 `None` 作为最终答案。我们将在下一章实现一种评分方法，用于计算给定答案的置信度，这可以用作平局决胜机制。

> **练习 4.3：在 MATH-500 上使用自一致性采样**
>
> 修改第 3 章第 3.9 节中的 `evaluate_math500_stream` 函数，以评估自一致性采样是否能提高基座模型在 MATH-500 上的准确性。使用样本大小为 3，并将温度和 top-p 值都设置为 0.9。
>
> 作为本练习的一部分，实现平局决胜，使得平局通过样本列表中首先出现的答案来解决。例如，如果样本答案是 13, 15, 13, 15, 16，则选择的答案应该是 13。
>
> 请注意，你不必修改 `self_consistency_vote` 本身来实现这种平局决胜，但你可以使用函数返回的 `results` 字典来实现这个简单的平局决胜规则。

> **练习 4.4：自一致性采样中的提前停止**
>
> 为了提高计算效率，实现自一致性的提前停止版本，一旦超过一半的答案达成一致，就结束采样。

### 选择温度和 top-p 设置

既然我们已经在本章前面讨论过温度，这里的实际问题是对于自一致性采样应该引入多少随机性。目标不是让模型尽可能随机，而是生成足够多的多样化答案，以便多数投票有所帮助，同时仍然保持大多数样本是合理的。

在实践中，温度值大约在 0.5 到 0.9 之间，top-p 值大约在 0.7 到 0.9 之间是合理的起点。

作为经验法则，如果所有（长）答案看起来几乎相同，那么稍微提高温度或 top-p 值以鼓励更多多样性是一个好主意。如果（长）答案开始变得偏离或无意义，那么设置可能过于激进，应该降低，通常先降低温度。这里，"长"答案指的是在提取最终框选结果之前的完整答案。你可以在 `self_consistency_vote` 函数返回的结果字典中检查长答案，或者使用 `show_long_answer=True` 设置运行函数。

你可能还记得，描述这种技术的上述论文标题是《Self-Consistency Improves Chain of Thought Reasoning in Language Models》。那么，"思维链推理"这个方面在哪里或如何融入这一切呢？

这里的思维链推理简单地指的是我们在本章前面介绍的思维链提示，当时我们通过 "\n\nExplain step by step." 修改了提示，使 LLM 生成更长的回复。

所以，让我们将思维链提示与自一致性采样结合起来：

```python
results = self_consistency_vote(
    model,
    tokenizer,
    prompt + "\n\nExplain step by step.",
    device=device,
    num_samples=5,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    seed=123,
    show_progress=True,
)
```

在这种情况下，所有 5 个答案都是 83（正确），很可能因为当使用思维链时，这个问题对 LLM 来说相对简单。

相反，更有趣的是看看模型在上一章的整个 MATH-500 数据集上的表现如何。各种实验的结果如表 4.1 所示。

**表 4.1 不同方法在 MATH-500 任务上的准确率**

| | 方法 | 模型 | 准确率 | 时间 |
|---|------|------|--------|------|
| 1 | 基线（第 3 章），贪婪解码 | Base | 15.2% | 10.1 min |
| 2 | 基线（第 3 章），贪婪解码 | Reasoning | 48.2% | 182.1 min |
| 3 | 思维链提示（"CoT"） | Base | 40.6% | 84.5 min |
| 4 | 温度和 top-p（"Top-p"） | Base | 17.8% | 30.7 min |
| 5 | "Top-p" + 自一致性（n=3） | Base | 29.6% | 97.6 min |
| 6 | "Top-p" + 自一致性（n=5） | Base | 27.8% | 116.8 min |
| 7 | "Top-p" + 自一致性（n=10） | Base | 31.6% | 300.4 min |
| 8 | "Top-p" + "CoT" | Base | 33.4% | 129.2 min |
| 9 | 自一致性（n=3）+ "Top-p" + "CoT" | Base | 42.2% | 211.6 min |
| 10 | 自一致性（n=5）+ "Top-p" + "CoT" | Base | 48.0% | 452.9 min |
| 11 | 自一致性（n=10）+ "Top-p" + "CoT" | Base | 52.0% | 862.6 min |
| 12 | 自一致性（n=3）+ "Top-p" + "CoT" | Reasoning | 55.2% | 544.4 min |

为简洁起见，表 4.1 中标记为 "Top-p" 的方法同时使用了温度缩放和 top-p 采样。

表 4.1 中显示的准确率值是使用 "cuda" GPU（DGX Spark）在 MATH-500 测试集的所有 500 个样本上计算的。让我们逐一查看结果。n=3 的缩写意味着我们在自一致性采样中使用了样本大小为 3。

第 1 行和第 2 行显示了使用第 3 章代码的基座模型和推理变体的结果，意味着我们使用没有温度缩放或 top-p 过滤的文本生成函数。这个文本函数只是在每一步选择分数最高的 token，这也被称为贪婪解码。我们可以看到，推理变体的准确率大约是基座模型的 3 倍，但运行时间也大幅增加，因为它生成了更多的 token。

接下来，在第 3 行，我们看到了通过 "\n\nExplain step by step." 提示修改的思维链提示方法的结果。正如我们所见，这将基座模型的准确率从大约 15% 提升到 40%。

第 4 行显示了当我们向基座模型添加温度缩放和 top-p 采样时会发生什么。（表 4.1 中所有涉及温度缩放和 top-p 采样的实验都将两者都设置为 0.9。）正如我们所见，与第 1 行的基座模型相比，准确率仅从 15.2% 适度提升到 17.8%。这是预期的，因为温度缩放和 top-p 缩放 merely 帮助我们控制采样多样性，但它们本身不是推理时扩展技术。我们还可以看到运行时间从 10.1 分钟增加到 30.7 分钟。这不是由于采样代码开销，而是因为模型在某些情况下现在生成了更长的回复。

第 5-7 行显示了自一致性扩展的结果。我们可以看到，将样本数量从 3 增加到 10 进一步将准确率提高到 31.6%，但它也显著增加了运行时间。在这种情况下，使用 10 个样本而不是 3 个样本几乎没有准确率优势。

第 8 行显示了温度缩放和 top-p 采样与思维链提示相结合。在这种情况下，采样使思维链的结果变差了（33.4% 对比第 3 行的 40.6%）。当将思维链提示与自一致性结合时，我们可以看到准确率以样本大小为 10 提高到 52%，但这也大幅增加了运行时间，达到惊人的 862.9 分钟。

请注意，在实践中，如果你有多个 GPU 的访问权限，自一致性采样中的不同样本可以并行计算而不是顺序计算。这仍然会消耗相同数量的计算，但可以分布和并行化以更快地生成结果。

最后，在第 12 行，我们可以看到推理变体也能从自一致性采样中受益，将推理变体（第 2 行）的性能从 48.2% 提高到 55.2% 的准确率。同样，这也以增加运行时间为代价。

结论是，表 4.1 中的结果很好地突出了推理时扩展中的权衡，即我们用更好的准确性换取更多的计算。

请注意，本章中自一致性采样方法的一个主要缺点是它依赖于我们可以提取用于多数投票的最终框选答案。这种方法对于没有数字或简短最终答案的问题来说更难应用。

在下一章中，如图 4.19 所示，我们将实现一种不同且更通用的推理时扩展方法，称为自我优化（self-refinement），其中模型迭代地改进自己的答案。

![图 4.19](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F19_raschka.webp)

**图 4.19** 本章聚焦于推理时技术的总结。这里，文本生成函数通过基于投票的方法来扩展以提高答案准确性。下一章将介绍自我优化，其中模型迭代地改进其回复。

## 4.7 总结

- 推理能力和答案准确性可以通过在推理时增加计算量（推理时扩展）来在不重新训练模型的情况下得到提升。
- 本章聚焦于两种这样的技术：思维链提示和自一致性；第三种方法，自我优化，已简要描述，将在下一章中介绍。
- 一个灵活的文本生成包装器（`generate_text_stream_concat_flex`），可以使用不同的采样策略，这些策略可以插入而无需更改周围的代码。
- 下一个 token 是通过 softmax 从 logits 生成的。
- 温度缩放改变 logits 以控制生成文本的多样性。
- Top-p（核）采样过滤掉低概率 token，以减少生成无意义答案的几率。
- 思维链提示（如 "Explain step by step." 或类似）通常通过鼓励模型写出中间推理过程来产生更准确的答案，尽管它增加了生成的 token 数量，从而增加了运行时间成本。
- 自一致性采样生成多个答案，从每个答案中提取最终的框选结果，并通过多数投票选择最频繁的答案以提高答案准确性。
- 在 MATH-500 数据集上的实验表明，与没有采样的基线相比，将思维链提示与自一致性结合可以显著提高准确率，但代价是更长的运行时间。
- 推理时扩展的核心权衡：以更多的计算换取更高的准确率。
