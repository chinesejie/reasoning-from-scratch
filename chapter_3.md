# 3 评估推理模型

本章涵盖以下内容：
- 从 LLM 响应中可靠地提取最终答案
- 通过符号数学求解器将 LLM 的输出与参考答案进行比较，从而验证答案的正确性
- 运行完整的评估流程：加载预训练模型、生成输出，并根据数据集进行评分

评估让我们能够区分那些听起来令人信服但实际上无法正确解决问题的 LLM。LLM 评估技术涵盖广泛的方法，从衡量任务准确性到确保 LLM 符合特定的安全标准。在本章中，"评估"指的是在大量示例上定量测试模型，并将其输出与参考答案进行评分。

更具体地说，我们专注于实现一种基于验证的方法，通过类似计算器的实现来检查 LLM 是否能准确解决数学问题，并将其答案与参考答案进行比较。这种验证器特别有用，因为它不仅可以评估数学任务上的表现，还引入了可验证奖励（verifiable rewards）的原则，这是我们在第 6 章将要实现的推理模型强化学习方法的基础。（感兴趣的读者可以在附录 F 中找到额外的评估方法。）

---

![图 3.1 本书所涵盖主题的心智模型。本章介绍评估方法（阶段 2），特别侧重于实现验证器，稍后我们将在第 6 章中将其作为可验证奖励的基础重复使用。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F01_raschka.webp?1)

## 3.1 构建数学验证器

在实践中，评估训练好的 LLM 有四种常见方法：多项选择、验证器、排行榜和 LLM 评判，如图 3.1 所示。这些方法广泛应用于研究论文、技术报告、营销材料和模型卡片（总结模型如何训练、评估及预期使用的文档）中，结果通常来自不止一个类别。

如图 3.1 所示，这些评估方法可以分为两种更广泛的类型：基于基准的评估和基于判断的评估。粗略地说，前者通常更定量，而后者更依赖于定性判断。所有四种评估方法在不同情境下都很有用，但验证器对推理模型尤其相关。

数学问题提供了一个自然的例子：根据问题的复杂程度，数学问题受益于逐步推理来解决，但评估却很直接，因为最终答案可以与正确答案进行核对。在这种情况下，验证器方法提供了一种简单而可靠的方式来衡量模型的推理步骤是否导致了正确的结果。

在本章中，我们专注于将验证器作为基于基准的方法来衡量数学问题中答案的正确性，如图 3.2 所示。

---

![图 3.2 在自由形式问答中使用基于验证的方法评估 LLM。模型生成自由形式的答案（可能包含多个步骤）和一个最终的方框答案，该答案被提取出来并与数据集中的正确答案进行比较。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F02_raschka.webp?1)

验证器将提取的答案与参考答案进行比较，如图 3.2 所示，通常依赖于代码解释器或计算器程序等外部工具。

虽然我们目前的重点是评估，但验证器将在本书后面再次出现。它们不仅作为衡量性能的方式，还提供了用于训练推理模型的强化学习方法中的反馈信号，我们将在第 6 章中探讨。

缺点是，验证器方法只能应用于可以轻松（最好是确定性地）验证的领域，例如数学和代码。此外，这种方法可能会引入额外的复杂性和依赖性，并可能将部分评估负担从模型本身转移到外部工具上。

由于数学问题求解可以通过编程生成无限变化，并且受益于逐步推理，这项任务已成为推理模型评估和开发的基石。

在本章的剩余部分，我们将按照图 3.3 所示的 8 个步骤逐步构建一个数学验证器。

---

![图 3.3 构建和应用数学验证器的逐步工作流程。从预训练的 LLM 开始，我们生成答案、提取并规范化它们，然后将其与真实值（ground-truth）解决方案进行比较。经验证的答案随后被评分，并在数据集（MATH-500）上重复该过程以评估整体模型性能。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F03_raschka.webp?1)

下一节将从图 3.3 所示的步骤 1 和 2 开始，即加载上一章介绍的预训练 LLM 并设置其生成答案。

## 3.2 加载预训练模型以生成文本

在本节中，我们通过遵循图 3.3 工作流程中的步骤 1 和 2 开始实现验证器。具体来说，我们将加载上一章介绍的预训练 LLM 并配置其生成答案。这为后续步骤奠定了基础，在这些步骤中，我们将提取、规范化和验证这些答案。

它还为后续章节建立了一个可复用的模型加载路径，在这些章节中，我们将比较不同的推理方法，并需要一种一致的方式来衡量它们是否真正改进了模型。

具体来说，我们使用与第 2 章相同的预训练基础模型。为了方便起见，也为了在将来的章节中复用，我们将模型加载逻辑包装在一个 `load_model_and_tokenizer` 函数调用中，如清单 3.1 所示：

---

**清单 3.1 加载预训练模型**

```python
from pathlib import Path
import torch
from reasoning_from_scratch.ch02 import (
    get_device
)
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B
)

def load_model_and_tokenizer(
    which_model, device, use_compile, local_dir="qwen3"
):
    if which_model == "base":
        download_qwen3_small(
            kind="base", tokenizer_only=False, out_dir=local_dir
        )
        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    elif which_model == "reasoning":
        download_qwen3_small(
            kind="reasoning", tokenizer_only=False, out_dir=local_dir
        )
        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )
    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    if use_compile:  #A
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)
    return model, tokenizer

WHICH_MODEL = "base"   #B
device = get_device()
# device = torch.device("cpu")  #C
model, tokenizer = load_model_and_tokenizer(
    which_model=WHICH_MODEL,
    device=device,
    use_compile=False
)
```

- #A 可选设置为 true 以启用模型编译
- #B 默认使用基础模型，类似于第 2 章
- #C 如果您的设备存在兼容性问题，请取消注释此行

默认情况下，清单 3.1 加载基础模型，正如第 2 章一样。一个可选的变体是官方 Qwen3 推理模型，这是 Qwen3 团队使用推理特定方法在基础模型之上训练的。这不是我们稍后在本书中将自己构建的自定义推理模型。相反，我们在这里将其作为一个参考点。可以通过在清单 3.1 中设置 `WHICH_MODEL = "reasoning"` 来加载它，以便我们稍后可以将其评估结果与基础模型的结果进行比较。

现在我们已经加载了模型，我们可以使用第 2 章的文本生成函数来生成文本，如清单 3.2 所示。

---

**清单 3.2 生成模型输出**

```python
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache
)

prompt = (  #A
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)
input_token_ids_tensor = torch.tensor(  #B
    tokenizer.encode(prompt),
    device=device
).unsqueeze(0)  #C

all_token_ids = []
for token in generate_text_basic_stream_cache(  #D
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0)  #E
    decoded_id = tokenizer.decode(token_id.tolist())
    print(  #F
        decoded_id,   
        end="",
        flush=True
    )
    all_token_ids.append(token_id)

all_tokens = tokenizer.decode(all_token_ids)  #G
```

- #A 将数学问题定义为字符串提示
- #B 将提示转换为模型可以处理的 token ID
- #C 添加批次维度
- #D 从模型逐个生成输出 token
- #E 移除批次维度
- #F 打印生成的 token
- #G 将完整生成的序列解码为文本

在清单 3.2 中，我们首先将一个简单的数学问题编码为模型可以处理的 token ID。然后，模型以流式方式逐个生成 token，我们立即打印它们，以便在生成过程中阅读输出。同时，我们将生成的 token 收集到一个列表中，以便稍后将它们解码为完整的最终答案字符串。这种同时流式传输和收集 token 的模式很方便，因为它让我们在监控生成过程的同时，仍然存储了可以稍后处理的完整答案文本（`all_tokens`）。

清单 3.2 中代码生成的响应如下：

```
To find the value of \( a^2 + b^2 \) given that \( a + b = 3 \) 
and \( ab = \frac{13}{6} \), we can use the following algebraic identity:
\[
a^2 + b^2 = (a + b)^2 - 2ab
\]
**Step 1:** Substitute the given values into the equation.
\[
a^2 + b^2 = (3)^2 - 2 \left( \frac{13}{6} \right)
\]
[...]  #A
**Final Answer:**
\[
\boxed{\dfrac{14}{3}}
\]
```

- #A 为简洁起见已缩短

正如我们所见，基于这个答案，即使它是一个基础模型，它也提供了类似推理模型的解释。这可能是因为 Qwen3 团队在技术报告中指出，他们在预训练阶段包含了思维链（chain-of-thought）数据。即使该模型具有一些类似推理模型的行为，添加额外的推理方法可以进一步提高这些能力。（请注意，响应可能会因您在 CPU、CUDA 或 MPS 设备上执行代码而有所不同。）

此外，如果您不熟悉数学中常用的 LaTeX 语法，上面的响应可能非常难以解读。如果是这种情况，您可以使用 IPython 的 `Latex` 类来渲染它，如下所示：

```python
from IPython.display import Latex, display
display(Latex(all_tokens))
```

在代码笔记本中执行上述代码将渲染响应，如图 3.4 所示。

---

![图 3.4 渲染后的响应，包含逐步计算和最终的方框答案。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F04_raschka.webp?1)

请注意，图 3.4 中给出的最终答案确实是这个问题的正确答案。

## 3.3 实现文本生成的包装器

在上一节中，我们加载了预训练的 LLM 并设置了文本生成功能（如图 3.5 所示），这是本章剩余部分所涵盖的评估过程的前两个步骤。

---

![图 3.5 验证器工作流程中步骤 1 和 2 的示意图。预训练的 LLM 被加载并用数学问题提示，产生原始 LaTeX 语法的输出。答案也以渲染后更易读的形式显示。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F05_raschka.webp?1)

为了在后续章节中更加方便，我们为文本生成函数创建了一个包装器（清单 3.3），这样我们只需要传入模型、分词器和提示，以及一些额外的设置，而不必每次都重复 tokenization 和输入准备步骤：

---

**清单 3.3 流式文本生成的包装器**

```python
def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
):
    input_ids = torch.tensor(                    #A
        tokenizer.encode(prompt), device=device  
    ).unsqueeze(0)                           
    generated_ids = []
    for token in generate_text_basic_stream_cache(  #B
        model=model,                                
        token_ids=input_ids,                        
        max_new_tokens=max_new_tokens,              
        eos_token_id=tokenizer.eos_token_id,        
    ):                                              
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose: #C
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)  #D
```

- #A 将提示文本编码为 token ID 并放置在设备上
- #B 使用缓存生成逐个流式传输 token
- #C 可选地在生成时打印 token
- #D 将所有生成的 ID 解码为最终文本字符串

清单 3.3 中的这个包装器函数处理文本生成的完整周期：它对输入提示进行 tokenization，从模型流式传输新 token，然后将结果解码为最终字符串。而且，如前所述，可选的 `verbose` 标志允许我们实时查看正在生成的 token。该函数可以如下使用：

```python
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt, device,
    max_new_tokens=2048,
    verbose=True  #A
)
```

- #A 使用 False 将抑制逐 token 的实时打印

这将打印与 3.2 节中完全相同的响应：

```
[...]  #A
**Final Answer:**
\[
\boxed{\dfrac{14}{3}}
\]
```

- #A 为简洁起见已缩短

## 3.4 提取最终答案方框

现在模型已加载并准备就绪，我们可以进入本章特有的有趣部分：评估模型。

在开始之前，值得回顾一下，本书采用从零开始的方法，这自然包括一些详细、有时繁琐的步骤。这是有意为之，因为我们想自己构建评估流程以更好地理解其工作原理，而不是仅仅调用预定义的包装函数和构建块。

在上一节中，我们看到模型在答案方框中返回了最终答案（在原始文本中写作 `r"\boxed{\dfrac{14}{3}}"`）。我们尚未明确强制该格式，但模型可能产生它，因为方框答案是数学基准和训练数据中的常见约定。这种行为不一定证明对我们稍后将要运行的评估任务过拟合。但它反映了预训练模型通常在网上遇到许多问题格式，并学会复制这些风格约定。

同样，作为一般规则，可以公平地假设模型训练时互联网上可用的任何信息都已成为训练数据的一部分。

尽管这里没有必要，但当我们稍后在 MATH-500 数据集上评估模型时，我们将添加一个特定提示，指示模型以这种方框形式返回答案，因为这是一种常见约定，可以使不同模型之间的评估更加一致，并使数据提取更容易。

虽然这个步骤是机械性的，但每当我们需要将模型响应转换为可以大规模自动评分的内容时，它就会不断出现。我们现在将编写代码来执行此答案方框内容的提取，如图 3.6 所示。

---

![图 3.6 如何从 LLM 输出中提取方框结果的示意图。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F06_raschka.webp?1)

具体来说，本节实现了图 3.6 中的步骤 3。下一节将实现步骤 4 的规范化方法。

由于您的模型可能会产生与上面显示的略有不同的响应（取决于您的硬件），我们将暂时使用一个硬编码的答案（假装这个答案是由模型生成的）。在后面的章节中，我们将重新访问模型，并让它为 MATH-500 数据集中的任务生成答案。

> **注意** 我们使用的是原始字符串（`r"""..."""` 而不是常规字符串 `"""..."""`）。原始字符串使处理 `\` 字符更容易，否则它们将被视为转义序列，需要每个反斜杠加倍。

接下来，让我们在清单 3.4 中定义一个函数，以从 `model_answer` 中提取答案方框：

```python
model_answer = (
    r"""... some explanation...
    **Final Answer:**
    \[                     #A
    \boxed{\dfrac{14}{3}}  #A
    \]                     #A
    """)
```

- #A 我们要提取的答案方框

---

**清单 3.4 提取答案方框**

```python
def get_last_boxed(text):
    boxed_start_idx = text.rfind(r"\boxed")  #A
    if boxed_start_idx == -1:
        return None
    current_idx = boxed_start_idx + len(r"\boxed")  #B
    #C
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1
    #D
    if current_idx >= len(text) or text[current_idx] != "{":
        return None
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx
    #E
    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1
    if brace_depth != 0:  #F
        return None
    return text[content_start_idx:current_idx-1]  #G
```

- #A 查找 `"\boxed"` 的最后一次出现
- #B 获取 `"\boxed"` 之后的位置
- #C 跳过 `"\boxed"` 后的任何空白字符
- #D 期望一个左大括号 `{`
- #E 解析带嵌套的大括号
- #F 处理不平衡的大括号
- #G 提取最外层大括号内的内容

清单 3.4 中的 `get_last_boxed` 辅助工具函数从模型输出中提取最后一个 `\boxed{...}` 表达式的内容。更具体地说，它扫描最后的 `\boxed`，跳过空白字符，检查大括号，并处理任何嵌套，以便我们捕获预期的答案字符串。

虽然它看起来有点繁琐，但当我们对 MATH-500 等数据集运行评估时，拥有这个解析器将会得到回报，因为在这些数据集中，提取正确的最终答案是衡量模型推理能力的第一步。（MATH-500 是一个包含 500 个问题的精选集合，广泛用作推理模型基准数据集，我们将在本章稍后使用。）

现在，让我们在模型答案上测试它：

```python
extracted_answer = get_last_boxed(model_answer)
print(extracted_answer)
```

此函数调用的输出是 `"\dfrac{14}{3}"`，这正是我们想要提取的方框答案。

### 渲染数学公式

我们可以通过前面介绍的 `Latex` 类来渲染数学公式。或者，对于不附带答案文本的单个数学公式，我们也可以使用更简单的 `Math` 类：

```python
from IPython.display import Math
display(Math(r"\dfrac{14}{3}"))
```

这将分数渲染为

$$\dfrac{14}{3}$$

虽然前面的 `get_last_boxed` 函数正确提取了文本，但我们将通过清单 3.5 中的 `extract_final_candidate` 函数使答案提取更加健壮，以应对最终答案方框缺失或不完整的情况：

---

**清单 3.5 提取最终答案候选**

```python
import re

RE_NUMBER = re.compile(  #A
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

def extract_final_candidate(text, fallback="number_then_full"):
    result = ""  #B
    if text:  #C
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")
        #D
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                result = m[-1]  #E
            elif fallback == "number_then_full":
                result = text  #F
    return result
```

- #A 用于从文本中提取数值的正则表达式
- #B 如果没有任何匹配项，则返回默认值
- #C 优先选择最后的方框表达式（如果存在）
- #D 如果没有方框表达式，尝试回退
- #E 使用最后一个数字
- #F 如果没有找到数字，则返回完整文本

清单 3.5 中的 `extract_final_candidate` 函数在找不到方框答案时提供回退设置，具体如下：
- `"number_then_full"`（默认）：选择最后一个简单数字，否则选择整个文本；
- `"number_only"`：选择最后一个简单数字，否则返回空字符串 `""`；
- `"none"`：仅提取方框内容，否则返回空字符串 `""`。

对于回退设置，清单 3.5 中的代码通过 Python 的 `re` 库使用正则表达式（简称 regex）。正则表达式是一种在文本中搜索模式的方法。在我们的例子中，正则表达式模式旨在识别数字，包括分数、小数和科学记数法。虽然正则表达式语法看起来令人生畏，但您无需担心这里的精确语法。重要的是，这为我们提供了一个可靠的工具，可以在没有方框答案时从模型输出中提取最后一个数字候选。

让我们在模型答案上尝试它：

```python
print(extract_final_candidate(model_answer))
```

这正确地返回 `"\dfrac{14}{3}"`。接下来，让我们尝试一些额外的示例。首先，另一个方框候选：

```python
print(extract_final_candidate(r"\boxed{ 14/3. }"))
```

这正确地返回 `"14/3."`，去除了额外的空白字符但保留了标点符号。标点字符将在我们稍后实现的相等性检查中被正确处理。

接下来，让我们尝试一个没有方框的候选，它应该触发回退设置，看看会发生什么：

```python
print(extract_final_candidate("abc < > 14/3 abc"))
```

由于默认的回退设置，它会在答案中找到最后一个数字，并正确地返回 `"14/3"`。

在本节中，我们定义了从 LLM 答案文本上下文中提取答案的实用函数。这使我们离实现验证此答案是否确实正确的总体目标更近了一步。在下一节中，我们将在实现检查功能之前，将响应规范化为更通用的标准形式。

> **为什么不用 LLM 进行答案提取？**
>
> 我们可以使用 LLM 本身来提取方框答案。但这会引入不必要的复杂性和潜在错误。假设 LLM 以特定格式输出答案，提取是一个简单的机械任务：我们只需要定位最后一个方框表达式，或者如果缺失，则回退到一个数字或原始文本。
>
> 正则表达式起初可能看起来复杂，但最终，我们得到了一个小的、可复用的实用函数，执行起来成本低廉，并且确定性和可复现地处理提取，而不依赖于另一个模型输出的可变性。

## 3.5 规范化提取的答案

之前，我们从模型的响应中提取了方框答案 `"\dfrac{14}{3}"`。模型可能以多种方式打印相同的值，例如 `"\frac{14}{3}"`、`"14/3"`、`"$14/3$"` 或 `"(14)/(3)"`。为了实现和使用一个可以检查答案是否正确的健壮检查系统，我们首先需要一个一致的结果比较方法。

在本节中，我们实现一个规范化（或标准化）过程（图 3.7 中的步骤 4），该过程去除格式并标准化答案。

---

![图 3.7 如何从 LLM 输出中提取方框结果并转换为规范纯文本形式的示意图。这个规范化的答案随后用于与正确答案进行验证。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F07_raschka.webp?1)

图 3.7 所示的规范化步骤通过清单 3.6 中的 `normalize_text` 函数实现。

---

**清单 3.6 规范化提取的答案**

```python
LATEX_FIXES = [  #A
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"·|×", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  #B

SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",  #C
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",    #C
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",              #C
}

def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()
    #D
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)  #D
    text = re.sub(r"\^\s*\\circ", "", text)            #E
    text = text.replace("°", "")                       #E
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)  #F
    if match:
        text = match.group("x")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)  #G
    for pat, rep in LATEX_FIXES:  #H
        text = re.sub(pat, rep, text)
    def convert_superscripts(s, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"
    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)
    #I
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    #J
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    #K
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )
    #L
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )
    return text.replace("{", "").replace("}", "").strip().lower()
```

- #A 要替换的 LaTeX 格式（左：原始值，右：新值）
- #B 去除聊天特殊 token，如 `"<|assistant|>"`
- #C 将 Unicode 上标转换为纯文本上标的字典映射
- #D 去除前导多项选择标签（例如，像 `"c. 3"` -> `3`）
- #E 去除角度度标记
- #F 如果整个字符串被包裹，则解包 `"\text{...}"`
- #G 去除行内/显示数学包装器：`\( \)` `\[ \]`
- #H LaTeX 规范化
- #I 规范化数字和根式表达式
- #J 将 LaTeX 分数转换为除法形式
- #K 处理指数和带分数
- #L 去除数字中的千位分隔符

清单 3.6 中的 `normalize_text` 函数获取一个提取的答案字符串，并将其重写为可以可靠地与参考答案进行比较的标准化格式。它首先去除特殊 token 和不必要的 LaTeX 杂乱内容，例如 `\left`、`\right` 或度符号。然后它解包像 `\text{...}` 这样的案例，去除行内数学标记符，并将常见结构重写为计算器样式形式。例如，它将 `\sqrt{a}` 转换为 `sqrt(a)`，将 `\frac{a}{b}` 转换为 `(a)/(b)`。最后，它规范化指数、带分数和千位分隔符，并清理大括号和大小写。简而言之，该函数将不同格式的 LaTeX 输出转换为干净、标准化的字符串表示。

现在让我们在模型答案上尝试 `normalize_text` 函数：

```python
print(normalize_text(extract_final_candidate(model_answer)))
```

结果，它不再使用 LaTeX 格式打印答案（`r"\dfrac{14}{3}"`），而是以标准化的、无 LaTeX 的形式返回答案：

```
(14)/(3)
```

接下来，让我们尝试一个不同格式的答案：

```python
print(normalize_text(r"\text{\[\frac{14}{3}\]}"))
```

这也按预期返回 `"(14)/(3)"`。

我们现在有一种健壮的方法从 LLM 的响应中提取答案文本。下一节将涵盖的下一个任务是实现一个函数，将 LLM 答案与正确的参考答案进行比较。

## 3.6 验证数学等价性

到目前为止，在本章中，我们实现了要求 LLM 生成答案、提取相关部分并规范化的步骤。下一步，如图 3.8 所示，是将提取的答案与正确的参考答案进行比较，这在技术上下文中被称为**真实值（ground truth）**。

这是我们在仔细构建验证器的主要原因之一：在第 6 章中，相同的基本思想将作为可验证的奖励信号重新出现，在后面的章节中，我们将使用它来检查我们的训练更改是否改进了模型。

---

![图 3.8 如何检查 LLM 生成的答案与正确参考答案（真实值）的示意图。最终方框答案被提取并规范化，然后与数据集中提供的正确答案进行比较。如果两者匹配，则响应被评为正确。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F08_raschka.webp?1)

请注意，如果我们想实现图 3.8 所示的相等性检查，仅使用 Python 的 `==` 运算符进行直接比较是不够的，因为像 `"14/3"` 和 `"(14)/(3)"` 这样的表达式不会匹配，而等价但未规范化的分数如 `"(28)/(6)"` 和 `"(14)/(3)"` 也会被视为不相等。

作为我们相等性检查的一部分，我们实现了一个额外的中间步骤：使用符号数学引擎解析提取和规范的答案。

为此，我们使用 SymPy 开源数学库（https://sympy.org），该库已经开发和测试了二十年，并已成为 Python 科学计算的主要工具。解析函数在清单 3.7 中实现。

> **注意** 如果您尚未安装第 2 章中的依赖项，可以通过 `uv pip install sympy`（或 `uv add sympy`）手动安装 SymPy。

---

**清单 3.7 用于数学相等性检查的 SymPy 解析器**

```python
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError

def sympy_parser(expr):
    if expr is None or len(expr) > 2000: #A
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,  #B
                #C
                spp.implicit_multiplication_application,
            ),
            evaluate=True,  #D
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None
```

- #A 避免在过长垃圾响应上崩溃
- #B 标准转换，如处理括号
- #C 允许省略乘号（例如，`2y` -> `2*y`）
- #D 在解析期间进行求值，使简单常数简化（例如，`2+3` -> `5`）

清单 3.7 中的 `sympy_parser` 函数获取一个输入表达式，例如我们从 LLM 响应中提取的规范化答案，并将其转换为可以可靠地进行数学等价性比较的 SymPy 对象。为此，它应用 SymPy 的标准解析规则，支持隐式乘法（如 `2y` 而不是 `2*y`），并简化基本算术（因此 `2+3` 变为 `5`）。

> **注意** `sympy_parser` 考虑到了看似过多的错误情况，但这些都是我在评估模型在所有 500 个 MATH-500 问题上时遇到的错误，因为模型并不总是生成完美格式化的输出。

让我们看看它的实际应用，并将其应用于规范化的答案候选：

```python
print(sympy_parser(normalize_text(
    extract_final_candidate(model_answer)
)))
```

这返回分数 `14/3`。接下来，让我们尝试一个未规范化的分数：

```python
print(sympy_parser("28/6"))
```

同样，这也返回 `14/3`。

使用 `sympy_parser`，我们现在可以在清单 3.8 中实现相等性检查函数：

---

**清单 3.8 使用 SymPy 的相等性检查函数**

```python
from sympy import simplify

def equality_check(expr_gtruth, expr_pred):
    if expr_gtruth == expr_pred:  #A
        return True
    #B
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)
    if gtruth is not None and pred is not None:  #C
        try:
            return simplify(gtruth - pred) == 0  #D
        except (SympifyError, TypeError):
            pass
    return False
```

- #A 首先，检查两个表达式是否是完全相同的字符串
- #B 将两个表达式解析为 SymPy 对象（如果解析失败则返回 None）
- #C 如果两个表达式都成功解析，尝试符号比较
- #D 如果差值为 0，则它们是等价的

清单 3.8 中的 `equality_check` 函数确定模型的答案是否与真实值解决方案匹配。它首先寻找精确的字符串匹配，这是最简单的情况。如果字符串不同，它将两个表达式解析为 SymPy 对象（通过我们在清单 3.7 中实现的 `sympy_parser` 函数），并检查它们的差值是否简化为零。这使我们能够识别表面上看起来不同（例如，`14/3` 和 `28/6`）但在数学上相同的答案。

让我们尝试清单 3.8 中的相等性检查器的一个示例：

```python
print(equality_check(
    normalize_text("13/4."),
    normalize_text(r"(13)/(4)")
))
```

如预期的那样，这忽略了格式并返回 `True`。接下来，让我们尝试一个更具挑战性的示例，看看符号数学解析器是否识别出 `0.5` 与 `1/2` 相同：

```python
print(equality_check(
    normalize_text("0.5"),
    normalize_text(r"(1)/(2)")
))
```

这也返回 `True`。现在，让我们尝试一个反面示例：

```python
print(equality_check(
    normalize_text("14/3"),
    normalize_text("15/3")
))
```

这返回 `False`，因为表达式不同。

到目前为止，一切顺利。基于上面令人鼓舞的结果，我们可能会得出结论，我们现在拥有一个健壮的相等性检查器，可用于在数学基准数据集上评估 LLM。为了确保它已准备好投入使用，让我们再尝试一个示例：

```python
print(equality_check(
    normalize_text("(14/3, 2/3)"),
    normalize_text("(14/3, 4/6)")
))
```

在这种情况下，我们比较两个元组。由于 `2/3` 和 `4/6` 在数学上是等价的，我们期望结果为 `True`。相反，函数返回 `False`，因为它目前只处理简单表达式，不处理元组。我们将在下一节中解决这个限制。

## 3.7 评分答案

现在，我们将在上一节的数学相等性检查函数的基础上，实现一个健壮的评分函数，该函数还可以处理元组类表达式，例如正确比较像 `"(14/3, 2/3)"` 和 `"(14/3, 4/6)"` 这样的表达式。

首先，我们通过清单 3.9 实现一个 Python 辅助函数，通过该函数将此类元组类表达式拆分为单独的子部分。

---

**清单 3.9 拆分元组类表达式的辅助函数**

```python
def split_into_parts(text):
    result = [text]
    if text:  #A
        if (
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            items = [p.strip() for p in text[1:-1].split(",")]  #B
            if all(items):
                result = items
        else:  #C
            result = []
    return result
```

- #A 检查文本是否看起来像元组或列表，例如 `"(a, b)"` 或 `"[a, b]"`
- #B 拆分括号内的逗号并去除空白字符
- #C 如果文本为空，则返回空列表

清单 3.9 中的 `split_into_parts` 函数帮助我们处理具有多个组件的答案。如果输入看起来像元组或列表，例如 `(a, b)` 或 `[a, b]`，它会按逗号拆分内容并返回各个部分。（如果字符串为空，它简单地返回一个空列表。）本质上，此函数将多部分答案分解为可以逐个检查的较小部分。

在下一节实现评分函数之前，让我们先对 `split_into_parts` 进行测试，尝试之前的元组类表达式：

```python
split_into_parts(normalize_text(r"(14/3, 2/3)"))
```

这按预期返回 `['14/3', '2/3']`。

现在，我们可以实现 `grade_answer` 函数（清单 3.10），该函数将元组类表达式（如果存在）拆分为子部分，然后使用上一节的 `equality_check` 函数将生成的答案与参考（真实值）答案进行比较。

---

**清单 3.10 对预测答案与真实值进行评分的函数**

```python
def grade_answer(pred_text, gt_text):
    result = False   #A
    if pred_text is not None and gt_text is not None:  #B
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )
        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )
        if (gt_parts and pred_parts                #C
            and len(gt_parts) == len(pred_parts)):  #C
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )  #D
    return result  #E
```

- #A 如果检查失败，默认结果
- #B 仅当两个输入都是非空字符串时才继续
- #C 确保两边具有相同数量的有效部分
- #D 检查每个部分的数学等价性
- #E 仅当所有检查通过时才为 True

清单 3.10 中的 `grade_answer` 函数实现首先假设预测是错误的（`False`），并且仅当预测和真实值都非空时才继续。然后它对每一侧进行规范化并将其拆分为子部分（例如，将 `"(14/3, 2/3)"` 拆分为 `["14/3", "2/3"]`）。如果子部分的数量匹配，它使用 `equality_check` 逐个比较它们。仅当所有对都在数学上匹配时，结果才返回为正确（`True`）。

我们可以将 `grade_answer` 函数视为上一节 `equality_check` 函数的高级版本。`grade_answer` 函数可以拆分元组类表达式并在应用 `equality_check` 函数之前规范化答案。

在简单表达式上，它的工作方式与 `equality_check` 类似，如果两个表达式在数学上等价，则返回 `True`：

```python
grade_answer("14/3", r"\frac{14}{3}")
```

此外，如上所述，它现在在两个数学上等价的元组类表达式的情况下也返回 `True`：

```python
grade_answer(r"(14/3, 2/3)", "(14/3, 4/6)")
```

为了更全面地检查 `grade_answer` 函数，清单 3.11 中的代码包含更多样化的测试用例。

---

**清单 3.11 测试用例和测试评分器的演示函数**

```python
tests = [  #A
    ("check_1", "3/4", r"\frac{3}{4}", True),
    ("check_2", "(3)/(4)", r"3/4", True),
    ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
    ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
    ("check_5", "(1, 2)", r"(1,2)", True),
    ("check_6", "(2, 1)", "(1, 2)", False),
    ("check_7", "(1, 2, 3)", "(1, 2)", False),
    ("check_8", "0.5", "1/2", True),
    ("check_9", "0.3333333333", "1/3", False),
    ("check_10", "1,234/2", "617", True),
    ("check_11", r"\text{2/3}", "2/3", True),
    ("check_12", "50%", "1/2", False),
    ("check_13", r"2\cdot 3/4", "3/2", True),
    ("check_14", r"90^\circ", "90", True),
    ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
    ("check_16", r"2²", "2**2", True),
]

def run_demos_table(tests):
    header = ("Test", "Expect", "Got", "Status")
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth)  #B
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))
    data = [header] + rows
    col_widths = [  #C
        max(len(row[i]) for row in data)
        for i in range(len(header))
    ]
    for row in data:  #D
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(header))
        )
        print(line)
    passed = sum(r[3] == "PASS" for r in rows)  #E
    print(f"\nPassed {passed}/{len(rows)}")     #E
```

- #A 定义测试用例：(名称, 预测, 真实值, 预期结果)
- #B 运行相等性检查
- #C 计算每列的最大宽度以整齐对齐表格
- #D 逐行打印表格
- #E 打印通过的测试摘要

清单 3.11 中的代码是一个简单的测试套件，它接收一组测试来检查 `grade_answer` 函数是否按预期工作。`tests` 列表包含涵盖分数、LaTeX 表示法、元组输入、小数、百分比和其他棘手格式的元组。

然后 `run_demos_table` 函数通过调用 `grade_answer` 运行每个测试，收集结果，并将结果组织成格式化的表格。

调用清单 3.11 中的 `run_demos_table(tests)` 函数会打印以下内容：

```
Test     | Expect | Got   | Status
check_1  | True   | True  | PASS  
check_2  | True   | True  | PASS  
check_3  | True   | True  | PASS  
check_4  | True   | True  | PASS  
check_5  | True   | True  | PASS  
check_6  | False  | False | PASS  
check_7  | False  | False | PASS  
check_8  | True   | True  | PASS  
check_9  | False  | False | PASS  
check_10 | True   | True  | PASS  
check_11 | True   | True  | PASS  
check_12 | False  | False | PASS  
check_13 | True   | True  | PASS  
check_14 | True   | True  | PASS  
check_15 | True   | True  | PASS  
check_16 | True   | True  | PASS  
Passed 16/16
```

基于上面显示的 PASS 结果，`grade_answer` 函数相对健壮，能够处理各种格式不同的表达式。

> **练习 3.1：添加更多测试用例**
>
> 尝试思考额外的测试用例，最好是具有挑战性的，并将它们添加到 `run_demos_table()` 函数中。你能找到检查错误地失败的情况吗？

随着 `grade_function` 的实现，我们现在已经具备了评估 LLM 的基本构建块。在下一节中，我们将加载一个数学数据集，在该数据集上评估 LLM。

## 3.8 加载评估数据集

正如我们在本章中看到的，实现一个健壮的验证流程可能是一项繁琐的任务。幸运的是，我们现在已经拥有了从答案提取到评分的所有部分，并准备在基准数据集上评估 LLM。为此，如图 3.9 所示，我们将使用 MATH-500 数据集（https://huggingface.co/datasets/HuggingFaceH4/MATH-500），这是一个广泛使用的推理模型基准。它是从原始 MATH 数据集中采样的 500 个问题的精选集合。

---

![图 3.9 加载评估数据集。在前面的章节中完成步骤 2–6（生成、提取、规范化、验证和评分答案）之后，剩下的两个步骤是加载完整数据集（步骤 7）并在所有问题上应用相同的过程以评估模型（步骤 8）。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F09_raschka.webp?1)

选择使用 MATH-500 出于三个实际原因。首先，MATH-500 足够大以具有意义，但仍然足够小，可以在本书中进行重复评估，而每次运行完整的 MATH 数据集会慢得多。其次，在后面的章节中，我们使用从原始 MATH 数据集派生的不重叠训练子集，因此将 MATH-500 作为保留评估集可以为我们提供一个干净的参考点。第三，MATH-500 是推理模型文献中的常见基准数据集，这使得本书中的结果更容易与先前的工作进行比较。我们在这里也更喜欢 MATH-500 而不是更简单的数据集（如 GSM8K），因为它对现代推理模型更具挑战性，并且更好地匹配我们在本章中构建的多步符号验证流程的类型。

我们将使用以下代码加载 MATH-500 数据集（图 3.9 中的步骤 7）：

---

**清单 3.12 加载 MATH-500 数据集**

```python
import json
import requests

def load_math500_test(local_path="math500_test.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if save_copy:  # Saves a local copy
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    return data

math_data = load_math500_test()
print("Number of entries:", len(math_data))
```

这打印：

```
Number of entries: 500
```

### 从 Hugging Face Model Hub 加载数据集

以下信息和代码示例是可选的，仅供参考，您不需要运行下面的代码。

MATH-500 数据集划分最初在 PRM800K 仓库（https://github.com/openai/prm800k/tree/main?tab=readme-ov-file#math-splits）中提出，也可在 Hugging Face Hub（https://huggingface.co/datasets/HuggingFaceH4/MATH-500）上获取。在这里，我们从代码仓库加载一个副本，以确保在外部来源发生变化时的可复现性。

如果您更喜欢直接从 Hugging Face 下载数据集，可以使用以下代码。请注意，这需要 `datasets` 库，可以通过 `pip install datasets` 或 `uv add datasets` 安装：

```python
from datasets import load_dataset
dset = load_dataset("HuggingFaceH4/MATH-500", split="test")
```

在跳到下一节实现模型评估流程之前，让我们通过打印其第一个条目来仔细查看数据集的结构（我们使用内置的 `pprint` 库以获得更好的格式）：

```python
from pprint import pprint
pprint(math_data[0])
```

这产生以下输出：

```python
{'answer': '\\left( 3, \\frac{\\pi}{2} \\right)',
 'level': 2,
 'problem': 'Convert the point $(0,3)$ in rectangular coordinates to polar '
            'coordinates.  Enter your answer in the form $(r,\\theta),$ where '
            '$r > 0$ and $0 \\le \\theta < 2 \\pi.$',
 'solution': 'We have that $r = \\sqrt{0^2 + 3^2} = 3.$  Also, if we draw the '
             'line connecting the origin and $(0,3),$ this line makes an angle '
             'of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n'
             '\n'
             '[asy]\n'
             'unitsize(0.8 cm);\n'
             '\n'
             'draw((-0.5,0)--(3.5,0));\n'
             'draw((0,-0.5)--(0,3.5));\n'
             'draw(arc((0,0),3,0,90),red,Arrow(6));\n'
             '\n'
             'dot((0,3), red);\n'
             'label("$(0,3)$", (0,3), W);\n'
             'dot((3,0), red);\n'
             '[/asy]\n'
             '\n'
             'Therefore, the polar coordinates are $\\boxed{\\left( 3, '
             '\\frac{\\pi}{2} \\right)}.$',
 'subject': 'Precalculus',
 'unique_id': 'test/precalculus/807.json'}
```

正如我们所见，数据集条目格式化为带有键和值的 Python 字典。相关的键是：
- `"problem"`：LLM 要解决的数学问题或题目；
- `"answer"`：正确的（真实值）答案，我们希望将 LLM 的答案与之比较；
- `"solution"`：问题的逐步详细解释（本章未使用，但对训练或分析有用）。

（请注意，输出包含按字母顺序排序的键：`"answer"`、`"problem"`、`"solution"`，但项目符号使用更具逻辑性的排序以提高可读性。）

现在我们有了预训练的 LLM、评估函数和一个可以使用的基准数据集，我们可以实现模型评估。

## 3.9 评估模型

在本节中，我们将图 3.10 中步骤 2–6 的 LLM 文本生成和评估工具付诸实践，并将其应用于我们在上一节加载的 MATH-500 数据集（图 3.10 中的步骤 8）。

这个完整的流程在本章之外也很重要，因为它将成为我们稍后比较提示方法和基于训练的改进时衡量进展的主要方式。

---

![图 3.10 MATH-500 数据集上的完整评估流程。加载数据集（步骤 7）后，步骤 2–6 系统地应用于所有问题，以获得最终的模型评估（步骤 8）。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F10_raschka.webp?1)

正如您可能从 3.4 节（提取最终答案方框）中回忆的那样，我们的答案检查流程期望模型以方框形式返回答案，这是评估推理模型在数学问题上的常见约定。为了增加模型遵循此格式的可能性，我们可以将提示格式化为如清单 3.13 所示：

---

**清单 3.13 用于数学评估的提示模板渲染函数**

```python
def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template
```

现在让我们将清单 3.13 中的提示模板应用于本章前面介绍的示例提示（第 3.2 节）。为了方便起见，我们在这里重新定义示例提示：

```python
prompt = (
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)
prompt_fmt = render_prompt(prompt)
print(prompt_fmt)
```

格式化后的提示现在如下：

```
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}

Question:
If $a+b=3$ and $ab=\tfrac{13}{6}$, what is the value of $a^2+b^2$?

Answer:
```

接下来，我们将提示传递给我们定义的文本生成包装器函数，如第 3.3 节清单 3.3 所示，在构建模型评估函数之前回顾一下文本生成过程：

```python
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt_fmt, device,
    max_new_tokens=2048,
    verbose=True
)
```

使用这个提示示例，模型以一个相对简短的答案响应：`"\boxed{10}"`。（请注意，生成的响应可能会因您在 CPU、CUDA 或 MPS 设备上执行代码而有所不同。）

虽然简洁性可以通过减少 token 数量来加快生成速度，但响应是不正确的。相比之下，在第 3.3 节中，没有提示模板的情况下，模型产生了一个更长的响应，从而得出了正确答案 `14/3`。

提示模板是否适合给定的模型和任务，理想情况下需要在更大的示例集上确定，然后我们才能得出任何结论，例如我们将在本节稍后评估模型的 MATH-500 数据集。

> **提示模板选择**
>
> 清单 3.13 中的提示模板在这里用于演示如何实现模型评估流程，并自动检查答案的正确性。所选的模板鼓励短输出，这让您在第一次阅读时高效地浏览本章。之后，我建议使用替代设置重新访问本章，以优化我们在本章中作为参考模型的官方 Qwen3 推理模型变体的准确性。
>
> 事实证明，不使用提示模板可以将基础模型性能提高 50%，但会将推理模型的准确性降低 40%。
>
> 此外，我们还可以尝试其他提示模板。例如，MATH-500 基准的常见标准提示是以下变体，将清单 3.13 中的 `"Question:"` 替换为 `"Problem:"`。这个看似微小的变化将基础模型的准确性提高了约 20%，可能是因为它更好地匹配了记忆的训练数据（假设 MATH-500 测试集包含在训练语料库中）。虽然基础模型受益于这一变化，但推理模型变体的准确性下降了 30%。
>
> 早些时候，我们提到答案提取是一个简单的机械任务，可以用确定性代码解决，而不需要雇用另一个 LLM 进行答案提取。基于不同提示模板导致的准确性变化，看起来我们的提取方法可能不可靠。情况不一定如此。
>
> 此外，LLM 生成格式错误的答案也不一定如此。它可能只是不正确，切换到另一个 LLM 进行提取也无法解决这个问题。较小的基础模型通常对提示措辞非常敏感。在下一章中，我们将看到一旦引入额外的提示变体，这一点变得更加明显。

接下来，在我们实现最终的模型评估函数之前，让我们通过清单 3.14 中的演示函数在一个较小的示例上端到端测试我们的模型评估流程：

---

**清单 3.14 运行评估流程的演示函数**

```python
def mini_eval_demo(model, tokenizer, device):
    ex = {  #A
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3"
    }
    prompt = render_prompt(ex["problem"])     #B
    gen_text = generate_text_stream_concat(   #C
        model, tokenizer, prompt, device,     #C
        max_new_tokens=64,                    #C
    )                                         #C
    pred_answer = extract_final_candidate(gen_text)  #D
    is_correct = grade_answer(                       #E
        pred_answer, ex["answer"]                    #E
    )                                                #E
    print(f"Device: {device}")
    print(f"Prediction: {pred_answer}")
    print(f"Ground truth: {ex['answer']}")
    print(f"Correct: {is_correct}")
```

- #A 带有 `"problem"` 和 `"answer"` 字段的测试示例
- #B 1. 应用提示模板
- #C 2. 生成响应
- #D 3. 提取并规范化答案
- #E 4. 评分答案

清单 3.14 中的 `mini_eval_demo` 函数本质上将评估组件连接成一个小函数，我们可以在为 MATH-500 数据集编码最终评估流程之前使用它来测试代码。代码从一个玩具示例（`ex`）开始，将问题渲染到提示模板（`prompt`）中，并从模型流式传输响应（`generate_text_stream_concat`）。然后它将模型输出解析为最终候选答案（`pred_answer`），并使用 `grade_answer` 对其进行真实值评分。最后，它打印结果供我们评估。

调用 `mini_eval_demo(model, tokenizer, device)` 函数会产生以下输出：

```
Device: mps
Prediction: 1/3
Ground truth: 2/3
Correct: False
```

我们可以看到，生成的答案（`"1/3"`）被正确提取，但它与正确答案（`"2/3"`）不匹配，因此检查返回 `False`。（请注意，结果可能会因您在 CPU、CUDA 或 MPS 设备上执行代码而有所不同。）

现在我们已经在一个更简单的示例上测试了我们的工作流程，让我们实现它在 MATH-500 数据集上运行。

---

**清单 3.15 MATH-500 数据集的端到端模型评估流程**

```python
import time

def eta_progress_message(  #A
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress",
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:          
        return progress.ljust(pad_width)
    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)
    remaining = max(total - processed, 0)
    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0
    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"
    message = f"{progress} | ETA: {eta}"
    return message.ljust(pad_width)

def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")      #B
        out_path = Path(f"math500-{dev_name}.jsonl")
    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()
    with open(out_path, "w", encoding="utf-8") as f:   #C
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])     #D
            gen_text = generate_text_stream_concat(    #E
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            extracted = extract_final_candidate(       #F
                gen_text
            )
            is_correct = grade_answer(                 #G
                extracted, row["answer"]
            )
            num_correct += int(is_correct)
            record = {                                 #H
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:                                #I
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc
```

- #A 打印进度的辅助函数，可选 ETA（预计到达时间）
- #B 使文件名与 Windows 兼容
- #C 保存结果以供检查
- #D 1. 应用提示模板
- #E 2. 生成响应
- #F 3. 提取并规范化答案
- #G 4. 评分答案
- #H 保存以供检查的记录
- #I 在生成期间打印响应

清单 3.15 中的 `evaluate_math500_stream` 函数使用与清单 3.14 中较小的演示函数相同的主要步骤：对于每个问题，它渲染提示，流式传输模型响应，提取答案候选，并根据参考答案进行评分。

除了迭代具有多个条目的数据集之外，它还添加了一些额外的功能。例如，它以 Python 字典格式将生成的响应保存到 JSON 文件中，以便记录和更仔细地检查。

现在让我们在子集上运行此函数，即 MATH-500 的前 10 个示例，这在一台配备 M4 芯片的 Mac Mini 上大约需要 0.7 分钟。（评估推理模型变体大约需要 7 分钟，因为它生成更长的响应。）

```python
print("Model:", WHICH_MODEL)
num_correct, num_examples, acc = evaluate_math500_stream(
    model, tokenizer, device, 
    math_data=math_data[:10],  #A
    max_new_tokens=2048,
    verbose=False              #B
)
```

- #A 仅评估前 10 个示例
- #B 设置为 true 以在生成时读取响应

在上面的代码示例中，我们将 `max_new_tokens` 设置为一个慷慨的 2048，因为推理模型变体按设计倾向于生成更长的响应，我们不希望过早地截断它。这导致评估时间更长，可能看起来生成卡住了。可选地，您可以设置 `verbose=True` 以逐 token 实时查看正在生成的响应。

运行 `evaluate_math500_stream` 函数的结果如下：

```
Model: base
Device: mps
MATH-500: 10/10 | ETA: 00s
Accuracy: 30.0% (3/10)
Total time: 0.4 min
```

（请注意，结果可能会因您在 CPU、CUDA 或 MPS 设备上执行代码而有所不同。）

正如我们所见，模型达到了相对较低的 30% 准确率。我们可以在文本编辑器中打开 `math500_base-mps.jsonl` 文件来分析结果，同时查看生成的响应。例如，我们发现所有情况下的答案都已成功提取，但它们明显是错误的，这表明模型没有非常强的数学问题解决能力（尚未）。这是预料之中的，因为它仅仅是一个基础模型。

### 以编程方式加载 .jsonl 文件

`.jsonl` 文件后缀是用于每行一个数据条目的 JSON 文件的约定。您可以在您喜欢的文本编辑器中查看它。可选地，我们可以使用以下代码在 Python 中加载评估期间创建的 `.jsonl` 文件：

```python
dev_name = str(device).replace(":", "-")
local_path = f"math500_{WHICH_MODEL}-{dev_name}.jsonl"
results = []
with open(local_path, "r") as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))
```

推理模型变体，您可以通过在第 3.2 节清单 3.1 中设置 `WHICH_MODEL = "reasoning"` 来启用，表现更好，在相同的 10 个样本子集上达到 90% 的准确率，在完整的 500 个样本数据集上达到 50.8%，如表 3.1 所示。

---

**表 3.1 不同设备上的 MATH-500 任务准确率**

| Mode      | Device   | Accuracy | MATH-500 size |
|-----------|----------|----------|---------------|
| Base      | CPU      | 30%      | 10            |
| Base      | CUDA     | 30%      | 10            |
| Base      | MPS      | 30%      | 10            |
| Reasoning | CPU      | 90%      | 10            |
| Reasoning | CUDA     | 90%      | 10            |
| Reasoning | MPS      | 80%      | 10            |
| Base      | CUDA     | 15.3%    | 500           |
| Reasoning | CUDA     | 50.8%    | 500           |

如表 3.1 所示，推理变体以其更长的响应，准确率大幅提高，但也大幅增加了计算强度和答案生成时间（在配备 M4 芯片的 Mac Mini 上的 10 个样本子集上，从基础模型的 0.4 分钟增加到推理模型的 7 分钟；在 H100 上的 500 个样本数据集上，从 13.3 分钟增加到 185.4 分钟），这凸显了使用推理模型的权衡之一。

请注意，这些数字是在 PyTorch 2.8 中获得的，在不同版本的 PyTorch 中可能会有所不同。

> **提示** 代码仓库包含一个额外脚本（https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500_batched.py），它以批处理模式运行本章中的代码。这意味着它在每次前向传递中处理多个示例以加速评估，同时需要更多的 RAM。使用 128 的批大小，这将基础模型评估所有 500 个样本的运行时间从 H100 上的 13.3 分钟减少到 3.3 分钟。同样，它将推理模型的运行时间从 185.4 分钟减少到 14.6 分钟。请注意，H100 仅用作示例，该脚本也兼容其他 GPU。

> **练习 3.2：计算平均响应长度**
>
> 尝试修改本章中的代码，以在清单 3.15 的 `evaluate_math500_stream` 函数中也报告平均响应长度。您也可以不直接修改函数，而是从生成的 JSON 报告文件中计算响应长度。

> **练习 3.3：扩展或更改评估数据集**
>
> 我们选择仅 10 个示例的子集是为了计算效率。鼓励读者考虑在更大或不同的数据集部分上运行代码，以观察 10 个样本子集是否具有代表性。理想情况下，您也可以尝试自己的数据。（作为参考，在完整的 MATH-500 数据集上评估基础模型在 H100 上大约需要 13.3 分钟，推理模型大约需要 185.4 分钟。）

> **练习 3.4：尝试不同的提示模板**
>
> 模型可能对不同的提示模板敏感。尝试清单 3.13 中的不同提示模板，看看它如何影响结果。此外，虽然 Qwen3 团队建议在不使用额外聊天模板的情况下使用基础模型，但您还可以在分词器中启用 `apply_chat_template=True` 设置（清单 3.1），并观察它是否改善了基础模型性能。

请注意，这结束了我们关于实现数学任务验证方法的章节（图 3.11）。我们选择数学是因为它不仅自然易于实现，而且在推理特定训练中广泛使用，特别是具有可验证奖励的强化学习，我们将在第 6 章中介绍。相同的概念可以扩展到其他领域，例如代码，尽管我们在这里没有探讨，因为执行代码需要额外设置安全虚拟环境。

在继续之前，值得注意的是，评估还有许多其他形式。在本章中，我们专注于数学问题的基于验证的准确性，因为它是一种流行的方法，而且我们将在第 6 章和第 7 章中重用相同的验证器作为强化学习流程的一部分。

如需更广泛的概述，附录 F 介绍了其他常见的评估策略，如多项选择基准、验证器、排行榜和 LLM 作为评判的设置。如果您想快速了解这些方法在实践中如何工作，该附录提供了概述和动手示例。

---

![图 3.11 本书所涵盖主题的心智模型。本章实现了基于验证器的评估流程。在下一章中，我们将通过更先进的推理技术来提高 LLM 的推理能力。](https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F11_raschka.webp?1)

现在，随着评估框架的到位，下一章如图 3.11 所示，专注于通过更先进的推理（文本生成）技术来提高推理能力。

## 3.10 总结

- LLM 有四种主要评估方法：多项选择、验证器、排行榜和 LLM 评判
- 基于验证的评估方法允许自由形式的答案，并使用外部工具检查正确性
- 本章通过构建一个提取、规范化并使用 SymPy 检查答案的数学验证器，专注于基于验证的评估
- 验证流程涉及从加载 LLM 到在数据集上运行评估的几个核心步骤
- 作为验证流程的一部分，答案提取使用字符串解析来定位方框内容（并对缺失的方框提供回退机制）
- 另一个步骤实现规范化，通过去除 LaTeX 和转换数学表示法来标准化各种答案格式
- 最后，流程使用数学等价性检查（通过 SymPy）来符号比较表达式
- MATH-500 数据集提供 500 个精选数学问题用于评估
- 提示模板显著影响模型性能
- 推理模型比基础模型达到更高的准确率，但需要更长的运行时间
