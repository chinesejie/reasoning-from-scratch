# 7 改进用于强化学习的 GRPO

本章涵盖以下内容：

- 解读训练曲线与评估指标
- 防止模型利用奖励信号
- 在任务正确性奖励之外增加响应格式奖励

此前，我们已经端到端地实现了用于可验证奖励强化学习（RLVR）的 GRPO 算法。现在，如图 7.1 所示，我们将从该基线出发，聚焦于延长训练时间后会发生什么。

**图 7.1** 本书所涵盖主题的心智模型。本章深入讲解用于可验证奖励强化学习的 GRPO 算法。

---

具体来说，我们将讨论哪些指标值得追踪（超越奖励和准确率）、如何尽早发现失败模式，以及为何即使代码"正确"，训练仍可能变得不稳定。此外，事实证明，基础 GRPO 可能导致训练不稳定，因此本章还会介绍推理模型训练中使用的实用 GRPO 扩展与修复方法。

## 7.1 改进 GRPO

在上一章实现 GRPO（组相对策略优化）之后，我们现在重新审视并更彻底地分析训练过程。同时，我们还将回顾上一章省略的 KL 损失项，并讨论一系列在实际训练运行中变得重要的实用技巧和算法选择。这些主题汇总在本章概览图 7.2 中。

**图 7.2** 本章概览，展示了本章涵盖的不同主题。

---

本章中的示例基于真实实验，但解读结果时需谨慎。若要得出关于单个设置效果的可靠结论，每次实验都需要重复多次并取平均结果，因为采样和优化中的随机性可能导致不同运行之间出现显著差异。尽管如此，这里展示的示例足以阐明主要思想和相关的权衡取舍。

> **注意** 本章聚焦于使用 GRPO 时的额外技术细节和实际考量。如果读者觉得本章内容过于技术化并希望跳过，也可以直接阅读下一章，因为下一章不依赖于本章内容。

## 7.2 追踪 GRPO 性能指标

在第 6 章中，我们运行了一个简短的 GRPO 训练循环并简要检查了结果。例如，一次短运行的输出结构如下：

```
[Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=5.5
[Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.8
[Step 3/50] loss=0.3592 reward_avg=0.250 avg_resp_len=7.8
[Step 4/50] loss=2.7401 reward_avg=0.250 avg_resp_len=56.5
[Step 5/50] loss=3.3214 reward_avg=0.500 avg_resp_len=251.2
# ...
```

让我们从这一点出发，讨论使用 GRPO 训练时应追踪哪些指标，以及它们如何帮助解读和调试训练行为。

### 7.2.1 执行 GRPO 训练运行

第 6 章的代码旨在完整实现 GRPO 流程与保持代码紧凑可读之间取得平衡。为方便起见，补充材料中包含了一个等效的脚本（`https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py`），其中包含相同的代码，可以从代码终端运行（稍后会详细介绍）。

对于本章介绍的每一项 GRPO 改进，我们都将使用补充材料中类似的参考脚本，以避免重复冗长的代码块，否则会使本章内容和相关讨论不必要地膨胀。

使用清单 7.1 中的辅助函数，我们可以根据需要下载补充材料中的相关脚本并保存在本地。这是为了避免重复冗长的代码段落，并专注于与之前 GRPO 实现相比的主要变化。

**清单 7.1** 用于下载补充材料的辅助函数

```python
from pathlib import Path
import requests

def download_from_github(rel_path, out=None):
    github_raw_base = (  #A
        "https://raw.githubusercontent.com/rasbt/"
        "reasoning-from-scratch/refs/heads/main/"
    )
    rel_path = Path(rel_path)
    out = Path(out) if out is not None else Path(rel_path.name)  #B
    if out.exists():  #C
        size_kb = out.stat().st_size / 1e3
        print(f"{out}: {size_kb:.1f} KB (cached)")
        return out
    r = requests.get(github_raw_base + rel_path.as_posix())  #D
    r.raise_for_status()
    out.write_bytes(r.content)
    size_kb = out.stat().st_size / 1e3
    print(f"{out}: {size_kb:.1f} KB")
```

- #A 基础 URL
- #B 使用 URL 文件名作为默认输出文件名
- #C 如果文件已在本地存在则跳过下载
- #D 下载文件

使用上述辅助函数，我们可以按如下方式下载包含第 6 章 GRPO 训练代码的脚本：

```python
download_from_github(
    "ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py"
)
```

执行上述代码时，你应该会看到以下输出：

```
rlvr_grpo_original_no_kl.py: 13.4 KB
```

---

如果下载的文件大小与上述显示的大小相差很大，则下载可能没有正确完成。在这种情况下，首先仔细检查 URL 是否有任何拼写错误。如果问题仍然存在，请参阅故障排除指南（`https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md`）获取更多建议。

生成的脚本可以从终端运行，如图 7.3 所示。

**图 7.3** 在终端中使用 GRPO 进行训练运行的输出，包含多项训练统计信息，如损失、平均奖励、每秒 token 吞吐量和平均响应长度。

也可以在 Jupyter Notebook 的代码单元中通过在命令前加上 "!" 来运行训练脚本，即 `!uv run rlvr_grpo_original_no_kl.py`。

如果你不是 uv 用户，请将图 7.3 中显示的 `uv run` 替换为 `python`，即 `python rlvr_grpo_original_no_kl.py`：

```bash
python rlvr_grpo_original_no_kl.py \
    --steps 500 \
    --max_new_tokens 1024
```

> **提示** 你可以在上述代码执行命令中添加 `--show_eta` 以显示脚本在你机器上的总运行时间估计。

---

如果你不想自己运行训练代码，鉴于其计算成本，这也是合理的，补充材料提供了此次运行的日志文件。我们可以按如下方式下载这些文件：

```python
download_from_github(
    "ch07/02_logs/rlvr_grpo_original_no_kl_metrics.txt"
)
download_from_github(
    "ch07/02_logs/ch06_rlvr_grpo_original_no_kl_metrics.csv"
)
```

第一个以 `.txt` 结尾的文件是纯文本输出文件，显示与图 7.3 类似的输出统计信息。第二个以 `.csv` 结尾的文件是逗号分隔值文件，为这些信息的提取和绘图分析提供了更多结构化数据。

### 7.2.2 检查 GRPO 训练运行

为了检查训练运行，我们使用 Matplotlib 绘制之前日志文件的结果。为此，我们定义了以下绘图函数，将在本章中多次使用：

**清单 7.2** 用于可视化日志文件中训练结果的绘图函数

```python
import csv
import matplotlib.pyplot as plt

def moving_average(values, window_fraction=0.25):
    window_size = max(1, int(window_fraction * len(values)))  #A
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window_mean = sum(values[start_idx : i + 1]) / (i - start_idx + 1)
        smoothed.append(window_mean)
    return smoothed

def plot_grpo_metrics(csv_path, columns, save_as=None):
    data = {name: {"steps": [], "values": []} for name in columns}
    with Path(csv_path).open(newline="") as f:  #B
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("step"):
                continue
            step = int(row["step"])  #C
            for name in columns:
                value_str = row.get(name)
                if value_str:
                    data[name]["steps"].append(step)
                    data[name]["values"].append(float(value_str))
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(6, 4))  #D
    axes = axes.ravel()
    for i, name in enumerate(columns):
        steps = data[name]["steps"]
        values = data[name]["values"]
        if not values:  #E
            fig.delaxes(axes[i])
            continue
        if name == "eval_acc":  #F
            axes[i].bar(steps, values, width=20)
        else:
            axes[i].plot(steps, values, alpha=0.4)
            axes[i].plot(steps, moving_average(values))
        axes[i].set_ylabel(name)
    for j in (2, 3):
        if axes[j] in fig.axes:
            axes[j].set_xlabel("Step")
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()
```

- #A 平滑嘈杂的训练信号，以揭示训练期间的长期趋势
- #B 打开并读取 CSV 日志文件
- #C 使用训练步数作为所有指标共享的 x 轴
- #D 创建固定网格，以便并排显示损失、奖励、响应长度等指标
- #E 跳过不存在的指标
- #F 评估准确率使用柱状图，因为我们没有每一步的数据

---

我们下载的 `.csv` 文件包含多个列。使用清单 7.2 中定义的 `plot_grpo_metrics` 函数，我们绘制损失、平均奖励、平均响应长度和评估准确率：

```python
plot_grpo_metrics(
    "rlvr_grpo_original_no_kl_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]
)
```

得到的绘图如图 7.4 所示。

**图 7.4** GRPO 训练运行期间追踪的四项指标（损失、平均奖励、平均响应长度和评估准确率）。橙色中心线表示最近 25% 数值的移动平均，有助于揭示原本嘈杂的训练信号中的整体趋势。评估准确率以柱状图显示，因为它每 50 步计算一次，而非每一步。

---

在图 7.4 所示的图表中，有几项总体观察值得注意。平均响应长度应该最初会增加，理想情况下准确率也会同时提高，这 largely 是我们在这里看到的，尽管在接近第 400 步时运行后期出现了明显的下降。与 LLM 预训练（这超出了本书范围，在《从零开始构建大型语言模型》中有更详细的介绍）相比，损失值本身信息量较少，主要作为合理性检查。总体而言，损失应该保持相对稳定。一些波动是预期的，但运行中期出现的较大峰值有些令人担忧。

平均奖励也应该随时间增加。原则上，平均奖励为 1.00 意味着所有采样响应都是正确的，这是理想的，但也意味着训练信号已经消失。此时，进一步训练不太可能有用，提前停止可以节省时间和资源。

最后，在外部目标任务上的性能，这里以 MATH-500 准确率衡量，应该提高。在此次运行中，准确率起初上升，但随后开始下降，这指向训练过程中的问题和不稳定。

---

总之，训练运行显示出早期的快速提升，随后在大约五十步之后出现收益递减。一个可能的原因是底层算法在较长运行中不是特别稳定。另一个可能的解释可能是后期的训练示例更困难，但这无法解释评估准确率的下降，因为评估准确率是每五十步从相同的 500 个 MATH-500 测试样本计算的。

请注意，本节的主要目的是介绍、分析和讨论各种训练指标。本节聚焦一些基本指标，在下一节中，我们将扩展这个列表并查看 GRPO 训练期间通常追踪的额外指标。

> **计算评估准确率**
>
> 评估准确率（`eval_acc`），这里在 MATH-500 基准上测量，默认不在训练期间追踪。可以通过在运行训练脚本时添加 `--eval_on_checkpoint 500` 设置来定期计算。不推荐这样做，因为这会显著减慢训练速度。或者，可以在训练后使用第 3 章介绍的 `evaluate_math500.py` 脚本单独计算：
>
> ```python
download_from_github(
    "ch03/02_math500-verifier-scripts/evaluate_math500.py"
)
```
>
> 然后可以在给定检查点上运行评估：
>
> ```bash
uv run evaluate_math500.py \
    --dataset_size 500 \
    --checkpoint_path \
    "checkpoints/rlvr_grpo_original_no_kl/\
    qwen3-0.6B-rlvr-grpo-step00050.pth"
```
>
> 在上述讨论的运行中，评估准确率已包含在日志文件中。（如果你不是 uv 用户，请将 `uv run` 替换为 `python`。）

## 7.3 追踪更高级的 GRPO 性能指标

除了用于监控 GRPO 训练运行的少量基本指标外，还有几个额外的指标有助于理解训练动态，如图 7.5 所示。

**图 7.5** 在分析基本 GRPO 训练指标后，我们现在添加更高级的指标来分析训练运行。

图 7.5 中提到的两个更高级指标的例子是 rollout 优势值（已在第 6 章的 `compute_grpo_loss` 函数中计算）和生成序列的熵。

### 7.3.1 优势追踪

作为 GRPO 算法的一部分，我们计算所谓的优势值，如第 6 章所讨论并在图 7.6 中所示。除了它们在损失计算中的作用外，这些值也有助于分析和理解训练动态。

**图 7.6** 来自第 6 章的 GRPO 概览图。优势值显示在第 3 步。

---

具体来说，我们计算图 7.6 中所示优势的两种汇总统计量，即它们的平均值（样本均值）和标准差：

**清单 7.3** 计算优势统计量

```python
import torch

def compute_advantage_stats(rewards_list):
    rewards = torch.tensor(rewards_list)  #A
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    adv_avg = advantages.mean().item()  #B
    adv_std = advantages.std().item()
    return advantages, adv_avg, adv_std

adv, adv_avg, adv_std = compute_advantage_stats([1., 1., 0., 0.])
print(f"Advantages: {adv}")
print(f"Advantage mean = {adv_avg:.4f}, std = {adv_std:.4f}")
```

- #A 下面的奖励和优势值已经是我们在 GRPO 中计算的
- #B 现在，我们还计算平均值（mean）和标准差（std）

作为简单说明，考虑四个奖励值的小样本，类似于前面图 7.6 中所示。

输出为：

```
Advantages: tensor([ 0.8659,  0.8659, -0.8659, -0.8659])
Advantage mean = 0.0000, std = 0.9998
```

---

由于 GRPO 中优势值的计算方式，它们的均值始终为零。在实践中，这使得均值主要作为实现是否按预期行为的合理性检查。标准差信息量更大。接近 1 的值表示梯度信号缩放良好，通常与稳定的更新相关。非常小的值指向消失的学习信号，这通常发生在奖励崩溃时。非常大的值表示过于尖锐的更新，可能使训练不稳定。

当所有奖励相同时会出现极端情况。当所有优势值相同时，这由标准差为零表示，策略梯度变为零，不会发生模型权重更新。换句话说，模型在这些情况下无法学习改进。例如，考虑以下场景：

```python
adv, adv_avg, adv_std = compute_advantage_stats([0., 0., 0., 0.])
print(f"Advantages: {adv}")
print(f"Advantage mean = {adv_avg:.4f}, std = {adv_std:.4f}")
```

输出如下：

```
Advantages: tensor([0., 0., 0., 0.])
Advantage mean = 0.0000, std = 0.0000
```

---

如果我们用全一奖励 `compute_advantage_stats([1., 1., 1., 1.])` 替换 `compute_advantage_stats([0., 0., 0., 0.])` 中的全零奖励，输出也类似。

在实践中，最好将优势统计量与平均奖励一起考虑。如前所述，平均奖励为 1.0 实际上是理想的结果，即使这意味着训练信号已经消失，因为模型所有答案都正确。此时，通常停止训练或切换到更具挑战性的示例是有意义的。

### 7.3.2 熵追踪

在追踪和分析上一节介绍的优势统计量之前，我们先介绍另一个要追踪的指标：熵。

熵衡量模型在生成下一个 token 时的不确定性。高熵意味着概率分布在许多可能的 token 上，这鼓励探索。低熵意味着大部分概率集中在单个 token 上，这使得模型越来越确定。这也可能预示着训练崩溃，模型停止探索并不断产生相同的输出。

在计算熵之前，简要回顾我们在第 5 章中如何计算对数概率值（logprobs）是有用的，如图 7.7 总结所示。

**图 7.7** LLM 生成答案中单个 token（"this"）的对数概率（logprob）计算。LLM 返回该 token 的 logits，然后通过 `torch.softmax()` 转换为 softmax 概率值，或通过 `torch.log_softmax()` 转换为 logprob 值。

---

在图 7.7 中，我们没有使用通过提示基础模型创建的真实 logits，而是使用示例 logits，假设词汇表大小为 7（否则，使用原始的 151,000 token 词汇表，在图中可视化这个概念将是不可能的）：

```python
logits = torch.tensor([
    0.6667, -2.0000,  1.3333, -0.0000, -0.6667,  2.0000, -1.3333
])
```

以下代码实现了前面图 7.7 中所示的计算：

```python
logprobs = torch.log_softmax(logits, dim=-1)
print("All token logprobs:", logprobs)
selected_token = torch.argmax(logprobs)
selected = logprobs[selected_token]
print("Selected token ID:", selected_token)
print("Selected token logprob:", selected)
```

输出为：

```
All token logprobs: tensor([-2.0442, -4.7109, -1.3776, -2.7109, -3.3776, -0.7109, -4.0442])
Selected token ID: tensor(5)
Selected token logprob: tensor(-0.7109)
```

---

简而言之，`torch.log_softmax()` 函数计算所有对数概率（logprobs），`torch.argmax()` 返回最大 logprob 的索引（token ID）（这里是 5），`logprobs[selected_token]` 返回该 token ID 的 logprob 值（-0.7109）。

熵与对数概率密切相关。我们通过将每个概率乘以其对数概率，然后求和这些乘积来计算它，如图 7.8 所示。

**图 7.8** 熵项通过将 token 概率与 token logprobs 相乘来计算。

图 7.8 显示熵是通过概率和 logprob 值相乘计算的。原则上，我们也可以在训练期间追踪比熵更简单的量，例如概率或 logprobs 的总和，但熵是衡量概率分布不确定性的广泛使用且易于解释的指标。

---

在前面的图中，我们看到熵为 1.37。作为粗略的经验法则：

- **非常低的熵（约 0-0.5）** 意味着一个 token 主导分布。模型高度自信，行为几乎确定性的。
- **中等熵（约 1-2）** 意味着概率在合理数量的小 token 集合上共享，这是稳定训练期间的典型情况。
- **高熵（远大于 2，接近 log(词汇表大小)；这里：log(7) = 1.9459）** 意味着概率分布在许多 token 上。在这种情况下，模型高度不确定，行为接近随机。

我们可以在代码中按如下方式计算熵：

**清单 7.4** 计算熵

```python
probs = torch.softmax(logits, dim=-1)
logprobs = torch.log_softmax(logits, dim=-1)
entropy = torch.sum(-(probs * logprobs))
print("Probs:", probs)
print("Entropy:", entropy)
```

结果输出为：

```
Probs: tensor([0.1295, 0.0090, 0.2522, 0.0665, 0.0341, 0.4912, 0.0175])
Entropy: tensor(1.3700)
```

---

如前所述，熵量化了模型在词汇表上的概率分布有多分散。在这个特定示例中，1.37 的熵表示中等程度的不确定性。一个 token（概率为 0.4912）明显占主导，但其他 token 仍然具有有意义的概率。

请注意，在前面的代码清单中，我们分别使用 `torch.softmax()` 和 `torch.log_softmax()` 计算 `probs` 和 `logprobs`。`torch.log_softmax()` 函数结合了两个独立的函数调用：`torch.log(torch.softmax())`。而 `torch.exp()` 函数是 `torch.log()` 的逆运算。这意味着，如果我们只有 `logprobs`，我们也可以通过应用 `torch.exp(logprobs)` 来计算 `probs`，如下所示：

```python
print("Probs:", torch.exp(logprobs))
```

输出与之前的 `probs` 值相同：

```
Probs: tensor([0.1295, 0.0090, 0.2522, 0.0665, 0.0341, 0.4912, 0.0175])
```

---

基于这个想法，我们可以将上一章的 `sequence_logprob` 函数扩展为 `sequence_logprob_and_entropy` 函数，该函数同时返回 logprobs 和熵。

**清单 7.5** 计算序列 logprob 和平均熵

```python
def sequence_logprob_and_entropy(model, token_ids, prompt_len):
    #A 以下代码与第 5 章中的 sequence_logprob 代码相同
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    targets = token_ids[1:]
    selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)
    #B 生成答案 token 的 logprob（在答案步骤上求和）
    selected_answer_logprobs = selected[prompt_len - 1:]
    logp_all_steps = torch.sum(selected_answer_logprobs)
    #C 以下是计算熵的新代码
    all_answer_logprobs = logprobs[:-1][prompt_len - 1:]
    if all_answer_logprobs.numel() == 0:  #D
        entropy_all_steps = logp_all_steps.new_tensor(0.0)
    else:
        all_answer_probs = torch.exp(all_answer_logprobs)  #E
        plogp = all_answer_probs * all_answer_logprobs  #F
        step_entropy = -torch.sum(plogp, dim=-1)  #G
        entropy_all_steps = torch.mean(step_entropy)  #H
    return logp_all_steps, entropy_all_steps
```

- #A 以下代码与第 5 章中的 sequence_logprob 代码相同
- #B 生成答案 token 的 logprob（在答案步骤上求和）
- #C 以下是计算熵的新代码
- #D 如果模型立即返回 EOS token，则触发的安全措施
- #E 将 logprob 转换为 prob
- #F 计算逐元素的 p * log p
- #G 通过对所有 plogp 值在词汇表上求和来计算单 token（生成步骤）的熵
- #H 在所有答案步骤上取平均，计算给定 LLM 答案的平均熵

---

请注意，前面的图说明了单个 rollout token 的熵计算（`step_entropy`），而 `sequence_logprob_and_entropy` 返回所有答案 token 的平均熵，即 `entropy_all_steps = torch.mean(step_entropy)`。例如，对于答案 "this is the LLM response"，`step_entropy` 指的是单个 token 位置的熵（例如在 "this" 处），而平均熵是在 "this is the LLM response" 中所有答案 token 上取的均值。

有了这个修改后的 `sequence_logprob_and_entropy` 函数，我们现在可以将熵用作模型生成行为的诊断工具。通过追踪生成答案 token 的平均熵，我们可以监控模型是随机行为（高熵）、保持探索性（中等熵），还是变得过于自信和确定性（低熵）。

特别是，我们预期随着模型变得更加自信，熵会逐渐降低。突然崩溃到非常低的熵可能是训练不稳定的警告信号。

### 7.3.3 绘制额外的 GRPO 指标

让我们在前面计算的优势统计量和熵的基础上，通过分析这些量在一次具体训练运行中的表现来进一步理解。为此，我们可以更新 `compute_grpo_loss` 以使用 `sequence_logprob_and_entropy`，然后在内部调用 `compute_grpo_loss` 的 `train_rlvr_grpo` 中打印并记录熵。为避免在此处重复冗长的代码清单，完整的修改版本在补充材料中提供，我们可以按如下方式下载：

```python
download_from_github(
    "ch07/03_rlvr_grpo_scripts_advanced/7_3_plus_tracking.py"
)
```

代码可以像之前使用的训练脚本一样运行：

```bash
uv run 7_3_plus_tracking.py \
    --steps 500  \
    --max_new_tokens 1024
```

---

由于训练需要很长时间，我们可以下载生成的 CSV 日志文件并按如下方式绘制新的优势统计量和熵：

```python
download_from_github(
    "ch07/02_logs/7_3_plus_tracking_metrics.csv"
)
plot_grpo_metrics(
    "7_3_plus_tracking_metrics.csv",
    columns=["reward_avg", "adv_avg", "adv_std", "entropy_avg"]
)
```

请注意，我们之前已经看过平均奖励，但这里为了比较目的再次包含它。得到的绘图如图 7.9 所示。这里，中等熵仍应理解为探索行为，只是比后来的高熵阶段随机性更小。

**图 7.9** GRPO 训练运行期间追踪的优势统计量和熵的可视化（旁边是我们之前追踪的平均奖励）。

---

让我们从分析图 7.9 中的优势开始。正如 GRPO 风格归一化所预期的那样，平均优势在整个训练过程中保持在零。由于优势是相对于组均值计算的，它们按设计求和为零。如前所述，在实践中，这个指标主要作为合理性检查。如果优势平均值偏离零，那将指向 bug 或归一化问题。

优势的标准差信息量更大。在训练早期，我们看到相对较高的方差，这表明模型生成的 rollout 质量范围很广。随着时间的推移，优势标准差逐渐降低并稳定，这意味着 rollout 在质量上变得更加相似。关键点是，只要优势标准差保持非零且相对稳定（意味着值变化不大），就仍有可用的学习信号和训练在进行。

接下来，让我们看看熵。在训练早期，熵相对较低且相当平坦，这表明模型行为在很大程度上是确定性的。在实践中，增加采样温度会导致更多样化的 rollout，尽管它不会改变模型本身的底层熵。

在训练后期，大约第 200 步之后，熵显著增加。这意味着下一个 token 的概率更加分散，模型行为更加随机。非常低的熵也可能是崩溃的迹象，模型反复产生相同或非常相似的输出。在此次运行中，考虑到熵与不断增加的平均奖励以及非消失的优势标准差一起，表明仍然是某种程度上的健康探索而非崩溃。这也与之前保持在 30%-40% 范围内的 MATH-500 评估准确率一致，这不是很好，但也不是接近零。

这里的主要要点是，每个指标讲述的故事略有不同，当综合考虑和结合上下文时，它们最有用。

## 7.4 使用裁剪策略比率稳定序列级 GRPO

到目前为止，我们主要专注于分析 GRPO 训练结果。接下来，我们开始对 GRPO 算法本身进行补充。我们到目前为止使用的版本是 GRPO 的简化形式，在本节中，我们在 GRPO 损失中引入裁剪策略比率，如图 7.10 的概览所示。

**图 7.10** 在绘制基本和高级 GRPO 训练指标后，我们现在修改 GRPO 算法并添加裁剪策略比率。

图 7.10 中提到的裁剪策略比率有助于限制过大的模型权重更新，并使训练更加稳定，尤其是在较长运行中。理想情况下，我们希望看到模型性能不会像之前那样明显下降。

### 7.4.1 计算裁剪策略比率

裁剪策略比率衡量当前策略（即正在训练的 LLM）相对于其早期版本的变化程度。具体而言，它比较更新步骤之前计算的序列 logprobs 与更新之后计算的序列 logprobs。你可以将其理解为问："如果我们之前调整了 LLM 的权重，它之前赋予某个答案的似然度是多少，现在它对同一答案的似然度又变化了多少？"

在图 7.11 所示的 GRPO 流程中，这对应于比较第 4 步的 logprobs（使用旧权重参数计算）与更新后模型产生的 logprobs（模型通过第 6 步更新）。

**图 7.11** 来自第 6 章的 GRPO 概览图。我们现在使用第 4 步的序列 logprobs 来计算策略比率和裁剪策略比率。

---

图 7.11 显示了通过正在训练的模型计算的 logprobs。然后通过第 6 步更新模型。策略比率是根据模型在两种不同状态下的 logprobs 计算的：权重更新前和权重更新后。一旦我们在代码中实现这个概念，这一点将变得更清楚。但首先，回顾一下，在第 6 章中我们按如下方式计算策略梯度损失：

**清单 7.6** 计算策略梯度

```python
# ...  #A 为简洁起见省略计算 rollout 的代码
#B
rewards = torch.tensor([1., 1., 0., 0.])
#C
logprobs = torch.tensor([-7.9243, -20.1546, -16.6130, -23.3677])
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
pg_loss = -(advantages.detach() * logprobs).mean()
print("Policy gradient loss:", pg_loss)
```

- #A 为简洁起见省略计算 rollout 的代码
- #B 计算奖励
- #C 计算序列 logprobs

得到的策略梯度值，也如图 7.11 所示，是 -2.5764。

接下来，我们计算策略比率和裁剪策略比率，如图 7.12 所示。

**图 7.12** 根据"新"和"旧" logprobs 计算添加到 GRPO 的策略比率（ratio）和裁剪策略比率（clipped ratio）。

---

策略比率和裁剪策略比率通过比较策略先前版本（"旧" logprobs）与当前策略（"新" logprobs）的对数概率来计算。数学推导超出了本章的范围，但感兴趣的读者可以在 Proximal Policy Optimization Algorithms 论文（`https://arxiv.org/pdf/1707.06347`）中找到更多细节。

对于图 7.12 中所示计算的以下基于代码的说明，我们重用上述示例中的 `logprobs` 作为 `old_logps`，假设已经进行了模型更新，然后使用更新后的模型计算相应的 `new_logps`。在实践中，两者都会使用相同的提示来计算，以确保旧策略和当前策略之间的公平、一对一比较：

**清单 7.7** 计算策略比率和裁剪策略比率：

```python
new_logps = logprobs  #A
old_logps = torch.tensor([
    -10.9243,   # -7.9243
    -20.3546,   # -20.1546
    -14.6130,   # -16.6130
    -23.3677,   # -23.3677
])
log_ratio = new_logps - old_logps
ratio = torch.exp(log_ratio)
clip_eps = 10.0
clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
print("Ratio:        ", ratio)
print("Clipped ratio:", clipped_ratio)
```

- #A `new_logps` 的值在注释中并排显示

---

得到的未裁剪和裁剪策略比率如下所示：

```
Ratio:         tensor([20.0855,  1.2214,  0.1353,  1.0000])
Clipped ratio: tensor([11.0000,  1.2214,  0.1353,  1.0000])
```

`ratio` 告诉我们旧 logprobs 和新 logprobs 之间的差异。如果它们相同，比率就是 1.0。

`clipped_ratio` 用于计算策略梯度损失的裁剪版本，限制新策略在单次更新中允许偏离旧策略的程度。具体而言，如果新模型突然对某个 rollout 分配的概率比旧模型高得多或低得多，原始比率可能变得非常大或非常小。没有裁剪的话，这会大幅缩放优势项，可能导致非常大的梯度步，从而可能使训练不稳定。

使用前面代码示例中的 `clip_eps` 参数，在实践中，通常将比率限制在 `1 ± clip_eps` 范围内。例如，DeepSeek-R1 使用 `clip_eps = 10`，对应非常弱的裁剪，而其他 RL 训练设置（例如使用 PPO 算法的人类反馈强化学习）通常使用小得多的值，例如 0.1，这导致激进的裁剪，从而大幅减小每步策略变化。

---

正如我们所见，基于这个非常宽松的 `clip_eps` 值，只有第一个值从 20.0855 被裁剪到 11.0000。

> **注意** `clip_eps` 中的 "eps" 是 epsilon（ε）的缩写，这是一个希腊字母，在数学中常用来表示一个小的正数。

接下来，我们在损失计算中应用 `ratio` 和 `clipped_ratio`。之前，我们直接将优势乘以 logprobs。现在，我们改为按策略比率缩放优势，并使用裁剪目标来限制每个 rollout 对更新的影响：

**清单 7.8** 计算裁剪策略梯度损失

```python
adv = advantages.detach()  #A
unclipped = ratio * adv
clipped = clipped_ratio * adv
obj = torch.where(
    adv >= 0,  #B
    torch.minimum(unclipped, clipped),  #C
    torch.maximum(unclipped, clipped),  #D
)
clipped_pg_loss = -torch.mean(obj)
policy_ratio = torch.mean(ratio)
print("Clipped policy gradient loss:", clipped_pg_loss)
print("Policy ratio:", policy_ratio)
```

- #A 将优势视为固定的学习信号（不通过奖励反向传播）
- #B 根据优势信号选择更保守的更新
- #C 限制大的正向更新
- #D 限制大的负向更新

---

得到的裁剪策略梯度损失是 -2.3998，略低于我们本节之前计算的常规策略梯度损失（-2.5764）。这个小差异是合理的，因为在我们的示例中，只有其中一个策略比率通过宽松的 `clip_eps=10.0` 比率被有效裁剪。一般来说，这种裁剪可以防止可能使训练不稳定过于激进的更新。大的、激进的更新可能在单步中将策略推得太远，这可能导致 token 概率的大幅变化，进而导致奖励崩溃和熵崩溃，或其他相关问题。

此外，我们还计算平均策略比率 `policy_ratio = torch.mean(ratio)`，我们可以在训练运行中追踪和报告它。这里，结果是一个相对较大的 5.6106。这个示例中的大策略比率意味着更新后的策略（模型）对采样 token 分配的概率比之前的策略高得多。

### 7.4.2 使用裁剪策略比率训练

我们可以应用前面清单 7.7 和 7.8 中讨论的裁剪策略比率和策略损失计算，看看它们是否有助于稳定训练。

修改后的代码可以从补充材料中下载：

```python
download_from_github(
    "ch07/03_rlvr_grpo_scripts_advanced/7_4_plus_clip_ratio.py"
)
```

我们可以像之前一样运行它：

```bash
uv run 7_4_plus_clip_ratio.py \
    --steps 500  \
    --max_new_tokens 1024
```

---

如前一节所讨论的，策略比率比较在当前策略下计算的 logprobs 与策略先前版本的 logprobs。

在实践中，这意味着我们首先需要一个批次采样响应，其 `old_logps` 在固定的参考策略下测量。然后我们可以将那些相同的响应与在优化过程中变化的当前模型进行比较，以形成裁剪策略比率更新。DeepSeek-R1 通过生成一个大的 rollout 池（8,192 个响应）一次，固定这些响应，然后在 16 个小批次中遍历它们来大规模实现这一点。在高层次上，过程如下：

1. 复制当前模型作为参考策略
2. 使用参考策略采样一个大的 rollout 池
3. 将这些固定的 rollout 分成小批次
4. 对于每个小批次：
5. 在参考模型下计算 `old_logps`
6. 在当前模型下计算 `new_logps`（在每个小批次更新后变化）
7. 更新当前模型
8. 更新参考模型

由于资源限制，我们无法生成像 DeepSeek-R1 那样多的 rollout 和小批次，因此我们使用一个较小的近似。我们不是生成一个巨大的固定 rollout 池并将其切片成小批次，而是生成一小批 rollout，在该批次上更新模型，然后用新的一批重复该过程。所以，`7_4_plus_clip_ratio.py` 脚本中的过程如下：

1. 复制当前模型作为参考策略
2. 使用参考策略采样 8 个 rollout
3. 然后，我们：
4. 在参考模型下计算 `old_logps`
5. 在当前模型下计算 `new_logps`
6. 更新当前模型
7. 用不同的 rollout 重复步骤 2 和 3（默认再重复一次），而不是使用小批次
8. 更新参考模型

---

简而言之，DeepSeek-R1 生成一个巨大的 rollout 池一次，然后在刷新参考模型之前跨小批次重用它。在我们的代码中，我们通过多次生成一小批 rollout 来更廉价地模仿相同的想法。

`7_4_plus_clip_ratio.py` 脚本训练运行的日志文件可以下载并可视化如下：

```python
download_from_github(
    "ch07/02_logs/7_4_plus_clip_ratio_metrics.csv"
)
plot_grpo_metrics(
    "7_4_plus_clip_ratio_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]
)
```

图 7.13 显示了得到的绘图。

**图 7.13** 使用裁剪策略比率的 GRPO 训练运行的选定指标。

---

与之前的训练运行相比，图 7.13 中使用裁剪策略比率的训练确实更加稳定，因为在第 400 步左右平均奖励或评估准确率没有明显下降。可选地，我们还可以绘制和检查其他指标，例如 `"policy_ratio"`、`"adv_avg"`、`"adv_std"`、`"entropy_avg"`，这留给读者作为练习。

## 7.5 使用 KL 项控制模型变化程度

此前，为了便于讲解，我们跳过了 GRPO 算法的 KL 损失项。为了完整实现 GRPO 算法，我们现在添加 KL 损失项，如图 7.14 所示。

> **注意** 熟悉强化学习的读者可能会认识到，到目前为止我们实现的本质上就是带有组归一化优势的 REINFORCE。

**图 7.14** 实现 KL 损失项，这是原始 GRPO 算法的一部分。

---

KL 是 Kullback-Leibler 散度的缩写，衡量当前策略（即正在训练的 LLM）与参考策略（通常是训练开始时的原始模型）的偏离程度。

KL 项充当约束（技术上称为正则化器），阻止过大的更新，并使模型保持接近其原始行为，以防止剧烈或不稳定的改变。这与我们之前介绍的裁剪策略比率密切相关。使用裁剪策略比率，我们限制了每个更新步骤可以有多大。KL 项是另一个控制模型可以变化多少的机制。虽然裁剪策略比率更侧重于限制步骤之间的变化，但 KL 项通常用于更长期地控制训练轨迹上的变化。

### 7.5.1 实现 KL 损失项

类似于我们计算裁剪策略比率时，计算 KL 项涉及我们比较的两组 logprobs，如图 7.15 所示。

**图 7.15** 添加了 KL 损失项计算的 GRPO 算法概览。

图 7.15 显示了作为 GRPO 算法一部分的 KL 损失项计算。KL 损失项计算涉及 logprobs 和参考 logprobs 之间的比较。它通过求这些对应 logprobs 之间的差异之和来计算。较小的 KL 损失项值表示当前模型和参考模型输出相对相似的 logprobs，意味着当前模型与参考模型相比变化不大。这通常对稳定性是可取的，因为它防止行为的大的、突然的变化。如果 KL 项在太长时间内保持太小，它也可能表明这里的学习有限。

---

得到的 KL 损失项然后被加到策略梯度损失中以计算总损失，因此即使导致高奖励，也会惩罚在训练期间强烈增加与参考模型差异（散度）的权重更新。

在裁剪策略比率部分，我们在每次迭代中替换参考模型。对于 KL 损失项，用于计算参考 logprobs 的参考模型是训练开始时的原始模型，要么保持固定，要么相对很少替换（在 DeepSeek-R1 的情况下，每 400 步）。

以下代码说明了如何修改 `compute_grpo_loss` 函数以包含此 KL 损失项。为避免代码重复，清单 7.9 中的代码没有显示完整的修改后的 `compute_grpo_loss_with_kl`，而是仅聚焦于我们必须对 `compute_grpo_loss` 函数所做的更改。

新添加的内容用注释标记或位于 `if kl_coeff` 块内。

**清单 7.9** 向 GRPO 损失计算添加 KL 项

```python
import copy

kl_coeff = 0.0  #A
if kl_coeff:  #B
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
else:
    ref_model = None

def compute_grpo_loss_with_kl(
    model,
    ref_model,  #C
    # ...
    kl_coeff=0.02,  #D
):
    roll_logps, roll_ref_logps, roll_rewards, samples = [], [], [], []
    # ...
    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            # ...
        )
        if kl_coeff:  #E
            with torch.no_grad():
                ref_logp = sequence_logprob(
                    ref_model, token_ids, prompt_len
                )
        else:
            ref_logp = None
        reward = reward_rlvr(text, example["answer"])
        roll_rewards.append(reward)
        roll_logps.append(logp)
        if kl_coeff:
            roll_ref_logps.append(ref_logp)
    # ...
    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)
    if kl_coeff:
        ref_logps = torch.stack(roll_ref_logps).detach()
    pg_loss = -(advantages.detach() * logps).mean()
    if kl_coeff:  #F
        kl_loss = kl_coeff * torch.mean(logps - ref_logps)
    else:
        kl_loss = torch.tensor(0.0, device=logps.device)
    loss = pg_loss + kl_loss  #G
```

- #A `kl_coeff = 0.0` 停用 KL 项，代码与之前类似
- #B 复制原始模型并禁用梯度，使其不被更新
- #C 新增：传递参考模型
- #D 新增：指定 KL 强度
- #E 使用参考模型计算 logprobs
- #F 计算 KL 损失项
- #G 新增：将 KL 损失项加到策略梯度损失上

---

请注意，添加 KL 损失项会增加资源使用，因为我们现在需要保留原始模型的额外副本。此惩罚的强度由 `kl_coeff` 控制。较大的 `kl_coeff` 值对当前策略允许偏离参考模型的程度施加了更强的约束。

清单 7.9 顶部的 `kl_coeff = 0.0` 设置仅是为了如果我们在代码环境中运行这个缩略的代码片段，它不会崩溃。在实践中，`kl_coeff` 通常是 0.001 到 0.05 之间的小值。

### 7.5.2 使用 KL 损失项训练

以下来自补充材料的脚本实现了我们之前讨论的 KL 损失项代码修改：

```python
download_from_github(
    "ch07/03_rlvr_grpo_scripts_advanced/7_5_plus_kl.py"
)
```

类似于之前，我们可以按如下方式运行它：

```bash
uv run 7_5_plus_kl.py \
    --steps 500  \
    --max_new_tokens 1024
```

---

让我们看看使用上述设置执行的训练运行的日志文件：

```python
download_from_github(
    "ch07/02_logs/"
    "7_5_plus_kl_metrics.csv"
)
plot_grpo_metrics(
    "7_5_plus_kl_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]
)
```

得到的绘图如图 7.16 所示。

**图 7.16** 添加 KL 损失项后 GRPO 训练运行的选定指标。这里，损失（左上方）指的是总 GRPO 损失，即策略梯度损失和 KL 损失项的总和。

---

我们可以看到，前五十步将模型性能从略高于 15% 的准确率 noticeably 提高到约 40% 的准确率。然后我们看到总崩溃，损失值爆炸，平均奖励趋向于 0。这伴随着评估准确率的崩溃，也趋向于 0%。

对手动检查生成的响应，模型最终开始产生无意义的文本和随机 token，例如 "framework.raises Self profess(import" 。

这种行为有几个可能的原因。

首先，KL 项使用答案 token 上的 logprobs 总和计算，即 `kl_loss = kl_coeff * mean(new_logps - ref_logps)` 。因此，更长的序列自然导致更大的 KL 值。这可能无意中鼓励非常长的输出，我们也在增长的响应长度中观察到，并可能使训练不稳定。

---

其次，一旦奖励在第 200 步左右崩溃到 0，KL 项就成为唯一剩余的梯度信号来源。当所有奖励为零时，优势也为零，因此策略梯度损失不再贡献任何梯度。KL 项仍然活跃并开始主导更新。这意味着 KL 损失项可以将模型推向更高的熵和接近均匀的 token 分布，这解释了越来越随机的输出和下降到 0.0% 的评估准确率。如果你愿意，你可以绘制其他评估指标，例如 `"policy_ratio"`、`"adv_avg"`、`"adv_std"`、`"entropy_avg"`，并确认熵变得非常大。

虽然 KL 项是 GRPO 算法的标准部分，这也是我们在这里介绍它的原因，但最近的几项工作报告称，没有它模型可以训练得更好。例子包括 Dr. GRPO、Olmo 3 和 DeepSeek-V3.2，它们都观察到当 KL 项被减少或省略时稳定性和性能得到改善（有关更详细的参考文献，请参见附录 A）。

> **使用 KL 项提高训练稳定性**
>
> 或者，除了移除 KL 项外，还有几种方法可以提高训练稳定性。首先，我们可以将 `kl_coeff` 从 0.02 降低到更小的值（例如 0.001）以减弱 KL 惩罚，尽管在我的实验中这并不足够。
>
> 由于我们看到响应长度不断增长直到达到最大值，另一个选项是通过除以生成的 token 数量来长度归一化序列 logprobs，这近似平均每个 token 的 logprobs。这可以减少增加响应长度以累积更大绝对 logprob 差异的激励。
>
> 第三个选项是使用基于重要性采样的重加权 KL 项，如 DeepSeek-V3.2 所提出的。不使用
>
> ```python
kl_loss = kl_coeff * torch.mean(logps - ref_logps)
```
>
> 我们使用
>
> ```python
kl_loss = (
    kl_coeff * torch.mean(torch.exp(new_logps - old_logps)
    * (new_logps - ref_logps)
)
```
>
> 这里，重要性权重 `torch.exp(new_logps - old_logps)` 纠正了 rollout 由旧策略生成的事实，并且它 downweights 当前策略很少产生的样本，upweights 那些它经常产生的样本。
>
> 最后，我们可以添加一个小的格式奖励来防止优势崩溃到零（我们将在下一节重新讨论格式奖励）。

---

也就是说，对于数学数据，常见的建议是完全省略 KL 项（例如，Dr. GRPO、Olmo 3 和 DeepSeek-V3.2），这也简化了代码并减少了资源需求，因为我们不必在内存中保留额外的参考模型。

## 7.6 添加显式格式奖励

到目前为止，我们在训练模型时只使用了一种可验证奖励：答案正确性奖励。在实践中，这足以用 RLVR 训练推理模型。使用一个或多个辅助奖励也很常见，例如，检查生成答案是否遵循某些格式指南的格式奖励（图 7.17）。

**图 7.17** 实现格式奖励，鼓励模型生成 `<think>...</think>` token

具体来说，我们将聚焦于鼓励模型使用所谓的 `<think>...</think>` token 的格式奖励，如图 7.17 所示。

### 7.6.1 使用 `<think>` token

一些推理模型，如 DeepSeek-R1 和 Qwen3，使用特殊的 `<think>` 和 `</think>` token。这些 `<think>` token 是标记模型中间推理开始和结束的普通文本 token。在实践中，我们可以提示模型在 `<think> ... </think>` 内逐步写出推理，然后在这些标签外提供最终答案。这整个方法是可选的，但它使输出结构明确，并有助于将推理与最终结果分开。

---

为了开始说明这个概念，我们加载之前使用的基础模型的 tokenizer。

**清单 7.10** 加载基础模型 tokenizer

```python
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer

device = get_device()
model, tokenizer_base = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)
print(tokenizer_base.encode("<think>"))
```

接下来，让我们使用 tokenizer 编码 `<think>`：

输出是 `[13708, 766, 29]`，这意味着 tokenizer 将 `<think>` token 分解成几个子词 token，这表明 `<think>` token 不是 tokenizer 词汇表的一部分。

如今，在开发 tokenizer 和创建 LLM 的嵌入层时，开发人员通常会留下一些未使用的占位符 token ID，这些可以在微调模型时用于特定目的。在这种情况下，Qwen3 基础模型 tokenizer 有一些空的占位符槽，我们可以用来添加 `<think>`：

```python
tokenizer_base._tok.add_special_tokens(
    ["<tool_response>", "</tool_response>", "<think>", "</think>"]
)
```

---

请注意，我们还在上面添加了 `<tool_response>` 和 `</tool_response>` token，以与 Qwen3 推理变体的 tokenizer 匹配，我们稍后会讨论。但首先，让我们检查修改后的基础 tokenizer 现在是否支持新添加的 `<think>` 和 `</think>` token：

```python
print(tokenizer_base.encode("<think>"))
print(tokenizer_base.encode("</think>"))
```

输出是单 token `[151667]` 和 `[151668]`，这表明 `<think>` 和 `</think>` token 现在是词汇表的一部分，不再被分解成子词 token。

这些新 token 也会被基础模型支持，该模型支持从 0 到 151935 的 token ID，因为它的词汇表大小为 151,936，我们可以通过打印嵌入层的大小来确认：

```python
print("Vocabulary size:", model.tok_emb.weight.shape[0])
```

或者，除了修改基础 tokenizer，推理模型变体的 tokenizer 已经原生支持这些 `<think>` token，这意味着我们不必手动修改 tokenizer。例如，类似于我们在第 2 章中所做的，我们可以单独加载 tokenizer：

**清单 7.11** 加载推理模型 tokenizer

```python
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
from reasoning_from_scratch.qwen3 import download_qwen3_small

download_qwen3_small(
    kind="reasoning", tokenizer_only=True, out_dir="qwen3"
)
tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
print(tokenizer.encode("<think>"))
print(tokenizer.encode("</think>"))
```

---

现在，让我们使用推理 tokenizer（`tokenizer`）而不是 `tokenizer_base` 来编码 token ID，类似于我们之前所做的：

类似于我们使用 `tokenizer_base` 时，这返回 `[151667]` 和 `[151668]` 。

现在我们有了一个支持这些 `<think> </think>` token 的 tokenizer，我们可以然后开发一个奖励函数，检查 LLM 是否使用这些 token，如果使用则提供非零奖励，如图 7.18 所示。

**图 7.18** 使用之前的正确性奖励（左）和正确性加格式奖励（右）。

类似于图 7.18，我们开发一个函数，如果输出中同时包含 `<think>` 和 `</think>` 且顺序正确，则返回 1.0；否则返回 0.0：

**清单 7.12** 实现格式奖励函数

```python
def reward_format(
    token_ids,
    prompt_len,
    start_think_id=151667,
    end_think_id=151668,
):
    try:
        gen = token_ids[prompt_len:].tolist()
        return float(
            gen.index(start_think_id) < gen.index(end_think_id)
        )
    except ValueError:
        return 0.0

prompt = "Calculate ..."
rollout = "Let's ... <think> ... </think> ..."
token_ids = tokenizer.encode(prompt + rollout)
reward_format(
    token_ids=torch.tensor(token_ids),
    prompt_len=len(tokenizer.encode(prompt))
)
```

---

让我们在一个简单的样本文本上试试：

运行以下代码按预期返回 1.0，因为答案（rollout）包含 `<think>` 和 `</think>` 。

> **练习 7.1：测试 `<THINK>` 格式奖励**
>
> 运行示例并确认它返回 1.0，因为 `<think>` 和 `</think>` 按正确顺序出现。
>
> 现在通过以下方式修改 `rollout` 文本：
>
> - 在 `<think>` `</think>` 标签中引入拼写错误
> - 反转标签顺序
> - 移除一个标签
>
> 验证每种情况下奖励都变为 0.0。

---

使用这个新的 `reward_format` 函数，我们可以使用格式奖励作为额外的奖励信号来训练模型。例如，我们可以将第 6 章的 `compute_grpo_loss` 转换为 `compute_grpo_loss_plus_format_reward` 函数如下（更改在注释中高亮显示）：

**清单 7.13** 使用格式奖励更新 GRPO 损失函数

```python
#A 一个设置参数，让我们可以调节格式奖励相对于现有正确性奖励的大小
def compute_grpo_loss_plus_format_reward(
    # ...
    format_reward_weight=1.0,  #A
):
    # ...
    logp = sequence_logprob(model, token_ids, prompt_len)
    rlvr_reward = reward_rlvr(text, example["answer"])
    #B 新的格式奖励行，修改之前的奖励，使其现在由正确性奖励和格式奖励组成
    format_reward = reward_format(token_ids, prompt_len)         # NEW
    reward = rlvr_reward + format_reward_weight * format_reward  # NEW
    # ...
```

- #A 一个设置参数，让我们可以调节格式奖励相对于现有正确性奖励的大小
- #B 新的格式奖励行，修改之前的奖励，使其现在由正确性奖励和格式奖励组成

此外，我们可以修改 `render_prompt` 以引导模型显式地发出这些 `<think>` 和 `</think>` token：

**清单 7.14** 用于 `<think>` token 的更新提示渲染函数

```python
def render_prompt_with_think_tokens(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "When solving the problem, first write your reasoning inside <think> and
"
        "</think> tags.\n"
        "Then write the final result on a new line in the exact format:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template
```

---

理论上，前面代码清单 7.11 到 7.14 中的修改应该足以训练一个带有额外格式奖励的推理模型。下一节将解释这在实践中如何工作并讨论额外的注意事项。

### 7.6.2 训练模型发出 `<think>` token

类似于前面的章节，训练包含格式奖励的推理模型所需的修改实现在一个脚本中，我们可以从补充材料下载：

```python
download_from_github(
    "ch07/03_rlvr_grpo_scripts_advanced/7_6_plus_format_reward.py"
)
```

关于这个脚本有一些注意事项需要记住。例如，如果我们用修改后的提示训练基础模型以发出 `<think>` 和 `</think>` token，如清单 7.14 所示，结果会相当差，因为基础模型以前从未见过这些 token。

相反，引入这些新 token 的正确方式是首先在包含 `<think>` 和 `</think>` token 的数据上预训练或指令微调模型（指令微调在下一章中介绍）。

所以，对于本章，我们聚焦于强化学习，我们训练现有的 "reasoning" 模型变体，它已经熟悉 `<think>` token。（这是我们在第 3 章介绍的相同的 "reasoning" 变体）。

类似于之前，我们可以按如下方式运行脚本：

```bash
uv run 7_6_plus_format_reward.py \
    --steps 500  \
    --max_new_tokens 1024
```

---

请注意，此脚本已配置为使用 "reasoning" 模型变体而不是 "base" 模型。

此外，类似于之前，我们可以下载此次运行的日志文件以进行进一步分析，因为运行实验可能资源密集且昂贵：

```python
download_from_github(
    "ch07/02_logs/7_6_plus_format_reward_metrics.csv"
)
plot_grpo_metrics(
    "7_6_plus_format_reward_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]
)
```

得到的绘图如图 7.19 所示。

**图 7.19** 带有格式奖励的 GRPO 训练运行的基本指标。

---

正如我们在图 7.19 中看到的，模型起初有所改善，但随后在评估准确率方面开始变差，而平均奖励保持大致恒定。评估准确率的下降似乎与模型此后产生的较短响应长度相关。

为了进一步调查，让我们也看看一些额外的指标：

```python
plot_grpo_metrics(
    "7_6_plus_format_reward_metrics.csv",
    columns=["reward_avg", "format_reward_avg", "adv_std", "entropy_avg"],
)
```

结果如图 7.20 所示。

**图 7.20** 带有格式奖励的 GRPO 训练运行的额外指标。

---

如图 7.20 所示，模型性能下降的一个可能解释是，模型从简单地遵循 `<think>` 标签格式中获得了太多奖励，如图 7.20 中的 `format_reward_avg` 绘图所示。换句话说，模型没有足够关注答案正确性（参见 `reward_avg`）。

一种潜在的解决方法是降低格式奖励的强度，例如将默认的 `format_reward_weight` 从 1.0 降低到 0.1。另一种选择是使格式奖励有条件，即仅在答案也正确时才应用。在当前设置中，无论正确性如何都会授予格式奖励。

> **关于 LOSS 和 KL_LOSS 的说明**
>
> 此实验使用 `7_6_plus_format_reward.py` 及其默认设置 `--kl_coeff 0.0` 和 `--inner_epochs 1`。因为 `--kl_coeff` 设置为 0.0，KL 惩罚被禁用，所以记录的 `kl_loss` 在整个运行期间保持为 0。
>
> 如果你想使用第 7.5 节中相同的设置启用 KL 正则化，请运行：
>
> ```bash
uv run 7_6_plus_format_reward.py \
    --steps 500 \
    --max_new_tokens 1024 \
    --kl_coeff 0.001 \
    --inner_epochs 2
```
>
> 记录的损失（图 7.19）在默认 7.6 设置下也为 0。这是因为运行仅使用一次内部更新（`--inner_epochs 1`），并将模型与同一步的自身副本进行比较。因此，策略比率从 1 开始，由于优势被归一化为均值 0，标量策略损失评估为 0。
>
> 所以在这种情况下，`loss = 0` 是默认设置导致的计算伪影。即，它主要反映运行使用单个内部 epoch 且没有 KL 项，这使得记录的标量损失没有信息量。
>
> 这里更有用的监控量是诸如 `reward_avg`、`format_reward_avg`、`adv_std` 和 `entropy_avg` 等指标。

---

> **练习 7.2：使格式奖励有条件**
>
> 修改格式奖励计算
>
> `reward = rlvr_reward + format_reward_weight * format_reward`
>
> 使得格式奖励仅在正确性奖励（`rlvr_reward`）非零时才授予。使用有条件格式奖励重复训练运行。有条件格式奖励会改变训练行为和最终准确率吗？
>
> 注意：由于运行这些实验可能资源密集，附录 B 中提供了参考结果。

## 7.6.3 更多 GRPO 修改、技巧与窍门

此前，为了入门目的，我们实现了一个没有 KL 损失项、裁剪策略比率和格式奖励的 GRPO 版本。然后，我们添加了这些额外的概念，以建立对原始 GRPO 算法的坚实基础和理解，该算法是大多数现代推理训练框架的基础。图 7.21 简要展示了这下一步，即额外的修改、技巧和窍门。

**图 7.21** 本章最后一节概述了近几个月出现的一些额外 GRPO 修改。

---

就使用 GRPO 进行 RLVR 而言，我们了解到有许多旋钮可以调整。除了更改基本设置，如学习率、最大响应长度、提示格式等，还有近乎无限数量的设置和修改可以尝试，其中彻底探索单个修改或少量修改对于一个小型研究团队来说可能是一个长达数月或数年的研究项目。

算法不断变化和发展是意料之中的。然而，重要的是理解其背后的基本原理和总体思路，以便也理解和欣赏这些改进。

在 DeepSeek-R1 成功之后的几个月里，许多研究人员提出了对原始 GRPO 算法的更改，提高了训练稳定性和最终性能。为了提供几项建议改进的概览，以下是从 DAPO、Dr. GRPO、DeepSeek-V3.2 等中选择的（你可以在附录 A 中找到完整的参考文献和链接列表）：

1. 零梯度信号过滤（DAPO）
2. 主动采样（DAPO）
3. 从序列级损失切换到 token 级损失（DAPO）
4. 无 KL 损失（DAPO 和 Dr. GRPO）
5. 更高裁剪（DAPO）
6. 截断重要性采样（VERL）
7. 无标准差归一化（Dr. GRPO）
8. 使用领域特定 KL 强度的 KL 调优；数学为零（DeepSeek-V3.2）
9. 重加权 KL（DeepSeek-V3.2）
10. 离线序列掩码（DeepSeek-V3.2）
11. 保留 top-p / top-k 的采样掩码（DeepSeek-V3.2）
12. 保留原始 GRPO 优势归一化（DeepSeek-V3.2）
13. 聚合前的每奖励组归一化（GDPO）
14. 序列级重要性采样和裁剪（GSPO）
15. 裁剪重要性采样权重而非 token 更新（CISPO）

鉴于这里已经相当长的篇幅，关于这些修改的详细解释、代码示例和训练运行结果超出了本章的范围。如果你感兴趣，可以在补充材料 `https://github.com/rasbt/reasoning-from-scratch/tree/main/ch07/03_rlvr_grpo_scripts_advanced` 中找到这些内容。

## 7.7 总结

- 使用 GRPO 训练推理模型在较长运行中可能变得不稳定，即使实现正确且奖励最初有所改善。
- 解读 GRPO 训练需要联合追踪多个指标（平均奖励、响应长度、评估准确率、优势统计量和熵）。
- 基本指标如损失在 GRPO 中主要作为合理性检查，不应孤立地过度解读。
- 优势统计量提供有用的诊断：均值按设计应保持在零附近，而标准差反映学习信号的强度和稳定性。
- 熵衡量模型在生成期间的不确定性。非常低的熵可能预示崩溃，非常大的熵可能表明不稳定的更新和模型响应中的随机性。
- 裁剪策略比率限制策略在更新之间可以变化的程度，并可以在较长运行中显著改善训练稳定性。
- 添加 KL 散度项约束与参考模型的长期漂移，但当奖励崩溃时可能使训练不稳定。
- 对于数学推理任务，几个最近的系统报告通过完全省略 KL 项获得了更好的稳定性和性能。
- 辅助格式奖励可以改善响应结构，例如鼓励使用 `<think>` 和 `</think>` token。
- 超越原始 GRPO 算法，许多最近的扩展修改了优势归一化、重要性采样、裁剪策略和 KL 处理，以提高稳定性和效率。
