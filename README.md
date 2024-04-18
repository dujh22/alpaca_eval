# `<a href="https://tatsu-lab.github.io/alpaca_eval/" target="_blank"><img src="https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/docs/AlpacaFarm_small.png" width="35">``</a>` [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) : 遵循指令的语言模型自动评估器

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![discord](https://img.shields.io/badge/discord-server-blue?logo=discord&logoColor=white)](https://discord.gg/GJMxJSVZZM)

采用长度控制胜率的 **AlpacaEval 2.0** 与 [ChatBot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 的斯皮尔曼相关性为 **0.98**，而运行成本不到 **10 美元** OpenAI 信用点，运行时间不到 3 分钟。我们的目标是为聊天 LLM 提供一个基准：快速（< 5 分钟）、便宜（< 10 美元）、与人类高度相关（0.98）。下面是与其他基准的比较：

<p float="left" align="middle">
<img src="figures/chat_correlations_no_ae.png" alt="LC AlpacaEval is the most highly correlated benchmark with Chat Arena." width="500"/>
</p>

---

更新：

🎉 **长度控制胜率** 已发布并默认使用！这将把与 ChatBot Arena 的相关性从 0.93 提高到 0.98，同时大大降低了长度游戏性。原始胜率仍显示在网站和 CLI 上。更多详情 [点击此处](#length-controlled-win-rates).

🎉 **AlpacaEval 2.0** 已发布并默认使用！我们改进了自动注释器（更好、更便宜），并使用 GPT-4 预览作为基准。更多详情 [点击此处](#alpacaeval-20). 要使用旧版本，请设置环境变量 `IS_ALPACA_EVAL_2=False`.

---

<details open>
  <summary><b>目录</b></summary>

1. [概述](#overview)
2. [快速开始](#快速开始)
3. [排行榜和如何解读排行榜](#排行榜和如何解读排行榜)
   - [模型](#模型)
   - [评估](#评估)
4. [使用案例](#使用案例)
   - [评估一个模型](#评估一个模型)
   - [制作新的排行榜](#制作新的排行榜)
   - [制作一个新的评估器](#制作一个新的评估器)
5. [贡献](#贡献)
   - [提供模型](#提供模型)
   - [验证模型](#验证模型)
   - [提供一名评估员](#提供一名评估员)
   - [贡献一个评估集](#贡献一个评估集)
   - [提供完成功能](#提供完成功能)
6. [限制](#限制)
7. [分析](#其他分析和绘图)
   - [分析一个评估器](#分析一个评估器)
   - [分析一个评估数据集](#分析一个评估数据集)
8. [引用](#引用)
9. [更多信息](#更多信息)
   - [长度控制胜率](#长度控制胜率)
   - [AlpacaEval 2.0](#alpacaeval-20)
   - [数据发布](#数据发布)
   - [与羊驼农场AlpacaFarm的区别](#与羊驼农场AlpacaFarm的区别)
   - [相关工作](#相关工作)
   - [解释注释](#解释注释)
   - [主要更新](#主要更新)

</details>

# 概述

对指令遵循模型(例如，ChatGPT)的评估通常需要人工交互。这既耗时又昂贵，而且很难复制。AlpacaEval是一个基于llm的自动评估，它快速、廉价、可复制，并针对20K个人工注释进行了验证。它对模型开发特别有用。尽管我们对之前的自动评估管道进行了改进，但仍然存在基本的[限制](#限制)，比如对更长的输出的偏好。

AlpacaEval提供以下功能：

- [**排行榜**](https://tatsu-lab.github.io/alpaca_eval/): AlpacaEval评估集上常用模型的排行榜。**警告**:自动评估器(如GPT-4)可能偏向于产生较长输出的模型和/或在评估器底层模型上进行微调的模型(如GPT-4)。
- [**自动评估器**](#评估): 与人类高度一致的自动评估器(在20K个注释上进行了验证)。我们通过测量强大的LLM(例如GPT-4)倾向于该模型的输出而不是参考模型的输出的次数来评估模型。默认情况下，我们的评估器启用缓存和输出随机化。
- [**用于构建自动评估器的工具包**](#analysis): 一个简单的接口，用于构建高级自动评估器(例如，使用缓存、批处理或多注释器)并分析它们(质量、价格、速度、统计能力、偏差、方差等)。
- [**人类评估数据**](#数据发布)： 在 [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm/tree/main)评估集上给定模型和参考模型之间的 20K 人类偏好。其中 2.5K 为交叉注释（4 人注释相同的 650 个示例）。
- [**AlpacaEval数据集**](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval.json)：[AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm/tree/main)评估集的简化版，其中 "指令 "和 "输入 "合并为一个字段，参考输出更长。[此处有详细信息](#data-release)。

<details>
  <summary><b>何时使用或不使用 AlpacaEval？</b></summary>

**何时使用 AlpacaEval？**

我们的自动评估器是一种快速、廉价的人工评估工具，可替代简单任务的人工评估。
如果您需要快速运行许多评估，例如在模型开发过程中，它就非常有用。

**何时不使用 AlpacaEval？**

与其他任何自动评估工具一样，AlpacaEval **不应在高风险决策**（如决定是否发布模型）中取代人工评估。AlpacaEval 尤其受到以下事实的限制

(1) 评估集中的指令可能无法代表 LLM 的高级用法；

(2) 自动评价器可能存在偏差，例如偏重风格而非事实性；

(3) AlpacaEval 无法衡量模型可能造成的风险。

具体细节参照[限制](#限制)。

</details>

# 快速开始

要安装稳定版，请运行

```bash
pip install alpaca-eval
```

要安装夜间版本，请运行（[**Nightly Build**（每日构建）通常是开发者自行维护的一个版本。白天，开发者们将各自的修改提交到一个中心代码库，然后在晚上进行一次编译，生成这个版本。一般来说，**Nightly Build** 包含了最新的错误修复和新增功能，因此适合那些关注特定错误或喜欢使用最新版本的用户](https://www.taholab.com/8350)[1](https://www.taholab.com/8350)）

```bash
pip install git+https://github.com/tatsu-lab/alpaca_eval
```

之后你可以按照如下方式使用它:

```bash
export OPENAI_API_KEY=<your_api_key> # 有关更复杂的配置，例如使用 Azure 或切换客户端，请参阅 client_configs/README.md  
alpaca_eval --model_outputs 'example/outputs.json' 
```

这会将排行榜打印到控制台，并将排行榜和注释保存到与 `model_outputs` 文件相同的目录中。重要参数如下：

- **model_outputs** ： 一个 json 文件的路径，用于将模型的输出添加到排行榜中。每个字典应包含关键字 "instruction "和 "output"。
- **annotators_config**： 这是要使用的注释器。我们建议使用 `weighted_alpaca_eval_gpt4_turbo` （AlpacaEval 2.0 的默认值），该注释器与我们的人工注释有很高的一致性，上下文规模大，而且相当便宜。所有注释器的比较见 [此处](#评估)。
- **reference_outputs**：  参考模型的输出。格式与 `model_outputs`相同。默认情况下为 AlpacaEval 2.0 的 `gpt4_turbo`。
- **output_path**： 用于保存注释和排行榜的路径。

如果没有模型输出，可以使用  [`evaluate_from_model`](https://github.com/tatsu-lab/alpaca_eval/tree/main#evaluating-a-model)  并传递本地路径或HuggingFace模型或标准 API（OpenAI、Anthropic、Cohere、google......）中的模型名称。其他命令:

<details open>
  <summary><code>>>> alpaca_eval -- --help</code></summary>

```
简述
    alpaca_eval 命令

命令
    COMMAND 是以下命令之一：

     evaluate（评估）
       根据输出结果评估模型。如果没有指定命令，这是默认的入口点。

     evaluate_from_model（从模型评估）
       从 HuggingFace 或 API 提供程序评估模型。这是对 `evaluate` 的封装，其中包括从所需模型生成。

     make_leaderboard（制作领航板）
       为给定的数据集/评估器/模型生成集预先计算并保存整个排行榜。

     analyze_evaluators（分析评估器）
       分析评价器并填充评价器排行榜（与人类的一致性、速度、价格......）。
```

</details>

有关每个函数的更多信息，请使用  `alpaca_eval <command> -- --help`.

# 排行榜和如何解读排行榜

## 模型

我们的排行榜是在 [AlpacaEval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) 数据集上计算得出的。
我们使用不同的基线模型和自动标注器为重要模型预先计算了排行榜。
我们的两个主要排行榜（"AlpacaEval 2.0 "和 "AlpacaEval"）[在这里](https://tatsu-lab.github.io/alpaca_eval/)。
"AlpacaEval 2.0 "使用加权 Alpaca_eval_gpt4_turbo 作为注释器，gpt4_turbo 作为基线。
"AlpacaEval "的注释器使用 alpaca_eval_gpt4，基准线使用 text_davinci_003。
有关所有预计算的排行榜，请参阅[此处](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/leaderboards)。
稍后我们还会展示如何将模型添加到排行榜 [将模型添加到排行榜](https://github.com/tatsu-lab/alpaca_eval#evaluating-a-model) ，以及如何为为你的评估器/数据集构造 [一个新的排行榜](https://github.com/tatsu-lab/alpaca_eval#making-a-new-leaderboard)。
所有模型的配置可以在 [这里](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/models_configs)找到。

**AlpacaEval 最小排行榜**:

|                       | Win Rate | Std Error |
| :-------------------- | -------: | --------: |
| gpt4                  |     95.3 |       0.7 |
| claude                |     88.4 |       1.1 |
| chatgpt               |     86.1 |       1.2 |
| guanaco-65b           |     71.8 |       1.6 |
| vicuna-13b            |     70.4 |       1.6 |
| text_davinci_003      |     50.0 |       0.0 |
| alpaca-farm-ppo-human |     41.2 |       1.7 |
| alpaca-7b             |     26.5 |       1.5 |
| text_davinci_001      |     15.2 |       1.2 |

<details>
  <summary><b>这些指标究竟是如何计算出来的？</b></summary>
**获胜率**：获胜率衡量的是模型输出比参考输出（AlpacaEval 的 `test-davinci-003` 和 AlpacaEval 2.0 的 `gpt4_turbo`）更受青睐的时间比例。
更具体地说，为了计算胜率，我们从 ApacaEval 数据集中收集了所需模型在每条指令上的输出对。
然后，我们将每个输出与我们的参考模型（例如 `text-davinci-003`）在同一指令上的输出配对。
然后，我们会询问我们的自动评估员他们更喜欢哪种输出。
参见 [AlpacaEval&#39;s](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4) 和 [AlpacaEval 2.0&#39;s](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo) 的提示和配置，特别是我们会随机调整输出的顺序，以避免位置偏差。
然后，我们对数据集中的所有指令的偏好进行平均，得出模型相对于基线的胜率。
如果两个输出完全相同，我们会对两个模型使用一半的偏好。

**标准误差**：这是胜率的标准误差（按 N-1 归一化），即不同指令的平均偏好。

</details>

<details>
  <summary><b>有关自动注释器的详细信息： <code>alpaca_eval_gpt4</code></b></summary>

我们的 alpaca_eval_gpt4（参见 [configs](#https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/configs.yaml#L5)）注释器对偏好进行平均，偏好的获取方式如下：

1. 它接收一条指令和一对输出（来自期望模型和参考模型）
2. 如果已经计算出这三者的偏好值，则返回该偏好值（即使用缓存）
3. 随机调整输出的顺序，以避免位置偏差
4. 将指令和输出格式化为零次提示 [following zero-shot prompt](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/alpaca_eval.txt),，要求按偏好顺序排列输出
5. 使用 GPT4 完成提示，温度=0
6. 解析完成后的偏好并返回 

注释器是 [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) and [Aviary](https://github.com/ray-project/aviary/tree/master) 评估器的混合体（并深受其影响）。
特别是，我们使用了与 AlpacaFarm 相同的代码（缓存/随机化/超参数），但使用了与 Aviary 类似的排名提示。
我们对 Aviary 的提示进行了修改，以减少较长输出的偏差。
详情请参见[相关工作](#相关工作)。

在 AlpacaEval 2.0 中，我们使用了 weighted_alpaca_eval_gpt4_turbo，它使用 logprobs 计算连续偏好，并使用 GPT4_turbo 作为模型（参见配置 [configs](#https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo/configs.yaml)）。

</details>

## 评估

我们在 AlpacaEval 集上评估了不同的自动注释器，将其与我们收集的 2.5K 条人类注释 [human annotations](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_farm_human_crossannotations.json)（每条指令约有 650 条，其中包含 4 条人类注释）进行比较。
下面我们展示了我们建议的评估器（weighted_alpaca_eval_gpt4_turbo,alpaca_eval_gpt4）、先前的自动评估器（[`alpaca_farm_greedy_gpt4`](https://github.com/tatsu-lab/alpaca_farm),[`aviary_gpt4`](https://aviary.anyscale.com/),[`lmsys_gpt4`](https://chat.lmsys.org/)）、人类（humans）以及具有基本相同提示的不同基础模型（gpt4,claude,text_davinci_003,chatgpt_fn,guanaco_33b,chatgpt）的指标。
有关开箱即用的所有评估器的配置及其相关指标，请参见[此处](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs)。

|                               | Human agreement | Price [$/1000 examples] | Time [seconds/1000 examples] | Spearman corr. | Pearson corr. | Bias | Variance | Proba. prefer longer |
| :---------------------------- | --------------: | ----------------------: | ---------------------------: | -------------: | ------------: | ---: | -------: | -------------------: |
| alpaca_eval_gpt4              |            69.2 |                    13.6 |                         1455 |           0.97 |          0.93 | 28.4 |     14.6 |                 0.68 |
| alpaca_eval_cot_gpt4_turbo_fn |            68.6 |                     6.3 |                         1989 |           0.97 |          0.90 | 29.3 |     18.4 |                 0.67 |
| alpaca_eval_gpt4_turbo_fn     |            68.1 |                     5.5 |                          864 |           0.93 |          0.82 | 30.2 |     15.6 |                 0.65 |
| gpt4                          |            66.9 |                    12.5 |                         1037 |           0.88 |          0.87 | 31.5 |     14.6 |                 0.65 |
| alpaca_farm_greedy_gpt4       |            66.4 |                    15.3 |                          878 |           0.85 |          0.75 | 30.2 |     19.3 |                 0.60 |
| alpaca_eval_cot_gpt4_turbo_fn |            65.7 |                     4.3 |                          228 |           0.78 |          0.77 | 33.9 |     23.7 |                 0.61 |
| humans                        |            65.7 |                   300.0 |                        36800 |           1.00 |          1.00 |  0.0 |     34.3 |                 0.64 |
| claude                        |            65.3 |                     3.3 |                          173 |           0.93 |          0.90 | 32.4 |     18.5 |                 0.66 |
| lmsys_gpt4                    |            65.3 |                    13.9 |                        17982 |           0.98 |          0.97 | 31.6 |     15.9 |                 0.74 |
| text_davinci_003              |            64.1 |                     8.7 |                          121 |           0.85 |          0.83 | 33.8 |     22.7 |                 0.70 |
| longest                       |            62.2 |                     0.0 |                            0 |           0.27 |          0.56 | 37.8 |      0.0 |                 1.00 |
| chatgpt                       |            57.3 |                     0.8 |                          285 |           0.72 |          0.71 | 39.4 |     34.1 |                 0.59 |

<details>
  <summary><b>这些指标究竟是如何计算出来的？</b></summary>

现在，我们将用文字解释如何计算上表中的指标。[代码在此](https://github.com/tatsu-lab/alpaca_eval/blob/f05cbd651b79ac93906b19d01fe443b45828b0f2/src/alpaca_eval/analyze.py#L366)。

**Human Agreement人类一致性**：这衡量的是当前注释者与我们[交叉注释集](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_farm_human_crossannotations.json)中约 650 个注释的大多数人类偏好之间的一致性，其中每个示例包含 4 个人类注释。
为了估算单个人类（上表中的 "human "行）与大多数人类之间的一致性，我们从 4 个注释中抽取一个，计算它在预测其他 3 个注释的模式时的准确率。
然后，我们对所有 4 个注释和 650 条指令求取平均值，从而得到人类的一致性，也就是说，我们计算的是预期的（人类和样本的）"遗漏一致性"。
如果模式不是唯一的，我们就随机抽取一个模式。
我们对自动注释器进行了完全相同的计算，因此最终的数字具有可比性。

**Price价格[$/1000 例]**：这是每 1000 个注释的平均价格。
对于人类，这是 [我们支付给 Mechanical Turkers](https://arxiv.org/abs/2305.14387) 收集这些注释的价格（21 美元/小时）。
如果价格取决于计算注释所使用的机器（如 Guanaco），则留空。

**Time时间[秒/1000 个示例]**：这是计算 1000 个注释所需的平均时间。
对于人类来说，这是每位 Mechanical Turker 对 1000 个示例进行注释所花费时间的估计中值。
对于自动注释器，这是我们运行注释所花费的平均时间。
请注意，这可能取决于不同用户的 API 限制以及集群正在处理的请求数量。

**Spearman corr.**：该指标衡量的是根据自动注释器偏好计算的排行榜与根据人类偏好计算的排行榜之间的斯皮尔曼相关性。与 "人类一致度 "一样，我们使用 AlpacaFarm 的人类注释，但现在我们考虑的是方法级一致度，而不仅仅是与人类的样本一致度。请注意，我们只使用了 9 个模型，因此相关性不是很可靠。

**Pearson corr.**：与 "Spearman corr. "相同，但使用皮尔逊相关性。

**Bias偏差**：最有可能的人工标注与最有可能的自动标注之间的一致性。
对于自动注释器，我们通过对每个示例的 4 个不同注释进行采样来估算。
这里的随机性来自提示中输出的顺序、从 LLM 中采样、批次中指令的顺序（如适用）以及池中注释器的选择。
然后，我们取 4 个注释的模式，计算该模式预测 4 个人类注释模式的准确度。
请注意，这很可能高估了如果我们有 "无限 "数量的交叉注释时的实际偏差。
低偏差意味着注释者的预期偏好与人类相同。
就人类而言，根据定义，偏差为零。
请注意，这与标准统计偏差有关，但不是标准统计偏差，因为我们采用的是模式而不是注释的平均值，我们考虑的是 0-1 损失而不是平方损失。

**Variance方差**：单个自动偏好与最可能偏好的预期一致度。我们估算它的方法与估算人类 "人类一致性 "的方法相同，即在使用第 4 个注释预测 3 个注释的模式时，我们取预期的遗漏误差。
低方差意味着注释者与其偏好是一致的，也就是说，如果用不同的种子进行采样，结果是一样的。
与偏差一样，这并不完全是标准的统计方差，因为我们采用的是模式而不是注释的平均值，我们考虑的是 0-1 损失而不是平方损失。

请注意，"人类一致性 "与偏差和方差密切相关。尤其是，方差衡量的是由于我们只使用单个注释而产生的误差，而偏差则旨在衡量当前注释者的不可还原误差。

**偏好较长**：这是当两个输出结果中的一个明显比另一个长（相差 30 个字符以上）时，注释者偏好较长输出结果的概率。

在 [完整表格](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md) 中，我们还提供了以下指标：

**Proba.偏好列表**：这是当一个输出包含列表/布尔点而另一个不包含时，注释者偏好包含列表/布尔点的输出的概率。

**Proba.偏好 1**：这是注释器偏好一对输出中第一个输出的概率。我们建议的所有注释器都会随机选择提示中的输出，因此该概率应为 0.5。先前的注释器，如 "lmsys "和 "aviary"，则不会这样做。

**#parsed已解析**：这是注释器能够解析的示例数量。

请注意，如果方差和偏差为空，则表示由于资源（时间和价格）限制，我们只对每个 648 个示例进行了一次注释。这就解释了为什么#parsed 是 648，否则应该是 2592。

</details>

<details>
  <summary><b>选择评估员的技巧</b></summary>
总的来说，如果您希望与人类达到较高的一致性，我们建议您使用 "annotators_config=weighted_alpaca_eval_gpt4_turbo"；如果您的预算有限，我们建议您使用 "annotators_config=chatgpt_fn"。

在选择注释器时，我们建议您考虑以下几点（前三点显而易见）：

- "人工同意率 [%]"
- "价格[$/1000 个示例]"
- "时间[秒/1000 个示例]"
- `"* corr."` 大约 > 0.7。相关性不能太低，这一点很重要，但我们不建议将其作为主要指标，因为相关性只根据 9 个模型计算。
- Proba.prefer longer "约<0.7。事实上，我们发现大多数人类注释者都非常偏爱较长的答案（如 "最长 "评价器的高[performance=62.2](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md)所示，该评价器总是偏爱最长的输出）。这表明人类注释者可能存在更多偏差。为了避免排行榜在长度方面出现强烈偏差，我们建议使用 "偏好较长 "小于 0.7 的自动注释器。
- 方差 "约小于 0.2。我们认为，一个好的评价器应该具有尽可能小的方差，这样评价结果才能具有很高的可重复性。请注意，如 [AlpacaFarm](https://arxiv.org/abs/2305.14387) 所示，在我们模拟人类的情况下，方差可能是可取的。

我们过滤了上表中不符合这些要求的注释者（除了人类/ChatGPT/003/lmsys，以供参考）。所有结果见 [此处](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md)。
总的来说，我们发现 "weighted_alpaca_eval_gpt4_turbo "在质量/价格/时间/方差/长度偏差之间进行了很好的权衡。

</details>

上述指标是根据人群工作者的注释计算得出的。这些注释虽然有用，但并不完美，例如，群众工作者往往偏爱风格而非事实。因此，我们建议用户根据自己的说明和人工注释来验证自动评估器。
详情请见 [限制](#limitations)。

# 使用案例

## 评估一个模型

<details>
  <summary><code>>>> alpaca_eval evaluate -- --help</code></summary>

```
名称
    alpaca_eval evaluate - 根据输出结果评估模型。如果没有指定命令，这是默认的入口点。

参数
    alpaca_eval evaluate <flags

描述
    根据输出结果评估模型。如果未指定任何命令，这是默认入口点。

参数
    --model_outputs=MODEL_OUTPUTS
        类型： 可选[联合］
        默认值： 无
        要添加到排行榜的模型输出。接受数据（字典列表、pd.dataframe、datasets.Dataset）或读取数据的路径（json、csv、tsv）或生成数据的函数。每个字典（或数据框的行）都应包含提示中格式化的键。例如，默认情况下，"指令 "和 "输出 "可选 "输入"。如果无，则只打印排行榜。
    -r, --reference_outputs=REFERENCE_OUTPUTS
        类型： 联合
        默认值： <func...
        参考模型的输出。格式与 `model_outputs`相同。如果为 "无"，则参考输出是 AlpacaEval 集上特定的 Davinci 003 输出集：
    --annotators_config=ANNOTATORS_CONFIG
        类型： 联合
        默认： "alpaca_eval_gpt4_turbo_fn
        注释器配置文件的路径（或 dict 列表）。详情请参见 `PairwiseAnnotator` 的 docstring。
    -n, --name=NAME
        类型： 可选[可选］
        默认： 无
        要添加到排行榜的模型名称。如果为 "无"，我们将检查 "生成器 "是否位于 "model_outputs "中，如果不在，我们将使用 "当前模型"。
    -o, --output_path=OUTPUT_PATH
        类型： 联合
        默认： "auto
        存储新排行榜和注释的目录路径。如果为 "无"，则不保存。如果 "auto "是路径，则使用 "model_outputs"，否则使用调用脚本的目录。
    -p, --precomputed_leaderboard=PRECOMPUTED_LEADERBOARD
        类型： 联合
        默认： 'auto' （自动
        预计算的排行榜或其路径（json、csv 或 tsv）。排行榜应至少包含 `win_rate` 列。如果 `auto`（自动），我们将尝试使用相应的排行榜作为参考输出（仅在 CORRESPONDING_OUTPUTS_LEADERBOARDS 中）。如果 "无"，我们将不会从排行榜中添加其他模型。
    --is_overwrite_leaderboard=IS_OVERWRITE_LEADERBOARD
        类型： bool
        默认： 假
        如果模型已在排行榜中，是否覆盖排行榜。
    -l, --leaderboard_mode_too_print=LEADERBOARD_MODE_TO_PRINT
        类型： 可选 可选
        默认： 'minimal
        要使用的排行榜模式。仅在预计算的排行榜有 "mode "列时使用，在这种情况下，它将根据该模式过滤排行榜。如果为 "无"，则保留所有模式。
    -c, --current_leaderboard_mode=CURRENT_LEADERBOARD_MODE
        类型：字符串
        默认："社区
        当前方法的排行榜模式。
    --is_return_instead_of_print=IS_RETURN_INSTEAD_OF_PRINT
        类型： bool
        默认值：false 假
        是否返回度量值而不是打印结果。
    -f, --fn_metric=FN_METRIC
        类型： 联合
        默认： "pairwise_too_winrate
        metrics.py "中用于将偏好值转换为度量值的函数或函数名。该函数应获取偏好值序列（0 表示平局，1 表示基本获胜，2 表示要比较的模型获胜），并返回一个度量值字典以及对排行榜进行排序的关键字。
    -s, --sort_by=SORT_BY
        类型：str
        默认："win_rate
        对排行榜进行排序的关键字。
    --is_cache_leaderboard=IS_CACHE_LEADERBOARD
        类型： 字符串 可选[可选］
        默认值： 无
        是否保存结果排行榜到 "precomputed_leaderboard"。如果为 "无"，则仅在 max_instances 不是 "无 "的情况下保存。将模型添加到排行榜的首选方法是将 `precomputed_leaderboard` 设置为先前保存在 `<output_path>/leaderboard.csv`的排行榜。
    -max_instances=MAX_INSTANCES
        类型： 可选[可选］
        默认值： 无
        要注释的实例的最大数量。用于测试。
    --annotation_kwargs=ANNOTATION_KWARGS
        类型： 可选[可选］
        默认值： 无
        传递给 `PairwiseAnnotator.annotate_head2head` 的附加参数。
    -A、--注释器=ANNOTATOR
        默认值： <class 'alpaca_eval.annotators.pairwise_evaluator.PairwiseAn...
        要使用的注解器类。
    接受附加标志。
        传递给 `PairwiseAnnotator` 的附加参数。
```

</details>

<details>
  <summary><code>>>> alpaca_eval evaluate_from_model -- --help</code></summary>

```
名称
    alpaca_eval evaluate_from_model - 评估来自 HuggingFace 或 API 提供者的模型。这是`evaluate`的一个包装器，包括从所需模型生成。

说明
    alpaca_eval evaluate_from_model MODEL_CONFIGS <flags> (从模型评估)

描述
    评估来自 HuggingFace 或 API 提供者的模型。它是 `evaluate` 的包装器，包括从所需模型生成。

位置参数
    MODEL_CONFIGS
        类型： 联合
        一个字典或一个 yaml 文件的路径（相对于 `models_configs`），该文件包含要解码的模型的配置。如果是一个目录，我们会在其中搜索 "configs.yml"。第一个字典的键应该是生成器的名称，值应该是生成器配置的字典，其中应该包含

FLAGS
    -r, --reference_model_configs=REFERENCE_MODEL_CONFIGS
        类型： 可选[联合］
        默认值： 无
        与 `model_configs` 相同，但用于参考模型。如果无，我们将使用默认的 Davinci003 输出。
    -e, --evaluation_dataset=EVALUATION_DATASET
        类型： 联合
        默认值： <func...
        评估数据集或返回数据帧的函数的路径。如果无，则使用默认评估
    -a, --annotators_config=ANNOTATORS_CONFIG
        类型： 联合
        默认值："alpaca_eval_gpt4_turbo_fn
        注释器配置或字典的路径。如果无，则使用默认注释器配置。
    -o, --output_path=OUTPUT_PATH
        类型： 联合
        默认："自动
        保存代、注释和排行榜的路径。如果自动保存，则保存在 `results/<model_name>` 处。
    -m, --max_instances=MAX_INSTANCES
        类型： 可选[int］
        默认值： 无
        生成和评估的最大实例数。如果无，我们将评估所有实例。
    --is_strip_output=IS_STRIP_OUTPUT
        类型： bool
        默认值：true 真
        是否从输出中去除尾部和前部空白。
    --is_load_outputs=IS_LOAD_OUTPUTS
        类型: bool
        默认值：true true
        是否尝试从输出路径加载输出。如果为 True 且存在输出，则只为尚未有输出的指令生成输出。
    -c, --chunksize=CHUNKSIZE
        类型：int
        默认值：64
        保存前生成的实例数。如果为 "无"，则在生成所有实例后保存。
    可接受其他标志。
        其他要 "评估 "的参数

注意
    您也可以对位置参数使用 flags 语法
```

</details>

要评估一个模型，您需要

1. 选择一个评估集，并计算指定为 `model_outputs` 的输出。默认情况下，我们使用 [AlpacaEval](#data-release) 中的 805 个示例。要在 AlpacaEval 上计算输出，请使用：

```python
import datasets

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
for example in eval_set:
    # generate here is a placeholder for your models generations
    example["output"] = generate(example["instruction"])
```

如果您的模型是 HuggingFace 模型或来自标准 API 提供商（OpenAI、Anthropic、Cohere）。然后，您可以直接使用 `alpaca_eval evaluate_from_model` 来生成输出。

2. 计算参考输出结果 `reference_outputs`。默认情况下，我们使用 [AlpacaEval 上的 `gpt4_turbo`](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) 的预计算输出。如果您想使用不同的模型或不同的数据集，请遵循与 (1.) 相同的步骤。
3. 选择通过 `annotators_config` 指定的评价器。我们推荐使用 `alpaca_eval_gpt4_turbo_fn`。有关其他选项和比较，请参阅 [本表](#evaluators)。根据评估器的不同，您可能需要在环境或 [client_configs](https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs) 中设置相应的 API_KEY。

一起运行:

```bash
alpaca_eval --model_outputs 'example/outputs.json' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'
```

如果没有解码输出，可以使用 `evaluate_from_model`，它会为您处理解码（模型和引用）。
下面是一个例子：

```bash
# need a GPU for local models
alpaca_eval evaluate_from_model \
  --model_configs 'oasst_pythia_12b' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'    
```

此处的 `model_configs` 和 `reference_model_configs`（可选）是指向指定提示符、模型提供者（此处为 HuggingFace）和解码参数的目录的路径。
示例请参见 [this directory](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/models_configs)。
关于所有开箱即用的模型提供程序，请参见 [此处](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/decoders)。

<details>
  <summary><b>关于注释者的信息</b></summary>

- 缓存**：默认情况下，所有注释都缓存在位于 `caching_path``的磁盘上。这样，注释就不会重新计算，从而使注释更快、更便宜，并具有可重复性。这甚至有助于评估不同的模型，因为许多模型都有相同的输出。
- 输出随机化*** 默认情况下，我们对输出示例进行随机化，因为我们发现注释者往往更喜欢他们看到的第一个示例。
- 批处理*** 我们提供了批处理注释的代码和示例，这样可以在提示较长的情况下减少注释的成本和时间。例如，请参见 [alpaca_farm_greedy_gpt4](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/alpaca_farm_greedy_gpt4)。
- 注释器库** 我们提供了使用自动注释器库进行评估的代码和示例，这有助于复制[人工注释](https://arxiv.org/abs/2305.14387)的差异。例如，请参阅 [羊驼农场](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/alpaca_farm)。
- 基于指令的播种*** 为提高可重复性和模型间的公平比较，我们根据指令播种所有随机性（输出顺序、批次顺序、池中每个注释者的示例）。

</details>

## 制作新的排行榜

<details>
  <summary><code>>>> alpaca_eval make_leaderboard -- --help</code></summary>

```
名称
    alpaca_eval make_leaderboard - 为给定的数据集/评估器/模型代集预先计算并保存整个排行榜。

说明
    alpaca_eval make_leaderboard <flags

描述
    预计算并保存给定数据集/评估器/模型代集的整个排行榜。

参数
    --leaderboard_path=LEADERBOARD_PATH
        类型： 可选[联合］
        默认值： 无
        保存排行榜的路径。排行榜将保存为 csv 文件，如果该文件已经存在，则将
    --annotators_config=ANNOTATORS_CONFIG
        类型： 联合
        默认： "alpaca_eval_gpt4_turbo_fn
        注释器配置文件的路径（或 dict 列表）。
    --all_model_outputs=ALL_MODEL_OUTPUTS
        类型： 联合
        默认值： <fu...
        要添加到排行榜的所有模型的输出。接受数据（字典列表、pd.dataframe、datasets.Dataset）或读取这些数据的路径（json、csv、tsv 可能带套叠）或生成这些数据的函数。如果路径中包含套叠模式，我们将读取与该模式匹配的所有文件并将它们连接起来。每个字典（或数据帧的行）都应包含提示中格式化的键。例如，默认情况下包含 `instruction` 和 `output` 以及可选的 `input`。它还应包含一列`生成器`，其中包含当前模型的名称。
    -r, --reference_outputs=REFERENCE_OUTPUTS
        类型： 联合
        默认值： <func...
        参考模型的输出。格式与 `all_model_outputs` 相同，但无需使用 `generator`。默认情况下，参考输出是 AlpacaEval 设置的 003 输出。
    -f, --fn_add_too_leaderboard=FN_ADD_TO_LEADERBOARD
        类型： 可调用
        默认值："evaluate
        用于将模型添加到排行榜的函数。如果是字符串，则应是 `main.py`中的函数名称。函数应包含以下参数 model_outputs`、`annotators_config`、`name`、`precomputed_leaderboard`、`is_return_instead_of_print`、`reference_outputs`。
    --leaderboard_mode=LEADERBOARD_MODE
        类型：str
        默认： "已验证
        保存所有新条目时使用的排行榜模式。
    -i, --is_return_instead_of_print=IS_RETURN_INSTEAD_OF_PRINT
        类型： bool
        默认值： 假
        是否返回指标而不是打印结果。
    接受其他标志。
        传递给 `fn_add_too_leaderboard` 的附加参数。
```

</details>

如果想使用一条命令（而不是多次调用 `alpaca_eval`）为所需的评价集和评价器创建一个新的排行榜，可以使用下面的命令：

```bash
alpaca_eval make_leaderboard \
  --leaderboard_path <path_to_save_leaderboard> \
  --all_model_outputs <model_outputs_path> \
  --reference_outputs <reference_outputs_path> \
  --annotators_config <path_to_config.yaml>
```

其中：

- leaderboard_path`：保存排行榜的路径。排行榜将以 csv 文件形式保存，如果该文件已经存在，则会添加。
- all_model_outputs`：要添加到排行榜的所有模型输出的 json 路径（作为单个文件或通过 globbing 多个文件）。每个字典都应包含提示格式的键（"指令 "和 "输出"），以及包含当前模型名称的 "生成器 "列。示例见 [this file](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval_all_outputs.json).
- `reference_outputs` 参考模型输出的路径。每个字典都应包含提示格式化的键（"指令 "和 "输出"）。默认情况下，参考输出是 AlpacaEval 设置的 003 个输出。
- `annotators_config`： 注释器配置文件的路径。默认为 `alpaca_eval_gpt4`。

</details>

## 制作一个新的评估器

<details>
  <summary><code>>>> alpaca_eval analyze_evaluators -- --help</code></summary>

```
AME
    alpaca_eval analyze_evaluators - 分析评价器并填充评价器排行榜（与人类的一致性、速度、价格......）。

说明
    alpaca_eval analyze_evaluators <flags> <标志

描述
    分析评价器并填充评价器排行榜（与人类的一致性、速度、价格......）。

标记
    --annotators_config=ANNOTATORS_CONFIG
        类型： 联合
        默认： "alpaca_eval_gpt4_turbo_fn
        注释器配置文件的路径（或 dict 列表）。
    -A, --Annotator=ANNOTATOR
        默认值： <类 'alpaca_eval.annotators.pairwise_evaluator.PairwiseAn...
        要使用的注释器类。
    --analyzer_kwargs=ANALYZER_KWARGS
        类型： 可选[可选］
        默认值： 无
        传递给分析器的附加参数。
    -p, --precomputed_leaderboard=PRECOMPUTED_LEADERBOARD
        类型： 联合
        默认值： PosixPath('/Users/yanndubois/Desktop/GitHub/alpaca_eval/src/...
        注释者的预计算（元）排行榜或其路径（json、csv 或 tsv）。
    --is_save_leaderboard=IS_SAVE_LEADERBOARD
        类型： bool
        默认值：false 假
        是否保存排行榜（即分析结果）。
    --is_return_instead_of_print=IS_RETURN_INSTEAD_OF_PRINT
        类型： bool
        默认值：false 假
        是否返回排行榜（即分析结果）。如果为 True，则不打印结果。
    --is_overwrite_leaderboard=IS_OVERWRITE_LEADERBOARD
        类型： bool
        默认值： False 假
        如果排行榜已存在，是否覆盖它。
    -m, --max_instances=MAX_INSTANCES
        类型： 可选 可选[可选］
        默认值： 无
        要分析的实例的最大数量。
    --is_single_annotator=IS_SINGLE_ANNOTATOR
        类型： bool
        默认值： False 假
        是否分析单个注释器。如果为 True，将无法估计注释者的偏差。
    -l, --leaderboard_mode_too_print=LEADERBOARD_MODE_TO_PRINT
        类型：字符串
        默认： 'minimal
        要打印的排行榜模式。
    -c, --current_leaderboard_mode=CURRENT_LEADERBOARD_MODE
        类型： 字符串
        默认： 'minimal
        保存所有新条目时使用的排行榜模式。
    -o, --output_path=OUTPUT_PATH
        类型： 字符串 联合
        默认："自动
        保存排行榜和注释的路径。如果为 "无"，则不保存。
    可接受其他标志。
        传递给 `Annotator`的附加参数。
```

</details>

AlpacaEval 提供了一种制作新评估器的简单方法。你只需创建一个新的 `configs.yaml` 配置文件，然后将其作为 `--annotators_config <path_to_config.yaml>` 传递给 `alpaca_eval`。
以下是一些制作新评估器的方法：

- 更改提示符**： 在文本文件中编写新的提示，并在配置文件的 `prompt_template` 中指定路径。路径是相对于配置文件而言的。
- 更改解码参数**： 在配置文件的 `completions_kwargs` 中指定所需的参数。要查看所有可用参数，请参阅配置文件中 `fn_completions` 指定的[本文件中](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py) 相应函数的 docstrings。
- 更改模型**： 在 `model_name` 中指定所需的模型，并在 `prompt_template` 中指定相应的提示。如果模型来自其他提供商，则必须更改 `fn_completions`，该函数映射到 [this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py) 中的相应函数。我们提供了 `fn_completions` 函数，以便使用 OpenAI、Anthropic、Cohere 或 HuggingFace 的模型。要安装所有提供者所需的软件包，请使用 `pip install alpaca_eval[all]`。

<details>
  <summary><b>配置文件中的其他参数</b></summary>

最简单的方法是查看 [`SinglePairwiseAnnotator`](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/annotators/pairwise_evaluator.py#L537) 的文档说明。
下面是一些重要的文档：

```
参数
----------
prompt_template : 路径
    将提供给 `fn_prompter` 的提示或提示路径。路径相对于
    `evaluators_configs/` 的相对路径。

fn_completion_parser : callable 或 str
    在 `completion_parsers.py` 中用于将完成信息解析为首选项的函数。对于每个完成
    首选项的数量应等于 batch_size，否则我们会将该批次中的所有首选项设置为
    NaN。

completion_parser_kwargs : dict
    fn_completion_parser 的 Kwargs。

fn_completions : callable 或 str
    用于解码输出的 `decoders.py` 中的函数。

completions_kwargs : dict
    用于 fn_completions 的 kwargs。例如：model_name、max_tokens、temperature、top_p、top_k、stop_seq。

is_randomize_output_order : bool
    格式化时是否随机化输出_1、输出_2。

batch_size : int
    单次提示中将添加的示例数量。
```

</details>

制作评价器后，您还可以使用以下命令对其进行分析，并将其添加到_评价器的_[排行榜](#evaluators)中：

```bash
alpaca_eval analyze_evaluators --annotators_config '<path_to_config.yaml>'  
```

为了估算偏差和方差，它会用 4 个种子对每个示例进行评估，即 2.5K 评估。
如果你想要更便宜的评估，可以使用 `--is_single_annotator True` 来使用单个种子，这样就可以跳过偏差和方差的估算。

</details>

# 贡献

除了错误修复外，我们还接受新模型、评估员和评估集的 PR。

我们将根据社区的新贡献定期更新排行榜网站 [leaderboard website](https://tatsu-lab.github.io/alpaca_eval/)。

我们还为 AlpacaEval 创建了一个[support discord](https://discord.gg/GJMxJSVZZM) ，以防您遇到任何问题并向社区寻求帮助。

要开始使用，请先 fork 软件仓库，然后从源代码安装软件包 pip install -e .

## 提供模型

首先，您需要在 [models_configs](src/alpaca_eval/models_configs/) 文件夹中添加模型配置定义。例如，您可以查看 [falcon-7b-instruct yaml]（src/alpaca_eval/models_configs/falcon-7b-instruct/configs.yml）。请确保 yaml 中的文件夹名称和密钥名称完全一致。

然后，请按照[评估模型](#evaluating-a-model)中的步骤在模型上运行推理，以在评估集上生成输出，并根据其中一个评估器对模型进行评分。
命令示例如下

```sh
alpaca_eval evaluate_from_model \
  --model_configs 'falcon-7b-instruct'
```

运行此命令后，您应该会生成一个输出 json，并在相应的 [leaderboardfile](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/leaderboards/data_AlpacaEval) 中生成一个新条目。请将配置、输出文件和更新的排行榜提交给 PR。

具体操作如下

1. 在 github 上叉仓库
2. 克隆分叉仓库 `git clone <URL>` 3.
3. 在 `src/alpaca_eval/models_configs/<model_name>` 中创建模型配置并对其进行评估 `evaluate_from_model --model_configs '<model_name>'`
4. 将模型配置、输出和排行榜条目添加到分叉版本库中

```sh
git add src/alpaca_eval/models_configs/<model_name>
git add src/alpaca_eval/leaderboards/ 
git add -f results/<model_name>/model_outputs.json
git add -f results/<model_name>/*/annotations.json
git commit -m "Add <model_name> to AlpacaEval"
git push
```

5. 在 AlpacaEval 上创建一个[pull request on AlpacaEval](https://github.com/tatsu-lab/alpaca_eval/pulls)。

注意：如果您要在 AlpacaEval 之外生成输出，则仍应添加模型配置，但应使用 `fn_completions:null`。
请参阅 [this config](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/models_configs/dolphin-2.2.1-mistral-7b/configs.yaml) 获取示例。

## 验证模型

<p align="center">
<img align="center" alt="verified.png" src="figures/verified.png" width="500"/>
</p>

AlpacaEval 中的验证结果表明，核心维护者已对模型输出进行解码并执行了评估。遗憾的是，我们 AlpacaEval 维护者缺乏验证所有模型的资源（时间/GPU/资金）。我们对由此造成的不便深表歉意，并感谢您的理解。要验证您的模型，请按照以下步骤操作：

1. 在 Discord 上联系 `@yann` 或 `@rtaori`，或给我们发送电子邮件（如果您有我们的邮箱），简要说明您的模型需要验证的原因。
2. 在继续之前等待我们的回复和批准。
3. 准备一个不需要 GPU 的脚本来解码您的模型，通常与您的模型贡献所用的脚本相同。该脚本应使用 `alpaca_eval evaluate_from_model --model_configs '<your_model_name>'` 运行，无需本地 GPU。
4. 生成用于运行脚本的临时 API 密钥并与我们共享。具体来说，我们需要用于解码模型和评估的密钥（例如 OpenAI 或 Anthropic 密钥）。
5. 我们将执行 "alpaca_eval evaluate_from_model --model_configs '<your_model_name>'"，更新结果并通知您，以便您撤销临时密钥。

请注意，我们不会重新评估同一个模型。由于抽样差异，结果可能与您的初始结果略有不同。我们将用经过验证的结果替换您之前的社区结果。

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">提供一名评估员</h2></summary>

请首先按照[创建新的评估器](#making-a-new-evaluator)中的说明进行操作。
创建注释器配置后，我们要求您通过评估最小模型集为注释器创建一个新的排行榜。这些模型的输出可通过下载 [alpaca_eval_all_outputs.json](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval_all_outputs.json) 找到。

```bash
alpaca_eval make_leaderboard \
  --leaderboard_path src/alpaca_eval/leaderboards/data_AlpacaEval/<evaluator>_leaderboard.csv \
  --all_model_outputs alpaca_eval_all_outputs.json \
  --annotators_config <evaluator_config>
```

然后，请创建包含注释器配置和排行榜 csv 的 PR。

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">贡献一个评估集</h2></summary>

要贡献一个新的评估集，您首先需要指定一组文字说明。
然后，您需要指定一组参考输出（模型胜率是根据此参考计算的）。
为方便使用，您可以使用默认的 [text-davinci_003](src/alpaca_eval/models_configs/text_davinci_003/) 参考配置。

将这些配置放到一个 json 中，其中每个条目都指定了 "指令"、"输出 "和 "生成器 "字段。您可以参考 [alpaca_eval.json](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval.json)（`dataset` 字段不是必需的）。

最后，我们要求您在这个新的评估集上创建一个最小的排行榜。您可以使用以下方法来完成这项工作：

```bash
alpaca_eval make_leaderboard \
  --leaderboard_path <src/alpaca_eval/leaderboards/data_AlpacaEval/your_leaderboard_name.csv> \
  --all_model_outputs alpaca_eval_all_outputs.json \
  --reference_outputs <path_to_json_file>
```

请提交 PR，并附上评估集 json 和相应的排行榜 csv。

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">提供完成功能</h2></summary>

目前，我们允许使用不同的完成函数，例如：`openai`, `anthropic`, `huggingface_local`, `huggingface_hub_api`... 如果您想贡献一个新的完成函数/API 来执行推理，请按照以下步骤操作：

1. 在 [decoder 文件夹](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/decoders) 中添加 `<name>`.py 文件和函数 `<name>_completions(prompts : Sequence[str], model_name :str, ...)`。该函数应将 prompts + kwargs 作为参数，并返回补全信息。请查看该目录中的其他补全函数模板。例如，[huggingface_local_completions](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/huggingface_local.py) 或 [anthropic](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/anthropic.py)。
2. 在 [__init__](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py)中添加 `<name>_completions` 和依赖项。同样，您也可以效仿 [huggingface_local_completions](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py#L30) 的做法。
3. 更新 [setup.py](https://github.com/tatsu-lab/alpaca_eval/blob/main/setup.py) 中的可选依赖项
4. 在 [models configs](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/models_configs) 中添加要评估的模型。
5. 使用 `alpaca_eval evaluate_from_model --model_configs '<model_configs>'` 评估模型
6. (可选）按照 [这些步骤](https://github.com/tatsu-lab/alpaca_eval/tree/main#contributing-a-model) 将上一个模型的结果推送到 AlpacaEval 排行榜上。

请随时提前启动 PR，我们将在此过程中提供一些帮助！

</details>

# 限制

AlpacaEval 评估流水线与目前的其他评估工具一样，都有很大的局限性，因此在一些重要的场合，如决定一个模型是否可以部署时，不应该用它来替代人工评估。
这些限制可大致分为三类：

1. **指令可能并不代表实际使用情况**：AlpacaEval 集包含来自各种数据集的示例（[self-instruct](https://github.com/yizhongw/self-instruct)、[open-assistant(https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/OpenAssistant--oasst1/validation)、[vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)、[koala](https://github.com/arnav-gudibande/koala-test-set)、[hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/Anthropic--hh-rlhf/test)），这些示例可能并不代表实际使用情况和更好模型（如 GPT4）的高级应用。这很可能使最好的封闭模型（GPT4 / Claude / ChatGPT / ......）看起来与开放模型更为相似。事实上，这些封闭模型似乎是在更多样化的数据上进行预训练/调整的。关于更复杂指令的初步结果，请参见 [this blog](https://medium.com/@marcotcr/exploring-chatgpt-vs-open-source-models-on-slightly-harder-tasks-aa0395c31610)。
   不过请注意，在 [AlpacaFarm](https://arxiv.org/abs/2305.14387) 中，我们表明我们的评估集的胜率与用户与 Alpaca Demo 交互指令的胜率高度相关（0.97 R2）。
   此外，AlpacaEval 排行榜显示开放模型与 OpenAI 模型之间的差距大于其他排行榜（例如 [lmsys](https://lmsys.org/blog/2023-03-30-vicuna/)）。
2. **自动注释器的偏差**：原始自动注释器似乎有隐含的偏差。特别是，我们发现他们倾向于较长的输出和包含列表的输出（例如，"alpaca_eval_gpt4 "为 0.68 / 0.69，"claude "为 0.62 / 0.58）。
   虽然我们发现人类也有类似的偏差（0.64 / 0.61），但我们认为这可能更多是我们使用的人类注释管道的限制，而不是真正的人类偏差。更广泛地说，通过定性分析，我们发现自动注释者更重视输出的风格而非内容（如事实性）。
   最后，我们发现自动评估者倾向于选择来自相似模型的输出结果（很可能是在相同的数据上训练出来的），这一点从 ChatGPT/GPT4 在 `claude` 和 `alpaca_eval_gpt4` 排行榜上的巨大差异可以看出。请注意，长度偏差在我们的长度控制胜率中得到了部分缓解。
3. **缺乏安全评估**：重要的是，AlpacaEval 只评估模型的指令执行能力，而不是它们可能造成的危害（如有毒行为或偏差）。因此，当前 ChatGPT 与最佳开源模型之间的微小差距***不应被解释为后者已经可以部署。

除了这些关于评估管道的局限性之外，我们对评估者的验证以及我们选择评估集的[建议方法]（#分析一个评估集）也存在局限性。

<details>
  <summary><b>我们验证管道的局限性</b></b></summary>

首先，我们基于人类交叉注释对评估者进行的验证存在以下局限性：(1) 我们定性地发现，我们的人群工作者也倾向于倾向于风格，如长度和列表的存在，而不是事实性；
(2) 这并不能证实对照参考模型的胜出率首先是否是一种好的评估策略；
(3) 16 名群众工作者的偏好并不能代表全人类的偏好。

其次，我们建议的基于统计能力选择评估集的方法存在以下局限性：(1) 统计能力并不能确保正确的方向，例如，您可能会得到一组不自然的指令，其中 Alpaca 比更好的模型 "表现 "更好；(2) 这可能会促使用户选择数据来支持他们想要验证的假设。

</details>

# 其他分析和绘图

**注意**：以下所有结果均与 AlpacaEval 1.0 有关，且自此之后未进行过更新

AlpacaEval 提供了一些可视化工具，帮助您分析和改进自动评估管道。我们简要说明并再次提供笔记本，以便进行更多分析。对于我们考虑的所有指标的描述，请参阅 [这些指标究竟是如何计算的](https://github.com/tatsu-lab/alpaca_eval#evaluators))

## 长度控制羊驼评估（LCAE)

**长度控制的 AlpacaEval 可视化：**
[![分析评价器](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_eval/blob/main/notebooks/figured_length_controlled.ipynb)

**长度可控的 AlpacaEval 开发：** [!
[！[分析评估器](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_eval/blob/main/notebooks/length_controlled.ipynb)

本笔记本展示了我们为减轻自动注释器的长度偏差而考虑的不同方案。

在此，我们简要总结了主要结果。即

- **LCAE 与 Chat Arena 的相关性从 AlpacaEval 2.0 的 0.94 提高到 0.98**。这使得 LCAE 成为与 Chat Arena 相关性最高的基准，如下图所示。

<p float="left" align="middle">
<img src="figures/chat_correlations.png" alt="LC AlpacaEval is the most highly correlated benchmark with Chat Arena." width="500"/>
</p>

- **LCAE 会降低游戏的可玩性** AlpacaEval 的一个主要问题是，您可以通过增加输出的长度来提高胜率。例如，在 AlpacaEval 2.0 中，当提示 "尽可能提供更多细节 "时，基线（50%）的胜率上升到 64%，而当提示 "尽可能简洁，同时仍提供回答问题所需的所有信息 "时，胜率下降到 23%。总体而言，AlpacaEval 的相对长度可玩性为 21%，而 LCAE 则降至 6%，因此通过提示长度可玩性降低了 3 倍。如下图所示。

<p float="left" align="middle">
<img src="figures/length_gameability.png" alt="LC AlpacaEval decreases length gameability of the benchmark." width="500"/>
</p>

- 我们可以预测不同基线的成绩** 使用 GLM 控制长度偏差还有一个好处。我们现在有了一个模型，可以预测模型在不同基线下的胜率。特别是，我们的 GLM 具有许多很好的特性，例如，"win_rate(m,b) = 1 - win_rate(b,m) （在 [0,1]〕 内）"和 "win_rate(m,m) = 0.5"。如下图所示。

<p float="left" align="middle">
<img src="figures/different_baselines.png" alt="Predicted win rate for different baselines" width="500"/>
</p>

最后，请注意我们只控制了长度偏差。我们没有控制其他已知的偏差，例如自动注释者更喜欢与其模型相似的输出结果。虽然我们可以控制这一点，但在实践中，我们发现这个问题不如长度偏差那么严重。原因有二：（1）这主要是排行榜中的单一模型，因为对自动注释器输出的微调似乎不会对胜率产生太大影响；（2）偏差实际上没有人们想象的那么强烈。例如，我们在下面展示了由三种不同模型自动标注的排行榜子集，我们可以看到模型的排名完全相同。特别是，"claude-3-opus "更喜欢 "gpt4_preview"，而 "mistral-large "更喜欢前两者。

<p float="left" align="middle">
<img src="figures/annotator_bias.png" alt="Leaderboard by different auto-annotators" width="500"/>
</p>
## 分析一个评估器

**注意**：以下所有结果均与 AlpacaEval 1.0 有关，此后未作更新

**分析评价器：**
[！[分析评价器](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_eval/blob/main/notebooks/analyzing_annotators.ipynb)

正如我们在[评估员排行榜](#evaluators)中所看到的，在选择评估员时有许多指标需要考虑，例如质量、价格和速度。为了帮助用户选择评估器，我们提供了一些绘制这些指标的函数。
例如，下图显示了不同评价器的价格/时间/协议。

![plot_quality_vs_price_and_time.png](figures%2Fplot_quality_vs_price_and_time.png)

在这里，我们看到 `alpaca_eval_gpt4` 的表现非常出色，在所有考虑的指标上都优于人类。

在此之前，我们只考虑了与人类注释者的整体一致性。
我们还可以做一个额外的验证，那就是检查使用我们的自动注释器制作排行榜是否能得到与人类排行榜相似的结果。
为了进行这样的分析，我们发布了来自 [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) 的 22 种方法输出的 [human annotations](#data-release) => 22*805 = ~18K annotations。因此，我们可以测试由人类和我们的自动注释器评估的 22 个模型的胜率之间的相关性。
需要注意的是，这可以说是比使用 "人类一致性 [%]"更好的选择自动评估器的方法，但由于需要 18K 个注释，因此成本较高。
下图显示了 `alpaca_eval_gpt4` 评估器的相关性。

<p float="left" align="middle">
<img src="figures/plot_winrate_correlations_alpaca_eval.png" alt="Correlation between humans and alpaca_eval_gpt4" width="400"/>
</p>

我们看到，"alpaca_eval_gpt4 "排行榜与来自人类的排行榜高度相关（0.94 Pearson correlation），这进一步表明自动评估是人类评估的良好代表。
有关代码和更多分析，请参阅 [this notebook](https://github.com/tatsu-lab/alpaca_eval/blob/main/notebooks/analyzing_annotators.ipynb) 或上面的 colab notebook。

</details>

## 分析一个评估数据集

**注意**：以下所有结果均与 AlpacaEval 1.0 有关，此后未作更新。

**制作评估集：**
[![分析评价器](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_eval/blob/main/notebooks/analyzing_evalset.ipynb)

在创建评估集时，需要考虑两个主要因素：使用多少数据？

回答这两个问题的一种方法是，将您认为质量不同的模型放在一个排行榜上，然后检查需要哪些数据以及需要多少数据才能在统计意义上将它们区分开来。
下面我们将使用配对 t 检验来测试每对模型之间的胜率差异是否具有统计学意义。

首先，让我们考虑使用多少数据的问题。
下面我们显示了在最小的 `alpaca_eval_gpt4` 排行榜中，为使每对模型的 p 值小于 0.05，需要从 AlpacaEval 进行配对 t 检验的随机样本数。
灰色单元格对应于在 805 个样本中没有显著差异的模型对。
y 轴和 x 轴分别按第一个和第二个模型的胜率排序。

<p float="left" align="middle">
<img src="figures/plot_paired_ttest_nsamples.png" alt="Number of samples needed to distinguish pairs in the Claude leaderboard" width="500"/>
</p>

我们看到，大多数模型只需 50 个样本就能区分，而 150 个样本就能区分大多数模型对（78 个中的 74 个）。这表明，当测试两个与最小 `alpaca_eval_gpt4` [leaderboard](#models) 上的性能差距相似的模型时，我们可以将评估集的大小减少 4 倍。

第二个问题是使用什么数据。我们也可以尝试从统计能力的角度来回答这个问题：
什么数据最能区分不同的模型。让我们考虑一下 AlpacaEval 中的所有数据集，但我们要控制评估集的大小，因为我们只关心数据的质量。下图显示了在 AlpacaEval 每个子集的 80 个示例上对每对模型进行配对 t 检验得出的 p 值。

![plot_paired_ttests_per_dataset.png](figures%2Fplot_paired_ttests_per_dataset.png)

例如，我们看到自我指导数据集产生的统计能力最小，这表明我们可以将该数据集从评估集中删除。
具体原因应在今后的工作中进行分析。
有关代码和更多分析，请参阅[本笔记本](https://github.com/tatsu-lab/alpaca_eval/blob/main/notebooks/analyzing_evalset.ipynb)或上面的colab笔记本。

</details>

# 引用

请根据您使用和引用的内容考虑引用以下内容：

- 代码、结果和一般基准： alpaca_eval（本软件仓库）。请指明您使用的是 AlpacaEval 还是 AlpacaEval 2.0。关于长度控制胜率，请参阅下文。
- **长度控制（LC）胜率**： 长度控制胜率： `alpaca_eval_length`.
- 人类注释： `dubois2023alpacafarm` ([AlpacaFarm](https://arxiv.org/abs/2305.14387))
- **AlpacaEval 评估集**： alpaca_eval`和[self-instruct](https://github.com/yizhongw/self-instruct)、[open-assistant](https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/OpenAssistant--oasst1/validation), [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [koala](https://github.com/arnav-gudibande/koala-test-set), [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/Anthropic--hh-rlhf/test).

以下是 bibtex 条目：

```
@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/alpaca_eval}}
}
```

```
@misc{alpaca_eval_length,
  author = {Yann Dubois and Balazs Galambosi and Percy Liang and Tatsunori B. Hashimoto },
  title = {Length-Corrected AlpacaEval: A Simple Debiasing of Automatic Evaluators},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/alpaca_eval}}
}
```

```
@misc{dubois2023alpacafarm,
  title={AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback}, 
  author={Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani and Jimmy Ba and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
  year={2023},
  eprint={2305.14387},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

# 更多信息

<details>
  <summary><h2 tabindex="-1" dir="auto">长度控制胜率</h2></summary>

长度控制（LC）胜率是控制输出长度的胜率的分化版本。

其主要思路是，对于每个模型，我们都将拟合一个逻辑回归来预测自动注释者的偏好，给定的偏好包括：（1）指令，（2）模型，（3）基线和模型输出之间的长度差异。
有了这样一个逻辑回归，我们就可以通过将长度差异设为 0 来尝试预测反事实 "如果模型输出与基线长度相同，偏好会是什么"。
通过对这一长度控制偏好进行平均，我们就得到了长度控制胜率。
逻辑回归的精确形式使得长度控制胜率的解释与原始胜率相似，例如，对于任何模型 "m1 "和 "m2"，我们有 "win_rate(m1, m2) = 1 - win_rate(m2, m1) \ in [0,100]` 和 "win_rate(m1, m1) = 0.5"。
受长度控制的胜率将 AlpacaEval 的排行榜与聊天竞技场之间的相关性从**0.93 提高到 0.98 Spearman 相关性，同时显著降低了注释者的长度游戏性**。
有关长度控制胜率的更多信息和结果，请参阅[本笔记本](https://github.com/tatsu-lab/alpaca_eval/blob/main/notebooks/length_correction.ipynb)。

通过预测结果，同时以中介因素（长度差异）为条件来估计受控直接效应，这种想法在统计推断中很常见。

要获取先前注释模型的 LC 胜率，可以使用下面的命令：

```bash
pip install -U alpaca_eval
alpaca_eval --model_outputs … --is_recompute_metrics_only True
```

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">AlpacaEval 2.0</h2></summary>
AlpacaEval 2.0 是 AlpacaEval 的新版本。以下是不同之处：

- **参考：gpt4_turbo**：我们将基准从text-davinci-003升级到了gpt4_turbo，以提高基准的挑战性，并采用更能反映当前技术水平的指标。
- **注释器：weighted_alpaca_eval_gpt4_turbo**：我们改进了注释器的质量和价格。首先，我们使用 "gpt4_turbo "模型进行注释，它比 "gpt4 "便宜约 2 倍。其次，我们改变了提示方式，使模型只输出一个标记，从而进一步降低了成本和速度。最后，我们不再使用二进制偏好，而是使用 logprobs 计算连续偏好，从而得出最终的加权胜率。需要注意的是，后两个变化产生了令人惊讶的效果，即减少了注释者的长度偏差。

默认情况下，从 `pip install alpaca_eval==0.5` 开始将使用 AlpacaEval 2.0。如果希望默认使用旧配置，可以在环境中设置 `IS_ALPACA_EVAL_2=False`。

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">数据发布</h2></summary>

作为 AlpacaEval 的一部分，我们发布以下数据：

- 人类注释 (17701)** 为了开发和理解自动评估器，我们发布了我们为 AlpacaFarm 收集的所有人类配对评估。其中包含 22 个模型与 AlpacaFarm 评估集上的 `text-davinci-003` 参考之间的比较。注释来自亚马逊 Mechanical Turk 上的 16 名人群工作者。不同的模型是 6 个来自 OpenAI，2 个来自 AlpacaFarm 的 SFT 模型，13 个来自 AlpacaFarm 的 RLHF 方法，以及 LLaMA 7B。
- 人类交叉注释（2596 个）** 为了进一步分析自动评估者，我们从 AlpacaFarm 评估集中选取了 650 个示例（通过对模型和数据集进行分层抽样），并为每个示例收集了 4 个人类注释。
- 羊驼评估集（805）** 我们对羊驼农场评估集略作修改/简化。特别是，我们首先将指令和输入字段合并为一个指令字段。这影响了 AlpacaFarm 评估集中 1/4 的示例，所有这些示例都来自 [自我指导评估集](https://arxiv.org/abs/2212.10560)。其次，我们重新生成了 text-davinci-003 参考输出，但没有限制其输出的长度。

有关人工注释的更多详情，请参阅[AlpacaFarm 论文](https://arxiv.org/abs/2305.14387)。

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">与羊驼农场AlpacaFarm的区别</h2></summary>

AlpacaEval 是对 [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) 中自动配对偏好模拟器的改进和简化。
在 AlpacaFarm 之外，您应该使用 AlpacaEval。
以下是主要区别：

- AlpacaEval 将指令和输入合并**： AlpacaEval 评估与 AlpacaFarm 评估相同，但指令和输入字段合并为 `{instruction}/n/n{input}`。这影响了 AlpacaFarm 评估集中 1/4 的示例（[self-instruct](https://arxiv.org/abs/2212.10560) 子集）。
  这种简化为那些没有通过区分两个字段来训练的模型提供了更公平的比较。
- **AlpacaEval 可处理更长的世代**： AlpacaFarm 中的模型最多只能生成 300 个代币。在 AlpacaEval 中，我们将这一数字改为 2000。请注意，这也会影响参考世代 (`text-davinci-003`)、
  因此 AlpacaEval 的结果无法与 AlpacaFarm 的结果相提并论，即使是在没有输入字段的示例中也是如此。
- AlpacaEval 消除了标注者内部和标注者之间的差异**： AlpacaFarm 模拟器在模式行为和多样性方面复制了人类注释。
  特别是，AlpacaFarm 模拟器使用模型和提示库，并添加噪音以复制人类标注者内部和标注者之间的差异。
  如果目标是使用自动注释器进行评估或仅仅是训练更好的模型，那么这种差异
  可能并不可取。因此，AlpacaEval 中的默认注释器没有这种差异。我们提供了一个选项，在创建评估器时使用 `--anotators_config 'alpaca_farm'` 和 `--p_label_flip 0.25` 将其添加回去。

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">相关工作</h2></summary>

目前已有多项研究提出了新的指令跟随模型自动注释器。在此，我们列出了我们所知道的几项工作，并讨论了它们与我们的工作有何不同。我们在[我们的评价器排行榜](https://github.com/tatsu-lab/alpaca_eval#evaluators)中对所有这些评价器进行了评估。

- 维库纳/lmsys** lmsys 注释器（"lmsys_gpt4"）通过要求注释者为每个输出从 1-10 分中打分，然后选择得分最高的输出作为首选来评估配对。他们不会对输出顺序进行随机化，而且会在评分后询问解释。总体而言，我们发现该注释器偏向于较长的输出（0.74），与人类注释的相关性相对较低（63.2）。
- AlpacaFarm** 最佳的 AlpacaFarm 注释器（"alpaca_farm_greedy_gpt4"）通过直接询问注释器它更喜欢哪种输出来评估配对。此外，它还会将 5 个示例集中在一起，以摊销提示的长度，并随机调整输出的顺序。总体而言，我们发现该注释器对较长输出的偏好要小得多（0.60），而且比其他注释器更快（878 秒/1000 个示例）。它与大多数人类注释的相关性（66.4）略高于人类注释本身（65.7）。
  不过，它的成本较高（15.3 美元/1000 个示例），而且由于批处理的原因，它无法处理很长的输出。
- Aviary**Aviary注释器（`aviary_gpt4`）要求注释者按照自己的偏好对输出排序，而不是简单地选择首选输出。它不会随机调整输出顺序，并使用高温进行解码（0.9）。总体而言，我们发现该注释器对较长输出的偏好相对较强（0.70），与人类注释的相关性非常高（69.1）。通过降低温度和随机化输出顺序，我们[进一步改善](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md)了相关性，达到 69.8（"improved_aviary_gpt4"），但长度偏差进一步增加到 0.73。

我们的 `alpaca_eval_gpt4` 是 AlpacaFarm 和 Aviary 注释器的混合体。它要求注释者按偏好对输出排序，但它使用温度 0，对输出进行随机化，并对提示进行了一些修改，从而将长度偏差降至 0.68。

其他相关工作包括最近发表的分析自动评价器的论文。
例如

- 羊驼农场应用程序 C](https://arxiv.org/abs/2305.14387)和 [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926v1) 都发现自动注释者有立场偏差。
- 羊驼农场第 5.2 节](https://arxiv.org/abs/2305.14387)和 [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717) 均发现自动注释者偏爱风格（如使用列表、语气、选词、长度）而非事实。

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">解释注释</h2></summary>

对于所有模型，您都可以在 `results/<model_name>/*/annotations.json`下找到自动注释。注释有以下几列：

- instruction`：提示
- 生成器_1"：基准模型
- 输出_1"：基准模型的输出
- 生成器_2"：正在评估的模型
- `output_2`: 被评估模型的输出结果
- 注释器`：自动注释器
- preference `：自动标注器的结果。这是一个介于 1 和 2 之间的浮点数。接近 1 表示自动注释器更喜欢`输出_1 `，接近 2 表示自动注释器更喜欢`输出_2 `。对于 AlpacaEval 2.0，"preference-1 "对应于 "输出_1 "被首选的概率。对于 AlpacaEval 1.0，如果 `输出_1 `是首选，则 `偏好 `为 1；如果 `输出_2 `是首选，则 `偏好 `为 2；如果两者相同，则 `偏好 `为 1.5。胜率总是`(preference-1).mean()`。
- raw_completion`：自动注释器的原始输出。

**思维链**

对于某些注释器，例如 `alpaca_eval_cot_gpt4_turbo_fn`，我们使用**思维链推理**来使模型偏好更易于解释。这些内容可以在 "concise_explanation "中找到。要解释它们，还应该查看 `referenced_models` ，它将临时模型名称（在提示中）转换为实际输出。下面，我们将就幕后发生的事情提供更多解释。

您可以查看 `annotations.json` 中的 `raw_annotations["concise_explanation]` 列（例如 [此处](https://github.com/tatsu-lab/alpaca_eval/tree/main/results/gpt4/alpaca_eval_cot_gpt4_turbo_fn/annotations.json)），其中包含自动注释器的思维推理链。请注意，raw_annotations 不会因输出顺序的随机化而改变。特别是，`"m"` 和 `"M"`有时指第一个模型（参考），有时指第二个模型（正在评估的模型）。要了解所指的是哪个模型，应使用列 `preference` 和 `ordered_models`。为了方便起见，我们添加了一列 `"referenced_models"`，将模型名称映射到相应的输出。例如，在下面的注释中，我们可以看到首选项是 1.0（即 `output_1`），对应于 `concise_explanation` 中的模型 `M`（见 `ordered_models`）。

```json
{
  "说明"： "美国各州的名称是怎么来的？"
  "输出_1"： "美国各州的名称来源于各种语言和资料，包括美国本土语言、欧洲语言（主要是英语、西班牙语和法语）以及其他历史影响。以下是一些州名由来的简要概述：\n\n1. **美国土著语言**： 许多州的名称来自美洲原住民的词汇或部落名称，通常来自生活在这些地区或附近的原住民的语言： Possibly derived from the Choctaw language, meaning \"thicket clearers.\"\n - **康涅狄格州**： 源于一个 Mohegan-Pequot 词，意思是 "潮汐长河"： [...]",
  "generator_1"： "gpt4_1106_preview",
  "数据集"： "helpful_base",
  "输出_2"： "美国 50 个州的名称有多种来源，包括美洲土著语言、欧洲语言和历史人物。下面简要介绍一些州的名称由来：\n\n1. 美洲原住民的起源： 许多州的名称源自美洲土著语言。例如，阿拉巴马州来自乔克托语（Choctaw），意思是 "植物采集者 "或 "草药采集者"。同样，密西西比（Mississippi）这个名字来自奥吉布韦语（Ojibwe），意思是大河。欧洲语言： [...].",
  "generator_2"： "gpt4",
  "annotator"： "alpaca_eval_cot_gpt4_turbo_fn",
  "偏好"： 1.0,
  "raw_completion"： {
    "concise_explanation"： "模式 M 提供了更详细、更有条理的答复，包括每个类别的粗体标题和更广泛的例 子。它还增加了 "其他欧洲语言 "和 "语言与影响的结合 "等类别，从而增加了解释的深度。模型 m 的答案准确，但不够全面，而且缺乏模型 M 答案中的清晰结构",
    "有序模型"： [
      {
        "模型"： "M",
        "等级"： 1
      },
      {
        "model"： "m",
        "等级"： 2
      }
    ]
  },
  "参考模型"： {
    "M": "output_1"、
    "m": "output_2"
  }
}
```

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">主要更新</h2></summary>

- 2024 年 3 月 12 日：更新为使用长度控制（LC）胜率。这是控制输出长度的赢率的去势版本。
- 2024 年 1 月 3 日：更新至 AlpacaEval 2.0，使用 GPT4-turbo 作为基准和注释器。
- 2024 年 1 月 2 日：添加了 Azure API 和更通用的客户端配置设置方法。参见 [此处](https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs/README.md)
- 2023 年 6 月 19 日：添加任何人都可使用的排行榜 `chatgpt_fn`（无等待名单）。
- 2023 年 6 月 19 日：更新为使用 [OpenAI 的函数调用](https://openai.com/blog/function-calling-and-other-api-updates)。
  举例说明： [`chatgpt_fn`](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/chatgpt_fn) 或 [`alpaca_eval_gpt4_fn`](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4_fn).

</details>
