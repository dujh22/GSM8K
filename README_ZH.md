# GSM小学数学

#### [[Blog Post]](https://openai.com/blog/grade-school-math/) [[Paper]](https://arxiv.org/abs/2110.14168)

最先进的[语言模型](https://so.csdn.net/so/search?q=%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020)可以在许多任务上与人类的表现相匹配，但它们仍然难以稳健地进行 **多步骤的数学推理** 。为了诊断当前模型的失败并支持研究，我们发布了GSM8K，一个由8.5K高质量的语言多样化的**小学数学**单词问题组成的数据集。我们发现，尽管这个问题分布在概念上很简单，但即使是最大的 [Transformer](https://so.csdn.net/so/search?q=Transformer&spm=1001.2101.3001.7020) 模型也不能达到很高的测试性能。

<p align="center">
    <img src="grade_school_math/img/example_problems.png" height="300"/>
</p>

## 数据集详情

GSM8K 由 **8.5K** 高质量的小学数学问题组成，这些问题都是由人类写手创造的。我们将这些问题分为  **7.5K 训练问题和 1K 测试问题** 。这些问题 **需要 2 到 8 个步骤来解决** ，解决方法主要是使用基本的算术运算（+ - / *）进行一连串的基本计算，以得出最终答案。一个聪明的中学生应该能够解决每个问题。

原始数据文件可以在以下地方找到。

- `grade_school_math/data/train.jsonl`
- `grade_school_math/data/test.jsonl`

这些文件的每一行都对应于一个单一的小学数学问题，保存为 json 字典（有一个 "问题 "键和一个 "答案 "键）。答案的格式是这样的：它使用了计算注释，并使最终的数字解成为解决方案的最后一行，前面是 `####`。

### 计算注释

我们的模型经常不能准确地进行计算。尽管大型模型比小型模型犯的算术错误要少，但这仍然是一个常见的错误来源。为了缓解这个问题，我们通过向训练集注入计算注释来训练我们的模型使用计算器。在训练时，我们只是在这个语言数据上按原样进行微调。在测试时，当模型选择使用这些注释时，计算器将覆盖采样。在 `calculator.py` 中可以找到一个计算器采样的实施例子。

如果你想删除计算器注释，只需删除任何以 `<<`开头和 `>>`结尾的字符串。

### 解的提取

要提取某个特定问题的最终数字解决方案，只需解析完成度，提取紧随 `####` 标记的数字值。在 `dataset.py:is_correct` 中显示了一些这样做的python代码示例。

### Socratic 数据集

在研究过程中，我们还研究了一种修改后的解决方案格式，在每个步骤之前注入自动生成的 “苏格拉底式子问题”。虽然我们最终没有在论文中的任何实验中使用这种格式，但我们向任何有兴趣的人提供了这些数据。

我们在下面展示了一个例子，其中苏格拉底式的子问题用粗体表示。

<pre>
A carnival snack booth made $50 selling popcorn each day. It made three times as much selling cotton candy. For a 5-day activity, the booth has to pay $30 rent and $75 for the cost of the ingredients. How much did the booth earn for 5 days after paying the rent and the cost of ingredients?
<b>How much did the booth make selling cotton candy each day? **</b> The booth made $50 x 3 = $<<50*3=150>>150 selling cotton candy each day.
<b>How much did the booth make in a day? **</b> In a day, the booth made a total of $150 + $50 = $<<150+50=200>>200.
<b>How much did the booth make in 5 days? **</b> In 5 days, they made a total of $200 x 5 = $<<200*5=1000>>1000.
<b>How much did the booth have to pay? **</b> The booth has to pay a total of $30 + $75 = $<<30+75=105>>105.
<b>How much did the booth earn after paying the rent and the cost of ingredients? **</b> Thus, the booth earned $1000 - $105 = $<<1000-105=895>>895.
</pre>

我们通过对解决方案中的每个基本事实（承包商提供的）步骤进行调节，使用专门为这项任务微调的模型（大约800个例子），生成每个苏格拉底式的子问题。为了构建完整的Socratic 数据集，解决方案中的每个步骤都以模型生成的Socratic子问题为前缀。其他步骤则不作任何改动。

这些数据文件可以在:

- `grade_school_math/data/train_socratic.jsonl`
- `grade_school_math/data/test_socratic.jsonl`

## 查看模型解决方案

对于每道测试题，我们提供由6B微调、6B验证、175B微调和175B验证产生的解决方案。这些数据可以在以下内容中找到。

- `grade_school_math/data/example_model_solutions.jsonl`

要逐个问题查看这些结果，请运行

```bash
python view_model_solutions.py
```

注意：这些模型生成的样本使用了一个稍旧版本的计算器。以前的实现错误导致计算器在大约1%的模型样本中失败。这些问题已经在代码库中被修复，但是由于样本没有被重新生成，偶尔会出现计算错误。

## 引用

Please use the below BibTeX entry to cite this dataset:

```
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

# 用法

我们介绍一个训练GPT2大小的模型的基本例子，并在采样过程中使用计算器。我们包括这个代码只是为了说明问题。这条管道没有用于本文的任何实验。

训练一个模型

```bash
python train.py
```

从模型中取样

```bash
python sample.py
```

核心的计算器采样逻辑可以在calculator.py:sample中找到。请注意，这段代码的实现是低效的。具体来说，该函数不支持批处理，也不缓存以前的令牌的激活。
