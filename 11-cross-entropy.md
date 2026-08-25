# 第十一课:Cross Entropy——大模型的唯一损失函数

> 模型输出概率,真相只有一个词。怎么量化"给真词的概率不够高"?
> 交叉熵——预训练一个 loss 从头用到尾,和第 1 课的 softmax 是天生的搭档。

---

## 1. 先不用代码——老师怎么打分

学生(模型)对下一道填空题给出信心分配:

```text
饭:65%, 面:24%, 菜:10%
```

标准答案:饭。老师打分有两种风格:

**风格 A(数数对错):** 对了满分错了零分。置信 65% 和置信 99% 得分一样——**分数里没有"多确信"的信息,梯度传不动。**

**风格 B(按置信度扣分):** 答案是饭,你给饭 65% 的信心——那我就扣你一个和 $-\log(0.65)$ 成正比的分。**信心越足,扣分越少;给错了对象,重罚。**

风格 B 就是交叉熵:

$$
CE = -\log p_{target}
$$

三个性质(为什么是 $-\log$):

- 答对且极有信心($p \to 1$):$-\log p \to 0$,不扣分;
- 给真词的概率很低($p \to 0$):$-\log p \to \infty$,重罚;
- 处处光滑可导——梯度能带着模型往"给真词更高概率"的方向爬。

---

## 2. 手算 + 真实运行

logits = `[2.0, 1.0, 0.1]`,目标词 = 第 0 个(真实运行):

```text
softmax → p = [0.659, 0.242, 0.099]

CE = -log(0.659) = 0.417
```

对照理解:

```text
如果 logits 是 [5.0, 1.0, 0.1]  → p₀ ≈ 0.976 → CE ≈ 0.024   ← 有信心,轻罚
如果 logits 是 [0.1, 2.0, 1.0]  → p₀ ≈ 0.099 → CE ≈ 2.31    ← 给低了,重罚
```

**训练就是全语料上千万亿次地把这个数字压小。**

---

## 3. 数值稳定版:和第 1 课是同一个坑

朴素写法 `loss = -log(softmax(logits)[t])` 在 logits 大时会溢出。稳定做法——**不要先算概率再取对数**,直接在对数域算(log_softmax):

$$
\log p_t = \underbrace{(x_t - \max)}_{\text{第1课的减max}} - \log\sum_j e^{x_j - \max}
$$

真实运行(与朴素法对照,两者一致):

```text
CE(朴素法) = 0.417
CE(稳定法) = 0.417     ← logits ±1000 时只有稳定法能活
```

生产的 `F.cross_entropy(logits, target)` 内部就是这么做的——**它根本不显式算 softmax**,logits 直接进,梯度直接回(见第 12 课的漂亮结论)。

---

## 4. 一个训练 batch 的 loss

真语料每个位置都有一个"下一个词"要预测。一个 batch 的 loss = 所有位置 CE 的平均:

```text
序列 [w0, w1, w2, w3]:
  w0 → 预测 w1,算一个 CE
  w1 → 预测 w2,算一个 CE
  w2 → 预测 w3,算一个 CE
loss = 三个 CE 的平均         ← 每个位置都是监督信号,一次前向全拿到
```

这就是"next token prediction"的全部监督来源——**没有一个人类标注,整部互联网的文本都是教材**。

### 进阶:label smoothing(面试加分项)

硬目标"100% 是饭"对模型太苛刻(数据本来就有噪声)。软化一点:

$$
CE_{smooth} = (1-\epsilon)\cdot(-\log p_t) + \epsilon \cdot \text{mean}_j(-\log p_j)
$$

真实运行,$\epsilon = 0.1$:CE 从 0.417 → 0.5137。给错误答案留 $\epsilon$ 的量,防过度自信。现代 LLM 预训练一般不用(数据量足够大),分类任务常用。

---

## 5. 代码

```python
import numpy as np

def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)

def cross_entropy(logits, target):
    # logits: [vocab]  target: 下标
    x = logits - logits.max()
    return -(x[target] - np.log(np.exp(x).sum()))      # 稳定版,一步到位
```

batch 版(生产形状 `[batch, seq, vocab]` 对 `[batch, seq]]` 目标)用向量化索引取对角:

```python
def cross_entropy_batch(logits, targets):              # [n, vocab], [n]
    x = logits - logits.max(-1, keepdims=True)
    log_p = x - np.log(np.exp(x).sum(-1, keepdims=True))
    return -log_p[np.arange(len(targets)), targets].mean()
```

---

## 6. 生产环境:看 loss 曲线读健康度

- **正常曲线**:初期 10 左右(词表 10 万时 ≈ ln(100000) ≈ 11.5,即"均匀瞎猜"的熵),一路下降到 2 上下,越降越慢。
- **loss 飙到 inf/NaN**:第 1 课的溢出在 loss 处爆发——查是否裸写了 `log(softmax)`。
- **loss 降不下去**:学习率、数据、目标错位(常见事故:targets 没对齐,预测的是 w_{t+2})。
- **eval loss 降、生成质量差**:loss 只衡量"下一个词的困惑度",不保证长文连贯——这就是为什么还要 14–17 课的采样策略。

---

## 7. 记住的故事

**现实:**

> "按置信度扣分:给真词的信心每翻一倍,扣分减一档(log 尺度)。"

**数学:**

$$
CE = -\log p_{target},\qquad \log p_t = (x_t - \max) - \log\textstyle\sum_j e^{x_j-\max}
$$

**工程:**

> "稳定版 = log_softmax,不显式算 softmax;batch 平均;初始 loss ≈ ln(vocab)。"

**代码:** 第 5 节 4 行。

---

## 作业

### A. 概念题

1. 为什么用 $-\log$ 而不是"对错计数"?(从梯度角度答)
2. $p \to 0$ 时 CE 趋向什么?这保证了什么训练行为?
3. 数值稳定版为什么要"在对数域一步算完"?
4. 预训练的监督信号从哪来?"next token prediction"一个 batch 有多少个监督点?

### B. 填空题

1. CE = -____ ;p=1 时 CE = ____,p=0.5 时 ≈ ____(ln 口径)。
2. 词表 100,000、完全瞎猜时 loss ≈ ____。
3. 稳定公式:log p_t = (x_t - ____) - log ____(...)。
4. label smoothing ε=0.1 给错误答案总共留 ____ 的概率质量。

### C. 代码手撕(关讲义)

1. 默写稳定版 `cross_entropy(logits, target)`,用 `[2.0, 1.0, 0.1]` 验证 0.417。
2. 默写 batch 版,构造 3 行 logits 和目标,验证与逐个调用平均一致。
3. 用 `logits*1000` 自测稳定版不炸,朴素版(`-log(softmax)`)炸或溢出警告。

### D. 生产环境实战

1. 词表 128,256:训练第一步 loss 应该在多少附近?如果你的模型第 1 步 loss 是 0.8,最先查什么?
2. 同事报"loss 训着训着变 inf"。列出你的排查顺序(提示:第 3 节 + 第 8 课)。
3. targets 错位一位(shift 错了)会发生什么?loss 大概在什么水平晃、为什么降不下去?

### E. 拓展

1. 算三个分布的熵:均匀 [1,1,1]、[0.7,0.15,0.15]、one-hot。CE 和熵的关系是什么?(预测分布=真实分布时 CE=熵。)
2. 预习钩子:loss 有了,梯度怎么"穿过" softmax 回到 logits?下一课反向传播——有一个美到犯规的结论等着你。

---

# 下一课预告:Backpropagation

链式法则把 loss 的梯度一层层往回传。而 softmax + 交叉熵这对搭档的反向梯度,化简后只剩四个字母:**p − onehot**。整条链最优雅的一环。

> **下一课:[Backpropagation](12-backprop.md)——梯度怎么往回爬。**
