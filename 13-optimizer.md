# 第十三课:SGD / Adam——参数更新的艺术

> 梯度说"往这边下坡",优化器决定"迈多大步、带多少惯性、要不要热身"。
> 现代大模型几乎清一色 AdamW。这一课手撕它,并算清"为什么它这么吃显存"。

---

## 1. 先不用代码——三种司机

下山(loss 曲面),三种开车风格:

**SGD(最朴素):** 每步都严格沿当前坡向走固定步长。

$$
w \leftarrow w - \eta \, g
$$

病:峡谷地形里来回震荡(这个方向陡、那个方向缓,同一学习率对陡的太大、对缓的太小)。

**Momentum(动量):** 记住之前的方向,惯性滑行,震荡相互抵消。

$$
m \leftarrow \beta m + g,\qquad w \leftarrow w - \eta\, m
$$

**Adam(现代默认):** 动量 + 自适应步长——每个参数**自己的**学习率(按自己梯度的历史波动调整),再配冷启动偏差修正。三件套齐活。

---

## 2. Adam 公式逐行手撕

对每个参数 $w$,维护两个滚动统计:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t & &\text{一阶矩:梯度的滑动平均(方向)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 & &\text{二阶矩:梯度平方的滑动平均(波动)}\\
\hat m_t &= \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t} & &\text{偏差修正(冷启动救火)}\\
w &\leftarrow w - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

人话版:

- **m**:"最近梯度平均往哪走"——噪声抵消,方向稳定(动量);
- **v**:"最近梯度抖多凶"——抖得凶的参数**自动小步**,稳的参数大步(自适应);
- **偏差修正**:m、v 从 0 起步,前几步系统性偏小;除以 $1-\beta^t$(t 步后趋于 1)把冷启动拉回正常量级;
- **ε**(1e-8):防除零,和第 8 课 LayerNorm 的 ε 一个道理。

默认超参 $\beta_1=0.9,\ \beta_2=0.999$——面试要背。

### AdamW:Adam 的必用修补

Adam 把权重衰减($L_2$ 正则)混进梯度里,数学上不等价且伤自适应。**AdamW 把衰减直接作用在权重上**:

$$
w \leftarrow w - \eta\left(\frac{\hat m}{\sqrt{\hat v}+\epsilon} + \lambda w\right)
$$

现代 LLM 标配:`AdamW(lr, weight_decay=0.1 左右)`。

---

## 3. 真实运行:Adam 300 步收敛

玩具任务:$w$ 从 5.0 收敛到 3.0(最小化 $(w-3)^2$),lr=0.05(真实运行):

```text
loss: 4.0 → 0.00000004      w: 5.0 → 3.0002
```

**偏差修正对照实验(真实数据,可能和直觉相反)**:本课 notebook 里去掉修正($m,\sqrt v$ 不除 $1-\beta^t$),前 5 步 loss 反而降得更快(0.71 vs 3.06)。原因:$v$ 的冷启动滞后($\beta_2=0.999$,收敛极慢)远大于 $m$($\beta_1=0.9$),不修正时有效步长被放大 $\sqrt{(1-0.9^t)/(1-0.999^t)}$ 倍(第 5 步约 **9 倍**)——步子过猛在这个良性二次函数上侥幸加速;在真实含噪 loss 上,这种不受控的放大就是发散源。**修正的本质:把 m、v 拉回无偏量纲,让步长按设计值执行。**

---

## 4. 生产账本:优化器状态的显存(面试高频)

Adam 的每个参数要多存 **m 和 v 两个浮点数**。以 fp16 混合精度训练计:

```text
每个参数:权重 2B(fp16)+ 梯度 2B + m 4B + v 4B(fp32)= 12 字节

7B 模型:7e9 × 12 ≈ 84 GB —— 权重只占 14GB,其余 70GB 全是"记账本"
```

这就是为什么大模型训练动辄几十张 A100/H100,也是 ZeRO/DeepSpeed(把优化器状态切片到多卡)存在的理由。**模型本身不贵,记账贵。**

---

## 5. 代码:手撕 AdamW

```python
import numpy as np

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, *betas, eps, wd
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0

    def step(self, grads):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)      # 偏差修正
        v_hat = self.v / (1 - self.b2 ** self.t)
        self.p = self.p - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * self.p)
```

---

## 6. 生产环境:学习率是第一超参

- **warmup(热身)**:前几百~几千步学习率从 0 线性升到峰值再衰减。Adam 的 m、v 冷启动不稳 + 残差主干尺度未成形,一开始就大步容易把模型推下悬崖。GPT-3 级训练 warmup 约 2% 总步数。
- **衰减日程**:峰值后 cosine 衰减到 10%(现代主流)。
- **典型 LLM 配置**:7B 预训练峰值 lr ≈ 3e-4(AdamW,cosine + warmup);SFT/RL 阶段小一到两个数量级(1e-5 ~ 5e-5)。
- **R1 案例**(第 21 课):RL 阶段的策略优化器超参极其敏感——训练不稳直接毁掉长链推理的涌现。

---

## 7. 记住的故事

**现实:**

> "动量记方向,二阶矩定步幅,修正救冷启动,衰减管在权重上。"

**数学:**

$$
w \leftarrow w - \eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}\;(+\lambda w \text{ for AdamW})
$$

**工程:**

> "每参数 12 字节;warmup+cosine;7B 峰值 3e-4;ZeRO 切记账本。"

**代码:** 第 5 节 AdamW 类 15 行。

---

## 作业

### A. 概念题

1. SGD 在"峡谷地形"的病是什么?Momentum 怎么治?
2. m 和 v 各统计什么?为什么 v 大的参数要小步?
3. 偏差修正解决什么问题?为什么 t 大了它自动失效?
4. AdamW 和 Adam + L2 的区别?为什么前者对?
5. 为什么优化器状态比权重本身还占显存?

### B. 填空题

1. Adam 默认 β₁=____、β₂=____、ε=____。
2. fp16 混合精度下每参数字节 = 权重__ + 梯度__ + m__ + v__ = 共 ____。
3. warmup 是训练初期把学习率从 ____ 升到 ____;主流衰减日程是 ____。
4. 7B 预训练 AdamW 峰值学习率约 ____ 量级。

### C. 代码手撕(关讲义)

1. 默写 `AdamW.step`(含偏差修正和权重衰减)。
2. 用它重跑第 3 节玩具(loss 4.0 → <1e-6)。
3. 对照实验:去掉偏差修正(`m_hat=m, v_hat=v`),前 50 步 loss 曲线怎么变?打印对比。
4. 把第 12 课的 MLP 接上你的 AdamW,训练 500 步,loss 降到多少?

### D. 生产环境实战

1. 13B 模型、AdamW、fp16 混合精度:优化器状态+梯度+权重共多少 GB?需要几张 80GB 卡(不考虑激活)?
2. 训练 200 步后 loss 突然翻倍并横盘。给出排查顺序(学习率日志、梯度范数、哪层溢出)。
3. 查证:ZeRO-1/2/3 各切分什么?一句话一个。

### E. 拓展

1. 推导:为什么 $m/(1-\beta_1^t)$ 在 t→∞ 时就等于 m?(β^t 趋向哪?)
2. 预习钩子:模型训好了,怎么从 logits 里"挑"出下一个词?永远选最大概率的吗?——下一课进入推理篇。

---

# 下一课预告:Token Sampling

训练篇结束,推理篇开始。模型吐出 logits,下一步是"选词"。永远贪心选最大?——会死循环复读。随机采样?——会精神分裂。生产系统在两者之间跳舞。

> **下一课:[Token Sampling](14-sampling.md)——生成的第一步。**
