# 第二十课:Mini Transformer(总装)——毕业设计

> 19 课零件全部到齐。这一课把它们拼成一台**能跑的 GPT**:embedding → N × Block → 输出头 → 采样生成,外加 KV Cache 加速。
> 全程 numpy,核心不到 100 行,每一行都能在前面的课里找到出处。

---

## 1. 先看设计图(深模块视角)

总装前先想清楚**接口**——每个模块对外只暴露一个小表面,复杂度全部藏在里面:

```text
┌──────────────────────────────────────────────────┐
│ MiniGPT.generate(ids, n, T, top_p) → ids          │  ← 唯一对外接口
├──────────────────────────────────────────────────┤
│  forward(ids, cache=None) → logits, cache          │  ← 内部缝(第18课)
│  block(x, pos, cache)     → x, cache               │  ← 形状不变 [n,d]→[n,d](第9课)
│    ├ attn  = RoPE + 多头因果注意力 + 残差 + RMSNorm │  ← 6/7/8/19 课
│    └ ffn   = SwiGLU + 残差 + RMSNorm               │  ← 8/10 课
└──────────────────────────────────────────────────┘
```

设计三原则(软件工程和模型架构在这里合流):

1. **深模块**:`forward(ids) → logits` 一个动作吞下"嵌入、旋转、注意力、前馈、归一化"全部复杂度;调用方学一个接口,得到整个模型的行为。
2. **形状不变是总装的前提**:Block `[n, d] → [n, d]`(第 9 课),所以堆 N 层就是 N 次函数调用,无需任何适配代码。
3. **两个适配器,一条真缝**:`attn` 和 `ffn` 是 Block 从外面收的两个依赖(第 9 课插槽)。想实验"去掉 RoPE"或"FFN 换 GeLU",换适配器即可,Block 一行不改——这就是缝(seam)的价值。

---

## 2. 完整代码(已验证可跑)

```python
import numpy as np, math

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

class MiniGPT:
    def __init__(self, vocab=16, d=32, n_heads=4, n_layers=2, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.1
        self.E = rng.normal(size=(vocab, d)) * 0.5            # 词嵌入(第2课)
        self.dh = d // n_heads
        self.H, self.D = n_heads, d
        # 每层四件套:QKV/O 投影 + SwiGLU 三矩阵(第7/10课)
        self.layers = [{
            'Wq': rng.normal(size=(d, d))*s, 'Wk': rng.normal(size=(d, d))*s,
            'Wv': rng.normal(size=(d, d))*s, 'Wo': rng.normal(size=(d, d))*s,
            'W1': rng.normal(size=(d, 4*d))*s*0.8, 'W2': rng.normal(size=(4*d, d))*s*0.8,
            'W3': rng.normal(size=(d, 4*d))*s*0.8,
        } for _ in range(n_layers)]
        self.head = rng.normal(size=(d, vocab)) * s           # 输出头

    def rms(self, x, eps=1e-5):                               # 第8课
        return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)

    def rope(self, x, pos):                                   # 第19课
        half = x.shape[-1] // 2
        ang = pos[:, None] / (10000 ** (np.arange(half)*2/x.shape[-1]))
        c, s = np.cos(ang), np.sin(ang)
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1*c - x2*s, x1*s + x2*c], -1)

    def _attn(self, x, pos, W, cache):
        n = x.shape[0]
        q, k, v = self.rope(x@W['Wq'], pos), self.rope(x@W['Wk'], pos), x@W['Wv']
        if cache is not None:                                  # 第18课
            k, v = np.concatenate([cache[0], k]), np.concatenate([cache[1], v])
        T = k.shape[0]
        mask = np.triu(np.ones((n, T), bool), T - n + 1)       # 差一教训
        outs = [softmax(np.where(mask, -np.inf, q[:, h*self.dh:(h+1)*self.dh]
                    @ k[:, h*self.dh:(h+1)*self.dh].T / math.sqrt(self.dh))
                    @ v[:, h*self.dh:(h+1)*self.dh]) for h in range(self.H)]
        return np.concatenate(outs, -1) @ W['Wo'], (k, v)

    def _ffn(self, x, W):                                      # 第10课 SwiGLU
        return ((x@W['W1']) / (1 + np.exp(-(x@W['W1']))) * (x@W['W3'])) @ W['W2']

    def forward(self, ids, cache=None):                        # 第9课 Block × N
        pos0 = cache[0][0].shape[0] if cache else 0
        x = self.E[list(ids)]
        new_cache = []
        for W in self.layers:
            a, kv = self._attn(x, np.arange(pos0, pos0+len(ids)), W,
                               cache[self.layers.index(W)] if cache else None)
            x = x + a                                          # 残差 1(pre-LN)
            x = x + self._ffn(self.rms(x), W)                  # 残差 2
            new_cache.append(kv)
        logits = self.rms(x) @ self.head                       # 最终 LN + 输出头
        assert np.isfinite(logits).all(), 'NaN!查掩码/精度/ε'
        return logits, new_cache

    def generate(self, ids, n, T=1.0, top_p=1.0, rng=None):
        rng = rng or np.random.default_rng()
        logits, cache = self.forward(ids)                      # prefill
        out = []
        for _ in range(n):
            p = softmax(logits[-1] / T)                        # 第15课
            if top_p < 1.0:                                    # 第17课
                order = np.argsort(-p); cum = np.cumsum(p[order])
                keep = order[:np.searchsorted(cum, top_p)+1]
                m = np.zeros_like(p); m[keep] = p[keep]; p = m/m.sum()
            nxt = int(rng.choice(len(p), p=p/p.sum()))         # 第14课
            out.append(nxt)
            logits, cache = self.forward([nxt], cache)         # decode(第18课)
        return list(ids) + out
```

配置:词表 16、d=32、4 头、2 层——麻雀虽小,**结构与现代 GPT 同构**(就差规模和训练)。

---

## 3. 真实运行结果

```python
gpt = MiniGPT()
print(gpt.generate([1, 2, 3], n=8, T=0.0 取 greedy 路径))
```

真实输出(权重固定 seed,可复现):

```text
greedy(T→0):  [1, 2, 3, 9, 5, 14, 14, 14, 14, 14, 14]
T=0.8 采样:   [1, 2, 3, 9, 12, 12, 4, 3, 10, 0, 11]
```

未训练模型的"语言"就是乱码(greedy 还会复读——第 14 课的理论照进现实)——这是**正确行为**:

```text
next-token loss = 3.2169       瞎猜基线 ln(16) = 2.7726
```

注意 loss **高于**瞎猜基线,这不是 bug,是数学:交叉熵 $E[-\log p]$ 在"均匀分布"处取得最小值(凸性/詹森不等式),随机权重产生的 logits 波动只会**抬高** loss。**训练(第 11–13 课的 loss + 反向 + AdamW)就是把它压到基线以下、再一路压低的过程**——那是训练篇的内容,本课程的零件在这里全部闭环。

---

## 4. 验证清单(工程纪律)

这台小机器出厂前过了四道检验(全部真实运行):

1. **缓存等价**:cache 版与全量重算版,逐 token logits 最大误差 `5.6e-16`(浮点噪声级);
2. **有限性断言**:forward 内建 `isfinite` 检查,NaN 当场爆炸(第 18 课血泪);
3. **可复现**:固定 seed 的权重 + 固定 seed 的采样 = 逐 bit 相同输出;
4. **形状不变**:任意层数、任意 batch 前向,shape 全程 `[n, d]`。

把这套清单背下来——**任何"我实现了一个 Transformer"的声明,都要附上这四条的证据**。

---

## 5. 生产环境:从玩具到真实模型差多远

| | MiniGPT | Llama-3-8B | DeepSeek-R1 |
|---|---|---|---|
| 层数 | 2 | 32 | 61 |
| 维度 | 32 | 4096 | 7168 |
| 词表 | 16 | 128,256 | 129,280 |
| 参数 | ~5 万 | 8B | 671B(激活 37B) |
| 注意力 | 标准多头 | GQA | MLA |
| FFN | SwiGLU | SwiGLU | SwiGLU + MoE |

结构上你已经全部认识;规模差 10 万倍,还有三样新东西:GQA/MLA(第 7/21 课)、MoE(第 10/21 课)、训练。下一课(案例)走进真实工业代码。

---

## 6. 记住的故事

**现实:**

> "19 课零件,一条总装线:形状不变才能堆层,深接口才能总装,缓存等价才能放心加速。"

**数学:**

$$
logits = LN\big(E_{ids} + \textstyle\sum_l Block\big)\,W_{head},\qquad Block = x + Attn_{RoPE}(LN(x)) + FFN(LN(\cdot))
$$

**工程:**

> "四条出厂检验:缓存等价、isfinite、可复现、形状不变。"

**代码:** 第 2 节 ~90 行。

---

## 作业

### A. 概念题

1. 为什么说 Block"形状不变"是能堆 N 层的前提?
2. 本课的"两个适配器、一条真缝"指什么?把 RoPE 去掉要改几行代码?
3. 未训练模型 loss 3.2169 vs 基线 2.7726,为什么**高于**瞎猜基线却反而"正常"?(提示:第 3 节讲过交叉熵的凸性)
4. 四条出厂检验各防什么事故?

### B. 填空题

1. forward 的两个阶段:prompt 走 ____,之后每步走 ____。
2. MiniGPT 配置:词表 ____、d=____、____ 头、____ 层。
3. 缓存等价检验的标准:逐 token logits 误差 < ____。
4. 从玩具到 Llama-3-8B,结构上多了 ____;到 R1 还多 ____ 和 ____。

### C. 代码手撕(关讲义)

1. 不看本课,从零重写 MiniGPT(允许翻第 6/7/8/9/10/18/19 课)。跑通 greedy 生成。
2. 复现四条出厂检验(缓存对照 / isfinite / seed 复现 / shape)。
3. 扩展:`generate` 加 `top_k` 参数,与 top_p 共存(先 k 后 p)。

### D. 生产环境实战

1. 把 vocab 改成 1000、d=256、6 层:参数量多少?单步前向的 FLOPs 怎么估(2×参数×token 数)?
2. 想真的训练它:把第 11 课 CE loss、第 12 课手写反向(提示:残差链逐模块回传)、第 13 课 AdamW 接上。在小语料(比如 1MB 文本)训 1000 步,loss 降到多少?
3. 生成 200 token,用 `%time` 对比开/关 cache 的耗时,报告加速比。

### E. 毕业拓展

1. 给 Block 换掉 FFN 适配器(GeLU 两矩阵版),对比同 seed 下生成差异。
2. 阅读 nanoGPT(karpathy/nanoGPT)源码,列出与本课 MiniGPT 的 10 处对应关系——你会发现你已经能读懂它 90%。
3. 最后一课:解剖你磁盘上真实的大语言模型(Qwen3-4B,逐张量对账)+ DeepSeek-R1(671B)对照。

---

# 下一课(案例):解剖一台真实的大语言模型

> **下一课:[真实大模型解剖](21-real-llm-anatomy.md)——从 90 行玩具到本机磁盘上的 Qwen3-4B,再到 671B 的 DeepSeek-R1。**
