# 第十八课:KV Cache——推理提速的第一杠杆

> 生成第 100 个 token 时,前 99 个词的 K、V 早就算过了,而且**一个数都不会变**。
> 那为什么还要每步全量重算?存起来。这就是 KV Cache——自回归推理最重要的优化,没有之一。

---

## 1. 先不用代码——重复记账的浪费

无缓存的朴素生成:

```text
生成第   1 个 token:前向 1 个 token
生成第   2 个 token:前向 2 个 token(第 1 个的全部重算!)
生成第 100 个 token:前向 100 个 token(前 99 个的全部重算!)
...
生成第 n 个:总计算量 = 1+2+...+n = O(n²)
```

关键观察:causal 注意力下,**旧 token 的 K、V 只取决于自己和更早的 token**——新增 token 不会改变它们。第 100 步算出的"第 1 个词的 K"和第 1 步算出的**一模一样**。

解法:每层把历史 K、V 存进缓存,每步只前向**新来的那一个** token:

```text
prompt(3 个词)     → 前向 3 个,缓存 K/V
生成第 1 个 token  → 前向 1 个,拼接缓存,缓存变 4
生成第 2 个 token  → 前向 1 个,缓存变 5
...
总计算量 = O(n)      ← 平方变线性
```

---

## 2. 真实运行:两种实现逐 token 等价

用本课 notebook 的迷你模型(第 20 课结构的单层版,真实运行),贪心生成 8 步:

```text
无缓存(每步全量重算): [1, 2, 3, 13, 13, 13, 0, 0, 0, 13, 0]
缓存版:                [1, 2, 3, 13, 13, 13, 0, 0, 0, 13, 0]
                        完全一致 ✓
```

更强的验证:把缓存版每步产出的 logits 和全量重算版**逐 token 对照**:

```text
logits 最大误差 = 5.0e-16        ← 仅浮点噪声,数学上完全等价
```

**KV Cache 是纯工程优化,不改变任何数学**——这就是它敢到处用的底气。

---

## 3. 实现要点:因果掩码差一(战争故事)

缓存版的新 token,query 要对"全部历史 + 自己"做注意力。掩码写成通用形式(新 token 占 n 行,历史共 T 列):

```python
def causal_mask(n, T):
    return np.triu(np.ones((n, T), bool), T - n + 1)   # 注意 +1!
```

`T - n + 1` 里那个 **+1** 是血泪教训:本课程开发时,这里差一(`T - n`),把"新 token 看到自己"也遮掉了——某行全 `-inf` → softmax 出 NaN → argmax 返回 0 → 表面上"能跑",实际生成全是垃圾。**NaN 不报错、只悄悄污染输出**——这类 bug 的两个教训:

1. 掩码逻辑必须单测:**"每个 query 行的可见列数"=[T-n+1, T-n+2, ..., T]**;
2. 生成循环要断言 `np.isfinite(logits).all()`,让 NaN 当场爆炸而不是静默流出。

---

## 4. 显存账:缓存不是免费的(生产必算)

KV Cache 每层要存所有历史 token 的 K 和 V:

$$
\text{字节数} = 2_{(K,V)} \times n_{layers} \times n_{kv\_heads} \times d_{head} \times seq \times bytes
$$

真实量级(Llama-3-8B,32 层、8 KV 头(GQA)、d_h=128、bf16、8K 上下文、batch=1):

```text
2 × 32 × 8 × 128 × 8192 × 2 ≈ 1.07 GB
```

batch=32 时 34 GB——**比模型权重(16GB)还大**。三个推论(全是生产决策):

1. **GQA 的价值在此**:KV 头 32→8,缓存缩 4 倍(第 7 课伏笔回收)。
2. **长上下文的成本主要是缓存**:128K 上下文的 KV Cache 轻松几十 GB。
3. **PagedAttention(vLLM)**:把缓存分页管理、按需分配,碎皮利用率大增——现代推理引擎标配。

---

## 5. 代码

```python
def forward(self, ids, cache=None):
    # cache: 每层一个 (K, V) 元组的列表;None 表示无缓存(prefill)
    x = self.embedding(ids)
    pos0 = cache[0][0].shape[0] if cache else 0      # 续写的起始位置
    kvs = []
    for l, block in enumerate(self.blocks):
        x, kv = block(x, np.arange(pos0, pos0 + len(ids)),
                      cache[l] if cache else None)     # 块内部:新 K/V 拼接旧缓存
        kvs.append(kv)
    return self.head(self.final_norm(x)), kvs

# 生成循环
logits, kvs = forward(prompt)                # prefill: 一次吃下整个 prompt
for _ in range(n):
    nxt = sample(logits[-1])
    logits, kvs = forward([nxt], kvs)         # decode: 每步只进一个 token
```

两阶段的名字要记:**prefill**(吃 prompt,算量大,可并行)和 **decode**(逐 token,访存密集,吃带宽)。推理引擎的一切优化(PD 分离、连续批处理)都围绕这两个阶段的不同脾气展开。

---

## 6. 生产环境

- **TPOT/T TPS**:decode 阶段每 token 延迟由显存带宽决定(每步要把全部权重和缓存读一遍)——HBM 带宽是推理引擎的生命线。
- **TTFT**:prefill 决定首字延迟。PD 分离(prefill/decode 用不同机器)是 2024 年以来推理架构的大趋势。
- **缓存命中的是前缀**:system prompt 相同的请求可共享缓存(前缀缓存,prompt caching)——API 计费里"缓存命中的输入便宜很多"就是这个原理。

---

## 7. 记住的故事

**现实:**

> "旧 token 的 K/V 永不变,存起来只算新的:O(n²) 变 O(n),数学一个数都不改。"

**数学:**

$$
bytes = 2 \times L \times H_{kv} \times d_h \times seq \times b
$$

**工程:**

> "prefill/decode 两阶段;GQA 省 4 倍;掩码差一 = NaN 静默事故;断言 isfinite。"

**代码:** 第 5 节骨架。

---

## 作业

### A. 概念题

1. 为什么旧 token 的 K、V 不会随生成改变?(用 causal 注意力的可见性答)
2. KV Cache 把计算量从 O(?) 降到 O(?)。显存代价是什么?
3. prefill 和 decode 的计算特点分别是什么?为什么优化思路不同?
4. GQA 为什么对 KV Cache 有 4 倍节省?

### B. 填空题

1. Llama-3-8B、8K、bf16、batch=1:KV Cache ≈ ____ GB(算出来)。
2. 掩码通用式:`np.triu(ones((n, T)), T - n + ____)`;差一会发生什么?
3. decode 阶段延迟主要受 ____ 限制;prefill 决定指标 ____。
4. prompt caching 便宜的原理:相同的 ____ 命中相同的缓存。

### C. 代码手撕(关讲义)

1. 把第六课的自注意力改造成带 cache 参数:`attn(x, pos, cache)` 返回 `(out, new_cache)`。
2. 写生成循环:prompt 走 prefill,之后每步只前向一个 token;与全量重算版逐 token 对照 logits(误差 < 1e-10 才算过)。
3. 给掩码写单测:断言每行可见列数 = `T-n+1+i`。

### D. 生产环境实战

1. 70B 模型(GQA:8 KV 头,d_h=128,80 层)、4K 上下文、batch=16、bf16:KV Cache 多少 GB?A100-80G 单卡放得下(模型权重 bf16 约 140GB 需先切两卡)?
2. 线上偶发"输出全是 token 0"。给出你的排查脚本(基于本课战争故事)。
3. 查证:vLLM 的 PagedAttention 解决什么问题?(两句话:碎片化/共享。)

### E. 拓展

1. 实验:测量本课 Mini-GPT 两种实现的生成耗时(100 token),验证加速比≈多少?
2. 预习钩子:attention 里至今有个悬案——模型怎么知道"谁在第几个位置"?下一课:RoPE,用旋转编码位置,且只看相对距离。

---

# 下一课预告:RoPE

注意力对词序天生不敏感(打乱句子,注意力数学不变——叫置换不变性)。位置信息必须人为注入。RoPE 的做法优雅到犯规:把 q、k 向量按位置**旋转不同角度**——点积只依赖角度差 = 只看相对位置。

> **下一课:[RoPE](19-rope.md)——旋转的位置编码。**
