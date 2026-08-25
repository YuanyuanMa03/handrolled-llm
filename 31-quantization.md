# 第三十一课:量化——把权重压进更小的整数

> 30 课把 KV 管好了,权重还是 bf16「全家桶」:7B 模型 14GB,24GB 的卡装下权重就装不下多少 KV。量化的思路简单到一句话:**权重其实不需要 16 位浮点,4–8 位整数就够**——但「够」是算出来的。本课在**本地 SmolLM2-135M 的真实权重**上逐张量实测:朴素 int8 误差 0.4%,朴素 int4 误差 7%,体积账 0.251 GB → 0.070 GB;生产级 4-bit(GPTQ/AWQ)为什么能几乎不掉点,账也在这一课。
> **本课数字**:全部实读本地 `model.safetensors`(272 个张量,逐张量对账 134,515,008 参数)。

---

## 1. 先不用代码——集装箱的磅秤

fp32/bf16 的权重像用**克级天平**称集装箱:精度高到浪费。量化就是换秤——吨级(int8)或十吨级(int4)的秤,再记一张「换算表」(**scale**):

$$x \approx s \cdot q,\qquad q \in \{-127,\dots,127\}\ (\text{int8})\ \text{或}\ \{-7,\dots,7\}\ (\text{int4})$$

存的是整数 $q$ 和少量的 $s$,用的时候乘回来。误差 = 换算表的粒度——同一张表管得越宽(整行/整组),误差越大:**scale 的粒度是量化误差的第一旋钮**。

---

## 2. 数学:三种量化姿势

**对称逐通道(RTN int8 的标准姿势)**:每个输出通道(权重矩阵的行)一个 scale:

$$s_i = \frac{\max_j |x_{ij}|}{2^{b-1}-1},\qquad q_{ij} = \mathrm{clip}\bigl(\mathrm{round}(x_{ij}/s_i)\bigr)$$

**分组量化(int4 的救命姿势)**:每 64 个权重一个 scale——组内动态范围小,scale 更贴。

**为什么 int4 朴素做不好**:4-bit 只有 16 个台阶,一行几百个权重挤 16 层楼,离群值(outlier)把 scale 撑大,其他值全挤成一两个台阶——这就是 7% 误差的来源。GPTQ 的解法:**量化一列,把产生的误差按 Hessian 信息摊给还没量化的列**(误差补偿);AWQ 的解法:**按激活重要性保护少数关键通道**(激活感知)。两者都把 int4 误差压到朴素法的一小部分。

## 3. 手撕(加载真实权重 + 量化器)

safetensors 加载是 21 课读头 + 本课读字节(bf16 是 fp32 的高 16 位,位运算还原):

```python
def load_tensor(st_path, name, header=None):
    """逐张量加载:safetensors 原始字节 → numpy fp32。"""
    header = header or read_header(st_path)
    meta = header[name]
    beg, end = meta["data_offsets"]
    with open(st_path, "rb") as f:
        f.seek(8 + hdrlen(st_path) + beg)
        raw = f.read(end - beg)
    arr = np.frombuffer(raw, dtype={"F32": np.float32, "BF16": np.uint16,
                                    "F16": np.float16}[meta["dtype"]])
    if meta["dtype"] == "BF16":                  # bf16 = fp32 的高 16 位
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.reshape(meta["shape"])

def quantize_sym(x, bits):                       # 对称逐通道
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(x).max(axis=1, keepdims=True) / qmax + 1e-12
    q = np.clip(np.round(x / scale), -qmax - 1, qmax)
    return q, scale

def quantize_group(x, bits, group=64):           # 分组(int4 用)
    qmax = 2 ** (bits - 1) - 1
    out, in_ = x.shape
    pad = (-in_) % group
    xp = np.concatenate([x, np.zeros((out, pad))], 1).reshape(out, -1, group)
    scale = np.abs(xp).max(axis=2, keepdims=True) / qmax + 1e-12
    q = np.clip(np.round(xp / scale), -qmax - 1, qmax)
    return (q.reshape(out, -1)[:, :in_],
            np.repeat(scale, group, axis=2).reshape(out, -1)[:, :in_])
```

## 4. 真实运行:真实张量上的误差表

本地 SmolLM2-135M-Instruct(272 个张量,单文件 `model.safetensors`),挑四类代表张量(相对误差 = 最大绝对误差 / 该张量最大幅值):

```text
tensor                                       shape          int8        int4(g64)
model.layers.0.self_attn.q_proj.weight      (576, 576)    3.94e-03    7.14e-02
model.layers.0.mlp.down_proj.weight         (576, 1536)   3.94e-03    7.12e-02
model.layers.5.self_attn.o_proj.weight      (576, 576)    3.93e-03    7.10e-02
model.embed_tokens.weight                   (49152, 576)  3.93e-03    7.01e-02
```

三个真实观察:

- **int8 朴素量化就够好**:0.4% 量级的相对误差,对语言模型输出几乎无感——所以 W8 是「免费午餐」;
- **int4 朴素(RTN)误差 7%**——直接上生产会掉点;这就是 GPTQ/AWQ 存在的理由(§2);
- **顺带一个解剖学发现**:文件里**没有 lm_head**——SmolLM2-135M 是词嵌入绑定(tied embeddings)的,输出层就是 embed_tokens 的转置;我们 20 课的 Mini-GPT 是不绑定的(21 课对过账)。

## 5. 体积账(逐张量对账后)

```text
全模型参数(逐张量求和):134,515,008     ← "135M" 名实相符
bf16 = 0.251 GB
int8 = 0.125 GB(scales 忽略)
int4 + g64 scales ≈ 0.063 + 0.008 = 0.070 GB    ← 降到 bf16 的 28%
```

外推到 7B:bf16 14GB → int4 ≈ 3.9GB——**24GB 的卡从「装不下 KV」变成「绰绰有余」**;再外推 Qwen3-4B(21 课:4.02B):bf16 7.5GB → int4 ≈ 2.1GB,Mac 16GB 内存也轻松跑(31 课作业 D1 让你自己算)。

## 6. 生产账本

- **重量化(W8A16 / W4A16)只省显存与带宽,计算仍在 fp16/bf16**——访存密集的 decode(29 课的两副脾气)受益最大,这解释了量化对 TPOT 的提速比对 TTFT 明显;
- **全量化(W8A8/FP8)连矩阵乘都在低位**——H100 的 FP8 TensorCore 是硬件级支持,训练推理都用;
- **GPTQ/AWQ 的一句话分工**:GPTQ 按 Hessian 逐列误差补偿;AWQ 按激活重要性缩放保护离群通道。现代引擎(vLLM/MLX)里 4-bit 权重已是默认选项之一;
- **量化粒度菜单**:逐张量 < 逐通道 < 逐组(64)——粒度越细误差越小、scales 越多,工程取折中。

---

## 7. 记住的故事

**现实:** 「权重不需要 16 位:int8 免费,int4 要付误差补偿的工程费。」

**数学:** $x \approx s\cdot q$;相对误差 ∝ 台阶宽度 / 动态范围;分组把动态范围切小。

**工程:** 「朴素 int4 的 7% 误差就是 GPTQ/AWQ 的存在理由——生产 4-bit 不掉点,不是玄学,是补偿算法。」

**代码:** 加载器 + 两个量化器约 40 行;全部对着真实权重跑。

---

## 作业

### A. 概念题

1. 对称与不对称量化差在哪?什么分布适合不对称(提示:偏一边的激活)?
2. 离群值(outlier)如何毁掉朴素 int4?GPTQ 和 AWQ 各怎么应对?
3. 重量化(W4A16)为什么对 decode 提速比对 prefill 明显?
4. 分组数从 64 改 16 或 256,误差与存储各怎么变?

### B. 填空题

1. 实测:int8 误差 ____ 量级,int4 朴素 ____;体积 bf16 ____ → int4+g64 ____ GB。
2. SmolLM2-135M 逐张量对账 ____ 参数;无 lm_head 是因为 ____。
3. bf16 → fp32 的转换 = ____。
4. 一块(16 token)KV 的账在 30 课;本课权重账:int4 后 7B ≈ ____ GB。

### C. 代码手撕(关讲义)

1. 默写 quantize_sym 与 quantize_group;对一个 (64,128) 随机矩阵手算第一组的 scale 对照。
2. 手撕 load_tensor 的 bf16 → fp32 位运算,并与「除以 2^16 的整数移位」解释等价。
3. 写一个量化前后的逐元素误差直方图,观察 int4 的台阶效应(值挤在少数台阶)。

### D. 生产环境实战

1. 容量题:Qwen3-4B(4.02B)int4+g64 总体积?scales 占多少?加上 30 课的 KV 账,16GB 内存的 Mac 能伺候几条 2K 上下文?
2. 事故题:int4 量化后模型在长算术题上掉点明显,短问答正常。解释(误差累积?哪类权重更敏感?)并给出定位实验。
3. 查证:FP8(E4M3/E5M2)是什么?H100 的 FP8 支持为什么训练推理都能用?

### E. 拓展

1. 实验:group ∈ {32,64,128,256} 扫描 int4 误差曲线,找「误差-scales 存储」的甜点(ipynb TODO)。
2. 预习钩子:权重压到 4-bit 了,**解码还是一个 token 一个 token 串行**——如果一个小模型先猜 4 个、大模型一次验证呢?下一课:投机解码,大模型亲笔签名的小模型代笔。

---

# 下一课预告:投机解码

draft 模型连写 k 个 token,target 模型一次前向全部验证,按拒绝采样规则接受/回退——数学上保证**分布与 target 独自解码完全相同**,加速全靠「串行变批处理」。本课在《西游记》上真训一大一小两个 MiniGPT,实测接受率与加速比。

> **下一课:[投机解码——小模型代笔,大模型签名](32-speculative.md)——分布不变,时间减半。**

## 参考来源

**本课实测**:272 张量全部实读自本地 `model.safetensors`(SmolLM2-135M-Instruct);误差与体积为真实运行结果。

**文献**:

- [Frantar et al. 2022, GPTQ (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 逐列误差补偿的 4-bit 量化
- [Lin et al. 2023, AWQ (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 激活感知的量化缩放
- [Dettmers et al. 2022, LLM.int8() (arXiv:2208.07339)](https://arxiv.org/abs/2208.07339) — 离群值问题与混合精度分解
