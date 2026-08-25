# 第二十四课:训练循环——真训《西游记》

> 第 20 课留了承诺:**训练就是把 loss「压到基线以下、再一路压低」的过程**。本课兑现它——在《西游记》全本上真训 2000 步:val loss 从 8.76 压到 5.33(瞎猜基线 8.42),模型从「生僻字汤」学会写半通顺的明朝白话文。
> **本课所有数字均来自本机真实运行**(约 2 分钟一次完整训练,固定种子,可复现)。

---

## 1. 先不用代码——学生刷题

把训练想成一个学生刷题:

- **题**:随机翻开语料一段,遮住下一个字,猜它;
- **对答案**:猜错的程度就是 loss(第 11 课的交叉熵);
- **长记性**:误差变成梯度(第 23 课),AdamW 调整直觉(第 13 课);
- **epoch**:整个题库刷完一遍,再来。

前 22–23 课我们备齐了题库(分词/字符表)、答案比对(CE)和长记性的机制(反向+优化器)。本课要解决的其实是**三个新的工程问题**:

1. **数据怎么喂**——切 batch 的规矩,训练/验证怎么分;
2. **步子怎么迈**——学习率先热身再退火(warmup + cosine);
3. **怎么知道没白学、没学坏**——train/val 两条曲线怎么读。

---

## 2. 数据管道:字符表与切片

第 22 课我们自训了 BPE;本课的模型太小(几万到几十万参数),**词表本身就会比模型大**——所以用最朴素的**字级词表**:语料里出现过的每个字符一个 id。《西游记》正好是个天然封闭王国:

```python
text = open("data/xiyouji.txt", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype=np.int32)
n_train = int(len(data) * 0.99)
train_data, val_data = data[:n_train], data[n_train:]
```

真实运行:`723,587 字 | 词表 V=4551 | 训练 716,351 / 验证 7,236 | 瞎猜基线 ln(4551) = 8.4231`

两个规矩:

- **训练/验证按位置切(99/1),不能随机切**。语料是连续小说,随机抽样会让同一回、甚至同一句的上下文同时出现在两边——验证就变成「默写」而不是「考试」。
- **x 与 y 错一位**:`x = ids[o:o+T]`,`y = ids[o+1:o+T+1]`——每个位置都在预测下一个字(第 11 课的 batch 框架)。

```python
def get_batch(split, B, T, rng):
    src = train_data if split == 'train' else val_data
    off = rng.integers(0, len(src) - T - 1, B)
    x = np.stack([src[o:o + T] for o in off])
    y = np.stack([src[o + 1:o + T + 1] for o in off])
    return x, y
```

---

## 3. 优化器与学习率:热身、退火、安全带

第 13 课的 AdamW 吃单个数组;这里把它扩成**参数组版**(`MultiAdamW`),一次更新全部 16 个权重张量,公式逐字不变:

```python
class MultiAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.ps, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, *betas, eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, gs):
        self.t += 1
        for p, g, m, v in zip(self.ps, gs, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g          # 一阶矩
            v *= self.b2; v += (1 - self.b2) * g * g      # 二阶矩
            m_hat = m / (1 - self.b1 ** self.t)           # 偏差修正
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p)
```

学习率先 **warmup**(前 100 步线性升到峰值)再 **cosine 退火**(降到峰值的 10%):

```python
def lr_at(step, total, base_lr=1e-3, warmup=100, floor_ratio=0.1):
    if step < warmup:                                    # 热身:冷启动时 m/v 还是 0,
        return base_lr * (step + 1) / warmup             # 大步长会乱踢
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
```

梯度先过第 23 课的**全局范数裁剪**(上限 1.0)再进优化器——安全带。

---

## 4. 训练循环本体

第 23 课的 `loss_and_grads` 是单序列的;batch 就是外面套一圈循环、梯度取平均:

```python
def batch_grads(gpt, x, y):
    B, T = x.shape
    gsum, tot = None, 0.0
    for b in range(B):
        loss, grads = loss_and_grads(gpt, list(x[b]), list(y[b]))
        tot += loss
        gs = flat_grads(grads)
        if gsum is None:
            gsum = [g / B for g in gs]
        else:
            for acc, g in zip(gsum, gs):
                acc += g / B
    return tot / B, gsum

gpt = MiniGPT(vocab=V, d=64, n_heads=4, n_layers=2, seed=0)
params = flat_params(gpt)
opt = MultiAdamW(params, lr=1e-3, wd=0.01)
rng = np.random.default_rng(42)
STEPS, B, T = 2000, 16, 64

for step in range(STEPS):
    opt.lr = lr_at(step, STEPS)
    x, y = get_batch('train', B, T, rng)
    loss, gs = batch_grads(gpt, x, y)
    gn = clip_norms(gs, 1.0)          # 第 23 课的安全带
    opt.step(gs)
```

配置:`d=64、4 头、2 层、713,600 参数;batch 16×T64;2000 步`。

---

## 5. 真实运行:2000 步全程

```text
未训练 loss = 8.7578(基线 8.4231 上方——第 20 课解释过:随机权重的 logits 波动只会抬高交叉熵)

 step    train      val   gnorm        lr
    1   8.7233   8.7574   0.645  1.00e-05
  200   5.7768   6.2412   0.418  9.94e-04
  400   5.5205   5.8918   0.518  9.46e-04
  600   5.0076   5.6586   0.602  8.55e-04
  800   4.9273   5.5345   0.612  7.31e-04
 1000   4.8032   5.4498   0.619  5.88e-04
 1200   4.9513   5.4105   0.707  4.40e-04
 1400   4.8850   5.3789   0.809  3.04e-04
 1600   4.6679   5.3510   0.683  1.95e-04
 1800   4.5357   5.3326   0.719  1.25e-04
 2000   4.4996   5.3266   0.742  1.00e-04

训练用时 2.1 min;tokens 见过 2,048,000 ≈ 2.9 个 epoch
```

读曲线(面试会问「你的 loss 曲线说明什么」):

- **前 200 步最陡**(8.7→5.8):高频字频、常用搭配(「的」「行者」「…道:」)先被学会——最便宜的规律最先被捡走;
- **train/val 差距 ~0.8 且稳定**:泛化 gap,正常;**val 仍在缓慢下降**——没过拟合,但也接近这份数据上这台模型的平台期;
- **梯度范数全程 0.4–0.8,从未触发裁剪上限 1.0**——安全带一次都没系上,不代表不需要系(换了更脏的数据/更大模型,它就是救命的那一行);
- **承诺兑现**:val 5.3266 < 基线 8.4231,压到基线的 **63%**(7.68 bits/char:每个汉字的信息量从完全随机的 12.15 bits 压到 7.68 bits,也远低于 UTF-8 的 24 bits/字——**训好的模型就是一个压缩器**)。

---

## 6. 毕业照:从乱码到明朝白话文

同一提示、同一采样种子,训练中途四张快照(真实运行):

```text
[step    0] 那猴王源骑同驷悭柏趄斫獒丕蓦犁抄裆报横同杓塘巷莹延泡齁鲁荙父往啐鹚牟割窟衬穴陋亵独母俪篷送矧帽蹊煠灼萨
[step  500] 那猴王道：「这个小真中，上正是上，出我也不得了火衣，都见行者光仙，就乃是水面，三藏怎么？」行者道：「有
[step 1000] 那猴王子，丢至天宫，喝道人还打。因你把他吃了。」众僧道：「徒弟啊，如今日粗细，不曾或上灵霄！」那小妖道
[step 2000] 那猴王子，与那国王见他散。真个徒弟子，个大圣僧，却娘转那里有个侍，将了山门，过不曾打。回来至城门外，有
```

三个真实观察:

- **step 0 的乱码偏爱生僻字**——随机权重下 4551 个字的分布近乎均匀,采样器在巨大空间里漫游,而常用字只占词表一小角;
- **step 500 词汇先于语法成型**——「行者」「三藏」「…道:」已就位(第 22 课 BPE 的频次视角:高频搭配最便宜),句子还是碎的;
- **step 2000 有了句式的骨架**——对话轮替、四六节奏,但语法树仍会断链:71 万参数买得到「明朝白话文的味道」,买不到《西游记》。

换个提示再来一段(T=0.8, top_p=0.95):

```text
行者道：「那些真之兵，却将三藏心儿，却又不敢出去，到此？快有何四个？」二魔道：「不期去。」
那怪见那山岁著一日，爬下行者。那行者慌
```

「二魔」都会接话了。这就是预训练的全部秘密:**没有魔法,只有 2000 步 gradient descent**。

---

## 7. 生产账本:从 2 分钟到 2 个月

我们:71 万参数、2 分钟、2.9 个 epoch。生产(Qwen3-4B 量级):40 亿参数、万亿级 token、几千卡、几周——三个本课直接放大就撞上的账:

- **数据/参数配比(Chinchilla 账)**。compute-optimal 经验律:每个参数喂约 20 个 token。我们 2,048,000 token / 713,600 参数 ≈ **2.9:1——远低于 20,数据喂少了**;但在玩具世界里语料只有 72 万字,继续喂就要反复刷题,迟早过拟合。真实预训练恰好相反:**数据比参数便宜**(互联网还有几十万亿 token),所以工业界一路把参数做大。
- **吞吐账**。本机纯 numpy ≈ 2M token/2min ≈ 1.6万 token/s;一张 A100 上 bf16 的同尺寸模型约为数百倍,7B 模型一天单卡 ~1 亿 token 量级——**千卡集群是把它乘一千,不是算法变了**。混合精度、ZeRO 分片(第 13 课账本)、数据并行 = 同一个循环的多份拷贝。
- **数据工程**。我们的语料是干净的公版书;生产语料是网络泥浆:去重、质量过滤、配比(代码/多语言/对话各占多少)、课程(先易后难)——「预训练工程师一半时间在数据上」不是玩笑。

---

## 8. 记住的故事

**现实:** 「预训练没有魔法:随机初始化 + 下一字预测 + 两千步梯度下降 = 一个会写明朝白话文的模型。」

**数学:** $\text{lr}(t)$:先线性热身、再余弦退火;梯度先裁范数再进 AdamW。

**工程:** 「val 曲线还在降就没过拟合;train/val 差距是泛化 gap 的体温计。」

**代码:** 第 3–4 节约 60 行,把 22(数据)、23(梯度)、13(优化器)三课焊成一台训练机。

---

## 作业

### A. 概念题

1. 训练/验证集为什么必须按位置连续切,不能随机切?随机切在什么任务里反而是对的?
2. warmup 解决什么问题?如果不热身直接用峰值 lr,训练早期会发生什么?
3. train loss 4.50 与 val loss 5.33 的差距说明什么?这个差距变大意味着什么?
4. 为什么未训练模型(step 0)的采样偏爱生僻字?

### B. 填空题

1. 本课词表 V=____,瞎猜基线 ln(V)=____;未训练 loss=____(在基线上方,是第 20 课解释的____)。
2. 2000 步后 val loss=____,压到基线的 ____;合 ____ bits/char(随机需 ____ bits)。
3. 训练配置:____ 参数、batch ____×T____、用时 ____ min、约 ____ 个 epoch。
4. 梯度范数全程在 ____~____,裁剪上限 ____(触发/未触发)。

### C. 代码手撕(关讲义)

1. 默写 `get_batch`(含 x/y 错一位与随机偏移),验证形状 [B,T]。
2. 默写 `lr_at`:画(或打印)0–2000 步的 lr 曲线,确认热身 100 步到峰值、终点为峰值的 10%。
3. 默写 `batch_grads` 的梯度平均;解释为什么除以 B 而不是 B×T(提示:第 23 课已对 T 取过平均)。

### D. 生产环境实战

1. 容量/吞吐题:7B 模型、1T token、单卡 A100 一天处理约 1 亿 token(数量级估算),需要多少卡天?若用 512 张卡,多少天?写出算式。
2. 事故题:训练到 5000 步,val loss 开始回升而 train loss 继续下降。给出处置顺序(学习率?数据?早停?正则?)。
3. 查证:Chinchilla(2022)的核心结论一句话;它之后工业界(Llama 3 等)实际怎么做配比,和 Chinchilla 有什么不同?

### E. 拓展

1. 实验:同配置训 6000 步,记录 val 曲线拐点——找到这台模型在这份数据上的过拟合起点,与 2000 步对照(ipynb TODO)。
2. 预习钩子:模型学会了「续写」,但你若问它「一加一等于几?」它只会继续写小说——**它不会「回答」**。下一课 SFT:教模型听指令,chat 模板和 loss mask 就是那间教室。

---

# 下一课预告:SFT

底座只会续写。指令微调(Supervised Fine-Tuning)用「问-答」格式的数据重塑它的行为:chat 模板(第 21 课在 Qwen 里见过的 `<|im_start|>` 全家福)划定角色,loss mask 只对回答计算梯度——底座的续写能力冻结在权重里,回应的礼貌生长在 mask 之上。

> **下一课:[SFT——教模型听指令](25-sft.md)——底座会写字,教室教它说话。**

## 参考来源

**本课实测**:全部数字来自本机真实运行(`data/xiyouji.txt` + 第 20/23 课代码,固定 seed=42,全文可复现)。

**文献**:

- [Hoffmann et al. 2022, Training Compute-Optimal Large Language Models (Chinchilla, arXiv:2203.15556)](https://arxiv.org/abs/2203.15556) — 20 tokens/参数 配比律
- [Karpathy, nanoGPT (GitHub)](https://github.com/karpathy/nanoGPT) — 同款玩具训练循环的业界参照(本课结构与其 train.py 同构,数据与模型不同)
- [Radford et al. 2019, GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — 「互联网是一台无监督多任务学习机」:预训练范式的出处
