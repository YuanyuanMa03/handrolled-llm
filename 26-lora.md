# 第二十六课:LoRA——低秩补丁

> 第 25 课的 SFT 为了学会「回答」这件事,动了全部 713,984 个参数。本课只训 **1.70%(12,160 个)**:LoRA 把权重补丁约束成低秩 $\Delta W = \frac{\alpha}{r}BA$——冻结整本书,只夹一张便签。真实运行:**任务 50/50、merge 后前向逐位不变**;同时本课还有一个反直觉发现:**LoRA 一样会「遗忘」,而且一半的遗忘是概率被特殊 token 偷走的**——账都算给你看。
> **本课所有数字均来自本机真实运行**;开发过程中的两个真 bug(补丁从未生效 / 梯度喂错矩阵)在 §8 战争故事。

---

## 1. 先不用代码——书的边注

全参数微调像什么?为了改书的第 37 页一句话,把整本书重新排版印刷一遍。LoRA(Low-Rank Adaptation)的洞察:**行为的改动是低维的**——你不需要重写整本书,只需要在页边夹便签:

$$W' = W + \Delta W, \qquad \Delta W = \frac{\alpha}{r}\,B A,\quad B \in \mathbb{R}^{d_{\text{out}} \times r},\ A \in \mathbb{R}^{r \times d_{\text{in}}},\ r \ll d$$

秩 $r$ 限制补丁的「自由度」:一个 $d \times d$ 的权重有 $d^2$ 个自由度,秩-4 补丁只有 $2dr$ 个。为什么敢这么省?因为微调改变的是**行为**(怎么调用已有知识),不是知识本身——而行为住在一个低得多的子空间里。

初始化有个关键机关:**$A$ 随机小值、$B$ 全零** $\Rightarrow$ 初始 $\Delta W = 0$,模型与底座**逐位相同**,训练从底座出发,不会第一步就把底座踢坏。

---

## 2. 数学:不需要新的反向传播

LoRA 最好的一件事:**23 课的整机反向原封不动就能用**。前向把 $W' = W + \frac{\alpha}{r}BA$ 算好放进 layers,反向拿到的就是 $\partial L / \partial W'$(记作 $gW$)。剩下只是链式法则的两行投影:

$$\frac{\partial L}{\partial B} = \frac{\alpha}{r}\, gW\, A^\top
\qquad
\frac{\partial L}{\partial A} = \frac{\alpha}{r}\, B^\top gW$$

(推导:$W'_{ij} = W_{ij} + \frac{\alpha}{r}\sum_k B_{ik}A_{kj}$,对 $B_{ik}$、$A_{kj}$ 各求一次偏导。)

## 3. 代码:四件套

```python
LORA_ALL = ('Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W3', 'W2')      # 挂全部线性层(§7 有为什么)

def init_lora(gpt, keys=LORA_ALL, r=4, alpha=8, seed=1):
    """A ~ N(0,0.02),B = 0 → 初始 ΔW = 0。同时快照 W_base。"""
    rng = np.random.default_rng(seed)
    adpts, W_base = {}, {}
    for i in range(gpt.NL):
        for k in keys:
            W = gpt.layers[i][k]
            out_, in_ = W.shape
            adpts[(i, k)] = (rng.normal(size=(r, in_)) * 0.02, np.zeros((out_, r)))
            W_base[(i, k)] = W.copy()
    return adpts, W_base, alpha / r

def apply_lora(gpt, adpts, scale, W_base):
    """每步从 W_base 快照干净重建合并权重——绝不在旧账上做加减(§8 战争故事①)。"""
    for key in W_base:
        i, k = key
        A, B = adpts[key]
        gpt.layers[i][k] = W_base[key] + scale * (B @ A)

def lora_grads(grads, adpts, scale):
    """gW(合并权重梯度)投影:gA = scale·BᵀgW,gB = scale·gW·Aᵀ。
    迭代序与 flat_adapters 一致(sorted)——zip 配对不许错位(§8 战争故事②)。"""
    out = []
    for key in sorted(adpts):
        i, k = key
        A, B = adpts[key]
        gW = grads['layers'][i][k]
        out.append(scale * (B.T @ gW))
        out.append(scale * (gW @ A.T))
    return out

def flat_adapters(adpts):
    out = []
    for key in sorted(adpts):
        out.extend(adpts[key])
    return out
```

训练循环与 25 课同构,只有两处变化:每步先 `apply_lora` 干净重建;可训练列表 = adapters + 3 个特殊 token 的 E 行与 head 列(新词表项必须可训,否则模板 token 永远是随机向量)。lr 用 **1e-3**(LoRA 参数少,惯用比全参 SFT 高一档)。

---

## 4. 出厂验收(真实运行)

```text
LoRA 梯度投影抽查 48 坐标 max|err| = 3.45e-10   ✓(数值裁判,B 取非零值后的非平凡解)
B=0 时挂上 LoRA 前后 logits max|diff| = 0.0      ✓(起点即底座,逐位相同)
参数账:全线性 r=4 → adapter 11,776 + 特殊 token 384 = 12,160,占 713,984 的 1.70%
```

## 5. 真实运行:主实验与遗忘分解

**LoRA-SFT(全线性 r=4,α=8,lr=1e-3,1000 步,15 秒):**

```text
任务准确率 50/50 | 全词表 val 5.3273 → 8.2272
```

50/50 达成——但 val 8.23?!LoRA 不是冻结底座吗,怎么也「遗忘」了?**做个分解实验**:把 3 个特殊 token 的 logits 剔除、在原 4551 词表上重归一化,再算 val loss——

```text
                全词表 val    去特殊 token val
基座              5.3273        5.3266        ← 特殊 token 本来无贡献
LoRA-SFT 后       8.2272        6.3551
                   ↑ 1.87 nats 是「概率被偷」:特殊 token 的 head 列被训高,
                     在普通文本上到处抢 softmax 质量(第 1 课的归一化代价)
                   ↑ 剩下 1.03 nats 是真实漂移:ΔW 无条件全局生效,不只作用于模板里
```

结论分两层:**ΔW 是无条件生效的**——它不知道自己只该在 `<|assistant|>` 后面发力,所以 LoRA 参数少 ≠ 不遗忘;但**底座知识的大头还在**(剔除概率偷窃后只漂了 1.03)。药方和 25 课同一张:**回放**。

**LoRA + 回放(半指令半西游记,其余同上):**

```text
任务准确率 49/50 | 全词表 val 5.3273 → 5.4090 | 去特殊 token 5.4052
```

49/50 与 5.41 兼得——和 25 课全参数 + 回放(50/50 与 5.37)同水位,但只动了 1.70% 的参数,而且:

## 6. merge:便签抄进书里,部署零开销

训练完,把 $\Delta W$ 直接加进 $W$,扔掉 adapter:

```text
merge 验收:合并权重 vs 带 adapter 前向 logits max|diff| = 0.0   ✓
```

部署意义:线上跑的是**普通模型**,没有额外矩阵乘、没有额外延迟——12,160 个参数的文件就能带走一个微调(2.9 KB 对 5.7 MB);反过来也可以**不 merge**,一份底座 + 多张 adapter 按请求热切换(多租户定制)。

贪心回答实录(merged 模型):

```text
反义词:大 → '小' ✓      重复:筋斗 → '筋斗' ✓
```

(但纯指令版续写「那猴王」已经输出 `######`——遗忘的直观样子,§5 的账。)

## 7. 挂对地方,选对秩(真实运行,同一纯底座 600 步)

| 配置 | 可训练 | 任务准确率 |
|---|---|---|
| 仅 attention(Wq/Wk/Wv/Wo)r=4 | 4,480 | **28/50** |
| 全线性 r=1 | 3,328 | 39/50 |
| 全线性 r=4 | 12,160 | **49/50** |
| 全线性 r=16 | 47,488 | 49/50 |

两个真实教训:

- **FFN 不能不挂**。attention 只能重排已有特征;「见『反义词:大』输出『小』」这种**新联想**,主要住在 FFN(Transformer 的键值记忆)里——这正是现代实践从 LoRA 原论文的「只挂 Wq/Wv」演进到「挂全部线性层」的原因;
- **秩的边际收益会封顶**。本任务 r=4 就到 49/50,r=16 不再涨(钱花在刀刃上:r 每翻倍,adapter 参数翻倍)。

## 8. 战争故事:补丁从未生效

本课第一版有两个真 bug,症状都极有教益:

**① merge 舞步在旧账上跳(补丁从未生效)。**第一版用 `strip(减 Δ) → apply(加 Δ)` 更新权重。第一步没问题;但从第二步起,layers 里已是**上一步的旧 Δ**,strip 却用**新 Δ** 去减——一减一加,净效果永远是 W_base,**adapter 的更新从未进过前向**。症状完全对得上:loss 卡在 3.7 降不下去(那点进步全是特殊 token 嵌入在学——它们不经过 merge 机制),A/B 的梯度算得一丝不苟,全在给一个不存在的通道发力。修法一句话:**每步从 W_base 快照干净重建**,不做增量加减。

**② zip 配对错位(梯度喂错矩阵)。**`lora_grads` 按 `LORA_KEYS` 的**定义序**(Wq,Wk,…)产出梯度,`flat_adapters` 按字典序(Wk,Wq,…)排参数——`zip` 一对,A 的梯度喂给了 B,Wk 的喂给了 Wq。数值裁判立刻按住:3e-2。统一成同一个 `sorted` 序后:3.45e-10。

两条合起来的教训:**梯度正确 ≠ 训练正确**——裁判验收的是「公式对」,训练还需要「数据流对」;而「loss 卡住但梯度全对」这种症状,先查管道,再查公式。

## 9. 生产账本

- **QLoRA**:底座量化到 4-bit(NF4)常驻,LoRA 在其上训练——7B 模型微调进单张 24GB 卡的标配(账:7B×0.5 字节 + LoRA + 优化器状态)。
- **多 adapter 服务**:不 merge,共享底座、按请求切换 adapter;merge 换取零延迟,不 merge 换取灵活性——部署形态的选择题。
- **什么时候仍要全参数**:注入新知识(新语言、新领域语料)时低秩补丁力不从心——LoRA 调**行为**最划算,灌**知识**还得开门。
- **生态**:HuggingFace PEFT 一行 `get_peft_model` 的事;本课手撕是为了让你知道那一行里面是什么。

## 10. 记住的故事

**现实:** 「全参数微调重印整本书;LoRA 夹一张低秩便签——1.70% 的参数,一样的任务成绩,还能 merge 回去。」

**数学:** $W' = W + \frac{\alpha}{r}BA$;梯度投影两行 $g_B = \frac{\alpha}{r}gW A^\top,\ g_A = \frac{\alpha}{r}B^\top gW$——23 课的反向原样复用。

**工程:** 「梯度正确 ≠ 训练正确:loss 卡住而梯度全对,先查管道(merge 在旧账上跳?zip 错位?),再查公式。」

**代码:** 第 3 节四件套约 40 行 + 25 课训练循环的两处修改。

---

## 作业

### A. 概念题

1. 为什么 B 初始化为零(而不是 A)?如果两个都随机初始化,训练开始会发生什么?
2. 「LoRA 参数少所以不会遗忘」——本课的实验怎么推翻这句话?遗忘的两种成分分别是什么?
3. 为什么「见问题给答案」这类新联想主要依赖给 FFN 挂 adapter,而不是 attention?
4. α 的作用是什么?保持 r 不变、把 α 翻倍,大致等价于改了什么?

### B. 填空题

1. 本课可训练参数 ____ 个,占 ____;其中 adapter ____ + 特殊 token ____。
2. 三项验收:梯度投影 ____、B=0 不变性 ____、merge ____。
3. 遗忘分解:全词表 5.3273→8.2272,其中 ____ nats 是概率被偷,____ 是真实漂移;LoRA+回放后 ____/50 且 val ____。
4. 600 步对照:仅 attention ____/50,全线性 r=1/4/16 分别 ____/____/____。

### C. 代码手撕(关讲义)

1. 默写 `init_lora`(含 B=0 与 W_base 快照)和 `apply_lora`(干净重建)。
2. 推导并默写 `lora_grads` 的两行投影;用数值裁判验收(阈值 1e-8)。
3. 手撕 merge:`W_merged = W_base + (α/r)·B@A`,验证与带 adapter 的前向逐位一致。

### D. 生产环境实战

1. 容量题:7B 模型(d=4096)、QLoRA r=16 挂全部 7×L 个线性层(L=32):adapter 共多少参数?4-bit 底座 + fp16 adapter + AdamW,微调共需多少显存(数量级)?
2. 事故题:同事的 LoRA 微调 loss 一直卡住不降,但梯度范数正常、gradcheck 也过。写出你的排查清单(提示:本课 §8)。
3. 查证:PEFT 里 `LoraConfig(r=…, lora_alpha=…, target_modules=…)` 各参数与本课哪些量对应?`target_modules` 默认挂哪些?

### E. 拓展

1. 实验:α∈{2,8,32} 扫描(固定 r=4),记录任务收敛速度与遗忘量——α 与有效学习率的关系(ipynb TODO)。
2. 预习钩子:SFT/LoRA 教会了「听指令」,但模型偶尔还是答错——**怎么让它偏爱更好的那个答案**?下一课 DPO:不需要奖励模型、不需要 RL,直接从偏好对(chosen/rejected)推出优化目标。

---

# 下一课预告:DPO

SFT 教行为,DPO(Direct Preference Optimization)教品味:同一个问题的两个回答,一个好一个坏,直接对它们的对数概率差做分类损失——推导源自 RLHF 目标的闭式解,绕开了奖励模型和强化学习。本课在 25 课的底座上,用合成偏好对真实训练一次,看 chosen 与 rejected 的概率差被拉开。

> **下一课:[DPO——偏好直接优化](27-dpo.md)——不训练奖励模型,直接教品味。**

## 参考来源

**本课实测**:基座为第 24 课配置的确定性复刻;对照与验收数字全部来自本机真实运行。

**文献**:

- [Hu et al. 2021, LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685)](https://arxiv.org/abs/2106.09685) — 低秩适配原始论文
- [Dettmers et al. 2023, QLoRA (arXiv:2305.14314)](https://arxiv.org/abs/2305.14314) — 4-bit 底座 + LoRA
- [Biderman et al. 2024, LoRA Learns Less and Forgets Less (arXiv:2405.09673)](https://arxiv.org/abs/2405.09673) — 实证:LoRA 学新知识弱于全参微调,但遗忘也更少
