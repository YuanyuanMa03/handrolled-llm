# 第二十七课:DPO——偏好直接优化

> SFT(25 课)教会「听指令」,LoRA(26 课)让微调变便宜——但模型答错了怎么办?RLHF 的经典答案是「训一个奖励模型 + 强化学习」,三台模型一起转。DPO(Direct Preference Optimization)的洞察:**RLHF 目标有闭式解,代回去,整个问题变成一个分类损失**——不要奖励模型,不要 RL。本课手撕全程,并在玩具上得到一个比「DPO 很神」更有用的结论:**DPO 是偏好分离器,不是知识注入器**(全部数字真实运行)。
> 开发战争故事:符号翻转 bug——逐行范数完全相等、坐标反号,数值裁判一眼看穿(§4)。

---

## 1. 先不用代码——两份答卷的品酒师

想象评卷老师同时拿到两份答卷:一份好(chosen),一份差(rejected)。他不需要给绝对分,只需要** reliably 说出谁好**——这正是人类标注最擅长、成本最低的形式:「A 比 B 好」。

- **SFT** 只见过好答案(「照着写」);
- **DPO** 同时见过好与坏(「往好的方向挪,往坏的反方向挪」)——信息量翻倍,且不需要告诉模型「好多少」。

生产里这就是 RLHF 数据的样子:同一个 prompt,采两个回答,人标哪个好。

---

## 2. 数学:三步推出 DPO

**第一步:偏好模型(Bradley-Terry)。**人类比较概率化:

$$P(y_w \succ y_l \mid x) = \sigma\bigl(r(x, y_w) - r(x, y_l)\bigr)$$

$r$ 是(未知的)奖励函数,$\sigma$ 是 sigmoid。训练奖励模型 = 拟合这个式子。

**第二步:RLHF 目标的闭式解。**RLHF 最大化奖励、同时用 KL 把策略拴在参考模型 $\pi_{\text{ref}}$(SFT 模型)附近:

$$\max_\pi\ \mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})
\quad\Longrightarrow\quad
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\, e^{r(x,y)/\beta}$$

反解出 $r$——**奖励被策略显式表达**:

$$r(x, y) = \beta \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$$

**第三步:代回 Bradley-Terry。**奖励消掉了,只剩策略与参考的 log 概率比——这就是 DPO 损失:

$$\boxed{\mathcal{L}_{\text{DPO}} = -\log \sigma\Bigl(\beta\,\Delta\Bigr),\qquad
\Delta = \underbrace{\bigl(\log \pi_w - \log \pi_{\text{ref},w}\bigr)}_{\text{chosen 隐式奖励}}
- \underbrace{\bigl(\log \pi_l - \log \pi_{\text{ref},l}\bigr)}_{\text{rejected 隐式奖励}}}$$

梯度(做题时推一遍,面试常考):

$$\nabla\mathcal{L} = -\beta\,\sigma(-\beta\Delta)\Bigl[\nabla\log\pi_w - \nabla\log\pi_l\Bigr]$$

人话版:系数 $\sigma(-\beta\Delta) \in (0,1)$ 是**难例权重**——已经赢麻了($\Delta$ 大)的对子梯度自动变小,模型把火力集中在还输着的偏好上;方括号里「抬 chosen、压 rejected」。

---

## 3. 代码:两遍法,反向机器零改动

先做一个一行的泛化:23/25 课反向里的监督 mask 推广为**逐 token 权重** $w_t$——

$$d\logits_t = (p_t - \mathbb{1}[y_t])\cdot w_t
\qquad\Longleftrightarrow\qquad
\mathcal{L} = -\textstyle\sum_t w_t \log p_t$$

CE 是 $w = \mathrm{mask}/N$ 的特例;DPO 只是换了权重:

```python
def dpo_step(pol, ref, ex_w, ex_l, beta=0.2):
    """一个偏好对上的 DPO:两遍前向定系数,两遍加权反向拿梯度。
    weighted 反向的语义是 loss=−Σ w·logp 的梯度,所以
    chosen 权重 = +βσ(−βΔ)(像 CE),rejected 权重 = −βσ(−βΔ)(负号=压低坏答案)。"""
    lp_w = seq_logprob(pol, *ex_w);  lp_l = seq_logprob(pol, *ex_l)
    lp_rw = seq_logprob(ref, *ex_w); lp_rl = seq_logprob(ref, *ex_l)
    delta = (lp_w - lp_rw) - (lp_l - lp_rl)              # 隐式奖励差
    loss = -math.log(1 / (1 + math.exp(-beta * delta)))   # −log σ(βΔ)
    sig_neg = 1 / (1 + math.exp(beta * delta))            # σ(−βΔ) 难例权重
    def ans_weights(ids, n, c):                          # 答案段各位置乘 c
        w = [0.0] * (len(ids) - 1)
        for j in range(1, n + 2):
            w[-j] = c
        return w
    g = add_grads(weighted_loss_and_grads(pol, ex_w[0][:-1], ex_w[0][1:],
                                          ans_weights(ex_w[0], ex_w[1],  beta * sig_neg)),
                  weighted_loss_and_grads(pol, ex_l[0][:-1], ex_l[0][1:],
                                          ans_weights(ex_l[0], ex_l[1], -beta * sig_neg)))
    return loss, delta, g
```

`seq_logprob` = 答案段各 token 的 log 概率之和(前向即可,不需要梯度);`ref` 是 SFT 模型的冻结拷贝,它的 logprob **每对只需算一次**(预计算缓存)。

---

## 4. 验收与战争故事:符号翻转

数值验收通过:**DPO 梯度抽查 72 坐标 max|err| = 1.66e-10** ✓。

但第一版不是这样的——第一版把 $c_w$ 直接当权重传,忘了加权反向的语义是 $-\sum w\log p$(CE 的负号世界)。诊断指纹极漂亮:

```text
逐行范数完全相等,但个别坐标 ana = +2.04 / num = −2.04 —— 符号整体翻转
```

范数相等说明幅值链路全对,符号错说明**语义约定**错:比较「+logp 的梯度」和「−logp 的梯度」,差一个负号。修正:chosen 权重取 $+\beta\sigma(-\beta\Delta)$、rejected 取 $-\beta\sigma(-\beta\Delta)$(负权重就是「对 rejected 做梯度上升」——DPO「压低坏答案」的那一半,在这里第一次现出原形)。

---

## 5. 真实运行:偏好分离器,不是知识注入器

实验设计:24 课基座 → **故意欠训的弱 SFT**(150 步,任务 18/50)→ 偏好对 50 个(chosen = 正确答案;rejected = **模型自己的错误输出**,修不了的才用随机错答)→ DPO。

**主表(全部真实运行):**

| 工况 | 任务准确率 | Δ 均值 | 西游记 val |
|---|---|---|---|
| 弱 SFT 起点 | 18/50 | 0.000 | 5.5982 |
| DPO β=1.0(最稳) | 16/50 | +4.997 | 5.6370 |
| DPO β=0.5 温和 | 14/50 | +6.110 | 5.6495 |
| DPO β=0.2 激进 500 步 | **11/50 ↓** | **+19.033** | 5.9756 |

三个层次读这张表:

- **DPO 百分百做自己被训练的事**:Δ(隐式奖励差)从 0 拉到 +5~+19,**Δ>0 的占比 100%**——chosen 与 rejected 被彻底分开,损失 0.19→0.05;
- **但它不保证任务变好**:所有工况的贪心准确率都降了。抽样看,有修好的(反义词:大,「后」→「小」✓)也有修坏的(重复:筋斗,「宝殿」→「叉,只见那」✗)——**DPO 优化的是偏好边际,不是答案正确性**:rejected 里的常用字被整体压低,顺带扭曲了整个答案分布;
- **β 是安全带,单调起效**:β 0.1→0.5→1.0,Δ +7.07→+6.11→+5.00(越拉越温和),val 5.667→5.650→5.637(漂移越小),准确率 14→14→16(伤得越轻)。β 就是「离参考模型多近」的那根皮筋。

**难例权重的自灭**(σ(−βΔ) 前半 vs 后半训练):0.296 → 0.098——赢麻的对子自动闭嘴,梯度集中到还输着的偏好上,§2 的公式照进现实。

**对照:更弱的起点也一样。**400 步弱基座 + 弱 SFT(7/50)+ 温和 DPO:7 → 8——margin 照样拉开,准确率只勉强挪一步。结论稳了:**错误是「偏好形状」的(两答案模型都会、只是排序错),DPO 能修;错误是「知识形状」的(正确答案模型根本不会),DPO 无能为力——它不创造知识,只重排已有知识**。这正是生产用法:SFT 练到足够好,再用 DPO 打磨风格与格式偏好;指望 DPO 教新技能,方向就错了(下一课 GRPO 用**真实奖励信号**正面解决这个问题)。

---

## 6. 生产账本

- **DPO vs PPO**:DPO 简单稳定(两个模型、离线数据、一个损失);PPO 类在线方法能持续探索新样本、上限更高但工程复杂(奖励模型 + 策略 + 价值 + 参考四模型同调)。工业界两者都用:DPO/变体是多数开源对齐的默认起点。
- **长度偏置**:DPO 倾向偏好长回答(chosen 通常更长,logprob 之和的差容易被长度主导)——生产要长度归一化或专门的变体。
- **变体一句话**:IPO(把 sigmoid 换成更稳的目标)、KTO(不需要成对比较,好/坏单标签即可)、迭代 DPO(多轮采样新偏好对)——查证作业。
- **数据质量 > 一切**:偏好标注的噪声(标注员不一致)对 DPO 伤害比 SFT 大,因为负样本也在提供梯度。

---

## 7. 记住的故事

**现实:** 「人类最便宜的标注是『A 比 B 好』;DPO 把整个 RLHF  pipeline 压缩成这个信号上的一个分类损失。」

**数学:** $r(x,y) = \beta\log\frac{\pi}{\pi_{\text{ref}}}$——奖励藏在策略里;$\mathcal{L} = -\log\sigma(\beta\Delta)$,梯度带难例权重 $\sigma(-\beta\Delta)$。

**工程:** 「DPO 是偏好分离器,不是知识注入器:Δ 100% 拉开,准确率可能反降——β 是安全带,别把它当引擎。」

**代码:** 第 3 节两遍法约 30 行,23 课反向机器零改动。

---

## 作业

### A. 概念题

1. 为什么 DPO 不需要奖励模型?奖励去哪了?(用 §2 第二步的式子回答)
2. σ(−βΔ) 这个系数在训练中起什么作用?训练后期它会怎样?
3. 「DPO 之后任务准确率反而下降」——从「DPO 优化什么」的角度解释这个现象。
4. β 变大,Δ 的增长、与参考模型的距离、难例权重,各往哪个方向变?

### B. 填空题

1. DPO 损失 = −log σ(____·Δ);Δ = (log π_w − log ____ ) − (log ____ − log π_ref,l)。
2. 本课验收:DPO 梯度抽查 max|err| = ____;第一版 bug 是____翻转(行范数____,坐标____)。
3. 主表:弱 SFT ____/50 → β=0.2 激进后 ____/50,但 Δ 从 0 拉到 ____;β=1.0 时 ____/50。
4. 难例权重 σ(−βΔ) 前半 ____ → 后半 ____;ref 模型每对只需 ____ 次 logprob(因为____)。

### C. 代码手撕(关讲义)

1. 默写 `dpo_step` 的两遍法(先定 Δ 与权重,再两次加权反向),过数值裁判(1e-8)。
2. 手撕 `seq_logprob`(答案段 logprob 之和),并与逐 token 手算对照一个 3 token 的例子。
3. 推导 $\nabla\mathcal{L}_{\text{DPO}}$ 的方括号展开式,并指认代码里哪两行分别是「抬 chosen」「压 rejected」。

### D. 生产环境实战

1. 容量题:DPO 训练 7B 模型:策略 + 参考两份 bf16 权重共多少 GB?为什么参考模型可以完全不需要梯度/优化器状态(算一笔省下的账)?
2. 事故题:DPO 后模型输出普遍变长,评测 ROUGE 反而降了。给出排查与处置(长度偏置?数据里 chosen 更长?归一化?)。
3. 查证:KTO 与 DPO 的关键区别一句话;迭代 DPO(iterative DPO)的循环结构是什么?

### E. 拓展

1. 实验:把偏好对全部换成「chosen=正确、rejected=随机错答」(不用模型自己的错误),重跑主表——哪种 rejected 学得更快?为什么?(ipynb TODO)
2. 预习钩子:DPO 的 Δ 里没有一点「答案对不对」的真实信号,只有偏好。如果任务**可自动判分**(答案对就是 1,错就是 0),能不能让奖励直接进场?下一课 GRPO:同一道题采一组回答,组内减基线,PPO 式裁剪——DeepSeek-R1 的选择,27 课欠的「真奖励」一次性补齐。

---

# 下一课预告:GRPO

DPO 的偏好来自人;GRPO(Group Relative Policy Optimization)的奖励来自**规则**:同一道题采 G 个回答,能自动判分的任务(算对没有?格式对没有?)直接给分,组内标准化当优势,替代价值网络——再配上 PPO 的裁剪与 KL 惩罚。训练篇的终章:base → SFT → DPO → GRPO,四段全部亲手跑过一遍。

> **下一课:[GRPO——组内基线的强化学习](28-grpo.md)——奖励来自规则,基线来自同伴。**

## 参考来源

**本课实测**:基座为第 24 课配置的确定性复刻;偏好对构造、四工况主表与弱基座对照全部来自本机真实运行。

**文献**:

- [Rafailov et al. 2023, Direct Preference Optimization (arXiv:2305.18290)](https://arxiv.org/abs/2305.18290) — DPO 原始论文(闭式解推导的出处)
- [Ouyang et al. 2022, InstructGPT (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — RLHF 三阶段的原始 pipeline(DPO 要替代的对象)
- [Schulman et al. 2017, PPO (arXiv:1707.06347)](https://arxiv.org/abs/1707.06347) — 裁剪代理目标(下一课的主角)
