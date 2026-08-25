# 第三十二课:投机解码——小模型代笔,大模型签名

> 解码的天花板是**串行**:一个 token 一个 token,大模型每步都要跑满全场。投机解码(speculative decoding)的思路:让小模型(draft)先连写 k 个,大模型(target)**一次前向**全部验证,对的采纳、错的纠正——数学上保证输出分布与「大模型独自解码」**完全相同**,加速纯粹来自「串行变批处理」。本课在《西游记》上真训一大一小(d=64 与 d=32,参数 713,600 与 324,032),实测:接受率 **30%**,大模型前向次数 **120 → 56(−53%)**,输出**逐位一致**;同时撞见一个诚实的教训——numpy 里它反而**慢** 0.55×(§5 有公式,有答案)。
> **本课数字**:全部来自本机真实运行(两个模型各训 2000 步,种子固定)。

---

## 1. 先不用代码——代笔与签名

大作家(target)写书一字一顿,每个字都要翻遍全部腹稿;实习生(draft)下笔飞快,但常写错。流程:

1. 实习生连写 k 个字(快,便宜);
2. 大作家**一次通读**这 k 个字加下一处(大模型一次前向能同时给出 k+1 个位置的裁决——第 5 课的因果掩码白送的能力);
3. 逐字核对:对到第几个错为止——前面全采纳;第一个错位**按大作家的版本纠正**;
4. 若全对,大作家亲笔补一个字(bonus token);
5. 从纠正/补位处继续。

每一轮至少产出 1 个「大作家亲笔」token,**永不倒退,永不出错**——错字也会被换成正确的。加速的全部来源:一次通读 k 个字的代价 ≈ 写一个字(prefill 的并行性,29 课)。

---

## 2. 数学:为什么分布不变

**贪心版(本课实现)**:接受 ⇔ draft 的 argmax == target 的 argmax;错位处取 target 的 argmax。显然输出 ≡ target 独自贪心。

**采样版(生产默认)**:拒绝采样规则——target 分布 $p$,draft 分布 $q$,draft 采样出 $x$,以 $\min(1, p(x)/q(x))$ 接受;拒绝时从残差 $(p-q)_+$ 归一化重采。证明一行:

$$P(\text{输出}=x) = q(x)\cdot\frac{p(x)}{q(x)} + \Bigl(1-\textstyle\sum_y q(y)\tfrac{p(y)}{q(y)}\Bigr)\cdot\frac{(p(x)-q(x))_+}{\sum_y (p(y)-q(y))_+} = p(x)$$

第一项是「draft 抽中且被接受」,第二项是「被拒后从残差重采」——两项相加恰好还原 $p$。**加速不牺牲分布**,这是投机解码相对「直接用小模型」的本质区别。

**加速条件**(本课的主角公式):记 target 单次前向成本 $F_t$、draft 为 $F_d$、逐位接受率 $\alpha$,则

$$\text{每 token 成本} = \frac{k F_d + F_t}{\alpha k + 1},\qquad
\text{加速} > 1 \iff \boxed{F_d / F_t < \alpha}$$

draft 必须比接受率**更便宜**,否则代笔比自己写还贵。

---

## 3. 代码:验证式投机解码

```python
def greedy_speculative(target, draft, prompt, n, k=4):
    ids = list(prompt)
    while len(ids) - len(prompt) < n:
        d_ids = list(ids)                       # ---- draft 连写 k 个(贪心)----
        draft_tokens = []
        for _ in range(k):
            dl, _ = draft.forward(d_ids)
            t = int(np.argmax(dl[-1]))
            draft_tokens.append(t)
            d_ids.append(t)
        logits, _ = target.forward(d_ids)       # ---- target 一次前向验证 ----
        new_ids = list(ids)
        for i, dt in enumerate(draft_tokens):
            tgt_t = int(np.argmax(logits[len(new_ids) - 1]))
            if tgt_t == dt:
                new_ids.append(dt)              # 采纳
            else:
                new_ids.append(tgt_t)           # 纠正,停在这里
                break
        else:
            new_ids.append(int(np.argmax(logits[len(d_ids) - 1])))   # 全对:bonus
        ids = new_ids
    return ids[:len(prompt) + n]
```

关键一行:`logits[len(new_ids) - 1]`——target 对 `prompt+draft` 的一次前向,第 $i$ 个位置的 logits 正是对第 $i$ 个 draft token 的裁决,第 $k+1$ 个位置是 bonus 的裁决。**因果掩码让一次前向同时当 k+1 次法官**(第 5/6 课)。

## 4. 真实运行:一大一小,全部实测

两个模型在《西游记》上各训 2000 步(24 课同款循环):大 = d=64(713,600 参数),小 = d=32(324,032 参数,45%)。提示「那猴王」,生成 120 token:

```text
[正确性] 投机输出与基线贪心逐位一致:True        ← 分布不变的直接验证
[接受率] 逐位接受率 30%(k=4)
[target 前向] 56 次(基线 120 次,−53%)         ← 大模型的算力账省一半
[输出样例] 子,只见那里去,只见那里去,只见那里去……  ← 2000 步模型的老毛病(24 课),两模型口径一致
```

k 扫描(同两模型):

```text
k=2:接受 32% | target 前向 74 次
k=4:接受 30% | target 前向 56 次   ← 本配置最优
k=6:接受 20% | target 前向 56 次   ← k 太长,错得更早,白写
```

## 5. 诚实的墙钟:本课的「反直觉」主发现

实测墙钟:基线 **1229 tok/s**,投机 **675 tok/s——慢了 0.55×**。用 §2 的公式反解:0.55 = (0.30×4+1)/(4r+1) ⟹ **r = F_d/F_t ≈ 75%**——d=32 的 draft 在 numpy 里单次前向竟然要大模型 75% 的代价!违反了加速条件 $r < \alpha$(75% ≫ 30%)。

为什么:numpy 的前向成本被**解释器与算子开销主导**,矩阵从 64×64 缩到 32×32 省不了多少时间。生产环境完全不同:7B target + 0.5B draft,$F_d/F_t ≈ 7\% \ll \alpha$(实测接受率常 60–80%),公式给出 $k{=}4, \alpha{=}0.7$ 时加速 ≈ 3×。**投机解码不是「小模型就快」,是「代笔足够便宜且足够像」**——本课的 numpy 实验恰好把这条边界画了出来。

三个生产注脚:

- **接受率是唯一要盯的指标**:它 = 「draft 与 target 的贪心/采样一致性」;同族蒸馏的 draft 接受率最高(表层数学能力类似,深层知识不同);
- **k 是旋钮**:k 大省「前向次数」但错得更早;生产自适应投机(动态 k)按最近接受率调;
- **与大模型算力解耦**:target 前向 −53% 意味着 GPU 占用降一半——同样算力可以伺候双倍并发,或给 KV(30 课)腾显存。

---

## 6. 记住的故事

**现实:** 「实习生代笔,大作家一次通读签名:输出与亲笔逐位相同,加速全靠串行变批。」

**数学:** 拒绝采样两项相加还原 $p$;加速条件 $F_d/F_t < \alpha$。

**工程:** 「本课 numpy 实测反而慢 0.55×:小模型在这儿不够便宜——加速条件是算出来的,不是许愿出来的。」

**代码:** greedy_speculative 约 25 行;因果掩码一次前向当 k+1 次法官。

---

## 作业

### A. 概念题

1. 投机解码为什么「永不倒退、永不出错」?每轮最少产出几个 token、由谁保证?
2. 采样版的残差重采 $(p-q)_+$ 为什么必要?只用 min(1, p/q) 会怎样?
3. 本课 numpy 实测变慢的原因是什么?给出公式与数值。
4. k 从 4 加到 8,接受率为什么下降?target 前向次数为什么可能不变?

### B. 填空题

1. 本课实测:接受率 ____%,target 前向 ____ → ____ 次(−____%),输出与基线____。
2. 加速条件:____ / ____ < ____;本课反解出 r ≈ ____%,违反条件。
3. 生产 7B+0.5B:r ≈ ____%,若 α=70%、k=4,加速 ≈ ____×。
4. 全对时 bonus token 来自 target 对第 ____ 位置的 logits。

### C. 代码手撕(关讲义)

1. 默写 greedy_speculative(含错位纠正与 bonus),并用「输出与基线逐位一致」验收。
2. 手撕采样版:实现拒绝采样规则,对玩具分布(p、q 已知)跑 1 万次,卡方/频率对照 p(分布不变的实证)。
3. 把 draft 换成 d=16,重测接受率与 r,画出「α−r 平面」上的加速边界。

### D. 生产环境实战

1. 容量/吞吐题:70B target + 7B draft,A100 上分别跑 bf16:F_d/F_t ≈ 多少?若 α=60%,k=5,理论加速多少?draft 常驻显存多花多少?
2. 事故题:线上投机解码吞吐反而下降。排查顺序(接受率掉了?draft 与 target 版本不匹配?k 配置?)。
3. 查证:Medusa(多头 draft)与 EAGLE 的思路和「独立小模型 draft」差在哪?各一句话。

### E. 拓展

1. 实验:接受率随生成位置的变化曲线(开头 vs 重复段)——哪里的草稿最像大作家?(ipynb TODO)
2. 预习钩子:课程最后一张牌——前 32 课的零件(加载、RoPE、GQA、SwiGLU、KV cache、分词)全部装到**一台真实模型**上:纯 numpy 加载 SmolLM2-135M 的真实权重,真实生成英文。毕业设计见。

---

# 下一课预告:毕业设计 II——纯 numpy 跑真模型

31 课读了它的权重,30 课算了它的 KV,现在把它**跑起来**:safetensors 逐张量加载、bf16→fp32、Llama 架构前向(GQA/RoPE/SwiGLU,与 20 课零件一一对齐)、KV cache 接上、BPE 编码(22 课)、真实生成、实测 tok/s。手撕课程的终点:你写的每一行,都站在真实模型里。

> **下一课:[毕业设计 II——纯 numpy 跑真模型](33-numpy-llm.md)——所有零件,一次点火。**

## 参考来源

**本课实测**:大/小模型各训 2000 步(24 课同款循环,种子固定);接受率/前向次数/墙钟为真实运行。

**文献**:

- [Leviathan et al. 2023, Fast Inference from Transformers via Speculative Decoding (arXiv:2211.15192)](https://arxiv.org/abs/2211.15192) — 拒绝采样式投机解码(与 Chen et al. 2023 独立同期的另半)
- [Chen et al. 2023, Accelerating LLM Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — DeepMind 同款
- [Cai et al. 2024, Medusa (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — 多头 draft 的代表
