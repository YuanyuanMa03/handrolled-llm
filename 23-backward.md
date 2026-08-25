# 第二十三课:Mini-GPT 反向传播手撕——数值梯度当裁判

> 训练一台模型,就是把 loss 对**每一个权重**的偏导数算出来,交给优化器(第 13 课)。第 20 课的 Mini-GPT 有 33,792 个坐标——本课手撕全部 33,792 条梯度,并请**数值梯度**当裁判:误差超过 1e-8,不许出厂。
> **本课所有数字均来自本机真实运行。**开发过程中,裁判当场抓住了两个真 bug——战争故事见 §7,先卖个关子:一个错在 `1/r`,一个错在乘了一半的乘积法则。

---

## 0. 任务书:从 forward 到 forward+backward

第 20 课的 Mini-GPT 只有 `forward`——它能算 loss,但不能算「loss 对每个权重的导数」。第 13 课的 AdamW 吃的正是这些导数:

```text
训练一步 = forward(ids) → loss
         + backward(loss) → grads        ← 本课要造的机器
         + AdamW.step(grads) → 更新权重   ← 第 13 课已有
```

为什么不用 PyTorch 的 autograd?因为**手撕一遍,才知道 autograd 每天在替你做什么**——面试问「RMSNorm 的反向是什么」,答案不在框架里,在这节课里。生产环境当然用框架(§8 有账本)。

整台机器的路线图(计算图,从上往下是 forward,从下往上是本课):

```text
ids ──► E 查表 ──► x ─┬─► Wq,Wk,Wv ─► RoPE ─► scores ─► softmax ─► attn ─► Wo ─┐
                      │                                                          + ◄┐ 残差
                      │            rms ─► W1,W3 ─► silu⊙ ─► W2 ────────────────+ ◄┘
                      └──(× n_layers)──────────────────────────────────────────┘
final rms ─► head ─► logits ─► softmax+CE ─► loss          ← 反向沿这条路回程
```

---

## 1. 先不用代码——工厂的召回流水线

把 forward 想成一条装配线:原料(embedding)经一站一站加工,出厂成品(logits),质检员给出一个损失(loss)。

反向传播就是**召回通知沿传送带逆行**:质检员说「成品超差 0.3」,通知逆流而上,每一站做两件事——

1. **认领自己那份责任**:哪些偏差是我这个工位的参数(权重)造成的?记进它的账户(grads);
2. **转发剩余责任**:剩下的责任是谁上游寄来的原料的?打包寄回去(输入梯度 dx)。

链式法则就是这份**责任分摊协议**:每站只算自己的局部导数,乘以上游寄来的梯度,就得到全局梯度。第 12 课的两条「万能模板」是协议里最常用的两条;本课再补四个新工位的局部导数,然后焊成整机。

---

## 2. 四个新工位的局部梯度(数学)

**线性层两模板(第 12 课,直接引用)。** $Y = XW$:$\ dW = X^\top dY$,$\ dX = dY\,W^\top$。全课反复使用,不再展开。

**softmax + 交叉熵(第 12 课,直接引用)。** $\ dz_i = p_i - \mathbb{1}[i{=}\text{target}]$。整机里再除以位置数 $T$ 取平均。

**RMSNorm(无 gamma 版)。** $y = x / r$,$\ r = \sqrt{\tfrac{1}{d}\sum_j x_j^2 + \epsilon}$。注意 $\epsilon$ 让 $r$ 对 $x_j$ 也有导数:

$$\frac{\partial r}{\partial x_j} = \frac{x_j}{d\,r}
\;\Longrightarrow\;
dx_j = \frac{dy_j}{r} - \frac{x_j}{d\,r^3}\sum_i dy_i\, x_i
\;\Longrightarrow\; \boxed{dx = \frac{dy - y\cdot\text{mean}(dy \odot y)}{r}}$$

(把 $x_j$ 换成 $y_j r$ 即得最后一行——本课第一个战争故事就错在这一步的 $r$ 上。)

**RoPE:旋转的转置。** 前向对每对分量 $(x_i, x_{i+\text{half}})$ 做角度 $m\theta_i$ 的二维旋转 $R$。旋转矩阵**正交**:$R^\top R = I$,所以反向就是转置——**角度取负再转一次**:

$$\text{forward: } \begin{pmatrix}x_c'\\x_s'\end{pmatrix} = \begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}\begin{pmatrix}x_c\\x_s\end{pmatrix}
\quad\Longrightarrow\quad
\text{backward: 角度取 } -m\theta$$

**SwiGLU(乘积法则!)。** $\text{gate} = \text{silu}(u) \odot z_v$,$\ u = x\,W_1$,$\ z_v = x\,W_3$。乘积法则,两个因子各拿一半:

$$\frac{\partial\,\text{gate}}{\partial u} = \text{silu}'(u) \odot z_v
\qquad
\frac{\partial\,\text{gate}}{\partial z_v} = \text{silu}(u)$$

$$\text{silu}'(u) = \sigma(u)\bigl(1 + u\,(1-\sigma(u))\bigr)$$

**attention 的 softmax Jacobian。** 每行 softmax 对 scores 的导数(第 1 课的推广):

$$ds = p \odot \Bigl(dp - \text{rowsum}(dp \odot p)\Bigr)$$

因果掩码的梯度**自动为零**:被掩成 $-\infty$ 的位置 $p=0$,于是 $ds=0$——不需要在反向里再掩一次。这是 forward 里那行 `np.where(mask, -inf, s)` 白送的。

---

## 3. 代码:loss_and_grads——tape 前向 + 整机反向

手撕反向的经典结构:**forward 时把中间量记在 tape 上**(激活),backward 沿 §0 路线图回程。这正是 PyTorch `retain_grad` 背后的东西,只是我们全部手写。

```python
def loss_and_grads(gpt, ids, targets):
    """带 tape 前向 + 整机反向。返回 (loss, grads)。
    grads = {'E', 'head', 'layers': [ {Wq..W3} ]},键与形状和 gpt 权重一一对应。"""
    T = len(ids)
    pos = np.arange(T)
    tape = []
    x = gpt.E[list(ids)]
    for W in gpt.layers:
        q, k, v = gpt.rope(x@W['Wq'], pos), gpt.rope(x@W['Wk'], pos), x@W['Wv']
        mask = np.triu(np.ones((T, T), bool), 1)          # 第 18 课的差一教训
        ps, outs = [], []
        for h in range(gpt.H):
            sl = slice(h*gpt.dh, (h+1)*gpt.dh)
            s = q[:, sl] @ k[:, sl].T / math.sqrt(gpt.dh)
            s = np.where(mask, -np.inf, s)
            p = softmax(s)                                 # tape:逐头概率
            ps.append(p); outs.append(p @ v[:, sl])
        concat_o = np.concatenate(outs, -1)
        x_mid = x + concat_o @ W['Wo']                     # tape:残差汇合点
        x_rms = gpt.rms(x_mid)
        u, zv = x_rms @ W['W1'], x_rms @ W['W3']
        gate = u / (1 + np.exp(-u)) * zv                   # SwiGLU
        x_out = x_mid + gate @ W['W2']
        tape.append(dict(x_in=x, x_mid=x_mid, x_rms=x_rms, u=u, zv=zv,
                         gate=gate, concat_o=concat_o, q=q, k=k, v=v, ps=ps))
        x = x_out

    x_final = x
    logits = gpt.rms(x_final) @ gpt.head
    loss = ce_loss(logits, targets)

    grads = {'E': np.zeros_like(gpt.E), 'head': np.zeros_like(gpt.head),
             'layers': [dict.fromkeys(W) for W in gpt.layers]}
    p = softmax(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(T), targets] = 1.0
    dlogits = (p - onehot) / T                              # ← 第 12 课 + 取平均
    grads['head'] = gpt.rms(x_final).T @ dlogits            # ← 模板一
    dx = rms_backward(dlogits @ gpt.head.T, x_final)        # ← 模板二

    for i in reversed(range(gpt.NL)):                       # 逆流水线
        W, t = gpt.layers[i], tape[i]
        # ---- FFN 支路:gate = silu(u)⊙zv ----
        dg = dx @ W['W2'].T                                 # 责任拆给 gate
        du = dg * silu_grad(t['u']) * t['zv']               # ← 乘积法则:带上 zv!
        dzv = dg * (t['u'] / (1 + np.exp(-t['u'])))
        grads['layers'][i]['W2'] = t['gate'].T @ dx
        grads['layers'][i]['W1'] = t['x_rms'].T @ du
        grads['layers'][i]['W3'] = t['x_rms'].T @ dzv
        dx_mid = dx + rms_backward(du @ W['W1'].T + dzv @ W['W3'].T, t['x_mid'])
        # ---- attention 支路 ----
        dconcat = dx_mid @ W['Wo'].T
        grads['layers'][i]['Wo'] = t['concat_o'].T @ dx_mid
        dvs, dqs, dks = [], [], []
        for h in range(gpt.H):
            sl = slice(h*gpt.dh, (h+1)*gpt.dh)
            p_h, v_h, q_h, k_h = t['ps'][h], t['v'][:, sl], t['q'][:, sl], t['k'][:, sl]
            dO = dconcat[:, sl]
            dvs.append(p_h.T @ dO)                          # dv:概率当权重
            dp = dO @ v_h.T
            ds = p_h * (dp - (dp * p_h).sum(-1, keepdims=True))   # softmax Jacobian
            dqs.append(ds @ k_h / math.sqrt(gpt.dh))
            dks.append(ds.T @ q_h / math.sqrt(gpt.dh))
        dv = np.concatenate(dvs, -1)
        dq = rope_backward(np.concatenate(dqs, -1), pos, gpt.D)   # 转置=角度取负
        dk = rope_backward(np.concatenate(dks, -1), pos, gpt.D)
        grads['layers'][i]['Wv'] = t['x_in'].T @ dv
        grads['layers'][i]['Wq'] = t['x_in'].T @ dq
        grads['layers'][i]['Wk'] = t['x_in'].T @ dk
        dx = dx_mid + dq @ W['Wq'].T + dk @ W['Wk'].T + dv @ W['Wv'].T  # 残差+三投影

    np.add.at(grads['E'], np.array(ids), dx)                # embedding:scatter-add
    return loss, grads
```

配套的四个局部梯度(`ce_loss` 是第 11 课的稳定写法):

```python
def rms_backward(dy, x, eps=1e-5):
    r = np.sqrt((x**2).mean(-1, keepdims=True) + eps)
    y = x / r
    return (dy - y * (dy * y).mean(-1, keepdims=True)) / r

def rope_backward(dy, pos, dim):
    """前向是正交旋转 R(m),反向 = R(m)ᵀ:角度取负再转一次。"""
    half = dim // 2
    ang = pos[:, None] / (10000 ** (np.arange(half)*2/dim))
    c, s = np.cos(ang), np.sin(ang)
    dy1, dy2 = dy[..., :half], dy[..., half:]
    return np.concatenate([dy1*c + dy2*s, -dy1*s + dy2*c], -1)

def silu_grad(u):
    sig = 1 / (1 + np.exp(-u))
    return sig * (1 + u * (1 - sig))

def ce_loss(logits, targets):
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return -logp[np.arange(len(targets)), targets].mean()
```

注意一个**结构事实**(20 课原码如此):attention 吃的是**原始 x**,只有 FFN 前面有 `rms`——反向必须按真实结构焊,不能按教科书里「处处 pre-norm」的想象焊。

---

## 4. 裁判:中心差分

数值梯度不依赖任何推导,直接问 loss 本身:

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + h) - L(\theta - h)}{2h} + O(h^2)$$

float64 下取 $h = 10^{-6}$,噪声地板约 $10^{-9}$。验收分两级:**逐模块**(局部梯度各自过裁判,隔离病灶)和**整机**(对 20 课原版 `forward` 求数值梯度,和手撕反向对账——原版 forward 是可信基准,它能抓住反向里任何一环的错误)。

```python
def num_grad(f, x, h=1e-6):
    """f() 返回标量且读的是 x 本体;逐坐标中心差分。"""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h; fp = f()
        x[i] = old - h; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g
```

---

## 5. 真实运行:出厂检验

**检验 0:taped 前向必须和 20 课原版逐位一致**(否则对账对象就错了):

```text
taped 前向 vs 原版 forward:max|diff| = 0.0        ← 逐位相同,对账对象成立
loss(ids=[1..6]) = 3.21532753                       ← 与 20 课 3.2169 同一量级(序列不同)
```

**检验 1:四个局部梯度各自过裁判**(随机输入,真实运行):

```text
rms_backward   max|err| = 2.59e-09   ✓
rope_backward  max|err| = 1.30e-09   ✓
silu_grad      max|err| = 1.64e-09   ✓
(softmax+CE 的 p−onehot 第 12 课已验,整机里隐式复检)
```

**检验 2:整机全参数扫描**——数值梯度对**全部 33,792 个坐标**逐个计算(数值基准用的是 20 课原版 `forward`),与手撕反向对账:

```text
整机梯度验收:全 33,792 坐标,max|err| = 9.08e-10 @ layers[0].W1,用时 9.3s   ✓ 出厂
```

**检验 3:换配置不藏私**——换两套完全不同的形状/头数/层数,抽查坐标:

```text
vocab=20 d=16 2头 1层:抽查  90 坐标 max|err| = 3.75e-10   ✓
vocab=11 d=24 3头 3层:抽查 230 坐标 max|err| = 5.73e-10   ✓
```

**检验 4:反向的代价**——工程口诀「反向 ≈ 2× 前向」实测:

```text
forward          0.18 ms
forward+backward 0.33 ms   → 反向/前向 ≈ 1.8×   ← 口诀的玩具版验证
```

顺手的真实观察:梯度绝对值最大的张量是 `head`(0.45),垫底的是 `Wq/Wk`(0.003,差近两个数量级)——softmax+CE 的梯度集中在出口;而 Q/K 只通过 scores 间接影响 loss,梯度最小。**出口学得快、深层慢慢来**,这是后面训练曲线上「不同层学习速度不同」的微观来源(第 24 课你会再见到它)。

---

## 6. 梯度裁剪:训练 loop 的安全带(第 24 课直用)

反向算出的梯度在训练里要先过一道**全局范数裁剪**再进优化器——防止个别 batch 的野梯度(d长序列、脏数据)把权重一步打飞(第 18 课 NaN 血泪的预防针):

```python
def clip_grad_norm(grads, max_norm):
    """所有参数梯度拼成一个全局范数;超上限则等比缩放。返回(裁前, 裁后)。"""
    total = math.sqrt(sum((g**2).sum() for g in _iter_grads(grads)))
    scale = min(1.0, max_norm / (total + 1e-6))
    for g in _iter_grads(grads):
        g *= scale
    return total, total * scale
```

真实运行(把本课梯度放大 100 倍模拟野梯度,上限 1.0):

```text
clip_grad_norm:范数 282.887 → 1.000(方向不变,步长封顶)
```

---

## 7. 战争故事:裁判抓住的两个真 bug

本课代码第一版有**两个真 bug**,都在 §5 的验收里被数值裁判当场按住。复现如下:

**Bug 1:RMSNorm 掉了一个 `1/r`。** 第一版写成 `dy/r − y·mean(dy⊙y)`,正确的是再除一个 `r`(§2 的 $x_j/(d\,r^3)$)。裁判报告:`rms_backward max|err| = 7.97e-01`——错得离谱但**只错在尺度上**,方向全对:推导时把 $x_j = y_j r$ 代换后忘了除回去。

**Bug 2:SwiGLU 乘积法则只乘了一半。** 第一版写 `du = dg * silu_grad(u)`,漏了另一个因子 $z_v$。裁判报告最妙的地方在**误差的形状**:

```text
L1.W2  6.15e-10 ✓    L1.W3  7.04e-10 ✓    ← 末层 W2/W3 全对
L1.W1  9.34e-02 ✗                        ← 只有 W1 错,两层都错
Wv/Wo/E/Wq/Wk  1e-3 ~ 6e-2 ✗             ← 下游全被污染(W1 的错梯度经 dx 传回)
```

W3 对、W1 错,而两者只差 `du` 和 `dzv`——病灶当场锁定在乘积法则那一行。**误差的模式就是指纹**:全对/全错告诉你有没有 bug,**哪里对哪里错**告诉你 bug 在哪。

两条教训(比代码本身值钱):手工推导的每一条反向公式,必须过数值裁判;裁判报告要**逐张量**看,不要只看一个 max。

---

## 8. 生产账本:手撕与框架的分工

- **生产没人手撕反向。**PyTorch 的 autograd 在每次 forward 自动记 tape(本课手写的那个 dict),`loss.backward()` 沿图回程;`torch.compile`/XLA 再把正反向编译成算子。但**面试必考**:RMSNorm/attention 的反向、梯度检查怎么做,就是本课内容。
- **训练显存 = 权重 + 梯度 + 优化器状态 + 激活。**前三项第 13 课算过(AdamW fp16 混合:12 字节/参数);第四项就是本课的 tape——每个中间量(`x_mid/x_rms/u/zv/gate/q/k/v/ps`)训练时都要留档,大模型用**激活重计算**(丢掉一部分 tape,反向时重算)换显存。我们的 tape 是显式 dict,框架里是隐式图,账是同一本。
- **精度。**数值梯度检查要 float64 + h≈1e-6;真训练用 bf16/fp16 混合精度,反向的数值敏感点(softmax、log)留在 fp32——第 1 课稳定 softmax 的教训在训练里加倍成立。

---

## 9. 记住的故事

**现实:** 「训练 = forward 拿 loss + backward 拿梯度 + optimizer 更新——本课补上中间那台机器。」

**数学:** $dx = \frac{dy - y\cdot\text{mean}(dy\odot y)}{r}$(RMSNorm);旋转反向=转置;乘积法则一个因子都不能丢。

**工程:** 「误差的模式就是指纹:W3 全对 W1 全错,病灶就在只差的那一行。」

**代码:** 第 3 节 `loss_and_grads` 约 70 行,33,792 个坐标全扫 max|err| 9.08e-10。

---

## 作业

### A. 概念题

1. 为什么因果掩码在反向传播里「白送」零梯度,不需要再掩一次?
2. RMSNorm 的 $\epsilon$ 不参与梯度公式?错——它通过 $r$ 进来了。指出具体位置。
3. RoPE 的反向为什么是「角度取负再转一次」?从旋转矩阵的什么性质来?
4. 数值梯度检查为什么用中心差分 $(f(x{+}h)-f(x{-}h))/2h$ 而不是前向 $(f(x{+}h)-f(x))/h$?

### B. 填空题

1. 本课整机验收覆盖 ____ 个坐标,max|err| = ____,阈值 ____ 才算出厂。
2. 反向/前向耗时比实测 ____,工程口诀约 ____。
3. 本课两个真 bug:RMSNorm 掉了 ____;SwiGLU 的 du 漏乘 ____。
4. 梯度绝对值最大的张量是 ____(0.45),clip 演示中范数从 282.887 缩到 ____。

### C. 代码手撕(关讲义)

1. 默写 `rms_backward`,过你自己的数值裁判(阈值 1e-8)。
2. 推导并手撕 softmax Jacobian 的行式 $ds = p \odot (dp - \text{rowsum}(dp \odot p))$,并用 2×3 的玩具矩阵验证。
3. 默写 `clip_grad_norm`(含 `_iter_grads` 的遍历),演示范数超限与不超限两种情况。

### D. 生产环境实战

1. 容量题:d=4096、L=32、T=4K、batch=8,估算训练一步的**激活**显存量级(每层至少留档哪些量?每个多大?),对比第 18 课 KV cache 的推理账。
2. 事故题:同事的手撕反向过你团队 gradcheck,只有 `Wk` 全错、其余全对。列出你的排查顺序(提示:从「只有 Wk 错」这条指纹反推)。
3. 查证:PyTorch 的 `retain_graph=True` 是干什么的?和本课 tape 里「哪些中间量必须留、哪些可以只留一次」怎么对应?

### E. 拓展

1. 实验:把 `loss_and_grads` 改成支持 batch(`ids` 变 [B,T]),梯度对 B 求平均;单样本 B=1 时与本题输出逐位对照。
2. 预习钩子:梯度机器有了,AdamW(第 13 课)、裁剪(本课 §6)都在桌上——万事俱备。下一课把它俩接上,在《西游记》上真训 1000 步,亲眼看着 loss 从 3.2 压到基线 ln(4551)≈8.42 的世界以下……等等,字级词表基线比 16 词表高这么多?——对,这就是下一课第一个要算的账。

---

# 下一课预告:训练循环

分词器(22)、梯度(23)、优化器(13)全部就位。下一课写 `train()`:数据切片、packed batch、warmup+cosine、梯度裁剪,在《西游记》上真训——并回答那个毕业问题:**loss 压到基线以下之后,乱码会变成什么?**

> **下一课:[训练循环——真训《西游记》](24-train-loop.md)——把第 20 课的承诺兑现。**

## 参考来源

**本课实测**:全部数字来自本机真实运行(MiniGPT 第 20 课原码 + 本课反向,双精度 float64,中心差分 h=1e-6)。

**文献**:

- [CS231n: Gradient Checks](https://cs231n.github.io/neural-networks-3/#gradcheck) — 数值梯度验收的传统做法(h 选择、双精度、逐项检查)
- [Baydin et al., Automatic Differentiation in Machine Learning (arXiv:1802.10575)](https://arxiv.org/abs/1802.10575) — autograd 的系统化视角(本课手写的就是它的反向模式)
