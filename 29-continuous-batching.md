# 第二十九课:连续批处理——推理服务的吞吐引擎

> 训练篇(22–28)把模型训好了;推理篇(29–33)把它**伺候好**。第一站:同一个模型,「读 64 字答 1 字」的请求和「一个字一个字往外蹦」的请求脾气完全不同(第 18 课埋的线)——拼车(static batching)会让快客等慢客,离散发车(continuous batching)让请求随到随走。本课用**真实测量的成本结构**驱动一个手撕调度模拟器:轻载时延 **25.2 → 4.0ms(6×)**,批占用率 **58% → 92%**,还顺手撞见一个生产级难题:散装 prefill 的代价(§5)。
> **本课数字**:成本测量来自本机真实运行(MiniGPT d=64);模拟器按该成本模型仿真,三种负载场景。

---

## 1. 先不用代码——拼车调度台

把推理服务想成一个拼车调度台:

- 乘客(请求)上车时说要去的里程(输出长度)——但**上车前没人知道确切里程**;
- **static batching**:凑满一车(8 人)才发车,全车到齐才回程接下一批——快客陪慢客干等,回程还常常空座;
- **continuous batching**:车位**每走一位就补一位**,候车队伍永不让车位空转。

一句话:static 的调度单位是**批**,continuous 的调度单位是**迭代(iteration)**——每生成一个 token 都是一次重新发车的机会。这就是 vLLM/Orca 这类引擎的核心(Orca 2021 首创 iteration-level scheduling)。

---

## 2. 真实成本结构:prefill 与 decode 的两副脾气

先拿真模型测出成本结构(第 18 课的 KV cache 在这里兑现为数字):

```text
MiniGPT d=64 2层,本机 numpy(真实运行):
prefill 一次性吃 64 token : 0.82 ms(13 µs/token,并行——一次矩阵乘吃完)
decode 带 cache 单步      : 0.14 ms(只算 1 个 token)
decode 无 cache 全量重算  : 0.84 ms(浪费 6×;第 18 课的账,此处实测复现)
```

两副脾气:prefill 是**计算密集**(一次吃一串,吞吐高、时延高);decode 是**访存密集**(每步一个 token,GPU 大部分时间在搬权重和 KV cache)。生产 PD 分离(18 课提过)就是把这两种脾气放到不同机器上伺候。模拟器的成本模型就取这三个实测数:prefill ∝ prompt 长度(同批取 max,GPU 并行),decode 每步近常数(批内近零边际——GPU 并行摊平)。

## 3. 手撕调度器

```python
def simulate(mode, n_req, max_batch=8, seed=0, arrival_gap=0.0, group_admit=False):
    rng = np.random.default_rng(seed)
    reqs = [dict(id=i, arrive=i*arrival_gap,                    # 均匀到达
                 prompt=int(rng.integers(16, 65)),               # 16-64 token
                 remaining=int(rng.integers(8, 41)),             # 输出 8-40 token
                 ttft=None, done=None) for i in range(n_req)]
    t, done_count, idle, busy = 0.0, 0, 0, 0
    iters_since_admit = 99
    running, waiting = [], list(reqs)
    while done_count < n_req:
        while waiting and waiting[0]['arrive'] <= t:             # 到达入候车队
            running.append(waiting.pop(0))
        if mode == 'static':
            if len(running) < max_batch and waiting:
                t = waiting[0]['arrive']                         # 拼车等客:凑满一批
                continue
            batch, running = running[:max_batch], running[max_batch:]
        else:
            batch, running = running[:max_batch], running[max_batch:]
            if not batch:
                t = waiting[0]['arrive']                         # 车位空,等下一位
                continue
        pre_cost = per_tok * max(r['prompt'] for r in batch)     # 批 prefill(并行)
        for r in batch:
            if r['ttft'] is None:
                r['ttft'] = t + pre_cost
        t += pre_cost
        while batch:                                             # decode 循环
            t += t_decode
            for r in batch[:]:
                r['remaining'] -= 1
                if r['remaining'] <= 0:
                    r['done'] = t
                    batch.remove(r)                              # 走客下车
                    done_count += 1
                    if mode == 'continuous':                     # 迟到者进候车队
                        while waiting and waiting[0]['arrive'] <= t \
                              and len(running) < max_batch:
                            running.append(waiting.pop(0))
            if mode == 'continuous':                             # 空位即补:同轮并行 prefill
                take = max_batch - len(batch)
                iters_since_admit += 1
                if group_admit and 0 < take < 3 and iters_since_admit < 8:
                    take = 0                                     # 成组准入:攒位再补
                admitted = running[:take]
                if admitted:
                    pc = per_tok * max(r['prompt'] for r in admitted)
                    for r in admitted:
                        if r['ttft'] is None:
                            r['ttft'] = t + pc
                    t += pc
                    iters_since_admit = 0
                batch += admitted
                running = running[take:]
            idle += max_batch - len(batch)
            busy += len(batch)
    ...
```

三种策略都在里面:static(等客+陪到终点)、continuous(逐位即补)、continuous+成组准入(攒 3 个空位或隔 8 步补一轮)。

---

## 4. 真实运行:三个场景

**闭环(60 请求全部 t=0 到达,队列压满——吞吐的净对比):**

```text
static              :makespan 47.2ms | 平均时延 23.7 | p99 46.2 | 批占用率 58%
continuous          :makespan 52.6ms | 平均时延 28.6 | p99 51.6 | 批占用率 92%
continuous(成组准入) :makespan 45.8ms | 平均时延 24.6 | p99 44.8 | 批占用率 84%
```

**轻载(40 请求,到达间隔 6ms——服务器等客):**

```text
static      :平均TTFT 21.7 | 平均时延 25.2 | p99 47.3
continuous  :平均TTFT  0.5 | 平均时延  4.0 | p99  6.3     ← 时延 6×,TTFT 43×
```

**重载(60 请求,到达间隔 1.5ms——排队压满):**

```text
static      :平均TTFT 5.8 | 平均时延 9.1 | p99 15.6
continuous  :平均TTFT 1.6 | 平均时延 5.8 | p99 10.4
```

---

## 5. 三个真实结论(比口号有用)

**① 时延是 continuous 的主场。**轻载下平均时延 25.2→4.0ms、TTFT 21.7→0.5ms:static 的「凑满一车」和「陪到终点」双重的等待被逐迭代调度消灭。重载下同样全赢(TTFT 5.8→1.6,p99 15.6→10.4)。

**② 吞吐没有免费的午餐——散装 prefill 的代价。**闭环里「逐位即补」的 continuous 占用率 92% 却比 static 慢(52.6 vs 47.2ms):**每补一位都要付一次串行 prefill**,散装插进 decode 流,反而吃掉快客攒下的时间。这正是生产引擎的痛点,两个工业解法:成组准入(同一轮并行 prefill 多位新客)与 **chunked prefill**(把长 prompt 切块,分多次混进 decode 流,不让任何一个请求的 prefill 独占机器)。

**③ 成组准入把两边都拿到。**攒位再补:占用率 84%(仍远高于 static 的 58%),makespan 45.8ms(三者最优)——**调度粒度本身就是超参数**:补得太勤 prefill 散装,补得太懒占用率漏走。

---

## 6. 生产账本

- **vLLM/SGLang/TensorRT-LLM 的默认都是 continuous batching**(vLLM 默认还开 chunked prefill);Orca(2021)是 iteration-level scheduling 的学术出处。
- **TTFT vs TPOT 的分账**(18 课的账本):TTFT ≈ 排队 + prefill;TPOT ≈ decode 步长 × 队列拥挤度。continuous 主要优化排队与占用;chunked prefill 用 TTFT 换 TPOT 的平稳。
- **批大小是吞吐-时延的旋钮**:批越大吞吐越高、单请求时延越差;生产按 SLO(如 p99 < X ms)反推最大批。
- **PD 分离**(18 课提过):本课的两副脾气放大到集群,就是 prefill 池与 decode 池分机器部署的趋势。

---

## 7. 记住的故事

**现实:** 「static 让快客陪慢客;continuous 每生成一个 token 都是一次重新发车的机会。」

**数学:** makespan ≈ Σ prefill(可并行) + 步数 × decode 步长;占用率 = busy/(busy+idle) 槽位步。

**工程:** 「占用率 92% 反而更慢:散装 prefill 的账要算——成组准入或 chunked prefill,调度粒度是超参数。」

**代码:** 第 3 节 simulate 约 60 行,三种策略一个函数。

---

## 作业

### A. 概念题

1. static batching 的两重等待分别是什么?continuous 各怎么消灭?
2. 为什么 decode 的批内边际成本近零,而 prefill 的并行以 max(prompt) 计费?
3. 「占用率 92% 的系统比 58% 的慢」——用散装 prefill 解释这个反直觉结果。
4. chunked prefill 用什么换什么?对 TTFT 和 TPOT 各是什么影响?

### B. 填空题

1. 实测:prefill 64 token ____ ms(____ µs/token),decode 单步 ____ ms,无 cache 重算浪费 ____×。
2. 轻载:平均时延 ____ → ____ ms(____×);TTFT ____ → ____ ms。
3. 闭环:static 占用率 ____%,continuous ____%,成组准入 ____% 且 makespan ____ ms 最优。
4. 三策略名:____ / ____ / ____;调度单位分别是 批 / ____ / ____。

### C. 代码手撕(关讲义)

1. 默写 simulate 的 continuous 分支(到达入队、走客下车、空位即补、成组准入)。
2. 给 simulate 加一个 `slo` 参数:时延超标的请求计入违约率,画批大小 vs 违约率曲线。
3. 把到达过程从均匀改成泊松(指数间隔),重跑轻载场景,结论变吗?

### D. 生产环境实战

1. 容量题:7B 模型 bf16、A100-80G,decode 每 token 需读 14GB 权重+KV;HBM 带宽 2TB/s,单请求 TPOT 下限多少?批 32 时呢(带宽被 32 个请求摊用)?
2. 事故题:上线 continuous batching 后 p99 时延反而恶化。列出三个排查方向(准入粒度?chunked prefill?批大小上限?)。
3. 查证:Orca(2021)的 iteration-level scheduling 一句话;vLLM 的 chunked prefill 默认参数?

### E. 拓展

1. 实验:max_batch ∈ {2,4,8,16,32} 扫描闭环 makespan 与平均时延,找吞吐-时延的帕累托前沿(ipynb TODO)。
2. 预习钩子:请求随到随走了,但每个请求的 KV cache 还是一整条**连续内存**——8K 上下文的请求先占 8K,写一半取消,碎片谁来管?下一课:PagedAttention,把 KV cache 切成块,像操作系统分页一样管理。

---

# 下一课预告:PagedAttention

KV cache 的显存账(18 课)只算了「要多少」,没算「怎么放」。连续内存预留 = 内部碎片 + 外部碎片 + 抵制共享;分页管理 = 逻辑块表 + 按需分配 + 前缀共享。vLLM 靠它把显存利用率翻倍——下一课手撕 block table 与分页分配器,实测碎片率对比,并把前缀缓存(prefix caching)一并拿下。

> **下一课:[PagedAttention——像操作系统一样管理 KV](30-paged-attention.md)——块表一立,碎片归零。**

## 参考来源

**本课实测**:prefill/decode 成本来自 MiniGPT 本机测量;三场景数字来自手撕模拟器(成本模型 §2,种子固定可复现)。

**文献**:

- [Yu et al. 2022, Orca: A Distributed Serving System (OSDI'22)](https://arxiv.org/abs/2208.04870) — iteration-level scheduling / continuous batching 学术出处
- [Kwon et al. 2023, Efficient Memory Management for LLM Serving with PagedAttention (vLLM, SOSP'23)](https://arxiv.org/abs/2309.06180) — 下一课主角,本课的工业容器
