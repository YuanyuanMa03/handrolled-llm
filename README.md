# 文科生版 LLM 算法面试课

> **目标:就业。** 按 LLM 内部真实结构学习,每课配五类作业——概念、填空、手撕、生产环境实战、拓展。
> 所有数字与代码输出均来自真实运行;纯 numpy/纯 Python 手撕,不依赖 torch。

方法论:**现实问题 → 数据 → 数学 → 算法 → 代码**。先懂"它解决什么",再学公式,最后翻译成代码。

## 课程目录(两卷 33 课,全部完成 ✅)

### 第一卷:零件与组装(01–21)

| # | 课 | 文件 |
|---|---|---|
| 01 | Softmax | [md](01-softmax.md) · [ipynb](01-softmax.ipynb) |
| 02 | 向量 / 矩阵 | [md](02-vectors.md) · [ipynb](02-vectors.ipynb) |
| 03 | 点积 | [md](03-dot-product.md) · [ipynb](03-dot-product.ipynb) |
| 04 | Cosine Similarity | [md](04-cosine.md) · [ipynb](04-cosine.ipynb) |
| 05 | Attention(一般形式) | [md](05-attention.md) · [ipynb](05-attention.ipynb) |
| 06 | Self-Attention 手撕 | [md](06-self-attention.md) · [ipynb](06-self-attention.ipynb) |
| 07 | Multi-Head Attention | [md](07-multi-head.md) · [ipynb](07-multi-head.ipynb) |
| 08 | LayerNorm | [md](08-layernorm.md) · [ipynb](08-layernorm.ipynb) |
| 09 | Transformer Block | [md](09-transformer-block.md) · [ipynb](09-transformer-block.ipynb) |
| 10 | MLP / FFN | [md](10-ffn.md) · [ipynb](10-ffn.ipynb) |
| 11 | Cross Entropy | [md](11-cross-entropy.md) · [ipynb](11-cross-entropy.ipynb) |
| 12 | Backpropagation | [md](12-backprop.md) · [ipynb](12-backprop.ipynb) |
| 13 | SGD / Adam | [md](13-optimizer.md) · [ipynb](13-optimizer.ipynb) |
| 14 | Token Sampling | [md](14-sampling.md) · [ipynb](14-sampling.ipynb) |
| 15 | Temperature | [md](15-temperature.md) · [ipynb](15-temperature.ipynb) |
| 16 | Top-K | [md](16-top-k.md) · [ipynb](16-top-k.ipynb) |
| 17 | Top-P | [md](17-top-p.md) · [ipynb](17-top-p.ipynb) |
| 18 | KV Cache | [md](18-kv-cache.md) · [ipynb](18-kv-cache.ipynb) |
| 19 | RoPE | [md](19-rope.md) · [ipynb](19-rope.ipynb) |
| 20 | Mini Transformer(总装) | [md](20-mini-gpt.md) · [ipynb](20-mini-gpt.ipynb) |
| 21 | 案例:解剖真实大模型(Qwen3-4B 实测 + DeepSeek-R1 对照) | [md](21-real-llm-anatomy.md) · [ipynb](21-real-llm-anatomy.ipynb) |

### 第二卷:训练篇 II + 推理工程(22–33)

| # | 课 | 文件 |
|---|---|---|
| 22 | BPE 分词器手撕(《西游记》自训词表 + Qwen 对照) | [md](22-bpe.md) · [ipynb](22-bpe.ipynb) |
| 23 | Mini-GPT 反向传播手撕(33,792 坐标数值验收) | [md](23-backward.md) · [ipynb](23-backward.ipynb) |
| 24 | 训练循环:真训《西游记》(val 8.76 → 5.33) | [md](24-train-loop.md) · [ipynb](24-train-loop.ipynb) |
| 25 | SFT:教模型听指令(0/50 → 50/50 + 灾难性遗忘) | [md](25-sft.md) · [ipynb](25-sft.ipynb) |
| 26 | LoRA:低秩补丁(1.70% 参数 + merge) | [md](26-lora.md) · [ipynb](26-lora.ipynb) |
| 27 | DPO:偏好直接优化(闭式解 + 符号战争) | [md](27-dpo.md) · [ipynb](27-dpo.ipynb) |
| 28 | GRPO:组内基线强化学习(训练篇终章) | [md](28-grpo.md) · [ipynb](28-grpo.ipynb) |
| 29 | 连续批处理:推理服务的吞吐引擎 | [md](29-continuous-batching.md) · [ipynb](29-continuous-batching.ipynb) |
| 30 | PagedAttention:像操作系统一样管理 KV | [md](30-paged-attention.md) · [ipynb](30-paged-attention.ipynb) |
| 31 | 量化:真实 SmolLM2-135M 权重逐张量实测 | [md](31-quantization.md) · [ipynb](31-quantization.ipynb) |
| 32 | 投机解码:小模型代笔,大模型签名 | [md](32-speculative.md) · [ipynb](32-speculative.ipynb) |
| 33 | 毕业设计 II:纯 numpy 跑真模型 SmolLM2-135M | [md](33-numpy-llm.md) · [ipynb](33-numpy-llm.ipynb) |

第二卷主线:**训练篇 II(22–28)** 把第 20 课的麻雀训活——分词、反向、真训、SFT、LoRA、DPO、GRPO 七步走完,终章结论「后训练放大既有能力,不创造知识」;**推理工程篇(29–33)** 把它伺候好——批处理、分页 KV、量化、投机解码,最后纯 numpy 点火真实模型。每课的真实运行数字、战争故事与验收记录都写在讲义里。

## 怎么用

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy regex opencc-python-reimplemented   # 全部纯 Python;后两者:33 课 BPE 预切分 / 语料构建
jupyter lab .               # 跑 notebook(可选)
```

- **md** = 讲义(先读);**ipynb** = 代码现场(边跑边学);**作业** = 每课 md 末尾,五类:
  - **A 概念题**——面试"讲清楚"的能力
  - **B 填空题**——关键常数和结论
  - **C 代码手撕**——关掉讲义白板复现
  - **D 生产环境实战**——显存、延迟、NaN、shape 等真问题
  - **E 拓展**——下一课/前沿的钩子
- 顺序不能乱:每课只依赖前面已学的零件。
- 第二卷的实验语料:`data/xiyouji.txt`——《西游记》全本(维基文库公版,繁转简,723,587 字),训练篇各课确定性重训复用。
- 31/33 课读本地 HuggingFace 缓存的真实模型权重(SmolLM2-135M-Instruct / Qwen3-4B),路径写死在 notebook 顶部,换机器改路径即可。

## 全部课程的一句话总结

> **第一卷拼零件,第二卷开机器:从 softmax 到纯 numpy 点火 1.345 亿参数的真模型——中间的每一步,都是你亲手写的。**
