# 第二十一课(案例):解剖一台真实的大语言模型

> 毕业设计(第 20 课)是 5 万参数的麻雀。这一课,我们把手术刀对准**一台 40 亿参数的真实生产模型**:
> **Qwen3-4B-Instruct-2507**(bf16,7.5GB,就在你磁盘 `/Volumes/Sam-APFS/huggingface/hub/` 里),用它的**真实文件**做逐字段、逐张量的解剖,再对照 DeepSeek-R1(671B)看工业前沿。
>
> **本课所有数字均来自本机真实文件的读取**(config.json、safetensors 张量表、tokenizer.json、generation_config.json),不是文档抄写。

---

## 1. 一台生产模型 = 几个 JSON + 权重文件

快照目录真实内容(`/Volumes/Sam-APFS/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-bf16/snapshots/f9e77d42…/`):

```text
chat_template.jinja              聊天模板(角色标记怎么拼)
config.json                      模型结构说明书        ← 本课主角
generation_config.json           采样默认参数          ← 14-17 课全家福
model-0000X-of-00002.safetensors 权重本体(2 个分片)
model.safetensors.index.json     分片索引(哪个张量在哪个文件)
tokenizer.json                   分词器
```

**没有魔法**。部署一个开源模型,就是下载这几个文件、按 config 搭积木、把权重灌进去——积木你全部学过。

---

## 2. config.json 逐字段对课(真实值)

```json
{"architectures": ["Qwen3ForCausalLM"],
 "hidden_size": 2560, "num_hidden_layers": 36,
 "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
 "intermediate_size": 9728, "vocab_size": 151936,
 "max_position_embeddings": 262144, "rope_theta": 5000000,
 "rms_norm_eps": 1e-06, "tie_word_embeddings": true,
 "hidden_act": "silu", "torch_dtype": "bfloat16"}
```

| 真实字段 | 真实值 | 对应课程 | 读出来的事实 |
|---|---|---|---|
| `hidden_size` | 2560 | 第2课 | 每个词一个 2560 维向量 |
| `num_hidden_layers` | 36 | 第9课 | 36 块积木堆叠 |
| `num_attention_heads` | 32 | 第7课 | 32 个注意力头 |
| `num_key_value_heads` | 8 | 第7课 | **GQA!KV 头只有 8,Q 头 32 → 4:1 共享** |
| `head_dim` | 128 | 第7课 | 每头 128 维 |
| `intermediate_size` | 9728 | 第10课 | FFN 膨胀到 9728 ≈ 3.8 倍 |
| `hidden_act` | silu | 第10课 | SwiGLU 的门控激活 |
| `rms_norm_eps` | 1e-06 | 第8课 | RMSNorm 的 ε(生产用 1e-6) |
| `rope_theta` | 5000000 | 第19课 | RoPE 基=500 万(原始 1 万;**为 256K 长上下文调大**) |
| `max_position_embeddings` | 262144 | 第19课 | 256K 上下文 |
| `vocab_size` | 151936 | 第2课 | 词表 15.2 万 |
| `tie_word_embeddings` | true | 第2/11课 | 输入嵌入 = 输出头,**省一份 151936×2560** |

**一页 JSON,十二个字段,课课都学过。** 注意 `rope_theta=5,000,000` 和第 19 课讲的"外推靠调 θ"——这台 4B 模型正是靠调大 RoPE 基换来了 256K 上下文,课程理论在生产文件里的直接落地。

---

## 3. safetensors 解剖:不加载权重,读张量表

工业技巧(本课 ipynb 现场实现):safetensors 文件头 8 字节存头部长度,后面跟着一段 JSON,记录**每一个张量的名字、形状、 dtype**——7.5GB 的模型,读几百字节就能拿到全部解剖图。

第 0 层的真实张量(共 398 个张量,此处为其中一层):

```text
model.layers.0.input_layernorm.weight            [2560]         ← 第8课 RMSNorm(只有 γ)
model.layers.0.post_attention_layernorm.weight   [2560]         ← 第9课 两个 LN
model.layers.0.self_attn.q_proj.weight           [4096, 2560]   ← Q 投影:32头×128维=4096!
model.layers.0.self_attn.k_proj.weight           [1024, 2560]   ← K 投影:8头×128维=1024 ← GQA 实锤
model.layers.0.self_attn.v_proj.weight           [1024, 2560]   ← V 投影:同 8 头
model.layers.0.self_attn.o_proj.weight           [2560, 4096]   ← 输出投影
model.layers.0.self_attn.q_norm.weight           [128]          ← 每头 Q 的 RMSNorm(QK-norm)
model.layers.0.self_attn.k_norm.weight           [128]          ← 每头 K 的 RMSNorm
model.layers.0.mlp.gate_proj.weight              [9728, 2560]   ← SwiGLU 门控支(W1)
model.layers.0.mlp.up_proj.weight                [9728, 2560]   ← SwiGLU 内容支(W3)
model.layers.0.mlp.down_proj.weight              [2560, 9728]   ← 压回(W2)
```

三个真实细节,教科书上很少讲:

1. **q_proj 是 [4096, 2560]——比 hidden 大!** 头维×头数(32×128=4096)**可以超过** hidden_size。第 7 课"d=d_h×H"只是简化,Qwen3 用"超宽头"换注意力容量。
2. **k_proj/v_proj 只有 [1024, 2560]**——GQA 不是嘴上说说,张量形状就是省下的参数和显存。
3. **q_norm/k_norm [128]**:对每个头的 Q、K 各做一次 RMSNorm(第 8 课的 per-head 版),稳定注意力打分的尺度。

---

## 4. 参数总账:与真实张量逐个核对

用第 2/7/10 课的公式分账,和"逐张量乘积求和"对账(**分毫不差**):

```text
词嵌入        151936 × 2560                =   388,956,160  (tie=True,lm_head 不另算)
每层注意力    q 4096×2560 + k 1024×2560
              + v 1024×2560 + o 2560×4096 =   26,214,400
每层 FFN      3 × 2560 × 9728              =   74,711,040
每层归一化    2×2560 + 2×128               =        5,376
36 层小计                                   = 3,633,315,840 + 193,536(层内 norm)
最终 norm                                  =         2,560
─────────────────────────────────────────
分账合计                                    = 4,022,468,096
逐张量求和(398 个张量)                     = 4,022,468,096   ✓ 完全一致
```

"4B"的名字名实相符。bf16 下 4.02e9 × 2 字节 = **7.49 GB**,和目录大小对上。

**这道对账题就是面试的"给你 config 算参数量"原题**——你现在会从两个方向算,还能互相对账。

---

## 5. generation_config.json:14–17 课的全家福

真实文件原文:

```json
{"do_sample": true, "temperature": 0.7, "top_k": 20, "top_p": 0.8,
 "eos_token_id": [151645, 151643], "pad_token_id": 151643}
```

四个采样课的参数在**同一个生产文件**里各就各位:T=0.7(第15课)、top_k=20(第16课)、top_p=0.8(第17课)、do_sample(第14课)。对照:DeepSeek-R1 模型卡建议 temperature 0.5–0.7、top_p 0.95——**推理模型的温度窗口更窄**(RL 把分布压尖了,第 15 课讲过)。

---

## 6. 真实显存账(第 18 课公式 × 真机参数)

$$
bytes = 2 \times 36_{layers} \times 8_{kv\ heads} \times 128_{d_h} \times seq \times 2_{bf16}
$$

```text
8K 上下文,  batch=1: 1.12 GB
256K 上下文,batch=1: 36.00 GB     ← 是模型本体的 4.8 倍!
```

现在你彻底理解:为什么 GQA(8 头而非 32 头)是刚需、为什么长上下文服务要 PagedAttention、为什么第 18 课说"缓存比模型贵"。

---

## 7. tokenizer.json:词表与 `<think>`

真实数据:

```text
基础词表(BPE 合并表): 151,643 条
特殊 token(added):    26 条
config.vocab_size:     151,936  = 对齐到 128 倍数的容量(硬件友好)

<|endoftext|>  id=151643     <|im_start|>  id=151644     <|im_end|>  id=151645
<think>        id=151667     </think>      id=151668
'attention' → id 53103(一个词一个 id,第 2 课查表)
```

`<think>`/`</think>` 真实存在于词表里——这就是 R1 式"显式思维链"的落点:推理模型把"思考过程"当成**普通 token 流**生成,chat_template 控制 `<think>` 何时出现,服务器按第 14 课的采样循环逐 token 吐出。**推理范式 = 你学的采样循环 + 两个特殊 token。**

---

## 8. 对照:同一血统的三台机器

| | MiniGPT(第20课) | **Qwen3-4B(本课实测)** | DeepSeek-R1(论文) |
|---|---|---|---|
| 层数 | 2 | 36 | 61 |
| hidden | 32 | 2560 | 7168 |
| 注意力 | 标准多头 | GQA 32Q/8KV | **MLA**(潜在压缩) |
| FFN | SwiGLU | SwiGLU 9728 | SwiGLU + **MoE**(256+1 专家) |
| 归一化 | RMSNorm | RMSNorm(+QK-norm) | RMSNorm |
| 位置 | RoPE | RoPE θ=5e6, 256K | RoPE(融入 MLA) |
| 参数 | 3.4 万 | **4.02B(逐张量对账)** | 671B / 激活 37B |
| 训练 | 未训练 | 预训练+SFT+RL | V3 底座 + **GRPO** 四阶段 |

R1 的三个新构件一句话版(详见文末来源):

- **MLA**:所有头共享一个低维潜向量,推理时才还原 K/V——KV Cache 压到接近"1 头"的量级,还不掉多头表达力。GQA 是砍头,MLA 是压扁再展开。
- **DeepSeekMoE**:每层 1 个共享专家 + 256 个路由专家,每 token 只激活 top-8——671B 存量、37B 算量,"大而不贵"。
- **GRPO**:强化学习里用"同题采一组、组内减均值当基线"替掉价值网络——规则可判的奖励(答案对错 + 格式)压出推理能力,R1-Zero 证明甚至不需要 SFT 冷启动也能涌现思维链。

**从右往左看这张表:每一列你都已经能读懂结构;差别只剩规模与三个工程发明。**

---

## 9. 作业

### A. 概念题

1. `num_attention_heads=32, num_key_value_heads=8` 各控制什么?为什么 KV 头可以少?
2. `q_proj` 形状 [4096, 2560] 说明这台机器的哪一点和第 7 课的简化不同?
3. `rope_theta=5,000,000`(对比原始 10,000)是为了什么?对应第 19 课哪个知识点?
4. `tie_word_embeddings=true` 省了多少参数?(算出来)
5. MLA、MoE、GRPO 各解决什么问题?一句话一个。

### B. 填空题

1. Qwen3-4B:hidden=____、____ 层、____Q/____KV 头、head_dim=____。
2. safetensors 文件头:前 ____ 字节是头部长度,随后是 ____ 格式的张量表。
3. 逐张量对账总参数 = ____(4.02B 精确到个位;作业里自己算一遍)。
4. KV Cache @256K、batch=1、bf16 = ____ GB(用公式)。
5. `<think>` 的 token id = ____。

### C. 代码手撕(本课 ipynb 现场即答案)

1. 写 `read_safetensors_header(path)`:返回张量名字典(名字 → (shape, dtype))。
2. 写 `count_params(config)`:纯从 config.json 算总参数,与逐张量求和对账(差值应恰好是各 norm 的几十万)。
3. 写 `kv_cache_gb(config, seq, batch, bytes_per=2)`。
4. 遍历本地 `models--*` 目录,打印每台模型的 (层数, hidden, Q/KV 头) 对照表。

### D. 生产环境实战

1. 同目录还有 Qwen3-1.7B、Qwen2.5-1.5B、Llama-3.2-1B:分别读 config,做本课 §4 的参数对账,验证与模型名(1.7B/1.5B/1B)是否相符。
2. 这台 4B 模型要在 16GB 内存的本机做 256K 上下文推理:权重 + KV Cache 共多少?给出量化(AWQ/INT4)后的方案估算。
3. `eos_token_id` 是一个**列表** [151645, 151643]:解码循环要怎么改?(第 14 课 generate 函数的多终止条件。)
4. 查证:QK-norm(q_norm/k_norm)解决什么训练不稳定?哪几家模型在用?

### E. 毕业拓展

1. 用 mlx-lm 或 llama.cpp 实际加载这台模型生成一段话,把 `temperature/top_k/top_p` 调成 0.1 / 5 / 0.5,观察第 15–17 课理论的真实表现。
2. 读 `chat_template.jinja`,画出一个真实请求的 prompt 拼装流程(system/user/assistant 标记)。
3. 全课程重做一遍 D 类题——现在的你和第一课的你,不是同一个工程师。

---

## 10. 课程终章

```text
config 的每个字段        → 你在第 2/7/8/9/10/19 课写过它的原理
safetensors 的每行张量   → 你在第 20 课亲手搭过它的同类
generation_config       → 你在第 14-17 课手撕过每个参数
KV Cache 显存账          → 你在第 18 课推导过公式
<think> token            → 你在第 14 课的采样循环里见过它的生成方式
R1 的 MLA/MoE/GRPO       → 是你已学构件的工程进化版
```

> **Agency 是训练出来的;理解,是把大机器拆到每一颗螺丝都眼熟。**

*(第一卷至此完结。第二卷 22–33 课——把第 20 课的麻雀训活(训练篇 II),再把它伺候好(推理工程篇),最后纯 numpy 点火真实模型——目录见 [README](README.md)。)*

---

## 参考来源

**本课实测数据来源(本机文件)**:
- `models--mlx-community--Qwen3-4B-Instruct-2507-bf16` 的 config.json / 两片 safetensors 头 / tokenizer.json / generation_config.json(HuggingFace 本地缓存,快照 f9e77d42)

**DeepSeek-R1 部分来源(文献)**:
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/html/2412.19437v1) — MLA/MoE/FP8,671B/37B
- [DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO、四阶段、R1-Zero
- [deepseek-ai/DeepSeek-V3 (GitHub)](https://github.com/deepseek-ai/deepseek-v3) · [R1 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1) · [Qwen3-4B-Instruct-2507 模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
