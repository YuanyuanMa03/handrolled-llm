# 第三十三课(毕业设计 II):纯 numpy 跑真模型

> 手撕课程的终点:把 22–32 课的全部零件装到**一台真实模型**上——本地 SmolLM2-135M-Instruct,1.345 亿参数,safetensors 逐张量加载,bf16→fp32 位运算,**一个 27 层 Llama 架构前向 + KV cache + BPE 分词,全部纯 numpy,真实生成英文**。贪心 9.5 tok/s,chat 模板对答,采样版甚至能蹦出「風格」——每行输出都出自本机真实运行。开发路上还有一个大陷阱(RoPE 的头内配对)被 MLX 交叉验证当场按住,战争故事 §5。
> **本课数字**:全部实读本地模型文件并真实运行;开发中用 MLX(已卸载)做了一次 logits 交叉验证。

---

## 1. 任务书:零件清单

| 零件 | 出自 | 在真模型里 |
|---|---|---|
| safetensors 读字节 + bf16→fp32 | 31 课 | 272 张量逐个加载 |
| byte-level BPE | 22 课 | 它的 vocab(49,152)/merges(48,900)与 `Ġ` 约定 |
| KV cache(18 课) | 18 课 | 逐 token 生成,`max|diff| = 8.7e-8` 对账 |
| RMSNorm / SwiGLU / GQA / RoPE | 5–8、19、20 课 | Llama 架构的标准件,一处细节差异见 §5 |
| 模型解剖(config/张量表) | 21 课 | hidden 576、30 层、9Q/3KV 头(GQA 3:1)、词表 49,152、**词嵌入绑定**(31 课发现) |

## 2. 加载:0.1 秒,5 亿字节

```text
272 张量 / 134,515,008 参数(fp32 0.50 GB),加载 0.1 s
```

31 课的加载器直接复用。注意 **lm_head 不存在**——词嵌入绑定,输出层就是 `embed_tokens` 的转置(代码里一行:`logits = x @ W["model.embed_tokens.weight"].T`)。

## 3. 分词器:22 课的思想 + 它的词表

BPE 的 encode(rank 贪心合并)、byte→unicode 映射、往返无损——22 课手撕过的每一件,现在接上真词表。唯一新依赖是 `regex`(纯 Python):GPT-2 的预切分正则要 `\p{L}` 类,标准库 `re` 不支持。真实运行:

```text
encode('The Journey to the West') = 5 tok:['The','ĠJourney','Ġto','Ġthe','ĠWest'] | 往返无损 ✓
encode('孙悟空 has 72 transformations!') = 14 tok:['åŃ','Ļ','æ','Ĥ',…]        | 往返无损 ✓
                                        ↑ 中文没有现成 token,退回 UTF-8 字节级——22 课的「永无 OOV」照进现实
```

## 4. 前向:27 层标准件(与 20 课零件对照)

每层与 MiniGPT 的差异逐件可数:RMSNorm 带 gamma(20 课没有)、RoPE 的 theta=100,000(20 课 10,000)、GQA 3:1(20 课满配 MHA)、SwiGLU 相同、词嵌入绑定(20 课不绑定)。生成实测:

```text
[贪心 30 tok,9.5 tok/s]
The Journey to the West tells the story of a young man named Kao Kao, who is
sent to the West Coast of America to escape the corrupt and evil-doers of …
                      ↑ 1.35 亿参数的知识水平:语法流畅,事实自由发挥——137M 的诚实样子

[chat 模板问答,T=0.6,60 tok](官方 chat_template 逐字拼接)
The journey to the West of the Yakuza in the 19th century was a complex,
clandestine, and treacherous path that involved a combination of cultural
exchange, corruption, and outright violence.
                      ↑ 真正的 instruct 行为:结构、口吻全对,内容是 135M 级的虚构

[T=0.8 采样]
… the Biblical adventurer, 風格, who has three brothers: Osiris, Caiban, and …
                      ↑ 采样放开了想象力,还蹦出日文汉字——byte-level BPE 无所不能

[18 课验收] KV cache 版 vs 全量重算 max|diff| = 8.74e-08  ✓
[prefill] 9 token,134 ms;下一 token 前5:['Ġa','Ġthe','Ġan','ĠJack','Ġtwo']
```

## 5. 战争故事:RoPE 的头内配对(本课差点翻车的地方)

第一版前向「看起来能跑」:9-token 提示的 top-5 像模像样,贪心也能输出半通顺英文——但 chat 模板下全是空白。请 MLX(同一模型、同一提示)当裁判:**top-1 从第 9 个 token 起就不一致**。

病根在 RoPE 的**配对维度**:半分配对 `(i, i+half)` 我在**展平的 [T, 576]** 上做——第 0 维与第 288 维配对,而 288 已在**别的头**里!真模型的约定是**每个头内部**前 32 维与后 32 维配对。修法一行:先 reshape 成 `[T, 头数, 64]` 再旋转。修复后三种长度 top-3 与 MLX **完全一致**。

三个教训:

- 20 课的 MiniGPT 在展平维度上做 RoPE 却一切正常——因为它**自己的权重就在这个约定下训练**,自洽即合法;换用别人的权重,约定必须跟着别人走。**「数学正确」之上还有「约定正确」**;
- 症状会骗人:短提示的 top-5 「像样」不代表前向正确——分布被扭曲后,高频 token 仍常排前;必须逐 token 对账;
- chat 全空白差点被误诊为「模型太弱」——先找独立裁判(MLX)对 logits,再谈模型。

另一个小陷阱:chat 模板必须**逐字**按官方 jinja 拼——首轮系统提示词(`You are a helpful AI assistant named SmolLM…`)、每个 `<|im_end|>` 后的换行,少一个都从「对答如流」变「白纸一张」(25 课模板铁律的现场版)。

---

## 6. 生产账本:手撕版与工业引擎的差距

| | 本课 numpy | vLLM / MLX |
|---|---|---|
| 算力 | CPU 单线程,9.5 tok/s | GPU/Metal,百倍以上 |
| 内存 | fp32 0.50 GB | bf16/int4(31 课),零拷贝 mmap |
| 调度 | 单请求串行 | 连续批处理 + 分页 KV(29/30 课) |

手撕的意义不在性能,在于**每一行都可解释**:面试里「GQA 怎么算」「RoPE 怎么转」「bf16 怎么读」——你全部亲手写过、跑过、对过账。

## 7. 记住的故事

**现实:** 「1.345 亿参数从磁盘字节到英文句子,零框架依赖——22–32 课的零件一张清单全数到场。」

**数学:** 每头内部的旋转配对;θ=10⁵ 的低频旋转;词嵌入绑定省下的 27M 参数。

**工程:** 「数学正确之上还有约定正确:RoPE 配对跨了头,短提示照样『像样』——独立裁判对 logits,别信症状。」

**代码:** 加载器 20 行 + 分词器 40 行 + 前向 50 行——全课程零件的第一次合体点火。

---

## 作业

### A. 概念题

1. 词嵌入绑定省了什么?绑定与不绑定各有什么权衡?
2. GQA 3:1 在本课代码里对应哪几行?相比满配 MHA 省了多少 K/V 显存(算 2K 上下文的账)?
3. RoPE 的 theta 从 10⁴(玩具)到 10⁵(真模型)变大了,旋转频率如何变、为什么长上下文模型倾向更大的 theta?
4. 为什么短提示 top-5「像样」不能证明前向正确?

### B. 填空题

1. 本课加载 ____ 张量 / ____ 参数(fp32 ____ GB),贪心 ____ tok/s。
2. 分词:中文「孙悟空」编码成 ____ 级 token;`Ġ` 代表 ____。
3. 两场战争故事:RoPE ____;chat 模板缺 ____ 与 ____ 导致空白。
4. KV cache 等价性验收:max|diff| = ____。

### C. 代码手撕(关讲义)

1. 默写 safetensors 逐张量加载(含 bf16 位运算),并打印任意一层的形状表。
2. 默写 Llama 注意力块(含 GQA 广播与每头 RoPE),用 MLX 或任何独立实现对账 top-3。
3. 把 31 课的 int4 量化接进本课:加载→量化→反量化→生成,对比输出质量与 tok/s。

### D. 生产环境实战

1. 容量题:本模型 fp32 0.50 GB + 2K 上下文 KV(30 课算过 0.04 GB):单请求总内存?int4 权重后?并发 16 条呢?
2. 事故题:线上把 SmolLM2 换成 Qwen3-4B,前向报形状错。列出你要对账的 config 字段清单(head 数、GQA、theta、绑定与否……)。
3. 查证:safetensors 格式规范(100 字内说清头部结构);mmap 零拷贝加载为什么快?

### E. 拓展(课程终章)

1. 实验:theta ∈ {10⁴, 10⁵, 10⁶} 对长上下文(1K+)生成的影响——位置外推的行为观察(ipynb TODO)。
2. **课程终章**。两卷 33 课,从 softmax 到真模型点火,全部手撕、全部真实运行。收官作业只有一条:**把两卷的 D 类题重做一遍**——现在的你,和第 1 课的你,已经不是同一个工程师。下次见面,应该在 mimi 的 agent 工程里——那是另一卷的故事。

---

## 参考来源

**本课实测**:272 张量、全部生成输出、速度与等价性数字均来自本机真实运行;开发期用 MLX(同权重同提示)做 logits 交叉验证(top-3 逐长度一致后卸载,不进课程依赖)。

**文献**:

- [Allal et al. 2024, SmolLM2 (arXiv:2502.02737)](https://arxiv.org/abs/2502.02737) — 本课真模型的模型卡/技术报告
- [HF safetensors 格式](https://github.com/huggingface/safetensors) — 8 字节头长 + JSON 张量表 + 裸字节
- [Su et al. 2021, RoFormer/RoPE (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — 旋转位置编码(19 课已引,此处是「每头内配对」的实现约定出处)
