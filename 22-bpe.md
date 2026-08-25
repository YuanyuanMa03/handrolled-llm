# 第二十二课:BPE 分词器——模型读到的不是字,是 token

> 第 2 课的 lookup 表(`小猫→0`)留了个坑:表里没有的词怎么办?本课把坑填死:**byte-level BPE**——GPT、Llama、Qwen 全在这一套上。跑完本课,你将拥有一个自己训出来的分词器:它在《西游记》上学会把「行者」「八戒」铸成单个 token。
> **本课所有数字均来自本机真实运行**:语料 `data/xiyouji.txt`(723,587 字,维基文库公版);对照数据实读本地 Qwen3-4B 的 `tokenizer.json`。

---

## 0. 第二卷开场:把第 20 课的麻雀训活

第一卷(01–21 课)拼好了全部零件:attention、Norm、FFN、loss、优化器、采样,KV cache,最后在 20 课总装出一台 3.4 万参数的 Mini-GPT——但它是**未训练**的,20 课结尾说:训练是「训练篇的内容」。第二卷就从这里接:

- **22(本课)**:模型吃什么——把《西游记》变成 id 序列;
- **23**:梯度谁来传——Mini-GPT 反向传播手撕;
- **24**:真的训——loss 压到瞎猜基线以下;
- **25–28**:SFT → LoRA → DPO → GRPO,后训练全家福;
- **29–33**:推理工程——批处理、PagedAttention、量化、投机解码、纯 numpy 跑真模型。

方法论不变:**现实问题 → 数据 → 数学 → 算法 → 代码**。

---

## 1. 先不用代码——排版工铸铅字

把分词器想象成一个活字印刷的排版工。他手里有两种字模:

- **单字模**:「行」「者」——什么文章都能排,但一行「行者行者行者」要取六次模;
- **整词模**:「行者」一块整模——取一次,顶两次。

排版工的策略非常朴素:**哪种搭配出现得多,就把它铸成一块整模**。「行者」在《西游记》里出现几千次,值得铸;「懒」只出现 61 次,不值得,继续用字节拼。铸多少块模,就是词表大小 V。

在铸模之前,先看三个候选方案的**真实数据**(《西游记》实测):

| 方案 | 唯一符号数 | 全文序列长度 | 问题 |
|---|---|---|---|
| 词级(整词一个 id) | 数万且无上限 | 最短 | 白话文没有空格,**词边界本身就是难题**;新词直接 OOV |
| 字级(一个汉字一个 id) | 4,551 | 723,587 token | 没有 OOV,但序列最长,attention 平方代价(18 课) |
| **BPE(子词)** | 自定(512…15 万) | 两者之间 | 频次铸模:高频词短 id、低频字用字节拼 |

**OOV(out-of-vocabulary)是词级方案的死穴**:推理时来了个训练从没见过的词,lookup 表查不到,模型当场失明。而 BPE 的兜底优雅到犯规:**任何字符串最后都能落到字节上**——字节只有 256 种,永无 OOV。

---

## 2. 数学:整个算法只有三条

**编码的原子——UTF-8 字节。** 一个汉字 = 3 字节。任意文本 → 字节串 → 256 种字节值,这是天然字符集:

$$\text{text} \xrightarrow{\text{UTF-8}} \{b_1, b_2, \dots\},\quad b_i \in \{0,\dots,255\}$$

**BPE 训练——频次合并。** 初始符号 = 单字节。每一轮,统计全语料所有**相邻符号对**的频次,把最高频的一对 $(a,b)$ 合并成新符号 $ab$,写进 merges 表;重复直到词表满:

$$\text{merge}_i = \arg\max_{(a,b)} \ \text{count}(a,b), \qquad i = 1, 2, \dots, V - 256$$

**压缩率。** 一句话的字节数除以 token 数,衡量「一块铅字平均顶几个字节」:

$$\text{compression} = \frac{\#\text{bytes}}{\#\text{tokens}}$$

人话版:

- **字节 = 油墨**,任何字都能印,永不断货(无 OOV);
- **合并 = 铸模**,频次说了算,语料里的世界决定铸什么;
- **rank = 铸模顺序**,编码时按这个顺序贪心,结果才和词表对得上。

---

## 3. byte→unicode:GPT-2 的小机关

有个工程细节:字节值里有一大半是控制字符/不可打印字符(换行、`\x00`…),直接当符号放进词表 JSON 既难看又容易出事。GPT-2 的办法:把 256 个字节**映射到 256 个可打印的 unicode 字符**上,纯显示用,底层永远是字节:

```python
def bytes_to_unicode():
    """可打印字节原样保留;其余映射到 256+ 的码位——避开控制字符,词表可存 JSON。"""
    bs = (list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

B2U = bytes_to_unicode()          # 字节 -> 可读符号
```

于是「，」的三个字节 `EF BC 8C` 显示为 `ï¼Ī`——难看,但可打印、可逆。空格变成 `Ġ`,你在 GPT 词表里见过的 `Ġthe` 就是这么来的。

---

## 4. 手撕训练:增量式 BPE

朴素实现:每合并一次,全语料重新数一遍所有相邻对。复杂度 $O(V \times N)$——我们 $N=210$ 万字节、$V=2048$,就是 **43 亿次**操作,纯 Python 要跑一个世纪。生产做法是**增量**:维护一张「对 → 频次」的账本和一张「对 → 含它的词」的索引,每次合并**只重扫受影响的词**,把旧对的账撤掉、新对的账记上:

```python
import re
from collections import Counter, defaultdict

# 预切分:汉字串 / 拉丁字母数字串 / 空白串 / 其他单字符(切词边界,合并不越界)
PRE_TOK = re.compile(r"[\u3400-\u9fff]+|[0-9A-Za-z]+|\s+|\S")


def train_bpe(text, vocab_size):
    """返回 merges:第 i 项是第 i 次合并的 (a, b)。
    id 约定:0..255 = 单字节,256+i = 第 i 次合并的产物。"""
    assert vocab_size >= 256
    word_counts = Counter(PRE_TOK.findall(text))
    words = [(tuple(B2U[b] for b in w.encode("utf-8")), c)   # 词 -> 符号元组,带频次
             for w, c in word_counts.items()]

    pair_counts = Counter()               # 相邻对 -> 总频次
    pair_to_words = defaultdict(set)      # 相邻对 -> 含它的词下标(增量索引)
    for i, (syms, c) in enumerate(words):
        for p in zip(syms, syms[1:]):
            pair_counts[p] += c
            pair_to_words[p].add(i)

    merges = []
    while len(merges) < vocab_size - 256:
        # max 遇平频取先插入者(dict 保序)——真实实现的「先见先得」
        best = max(pair_counts, key=pair_counts.get)
        if pair_counts[best] < 2:          # 没有成对的频次了,提前收工
            break
        a, b = best
        merges.append(best)
        for i in list(pair_to_words[best]):      # 只重扫受影响的词 ← 增量的关键
            syms, c = words[i]
            j, new = 0, []
            while j < len(syms):
                if j < len(syms) - 1 and (syms[j], syms[j + 1]) == best:
                    new.append(a + b)            # 合并!
                    j += 2
                else:
                    new.append(syms[j]); j += 1
            new = tuple(new)
            if new == syms:
                continue
            for p in zip(syms, syms[1:]):        # 撤旧对的账
                pair_counts[p] -= c
                if not pair_counts[p]: del pair_counts[p]
                pair_to_words[p].discard(i)
            for p in zip(new, new[1:]):          # 记新对的账
                pair_counts[p] += c
                pair_to_words[p].add(i)
            words[i] = (new, c)
        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)
    return merges
```

---

## 5. 手撕编码 / 解码

编码:对每个预切分出的词,字节变符号,**反复应用 rank 最小(最早学到)的可合并对**,直到无可合并。为什么必须按 rank?因为「行者」是先铸「(行字节1,行字节2)」再铸「行者」的——顺序错了,同样的字节会拼出不同的切分。解码:id → 符号 → 字节串 → 按 UTF-8 拼回文本。

```python
class BPETokenizer:
    def __init__(self, merges):
        self.merges = merges
        self.rank = {m: i for i, m in enumerate(merges)}
        self.sym_to_id = {B2U[b]: b for b in range(256)}
        sym_to_bytes = {B2U[b]: bytes([b]) for b in range(256)}
        for i, (a, b) in enumerate(merges):
            self.sym_to_id[a + b] = 256 + i
            sym_to_bytes[a + b] = sym_to_bytes[a] + sym_to_bytes[b]
        self.sym_to_bytes = sym_to_bytes
        self.id_to_sym = {i: s for s, i in self.sym_to_id.items()}

    def _merge_word(self, syms):
        """按训练顺序(rank)贪心合并——顺序与训练一致,结果才与词表对齐。"""
        while len(syms) > 1:
            best_k, best_rank = None, None
            for k, p in enumerate(zip(syms, syms[1:])):
                r = self.rank.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_k, best_rank = k, r
            if best_k is None:
                return syms
            syms[best_k:best_k + 2] = [syms[best_k] + syms[best_k + 1]]
        return syms

    def encode(self, text):
        ids = []
        for w in PRE_TOK.findall(text):
            syms = [B2U[b] for b in w.encode("utf-8")]
            for s in self._merge_word(syms):
                ids.append(self.sym_to_id[s])
        return ids

    def decode(self, ids):
        raw = b"".join(self.sym_to_bytes[self.id_to_sym[i]] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def id_to_token(self, i):
        return self.sym_to_bytes[self.id_to_sym[i]].decode("utf-8", errors="replace")
```

---

## 6. 真实运行:《西游记》分词器出厂

**实验 1:词表换序列长度的汇率**(全语料,真实运行):

```text
词表   512:   7.9s | 1,094,677 token | 1.97 字节/token | 往返无损 True
词表  1024:  11.7s |   829,251 token | 2.60 字节/token | 往返无损 True
词表  2048:  19.7s |   693,028 token | 3.11 字节/token | 往返无损 True  ← 本课主角
词表  4096:  39.8s |   602,897 token | 3.58 字节/token | 往返无损 True
(字级对照:723,587 token,无压缩;每级都全语料 decode(encode(x))==x ✓)
```

词表 ×8,序列 −45%,压缩率 1.97→3.58——**边际收益递减**,这是所有 BPE 曲线的形状。

**实验 2:主角登场,词表 2048**(训练 18.3s,学到 1792 个合并):

```text
「那猴王信受奉行，日日在洞中修炼。行者道：师父，八戒又在偷懒！」
30 字 → 26 token:
['那','猴王','信','受','奉','行','，','日','日','在','洞','中','修','炼','。',
 '行者道','：','师父','，','八戒','又','在','偷','�','�','！']
                                 ↑ 懒:语料只出现 61 次,不够铸整字模,
                                   退回 3 字节里的 2 个已学合并片段(往返仍无损 ✓)
```

铸模的先后有讲究——**第 15 次合并才铸出第一个整汉字「道」**,此前全在拼高频字的 UTF-8 字节前缀(比如「，」的前两个字节);到第 500 次合并附近,已经是词的世界:

```text
合并 #500 附近:['送','城','转','惊','兵','国王','内','呆','性','尽']
```

而主角团全部铸成了**单 token**(真实运行):

```text
行者: 1 token   八戒: 1 token   唐僧: 1 token   菩萨: 1 token
沙僧: 1 token   悟空: 1 token   妖精: 1 token   孙悟空: 1 token ← 三连铸!
```

---

## 7. 对照 Qwen:词表是语料的自画像

拿本地 Qwen3-4B 的 `tokenizer.json` 实读(基础词表 **151,643** 条 + 特殊 token 26 条,merges **151,387** 条;21 课对过账),查同一批词在 Qwen 那边的待遇:

```text
'行者' 是 Qwen 单 token: False      '进行' 是 Qwen 单 token: True
'八戒' 是 Qwen 单 token: False      '银行' 是 Qwen 单 token: True
'唐僧' 是 Qwen 单 token: False      '行政' 是 Qwen 单 token: True
'菩萨' 是 Qwen 单 token: True       '悟空' 是 Qwen 单 token: False
含'行'的 Qwen token 135 个:['之行','一行','下行','不行','举行','举行了',…]
                        ↑ 连日文搭配 'に行く' 都铸了模——Qwen 的语料里有整个互联网
Qwen 词表中含汉字的 token:25,511 条
```

**同一个算法,喂不同的语料,铸出完全不同的模。** 我们的玩具在《西游记》上学会了「行者」,Qwen 没学——不是它不行,是它的排版工没见过几百万字的明朝白话文。**词表是训练语料的化石**:换语料就要重训分词器(或至少扩词表),这是中文模型圈的老话题。

---

## 8. 生产账本:词表大小 V 的三本账(面试高频)

词表不是越大越好,三本账同时打开(以 Qwen3-4B 实测数字算):

**参数账。** embedding + lm_head 都是 $V \times d$ 的矩阵(21 课张量表里 `embed_tokens` 与 `lm_head`):

```text
Qwen3-4B:hidden_size=2560, vocab_size=151,936(实读 config.json)
embedding + lm_head = 2 × 151,936 × 2560 = 777,912,320 参数
                     ≈ 778M,占总参数 4.02B 的 19%   ← 每 5 个参数就有 1 个在管词表
```

**序列账。** 词表越大 token 越短(实验 1 的汇率),attention 的平方计算和 KV cache(18 课的显存公式)都跟着省。

**softmax 账。** 生成每个 token 都要在 V 个 logits 上做一次 softmax(第 1 课)——V 从 4K 到 151K,这一步的成本 ×37,而且 151K 里大半 token 在你的语料里根本不会被选。

三本账打架,才有各家不同的选择:Llama 系 128K、Qwen 151K、很多小模型刻意用小词表(如 32K)省参数。**没有免费的词表**。

生产环境还要记住三件事:

- **token ≠ 字符**。模型看见的是 id,不是笔画——「strawberry 里有几个 r」数错,不是模型笨,是这个词在它眼里可能只有 1–2 个 token;
- **大小写、空格、全半角都是不同字节**。「Medically」和「medically」在不同词表里的切分可以完全不同,预处理管线必须和训练时严格一致,否则线上就是静默的性能衰退;
- **特殊 token 是分词器的一部分**(21 课的 `<|im_start|>`、`<think>`)。25 课做 SFT 时,chat 模板就架在这些特殊 token 上——本课的伏笔。

---

## 9. 记住的故事

**现实:** 「OOV 不需要解决,需要消灭——把一切落到字节上。」

**数学:** $\text{merge}_i = \arg\max_{(a,b)}\ \text{count}(a,b)$,重复 $V-256$ 次;编码按 rank 贪心。

**工程:** 「增量账本:每次合并只重扫受影响的词——朴素法 $O(V{\cdot}N)$ 是一个世纪,增量法是 18 秒。」

**代码:** 第 4–5 节,`train_bpe` + `BPETokenizer` 共约 90 行——GPT/Qwen/Llama 同款算法的玩具版。

---

## 作业

### A. 概念题

1. 为什么 byte-level BPE 能彻底消灭 OOV?这个兜底能力付出了什么代价?
2. 预切分(pretokenize)这步是干什么的?如果训练时用 A 规则切、编码时用 B 规则切,会发生什么?
3. 为什么 encode 必须按 rank(合并的先后顺序)贪心,而不是「随便挑一个可合并的对」?
4. 词表大小 V 变大,参数账、序列账、softmax 账各往哪个方向走?

### B. 填空题

1. 一个汉字的 UTF-8 编码是 ____ 字节;本课 2048 词表在《西游记》上的压缩率是 ____ 字节/token(字级是 ____ token 全文)。
2. Qwen3-4B 基础词表 ____ 条 + 特殊 token ____ 条;embedding + lm_head 共 ____ 参数,占总参数约 ____。
3. 本课分词器第 ____ 次合并才铸出第一个整汉字「____」;「行者」编码为 ____ 个 token。
4. 「行者」____(是/不是)Qwen 的单 token;Qwen 词表含汉字的 token 有 ____ 条。

### C. 代码手撕(关讲义)

1. 默写 `train_bpe` 的增量更新:为什么只重扫 `pair_to_words[best]` 里的词?全量重扫的复杂度是多少?
2. 默写 `_merge_word` 的贪心合并循环;然后故意改成「任选可合并对」,构造一个切分与词表不一致的反例。
3. 写 round-trip 压力测试:从语料随机抽 100 段,断言 `decode(encode(x)) == x` 全部成立。

### D. 生产环境实战

1. 容量题:自训 32K 词表、d=4096 的模型,embedding + lm_head 共多少参数?bf16 下多大?对 7B 模型这买卖划算吗,写出你的判断依据。
2. 事故题:线上发现英文长词平均切成 5+ 个 token,推理延迟涨了 30%。给出排查顺序(大小写?空格?词表覆盖?预处理一致性?)。
3. 查证:SentencePiece 的 Unigram 与 BPE 的本质区别(一句话)?为什么主流开源模型(Llama、Qwen)都选 byte-level BPE?

### E. 拓展

1. 实验:把 vocab_size 提到 8192 再训一次,压缩率涨到多少?对照实验 1 画出边际收益曲线。
2. 预习钩子:词表有了,《西游记》已能变成 id 序列——可第 20 课的 Mini-GPT 只有前向。loss 谁来算(第 11 课有现成的)、**梯度谁来传**?下一课,把第 12 课的两行模板焊成整台机器。

---

# 下一课预告:反向传播手撕

二十万参数的 Mini-GPT,梯度要沿着 embedding → attention → FFN → 残差链一路回传。第 12 课你手撕过两条万能模板;下一课把它们焊成整台反向机器,并让**数值梯度**当裁判,逐模块验收——误差超过 1e-8 就不许出厂。

> **下一课:[Mini-GPT 反向传播](23-backward.md)——数值梯度当裁判,1e-8 才准出厂。**

## 参考来源

**本课实测数据来源(本机文件)**:

- `data/xiyouji.txt`——维基文库《西遊記》100 回公版全文(繁转简),723,587 字
- `models--mlx-community--Qwen3-4B-Instruct-2507-bf16` 的 tokenizer.json / config.json(HuggingFace 本地缓存,快照 f9e77d4283)

**文献**:

- [Sennrich et al. 2016, Neural Machine Translation of Rare Words with Subword Units (arXiv:1608.08622)](https://arxiv.org/abs/1608.08622) — 把 BPE 引入 NLP 的原始论文
- [Radford et al. 2019, Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — byte-level 映射的出处
- [Karpathy, minbpe (GitHub)](https://github.com/karpathy/minbpe) — 教学版 BPE 参考实现
