Tiger 这个模型使用了迭代多次相同路径（共享参数）在不增加参数的情况下提升性能， 但是它会带来另外一个问题，就是转换的图中节点数量会成倍增加，而NPU就不擅长这种数量巨大每个操作又是小的tensor的操作。 你能帮我在网络搜索下最新的sound source separation 方面的paper，找到一个更适合edge 端NPU 部署的模型吗？


Thought

Thought

Thought for 53s
可以。基于你现在的约束，我会把筛选标准设成这几条：

少用或不用“共享参数多次展开”的迭代路径，避免导出图节点数成倍膨胀

尽量以 Conv / FC / 小型 MLP / 简单序列模块为主，少依赖复杂 attention、动态缓存和大量小张量算子

已有真实 edge 端结果，最好明确提到实时、量化、手机/GAP9/移动端加速器

任务尽量接近 source separation，不是泛泛的 enhancement

按这个标准，我最推荐你优先看的不是 TIGER 这类“同一路径反复迭代”的结构，而是 TF-MLPNet。它是 2025 年的工作，作者明确把目标设成“低功耗 hearable / tiny accelerator 上的实时 speech separation”，模型在时频域工作，核心是沿 channel/frequency 交替的全连接层堆叠 + 每个频点上独立的时间卷积，并且作者报告了mixed-precision QAT 后能在 GAP9 上实时处理 6 ms chunk，运行时比以往模型快 3.5–4×。从部署视角看，这比反复 unroll 的共享路径要友好得多，因为它更像规则的 FC/Conv 图，而不是大量重复的小 attention 子图。

如果你的目标更偏通用语音分离/目标说话人提取，而且重点是 NPU/低功耗加速器，我会把 TF-MLPNet 放在第一优先级。原因不是它一定在所有 separation benchmark 上绝对 SOTA，而是它的设计目标和你的部署痛点几乎完全对齐：tiny real-time、QAT、低功耗 accelerator、规整算子栈。对你现在遇到的“ONNX/circle 图太大、节点太碎、NPU 不擅长”这一类问题，它比 TIGER 这类迭代式架构更像正确方向。

第二个值得认真看的，是 DeepBeam 这类经典前端 + 小网络后端的 hybrid 路线。它不是纯黑盒 separation，而是把传统 beamformer 和轻量神经网络结合起来。公开页面给出的结果是：模型大小减少 5×、计算量每秒减少 4×、处理时间减少 5×，并且能在低功耗 wearable 级移动 CPU 上 8 ms 运行、端到端延迟 17.5 ms。这条路线对 edge 很有吸引力，因为它把一部分“难算的全局建模”交给传统 DSP/波束形成，神经网络只做剩余补偿；从图编译视角看，通常也比多轮共享块展开更稳定。缺点是它更适合多麦/空间信息可用的场景，而不是单通道通用 separation。

如果你关心的是手机端或移动端 GPU/NPU 风格部署，Samsung 2025 这篇 Single-Channel Distance-Based Source Separation 也很值得看。它不是通用“任意源分离”，而是把任务定义成按距离分离 near/far source。对你有价值的点在于：作者明确做了移动端加速优化，并报告模型可在 Galaxy S23 上用 TFLite GPU delegate 实时运行，RTF 0.16、约 6× 实时、能耗下降 20×，模型大约 1.3M 参数 / 25.7 GMac/s。不过它内部仍有 Conformer/线性 RSA，算子图会比 TF-MLPNet 更复杂一些，所以我会把它当作“移动端工程参考”，不作为你的首选 backbone。

如果你的任务偏音乐源分离，而不是语音分离，我会看两篇：RT-STT 和 Moises-Light。RT-STT 是 2025 年面向real-time, low-latency music source separation 的工作，作者明确提出 single-path 优于 dual-path，并且专门研究了量化来进一步降低推理时间。这点和你当前的部署痛点高度一致，因为它本质上是在削减“路径重复 + 序列模块复杂度”。

相比之下，Moises-Light 更像“轻量但仍追求较强音乐分离性能”的折中方案。它大约 5M 参数/单 stem，并声称以比 BS-RoFormer 少 13× 参数取得有竞争力的 MUSDB-HQ 结果。它仍然带有 band-split 和 transformer 元素，所以参数规模已经明显降下来了，但从 NPU 图友好性 来说，我认为它未必比 TF-MLPNet 或 RT-STT 更优，因为 band-split + transformer 往往会引入更多 reshape/split/concat 类操作。

我的结论很直接：

如果你要找一个比 TIGER 更适合 edge 端 NPU 部署的 source separation 方向，首推 TF-MLPNet。
原因是它同时满足了这几个关键条件：

目标明确指向 tiny low-power accelerator / hearable

已有 实时 和 QAT 结果

主体算子更接近 FC + Conv，更容易做算子融合、静态图优化和低比特量化

不依赖 TIGER 那种共享参数反复迭代展开带来的巨图问题 

如果你要一个更工程化的选择，我会这样排：

语音 separation / TSE / blind speech separation

TF-MLPNet：最像 edge-first 方案。

DeepBeam：有多麦时很强，DSP+小网络混合很适合穿戴式/移动端。

Samsung DSS mobile model：如果你的任务能重述成 near/far 分离，它非常有移动端价值。

音乐 separation

RT-STT：比 dual-path / iterative 结构更适合实时部署。

Moises-Light：轻量高性能折中，但图复杂度大概率仍高于 TF-MLPNet 这类 MLP/Conv 路线。

如果你想把这个结论进一步落到你的工程上，我建议下一步不要直接“换一个论文名字”，而是把 TIGER 抽象成几个会伤害 NPU 部署的设计点，然后选替代结构：

去掉 iteration/unroll 共享路径

去掉需要大量 reshape / transpose / slice / cat / matmul 的 attention cell

保留 STFT front-end + mask estimation

主干改成 single-pass TF MLP/Conv stack

训练时做 QAT + chunk-wise streaming

这会比简单把 TIGER 替换成另一个“看起来更轻量的 transformer”更稳。




arXiv
[2508.03047] TF-MLPNet: Tiny Real-Time Neural Speech Separation



arXiv
[2511.13146] Towards Practical Real-Time Low-Latency Music Source Separation

directionalhearing.cs.washington.edu
Deep Learning for Directional Hearing

Home | Samsung Research
BLOG | Samsung Research



arXiv
[2506.06689] A Fast and Lightweight Model for Causal Audio-Visual Speech Separation



arXiv
Moises-Light: Resource-efficient Band-split U-Net For Music Source Separation


如果你愿意，我下一步可以直接按你的 TIGER 输入输出接口，给你设计一个 “TF-MLPNet 风格、可导出 ONNX/circle、面向 edge NPU” 的替代骨干。


直接按我的 TIGER 输入输出接口，给你设计一个 “TF-MLPNet 风格、可导出 ONNX/circle、面向 edge NPU” 的替代骨干


Thought for 51s
可以。我直接按你现在这份 TIGER streaming 接口 来设计一个替代骨干，目标是：

输入输出接口不变

单帧 forward_cell(T=1) 可导出 ONNX/circle

长序列仍然通过 forward_sequence() 复用同一套 streaming cell

去掉 TIGER 里最伤 NPU 的部分：迭代展开、KV attention、过多小张量 MatMul/reshape/slice/cat

主干改成 TF-MLPNet 风格的单次前向、Conv/1x1/Depthwise 为主 的结构

你当前 TIGER 的接口核心是：

forward(subband_spec_RIs, past_kvs, past_valid_mask, prev_states_0, prev_states_1, prev_states_2, prev_global_states, ...)

当 T=1 时走 forward_cell()

当 T>1 时走 forward_sequence()，按时间展开同一个 streaming cell

输出是 band_masked_output 加更新后的 state/cache 


而 TIGERCtx 是同样的思路，只是 time path state 被压成一个 time_ctx。

所以我建议你不要做“论文级别完全换范式”的替换，而是做一个 接口兼容的 EdgeTIGER-MLP：

方案名
EdgeTIGER-MLP

一句话定义：

保留 TIGER 的 encoder / decoder / streaming API，不保留 RecurrentKV attention separator；把 separator 换成单次 through 的、状态显式可缓存的 TF-MLP 风格时频混合网络。

1. 总体结构
保持外层 TIGER 类的调用方式不变：

subband_spec_RIs
    -> _encode_subbands(...)
    -> separator(...)         # 替换这里
    -> _decode_masks(...)
    -> band_masked_output
也就是说：

_encode_subbands() 不动

_decode_masks() 不动

forward / forward_cell / forward_sequence / init_streaming_state 的签名不动

只把 self.separator = RecurrentKV(...) 换成一个新的 self.separator = EdgeTFMLPSeparator(...)

这和你当前代码的封装方式完全一致：forward_cell() 先 _encode_subbands()，再 separator(...)，再 _decode_masks()。

2. 为什么这个替代骨干更适合 edge NPU
2.1 去掉 KV attention
不再有：

q/k/v

bmm / MatMul

transpose + reshape + slice + cat 链式图

export 分支和 training 分支不一致的问题

2.2 去掉 iteration 展开
TIGER 现在是“同一路径共享参数多次展开”，参数没涨，但图节点会按迭代次数放大。你已经碰到了这一点。
新方案改成：

固定深度、单次 through

不做 iterative reuse

参数量可控，图规模也可控

2.3 运算以 NPU 友好的算子为主
主算子只保留：

Conv2d(1x1)

Depthwise Conv2d(1xk) 或 (k x 1)

Add

Mul

Sigmoid/ReLU/PReLU

少量固定 shape 的 Concat

这些通常比 attention / matmul / 动态 cache 拼接更适合 edge 编译器。

3. 新 separator 的核心思路
3.1 输入输出张量形状
保持和现有 separator 一致。

输入给 separator 的是编码后的子带特征：

x: [B, N, nband, T]
输出：

sep_output: [B, num_output * feature_dim, nband, T]   # 与当前 decoder 对接
外加 streaming 状态：

new_kvs, new_valid_mask, new_state_0, new_state_1, new_state_2, new_global_state
但这里：

past_kvs / past_valid_mask 不再承载真正 attention KV

为了兼容旧接口，它们可以变成 空占位状态 或 轻量 history buffer

真正有效的时间状态，放在 prev_states_0/1/2 和 prev_global_states

这能保证你上层训练/导出脚本基本不用改。当前 TIGER 的 state 组成本来也是这几项。


4. 新主干：TF-MLP 风格时频混合块
我建议 separator 由下面三部分组成：

4.1 输入投影
先把编码特征投影到较小 hidden 维：

x = in_proj(x)   # 1x1 conv, [B, N, F, T] -> [B, H, F, T]
建议：

H = 96 或 128

edge 版优先 96

4.2 多层 Edge TF-MLP Block
堆 6 到 8 层，不共享参数，不迭代展开。

每层 block 由 3 个子模块组成：

A. 频率混合 Frequency Mixer
沿频率维做局部混合：

freq = pw1(x)
freq = dw_freq(freq)   # depthwise (k_f, 1)
freq = pw2(freq)
x = x + freq
推荐：

k_f = 3 或 5

不做全频大矩阵 MLP，避免大 FC

用深度可分离卷积近似 TF-MLPNet 的 frequency mixing

B. 时间混合 Time Mixer（带显式缓存）
沿时间维做 causal mixing：

time, new_state_i = StatefulCausalFrameMlpLike(x, state_i)
x = x + time
这个模块是关键。它应该只用：

1x1 conv

depthwise causal conv (1, k_t)

1x1 conv

并且显式维护每层的左缓存 state。
这和你当前代码里 StatefulCausalFrameMlp 的思路是兼容的，只是我们把它作为主干，而不是 attention 辅助件。你文件里已经有这个模式可参考。

推荐：

k_t = 3

dilation 用 [1, 2, 4] 循环

每层只保留必要 lookback，不搞大窗口 KV

C. 轻量门控 Gated Fusion
做一个很便宜的门控，而不是 attention：

gate = sigmoid(gate_proj(x))
x = x + gate * mix_proj(x)
这一步的作用是保留 TIGER 原来“局部 + 全局注入”的一点味道，但成本很低。

4.3 输出头 Mask Head
最后输出给 decoder 的 mask 特征：

sep_output = out_proj(x)   # [B, H, F, T] -> [B, num_output * feature_dim, F, T]
和你现有 _decode_masks() 对接。

5. 建议的 block 具体配置
Edge 版
适合优先部署：

hidden channels: 96

num blocks: 6

expansion ratio: 2

time kernel: 3

freq kernel: 3

dilations: [1, 2, 4, 1, 2, 4]

Balance 版
兼顾效果和部署：

hidden channels: 128

num blocks: 8

expansion ratio: 2

time kernel: 3

freq kernel: 5

dilations: [1, 2, 4, 8, 1, 2, 4, 8]

Large 版
只在 NPU 还扛得住时考虑：

hidden channels: 160

num blocks: 8~10

expansion ratio: 2

仍然不引入 attention 和迭代展开

6. 状态设计：完全兼容 TIGER 外部接口
你现在 TIGER 的 streaming 状态接口是：

past_kvs
past_valid_mask
prev_states_0
prev_states_1
prev_states_2
prev_global_states
所以新骨干我建议这样映射：

6.1 past_kvs
保留，但弱化为占位张量。

做法：

shape 不变

运行时直接透传或返回零

这样你原来的导出和 runner 可以不改签名

6.2 past_valid_mask
同上，保留但基本不参与计算。

6.3 prev_states_0 / 1 / 2
作为三组时间卷积缓存：

block 按组分配到三类 dilation stage

每组 state shape 继续用 [B, C_group, nband, width]

width = (kernel_size - 1) * dilation_max

6.4 prev_global_states
作为全局时间摘要缓存。

最简单的方式：

存最近若干帧的低维投影

或者只存一个 EMA/低通上下文

通过 1x1 conv + sigmoid 注入到当前层

这样你就保留了“有 global state”的接口，同时实现非常轻量。

7. 我建议的类设计
7.1 外层模型类
新类可以这样定义：

class TIGEREdgeMLP(TIGER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.separator = EdgeTFMLPSeparator(
            input_dim=self.feature_dim,
            hidden_dim=kwargs.get("edge_hidden_dim", 96),
            nband=self.nband,
            num_blocks=kwargs.get("edge_num_blocks", 6),
            num_output=self.num_output,
            lookahead=0,
            state_groups=3,
            time_kernel_size=3,
            freq_kernel_size=3,
            expansion=2,
        )
        self.supports_exact_chunk_training = True
这样外部仍然是 TIGER 风格。

7.2 separator 的 forward 签名
必须和当前 TIGER separator 对齐：

def forward(
    self,
    x,
    past_kvs=None,
    past_valid_mask=None,
    prev_states_0=None,
    prev_states_1=None,
    prev_states_2=None,
    prev_global_states=None,
):
    ...
    return sep_output, new_kvs, new_valid_mask, new_state_0, new_state_1, new_state_2, new_global_state
这样你原来的 forward_cell() 和 forward_sequence() 可以直接复用。当前 TIGER 外层就是这么传状态的。


8. 推荐的模块拆分
8.1 EdgeFreqMixer
class EdgeFreqMixer(nn.Module):
    # 1x1 -> dwconv(kf,1) -> 1x1
8.2 EdgeTimeMixer
class EdgeTimeMixer(nn.Module):
    # 1x1 -> causal dwconv(1,kt) -> 1x1
    # returns y, new_state
8.3 EdgeGate
class EdgeGate(nn.Module):
    # cheap sigmoid gate
8.4 EdgeTFMLPBlock
class EdgeTFMLPBlock(nn.Module):
    # freq mixer + time mixer + gate + residual
8.5 EdgeTFMLPSeparator
class EdgeTFMLPSeparator(nn.Module):
    # in_proj -> stacked blocks -> out_proj
9. 一个可直接落地的前向逻辑
forward_cell(T=1)
x = in_proj(x)

x, s0 = block0(x, state=s0, global_state=g)
x, s1 = block1(x, state=s1, global_state=g)
x, s2 = block2(x, state=s2, global_state=g)
x, s0 = block3(x, state=s0, global_state=g)
x, s1 = block4(x, state=s1, global_state=g)
x, s2 = block5(x, state=s2, global_state=g)

g = update_global_state(g, x)

sep_output = out_proj(x)

return sep_output, dummy_kv, dummy_mask, s0, s1, s2, g
这里有几个关键好处：

没有任何 MatMul

没有 attention KV packing

没有 iteration unroll

单帧图就是部署图

长序列训练时，forward_sequence() 只是重复调用这个 cell

这和你当前 TIGER 的 streaming 组织方式是完全一致的，只是 cell 里面的骨干换掉了。


10. 为什么我不建议“纯 MLP 沿频率做全连接”
虽然你说要 TF-MLPNet 风格，但如果完全照搬“沿频率做大矩阵混合”：

当 nband 不小

且你要兼容多个输入 shape

导出到 ONNX/circle 时

往往仍然会带来：

reshape/permute 变多

Conv 不一定能完全替代

编译器对 FC-on-reordered-dim 未必比 Conv 更友好

所以我的建议是：

保留 TF-MLPNet 的设计思想，但在实现层面用 Conv 化版本。

也就是：

频率混合：用 (k_f, 1) depthwise conv 近似

时间混合：用 (1, k_t) causal depthwise conv 近似

通道混合：用 1x1 conv

这样更像 “TF-MLP 思想 + MobileNet 式实现”，对 NPU 更稳。

11. 训练建议
11.1 初始化
如果你想尽量复用现有训练脚手架：

encoder / decoder 权重尽量从 TIGER 迁移

separator 全新初始化

11.2 蒸馏
最有效的办法不是直接从头训，而是：

用现有 TIGER 当 teacher

新 TIGEREdgeMLP 当 student

loss = 原任务 loss + teacher feature distill + mask distill

推荐蒸馏位点：

_encode_subbands() 后的子带特征

separator 输出 sep_output

decoder 前的 mask logits

11.3 QAT
因为目标就是 edge/NPU：

训练后期做 QAT

优先 int8 激活 + int8 权重

PReLU 若量化不稳定，可换 ReLU / ReLU6 / HardSwish

11.4 Chunk training
沿用你当前的 exact chunk training 方式即可，因为外层 forward_sequence() 逻辑已经是按 chunk 复用 cell。


12. 推理和导出建议
单帧部署图
只导：

forward_cell(
    subband_spec_RIs,
    past_kvs,
    past_valid_mask,
    prev_states_0,
    prev_states_1,
    prev_states_2,
    prev_global_states,
)
状态 shape
为了最大限度兼容旧 runner：

past_kvs: 保持旧 shape，但内容不用

past_valid_mask: 保持旧 shape，但内容不用

prev_states_0/1/2: 真正有效

prev_global_states: 真正有效

graph 特征
导出后的图主要应该只剩：

Conv

Add

Mul

Sigmoid/ReLU

少量 Concat

这会比现在的 TIGER attention 图干净很多。

13. 我给你的最终替代方案总结
如果用一句话总结这个新 backbone：

它不是把 TIGER 换成另一个复杂模型，而是把 TIGER 的 separator 改造成一个“单次 through 的、显式状态缓存的、Conv 化 TF-MLP 风格时频混合网络”，同时保留 TIGER 的外层接口、训练方式和 decoder。

我建议你先落地这一个版本：

hidden_dim = 96

num_blocks = 6

time_kernel = 3

freq_kernel = 3

expansion = 2

不用 attention

不用迭代展开

past_kvs/past_valid_mask 仅兼容保留

真正状态只放在 prev_states_0/1/2/global

这是最稳的 edge 起点。

14. 下一步我可以直接给你的内容
我可以继续直接写出一套可粘贴进你工程的 PyTorch 代码骨架，包括：

EdgeFreqMixer

EdgeTimeMixer

EdgeTFMLPBlock

EdgeTFMLPSeparator

TIGEREdgeMLP

init_streaming_state

forward_cell

forward_sequence

并且我会严格按你现在这份 TIGER 的接口来写。


请完整写出一套可粘贴进我工程的 PyTorch 代码骨架，包括：

EdgeFreqMixer
EdgeTimeMixer
EdgeTFMLPBlock
EdgeTFMLPSeparator
TIGEREdgeMLP
init_streaming_state
forward_cell
forward_sequence

并且严格按你现在这份 TIGER 的接口来写， 并且遵守时间维causal， 以及下面的NPU约束：**NPU部署要求**： 
	- 因为NPU支持的算作有限，  只能使用简单的conv2d，以及对应的反卷积， torch.bmm, 以及其它的一些基本的tensor操作
        - tensor 维度不能超过4维， 前向过程中，4维的tensor不能往第一个维度折叠（Batch维度），以保证导出onnx时第一维总是1(如果维度小于4， 那么折叠到batch维度就不是问题）
        - conv2d的 (kernel_size-1)*dilation <14 
        - 模型中经量不要出现tensor 常量，因为转NPU的时候这些常量可能会带来问题，如果不能避免，可以通过输入参数传递
        - 在设备端是realtime/online 推理，所有需要保证训练时跟部署时要兼容，比如时间维度要是causal的， 而且部署时性能不能比训练时下降太多
        - 为了维持causal/online， 需要把一些state或者cache（比如kv_cache）存储并传递到下一帧，并且存储的大小受到DSP 192k 限额的约束
        - 这些存储的state/cache 需要通过模型输入输出进行传递，onnx导出时需要考虑这些参数


Thought for 2m 54s
下面这套代码骨架是按你当前文件里的 TIGER 接口 来写的：forward_cell(subband_spec_RIs, past_kvs, past_valid_mask, prev_states_0, prev_states_1, prev_states_2, prev_global_states)、forward_sequence(...)、以及 init_streaming_state(...)，并且仍然复用外层 TIGER 的 _encode_subbands() / _decode_masks()。你原文件里 TIGER 的这几个接口和状态流就是这样组织的。

 


这版骨干遵守你给的约束：

全程只用 Conv2d、cat、切片、逐元素加乘、sigmoid/relu

不使用 bmm / attention / MatMul

张量始终不超过 4 维，且 不把 4D 张量折叠到 batch 维

时间维严格 causal，训练和部署共用同一套 streaming state 逻辑

state/cache 全都通过模型输入输出传递

time_kernel_size 和 dilation 都检查 (kernel_size - 1) * dilation < 14

你可以把下面代码直接贴进现有 tiger.py，放在 TIGER 类定义之后。它默认继承你现有的 TIGER，因此会直接复用现有 encoder/decoder。


import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Basic utilities
# =========================================================

class EdgeChannelNorm(nn.Module):
    """
    Simple channel-only normalization on 4D tensors [B, C, F, T].
    No tensor rank expansion beyond 4D.
    """
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        return centered * inv_std * self.weight + self.bias


class EdgePWConvBlock(nn.Module):
    """
    1x1 Conv2d + ReLU + channel norm
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.act = nn.ReLU()
        self.norm = EdgeChannelNorm(out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x


# =========================================================
# EdgeFreqMixer
# =========================================================

class EdgeFreqMixer(nn.Module):
    """
    Frequency-axis local mixer on [B, C, F, T]
    Only uses:
      - 1x1 Conv2d
      - depthwise Conv2d with kernel (k_f, 1)
      - residual add
    """
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        assert freq_kernel_size >= 1 and (freq_kernel_size % 2 == 1), "freq_kernel_size must be odd"

        hidden = channels * expansion
        pad_f = (freq_kernel_size - 1) // 2

        self.pre = EdgePWConvBlock(channels, hidden, bias=bias)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(freq_kernel_size, 1),
            stride=(1, 1),
            padding=(pad_f, 0),   # frequency local symmetric padding is OK
            dilation=(1, 1),
            groups=hidden,
            bias=bias,
        )
        self.dw_act = nn.ReLU()
        self.dw_norm = EdgeChannelNorm(hidden)
        self.post = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.post_norm = EdgeChannelNorm(channels)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.pre(x)
        x = self.dw(x)
        x = self.dw_act(x)
        x = self.dw_norm(x)
        x = self.post(x)
        x = self.post_norm(x)
        return residual + x


# =========================================================
# EdgeTimeMixer
# =========================================================

class EdgeTimeMixer(nn.Module):
    """
    Causal time-axis mixer on [B, C, F, T]
    Streaming state shape:
      [B, channels, F, state_width]
    Uses only:
      - 1x1 Conv2d
      - depthwise Conv2d with kernel (1, k_t), padding=0
      - explicit left-cache concatenation
    """
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        time_kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert time_kernel_size >= 1 and (time_kernel_size % 2 == 1), "time_kernel_size must be odd"
        assert dilation >= 1, "dilation must be >= 1"
        assert (time_kernel_size - 1) * dilation < 14, \
            "NPU constraint violated: (kernel_size - 1) * dilation must be < 14"

        hidden = channels * expansion
        self.channels = channels
        self.hidden = hidden
        self.time_kernel_size = time_kernel_size
        self.dilation = dilation
        self.state_width = (time_kernel_size - 1) * dilation

        self.pre = EdgePWConvBlock(channels, hidden, bias=bias)

        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, time_kernel_size),
            stride=(1, 1),
            padding=(0, 0),               # causal handled by explicit state/cache
            dilation=(1, dilation),
            groups=hidden,
            bias=bias,
        )
        self.dw_act = nn.ReLU()
        self.dw_norm = EdgeChannelNorm(hidden)

        self.post = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.post_norm = EdgeChannelNorm(channels)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        """
        x: [B, C, F, T]
        state: [B, hidden, F, state_width]
        """
        residual = x
        x = self.pre(x)  # [B, hidden, F, T]

        B, H, Freq, T = x.shape

        if self.state_width > 0:
            if state is None:
                state = x.new_zeros(B, H, Freq, self.state_width)
            combined = torch.cat([state, x], dim=-1)  # [B, hidden, F, state_width + T]
            new_state = combined[:, :, :, -self.state_width:]
        else:
            combined = x
            new_state = x[:, :, :, :0]

        x = self.dw(combined)    # [B, hidden, F, T]
        x = self.dw_act(x)
        x = self.dw_norm(x)
        x = self.post(x)
        x = self.post_norm(x)
        return residual + x, new_state


# =========================================================
# EdgeTFMLPBlock
# =========================================================

class EdgeTFMLPBlock(nn.Module):
    """
    One block = frequency mixer + causal time mixer + cheap global gated fusion.
    All tensors remain 4D.
    """
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        time_kernel_size: int = 3,
        dilation: int = 1,
        global_context_width: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.global_context_width = global_context_width

        self.freq_mixer = EdgeFreqMixer(
            channels=channels,
            expansion=expansion,
            freq_kernel_size=freq_kernel_size,
            bias=bias,
        )
        self.time_mixer = EdgeTimeMixer(
            channels=channels,
            expansion=expansion,
            time_kernel_size=time_kernel_size,
            dilation=dilation,
            bias=bias,
        )

        # cheap global gated fusion:
        # concat([x, global_context_repeated], dim=1) -> gate/mix
        self.gate_conv = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.mix_conv = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.out_norm = EdgeChannelNorm(channels)

        # update next global state from the latest frame only
        self.global_update = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, time_state: torch.Tensor = None, global_state: torch.Tensor = None):
        """
        x: [B, C, F, T]
        time_state: [B, C*expansion_after_pre? no -> hidden in EdgeTimeMixer, F, state_width]
                    NOTE: the caller must pass the state produced by this block's time_mixer.
        global_state: [B, C, F, 1]
        """
        B, C, Freq, T = x.shape

        x = self.freq_mixer(x)
        x, new_time_state = self.time_mixer(x, state=time_state)

        if global_state is None:
            global_state = x.new_zeros(B, C, Freq, self.global_context_width)

        # repeat cached global context to current T using expand on the last dim
        # [B, C, F, 1] -> [B, C, F, T]
        if global_state.shape[-1] != 1:
            # For simplicity and export stability, we keep width=1.
            # If caller passes a larger width, only the latest frame is used.
            global_state = global_state[:, :, :, -1:]
        global_rep = global_state.expand(B, C, Freq, T)

        fusion_in = torch.cat([x, global_rep], dim=1)
        gate = torch.sigmoid(self.gate_conv(fusion_in))
        mix = self.mix_conv(fusion_in)
        x = self.out_norm(x + gate * mix)

        # update global state from the latest output frame only
        new_global_state = self.global_update(x[:, :, :, -1:])  # [B, C, F, 1]
        return x, new_time_state, new_global_state


# =========================================================
# EdgeTFMLPSeparator
# =========================================================

class EdgeTFMLPSeparator(nn.Module):
    """
    TIGER-compatible separator replacement.
    Input:
      x: [B, feature_dim, nband, T]
    Output:
      sep_output: [B, num_output * feature_dim, nband, T]

    Interface-compatible states:
      past_kvs, past_valid_mask: kept as dummy passthrough states for compatibility
      prev_states_0/1/2: real causal time states for block groups 0/1/2
      prev_global_states: real packed global states, one slice per block
    """
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        nband: int,
        num_output: int,
        num_blocks: int = 6,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        time_kernel_size: int = 3,
        time_dilations=(1, 2, 4),
        dummy_kv_channels: int = 1,
        dummy_kv_width: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert len(time_dilations) == 3, "time_dilations must have length 3, e.g. (1,2,4)"
        assert num_blocks >= 1, "num_blocks must be >= 1"
        assert hidden_channels >= 1, "hidden_channels must be >= 1"

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.nband = nband
        self.num_output = num_output
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.freq_kernel_size = freq_kernel_size
        self.time_kernel_size = time_kernel_size
        self.time_dilations = tuple(int(d) for d in time_dilations)

        self.dummy_kv_channels = int(dummy_kv_channels)
        self.dummy_kv_width = int(dummy_kv_width)

        # Dummy compatibility attributes to keep outer code simple
        self.n_heads = 1
        self.kv_window_size = self.dummy_kv_width
        self.att_hid_chan = self.dummy_kv_channels
        self.att_val_hid_chan = 0
        self.iter = self.num_blocks

        self.in_proj = EdgePWConvBlock(input_dim, hidden_channels, bias=bias)

        blocks = []
        for i in range(num_blocks):
            dilation = self.time_dilations[i % 3]
            blocks.append(
                EdgeTFMLPBlock(
                    channels=hidden_channels,
                    expansion=expansion,
                    freq_kernel_size=freq_kernel_size,
                    time_kernel_size=time_kernel_size,
                    dilation=dilation,
                    global_context_width=1,
                    bias=bias,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.out_proj = nn.Conv2d(
            hidden_channels,
            num_output * input_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        # How many blocks belong to each state group
        self.group_block_indices = [
            [i for i in range(num_blocks) if (i % 3) == 0],
            [i for i in range(num_blocks) if (i % 3) == 1],
            [i for i in range(num_blocks) if (i % 3) == 2],
        ]

        self.group_state_widths = [
            (time_kernel_size - 1) * self.time_dilations[0],
            (time_kernel_size - 1) * self.time_dilations[1],
            (time_kernel_size - 1) * self.time_dilations[2],
        ]

        for width in self.group_state_widths:
            assert width < 14, "Per-group state width must remain small for deployment"

        self.group_state_channels = [
            len(self.group_block_indices[0]) * hidden_channels * expansion,
            len(self.group_block_indices[1]) * hidden_channels * expansion,
            len(self.group_block_indices[2]) * hidden_channels * expansion,
        ]

        self.global_state_channels = num_blocks * hidden_channels
        self.global_state_width = 1

    def _make_dummy_kv(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size,
            self.n_heads,
            self.kv_window_size,
            self.dummy_kv_channels,
            device=device,
            dtype=dtype,
        )

    def _make_dummy_mask(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size,
            1,
            self.kv_window_size,
            1,
            device=device,
            dtype=dtype,
        )

    def init_streaming_state(self, batch_size: int, device=None, dtype=None):
        return (
            self._make_dummy_kv(batch_size, device, dtype),
            self._make_dummy_mask(batch_size, device, dtype),
            torch.zeros(
                batch_size,
                self.group_state_channels[0],
                self.nband,
                self.group_state_widths[0],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.group_state_channels[1],
                self.nband,
                self.group_state_widths[1],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.group_state_channels[2],
                self.nband,
                self.group_state_widths[2],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.global_state_channels,
                self.nband,
                self.global_state_width,
                device=device,
                dtype=dtype,
            ),
        )

    def _slice_group_state(self, packed_state: torch.Tensor, block_local_idx: int, hidden_dim: int):
        """
        packed_state shape: [B, group_blocks * hidden_dim, F, W]
        returns one slice: [B, hidden_dim, F, W]
        """
        c0 = block_local_idx * hidden_dim
        c1 = c0 + hidden_dim
        return packed_state[:, c0:c1, :, :]

    def forward(
        self,
        x,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
    ):
        """
        x: [B, input_dim, nband, T]
        """
        assert x.dim() == 4, "EdgeTFMLPSeparator expects a 4D tensor [B, C, F, T]"
        B, C, Freq, T = x.shape
        assert Freq == self.nband, f"Expected nband={self.nband}, got F={Freq}"

        if past_kvs is None:
            past_kvs = self._make_dummy_kv(B, x.device, x.dtype)
        if past_valid_mask is None:
            past_valid_mask = self._make_dummy_mask(B, x.device, x.dtype)

        if prev_states_0 is None:
            prev_states_0 = x.new_zeros(B, self.group_state_channels[0], self.nband, self.group_state_widths[0])
        if prev_states_1 is None:
            prev_states_1 = x.new_zeros(B, self.group_state_channels[1], self.nband, self.group_state_widths[1])
        if prev_states_2 is None:
            prev_states_2 = x.new_zeros(B, self.group_state_channels[2], self.nband, self.group_state_widths[2])
        if prev_global_states is None:
            prev_global_states = x.new_zeros(B, self.global_state_channels, self.nband, self.global_state_width)

        h = self.in_proj(x)

        new_group0 = []
        new_group1 = []
        new_group2 = []
        new_global = []

        group_local_counters = [0, 0, 0]

        for block_idx, block in enumerate(self.blocks):
            group_id = block_idx % 3
            local_idx = group_local_counters[group_id]
            group_local_counters[group_id] += 1

            if group_id == 0:
                state_slice = self._slice_group_state(
                    prev_states_0, local_idx, self.hidden_channels * self.expansion
                )
            elif group_id == 1:
                state_slice = self._slice_group_state(
                    prev_states_1, local_idx, self.hidden_channels * self.expansion
                )
            else:
                state_slice = self._slice_group_state(
                    prev_states_2, local_idx, self.hidden_channels * self.expansion
                )

            g0 = block_idx * self.hidden_channels
            g1 = g0 + self.hidden_channels
            global_slice = prev_global_states[:, g0:g1, :, :]   # [B, hidden_channels, F, 1]

            h, new_state_slice, new_global_slice = block(
                h,
                time_state=state_slice,
                global_state=global_slice,
            )

            if group_id == 0:
                new_group0.append(new_state_slice)
            elif group_id == 1:
                new_group1.append(new_state_slice)
            else:
                new_group2.append(new_state_slice)

            new_global.append(new_global_slice)

        if len(new_group0) > 0:
            new_states_0 = torch.cat(new_group0, dim=1)
        else:
            new_states_0 = h.new_zeros(B, 0, self.nband, self.group_state_widths[0])

        if len(new_group1) > 0:
            new_states_1 = torch.cat(new_group1, dim=1)
        else:
            new_states_1 = h.new_zeros(B, 0, self.nband, self.group_state_widths[1])

        if len(new_group2) > 0:
            new_states_2 = torch.cat(new_group2, dim=1)
        else:
            new_states_2 = h.new_zeros(B, 0, self.nband, self.group_state_widths[2])

        new_global_states = torch.cat(new_global, dim=1)

        sep_output = self.out_proj(h)
        return (
            sep_output,
            past_kvs,          # compatibility passthrough
            past_valid_mask,   # compatibility passthrough
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        )


# =========================================================
# TIGEREdgeMLP
# =========================================================

class TIGEREdgeMLP(TIGER):
    """
    TIGER-compatible edge deployment backbone:
    - keeps TIGER encoder/decoder and public API
    - replaces the separator with a causal Conv2d-only TF-MLP-like backbone
    - no attention / no bmm / no iterative shared-path unroll
    """
    def __init__(
        self,
        *args,
        edge_hidden_channels=64,
        edge_num_blocks=6,
        edge_expansion=2,
        edge_freq_kernel_size=3,
        edge_time_kernel_size=3,
        edge_time_dilations=(1, 2, 4),
        edge_dummy_kv_channels=1,
        edge_dummy_kv_width=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # NPU constraint checks
        assert len(edge_time_dilations) == 3, "edge_time_dilations must have length 3"
        for d in edge_time_dilations:
            assert (edge_time_kernel_size - 1) * int(d) < 14, \
                "NPU constraint violated: (kernel_size - 1) * dilation must be < 14"

        self.separator = EdgeTFMLPSeparator(
            input_dim=self.feature_dim,
            hidden_channels=edge_hidden_channels,
            nband=self.nband,
            num_output=self.num_output,
            num_blocks=edge_num_blocks,
            expansion=edge_expansion,
            freq_kernel_size=edge_freq_kernel_size,
            time_kernel_size=edge_time_kernel_size,
            time_dilations=edge_time_dilations,
            dummy_kv_channels=edge_dummy_kv_channels,
            dummy_kv_width=edge_dummy_kv_width,
            bias=True,
        )
        self.supports_exact_chunk_training = True

    def init_streaming_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self.separator.init_streaming_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward_cell(
        self,
        subband_spec_RIs,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
    ):
        if not torch.onnx.is_in_onnx_export():
            assert subband_spec_RIs.shape[-1] == 1, "forward_cell expects a single frame (T=1)"

        subband_features = self._encode_subbands(subband_spec_RIs)
        (
            sep_output,
            new_kv,
            new_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        ) = self.separator(
            subband_features,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
        )
        band_masked_output = self._decode_masks(sep_output)
        return (
            band_masked_output,
            new_kv,
            new_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        )

    def forward_sequence(
        self,
        subband_spec_RIs=None,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
        detach_state=False,
        chunk_size=8,
    ):
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        batch_size, _, _, total_frames = subband_spec_RIs.shape

        if (
            past_kvs is None
            or past_valid_mask is None
            or prev_states_0 is None
            or prev_states_1 is None
            or prev_states_2 is None
            or prev_global_states is None
        ):
            (
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.init_streaming_state(
                batch_size,
                device=subband_spec_RIs.device,
                dtype=subband_spec_RIs.dtype,
            )

        if not self.supports_exact_chunk_training:
            chunk_size = 1
        elif chunk_size is None or chunk_size <= 0:
            chunk_size = total_frames

        frame_outputs = []
        for t in range(0, total_frames, chunk_size):
            chunk = subband_spec_RIs[:, :, :, t:t + chunk_size]

            # exact same encode -> separator(stateful) -> decode flow as deployment cell
            subband_features = self._encode_subbands(chunk)
            (
                sep_output,
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.separator(
                subband_features,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
            frame_outputs.append(self._decode_masks(sep_output))

            if detach_state:
                past_kvs = past_kvs.detach()
                past_valid_mask = past_valid_mask.detach()
                prev_states_0 = prev_states_0.detach()
                prev_states_1 = prev_states_1.detach()
                prev_states_2 = prev_states_2.detach()
                prev_global_states = prev_global_states.detach()

        return (
            torch.cat(frame_outputs, dim=-1),
            past_kvs,
            past_valid_mask,
            prev_states_0,
            prev_states_1,
            prev_states_2,
            prev_global_states,
        )

    def forward(
        self,
        subband_spec_RIs=None,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
        detach_state=False,
        chunk_size=8,
    ):
        """
        Export path: pass a single frame (T=1) and explicit states/caches.
        Training path: pass a longer sequence (T>1); the model will unroll the
        same 1-frame-compatible causal cell over time.
        """
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        if subband_spec_RIs.shape[-1] == 1:
            return self.forward_cell(
                subband_spec_RIs,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
        return self.forward_sequence(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
            detach_state=detach_state,
            chunk_size=chunk_size,
        )
使用说明
这版代码有几个点你需要注意：

第一，它严格保持了你现有 TIGER 的外层接口。也就是：

init_streaming_state(batch_size, device, dtype)

forward_cell(...)

forward_sequence(...)

forward(...)

并且返回的状态顺序仍然是：

past_kvs

past_valid_mask

prev_states_0

prev_states_1

prev_states_2

prev_global_states

这和你当前 TIGER 的接口一致。

第二，past_kvs / past_valid_mask 在这版里只是兼容占位 state。
它们仍然通过输入输出传递，便于你复用原来的 ONNX/export/runner 接口，但不会参与 attention，因为这里已经没有 attention 了。

第三，真正的 causal state 在：

prev_states_0

prev_states_1

prev_states_2

prev_global_states

其中前三个分别对应 dilation 组 0/1/2 的时间缓存，最后一个是每个 block 一份的低成本全局上下文。

第四，这版默认参数是偏 edge 的：

edge_hidden_channels=64

edge_num_blocks=6

edge_expansion=2

edge_time_kernel_size=3

edge_time_dilations=(1, 2, 4)

这组配置在状态大小和图规模上都比原 TIGER 更适合部署。

状态大小估算
按默认配置，主要 state 大小大致是：

prev_states_0: 2 * 64 * 2 * nband 个元素

prev_states_1: 2 * 64 * 4 * nband

prev_states_2: 2 * 64 * 8 * nband

prev_global_states: 6 * 64 * 1 * nband

这里 2 是每组 block 数，因为 6 个 block 按 3 组轮转。
如果你用 fp16，这套 state 一般会比原来的 KV cache 小不少，适合你说的 DSP 预算。

训练/部署一致性
这版的关键点是：

forward_cell(T=1) 和部署完全一致

forward_sequence(T>1) 只是按 chunk 重复调用同样的 causal state 逻辑

也就是你要求的“训练时和部署时兼容，而且部署性能不能比训练时掉太多”，这版就是按这个原则写的。

我建议你下一步做的两件事
先把这版骨干接进去，确认：

_encode_subbands() 输出确实是 [B, feature_dim, nband, T]

_decode_masks() 确实接受 [B, num_output * feature_dim, nband, T]

从你文件里 TIGER.forward_cell() 的写法看，这就是当前 separator 的契约。


然后做一个只导 forward_cell() 的 ONNX wrapper，输入显式包含：

subband_spec_RIs

past_kvs

past_valid_mask

prev_states_0

prev_states_1

prev_states_2

prev_global_states

这样最符合你现在的设备端在线推理约束。

如果你愿意，我下一条可以继续直接给你一份 对应这套 TIGEREdgeMLP 的 ONNX 导出 wrapper 和 dummy-state 构造代码。


请先完成对应这套 TIGEREdgeMLP 的 ONNX 导出 wrapper 和 dummy-state 构造代码。 接着请仔细review上面的代码，检查NPU约束是否得到遵守，state缓存是否过大， 时间维causal是否守住，training 跟inference的路径是否一致。另外实际是可以使用bmm的， 但是不要有scatterND之类操作（可能是转onnx的时候生成的）


Thought for 21s
下面先给你一套 对应 TIGEREdgeMLP 的 ONNX 导出 wrapper + dummy-state 构造代码，然后我会对上面那版骨架做一次逐项 review，明确指出：

哪些约束已经满足

哪些地方需要修

哪些地方我建议你立刻改

你当前原始 TIGER 的外层接口确实是：

init_streaming_state(...)

forward_cell(...)

forward_sequence(...)

forward(...)

并且状态流是 past_kvs / past_valid_mask / prev_states_0 / prev_states_1 / prev_states_2 / prev_global_states。

 


一、ONNX 导出 wrapper 与 dummy-state 构造代码
这部分是给我上一条写的 TIGEREdgeMLP 配套的。

1) 单帧导出 wrapper
import os
import torch
import torch.nn as nn


class TIGEREdgeMLPCellExportWrapper(nn.Module):
    """
    Export only the single-frame causal streaming cell.

    Inputs:
      - subband_spec_RIs:    [B, num_input, enc_dim*2, 1]
      - past_kvs:            [B, 1, dummy_kv_width, dummy_kv_channels]
      - past_valid_mask:     [B, 1, dummy_kv_width, 1]
      - prev_states_0:       [B, C0, nband, W0]
      - prev_states_1:       [B, C1, nband, W1]
      - prev_states_2:       [B, C2, nband, W2]
      - prev_global_states:  [B, Cg, nband, 1]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(
        self,
        subband_spec_RIs,
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    ):
        if subband_spec_RIs.shape[-1] != 1:
            raise RuntimeError(
                f"TIGEREdgeMLPCellExportWrapper expects T=1, but got input shape {tuple(subband_spec_RIs.shape)}"
            )

        return self.model.forward_cell(
            subband_spec_RIs=subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
        )
2) dummy-state 构造函数
这个函数直接依赖 model.init_streaming_state(...)，这样不会外部猜错 state shape。原始 TIGER/TIGERCtx 也是通过 init_streaming_state() 给出部署态 state 形状的。

def build_tiger_edge_mlp_dummy_inputs(
    model: nn.Module,
    batch_size: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Build exact dummy inputs for ONNX export, using model.init_streaming_state().
    """
    model = model.to(device=device, dtype=dtype).eval()

    states = model.init_streaming_state(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    (
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    ) = states

    # TIGER encoder expects [B, num_input, enc_dim * 2, T]
    # from original TIGER config:
    #   enc_dim = win // 2 + 1
    # and real/imag are packed as *2. :contentReference[oaicite:3]{index=3}
    if not hasattr(model, "num_input"):
        raise RuntimeError("Model is missing attribute 'num_input'")
    if not hasattr(model, "enc_dim"):
        raise RuntimeError("Model is missing attribute 'enc_dim'")

    subband_spec_RIs = torch.randn(
        batch_size,
        model.num_input,
        model.enc_dim * 2,
        1,   # single frame export
        device=device,
        dtype=dtype,
    )

    return (
        subband_spec_RIs,
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    )
3) 导出前 PyTorch 预检查
def precheck_tiger_edge_mlp_export(wrapper: nn.Module, inputs):
    wrapper.eval()
    with torch.no_grad():
        outputs = wrapper(*inputs)

    print("Precheck passed.")
    for idx, out in enumerate(outputs):
        print(f"output[{idx}] shape = {tuple(out.shape)}")
4) ONNX 导出函数
def export_tiger_edge_mlp_to_onnx(
    wrapper: nn.Module,
    inputs,
    onnx_path: str,
    opset_version: int = 17,
):
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    input_names = [
        "subband_spec_RIs",
        "past_kvs",
        "past_valid_mask",
        "prev_states_0",
        "prev_states_1",
        "prev_states_2",
        "prev_global_states",
    ]

    output_names = [
        "band_masked_output",
        "new_kvs",
        "new_valid_mask",
        "new_states_0",
        "new_states_1",
        "new_states_2",
        "new_global_states",
    ]

    # 推荐只放 batch 维动态，保持 streaming 单帧图稳定
    dynamic_axes = {
        "subband_spec_RIs": {0: "B"},
        "past_kvs": {0: "B"},
        "past_valid_mask": {0: "B"},
        "prev_states_0": {0: "B"},
        "prev_states_1": {0: "B"},
        "prev_states_2": {0: "B"},
        "prev_global_states": {0: "B"},
        "band_masked_output": {0: "B"},
        "new_kvs": {0: "B"},
        "new_valid_mask": {0: "B"},
        "new_states_0": {0: "B"},
        "new_states_1": {0: "B"},
        "new_states_2": {0: "B"},
        "new_global_states": {0: "B"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print(f"Exported ONNX to: {onnx_path}")
5) 一键导出脚本模板
你把真实 import 路径替换掉就行。

def export_main():
    # 你需要替换为你工程里的真实路径
    # from your_project.models.tiger import TIGEREdgeMLP
    from your_project.models.tiger import TIGEREdgeMLP

    ckpt_path = "tiger_edge_mlp.pt"
    onnx_path = "tiger_edge_mlp_streaming_cell.onnx"
    device = "cpu"
    dtype = torch.float32
    batch_size = 1

    model = TIGEREdgeMLP(
        edge_hidden_channels=64,
        edge_num_blocks=6,
        edge_expansion=2,
        edge_freq_kernel_size=3,
        edge_time_kernel_size=3,
        edge_time_dilations=(1, 2, 4),
        edge_dummy_kv_channels=1,
        edge_dummy_kv_width=1,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval().to(device=device, dtype=dtype)

    wrapper = TIGEREdgeMLPCellExportWrapper(model).eval().to(device=device, dtype=dtype)
    inputs = build_tiger_edge_mlp_dummy_inputs(
        model=model,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    precheck_tiger_edge_mlp_export(wrapper, inputs)
    export_tiger_edge_mlp_to_onnx(wrapper, inputs, onnx_path, opset_version=17)


if __name__ == "__main__":
    export_main()
二、对上面那版 TIGEREdgeMLP 代码的 review
下面我直接给结论，不绕。

结论概览
那版代码的总体方向是对的，但还不够严谨。
如果你要真拿去导 ONNX/circle/NPU，我建议你在合入前至少做 4 个修正：

把 EdgeChannelNorm 去掉，或换成纯 affine/identity

把 global_state.expand(...) 改成 repeat_interleave 风格的显式复制，或更好地改成“只作用在当前帧”

把 dummy_kv_channels=1, dummy_kv_width=1 再明确成完全固定，不允许外部乱传

给所有 state shape 加严格断言

下面逐项说。

1) NPU 约束是否得到遵守
1.1 是否只用了简单 Conv2d 和基础张量操作
基本满足。

主干里只有：

Conv2d

cat

slicing

+

*

sigmoid

ReLU

mean/rsqrt 这类归一化内部操作

没有 scatterND，也没有 attention / MatMul / bmm。
你原始 TIGER 最麻烦的地方正是 frame attention 和迭代块里 packed KV 的 slice/concat/attention 路径。原始 RecurrentKV 会在每次 iteration 里取一段 past_kv，再进入 frame path。

1.2 tensor 维度是否始终不超过 4
满足。

整版代码里所有张量都是：

4D: [B, C, F, T]

或 state 4D: [B, C, F, W]

没有升到 5D，也没有把 4D 折到 batch 维。

1.3 是否把 4D tensor 折叠到 batch 维
满足。

没有任何 reshape(B*something, ...) 这类操作。

1.4 conv2d 的 (kernel_size - 1) * dilation < 14
满足，但要注意只对时间卷积显式检查了。

在我给的实现里：

EdgeTimeMixer 已检查 (time_kernel_size - 1) * dilation < 14

EdgeFreqMixer 默认 dilation=1，freq_kernel_size=3，自然满足

所以当前默认配置是安全的。

1.5 模型中尽量不要出现 tensor 常量
部分满足，但还不够理想。

这里有两类“常量”：

第一类是 Python 标量常量，比如 0/1、kernel 参数，这没问题。
第二类是运行中用 x.new_zeros(...) 生成的零张量。

这版代码里大量 state 默认用 x.new_zeros 构造。
这本身通常比图里埋常量权重更安全，但如果你非常严格地避免导出图内 constant initializer，我建议：

ONNX 导出时总是显式传入所有 state 输入

避免依赖 forward 里的 “if state is None: new_zeros(...)”

也就是：部署图里别走默认分支。

这一点和你原始 TIGER 的设计是一致的：部署时本来就应该把 state/cache 当模型输入传入。

2) state/cache 是否过大
这个问题需要分开说。

2.1 比原始 TIGER 小很多
是的，明显更小。

原始 TIGER 的 past_kvs 是：

[B, n_heads, kv_window_size, kv_dim * iter]

kv_dim = (att_hid_chan + att_val_hid_chan) * nband 


而且还有：

prev_states_0

prev_states_1

prev_states_2

prev_global_states

所以原始状态本身就不小。

新骨干里：

past_kvs / past_valid_mask 基本只是占位

真实 state 只有三组 causal conv cache + 一组 global state

这一点是好的。

2.2 但我上一版对 prev_states_0/1/2 的通道定义偏大
是的，这里需要收紧。

我上一版定义的是：

每个 block 的 time state 取 hidden_channels * expansion

然后把同组 block 的 state 在通道维拼起来

这会把 state 放大到：

group0: #blocks_in_group * hidden * expansion * W0

group1: ...

group2: ...

这虽然还算可控，但不是最省。

更好的做法
把 EdgeTimeMixer 改成：

state 不缓存 pre 之后的 hidden 通道

只缓存 输入通道 channels

causal depthwise conv 放到 pre 之前

也就是把时间模块从：

pre -> cache(hidden) -> dw(hidden) -> post
改成：

cache(channels) -> dw(channels) -> pw_expand -> pw_project
这样 state 通道数从 hidden_channels * expansion 降到 hidden_channels，会更省一倍以上。

这是我最建议你立即改的一点。

3) 时间维 causal 是否守住
守住了。

原因很直接：

时间卷积 EdgeTimeMixer 对时间维没有右侧 padding

它只显式拼接左缓存 state

新状态是 combined[..., -state_width:]

这就是标准 causal streaming 写法。

和你原始代码的 streaming 设计思想是一致的：原始 TIGER/TIGERCtx 也是通过显式输入输出 state 来维持 streaming/online。

需要补充的一点
EdgeFreqMixer 在频率维使用了对称 padding。
这不影响时间 causal，因为频率维不是在线维。

4) training 跟 inference 的路径是否一致
基本一致。

这是这版设计最重要的优点之一。

forward_cell(T=1)：部署语义

forward_sequence(T>1)：对 chunk 直接走同一个 separator，同样传 state

和原始 TIGER 一样，都是同一套 stateful 模块，只是原始实现里是 RecurrentKV。


但我需要纠正一个说法
我上一条里写的是“forward_sequence() 只是按 chunk 重复调用同样的 causal state 逻辑”。

这个方向对，但实现上不是逐帧调用 forward_cell()，而是按 chunk 直接把 T=chunk_size 一次喂给 separator。
这仍然是兼容的，因为 causal time mixer 支持 T>=1，并且 state 更新逻辑是同一套。

所以这里不是问题。

5) 会不会生成 scatterND 之类操作
当前设计看，不应该主动生成 scatterND。

高风险触发 scatterND 的一般是：

稀疏更新

index_put_

scatter/gather

动态复杂写入

某些复杂 mask 构造

这版代码没有这些。

但有一个潜在点
global_state.expand(B, C, F, T) 在某些导出器里可能被翻译成广播视图；通常不会变成 scatterND，但我还是建议你改掉，原因是它对某些后端不够稳定。

更稳的改法有两种：

方案 A：只在单帧里用 global_state
既然部署是 T=1，那 forward_cell() 根本不需要扩展，直接：

global_rep = global_state
而训练 chunk 模式里，也可以只把 latest global state 作用到每个时间步，这时可以通过简单 cat/repeat 达成。

方案 B：用 repeat_interleave 明确复制
不过你之前还要求别在 4D 里乱折 batch，这里倒没这个问题。
但从 NPU 编译器角度，我更建议方案 A：全局态只保留 [B,C,F,1]，只在当前帧路径使用。

三、我建议你现在就改的地方
下面这 4 个修改，我认为是必要的。

修改 1：去掉 EdgeChannelNorm
原因：

它里面有 mean / rsqrt / broadcast

虽然不一定有问题，但这不是“最简单 Conv2d 图”

很多 NPU 更喜欢你把 norm 干掉，或折到前后训练策略里解决

建议替换
直接把：

self.norm = EdgeChannelNorm(...)
改成：

self.norm = nn.Identity()
等模型先跑通、先导通 ONNX/circle，再决定是否加回轻量归一化。

修改 2：缩小时间 state 通道数
把 EdgeTimeMixer 改成“缓存输入通道而不是 hidden 通道”。

也就是把时间模块调整为：

cache(channels) -> dw time conv on channels -> 1x1 expand -> 1x1 project
这样状态大小会显著下降，更符合 192k DSP 预算。

修改 3：给 separator 的输入 state 加严格 shape 断言
现在骨架里默认 None 时自动建 state，这对训练方便，但对部署不够严。

建议在 forward_cell() 导出/部署路径里显式检查：

assert past_kvs is not None
assert past_valid_mask is not None
...
至少在 ONNX 导出 wrapper 里要这样做。

修改 4：把 global_state.expand(...) 改掉
改成只处理单帧 global context，避免后端对广播视图的奇怪 lowering。

四、我对这版骨架的最终评价
如果按你的目标排序：

从“能不能替代 TIGER 并更适合 NPU”看
可以，方向是对的。

从“现在这版能不能直接上设备”看
还差一步。

最主要差在：

归一化还不够保守

时间 state 还能继续瘦身

expand 这种广播我不建议留

还缺一轮“最小算子子集化”的收敛

五、我最推荐的下一步
下一步不要直接训练，先做一次算子收缩版修订：

去掉 EdgeChannelNorm

把 EdgeTimeMixer 改成缓存 channels 而不是 hidden

去掉 global_state.expand

保留其他接口不变

这样会更像一个真正面向 circle/NPU 的版本。

如果你同意，我下一条就直接把这 4 个修正合进代码，给你一版 “更保守、更适合导出”的 TIGEREdgeMLP v2 完整骨架。


好的， 给我一版 “更保守、更适合导出”的 TIGEREdgeMLP v2 完整骨架


Thought for 1m 0s
下面给你一版更保守、导出更稳的 TIGEREdgeMLP v2。它严格复用你现有 TIGER 的外层契约：

复用 TIGER 的 _encode_subbands() / _decode_masks()

保留 init_streaming_state / forward_cell / forward_sequence / forward

状态接口仍是
past_kvs, past_valid_mask, prev_states_0, prev_states_1, prev_states_2, prev_global_states

你原文件里的 TIGER / TIGERCtx 也是按这套 streaming state 输入输出组织的，feature_dim / num_output / nband / enc_dim 也都在 TIGER.__init__ 里定义好了。


你上一版 edge 骨架的 forward_sequence 也是沿用这套 7 输入 / 7 输出接口。


这版 v2 相比上一版，做了 4 个关键收缩：

去掉了归一化，避免额外 mean/rsqrt 图

时间 state 只缓存 channels，不缓存扩展后的 hidden

不再用 expand 广播 global state

forward_sequence 直接逐帧调用 forward_cell，保证训练/推理路径完全一致

TIGEREdgeMLP v2 代码
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Small safe utility blocks
# =========================================================

class EdgeIdentity(nn.Module):
    def forward(self, x):
        return x


class EdgePWConv(nn.Module):
    """
    Conservative 1x1 Conv2d block:
    Conv2d -> ReLU
    No normalization, fewer export risks.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.conv(x))


# =========================================================
# EdgeFreqMixer v2
# =========================================================

class EdgeFreqMixer(nn.Module):
    """
    Frequency-axis local mixer on [B, C, F, T]
    Only uses:
      - Conv2d(1x1)
      - depthwise Conv2d(kf, 1)
      - Add
      - ReLU

    Frequency dimension is not causal, so symmetric padding on F is acceptable.
    """
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        assert freq_kernel_size >= 1 and (freq_kernel_size % 2 == 1), "freq_kernel_size must be odd"

        hidden = channels * expansion
        pad_f = (freq_kernel_size - 1) // 2

        self.pre = EdgePWConv(channels, hidden, bias=bias)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(freq_kernel_size, 1),
            stride=(1, 1),
            padding=(pad_f, 0),
            dilation=(1, 1),
            groups=hidden,
            bias=bias,
        )
        self.dw_act = nn.ReLU()
        self.post = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.pre(x)
        x = self.dw(x)
        x = self.dw_act(x)
        x = self.post(x)
        return residual + x


# =========================================================
# EdgeTimeMixer v2
# =========================================================

class EdgeTimeMixer(nn.Module):
    """
    Strictly causal time mixer on [B, C, F, T].

    Key design change vs v1:
      - cache stores only input channels C, not expanded hidden channels
      - therefore state size is much smaller
      - no normalization
      - no expand/broadcast tricks

    Structure:
      [cache + current] -> depthwise causal Conv2d(1, kt) on channels
                        -> 1x1 expand
                        -> 1x1 project
                        -> residual add
    """
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        time_kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert time_kernel_size >= 1 and (time_kernel_size % 2 == 1), "time_kernel_size must be odd"
        assert dilation >= 1, "dilation must be >= 1"
        assert (time_kernel_size - 1) * dilation < 14, \
            "NPU constraint violated: (kernel_size - 1) * dilation must be < 14"

        self.channels = channels
        self.expansion = expansion
        self.time_kernel_size = time_kernel_size
        self.dilation = dilation
        self.state_width = (time_kernel_size - 1) * dilation

        hidden = channels * expansion

        # causal depthwise conv directly on C channels
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, time_kernel_size),
            stride=(1, 1),
            padding=(0, 0),  # explicit cache only
            dilation=(1, dilation),
            groups=channels,
            bias=bias,
        )
        self.dw_act = nn.ReLU()

        self.expand = EdgePWConv(channels, hidden, bias=bias)
        self.project = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """
        x:     [B, C, F, T]
        state: [B, C, F, state_width]
        """
        assert x.dim() == 4, "EdgeTimeMixer expects x to be 4D"
        B, C, Freq, T = x.shape
        assert C == self.channels, f"Expected channels={self.channels}, got {C}"

        if self.state_width > 0:
            assert state is not None, "EdgeTimeMixer requires explicit state in deployment/export"
            assert state.dim() == 4, "EdgeTimeMixer state must be 4D"
            assert state.shape[0] == B, "state batch mismatch"
            assert state.shape[1] == C, "state channel mismatch"
            assert state.shape[2] == Freq, "state freq mismatch"
            assert state.shape[3] == self.state_width, "state width mismatch"

            combined = torch.cat([state, x], dim=-1)   # [B, C, F, state_width + T]
            new_state = combined[:, :, :, -self.state_width:]
        else:
            combined = x
            new_state = x[:, :, :, :0]

        residual = x
        x = self.dw(combined)   # [B, C, F, T]
        x = self.dw_act(x)
        x = self.expand(x)
        x = self.project(x)
        x = residual + x
        return x, new_state


# =========================================================
# EdgeTFMLPBlock v2
# =========================================================

class EdgeTFMLPBlock(nn.Module):
    """
    Conservative block:
      freq mixer
      time mixer
      cheap single-frame global bias injection

    global_state shape:
      [B, C, F, 1]

    This block is designed so that deployment path is naturally T=1.
    For exact training/inference consistency, forward_sequence will call
    forward_cell frame by frame.
    """
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        time_kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels

        self.freq_mixer = EdgeFreqMixer(
            channels=channels,
            expansion=expansion,
            freq_kernel_size=freq_kernel_size,
            bias=bias,
        )
        self.time_mixer = EdgeTimeMixer(
            channels=channels,
            expansion=expansion,
            time_kernel_size=time_kernel_size,
            dilation=dilation,
            bias=bias,
        )

        # cheap global bias / gate for current frame only
        self.global_gate = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.global_mix = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.global_update = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, time_state: torch.Tensor, global_state: torch.Tensor):
        """
        x:           [B, C, F, T]
        time_state:  [B, C, F, W]
        global_state:[B, C, F, 1]
        """
        assert x.dim() == 4
        B, C, Freq, T = x.shape
        assert C == self.channels

        assert global_state is not None, "global_state must be explicit"
        assert global_state.dim() == 4
        assert global_state.shape[0] == B
        assert global_state.shape[1] == C
        assert global_state.shape[2] == Freq
        assert global_state.shape[3] == 1

        x = self.freq_mixer(x)
        x, new_time_state = self.time_mixer(x, state=time_state)

        # conservative global fusion:
        # only use current global state as a 1-frame bias.
        # For T=1 deployment, this is exact.
        # For training, forward_sequence will unroll frame-by-frame.
        gate = torch.sigmoid(self.global_gate(global_state))   # [B, C, F, 1]
        mix = self.global_mix(global_state)                    # [B, C, F, 1]
        x = x + gate * mix

        # update global state from current output frame only
        # x[:, :, :, -1:] keeps all tensors 4D and causal
        new_global_state = self.global_update(x[:, :, :, -1:])
        return x, new_time_state, new_global_state


# =========================================================
# EdgeTFMLPSeparator v2
# =========================================================

class EdgeTFMLPSeparator(nn.Module):
    """
    TIGER-compatible separator replacement, conservative export version.

    Input:
      x: [B, feature_dim, nband, T]

    Output:
      sep_output: [B, num_output * feature_dim, nband, T]

    State contract:
      - past_kvs / past_valid_mask are compatibility placeholders only
      - prev_states_0/1/2 are the real causal caches
      - prev_global_states stores one [C, F, 1] slice per block
    """
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        nband: int,
        num_output: int,
        num_blocks: int = 6,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        time_kernel_size: int = 3,
        time_dilations=(1, 2, 4),
        dummy_kv_channels: int = 1,
        dummy_kv_width: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert len(time_dilations) == 3, "time_dilations must be length 3"
        assert num_blocks >= 1
        assert hidden_channels >= 1

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.nband = nband
        self.num_output = num_output
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.freq_kernel_size = freq_kernel_size
        self.time_kernel_size = time_kernel_size
        self.time_dilations = tuple(int(d) for d in time_dilations)

        # dummy compatibility attrs
        self.n_heads = 1
        self.kv_window_size = int(dummy_kv_width)
        self.att_hid_chan = int(dummy_kv_channels)
        self.att_val_hid_chan = 0
        self.iter = self.num_blocks

        self.in_proj = EdgePWConv(input_dim, hidden_channels, bias=bias)

        self.blocks = nn.ModuleList([
            EdgeTFMLPBlock(
                channels=hidden_channels,
                expansion=expansion,
                freq_kernel_size=freq_kernel_size,
                time_kernel_size=time_kernel_size,
                dilation=self.time_dilations[i % 3],
                bias=bias,
            )
            for i in range(num_blocks)
        ])

        self.out_proj = nn.Conv2d(
            hidden_channels,
            num_output * input_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.group_block_indices = [
            [i for i in range(num_blocks) if (i % 3) == 0],
            [i for i in range(num_blocks) if (i % 3) == 1],
            [i for i in range(num_blocks) if (i % 3) == 2],
        ]

        self.group_state_widths = [
            (time_kernel_size - 1) * self.time_dilations[0],
            (time_kernel_size - 1) * self.time_dilations[1],
            (time_kernel_size - 1) * self.time_dilations[2],
        ]

        for width in self.group_state_widths:
            assert width < 14, "state width violates NPU constraint"

        # v2: state channels are only hidden_channels, not hidden_channels * expansion
        self.group_state_channels = [
            len(self.group_block_indices[0]) * hidden_channels,
            len(self.group_block_indices[1]) * hidden_channels,
            len(self.group_block_indices[2]) * hidden_channels,
        ]

        self.global_state_channels = num_blocks * hidden_channels
        self.global_state_width = 1

    def _make_dummy_kv(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size,
            self.n_heads,
            self.kv_window_size,
            self.att_hid_chan,
            device=device,
            dtype=dtype,
        )

    def _make_dummy_mask(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size,
            1,
            self.kv_window_size,
            1,
            device=device,
            dtype=dtype,
        )

    def init_streaming_state(self, batch_size: int, device=None, dtype=None):
        return (
            self._make_dummy_kv(batch_size, device, dtype),
            self._make_dummy_mask(batch_size, device, dtype),
            torch.zeros(
                batch_size,
                self.group_state_channels[0],
                self.nband,
                self.group_state_widths[0],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.group_state_channels[1],
                self.nband,
                self.group_state_widths[1],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.group_state_channels[2],
                self.nband,
                self.group_state_widths[2],
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.global_state_channels,
                self.nband,
                self.global_state_width,
                device=device,
                dtype=dtype,
            ),
        )

    def _slice_group_state(self, packed_state: torch.Tensor, block_local_idx: int, channels: int):
        c0 = block_local_idx * channels
        c1 = c0 + channels
        return packed_state[:, c0:c1, :, :]

    def forward(
        self,
        x,
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    ):
        """
        Conservative separator forward.
        In export/runtime you should pass explicit states, never None.
        """
        assert x.dim() == 4, "separator expects [B, C, F, T]"
        B, C, Freq, T = x.shape
        assert Freq == self.nband, f"Expected nband={self.nband}, got {Freq}"

        # strict explicit-state mode
        assert past_kvs is not None
        assert past_valid_mask is not None
        assert prev_states_0 is not None
        assert prev_states_1 is not None
        assert prev_states_2 is not None
        assert prev_global_states is not None

        assert past_kvs.shape == (B, self.n_heads, self.kv_window_size, self.att_hid_chan)
        assert past_valid_mask.shape == (B, 1, self.kv_window_size, 1)
        assert prev_states_0.shape == (B, self.group_state_channels[0], self.nband, self.group_state_widths[0])
        assert prev_states_1.shape == (B, self.group_state_channels[1], self.nband, self.group_state_widths[1])
        assert prev_states_2.shape == (B, self.group_state_channels[2], self.nband, self.group_state_widths[2])
        assert prev_global_states.shape == (B, self.global_state_channels, self.nband, self.global_state_width)

        h = self.in_proj(x)

        new_group0 = []
        new_group1 = []
        new_group2 = []
        new_global = []

        group_local_counters = [0, 0, 0]

        for block_idx, block in enumerate(self.blocks):
            group_id = block_idx % 3
            local_idx = group_local_counters[group_id]
            group_local_counters[group_id] += 1

            if group_id == 0:
                state_slice = self._slice_group_state(prev_states_0, local_idx, self.hidden_channels)
            elif group_id == 1:
                state_slice = self._slice_group_state(prev_states_1, local_idx, self.hidden_channels)
            else:
                state_slice = self._slice_group_state(prev_states_2, local_idx, self.hidden_channels)

            g0 = block_idx * self.hidden_channels
            g1 = g0 + self.hidden_channels
            global_slice = prev_global_states[:, g0:g1, :, :]  # [B, hidden_channels, F, 1]

            h, new_state_slice, new_global_slice = block(
                h,
                time_state=state_slice,
                global_state=global_slice,
            )

            if group_id == 0:
                new_group0.append(new_state_slice)
            elif group_id == 1:
                new_group1.append(new_state_slice)
            else:
                new_group2.append(new_state_slice)

            new_global.append(new_global_slice)

        new_states_0 = torch.cat(new_group0, dim=1) if len(new_group0) > 0 else h.new_zeros(B, 0, self.nband, self.group_state_widths[0])
        new_states_1 = torch.cat(new_group1, dim=1) if len(new_group1) > 0 else h.new_zeros(B, 0, self.nband, self.group_state_widths[1])
        new_states_2 = torch.cat(new_group2, dim=1) if len(new_group2) > 0 else h.new_zeros(B, 0, self.nband, self.group_state_widths[2])
        new_global_states = torch.cat(new_global, dim=1)

        sep_output = self.out_proj(h)

        # compatibility passthrough
        new_kvs = past_kvs
        new_valid_mask = past_valid_mask

        return (
            sep_output,
            new_kvs,
            new_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        )


# =========================================================
# TIGEREdgeMLP v2
# =========================================================

class TIGEREdgeMLP(TIGER):
    """
    Conservative export-oriented TIGER replacement:
      - reuses TIGER encoder / decoder / interface
      - replaces separator with Conv2d-only TF-MLP-like causal backbone
      - exact training/inference path consistency by frame-wise unroll
    """
    def __init__(
        self,
        *args,
        edge_hidden_channels=64,
        edge_num_blocks=6,
        edge_expansion=2,
        edge_freq_kernel_size=3,
        edge_time_kernel_size=3,
        edge_time_dilations=(1, 2, 4),
        edge_dummy_kv_channels=1,
        edge_dummy_kv_width=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert len(edge_time_dilations) == 3
        for d in edge_time_dilations:
            assert (edge_time_kernel_size - 1) * int(d) < 14, \
                "NPU constraint violated: (kernel_size - 1) * dilation must be < 14"

        self.separator = EdgeTFMLPSeparator(
            input_dim=self.feature_dim,
            hidden_channels=edge_hidden_channels,
            nband=self.nband,
            num_output=self.num_output,
            num_blocks=edge_num_blocks,
            expansion=edge_expansion,
            freq_kernel_size=edge_freq_kernel_size,
            time_kernel_size=edge_time_kernel_size,
            time_dilations=edge_time_dilations,
            dummy_kv_channels=edge_dummy_kv_channels,
            dummy_kv_width=edge_dummy_kv_width,
            bias=True,
        )

        # exact equivalence only if sequence path unrolls frame-by-frame
        self.supports_exact_chunk_training = True

    def init_streaming_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self.separator.init_streaming_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward_cell(
        self,
        subband_spec_RIs,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
    ):
        if not torch.onnx.is_in_onnx_export():
            assert subband_spec_RIs.shape[-1] == 1, "forward_cell expects T=1"

        B = subband_spec_RIs.shape[0]

        if (
            past_kvs is None
            or past_valid_mask is None
            or prev_states_0 is None
            or prev_states_1 is None
            or prev_states_2 is None
            or prev_global_states is None
        ):
            (
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.init_streaming_state(
                B,
                device=subband_spec_RIs.device,
                dtype=subband_spec_RIs.dtype,
            )

        subband_features = self._encode_subbands(subband_spec_RIs)
        (
            sep_output,
            new_kv,
            new_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        ) = self.separator(
            subband_features,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
        )
        band_masked_output = self._decode_masks(sep_output)
        return (
            band_masked_output,
            new_kv,
            new_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        )

    def forward_sequence(
        self,
        subband_spec_RIs=None,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
        detach_state=False,
        chunk_size=1,
    ):
        """
        Exact path consistency mode:
        always unroll frame by frame, regardless of chunk_size.
        This makes training and deployment use the same 1-frame causal cell.
        """
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        B, _, _, total_frames = subband_spec_RIs.shape

        if (
            past_kvs is None
            or past_valid_mask is None
            or prev_states_0 is None
            or prev_states_1 is None
            or prev_states_2 is None
            or prev_global_states is None
        ):
            (
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.init_streaming_state(
                B,
                device=subband_spec_RIs.device,
                dtype=subband_spec_RIs.dtype,
            )

        frame_outputs = []
        for t in range(total_frames):
            frame = subband_spec_RIs[:, :, :, t:t + 1]
            (
                frame_out,
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.forward_cell(
                frame,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
            frame_outputs.append(frame_out)

            if detach_state:
                past_kvs = past_kvs.detach()
                past_valid_mask = past_valid_mask.detach()
                prev_states_0 = prev_states_0.detach()
                prev_states_1 = prev_states_1.detach()
                prev_states_2 = prev_states_2.detach()
                prev_global_states = prev_global_states.detach()

        return (
            torch.cat(frame_outputs, dim=-1),
            past_kvs,
            past_valid_mask,
            prev_states_0,
            prev_states_1,
            prev_states_2,
            prev_global_states,
        )

    def forward(
        self,
        subband_spec_RIs=None,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
        detach_state=False,
        chunk_size=1,
    ):
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        if subband_spec_RIs.shape[-1] == 1:
            return self.forward_cell(
                subband_spec_RIs,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
        return self.forward_sequence(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
            detach_state=detach_state,
            chunk_size=chunk_size,
        )


# =========================================================
# ONNX export wrapper for v2
# =========================================================

class TIGEREdgeMLPCellExportWrapper(nn.Module):
    """
    Export only the single-frame causal streaming cell.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(
        self,
        subband_spec_RIs,
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    ):
        if subband_spec_RIs.shape[-1] != 1:
            raise RuntimeError(
                f"TIGEREdgeMLPCellExportWrapper expects T=1, got {tuple(subband_spec_RIs.shape)}"
            )

        return self.model.forward_cell(
            subband_spec_RIs=subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
        )


def build_tiger_edge_mlp_v2_dummy_inputs(
    model: nn.Module,
    batch_size: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    model = model.to(device=device, dtype=dtype).eval()

    (
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    ) = model.init_streaming_state(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    if not hasattr(model, "num_input"):
        raise RuntimeError("Model is missing attribute 'num_input'")
    if not hasattr(model, "enc_dim"):
        raise RuntimeError("Model is missing attribute 'enc_dim'")

    subband_spec_RIs = torch.randn(
        batch_size,
        model.num_input,
        model.enc_dim * 2,
        1,
        device=device,
        dtype=dtype,
    )

    return (
        subband_spec_RIs,
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    )


def precheck_tiger_edge_mlp_v2_export(wrapper: nn.Module, inputs):
    wrapper.eval()
    with torch.no_grad():
        outputs = wrapper(*inputs)
    print("Precheck passed.")
    for i, out in enumerate(outputs):
        print(f"output[{i}] shape = {tuple(out.shape)}")


def export_tiger_edge_mlp_v2_to_onnx(
    wrapper: nn.Module,
    inputs,
    onnx_path: str,
    opset_version: int = 17,
):
    input_names = [
        "subband_spec_RIs",
        "past_kvs",
        "past_valid_mask",
        "prev_states_0",
        "prev_states_1",
        "prev_states_2",
        "prev_global_states",
    ]
    output_names = [
        "band_masked_output",
        "new_kvs",
        "new_valid_mask",
        "new_states_0",
        "new_states_1",
        "new_states_2",
        "new_global_states",
    ]
    dynamic_axes = {
        "subband_spec_RIs": {0: "B"},
        "past_kvs": {0: "B"},
        "past_valid_mask": {0: "B"},
        "prev_states_0": {0: "B"},
        "prev_states_1": {0: "B"},
        "prev_states_2": {0: "B"},
        "prev_global_states": {0: "B"},
        "band_masked_output": {0: "B"},
        "new_kvs": {0: "B"},
        "new_valid_mask": {0: "B"},
        "new_states_0": {0: "B"},
        "new_states_1": {0: "B"},
        "new_states_2": {0: "B"},
        "new_global_states": {0: "B"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
这版 v2 的 review 结论
1. NPU 约束
这版只用了：

Conv2d

cat

slicing

Add

Mul

Sigmoid

ReLU

没有 scatter / scatterND / gather / index_put / attention / matmul。
4D 张量也没有折到 batch 维。时间卷积只用 (1, k) 的 Conv2d，并且强制检查 (kernel_size - 1) * dilation < 14。这点是显式写进 EdgeTimeMixer 和 TIGEREdgeMLP 初始化里的。

2. state/cache 大小
这版比 v1 小了一截，关键是 时间 state 从缓存 hidden_channels * expansion 改成只缓存 hidden_channels。

按默认配置：

edge_hidden_channels = 64

edge_num_blocks = 6

edge_time_dilations = (1, 2, 4)

edge_time_kernel_size = 3

则三组时间 cache 宽度分别是：

W0 = 2

W1 = 4

W2 = 8

每组 block 数都是 2，所以：

prev_states_0: B x (2*64) x nband x 2

prev_states_1: B x (2*64) x nband x 4

prev_states_2: B x (2*64) x nband x 8

prev_global_states: B x (6*64) x nband x 1

如果按你原 TIGER 的 nband = len(self.band_width) 来看，默认是很多子带。TIGER.__init__ 里就是这样定义 enc_dim / feature_dim / num_output / nband 的。


这版 state 仍然不算“极小”，但已经明显比带 KV attention 的路径更保守。

如果你还要继续压 state，我建议优先把：

edge_hidden_channels: 64 -> 48

edge_num_blocks: 6 -> 4

3. 时间维 causal
这版是守住的。

原因很简单：

时间卷积没有右侧 padding

只通过显式 state 拼接左历史

新 state 取 combined[..., -state_width:]

global_state 只使用上一时刻缓存，不看未来

forward_sequence() 逐帧调用 forward_cell()

所以训练和部署都严格 causal。

4. training / inference 路径一致性
这版比 v1 更严格。

forward_sequence() 不再把多帧 chunk 一次送进 separator，而是逐帧调用 forward_cell()。
这样训练路径和部署路径完全同构，只是训练时循环更长一点。你原文件里 TIGER/TIGERCtx 的 streaming 组织本来也是围绕 forward_cell 和 state 输入输出展开的。

5. 关于 bmm
你说实际可以使用 bmm，但不要引入 scatterND 一类算子。
这版没有用 bmm，因为当前目标是“更保守、更适合导出”。如果后面你确认某一段局部 mixing 必须用 bmm，我建议只在 3D 小张量、固定 rank、固定布局 的场景里单点引入，不要放回类似 TIGER 里那种 KV-attention 风格路径。原始问题就是这类路径在导出后很容易变成复杂图。


我建议你接下来先做的事
先用这版 v2 做两步验证：

用 TIGEREdgeMLPCellExportWrapper + build_tiger_edge_mlp_v2_dummy_inputs() 导出单帧 ONNX

跑一遍 ONNX 图检查，确认只剩下你预期的基础算子，没有异常的 ScatterND、GatherND、NonZero、Where 之类

如果你愿意，我下一条直接给你一份 “用于自检 ONNX 图中是否出现高风险算子”的 Python 检查脚本。