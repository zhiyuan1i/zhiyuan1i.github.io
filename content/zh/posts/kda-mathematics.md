---
title: 'KDA（Kimi Delta Attention）的数学原理：从矩阵乘法到 Affine 变换'
date: '2026-02-17T03:00:00Z'
draft: false
math: true
translationKey: kda-mathematics
tags: ['KDA', 'Linear Attention', 'Delta Rule', 'Affine 变换', 'Kimi', 'Kimi Delta Attention']
categories: ['技术']
description: '深入推导 KDA 的 chunk-wise 并行算法，从矩阵乘法的基本引理出发，建立 Affine 变换的理论框架，理解 WY 表示和 CP 并行的数学基础'
---

> 本文假设读者熟悉线性代数（矩阵乘法、外积、逆矩阵）和基本的序列模型概念。

## 摘要

本文推导了 KDA（Kimi Delta Attention）的 chunk-wise 并行算法。核心贡献：

1. 证明 KDA 的 chunk 状态更新可表示为 **Affine 变换**：$\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$
2. 通过 **WY 表示** 将残差计算分解为与历史状态无关的部分，实现并行计算
3. 基于 Affine 变换的复合性质，推导出 **CP（Context Parallel，上下文并行）** 的数学基础

KDA 相比标准 Attention 的优势：$O(N)$ 复杂度、常数内存状态、适合超长序列。

---

## 目录

1. [引言：从 Transformer 到 Linear Attention](#引言从-transformer-到-linear-attention)
2. [Linear Attention 的发展历程](#linear-attention-的发展历程)
3. [符号表与约定](#符号表与约定)
4. [线性注意力：简单的起点](#线性注意力简单的起点)
5. [背景：从 GDN 到 KDA](#背景从-gdn-到-kda)
6. [核心引理](#核心引理)
7. [KDA 的状态更新机制](#kda-的状态更新机制)
8. [WY 表示：依赖的分离](#wy-表示依赖的分离)
9. [核心定理：Chunk-wise Affine 形式](#核心定理chunk-wise-affine-形式)
10. [算法实现：从理论到代码](#算法实现从理论到代码)
11. [CP 并行与 SM 并行](#cp-并行与-sm-并行)
12. [总结](#总结)
13. [附录：GDN vs KDA](#附录gdn-vs-kda)
14. [参考资料](#参考资料)

---

## 引言：从 Transformer 到 Linear Attention

### 标准 Attention 的瓶颈

Transformer 架构自 2017 年提出以来，已成为自然语言处理和序列建模的主流方法。其核心组件 **Self-Attention** 机制通过计算序列中所有 token 两两之间的注意力权重来捕获长距离依赖：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

然而，这种标准的 Softmax Attention 存在显著的计算瓶颈：

- **$O(N^2)$ 复杂度**：计算注意力矩阵需要 $O(N^2)$ 的时间和空间复杂度
- **内存墙问题**：当序列长度 $N$ 增加时，内存占用呈平方增长
- **推理效率低**：自回归生成时需要缓存所有历史 KV，内存开销巨大

对于长序列任务（如文档理解、代码生成、多轮对话），$N$ 可能达到数十万甚至上百万，这使得标准 Attention 变得不可行。

### Linear Attention 的动机

Linear Attention [^6] 通过去掉 Softmax，将注意力重写为 RNN 形式。其完整形式包含分子（值累积）和分母（归一化累积）：

$$\mathbf{o}_t = \frac{\phi(\mathbf{q}_t)^T \mathbf{S}_t}{\phi(\mathbf{q}_t)^T \mathbf{Z}_t}$$

其中两个状态分别递推更新：
$$
\begin{aligned}
\mathbf{S}_t &= \mathbf{S}_{t-1} + \phi(\mathbf{k}_t) \otimes \mathbf{v}_t \\
\mathbf{Z}_t &= \mathbf{Z}_{t-1} + \phi(\mathbf{k}_t)
\end{aligned}
$$

这里 $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$ 是状态矩阵，$\mathbf{Z}_t \in \mathbb{R}^{d_k}$ 是归一化向量。**在实际应用中，分母的归一化可以通过后续的 RMSNorm 等层近似，因此常被省略以简化计算**，得到更简洁的形式：

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t) \otimes \mathbf{v}_t, \quad \mathbf{o}_t = \phi(\mathbf{q}_t)^T \mathbf{S}_t$$

这种形式的复杂度仅为 $O(N)$，且推理时只需要维护固定大小的状态矩阵。

### 本文的工作

本文聚焦于 **Kimi Linear** [^16] 中提出的 **Kimi Delta Attention (KDA)**，这是一种新一代 Linear Attention 变体，结合了：

1. **Delta Rule**：仅更新与预测误差相关的信息
2. **Per-dimension Decay**：不同维度可以有独立的遗忘速率
3. **Chunk-wise 并行**：通过 WY 表示实现硬件高效的并行训练

我们将从最基本的矩阵乘法引理出发，逐步建立 KDA 的完整数学理论。

---

## Linear Attention 的发展历程

Linear Attention 的研究从早期的模仿 Softmax Attention，到逐渐形成自身特色，再到最近探索更上层的指导原则（如 Delta Rule），经历了多个重要阶段。

### 1. 奠基时期 (2020)：从近似到重构

**Katharopoulos et al. [^6]** 在 ICML 2020 上发表了开创性工作 "Transformers are RNNs"，首次将 Transformer 重新表述为 RNN 形式。他们证明了通过特征映射 $\phi$，可以构造线性复杂度的注意力机制。

早期的 Linear Attention 主要是**模仿和近似 Softmax Attention**：
- 直接去掉 softmax 的 exp，得到 $O = (QK^\top \odot M)V$
- 为了数值稳定性，给 Q, K 加上非负激活函数（如 elu+1）
- **Performer** [^8] 使用随机傅里叶特征近似 Softmax

然而后续研究发现，在序列长度维度归一化并不能完全避免数值不稳定性，倒不如直接事后归一化（如 RMSNorm），而且给 Q, K 加激活函数也非必须。

### 2. 遗忘机制的引入 (2021-2023)

纯 Linear Attention 本质上是 cumsum，会将所有历史信息等权叠加，导致远距离 token 的信息占比极小。**遗忘机制**的引入解决了这个问题：

- **LRU** (2023)：线性循环单元，引入标量 decay（衰减） 因子
- **RetNet** (2023)：首次将遗忘因子与 Linear Attention 结合，$S_t = \gamma S_{t-1} + v_t k_t^\top$，其中 $\gamma \in (0,1)$ 是常数 decay
- **RWKV-4** [^10] (2023)：纯 RNN 架构，结合 RNN 的常数推理内存和 Transformer 的并行训练优点，使用 channel-wise decay（通道级衰减）

RetNet 的一个细节是给 Q, K 加上 RoPE，相当于将 decay 推广到复数域，从 LRU 角度看是考虑了复数特征值。

### 3. Data-Dependent Decay (2023-2024)

将静态 decay 推广为与输入相关的动态 decay，形成了一系列工作：

- **Mamba** [^13]：引入输入依赖的门控机制
- **Mamba2** [^14][^15]：提出 SSD 框架，从状态空间模型角度重新解释
- **GLA** [^7]：使用外积形式的遗忘门，实现 GPU 高效的矩阵乘法并行
- **RWKV-5/6** [^18] (2024)：Eagle 和 Finch 架构，引入矩阵值状态和动态递推

这一阶段的工作与 GRU、LSTM 等传统 RNN 的"遗忘门"已经非常相似，只是为了保持线性性，去除了门控对 State 的依赖。

### 4. RWKV：独立的纯 RNN 架构

**RWKV**（Receptance Weighted Key Value）是由 Peng Bo 等人提出的一系列纯 RNN 架构 LLM，与 Linear Attention 的发展并行，但采用了不同的技术路线——RWKV 强调保持纯粹的 RNN 形式（仅通过固定大小的状态传递历史信息），而 Linear Attention 则侧重于利用矩阵乘法实现 chunk-wise 并行计算。

| 版本 | 时间 | 核心特性 |
|------|------|----------|
| **RWKV-4** [^10] | 2023 | 基础架构，引入 Receptance 机制和 channel-wise 时间衰减 |
| **RWKV-5 (Eagle)** [^18] | 2024 | 矩阵值状态（Matrix-Valued States），增强表达能力 |
| **RWKV-6 (Finch)** [^18] | 2024 | 数据依赖的 token shift 和动态递推 |
| **RWKV-7** [^19] | 2025 | **引入广义 Delta Rule**（generalized delta rule），向量值门控和上下文学习率 |

RWKV 的独特之处在于其完全基于 RNN 形式，通过精心设计的状态更新机制实现高效的序列建模。

### 5. Delta Rule 的兴起 (2024-2025)

Delta Rule 最初是神经网络中的参数更新规则（Widrow-Hoff 规则），近年来被引入序列建模作为"测试时训练"（Test Time Training）的一种形式：

- **TTT** (2024)：将序列模型构建视为在线学习问题，用优化器构建 RNN
- **DeltaNet** [^4] (NeurIPS 2024)：将 Delta Rule 应用于 Linear Attention
- **Gated DeltaNet (GDN)** [^5] (2024)：引入门控机制控制信息流动
- **RWKV-7** [^19] (2025)：独立地引入广义 Delta Rule
- **KDA** [^16] (2025)：在 Kimi Linear 中提出，将标量 decay（衰减） 推广到 per-dimension decay

Delta Rule 的核心思想是**仅当新信息与历史预测不同时才更新状态**，这与人类的增量学习过程相似，也与 TTT 的"在线学习"视角高度契合。

### 变体对比

| 方法 | 更新规则 | 复杂度 | 关键特性 |
|------|----------|--------|----------|
| Softmax Attention | $\text{softmax}(QK^T)V$ | $O(N^2)$ | 全局依赖，准确但慢 |
| Linear Attention | $\phi(Q)^T \sum \phi(K)V^T$ | $O(N)$ | 固定状态，高效但弱表达 |
| RetNet | $S_t = \gamma S_{t-1} + v_t k_t^\top$ | $O(N)$ | 常数 decay + RoPE |
| RWKV-4/5/6 | Receptance + 时间衰减 | $O(N)$ | 纯 RNN 架构，并行训练 |
| Mamba | 输入依赖的状态转移 | $O(N)$ | 选择性，硬件优化 |
| GLA | 门控线性注意力 | $O(N)$ | 外积形式，GPU 高效 |
| DeltaNet | Delta Rule | $O(N)$ | 内容感知增量更新 |
| GDN | Delta + 标量门控 | $O(N)$ | 全局遗忘控制 |
| RWKV-7 | 广义 Delta Rule | $O(N)$ | 向量值门控 |
| **KDA** | Delta + Per-dim 门控 | $O(N)$ | 维度选择性遗忘 |


---

## 符号表与约定

| 符号 | 维度 | 含义 |
|------|------|------|
| $\mathbf{s}_t$ | $\mathbb{R}^{K \times V}$ | token-level 状态矩阵 |
| $\mathbf{S}$ | $\mathbb{R}^{K \times V}$ | chunk-level 初始状态 |
| $\mathbf{S}'$ | $\mathbb{R}^{K \times V}$ | chunk-level 结束状态 |
| $\mathbf{k}_t, \mathbf{q}_t$ | $\mathbb{R}^{1 \times K}$（行向量） | token-level key/query |
| $\mathbf{v}_t$ | $\mathbb{R}^{1 \times V}$（行向量） | token-level value |
| $\mathbf{K}, \mathbf{Q}, \mathbf{V}$ | $\mathbb{R}^{C \times K}$ / $\mathbb{R}^{C \times V}$ | chunk-level 矩阵，第 $i$ 行为 $\mathbf{k}_i$ |
| $\mathbf{g}_t^{\text{raw}}$ | $\mathbb{R}^K$ | 原始 log decay |
| $\mathbf{g}_t$ | $\mathbb{R}^K$ | 累积 log decay（cumsum 后）|
| $\boldsymbol{\lambda}_t = \exp(\mathbf{g}_t^{\text{raw}})$ | $\mathbb{R}^K$ | per-dimension decay（逐维衰减） 因子（原始 decay）|
| $\beta_t$ | 标量 | Delta Rule 权重 |
| $\mathbf{A}_{kk}$ | $\mathbb{R}^{C \times C}$ | 严格下三角权重矩阵 |
| $\mathbf{W}, \mathbf{U}$ | $\mathbb{R}^{C \times K}$ / $\mathbb{R}^{C \times V}$ | WY 表示的加权 keys/values |
| $\mathbf{M}$ | $\mathbb{R}^{K \times K}$ | Affine transition 矩阵 |
| $\mathbf{B}$ | $\mathbb{R}^{K \times V}$ | Affine bias 矩阵 |
| $\otimes$ | - | 外积：$(\mathbf{k}\otimes\mathbf{v})_{ab} = k_a \cdot v_b$ |
| $\odot$ | - | Hadamard 积（逐元素乘）|

**约定**：
- 小写粗体（$\mathbf{s}, \mathbf{k}, \mathbf{v}$）表示 token-level 行向量
- 大写粗体（$\mathbf{S}, \mathbf{K}, \mathbf{V}$）表示 chunk-level 矩阵
- 矩阵 $\mathbf{K} \in \mathbb{R}^{C \times K}$ 的第 $i$ 行为 $\mathbf{k}_i \in \mathbb{R}^{1 \times K}$
- 矩阵 $\mathbf{V} \in \mathbb{R}^{C \times V}$ 的第 $i$ 行为 $\mathbf{v}_i \in \mathbb{R}^{1 \times V}$
- 状态 $\mathbf{s}_t \in \mathbb{R}^{K \times V}$ 和 $\mathbf{S} \in \mathbb{R}^{K \times V}$ 是矩阵（非向量）

### 关于 Chunk

**Chunk** 是指将长序列分割为固定长度的连续段（通常 $C = 64$ 或 $128$），每段包含 $C$ 个 token。选择 $C = 64$ 或 $128$ 的原因与 **GPU Tensor Core** 的矩阵乘法维度有关：

- Tensor Core 的矩阵乘法最优维度通常满足 $M, N, K \in \{64, 128, 256\}$
- Chunk size $C$ 对应矩阵乘的 $M$ 或 $N$ 维度
- 过大的 $C$（如 256）会增加 Shared Memory 内存占用；过小的 $C$（如 16）无法充分利用 Tensor Core 的并行度

---

## 线性注意力：简单的起点

作为热身，我们先看**线性注意力**（Linear Attention），它是最简单的 recurrent 注意力形式。

### 定义

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{k}_t \otimes \mathbf{v}_t, \quad \mathbf{o}_t = \mathbf{q}_t^\top \mathbf{s}_t$$

其中 $\mathbf{s}_t \in \mathbb{R}^{K \times V}$ 是状态矩阵。

### Chunk-wise 形式

将序列分成每 $C$ 个 token 为一个 chunk。设 $\mathbf{S} \in \mathbb{R}^{K \times V}$ 是 chunk 开始时的状态，则 chunk 内第 $i$ 个位置的状态为：

$$\mathbf{s}_i = \mathbf{S} + \sum_{j=1}^i \mathbf{k}_j \otimes \mathbf{v}_j$$

chunk 的输出 $\mathbf{O} \in \mathbb{R}^{C \times V}$（第 $i$ 行是 $\mathbf{o}_i^\top$）：

$$\mathbf{O} = \mathbf{Q} \mathbf{S} + \text{mask}(\mathbf{Q} \mathbf{K}^\top) \mathbf{V}$$

其中 $\text{mask}(\cdot)$ 表示因果掩码（下三角部分）。上述形式完全由矩阵乘法构成。

> **参考资料**：线性注意力的奠基性工作见 Katharopoulos et al. (ICML 2020) [^6]，首次将 Transformer 重新表述为 RNN 形式。硬件高效的 chunk-wise 并行训练方法见 Yang et al. (ICML 2024) [^7]。

## 背景：从 GDN 到 KDA

### Gated DeltaNet (GDN)

**Gated DeltaNet (GDN)** 是一种基于 Delta Rule 的序列建模方法，使用**标量 decay（衰减）**：

$$\mathbf{s}_t = \lambda_t \cdot \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\lambda_t \cdot \mathbf{s}_{t-1}))$$

其中 $\lambda_t = \exp(g_t)$ 是**标量**（每个 head 一个值），所有维度共享相同的遗忘速率。

### Kimi Delta Attention (KDA)

**KDA** 是 GDN 的扩展，将标量 decay（衰减） 推广为**per-dimension decay**：

$$\mathbf{s}_t = \boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}))$$

其中 $\boldsymbol{\lambda}_t \in \mathbb{R}^K$ 是**向量**（每个维度一个值），不同维度可以有独立的遗忘速率。

### 本文的目标

本文以 **KDA** 为主要对象，建立其 chunk-wise 并行和 CP 并行的数学理论。GDN 作为 KDA 的特例（标量 decay（衰减））在附录中讨论。

---


## 核心引理

### 引理 1：外积累加的矩阵形式

**引理 1**：设 $\mathbf{k}_1, \ldots, \mathbf{k}_C \in \mathbb{R}^K$ 和 $\mathbf{v}_1, \ldots, \mathbf{v}_C \in \mathbb{R}^V$ 是两组向量。则

$$\sum_{i=1}^C \mathbf{k}_i \otimes \mathbf{v}_i = \mathbf{K}^\top \mathbf{V}$$

其中：
- $\mathbf{K} \in \mathbb{R}^{C \times K}$ 是以 $\mathbf{k}_i^\top$ 为第 $i$ 行的矩阵
- $\mathbf{V} \in \mathbb{R}^{C \times V}$ 是以 $\mathbf{v}_i^\top$ 为第 $i$ 行的矩阵
- $\otimes$ 表示外积：$(\mathbf{k} \otimes \mathbf{v})_{ab} = k_a \cdot v_b$

**证明**：直接计算右边矩阵的 $(a, b)$ 元素：

$$(\mathbf{K}^\top \mathbf{V})_{ab} = \sum_{i=1}^C K_{ia} V_{ib} = \sum_{i=1}^C k_{i,a} \cdot v_{i,b} = \sum_{i=1}^C (\mathbf{k}_i \otimes \mathbf{v}_i)_{ab}$$

由引理 1，chunk 内的外积累加可表示为矩阵乘法（GEMM，General Matrix Multiply），这为 chunk-wise 并行提供了数学基础。

### 引理 2：下三角矩阵的逆

**引理 2**：设 $\mathbf{L} \in \mathbb{R}^{C \times C}$ 是单位下三角矩阵（对角线为 1，上三角为 0），则 $\mathbf{L}^{-1}$ 也是单位下三角矩阵，且可以通过前向替换计算。

特别地，若 $\mathbf{L} = \mathbf{I} - \mathbf{N}$，其中 $\mathbf{N}$ 是严格下三角矩阵（对角线为 0），则

$$\mathbf{L}^{-1} = \mathbf{I} + \mathbf{N} + \mathbf{N}^2 + \cdots + \mathbf{N}^{C-1}$$

**证明**：直接验证 $(\mathbf{I} - \mathbf{N})(\mathbf{I} + \mathbf{N} + \cdots + \mathbf{N}^{C-1}) = \mathbf{I} - \mathbf{N}^C = \mathbf{I}$（因为 $\mathbf{N}^C = 0$，严格下三角矩阵的 $C$ 次幂为零）。

### 引理 3：对数衰减矩阵的线性分解（exp g 与 exp -g）

**引理 3**：对于给定的**累积对数衰减向量** $\mathbf{g}_1, \dots, \mathbf{g}_C \in \mathbb{R}^K$（已通过 `cumsum` 计算），Attention 矩阵中的衰减项可以分解为：

$$\exp(\mathbf{g}_i - \mathbf{g}_j) = \exp(\mathbf{g}_i) \odot \exp(-\mathbf{g}_j)$$

这使得原本需要针对每个位置进行循环计算的逻辑可以直接写为两个"门控矩阵"的**标准矩阵乘法**：

$$\mathbf{A} = (\mathbf{K} \odot \exp(\mathbf{G})) \cdot (\mathbf{K} \odot \exp(-\mathbf{G}))^\top$$

**维度说明**：
- $\mathbf{K} \in \mathbb{R}^{C \times K}$：chunk 内的 keys 矩阵，第 $i$ 行为 $\mathbf{k}_i$
- $\mathbf{G} \in \mathbb{R}^{C \times K}$：累积 log decay 矩阵，第 $i$ 行为 $\mathbf{g}_i$
- $\mathbf{A} \in \mathbb{R}^{C \times C}$：中间 Attention 矩阵（尚未应用 $\beta$ 和因果 mask）

**分解形式**：
- $\mathbf{K}_{\text{exp}} = \mathbf{K} \odot \exp(\mathbf{G})$：Forward decay（累积衰减后的 keys）
- $\mathbf{K}_{\text{inv}} = \mathbf{K} \odot \exp(-\mathbf{G})$：Reverse decay（逆向衰减后的 keys）
- $$\mathbf{A} = \mathbf{K}_{\text{exp}} \cdot \mathbf{K}_{\text{inv}}^\top$$

**意义**：
1. **消除循环**：将 $O(C)$ 的循环和复杂的 `einsum` 转化为了单次标准的 **矩阵乘法 (GEMM)**
2. **硬件加速**：利用 GPU 的 **Tensor Core** 硬件加速，计算效率从访存受限（Memory-bound）转为计算受限（Compute-bound）
3. **内存节省**：不需要存储 $C \times C \times K$ 的中间张量，只需要存储 $C \times K$ 的门控矩阵

---



## KDA 的状态更新机制

### Delta Rule 的来源

**Delta Rule**（又称 Widrow-Hoff 学习规则或 LMS 算法）最初是神经网络中的参数更新规则：

$$\Delta w = \eta \cdot (y - \hat{y}) \cdot x$$

其中 $(y - \hat{y})$ 是预测误差（delta），$\eta$ 是学习率。该规则用误差信号修正权重。

在序列模型中，Delta Rule 被重新诠释为**状态更新机制**：
- 将历史状态 $\mathbf{s}_{t-1}$ 视为对当前输入的"预测"
- 用 $\mathbf{k}_t^\top \mathbf{s}_{t-1}$ 计算"预期 value"
- 残差 $\mathbf{v}_t - \mathbf{k}_t \mathbf{s}_{t-1}$（行向量 $\mathbb{R}^{1 \times V}$）表示"新信息"与"历史预期"的差异，外积 $\mathbf{k}_t^\top (\cdot)$ 将结果映射回状态矩阵 $\mathbb{R}^{K \times V}$
- 仅用这个差异（而非完整 value）更新状态

### KDA 的递推公式

**KDA** 的状态更新机制（Delta Rule + per-dim gate）：

$$\mathbf{s}_t = \boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}))$$

其中：
- $\boldsymbol{\lambda}_t = \exp(\mathbf{g}_t^{\text{raw}}) \in \mathbb{R}^K$ 是 per-dimension decay（逐维衰减） 因子（向量）
- $\beta_t$ 是 delta rule 的权重
- 残差项 $\mathbf{v}_t - \mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1})$ 中：
  - $\mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}) \in \mathbb{R}^{1 \times V}$（行向量）是预期 value
  - 与 $\mathbf{v}_t$ 对比得到残差（行向量形式）
  - 乘积 $\mathbf{k}_t^\top (\cdot)$ 将结果映射回状态矩阵 $\mathbb{R}^{K \times V}$

**注意**：
1. 残差中的预期 value 是用 **gate 之后的状态** $\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}$ 计算的
2. $\boldsymbol{\lambda}_t$ 是向量，每个维度 $i$ 有独立的 decay 率 $\lambda_{t,i}$
3. 当 $\boldsymbol{\lambda}_t = \lambda_t \cdot \mathbf{1}$（所有维度相同），KDA 退化为 GDN

### 对比：Linear Attention vs KDA

| 机制 | 更新规则 | 特性 |
|------|----------|------|
| Linear Attention | $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{k}_t \otimes \mathbf{v}_t$ | 累积所有历史信息 |
| GDN | $\mathbf{s}_t = \lambda_t \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\lambda_t \mathbf{s}_{t-1}))$ | 标量 decay（衰减），全局遗忘 |
| **KDA** | $\mathbf{s}_t = \boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}))$ | per-dimension decay（逐维衰减），维度选择性遗忘 |

### 问题：残差依赖于历史状态

展开 recurrent 的前两步（注意残差中用的是 gated 状态）：

$$\mathbf{s}_1 = \boldsymbol{\lambda}_1 \odot \mathbf{s}_0 + \beta_1 \cdot \mathbf{k}_1^\top (\mathbf{v}_1 - \mathbf{k}_1 (\boldsymbol{\lambda}_1 \odot \mathbf{s}_0))$$
$$\mathbf{s}_2 = \boldsymbol{\lambda}_2 \odot \mathbf{s}_1 + \beta_2 \cdot \mathbf{k}_2^\top (\mathbf{v}_2 - \mathbf{k}_2 (\boldsymbol{\lambda}_2 \odot \mathbf{s}_1))$$

每个 $\mathbf{s}_i$ 都复杂地依赖于 $\mathbf{S}$，无法直接用引理 1 写成 $\mathbf{K}^\top \mathbf{V}$ 的形式。

需要解决的问题：将"依赖于 $\mathbf{S}$"和"不依赖于 $\mathbf{S}$"的部分分离开。

---

## WY 表示：依赖的分离

### 目标

让我们把 $\mathbf{s}_i$ 对 $\mathbf{S}$ 的依赖显式写出来。定义修正后的 value：

$$\tilde{\mathbf{v}}_i = \mathbf{v}_i - \mathbf{k}_i (\boldsymbol{\lambda}_i \odot \mathbf{s}_{i-1}) \in \mathbb{R}^{1 \times V}$$

由于 $\mathbf{s}_{i-1}$ 本身依赖于 $\mathbf{S}$，需要找到满足下式的表示：

$$\tilde{\mathbf{v}}_i = \mathbf{u}_i - \mathbf{w}_i \mathbf{S}$$

其中 $\mathbf{u}_i, \mathbf{w}_i$ 仅依赖于 chunk 内的 $\{\mathbf{k}_j, \mathbf{v}_j\}$，与 $\mathbf{S}$ 无关。

### 推导 WY 表示

**步骤 1**：写出 $\mathbf{s}_i$ 的递推式

$$\mathbf{s}_i = \boldsymbol{\lambda}_i \odot \mathbf{s}_{i-1} + \beta_i \cdot \mathbf{k}_i^\top (\mathbf{v}_i - \mathbf{k}_i (\boldsymbol{\lambda}_i \odot \mathbf{s}_{i-1}))$$

---

**步骤 2**：定义累积量

令 $\boldsymbol{\Lambda}^{(i)} = \prod_{j=1}^i \text{diag}(\boldsymbol{\lambda}_j) \in \mathbb{R}^{K \times K}$（对角累积 decay 矩阵），并定义归一化状态：

$$\hat{\mathbf{s}}_i = (\boldsymbol{\Lambda}^{(i)})^{-1} \mathbf{s}_i$$

---

**步骤 3**：转化为下三角线性系统

将归一化状态 $\hat{\mathbf{s}}_i = (\boldsymbol{\Lambda}^{(i)})^{-1} \mathbf{s}_i$ 代入递推式，整理得到：

$$\hat{\mathbf{s}}_i = \hat{\mathbf{s}}_{i-1} + \beta_i \cdot \hat{\mathbf{k}}_i^\top (\hat{\mathbf{v}}_i - \hat{\mathbf{k}}_i \hat{\mathbf{s}}_{i-1})$$

定义归一化后的 key/value（注意 value 不需要相对于状态的 decay）：
$$\hat{\mathbf{k}}_i = \mathbf{k}_i \odot \exp(\mathbf{g}_i), \quad \hat{\mathbf{v}}_i = \mathbf{v}_i$$

则残差可写为（行向量）：
$$\tilde{\mathbf{v}}_i = \hat{\mathbf{v}}_i - \hat{\mathbf{k}}_i \hat{\mathbf{s}}_{i-1} \in \mathbb{R}^{1 \times V}$$

展开 $\hat{\mathbf{s}}_{i-1}$ 的递归形式（以 $\hat{\mathbf{s}}_0 = \mathbf{S}$ 为初始状态）：
$$\hat{\mathbf{s}}_{i-1} = \mathbf{S} + \sum_{j=1}^{i-1} \beta_j \cdot \hat{\mathbf{k}}_j \otimes \tilde{\mathbf{v}}_j$$

代入残差表达式：
$$\tilde{\mathbf{v}}_i = \hat{\mathbf{v}}_i - \hat{\mathbf{k}}_i \mathbf{S} - \sum_{j=1}^{i-1} \beta_j \cdot \hat{\mathbf{k}}_i \hat{\mathbf{k}}_j^\top \cdot \tilde{\mathbf{v}}_j$$

**注**：这里 $\tilde{\mathbf{v}}_j \in \mathbb{R}^{1 \times V}$ 是行向量，$\hat{\mathbf{k}}_i \hat{\mathbf{k}}_j^\top$ 是标量（$K$ 维内积）。

整理为矩阵形式。定义：
- 矩阵 $\tilde{\mathbf{V}}, \hat{\mathbf{V}} \in \mathbb{R}^{C \times V}$ 分别以 $\tilde{\mathbf{v}}_i, \hat{\mathbf{v}}_i$ 为第 $i$ 行
- 矩阵 $\mathbf{A}_{kk} \in \mathbb{R}^{C \times C}$ 为严格下三角矩阵，对于 $i > j$：$A_{ij} = \beta_j (\mathbf{k}_i \odot \exp(\mathbf{g}_i)) (\mathbf{k}_j \odot \exp(-\mathbf{g}_j))^\top$

则得到线性系统：
$$\tilde{\mathbf{V}} = \hat{\mathbf{V}} - \mathbf{K}_{\text{gated}} \mathbf{S} - \mathbf{A}_{kk} \tilde{\mathbf{V}}$$

即：
$$(\mathbf{I} + \mathbf{A}_{kk}) \tilde{\mathbf{V}} = \hat{\mathbf{V}} - \mathbf{K}_{\text{gated}} \mathbf{S}$$

其中 $\mathbf{K}_{\text{gated}}$ 的第 $i$ 行为 $\mathbf{k}_i \odot \exp(\mathbf{g}_i)$。

---

**步骤 4**：应用引理 2

由引理 2，$\mathbf{L} = \mathbf{I} + \mathbf{A}_{kk}$ 是单位下三角矩阵，其逆 $\mathbf{L}^{-1} = (\mathbf{I} + \mathbf{A}_{kk})^{-1}$ 也是单位下三角矩阵。求解线性系统：

$$\tilde{\mathbf{V}} = (\mathbf{I} + \mathbf{A}_{kk})^{-1} \cdot \hat{\mathbf{V}} - (\mathbf{I} + \mathbf{A}_{kk})^{-1} \cdot \mathbf{K} \mathbf{S}$$

---

**步骤 5**：定义 WY 表示

定义加权矩阵（对应代码中的 `u = A @ v` 和 `w = A @ (exp(g) * k)`）：
$$\mathbf{U} = (\mathbf{I} + \mathbf{A}_{kk})^{-1} \text{diag}(\boldsymbol{\beta}) \mathbf{V}$$
$$\mathbf{W} = (\mathbf{I} + \mathbf{A}_{kk})^{-1} \text{diag}(\boldsymbol{\beta}) (\mathbf{K} \odot \exp(\mathbf{G}))$$

其中 $\hat{\mathbf{V}}$ 是归一化后的 values（包含 $\beta$ 和相对 decay），则得到分离形式：
$$\tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W} \mathbf{S}$$

这就是 **WY 表示**。

> **参考资料**：WY 表示最初由 Bischof & Van Loan (1987) [^1] 提出用于 Householder 矩阵乘积的表示，后被 Schreiber & Van Loan (1989) [^2] 改进为紧凑形式。在序列模型中，DeltaNet [^4] 首次将这一技术应用于线性注意力的并行计算，Gated DeltaNet [^5] 进一步引入了门控机制。

### WY 表示的说明

- $\mathbf{W} \in \mathbb{R}^{C \times K}$：加权 keys，第 $i$ 行为 $\mathbf{w}_i \in \mathbb{R}^{1 \times K}$
- $\mathbf{U} \in \mathbb{R}^{C \times V}$：加权 values，第 $i$ 行为 $\mathbf{u}_i \in \mathbb{R}^{1 \times V}$
- $\tilde{\mathbf{v}}_i = \mathbf{u}_i - \mathbf{w}_i \mathbf{S}$：修正后的 value（行向量 $\mathbb{R}^{1 \times V}$）

由上述推导，$\mathbf{U}, \mathbf{W}$ 与 $\mathbf{S}$ 无关，可在计算 $\mathbf{S}$ 之前预先算出。

---

## 核心定理：Chunk-wise Affine 形式

现在我们可以陈述核心定理了。

### 定理（KDA/GDN 的 Chunk-wise Affine 形式）

设 chunk 开始时状态为 $\mathbf{S} \in \mathbb{R}^{K \times V}$，则 chunk 结束时的状态为：

$$\mathbf{S}' = \mathbf{M} \cdot \mathbf{S} + \mathbf{B}$$

其中：
- **Transition 矩阵** $\mathbf{M} \in \mathbb{R}^{K \times K}$：
  $$\mathbf{M} = \text{diag}(\exp(\mathbf{g}_{\text{last}})) - \mathbf{K}_{\text{decayed}}^\top \mathbf{W}$$
- **Bias 矩阵**：$\mathbf{B} = \mathbf{K}_{\text{decayed}}^\top \mathbf{U} \in \mathbb{R}^{K \times V}$
- 其中 $\mathbf{K}_{\text{decayed}}$ 的第 $i$ 行为 $\mathbf{k}_i \odot \exp(\mathbf{g}_{\text{last}} - \mathbf{g}_i)$，$\mathbf{g}_{\text{last}}$ 表示 chunk 最后一个位置的累积 log decay

且 chunk 的输出为：

$$\mathbf{O} = (\mathbf{Q} \odot \exp(\mathbf{g}_q)) \cdot \mathbf{S} + \text{mask}(\mathbf{A}_{qk}) \cdot (\mathbf{U} - \mathbf{W} \mathbf{S})$$

其中 $\mathbf{g}_q$ 是 query 的累积 gate，$\odot$ 表示 广播（broadcasting）乘法。

### 证明

**状态更新**（以 KDA 为例）：

$$\begin{aligned}
\mathbf{S}' &= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \sum_{i=1}^C \exp(\mathbf{g}_{\text{last}} - \mathbf{g}_i) \odot (\mathbf{k}_i^\top \tilde{\mathbf{v}}_i) \\
&= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \mathbf{K}_{\text{decayed}}^\top \tilde{\mathbf{V}} \quad \text{（引理 1：外积累加 = 矩阵乘）} \\
&= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \mathbf{K}_{\text{decayed}}^\top (\mathbf{U} - \mathbf{W} \mathbf{S}) \quad \text{（代入 WY 表示 } \tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W} \mathbf{S} \text{）} \\
&= (\text{diag}(\exp(\mathbf{g}_{\text{last}})) - \mathbf{K}_{\text{decayed}}^\top \mathbf{W}) \mathbf{S} + \mathbf{K}_{\text{decayed}}^\top \mathbf{U} \\
&= \mathbf{M} \mathbf{S} + \mathbf{B}
\end{aligned}$$

对于 GDN，将对角矩阵 $\text{diag}(\boldsymbol{\lambda}^{\text{last}})$ 替换为标量 $\lambda^{\text{last}} \mathbf{I}$ 即可。

**输出计算**类似可得。

### Affine 变换的形式

$$\mathbf{S}' = \underbrace{\mathbf{M}}_{K \times K} \cdot \underbrace{\mathbf{S}}_{K \times V} + \underbrace{\mathbf{B}}_{K \times V}$$

上述形式为仿射变换（Affine Transformation）：
- 线性部分：$\mathbf{M} \cdot \mathbf{S}$ 表示历史状态的衰减与投影
- 平移部分：$\mathbf{B}$ 表示当前 chunk 引入的新信息

---

## 算法实现：从理论到代码

基于上述定理，我们可以写出 chunk-wise 算法：

```python
def chunk_kda(K, V, Q, g, beta):
    """
    K, V, Q: [C, K] or [C, V]  # chunk 内的 keys, values, queries
    g: [C, K]                  # 累积 gate (cumsum of log decay)
    beta: [C]                  # delta rule 的权重
    """
    # Step 1: 计算下三角矩阵 A (不含 beta)
    # 利用引理 3 的分解：A = (K * exp(g)) @ (K * exp(-g)).T
    K_exp = K * exp(g)
    K_inv = K * exp(-g)
    A = (K_exp @ K_inv.T).masked_fill(diagonal_mask, 0)
    
    # Step 2: 计算 (I + A)^{-1} 通过前向替换（引理 2）
    # 实际上由于 A = K_exp @ K_inv.T，这就是典型的 WY 表示形式
    L = I + A * beta[:, None]  # 包含 beta 的单位下三角矩阵
    
    # Step 3: 准备 gated 输入
    K_gated = K * exp(g)          # [C, K], gated keys
    V_weighted = V * beta[:, None]  # [C, V], V * beta
    K_weighted = K_gated * beta[:, None]  # [C, K], gated K * beta
    
    # Step 4: WY 表示（通过前向替换求解 L @ X = Y）
    # U = L^{-1} @ (V * beta)
    # W = L^{-1} @ (K * exp(g) * beta)
    U = forward_substitution(L, V_weighted)   # [C, V]
    W = forward_substitution(L, K_weighted)   # [C, K]
    
    # Step 5: 计算 Affine 参数
    # 注意：K_decayed 的第 i 行为 k_i * exp(g_last - g_i)
    K_decayed = K * exp(g[-1] - g)  # [C, K]
    decay_last = exp(g[-1])     # [K], 最后一个位置的累积 decay (per-dim)
    M = diag(decay_last) - K_decayed.T @ W    # [K, K]
    B = K_decayed.T @ U         # [K, V]
    
    # Step 6: 假设初始状态 S=0，计算 local 状态
    S_next = B                  # 如果 S=0
    
    # Step 7: 计算 chunk 输出（假设 S=0，实际需加上 S 的贡献）
    Q_gated = Q * exp(g)        # [C, K], gated queries
    O_local = mask(Q_gated @ K.T) @ U   # [C, V]
    
    return M, B, O_local, S_next, W, U
```

**说明**：
1. KDA 使用 per-dimension decay（逐维衰减） `diag(decay_last)`，GDN 使用标量 `decay_last * I`
2. Query 和 Key 都需要应用 gate，分别用于输出计算和残差计算
3. `g` 是累积 gate，维度 `[C, K]`，表示 per-dim 的 log decay

---


## CP 并行与 SM 并行

### CP 并行：Affine 链式法则

现在我们有了一致的 Affine 接口，可以自然地扩展到 **Context Parallel (CP)**。

#### Affine 变换的复合性质

**引理 4**：两个 Affine 变换的复合仍是 Affine 变换。

设：
- $\mathbf{S}_1 = \mathbf{M}_0 \mathbf{S}_0 + \mathbf{B}_0$
- $\mathbf{S}_2 = \mathbf{M}_1 \mathbf{S}_1 + \mathbf{B}_1$

则：
$$\mathbf{S}_2 = \underbrace{(\mathbf{M}_1 \mathbf{M}_0)}_{\mathbf{M}_{01}} \mathbf{S}_0 + \underbrace{(\mathbf{M}_1 \mathbf{B}_0 + \mathbf{B}_1)}_{\mathbf{B}_{01}}$$

#### CP 算法

假设有 $R$ 个 rank，rank $r$ 持有 chunk $r$。

**步骤 1：Local 计算**

每个 rank 假设 $\mathbf{S} = \mathbf{0}$，计算：
- $(\mathbf{M}_r, \mathbf{B}_r)$：Affine 参数
- $\mathbf{B}_r$：假设零初始状态时的最终状态（即 local accumulation，对应 KCP 中的 $h_{ext}$）

**步骤 2：All-Gather**

收集所有 rank 的 $\{ (\mathbf{M}_r, \mathbf{B}_r) \}_{r=0}^{R-1}$。

**步骤 3：Prefix Scan（Fold）**

Rank $r$ 计算真正的初始状态：

$$\mathbf{S}_r = \sum_{j=0}^{r-1} \left( \prod_{k=j+1}^{r-1} \mathbf{M}_k \right) \mathbf{B}_j$$

**步骤 4：Local 重算**

用正确的 $\mathbf{S}_r$ 重新计算 chunk 输出：

$$\mathbf{O}_r = \mathbf{O}_r^{\text{local}} + \mathbf{Q}_r \mathbf{S}_r - \text{mask}(\mathbf{A}_{qk}) \mathbf{W}_r \mathbf{S}_r$$

#### CP 并行的数学基础

CP 并行能够实现，其数学基础在于 Affine 变换的复合性质：
- 每个 chunk 是一个 Affine 变换
- 多个 chunk 的连续作用 = Affine 变换的乘积
- 跨 rank 的状态传递 = Affine 参数的累积

### SM 并行：单卡内的细粒度并行

#### 问题背景

在单卡（Intra-Card）推理场景中，当序列很长时会出现 **SM 利用率不足** 的问题：

- GPU 有固定数量的 SM（Streaming Multiprocessors，如 A100 有 108 个 SM）
- 每个 head 的 chunk 数量 = $T / (H \times C)$，其中 $T$ 是序列长度，$H$ 是 head 数，$C$ 是 chunk size
- 当序列很长但 head 数较少时，单个 head 的 chunk 数可能超过 SM 数，导致部分 SM 空闲

#### 解决方案：Subsequence 分割

**SM 并行**（SM Parallel）将长序列分割为多个 **subsequence**，使得：

$$\text{subseq\_len} = \text{target\_chunks} \times C \approx \text{num\_sms} \times C$$

其中：
- $\text{num\_sms}$：GPU 的 SM 数量
- $C$：chunk size（通常为 64）
- 每个 subsequence 包含足够多的 chunks 来饱和所有 SM

#### 数学形式

设原始序列被分割为 $M$ 个 subsequence，每个 subsequence $m$ 有初始状态 $\mathbf{S}_m$。

**步骤 1：Intra-subsequence CP**

每个 subsequence 内部执行标准的 CP Pre-process：
- 计算 $(\mathbf{M}_m^{\text{local}}, \mathbf{B}_m^{\text{local}})$：假设 $\mathbf{S}_m = \mathbf{0}$ 时的 local accumulation

**步骤 2：Inter-subsequence Merge**

同一原始序列的多个 subsequence 之间进行状态合并：
$$\mathbf{S}_{m+1} = \mathbf{M}_m^{\text{local}} \cdot \mathbf{S}_m + \mathbf{B}_m^{\text{local}}$$

这仍然是 Affine 变换的链式复合。

**步骤 3：Final Computation**

用正确的初始状态重新计算每个 subsequence 的输出。

#### 与 CP 并行的关系

| 并行级别 | 分割维度 | 通信方式 | 适用场景 |
|----------|----------|----------|----------|
| **CP 并行** | 跨 GPU（inter-card） | NCCL All-Gather | 多卡训练/推理 |
| **SM 并行** | 单卡内（intra-card） | 共享内存 | 单卡长序列推理 |

两者的数学本质相同：都是 Affine 变换的链式复合，只是粒度不同：
- CP 并行：rank 级别
- SM 并行：subsequence 级别

#### 实现要点

1. **动态分割**：根据序列长度和 SM 数量动态计算 `subseq_len`
2. **Split Info 管理**：维护 subsequence 与原序列的映射关系
3. **两级计算**：
   - `intracard_pre_scan`：并行计算所有 subsequence 的 local $(\mathbf{M}, \mathbf{B})$
   - `intracard_merge`：合并同一原序列的 subsequence 状态

> **实现参考**：`fla/ops/common/intracard_cp.py`

---

## 总结

我们从最基本的引理出发，建立了 **KDA**（及作为其特例的 **GDN**）的完整数学框架：

1. **引理 1**：外积累加 = 矩阵乘 → chunk-wise 并行的动机
2. **引理 2**：下三角矩阵的逆 → WY 表示的理论基础
3. **引理 3**：对数衰减的分解 → 矩阵乘法形式的 decay 计算
4. **KDA 的挑战**：残差依赖于历史状态
5. **WY 表示**：分离依赖，得到 $\tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W} \mathbf{S}$
6. **核心定理**：Chunk-wise Affine 形式 $\mathbf{S}' = \mathbf{M} \mathbf{S} + \mathbf{B}$
7. **CP 并行**：Affine 变换的链式复合

### 关键洞察

- **WY 表示**的本质：将依赖于历史状态 $\mathbf{S}$ 的部分显式分离，使得并行计算成为可能
- **Affine 形式**的作用：提供统一的状态更新接口，天然支持多级并行（CP、SM）
- **Per-dim decay**的优势：允许不同特征维度有独立的遗忘速率，增强表达能力

### 符号约定

- 小写 $\mathbf{s}, \mathbf{k}, \mathbf{v}$：token-level 向量
- 大写 $\mathbf{S}, \mathbf{K}, \mathbf{V}, \mathbf{M}, \mathbf{B}$：chunk-level 矩阵
- 区分 GDN（标量 decay（衰减））和 KDA（per-dimension decay（逐维衰减））只在 transition 矩阵的对角部分

---

## 附录：GDN vs KDA

| 特性 | GDN | KDA |
|------|-----|-----|
| Decay | 标量 $\lambda$ | 向量 $\boldsymbol{\lambda} \in \mathbb{R}^K$ |
| Transition | $\mathbf{M} = \lambda \mathbf{I} - \mathbf{K}^\top \mathbf{W}$ | $\mathbf{M} = \text{diag}(\boldsymbol{\lambda}) - \mathbf{K}^\top \mathbf{W}$ |
| 表达力 | 全局遗忘 | 维度选择性遗忘 |
| 计算 | 稍快 | 稍慢 |

两者都是 Affine 形式，只是 $\mathbf{M}$ 的对角部分不同。

> **参考资料**：Gated DeltaNet 详见 Yang et al. (2024) [^5]，Kimi Delta Attention (KDA) 是其在 per-dimension decay（逐维衰减） 方向的扩展。

---

## 参考资料

[^1]: Bischof, C., & Van Loan, C. (1987). "The WY Representation for Products of Householder Matrices". *SIAM Journal on Scientific and Statistical Computing*, 8(1). https://epubs.siam.org/doi/abs/10.1137/0908009

[^2]: Schreiber, R., & Van Loan, C. (1989). "A Storage-Efficient WY Representation for Products of Householder Transformations". *SIAM Journal on Scientific and Statistical Computing*, 10(1). https://epubs.siam.org/doi/10.1137/0910005


[^4]: Yang, S., et al. (NeurIPS 2024). "Parallelizing Linear Transformers with the Delta Rule over Sequence Length". *NeurIPS 2024*. https://arxiv.org/abs/2406.06484

[^5]: Yang, S., Kautz, J., & Hatamizadeh, A. (2024). "Gated Delta Networks: Improving Mamba2 with Delta Rule". arXiv:2412.06464. https://arxiv.org/abs/2412.06464

[^6]: Katharopoulos, A., et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention". *ICML 2020*. https://arxiv.org/abs/2006.16236

[^7]: Yang, S., et al. (2024). "Gated Linear Attention Transformers with Hardware-Efficient Training". *ICML 2024*. https://arxiv.org/abs/2312.06635

[^8]: Choromanski, K., et al. (2021). "Rethinking Attention with Performers". *ICLR 2021*. https://arxiv.org/abs/2009.14794


[^10]: Peng, B., et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era". *EMNLP 2023*. https://arxiv.org/abs/2305.13048




[^13]: Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". https://arxiv.org/abs/2312.00752

[^14]: Dao, T., & Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality". https://arxiv.org/abs/2405.21060

[^15]: Dao, T., & Gu, A. (2024). "Mamba2" (in "Transformers are SSMs"). https://arxiv.org/abs/2405.21060

[^16]: Kimi Team. (2025). "Kimi Linear: An Expressive, Efficient Attention Architecture". *arXiv:2510.26692*. https://arxiv.org/abs/2510.26692

[^18]: Peng, B., et al. (2024). "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence". *arXiv:2404.05892*. https://arxiv.org/abs/2404.05892

[^19]: Peng, B., et al. (2025). "RWKV-7 'Goose' with Expressive Dynamic State Evolution". *arXiv:2503.14456*. https://arxiv.org/abs/2503.14456

---

*本文的数学推导和算法描述基于 Flash Linear Attention (FLA) 框架的实现。*
