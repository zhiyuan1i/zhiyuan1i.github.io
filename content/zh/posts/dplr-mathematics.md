---
title: 'DPLR（Diagonal Plus Low Rank）的数学原理：显式转移矩阵的并行计算'
date: '2026-02-21T10:44:23Z'
draft: false
math: true
translationKey: dplr-mathematics
tags: ['DPLR', 'Linear Attention', 'RWKV-7', 'Low Rank', 'WY 表示']
categories: ['技术']
description: '深入推导 DPLR 的 chunk-wise 并行算法，理解显式对角+低秩转移矩阵的 WY 表示，探讨其与 KDA/IPLR 的统一框架'
---

> 本文假设读者熟悉线性代数（矩阵乘法、外积、逆矩阵）和基本的序列模型概念，建议先阅读 [KDA 数学原理](/posts/kda-mathematics/)。

## 摘要

本文推导了 **DPLR（Diagonal Plus Low Rank）** 的 chunk-wise 并行算法。DPLR 是广义 Delta Rule 的重要变体，被应用于 **RWKV-7** 等架构中。核心贡献：

1. 建立 DPLR 的显式转移矩阵形式：$\mathbf{P}_t = \text{diag}(\exp(\mathbf{g}_t)) + \mathbf{b}_t \mathbf{a}_t^T$
2. 推导 DPLR 的 **WY 表示**，将累积转移矩阵分解为对角部分与低秩部分之和
3. 证明 DPLR 同样满足 **Affine 变换**形式，天然支持 CP 并行
4. 对比 DPLR、KDA、IPLR 的异同，揭示线性注意力家族的统一数学框架

DPLR 相比标准 Delta Rule 的优势：显式控制对角衰减（dim-wise forgetting）和低秩更新，表达力更强，但在 chunk 形式下显著的引入了额外的计算复杂度，需要更多的 HBM 空间来存储中间变量。

---

## 目录

1. [引言：从 Delta Rule 到 DPLR](#引言从-delta-rule-到-dplr)
2. [符号表与约定](#符号表与约定)
3. [核心引理](#核心引理)
4. [DPLR 的递推形式](#dplr-的递推形式)
5. [WY 表示：累积转移矩阵的分解](#wy-表示累积转移矩阵的分解)
6. [核心定理：Chunk-wise Affine 形式](#核心定理chunk-wise-affine-形式)
7. [算法实现：从理论到代码](#算法实现从理论到代码)
8. [DPLR vs KDA vs IPLR](#dplr-vs-kda-vs-iplr)
9. [CP 并行与多级并行](#cp-并行与多级并行)
10. [总结](#总结)

---

## 引言：从 Delta Rule 到 DPLR

### Delta Rule 的局限性

标准 Delta Rule（以及没有遗忘门的 GDN/KDA）的状态更新可以写成：

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^T (\mathbf{v}_t - \mathbf{k}_t \mathbf{s}_{t-1})$$

这种形式的转移矩阵是隐式的：
- 通过残差 $(\mathbf{v}_t - \mathbf{k}_t \mathbf{s}_{t-1})$ 间接影响状态更新
- 遗忘机制通过门控 $\boldsymbol{\lambda}_t$ 实现

从数学上看，这等价于：

$$\mathbf{s}_t = (\mathbf{I} - \beta_t \mathbf{k}_t^T \mathbf{k}_t)\mathbf{s}_{t-1} + \beta_t \mathbf{k}_t^T \mathbf{v}_t$$

转移矩阵 $\mathbf{I} - \beta_t \mathbf{k}_t^T \mathbf{k}_t$ 是单位矩阵 + 低秩（rank-1）的形式，即IPLR（Identity Plus Low Rank）结构。

### DPLR 的核心思想

**DPLR（Diagonal Plus Low Rank）** 采用**显式的转移矩阵**形式：

$$\mathbf{S}_t = \exp(\mathbf{g}_t) \odot \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t + \mathbf{b}_t (\mathbf{a}_t^T \mathbf{S}_{t-1})$$

或者更紧凑地写成：

$$\mathbf{S}_t = (\mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T) \mathbf{S}_{t-1} + \mathbf{k}_t^T\mathbf{v}_t$$

其中：
- $\mathbf{D}_t = \text{diag}(\exp(\mathbf{g}_t)) \in \mathbb{R}^{K \times K}$ 是对角衰减矩阵
- $\mathbf{a}_t, \mathbf{b}_t \in \mathbb{R}^{K \times 1}$（列向量）是低秩更新的两个向量
- 转移矩阵 $\mathbf{P}_t = \mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T$ 是**对角+低秩（DPLR）**结构

### 为什么叫 "Diagonal Plus Low Rank"？

矩阵 $\mathbf{P}_t = \mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T$ 的结构：
1. **对角部分** $\mathbf{D}_t$：控制每个维度的独立衰减
2. **低秩部分** $\mathbf{b}_t \mathbf{a}_t^T$：秩为 1 的更新，提供跨维度的耦合

这种结构在数值线性代数中有深入研究，特别适合快速矩阵-向量乘法。

### 与 RWKV-7 的关系

RWKV-7 采用了基于 DPLR 思想的动态状态演化（Dynamic State Evolution）架构。在我们的底层并行实现中，RWKV-7 的状态更新公式实际上是 DPLR 框架的一个具体实例化：

传统线性注意力试图直接拟合 $\{k, v\}$ 对，而 RWKV-7 在 L2 loss $L=\frac{1}{2} \left\Vert v - S k \right\Vert^2$ 的指导下，通过模拟动态梯度下降来更新状态 $S$。其理论更新公式为：

$$S_t = S_{t-1} \text{Diag}(d_t) - \eta_t \cdot S_{t-1} k_t k_t^{\top} + \eta_t \cdot v_t k_t^{\top}$$

在算法实现中，这个基于梯度的更新被泛化为更灵活的 DPLR 形式：

$$S_t = S_{t-1} \odot \exp(-e^{w_t}) + (S_{t-1} a_t) b_t^T + v_t k_t^T$$

对应到我们在并行系统中的参数映射为：
- **$w_t$** 对应对数衰减项（具体为 $-\exp(w_t)$）
- **$a_t$** 对应低秩更新向量 $a$（动态学习率调节器 / in-context learning rate）
- **$b_t$** 对应低秩更新向量 $b$（状态更新调节器）

这使得 RWKV-7 ：
- **动态的衰减与学习率**：$w_t, a_t, b_t$ 都是 data-dependent 的，允许模型根据上下文动态决定遗忘和更新的强度。
- **表达能力有所提升**：由于引入了显式的状态演化，RWKV-7 能够识别所有正则语言（regular languages），其理论表达能力超越了 TC0 的 Transformer，达到了 NC1。
- **无缝接入 DPLR Chunk 并行**：由于其本质是 DPLR 结构，RWKV-7 可以直接复用 DPLR 的 chunk-wise 算法来实现高效的长序列并行训练。

---

## 符号表与约定

| 符号 | 维度 | 含义 |
|------|------|------|
| $\mathbf{s}_t$ | $\mathbb{R}^{K \times V}$ | token-level 状态矩阵 |
| $\mathbf{S}$ | $\mathbb{R}^{K \times V}$ | chunk-level 初始状态 |
| $\mathbf{S}'$ | $\mathbb{R}^{K \times V}$ | chunk-level 结束状态 |
| $\mathbf{k}_t, \mathbf{q}_t$ | $\mathbb{R}^{1 \times K}$（行向量）| token-level key/query |
| $\mathbf{v}_t$ | $\mathbb{R}^{1 \times V}$（行向量）| token-level value |
| $\mathbf{a}_t, \mathbf{b}_t$ | $\mathbb{R}^{K \times 1}$（列向量）| 低秩更新的两个向量 |
| $\mathbf{K}, \mathbf{V}$ | $\mathbb{R}^{C \times K}$ / $\mathbb{R}^{C \times V}$ | chunk-level key/value 矩阵，第 $i$ 行为 $\mathbf{k}_i$ / $\mathbf{v}_i$ |
| $\mathbf{A}^{\text{lr}} \in \mathbb{R}^{C \times K}$ | 第 $i$ 行为 $\mathbf{a}_i^T$ | 低秩向量 $\mathbf{a}$ 的矩阵形式（列向量转行排列）|
| $\mathbf{B}^{\text{lr}} \in \mathbb{R}^{C \times K}$ | 第 $i$ 行为 $\mathbf{b}_i^T$ | 低秩向量 $\mathbf{b}$ 的矩阵形式（列向量转行排列）|
| $\mathbf{g}_t$ | $\mathbb{R}^{K}$ | log decay 向量（累积前）|
| $\mathbf{g}_t^{\text{cum}}$ | $\mathbb{R}^{K}$ | 累积 log decay（cumsum 后）|
| $\mathbf{D}_t = \text{diag}(\exp(\mathbf{g}_t^{\text{cum}}))$ | $\mathbb{R}^{K \times K}$ | 对角衰减矩阵 |
| $\boldsymbol{\Gamma}_i^t = \prod_{j=i}^t \mathbf{D}_j$ | $\mathbb{R}^{K \times K}$ | 累积对角衰减矩阵 |
| $\mathbf{P}_t = \mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T$ | $\mathbb{R}^{K \times K}$ | 转移矩阵（低秩外积形式）|
| $\mathbf{A}_{ab}, \mathbf{A}_{ak}$ | $\mathbb{R}^{C \times C}$ | 严格下三角注意力矩阵 |
| $\mathbf{W}, \mathbf{U}$ | $\mathbb{R}^{C \times K}$ / $\mathbb{R}^{C \times V}$ | WY 表示的加权矩阵 |
| $\mathbf{w}_i, \mathbf{u}_i$ | $\mathbb{R}^{K}$ / $\mathbb{R}^{V}$ | WY 表示的加权向量（第 $i$ 个分量）|
| $\tilde{\mathbf{u}}_i$ | $\mathbb{R}^{V}$ | 包含历史状态贡献的修正向量 |
| $\mathbf{M}$ | $\mathbb{R}^{K \times K}$ | Affine transition 矩阵 |
| $\mathbf{B}$ | $\mathbb{R}^{K \times V}$ | Affine bias 矩阵 |
| $\odot$ | - | Hadamard 积（逐元素乘）|

**重要约定**：
- DPLR 在 `flash-linear-attention` 的实现中采用**左乘**形式：$\mathbf{S}_t = \mathbf{P}_t \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t$
- 状态矩阵 $\mathbf{S} \in \mathbb{R}^{K \times V}$（key dim × value dim）

*注：原生的 RWKV-7 公式是其对偶的**右乘**形式，状态矩阵为 $\mathbf{S}_{\text{rwkv}} \in \mathbb{R}^{V \times K}$，更新公式为 $\mathbf{S}_t = \mathbf{S}_{t-1} \mathbf{P}_t^T + \mathbf{v}_t \mathbf{k}_t^T$。在 FLA 框架中，为了与 KDA 等其他线性注意力机制保持统一，我们对状态矩阵进行了转置，统一采用左乘形式。*

**与 KDA 的对比**：

| 特性 | KDA | DPLR (FLA 实现) | RWKV-7 原生 |
|------|-----|------|------|
| 乘法方向 | 左乘 | 左乘 | 右乘 |
| 状态维度 | $\mathbb{R}^{K \times V}$ | $\mathbb{R}^{K \times V}$ | $\mathbb{R}^{V \times K}$ |
| Affine 形式 | $\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$ | $\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$ | $\mathbf{S}' = \mathbf{S}\mathbf{M}^T + \mathbf{B}^T$ |
| 转移矩阵 | 隐式（Delta Rule） | 显式（DPLR） |

---

## 核心引理

### 引理 1：下三角矩阵的逆

设 $\mathbf{L} \in \mathbb{R}^{C \times C}$ 是单位下三角矩阵（对角线为 1，上三角为 0），则 $\mathbf{L}^{-1}$ 也是单位下三角矩阵，且可以通过前向替换计算。

特别地，若 $\mathbf{L} = \mathbf{I} - \mathbf{N}$，其中 $\mathbf{N}$ 是严格下三角矩阵（对角线为 0），则

$$\mathbf{L}^{-1} = \mathbf{I} + \mathbf{N} + \mathbf{N}^2 + \cdots + \mathbf{N}^{C-1}$$

**证明**：直接验证 $(\mathbf{I} - \mathbf{N})(\mathbf{I} + \mathbf{N} + \cdots + \mathbf{N}^{C-1}) = \mathbf{I} - \mathbf{N}^C = \mathbf{I}$（因为 $\mathbf{N}^C = 0$）。

### 引理 2：DPLR 矩阵的乘积结构

设 $\mathbf{P}_i = \mathbf{D}_i + \mathbf{b}_i \mathbf{a}_i^T$，其中 $\mathbf{D}_i$ 是对角矩阵。则**反向**累积乘积 $\mathbf{P}_{t:1} = \prod_{i=t}^1 \mathbf{P}_i = \mathbf{P}_t \mathbf{P}_{t-1} \cdots \mathbf{P}_1$ 可以表示为：

$$\mathbf{P}_{t:1} = \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot (\mathbf{a}_i^T \boldsymbol{\Gamma}_1^{i-1})$$

**注意乘积方向**：这里的乘积是从右到左累乘（$\mathbf{P}_t$ 在最左边），与状态递推 $\mathbf{S}_t = \mathbf{P}_t \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t$ 展开后的形式一致。在展开后的求和项中，$\boldsymbol{\Gamma}_{i+1}^t$ 是 $\mathbf{b}_i$ 左侧的累积衰减（从 $i+1$ 到 $t$），$\boldsymbol{\Gamma}_1^{i-1}$ 是 $\mathbf{a}_i^T$ 右侧的累积衰减（从 $1$ 到 $i-1$）。

**意义**：这个引理保证了 DPLR 结构在矩阵乘法下的封闭性，是 WY 表示存在的基础。具体形式表明累积乘积保持"对角+低秩"的结构。

### 引理 3：对数衰减的分解

对于累积对数衰减，有：

$$\exp(\mathbf{g}_i^{\text{cum}} - \mathbf{g}_j^{\text{cum}}) = \exp(\mathbf{g}_i^{\text{cum}}) \odot \exp(-\mathbf{g}_j^{\text{cum}})$$

这使得衰减计算可以表示为两个门控向量的外积形式。

---

## DPLR 的递推形式

### 基本递推

DPLR 的状态更新方程为：

$$\mathbf{S}_t = \exp(\mathbf{g}_t) \odot \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t + \mathbf{b}_t (\mathbf{a}_t^T \mathbf{S}_{t-1})$$

或写成矩阵形式：

$$\mathbf{S}_t = (\mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T) \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t$$

其中：
- 第一项 $\mathbf{S}_{t-1} \odot \exp(\mathbf{g}_t)$：维度级衰减（Hadamard 积形式）
- 第二项 $\mathbf{k}_t^T \mathbf{v}_t$：标准的 key-value 外积更新
- 第三项 $\mathbf{b}_t (\mathbf{a}_t^T \mathbf{S}_{t-1})$：低秩更新，通过 $\mathbf{a}_t^T$ 投影状态（得到 $1 \times V$），再通过 $\mathbf{b}_t$ 扩展（得到 $K \times V$）

### 展开递推

为了理解 chunk-wise 并行，我们先展开前几个时间步：

$$
\begin{aligned}
\mathbf{S}_1 &= \mathbf{P}_1 \mathbf{S}_0 + \mathbf{k}_1^T \mathbf{v}_1 \\
\mathbf{S}_2 &= \mathbf{P}_2 \mathbf{S}_1 + \mathbf{k}_2^T \mathbf{v}_2 \\
&= \mathbf{P}_2 (\mathbf{P}_1 \mathbf{S}_0 + \mathbf{k}_1^T \mathbf{v}_1) + \mathbf{k}_2^T \mathbf{v}_2 \\
&= \mathbf{P}_2 \mathbf{P}_1 \mathbf{S}_0 + \mathbf{P}_2 \mathbf{k}_1^T \mathbf{v}_1 + \mathbf{k}_2^T \mathbf{v}_2
\end{aligned}
$$

一般形式：
$$\mathbf{S}_t = \left( \prod_{i=t}^1 \mathbf{P}_i \right) \mathbf{S}_0 + \sum_{i=1}^t \left( \prod_{j=t}^{i+1} \mathbf{P}_j \right) \mathbf{k}_i^T \mathbf{v}_i$$

**挑战**：直接计算累积转移矩阵 $\mathbf{P}_{t:1} = \prod_{i=t}^1 \mathbf{P}_i$ 需要 $O(t)$ 的矩阵乘法，如何实现并行？

---

## WY 表示：累积转移矩阵的分解

### 核心问题

我们需要高效地表示累积转移矩阵的乘积（注意左乘顺序，从右到左累乘）：
$$\mathbf{P}_{t:1} = \prod_{i=t}^1 (\mathbf{D}_i + \mathbf{b}_i \mathbf{a}_i^T)$$

**关键洞察**：对角+低秩矩阵的乘积仍然保持"对角+低秩"的结构，可以分解为对角累积加上加权的低秩外积之和。

### 定义累积对角衰减

令：
$$\boldsymbol{\Gamma}_i^t = \prod_{j=i}^t \mathbf{D}_j = \text{diag}\left(\exp\left(\sum_{j=i}^t \mathbf{g}_j\right)\right)$$

当 $i > t$ 时，定义 $\boldsymbol{\Gamma}_i^t = \mathbf{I}$（单位矩阵）。

### 定理（DPLR 的 WY 表示）

累积转移矩阵可以分解为：

$$\mathbf{P}_{t:1} = \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot (\mathbf{a}_i^T \boldsymbol{\Gamma}_1^{i-1})$$

> **定义动机**：为了使 WY 表示更紧凑，我们定义加权向量 $\mathbf{w}_i^T$（行向量），它将历史所有低秩更新的影响累积到第 $i$ 步。这类似于经典 WY 表示中累积 Householder 变换的权重。
>
> 或等价地，定义 $\mathbf{w}_i^T = \mathbf{a}_i^T \boldsymbol{\Gamma}_1^{i-1} + \sum_{j=1}^{i-1} (\mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j) \cdot \mathbf{w}_j^T$：

$$\mathbf{P}_{t:1} = \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot \mathbf{w}_i^T$$

其中系数 $(\mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j)$ 是标量。

> **与经典 WY 表示的联系**：经典 WY 表示将 Householder 矩阵的乘积分解为 $\mathbf{Q} = \mathbf{I} - \mathbf{W}\mathbf{Y}^T$。DPLR 的 WY 表示是其推广：以 $\boldsymbol{\Gamma}_1^t$（对角累积）替代 $\mathbf{I}$，以加权的低秩和替代标准低秩外积。

### 证明（归纳法）

**基例** $t=1$：
$$\mathbf{P}_1 = \mathbf{D}_1 + \mathbf{b}_1 \mathbf{a}_1^T = \boldsymbol{\Gamma}_1^1 + (\boldsymbol{\Gamma}_2^1 \mathbf{b}_1) \cdot \mathbf{w}_1^T$$

由于 $\boldsymbol{\Gamma}_1^1 = \mathbf{D}_1$，$\boldsymbol{\Gamma}_2^1 = \mathbf{I}$，$\mathbf{w}_1^T = \mathbf{a}_1^T$，成立。

**归纳步**：假设对 $t$ 成立，证明对 $t+1$ 成立。

$$
\begin{aligned}
\mathbf{P}_{t+1:1} &= \mathbf{P}_{t+1} \cdot \mathbf{P}_{t:1} \\
&= (\mathbf{D}_{t+1} + \mathbf{b}_{t+1} \mathbf{a}_{t+1}^T)\left(\boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot \mathbf{w}_i^T\right) \\
&= \boldsymbol{\Gamma}_1^{t+1} + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^{t+1} \mathbf{b}_i) \cdot \mathbf{w}_i^T \\
&\quad + \mathbf{b}_{t+1} \cdot \underbrace{\left(\mathbf{a}_{t+1}^T \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\mathbf{a}_{t+1}^T \boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot \mathbf{w}_i^T\right)}_{\eqqcolon \mathbf{w}_{t+1}^T} \\
&= \boldsymbol{\Gamma}_1^{t+1} + \sum_{i=1}^{t+1} (\boldsymbol{\Gamma}_{i+1}^{t+1} \mathbf{b}_i) \cdot \mathbf{w}_i^T
\end{aligned}
$$

其中使用了 $\boldsymbol{\Gamma}_{t+2}^{t+1} = \mathbf{I}$。证毕。

### 状态的 WY 表示

将 WY 表示代入状态递推，我们可以得到：

$$\mathbf{S}_t = \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{k}_i^T \mathbf{v}_i + \boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i \mathbf{u}_i^T)$$

其中 $\mathbf{u}_i^T$（$1 \times V$ 行向量）满足：

$$
\mathbf{u}_i^T = \begin{cases}
\mathbf{0}, & i=1 \\
\sum_{j=1}^{i-1} (\mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{k}_j^T \mathbf{v}_j + \mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j \mathbf{u}_j^T), & i \geq 2
\end{cases}
$$

### 矩阵形式的线性系统

定义 chunk 内的矩阵（第 $i$ 行为对应向量，以下计算适用于左乘的 DPLR）：
- $\mathbf{A}_{ab} \in \mathbb{R}^{C \times C}$：$[\mathbf{A}_{ab}]_{ij} = \mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j$ for $i > j$
- $\mathbf{A}_{ak} \in \mathbb{R}^{C \times C}$：$[\mathbf{A}_{ak}]_{ij} = \mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{k}_j^T$ for $i > j$

则 $(\mathbf{I} + \mathbf{A}_{ab})$ 是单位下三角矩阵。令：
- $\mathbf{A}^{\text{gate}} = \mathbf{A}^{\text{lr}} \odot \exp(\mathbf{G}^{\text{cum}}) \in \mathbb{R}^{C \times K}$（门控后的低秩向量矩阵），其中 $\mathbf{A}^{\text{lr}} \in \mathbb{R}^{C \times K}$ 的第 $i$ 行为 $\mathbf{a}_i^T$，$\mathbf{G}^{\text{cum}}$ 的第 $i$ 行为 $\mathbf{g}_i^{\text{cum}}$

则 WY 表示的矩阵形式为：

$$\mathbf{W} = (\mathbf{I} + \mathbf{A}_{ab})^{-1} \mathbf{A}^{\text{gate}}$$

$$\mathbf{U} = (\mathbf{I} + \mathbf{A}_{ab})^{-1} \mathbf{A}_{ak} \mathbf{V}$$

这与 KDA 中的 WY 表示形式结构类似，区别在于：KDA 中 $\tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W}\mathbf{S}$（减号，来自 Delta Rule 残差），而 DPLR 中 $\tilde{\mathbf{U}} = \mathbf{U} + \mathbf{W}\mathbf{S}$（加号，来自低秩叠加）。这导致 Affine 参数 $\mathbf{M}$ 中的符号也不同：KDA 为 $\text{diag}(\cdot) - \mathbf{K}^T \mathbf{W}$，DPLR 为 $\text{diag}(\cdot) + \mathbf{B}^T \mathbf{W}$。

---

## 核心定理：Chunk-wise Affine 形式

### 定理（DPLR 的 Chunk-wise Affine 形式）

设 chunk 开始时状态为 $\mathbf{S} \in \mathbb{R}^{K \times V}$，则 chunk 结束时的状态为：

$$\mathbf{S}' = \mathbf{M} \mathbf{S} + \mathbf{B}$$

其中：
- **Transition 矩阵** $\mathbf{M} \in \mathbb{R}^{K \times K}$：
  $$\mathbf{M} = \text{diag}(\exp(\mathbf{g}_{\text{last}})) + \mathbf{B}_{\text{decayed}}^T \mathbf{W}$$
- **Bias 矩阵** $\mathbf{B} \in \mathbb{R}^{K \times V}$：
  $$\mathbf{B} = \mathbf{K}_{\text{decayed}}^T \mathbf{V} + \mathbf{B}_{\text{decayed}}^T \mathbf{U}$$

且 chunk 的输出为：

$$\mathbf{O} = \mathbf{Q} \mathbf{S} + \text{mask}(\mathbf{A}_{qk}) \mathbf{V} + \text{mask}(\mathbf{A}_{qb}) (\mathbf{U} + \mathbf{W} \mathbf{S})$$

### 证明

**状态更新**：

$$
\begin{aligned}
\mathbf{S}' &= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \sum_{i=1}^C \exp(\mathbf{g}_{\text{last}} - \mathbf{g}_i) \odot (\mathbf{k}_i^T \mathbf{v}_i + \mathbf{b}_i \tilde{\mathbf{u}}_i) \\
&= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \mathbf{K}_{\text{decayed}}^T \mathbf{V} + \mathbf{B}_{\text{decayed}}^T \tilde{\mathbf{U}}
\end{aligned}
$$

其中 $\tilde{\mathbf{u}}_i = \mathbf{u}_i + \mathbf{w}_i \mathbf{S}$（$1 \times V$ 行向量）是包含历史状态贡献的修正向量。这里 $\mathbf{w}_i \in \mathbb{R}^{1 \times K}$（行向量），$\mathbf{S} \in \mathbb{R}^{K \times V}$，乘积 $\mathbf{w}_i \mathbf{S} \in \mathbb{R}^{1 \times V}$，维度匹配。

代入 WY 表示的矩阵形式 $\tilde{\mathbf{U}} = \mathbf{U} + \mathbf{W} \mathbf{S}$（注意这里是**加号**，与 KDA 中 $\tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W} \mathbf{S}$ 的**减号**不同。原因在于 KDA 的 WY 表示分离的是 Delta Rule 的*残差* $\mathbf{v}_i - \mathbf{k}_i \mathbf{S}$，减号来源于"减去历史预测"；而 DPLR 没有 Delta Rule 结构，低秩部分 $\mathbf{b}_i \mathbf{a}_i^T$ 是直接*叠加*到状态上的，因此历史状态的贡献是正向累积的）：

$$
\begin{aligned}
\mathbf{S}' &= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \mathbf{K}_{\text{decayed}}^T \mathbf{V} + \mathbf{B}_{\text{decayed}}^T (\mathbf{U} + \mathbf{W} \mathbf{S}) \\
&= \underbrace{(\text{diag}(\exp(\mathbf{g}_{\text{last}})) + \mathbf{B}_{\text{decayed}}^T \mathbf{W})}_{\mathbf{M}} \mathbf{S} + \underbrace{(\mathbf{K}_{\text{decayed}}^T \mathbf{V} + \mathbf{B}_{\text{decayed}}^T \mathbf{U})}_{\mathbf{B}}
\end{aligned}
$$

（注：详细的交叉项推导需要考虑 $\mathbf{W}$ 和 $\mathbf{K}_{\text{decayed}}$ 的具体关系，此处给出主要结构。）

**输出计算**类似可得。

---

## 算法实现：从理论到代码

基于上述定理，DPLR 的 chunk-wise 算法流程如下：

```python
def chunk_dplr(K, V, A, B, G, chunk_size=64):
    """
    K, V: [C, K], [C, V] - keys, values
    A, B: [C, K] - low-rank vectors a, b
    G: [C, K] - cumulative log decay
    """
    # Step 1: 计算门控后的输入
    # 注意：代码中使用相对衰减技巧
    ag = A * exp(G)           # gated a (使用 ge，即 shifted cumsum)
    bg = B * exp(-G + G[-1])  # gated b (相对衰减)
    kg = K * exp(-G + G[-1])  # gated k (相对衰减)
    qg = Q * exp(G)           # gated q (正向门控)
    
    # Step 2: 计算下三角矩阵 A_ab 和 A_ak
    # A_ab[i,j] = dot(a_i * exp(g_i - g_j), b_j) for i > j
    A_ab = (ag @ (B * exp(-G)).T).masked_fill_(triu_mask, 0)
    A_ak = (ag @ (K * exp(-G)).T).masked_fill_(triu_mask, 0)
    
    # Step 3: 计算 (I + A_ab)^{-1} 通过前向替换
    A_ab_inv = forward_substitution_inverse(I + A_ab)
    
    # Step 4: WY 表示
    # w = A_ab_inv @ ag
    # u = A_ab_inv @ (A_ak @ v)
    W = A_ab_inv @ ag    # [C, K]
    U = A_ab_inv @ (A_ak @ V)  # [C, V]
    
    # Step 5: 计算 Affine 参数
    decay_last = exp(G[-1])  # [K]
    K_decayed = K * exp(G[-1] - G)  # [C, K]
    B_decayed = B * exp(G[-1] - G)  # [C, K]
    
    # Transition 矩阵 M
    M = diag(decay_last) + B_decayed.T @ W  # [K, K]
    
    # Bias 矩阵 B（包含 k 和 b 的贡献）
    B_mat = K_decayed.T @ V + B_decayed.T @ U  # [K, V]
    
    # Step 6: 状态更新（如果初始状态 S=0，则 S_next = B_mat）
    S_next = M @ S + B_mat
    
    # Step 7: 计算 chunk 输出
    # O = Q @ S + masked_attention
    # 注意：qg 是门控后的 query，与 QG 相同
    O_local = mask(qg @ K.T) @ V + mask(qg @ B_lr.T) @ U
    
    return M, B_mat, S_next, W, U
```

### 关键实现细节

1. **矩阵求逆**：$(\mathbf{I} + \mathbf{A}_{ab})^{-1}$ 是单位下三角矩阵的逆，可通过前向替换在 $O(C^3)$ 内完成（$C$ 是 chunk size，通常 64 或 128）

2. **相对衰减技巧**：代码中使用 $\exp(-\mathbf{g} + \mathbf{g}_{\text{last}})$ 而非直接使用 $\exp(\mathbf{g})$，这是为了数值稳定性

3. **索引吸收约定**：代码中 `ag = A * exp(G)` 将 $\exp(\mathbf{g}_i)$ 吸收进了 $\mathbf{a}_i$，因此代码计算的 $\mathbf{A}_{ab}$ 实际上是 $[\mathbf{A}_{ab}]_{ij} = \mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i} \mathbf{b}_j$（包含 $\mathbf{g}_i$ 因子），而非数学定义中的 $\mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j$。相应地，$\mathbf{W}$ 的计算结果也吸收了这个额外因子，使得最终的 Affine 参数 $\mathbf{M}, \mathbf{B}$ 保持正确。这种吸收简化了代码实现，避免了显式的索引偏移

4. **分块计算**：当 $K$ 较大时，需要将 key/value 维度分块以适配 GPU Shared Memory

5. **精度控制**：类似 KDA，中间计算使用 float32，存储使用 bf16/fp16

---

## DPLR vs KDA vs IPLR

### 三种变体的统一视角

| 变体 | 转移矩阵 | 乘法方向 | 核心特征 |
|------|----------|----------|----------|
| **IPLR** | $\mathbf{I} + \mathbf{b}\mathbf{a}^T$ | 右乘 | Identity + Low Rank，无显式衰减 |
| **KDA** | 隐式（通过 Delta Rule） | 左乘 | Per-dim decay + Delta Rule |
| **DPLR** | $\text{diag}(\exp(\mathbf{g})) + \mathbf{b}\mathbf{a}^T$ | 左乘 | Diagonal decay + Low Rank |

### 数学联系

1. **IPLR 是 DPLR 的特例**：当 $\mathbf{g}_t = \mathbf{0}$（即 $\mathbf{D}_t = \mathbf{I}$）时，DPLR 退化为 IPLR

2. **RWKV-7 与 DPLR 的对偶性**：
   - DPLR（FLA 实现）：$\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$（左乘，column-space 更新）
   - RWKV-7（原生）：$\mathbf{S}' = \mathbf{S}\mathbf{M}^T + \mathbf{B}^T$（右乘，row-space 更新）

3. **统一框架**：两者最终都归结为 **Affine 变换**形式

---

## CP 并行与多级并行

### Affine 链式法则（左乘版本）

DPLR 的状态更新也是 Affine 形式，满足链式复合：

设：
- $\mathbf{S}_1 = \mathbf{M}_0 \mathbf{S}_0 + \mathbf{B}_0$
- $\mathbf{S}_2 = \mathbf{M}_1 \mathbf{S}_1 + \mathbf{B}_1$

则：
$$\mathbf{S}_2 = \underbrace{(\mathbf{M}_1 \mathbf{M}_0)}_{\mathbf{M}_{01}} \mathbf{S}_0 + \underbrace{(\mathbf{M}_1 \mathbf{B}_0 + \mathbf{B}_1)}_{\mathbf{B}_{01}}$$

### CP 并行算法

与 KDA 完全类似：

1. **Local 计算**：每个 rank 假设 $\mathbf{S} = \mathbf{0}$，计算 $(\mathbf{M}_r, \mathbf{B}_r)$
2. **All-Gather**：收集所有 Affine 参数
3. **Prefix Scan**：Rank $r$ 计算真正的初始状态
   $$\mathbf{S}_r = \sum_{j=0}^{r-1} \left( \prod_{k=j+1}^{r-1} \mathbf{M}_k \right) \mathbf{B}_j$$
4. **Local 重算**：用正确的 $\mathbf{S}_r$ 重新计算输出

### SM 并行

同样适用，将长序列分割为多个 subsequence，通过两级 Affine 复合实现。

---

## 总结

我们从显式转移矩阵的角度建立了 **DPLR** 的完整数学理论：

1. **DPLR 的核心**：对角+低秩转移矩阵 $\mathbf{P}_t = \text{diag}(\exp(\mathbf{g}_t)) + \mathbf{b}_t \mathbf{a}_t^T$
2. **WY 表示**：将累积转移矩阵分解为对角部分与低秩部分之和
   $$\mathbf{P}_{t:1} = \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot \mathbf{w}_i^T$$
3. **Chunk-wise Affine**：$\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$
4. **统一框架**：DPLR、KDA、IPLR 都是 Affine 变换的特例，支持相同的并行范式

---

*本文的数学推导基于我们在 Flash Linear Attention (FLA) 框架中的理论构建与代码实现。*
