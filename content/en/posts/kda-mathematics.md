---
title: 'KDA (Kimi Delta Attention): From Matrix Multiplication to Affine Transformation'
date: '2026-02-17T03:00:00Z'
draft: false
math: true
translationKey: kda-mathematics
tags: ['KDA', 'Linear Attention', 'Delta Rule', 'Affine Transformation', 'Kimi']
categories: ['Technical']
description: 'A deep dive into the chunk-wise parallel algorithm of KDA, establishing the theoretical framework of Affine transformations from basic matrix multiplication lemmas'
---

> This article assumes familiarity with linear algebra (matrix multiplication, outer product, inverse matrices) and basic sequence modeling concepts.

## Abstract

This article derives the chunk-wise parallel algorithm for KDA (Kimi Delta Attention). Core contributions:

1. Proving that KDA's chunk state update can be expressed as an **Affine transformation**: $\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$
2. Decomposing residual computation into history-independent components via **WY representation** to enable parallel computation
3. Deriving the mathematical foundation for **CP (Context Parallel)** based on the compositional properties of Affine transformations

Advantages of KDA over standard Attention: $O(N)$ complexity, constant memory state, suitable for ultra-long sequences.

---

## Table of Contents

1. [Introduction: From Transformer to Linear Attention](#introduction-from-transformer-to-linear-attention)
2. [The Development of Linear Attention](#the-development-of-linear-attention)
3. [Notation and Conventions](#notation-and-conventions)
4. [Background: From GDN to KDA](#background-from-gdn-to-kda)
5. [Core Lemmas](#core-lemmas)
6. [State Update Mechanism of KDA](#state-update-mechanism-of-kda)
7. [WY Representation: Separation of Dependencies](#wy-representation-separation-of-dependencies)
8. [Core Theorem: Chunk-wise Affine Form](#core-theorem-chunk-wise-affine-form)
9. [Algorithm Implementation: From Theory to Code](#algorithm-implementation-from-theory-to-code)
10. [CP Parallelism and SM Parallelism](#cp-parallelism-and-sm-parallelism)
11. [Summary](#summary)
12. [Appendix: GDN vs KDA](#appendix-gdn-vs-kda)
13. [References](#references)

---

## Introduction: From Transformer to Linear Attention

### Bottleneck of Standard Attention

Since its introduction in 2017, the Transformer architecture has become the mainstream method for natural language processing and sequence modeling. Its core component, the **Self-Attention** mechanism, captures long-range dependencies by computing attention weights between all pairs of tokens in a sequence:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

However, this standard Softmax Attention has significant computational bottlenecks:

- **$O(N^2)$ complexity**: Computing the attention matrix requires $O(N^2)$ time and space complexity
- **Memory wall problem**: As sequence length $N$ increases, memory usage grows quadratically
- **Low inference efficiency**: Autoregressive generation requires caching all historical KV, resulting in huge memory overhead

For long sequence tasks (e.g., document understanding, code generation, multi-turn dialogue), $N$ can reach hundreds of thousands or even millions, making standard Attention infeasible.

### Motivation for Linear Attention

Linear Attention [^6] removes Softmax and rewrites attention in RNN form. The complete form includes both numerator (value accumulation) and denominator (normalization accumulation):

$$\mathbf{o}_t = \frac{\phi(\mathbf{q}_t)^T \mathbf{S}_t}{\phi(\mathbf{q}_t)^T \mathbf{Z}_t}$$

where both states are updated recursively:
$$
\begin{aligned}
\mathbf{S}_t &= \mathbf{S}_{t-1} + \phi(\mathbf{k}_t) \otimes \mathbf{v}_t \\
\mathbf{Z}_t &= \mathbf{Z}_{t-1} + \phi(\mathbf{k}_t)
\end{aligned}
$$

Here $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$ is the state matrix and $\mathbf{Z}_t \in \mathbb{R}^{d_k}$ is the normalizer vector. **In practice, the denominator normalization can be approximated by subsequent layers such as RMSNorm, so it is often omitted to simplify computation**, yielding a cleaner form:

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t) \otimes \mathbf{v}_t, \quad \mathbf{o}_t = \phi(\mathbf{q}_t)^T \mathbf{S}_t$$

This form has $O(N)$ complexity, and inference only requires maintaining a fixed-size state matrix.

### Contributions of This Article

This article focuses on **Kimi Delta Attention (KDA)** introduced in **Kimi Linear** [^16], a new generation of Linear Attention architecture that combines:

1. **Delta Rule**: Only updates information related to prediction errors
2. **Per-dimension Decay**: Different dimensions can have independent forgetting rates
3. **Chunk-wise parallelism**: Hardware-efficient parallel training through WY representation

We will build the complete mathematical theory of KDA from the most basic matrix multiplication lemmas.

---

## The Development of Linear Attention

Linear Attention research has evolved from early attempts to mimic Softmax Attention, to gradually developing its own characteristics, and recently exploring higher-level guiding principles (such as the Delta Rule), going through several important stages.

### 1. Foundational Period (2020): From Approximation to Reconstruction

**Katharopoulos et al. [^6]** published the groundbreaking work "Transformers are RNNs" at ICML 2020, first reformulating Transformers into RNN form. They proved that through feature mapping $\phi$, linear-complexity attention mechanisms can be constructed.

Early Linear Attention mainly **mimicked and approximated Softmax Attention**:
- Directly removing exp from softmax to obtain $O = (QK^\top \odot M)V$
- Adding non-negative activation functions (e.g., elu+1) to Q, K for numerical stability
- **Performer** [^8] used random Fourier features to approximate Softmax

However, subsequent research found that normalization along the sequence dimension cannot completely avoid numerical instability; it's better to use post-hoc normalization (e.g., RMSNorm), and activation functions for Q, K are not strictly necessary.

### 2. Introduction of Forgetting Mechanisms (2021-2023)

Pure Linear Attention is essentially cumsum, equally weighting all historical information, causing information from distant tokens to have minimal contribution. The introduction of **forgetting mechanisms** solved this problem:

- **LRU** (2023): Linear Recurrent Unit, introducing scalar decay factors
- **RetNet** (2023): First combining forgetting factors with Linear Attention, $S_t = \gamma S_{t-1} + v_t k_t^\top$, where $\gamma \in (0,1)$ is a constant decay
- **RWKV-4** [^10] (2023): Pure RNN architecture combining constant inference memory of RNNs with parallel training advantages of Transformers, using channel-wise decay

A detail of RetNet is adding RoPE to Q, K, equivalent to generalizing decay to the complex domain; from the LRU perspective, this considers complex eigenvalues.

### 3. Data-Dependent Decay (2023-2024)

Extending static decay to input-dependent dynamic decay led to a series of works:

- **Mamba** [^13]: Introducing input-dependent gating mechanisms
- **Mamba2** [^14][^15]: Proposing the SSD framework, reinterpreting from the state space model perspective
- **GLA** [^7]: Using outer product form of forgetting gates, enabling GPU-efficient matrix multiplication parallelism
- **RWKV-5/6** [^18] (2024): Eagle and Finch architectures, introducing matrix-valued states and dynamic recurrence

Works at this stage are very similar to "forgetting gates" in traditional RNNs like GRU and LSTM, except that to maintain linearity, the gating's dependence on State is removed.

### 4. RWKV: An Independent Pure RNN Architecture

**RWKV** (Receptance Weighted Key Value) is a series of pure RNN architecture LLMs proposed by Peng Bo et al., developed in parallel with Linear Attention but adopting a different technical route—RWKV emphasizes maintaining a pure RNN form (only passing historical information through a fixed-size state), while Linear Attention focuses on using matrix multiplication to achieve chunk-wise parallel computation.

| Version | Time | Core Features |
|---------|------|---------------|
| **RWKV-4** [^10] | 2023 | Basic architecture, introducing Receptance mechanism and channel-wise time decay |
| **RWKV-5 (Eagle)** [^18] | 2024 | Matrix-Valued States, enhanced expressiveness |
| **RWKV-6 (Finch)** [^18] | 2024 | Data-dependent token shift and dynamic recurrence |
| **RWKV-7** [^19] | 2025 | **Introduction of generalized Delta Rule**, vector-valued gating and context learning rate |

The unique aspect of RWKV is its complete RNN-based form, achieving efficient sequence modeling through carefully designed state update mechanisms.

### 5. The Rise of Delta Rule (2024-2025)

The Delta Rule was originally a parameter update rule in neural networks (Widrow-Hoff rule), recently introduced into sequence modeling as a form of "Test Time Training":

- **TTT** (2024): Treating sequence model construction as an online learning problem, building RNNs with optimizers
- **DeltaNet** [^4] (NeurIPS 2024): Applying Delta Rule to Linear Attention
- **Gated DeltaNet (GDN)** [^5] (2024): Introducing gating mechanisms to control information flow
- **RWKV-7** [^19] (2025): Independently introducing generalized Delta Rule
- **KDA** [^16] (2025): Introduced in Kimi Linear, extending scalar decay to per-dimension decay

The core idea of Delta Rule is to **only update the state when new information differs from historical predictions**, similar to human incremental learning processes and highly aligned with TTT's "online learning" perspective.

### Comparison of Variants

| Method | Update Rule | Complexity | Key Features |
|--------|-------------|------------|--------------|
| Softmax Attention | $\text{softmax}(QK^T)V$ | $O(N^2)$ | Global dependencies, accurate but slow |
| Linear Attention | $\phi(Q)^T \sum \phi(K)V^T$ | $O(N)$ | Fixed state, efficient but weak expressiveness |
| RetNet | $S_t = \gamma S_{t-1} + v_t k_t^\top$ | $O(N)$ | Constant decay + RoPE |
| RWKV-4/5/6 | Receptance + time decay | $O(N)$ | Pure RNN architecture, parallel training |
| Mamba | Input-dependent state transition | $O(N)$ | Selective, hardware-optimized |
| GLA | Gated Linear Attention | $O(N)$ | Outer product form, GPU-efficient |
| DeltaNet | Delta Rule | $O(N)$ | Content-aware incremental updates |
| GDN | Delta + scalar gating | $O(N)$ | Global forgetting control |
| RWKV-7 | Generalized Delta Rule | $O(N)$ | Vector-valued gating |
| **KDA** | Delta + per-dim gating | $O(N)$ | Dimension-selective forgetting |

---

## Notation and Conventions

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $\mathbf{s}_t$ | $\mathbb{R}^{K \times V}$ | token-level state matrix |
| $\mathbf{S}$ | $\mathbb{R}^{K \times V}$ | chunk-level initial state |
| $\mathbf{S}'$ | $\mathbb{R}^{K \times V}$ | chunk-level final state |
| $\mathbf{k}_t, \mathbf{q}_t$ | $\mathbb{R}^{1 \times K}$ (row vector) | token-level key/query |
| $\mathbf{v}_t$ | $\mathbb{R}^{1 \times V}$ (row vector) | token-level value |
| $\mathbf{K}, \mathbf{Q}, \mathbf{V}$ | $\mathbb{R}^{C \times K}$ / $\mathbb{R}^{C \times V}$ | chunk-level matrices, row $i$ is $\mathbf{k}_i$ |
| $\mathbf{g}_t^{\text{raw}}$ | $\mathbb{R}^K$ | raw log decay |
| $\mathbf{g}_t$ | $\mathbb{R}^K$ | cumulative log decay (after cumsum) |
| $\boldsymbol{\lambda}_t = \exp(\mathbf{g}_t^{\text{raw}})$ | $\mathbb{R}^K$ | per-dimension decay factor (raw decay) |
| $\beta_t$ | scalar | Delta Rule weight |
| $\mathbf{A}_{kk}$ | $\mathbb{R}^{C \times C}$ | strictly lower triangular weight matrix |
| $\mathbf{W}, \mathbf{U}$ | $\mathbb{R}^{C \times K}$ / $\mathbb{R}^{C \times V}$ | WY representation weighted keys/values |
| $\mathbf{M}$ | $\mathbb{R}^{K \times K}$ | Affine transition matrix |
| $\mathbf{B}$ | $\mathbb{R}^{K \times V}$ | Affine bias matrix |
| $\otimes$ | - | outer product: $(\mathbf{k}\otimes\mathbf{v})_{ab} = k_a \cdot v_b$ |
| $\odot$ | - | Hadamard product (element-wise multiplication) |

**Conventions**:
- Lowercase bold ($\mathbf{s}, \mathbf{k}, \mathbf{v}$) denotes token-level row vectors
- Uppercase bold ($\mathbf{S}, \mathbf{K}, \mathbf{V}$) denotes chunk-level matrices
- Matrix $\mathbf{K} \in \mathbb{R}^{C \times K}$ has row $i$ as $\mathbf{k}_i \in \mathbb{R}^{1 \times K}$
- Matrix $\mathbf{V} \in \mathbb{R}^{C \times V}$ has row $i$ as $\mathbf{v}_i \in \mathbb{R}^{1 \times V}$
- States $\mathbf{s}_t \in \mathbb{R}^{K \times V}$ and $\mathbf{S} \in \mathbb{R}^{K \times V}$ are matrices (not vectors)

### About Chunks

**Chunk** refers to dividing long sequences into fixed-length continuous segments (typically $C = 64$ or $128$), each containing $C$ tokens. The choice of $C = 64$ or $128$ is related to **GPU Tensor Core** matrix multiplication dimensions:

- Optimal dimensions for Tensor Core matrix multiplication typically satisfy $M, N, K \in \{64, 128, 256\}$
- Chunk size $C$ corresponds to the $M$ or $N$ dimension in matrix multiplication
- Larger $C$ (e.g., 256) increases shared memory usage; smaller $C$ (e.g., 16) cannot fully utilize Tensor Core parallelism

---

## Linear Attention: A Simple Starting Point

As a warm-up, let's first look at **Linear Attention**, the simplest recurrent attention form.

### Definition

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{k}_t \otimes \mathbf{v}_t, \quad \mathbf{o}_t = \mathbf{q}_t^\top \mathbf{s}_t$$

where $\mathbf{s}_t \in \mathbb{R}^{K \times V}$ is the state matrix.

### Chunk-wise Form

Divide the sequence into chunks of $C$ tokens each. Let $\mathbf{S} \in \mathbb{R}^{K \times V}$ be the state at the beginning of the chunk; then the state at position $i$ within the chunk is:

$$\mathbf{s}_i = \mathbf{S} + \sum_{j=1}^i \mathbf{k}_j \otimes \mathbf{v}_j$$

The chunk output $\mathbf{O} \in \mathbb{R}^{C \times V}$ (row $i$ is $\mathbf{o}_i^\top$):

$$\mathbf{O} = \mathbf{Q} \mathbf{S} + \text{mask}(\mathbf{Q} \mathbf{K}^\top) \mathbf{V}$$

where $\text{mask}(\cdot)$ denotes causal masking (lower triangular part). This form is entirely composed of matrix multiplications.

> **Reference**: The foundational work on Linear Attention is Katharopoulos et al. (ICML 2020) [^6], which first reformulated Transformers into RNN form. Hardware-efficient chunk-wise parallel training methods are described in Yang et al. (ICML 2024) [^7].

## Background: From GDN to KDA

### Gated DeltaNet (GDN)

**Gated DeltaNet (GDN)** is a Delta Rule-based sequence modeling method using **scalar decay**:

$$\mathbf{s}_t = \lambda_t \cdot \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\lambda_t \cdot \mathbf{s}_{t-1}))$$

where $\lambda_t = \exp(g_t)$ is a **scalar** (one value per head), with all dimensions sharing the same forgetting rate.

### Kimi Delta Attention (KDA)

**KDA** extends GDN by generalizing scalar decay to **per-dimension decay**:

$$\mathbf{s}_t = \boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}))$$

where $\boldsymbol{\lambda}_t \in \mathbb{R}^K$ is a **vector** (one value per dimension), allowing different dimensions to have independent forgetting rates.

### Objective of This Article

This article focuses on **KDA** as the main subject, establishing its mathematical theory for chunk-wise parallelism and CP parallelism. GDN, as a special case of KDA (scalar decay), is discussed in the appendix.

---

## Core Lemmas

### Lemma 1: Matrix Form of Outer Product Accumulation

**Lemma 1**: Let $\mathbf{k}_1, \ldots, \mathbf{k}_C \in \mathbb{R}^K$ and $\mathbf{v}_1, \ldots, \mathbf{v}_C \in \mathbb{R}^V$ be two sets of vectors. Then:

$$\sum_{i=1}^C \mathbf{k}_i \otimes \mathbf{v}_i = \mathbf{K}^\top \mathbf{V}$$

where:
- $\mathbf{K} \in \mathbb{R}^{C \times K}$ is the matrix with $\mathbf{k}_i^\top$ as row $i$
- $\mathbf{V} \in \mathbb{R}^{C \times V}$ is the matrix with $\mathbf{v}_i^\top$ as row $i$
- $\otimes$ denotes outer product: $(\mathbf{k} \otimes \mathbf{v})_{ab} = k_a \cdot v_b$

**Proof**: Directly compute element $(a, b)$ of the right-hand side matrix:

$$(\mathbf{K}^\top \mathbf{V})_{ab} = \sum_{i=1}^C K_{ia} V_{ib} = \sum_{i=1}^C k_{i,a} \cdot v_{i,b} = \sum_{i=1}^C (\mathbf{k}_i \otimes \mathbf{v}_i)_{ab}$$

By Lemma 1, outer product accumulation within a chunk can be expressed as matrix multiplication (GEMM, General Matrix Multiply), providing the mathematical foundation for chunk-wise parallelism.

### Lemma 2: Inverse of Lower Triangular Matrix

**Lemma 2**: Let $\mathbf{L} \in \mathbb{R}^{C \times C}$ be a unit lower triangular matrix (diagonal is 1, upper triangle is 0), then $\mathbf{L}^{-1}$ is also a unit lower triangular matrix, and can be computed via forward substitution.

In particular, if $\mathbf{L} = \mathbf{I} - \mathbf{N}$, where $\mathbf{N}$ is a strictly lower triangular matrix (diagonal is 0), then:

$$\mathbf{L}^{-1} = \mathbf{I} + \mathbf{N} + \mathbf{N}^2 + \cdots + \mathbf{N}^{C-1}$$

**Proof**: Directly verify $(\mathbf{I} - \mathbf{N})(\mathbf{I} + \mathbf{N} + \cdots + \mathbf{N}^{C-1}) = \mathbf{I} - \mathbf{N}^C = \mathbf{I}$ (since $\mathbf{N}^C = 0$, the $C$-th power of a strictly lower triangular matrix is zero).

### Lemma 3: Linear Decomposition of Log-Decay Matrix (exp g and exp -g)

**Lemma 3**: For given **cumulative log-decay vectors** $\mathbf{g}_1, \dots, \mathbf{g}_C \in \mathbb{R}^K$ (computed via `cumsum`), the decay terms in the attention matrix can be decomposed as:

$$\exp(\mathbf{g}_i - \mathbf{g}_j) = \exp(\mathbf{g}_i) \odot \exp(-\mathbf{g}_j)$$

This allows logic originally requiring per-position loops to be written directly as **standard matrix multiplication** of two "gating matrices":

$$\mathbf{A} = (\mathbf{K} \odot \exp(\mathbf{G})) \cdot (\mathbf{K} \odot \exp(-\mathbf{G}))^\top$$

**Dimension notes**:
- $\mathbf{K} \in \mathbb{R}^{C \times K}$: keys matrix within chunk, row $i$ is $\mathbf{k}_i$
- $\mathbf{G} \in \mathbb{R}^{C \times K}$: cumulative log-decay matrix, row $i$ is $\mathbf{g}_i$
- $\mathbf{A} \in \mathbb{R}^{C \times C}$: intermediate attention matrix (before applying $\beta$ and causal mask)

**Decomposition form**:
- $\mathbf{K}_{\text{exp}} = \mathbf{K} \odot \exp(\mathbf{G})$: Forward decay (keys after cumulative decay)
- $\mathbf{K}_{\text{inv}} = \mathbf{K} \odot \exp(-\mathbf{G})$: Reverse decay (keys after inverse decay)
- $$\mathbf{A} = \mathbf{K}_{\text{exp}} \cdot \mathbf{K}_{\text{inv}}^\top$$

**Significance**:
1. **Eliminates loops**: Transforms $O(C)$ loops and complex `einsum` into a single standard **matrix multiplication (GEMM)**
2. **Hardware acceleration**: Leverages GPU **Tensor Core** hardware acceleration, shifting computational efficiency from memory-bound to compute-bound
3. **Memory savings**: No need to store $C \times C \times K$ intermediate tensors, only need to store $C \times K$ gating matrices

---



## State Update Mechanism of KDA

### Origin of Delta Rule

**Delta Rule** (also known as Widrow-Hoff learning rule or LMS algorithm) was originally a parameter update rule in neural networks:

$$\Delta w = \eta \cdot (y - \hat{y}) \cdot x$$

where $(y - \hat{y})$ is the prediction error (delta), and $\eta$ is the learning rate. This rule corrects weights using error signals.

In sequence models, Delta Rule is reinterpreted as a **state update mechanism**:
- Historical state $\mathbf{s}_{t-1}$ is viewed as a "prediction" of current input
- $\mathbf{k}_t^\top \mathbf{s}_{t-1}$ computes the "expected value"
- Residual $\mathbf{v}_t - \mathbf{k}_t \mathbf{s}_{t-1}$ (row vector $\mathbb{R}^{1 \times V}$) represents the difference between "new information" and "historical expectation", outer product $\mathbf{k}_t^\top (\cdot)$ maps the result back to state matrix $\mathbb{R}^{K \times V}$
- Only this difference (not the full value) updates the state

### Recurrence Formula of KDA

**KDA** state update mechanism (Delta Rule + per-dim gate):

$$\mathbf{s}_t = \boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}))$$

where:
- $\boldsymbol{\lambda}_t = \exp(\mathbf{g}_t^{\text{raw}}) \in \mathbb{R}^K$ is the per-dimension decay factor (vector)
- $\beta_t$ is the delta rule weight
- In the residual term $\mathbf{v}_t - \mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1})$:
  - $\mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}) \in \mathbb{R}^{1 \times V}$ (row vector) is the expected value
  - Comparison with $\mathbf{v}_t$ yields the residual (row vector form)
  - Product $\mathbf{k}_t^\top (\cdot)$ maps the result to state matrix $\mathbb{R}^{K \times V}$

**Note**:
1. The expected value in the residual is computed using **gated state** $\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}$
2. $\boldsymbol{\lambda}_t$ is a vector; each dimension $i$ has an independent decay rate $\lambda_{t,i}$
3. When $\boldsymbol{\lambda}_t = \lambda_t \cdot \mathbf{1}$ (all dimensions identical), KDA reduces to GDN

### Comparison: Linear Attention vs KDA

| Mechanism | Update Rule | Features |
|-----------|-------------|----------|
| Linear Attention | $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{k}_t \otimes \mathbf{v}_t$ | Accumulates all historical information |
| GDN | $\mathbf{s}_t = \lambda_t \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\lambda_t \mathbf{s}_{t-1}))$ | Scalar decay, global forgetting |
| **KDA** | $\mathbf{s}_t = \boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t (\boldsymbol{\lambda}_t \odot \mathbf{s}_{t-1}))$ | per-dimension decay, dimension-selective forgetting |

### Problem: Residual Depends on Historical State

Expanding the first two steps of recurrence (note: gated state is used in the residual):

$$\mathbf{s}_1 = \boldsymbol{\lambda}_1 \odot \mathbf{s}_0 + \beta_1 \cdot \mathbf{k}_1^\top (\mathbf{v}_1 - \mathbf{k}_1 (\boldsymbol{\lambda}_1 \odot \mathbf{s}_0))$$
$$\mathbf{s}_2 = \boldsymbol{\lambda}_2 \odot \mathbf{s}_1 + \beta_2 \cdot \mathbf{k}_2^\top (\mathbf{v}_2 - \mathbf{k}_2 (\boldsymbol{\lambda}_2 \odot \mathbf{s}_1))$$

Each $\mathbf{s}_i$ complexly depends on $\mathbf{S}$ and cannot be directly written in $\mathbf{K}^\top \mathbf{V}$ form using Lemma 1.

Problem to solve: Separate "depends on $\mathbf{S}$" and "independent of $\mathbf{S}$" parts.

---

## WY Representation: Separation of Dependencies

### Objective

Let's explicitly write out $\mathbf{s}_i$'s dependence on $\mathbf{S}$. Define the corrected value:

$$\tilde{\mathbf{v}}_i = \mathbf{v}_i - \mathbf{k}_i (\boldsymbol{\lambda}_i \odot \mathbf{s}_{i-1}) \in \mathbb{R}^{1 \times V}$$

Since $\mathbf{s}_{i-1}$ itself depends on $\mathbf{S}$, we need to find a representation satisfying:

$$\tilde{\mathbf{v}}_i = \mathbf{u}_i - \mathbf{w}_i \mathbf{S}$$

where $\mathbf{u}_i, \mathbf{w}_i$ only depend on $\{\mathbf{k}_j, \mathbf{v}_j\}$ within the chunk, independent of $\mathbf{S}$.

### Deriving WY Representation

**Step 1**: Write the recurrence for $\mathbf{s}_i$

$$\mathbf{s}_i = \boldsymbol{\lambda}_i \odot \mathbf{s}_{i-1} + \beta_i \cdot \mathbf{k}_i^\top (\mathbf{v}_i - \mathbf{k}_i (\boldsymbol{\lambda}_i \odot \mathbf{s}_{i-1}))$$

---

**Step 2**: Define cumulative quantities

Let $\boldsymbol{\Lambda}^{(i)} = \prod_{j=1}^i \text{diag}(\boldsymbol{\lambda}_j) \in \mathbb{R}^{K \times K}$ (diagonal cumulative decay matrix), and define normalized state:

$$\hat{\mathbf{s}}_i = (\boldsymbol{\Lambda}^{(i)})^{-1} \mathbf{s}_i$$

---

**Step 3**: Transform to lower triangular linear system

Substituting normalized state $\hat{\mathbf{s}}_i = (\boldsymbol{\Lambda}^{(i)})^{-1} \mathbf{s}_i$ into the recurrence and rearranging:

$$\hat{\mathbf{s}}_i = \hat{\mathbf{s}}_{i-1} + \beta_i \cdot \hat{\mathbf{k}}_i^\top (\hat{\mathbf{v}}_i - \hat{\mathbf{k}}_i \hat{\mathbf{s}}_{i-1})$$

Define normalized key/value (note: value does not need decay relative to state):
$$\hat{\mathbf{k}}_i = \mathbf{k}_i \odot \exp(\mathbf{g}_i), \quad \hat{\mathbf{v}}_i = \mathbf{v}_i$$

Then the residual can be written as (row vector):
$$\tilde{\mathbf{v}}_i = \hat{\mathbf{v}}_i - \hat{\mathbf{k}}_i \hat{\mathbf{s}}_{i-1} \in \mathbb{R}^{1 \times V}$$

Expanding $\hat{\mathbf{s}}_{i-1}$ in recursive form (with initial state $\hat{\mathbf{s}}_0 = \mathbf{S}$):
$$\hat{\mathbf{s}}_{i-1} = \mathbf{S} + \sum_{j=1}^{i-1} \beta_j \cdot \hat{\mathbf{k}}_j \otimes \tilde{\mathbf{v}}_j$$

Substituting into the residual expression:
$$\tilde{\mathbf{v}}_i = \hat{\mathbf{v}}_i - \hat{\mathbf{k}}_i \mathbf{S} - \sum_{j=1}^{i-1} \beta_j \cdot \hat{\mathbf{k}}_i \hat{\mathbf{k}}_j^\top \cdot \tilde{\mathbf{v}}_j$$

**Note**: Here $\tilde{\mathbf{v}}_j \in \mathbb{R}^{1 \times V}$ is a row vector, $\hat{\mathbf{k}}_i \hat{\mathbf{k}}_j^\top$ is a scalar ($K$-dimensional inner product).

Rearranging into matrix form. Define:
- Matrices $\tilde{\mathbf{V}}, \hat{\mathbf{V}} \in \mathbb{R}^{C \times V}$ with rows $\tilde{\mathbf{v}}_i, \hat{\mathbf{v}}_i$ respectively
- Matrix $\mathbf{A}_{kk} \in \mathbb{R}^{C \times C}$ as strictly lower triangular, for $i > j$: $A_{ij} = \beta_j (\mathbf{k}_i \odot \exp(\mathbf{g}_i)) (\mathbf{k}_j \odot \exp(-\mathbf{g}_j))^\top$

This yields the linear system:
$$\tilde{\mathbf{V}} = \hat{\mathbf{V}} - \mathbf{K}_{\text{gated}} \mathbf{S} - \mathbf{A}_{kk} \tilde{\mathbf{V}}$$

That is:
$$(\mathbf{I} + \mathbf{A}_{kk}) \tilde{\mathbf{V}} = \hat{\mathbf{V}} - \mathbf{K}_{\text{gated}} \mathbf{S}$$

where row $i$ of $\mathbf{K}_{\text{gated}}$ is $\mathbf{k}_i \odot \exp(\mathbf{g}_i)$.

---

**Step 4**: Apply Lemma 2

By Lemma 2, $\mathbf{L} = \mathbf{I} + \mathbf{A}_{kk}$ is a unit lower triangular matrix; its inverse $\mathbf{L}^{-1} = (\mathbf{I} + \mathbf{A}_{kk})^{-1}$ is also unit lower triangular. Solving the linear system:

$$\tilde{\mathbf{V}} = (\mathbf{I} + \mathbf{A}_{kk})^{-1} \cdot \hat{\mathbf{V}} - (\mathbf{I} + \mathbf{A}_{kk})^{-1} \cdot \mathbf{K} \mathbf{S}$$

---

**Step 5**: Define WY representation

Define weighted matrices (corresponding to `u = A @ v` and `w = A @ (exp(g) * k)` in code):
$$\mathbf{U} = (\mathbf{I} + \mathbf{A}_{kk})^{-1} \text{diag}(\boldsymbol{\beta}) \mathbf{V}$$
$$\mathbf{W} = (\mathbf{I} + \mathbf{A}_{kk})^{-1} \text{diag}(\boldsymbol{\beta}) (\mathbf{K} \odot \exp(\mathbf{G}))$$

where $\hat{\mathbf{V}}$ is the normalized values (including $\beta$ and relative decay), yielding the separated form:
$$\tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W} \mathbf{S}$$

This is the **WY representation**.

> **Reference**: WY representation was originally proposed by Bischof & Van Loan (1987) [^1] for representing products of Householder matrices, later improved to a compact form by Schreiber & Van Loan (1989) [^2]. In sequence models, DeltaNet [^4] first applied this technique to parallel computation of linear attention; Gated DeltaNet [^5] further introduced gating mechanisms.

### Explanation of WY Representation

- $\mathbf{W} \in \mathbb{R}^{C \times K}$: weighted keys, row $i$ is $\mathbf{w}_i \in \mathbb{R}^{1 \times K}$
- $\mathbf{U} \in \mathbb{R}^{C \times V}$: weighted values, row $i$ is $\mathbf{u}_i \in \mathbb{R}^{1 \times V}$
- $\tilde{\mathbf{v}}_i = \mathbf{u}_i - \mathbf{w}_i \mathbf{S}$: corrected value (row vector $\mathbb{R}^{1 \times V}$)

From the above derivation, $\mathbf{U}, \mathbf{W}$ are independent of $\mathbf{S}$ and can be precomputed before computing $\mathbf{S}$.

---

## Core Theorem: Chunk-wise Affine Form

Now we can state the core theorem.

### Theorem (Chunk-wise Affine Form of KDA/GDN)

Let the state at chunk start be $\mathbf{S} \in \mathbb{R}^{K \times V}$; then the state at chunk end is:

$$\mathbf{S}' = \mathbf{M} \cdot \mathbf{S} + \mathbf{B}$$

where:
- **Transition matrix** $\mathbf{M} \in \mathbb{R}^{K \times K}$:
  $$\mathbf{M} = \text{diag}(\exp(\mathbf{g}_{\text{last}})) - \mathbf{K}_{\text{decayed}}^\top \mathbf{W}$$
- **Bias matrix**: $\mathbf{B} = \mathbf{K}_{\text{decayed}}^\top \mathbf{U} \in \mathbb{R}^{K \times V}$
- Row $i$ of $\mathbf{K}_{\text{decayed}}$ is $\mathbf{k}_i \odot \exp(\mathbf{g}_{\text{last}} - \mathbf{g}_i)$, where $\mathbf{g}_{\text{last}}$ denotes the cumulative log decay at the last position of the chunk

And the chunk output is:

$$\mathbf{O} = (\mathbf{Q} \odot \exp(\mathbf{g}_q)) \cdot \mathbf{S} + \text{mask}(\mathbf{A}_{qk}) \cdot (\mathbf{U} - \mathbf{W} \mathbf{S})$$

where $\mathbf{g}_q$ is the cumulative gate for queries, and $\odot$ denotes broadcasting multiplication.

### Proof

**State update** (taking KDA as example):

$$\begin{aligned}
\mathbf{S}' &= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \sum_{i=1}^C \exp(\mathbf{g}_{\text{last}} - \mathbf{g}_i) \odot (\mathbf{k}_i^\top \tilde{\mathbf{v}}_i) \\
&= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \mathbf{K}_{\text{decayed}}^\top \tilde{\mathbf{V}} \quad \text{(Lemma 1: outer product accumulation = matrix multiplication)} \\
&= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \mathbf{K}_{\text{decayed}}^\top (\mathbf{U} - \mathbf{W} \mathbf{S}) \quad \text{(substitute WY representation } \tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W} \mathbf{S} \text{)} \\
&= (\text{diag}(\exp(\mathbf{g}_{\text{last}})) - \mathbf{K}_{\text{decayed}}^\top \mathbf{W}) \mathbf{S} + \mathbf{K}_{\text{decayed}}^\top \mathbf{U} \\
&= \mathbf{M} \mathbf{S} + \mathbf{B}
\end{aligned}$$

For GDN, simply replace diagonal matrix $\text{diag}(\boldsymbol{\lambda}^{\text{last}})$ with scalar $\lambda^{\text{last}} \mathbf{I}$.

**Output computation** follows similarly.

### Form of Affine Transformation

$$\mathbf{S}' = \underbrace{\mathbf{M}}_{K \times K} \cdot \underbrace{\mathbf{S}}_{K \times V} + \underbrace{\mathbf{B}}_{K \times V}$$

The above form is an Affine transformation:
- Linear part: $\mathbf{M} \cdot \mathbf{S}$ represents decay and projection of historical state
- Translation part: $\mathbf{B}$ represents new information introduced by the current chunk

---

## Algorithm Implementation: From Theory to Code

Based on the above theorem, we can write the chunk-wise algorithm:

```python
def chunk_kda(K, V, Q, g, beta):
    """
    K, V, Q: [C, K] or [C, V]  # keys, values, queries within chunk
    g: [C, K]                  # cumulative gate (cumsum of log decay)
    beta: [C]                  # weight for delta rule
    """
    # Step 1: Compute lower triangular matrix A (without beta)
    # Using Lemma 3 decomposition: A = (K * exp(g)) @ (K * exp(-g)).T
    K_exp = K * exp(g)
    K_inv = K * exp(-g)
    A = (K_exp @ K_inv.T).masked_fill(diagonal_mask, 0)
    
    # Step 2: Compute (I + A)^{-1} via forward substitution (Lemma 2)
    # Since A = K_exp @ K_inv.T, this is the typical WY representation form
    L = I + A * beta[:, None]  # Unit lower triangular matrix including beta
    
    # Step 3: Prepare gated inputs
    K_gated = K * exp(g)          # [C, K], gated keys
    V_weighted = V * beta[:, None]  # [C, V], V * beta
    K_weighted = K_gated * beta[:, None]  # [C, K], gated K * beta
    
    # Step 4: WY representation (solve L @ X = Y via forward substitution)
    # U = L^{-1} @ (V * beta)
    # W = L^{-1} @ (K * exp(g) * beta)
    U = forward_substitution(L, V_weighted)   # [C, V]
    W = forward_substitution(L, K_weighted)   # [C, K]
    
    # Step 5: Compute Affine parameters
    # Note: row i of K_decayed is k_i * exp(g_last - g_i)
    K_decayed = K * exp(g[-1] - g)  # [C, K]
    decay_last = exp(g[-1])     # [K], cumulative decay at last position (per-dim)
    M = diag(decay_last) - K_decayed.T @ W    # [K, K]
    B = K_decayed.T @ U         # [K, V]
    
    # Step 6: Assume initial state S=0, compute local state
    S_next = B                  # If S=0
    
    # Step 7: Compute chunk output (assuming S=0; actual output needs S contribution)
    Q_gated = Q * exp(g)        # [C, K], gated queries
    O_local = mask(Q_gated @ K.T) @ U   # [C, V]
    
    return M, B, O_local, S_next, W, U
```

**Notes**:
1. KDA uses per-dimension decay `diag(decay_last)`; GDN uses scalar `decay_last * I`
2. Both queries and keys need gating applied, for output computation and residual computation respectively
3. `g` is cumulative gate with dimension `[C, K]`, representing per-dim log decay

---

## CP Parallelism and SM Parallelism

### CP Parallelism: Affine Chain Rule

Now that we have a consistent Affine interface, we can naturally extend to **Context Parallel (CP)**.

#### Compositional Properties of Affine Transformations

**Lemma 4**: The composition of two Affine transformations is still an Affine transformation.

Let:
- $\mathbf{S}_1 = \mathbf{M}_0 \mathbf{S}_0 + \mathbf{B}_0$
- $\mathbf{S}_2 = \mathbf{M}_1 \mathbf{S}_1 + \mathbf{B}_1$

Then:
$$\mathbf{S}_2 = \underbrace{(\mathbf{M}_1 \mathbf{M}_0)}_{\mathbf{M}_{01}} \mathbf{S}_0 + \underbrace{(\mathbf{M}_1 \mathbf{B}_0 + \mathbf{B}_1)}_{\mathbf{B}_{01}}$$

#### CP Algorithm

Assume $R$ ranks, where rank $r$ holds chunk $r$.

**Step 1: Local Computation**

Each rank assumes $\mathbf{S} = \mathbf{0}$ and computes:
- $(\mathbf{M}_r, \mathbf{B}_r)$: Affine parameters
- $\mathbf{B}_r$: Final state assuming zero initial state (i.e., local accumulation, corresponding to $h_{ext}$ in KCP)

**Step 2: All-Gather**

Collect all ranks' $\{ (\mathbf{M}_r, \mathbf{B}_r) \}_{r=0}^{R-1}$.

**Step 3: Prefix Scan (Fold)**

Rank $r$ computes the true initial state:

$$\mathbf{S}_r = \sum_{j=0}^{r-1} \left( \prod_{k=j+1}^{r-1} \mathbf{M}_k \right) \mathbf{B}_j$$

**Step 4: Local Recomputation**

Recompute chunk output with correct $\mathbf{S}_r$:

$$\mathbf{O}_r = \mathbf{O}_r^{\text{local}} + \mathbf{Q}_r \mathbf{S}_r - \text{mask}(\mathbf{A}_{qk}) \mathbf{W}_r \mathbf{S}_r$$

#### Mathematical Foundation of CP Parallelism

CP parallelism is possible due to the compositional properties of Affine transformations:
- Each chunk is an Affine transformation
- Continuous application of multiple chunks = product of Affine transformations
- Cross-rank state transfer = accumulation of Affine parameters

### SM Parallelism: Fine-grained Parallelism within Single Card

#### Problem Background

In single-card (Intra-Card) inference scenarios, **SM underutilization** occurs when sequences are very long:

- GPUs have a fixed number of SMs (Streaming Multiprocessors, e.g., A100 has 108 SMs)
- Number of chunks per head = $T / (H \times C)$, where $T$ is sequence length, $H$ is number of heads, $C$ is chunk size
- When sequences are long but the number of heads is small, chunks per head may exceed the number of SMs, leaving some SMs idle

#### Solution: Subsequence Splitting

**SM Parallelism** splits long sequences into multiple **subsequences** such that:

$$\text{subseq\_len} = \text{target\_chunks} \times C \approx \text{num\_sms} \times C$$

where:
- $\text{num\_sms}$: Number of SMs in GPU
- $C$: chunk size (typically 64)
- Each subsequence contains enough chunks to saturate all SMs

#### Mathematical Form

Let the original sequence be split into $M$ subsequences, each subsequence $m$ having initial state $\mathbf{S}_m$.

**Step 1: Intra-subsequence CP**

Each subsequence internally executes standard CP Pre-process:
- Compute $(\mathbf{M}_m^{\text{local}}, \mathbf{B}_m^{\text{local}})$: local accumulation assuming $\mathbf{S}_m = \mathbf{0}$

**Step 2: Inter-subsequence Merge**

States are merged between multiple subsequences of the same original sequence:
$$\mathbf{S}_{m+1} = \mathbf{M}_m^{\text{local}} \cdot \mathbf{S}_m + \mathbf{B}_m^{\text{local}}$$

This is still chain composition of Affine transformations.

**Step 3: Final Computation**

Recompute output for each subsequence with correct initial state.

#### Relationship with CP Parallelism

| Parallelism Level | Split Dimension | Communication | Applicable Scenario |
|-------------------|-----------------|---------------|---------------------|
| **CP Parallelism** | Cross-GPU (inter-card) | NCCL All-Gather | Multi-card training/inference |
| **SM Parallelism** | Within single card (intra-card) | Shared memory | Single-card long sequence inference |

Both have the same mathematical essence: chain composition of Affine transformations, just at different granularities:
- CP Parallelism: rank level
- SM Parallelism: subsequence level

#### Implementation Points

1. **Dynamic splitting**: Dynamically compute `subseq_len` based on sequence length and number of SMs
2. **Split info management**: Maintain mapping between subsequences and original sequence
3. **Two-level computation**:
   - `intracard_pre_scan`: Parallelly compute local $(\mathbf{M}, \mathbf{B})$ for all subsequences
   - `intracard_merge`: Merge subsequence states of the same original sequence

> **Implementation reference**: `fla/ops/common/intracard_cp.py`

---

## Summary

We have established the complete mathematical framework for **KDA** (and **GDN** as its special case) from the most basic lemmas:

1. **Lemma 1**: Outer product accumulation = matrix multiplication → motivation for chunk-wise parallelism
2. **Lemma 2**: Inverse of lower triangular matrix → theoretical foundation for WY representation
3. **Lemma 3**: Decomposition of log-decay → matrix multiplication form of decay computation
4. **Challenge of KDA**: Residual depends on historical state
5. **WY Representation**: Separate dependencies to obtain $\tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W} \mathbf{S}$
6. **Core Theorem**: Chunk-wise Affine form $\mathbf{S}' = \mathbf{M} \mathbf{S} + \mathbf{B}$
7. **CP Parallelism**: Chain composition of Affine transformations

### Key Insights

- Essence of **WY Representation**: Explicitly separate parts dependent on historical state $\mathbf{S}$, making parallel computation possible
- Role of **Affine Form**: Provides a unified state update interface, naturally supporting multi-level parallelism (CP, SM)
- Advantage of **Per-dim decay**: Allows different feature dimensions to have independent forgetting rates, enhancing expressiveness

### Notation Conventions

- Lowercase $\mathbf{s}, \mathbf{k}, \mathbf{v}$: token-level vectors
- Uppercase $\mathbf{S}, \mathbf{K}, \mathbf{V}, \mathbf{M}, \mathbf{B}$: chunk-level matrices
- Distinguishing GDN (scalar decay) and KDA (per-dimension decay) only differs in the diagonal part of the transition matrix

---

## Appendix: GDN vs KDA

| Feature | GDN | KDA |
|---------|-----|-----|
| Decay | Scalar $\lambda$ | Vector $\boldsymbol{\lambda} \in \mathbb{R}^K$ |
| Transition | $\mathbf{M} = \lambda \mathbf{I} - \mathbf{K}^\top \mathbf{W}$ | $\mathbf{M} = \text{diag}(\boldsymbol{\lambda}) - \mathbf{K}^\top \mathbf{W}$ |
| Expressiveness | Global forgetting | Dimension-selective forgetting |
| Computation | Slightly faster | Slightly slower |

Both are Affine forms; only the diagonal part of $\mathbf{M}$ differs.

> **Reference**: Gated DeltaNet is detailed in Yang et al. (2024) [^5]; Kimi Delta Attention (KDA) is its extension in the per-dimension decay direction.

---

## References

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

*The mathematical derivations and algorithm descriptions in this article are based on the Flash Linear Attention (FLA) framework implementation.*
