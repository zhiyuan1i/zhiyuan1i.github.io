---
title: 'The Mathematics of DPLR (Diagonal Plus Low Rank): Parallel Computing with Explicit Transition Matrices'
date: '2026-02-21T10:44:23Z'
draft: false
math: true
translationKey: dplr-mathematics
tags: ['DPLR', 'Linear Attention', 'RWKV-7', 'Low Rank', 'WY Representation']
categories: ['Technical']
description: 'A deep dive into the chunk-wise parallel algorithm for DPLR, understanding the WY representation of explicit diagonal-plus-low-rank transition matrices, and exploring the unified framework with KDA/IPLR'
---

> This article assumes familiarity with linear algebra (matrix multiplication, outer products, inverse matrices) and basic sequence modeling concepts. It is recommended to read [The Mathematics of KDA](/en/posts/kda-mathematics/) first.

## Abstract

This article derives the **chunk-wise parallel algorithm** for **DPLR (Diagonal Plus Low Rank)**. DPLR is an important variant of the generalized Delta Rule, applied in architectures such as **RWKV-7**. The core contributions are:

1. Establishing the explicit transition matrix form of DPLR: $\mathbf{P}_t = \text{diag}(\exp(\mathbf{g}_t)) + \mathbf{b}_t \mathbf{a}_t^T$
2. Deriving the **WY representation** for DPLR, decomposing the cumulative transition matrix into diagonal and low-rank components
3. Proving that DPLR also satisfies the **Affine transformation** form, naturally supporting Context Parallelism (CP)
4. Comparing DPLR, KDA, and IPLR, revealing the unified mathematical framework of the linear attention family

Advantages of DPLR over standard Delta Rule: explicit control of diagonal decay (dim-wise forgetting) and low-rank updates, providing stronger expressiveness. However, in chunk form, it significantly introduces additional computational complexity and requires more HBM space to store intermediate variables.

---

## Table of Contents

1. [Introduction: From Delta Rule to DPLR](#introduction-from-delta-rule-to-dplr)
2. [Notation and Conventions](#notation-and-conventions)
3. [Core Lemmas](#core-lemmas)
4. [The Recurrent Form of DPLR](#the-recurrent-form-of-dplr)
5. [WY Representation: Decomposition of Cumulative Transition Matrices](#wy-representation-decomposition-of-cumulative-transition-matrices)
6. [Core Theorem: Chunk-wise Affine Form](#core-theorem-chunk-wise-affine-form)
7. [Algorithm Implementation: From Theory to Code](#algorithm-implementation-from-theory-to-code)
8. [DPLR vs KDA vs IPLR](#dplr-vs-kda-vs-iplr)
9. [CP Parallelism and Multi-Level Parallelism](#cp-parallelism-and-multi-level-parallelism)
10. [Summary](#summary)

---

## Introduction: From Delta Rule to DPLR

### Limitations of Delta Rule

The state update of standard Delta Rule (and KDA) can be written as:

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \beta_t \cdot \mathbf{k}_t^T (\mathbf{v}_t - \mathbf{k}_t \mathbf{s}_{t-1})$$

The transition matrix in this form is implicit:
- The state update is indirectly affected through the residual $(\mathbf{v}_t - \mathbf{k}_t \mathbf{s}_{t-1})$
- The forgetting mechanism is implemented through the gate $\boldsymbol{\lambda}_t$

Mathematically, this is equivalent to:

$$\mathbf{s}_t = (\mathbf{I} - \beta_t \mathbf{k}_t^T \mathbf{k}_t)\mathbf{s}_{t-1} + \beta_t \mathbf{k}_t^T \mathbf{v}_t$$

The transition matrix $\mathbf{I} - \beta_t \mathbf{k}_t^T \mathbf{k}_t$ is in the form of **identity matrix + low-rank (rank-1)**, known as the **IPLR (Identity Plus Low Rank)** structure.

### The Core Idea of DPLR

**DPLR (Diagonal Plus Low Rank)** adopts an **explicit transition matrix** form:

$$\mathbf{S}_t = \exp(\mathbf{g}_t) \odot \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t + \mathbf{b}_t (\mathbf{a}_t^T \mathbf{S}_{t-1})$$

Or more compactly:

$$\mathbf{S}_t = (\mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T) \mathbf{S}_{t-1} + \mathbf{k}_t^T\mathbf{v}_t$$

Where:
- $\mathbf{D}_t = \text{diag}(\exp(\mathbf{g}_t)) \in \mathbb{R}^{K \times K}$ is the diagonal decay matrix
- $\mathbf{a}_t, \mathbf{b}_t \in \mathbb{R}^{K \times 1}$ (column vectors) are the two vectors for low-rank update
- The transition matrix $\mathbf{P}_t = \mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T$ has the **Diagonal Plus Low Rank (DPLR)** structure

### Why "Diagonal Plus Low Rank"?

The structure of matrix $\mathbf{P}_t = \mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T$:
1. **Diagonal part** $\mathbf{D}_t$: Controls independent decay for each dimension
2. **Low-rank part** $\mathbf{b}_t \mathbf{a}_t^T$: Rank-1 update providing cross-dimensional coupling

This structure has been extensively studied in numerical linear algebra and is particularly suitable for fast matrix-vector multiplication.

### Relationship with RWKV-7

RWKV-7 adopts a **Dynamic State Evolution** architecture based on the DPLR concept. In our underlying parallel implementation, RWKV-7's state update formula is essentially a specific instantiation of the DPLR framework.

While traditional linear attention tries to directly match $\{k, v\}$ pairs, RWKV-7 simulates dynamic gradient descent to update the state $S$, guided by the L2 loss $L=\frac{1}{2} \left\Vert v - S k \right\Vert^2$. The theoretical update formula is:

$$S_t = S_{t-1} \text{Diag}(d_t) - \eta_t \cdot S_{t-1} k_t k_t^{\top} + \eta_t \cdot v_t k_t^{\top}$$

In the algorithm implementation, this gradient-based update is generalized into a more flexible DPLR form:

$$S_t = S_{t-1} \odot \exp(-e^{w_t}) + (S_{t-1} a_t) b_t^T + v_t k_t^T$$

The parameter mapping in our parallel system is as follows:
- **$w_t$** maps to the logarithmic decay term (specifically $-\exp(w_t)$)
- **$a_t$** maps to the low-rank update vector $a$ (dynamic learning rate modulator / in-context learning rate)
- **$b_t$** maps to the low-rank update vector $b$ (state update modulator)

These features enable RWKV-7 to achieve:
- **Dynamic Decay and Learning Rate**: $w_t, a_t, b_t$ are all data-dependent, allowing the model to dynamically determine the strength of forgetting and updating based on the context.
- **Enhanced Expressiveness**: By introducing explicit state evolution, RWKV-7 can recognize all regular languages. Its theoretical expressiveness surpasses TC0 (Transformers) and reaches NC1.
- **Seamless Integration with DPLR Chunk Parallelism**: Because its core is a DPLR structure, RWKV-7 can directly reuse the DPLR chunk-wise algorithm to achieve highly efficient parallel training for long sequences.

---

## Notation and Conventions

| Symbol | Dimensions | Meaning |
|--------|------------|---------|
| $\mathbf{s}_t$ | $\mathbb{R}^{K \times V}$ | Token-level state matrix |
| $\mathbf{S}$ | $\mathbb{R}^{K \times V}$ | Chunk-level initial state |
| $\mathbf{S}'$ | $\mathbb{R}^{K \times V}$ | Chunk-level final state |
| $\mathbf{k}_t, \mathbf{q}_t$ | $\mathbb{R}^{1 \times K}$ (row vectors) | Token-level key/query |
| $\mathbf{v}_t$ | $\mathbb{R}^{1 \times V}$ (row vector) | Token-level value |
| $\mathbf{a}_t, \mathbf{b}_t$ | $\mathbb{R}^{K \times 1}$ (column vectors) | Two vectors for low-rank update |
| $\mathbf{K}, \mathbf{V}$ | $\mathbb{R}^{C \times K}$ / $\mathbb{R}^{C \times V}$ | Chunk-level key/value matrices, row $i$ is $\mathbf{k}_i$ / $\mathbf{v}_i$ |
| $\mathbf{A}^{\text{lr}} \in \mathbb{R}^{C \times K}$ | Row $i$ is $\mathbf{a}_i^T$ | Matrix form of low-rank vector $\mathbf{a}$ (column vectors arranged as rows) |
| $\mathbf{B}^{\text{lr}} \in \mathbb{R}^{C \times K}$ | Row $i$ is $\mathbf{b}_i^T$ | Matrix form of low-rank vector $\mathbf{b}$ (column vectors arranged as rows) |
| $\mathbf{g}_t$ | $\mathbb{R}^{K}$ | Log decay vector (before cumsum) |
| $\mathbf{g}_t^{\text{cum}}$ | $\mathbb{R}^{K}$ | Cumulative log decay (after cumsum) |
| $\mathbf{D}_t = \text{diag}(\exp(\mathbf{g}_t^{\text{cum}}))$ | $\mathbb{R}^{K \times K}$ | Diagonal decay matrix |
| $\boldsymbol{\Gamma}_i^t = \prod_{j=i}^t \mathbf{D}_j$ | $\mathbb{R}^{K \times K}$ | Cumulative diagonal decay matrix |
| $\mathbf{P}_t = \mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T$ | $\mathbb{R}^{K \times K}$ | Transition matrix (low-rank outer product form) |
| $\mathbf{A}_{ab}, \mathbf{A}_{ak}$ | $\mathbb{R}^{C \times C}$ | Strictly lower-triangular attention matrices |
| $\mathbf{W}, \mathbf{U}$ | $\mathbb{R}^{C \times K}$ / $\mathbb{R}^{C \times V}$ | Weighted matrices in WY representation |
| $\mathbf{w}_i, \mathbf{u}_i$ | $\mathbb{R}^{K}$ / $\mathbb{R}^{V}$ | Weighted vectors in WY representation (the $i$-th component) |
| $\tilde{\mathbf{u}}_i$ | $\mathbb{R}^{V}$ | Corrected vector including historical state contributions |
| $\mathbf{M}$ | $\mathbb{R}^{K \times K}$ | Affine transition matrix |
| $\mathbf{B}$ | $\mathbb{R}^{K \times V}$ | Affine bias matrix |
| $\odot$ | - | Hadamard product (element-wise multiplication) |

**Important Conventions**:
- In the `flash-linear-attention` implementation, DPLR adopts the **left-multiplication** form: $\mathbf{S}_t = \mathbf{P}_t \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t$
- State matrix $\mathbf{S} \in \mathbb{R}^{K \times V}$ (key dim × value dim)

*Note: The native RWKV-7 formula uses the dual **right-multiplication** form, where the state matrix is $\mathbf{S}_{\text{rwkv}} \in \mathbb{R}^{V \times K}$ and the update is $\mathbf{S}_t = \mathbf{S}_{t-1} \mathbf{P}_t^T + \mathbf{v}_t \mathbf{k}_t^T$. In the FLA framework, to maintain consistency with KDA and other linear attention mechanisms, we transposed the state matrix to unify under the left-multiplication form.*

**Comparison with KDA**:

| Property | KDA | DPLR (FLA Implementation) | RWKV-7 Native |
|----------|-----|------|------|
| Multiplication Direction | Left | Left | Right |
| State Dimensions | $\mathbb{R}^{K \times V}$ | $\mathbb{R}^{K \times V}$ | $\mathbb{R}^{V \times K}$ |
| Affine Form | $\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$ | $\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$ | $\mathbf{S}' = \mathbf{S}\mathbf{M}^T + \mathbf{B}^T$ |
| Transition Matrix | Implicit (Delta Rule) | Explicit (DPLR) |

---

## Core Lemmas

### Lemma 1: Inverse of Lower Triangular Matrices

Let $\mathbf{L} \in \mathbb{R}^{C \times C}$ be a unit lower triangular matrix (diagonal entries are 1, upper triangle is 0), then $\mathbf{L}^{-1}$ is also unit lower triangular and can be computed via forward substitution.

In particular, if $\mathbf{L} = \mathbf{I} - \mathbf{N}$, where $\mathbf{N}$ is strictly lower triangular (diagonal entries are 0), then:

$$\mathbf{L}^{-1} = \mathbf{I} + \mathbf{N} + \mathbf{N}^2 + \cdots + \mathbf{N}^{C-1}$$

**Proof**: Directly verify that $(\mathbf{I} - \mathbf{N})(\mathbf{I} + \mathbf{N} + \cdots + \mathbf{N}^{C-1}) = \mathbf{I} - \mathbf{N}^C = \mathbf{I}$ (since $\mathbf{N}^C = 0$).

### Lemma 2: Product Structure of DPLR Matrices

Let $\mathbf{P}_i = \mathbf{D}_i + \mathbf{b}_i \mathbf{a}_i^T$, where $\mathbf{D}_i$ is a diagonal matrix. Then the **reverse** cumulative product $\mathbf{P}_{t:1} = \prod_{i=t}^1 \mathbf{P}_i = \mathbf{P}_t \mathbf{P}_{t-1} \cdots \mathbf{P}_1$ can be expressed as:

$$\mathbf{P}_{t:1} = \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot (\mathbf{a}_i^T \boldsymbol{\Gamma}_1^{i-1})$$

**Note on product direction**: The product here accumulates from right to left ($\mathbf{P}_t$ on the leftmost), consistent with the form obtained by expanding the state recurrence $\mathbf{S}_t = \mathbf{P}_t \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t$. In the expanded summation terms, $\boldsymbol{\Gamma}_{i+1}^t$ is the cumulative decay to the left of $\mathbf{b}_i$ (from $i+1$ to $t$), and $\boldsymbol{\Gamma}_1^{i-1}$ is the cumulative decay to the right of $\mathbf{a}_i^T$ (from $1$ to $i-1$).

**Significance**: This lemma guarantees that the DPLR structure is closed under matrix multiplication, forming the foundation for the existence of the WY representation. The specific form shows that the cumulative product maintains a "diagonal + low-rank" structure.

### Lemma 3: Decomposition of Logarithmic Decay

For cumulative logarithmic decay, we have:

$$\exp(\mathbf{g}_i^{\text{cum}} - \mathbf{g}_j^{\text{cum}}) = \exp(\mathbf{g}_i^{\text{cum}}) \odot \exp(-\mathbf{g}_j^{\text{cum}})$$

This allows the decay computation to be expressed as the outer product of two gated vectors.

---

## The Recurrent Form of DPLR

### Basic Recurrence

The state update equation for DPLR is:

$$\mathbf{S}_t = \exp(\mathbf{g}_t) \odot \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t + \mathbf{b}_t (\mathbf{a}_t^T \mathbf{S}_{t-1})$$

Or in matrix form:

$$\mathbf{S}_t = (\mathbf{D}_t + \mathbf{b}_t \mathbf{a}_t^T) \mathbf{S}_{t-1} + \mathbf{k}_t^T \mathbf{v}_t$$

Where:
- First term $\mathbf{S}_{t-1} \odot \exp(\mathbf{g}_t)$: Dimension-wise decay (Hadamard product form)
- Second term $\mathbf{k}_t^T \mathbf{v}_t$: Standard key-value outer product update
- Third term $\mathbf{b}_t (\mathbf{a}_t^T \mathbf{S}_{t-1})$: Low-rank update, projecting state through $\mathbf{a}_t^T$ (yielding $1 \times V$) and expanding through $\mathbf{b}_t$ (yielding $K \times V$)

### Expanding the Recurrence

To understand chunk-wise parallelism, let's expand the first few time steps:

$$
\begin{aligned}
\mathbf{S}_1 &= \mathbf{P}_1 \mathbf{S}_0 + \mathbf{k}_1^T \mathbf{v}_1 \\
\mathbf{S}_2 &= \mathbf{P}_2 \mathbf{S}_1 + \mathbf{k}_2^T \mathbf{v}_2 \\
&= \mathbf{P}_2 (\mathbf{P}_1 \mathbf{S}_0 + \mathbf{k}_1^T \mathbf{v}_1) + \mathbf{k}_2^T \mathbf{v}_2 \\
&= \mathbf{P}_2 \mathbf{P}_1 \mathbf{S}_0 + \mathbf{P}_2 \mathbf{k}_1^T \mathbf{v}_1 + \mathbf{k}_2^T \mathbf{v}_2
\end{aligned}
$$

General form:
$$\mathbf{S}_t = \left( \prod_{i=t}^1 \mathbf{P}_i \right) \mathbf{S}_0 + \sum_{i=1}^t \left( \prod_{j=t}^{i+1} \mathbf{P}_j \right) \mathbf{k}_i^T \mathbf{v}_i$$

**Challenge**: Directly computing the cumulative transition matrix $\mathbf{P}_{t:1} = \prod_{i=t}^1 \mathbf{P}_i$ requires $O(t)$ matrix multiplications. How can we achieve parallelism?

---

## WY Representation: Decomposition of Cumulative Transition Matrices

### Core Problem

We need to efficiently represent the product of cumulative transition matrices (note the left-multiplication order, accumulating from right to left):
$$\mathbf{P}_{t:1} = \prod_{i=t}^1 (\mathbf{D}_i + \mathbf{b}_i \mathbf{a}_i^T)$$

**Key Insight**: The product of diagonal-plus-low-rank matrices retains the "diagonal + low-rank" structure and can be decomposed into diagonal accumulation plus weighted sums of low-rank outer products.

### Defining Cumulative Diagonal Decay

Let:
$$\boldsymbol{\Gamma}_i^t = \prod_{j=i}^t \mathbf{D}_j = \text{diag}\left(\exp\left(\sum_{j=i}^t \mathbf{g}_j\right)\right)$$

When $i > t$, define $\boldsymbol{\Gamma}_i^t = \mathbf{I}$ (identity matrix).

### Theorem (WY Representation for DPLR)

The cumulative transition matrix can be decomposed as:

$$\mathbf{P}_{t:1} = \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot (\mathbf{a}_i^T \boldsymbol{\Gamma}_1^{i-1})$$

> **Motivation for Definition**: To make the WY representation more compact, we define the weighted vector $\mathbf{w}_i^T$ (row vector), which accumulates the influence of all historical low-rank updates up to step $i$. This is analogous to how the classical WY representation accumulates the weights of Householder transformations.
>
> Or equivalently, define $\mathbf{w}_i^T = \mathbf{a}_i^T \boldsymbol{\Gamma}_1^{i-1} + \sum_{j=1}^{i-1} (\mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j) \cdot \mathbf{w}_j^T$:

$$\mathbf{P}_{t:1} = \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot \mathbf{w}_i^T$$

Where the coefficient $(\mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j)$ is a scalar.

> **Connection to Classical WY Representation**: The classical WY representation decomposes the product of Householder matrices as $\mathbf{Q} = \mathbf{I} - \mathbf{W}\mathbf{Y}^T$. The DPLR WY representation is its generalization: replacing $\mathbf{I}$ with $\boldsymbol{\Gamma}_1^t$ (diagonal accumulation) and replacing the standard low-rank outer product with a weighted sum.

### Proof (by Induction)

**Base case** $t=1$:
$$\mathbf{P}_1 = \mathbf{D}_1 + \mathbf{b}_1 \mathbf{a}_1^T = \boldsymbol{\Gamma}_1^1 + (\boldsymbol{\Gamma}_2^1 \mathbf{b}_1) \cdot \mathbf{w}_1^T$$

Since $\boldsymbol{\Gamma}_1^1 = \mathbf{D}_1$, $\boldsymbol{\Gamma}_2^1 = \mathbf{I}$, $\mathbf{w}_1^T = \mathbf{a}_1^T$, the equality holds.

**Inductive step**: Assume the formula holds for $t$, prove for $t+1$.

$$
\begin{aligned}
\mathbf{P}_{t+1:1} &= \mathbf{P}_{t+1} \cdot \mathbf{P}_{t:1} \\
&= (\mathbf{D}_{t+1} + \mathbf{b}_{t+1} \mathbf{a}_{t+1}^T)\left(\boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot \mathbf{w}_i^T\right) \\
&= \boldsymbol{\Gamma}_1^{t+1} + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^{t+1} \mathbf{b}_i) \cdot \mathbf{w}_i^T \\
&\quad + \mathbf{b}_{t+1} \cdot \underbrace{\left(\mathbf{a}_{t+1}^T \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\mathbf{a}_{t+1}^T \boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot \mathbf{w}_i^T\right)}_{\eqqcolon \mathbf{w}_{t+1}^T} \\
&= \boldsymbol{\Gamma}_1^{t+1} + \sum_{i=1}^{t+1} (\boldsymbol{\Gamma}_{i+1}^{t+1} \mathbf{b}_i) \cdot \mathbf{w}_i^T
\end{aligned}
$$

Where we used $\boldsymbol{\Gamma}_{t+2}^{t+1} = \mathbf{I}$. Q.E.D.

### WY Representation of State

Substituting the WY representation into the state recurrence, we obtain:

$$\mathbf{S}_t = \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{k}_i^T \mathbf{v}_i + \boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i \mathbf{u}_i^T)$$

Where $\mathbf{u}_i^T$ ($1 \times V$ row vector) satisfies:

$$
\mathbf{u}_i^T = \begin{cases}
\mathbf{0}, & i=1 \\
\sum_{j=1}^{i-1} (\mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{k}_j^T \mathbf{v}_j + \mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j \mathbf{u}_j^T), & i \geq 2
\end{cases}
$$

### Matrix Form of Linear System

Define matrices within a chunk (row $i$ is the corresponding vector, the following applies to the left-multiplication DPLR):
- $\mathbf{A}_{ab} \in \mathbb{R}^{C \times C}$: $[\mathbf{A}_{ab}]_{ij} = \mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j$ for $i > j$
- $\mathbf{A}_{ak} \in \mathbb{R}^{C \times C}$: $[\mathbf{A}_{ak}]_{ij} = \mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{k}_j^T$ for $i > j$

Then $(\mathbf{I} + \mathbf{A}_{ab})$ is a unit lower triangular matrix. Let:
- $\mathbf{A}^{\text{gate}} = \mathbf{A}^{\text{lr}} \odot \exp(\mathbf{G}^{\text{cum}}) \in \mathbb{R}^{C \times K}$ (gated low-rank vector matrix), where $\mathbf{A}^{\text{lr}} \in \mathbb{R}^{C \times K}$ has row $i$ as $\mathbf{a}_i^T$, and $\mathbf{G}^{\text{cum}}$ has row $i$ as $\mathbf{g}_i^{\text{cum}}$

The matrix form of the WY representation is:

$$\mathbf{W} = (\mathbf{I} + \mathbf{A}_{ab})^{-1} \mathbf{A}^{\text{gate}}$$

$$\mathbf{U} = (\mathbf{I} + \mathbf{A}_{ab})^{-1} \mathbf{A}_{ak} \mathbf{V}$$

This is structurally similar to the WY representation in KDA. The difference is: in KDA $\tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W}\mathbf{S}$ (minus sign, from Delta Rule residual), while in DPLR $\tilde{\mathbf{U}} = \mathbf{U} + \mathbf{W}\mathbf{S}$ (plus sign, from low-rank superposition). This leads to different signs in the Affine parameter $\mathbf{M}$: KDA uses $\text{diag}(\cdot) - \mathbf{K}^T \mathbf{W}$, while DPLR uses $\text{diag}(\cdot) + \mathbf{B}^T \mathbf{W}$.

---

## Core Theorem: Chunk-wise Affine Form

### Theorem (Chunk-wise Affine Form for DPLR)

Let the state at the beginning of a chunk be $\mathbf{S} \in \mathbb{R}^{K \times V}$, then the state at the end of the chunk is:

$$\mathbf{S}' = \mathbf{M} \mathbf{S} + \mathbf{B}$$

Where:
- **Transition matrix** $\mathbf{M} \in \mathbb{R}^{K \times K}$:
  $$\mathbf{M} = \text{diag}(\exp(\mathbf{g}_{\text{last}})) + \mathbf{B}_{\text{decayed}}^T \mathbf{W}$$
- **Bias matrix** $\mathbf{B} \in \mathbb{R}^{K \times V}$:
  $$\mathbf{B} = \mathbf{K}_{\text{decayed}}^T \mathbf{V} + \mathbf{B}_{\text{decayed}}^T \mathbf{U}$$

And the chunk output is:

$$\mathbf{O} = \mathbf{Q} \mathbf{S} + \text{mask}(\mathbf{A}_{qk}) \mathbf{V} + \text{mask}(\mathbf{A}_{qb}) (\mathbf{U} + \mathbf{W} \mathbf{S})$$

### Proof

**State Update**:

$$
\begin{aligned}
\mathbf{S}' &= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \sum_{i=1}^C \exp(\mathbf{g}_{\text{last}} - \mathbf{g}_i) \odot (\mathbf{k}_i^T \mathbf{v}_i + \mathbf{b}_i \tilde{\mathbf{u}}_i) \\
&= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \mathbf{K}_{\text{decayed}}^T \mathbf{V} + \mathbf{B}_{\text{decayed}}^T \tilde{\mathbf{U}}
\end{aligned}
$$

Where $\tilde{\mathbf{u}}_i = \mathbf{u}_i + \mathbf{w}_i \mathbf{S}$ ($1 \times V$ row vector) is the corrected vector including historical state contributions. Here $\mathbf{w}_i \in \mathbb{R}^{1 \times K}$ (row vector), $\mathbf{S} \in \mathbb{R}^{K \times V}$, and the product $\mathbf{w}_i \mathbf{S} \in \mathbb{R}^{1 \times V}$, with matching dimensions.

Substituting the WY representation's matrix form $\tilde{\mathbf{U}} = \mathbf{U} + \mathbf{W} \mathbf{S}$ (note the **plus sign** here, different from KDA where $\tilde{\mathbf{V}} = \mathbf{U} - \mathbf{W} \mathbf{S}$ uses a **minus sign**. The reason is that KDA's WY representation separates the *residual* $\mathbf{v}_i - \mathbf{k}_i \mathbf{S}$ from the Delta Rule, where the minus comes from "subtracting historical prediction"; DPLR has no Delta Rule structure, and the low-rank part $\mathbf{b}_i \mathbf{a}_i^T$ is directly *superimposed* onto the state, so the contribution from historical states accumulates positively):

$$
\begin{aligned}
\mathbf{S}' &= \text{diag}(\exp(\mathbf{g}_{\text{last}})) \mathbf{S} + \mathbf{K}_{\text{decayed}}^T \mathbf{V} + \mathbf{B}_{\text{decayed}}^T (\mathbf{U} + \mathbf{W} \mathbf{S}) \\
&= \underbrace{(\text{diag}(\exp(\mathbf{g}_{\text{last}})) + \mathbf{B}_{\text{decayed}}^T \mathbf{W})}_{\mathbf{M}} \mathbf{S} + \underbrace{(\mathbf{K}_{\text{decayed}}^T \mathbf{V} + \mathbf{B}_{\text{decayed}}^T \mathbf{U})}_{\mathbf{B}}
\end{aligned}
$$

(Note: Detailed derivation of cross terms requires considering the specific relationship between $\mathbf{W}$ and $\mathbf{K}_{\text{decayed}}$; the main structure is presented here.)

**Output computation** follows similarly.

---

## Algorithm Implementation: From Theory to Code

Based on the above theorems, the chunk-wise algorithm for DPLR proceeds as follows:

```python
def chunk_dplr(K, V, A, B_lr, G, chunk_size=64):
    """
    K, V: [C, K], [C, V] - keys, values
    A, B_lr: [C, K] - low-rank vectors a, b
    G: [C, K] - cumulative log decay
    """
    # Step 1: Compute gated inputs
    # Note: Code uses relative decay trick for numerical stability
    ag = A * exp(G)           # gated a (using ge, i.e., shifted cumsum)
    bg = B_lr * exp(-G + G[-1])  # gated b (relative decay)
    kg = K * exp(-G + G[-1])  # gated k (relative decay)
    qg = Q * exp(G)           # gated q (forward gating)
    
    # Step 2: Compute lower triangular matrices A_ab and A_ak
    # A_ab[i,j] = dot(a_i * exp(g_i - g_j), b_j) for i > j
    A_ab = (ag @ (B_lr * exp(-G)).T).masked_fill_(triu_mask, 0)
    A_ak = (ag @ (K * exp(-G)).T).masked_fill_(triu_mask, 0)
    
    # Step 3: Compute (I + A_ab)^{-1} via forward substitution
    A_ab_inv = forward_substitution_inverse(I + A_ab)
    
    # Step 4: WY representation
    # w = A_ab_inv @ ag
    # u = A_ab_inv @ (A_ak @ v)
    W = A_ab_inv @ ag    # [C, K]
    U = A_ab_inv @ (A_ak @ V)  # [C, V]
    
    # Step 5: Compute Affine parameters
    decay_last = exp(G[-1])  # [K]
    K_decayed = K * exp(G[-1] - G)  # [C, K]
    B_decayed = B_lr * exp(G[-1] - G)  # [C, K]
    
    # Transition matrix M
    M = diag(decay_last) + B_decayed.T @ W  # [K, K]
    
    # Bias matrix B (contributions from k and b)
    B_mat = K_decayed.T @ V + B_decayed.T @ U  # [K, V]
    
    # Step 6: State update (if initial state S=0, then S_next = B_mat)
    S_next = M @ S + B_mat
    
    # Step 7: Compute chunk output
    # O = Q @ S + masked_attention
    # Note: qg is gated query
    O_local = mask(qg @ K.T) @ V + mask(qg @ B_lr.T) @ U
    
    return M, B_mat, S_next, W, U
```

### Key Implementation Details

1. **Matrix Inversion**: $(\mathbf{I} + \mathbf{A}_{ab})^{-1}$ is the inverse of a unit lower triangular matrix, which can be computed via forward substitution in $O(C^3)$ time ($C$ is the chunk size, typically 64 or 128)

2. **Relative Decay Trick**: The code uses $\exp(-\mathbf{g} + \mathbf{g}_{\text{last}})$ rather than directly using $\exp(\mathbf{g})$, for numerical stability

3. **Index Absorption Convention**: In the code, `ag = A * exp(G)` absorbs $\exp(\mathbf{g}_i)$ into $\mathbf{a}_i$, so the computed $\mathbf{A}_{ab}$ is actually $[\mathbf{A}_{ab}]_{ij} = \mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i} \mathbf{b}_j$ (including the $\mathbf{g}_i$ factor), rather than $\mathbf{a}_i^T \boldsymbol{\Gamma}_{j+1}^{i-1} \mathbf{b}_j$ from the mathematical definition. Correspondingly, the computed $\mathbf{W}$ also absorbs this extra factor, ensuring the final Affine parameters $\mathbf{M}, \mathbf{B}$ remain correct. This absorption simplifies code implementation by avoiding explicit index shifts

4. **Block-wise Computation**: When $K$ is large, key/value dimensions need to be blocked to fit GPU Shared Memory

5. **Precision Control**: Similar to KDA, intermediate computations use float32, while storage uses bf16/fp16

---

## DPLR vs KDA vs IPLR

### A Unified Perspective on Three Variants

| Variant | Transition Matrix | Multiplication Direction | Core Feature |
|---------|-------------------|--------------------------|--------------|
| **IPLR** | $\mathbf{I} + \mathbf{b}\mathbf{a}^T$ | Right (historically) | Identity + Low Rank, no explicit decay |
| **KDA** | Implicit (via Delta Rule) | Left | Per-dim decay + Delta Rule |
| **DPLR** | $\text{diag}(\exp(\mathbf{g})) + \mathbf{b}\mathbf{a}^T$ | Left | Diagonal decay + Low Rank |

### Mathematical Connections

1. **IPLR is a special case of DPLR**: When $\mathbf{g}_t = \mathbf{0}$ (i.e., $\mathbf{D}_t = \mathbf{I}$), DPLR reduces to IPLR

2. **Duality between RWKV-7 and DPLR**:
   - DPLR (FLA): $\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$ (left multiplication, column-space update)
   - RWKV-7: $\mathbf{S}' = \mathbf{S}\mathbf{M}^T + \mathbf{B}^T$ (right multiplication, row-space update)

3. **Unified Framework**: Both ultimately reduce to the **Affine transformation** form

---

## CP Parallelism and Multi-Level Parallelism

### Affine Chain Rule (Left-Multiplication Version)

DPLR state updates also satisfy the Affine form and permit chain composition:

Let:
- $\mathbf{S}_1 = \mathbf{M}_0 \mathbf{S}_0 + \mathbf{B}_0$
- $\mathbf{S}_2 = \mathbf{M}_1 \mathbf{S}_1 + \mathbf{B}_1$

Then:
$$\mathbf{S}_2 = \underbrace{(\mathbf{M}_1 \mathbf{M}_0)}_{\mathbf{M}_{01}} \mathbf{S}_0 + \underbrace{(\mathbf{M}_1 \mathbf{B}_0 + \mathbf{B}_1)}_{\mathbf{B}_{01}}$$

### CP Parallelism Algorithm

Similar to KDA:

1. **Local Computation**: Each rank assumes $\mathbf{S} = \mathbf{0}$ and computes $(\mathbf{M}_r, \mathbf{B}_r)$
2. **All-Gather**: Collect Affine parameters from all ranks
3. **Prefix Scan**: Rank $r$ computes the true initial state
   $$\mathbf{S}_r = \sum_{j=0}^{r-1} \left( \prod_{k=j+1}^{r-1} \mathbf{M}_k \right) \mathbf{B}_j$$
4. **Local Recomputation**: Recompute chunk outputs with correct $\mathbf{S}_r$

### SM Parallelism

Also applicable. Long sequences are divided into multiple subsequences, and states are merged through two-level Affine composition.

---

## Summary

We have established a complete mathematical theory for **DPLR** from the perspective of explicit transition matrices:

1. **Core of DPLR**: Diagonal-plus-low-rank transition matrix $\mathbf{P}_t = \text{diag}(\exp(\mathbf{g}_t)) + \mathbf{b}_t \mathbf{a}_t^T$
2. **WY Representation**: Decomposing the cumulative transition matrix into diagonal and low-rank components
   $$\mathbf{P}_{t:1} = \boldsymbol{\Gamma}_1^t + \sum_{i=1}^t (\boldsymbol{\Gamma}_{i+1}^t \mathbf{b}_i) \cdot \mathbf{w}_i^T$$
3. **Chunk-wise Affine**: $\mathbf{S}' = \mathbf{M}\mathbf{S} + \mathbf{B}$
4. **Unified Framework**: DPLR, KDA, and IPLR are all special cases of Affine transformations, supporting the same parallel paradigms

---

*The mathematical derivations in this article are based on our theoretical framework and implementations in Flash Linear Attention (FLA).*
