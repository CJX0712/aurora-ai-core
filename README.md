# Aurora AI Core

> **世界顶级、可验证、零依赖的 AI 计算内核**
> 数值线性代数 · 机器学习 · 深度学习 · 微指令验证层
> Author: **晨星**

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![deps](https://img.shields.io/badge/dependencies-0-green.svg)](./package.json)
[![tests](https://img.shields.io/badge/verification-35%2F35-brightgreen.svg)](./tests/run.js)

---

## 为什么是 Aurora

绝大多数 AI 库只回答「能不能跑」。Aurora 回答更难的问题：**「凭什么相信它是对的？」**

每个算法都附带一条可被独立交叉验证的**数学不变量**（解析解 vs 数值解、经典定理 vs 实现输出），
全部在无头 Node 中自动执行 —— 当前 **35/35 全绿**，数值精度稳定在机器精度量级。

| 核心原则 | 落地方式 |
|---|---|
| **零重写** | 直接复用世界顶级经典算法：Jacobi(1846)、Cholesky(1924)、EM(Dempster 1977)、Lloyd k-Means、CART(Breiman 1984)、反向模式自动微分(Griewank 1989)、缩放点积注意力(Vaswani 2017)。以纯 JS 零依赖实现，而非造轮子。 |
| **集大成** | 数值内核 → 机器学习 → 深度学习 → 验证层 → 微指令层，五层贯通，共享同一套不变量契约。 |
| **可验证** | 35 条数学不变量自动校验；梯度检验误差 ~1e-9 量级。 |
| **闭环** | 无头 Node 可复现全绿 + 浏览器一键自检，零构建、零依赖。 |

---

## 快速开始

```bash
git clone https://github.com/CJX0712/aurora-ai-core.git
cd aurora-ai-core
npm test        # 或 node tests/run.js
```

零依赖，无需 `npm install`。预期输出：

```
[num]    11/11 通过
[ml]     16/16 通过
[dl]      5/5  通过
[micro]   3/3  通过
总计: 35/35 通过, 失败 0
```

### Node 使用

```js
const { Num, ML, DL, Verify } = require('aurora-ai-core');

// PCA：协方差对称半正定，主成分正交
const X = Num.randn(100, 4, 42);
const p = ML.pca(X, { components: 2 });
console.log(p.explainedVariance);   // 特征值 (降序)
console.log(p.Z);                   // 降维后的投影

// 多头注意力 (带梯度自检)
const mha = DL.multiHeadAttention(16, 4);
const gc  = DL.gradCheck((ps) => DL.sumAll(mha.forward(X)), mha.params);
console.log(gc.maxRel);             // ≈ 1e-9
```

### 浏览器使用

打开 `index.html`（双击即可，file:// 直接可跑，无需服务器、无需构建）：
按顺序引入 `src/num.js → ml.js → dl.js → verify.js`，使用全局 `Aurora`。

---

## 架构

```
┌──────────────────────────────────────────────────────────┐
│  L5  微指令层   verify.js  MicroEngine · 逐元素可审计微指令  │
├──────────────────────────────────────────────────────────┤
│  L4  验证层     verify.js  Invariant Harness (35 条不变量)  │
├──────────────────────────────────────────────────────────┤
│  L3  深度学习   dl.js      Autograd · MLP · Multi-Head Attn│
├──────────────────────────────────────────────────────────┤
│  L2  机器学习   ml.js      PCA·GMM·kMeans·kNN·NB·Reg·CART  │
├──────────────────────────────────────────────────────────┤
│  L1  数值内核   num.js     Cholesky·Jacobi·SVD·logsumexp   │
└──────────────────────────────────────────────────────────┘
```

详见 [`docs/architecture.md`](./docs/architecture.md)。

---

## 验证报告（实测）

| 分组 | 通过 | 代表性不变量与实测值 |
|---|---:|---|
| **num** | 11/11 | Cholesky 重构误差 `0.00e+0` · Jacobi 特征残差 `2.26e-15` · 特征向量正交 `2.22e-16` · SVD 重构 `8.88e-16` · 逆×矩阵=I `1.33e-15` |
| **ml** | 16/16 | PCA 主成分正交 `2.22e-16` · GMM 权重Σ=1 · GMM 对数似然单调不减 · 线性回归残差⊥X `2.70e-14` · 逻辑回归梯度检验 `2.64e-11` · 决策树训练准确率 `1.00` |
| **dl** | 5/5 | 自动微分梯度检验 `3.05e-9` · 多头注意力梯度检验 `2.85e-9` · MLP 解 XOR 准确率 `1.00`（loss `0.695 → -0.000`） |
| **micro** | 3/3 | 微指令 52 条全部通过不变量校验，0 失败 |

完整不变量目录见 [`docs/invariants.md`](./docs/invariants.md)。

---

## 创新点：微指令验证层

传统库是一个黑盒函数。Aurora 把算法执行**分解为可审计的微指令**，每一步带数学不变量自动校验：

```js
const { createMicro, microMatmul } = require('aurora-ai-core').Verify;
const M = createMicro();
const C = microMatmul(M, A, B);   // 每个 MAC 微指令都被记录并校验
console.log(M.report());
// { total: 52, failed: 0, passed: 52, byOp: { MAC: {n:48,f:0}, SOFTMAX_ROW: {n:4,f:0} }, catalogSize: 6 }
```

微指令目录（`send = 6` 类，均带数学不变量）：

| 微指令 | 数学不变量 |
|---|---|
| `MAC` | `C[i][j] == Σ_k A[i][k]·B[k][j]` |
| `SOFTMAX_ROW` | `Σp == 1`，且 `0 ≤ p ≤ 1` |
| `CHOL_DIAG` | `L_ii² == A_ii - Σ_{k<i} L_ik²` |
| `CHOL_OFF` | `L_ij·L_jj == A_ij - Σ L_ik·L_jk` |
| `JACOBI_ROT` | 旋转后非对角元不增 |
| `COV_SYM` | 协方差对称 `C == Cᵀ` |

详见 [`docs/micro-instructions.md`](./docs/micro-instructions.md)。

---

## 模块一览

| 模块 | 能力 |
|---|---|
| `src/num.js` | 矩阵运算、Cholesky 分解与求解、SPD 求逆、Jacobi 对称特征分解、SVD、`logsumexp`/`softmax`/`sigmoid`、对称性与 PSD 判定、正交性与残差校验 |
| `src/ml.js` | PCA、GMM(EM)、k-Means(k-means++ 初始化)、k-NN、高斯朴素贝叶斯、线性回归(正规方程)、逻辑回归(GD)、CART 决策树 |
| `src/dl.js` | 反向模式自动微分（`V` 张量 + 14 个可微算子）、MLP（含 XOR 求解）、多头缩放点积注意力、通用 `gradCheck` |
| `src/verify.js` | 不变量测试框架 + 微指令引擎（`MicroEngine`）、微指令版矩阵乘与 softmax |

---

## 设计取舍（诚实记录）

- **精度优先于速度**：矩阵为 `number[][]`，可读性/可验证性优先；生产级性能场景应替换为 BLAS/WASM 后端（架构已留出接口）。
- **EM 的局部最优**：GMM 使用 k-means++ 初始化以降低局部最优风险；不变量保证「对数似然单调不减」，不保证全局最优。
- **SVD 路径**：对一般矩阵走 `AᵀA` 的对称特征分解，数值稳健且复用同一 Jacobi 内核；极病态矩阵建议改用双边 Jacobi。

---

## 路线图

- [ ] 卷积 / RNN / LSTM 与对应不变量
- [ ] 强化学习（Q-Learning、PPO）与 Bellman 残差不变量
- [ ] 微指令目录扩展至 1024 项原子操作（当前 6 类语义微指令，运行时逐元素展开）
- [ ] WASM/BLAS 后端，保持同一套不变量契约

---

## 作者

**晨星** · GitHub [@CJX0712](https://github.com/CJX0712)

## 许可

[MIT](./LICENSE)
