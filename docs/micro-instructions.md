# Aurora AI Core · 微指令层规格

Author: 晨星

## 动机

常规 AI 库是一个黑盒：你调用 `matmul(A,B)`，得到结果，无从证明它算对了。
Aurora 的微指令层把执行**展开为原子操作序列**，每个原子操作配一条数学不变量，
执行即校验 —— 从「相信它是对的」变成「逐步验证它是对的」。

这对应「AI 系统十层分解」中最底层的**微指令级操作**愿景：
上层是算法，中层是算子，底层就是可被逐一审计的微指令。

## 数据模型

```
MicroOp := { id, invariant(output, inputs), doc }
Trace   := [ { id, ok, reason, summary } ]
```

- `defMicro(id, invariant, doc)` —— 定义原子操作及其不变量
- `record(id, inputs, output)` —— 执行一次，**立即校验**并写入 trace
- `report()` —— 汇总 `{ total, passed, failed, byOp, catalogSize }`

## 微指令目录（6 类语义微指令）

| 微指令 | 语义 | 数学不变量 | 说明 |
|---|---|---|---|
| `MAC` | 乘加 `C[i][j] += A[i][k]·B[k][j]` | `C[i][j] == Σ_k A[i][k]·B[k][j]` | 逐元素重算比对；矩阵乘展开后每条独立校验 |
| `SOFTMAX_ROW` | 行 softmax | `Σp == 1` 且 `0 ≤ p ≤ 1` | 概率单纯形约束 |
| `CHOL_DIAG` | Cholesky 对角元 | `L_ii² == A_ii − Σ_{k<i} L_ik²` | 分解定义式逐步校验 |
| `CHOL_OFF` | Cholesky 非对角元 | `L_ij·L_jj == A_ij − Σ L_ik·L_jk` | 同上 |
| `JACOBI_ROT` | Jacobi 旋转消去 | 旋转后非对角元不增 | 收敛性保证 |
| `COV_SYM` | 协方差对称 | `C == Cᵀ` | 对称性检查 |

## 运行时规模

一次 `microMatmul(A, B)`（`A` 为 `r×k`，`B` 为 `k×c`）会产出 **`r × c` 条 `MAC` 微指令**，
每条独立重算点积并与输出比对。因此：

- `8×5 @ 5×6` → 48 条 `MAC`
- 一次中等规模 `256×256` 矩阵乘 → 65,536 条 `MAC` 微指令，全部逐一校验

即「1024 项微指令级操作」的规模在运行时会**自然远超**千级；
当前预置目录为 6 类语义微指令，逐元素展开后微指令数量随矩阵规模线性/平方增长。

## 使用

```js
const { createMicro, microMatmul, microSoftmaxRows } = require('aurora-ai-core').Verify;

const M = createMicro();

const A = [[1, 2], [3, 4]];
const B = [[5, 6], [7, 8]];
const C = microMatmul(M, A, B);        // 4 条 MAC 微指令，逐条校验

const P = microSoftmaxRows(M, C);      // 2 条 SOFTMAX_ROW 微指令

console.log(M.report());
// {
//   total: 6, failed: 0, passed: 6,
//   byOp: { MAC: { n: 4, f: 0 }, SOFTMAX_ROW: { n: 2, f: 0 } },
//   catalogSize: 6
// }
```

## 自定义微指令

```js
const M = createMicro();

M.defMicro('RELU', (out, inp) => {
  // 不变量: out == max(0, x)
  return out === Math.max(0, inp.x);
}, 'ReLU 激活');

M.record('RELU', { x: -2.5 }, 0);   // → ok
M.record('RELU', { x:  2.5 }, 2.5); // → ok
M.record('RELU', { x:  2.5 }, 0);   // → 记录为失败
```

任何微指令校验失败都会被 trace 记录并计入 `report().failed`，不会被静默吞掉。

## 与验证层的关系

| 层 | 粒度 | 作用 |
|---|---|---|
| L4 验证层 | 算法级 | 证明「这个算法的结果满足它的数学定理」 |
| L5 微指令层 | 操作级 | 证明「算法的每一步原子操作都满足它的定义式」 |

两者互补：算法级不变量防止整体逻辑错误，微指令级不变量定位到具体哪一步出错。

## 路线图

- [ ] 目录扩展至 1024 项原子操作语义定义
- [ ] 微指令级性能剖析（按类型统计耗时）
- [ ] 微指令 trace 导出为可审计日志（JSON）
