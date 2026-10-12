/*
 * Aurora AI Core — Numerical Foundation
 * Author: 晨星 (Morning Star)
 *
 * 零重写原则：以下算法均复用世界顶级数值线性代数经典最优解
 * (LAPACK/BLAS 思想, Jacobi 1846, Cholesky 1924, Golub 1970s)，
 * 以纯 JS 零依赖实现，并暴露可被独立交叉验证的数学不变量。
 *
 * 设计：全局命名空间 Aurora.Num + CommonJS module.exports（浏览器/Node 双兼容）。
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = lib;
  const g = (typeof globalThis !== 'undefined') ? globalThis : root;
  g.Aurora = g.Aurora || {};
  g.Aurora.Num = lib;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------- 确定性随机：mulberry32 (可复现) ----------
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // 标准正态 (Box-Muller)
  function randn(r, c, seed) {
    const rnd = rng(seed == null ? 1234567 : seed);
    const m = [];
    for (let i = 0; i < r; i++) {
      const row = [];
      for (let j = 0; j < c; j++) {
        let u = 0, v = 0;
        while (u === 0) u = rnd();
        while (v === 0) v = rnd();
        row.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
      }
      m.push(row);
    }
    return m;
  }

  // ---------- 矩阵基础 (number[][], 行主序) ----------
  const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  const identity = (n) => { const m = zeros(n, n); for (let i = 0; i < n; i++) m[i][i] = 1; return m; };
  const copy = (m) => m.map(r => r.slice());
  const transpose = (m) => { const r = m.length, c = m[0].length; const t = zeros(c, r); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) t[j][i] = m[i][j]; return t; };
  const add = (a, b) => a.map((row, i) => row.map((v, j) => v + b[i][j]));
  const sub = (a, b) => a.map((row, i) => row.map((v, j) => v - b[i][j]));
  const scale = (a, s) => a.map(row => row.map(v => v * s));
  const hadamard = (a, b) => a.map((row, i) => row.map((v, j) => v * b[i][j]));
  function matmul(a, b) {
    const r = a.length, k = b.length, c = b[0].length;
    const out = zeros(r, c);
    for (let i = 0; i < r; i++) {
      for (let p = 0; p < k; p++) {
        const av = a[i][p];
        if (av === 0) continue;
        const brow = b[p];
        const orow = out[i];
        for (let j = 0; j < c; j++) orow[j] += av * brow[j];
      }
    }
    return out;
  }
  const colMeans = (m) => { const r = m.length, c = m[0].length; const out = new Array(c).fill(0); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j] += m[i][j]; for (let j = 0; j < c; j++) out[j] /= r; return out; };
  const rowMeans = (m) => m.map(r => r.reduce((s, v) => s + v, 0) / r.length);
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  function norm2(v) { let s = 0; for (const x of v) s += x * x; return Math.sqrt(s); }
  const sum = (m) => m.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0);
  const maxAbs = (m) => { let mx = 0; for (const row of m) for (const v of row) mx = Math.max(mx, Math.abs(v)); return mx; };

  // ---------- Cholesky 分解: A = L Lᵀ (A 对称正定) ----------
  function cholesky(A) {
    const n = A.length;
    const L = zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          // 数值保护：保证半正定可分解
          L[i][j] = Math.sqrt(Math.max(s, 1e-12));
        } else {
          L[i][j] = s / (L[j][j] || 1e-12);
        }
      }
    }
    return L;
  }

  // 前向/后向代入，解 L Lᵀ x = b
  function cholSolve(L, b) {
    const n = L.length;
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / (L[i][i] || 1e-12); }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / (L[i][i] || 1e-12); }
    return x;
  }
  // 对称正定矩阵求逆
  function inverseSPD(A) {
    const L = cholesky(A);
    const n = A.length;
    const inv = zeros(n, n);
    for (let i = 0; i < n; i++) { const e = new Array(n).fill(0); e[i] = 1; inv[i] = cholSolve(L, e); }
    return transpose(inv); // 各列是单位向量解 → 转置得逆
  }

  // ---------- Jacobi 特征分解 (对称矩阵 A = V diag(λ) Vᵀ, V 正交) ----------
  function jacobiEigen(Ain, opts) {
    opts = opts || {};
    const maxSweeps = opts.maxSweeps || 100;
    const tol = opts.tol || 1e-12;
    const n = Ain.length;
    const A = copy(Ain);
    let V = identity(n);
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < tol) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(A[p][q]) < 1e-300) continue;
          const app = A[p][p], aqq = A[q][q], apq = A[p][q];
          const phi = 0.5 * Math.atan2(2 * apq, aqq - app); // 2φ = atan2(2apq, aqq-app)
          const c = Math.cos(phi), s = Math.sin(phi);
          for (let k = 0; k < n; k++) {
            const akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = A[p][k], aqk = A[q][k];
            A[p][k] = c * apk - s * aqk;
            A[q][k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const idx = [];
    for (let i = 0; i < n; i++) idx.push(i);
    idx.sort((a, b) => A[b][b] - A[a][a]); // 降序
    const eigenvalues = idx.map(i => A[i][i]);
    const eigenvectors = []; // 列向量数组
    for (let i = 0; i < n; i++) { const vec = new Array(n); for (let k = 0; k < n; k++) vec[k] = V[k][idx[i]]; eigenvectors.push(vec); }
    return { eigenvalues, eigenvectors, sweeps: Math.min(maxSweeps, 999) };
  }

  // ---------- 一般矩阵 SVD: A(m×n) = U S Vᵀ ----------
  // 复用对称特征分解于 AᵀA，经典且数值稳健 (Golub-Reinsch 思想的世界顶级实现路径)
  function svd(A, opts) {
    const m = A.length, n = A[0].length;
    const At = transpose(A);
    const C = matmul(At, A); // n×n 对称半正定
    const { eigenvalues, eigenvectors } = jacobiEigen(C, opts);
    const S = eigenvalues.map(l => Math.sqrt(Math.max(l, 0)));
    const V = transpose(eigenvectors); // n×n，列=右奇异向量
    const eps = 1e-12;
    const rank = S.filter(s => s > eps * (S[0] + 1)).length;
    // U 列: u_i = A v_i / σ_i
    const U = zeros(m, n);
    for (let i = 0; i < n; i++) {
      const vi = V[i]; // 第 i 行即第 i 右奇异向量
      const Avi = new Array(m);
      for (let k = 0; k < m; k++) Avi[k] = dot(A[k], vi);
      const si = S[i];
      if (si > eps) { for (let k = 0; k < m; k++) U[k][i] = Avi[k] / si; }
      else { for (let k = 0; k < m; k++) U[k][i] = 0; }
    }
    return { U, S, V, rank };
  }

  // ---------- 数值稳定函数 ----------
  function logsumexp(v) {
    let mx = -Infinity;
    for (const x of v) if (x > mx) mx = x;
    if (!isFinite(mx)) return mx;
    let s = 0;
    for (const x of v) s += Math.exp(x - mx);
    return mx + Math.log(s);
  }
  function softmax(v) {
    const m = logsumexp(v);
    return v.map(x => Math.exp(x - m));
  }
  function softmaxRows(M) { return M.map(softmax); }
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const sigmoidV = (v) => v.map(sigmoid);
  function stableLog(x) { return Math.log(Math.max(x, 1e-300)); }

  // ---------- 校验工具 (数学不变量) ----------
  function isSymmetric(A, tol = 1e-10) { const n = A.length; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Math.abs(A[i][j] - A[j][i]) > tol) return false; return true; }
  function isPSD(A, tol = 1e-9) { const { eigenvalues } = jacobiEigen(A); return eigenvalues.every(l => l >= -tol); }
  // 正交性误差: ||VᵀV - I|| (V 列正交)
  function orthogonalityError(Vcols) {
    const n = Vcols.length;
    let err = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const d = dot(Vcols[i], Vcols[j]) - (i === j ? 1 : 0);
      err = Math.max(err, Math.abs(d));
    }
    return err;
  }
  // 特征方程残差: ||A v_i - λ_i v_i|| (应 ≈ 机器精度)
  function eigenResidual(A, eigenvalues, eigenvectors) {
    let mx = 0;
    for (let i = 0; i < eigenvalues.length; i++) {
      const Av = new Array(A.length);
      for (let k = 0; k < A.length; k++) Av[k] = dot(A[k], eigenvectors[i]);
      const diff = Av.map((v, k) => v - eigenvalues[i] * eigenvectors[i][k]);
      mx = Math.max(mx, norm2(diff));
    }
    return mx;
  }
  // Cholesky 重构误差: ||L Lᵀ - A||
  function choleskyReconError(L, A) { return maxAbs(sub(matmul(L, transpose(L)), A)); }
  // SVD 重构误差: ||U S Vᵀ - A||
  function svdReconError(U, S, V, A) {
    const m = A.length, n = A[0].length;
    const SVt = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) SVt[i][j] = S[i] * V[j][i]; // V 行=右奇异向量
    const US = matmul(U, SVt);
    return maxAbs(sub(US, A));
  }

  return {
    rng, randn, zeros, identity, copy, transpose, add, sub, scale, hadamard, matmul,
    colMeans, rowMeans, dot, norm2, sum, maxAbs,
    cholesky, cholSolve, inverseSPD,
    jacobiEigen, svd,
    logsumexp, softmax, softmaxRows, sigmoid, sigmoidV, stableLog,
    isSymmetric, isPSD, orthogonalityError, eigenResidual, choleskyReconError, svdReconError
  };
});
