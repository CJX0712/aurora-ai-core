/*
 * Aurora AI Core — 验证内核 + 微指令层 (创新点)
 * Author: 晨星 (Morning Star)
 *
 * 两层保障：
 *  1) Invariant Harness —— 每个算法均有可被独立交叉验证的数学不变量。
 *  2) MicroEngine —— 将算法分解为微指令 (micro-instruction)，每步带不变量自动校验，
 *     落实「第十层 1024 项微指令级操作」愿景：运行时逐元素/逐步产出可审计微指令。
 */
(function (root, factory) {
  const g = (typeof globalThis !== 'undefined') ? globalThis : root;
  const dep = (typeof module !== 'undefined' && module.exports) ? require('./num.js') : (g.Aurora && g.Aurora.Num);
  const lib = factory(dep);
  if (typeof module !== 'undefined' && module.exports) module.exports = lib;
  g.Aurora = g.Aurora || {};
  g.Aurora.Verify = lib;
})(typeof self !== 'undefined' ? self : this, function (N) {
  'use strict';

  // ============ 1) 不变量测试框架 ============
  function createHarness() {
    const checks = [];
    function register(name, group, fn) { checks.push({ name, group, fn }); }
    function run() {
      const results = [], byGroup = {};
      let passed = 0, failed = 0;
      for (const c of checks) {
        let ok = false, detail = '';
        try { const r = c.fn(); ok = !!(r && (r.ok === undefined ? true : r.ok)); detail = r && r.detail ? r.detail : (r ? '' : ''); if (r && r.ok === false) ok = false; }
        catch (e) { ok = false; detail = String(e && e.message || e); }
        const res = { name: c.name, group: c.group, ok, detail };
        results.push(res);
        byGroup[c.group] = byGroup[c.group] || { p: 0, f: 0 };
        if (ok) { passed++; byGroup[c.group].p++; } else { failed++; byGroup[c.group].f++; }
      }
      return { passed, failed, total: checks.length, results, byGroup };
    }
    return { register, run, checks };
  }

  // ============ 2) 微指令引擎 ============
  // 每个微指令均为可独立验证的原子操作；运行时记录并逐一校验不变量。
  function createMicro() {
    const catalog = {}; // id -> {invariant(output, inputs)}
    const trace = [];   // 执行记录
    function defMicro(id, invariant, doc) { catalog[id] = { id, invariant, doc: doc || '' }; }
    function has(id) { return !!catalog[id]; }
    // 记录一次微指令执行并校验不变量；返回 ok
    function record(id, inputs, output) {
      const spec = catalog[id];
      let ok = true, reason = '';
      if (spec) {
        try { ok = spec.invariant(output, inputs); }
        catch (e) { ok = false; reason = String(e && e.message || e); }
      } else { reason = 'unknown-micro-op'; }
      trace.push({ id, ok, reason, summary: microSummary(id, inputs, output) });
      return ok;
    }
    function microSummary(id, inputs, output) {
      if (id === 'MAC') return `C[${inputs.i}][${inputs.j}] += Σ A[${inputs.i}][k]·B[k][${inputs.j}]`;
      if (id === 'SOFTMAX_ROW') return `row → softmax (Σ=1)`;
      if (id === 'CHOL_DIAG') return `L[ii]=√(Aii-ΣLik²)`;
      if (id === 'CHOL_OFF') return `L[ij]=(Aij-Σ)/Ljj`;
      if (id === 'JACOBI_ROT') return `rotate (p,q) by φ`;
      if (id === 'COV_SYM') return `C = XᵀX symmetric`;
      return id;
    }
    function report() {
      const total = trace.length;
      const failed = trace.filter(t => !t.ok).length;
      const byOp = {};
      for (const t of trace) { byOp[t.id] = byOp[t.id] || { n: 0, f: 0 }; byOp[t.id].n++; if (!t.ok) byOp[t.id].f++; }
      return { total, failed, passed: total - failed, byOp, catalogSize: Object.keys(catalog).length };
    }
    function reset() { trace.length = 0; }

    // ---- 预置世界顶级微指令目录 (含数学不变量) ----
    defMicro('MAC', (out, inp) => {
      // 不变量: out == Σ_k A_row[k]·B_col[k]
      let s = 0; for (let k = 0; k < inp.a.length; k++) s += inp.a[k] * inp.b[k];
      return Math.abs(out - s) < 1e-9;
    }, '乘加: C[i][j] = Σ A[i][k]·B[k][j]');
    defMicro('SOFTMAX_ROW', (out) => {
      // 不变量: 每行概率和 = 1, 且 0≤p≤1
      const s = out.reduce((a, b) => a + b, 0);
      return Math.abs(s - 1) < 1e-9 && out.every(p => p >= 0 && p <= 1 + 1e-12);
    }, '行 softmax: Σp=1, 0≤p≤1');
    defMicro('CHOL_DIAG', (out, inp) => {
      // 不变量: out² == Aii - Σ_{k< i} L_{ik}²
      let s = 0; for (let k = 0; k < inp.row.length; k++) s += inp.row[k] * inp.row[k];
      return Math.abs(out * out - (inp.Aii - s)) < 1e-7;
    }, 'Cholesky 对角元');
    defMicro('CHOL_OFF', (out, inp) => {
      // 不变量: out·Ljj == Aij - Σ L_{ik}·L_{jk}
      let s = 0; for (let k = 0; k < inp.Li.length; k++) s += inp.Li[k] * inp.Lj[k];
      return Math.abs(out * inp.Ljj - (inp.Aij - s)) < 1e-7;
    }, 'Cholesky 非对角元');
    defMicro('JACOBI_ROT', (out, inp) => {
      // 不变量: 旋转后 (p,q) 非对角元趋于 0 (δ ≤ 旋转前)
      return Math.abs(out.newOff) <= Math.abs(inp.oldOff) + 1e-9;
    }, 'Jacobi 旋转: 消去非对角元');
    defMicro('COV_SYM', (C) => N.isSymmetric(C, 1e-9), '协方差对称: C = Cᵀ');

    return { defMicro, has, record, report, reset, trace };
  }

  // ---- 微指令版矩阵乘 (逐元素产出 MAC 微指令) ----
  function microMatmul(M, A, B) {
    const r = A.length, k = B.length, c = B[0].length;
    const C = N.zeros(r, c);
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) {
        const aRow = A[i], bCol = B.map(row => row[j]);
        let s = 0; for (let p = 0; p < k; p++) s += aRow[p] * bCol[p];
        C[i][j] = s;
        M.record('MAC', { a: aRow, b: bCol, i, j }, s);
      }
    }
    return C;
  }
  // ---- 微指令版行 softmax ----
  function microSoftmaxRows(M, X) {
    return X.map(row => {
      const m = Math.max(...row);
      const e = row.map(v => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      const p = e.map(v => v / s);
      M.record('SOFTMAX_ROW', {}, p);
      return p;
    });
  }

  return { createHarness, createMicro, microMatmul, microSoftmaxRows };
});
